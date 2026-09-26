import os
import time
import logging
import asyncio
import bisect
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

REPO_ID = "darkdevjlfjfmf/Tgdata-bucket"
CACHE_DIR = os.getenv("CACHE_DIR", "/data/tgdb_cache")

dataset = None
user_id_index = None
is_ready = False
init_error = None

stats = {
    "startup_time": 0,
    "total_files": 0,
    "total_row_groups": 0,
    "total_rows": 0,
    "queries": 0,
}

ALL_COLS = [
    "user_id",
    "username",
    "first_name",
    "last_name",
    "phone",
    "email",
    "status",
    "linked_id",
    "linked_name",
    "linked_handle",
]


def find_parquet(base):
    files = []
    for root, _, names in os.walk(base):
        for name in names:
            if name.endswith(".parquet"):
                files.append(os.path.join(root, name))
    return sorted(files)


class UserIdIndex:
    def __init__(self):
        self.entries = []
        self.keys = []

    def build(self, arrow_dataset):
        logger.info("Building user_id index...")

        schema = arrow_dataset.schema

        try:
            uid_idx = schema.get_field_index("user_id")
        except Exception:
            uid_idx = -1

        if uid_idx < 0:
            logger.warning("user_id column not found")
            return

        for fragment in arrow_dataset.get_fragments():
            metadata = fragment.metadata

            if metadata is None:
                continue

            for row_group in range(metadata.num_row_groups):
                column = metadata.row_group(row_group).column(uid_idx)
                statistics = column.statistics

                if (
                    statistics
                    and statistics.min is not None
                    and statistics.max is not None
                ):
                    try:
                        self.entries.append(
                            (
                                int(statistics.min),
                                int(statistics.max),
                                fragment.path,
                                row_group,
                            )
                        )
                    except Exception:
                        pass

        self.entries.sort(key=lambda x: x[0])
        self.keys = [x[0] for x in self.entries]

        logger.info(
            "Index built: %s row groups",
            len(self.entries)
        )

    def find(self, user_id):
        if not self.entries:
            return None

        index = bisect.bisect_right(
            self.keys,
            user_id
        ) - 1

        if 0 <= index < len(self.entries):
            minimum, maximum, path, row_group = self.entries[index]

            if minimum <= user_id <= maximum:
                return path, row_group

        return None


def init_dataset():
    global dataset
    global user_id_index
    global is_ready
    global init_error

    try:
        os.makedirs(CACHE_DIR, exist_ok=True)

        files = find_parquet(CACHE_DIR)

        if not files:
            logger.info(
                "Downloading dataset: %s",
                REPO_ID
            )

            start = time.time()

            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=CACHE_DIR,
                local_dir_use_symlinks=False,
            )

            logger.info(
                "Download completed in %.1fs",
                time.time() - start
            )

            files = find_parquet(CACHE_DIR)

        if not files:
            raise RuntimeError(
                "No .parquet files found in Tgdata-bucket"
            )

        logger.info(
            "Loading %s parquet files",
            len(files)
        )

        arrow_dataset = ds.dataset(
            files,
            format="parquet"
        )

        row_groups = 0
        rows = 0

        for fragment in arrow_dataset.get_fragments():
            metadata = fragment.metadata

            if metadata:
                row_groups += metadata.num_row_groups

            try:
                rows += fragment.count_rows()
            except Exception:
                pass

        index = UserIdIndex()
        index.build(arrow_dataset)

        dataset = arrow_dataset
        user_id_index = index

        stats["total_files"] = len(files)
        stats["total_row_groups"] = row_groups
        stats["total_rows"] = rows
        stats["startup_time"] = time.time()

        is_ready = True

        logger.info(
            "API READY | files=%s | rows=%s | row_groups=%s",
            len(files),
            rows,
            row_groups
        )

    except Exception as e:
        init_error = str(e)
        logger.exception("Dataset initialization failed")


async def background_init():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, init_dataset)


@asynccontextmanager
async def lifespan(app):
    asyncio.create_task(background_init())
    yield


app = FastAPI(
    title="Tgdata API",
    description="Tgdata Hugging Face API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_ready():
    if not is_ready:
        if init_error:
            raise HTTPException(
                status_code=503,
                detail=init_error
            )

        raise HTTPException(
            status_code=503,
            detail="Dataset is still loading"
        )


def query_user(user_id: int):
    stats["queries"] += 1
    start = time.time()

    result = user_id_index.find(user_id)

    if result:
        path, row_group = result

        try:
            pf = pq.ParquetFile(path)

            table = pf.read_row_group(
                row_group,
                columns=ALL_COLS
            )

            table = table.filter(
                pc.equal(
                    pc.field("user_id"),
                    user_id
                )
            )

            if len(table):
                return (
                    table.to_pylist()[0],
                    time.time() - start
                )

        except Exception as e:
            logger.warning(
                "Indexed lookup failed: %s",
                e
            )

    table = dataset.to_table(
        filter=pc.equal(
            pc.field("user_id"),
            user_id
        ),
        columns=ALL_COLS
    )

    elapsed = time.time() - start

    if len(table):
        return table.to_pylist()[0], elapsed

    return None, elapsed


@app.get("/")
async def root():
    return {
        "success": True,
        "service": "Tgdata API",
        "dataset": REPO_ID,
        "ready": is_ready,
        "endpoints": {
            "user": "/user/{user_id}",
            "search": "/search",
            "health": "/health"
        }
    }


@app.get("/user/{user_id}")
async def get_user(user_id: int):
    require_ready()

    user, elapsed = query_user(user_id)

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    return {
        "success": True,
        "found": True,
        "query_time_ms": round(
            elapsed * 1000,
            2
        ),
        "user": user
    }


@app.get("/search")
async def search(
    username: str = Query(None),
    phone: str = Query(None),
    email: str = Query(None),
    first_name: str = Query(None),
    last_name: str = Query(None),
    limit: int = Query(10, ge=1, le=100),
):
    require_ready()

    params = {
        "username": username,
        "phone": phone,
        "email": email,
        "first_name": first_name,
        "last_name": last_name,
    }

    params = {
        k: v for k, v in params.items()
        if v
    }

    if not params:
        raise HTTPException(
            status_code=400,
            detail="Provide a search parameter"
        )

    conditions = []

    for key, value in params.items():
        conditions.append(
            pc.equal(
                pc.field(key),
                value
            )
        )

    combined = conditions[0]

    for condition in conditions[1:]:
        combined = combined & condition

    columns = [
        "user_id",
        "username",
        "first_name",
        "last_name",
        "phone",
        "email",
        "status",
    ]

    start = time.time()

    table = dataset.to_table(
        filter=combined,
        columns=columns
    )

    elapsed = time.time() - start

    rows = table.to_pylist()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No users found"
        )

    return {
        "success": True,
        "count": len(rows),
        "returned": min(len(rows), limit),
        "query_time_ms": round(
            elapsed * 1000,
            2
        ),
        "users": rows[:limit]
    }


@app.get("/health")
async def health():
    return {
        "status": (
            "ok"
            if is_ready
            else "error"
            if init_error
            else "loading"
        ),
        "ready": is_ready,
        "error": init_error,
        "dataset": REPO_ID,
        "files": stats["total_files"],
        "row_groups": stats["total_row_groups"],
        "rows": stats["total_rows"],
        "queries": stats["queries"],
    }
