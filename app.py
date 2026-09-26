import os
import glob
import asyncio
from typing import Optional

from fastapi import FastAPI, Query
from huggingface_hub import snapshot_download
import pyarrow.dataset as ds
import pyarrow.compute as pc


app = FastAPI(
    title="TG Data API",
    version="1.0.0"
)

# Hugging Face Dataset
REPO_ID = "sauravsingh2111/Tgdata"

# Railway local cache
CACHE_DIR = os.getenv("CACHE_DIR", "/data/tgdb_cache")

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

dataset = None
loading = False
load_error = None


def find_parquet():
    return glob.glob(
        os.path.join(CACHE_DIR, "**", "*.parquet"),
        recursive=True
    )


def init_dataset():
    global dataset, loading, load_error

    loading = True

    try:
        os.makedirs(CACHE_DIR, exist_ok=True)

        files = find_parquet()

        # Download dataset from Hugging Face if not already cached
        if not files:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=CACHE_DIR,
                local_dir_use_symlinks=False,
            )

        files = find_parquet()

        if not files:
            raise RuntimeError(
                "No .parquet files found in Hugging Face dataset"
            )

        dataset = ds.dataset(
            files,
            format="parquet"
        )

        print(f"Loaded {len(files)} parquet file(s)")

    except Exception as e:
        load_error = str(e)
        print("Dataset loading error:", e)

    finally:
        loading = False


@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, init_dataset)


@app.get("/")
def home():
    return {
        "status": "online",
        "api": "TG Data API",
        "dataset": REPO_ID,
        "endpoints": {
            "health": "/health",
            "user": "/user/{user_id}",
            "search": "/search",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    if loading:
        return {
            "status": "loading",
            "dataset": REPO_ID
        }

    if load_error:
        return {
            "status": "error",
            "error": load_error
        }

    return {
        "status": "ready",
        "dataset": REPO_ID
    }


@app.get("/user/{user_id}")
def get_user(user_id: str):

    if dataset is None:
        return {
            "status": "error",
            "message": "Dataset is not ready"
        }

    try:
        table = dataset.to_table(
            columns=ALL_COLS,
            filter=pc.equal(
                ds.field("user_id"),
                user_id
            )
        )

        rows = table.to_pylist()

        return {
            "status": "success",
            "count": len(rows),
            "data": rows
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/search")
def search(
    username: Optional[str] = Query(None),
    phone: Optional[str] = Query(None),
    email: Optional[str] = Query(None),
    first_name: Optional[str] = Query(None),
    last_name: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100)
):

    if dataset is None:
        return {
            "status": "error",
            "message": "Dataset is not ready"
        }

    filters = []

    if username:
        filters.append(
            pc.equal(ds.field("username"), username)
        )

    if phone:
        filters.append(
            pc.equal(ds.field("phone"), phone)
        )

    if email:
        filters.append(
            pc.equal(ds.field("email"), email)
        )

    if first_name:
        filters.append(
            pc.equal(ds.field("first_name"), first_name)
        )

    if last_name:
        filters.append(
            pc.equal(ds.field("last_name"), last_name)
        )

    if not filters:
        return {
            "status": "error",
            "message": "Provide at least one search parameter"
        }

    filter_expr = filters[0]

    for f in filters[1:]:
        filter_expr = filter_expr & f

    try:
        table = dataset.to_table(
            columns=ALL_COLS,
            filter=filter_expr
        )

        rows = table.to_pylist()[:limit]

        return {
            "status": "success",
            "count": len(rows),
            "data": rows
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
