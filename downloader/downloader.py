# downloader/downloader.py
"""
Download .nc files listed by the crawler, but skip files already recorded in the DB.
Returns a list of dicts for newly downloaded files:
  { "url": ..., "local_path": ..., "file_name": ..., "checksum": ... }
This file uses the SQLAlchemy engine from db/db_helpers.py to query the DB.
"""

import os
import hashlib
import asyncio
import aiohttp
import aiofiles
from urllib.parse import urlparse

from db import db_helpers
from sqlalchemy import text

DATA_DIR = os.getenv("DATA_DIR", "./data/netcdf")
os.makedirs(DATA_DIR, exist_ok=True)

def compute_checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

async def download_file(session, url, dst_path):
    async with session.get(url, timeout=120) as resp:
        resp.raise_for_status()
        async with aiofiles.open(dst_path, "wb") as f:
            async for chunk in resp.content.iter_chunked(64 * 1024):
                await f.write(chunk)

async def download_if_new(session, url, conn):
    # Check DB for existence
    already = conn.execute(text("SELECT id FROM netcdf_files WHERE file_url = :u"), {"u": url}).fetchone()
    if already:
        return None

    file_name = os.path.basename(urlparse(url).path)
    dst_path = os.path.join(DATA_DIR, file_name)

    try:
        await download_file(session, url, dst_path)
    except Exception as e:
        # On failure, ensure partial file is removed
        if os.path.exists(dst_path):
            try:
                os.remove(dst_path)
            except Exception:
                pass
        raise

    checksum = compute_checksum(dst_path)
    return {"url": url, "local_path": dst_path, "file_name": file_name, "checksum": checksum}

async def download_batch(urls, concurrency=4):
    """
    Download a batch of URLs concurrently, skipping those already in DB.
    Returns list of successful download dicts.
    """
    results = []
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        async def worker(u):
            async with sem:
                # open a local DB connection for this check/download
                with db_helpers.engine.connect() as conn:
                    try:
                        res = await download_if_new(session, u, conn)
                        if res:
                            results.append(res)
                    except Exception as e:
                        # log and continue
                        print("Failed to download", u, e)

        tasks = [asyncio.create_task(worker(u)) for u in urls]
        await asyncio.gather(*tasks)
    return results

# convenience sync wrapper for quick tests
def download_batch_sync(urls, concurrency=4):
    return asyncio.run(download_batch(urls, concurrency=concurrency))

if __name__ == "__main__":
    # quick local test (won't run in Actions without network)
    sample = [
        "https://data-argo.ifremer.fr/geo/atlantic_ocean/2023/sample.nc"
    ]
    print("Testing download (no real download in CI)...")
    try:
        res = download_batch_sync(sample)
        print("Downloaded:", res)
    except Exception as e:
        print("Test failed:", e)
