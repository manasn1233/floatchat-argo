# downloader/downloader.py
"""
Robust downloader: check Content-Length first, skip very large files, retry on transient errors,
and stream the file to disk. Designed for CI runs where a single stuck download can hang the job.
"""

import os
import hashlib
import asyncio
import aiohttp
import aiofiles
from urllib.parse import urlparse
import time

from db import db_helpers
from sqlalchemy import text

DATA_DIR = os.getenv("DATA_DIR", "./data/netcdf")
os.makedirs(DATA_DIR, exist_ok=True)

# tune these as needed
MAX_SIZE_BYTES = int(os.getenv("MAX_SIZE_BYTES", 60 * 1024 * 1024))  # 60 MB default
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", 120))  # seconds per request
RETRIES = int(os.getenv("DOWNLOAD_RETRIES", 3))
CONCURRENCY = int(os.getenv("DOWNLOAD_CONCURRENCY", 3))

def compute_checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

async def head_size(session, url):
    """
    Try to perform a HEAD to get Content-Length. Return int bytes or None.
    Some servers may not respond to HEAD; handle gracefully.
    """
    try:
        async with session.head(url, timeout=DOWNLOAD_TIMEOUT) as resp:
            if resp.status == 200:
                cl = resp.headers.get("Content-Length")
                if cl and cl.isdigit():
                    return int(cl)
    except Exception as e:
        # HEAD may be blocked or unsupported; ignore and return None
        print("HEAD failed for", url, ":", e)
    return None

async def download_file(session, url, dst_path):
    """
    Stream download with per-chunk writes. Raises on non-200 or network errors.
    """
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=DOWNLOAD_TIMEOUT, sock_read=DOWNLOAD_TIMEOUT)
    async with session.get(url, timeout=timeout) as resp:
        resp.raise_for_status()
        async with aiofiles.open(dst_path, "wb") as f:
            async for chunk in resp.content.iter_chunked(64 * 1024):
                if not chunk:
                    break
                await f.write(chunk)

async def download_if_new(session, url, conn):
    # DB check
    already = conn.execute(text("SELECT id FROM netcdf_files WHERE file_url = :u"), {"u": url}).fetchone()
    if already:
        print("Already in DB, skipping:", url)
        return None

    file_name = os.path.basename(urlparse(url).path)
    dst_path = os.path.join(DATA_DIR, file_name)

    # check size via HEAD
    size = await head_size(session, url)
    if size is not None:
        print("Content-Length for", url, "=", size, "bytes")
        if size > MAX_SIZE_BYTES:
            print(f"Skipping {url} because size {size} > MAX_SIZE_BYTES {MAX_SIZE_BYTES}")
            return None

    # attempt with retries
    last_exc = None
    for attempt in range(1, RETRIES + 1):
        try:
            await download_file(session, url, dst_path)
            checksum = compute_checksum(dst_path)
            print("Downloaded", url, "->", dst_path, "checksum", checksum)
            return {"url": url, "local_path": dst_path, "file_name": file_name, "checksum": checksum}
        except Exception as e:
            last_exc = e
            print(f"Attempt {attempt} failed for {url}: {e}")
            # clean partial file
            if os.path.exists(dst_path):
                try:
                    os.remove(dst_path)
                except Exception:
                    pass
            # small backoff
            await asyncio.sleep(2 * attempt)
    # all retries failed
    print("All retries failed for", url, "last_exc:", last_exc)
    return None

async def download_batch(urls, concurrency=CONCURRENCY):
    results = []
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        async def worker(u):
            async with sem:
                with db_helpers.engine.connect() as conn:
                    try:
                        res = await download_if_new(session, u, conn)
                        if res:
                            results.append(res)
                    except Exception as e:
                        print("Failed to download", u, e)

        tasks = [asyncio.create_task(worker(u)) for u in urls]
        await asyncio.gather(*tasks)
    return results

# convenience sync wrapper
def download_batch_sync(urls, concurrency=CONCURRENCY):
    return asyncio.run(download_batch(urls, concurrency=concurrency))

if __name__ == "__main__":
    sample = [
        "https://data-argo.ifremer.fr/geo/atlantic_ocean/2019/08/20190801_prof.nc"
    ]
    print("Testing download (local)...")
    try:
        res = download_batch_sync(sample)
        print("Downloaded:", res)
    except Exception as e:
        print("Test failed:", e)
