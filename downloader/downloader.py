# downloader/downloader.py
"""
Robust downloader with retries and server-error handling.

Behavior:
- HEAD check for Content-Length (skip if too large)
- Stream download to .part file then rename on success
- Retry on 5xx and 429 (honor Retry-After if provided)
- Exponential backoff with jitter
- Logs actions so CI shows why a file was skipped/failed
"""

import os
import hashlib
import asyncio
import aiohttp
import aiofiles
from urllib.parse import urlparse
import random
import datetime

from db import db_helpers
from sqlalchemy import text

DATA_DIR = os.getenv("DATA_DIR", "./data/netcdf")
os.makedirs(DATA_DIR, exist_ok=True)

# Tunables (can be overridden via env in Actions)
MAX_SIZE_BYTES = int(os.getenv("MAX_SIZE_BYTES", 60 * 1024 * 1024))  # 60 MB
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", 120))  # seconds for connect/read
RETRIES = int(os.getenv("DOWNLOAD_RETRIES", 4))
CONCURRENCY = int(os.getenv("DOWNLOAD_CONCURRENCY", 3))

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def compute_checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# Custom exceptions
class ServerError(Exception):
    pass

class RateLimitError(Exception):
    def __init__(self, retry_after):
        super().__init__("Rate limited")
        self.retry_after = retry_after

class ClientError(Exception):
    pass

def backoff_seconds(attempt, base=1.0, cap=60.0):
    delay = min(cap, base * (2 ** (attempt - 1)))
    jitter = random.uniform(0, delay * 0.5)
    return delay + jitter

async def head_size(session, url):
    try:
        async with session.head(url, timeout=DOWNLOAD_TIMEOUT, headers=DEFAULT_HEADERS) as resp:
            if resp.status == 200:
                cl = resp.headers.get("Content-Length")
                if cl and cl.isdigit():
                    return int(cl)
    except Exception as e:
        print("HEAD failed for", url, ":", e)
    return None

async def download_stream(session, url, dst_path):
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=DOWNLOAD_TIMEOUT, sock_read=DOWNLOAD_TIMEOUT)
    async with session.get(url, timeout=timeout, headers=DEFAULT_HEADERS) as resp:
        status = resp.status
        if status >= 500:
            body = await resp.text()[:400]
            raise ServerError(f"{status}: {body}")
        if status == 429:
            ra = resp.headers.get("Retry-After")
            raise RateLimitError(ra)
        if status >= 400:
            body = await resp.text()[:400]
            raise ClientError(f"{status}: {body}")

        part = dst_path + ".part"
        async with aiofiles.open(part, "wb") as f:
            async for chunk in resp.content.iter_chunked(64 * 1024):
                if not chunk:
                    break
                await f.write(chunk)
        os.replace(part, dst_path)

async def download_if_new(session, url, conn):
    already = conn.execute(text("SELECT id FROM netcdf_files WHERE file_url = :u"), {"u": url}).fetchone()
    if already:
        print("Already in DB, skipping:", url)
        return None

    file_name = os.path.basename(urlparse(url).path)
    dst_path = os.path.join(DATA_DIR, file_name)

    size = await head_size(session, url)
    if size is not None:
        print("Content-Length for", url, "=", size, "bytes")
        if size > MAX_SIZE_BYTES:
            print(f"Skipping {url} because size {size} > MAX_SIZE_BYTES {MAX_SIZE_BYTES}")
            return None

    last_exc = None
    for attempt in range(1, RETRIES + 1):
        try:
            await download_stream(session, url, dst_path)
            checksum = compute_checksum(dst_path)
            print("Downloaded", url, "->", dst_path, "checksum", checksum)
            return {"url": url, "local_path": dst_path, "file_name": file_name, "checksum": checksum}
        except RateLimitError as rl:
            ra = rl.retry_after
            if ra:
                try:
                    wait = int(ra)
                except Exception:
                    try:
                        wait = int((datetime.datetime.strptime(ra, "%a, %d %b %Y %H:%M:%S %Z") - datetime.datetime.utcnow()).total_seconds())
                    except Exception:
                        wait = int(backoff_seconds(attempt))
            else:
                wait = int(backoff_seconds(attempt))
            print(f"Rate limited on {url}, waiting {wait}s (attempt {attempt})")
            await asyncio.sleep(wait)
            last_exc = rl
            continue
        except ServerError as se:
            wait = backoff_seconds(attempt)
            print(f"Server error for {url}: {se}. Backing off {wait:.1f}s (attempt {attempt})")
            await asyncio.sleep(wait)
            last_exc = se
            continue
        except ClientError as ce:
            print(f"Client error for {url}, skipping: {ce}")
            last_exc = ce
            break
        except Exception as e:
            wait = backoff_seconds(attempt)
            print(f"Transient error for {url}: {e}. Backing off {wait:.1f}s (attempt {attempt})")
            await asyncio.sleep(wait)
            last_exc = e
            continue

    print("Failed to download after retries:", url, "last_exc:", last_exc)
    try:
        conn.execute(text(
            "INSERT INTO download_failures (file_url, last_error, attempts, last_attempt) "
            "VALUES (:u, :err, :a, now()) "
            "ON CONFLICT (file_url) DO UPDATE SET last_error = EXCLUDED.last_error, attempts = download_failures.attempts + 1, last_attempt = now()"
        ), {"u": url, "err": str(last_exc), "a": RETRIES})
        conn.commit()
    except Exception:
        pass

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

def download_batch_sync(urls, concurrency=CONCURRENCY):
    return asyncio.run(download_batch(urls, concurrency=concurrency))

if __name__ == "__main__":
    sample = [
        "https://data-argo.ifremer.fr/geo/atlantic_ocean/2019/08/20190801_prof.nc"
    ]
    print("Local test (not in CI):")
    try:
        res = download_batch_sync(sample)
        print("Downloaded:", res)
    except Exception as e:
        print("Test failed:", e)
