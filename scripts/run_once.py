# scripts/run_once.py
"""
Orchestration script: crawl → download → ingest for IFREMER roots.
Designed to be run in CI (GitHub Actions) or locally (with network access).
It processes two roots: atlantic_ocean and indian_ocean by default.
"""

import asyncio
import os
from downloader.crawler import crawl_root_for_files, list_files_sync
from downloader.downloader import download_batch, download_batch_sync
from downloader.ingest import ingest_file_entry
from urllib.parse import urlparse

# Roots to crawl
ROOTS = [
    ("https://data-argo.ifremer.fr/geo/atlantic_ocean/", "atlantic"),
    ("https://data-argo.ifremer.fr/geo/indian_ocean/", "indian"),
]

# Limit years for quick CI runs (set to None for full)
CI_MAX_YEARS = 1  # set to 1 or 2 to test quickly

def process_sync_for_root(root_url, region, max_years=None):
    print("Listing files for", root_url)
    files = list_files_sync(root_url, max_years=max_years)
    print("Found", len(files), "files (limited view)")
    if not files:
        return
    # Download only new (sync)
    downloaded = download_batch_sync(files, concurrency=4)
    print("Downloaded new files:", len(downloaded))
    for entry in downloaded:
        try:
            ingest_file_entry(entry, ocean_region=region)
        except Exception as e:
            print("Ingest failed for", entry.get("url"), e)

async def process_async_for_root(root_url, region, max_years=None):
    print("Crawling", root_url)
    files = await crawl_root_for_files(root_url, max_years=max_years)
    print("Found", len(files), "files")
    if not files:
        return
    downloaded = await download_batch(files)
    print("Downloaded new files:", len(downloaded))
    for entry in downloaded:
        try:
            ingest_file_entry(entry, ocean_region=region)
        except Exception as e:
            print("Ingest failed for", entry.get("url"), e)

def main():
    # If running in CI where network is allowed, use async runner.
    # For quick local testing without heavy downloads, you can set CI_MAX_YEARS to 1.
    max_years = CI_MAX_YEARS

    # Prefer async orchestration where possible
    loop = asyncio.get_event_loop()
    for root_url, region in ROOTS:
        try:
            loop.run_until_complete(process_async_for_root(root_url, region, max_years=max_years))
        except Exception as e:
            print("Async run failed, falling back to sync for", root_url, e)
            try:
                process_sync_for_root(root_url, region, max_years=max_years)
            except Exception as ee:
                print("Sync fallback also failed for", root_url, ee)

if __name__ == "__main__":
    main()
