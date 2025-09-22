# scripts/run_once.py
"""
Orchestration script: crawl → download → ingest for IFREMER roots.
This version uses lazy imports to avoid module import-time errors in CI.
"""

import asyncio

# Limit years for quick CI runs (set to None for full)
CI_MAX_YEARS = 1  # set to 1 for testing; change to None to process all years

ROOTS = [
    ("https://data-argo.ifremer.fr/geo/atlantic_ocean/", "atlantic"),
    ("https://data-argo.ifremer.fr/geo/indian_ocean/", "indian"),
]

def process_sync_for_root(root_url, region, max_years=None):
    from downloader.crawler import list_files_sync
    from downloader.downloader import download_batch_sync
    from downloader.ingest import ingest_file_entry

    print("Listing files for", root_url)
    files = list_files_sync(root_url, max_years=max_years)
    print("Found", len(files), "files (limited view)")
    if not files:
        return
    downloaded = download_batch_sync(files, concurrency=4)
    print("Downloaded new files:", len(downloaded))
    for entry in downloaded:
        try:
            ingest_file_entry(entry, ocean_region=region)
        except Exception as e:
            print("Ingest failed for", entry.get("url"), e)

async def process_async_for_root(root_url, region, max_years=None):
    from downloader.crawler import crawl_root_for_files
    from downloader.downloader import download_batch
    from downloader.ingest import ingest_file_entry

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
    max_years = CI_MAX_YEARS
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
