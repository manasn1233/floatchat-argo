# scripts/run_once.py
"""
Orchestration script (restricted): crawl → download → ingest for a single folder.
This version is temporarily configured to process only Indian Ocean 2025 data.
"""

import asyncio

# For this run we explicitly target the 2025 folder under the Indian Ocean
ROOTS = [
    ("https://data-argo.ifremer.fr/geo/indian_ocean/2025/", "indian_2025"),
]

# CI_MAX_YEARS is ignored when ROOTS contains full year folder URLs,
# but we keep it here for compatibility with the rest of the code.
CI_MAX_YEARS = 1

def process_sync_for_root(root_url, region, max_years=None):
    # Lazy imports
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
    # Lazy imports
    from downloader.crawler import crawl_root_for_files
    from downloader.downloader import download_batch
    from downloader.ingest import ingest_file_entry

    print("Crawling", root_url)
    # When root_url already points to a year folder we pass max_years=None (crawler will crawl the folder)
    files = await crawl_root_for_files(root_url, max_years=None)
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
    # We don't use CI_MAX_YEARS here because ROOTS points directly at year folder(s)
    loop = asyncio.get_event_loop()
    for root_url, region in ROOTS:
        try:
            loop.run_until_complete(process_async_for_root(root_url, region, max_years=None))
        except Exception as e:
            print("Async run failed, falling back to sync for", root_url, e)
            try:
                process_sync_for_root(root_url, region, max_years=None)
            except Exception as ee:
                print("Sync fallback also failed for", root_url, ee)

if __name__ == "__main__":
    main()
