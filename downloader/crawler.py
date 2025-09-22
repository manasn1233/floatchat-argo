# downloader/crawler.py
"""
Robust crawler for IFREMER index pages.
- Uses requests-style headers via aiohttp to present a browser-like User-Agent.
- Uses BeautifulSoup for reliable link extraction instead of fragile regex.
- Logs the first N characters of the fetched HTML so we can inspect listing format.
"""

import asyncio
from urllib.parse import urljoin
import aiohttp
from bs4 import BeautifulSoup

# How many characters of the index HTML to log (helps debug in CI logs)
LOG_HTML_CHARS = 2000

# Headers to look like a browser (some servers return different content for bots)
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

async def fetch_text(session, url, timeout=30):
    async with session.get(url, timeout=timeout, headers=DEFAULT_HEADERS) as resp:
        resp.raise_for_status()
        return await resp.text()

async def list_files_in_folder(session, folder_url):
    """
    Parse folder index and return absolute .nc links found there.
    """
    html = await fetch_text(session, folder_url)
    # Debug: log the beginning of the HTML so we can inspect index format in CI logs
    print(f"--- BEGIN HTML SNIPPET for {folder_url} ---")
    print(html[:LOG_HTML_CHARS])
    print(f"--- END HTML SNIPPET for {folder_url} ---")
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".nc"):
            links.add(urljoin(folder_url, href))
    return sorted(links)

async def list_year_dirs(session, root_url):
    """
    Parse root listing and return absolute year directory URLs (YYYY/).
    """
    html = await fetch_text(session, root_url)
    print(f"--- BEGIN ROOT HTML SNIPPET for {root_url} ---")
    print(html[:LOG_HTML_CHARS])
    print(f"--- END ROOT HTML SNIPPET for {root_url} ---")
    soup = BeautifulSoup(html, "html.parser")
    years = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # treat links that are exactly a 4-digit year or end with 'YYYY/'
        if href.isdigit() and len(href) == 4:
            years.add(urljoin(root_url, href + "/"))
        elif href.endswith("/") and href[:-1].isdigit() and len(href[:-1]) == 4:
            years.add(urljoin(root_url, href))
    return sorted(years)

async def crawl_root_for_files(root_url, max_years=None):
    files = []
    async with aiohttp.ClientSession() as session:
        try:
            years = await list_year_dirs(session, root_url)
        except Exception as e:
            print("Failed to fetch or parse root URL:", root_url, e)
            return []

        if not years:
            print("No year directories found at root:", root_url)
            return []

        if max_years:
            years = years[-max_years:]

        for yurl in years:
            try:
                found = await list_files_in_folder(session, yurl)
                files.extend(found)
            except Exception as e:
                print("Failed to parse year folder", yurl, e)
                continue

    return sorted(files)

# Sync wrapper for convenience
def list_files_sync(root_url, max_years=None):
    return asyncio.run(crawl_root_for_files(root_url, max_years=max_years))

if __name__ == "__main__":
    # quick manual test (not run in Actions)
    roots = [
        "https://data-argo.ifremer.fr/geo/atlantic_ocean/",
        "https://data-argo.ifremer.fr/geo/indian_ocean/"
    ]
    for r in roots:
        try:
            res = list_files_sync(r, max_years=1)
            print(r, "->", len(res), "files sample:", res[:5])
        except Exception as e:
            print("crawl failed for", r, e)
