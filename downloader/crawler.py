# downloader/crawler.py
"""
Simple crawler to list year folders and .nc files from IFREMER directory pages.
This is intentionally conservative: it parses HTML index pages and extracts links
ending with .nc and year directories (YYYY/). If IFREMER exposes a JSON API,
replace the parsing with JSON fetching for robustness.
"""

import re
from urllib.parse import urljoin
import aiohttp
import asyncio

NC_RE = re.compile(r'href=["\']([^"\']+\.nc)["\']', re.IGNORECASE)
YEAR_DIR_RE = re.compile(r'href=["\']([0-9]{4})/["\']')

async def fetch_text(session, url, timeout=30):
    async with session.get(url, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.text()

async def list_files_in_folder(session, folder_url):
    """
    Parse an index HTML page at folder_url and return absolute .nc file URLs found there.
    """
    text = await fetch_text(session, folder_url)
    links = set()
    for m in NC_RE.finditer(text):
        href = m.group(1)
        full = urljoin(folder_url, href)
        links.add(full)
    return sorted(links)

async def list_years(root_url):
    """
    Return a list of year folder URLs discovered at root_url (e.g., .../atlantic_ocean/).
    """
    async with aiohttp.ClientSession() as session:
        text = await fetch_text(session, root_url)
    years = []
    for m in YEAR_DIR_RE.finditer(text):
        y = m.group(1)
        years.append(urljoin(root_url, f"{y}/"))
    years = sorted(set(years))
    return years

async def crawl_root_for_files(root_url, max_years=None):
    """
    Crawl the root IFREMER folder and return a list of .nc file URLs found under year folders.
    max_years: if set, limit to the newest N years discovered (helps testing).
    """
    files = []
    async with aiohttp.ClientSession() as session:
        # discover years
        text = await fetch_text(session, root_url)
        year_dirs = YEAR_DIR_RE.findall(text)
        year_dirs = sorted(set(year_dirs))
        if max_years:
            year_dirs = year_dirs[-max_years:]
        for y in year_dirs:
            year_url = urljoin(root_url, f"{y}/")
            try:
                found = await list_files_in_folder(session, year_url)
                files.extend(found)
            except Exception:
                # skip problematic year pages but continue
                continue
    return sorted(files)

# Simple sync wrapper for convenience (useful for tests / running locally)
def list_files_sync(root_url, max_years=None):
    return asyncio.run(crawl_root_for_files(root_url, max_years=max_years))

if __name__ == "__main__":
    # quick local test (won't run in Actions without network)
    roots = [
        "https://data-argo.ifremer.fr/geo/atlantic_ocean/",
        "https://data-argo.ifremer.fr/geo/indian_ocean/"
    ]
    for r in roots:
        try:
            files = list_files_sync(r, max_years=1)  # limit for a quick test
            print(r, "->", len(files), "files (sample):")
            for f in files[:5]:
                print("  ", f)
        except Exception as e:
            print("Error crawling", r, e)
