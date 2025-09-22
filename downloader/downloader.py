# downloader/crawler.py
"""
Robust crawler for IFREMER index pages (newest-year-first).
- Uses aiohttp with a browser-like User-Agent.
- Uses BeautifulSoup for reliable link extraction.
- Finds year directories, sorts them newest-first, and crawls only the newest N years when requested.
- Recursively follows links up to a depth limit to find .nc files.
- Logs HTML snippets and discovered hrefs for debugging in CI logs.
"""

import asyncio
from urllib.parse import urljoin, urlparse
import aiohttp
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
LOG_HTML_CHARS = 2000

async def fetch_text(session, url, timeout=30):
    async with session.get(url, timeout=timeout, headers=DEFAULT_HEADERS) as resp:
        resp.raise_for_status()
        return await resp.text()

async def extract_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(base_url, href)
        links.append((href, full))
    return links

async def list_year_dirs(session, root_url):
    """
    Parse root listing and return a list of year directory absolute URLs,
    sorted newest-first (descending).
    """
    html = await fetch_text(session, root_url)
    print(f"--- BEGIN ROOT HTML SNIPPET for {root_url} ---")
    print(html[:LOG_HTML_CHARS])
    print(f"--- END ROOT HTML SNIPPET for {root_url} ---")
    soup = BeautifulSoup(html, "html.parser")
    years = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # accept 'YYYY' and 'YYYY/' patterns
        if href.isdigit() and len(href) == 4:
            years.add(href)
        elif href.endswith("/") and href[:-1].isdigit() and len(href[:-1]) == 4:
            years.add(href[:-1])
    # sort descending (newest first)
    sorted_years = sorted(years, reverse=True)
    return [urljoin(root_url, f"{y}/") for y in sorted_years]

async def list_files_in_folder(session, folder_url):
    """
    Parse a folder index page and return absolute .nc links found there.
    """
    html = await fetch_text(session, folder_url)
    print(f"--- BEGIN HTML SNIPPET for {folder_url} ---")
    print(html[:LOG_HTML_CHARS])
    print(f"--- END HTML SNIPPET for {folder_url} ---")
    links = await extract_links(html, folder_url)
    nc_links = []
    for raw, full in links:
        if ".nc" in raw.lower():
            nc_links.append(full)
    return sorted(set(nc_links))

async def crawl(root_url, max_depth=3, max_pages=500):
    """
    BFS crawl root_url up to max_depth to find .nc files.
    Returns sorted list of unique .nc URLs.
    """
    found_nc = set()
    seen = set()
    queue = [(root_url, 0)]
    pages_visited = 0
    parsed_root = urlparse(root_url).netloc

    async with aiohttp.ClientSession() as session:
        while queue and pages_visited < max_pages:
            url, depth = queue.pop(0)
            if url in seen:
                continue
            seen.add(url)
            pages_visited += 1
            try:
                html = await fetch_text(session, url)
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")
                continue

            # log HTML snippet for debugging
            print(f"--- BEGIN HTML SNIPPET for {url} ---")
            print(html[:LOG_HTML_CHARS])
            print(f"--- END HTML SNIPPET for {url} ---")

            links = await extract_links(html, url)
            if links:
                print(f"Found {len(links)} <a> tags on {url}. Sample hrefs:")
                for raw, full in links[:200]:
                    print("  href:", raw, "->", full)

            for raw, full in links:
                lower = raw.lower()
                if ".nc" in lower or ".nc?" in lower:
                    found_nc.add(full)
                else:
                    # only follow links within same host
                    parsed = urlparse(full)
                    if not parsed.netloc or parsed.netloc != parsed_root:
                        continue
                    # follow plausible directory or index links
                    if depth + 1 <= max_depth and (raw.endswith("/") or raw.isdigit() or raw.endswith(".html") or "/" in raw):
                        if full not in seen:
                            queue.append((full, depth + 1))

    return sorted(found_nc)

async def crawl_root_for_files(root_url, max_years=None):
    """
    Crawl the root_url and return .nc file URLs found under the newest year directories.
    If max_years is provided, only the newest `max_years` years are traversed.
    """
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

        # pick newest-first top N years if requested
        if max_years:
            years = years[:max_years]

        print(f"Will crawl these year folders for {root_url}: {years}")

        for yurl in years:
            try:
                found = await list_files_in_folder(session, yurl)
                files.extend(found)
            except Exception as e:
                print("Failed to parse year folder", yurl, e)
                continue

    return sorted(files)

# synchronous convenience wrapper for local tests / CI compatibility
def list_files_sync(root_url, max_years=None):
    return asyncio.run(crawl_root_for_files(root_url, max_years=max_years))

if __name__ == "__main__":
    roots = [
        "https://data-argo.ifremer.fr/geo/atlantic_ocean/",
        "https://data-argo.ifremer.fr/geo/indian_ocean/"
    ]
    for r in roots:
        try:
            res = list_files_sync(r, max_years=1)
            print(r, "-> found", len(res), "nc files (sample):")
            for f in res[:10]:
                print("  ", f)
        except Exception as e:
            print("crawl failed for", r, e)
