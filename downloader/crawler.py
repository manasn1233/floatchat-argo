# downloader/crawler.py
"""
Robust crawler for IFREMER index pages (restricted to a root path).
- Uses aiohttp with a browser-like User-Agent.
- Uses BeautifulSoup for reliable link extraction.
- Finds .nc files by crawling only within the path prefix of the given root_url.
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
    BFS crawl starting from root_url, depth-limited.
    Only follows links whose path begins with the root_url path (prevents climbing to parent dirs).
    Returns sorted list of found .nc file URLs (unique).
    """
    found_nc = set()
    seen = set()
    queue = [(root_url, 0)]
    pages_visited = 0

    parsed_root = urlparse(root_url)
    root_netloc = parsed_root.netloc
    # ensure root_path ends with a slash for prefix checks
    root_path = parsed_root.path if parsed_root.path.endswith("/") else parsed_root.path + "/"

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

            # debug: print HTML snippet for the first few pages
            print(f"--- BEGIN HTML SNIPPET for {url} ---")
            print(html[:LOG_HTML_CHARS])
            print(f"--- END HTML SNIPPET for {url} ---")

            links = await extract_links(html, url)
            if links:
                print(f"Found {len(links)} <a> tags on {url}. Sample hrefs:")
                for raw, full in links[:200]:
                    print("  href:", raw, "->", full)

            for raw, full in links:
                # normalize and check the URL belongs under the same host
                parsed = urlparse(full)
                if parsed.netloc != root_netloc:
                    # external host -> skip
                    continue

                # enforce path prefix restriction: do not follow if path climbs above root_path
                # e.g., parent directory links like '/geo/indian_ocean/' won't be followed when root_path is '/geo/indian_ocean/2025/'
                p = parsed.path
                # ensure trailing slash for directory comparisons
                p_slash = p if p.endswith("/") else p + "/"
                if not p_slash.startswith(root_path):
                    # skip links outside the specified root path
                    continue

                lower = raw.lower()
                if ".nc" in lower or ".nc?" in lower:
                    found_nc.add(full)
                else:
                    # follow plausible directory/index links only if depth allows
                    if depth + 1 <= max_depth and (raw.endswith("/") or raw.isdigit() or raw.endswith(".html") or "/" in raw):
                        if full not in seen:
                            queue.append((full, depth + 1))
        # end while
    return sorted(found_nc)

async def crawl_root_for_files(root_url, max_years=None):
    """
    Crawl the root_url and return .nc file URLs found under this root path.
    If max_years is provided and the root_url is a true root (not a specific year), the caller
    logic can still choose years — but when root_url points directly at a year folder, this function
    will only crawl within that year folder because of the path-prefix restriction.
    """
    # For compatibility, we still return the crawl() result; caller must pass the correct root_url
    return await crawl(root_url, max_depth=3)

# synchronous convenience wrapper for local tests / CI compatibility
def list_files_sync(root_url, max_years=None):
    return asyncio.run(crawl_root_for_files(root_url, max_years=max_years))

if __name__ == "__main__":
    roots = [
        "https://data-argo.ifremer.fr/geo/indian_ocean/2025/",
    ]
    for r in roots:
        try:
            res = list_files_sync(r, max_years=1)
            print(r, "-> found", len(res), "nc files (sample):")
            for f in res[:20]:
                print("  ", f)
        except Exception as e:
            print("crawl failed for", r, e)
