# downloader/crawler.py
"""
Recursive crawler for IFREMER index pages (more robust):
- Follows links up to `max_depth` to find .nc files in nested folders.
- Uses aiohttp with a browser-like User-Agent.
- Uses BeautifulSoup to parse links and logs all hrefs it sees (for debugging).
- Accepts any href containing '.nc' (to catch query-strings or unusual naming).
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
        # make absolute
        full = urljoin(base_url, href)
        links.append((href, full))
    return links

async def crawl(root_url, max_depth=3, max_pages=500):
    """
    BFS crawl starting from root_url, depth-limited.
    Returns sorted list of found .nc file URLs (unique).
    Logs discovered hrefs and HTML snippets for debugging in CI logs.
    """
    found_nc = set()
    seen = set()
    queue = [(root_url, 0)]
    pages_visited = 0

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

            # debug: print snippet of HTML for the first few pages
            print(f"--- BEGIN HTML SNIPPET for {url} ---")
            print(html[:LOG_HTML_CHARS])
            print(f"--- END HTML SNIPPET for {url} ---")

            links = await extract_links(html, url)
            # log what hrefs we saw (first 200 links to avoid huge logs)
            if links:
                print(f"Found {len(links)} <a> tags on {url}. Sample hrefs:")
                for raw, full in links[:200]:
                    print("  href:", raw, "->", full)

            # process links
            for raw, full in links:
                lower = raw.lower()
                if ".nc" in lower or ".nc?" in lower:
                    found_nc.add(full)
                else:
                    # consider following this link if it's likely a directory or HTML page
                    # skip if it points to an external host
                    parsed = urlparse(full)
                    parsed_root = urlparse(root_url).netloc
                    if parsed.netloc and parsed.netloc != parsed_root:
                        continue
                    # follow if depth allows and link ends with '/' or looks like a folder
                    if depth + 1 <= max_depth and (raw.endswith("/") or raw.isdigit() or raw.endswith(".html") or "/" in raw):
                        if full not in seen:
                            queue.append((full, depth + 1))
            # end for links
        # end while
    return sorted(found_nc)

# helper wrappers
def crawl_root_for_files(root_url, max_years=None):
    # note: max_years is ignored here because we do a general crawl — keep for back-compat.
    return asyncio.run(crawl(root_url, max_depth=3))

def list_files_sync(root_url, max_years=None):
    return crawl_root_for_files(root_url, max_years=max_years)

if __name__ == "__main__":
    roots = [
        "https://data-argo.ifremer.fr/geo/atlantic_ocean/",
        "https://data-argo.ifremer.fr/geo/indian_ocean/"
    ]
    for r in roots:
        try:
            res = list_files_sync(r)
            print(r, "-> found", len(res), "nc files (sample):")
            for f in res[:10]:
                print("  ", f)
        except Exception as e:
            print("crawl failed for", r, e)
