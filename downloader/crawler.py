# downloader/crawler.py
"""
Robust crawler for IFREMER index pages (restricted to a root path).
- Uses aiohttp with a browser-like User-Agent.
- Uses BeautifulSoup for reliable link extraction.
- Finds .nc files by crawling only within the path prefix of the given root_url.
- If root_url points to a region root (e.g. .../indian_ocean/), the crawler will
  discover year subdirectories automatically and optionally limit to the most
  recent N years or a MIN_YEAR environment cutoff.
- Recursively follows links up to a depth limit to find .nc files.
- Logs HTML snippets and discovered hrefs for debugging in CI logs.
"""

import os
import asyncio
import re
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
                p = parsed.path
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


async def list_year_dirs_from_root(session, root_url):
    """
    Fetch root_url and return a list of year-directory absolute URLs (e.g. .../2025/).
    Only returns directories that look like 4-digit years.
    """
    try:
        html = await fetch_text(session, root_url)
    except Exception as e:
        print(f"Failed to fetch root {root_url} for year discovery: {e}")
        return []

    links = await extract_links(html, root_url)
    years = set()
    for raw, full in links:
        # match hrefs like '2025/' or '/geo/.../2025/'
        m = re.match(r"^\d{4}/?$", raw)
        if m:
            years.add(full if full.endswith('/') else full + '/')
        else:
            # try to parse year from path
            parsed = urlparse(full)
            parts = [p for p in parsed.path.split('/') if p]
            for part in parts:
                if re.match(r"^\d{4}$", part):
                    # build absolute year dir URL
                    year_idx = parsed.path.index(part)
                    base = full[:full.rfind(part)+len(part)]
                    years.add(base if base.endswith('/') else base + '/')
                    break
    # return sorted ascending
    try:
        return sorted(years)
    except Exception:
        return list(years)


async def crawl_root_for_files(root_url, max_years=None, min_year=None):
    """
    Crawl the root_url and return .nc file URLs found under this root path.

    Behavior:
    - If root_url appears to be a region root (path ends with the region folder, not a specific year),
      the function will attempt to discover year subdirectories and then crawl each year folder.
    - max_years: if provided (int), limit to that many most recent years.
    - min_year: if provided (int), only include years >= min_year.
    """
    parsed_root = urlparse(root_url)
    path = parsed_root.path
    # detect if root_url already points to a year folder (ends with /YYYY/)
    year_match = re.search(r"/(\d{4})/+$", path)

    async with aiohttp.ClientSession() as session:
        if year_match:
            # it's already year-specific; just crawl this folder
            return await crawl(root_url, max_depth=3)

        # Otherwise try to discover year dirs under root_url
        candidate_years = await list_year_dirs_from_root(session, root_url)
        # candidate_years are absolute URLs like .../2019/
        # parse numeric year and filter by min_year
        year_map = []  # list of tuples (year_int, url)
        for yurl in candidate_years:
            try:
                y = int(re.search(r"(\d{4})", yurl).group(1))
            except Exception:
                continue
            if min_year and y < int(min_year):
                continue
            year_map.append((y, yurl))

        if not year_map:
            # fallback: crawl root directly
            return await crawl(root_url, max_depth=3)

        # sort descending (most recent first)
        year_map.sort(key=lambda x: x[0], reverse=True)

        # apply max_years if provided
        if max_years is not None:
            year_map = year_map[:int(max_years)]

        # iterate years and crawl each year folder, aggregate results
        all_files = set()
        for y, yurl in year_map:
            print(f"Crawling year {y} -> {yurl}")
            try:
                files = await crawl(yurl, max_depth=3)
                for f in files:
                    all_files.add(f)
            except Exception as e:
                print(f"Failed crawling {yurl}: {e}")
        return sorted(all_files)


# synchronous convenience wrapper for local tests / CI compatibility
def list_files_sync(root_url, max_years=None, min_year=None):
    return asyncio.run(crawl_root_for_files(root_url, max_years=max_years, min_year=min_year))


if __name__ == "__main__":
    roots = [
        "https://data-argo.ifremer.fr/geo/indian_ocean/",
    ]
    # optional env driven defaults
    MAX_YEARS = os.getenv("CI_MAX_YEARS")
    MIN_YEAR = os.getenv("MIN_YEAR")
    for r in roots:
        try:
            res = list_files_sync(r, max_years=MAX_YEARS, min_year=MIN_YEAR)
            print(r, "-> found", len(res), "nc files (sample):")
            for f in res[:20]:
                print("  ", f)
        except Exception as e:
            print("crawl failed for", r, e)
