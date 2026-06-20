"""
crawl_to_pdf.py
===============
Crawls a website starting from a root URL, follows all internal links
up to 2 levels deep, and saves each page as a PDF in an output folder.

Usage:
    python crawl_to_pdf.py --url https://mimic.mit.edu/docs/IV/ --output knowledge

Arguments:
    --url       Root URL to start crawling from
    --output    Output folder for PDFs (default: knowledge)
    --depth     Max crawl depth (default: 2)
    --delay     Seconds to wait between page loads (default: 1.5)

Ignored links (configurable in IGNORE_PATTERNS below):
    - accessibility.mit.edu
    - github.com/MIT-LCP/mimic-website/edit  (edit on GitHub links)
    - Any non-HTTP links (mailto:, tel:, #anchors)
"""

import argparse
import re
import time
import os
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# ── Configuration ─────────────────────────────────────────────────────────────

# Links matching any of these patterns are silently skipped
IGNORE_PATTERNS = [
    r"accessibility\.mit\.edu",
    r"github\.com/.*?/edit",
    r"github\.com/.*?/blob",
    r"github\.com/.*?/tree",
    r"twitter\.com",
    r"linkedin\.com",
    r"facebook\.com",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def should_ignore(url: str) -> bool:
    for pattern in IGNORE_PATTERNS:
        if re.search(pattern, url):
            return True
    return False


def is_internal(url: str, base_domain: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == base_domain or parsed.netloc == ""


def url_to_filename(url: str, visited_order: int) -> str:
    """Convert a URL to a safe filename."""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "__") or "index"
    # Remove special chars
    path = re.sub(r"[^\w\-.]", "_", path)
    # Prefix with order number for easy sorting
    return f"{visited_order:03d}_{path}.pdf"


def extract_links(page, base_url: str, base_domain: str) -> list[str]:
    """Extract all valid internal links from the current page."""
    content = page.content()
    soup = BeautifulSoup(content, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        # Skip anchors, mailto, tel, javascript
        if not href or href.startswith("#") or ":" in href.split("/")[0] and not href.startswith("http"):
            continue
        full_url = urljoin(base_url, href)
        # Strip fragment
        full_url = full_url.split("#")[0]
        if not full_url.startswith("http"):
            continue
        if not is_internal(full_url, base_domain):
            continue
        if should_ignore(full_url):
            continue
        links.append(full_url)
    return list(dict.fromkeys(links))  # deduplicate preserving order


def save_pdf(page, url: str, out_path: Path, delay: float):
    """Navigate to URL and save as PDF."""
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
        time.sleep(delay)
        page.pdf(
            path=str(out_path),
            format="A4",
            print_background=True,
            margin={"top": "1cm", "bottom": "1cm", "left": "1cm", "right": "1cm"},
        )
        return True
    except Exception as e:
        print(f"    [ERROR] Could not save {url}: {e}")
        return False


# ── Main crawler ──────────────────────────────────────────────────────────────

def crawl(root_url: str, output_dir: str, max_depth: int, delay: float):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_domain = urlparse(root_url).netloc

    # queue entries: (url, depth)
    queue = [(root_url, 0)]
    visited = set()
    counter = 0

    print(f"\n{'='*60}")
    print(f"Root URL   : {root_url}")
    print(f"Domain     : {base_domain}")
    print(f"Max depth  : {max_depth}")
    print(f"Output dir : {out.resolve()}")
    print(f"{'='*60}\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent="Mozilla/5.0 (compatible; crawl_to_pdf/1.0)",
        )
        page = context.new_page()

        while queue:
            url, depth = queue.pop(0)

            if url in visited:
                continue
            visited.add(url)

            counter += 1
            filename = url_to_filename(url, counter)
            out_path = out / filename

            print(f"[{counter:03d}] depth={depth}  {url}")
            print(f"      -> {filename}")

            ok = save_pdf(page, url, out_path, delay)
            if not ok:
                counter -= 1
                continue

            # Extract links from this page if not at max depth
            if depth < max_depth:
                links = extract_links(page, url, base_domain)
                new_links = [(lnk, depth + 1) for lnk in links if lnk not in visited]
                # Insert at front to do depth-first; use append for breadth-first
                queue = new_links + queue
                print(f"      Found {len(new_links)} new links to follow")

        browser.close()

    print(f"\n{'='*60}")
    print(f"Done. {counter} PDFs saved to: {out.resolve()}")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl a website and save each page as PDF."
    )
    parser.add_argument(
        "--url",
        default="https://mimic.mit.edu/docs/IV/",
        help="Root URL to start crawling from",
    )
    parser.add_argument(
        "--output",
        default="knowledge",
        help="Output folder for PDFs (default: knowledge)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Max crawl depth (default: 2)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds to wait after page load before PDF (default: 1.5)",
    )

    args = parser.parse_args()
    crawl(
        root_url=args.url,
        output_dir=args.output,
        max_depth=args.depth,
        delay=args.delay,
    )