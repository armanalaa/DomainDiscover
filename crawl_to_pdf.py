"""
crawl_to_pdf.py
===============
Crawls a website starting from a root URL, follows all internal links
up to a given depth, and saves each page as a PDF in the dataset's
knowledge/ folder.

Run from the DomainMiner root folder:

    python crawl_to_pdf.py --dataset_dir Airportdb --url https://dev.mysql.com/doc/airportdb/en/
    python crawl_to_pdf.py --dataset_dir Synthea   --url https://synthea.mitre.org/about --depth 1

The PDFs are saved to:
    <dataset_dir>/knowledge/

Multiple crawls into the same folder are supported — run the command
again with a different --url to add more PDFs before running
extract_knowledge.py.

Arguments:
    --dataset_dir   Dataset subfolder inside DomainMiner (e.g. Airportdb)
    --url           Root URL to start crawling from
    --depth         Max crawl depth (default: 2)
    --delay         Seconds to wait between page loads (default: 1.5)
"""

import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# ── Configuration ─────────────────────────────────────────────────────────────

IGNORE_PATTERNS = [
    r"accessibility\.mit\.edu",
    r"github\.com/.*?/edit",
    r"github\.com/.*?/blob",
    r"github\.com/.*?/tree",
    r"twitter\.com",
    r"linkedin\.com",
    r"facebook\.com",
]

# Network domains that serve cookie/consent banner scripts — aborted entirely.
# No script loading = no banner injected, regardless of timing.
BLOCKED_DOMAINS = [
    "cdn.cookielaw.org",              # OneTrust (Oracle, MySQL, many enterprise sites)
    "optanon.blob.core.windows.net",
    "geolocation.onetrust.com",
    "cdn-ukwest.onetrust.com",
    "privacyportal.onetrust.com",
    "consent.cookiebot.com",          # Cookiebot
    "cookieconsentpro.com",
    "consent.trustarc.com",           # TrustArc / TRUSTe (Oracle docs)
    "choices.trustarc.com",
    "preferences.trustarc.com",
    "consent-pref.trustarc.com",
    "privacy-policy.truste.com",
    "consent.truste.com",
]

# CSS injected before PDF capture.
# Wrapped in both screen and print media so it applies when page.pdf() renders.
HIDE_OVERLAY_CSS = """
    #onetrust-consent-sdk, #onetrust-banner-sdk, #onetrust-pc-sdk,
    .onetrust-pc-dark-filter, .ot-sdk-container, .ot-floating-button,
    .truste_overlay, .truste_popframe, .trustarc_newcm_container,
    [id^='pop-div'], [id^='pop-frame'], [id^='truste-consent'],
    iframe[src*='trustarc'], iframe[src*='truste'],
    [id*='cookie-banner'], [id*='cookie_banner'], [id*='cookieBanner'],
    [class*='cookie-banner'], [class*='cookieBanner'],
    [id*='gdpr'], [class*='gdpr'],
    [id*='consent-banner'], [class*='consent-banner'] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }
    body { overflow: auto !important; }

    @media print {
        #onetrust-consent-sdk, #onetrust-banner-sdk, #onetrust-pc-sdk,
        .onetrust-pc-dark-filter, .ot-sdk-container, .ot-floating-button,
        .truste_overlay, .truste_popframe, .trustarc_newcm_container,
        [id^='pop-div'], [id^='pop-frame'], [id^='truste-consent'],
        iframe[src*='trustarc'], iframe[src*='truste'],
        [id*='cookie-banner'], [class*='cookie-banner'],
        [id*='consent-banner'], [class*='consent-banner'],
        [id*='gdpr'], [class*='gdpr'] {
            display: none !important;
        }
    }
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def should_ignore(url: str) -> bool:
    return any(re.search(p, url) for p in IGNORE_PATTERNS)


def is_internal(url: str, base_domain: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == base_domain or parsed.netloc == ""


def is_under_root_path(url: str, root_path: str) -> bool:
    return urlparse(url).path.startswith(root_path)


def normalize_url(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}{p.path.rstrip('/') or '/'}"


def url_to_filename(url: str, order: int) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "__") or "index"
    path = re.sub(r"[^\w\-.]", "_", path)
    return f"{order:03d}_{path}.pdf"


def extract_links(page, base_url: str, base_domain: str, root_path: str) -> list:
    soup = BeautifulSoup(page.content(), "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith("#"):
            continue
        if ":" in href.split("/")[0] and not href.startswith("http"):
            continue
        full_url = urljoin(base_url, href).split("#")[0]
        if not full_url.startswith("http"):
            continue
        if not is_internal(full_url, base_domain):
            continue
        if not is_under_root_path(full_url, root_path):
            continue
        if should_ignore(full_url):
            continue
        links.append(full_url)
    return list(dict.fromkeys(links))


# ── Consent banner suppression ───────────────────────────────────────────────
#
# Oracle/MySQL serves OneTrust from their own domain, so blocking external CDNs
# is not enough. The definitive fix is add_init_script(): Playwright injects our
# JS BEFORE any page script runs, sets up a MutationObserver, and removes banner
# nodes the instant they are added — completely race-condition-proof.

INIT_SCRIPT = """
(function() {

    // ── 1. Stub out OneTrust globals before its script runs ──────────────────
    // OneTrust checks for these on init; if already defined it exits early.
    var noop = function() { return { then: noop, catch: noop }; };
    var stubOT = {
        Init: noop, initializeCookiePolicyHtml: noop,
        GetDomainData: noop, IsAlertBoxClosed: function() { return true; },
        IsAlertBoxClosedAndValid: function() { return true; },
        TriggerGoogleAnalyticsEvent: noop,
        OnConsentChanged: noop, Close: noop, LoadBanner: noop
    };
    try { Object.defineProperty(window, 'OneTrust',        { value: stubOT,  writable: false, configurable: false }); } catch(e) {}
    try { Object.defineProperty(window, 'OnetrustActiveGroups', { value: '1,2,3,4', writable: false }); } catch(e) {}
    try { Object.defineProperty(window, 'OptanonWrapper',  { value: noop,    writable: false, configurable: false }); } catch(e) {}
    try { Object.defineProperty(window, 'Optanon',         { value: stubOT,  writable: false, configurable: false }); } catch(e) {}

    // ── 2. Pre-fill localStorage consent keys ────────────────────────────────
    try {
        var now = new Date().toISOString();
        localStorage.setItem('OptanonAlertBoxClosed', now);
        localStorage.setItem('OptanonConsent', 'isIABGlobal=false&interactionCount=1&groups=1:1,2:1,3:1,4:1');
        localStorage.setItem('notice_behavior', 'implied,eu');
        localStorage.setItem('truste.eu.cookie.notice_preferences', '0,1,2');
        localStorage.setItem('truste.eu.cookie.notice_gdpr_prefs', '0,1,2');
    } catch(e) {}

    // ── 3. MutationObserver — remove any banner nodes that slip through ───────
    var SELECTORS = [
        '#onetrust-consent-sdk', '#onetrust-banner-sdk',
        '#onetrust-pc-sdk', '.onetrust-pc-dark-filter',
        '.ot-floating-button', '.truste_overlay', '.truste_popframe',
        '.trustarc_newcm_container', '[id^="pop-div"]', '[id^="pop-frame"]',
        '[id^="truste-consent"]', 'iframe[src*="trustarc"]',
        'iframe[src*="truste"]', '[id*="cookie-banner"]',
        '[class*="cookie-banner"]', '[id*="consent-banner"]'
    ];
    function removeOverlays() {
        SELECTORS.forEach(function(sel) {
            document.querySelectorAll(sel).forEach(function(el) { el.remove(); });
        });
        if (document.body) document.body.style.overflow = 'auto';
    }
    removeOverlays();
    new MutationObserver(function(mutations) {
        var changed = false;
        for (var i = 0; i < mutations.length; i++) {
            if (mutations[i].addedNodes.length) { changed = true; break; }
        }
        if (changed) removeOverlays();
    }).observe(document.documentElement, { childList: true, subtree: true });

})();
"""


def setup_banner_suppression(context, page):
    """
    Layer 1 — Script content interception:
      Intercept every JS response. If the body contains OneTrust/Optanon code,
      replace it with an empty stub. This neutralises the banner script
      regardless of which domain serves it (including the page's own domain).

    Layer 2 — add_init_script MutationObserver + global stubs (belt & suspenders).
    """
    context.add_init_script(script=INIT_SCRIPT)

    def handle(route):
        req = route.request
        # For JS files, fetch the response and check if it's OneTrust
        if req.resource_type == "script":
            try:
                resp = route.fetch()
                body = resp.body()
                keywords = [b"OneTrust", b"onetrust", b"OptanonWrapper",
                            b"cookielaw", b"CookiePolicyHtml", b"cookiebanner"]
                if any(kw in body for kw in keywords):
                    print(f"      [BLOCK] OneTrust script neutralised: {req.url[:80]}")
                    route.fulfill(
                        status=200,
                        content_type="application/javascript",
                        body="/* consent script blocked */"
                    )
                    return
                route.fulfill(response=resp)
                return
            except Exception:
                pass
        # All other requests: block known CDNs, continue the rest
        if any(d in req.url for d in BLOCKED_DOMAINS):
            route.abort()
        else:
            route.continue_()

    page.route("**/*", handle)


# ── PDF saving ────────────────────────────────────────────────────────────────

def save_pdf(page, url: str, out_path: Path, delay: float, retries: int = 2):
    """Navigate to URL and save as PDF."""
    for attempt in range(1, retries + 2):
        try:
            wait_strategy = "domcontentloaded" if attempt <= retries else "commit"
            page.goto(url, wait_until=wait_strategy, timeout=60000)
            time.sleep(delay)
            # Nuclear cleanup right before capture:
            # 1. Inject CSS covering both screen + @media print
            # 2. Remove known banner selectors from DOM
            # 3. Hide ALL fixed/sticky high-z-index overlays by computed style
            try:
                page.add_style_tag(content=HIDE_OVERLAY_CSS)
            except Exception:
                pass
            try:
                page.evaluate("""
                    (function() {
                        // Remove known selectors
                        var sels = [
                            '#onetrust-consent-sdk','#onetrust-banner-sdk',
                            '#onetrust-pc-sdk','.onetrust-pc-dark-filter',
                            '.ot-floating-button','.truste_overlay',
                            '.truste_popframe','.trustarc_newcm_container',
                            '[id^="pop-div"]','[id^="pop-frame"]',
                            '[id^="truste-consent"]',
                            'iframe[src*="trustarc"]','iframe[src*="truste"]',
                            '[id*="cookie-banner"]','[class*="cookie-banner"]',
                            '[id*="consent-banner"]','[class*="consent-banner"]',
                            '[id*="gdpr"]','[class*="gdpr"]'
                        ];
                        sels.forEach(function(s) {
                            document.querySelectorAll(s).forEach(function(el) {
                                el.remove();
                            });
                        });
                        // Hide any remaining fixed/sticky overlay by computed style
                        document.querySelectorAll('body *').forEach(function(el) {
                            var cs = window.getComputedStyle(el);
                            var z = parseInt(cs.zIndex, 10);
                            if ((cs.position === 'fixed' || cs.position === 'sticky')
                                    && !isNaN(z) && z > 100) {
                                el.style.setProperty('display', 'none', 'important');
                            }
                        });
                        document.body.style.overflow = 'auto';
                    })();
                """)
            except Exception:
                pass
            page.pdf(
                path=str(out_path),
                format="A4",
                print_background=True,
                margin={"top": "1cm", "bottom": "1cm", "left": "1cm", "right": "1cm"},
            )
            return True
        except Exception as e:
            if attempt <= retries:
                print(f"    [RETRY {attempt}] {url} — {e}")
                time.sleep(2)
            else:
                print(f"    [ERROR] Could not save {url}: {e}")
                return False


# ── Main crawler ──────────────────────────────────────────────────────────────

def crawl(root_url: str, output_dir: str, max_depth: int, delay: float):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    parsed_root = urlparse(root_url)
    base_domain = parsed_root.netloc
    root_path   = parsed_root.path

    queue   = [(root_url, 0)]
    visited = set()
    counter = 0

    print(f"\n{'='*60}")
    print(f"Root URL   : {root_url}")
    print(f"Domain     : {base_domain}")
    print(f"Path prefix: {root_path}")
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

        # Block consent scripts — MutationObserver + route blocking
        setup_banner_suppression(context, page)
        print("  [BANNER] Consent suppression active (init_script + route blocking)\n")

        while queue:
            url, depth = queue.pop(0)
            norm = normalize_url(url)
            if norm in visited:
                continue
            visited.add(norm)

            counter += 1
            filename = url_to_filename(url, counter)
            out_path = out / filename

            print(f"[{counter:03d}] depth={depth}  {url}")
            print(f"      -> {filename}")

            ok = save_pdf(page, url, out_path, delay)
            if not ok:
                counter -= 1
                continue

            if depth < max_depth:
                links = extract_links(page, url, base_domain, root_path)
                new_links = [
                    (lnk, depth + 1)
                    for lnk in links
                    if normalize_url(lnk) not in visited
                ]
                queue = new_links + queue
                print(f"      Found {len(new_links)} new links to follow")

        browser.close()

    print(f"\n{'='*60}")
    print(f"Done. {counter} PDFs saved to: {out.resolve()}")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl a website and save each page as PDF into "
                    "<dataset_dir>/knowledge/. Run from DomainMiner root."
    )
    parser.add_argument(
        "--dataset_dir", required=True,
        help="Dataset subfolder name inside DomainMiner (e.g. Airportdb, Synthea)."
    )
    parser.add_argument(
        "--url", required=True,
        help="Root URL to start crawling from."
    )
    parser.add_argument("--depth",  type=int,   default=2,
                        help="Max crawl depth (default: 2).")
    parser.add_argument("--delay",  type=float, default=1.5,
                        help="Seconds between page loads (default: 1.5).")
    args = parser.parse_args()

    script_dir  = Path(__file__).resolve().parent
    dataset_dir = script_dir / args.dataset_dir
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}\n"
            f"Make sure you are running from the DomainMiner root and "
            f"--dataset_dir matches an existing subfolder."
        )

    output_dir = dataset_dir / "knowledge"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_dir}")

    crawl(root_url=args.url, output_dir=str(output_dir),
          max_depth=args.depth, delay=args.delay)