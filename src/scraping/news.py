"""Scrape comments from Tunisian news sites.

STUB — fill in per-site selectors in config/config.yaml (scraping.news.sites).
Static pages: requests + BeautifulSoup. JS-heavy pages: set requires_js=true to use
Playwright (install with `playwright install chromium`).

ALWAYS check the site's robots.txt and Terms of Service, rate-limit, and store no PII.

Run:  python -m src.scraping.news
"""
from __future__ import annotations

import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from ..config import RAW, ensure_data_dirs, settings
from .storage import make_comment, write_jsonl

OUT = RAW / "news.jsonl"
UA = "text-classifier-research/0.1 (+contact in README)"


def robots_allows(url: str) -> bool:
    parsed = urlparse(url)
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
    try:
        rp.read()
    except Exception:
        return False  # fail closed: if we can't read robots.txt, don't scrape
    return rp.can_fetch(UA, url)


def fetch_static(url: str) -> str:
    resp = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    resp.raise_for_status()
    return resp.text


def fetch_js(url: str) -> str:
    """Render a JS-heavy page with Playwright. Lazy import so it's optional."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=UA)
        page.goto(url, wait_until="networkidle", timeout=30000)
        html = page.content()
        browser.close()
    return html


def scrape(out: Path = OUT) -> int:
    ensure_data_dirs()
    cfg = settings()["scraping"]
    sites = cfg["news"].get("sites") or []
    sleep = cfg["request_sleep_seconds"]
    if not sites:
        print("No news sites configured (scraping.news.sites). Nothing to do.")
        return 0

    total = 0
    for site in sites:
        url = site["url"]
        if not robots_allows(url):
            print(f"  robots.txt disallows {url} for our UA — skipping.")
            continue
        html = fetch_js(url) if site.get("requires_js") else fetch_static(url)
        soup = BeautifulSoup(html, "lxml")
        rows = []
        for i, el in enumerate(soup.select(site["comment_selector"])):
            text = el.get_text(" ", strip=True)
            if text:
                rows.append(make_comment(
                    id=f"nw_{site['name']}_{i}",
                    platform="news", text=text, source=site["name"], timestamp=None,
                ))
        total += write_jsonl(out, rows)
        time.sleep(sleep)
    print(f"Wrote {total} new comments -> {out}")
    return total


if __name__ == "__main__":
    scrape()
