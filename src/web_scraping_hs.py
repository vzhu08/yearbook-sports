#!/usr/bin/env python3
"""
DigitalNC Yearbook PDF Downloader — HIGH SCHOOLS ONLY.

Same job as web_scraping.py, but walks the yearbooks collection page and
keeps only the schools whose DigitalNC badge reads "High School" (the
authoritative classification), so colleges/universities/community colleges
are never downloaded.

Built for long unattended runs (HPC):
  - lib.digitalnc.org answers request bursts with empty HTTP 202 — every
    request backs off and retries instead of dying;
  - fully resumable: already-downloaded PDFs are skipped, and schools whose
    records were fully enumerated are cached in hs_scrape_state.json, so a
    killed job continues where it stopped on the next run.

Usage:
    python src/web_scraping_hs.py [download_dir]     (default: yearbook_downloads)
"""

import json
import os
import sys
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

COLLECTIONS_URL = "https://www.digitalnc.org/collections/yearbooks/"
STATE = "hs_scrape_state.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

session = requests.Session()
session.headers.update(HEADERS)


def get_with_backoff(url: str, tries: int = 6) -> requests.Response:
    """lib.digitalnc.org answers bursts with empty HTTP 202 — back off and retry."""
    delay = 5
    for _ in range(tries):
        r = session.get(url, timeout=30)
        if r.status_code == 200 and r.text:
            return r
        time.sleep(delay)
        delay = min(delay * 2, 120)
    raise RuntimeError(f"still throttled after {tries} tries: {url}")


def high_school_rows():
    """Yield (school_name, view_all_url) for every row badged "High School".

    One row can name several schools joined by ';' under a single View All
    link — that's fine here, the row is just the crawl unit."""
    soup = BeautifulSoup(get_with_backoff(COLLECTIONS_URL).text, "html.parser")
    for row in soup.select("div.dnc-datagrid-row"):
        badge = row.select_one("span.badge.bg-azure.text-white")
        if not badge or badge.get_text(strip=True) != "High School":
            continue
        view_all = row.select_one("a.btn.btn-sm.btn-outline-azure[href]")
        if not view_all:
            continue
        name_el = row.select_one(".dnc-datagrid-title, h3, strong, a")
        name = name_el.get_text(strip=True) if name_el else "?"
        yield name, urljoin(COLLECTIONS_URL, view_all["href"])


def record_urls(school_url: str):
    """All /record/ URLs for a school, following result pagination."""
    curr = school_url
    while curr:
        soup = BeautifulSoup(get_with_backoff(curr).text, "html.parser")
        for link in soup.select('div.result-title a[href^="/record/"]'):
            yield urljoin(curr, link["href"])
        nxt = soup.find("img", alt="next")
        curr = (
            urljoin(curr, nxt.parent.get("href"))
            if nxt and nxt.parent.name == "a" and nxt.parent.get("href")
            else None
        )


def get_pdf_link(record_url: str):
    soup = BeautifulSoup(get_with_backoff(record_url).text, "html.parser")
    for elem in soup.select("div.metadata-row"):
        title = elem.find("div", class_="title")
        if not title or title.get_text(strip=True) != "Linked Resources":
            continue
        value = elem.find("div", class_="value")
        a = value.find("a") if value else None
        if a and a.get("href"):
            return urljoin(record_url, a["href"])
    return None


def download_yearbook(pdf_url: str, out_dir: str) -> bool:
    """Download one PDF; returns False if it was already on disk."""
    filename = os.path.basename(urlparse(pdf_url).path)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return False

    r = session.get(pdf_url, timeout=120, stream=True)
    r.raise_for_status()
    tmp_path = out_path + ".part"  # never leave a truncated .pdf behind
    with open(tmp_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    os.replace(tmp_path, out_path)
    return True


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "yearbook_downloads"
    os.makedirs(out_dir, exist_ok=True)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {"done_schools": []}
    failed = []

    for name, school_url in high_school_rows():
        if name in state["done_schools"]:
            continue
        print(f"== {name}", flush=True)
        new = skipped = 0
        fails_before = len(failed)
        try:
            for record_url in record_urls(school_url):
                try:
                    pdf_url = get_pdf_link(record_url)
                except (requests.RequestException, RuntimeError) as e:
                    failed.append((record_url, f"record page error: {e}"))
                    continue
                if not pdf_url:
                    continue
                try:
                    if download_yearbook(pdf_url, out_dir):
                        new += 1
                        print(f"   + {os.path.basename(urlparse(pdf_url).path)}", flush=True)
                    else:
                        skipped += 1
                except requests.HTTPError as e:
                    code = e.response.status_code if e.response is not None else "?"
                    failed.append((pdf_url, f"HTTP {code}"))
                except requests.RequestException as e:
                    failed.append((pdf_url, f"request error: {e}"))
                time.sleep(0.5)
        except (requests.RequestException, RuntimeError) as e:
            # listing this school failed mid-way: don't mark done, keep going
            failed.append((school_url, f"school listing error: {e}"))
            print(f"   ! listing failed, will retry next run: {e}", flush=True)
            continue
        if len(failed) == fails_before:
            # only a fully clean school is skipped on the next run — a school
            # with failures gets re-listed so the misses are retried (already
            # downloaded files are skipped by filename, so retries are cheap)
            state["done_schools"].append(name)
            json.dump(state, open(STATE, "w"), indent=1)
        print(f"   {new} downloaded, {skipped} already on disk, "
              f"{len(failed) - fails_before} failed", flush=True)

    print("\n✓ Done!")
    if failed:
        print("These items failed — rerun the script to retry them:")
        for url, reason in failed:
            print(f"{url}: {reason}")


if __name__ == "__main__":
    main()
