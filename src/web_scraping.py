#!/usr/bin/env python3
"""
DigitalNC Yearbook PDF Downloader - Download all yearbooks from a school
Usage:
  Single yearbook: python scraper.py https://lib.digitalnc.org/record/38440
  All yearbooks: python scraper.py https://lib.digitalnc.org/search?ln=en&p=691:%22A.+L.+Brown+High+School%22
"""

from bs4 import BeautifulSoup
import requests
from typing import List
import time
import os
from urllib.parse import urljoin, urlparse


# Step 1 - get the "View All" url for all high schools
def get_school_urls(url: str) -> List[str]:
    '''
    this function will collect the "View All" url for all high schools
    '''
    school_view_all_urls = []
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to load page: Status code {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")
    tags = soup.select("a.btn.btn-sm.btn-outline-azure")
    for tag in tags:
        href = tag.get("href")
        if href:
            school_view_all_urls.append(href)
    return school_view_all_urls

# from each school we get the url for yearbook for each year
def get_yearbook_url(url: str) -> List[str]:
    if url is None:
        raise Exception("URL cannot be None")
    years_url = []
    curr_url = url

    while curr_url:
        response = requests.get(curr_url)
        if response.status_code != 200:
            raise Exception(f"Failed to load page: Status code {response.status_code}")

        soup = BeautifulSoup(response.text, "html.parser")
        # find the link that meets (1) inside <div> and has the class "result-title" (2) the href attributes starts with /record/
        for link in soup.select('div.result-title a[href^="/record/"]'):
            href = link.get("href")
            if href:
                full_link = urljoin(curr_url, href)
                years_url.append(full_link)
        # need to deal with multipage issue, so check if there's a next page button, if so update the url and then repeat the process
        next_page = None
        img_next_page = soup.find("img", alt="next")
        if img_next_page and img_next_page.parent.name == "a":
            href = img_next_page.parent.get("href")
            if href:
                next_page = urljoin(url, href)

        if not next_page:
            break

        curr_url = next_page
    print(years_url)
    return years_url


# grad the download PDF link from each record page
def get_pdf_link(record_url: str) -> str | None:
    if record_url is None:
        raise Exception("URL cannot be None")
    response = requests.get(record_url)
    if response.status_code != 200:
        raise Exception(f"Failed to load page: Status code {response.status_code}")
    soup = BeautifulSoup(response.text, "html.parser")
    for elem in soup.select("div.metadata-row"):
        title = elem.find("div", class_="title")
        if not title or title.get_text(strip=True) != "Linked Resources":
            continue
        value = elem.find("div", class_="value")
        if not value:
            raise Exception(f"Failed to extract value: {title.getText.strip()}")
        link = value.find("a").get("href")
        if not link:
            raise Exception(f"Failed to extract link: {title.getText.strip()}")
        full_link = urljoin(record_url, link)
        print(full_link)
        return full_link
    return None

def download_yearbook(pdf_url: str):
    # create a directory to store the downloaded pdf
    os.makedirs("yearbook_downloads", exist_ok=True)

    # keep the original file name
    filename = os.path.basename(urlparse(pdf_url).path)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    out_path = os.path.join("yearbook_downloads", filename)

    r = requests.get(pdf_url, timeout=60, stream=True)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("file saved successfully")

def main():
    url ="https://www.digitalnc.org/collections/yearbooks/"

    school_urls = get_school_urls(url)
    seen = set()
    failed = []

    for school_url in school_urls:
        record_urls = get_yearbook_url(school_url)
        for record_url in record_urls:
            try:
                pdf_url = get_pdf_link(record_url)
            except requests.RequestException as e:
                failed.append((record_url, f'Record page error: {e}'))
                continue
            if not pdf_url or pdf_url in seen:
                continue
            seen.add(pdf_url)
            try:
                download_yearbook(pdf_url)
            except requests.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else "?"
                failed.append((pdf_url, f'HTTP {status_code}'))
            except requests.RequestException as e:
                failed.append((pdf_url, f'Request Error: {e}'))
            time.sleep(0.5)
    print("\n✓ Done!")

    # For pdfs that we failed to download
    if failed:
        print("These items failed to download:")
        for url, reason in failed:
            print(f'{url}: {reason}')

if __name__ == "__main__":
    main()
