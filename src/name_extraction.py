# src/name_extraction.py

"""
Step 3 — Name Extraction via headers.

Consumes:
  - compiled_ocr.json
  - corrected_names.json

Produces:
  - headers.json
  - headers_with_names.json
  - roster.csv
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from src.common.io_utils import read_json, write_json


# -------------------------- Internal --------------------------

def _detect_headers(compiled: Dict[str, Any], header_thresh: float) -> List[Dict[str, Any]]:
    heights: List[float] = []
    for page in compiled.get("pages", []):
        for it in page.get("items", []):
            h = it.get("height", None)
            if h:
                heights.append(float(h))
    if not heights:
        return []

    med = statistics.median(heights)
    cutoff = med * header_thresh

    headers: List[Dict[str, Any]] = []
    for pg_idx, page in enumerate(compiled.get("pages", [])):
        for it in page.get("items", []):
            h = float(it.get("height", 0.0))
            if h >= cutoff:
                headers.append(
                    {
                        "page": pg_idx,
                        "text": it.get("text", ""),
                        "bbox": it.get("bbox", None),
                        "height": h,
                    }
                )
    # Attach diagnostics for printing
    headers._median = med  # type: ignore[attr-defined]
    headers._cutoff = cutoff  # type: ignore[attr-defined]
    return headers


def _associate_names(headers: List[Dict[str, Any]], corrected_names: List[Dict[str, str]], compiled: Dict[str, Any]) -> List[Dict[str, Any]]:
    def _y_of(it: Dict[str, Any]) -> float:
        bbox = it.get("bbox", None)
        return float(bbox[1]) if bbox else 0.0

    by_page_headers: Dict[int, List[Dict[str, Any]]] = {}
    for h in headers:
        by_page_headers.setdefault(h["page"], []).append(h)
    for lst in by_page_headers.values():
        lst.sort(key=_y_of)

    roster: List[Dict[str, Any]] = []
    pages = compiled.get("pages", [])

    assigned = 0
    unassigned = 0

    for c in corrected_names:
        name = c["corrected"]
        best_page = None
        best_y = None

        for pg_idx, page in enumerate(pages):
            found = False
            for it in page.get("items", []):
                if name and name.lower() in it.get("text", "").lower():
                    y = _y_of(it)
                    best_page, best_y = pg_idx, y
                    found = True
                    break
            if found:
                break

        if best_page is None:
            roster.append({"page": None, "header": None, "name": name})
            unassigned += 1
            continue

        candidates = by_page_headers.get(best_page, [])
        candidates = [h for h in candidates if _y_of(h) <= float(best_y)]
        candidates.sort(key=lambda h: float(best_y) - _y_of(h))

        header_text = candidates[0]["text"] if candidates else None
        roster.append({"page": best_page, "header": header_text, "name": name})
        assigned += 1

    # Attach diagnostics for printing
    roster._assigned = assigned  # type: ignore[attr-defined]
    roster._unassigned = unassigned  # type: ignore[attr-defined]
    return roster


# -------------------------- Public API --------------------------

def extract_names(
    pdf_path: str,
    out_dir: str,
    header_thresh: float = 1.8,
    max_gap: float = 0.25,  # reserved for future geometric gating
) -> None:
    """
    Integrated run function for Step 3. Prints progress to console.
    """
    t0 = time.time()
    book_dir = Path(out_dir) / Path(pdf_path).stem

    compiled_path = book_dir / "compiled_ocr.json"
    corrected_path = book_dir / "corrected_names.json"
    if not compiled_path.exists() or not corrected_path.exists():
        print(f"[names] skip: {book_dir.name} missing compiled_ocr.json or corrected_names.json")
        return

    compiled = read_json(compiled_path)
    corrected = read_json(corrected_path).get("corrected", [])

    print(f"[names] start: {book_dir.name}, header_thresh={header_thresh}, corrected_names={len(corrected)}")

    t_hdr = time.time()
    headers = _detect_headers(compiled, header_thresh=header_thresh)
    median_h = getattr(headers, "_median", None)  # type: ignore[attr-defined]
    cutoff_h = getattr(headers, "_cutoff", None)  # type: ignore[attr-defined]
    print(f"[names] headers: detected={len(headers)}, median_height={median_h}, cutoff={cutoff_h} time={time.time() - t_hdr:.2f}s")

    t_assoc = time.time()
    roster = _associate_names(headers, corrected, compiled)
    assigned = getattr(roster, "_assigned", 0)    # type: ignore[attr-defined]
    unassigned = getattr(roster, "_unassigned", 0)  # type: ignore[attr-defined]
    print(f"[names] associate: assigned={assigned}, unassigned={unassigned} time={time.time() - t_assoc:.2f}s")

    write_json({"headers": headers}, book_dir / "headers.json")
    write_json({"roster": roster}, book_dir / "headers_with_names.json")

    # CSV
    import csv
    with open(book_dir / "roster.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page", "header", "name"])
        for row in roster:
            w.writerow([row.get("page"), row.get("header"), row.get("name")])

    print(f"[names] save: headers.json, headers_with_names.json, roster.csv")
    print(f"[names] done: {book_dir.name} total_time={time.time() - t0:.2f}s")
