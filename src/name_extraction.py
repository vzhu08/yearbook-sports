# src/section_name_extraction.py
"""
Section mapping and name extraction from compiled_ocr.json using spaCy PERSON NER.

Inputs (under <out_dir>/<pdf_stem>/):
  - compiled_ocr.json

Outputs (same folder):
  - sections_with_text.json       # sections between headers with page spans + raw texts
  - sections_with_names.json      # same spans, but with deduped person names instead of texts

Rules:
  1) A section spans from one header to the next header.
  2) Ignore headers that contain "starter", "player", or "captain" (case-insensitive).
     Ignored headers neither start nor end sections.
  3) Text collection excludes blocks labeled: header, paragraph_title, footer, image.

Robust to legacy formats:
  - Supports PPStructure 'parsing_res_list' and legacy 'items' layouts.
  - Page numbers are inferred by list order, 1-indexed.

Public entry:
  extract_names(pdf_path: str, out_dir: str) -> dict | None
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from collections import OrderedDict

from src.common.io_utils import read_json, write_json


# ------------------------------- Config -------------------------------

IGNORE_HEADER_SUBSTRINGS = {"starter", "player", "captain"}   # case-insensitive
EXCLUDE_TEXT_LABELS = {"header", "paragraph_title", "footer", "image"}  # labels not treated as body text

# Domain stopwords to filter false-positive "names"
NAME_STOPWORDS = {
    # roles and grades
    "Varsity", "Junior", "JV", "Freshman", "Sophomore", "Senior",
    "Captain", "Captains", "Assistant", "Coach", "Manager", "Starter", "Player", "Bench",
    "Team", "Roster", "Schedule", "Record", "Season",
    # sports
    "Basketball", "Football", "Soccer", "Baseball", "Softball", "Lacrosse", "Hockey",
    "Tennis", "Golf", "Swim", "Swimming", "Diving", "Track", "Field", "Cross", "Country",
    "Volleyball", "Wrestling", "Cheer", "Cheerleading",
    # time and common non-names
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
    # school words
    "High", "School", "Yearbook", "Club", "Clubs", "Honor", "Honors",
    # punctuation placeholders
    "—", "-", "•",
}

SUFFIXES = {"Jr", "Jr.", "Sr", "Sr.", "II", "III", "IV", "V"}


# --------------------------- spaCy Loader ---------------------------

_NLP = None
_SPACY_MODEL_CANDIDATES = [
    "en_core_web_trf",
    "en_core_web_lg",
    "en_core_web_md",
    "en_core_web_sm",
]


def _get_spacy_nlp():
    """
    Lazy-load a spaCy English pipeline. Requires a model to be installed.
    """
    global _NLP
    if _NLP is not None:
        return _NLP

    try:
        import spacy  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "spaCy is required for name extraction. Install with: pip install spacy && python -m spacy download en_core_web_sm"
        ) from e

    last_err: Optional[Exception] = None
    for model in _SPACY_MODEL_CANDIDATES:
        try:
            _NLP = spacy.load(model)
            return _NLP
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        "No spaCy English model found. Install one, e.g.: python -m spacy download en_core_web_sm"
    ) from last_err


# --------------------------- Helpers: blocks ---------------------------

def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _page_blocks(page: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Yield blocks with unified keys: block_label, block_content, block_bbox.
    Supports PPStructure 'parsing_res_list' and legacy 'items'.
    """
    if not isinstance(page, dict):
        return

    # PPStructure
    prl = page.get("parsing_res_list")
    if isinstance(prl, list) and prl:
        for blk in prl:
            if not isinstance(blk, dict):
                continue
            yield {
                "block_label": blk.get("block_label"),
                "block_content": blk.get("block_content", "") if isinstance(blk.get("block_content"), str) else "",
                "block_bbox": blk.get("block_bbox") if isinstance(blk.get("block_bbox"), (list, tuple)) else None,
            }
        return

    # Legacy
    items = page.get("items")
    if isinstance(items, list) and items:
        for it in items:
            if not isinstance(it, dict):
                continue
            yield {
                "block_label": "text",
                "block_content": it.get("text", "") if isinstance(it.get("text"), str) else "",
                "block_bbox": it.get("bbox") if isinstance(it.get("bbox"), (list, tuple)) else None,
            }


def _y1(block: Dict[str, Any]) -> float:
    bbox = block.get("block_bbox")
    try:
        return float(bbox[1]) if bbox else -1.0
    except Exception:
        return -1.0


def _is_header(block: Dict[str, Any]) -> bool:
    lbl = str(block.get("block_label", "")).lower()
    return lbl in {"header", "paragraph_title"}


def _is_ignored_header_text(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(key in t for key in IGNORE_HEADER_SUBSTRINGS)


# ------------------------- Collect headers/text ------------------------

def _collect_headers_with_pos(compiled_clean: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ordered header markers: {"page": int, "y": float, "text": str}
    Ignores headers with undesired substrings.
    """
    markers: List[Dict[str, Any]] = []

    for page_idx, page in enumerate(compiled_clean.get("pages", []), start=1):
        blocks = list(_page_blocks(page))
        blocks.sort(key=_y1)  # top-to-bottom
        for blk in blocks:
            if _is_header(blk):
                txt = _normalize_space(blk.get("block_content", ""))
                if txt and not _is_ignored_header_text(txt):
                    markers.append({"page": page_idx, "y": _y1(blk), "text": txt})

    markers.sort(key=lambda m: (m["page"], m["y"]))
    return markers


def _collect_text_blocks_by_page(compiled_clean: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """
    For each page, collect textual blocks to include in sections.
    Excludes labels in EXCLUDE_TEXT_LABELS. Keeps string contents only.
    Returns: page_num -> [ { "text": str, "y": float } ... ]
    """
    out: Dict[int, List[Dict[str, Any]]] = {}

    for page_idx, page in enumerate(compiled_clean.get("pages", []), start=1):
        lst: List[Dict[str, Any]] = []
        for blk in _page_blocks(page):
            label = str(blk.get("block_label", "")).lower()
            if label in EXCLUDE_TEXT_LABELS:
                continue
            text = _normalize_space(blk.get("block_content", ""))
            if not text:
                continue
            lst.append({"text": text, "y": _y1(blk)})
        lst.sort(key=lambda r: r["y"])
        out[page_idx] = lst

    return out


# --------------------------- Section slicing ---------------------------

def _slice_section_texts(
    texts_by_page: Dict[int, List[Dict[str, Any]]],
    start_marker: Dict[str, Any],
    next_marker: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[int]]:
    """
    Return (texts, pages_involved) for section from start_marker up to (not including) next_marker.
    - Start page: include y >= start_y
    - Middle pages: include all
    - End page: include y < end_y
    """
    start_p, start_y = start_marker["page"], start_marker["y"]
    end_p: Optional[int] = next_marker["page"] if next_marker else None
    end_y: Optional[float] = next_marker["y"] if next_marker else None

    texts: List[str] = []
    pages_involved: Set[int] = set()

    if end_p is None:
        for p in sorted(texts_by_page.keys()):
            if p < start_p:
                continue
            if p == start_p:
                for r in texts_by_page.get(p, []):
                    if r["y"] >= start_y:
                        texts.append(r["text"])
                        pages_involved.add(p)
            else:
                for r in texts_by_page.get(p, []):
                    texts.append(r["text"])
                    pages_involved.add(p)
        return texts, sorted(pages_involved)

    for p in sorted(texts_by_page.keys()):
        if p < start_p or p > end_p:
            continue
        if p == start_p and p == end_p:
            for r in texts_by_page.get(p, []):
                if r["y"] >= start_y and r["y"] < (end_y or float("inf")):
                    texts.append(r["text"])
                    pages_involved.add(p)
        elif p == start_p:
            for r in texts_by_page.get(p, []):
                if r["y"] >= start_y:
                    texts.append(r["text"])
                    pages_involved.add(p)
        elif p == end_p:
            for r in texts_by_page.get(p, []):
                if end_y is None or r["y"] < end_y:
                    texts.append(r["text"])
                    pages_involved.add(p)
        else:
            for r in texts_by_page.get(p, []):
                texts.append(r["text"])
                pages_involved.add(p)

    return texts, sorted(pages_involved)


def _build_sections(compiled_clean: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build sections list with header text, page span, and collected texts.
    """
    markers = _collect_headers_with_pos(compiled_clean)
    texts_by_page = _collect_text_blocks_by_page(compiled_clean)

    sections: List[Dict[str, Any]] = []

    if not markers:
        # Single section for the whole document if no headers
        all_texts: List[str] = []
        pages_involved: Set[int] = set()
        for p, rows in texts_by_page.items():
            pages_involved.add(p)
            for r in rows:
                all_texts.append(r["text"])
        if all_texts:
            pages_sorted = sorted(pages_involved)
            sections.append({
                "header": "(no headers found)",
                "start_page": pages_sorted[0] if pages_sorted else 1,
                "end_page": pages_sorted[-1] if pages_sorted else 1,
                "pages": pages_sorted if pages_sorted else [1],
                "texts": all_texts,
            })
        return sections

    for i, start in enumerate(markers):
        nxt = markers[i + 1] if i + 1 < len(markers) else None
        texts, pages = _slice_section_texts(texts_by_page, start, nxt)
        if not texts and not pages:
            pages = [start["page"]]  # keep traceable even if empty
        sections.append({
            "header": start["text"],
            "start_page": pages[0] if pages else start["page"],
            "end_page": pages[-1] if pages else (nxt["page"] if nxt else start["page"]),
            "pages": pages if pages else [start["page"]],
            "texts": texts,
        })

    return sections


# --------------------------- Name utilities ---------------------------

def _is_bad_token(tok: str) -> bool:
    if not tok or len(tok) < 2:
        return True
    if any(ch.isdigit() for ch in tok):
        return True
    if tok in SUFFIXES:
        return False
    if tok in NAME_STOPWORDS:
        return True
    return False


def _post_filter_name_str(name: str) -> Optional[str]:
    """
    Token-level sanity checks and domain stopword filtering.
    Allows suffixes. Drops names containing obvious non-person terms.
    """
    name = _normalize_space(name).strip(".,;:()[]{}")
    if not name:
        return None

    # Title-case if the span is ALL CAPS to normalize OCR variants.
    if name.isupper():
        name = name.title()

    parts = name.split()
    clean_parts: List[str] = []
    for p in parts:
        if _is_bad_token(p):
            return None
        clean_parts.append(p)

    full = " ".join(clean_parts)
    for sw in NAME_STOPWORDS:
        if re.search(rf"\b{re.escape(sw)}\b", full):
            return None
    return full


def _extract_names_spacy(texts: List[str]) -> List[str]:
    """
    Run spaCy NER over a list of lines. Collect PERSON entities.
    """
    if not texts:
        return []

    nlp = _get_spacy_nlp()

    found: List[str] = []
    # Use pipe for speed and memory efficiency.
    for doc in nlp.pipe(texts):
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                cand = _post_filter_name_str(ent.text)
                if cand:
                    found.append(cand)

    # Deduplicate then sort for stability.
    uniq_sorted = sorted(set(found))
    return uniq_sorted


def _names_from_section_texts(texts: List[str]) -> List[str]:
    return _extract_names_spacy(texts)


# ------------------------------- Entry Point -------------------------------

def extract_names(pdf_path: str, out_dir: str) -> Dict[str, Any] | None:
    """
    Read compiled_ocr.json and emit:
      - sections_with_text.json
      - sections_with_names.json

    Args:
      pdf_path: path to the original PDF (used to locate <out_dir>/<pdf_stem>/)
      out_dir: root output directory used in earlier steps

    Returns:
      dict with file paths and counts, or None if inputs are missing.
    """
    book_dir = Path(out_dir) / Path(pdf_path).stem
    compiled_clean_path = book_dir / "compiled_ocr.json"

    if not compiled_clean_path.exists():
        print(f"[names] missing {compiled_clean_path}")
        return None

    compiled_clean = read_json(compiled_clean_path)

    # Build sections with texts
    sections = _build_sections(compiled_clean)
    sections_text_path = book_dir / "sections_with_text.json"
    write_json({"sections": sections}, sections_text_path)
    print(f"[names] wrote {sections_text_path.name}  sections={len(sections)}")

    # Build sections with names via spaCy PERSON NER
    sections_names: List[Dict[str, Any]] = []
    for sec in sections:
        names = _names_from_section_texts(sec.get("texts", []))
        sections_names.append({
            "header": sec["header"],
            "start_page": sec["start_page"],
            "end_page": sec["end_page"],
            "pages": sec["pages"],
            "names": names,
        })

    sections_names_path = book_dir / "sections_with_names.json"
    write_json({"sections": sections_names}, sections_names_path)
    print(f"[names] wrote {sections_names_path.name}")

    return {
        "sections_with_text": str(sections_text_path),
        "sections_with_names": str(sections_names_path),
        "sections_count": len(sections),
    }
