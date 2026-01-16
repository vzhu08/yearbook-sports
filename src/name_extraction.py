# src/section_name_extraction.py
"""
Pre/Post processing: section building + text/header correction + candidate generation,
then assign name tokens to sections via spaCy NER (src.name_ner) and identify sports.

Inputs (under <out_dir>/<pdf_stem>/):
  - compiled_ocr.json

Outputs (same folder):
  - sections_with_text.json
  - sections_with_names.json
  - sports_sections.json  (sports-only reduced view of sections_with_names)

Rules:
  1) A section spans from one header to the next header.
  2) Ignore headers containing "starter", "player", or "captain" (case-insensitive).
  3) Exclude blocks labeled: header, paragraph_title, footer, image.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.common.io_utils import read_json, write_json
from src.name_ner import extract_person_names_per_candidate, dedupe_person_names

# ---------------- CPU oversubscription guard ----------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ------------------------------- Config -------------------------------

IGNORE_HEADER_SUBSTRINGS = {"starter", "player", "captain"}
EXCLUDE_TEXT_LABELS = {"header", "paragraph_title", "footer", "image"}

# ---------------- Sports section detection ----------------
_SPORTS_SINGLE_WORD = {
    "athletics",
    "sports",
    "varsity",
    "jv",
    "baseball",
    "softball",
    "basketball",
    "football",
    "soccer",
    "lacrosse",
    "volleyball",
    "tennis",
    "golf",
    "swimming",
    "diving",
    "wrestling",
    "cheer",
    "gymnastics",
    "rowing",
    "crew",
    "badminton",
    "squash",
    "fencing",
    "bowling",
    "skiing",
    "rugby",
    "ultimate",
    "hockey",
    "track",
}
_SPORTS_MULTI_WORD = {
    "junior varsity",
    "cross country",
    "track and field",
    "field hockey",
    "water polo",
    "flag football",
}

# --------------------------- Helpers: blocks ---------------------------

def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _page_blocks(page: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield unified blocks from PPStructure 'parsing_res_list' or legacy 'items'."""
    if not isinstance(page, dict):
        return

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


# ------------------------- Text correction helpers -------------------------

_PREFIX_EXCEPTIONS = ("Mc", "Mac", "O")
_SPLIT_ON_PUNCT = re.compile(r"[.,]+")


def _looks_all_caps(s: str) -> bool:
    """True if the string has letters and none are lowercase (ignores digits/punct/spaces)."""
    if not s:
        return False
    if not re.search(r"[A-Za-z]", s):
        return False
    return not bool(re.search(r"[a-z]", s))


def _collapse_spaced_letters(s: str) -> str:
    """Collapse sequences like 'A T H L E T I C S' -> 'ATHLETICS'."""
    if not s:
        return s

    def repl(m: re.Match) -> str:
        return m.group(0).replace(" ", "")

    return re.sub(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b", repl, s)


def _split_joined_capitals_left_to_right(s: str) -> str:
    """
    Split words on internal capital letters from left to right.
    Do NOT split if the segment before the capital equals 'Mc', 'Mac', or 'O'.
    Works per-token while preserving whitespace and punctuation.
    """
    if not s:
        return s

    parts = re.split(r"(\s+)", s)
    out: List[str] = []

    for part in parts:
        if not part or part.isspace():
            out.append(part)
            continue

        tokens = re.split(r"([A-Za-z]+)", part)
        rebuilt: List[str] = []
        for tok in tokens:
            if not tok or not tok.isalpha():
                rebuilt.append(tok)
                continue

            if tok.isupper() or len(tok) < 6:
                rebuilt.append(tok)
                continue

            segments: List[str] = []
            start = 0
            i = 1
            while i < len(tok):
                if tok[i].isupper():
                    prefix = tok[start:i]
                    if prefix in _PREFIX_EXCEPTIONS:
                        i += 1
                        continue
                    segments.append(prefix)
                    start = i
                i += 1
            segments.append(tok[start:])
            rebuilt.append(" ".join(seg for seg in segments if seg))

        out.append("".join(rebuilt))

    return "".join(out)


def _correct_candidate_text_pre_ner(s: str) -> str:
    """
    Candidate correction:
      - If ALL CAPS: do nothing beyond punctuation splitting already done upstream
      - Else: split joined capitals with prefix exceptions
    """
    s = _normalize_space(s)
    if not s:
        return s
    if _looks_all_caps(s):
        return s
    return _split_joined_capitals_left_to_right(s)


def _correct_header_text(s: str) -> str:
    """
    Header correction:
      - normalize spaces
      - collapse spaced-letter headers ("A T H L E T I C S" -> "ATHLETICS")
      - if ALL CAPS: do not split joined capitals
      - else: split joined capitals with prefix exceptions
    """
    s = _normalize_space(s)
    s = _collapse_spaced_letters(s)
    if not s:
        return s
    if _looks_all_caps(s):
        return s
    return _split_joined_capitals_left_to_right(s)


# ------------------------- Collect headers/text ------------------------

def _collect_headers_with_pos(compiled_clean: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ordered header markers: {'page', 'y', 'text'}."""
    markers: List[Dict[str, Any]] = []

    for page_idx, page in enumerate(compiled_clean.get("pages", []), start=1):
        blocks = list(_page_blocks(page))
        blocks.sort(key=_y1)
        for blk in blocks:
            if _is_header(blk):
                raw = blk.get("block_content", "")
                txt = _correct_header_text(raw if isinstance(raw, str) else "")
                if txt and not _is_ignored_header_text(txt):
                    markers.append({"page": page_idx, "y": _y1(blk), "text": txt})

    markers.sort(key=lambda m: (m["page"], m["y"]))
    return markers


def _collect_text_blocks_by_page(compiled_clean: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """Collect textual blocks per page, excluding headers/titles/footers/images."""
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
    """Return (texts, pages_involved) for section from start_marker to next_marker."""
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
    """Build sections with header, page span, and collected texts."""
    markers = _collect_headers_with_pos(compiled_clean)
    texts_by_page = _collect_text_blocks_by_page(compiled_clean)

    sections: List[Dict[str, Any]] = []

    if not markers:
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
            pages = [start["page"]]
        sections.append({
            "header": start["text"],
            "start_page": pages[0] if pages else start["page"],
            "end_page": pages[-1] if pages else (nxt["page"] if nxt else start["page"]),
            "pages": pages if pages else [start["page"]],
            "texts": texts,
        })

    return sections


# --------------------------- Candidate generation ---------------------------

def _chunk_candidates(texts: List[str]) -> List[str]:
    """
    Step 1:
      - Split each text span on commas/periods into short candidate strings.
      - Then apply joined-capital splitting BEFORE NER (unless ALL CAPS).
    """
    cands: List[str] = []
    for t in texts:
        t = _normalize_space(t)
        if not t:
            continue
        for chunk in _SPLIT_ON_PUNCT.split(t):
            chunk = _normalize_space(chunk)
            if not chunk:
                continue
            cands.append(_correct_candidate_text_pre_ner(chunk))
    return cands


# --------------------------- Sports header matching ---------------------------

def _is_sports_section_header(header_text: str) -> bool:
    """
    Case-insensitive match of a section header against sports keywords.
    Now supports partial matches (substring) on normalized text.
    Example: "Track & Field" or "TRACKANDFIELD" matches keyword "track".
    """
    t = (header_text or "").strip().lower()
    if not t:
        return False

    norm = re.sub(r"[^a-z0-9\s-]+", " ", t)
    norm = re.sub(r"\s+", " ", norm).strip()

    for kw in _SPORTS_MULTI_WORD:
        if kw in norm:
            return True

    # partial match for single-word sports keys
    for kw in _SPORTS_SINGLE_WORD:
        if kw in norm:
            return True

    return False


# ------------------------------- Entry Point -------------------------------

def extract_names(pdf_path: str, out_dir: str, verbose_ner: bool = False) -> Dict[str, Any] | None:
    """
    Read compiled_ocr.json and emit:
      - sections_with_text.json
      - sections_with_names.json
      - sports_sections.json
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

    # Sports-only reduced view (decide sports sections BEFORE NER)
    sports_section_mask: List[bool] = [
        _is_sports_section_header(sec.get("header", "")) for sec in sections
    ]
    sports_section_indices: List[int] = [i for i, is_sports in enumerate(sports_section_mask) if is_sports]

    # Build candidates only for sports sections (single NER call)
    flat_candidates: List[str] = []
    flat_section_idx: List[int] = []
    candidates_per_sports_section: List[int] = []

    for i in sports_section_indices:
        sec = sections[i]
        cands = _chunk_candidates(sec.get("texts", []))
        candidates_per_sports_section.append(len(cands))
        for c in cands:
            flat_candidates.append(c)
            flat_section_idx.append(i)

    # Run NER once over sports candidates (or skip if none)
    section_names_raw: List[List[str]] = [[] for _ in sections]
    if flat_candidates:
        names_per_candidate = extract_person_names_per_candidate(flat_candidates, verbose=verbose_ner)

        # Group back to original section indices
        for idx, names_here in zip(flat_section_idx, names_per_candidate):
            section_names_raw[idx].extend(names_here)

    # Per-section dedupe (nested-name removal, etc.)
    sections_names: List[Dict[str, Any]] = []
    for sec, raw_names in zip(sections, section_names_raw):
        names = dedupe_person_names(raw_names) if raw_names else []
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

    sports_only = [sections_names[i] for i in sports_section_indices]
    sports_sections_path = book_dir / "sports_sections.json"
    write_json({"sections": sports_only}, sports_sections_path)
    print(f"[names] wrote {sports_sections_path.name}  sports_sections={len(sports_only)}")

    return {
        "sections_with_text": str(sections_text_path),
        "sections_with_names": str(sections_names_path),
        "sports_sections": str(sports_sections_path),
        "sections_count": len(sections),
        "sports_sections_count": len(sports_only),
        "total_candidates": len(flat_candidates),
        "candidates_per_sports_section": candidates_per_sports_section,
    }

