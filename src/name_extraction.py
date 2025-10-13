# src/section_name_extraction.py
"""
Section mapping and name extraction from compiled_ocr.json using spaCy PERSON NER.

Pipeline for names per your spec:
  1) From each section's text blocks, split on commas/periods to get short candidate strings.
  2) Run spaCy NER on candidates (CPU-parallel).
  3) For each PERSON span, split joined capitals left->right (JakeSmith -> Jake Smith),
     but DO NOT split if the first chunk would be 'Mc', 'Mac', or 'O' (prefix exceptions).
  4) Normalize and dedupe.

Inputs (under <out_dir>/<pdf_stem>/):
  - compiled_ocr.json

Outputs (same folder):
  - sections_with_text.json
  - sections_with_names.json

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

# ---------------- CPU oversubscription guard ----------------
# Prevent BLAS thread explosion when using spaCy with n_process>1
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ------------------------------- Config -------------------------------

IGNORE_HEADER_SUBSTRINGS = {"starter", "player", "captain"}
EXCLUDE_TEXT_LABELS = {"header", "paragraph_title", "footer", "image"}

NAME_STOPWORDS: Set[str] = set()
SUFFIXES = {"Jr", "Jr.", "Sr", "Sr.", "II", "III", "IV", "V"}

# NER parallel knobs (CPU only)
NER_CPU_WORKERS = max((os.cpu_count() or 2) - 1, 1)
NER_BATCH_CPU = 256
NER_BATCH_GPU = 1024  # kept for completeness; we keep n_process=1 on GPU

# --------------------------- spaCy Loader ---------------------------

_NLP = None
_SPACY_MODEL_CANDIDATES = [
    "en_core_web_trf",
    "en_core_web_lg",
    "en_core_web_md",
    "en_core_web_sm",
]


def _get_spacy_nlp():
    """Lazy-load a spaCy English pipeline."""
    global _NLP
    if _NLP is not None:
        return _NLP

    try:
        import spacy  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "spaCy is required. Install: pip install spacy && python -m spacy download en_core_web_sm"
        ) from e

    last_err: Optional[Exception] = None
    for model in _SPACY_MODEL_CANDIDATES:
        try:
            _NLP = spacy.load(model)
            print("Using model:", model)
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


# ------------------------- Collect headers/text ------------------------

def _collect_headers_with_pos(compiled_clean: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ordered header markers: {'page', 'y', 'text'}."""
    markers: List[Dict[str, Any]] = []

    for page_idx, page in enumerate(compiled_clean.get("pages", []), start=1):
        blocks = list(_page_blocks(page))
        blocks.sort(key=_y1)
        for blk in blocks:
            if _is_header(blk):
                txt = _normalize_space(blk.get("block_content", ""))
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


# --------------------------- Name utilities ---------------------------

_SPLIT_ON_PUNCT = re.compile(r"[.,]+")  # split candidates on commas/periods


def _chunk_candidates(texts: List[str]) -> List[str]:
    """
    Split each text span on commas/periods into short candidate strings.
    Do NOT pre-split joined capitals here; we apply that AFTER NER per spec.
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
            cands.append(chunk)
    return cands


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
    """Normalize casing, remove punctuation, and reject obvious non-names."""
    name = _normalize_space(name).strip(".,;:()[]{}")
    if not name:
        return None
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


# ---- Post-NER joined-capitals splitter ----

_PREFIX_EXCEPTIONS = ("Mc", "Mac", "O")


def _split_joined_capitals_left_to_right(s: str) -> str:
    """
    Split words on internal capital letters from left to right.
    Do NOT split if the segment before the capital equals 'Mc', 'Mac', or 'O'.
    Works per-token while preserving whitespace and punctuation.
    """
    if not s:
        return s

    # Preserve whitespace tokens
    parts = re.split(r"(\s+)", s)
    out: List[str] = []

    for part in parts:
        if not part or part.isspace():
            out.append(part)
            continue

        # Process contiguous alphabetic chunks; leave non-alpha as-is
        tokens = re.split(r"([A-Za-z]+)", part)
        rebuilt: List[str] = []
        for tok in tokens:
            if not tok or not tok.isalpha():
                rebuilt.append(tok)
                continue

            # Skip all-caps tokens (e.g., II, III, JR) and short ones
            if tok.isupper() or len(tok) < 6:
                rebuilt.append(tok)
                continue

            # Left-to-right scan
            segments: List[str] = []
            start = 0
            i = 1
            while i < len(tok):
                ch = tok[i]
                if ch.isupper():
                    prefix = tok[start:i]
                    # Apply exception only if the split would directly follow the exception prefix
                    if prefix in _PREFIX_EXCEPTIONS:
                        # Do not split here; continue scanning
                        i += 1
                        continue
                    # Split here
                    segments.append(prefix)
                    start = i
                i += 1
            segments.append(tok[start:])

            # Join segments with spaces
            rebuilt.append(" ".join(seg for seg in segments if seg))
        out.append("".join(rebuilt))

    return "".join(out)


def _extract_names_spacy_from_candidates(candidates: List[str]) -> List[str]:
    """
    Run spaCy NER over candidates with CPU parallelism and post-NER splitting.
    - CPU: n_process = NER_CPU_WORKERS, batch_size = NER_BATCH_CPU
    - GPU+transformer: n_process = 1, batch_size = NER_BATCH_GPU
    After extracting PERSON spans, apply joined-capital splitting and normalize.
    """
    if not candidates:
        return []

    nlp = _get_spacy_nlp()

    # Keep only transformer+ner if present, to reduce overhead
    keep = {"ner", "transformer"} & set(nlp.pipe_names)
    disable = [p for p in nlp.pipe_names if p not in keep]

    # Decide CPU vs GPU
    is_gpu = False
    try:
        import torch  # type: ignore
        is_gpu = torch.cuda.is_available() and ("transformer" in nlp.pipe_names)
    except Exception:
        is_gpu = False

    n_proc = 1 if is_gpu else NER_CPU_WORKERS
    bsz = NER_BATCH_GPU if is_gpu else NER_BATCH_CPU

    found: List[str] = []
    with nlp.select_pipes(disable=disable):
        for doc in nlp.pipe(candidates, batch_size=bsz, n_process=n_proc):
            for ent in doc.ents:
                if ent.label_ != "PERSON":
                    continue
                # Post-NER split of joined capitals with prefix exceptions
                split = _split_joined_capitals_left_to_right(ent.text)
                cand = _post_filter_name_str(split)
                if cand:
                    found.append(cand)

    return sorted(set(found))


def _names_from_section_texts(texts: List[str]) -> List[str]:
    """Chunk -> NER -> split joined capitals -> normalize -> dedupe."""
    candidates = _chunk_candidates(texts)
    return _extract_names_spacy_from_candidates(candidates)


# ------------------------------- Entry Point -------------------------------

def extract_names(pdf_path: str, out_dir: str) -> Dict[str, Any] | None:
    """
    Read compiled_ocr.json and emit:
      - sections_with_text.json
      - sections_with_names.json
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

    # Build sections with names
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
