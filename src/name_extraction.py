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
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.common.io_utils import read_json, write_json
from src.name_ner import extract_person_names_per_candidate, dedupe_person_names, _ROSTER_NOISE_WORDS

# ---------------- CPU oversubscription guard ----------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ------------------------------- Config -------------------------------

IGNORE_HEADER_SUBSTRINGS = {"starter", "player", "captain"}
EXCLUDE_TEXT_LABELS = {"header", "paragraph_title", "footer", "image"}

# Layout labels PP-StructureV3 uses for section-title-like blocks. "doc_title"
# is included alongside the primary "header"/"paragraph_title" labels because
# some yearbooks' section titles get classified there instead.
_HEADER_LAYOUT_LABELS = {"header", "paragraph_title", "doc_title"}

# Some yearbooks' large/spaced stylized heading fonts get OCR'd as full-width
# Unicode glyphs (e.g. "ＡＴＨＬＥＴＩＣＳ") or underscore-joined letters
# instead of being classified as a header block by the layout model — they
# come through as plain "text"/"doc_title" blocks. This range/marker check is
# a narrow, high-precision signal for that artifact (real body text and names
# never contain full-width glyphs or underscores), so it catches those
# mislabeled headers without risking misclassifying ordinary short all-caps
# lines (e.g. "OFFICERS", a coach's name) as section headers.
_FULLWIDTH_RE = re.compile(r"[＀-￯]")
_STYLIZED_HEADER_MAX_WORDS = 6

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
    # NFKC folds OCR'd full-width Unicode glyphs (e.g. "Ａ" -> "A") to their
    # normal ASCII form, so downstream keyword matching and NER see plain text.
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", " ", s.strip())


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
    if lbl in _HEADER_LAYOUT_LABELS:
        return True
    if lbl == "text":
        raw = block.get("block_content", "")
        return _looks_like_stylized_header(raw if isinstance(raw, str) else "")
    return False


def _is_ignored_header_text(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(key in t for key in IGNORE_HEADER_SUBSTRINGS)


# ------------------------- Text correction helpers -------------------------

_PREFIX_EXCEPTIONS = ("Mc", "Mac", "O")
_SPLIT_ON_PUNCT = re.compile(r"[.,]+")

# A single letter followed by "." (an initial, e.g. "R." in "R. E. Foard")
# would otherwise be torn apart by _SPLIT_ON_PUNCT into meaningless
# single-letter fragments. We temporarily swap that period for a placeholder
# before splitting, then restore it (with a following space, since OCR often
# drops the space after an initial, e.g. "E.Foard") once chunking is done.
_INITIAL_DOT_RE = re.compile(r"\b([A-Za-z])\.(?=\s?[A-Za-z])")
_INITIAL_DOT_PLACEHOLDER = "\x00"


def _protect_initials(s: str) -> str:
    return _INITIAL_DOT_RE.sub(lambda m: m.group(1) + _INITIAL_DOT_PLACEHOLDER, s)


def _restore_initials(s: str) -> str:
    return re.sub(re.escape(_INITIAL_DOT_PLACEHOLDER) + r"\s*", ". ", s)


def _looks_all_caps(s: str) -> bool:
    """True if the string has letters and none are lowercase (ignores digits/punct/spaces)."""
    if not s:
        return False
    if not re.search(r"[A-Za-z]", s):
        return False
    return not bool(re.search(r"[a-z]", s))


def _looks_like_stylized_header(raw_text: str) -> bool:
    """
    Catch section-title blocks that PP-StructureV3 labeled as plain
    "text"/"doc_title" instead of "header"/"paragraph_title" — seen on
    yearbooks whose large/spaced heading font gets OCR'd as full-width
    Unicode glyphs (e.g. "ATHLETICS") or underscore-joined letters.

    Deliberately narrow: only fires when the ORIGINAL (pre-normalization)
    text contains one of those specific artifacts, so ordinary short
    all-caps lines (a coach's name, "OFFICERS") aren't misclassified as
    section headers.
    """
    raw = raw_text or ""
    if not (_FULLWIDTH_RE.search(raw) or "_" in raw):
        return False

    t = _normalize_space(raw).replace("_", " ")
    if not t or len(t.split()) > _STYLIZED_HEADER_MAX_WORDS:
        return False
    return _looks_all_caps(t)


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


_GRID_MAX_WIDTH_FRACTION = 0.35


def _looks_like_grid_block(raw_content: str, block_bbox: Any, page_width: Any) -> bool:
    """
    True when a block's RAW (pre-normalization) content looks like a narrow,
    multi-line, punctuation-free grid/column of short items — e.g. a roster
    laid out in columns, where PP-StructureV3 still preserves the original
    per-row "\n" breaks inside block_content — rather than ordinary wrapped
    prose or a comma-delimited list. Requires ALL of:
      - more than one real line (ignoring a purely trailing newline)
      - no periods anywhere in the block (in this pipeline's data, prose
        paragraphs and comma-lists always have at least one — a trailing
        sentence period or a period-abbreviated initial like "R. E. Foard")
      - the block is narrow relative to the page (a paragraph spanning most
        of the page width is prose, not a name column)

    Only when ALL hold do we split the block on "\n" into separate per-line
    candidates; otherwise newlines are collapsed to spaces exactly as
    before, so ordinary wrapped prose/comma-lists are never affected (a
    name like "R. E. Foard" can't be split mid-name by this).
    """
    if "." in raw_content:
        return False
    if "\n" not in raw_content.strip("\n"):
        return False
    if not isinstance(block_bbox, (list, tuple)) or len(block_bbox) < 3 or not page_width:
        return False
    width = block_bbox[2] - block_bbox[0]
    return width <= _GRID_MAX_WIDTH_FRACTION * page_width


def _collect_text_blocks_by_page(compiled_clean: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """Collect textual blocks per page, excluding headers/titles/footers/images."""
    out: Dict[int, List[Dict[str, Any]]] = {}

    for page_idx, page in enumerate(compiled_clean.get("pages", []), start=1):
        lst: List[Dict[str, Any]] = []
        page_width = page.get("width") if isinstance(page, dict) else None
        for blk in _page_blocks(page):
            label = str(blk.get("block_label", "")).lower()
            if label in EXCLUDE_TEXT_LABELS:
                continue
            raw = blk.get("block_content", "")
            if not raw:
                continue

            if _looks_like_grid_block(raw, blk.get("block_bbox"), page_width):
                for line in raw.split("\n"):
                    text = _normalize_space(line)
                    if text:
                        lst.append({"text": text, "y": _y1(blk)})
            else:
                text = _normalize_space(raw)
                if text:
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

_ROLE_WORD_SPLIT_RE = re.compile(r"[\s.,]")
_CLAUSE_SPLIT_RE = re.compile(r";")


def _is_single_role_word_caption(text: str) -> bool:
    """
    True only for a whole block that is itself a single role/organizational
    caption for one person (e.g. "COACHEP.FAULKNER", "COACHPASCO GETTLE") —
    used to drop such blocks entirely before candidate splitting, so a
    coach's/officer's name isn't fragmented and accidentally salvaged as a
    bare surname by NER later. Prefix match on the first word handles OCR
    dropping the space after the role word (e.g. "COACHPASCO" for "Coach
    Pasco").

    Deliberately requires NO comma anywhere in the block, so this doesn't
    also swallow multi-person list labels like "MEMBERS: Eleanor Bennett,
    Dot Fulk, ..." — those get comma-split into per-person candidates
    normally, and NER already handles the "MEMBERS:" noise prefix fine on
    just the first chunk.
    """
    t = _normalize_space(text)
    if not t or "," in t:
        return False
    first = _ROLE_WORD_SPLIT_RE.split(t, maxsplit=1)[0].lower()
    return any(first == w or first.startswith(w) for w in _ROSTER_NOISE_WORDS)


def _is_role_caption_clause(clause: str) -> bool:
    """
    True for a single ';'-delimited clause that is itself one "Name, role"
    pair (e.g. "Mr.Cordell,Athletic Director", "Mr.Savage,HeadCoach") — used
    to drop the whole clause so a coach's/manager's bare surname isn't
    fragmented off from its role word and salvaged as a PERSON by NER, the
    same way _is_single_role_word_caption already does for a whole undivided
    block. Multi-person captions join each person's "Name, role" with ';'
    (e.g. "Mr.Cordell,Athletic Director;Mr. Davis, Asst.Coach"), which
    defeats _is_single_role_word_caption's whole-block check (it bails on
    ANY comma) since each clause here has a comma between name and role.

    Deliberately requires the clause to have EXACTLY ONE comma, so it never
    touches a genuinely comma-delimited multi-person list that happens to
    end up glued into one clause (e.g. a "STANDING: A, B, C...KNEELING: D,
    manager" clause has many commas and is left untouched here).
    """
    clause = _normalize_space(clause)
    if clause.count(",") != 1:
        return False
    _, role_part = clause.split(",", 1)
    role_part = _normalize_space(role_part).lower()
    if not role_part:
        return False
    return any(w in role_part for w in _ROSTER_NOISE_WORDS)


def _chunk_candidates(texts: List[str], header: str = "") -> List[str]:
    """
    Step 1:
      - Drop any text block that's just the section's own header repeated
        as a caption (e.g. a photo caption that duplicates "Varsity
        Basketball" verbatim) — never a real name, but shaped enough like
        one (2+ capitalized words) to otherwise slip through the NER-miss
        fallback in name_ner.py.
      - Drop whole blocks that start with a role/organizational word (e.g. a
        coach's caption), so they never enter candidate generation at all.
      - Split each remaining text span on ';' into per-person clauses, and
        drop any clause that is itself a single "Name, role" pair (see
        _is_role_caption_clause) before further splitting.
      - Split each remaining clause on commas/periods into short candidate
        strings.
      - Then apply joined-capital splitting BEFORE NER (unless ALL CAPS).
    """
    header_norm = _normalize_space(header).casefold()
    cands: List[str] = []
    for t in texts:
        t = _normalize_space(t)
        if not t:
            continue
        if header_norm and t.casefold() == header_norm:
            continue
        if _is_single_role_word_caption(t):
            continue
        for clause in _CLAUSE_SPLIT_RE.split(t):
            clause = _normalize_space(clause)
            if not clause or _is_role_caption_clause(clause):
                continue
            protected = _protect_initials(clause)
            for chunk in _SPLIT_ON_PUNCT.split(protected):
                chunk = _normalize_space(_restore_initials(chunk))
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

    # Hyphens are folded to spaces (not kept) so a hyphenated header like
    # "Cross-Country Team" still matches the space-separated
    # _SPORTS_MULTI_WORD keyword "cross country".
    norm = re.sub(r"[^a-z0-9\s]+", " ", t)
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
        cands = _chunk_candidates(sec.get("texts", []), sec.get("header", ""))
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

