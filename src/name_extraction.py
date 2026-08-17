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
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from rapidfuzz.distance import Levenshtein

from src.common.io_utils import read_json, write_json
from src.name_ner import extract_person_names_per_candidate, dedupe_person_names, _ROSTER_NOISE_WORDS, _CAPTION_STOPWORDS

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
    "karate",
    "riflery",
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
    # OCR sometimes reads the gap in a spaced-out name as "_" ("LoIs_Cox");
    # underscores never appear in real yearbook text. (Header detection
    # checks the RAW text for "_" before this runs, so it's unaffected.)
    s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s.strip())


_ORPHAN_MIN_SCORE = 0.8


def _orphan_line_blocks(page: Dict[str, Any], blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Synthetic caption blocks for OCR lines the layout model left OUTSIDE
    every block (label "figure_title", so they flow through the same
    caption/title paths as real captions).

    PP-StructureV3's overall_ocr_res covers the whole page, but its layout
    pass regularly fails to wrap small captions in any block: on the 1975
    Ayantee the BASEBALL page (PDF p. 140) had six player captions and the
    "BASEBALL" title recognized in overall_ocr_res while parsing_res_list held
    ONE of them; the wrestling pages lost "Darnell Glover"/"Horace McDonald"
    the same way, and the 1972 Belles Memoires lost three cheerleaders. Only
    lines whose centre lies outside every block bbox are taken, so nothing
    already inside a block (table rows, image lettering, captions) can enter
    twice. Lines with fewer than two letters (page numbers, jersey digits)
    are skipped.

    Lines on one visual row are re-joined left-to-right ONLY when nearly
    touching (gap <= row height): a multi-word title OCR'd word by word ("The"
    "Track" "Team" "of" "'72") becomes one title again, while three player
    captions sharing a row under separate photos stay three captions.
    """
    ocr = page.get("overall_ocr_res") or {}
    texts = ocr.get("rec_texts") or []
    boxes = ocr.get("rec_boxes") or []
    scores = ocr.get("rec_scores") or []
    if not texts or len(boxes) != len(texts):
        return []
    bboxes = [
        tuple(float(v) for v in b["block_bbox"])
        for b in blocks
        if isinstance(b.get("block_bbox"), (list, tuple)) and len(b["block_bbox"]) == 4
    ]
    lines: List[Tuple[float, float, float, float, str]] = []
    for i, (text, box) in enumerate(zip(texts, boxes)):
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        t = str(text).strip()
        if len(re.findall(r"[A-Za-z]", t)) < 2:
            continue
        # a line the layout model ignored AND the recognizer was unsure of
        # ("TEDWATTS" 0.71, a photo credit) is not worth speculating on
        if i < len(scores) and float(scores[i]) < _ORPHAN_MIN_SCORE:
            continue
        x0, y0, x1, y1 = (float(v) for v in box)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if any(bx0 <= cx <= bx1 and by0 <= cy <= by1 for bx0, by0, bx1, by1 in bboxes):
            continue
        lines.append((x0, y0, x1, y1, t))
    if not lines:
        return []

    lines.sort(key=lambda l: (l[1], l[0]))
    rows: List[List[Tuple[float, float, float, float, str]]] = []
    for ln in lines:
        cy = (ln[1] + ln[3]) / 2.0
        for row in rows:
            ry0 = min(l[1] for l in row)
            ry1 = max(l[3] for l in row)
            rcy = (ry0 + ry1) / 2.0
            if ry0 <= cy <= ry1 and ln[1] <= rcy <= ln[3]:
                row.append(ln)
                break
        else:
            rows.append([ln])

    out: List[Dict[str, Any]] = []
    for row in rows:
        row.sort(key=lambda l: l[0])
        # split the row where neighbouring boxes are far apart
        groups: List[List[Tuple[float, float, float, float, str]]] = [[row[0]]]
        for ln in row[1:]:
            prev = groups[-1][-1]
            height = max(prev[3] - prev[1], ln[3] - ln[1], 1.0)
            if ln[0] - prev[2] <= height:
                groups[-1].append(ln)
            else:
                groups.append([ln])
        for g in groups:
            out.append({
                "block_label": "figure_title",
                "block_content": " ".join(l[4] for l in g),
                "block_bbox": [min(l[0] for l in g), min(l[1] for l in g),
                               max(l[2] for l in g), max(l[3] for l in g)],
                "orphan": True,
            })
    return out


def _page_blocks(page: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield unified blocks from PPStructure 'parsing_res_list' or legacy 'items'."""
    if not isinstance(page, dict):
        return
    if page.get("layout") is False:
        # plain-OCR-only page (extract_text(layout=False)): index material
        # for name_resolution, never section text
        return

    prl = page.get("parsing_res_list")
    if isinstance(prl, list):
        blocks: List[Dict[str, Any]] = []
        for blk in prl:
            if not isinstance(blk, dict):
                continue
            blocks.append({
                "block_label": blk.get("block_label"),
                "block_content": blk.get("block_content", "") if isinstance(blk.get("block_content"), str) else "",
                "block_bbox": blk.get("block_bbox") if isinstance(blk.get("block_bbox"), (list, tuple)) else None,
            })
        for b in blocks:
            yield b
        for b in _orphan_line_blocks(page, blocks):
            yield b
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


_QUOTE_CHARS = "\"'“”‘’"


def _is_person_name_shaped_header(text: str) -> bool:
    """
    True for a title block that is really a PLAYER'S NAME, not a section
    title (1975 Ayantee: each player-profile page uses the player's name as
    its paragraph_title, e.g. "Al Holland", "Artis Stanfield"). Such blocks
    must not become section markers — and their text should flow into the
    enclosing sport's section so the name is extracted as a name.
    Requires 2-3 capitalized alphabetic tokens, no digits, no sports match.
    """
    t = _normalize_space(text)
    if not t or re.search(r"\d", t):
        return False
    toks = t.split()
    if not (2 <= len(toks) <= 3):
        return False
    if not all(re.fullmatch(r"[A-Za-z'.\-]+", tok) and tok[0].isupper() for tok in toks):
        return False
    if any(tok.rstrip(".").lower() in _ROSTER_NOISE_WORDS for tok in toks):
        return False
    # "The Players", "Oak Hill's Best", "Mighty Dogs" are titles, not people
    if any(tok.rstrip(".").lower() in _CAPTION_STOPWORDS for tok in toks):
        return False
    if any(tok.lower().endswith(("'s", "’s")) for tok in toks):
        return False
    return not _is_sports_section_header(t)


def _is_junk_section_header(text: str) -> bool:
    """
    Headers that are layout noise, never real sections:
      - 1-2 characters ("S" from a stylized SPORTS divider)
      - digit-bearing non-sports titles ("6-4, 245 lb. Senior Defensive
        End") — real sports titles with years ("Wrestling '69",
        "...BASKETBALLTEAM 1975") survive via the sports match.
    """
    t = _normalize_space(text)
    if len(t) <= 2:
        return True
    if re.search(r"\d", t) and not _is_sports_section_header(t):
        return True
    # lettering fragments with no vowel at all ("SGS", "GGS")
    if not re.search(r"[AEIOUYaeiouy]", t):
        return True
    return False


def _is_caption_like_header_text(text: str) -> bool:
    """
    True when a header-labeled block is really a photo caption, not a
    section title (1970 Archway labels captions as header/doc_title):
    it ends like a sentence ("Wilson carries for gain of yards.") or is a
    "Label: ..." line ("Seniors: F. Sauls, K. Bryan.", "Coach: D. Scott").
    Real yearbook section titles never end with sentence punctuation and
    never contain a colon. A caption-like block is NOT a section marker;
    list-like ones re-enter the section's text so their rosters survive.

    A sports-matching title is never demoted (belt-and-braces so a title
    like "Track." could not lose its section).
    """
    t = _normalize_space(text)
    if not t:
        return False
    if t[0] in _QUOTE_CHARS:
        # A quoted exclamation ('"Sink it, Vicky, sink it!"') is always a
        # caption.
        return True
    if ":" in t or "：" in t:
        # "Label: ..." is always a caption, even when the label mentions a
        # sport ("Head Cheerleader: Valencia Willoughby").
        return True
    if not t.rstrip(_QUOTE_CHARS).endswith((".", "!", "?")):
        return False
    return not _is_sports_section_header(t)


# ------------------------- Text correction helpers -------------------------

_PREFIX_EXCEPTIONS = ("Mc", "Mac", "O")
# Also split on double quotes and on a dash set off by spaces or followed
# by a quote ('Charles Davis-"Dancing to the music."', "DEBRA CLONTZ -
# Senior - Chief") — never inside a hyphenated name ("Co-Captain",
# "Lower-the-Boom").
_SPLIT_ON_PUNCT = re.compile(r"[.,\"“”]+|\s[-–—]+\s|[-–—]+(?=[\"“”'‘’])|[-–—]+$")

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

            # "McCALL" / "MacDONALD" in an all-caps caption is one name,
            # not a run of joined capitals
            if tok.isupper() or len(tok) < 6 or re.fullmatch(r"(?:Mc|Mac|O)[A-Z]+", tok):
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
    if len(s.split()) <= 4:
        s = _join_split_fragments(s)  # "James Jeff ries" -> "James Jeffries"
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

# An ALL-CAPS label ending in ':' at the very start of a text block (e.g.
# "JV FOOTBALL: ROW 1: Don Hinson, ..."). When the label names a sport, the
# layout model gave the sub-team no header block of its own, so its roster
# would silently merge into the enclosing section (JV Football lost inside
# FOOTBALL). No digits allowed in the label, so "ROW 1:" can never match.
_INLINE_SPORTS_LABEL_RE = re.compile(r"^\s*([A-Z][A-Z .&'\-]{1,40}?)\s*[:：]")


def _inline_sports_label(text: str) -> Optional[re.Match]:
    """Match an inline sports sub-team label prefix, else None."""
    m = _INLINE_SPORTS_LABEL_RE.match(text or "")
    if m and _is_sports_section_header(m.group(1)):
        return m
    return None


def _split_sport_labelled_block(raw: str) -> List[Tuple[str, str]]:
    """
    A staff/directory block that alternates an ALL-CAPS sport label with a
    short mixed-case name — the 1975 Ayantee's coaches page: "FOOTBALL
    Hornsby Howell BASKETBALL Warren Reynolds WRESTLING Melvin Pinckney ..."
    (one line, or one line per item). Returns [(label, text), ...] so each
    label becomes its own inline section marker with its name(s) as text;
    [] when the block isn't that pattern (text before the first label, a
    label with nothing or with more than four words after it, an all-caps
    "name", or fewer than two labels).
    """
    toks = _normalize_space(raw).split()
    segs: List[Tuple[str, List[str]]] = []
    i = 0
    while i < len(toks):
        label = None
        step = 1
        two = " ".join(toks[i:i + 2]).lower()
        if i + 1 < len(toks) and two in _SPORTS_MULTI_WORD and toks[i].isupper() and toks[i + 1].isupper():
            label, step = " ".join(toks[i:i + 2]), 2
        elif toks[i].isalpha() and toks[i].isupper() and len(toks[i]) >= 4 and _is_sports_section_header(toks[i]):
            label = toks[i]
        if label:
            segs.append((label, []))
            i += step
            continue
        if not segs:
            return []
        segs[-1][1].append(toks[i])
        i += 1
    if len(segs) < 2:
        return []
    out: List[Tuple[str, str]] = []
    for label, words in segs:
        if not (1 <= len(words) <= 4) or not all(w[0].isupper() for w in words):
            return []
        text = " ".join(words)
        if _looks_all_caps(text):
            return []
        out.append((label, text))
    return out


def _looks_like_sports_title_block(text: str) -> bool:
    """
    True for a short title-like block naming a sport — used to promote
    figure_title blocks to section markers. The 1970 Archway's script
    section titles are labeled figure_title and OCR'd loosely ("restling",
    "Bolf", "Harsity Cheer leaders"), so they'd otherwise never start a
    section. Requires ≤3 words, no colon, no sentence-ending punctuation;
    then matches sports keywords with a slightly looser fuzzy rule (words
    down to 4 letters, 1 edit) than _is_sports_section_header — safe here
    because the whole block must already look like a bare title, which a
    caption sentence or a person's name in a roster never does.
    """
    t = _normalize_space(text)
    if not t or ":" in t or "：" in t:
        return False
    if t.rstrip(_QUOTE_CHARS).endswith((".", "!", "?")):
        return False
    words = re.sub(r"[^a-z0-9\s]+", " ", t.lower()).split()
    if not words:
        return False
    # A long-form institutional team title ("N.C. A&T STATE UNIVERSITY
    # FOOTBALLTEAM 1974-75") — sports keyword plus an explicit team word
    # is unambiguous at any length.
    if _is_sports_section_header(t) and any(
        w in ("team", "roster", "squad") or w.endswith(("team", "roster", "squad"))
        for w in words
    ):
        return True
    if len(words) > 3:
        return False
    if _is_sports_section_header(t):
        return True
    for w in words:
        if len(w) < 4:
            continue
        for kw in _SPORTS_SINGLE_WORD:
            if len(kw) >= 4 and Levenshtein.distance(w, kw, score_cutoff=1) <= 1:
                return True
    return False


# Words that name a yearbook's big NON-sports divisions. A title-only page
# whose title is made of these words closes the sports zone opened by a
# "SPORTS"/"ATHLETICS" divider (see _collect_headers_with_pos).
_NON_SPORTS_DIVIDER_WORDS = {
    "classes", "class", "seniors", "juniors", "sophomores", "freshmen",
    "underclassmen", "graduates", "faculty", "administration", "staff",
    "organizations", "organization", "clubs", "activities", "academics",
    "curriculum", "departments", "advertisements", "advertising", "ads",
    "patrons", "sponsors", "index", "honors", "features", "dedication",
    "student", "life", "campus", "government", "greeks", "fraternities",
    "sororities", "military", "rotc", "music", "drama", "arts", "closing",
    "epilogue", "memories", "events", "publications", "directory",
    "personalities", "royalty", "queens", "favorites", "school",
}
_DIVIDER_STOPWORDS = {"the", "of", "and", "&", "our", "in", "a"}
_SPORTS_UMBRELLA_WORDS = {"sports", "athletics", "athletic"}


def _divider_words(text: str) -> List[str]:
    words = re.sub(r"[^a-z0-9\s]+", " ", _normalize_space(text).lower()).split()
    return [w for w in words if w not in _DIVIDER_STOPWORDS]


def _fuzzy_in(word: str, vocab: Set[str]) -> bool:
    if word in vocab:
        return True
    if len(word) < 6:
        return False
    return any(len(v) >= 6 and Levenshtein.distance(word, v, score_cutoff=1) <= 1 for v in vocab)


def _is_sports_umbrella_title(text: str) -> bool:
    """A bare "SPORTS" / "ATHLETICS" divider title (<=3 content words)."""
    words = _divider_words(text)
    return 0 < len(words) <= 3 and any(_fuzzy_in(w, _SPORTS_UMBRELLA_WORDS) for w in words)


def _is_non_sports_divider_title(text: str) -> bool:
    """A bare non-sports division title ("CLASSES", "ORGANIZATIONS",
    "STUDENT LIFE"): every content word is a known division word and none is
    a sport."""
    words = _divider_words(text)
    if not (0 < len(words) <= 3):
        return False
    if _is_sports_section_header(text):
        return False
    return all(_fuzzy_in(w, _NON_SPORTS_DIVIDER_WORDS) for w in words)


def _has_letters(blk: Dict[str, Any]) -> bool:
    raw = blk.get("block_content", "")
    return len(re.findall(r"[A-Za-z]", raw if isinstance(raw, str) else "")) >= 2


def _marker_kind(text: str, blk: Dict[str, Any], lettered_blocks: List[Dict[str, Any]]) -> Optional[str]:
    """
    Classify a header marker as a book-level DIVIDER page or not.

    Yearbooks open each big division with a title-only page ("ATHLETICS" over
    a drawing, "SPORTS" over a photo collage) and then run sub-sections whose
    titles need not name a sport at all (1972 Belles Memoires: "The Head
    Bulldog", "Striving Seniors", "in Action" for its football pages). Every
    section between a sports divider and the next non-sports divider is
    therefore a sports section, whatever its own title says.

    A divider is a marker whose block is the ONLY lettered block on its page
    (page numbers and empty photo blocks aside), so a titled photo-collage
    page ("SENIOR AGGIES") or a titled roster page can't pass by accident.
    """
    key = _normalize_space(text).casefold()
    others = [
        b for b in lettered_blocks
        if b is not blk
        and _correct_header_text(b.get("block_content", "")).strip(" .").casefold() != key
    ]
    if others or blk not in lettered_blocks:
        return None
    if _is_sports_umbrella_title(text):
        return "sports_divider"
    if _is_non_sports_divider_title(text):
        return "other_divider"
    return None


def _collect_headers_with_pos(compiled_clean: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ordered header markers: {'page', 'y', 'text', 'kind'}."""
    markers: List[Dict[str, Any]] = []

    for page_idx, page in enumerate(compiled_clean.get("pages", []), start=1):
        blocks = list(_page_blocks(page))
        blocks.sort(key=_y1)
        lettered = [
            b for b in blocks
            if _has_letters(b) and str(b.get("block_label", "")).lower() != "number"
        ]
        page_markers: List[Dict[str, Any]] = []

        def _add(txt: str, blk: Dict[str, Any], y: Optional[float] = None) -> None:
            key = txt.casefold()
            if any(m["text"].casefold() == key for m in page_markers):
                return  # same title seen twice on the page (image lettering + orphan line)
            page_markers.append({
                "page": page_idx, "y": _y1(blk) if y is None else y, "text": txt,
                "kind": _marker_kind(txt, blk, lettered),
            })

        for blk in blocks:
            lbl = str(blk.get("block_label", "")).lower()
            if lbl == "figure_title":
                raw = blk.get("block_content", "")
                txt = _correct_header_text(raw if isinstance(raw, str) else "")
                if txt and _looks_like_sports_title_block(txt):
                    # An orphan OCR line (see _orphan_line_blocks) is not a
                    # layout-detected caption, so it doesn't get the fuzzy
                    # sports match real figure_titles do ("RUBY POOLE" ~
                    # rugby): the sport must be named outright.
                    if not blk.get("orphan") or _is_sports_section_header(txt):
                        _add(txt, blk)
                continue
            if lbl == "image":
                # A section title printed over/inside a photo, or a divider
                # page whose title the layout model folded into its image
                # block ("ATHLETICS", "WRESTLING", "TRACK") — the ONLY text
                # in the block must be the short sports title.
                # Checked line by line: the title line shares the block
                # with whatever else is lettered on the photo ("OUR
                # CHEERLEADERS..." above a first name, jersey numbers).
                raw = blk.get("block_content", "")
                # Only an ALL-CAPS line naming a sport outright counts —
                # bio prose printed over a photo ("...National Football
                # League") and OCR fragments ("CHER YL") must not.
                for line in (raw if isinstance(raw, str) else "").split("\n"):
                    txt = _correct_header_text(line).strip(" .")
                    if (
                        txt
                        and len(txt.split()) <= 3
                        and _looks_all_caps(txt)
                        and _is_sports_section_header(txt)
                        and not _is_junk_section_header(txt)
                    ):
                        _add(txt, blk)
                        break
                continue
            if _is_header(blk):
                raw = blk.get("block_content", "")
                txt = _correct_header_text(raw if isinstance(raw, str) else "")
                if (
                    txt
                    and not _is_ignored_header_text(txt)
                    and not _is_caption_like_header_text(txt)
                    and not _is_person_name_shaped_header(txt)
                    and not _is_junk_section_header(txt)
                ):
                    _add(txt, blk)
                continue
            # Inline sub-team label mislaid inside a text block
            raw = blk.get("block_content", "")
            m = _inline_sports_label(_normalize_space(raw if isinstance(raw, str) else ""))
            if m:
                txt = _correct_header_text(m.group(1))
                if txt and not _is_ignored_header_text(txt):
                    _add(txt, blk)
                continue
        markers.extend(page_markers)

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
    if width > _GRID_MAX_WIDTH_FRACTION * page_width:
        return False
    # every line must read like a roster item (capitalized tokens only);
    # a lowercase word means wrapped prose ("Our Puppette Coaches are /
    # GLENN CHILDRES and / LAWRENCE CARPENTER"), which must stay one text
    for line in raw_content.split("\n"):
        toks = line.split()
        if any(t[0].islower() for t in toks):
            return False
    return True


def _x0(block: Dict[str, Any]) -> float:
    bbox = block.get("block_bbox")
    try:
        return float(bbox[0]) if bbox else -1.0
    except Exception:
        return -1.0


# Two blocks whose top edges are within this many pixels are treated as the
# same visual row (e.g. the two columns of a caption split across the page)
# and ordered left-to-right instead of by their tiny y difference. 80px at
# 400 DPI is ~0.2in — far smaller than a line of roster text, so distinct
# rows can't be banded together.
_ROW_BAND_PX = 80


def _sort_reading_order(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort rows top-to-bottom, but order same-row bands left-to-right.

    A plain sort by y put the RIGHT column of a two-column caption first
    whenever its bbox top was a few pixels higher, severing names that wrap
    across the column break (e.g. "...,Steve " | "McCorkle,...")."""
    rows = sorted(rows, key=lambda r: r["y"])
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(rows):
        j = i + 1
        while j < len(rows) and rows[j]["y"] - rows[i]["y"] <= _ROW_BAND_PX:
            j += 1
        out.extend(sorted(rows[i:j], key=lambda r: r["x"]))
        i = j
    return out


# US state names/abbreviations, used to tell a roster table's name cell
# ("Ryan, Steve") from a hometown cell ("Housron, Texas") — both are
# "Word, Word" shaped, and only the first is a person.
_STATE_WORDS = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "ohio", "oklahoma", "oregon",
    "pennsylvania", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington", "wisconsin", "wyoming",
    "ala", "ariz", "ark", "cal", "calif", "colo", "conn", "del", "fla", "ga",
    "ida", "ill", "ind", "kan", "kans", "ky", "la", "md", "mass", "mich",
    "minn", "miss", "mo", "mont", "neb", "nebr", "nev", "okla", "ore", "oreg",
    "pa", "tenn", "tex", "va", "vt", "wash", "wis", "wisc", "wyo",
    "nc", "nd", "nh", "nj", "nm", "ny", "ri", "sc", "sd", "dc", "wv",
}

# Class-year/role words that trail a profile panel's name line
# ("Paul McKibbins Senior") and are not part of the name.
_PANEL_TRAILING_WORDS = {
    "senior", "junior", "sophomore", "freshman", "seniors", "juniors",
    "coach", "manager", "trainer", "captain", "co-captain",
}

# A roster table's name cell: surname, separator, given name(s). The separator
# is a comma in print but OCR often reads it as a period ("Holland. Glenn"),
# and the space after it is sometimes missing ("Turner,Elsworth").
_TABLE_NAME_CELL_RE = re.compile(
    r"^([A-Z][A-Za-z'’\-]{1,})\s*[.,]\s*([A-Z][A-Za-z'’\-]{1,}(?:\s+[A-Z][A-Za-z'’\-]*)?)$"
)


def _table_cell_to_name(cell: str) -> Optional[str]:
    """
    Put a roster-table name cell in normal order: "Ryan, Steve" -> "Steve Ryan".

    Returns None for anything that isn't a person. Hometown cells share the
    "Word, Word" shape, so a half that is a state name or abbreviation
    ("Housron, Texas", "Gaithersburg, Md.", "Norwood. N. C.") is rejected.
    """
    m = _TABLE_NAME_CELL_RE.match(cell.strip())
    if not m:
        return None
    surname, given = m.group(1).strip(), m.group(2).strip()
    if given.lower() in _STATE_WORDS or surname.lower() in _STATE_WORDS:
        return None
    if len(given) < 2 or len(surname) < 2:
        return None
    return f"{given} {surname}"


def _recover_table_block_lines(
    page: Dict[str, Any], blk: Dict[str, Any]
) -> List[Tuple[str, float, float]]:
    """
    Name cells for a table block PP-StructureV3 left empty, as (name, y, x).

    text_extraction builds the pipeline with use_table_recognition=False, so a
    detected table region comes back with empty block_content even though every
    line inside it was recognized — the 1975 Ayantee's football roster (printed
    p. 95) is one empty table block sitting over 416 recognized strings. Those
    lines survive in overall_ocr_res, which covers the WHOLE page, so only
    boxes whose centre falls inside this block's bbox are taken; without that
    test the page's headers, footers and prose would leak into section text.

    Only name-shaped cells are returned, so the neighbouring columns (numbers,
    positions, heights, hometowns) never reach NER.
    """
    bbox = blk.get("block_bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return []
    bx0, by0, bx1, by1 = (float(v) for v in bbox)

    ocr = page.get("overall_ocr_res") or {}
    texts = ocr.get("rec_texts") or []
    boxes = ocr.get("rec_boxes") or []
    if not texts or len(boxes) != len(texts):
        return []

    cells: List[Tuple[str, float, float, float, float]] = []
    for text, box in zip(texts, boxes):
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        x0, y0, x1, y1 = (float(v) for v in box)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if not (bx0 <= cx <= bx1 and by0 <= cy <= by1):
            continue
        cells.append((_normalize_space(str(text)), x0, y0, x1, y1))

    # A "NAME"/"NAMES"/"PLAYER(S)" column heading tells us which column
    # holds the people; then plain "First Last" cells in that column are
    # names too (the 1975 Ayantee's wrestling, baseball and Aggiettes
    # rosters print given name first, with no comma to recognise them by).
    # Without such a heading only the unambiguous "Surname, Given" cells
    # are taken, as before.
    heading = next(
        (c for c in cells if re.fullmatch(r"(?i)names?|players?", c[0].strip(" .:"))),
        None,
    )

    out: List[Tuple[str, float, float]] = []
    for text, x0, y0, x1, y1 in cells:
        name = _table_cell_to_name(text)
        if not name and heading is not None:
            _, hx0, hy0, hx1, hy1 = heading
            in_column = (y0 + y1) / 2.0 > (hy0 + hy1) / 2.0 and x0 < hx1 and x1 > hx0
            text = _join_split_fragments(text)
            if in_column and _is_plain_full_name_cell(text):
                name = _normalize_name_casing(text)
        if name:
            out.append((name, y1, x0))
    return out


_PLAIN_NAME_TOKEN_RE = re.compile(r"[A-Z][A-Za-z'’\-]*\.?")

# A surname OCR-split by a stray space ("James Jeff ries", "Lit tle"): a
# capitalized token followed by a short all-lowercase fragment. Real
# lowercase words that can follow a name are excluded.
_SPLIT_FRAGMENT_RE = re.compile(r"\b([A-Z][a-z]+) ([a-z]{2,4})\b(?![-'’])")
_FRAGMENT_STOP = {
    "and", "of", "the", "or", "in", "on", "at", "to", "for", "jr", "sr", "de",
    "la", "van", "von", "del", "di", "da", "du", "le", "st", "is", "as", "by",
    "with", "was", "his", "her", "who", "has", "had", "are", "if", "it", "an",
    "our", "all", "not", "but", "so", "up", "out", "off", "no", "yes",
}


def _join_split_fragments(text: str) -> str:
    return _SPLIT_FRAGMENT_RE.sub(
        lambda m: m.group(0) if m.group(2) in _FRAGMENT_STOP else m.group(1) + m.group(2),
        text,
    )


def _is_plain_full_name_cell(text: str) -> bool:
    """"First Last" / "First M. Last" roster cell: 2-4 capitalized alphabetic
    tokens, no digits/commas, none a role or sports word."""
    t = _normalize_space(text)
    if not t or "," in t or re.search(r"\d", t):
        return False
    toks = t.split()
    if not (2 <= len(toks) <= 4):
        return False
    if not all(_PLAIN_NAME_TOKEN_RE.fullmatch(tok) for tok in toks):
        return False
    if any(tok.rstrip(".").lower() in _ROSTER_NOISE_WORDS for tok in toks):
        return False
    if any(tok.lower() in _STATE_WORDS for tok in toks):
        return False
    return not _is_sports_section_header(t)


def _image_block_heading(raw: str) -> str:
    """
    The heading lines of a player-profile panel, rejoined into one line.

    PP-StructureV3 labels a photo with text printed over it as "image" while
    still OCR'ing that text into block_content. On the 1975 Ayantee's profile
    pages the player's name is set one word per line ("Paul" / "McKibbins"), so
    the lines only read as a person once rejoined, and stopping at the first
    vitals or bio line keeps the narrative out of NER.

    A real panel always prints vitals (height/weight/number) under the name;
    decorative lettering that OCR'd into an image block ("AGGIES") does not, so
    requiring vitals keeps banner fragments out.
    """
    head: List[str] = []
    saw_vitals = False
    for line in raw.split("\n"):
        line = _normalize_space(line)
        if not line:
            continue
        if len(line.split()) > 4 or any(ch.isdigit() for ch in line):
            saw_vitals = any(ch.isdigit() for ch in line)
            break
        head.append(line)
    if not saw_vitals:
        return ""
    while head and head[-1].lower().strip(".,") in _PANEL_TRAILING_WORDS:
        head.pop()
    text = " ".join(head).lstrip("-—–•· ").strip()
    # Decorative lettering that bled into an image block ("AGGIE", "GGIESI
    # AGGIES", "-GGI GGIE") is set in all caps; a profile panel prints the
    # player's name in title case, so an all-caps heading is never a name.
    if not text or text == text.upper():
        return ""
    return text


# An awards / honours list line: label, dot leader, name
# ("Most Valuable. . Allen Spruill", "Hustle......Raymond Perry").
_DOT_LEADER_RE = re.compile(r"(?:\.\s*){2,}|…")


def _dot_leader_names(raw: str) -> List[str]:
    """
    Right-hand sides of a dot-leader list block, one per line, or [] when
    the block isn't one. The 1975 Ayantee's "ATHLETIC AWARDS SPECIAL" page
    is a single "content" block of ~40 such lines; read as one paragraph it
    is comma-free prose and gets dropped, and split blindly its labels
    ("Most Valuable") are name-shaped enough to be rescued as people. Needs
    >= 3 lines with a leader in at least half of them.
    """
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    if len(lines) < 3:
        return []
    rhs_of: List[Optional[str]] = []
    n_leader = 0
    for ln in lines:
        if _DOT_LEADER_RE.search(ln):
            rhs_of.append(_normalize_space(_DOT_LEADER_RE.split(ln, maxsplit=1)[-1]))
            n_leader += 1
            continue
        # OCR often collapses the leader to one period: "Hustle. Raymond
        # Perry" — a short label, a period, then a 2-3 token name.
        m = _SINGLE_DOT_LEADER_RE.match(ln)
        if m:
            rhs_of.append(m.group(2))
            n_leader += 1
            continue
        # ...or drops it entirely: "Leadership Alfred Holland", "Most
        # Valuable Wilbert Stancel" — label word(s) then a two-token name
        # (three when the middle one is an initial).
        m = _BARE_LABEL_NAME_RE.match(ln)
        rhs_of.append(m.group(2) if m else None)
    n_bare = sum(1 for r in rhs_of if r is not None) - n_leader
    if n_leader < 3 or (n_leader + n_bare) * 2 < len(lines):
        return []
    return [r for r in rhs_of if r and re.search(r"[A-Za-z]", r)]


_BARE_LABEL_NAME_RE = re.compile(
    r"^([A-Z][a-z]+(?: [A-Z][a-z]+)?)\s+([A-Z][A-Za-z'’\-]+(?:\s+[A-Z]\.)?\s+[A-Z][A-Za-z'’\-]+)$"
)


_SINGLE_DOT_LEADER_RE = re.compile(
    r"^([A-Za-z][A-Za-z ]{1,30}?)\.\s*([A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-.]+){1,2})$"
)


def _image_caps_tokens_book(compiled_clean: Dict[str, Any]) -> Counter:
    """Book-wide count of ALL-CAPS single-word lines inside image blocks."""
    cnt: Counter = Counter()
    for page in compiled_clean.get("pages", []):
        for blk in _page_blocks(page):
            if str(blk.get("block_label", "")).lower() != "image":
                continue
            raw = blk.get("block_content", "")
            for tok in _image_caps_lines(raw if isinstance(raw, str) else ""):
                cnt[tok] += 1
    return cnt


def _image_caps_lines(raw: str) -> List[str]:
    """Single ALL-CAPS alphabetic words set one per line in an image block,
    plus runs of >= 3 single-letter lines (a name set vertically, "R/E/B/A")
    rejoined into one word."""
    out: List[str] = []
    run: List[str] = []

    def _flush() -> None:
        if len(run) >= 3:
            out.append("".join(run))
        run.clear()

    for line in raw.split("\n"):
        t = line.strip()
        if not t:
            continue  # blank lines interleave the letters of a vertical name
        if re.fullmatch(r"[A-Z]", t):
            run.append(t)
            continue
        _flush()
        if re.fullmatch(r"[A-Z]{4,}", t):
            out.append(t)
    _flush()
    return out


def _is_decorative_caps(tok: str, book_counts: Counter) -> bool:
    """
    True when an ALL-CAPS word is (a variant of) lettering that recurs in
    the book's photos — the school/mascot name on every jersey ("AGGIE",
    "AGGIES", "AGGIETTES", "BULLOOG"/"BULLOOGS"/"ILLOOGS") — judged by
    fuzzy clustering of the image-block caps words: any other caps word
    within 2 edits (1 for short words), or one contained in this word,
    makes it decoration. A person's captioned first name has no such
    neighbours.
    """
    tok = tok.upper()
    for w, c in book_counts.items():
        if w == tok:
            if c >= 2:
                return True
            continue
        if len(w) >= 4 and w in tok:
            return True
        budget = 2 if min(len(w), len(tok)) >= 6 else 1
        if abs(len(w) - len(tok)) <= budget and Levenshtein.distance(w, tok, score_cutoff=budget) <= budget:
            return True
    return False


def _image_caps_names(raw: str, book_counts: Counter) -> List[str]:
    """
    First names printed as the only caption of a photo INSIDE an image block
    (1975 Ayantee cheerleader pages: "CASSANDRA", "ROBIN", vertical "REBA").
    A word is a candidate only when it appears ONCE in the whole book's
    image lettering — decorative words recur (school name, "AGGIES" on every
    jersey) while a person's caption doesn't — and is not near (<= 1 edit,
    or 2 for long words) any recurring word, sport, or role word. NER still
    has to accept the word as a PERSON downstream.
    """
    out: List[str] = []
    for tok in _image_caps_lines(raw):
        low = tok.lower()
        if low in _ROSTER_NOISE_WORDS or _is_sports_section_header(tok):
            continue
        if _POSITION_MARKER_RE.match(tok) or low in _PANEL_TRAILING_WORDS:
            continue
        if _is_decorative_caps(tok, book_counts):
            continue
        out.append(tok)
    # A page of individually captioned photos carries several such names
    # (or a sports title line, "OUR CHEERLEADERS", alongside one); one or
    # two words on a photo ("GREEN" on an opponent's jersey, "OAK"/"HILL"
    # on a banner) are lettering.
    has_title = any(
        _looks_all_caps(ln.strip()) and _is_sports_section_header(ln)
        and len(ln.split()) <= 3
        for ln in raw.split("\n")
    )
    if len(out) < 3 and not has_title:
        return []
    # Title-cased so downstream sees a caption name, not decorative caps.
    return [t.title() for t in out]


def _collect_text_blocks_by_page(compiled_clean: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """Collect textual blocks per page, excluding headers/titles/footers/images."""
    out: Dict[int, List[Dict[str, Any]]] = {}
    caps_counts = _image_caps_tokens_book(compiled_clean)

    for page_idx, page in enumerate(compiled_clean.get("pages", []), start=1):
        lst: List[Dict[str, Any]] = []
        page_width = page.get("width") if isinstance(page, dict) else None
        for blk in _page_blocks(page):
            label = str(blk.get("block_label", "")).lower()
            raw = blk.get("block_content", "")

            # A table region left empty by use_table_recognition=False: its
            # rows are the roster, recovered from overall_ocr_res.
            if label == "table" and not (raw or "").strip():
                for name, y, x in _recover_table_block_lines(page, blk):
                    lst.append({"text": name, "y": y, "x": x, "label": "figure_title"})
                continue

            # A photo with the player's name printed over it.
            if label == "image" and isinstance(raw, str) and raw.strip():
                text = _image_block_heading(raw)
                if text:
                    lst.append({"text": text, "y": _y1(blk), "x": _x0(blk), "label": "figure_title"})
                else:
                    for tok in _image_caps_names(raw, caps_counts):
                        lst.append({"text": tok, "y": _y1(blk), "x": _x0(blk), "label": "figure_title"})
                continue

            # An awards/honours dot-leader list: one name per line.
            if isinstance(raw, str) and label not in EXCLUDE_TEXT_LABELS:
                leader_names = _dot_leader_names(raw)
                if leader_names:
                    for text in leader_names:
                        lst.append({"text": text, "y": _y1(blk), "x": _x0(blk), "label": label})
                    continue

            if label in EXCLUDE_TEXT_LABELS or label == "doc_title":
                # A header-labeled block that is really a caption is not a
                # marker; when list-like (has ':' or ',') it can carry a
                # roster ("Seniors: F. Sauls, K. Bryan."), so route it back
                # into section text. Pure sentence captions ("Purvis shoots
                # for two!") stay excluded — no roster content.
                if label in ("header", "paragraph_title", "doc_title") and isinstance(raw, str):
                    t = _normalize_space(raw)
                    if _is_caption_like_header_text(t) and ("," in t or ":" in t or "：" in t):
                        lst.append({"text": t, "y": _y1(blk), "x": _x0(blk), "label": "figure_title"})
                    elif _is_person_name_shaped_header(t):
                        # a player's name used as a page title (1975
                        # Ayantee profile pages) — it's roster data
                        lst.append({"text": t, "y": _y1(blk), "x": _x0(blk), "label": "figure_title"})
                continue
            if not raw:
                continue

            if _split_sport_labelled_block(raw):
                # a coaches' directory ("FOOTBALL Hornsby Howell BASKETBALL
                # Warren Reynolds ..."): staff, not players — see
                # _collect_staff_candidates
                continue

            if _looks_like_grid_block(raw, blk.get("block_bbox"), page_width):
                for line in raw.split("\n"):
                    text = _normalize_space(line)
                    if text:
                        lst.append({"text": text, "y": _y1(blk), "x": _x0(blk), "label": label})
            else:
                text = _normalize_space(raw)
                # Drop an inline sub-team label prefix ("JV FOOTBALL: ...");
                # it became a section marker in _collect_headers_with_pos.
                m = _inline_sports_label(text)
                if m:
                    text = text[m.end():].strip()
                if text:
                    lst.append({"text": text, "y": _y1(blk), "x": _x0(blk), "label": label})
        out[page_idx] = _sort_reading_order(lst)

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


def _reassign_captions_above_headers(
    markers: List[Dict[str, Any]],
    texts_by_page: Dict[int, List[Dict[str, Any]]],
) -> None:
    """
    A photo + its figure_title caption often sit ABOVE the section title on
    the same page (1969 Acorn: tennis caption above "Tennis Team", wrestling
    caption above "Senior Grapplers"). Reading-order slicing would hand that
    caption to the PRECEDING section, misattributing every name in it.

    For the FIRST marker on a page, if the nearest text row above it is a
    figure_title, bump that row's y to the marker's y so it lands in the
    section the header opens. Restricted to the first marker on the page so
    a caption between two same-page sections keeps its normal attribution.
    """
    for mk in markers:
        if any(m2["page"] == mk["page"] and m2["y"] < mk["y"] for m2 in markers):
            continue  # not the first marker on this page
        above = [r for r in texts_by_page.get(mk["page"], []) if r["y"] < mk["y"]]
        # Walk upward through CONSECUTIVE captions (nearest first): a page
        # can stack several caption rows between the photo and the title
        # ("Head Cheerleader: ..." over "Advisor: ..." over the title).
        for r in sorted(above, key=lambda r: r["y"], reverse=True):
            if r.get("label") != "figure_title":
                break
            r["y"] = mk["y"]


def _build_sections(compiled_clean: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build sections with header, page span, and collected texts."""
    markers = _collect_headers_with_pos(compiled_clean)
    texts_by_page = _collect_text_blocks_by_page(compiled_clean)
    _reassign_captions_above_headers(markers, texts_by_page)

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
            "kind": start.get("kind"),
        })

    return sections


# --------------------------- Candidate generation ---------------------------

_ROLE_WORD_SPLIT_RE = re.compile(r"[\s.,]")
_CLAUSE_SPLIT_RE = re.compile(r";")

# A text span ending in a comma followed by ONE capitalized token — the
# signature of a name severed at a column break ("... Cottingham,Steve ").
_DANGLING_NAME_END_RE = re.compile(r",\s*[A-Z][A-Za-z'’\-]*$")

# Staff roles whose "Role: Name(s)" caption lists non-players only. Kept
# separate from _ROSTER_NOISE_WORDS because some of those words ("members",
# "captain") label actual players.
_NON_PLAYER_ROLES = {
    "coach", "manager", "advisor", "adviser", "sponsor", "director",
    "coordinator", "trainer", "instructor", "scorer", "statistician",
    "statistian", "assistant",
}

# Honorific titles marking adults (coaches/faculty) in rosters. Punctuation
# splitting tears "Mr.Campbell" into chunks "Mr" + "Campbell"; the chunk
# after a bare honorific is that adult's surname, not a student.
_HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "rev", "fr", "sr"}

# Photo-caption positional markers ("ROW 2:", "STANDING–", "KNEELING-",
# "CENTER TO RIGHT:", "FIRST ROW:") that prefix name runs. Stripped from
# each candidate chunk so they can't leak into names — either fused onto a
# real name ("KNEELING-Tommy Shaw") or rescued on their own as a fake name
# by the NER-miss fallback ("First Row"). Matching notes:
#   - OCR often glues words ("THiRDROW:", "TORIGHT"), so \s* joins word
#     pairs rather than \s+.
#   - Bare "ROW" requires a following digit, colon/dash, or end-of-chunk so
#     surnames like "Rowland"/"Rowe" can never match.
#   - Bare direction words (LEFT/RIGHT/CENTER alone) are deliberately NOT
#     stripped — only the compound "X TO Y" caption form — to avoid ever
#     eating an unusual real name.
_POSITION_MARKER_RE = re.compile(
    r"^(?:"
    r"(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|FRONT|BACK|TOP|BOTTOM|MIDDLE)\s*ROW\s*\d*"
    r"|ROW\s*\d*(?=\s*(?:[:：;–—-]|$))"
    r"|STANDING|KNEELING|SITTING|SEATED"
    r"|(?:CENTER|CENTRE|LEFT|RIGHT)\s*TO\s*(?:RIGHT|LEFT)"
    r")\s*(?:[:：;,.–—-]+\s*|\s+|$)",
    re.IGNORECASE,
)


def _strip_position_markers(s: str) -> str:
    """Remove leading positional caption markers, repeatedly (a chunk can
    stack them, e.g. "ROW 1: STANDING: Bob Smith")."""
    prev = None
    while prev != s:
        prev = s
        s = _POSITION_MARKER_RE.sub("", s, count=1).lstrip()
    return s


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
    if not t:
        return False

    # "Coach: D. Scott" / "Managers: Bobby Erwin, Rodney Evans." — a
    # NON-PLAYER role label with ':' applies to every name after it, so the
    # whole block goes, commas and all. Restricted to staff roles:
    # "MEMBERS: ..." (also a _ROSTER_NOISE_WORDS word) is an actual roster
    # and must survive, and captains are players.
    label_part = re.split(r"[:：]", t, maxsplit=1)
    if len(label_part) == 2:
        label_first = _ROLE_WORD_SPLIT_RE.split(label_part[0].strip(), maxsplit=1)[0].lower()
        if any(label_first in (w, w + "s", w + "es") for w in _NON_PLAYER_ROLES):
            return True

    if "," in t:
        return False
    words = [w for w in _ROLE_WORD_SPLIT_RE.split(t) if w]
    # "Name + role" photo caption with a staff role word ANYWHERE after the
    # first two tokens ("Tom Bynum Trainer", "Wilev Harris Defensive
    # Coordinator", "Hornsby Howell Head Coach", "Mel Groomes Defensie
    # Secondary nd Director of Scoutng") — a staff member's photo caption,
    # drop the whole block. Only scans tokens after the first two (the name)
    # so a student named e.g. "Coach Smith" doesn't false-positive.
    if len(words) >= 3:
        for w in words[2:]:
            if any(w.lower() in (r, r + "s", r + "es") for r in _NON_PLAYER_ROLES):
                return True
    first = words[0].lower() if words else ""
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


_PROSE_MIN_WORDS = 25
_PROSE_MAX_WORDS_PER_COMMA = 8

# Shorter biographical/narrative sentences (10-24 words) that start with
# articles, pronouns, or adverbs typical of prose but never of rosters.
_PROSE_STARTERS_RE = re.compile(
    r"^(?:he|she|they|his|her|a|an|the|this|that|in|on|at|as|"
    r"with|from|by|after|before|during|although|while|also)\s",
    re.IGNORECASE,
)


def _looks_like_prose_block(text: str) -> bool:
    """
    True for narrative paragraphs (season summaries, divider-page intros)
    that should never enter name extraction — NER picks embedded people
    (coaches) and even school names out of them. Roster captions can be just
    as long but are comma-dense (a comma every 2-3 words), while prose runs
    many words between commas.

    Also catches shorter biographical sentences (10+ words) that start with
    prose-typical function words and end with sentence punctuation — player
    profile bios in yearbooks like Ayantee 1975.
    """
    words = len(text.split())
    if words <= 8:
        return False
    # Short sentence heuristic: 10+ words, starts with a prose word, ends
    # with sentence punctuation → almost certainly a bio/narrative sentence,
    # never a roster line.
    if (
        words >= 10
        and _PROSE_STARTERS_RE.match(text)
        and text.rstrip().rstrip('"\'').endswith((".", "!", "?"))
    ):
        return True
    if words <= _PROSE_MIN_WORDS:
        return False
    commas = text.count(",") + text.count("，")
    return commas == 0 or words / commas > _PROSE_MAX_WORDS_PER_COMMA


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

    # Re-join names severed at a column break: when a caption wraps across
    # two side-by-side blocks, the first block can end mid-name with a lone
    # capitalized token right after a comma ("...,Steve ") and the next
    # block open with its continuation ("McCorkle, ..."). Requiring the
    # dangling token to immediately follow a comma keeps this from ever
    # gluing complete entries together (grid lines / full names never end
    # ",Word").
    merged: List[str] = []
    for t in texts:
        t = _normalize_space(t)
        if not t:
            continue
        if merged and _DANGLING_NAME_END_RE.search(merged[-1]) and t[:1].isupper():
            merged[-1] = merged[-1] + " " + t
        else:
            merged.append(t)

    cands: List[str] = []
    for t in merged:
        if header_norm and t.casefold() == header_norm:
            continue
        if _is_single_role_word_caption(t):
            continue
        if _looks_like_prose_block(t):
            continue
        for clause in _CLAUSE_SPLIT_RE.split(t):
            clause = _normalize_space(clause)
            if not clause or _is_role_caption_clause(clause):
                continue
            clause_start = len(cands)
            protected = _protect_initials(clause)
            skip_after_honorific = False
            for chunk in _SPLIT_ON_PUNCT.split(protected):
                chunk = _normalize_space(_restore_initials(chunk))
                chunk = _strip_position_markers(chunk)
                if not chunk:
                    continue
                if skip_after_honorific:
                    # Surname of an "Mr./Mrs. X" adult — not a student.
                    skip_after_honorific = False
                    continue
                chunk_toks = chunk.split()
                if (
                    chunk_toks
                    and chunk_toks[-1].lower() in _HONORIFICS
                    # honorific alone, or prose ending at the honorific's
                    # period ("coached by Mr", '"But Mr, Lawrence, ...') —
                    # but NOT a real name that happens to end in a suffix
                    # ("John Smith Sr")
                    and (
                        len(chunk_toks) == 1
                        or any(t[0].islower() or t.lower() in _CAPTION_STOPWORDS for t in chunk_toks[:-1])
                    )
                ):
                    # Next chunk opens with that adult's name — skip it.
                    skip_after_honorific = True
                    continue
                first_word = _ROLE_WORD_SPLIT_RE.split(chunk, maxsplit=1)[0].lower()
                if first_word in _ROSTER_NOISE_WORDS:
                    # A bare role word ("manager") is the role of the
                    # preceding comma-separated name ("Tommy Shaw,manager")
                    # — drop that name too. A role word starting a longer
                    # chunk ("Coach Campbell") names a non-player; drop the
                    # chunk but keep prior candidates.
                    if chunk.lower() in _ROSTER_NOISE_WORDS and len(cands) > clause_start:
                        cands.pop()
                    continue
                if chunk_toks and chunk_toks[-1].lower() in _NON_PLAYER_ROLES:
                    # "...basketball coach, Pete Campbell, fielded..." —
                    # the chunk after a role-word-final chunk is that staff
                    # member's name. (Checked AFTER the bare-role-word pop
                    # above, which must win for "Name,manager" chunks.)
                    skip_after_honorific = True
                    continue
                # An ALL-CAPS comma-free row of an even number of alphabetic
                # tokens (>=4) is a photo-caption grid row of First-Last
                # pairs ("GENE HENDERSON BLACKIE HAGER TOMMY COURSEY PUG
                # PEARMAN"). Split deterministically into pairs — NER over
                # the undelimited run misses unusual names ("PUG PEARMAN")
                # and its gap-fill refuses gaps > 2 tokens as ambiguous.
                letters = [c for c in chunk if c.isalpha()]
                caps_ratio = (
                    sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
                )
                if (
                    caps_ratio >= 0.8  # tolerate stray OCR lowercase ("FrED")
                    and len(chunk_toks) >= 4
                    and len(chunk_toks) % 2 == 0
                    and all(re.fullmatch(r"[A-Za-z'\-]+", t) for t in chunk_toks)
                ):
                    for i in range(0, len(chunk_toks), 2):
                        cands.append(_correct_candidate_text_pre_ner(
                            f"{chunk_toks[i]} {chunk_toks[i + 1]}"
                        ))
                    continue
                chunk = _strip_trailing_role_words(chunk)
                if chunk:
                    cands.append(_correct_candidate_text_pre_ner(chunk))
    return cands


# Position/class words that trail a caption's name ("KENNETH BARRIER Tackle
# Def. End", "RICKY WHISNANT Forward", "DEBRA CLONTZ Senior") — never part of
# the name, and left in place they make NER cut the name short ("Roy Lee"
# from "ROY LEE CORPENING Guard Def") or swallow the role ("Ricky Whisnant
# Forward").
# Everything after one of these words in a caption names the OPPONENT
# ("...tosses in two points against George Hildebran").
_OPPONENT_CUE_RE = re.compile(r"\s(?:against|vs\.?|versus)\s.*$", re.IGNORECASE)


def _strip_trailing_role_words(chunk: str) -> str:
    chunk = _OPPONENT_CUE_RE.sub("", chunk).strip()
    toks = chunk.split()
    # never strip a two-token chunk down to one: "Bill Strong" is a name
    while len(toks) > 2 and toks[-1].strip(".,").lower() in _POSITION_WORDS:
        toks.pop()
    return " ".join(toks)


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

    # Fuzzy per-word match for misOCR'd stylized titles (1970 Archway's
    # script font: "Harsity Honthall" -> varsity, "Baskethall" ->
    # basketball). Both words must be reasonably long so short words can't
    # accidentally fuzz onto a sport, and the edit budget is tight (1, or 2
    # for keywords of 9+ letters).
    for word in norm.split():
        if len(word) < 6:
            continue
        for kw in _SPORTS_SINGLE_WORD:
            if len(kw) < 6:
                continue
            thr = 2 if len(kw) >= 9 else 1
            if Levenshtein.distance(word, kw, score_cutoff=thr) <= thr:
                return True

    return False


def _looks_like_glued_caps_name(chunk: str) -> bool:
    """
    True for a lone ALL-CAPS alphabetic token (>=6 letters) that came
    through candidate splitting by itself — in a sports section this is
    almost always a roster name whose internal space OCR dropped
    ("MIMIGADD", "SISGROSE"). Sports keywords, role words, and position
    markers are excluded so labels can never slip through.
    """
    if not re.fullmatch(r"[A-Z]{6,}", chunk):
        return False
    low = chunk.lower()
    if low in _ROSTER_NOISE_WORDS or any(low.startswith(w) for w in _ROSTER_NOISE_WORDS):
        return False
    if _is_sports_section_header(chunk):
        return False
    if _POSITION_MARKER_RE.match(chunk):
        return False
    return True


_MC_PREFIX_RE = re.compile(r"^(Mc|Mac|O')(\w)")


def _normalize_name_casing(name: str) -> str:
    """
    Repair OCR-mangled casing inside otherwise mixed-case names:
    "DoT McKEE" -> "Dot McKee", "LoIs Cox" -> "Lois Cox". A token is
    rewritten only when MORE than a normal share of its letters are
    uppercase (ratio >= 0.5 with an uppercase beyond position 0), so
    legitimate internal capitals ("DeWanda", "McCoy") are untouched.

    Also fixes capital-I-for-lowercase-l OCR error in short tokens:
    "AI" -> "Al" (common OCR confusion between I and l).
    """
    out: List[str] = []
    for tok in name.split():
        letters = [c for c in tok if c.isalpha()]
        if len(letters) > 1:
            upper = sum(c.isupper() for c in letters)
            # Capital I for lowercase l: "AIbert" -> "Albert", "DoIores"
            # -> "Dolores". Only when I is medial between letters (not the
            # first letter, not a suffix like "II"/"III").
            if re.search(r"(?<=[A-Za-z])I(?=[a-z])", tok):
                tok = re.sub(r"(?<=[A-Za-z])I(?=[a-z])", "l", tok)
                letters = [c for c in tok if c.isalpha()]
                upper = sum(c.isupper() for c in letters)
            # Short all-caps first name ending in I: "AI" -> "Al" (OCR
            # confuses capital I with lowercase l). Not suffixes like "II".
            elif (
                len(tok) in (2, 3)
                and tok.isupper()
                and tok.endswith("I")
                and tok != "II"
                and tok not in {"JR", "SR", "III", "IV"}
            ):
                tok = tok[:-1] + "l"
            if upper / len(letters) >= 0.5 and any(c.isupper() for c in letters[1:]):
                fixed = tok.capitalize()
                fixed = _MC_PREFIX_RE.sub(lambda m: m.group(1) + m.group(2).upper(), fixed)
                tok = fixed
            # "Mcginnis"/"O'neal" (an ALL-CAPS source title-cased upstream)
            # -> "McGinnis"/"O'Neal"; "Mac" is left alone ("Mackey").
            tok = re.sub(r"^(Mc|O')([a-z])", lambda m: m.group(1) + m.group(2).upper(), tok)
        out.append(tok)
    return " ".join(out)


_POSITION_LABELS = {
    "running back", "quarterback", "linebacker", "wide receiver",
    "tight end", "defensive end", "offensive line", "defensive line",
    "safety", "cornerback", "fullback", "halfback", "tailback",
    "center", "guard", "tackle", "punter", "kicker",
    "point guard", "shooting guard", "small forward", "power forward",
    "pitcher", "catcher", "shortstop", "outfielder", "infielder",
    "goalie", "goalkeeper", "midfielder", "defender", "forward",
    "senior", "junior", "sophomore", "freshman",
}

_POSITION_WORDS = {
    w for label in _POSITION_LABELS for w in label.split()
} | {
    "def", "off", "defensive", "offensive", "end", "back", "line", "wing",
    "split", "tight", "wide", "half", "full", "receiver", "linebacker",
    "nose", "corner", "free", "strong", "flanker", "utility", "chief",
    "co-captain", "captain", "manager", "coach", "trainer",
    "senior", "junior", "sophomore", "freshman", "seniors", "juniors",
    "sophomores", "freshmen",
}


def _is_junk_name(name: str) -> bool:
    """
    Final sanity filter on NER output:
      - any token starting lowercase means a sentence fragment slipped
        through ("The end of the season"), never a roster name;
      - a "name" that itself matches a sports-section phrase ("Boy's
        Basketball", "Head Cheerleader") is a label, not a person;
      - a "name" that matches a football/basketball position label
        ("Running Back", "Quarterback") is never a person.
    """
    toks = name.split()
    if not toks:
        return True
    if any(t[0].islower() for t in toks):
        return True
    if name.lower() in _POSITION_LABELS:
        return True
    # a lone 1-2 letter token ("Pa" from "Philadelphia, Pa.") is never a name
    if len(toks) == 1 and len(toks[0].strip(".")) < 3:
        return True
    # position / class-year words only ("Tackle Def", "Guard Linebacker",
    # "Striving Seniors")
    lows = [t.strip(".,").lower() for t in toks]
    if all(t in _POSITION_WORDS for t in lows):
        return True
    if any(t in _CLASS_YEAR_WORDS for t in lows):
        return True
    # a team mascot NER took for a person ("Panther clings on for life")
    if len(toks) == 1 and lows[0] in _MASCOT_WORDS:
        return True
    return _is_sports_section_header(name)


_MASCOT_WORDS = {
    "panther", "panthers", "bulldog", "bulldogs", "tiger", "tigers", "eagle",
    "eagles", "lion", "lions", "bear", "bears", "wildcat", "wildcats",
    "hornet", "hornets", "devil", "devils", "aggie", "aggies", "ram", "rams",
    "knight", "knights", "falcon", "falcons", "hawk", "hawks", "cougar",
    "cougars", "warrior", "warriors", "raider", "raiders", "pirate",
    "pirates", "viking", "vikings", "spartan", "spartans", "trojan",
    "trojans", "wave", "indians", "cardinal", "cardinals", "bruin", "bruins",
    "mustang", "mustangs", "bronco", "broncos", "rebel", "rebels", "patriot",
    "patriots", "titan", "titans", "giant", "giants", "jaguar", "jaguars",
    "wolverine", "wolverines", "badger", "badgers", "owl", "owls", "bobcat",
    "bobcats", "cyclone", "cyclones", "hurricane", "hurricanes", "tornado",
    "demon", "demons", "saint", "saints", "crusader", "crusaders",
    "cavalier", "cavaliers", "colonel", "colonels", "general", "generals",
    "senator", "senators", "commodore", "commodores", "wolfpack", "wolves",
    "wolf", "gator", "gators", "seminole", "seminoles", "buccaneer",
    "buccaneers", "rocket", "rockets", "jet", "jets", "comet", "comets",
    "ranger", "rangers", "chief", "chiefs", "brave", "braves",
    "mountaineer", "mountaineers", "miner", "miners", "yellowjacket",
    "yellowjackets", "greenwave", "bluedevil", "bluedevils", "dog", "dogs",
    "cat", "cats", "pup", "pups", "puppettes",
}


_CLASS_YEAR_WORDS = {
    "senior", "seniors", "junior", "juniors", "sophomore", "sophomores",
    "freshman", "freshmen",
}


# ----------------------- OCR-variant name merging -----------------------

_VARIANT_MAX_EDIT_DISTANCE = 3


def _merge_ocr_variant_names(sections_names: List[Dict[str, Any]]) -> None:
    """
    Merge near-identical name pairs WITHIN a section that are almost surely
    the same person read twice by OCR (photo caption + roster grid), e.g.
    "Steve Kisncr"/"Steve Kisner", "Domnid Robinson"/"Donald Robinson".

    A pair merges only when ALL hold:
      - same token count and same first letter per token;
      - Levenshtein distance 1..3 over the whole name;
      - the two spellings appear in a DIFFERENT number of sections
        book-wide — the majority spelling wins. Equal counts mean no
        external evidence either way (could be two real teammates with
        similar names), so the pair is left alone.
    """
    counts: Counter = Counter()
    token_counts: Counter = Counter()
    for sec in sections_names:
        for n in set(sec.get("names", [])):
            counts[n] += 1
            for tok in n.split():
                token_counts[tok.lower()] += 1

    def _token_vote(a: str, b: str) -> Optional[str]:
        """Which of two same-shape spellings the book supports: at every
        token position where they differ, the token seen more often in
        OTHER names wins ("Erc Harris"/"Eric Harris": "Eric" also in "Eric
        Wallace"). None when the evidence ties or conflicts."""
        va = vb = 0
        for x, y in zip(a.split(), b.split()):
            if x.lower() == y.lower():
                continue
            cx = token_counts[x.lower()] - 1  # minus this name itself
            cy = token_counts[y.lower()] - 1
            if cx > cy:
                va += 1
            elif cy > cx:
                vb += 1
        if va and not vb:
            return a
        if vb and not va:
            return b
        return None

    for sec in sections_names:
        names = sec.get("names", [])
        drop: Set[str] = set()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if a in drop or b in drop:
                    continue
                ta, tb = a.split(), b.split()
                if len(ta) != len(tb):
                    continue
                if any(x[0].lower() != y[0].lower() for x, y in zip(ta, tb)):
                    continue
                dist = Levenshtein.distance(a, b)
                if not (1 <= dist <= _VARIANT_MAX_EDIT_DISTANCE):
                    continue
                if counts[a] == counts[b]:
                    # Equal section counts: let the book's other names vote
                    # on the spelling; still a tie -> two real people.
                    winner = _token_vote(a, b) if dist <= 2 else None
                    if winner is None:
                        continue
                    drop.add(b if winner == a else a)
                    continue
                drop.add(b if counts[a] > counts[b] else a)

        # Nickname vs formal given name within one section ("Ray Perry" /
        # "Raymond Perry", "Vince Butler" / "Vincent Butler"): the roster table
        # prints the formal name and the photo caption the short form, so the
        # same player lands twice. Keep the longer spelling. Edit distance is
        # too large for the rule above, and both spellings appear in the same
        # number of sections, so its majority test can't break the tie either.
        # Pairs in DIFFERENT sections are left alone — sections are allowed to
        # print the same person differently.
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if a in drop or b in drop:
                    continue
                ta, tb = a.split(), b.split()
                if len(ta) != len(tb) or len(ta) < 2:
                    continue
                if [t.lower() for t in ta[1:]] != [t.lower() for t in tb[1:]]:
                    continue
                ga, gb = ta[0].lower(), tb[0].lower()
                if ga == gb:
                    continue
                short = ga if len(ga) < len(gb) else gb
                longer = gb if short == ga else ga
                if len(short) >= 3 and longer.startswith(short):
                    drop.add(a if ga == short else b)

        if drop:
            sec["names"] = [n for n in names if n not in drop]


# ----------------------- Partial-name resolution -----------------------

# Common given-name short forms -> formal names, for matching a caption's
# "Charlie" to the roster's "Charles Davis". Deliberately conservative.
_NICKNAMES: Dict[str, Set[str]] = {
    "al": {"albert", "alfred", "alan", "allen", "alvin"},
    "bill": {"william"}, "billy": {"william"}, "will": {"william"}, "willie": {"william"},
    "bob": {"robert"}, "bobby": {"robert"}, "rob": {"robert"}, "robbie": {"robert"},
    "charlie": {"charles"}, "charley": {"charles"}, "chuck": {"charles"},
    "dick": {"richard"}, "rick": {"richard"}, "ricky": {"richard"}, "rich": {"richard"},
    "jim": {"james"}, "jimmy": {"james"}, "jimmie": {"james"},
    "tom": {"thomas"}, "tommy": {"thomas"},
    "mike": {"michael"}, "mickey": {"michael"},
    "dave": {"david"}, "davey": {"david"},
    "joe": {"joseph"}, "joey": {"joseph"},
    "ron": {"ronald"}, "ronnie": {"ronald"},
    "don": {"donald"}, "donnie": {"donald"},
    "ed": {"edward", "edwin", "edgar"}, "eddie": {"edward", "edwin", "edgar"},
    "ken": {"kenneth"}, "kenny": {"kenneth"},
    "steve": {"steven", "stephen"},
    "ray": {"raymond"}, "stan": {"stanley"}, "vince": {"vincent"},
    "dan": {"daniel"}, "danny": {"daniel"},
    "larry": {"lawrence"}, "jerry": {"gerald", "jerome"},
    "greg": {"gregory"}, "jeff": {"jeffrey", "jeffery"},
    "pat": {"patrick", "patricia"}, "patty": {"patricia"},
    "sam": {"samuel"}, "sammy": {"samuel"},
    "ben": {"benjamin"}, "benny": {"benjamin"},
    "fred": {"frederick", "alfred"}, "freddie": {"frederick", "alfred"},
    "hank": {"henry"}, "harry": {"harold", "henry"},
    "jack": {"john"}, "johnny": {"john"},
    "tony": {"anthony"}, "andy": {"andrew"}, "drew": {"andrew"},
    "chris": {"christopher", "christine"},
    "matt": {"matthew"}, "nick": {"nicholas"},
    "pete": {"peter"}, "phil": {"philip", "phillip"},
    "ted": {"theodore", "edward"}, "teddy": {"theodore", "edward"},
    "tim": {"timothy"}, "timmy": {"timothy"},
    "walt": {"walter"}, "wally": {"walter"},
    "gene": {"eugene"}, "herb": {"herbert"}, "art": {"arthur"},
    "bert": {"albert", "herbert", "robert"},
    "liz": {"elizabeth"}, "beth": {"elizabeth"}, "betty": {"elizabeth"}, "betsy": {"elizabeth"},
    "peggy": {"margaret"}, "maggie": {"margaret"}, "meg": {"margaret"},
    "sue": {"susan"}, "susie": {"susan"},
    "kathy": {"katherine", "kathleen"}, "kate": {"katherine"},
    "debbie": {"deborah", "debra"}, "deb": {"deborah", "debra"},
    "becky": {"rebecca"}, "cindy": {"cynthia"}, "jenny": {"jennifer"},
    "barb": {"barbara"}, "barbie": {"barbara"},
    "dot": {"dorothy"}, "dottie": {"dorothy"}, "kim": {"kimberly"},
}

_POSSESSIVE_RE = re.compile(r"['’]s$")


def _given_names_match(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    return b in _NICKNAMES.get(a, ()) or a in _NICKNAMES.get(b, ())


def _resolve_partial_names(
    sections_names: List[Dict[str, Any]],
    sports_idx: List[int],
    caps_singles: Optional[Dict[int, Set[str]]] = None,
) -> None:
    """
    Map partial names to the full names the book itself prints ("mapping":
    action captions say "Beck grounds out...", "Charlie lays up..." while the
    roster page prints "EDDIE BECK", "CHARLES DAVIS").

    Within each sports section, a one-token name (or an initial+surname) is
    replaced by a full name from the same section — or, failing that, from
    the neighbouring sports sections (the sub-sections of one sport sit side
    by side; the far end of the book is another team) — when exactly one
    person fits: surname match, or given-name match allowing common short
    forms (Charlie/Charles, Jimmy/James). Two candidates that share a surname
    and equivalent given names ("Charles Davis" / "Charlie Davis") are one
    person; the longer, more formal spelling wins. A glued caps caption
    ("ALHOLLAND", "EARNESTGASKINS") is matched against the concatenated full
    names, and a surname OCR-glued to a following "AND" ("Patrick
    Matthewsand") is repaired when the book prints "Patrick Matthews".
    Anything ambiguous or unknown is left exactly as printed — except a
    one-token name NER pulled out of an ALL-CAPS candidate (`caps_singles`):
    those are decorative lettering or jersey text ("AGGIETTES", "HILL") far
    more often than a real bare surname, so they are dropped unless
    resolved. (A glued caps token kept by the NER-miss fallback is NOT in
    that set: it stays, as before, for the portrait-index step to split.)
    """
    caps_singles = caps_singles or {}

    def _full(name: str) -> bool:
        toks = name.split()
        return len(toks) >= 2 and len(toks[0].rstrip(".")) > 1

    full_by_sec: Dict[int, List[str]] = {
        i: [n for n in sections_names[i].get("names", []) if _full(n)] for i in sports_idx
    }
    order = {i: k for k, i in enumerate(sports_idx)}

    def _neighbour_pool(i: int) -> List[str]:
        k = order[i]
        pool: List[str] = []
        for j in (k - 1, k + 1):
            if 0 <= j < len(sports_idx):
                pool.extend(full_by_sec[sports_idx[j]])
        return pool

    def _glued(tok: str, pool: List[str]) -> Optional[str]:
        t = re.sub(r"[^a-z]", "", tok.lower())
        if len(t) < 6:
            return None
        hits = []
        for n in pool:
            g = re.sub(r"[^a-z]", "", n.lower())
            if abs(len(g) - len(t)) <= 1 and Levenshtein.distance(g, t, score_cutoff=1) <= 1:
                hits.append(n)
        hits = list(dict.fromkeys(hits))
        return hits[0] if len(hits) == 1 else None

    def _pick(cands: List[str]) -> Optional[str]:
        cands = list(dict.fromkeys(cands))
        if not cands:
            return None
        first = cands[0]
        for c in cands[1:]:
            if c.split()[-1].lower() != first.split()[-1].lower():
                return None
            if not _given_names_match(c.split()[0], first.split()[0]):
                return None
        return max(cands, key=lambda n: (len(n.split()[0]), len(n)))

    def _matches(pool: List[str], surname: Optional[str], given: Optional[str], initial: Optional[str]) -> List[str]:
        hits: List[str] = []
        for n in pool:
            toks = n.split()
            if surname is not None:
                if toks[-1].lower() != surname.lower():
                    continue
                if initial and toks[0][0].lower() != initial.lower():
                    continue
                hits.append(n)
            elif given is not None and _given_names_match(toks[0], given):
                hits.append(n)
        return hits

    for i in sports_idx:
        sec = sections_names[i]
        names = sec.get("names", [])
        local_full = full_by_sec[i]
        pools = (local_full, _neighbour_pool(i))
        drop_unresolved = caps_singles.get(i, set())
        resolved: List[str] = []
        changed = False
        for name in names:
            toks = name.split()
            new = None
            if len(toks) >= 2 and _POSSESSIVE_RE.search(toks[-1]):
                # "Claude Warren's" (from "...Warren's shot") -> "Claude Warren"
                name = " ".join(toks[:-1] + [_POSSESSIVE_RE.sub("", toks[-1])])
                toks = name.split()
                changed = True
            if len(toks) == 1:
                tok = _POSSESSIVE_RE.sub("", toks[0])
                for pool in pools:
                    new = (
                        _pick(_matches(pool, tok, None, None))
                        or _pick(_matches(pool, None, tok, None))
                        or _glued(tok, pool)
                    )
                    if new:
                        break
                if not new:
                    if name in drop_unresolved:
                        changed = True
                        continue
                    if tok != toks[0]:
                        new = tok  # at least drop the possessive
            elif len(toks) == 2 and len(toks[0].rstrip(".")) == 1:
                for pool in pools:
                    new = _pick(_matches(pool, toks[1], None, toks[0][0]))
                    if new:
                        break
            elif len(toks) >= 2 and toks[-1].lower().endswith("and") and len(toks[-1]) > 6:
                # "Patrick Matthewsand" <- "PATRICK MATTHEWSAND RODNEY ..."
                trimmed = " ".join(toks[:-1] + [toks[-1][:-3]])
                for pool in pools:
                    if any(n.lower() == trimmed.lower() for n in pool):
                        new = trimmed
                        break
            if new and new != name:
                changed = True
                resolved.append(new)
            else:
                resolved.append(name)
        if changed:
            sec["names"] = dedupe_person_names(resolved)


# Common American given names and surnames (lower-case) — the "common
# sense" tie-breaker for an a<->o OCR confusion the book itself cannot settle
# ("Margan" -> "Morgan", "Raberts" -> "Roberts", "Jerame" -> "Jerome"). Only
# ever consulted for a token that is NOT itself in this list and whose a/o
# variant is; nothing else is touched.
_COMMON_NAME_TOKENS = set("""
james john robert michael william david richard charles joseph thomas
christopher daniel paul mark donald george kenneth steven edward brian
ronald anthony kevin jason matthew gary timothy jose larry jeffrey frank
scott eric stephen andrew raymond gregory joshua jerry dennis walter patrick
peter harold douglas henry carl arthur ryan roger joe juan jack albert
jonathan justin terry gerald keith samuel willie ralph lawrence nicholas roy
benjamin bruce brandon adam harry fred wayne billy steve louis jeremy aaron
randy howard eugene carlos russell bobby victor martin ernest phillip todd
jesse craig alan shawn clarence sean philip chris johnny earl jimmy antonio
danny bryan tony luis mike stanley leonard nathan dale manuel rodney curtis
norman allen marvin vincent glenn jeffery travis jeff chad jacob lee melvin
alfred kyle francis bradley jesus herbert frederick ray joel edwin don eddie
ricky troy randall barry alexander bernard mario leroy francisco marcus
micheal theodore clifford miguel oscar jay jim tom calvin alex jon ronnie
bill lloyd tommy leon derek warren darrell jerome floyd leo alvin tim wesley
gordon dean greg jorge dustin pedro derrick dan lewis zachary corey herman
maurice vernon roberto clyde glen hector shane ricardo sam rick lester
brent ramon charlie tyler gilbert gene marc reginald ruben brett angel
nathaniel rafael leslie edgar milton raul ben chester cecil duane franklin
andre elmer brad gabriel ron mitchell roland arnold harvey jared adrian karl
cory claude erik darryl jamie neil jessie christian javier fernando clinton
ted mathew tyrone darren lonnie lance cody julio kelly kurt allan nelson guy
clayton hugh max dwayne dwight armando felix jimmie everett jordan ian
wallace ken bob jaime casey alfredo alberto dave ivan johnnie sidney byron
julian isaac morris clifton willard daryl ross virgil andy marshall salvador
perry kirk sergio marion tracy seth kent terrance rene eduardo terrence
enrique freddie wade austin stuart fredrick arturo alejandro jackie joey
nick luther wendell jeremiah evan julius dana donnie otis shannon trevor
oliver luke homer gerard doug kenny hubert angelo shaun lyle matt lynn
alfonso orlando rex carlton ernesto cameron neal pablo lorenzo omar wilbur
blake grant horace roderick kerry abraham willis rickey jean ira andres
cesar johnathan malcolm rudolph damon kelvin rudy preston alton archie
marco wm pete randolph garry geoffrey jonathon felipe bennie gerardo ed
dominic robin loren delbert colin guillermo earnest lucas benny noel
spencer rodolfo myron edmund garrett salvatore cedric lowell gregg sherman
wilson devin sylvester kim roosevelt israel jermaine forrest wilbert leland
simon irving owen rufus woodrow sammy kristopher levi marcos gustavo jake
lionel marty taylor ellis dallas gilberto clint nicolas laurence ismael
orville drew jody ervin dewey al wilfred josh hugo ignacio caleb tomas
sheldon erick frankie stewart doyle darrel rogelio terence santiago
alonzo elias bert elbert ramiro conrad pat noah grady phil cornelius
lamar rolando clay percy dexter bradford merle darin amos terrell moses
irvin saul roman darnell randal tomas elijah lynn kip mel monty
mary patricia linda barbara elizabeth jennifer maria susan margaret dorothy
lisa nancy karen betty helen sandra donna carol ruth sharon michelle laura
sarah kimberly deborah jessica shirley cynthia angela melissa brenda amy anna
rebecca virginia kathleen pamela martha debra amanda stephanie carolyn
christine marie janet catherine frances ann joyce diane alice julie heather
teresa doris gloria evelyn jean cheryl mildred katherine joan ashley judith
rose janice kelly nicole judy christina kathy theresa beverly denise tammy
irene jane lori rachel marilyn andrea kathryn louise sara anne jacqueline
wanda bonnie julia ruby lois tina phyllis norma paula diana annie lillian
emily robin peggy crystal gladys rita dawn connie florence tracy edna
tiffany carmen rosa cindy grace wendy victoria edith kim sherry sylvia
josephine thelma shannon sheila ethel ellen elaine marjorie carrie charlotte
monica esther pauline emma juanita anita rhonda hazel amber eva debbie
april leslie clara lucille jamie joanne eleanor valerie danielle megan
alicia suzanne michele gail bertha darlene veronica jill erin geraldine
lauren cathy joann lorraine lynn sally regina erica beatrice dolores
bernice audrey yvonne annette june samantha marion dana stacy ana renee
ida vivian roberta holly brittany melanie loretta yolanda jeanette laurie
katie kristen vanessa alma sue elsie beth jeanne vicki carla tara rosemary
eileen terri gertrude lucy tonya ella stacey wilma gina kristin jessie
natalie agnes vera willie charlene bessie delores melinda pearl arlene
maureen colleen allison tamara joy georgia constance lillie claudia jackie
marcia tanya nellie minnie marlene heidi glenda lydia viola courtney marian
stella caroline dora jo vickie mattie terry maxine irma mabel marsha myrtle
lena christy deanna patsy hilda gwendolyn jennie nora margie nina cassandra
leah penny kay priscilla naomi carole brandy olga billie dianne tracey
leona jenny felicia sonia miriam velma becky bobbie violet kristina toni
misty mae shelly daisy ramona sherri erika katrina claire lindsey lindsay
geneva guadalupe belinda margarita sheryl cora faye ada natasha sabrina
isabel marguerite hattie harriet molly cecilia kristi brandi blanche sandy
rosie joanna iris eunice angie inez lynda madeline amelia alberta genevieve
monique jodi janie maggie kayla sonya jan lee kristine candace fannie
maryann opal alison yvette melody luz susie olivia flora shelley kristy
mamie lula lola verna beulah antoinette candice juana jeannette pam kelli
hannah whitney bridget karla celia latoya patty shelia gayle della vicky
lynne sheri marianne kara jacquelyn erma blanca myra leticia pat krista
roxanne angelica johnnie robyn francis adrienne rosalie alexandra brooke
bethany sadie bernadette traci jody kendra jasmine nichole rachael chelsea
mable ernestine muriel marcella elena krystal angelina nadine kari estelle
dianna paulette lora mona doreen rosemarie angel desiree antonia hope
ginger janis betsy christie freda mercedes meredith lynette teri cristina
eula leigh meghan sophia eloise rochelle gretchen cecelia raquel henrietta
alyssa jana kelley gwen kerry jenna tricia laverne olive alexis tasha
silvia elvira casey delia sophie kate patti lorena kellie sonja lila lana
darla may mindy essie mandy lorene elsa josefina jeannie miranda dixie lucia
marta faith lela johanna shari camille tami shawna elisa ebony melba ora
nettie tabitha ollie jaime winifred kristie marina alisha aimee rena myrna
marla tammie latasha bonita patrice ronda sherrie addie francine deloris
stacie adriana cheri shelby abigail celeste jewel cara adele rebekah lucinda
dorthy chris effie trina reba shawn sallie aurora lenora etta lottie kerri
trisha nikki estella francisca josie tracie marissa karin brittney janelle
lourdes laurel helene fern elva corinne kelsey ina bettie elisabeth aida
caitlin ingrid iva eugenia christa goldie cassie maude jenifer therese
frankie dena lorna janette latonya candy morgan consuelo tamika rosetta
debora cherie polly dina jewell fay jillian dorothea nell trudy esperanza
patrica kimberley shanna helena carolina cleo stefanie rosario ola janine
mollie lupe alisa lou maribel susanne bette susana elise cecile isabelle
lesley jocelyn paige joni rachelle leola daphne alta ester petra graciela
imogene jolene keisha lacey glenna gabriela cecelia carmela nona margarita
smith johnson williams brown jones miller davis garcia rodriguez wilson
martinez anderson taylor thomas hernandez moore martin jackson thompson
white lopez lee gonzalez harris clark lewis robinson walker perez hall
young allen sanchez wright king scott green baker adams nelson hill ramirez
campbell mitchell roberts carter phillips evans turner torres parker
collins edwards stewart flores morris nguyen murphy rivera cook rogers
morgan peterson cooper reed bailey bell gomez kelly howard ward cox diaz
richardson wood watson brooks bennett gray james reyes cruz hughes price
myers long foster sanders ross morales powell sullivan russell ortiz
jenkins gutierrez perry butler barnes fisher henderson coleman simmons
patterson jordan reynolds hamilton graham kim gonzales alexander ramos
wallace griffin west cole hayes chavez gibson bryant ellis stevens murray
ford marshall owens mcdonald harrison ruiz kennedy wells alvarez woods
mendoza castillo olson webb washington tucker freeman burns henry vasquez
snyder simpson crawford jimenez porter mason shaw gordon wagner hunter
romero hicks dixon hunt palmer robertson black holmes stone meyer boyd
mills warren fox rose rice moreno schmidt patel ferguson nichols herrera
medina ryan fernandez weaver daniels stephens gardner payne kelley dunn
pierce arnold tran spencer peters hawkins grant hansen castro hoffman hart
elliott cunningham knight bradley carroll hudson duncan armstrong berry
andrews johnston ray lane riley carpenter perkins aguilar silva richards
willis matthews chapman lawrence garza vargas watkins wheeler larson carlson
harper george greene burke guzman morrison munoz jacobs obrien lawson
franklin lynch bishop carr salazar austin mendez gilbert jensen williamson
montgomery harvey oliver howell dean hanson weber garrett sims burton
fuller soto mccoy welch chen schultz walters reid fields walsh little
fowler bowman davidson may day schneider newman brewer lucas holland wong
banks santos curtis pearson delgado valdez pena rios douglas sandoval
barrett hopkins keller guerrero stanley bates alvarado beck ortega wade
estrada contreras barnett caldwell santiago lambert powers chambers nunez
craig leonard lowe rhodes byrd gregory shelton frazier becker maldonado
fleming vega sutton cohen jennings parks mcdaniel watts barker norris
vaughn vazquez holt schwartz steele benson neal dominguez horton terry
wolfe hale lyons graves haynes miles park warner padilla bush thornton
mccarthy mann zimmerman erickson fletcher mckinney page dawson joseph
marquez reeves klein espinoza baldwin moran love robbins higgins ball cortez
le griffith bowen sharp cummings ramsey hardy swanson barber acosta luna
chandler blair daniel cross simon dennis oconnor quinn gross navarro moss
fitzgerald doyle mclaughlin rojas rodgers stevenson singh yang figueroa
harmon newton paul manning garner mcgee reese francis burgess adkins
goodman curry brady christensen potter walton goodwin mullins molina webster
fischer campos avila sherman todd chang blake malone wolf hodges juarez
gill farmer hines gallagher duran hubbard cannon miranda wang saunders tate
mack hammond carrillo townsend wise ingram barton mejia ayala schroeder
hampton rowe parsons frank waters strickland osborne maxwell chan deleon
norman harrington casey patton logan bowers mueller glover floyd hartman
buchanan cobb french kramer mccormick clarke tyler gibbs moody conner
sparks mcguire leon bauer norton pope flynn hogan robles salinas yates
lindsey lloyd marsh mcbride owen solis pham lang pratt lara brock ballard
trujillo shaffer drake roman aguirre morton stokes lamb pacheco patrick
cochran shepherd cain burnett hess li cervantes olsen briggs ochoa cabrera
velasquez montoya roth meyers cardenas fuentes weiss hoover wilkins nicholson
underwood short carson morrow colon holloway summers bryan petersen
mckenzie serrano wilcox carey clayton poole calderon gallegos greer rivas
guerra decker collier wall whitaker bass flowers davenport conley houston
huff copeland hood monroe massey roberson combs franco larsen pittman
randall skinner wilkinson kirby cameron bridges anthony wong dalton
huffman kane spears carter atkins nash mcclure oneal wooten sherman
holland hairston hampton pinnex boone barnhill bellamy blue coefield
kendrick reliford westmoreland ragsdale feaster stakes lilly bloomfield
porchia simpson christian jones medley mckie bell aiken merrick pridgen
crosby simmons wall pride bennett fuller alston bostick tuck king jackson
""".split())


def _ao_variants(word: str) -> List[str]:
    """All words obtained by flipping any subset of a/o letters (bounded)."""
    pos = [i for i, ch in enumerate(word) if ch in "ao"]
    if not pos or len(pos) > 4:
        return []
    out = []
    for mask in range(1, 1 << len(pos)):
        chars = list(word)
        for k, i in enumerate(pos):
            if mask >> k & 1:
                chars[i] = "o" if chars[i] == "a" else "a"
        out.append("".join(chars))
    return out


def _vote_correct_ao(sections_names: List[Dict[str, Any]], sports_idx: List[int]) -> None:
    """
    Fix the one OCR confusion the italic roster type produces constantly —
    'a' read for 'o' or vice versa ("Thamas", "Baone", "Halland") — using
    the book as its own dictionary: a name token (>= 4 letters) that occurs
    ONCE in the book's sports names is rewritten to a token that occurs at
    least TWICE and differs from it only by a<->o substitutions. Nothing
    else is ever changed (so "Dan"/"Don", both real and short, are safe,
    and a genuinely rare surname without an a/o twin is left as printed).
    """
    token_counts: Counter = Counter()
    for i in sports_idx:
        for n in set(sections_names[i].get("names", [])):
            for tok in n.split():
                token_counts[tok] += 1

    def _ao_key(tok: str) -> str:
        return tok.lower().replace("a", "o")

    by_key: Dict[str, List[str]] = {}
    for tok, c in token_counts.items():
        if c >= 2 and len(tok) >= 4:
            by_key.setdefault(_ao_key(tok), []).append(tok)

    fixes: Dict[str, str] = {}
    for tok, c in token_counts.items():
        if c != 1 or len(tok) < 4:
            continue
        cands = [t for t in by_key.get(_ao_key(tok), []) if t != tok and t[0] == tok[0]]
        if len(cands) == 1:
            fixes[tok] = cands[0]
            continue
        if cands or tok.lower() in _COMMON_NAME_TOKENS or not re.fullmatch(r"[A-Z][a-z]+", tok):
            continue
        # No evidence in the book: "common sense" — the token isn't a name
        # anyone has, but exactly one a/o variant of it is a common name.
        variants = _ao_variants(tok.lower())
        common = [v for v in variants if v in _COMMON_NAME_TOKENS]
        if len(common) == 1:
            fixes[tok] = common[0].capitalize()
    if not fixes:
        return
    for i in sports_idx:
        sec = sections_names[i]
        new_names = [" ".join(fixes.get(t, t) for t in n.split()) for n in sec.get("names", [])]
        if new_names != sec.get("names", []):
            sec["names"] = dedupe_person_names(new_names)


# ------------------------------ Staff exclusion ------------------------------

_ROLE_WORD_RE = re.compile(
    r"\b(?:head\s+)?(?:coach|coaches|coached|manager|managers|trainer|trainers|sponsor|sponsors|"
    r"advisor|advisors|adviser|advisers|instructor|instructors|director|directors|"
    r"statistician|statistian|scorer|coordinator|coordinators)\b",
    re.IGNORECASE,
)


def _collect_staff_candidates(compiled_clean: Dict[str, Any]) -> List[str]:
    """
    Text spans anywhere in the book that name STAFF: any block/line
    containing a role word ("Hornsby Howell Head Coach", "Coach: Mel
    Groomes", "Our Puppette Coaches are GLENN CHILDRES and LAWRENCE
    CARPENTER", "A word from his coach, Warren Reynolds") plus each name of
    a sport-labelled coaches' directory ("FOOTBALL Hornsby Howell BASKETBALL
    Warren Reynolds ..."). NER over these gives the staff name set that
    _drop_staff_names removes from every sports section, so a coach never
    lands in a roster just because one page prints him without his title.
    """
    out: List[str] = []
    for page in compiled_clean.get("pages", []):
        for blk in _page_blocks(page):
            raw = blk.get("block_content", "")
            if not isinstance(raw, str) or not raw.strip():
                continue
            for _label, text in _split_sport_labelled_block(raw):
                out.append(text)
            for line in raw.split("\n"):
                out.extend(_staff_spans(_normalize_space(line)))
    return out


_STAFF_WINDOW = 4
_STAFF_LIST_MAX = 14


def _staff_spans(text: str) -> List[str]:
    """
    The part of a line that names staff: a window of a few words around
    each role word ("Hornsby Howell Head Coach", "his coach, Warren
    Reynolds", "COACH GROOMES (Above) WITH SENIORS AL HOLLAND, ..." -> only
    "COACH GROOMES (Above) WITH"), or, after "Coach:" / "Coaches are" /
    "TRAINERS:", the list that follows ("Art Word and Chris Allen"). Players
    named further along the same caption stay out.
    """
    if not text:
        return []
    toks = text.split()
    spans: List[str] = []
    for i, tok in enumerate(toks):
        if not _ROLE_WORD_RE.fullmatch(tok.strip(",.;:()")):
            continue
        prev = toks[i - 1].lower().strip(",.;:()") if i > 0 else ""
        if tok.lower().startswith("scorer") and prev in ("leading", "top", "high", "second", "third", "best", "our"):
            continue  # "Leading scorer Ricky Whisnant" is a player
        after = toks[i + 1] if i + 1 < len(toks) else ""
        if tok.endswith(":") or after.lower() in ("are", "is", "were", "was", "-") or after in (":", "-"):
            start = i + 1 + (0 if tok.endswith(":") else 1)
            end = start
            while end < len(toks) and end - start < _STAFF_LIST_MAX:
                nxt = toks[end]
                if end > start and _ROLE_WORD_RE.fullmatch(nxt.strip(",.;:()")):
                    break  # next "Manager: ..." label
                end += 1
            spans.append(" ".join(toks[start:end]))
        else:
            # window around the role word, never across a ';' and, after
            # the role, only across the comma that directly follows it
            # ("his coach, Warren Reynolds") — "Mr. Davis, coach; Stan
            # Patterson" and "Burkhead, Manager; Wiles, Patterson" list
            # players after the separator.
            lo = i
            while lo > 0 and i - lo < _STAFF_WINDOW and not toks[lo - 1].endswith(";"):
                lo -= 1
            hi = i + 1
            if not tok.endswith((";", ".")):
                while hi < len(toks) and hi - (i + 1) < _STAFF_WINDOW:
                    hi += 1
                    prev_tok = toks[hi - 1]
                    if prev_tok.endswith(";") or (prev_tok.endswith(",") and hi - 1 > i + 1):
                        break
                    if prev_tok.endswith(",") and hi - 1 == i + 1 and not tok.endswith(","):
                        break
            spans.append(" ".join(toks[lo:hi]))
    return [sp for sp in spans if sp.strip()]


def _drop_staff_names(sections_names: List[Dict[str, Any]], sports_idx: List[int], staff: Set[str]) -> None:
    """Remove staff (see _collect_staff_candidates) from the sports sections:
    exact full-name matches, and one-token names equal to a staff member's
    surname or given name."""
    if not staff:
        return
    full = {n.lower() for n in staff}
    parts = {tok.lower() for n in staff for tok in n.split() if len(tok) > 2}
    for i in sports_idx:
        sec = sections_names[i]
        kept = []
        for n in sec.get("names", []):
            low = n.lower()
            toks = low.split()
            if low in full or (len(toks) == 1 and toks[0] in parts):
                continue
            kept.append(n)
        if len(kept) != len(sec.get("names", [])):
            sec["names"] = kept


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
    write_json(
        {"sections": [{k: v for k, v in sec.items() if k != "kind"} for sec in sections]},
        sections_text_path,
    )
    print(f"[names] wrote {sections_text_path.name}  sections={len(sections)}")

    # Sports-only reduced view (decide sports sections BEFORE NER).
    # A section is sports when its own title names a sport, OR when it lies
    # inside the book's sports zone: after a title-only "SPORTS"/"ATHLETICS"
    # divider page and before the next title-only non-sports divider
    # ("CLASSES", "ORGANIZATIONS", ...). See _marker_kind.
    sports_section_mask: List[bool] = []
    in_sports_zone = False
    for sec in sections:
        kind = sec.get("kind")
        if kind == "sports_divider":
            in_sports_zone = True
        elif kind == "other_divider":
            in_sports_zone = False
        sports_section_mask.append(
            in_sports_zone
            or _is_sports_section_header(sec.get("header", ""))
            or _looks_like_sports_title_block(sec.get("header", ""))
        )
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

    # Staff spans (coaches, managers, ...) — NER'd in the same batch and
    # excluded from every sports section afterwards.
    staff_candidates = _collect_staff_candidates(compiled_clean)
    staff_names: Set[str] = set()

    # Run NER once over sports candidates (or skip if none)
    section_names_raw: List[List[str]] = [[] for _ in sections]
    # one-token names NER pulled out of an ALL-CAPS candidate, per section
    # (see _resolve_partial_names: kept only if they resolve to a person)
    caps_singles: Dict[int, Set[str]] = {}
    caps_counts = _image_caps_tokens_book(compiled_clean)
    if flat_candidates:
        names_per_candidate = extract_person_names_per_candidate(
            flat_candidates + staff_candidates, verbose=verbose_ner
        )
        for names_here in names_per_candidate[len(flat_candidates):]:
            for n in names_here:
                if not _is_junk_name(n):
                    staff_names.add(_normalize_name_casing(n))
        names_per_candidate = names_per_candidate[: len(flat_candidates)]

        # Group back to original section indices
        for cand, idx, names_here in zip(flat_candidates, flat_section_idx, names_per_candidate):
            kept = [
                _normalize_name_casing(n) for n in names_here if not _is_junk_name(n)
            ]
            if (
                kept
                and _looks_all_caps(cand)
                # a plausible glued name ("SISGROSE") is kept whichever
                # path found it, unless it is jersey/mascot lettering
                and not (
                    _looks_like_glued_caps_name(cand)
                    and not _is_decorative_caps(cand, caps_counts)
                )
            ):
                for n in kept:
                    if len(n.split()) == 1:
                        caps_singles.setdefault(idx, set()).add(n)
            if (
                not kept
                and _looks_like_glued_caps_name(cand)
                and not _is_decorative_caps(cand, caps_counts)
            ):
                # A single glued ALL-CAPS token in a roster line
                # ("MIMIGADD" for "MIMI GADD") — keep it (normalized) so
                # the person isn't silently lost; the portrait-index
                # resolution step can split it against real full names.
                kept = [cand.capitalize()]
            section_names_raw[idx].extend(kept)

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

    _merge_ocr_variant_names(sections_names)
    _resolve_partial_names(sections_names, sports_section_indices, caps_singles)
    _vote_correct_ao(sections_names, sports_section_indices)
    _drop_staff_names(sections_names, sports_section_indices, staff_names)

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

    