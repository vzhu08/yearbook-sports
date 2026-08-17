# src/name_resolution.py
# -*- coding: utf-8 -*-
"""
Resolve partial names in sports sections against a class-portrait index.

Many yearbooks caption sports photos with surnames only ("Fuquay gets ready
to tee off", "Sitting: Ingram, Paisley, Lloyd...") or initial+surname
("F. Sauls"). The class-portrait pages of the same book list every student's
full name, so they serve as the authoritative roster to resolve against.

Inputs:
  - the sports book dir's sports_sections.json + sections_with_text.json
    (produced by src.name_extraction.extract_names)
  - a portraits compiled_ocr.json (produced by src.text_extraction.extract_text
    over the book's class-portrait page range)

Output (written next to sports_sections.json):
  - sports_sections.resolved.json —
      "sections": same sections with plain name strings (resolved partials
        replaced by the full name, unmatched partials kept as printed), plus
        a final "NEEDS MAPPING" section listing each partial name that has
        several possible people, with all candidates;
      "details": per-name records {"name", "status": full|resolved|
        ambiguous|no_match, "resolved", "candidates"} incl. caption-lead
        recoveries (surnames opening an action caption, validated against
        the index so junk like "Devils" cannot enter).
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from rapidfuzz.distance import Levenshtein

from src.common.io_utils import read_json, write_json
from src.name_extraction import _given_names_match
from src.name_ner import _CAPTION_STOPWORDS

# --------------------------- Index building ---------------------------

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

# Portrait-page block labels that never contain portrait names.
_INDEX_SKIP_LABELS = {"image", "number", "footer", "header"}


_INVERTED_RE = re.compile(r"^([A-Z][A-Za-z'’\-]+)\s*[.,]\s+([A-Z][A-Za-z'’\-]+(?:\s+[A-Z][A-Za-z'’\-.]*)?)\s*$")


def _clean_portrait_line(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip(" ·.,")
    # index-style portrait captions "Tyler. Bettye" / "Wright, Cassandra"
    m = _INVERTED_RE.match(s)
    if m:
        s = f"{m.group(2)} {m.group(1)}"
    # capital I between lowercase letters is OCR'd lowercase l ("StrickIand")
    s = re.sub(r"(?<=[a-z])I(?=[a-z])", "l", s)
    # merge a stray space splitting one name token: "Lit tle" -> "Little"
    s = re.sub(r"\b([A-Z][a-z]+) ([a-z]{2,4})\b", r"\1\2", s)
    toks = s.split()
    while toks and toks[-1].strip(".,").lower() in _NAME_SUFFIXES:
        toks.pop()
    return " ".join(toks)


def _is_full_name_line(s: str) -> bool:
    toks = s.split()
    if not (2 <= len(toks) <= 4):
        return False
    if re.search(r"\d", s):
        return False
    if len(toks[-1].strip(".")) < 2:
        return False  # "Carolyn G", "Mary B": a cut-off line, not a surname
    return all(t[0].isupper() for t in toks if t[0].isalpha())


def _fix_erratic_caps(name: str) -> str:
    """Normalize small-caps OCR casing ("NEAL WesTMOReLANd" -> "Neal
    Westmoreland", "MaRGAReT SiKES" -> "Margaret Sikes"). Applied to index
    names, which are display/lookup values, so title-casing is safe."""
    out = []
    for tok in name.split():
        letters = [c for c in tok if c.isalpha()]
        if len(letters) > 1 and sum(c.isupper() for c in letters) >= 2:
            fixed = tok.capitalize()
            fixed = re.sub(r"^(Mc|Mac|O')(\w)", lambda m: m.group(1) + m.group(2).upper(), fixed)
            tok = fixed
        out.append(tok)
    return " ".join(out)


# Leading run of 2-4 ALL-CAPS tokens at the start of a senior portrait
# block, before the mixed-case trait line ("IRIS JEAN OVERCASH Placid,
# Pleasing, Patient." -> "IRIS JEAN OVERCASH").
_LEADING_CAPS_NAME_RE = re.compile(
    r"^([A-Z][A-Z'\-]{1,}(?:\s+[A-Z][A-Z'\-]{1,}){1,3})(?=\s+[A-Z][a-z]|\s*$)"
)


def _page_text_lines(page: Dict[str, Any]) -> List[str]:
    """Text lines of a page: block contents when the layout pass ran, else
    the raw OCR lines (pages OCR'd with extract_text(layout=False))."""
    blocks = page.get("parsing_res_list") or []
    if blocks:
        out: List[str] = []
        for blk in blocks:
            if blk.get("block_label") in _INDEX_SKIP_LABELS:
                continue
            content = blk.get("block_content") or ""
            if ":" in content:
                continue  # officer/marshal captions, not portrait names
            out.extend(content.split("\n"))
        return out
    ocr = page.get("overall_ocr_res") or {}
    return [str(t) for t in (ocr.get("rec_texts") or []) if ":" not in str(t)]


def build_student_index(
    portraits_compiled: Dict[str, Any],
    pages: Optional[set] = None,
) -> Dict[str, List[str]]:
    """
    Extract full student names from a portraits compiled_ocr.json and index
    them by lowercase surname.

    Portrait layouts handled (one book may mix all of them):
      - one name per block (archway senior pages: figure_title per portrait)
      - grid rows of First-Last pairs ("Wilton Baker Paula Batts ...")
      - all-caps name followed by mixed-case description on the same line
        (1945 Acorn senior pages: "IRIS JEAN OVERCASH Placid, Pleasing, ...")
      - comma-separated full-name lists (grade/homeroom group captions:
        "Virginia Armstrong, Wallace Burgess, Joe Church, ...")
    """
    full_names: set = set()
    for page in portraits_compiled.get("pages", []):
        if pages is not None and page.get("page_number") not in pages:
            continue
        for _once in (0,):
            for raw_ln in _page_text_lines(page):
                # all-caps leading name before a mixed-case description
                m = _LEADING_CAPS_NAME_RE.match(raw_ln.strip())
                if m and not _is_full_name_line(_clean_portrait_line(raw_ln)):
                    cand = _clean_portrait_line(m.group(1))
                    if _is_full_name_line(cand):
                        full_names.add(_fix_erratic_caps(cand))
                        continue

                # comma-separated name list, or a single name
                for seg in re.split(r"[,，]", raw_ln):
                    ln = _clean_portrait_line(seg)
                    if not ln:
                        continue
                    if _is_full_name_line(ln):
                        full_names.add(_fix_erratic_caps(ln))
                        continue
                    # grid row of several "First Last" pairs
                    toks = ln.split()
                    if (
                        len(toks) >= 4
                        and len(toks) % 2 == 0
                        and not re.search(r"\d", ln)
                        and all(t[0].isupper() for t in toks if t[0].isalpha())
                    ):
                        for i in range(0, len(toks), 2):
                            full_names.add(
                                _fix_erratic_caps(f"{toks[i]} {toks[i + 1]}")
                            )

    index: Dict[str, set] = defaultdict(set)
    for n in full_names:
        index[n.split()[-1].lower()].add(n)
    return {k: sorted(v) for k, v in sorted(index.items())}


def _all_names(index: Dict[str, List[str]]) -> List[str]:
    return sorted({n for v in index.values() for n in v})


# ----------------------------- Resolution -----------------------------

class NameResolver:
    def __init__(self, index: Dict[str, List[str]]):
        self.index = index

    def _fuzzy_surname(self, surname: str) -> List[str]:
        s = surname.lower()
        if s in self.index:
            return self.index[s]
        if len(s) < 5:
            return []  # "Ben" ~ "Bean", "Reba" ~ "Rena": too short to fuzz
        # OCR-tolerant fallback ("Englehardt" -> "Engelhardt"); 2 edits only
        # for long surnames — at 8 letters "Charlene" is 2 from "Charles"
        budget = 2 if len(s) >= 10 else 1
        hits: List[str] = []
        for k, names in self.index.items():
            if abs(len(k) - len(s)) <= budget and Levenshtein.distance(
                k, s, score_cutoff=budget
            ) <= budget:
                hits.extend(names)
        return sorted(hits)

    def resolve(self, surname: str, initial: Optional[str] = None) -> List[str]:
        matches = self._fuzzy_surname(surname)
        if initial and len(matches) > 1:
            # match against first AND middle names: "G. Pierce" should find
            # "Eleanor Gayle Pierce"
            narrowed = [
                m for m in matches
                if any(t[0].lower() == initial.lower() for t in m.split()[:-1])
            ]
            if narrowed:
                matches = narrowed
        return matches

    def resolve_given(self, given: str) -> List[str]:
        """Full names whose FIRST name is `given` (a caption printing first
        names only: "CASSANDRA", "REBA"), allowing common short forms
        (Debbie/Deborah, Charlie/Charles)."""
        return sorted({
            n for n in _all_names(self.index) if _given_names_match(n.split()[0], given)
        })

    def resolve_glued(self, token: str) -> List[str]:
        """
        Match a glued OCR token ("Mimigadd") against index names whose
        concatenated tokens equal it (edit distance <=1 for OCR tolerance):
        "Mimi Gadd" -> "mimigadd". The index is the only safe way to split
        such tokens — every possible 2-way split looks name-shaped to NER.
        """
        s = re.sub(r"[^a-z]", "", token.lower())
        if len(s) < 6:
            return []
        hits = []
        for names in self.index.values():
            for m in names:
                glued = re.sub(r"[^a-z]", "", m.lower())
                if abs(len(glued) - len(s)) <= 1 and Levenshtein.distance(
                    glued, s, score_cutoff=1
                ) <= 1:
                    hits.append(m)
        if hits:
            return sorted(set(hits))

        # Fallback: the glued token ENDS with a known surname and the
        # remainder is a plausible nickname the index doesn't know
        # ("Sisgrose" -> "Sis" + known surname "Grose"). Returns the split
        # in its printed form rather than the portrait full name, since the
        # nickname is what the yearbook printed.
        for surname in self.index:
            if len(surname) >= 4 and s.endswith(surname):
                first = s[: -len(surname)]
                if len(first) >= 2:
                    hits.append(f"{first.capitalize()} {surname.capitalize()}")
        return sorted(set(hits)) if len(set(hits)) == 1 else []


# Leading capitalized word of a sentence-shaped action caption
# ("Fuquay gets ready to tee off.").
_CAPTION_LEAD_RE = re.compile(r"^([A-Z][a-z]+)\s+[a-z]")

# Caption openers that are never player surnames; index validation already
# rejects most junk, this guards against a student surname colliding with a
# common caption opener.
_CAPTION_LEAD_STOP = {
    "the", "a", "an", "coach", "coaches", "mr", "mrs", "miss", "seniors",
    "managers", "manager", "advisor", "adviser", "head", "alternates",
}


def _cross_section_narrow(matches: List[str], seen_full: set) -> List[str]:
    """
    Narrow an ambiguous surname using book-level evidence: if exactly one
    candidate's first+last name already appears as a FULL name somewhere in
    the book's sports sections (e.g. Athletic Honors names "Mickey
    Bradshaw"), that candidate wins.
    """
    if len(matches) <= 1:
        return matches
    hits = [
        m for m in matches
        if any(
            m.split()[0].lower() in f and m.split()[-1].lower() in f
            for f in seen_full
        )
    ]
    return hits if len(hits) == 1 else matches


def _entry(name: str, matches: List[str], was_full: bool = False) -> Dict[str, Any]:
    if was_full:
        return {"name": name, "status": "full", "resolved": name, "candidates": []}
    # "Bettye Tyler" / "Bettye L. Tyler" (the same student on two portrait
    # pages, or read twice) are one person: collapse candidates that share
    # first name AND surname, keeping the fullest spelling.
    if len(matches) > 1:
        groups: Dict[tuple, List[str]] = {}
        for m in matches:
            toks = m.split()
            groups.setdefault((toks[0].lower(), toks[-1].lower()), []).append(m)
        matches = [max(g, key=len) for g in groups.values()]
    if len(matches) == 1:
        return {"name": name, "status": "resolved", "resolved": matches[0], "candidates": matches}
    if not matches:
        return {"name": name, "status": "no_match", "resolved": None, "candidates": []}
    return {"name": name, "status": "ambiguous", "resolved": None, "candidates": matches}


def resolve_sections(
    sports_sections: Dict[str, Any],
    sections_with_text: Dict[str, Any],
    index: Dict[str, List[str]],
) -> Dict[str, Any]:
    resolver = NameResolver(index)
    texts_by_header = {
        s["header"]: s.get("texts", [])
        for s in sections_with_text.get("sections", [])
    }

    # Book-level full names for cross-section disambiguation
    seen_full: set = set()
    for sec in sports_sections.get("sections", []):
        for name in sec.get("names", []):
            toks = name.split()
            if len(toks) >= 2 and not toks[0].endswith("."):
                seen_full.add(name.lower())

    out_sections: List[Dict[str, Any]] = []
    for sec in sports_sections.get("sections", []):
        entries: List[Dict[str, Any]] = []
        seen_surnames: set = set()

        for name in sec.get("names", []):
            toks = name.split()
            if len(toks) == 1:
                # exact surname, then first name ("Cassandra"), then a
                # fuzzy surname, then a glued "FirstLast" token
                m = _cross_section_narrow(index.get(toks[0].lower(), []), seen_full)
                if not m:
                    m = resolver.resolve_given(toks[0])
                if not m:
                    m = _cross_section_narrow(resolver.resolve(toks[0]), seen_full)
                if not m:
                    m = resolver.resolve_glued(toks[0])
                seen_surnames.add(toks[0].lower())
                entries.append(_entry(name, m))
            elif len(toks) == 2 and toks[0].endswith("."):
                m = _cross_section_narrow(
                    resolver.resolve(toks[1], initial=toks[0][0]), seen_full
                )
                seen_surnames.add(toks[1].lower())
                entries.append(_entry(name, m))
            else:
                seen_surnames.add(toks[-1].lower())
                entries.append(_entry(name, [], was_full=True))

        # Caption-lead recovery, validated against the index
        for t in texts_by_header.get(sec.get("header", ""), []):
            m = _CAPTION_LEAD_RE.match(t.strip())
            if not m:
                continue
            lead = m.group(1)
            if lead.lower() in _CAPTION_LEAD_STOP or lead.lower() in _CAPTION_STOPWORDS or lead.lower() in seen_surnames:
                continue
            if len(lead) < 4:
                continue  # "Ben", "He": too short to trust as a surname lead
            matches = resolver._fuzzy_surname(lead)
            if matches:
                seen_surnames.add(lead.lower())
                entry = _entry(lead, _cross_section_narrow(matches, seen_full))
                entry["source"] = "caption"
                entries.append(entry)

        out_sections.append({
            "header": sec.get("header"),
            "start_page": sec.get("start_page"),
            "end_page": sec.get("end_page"),
            "pages": sec.get("pages"),
            "names": entries,
        })

    # Final shape: every section lists plain name strings (full names as
    # printed, resolved partials replaced by their full name, unmatched
    # partials kept as printed); partials with SEVERAL possible people are
    # pulled out into one last "NEEDS MAPPING" section that keeps every
    # candidate, so a reader can settle them by hand.
    final_sections: List[Dict[str, Any]] = []
    needs_mapping: List[Dict[str, Any]] = []
    for sec in out_sections:
        names: List[str] = []
        for e in sec["names"]:
            if e["status"] == "ambiguous":
                needs_mapping.append({
                    "name": e["name"],
                    "section": sec["header"],
                    "pages": sec["pages"],
                    "candidates": e["candidates"],
                })
                continue
            n = e["resolved"] if e["status"] == "resolved" else e["name"]
            if n not in names:
                names.append(n)
        final_sections.append({
            "header": sec["header"],
            "start_page": sec["start_page"],
            "end_page": sec["end_page"],
            "pages": sec["pages"],
            "names": names,
        })
    if needs_mapping:
        final_sections.append({
            "header": "NEEDS MAPPING (partial names with several possible matches)",
            "start_page": None,
            "end_page": None,
            "pages": [],
            "names": needs_mapping,
        })
    return {"sections": final_sections, "details": out_sections}


# ----------------------------- Entry point -----------------------------

def resolve_names(
    sports_pdf_path: str,
    portraits_pdf_path: Optional[str] = None,
    out_dir: str = "output",
    portrait_pages: Optional[List[int]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Read <out_dir>/<sports_stem>/sports_sections.json (+ sections_with_text)
    and a portraits compiled_ocr.json; write
    <out_dir>/<sports_stem>/sports_sections.resolved.json.

    Portraits come from `portraits_pdf_path`'s book dir when given, else
    from the SAME book (whole-PDF runs): its compiled_ocr.json restricted to
    `portrait_pages` (1-indexed) when given, otherwise to every page outside
    the sports sections.
    """
    sports_dir = Path(out_dir) / Path(sports_pdf_path).stem
    portraits_dir = Path(out_dir) / Path(portraits_pdf_path).stem if portraits_pdf_path else sports_dir

    sports_path = sports_dir / "sports_sections.json"
    texts_path = sports_dir / "sections_with_text.json"
    portraits_path = portraits_dir / "compiled_ocr.json"

    for p in (sports_path, texts_path, portraits_path):
        if not p.exists():
            print(f"[resolve] missing {p}")
            return None

    pages: Optional[set] = set(portrait_pages) if portrait_pages else None
    if pages is None and not portraits_pdf_path:
        sports_pages = {
            pg for sec in read_json(sports_path).get("sections", []) for pg in sec.get("pages", [])
        }
        pages = {
            pg.get("page_number") for pg in read_json(portraits_path).get("pages", [])
        } - sports_pages
    index = build_student_index(read_json(portraits_path), pages=pages)
    n_names = sum(len(v) for v in index.values())
    print(f"[resolve] student index: {n_names} names, {len(index)} surnames")

    resolved = resolve_sections(
        read_json(sports_path), read_json(texts_path), index
    )

    out_path = sports_dir / "sports_sections.resolved.json"
    write_json(resolved, out_path)

    counts: Dict[str, int] = defaultdict(int)
    for sec in resolved["details"]:
        for e in sec["names"]:
            counts[e["status"]] += 1
    print(f"[resolve] wrote {out_path.name}  {dict(counts)}")
    return {"resolved_path": str(out_path), "index_size": n_names, "counts": dict(counts)}
