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
    s = re.sub(r"[()\[\]\"“”]", " ", s)
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
# Pleasing, Patient." -> "IRIS JEAN OVERCASH"). -- this is a special case
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

    from src.name_extraction import _is_junk_name  # "Pfeiffer News" is not a student
    index: Dict[str, set] = defaultdict(set)
    for n in full_names:
        if _is_junk_name(n) or not _index_ok(n):
            continue
        index[n.split()[-1].lower()].add(n)
    return {k: sorted(v) for k, v in sorted(index.items())}


_INDEX_STOP_TOKENS = {
    "and", "&", "or", "the", "of", "mr", "mrs", "miss", "ms", "dr", "rev",
    "mills", "mill", "band", "cross", "hosiery", "company", "co", "inc",
    "store", "stores", "shop", "bank", "church", "club", "class", "grade",
    "school", "high", "junior", "senior", "red", "white", "blue", "jersey",
    "milk", "dairy", "motor", "motors", "furniture", "insurance", "drug",
    "drugs", "supply", "hardware", "lumber", "oil", "gas", "cafe", "hotel",
    "studio", "studios", "bros", "brothers", "sons", "son", "president",
    "secretary", "treasurer", "sponsor", "coach", "captain", "manager",
    "majorettes", "cheerleaders", "team", "staff", "faculty", "editor",
}


def _index_ok(name: str) -> bool:
    """A portrait/roster index entry must look like a real full name: 2-4
    alphabetic tokens, first token a real given name (not an initial or
    honorific), no business/organisation words ("Rich & Thompson", "Hosiery
    Mills", "Mrs. W. G. Coleman", "Elkins And Terrell", "B. Stearns")."""
    toks = name.split()
    if not (2 <= len(toks) <= 4):
        return False
    first = toks[0].strip(".").lower()
    if len(first) < 2 or first in _INDEX_STOP_TOKENS:
        return False
    for t in toks:
        core = t.strip(".,'’\"()")
        if not re.fullmatch(r"[A-Za-z][A-Za-z'’\-]*", core):
            return False
        if core.lower() in _INDEX_STOP_TOKENS:
            return False
        if re.search(r"[a-z]and$", core) and len(core) > 5:
            return False  # "Louiseand"
    return True

# get all names from index without
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
        # for long surnames — at 8 letters "Charlene" is 2 from "Charles".
        # A fuzzy hit is speculative, so its given name must look like one
        # ("R. Cates" must not fuzz onto the ad "Birthday Cakes").
        from src.name_extraction import _COMMON_FIRST_NAMES, _NICKNAMES
        budget = 2 if len(s) >= 10 else 1
        hits: List[str] = []
        for k, names in self.index.items():
            if abs(len(k) - len(s)) <= budget and Levenshtein.distance(
                k, s, score_cutoff=budget
            ) <= budget:
                for n in names:
                    first = n.split()[0].strip(".").lower()
                    if len(first) == 1 or first in _COMMON_FIRST_NAMES or first in _NICKNAMES:
                        hits.append(n)
        return sorted(hits)

    def resolve(self, surname: str, initial: Optional[str] = None) -> List[str]:
        matches = self._fuzzy_surname(surname)
        if initial and matches:
            # The initial is a hard constraint, checked even against a lone
            # surname match — "B. Cooper" must NOT resolve to the only
            # fuzzy hit "Mary Hooper" (1973 Hanoverian football).
            want = initial.lower()
            first = [m for m in matches if m.split()[0][0].lower() == want]
            if first:
                return first
            # A middle-name hit ("G. Pierce" ~ "Eleanor Gayle Pierce") is
            # weak evidence — the cheerleader "Mary Beth Hickman" is NOT
            # the football roster's "B. Hickman". Offer such hits only as
            # candidates for hand-mapping, never as a unique resolution.
            mid = [
                m for m in matches
                if any(t[0].lower() == want for t in m.split()[1:-1])
            ]
            return mid if len(mid) > 1 else []
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


def _index_spellfix(name: str, resolver: "NameResolver", quality: Optional[Dict[str, float]] = None) -> Optional[str]:
    """
    Verify a FULL name's surname spelling against the book's own index: when
    the surname is unknown but within 1-2 edits of an indexed surname AND
    that indexed family has a member whose given name matches (allowing
    nicknames), the printed given name is kept with the index's surname
    spelling — "Tom Campbelf" -> "Tom Campbell", "Harry Grav" -> "Harry
    Gray", "Grant Upehurch" -> "Grant Upchurch". Anything without a
    same-given corroborating index entry is left as printed.
    """
    from src.name_extraction import _given_names_match
    toks = name.split()
    if len(toks) < 2 or not toks[0][0].isupper():
        return None
    given = toks[0]
    if len(given.rstrip(".")) < 2:
        return None
    from src.name_extraction import _COMMON_FIRST_NAMES, _COMMON_SURNAMES
    surname = toks[-1].lower().strip(".,")
    if len(surname) < 4:
        return None

    def _clear(word: str) -> bool:
        # A token is trusted as printed when the OCR read it confidently
        # AND the book prints it more than once or knows it as a family —
        # a solitary confident misread ("Campbelf" at 0.99 beside a real
        # "Campbell") is still fixable, while "Joe Tiller"/"Robert Masse"
        # (families in the index) are protected.
        if quality is None:
            return False
        w = word.lower()
        if w in resolver.index:
            return True
        q, c = quality.get(w, (0.0, 0))
        return q >= 0.97 and c >= 2
    if surname in resolver.index:
        # surname is right; check the GIVEN name against that family
        # ("Beeky Gantt" -> index "Becky Gantt")
        g = given.lower().strip(".")
        if g in _COMMON_FIRST_NAMES or g in _NICKNAMES_LOCAL or _clear(g):
            return None
        if len(resolver.index[surname]) > 3:
            return None  # big family: too many chances for a false match
        near1, near2 = set(), set()
        for n in resolver.index[surname]:
            f = n.split()[0].strip(".")
            if len(f) <= 2 or f.lower() == g:
                continue
            d = Levenshtein.distance(f.lower(), g, score_cutoff=2)
            if d <= 1 and len(f) == len(given.strip(".")):
                near1.add(f)  # substitutions only: "Beeky"->"Becky"
            elif d <= 2:
                near2.add(f)
        if len(near1) == 1 and not near2:
            f = near1.pop()
            if f[:1].lower() != g[:1]:
                return None  # different first letter: too speculative
            return " ".join([f] + toks[1:])
        return None
    if _clear(surname):
        return None
    budget = 2 if len(surname) >= 7 else 1
    hits: List[str] = []
    for k, names_k in resolver.index.items():
        # substitutions only ("Campbelf"->"Campbell", "Paiham"->"Parham"):
        # insertions/deletions matched real different names ("Easter"->
        # "Feaster", "Roberts"->"Robert")
        if len(k) == len(surname) and k[:1] == surname[:1] and Levenshtein.distance(k, surname, score_cutoff=budget) <= budget:
            for n in names_k:
                if _given_names_match(n.split()[0], given):
                    hits.append(n.split()[-1])
    fixed_surnames = sorted(set(h.lower() for h in hits))
    if len(fixed_surnames) == 1:
        fixed = hits[0]
        if fixed.lower() == surname:
            return None  # case-only difference: keep the printed casing
        return " ".join(toks[:-1] + [_fix_case(fixed)])
    if fixed_surnames:
        return None
    # No index family at all: fall back to the common-surname list, but
    # only when the printed surname is not itself a known family in the
    # book (1-edit protected: "Stroup" stays when the index has "Stroupe")
    # and the given name is a common one ("Tom Campbelf" -> "Tom Campbell").
    if given.lower().strip(".") not in _COMMON_FIRST_NAMES and given.lower() not in _NICKNAMES_LOCAL:
        return None
    if any(
        abs(len(k) - len(surname)) <= 1 and Levenshtein.distance(k, surname, score_cutoff=1) <= 1
        for k in resolver.index
    ):
        return None
    common_hits = {
        w for w in _COMMON_SURNAMES
        if len(w) >= 6 and len(w) == len(surname)
        and Levenshtein.distance(w, surname, score_cutoff=1) <= 1
    }
    common_hits = {w for w in common_hits if w[:1] == surname[:1]}
    if len(common_hits) == 1:
        return " ".join(toks[:-1] + [_fix_case(common_hits.pop())])
    return None


def _fix_case(surname: str) -> str:
    out = surname[0].upper() + surname[1:]
    return re.sub(r"^(Mc|O')(\w)", lambda m: m.group(1) + m.group(2).upper(), out)


from src.name_extraction import _NICKNAMES as _NICKNAMES_LOCAL


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
    quality: Optional[Dict[str, float]] = None,
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
        # narrowing evidence comes from THIS section only: the football
        # roster's "Joe Brookshire" must not settle the girls' "Brookshire"
        seen_full = {
            n.lower() for n in sec.get("names", [])
            if len(n.split()) >= 2 and not n.split()[0].endswith(".")
        }

        # a caption full of quoted jersey nicknames ('"Hank," "Suzie," ...')
        # means bare first names here are NICKNAMES as printed — mapping
        # them onto the index invented people ("Hank" -> Henry Stewart)
        quoted_caption = sum(t.count('"') for t in texts_by_header.get(sec.get("header", ""), [])) >= 6

        for name in sec.get("names", []):
            toks = name.split()
            if len(toks) == 1:
                # exact surname, then first name ("Cassandra"), then a
                # fuzzy surname, then a glued "FirstLast" token. In a
                # surname-only roster (1942-style), EXACT index matches
                # only: fuzzy uniqueness invented people ("Hurley" ->
                # "Elmer Eugene Hunley").
                m = _cross_section_narrow(index.get(toks[0].lower(), []), seen_full)
                if sec.get("surname_mode"):
                    pass
                else:
                    if not m and not quoted_caption:
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
                fixed = _index_spellfix(name, resolver, quality)
                if fixed and fixed != name:
                    entries.append({"name": name, "status": "resolved", "resolved": fixed, "candidates": [fixed]})
                else:
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

        osec = {
            "header": sec.get("header"),
            "start_page": sec.get("start_page"),
            "end_page": sec.get("end_page"),
            "pages": sec.get("pages"),
            "names": entries,
        }
        for k in ("human_check", "human_check_reason", "surname_mode"):
            if sec.get(k):
                osec[k] = sec[k]
        out_sections.append(osec)

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
                # the roster keeps the entry AS PRINTED (page order stays
                # complete); NEEDS MAPPING lists it again with candidates
                needs_mapping.append({
                    "name": e["name"],
                    "section": sec["header"],
                    "pages": sec["pages"],
                    "candidates": e["candidates"],
                })
            n = e["resolved"] if e["status"] == "resolved" else e["name"]
            if n not in names:
                names.append(n)
        out_sec = {
            "header": sec["header"],
            "start_page": sec["start_page"],
            "end_page": sec["end_page"],
            "pages": sec["pages"],
            "names": names,
        }
        if sec.get("human_check"):
            out_sec["human_check"] = True
            out_sec["human_check_reason"] = sec.get("human_check_reason", "")
        final_sections.append(out_sec)
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

    from src.name_extraction import _token_read_quality_counts
    quality = _token_read_quality_counts(read_json(portraits_path))
    resolved = resolve_sections(
        read_json(sports_path), read_json(texts_path), index, quality
    )

    # user-confirmed corrections, applied last: <book>/name_overrides.json
    # maps an extracted name (exact string) to its verified form
    ov_path = sports_dir / "name_overrides.json"
    if ov_path.exists():
        try:
            raw_ov = read_json(ov_path)
            ov = {str(k): v for k, v in raw_ov.items()}
        except Exception:
            ov = {}
        adds = ov.pop("__add__", {}) if isinstance(ov.get("__add__", None), dict) else {}
        removes = ov.pop("__remove__", {}) if isinstance(ov.get("__remove__", None), dict) else {}
        # section-scoped renames ("__rename__": {header: {old: new}}) for a
        # spelling that is correct in OTHER sections; a rename value (here
        # or global) may be a LIST — the names splice in at the original
        # position, so one glued OCR token ("Jimmy Lee Herbert Corey") can
        # become two people without losing the printed page order.
        sec_renames = ov.pop("__rename__", {}) if isinstance(ov.get("__rename__", None), dict) else {}
        # "__human_check__": {header: reason} — force the human-check flag
        # on a section whose print the user found illegible (overlapping
        # photos, ink) even though the OCR text itself looked clean
        hc_marks = ov.pop("__human_check__", {}) if isinstance(ov.get("__human_check__", None), dict) else {}
        for sec in resolved["sections"]:
            hdr_key = str(sec.get("header", ""))
            if hdr_key not in hc_marks:
                continue
            r = hc_marks[hdr_key]
            if r:
                sec["human_check"] = True
                sec["human_check_reason"] = r
            else:
                # "" clears an auto-set flag the user has verified by eye
                sec.pop("human_check", None)
                sec.pop("human_check_reason", None)
        if ov or adds or removes or sec_renames:
            for sec in resolved["sections"]:
                hdr = str(sec.get("header", ""))
                if hdr.startswith("NEEDS"):
                    # entries here are dicts; ANY override on the printed
                    # name settles it — "" (a chunking artifact) and a
                    # rename (the user mapped it; the roster entry renames
                    # in place) both drop the worklist row
                    sec["names"] = [
                        e for e in sec["names"]
                        if not (isinstance(e, dict) and str(e.get("name")) in ov)
                    ]
                    continue
                def _apply(name: str, mapping: Dict[str, Any]) -> List[str]:
                    if name not in mapping:
                        return [name]
                    v = mapping[name]
                    if isinstance(v, list):
                        return [x for x in v if x]
                    return [v] if v else []  # ""/[] removes the entry
                local = sec_renames.get(hdr, {})
                renamed = [
                    x
                    for n in sec["names"]
                    for mid in _apply(n, local)
                    for x in _apply(mid, ov)
                ]
                # section-scoped removals ("__remove__": {header: [names]})
                # for names that are correct in OTHER sections (a global ""
                # removal would strip them everywhere)
                rm_here = set(removes.get(hdr, []))
                renamed = [n for n in renamed if n not in rm_here]
                seen: set = set()
                sec["names"] = [n for n in renamed if not (n in seen or seen.add(n))]
            # __add__ second pass: each header's names go to ONE section —
            # the first matching one that already has names (a divider page
            # often shares its header with the real roster section), else
            # the first match; a header matching nothing creates a section
            # (rosters reconstructed from elsewhere in the book, 1932
            # Laurel style), inserted before NEEDS MAPPING.
            needs_idx = next(
                (i for i, s in enumerate(resolved["sections"])
                 if str(s.get("header", "")).startswith("NEEDS")),
                len(resolved["sections"]),
            )
            for hdr, extras in adds.items():
                if not extras:
                    continue
                targets = [
                    s for s in resolved["sections"]
                    if str(s.get("header", "")) == hdr
                    and not str(s.get("header", "")).startswith("NEEDS")
                ]
                if targets:
                    tgt = next((s for s in targets if s.get("names")), targets[0])
                    for extra in extras:
                        if extra not in tgt["names"]:
                            tgt["names"].append(extra)
                else:
                    resolved["sections"].insert(
                        needs_idx,
                        {"header": hdr, "pages": [], "names": list(dict.fromkeys(extras))},
                    )
                    needs_idx += 1
            print(f"[resolve] applied {len(ov)} name override(s), {sum(len(v) for v in adds.values())} addition(s)")

    # Final summary section: every section flagged human_check, gathered
    # in one place at the very end so a reader can find what still needs
    # eyes on the page without scanning every section.
    hc_rows = [
        {
            "section": s.get("header"),
            "pages": s.get("pages"),
            "reason": s.get("human_check_reason", ""),
        }
        for s in resolved["sections"]
        if s.get("human_check")
    ]
    if hc_rows:
        resolved["sections"].append({
            "header": "NEEDS HUMAN CHECK (sections with garbled/obscured print)",
            "start_page": None,
            "end_page": None,
            "pages": [],
            "names": hc_rows,
        })

    out_path = sports_dir / "sports_sections.resolved.json"
    write_json(resolved, out_path)

    counts: Dict[str, int] = defaultdict(int)
    for sec in resolved["details"]:
        for e in sec["names"]:
            counts[e["status"]] += 1
    print(f"[resolve] wrote {out_path.name}  {dict(counts)}")
    return {"resolved_path": str(out_path), "index_size": n_names, "counts": dict(counts)}
