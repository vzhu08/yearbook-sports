#!/usr/bin/env python3
"""
Map yearbook PDFs to schools (and High School / non-HS tags) WITHOUT the
DigitalNC website: read each book's own front pages and fuzzy-match the
printed school name against school_types.json (the exhaustive school ->
badge list scraped from DigitalNC).

Usage:
    python src/identify_schools.py scan  <pdf_folder> [out.csv]
        OCRs the first/last few pages of every PDF that has no cached text
        (plain OCR via the pipeline's own extractor), then matches.
    python src/identify_schools.py cached [out.csv]
        Match using already-extracted text in output/<book>/ocr_json only
        (no new OCR) — used to validate the matcher on processed books.

Output CSV columns:
    filename, school, badge, is_high_school, printed_name, title_suggests,
    score, runner_up, margin, flag
A row is flagged NEEDS-REVIEW when the score is low or the runner-up is
close, and NOT-IN-LIST? when the school name printed on the book's own
title pages (printed_name) doesn't resemble the matched list entry — that
usually means the true school is absent from school_types.json (e.g. West
Hyde High School) or ad/faculty text out-scored the title page; in that
case title_suggests gives the list entry closest to the printed name.
Flagged rows are the only ones a human needs to look at.
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent
SCHOOLS = json.loads((ROOT / "school_types.json").read_text())

# Words that carry no identity on their own.
_GENERIC = {"high", "school", "senior", "junior", "college", "university",
            "community", "the", "of", "at", "and", "north", "carolina",
            "state", "county", "technical", "institute", "memorial"}

# A lone city token is weak evidence (printers/engravers in Asheville,
# Charlotte, etc. appear in many books' credits).
_CITY_ONLY_CAP = 70.0
_BIG_CITIES = {"asheville", "charlotte", "durham", "greensboro", "raleigh",
               "wilmington", "winston", "salem", "fayetteville", "goldsboro"}

PAGES_FRONT = 8    # title page / frontispiece usually lands here
PAGES_BACK = 3     # some books only print the school on the colophon/ads

# Historical names -> the name used in school_types.json today.
ALIASES = {
    "North Carolina College for Women": "University of North Carolina at Greensboro",
    "Woman's College of the University of North Carolina": "University of North Carolina at Greensboro",
    "North Carolina College at Durham": "North Carolina Central University",
    "North Carolina College for Negroes": "North Carolina Central University",
    "Appalachian State Teachers College": "Appalachian State University",
    "Pembroke State University": "University of North Carolina at Pembroke",
    "Pembroke State College": "University of North Carolina at Pembroke",
    "East Carolina Teachers College": "East Carolina University",
    "East Carolina College": "East Carolina University",
    "Asheville-Biltmore College": "University of North Carolina at Asheville",
    "Mars Hill College": "Mars Hill University",
    "Chowan College": "Chowan University",
    "Pfeiffer College": "Pfeiffer University",
    "Saint Augustine's College": "Saint Augustine's University",
    "Richmond Technical College": "Richmond Community College",
    "Richmond Technical Institute": "Richmond Community College",
    "Salem Academy and College": "Salem College",
}
CANDIDATES = {**{s: s for s in SCHOOLS}, **ALIASES}

_BOILER = re.compile(
    r"[^.]*(?:Internet Archive|archive\.org|Digitized|digitization)[^.]*\.?",
    re.I)


def _strip_boiler(t: str) -> str:
    return _BOILER.sub(" ", t)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z ]+", " ", s.lower()).strip()


def _distinct_tokens(name: str):
    return [t for t in _norm(name).split() if t not in _GENERIC and len(t) > 2]


def _score(page_text: str, school: str) -> float:
    """Fuzzy presence of the school name in the page text, demanding that
    the school's DISTINCT tokens (e.g. 'yadkinville') actually appear --
    'High School' alone must never match."""
    text = _norm(page_text)
    base = fuzz.partial_ratio(_norm(school), text)
    distinct = _distinct_tokens(school)
    if not distinct:
        # name is all-generic tokens (e.g. "University of North Carolina
        # at Chapel Hill" minus its town): demand a near-verbatim phrase
        return base if base >= 95 else 0.0
    hits = sum(1 for t in distinct if t in text)
    frac = hits / len(distinct)
    if frac == 0:
        return 0.0
    return 0.6 * base + 40.0 * frac


def _page_texts_from_cache(book_dir: Path):
    texts = []
    for f in sorted((book_dir / "ocr_json").glob("page*.json")):
        n = int(f.stem[4:])
        if n <= PAGES_FRONT or n > _count_pages(book_dir) - PAGES_BACK:
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            parts = [t for t in (d.get("overall_ocr_res", {}).get("rec_texts") or [])
                     if isinstance(t, str)]
            m = d.get("plain_ocr_merged")
            if isinstance(m, str):
                parts.append(m)
            texts.append(" ".join(parts))
    return texts


def _count_pages(book_dir: Path) -> int:
    pages = [int(f.stem[4:]) for f in (book_dir / "ocr_json").glob("page*.json")]
    return max(pages) if pages else 0


def _page_texts_from_pdf(pdf: Path, out_dir: Path):
    """OCR just the front/back pages (plain pass) into the normal output
    layout, then read them back. Reuses the pipeline's extractor so the
    cache also serves any later full run."""
    import fitz
    from src.text_extraction import extract_text
    n = len(fitz.open(pdf))
    extract_text(str(pdf), str(out_dir), dpi=200,
                 page_range=(1, min(PAGES_FRONT, n)), layout=False)
    if n > PAGES_FRONT:
        extract_text(str(pdf), str(out_dir), dpi=200,
                     page_range=(max(1, n - PAGES_BACK + 1), n), layout=False)
    return _page_texts_from_cache(out_dir / pdf.stem)


_PRINTED = re.compile(
    r"([A-Z][A-Za-z .&'\-]{2,40}?\s+(?:HIGH SCHOOL|High School|COLLEGE|College|UNIVERSITY|University|SCHOOL OF NURSING))")


# Words that mark a "<...> High School / College" hit as NOT the school's own
# name (activity lists, faculty bios, library stamps, ads).
_PRINTED_JUNK = {"yearbook", "annual", "presented", "band", "coach", "glee",
                 "club", "library", "service", "education", "vice",
                 "president", "athletic", "faculty", "staff"}


def _printed_name(pages) -> str:
    """The school name as the book itself prints it (first good hit on the
    earliest pages) — lets us flag matches whose list entry disagrees."""
    for pg in pages[:5]:
        for m in _PRINTED.finditer(pg):
            cand = re.sub(r"\s+", " ", m.group(1)).strip()
            # "Published by the Senior Class of X High School" -> keep X on
            cand = re.split(r"\b(?:of|OF|by|BY)\b", cand)[-1].strip()
            toks = _distinct_tokens(cand)
            if toks and not any(t in _PRINTED_JUNK for t in toks):
                return cand
    return ""


def _printed_disagrees(printed: str, matched: str) -> bool:
    """True when no distinct token of the printed name resembles any distinct
    token of the matched list entry (fuzzy, so OCR crops like 'EMBROKE' still
    count as Pembroke)."""
    pt, bt = _distinct_tokens(printed), _distinct_tokens(matched)
    if not pt or not bt:
        return False

    def similar(a, b):
        # partial_ratio forgives OCR crops ('EMBROKE'/'PISGAHSENOP') but is
        # too loose on short words ('west'~'east' scores 86), so short
        # tokens must pass the plain ratio
        if min(len(a), len(b)) >= 6:
            return fuzz.partial_ratio(a, b) >= 80
        return fuzz.ratio(a, b) >= 80

    return not any(similar(a, b) for a in pt for b in bt)


def identify(texts) -> dict:
    pages = [_strip_boiler(t)[:8000] for t in texts if t.strip()]
    if not pages:
        return {"school": "", "badge": "", "is_high_school": "", "printed_name": "",
                "title_suggests": "", "score": 0, "runner_up": "", "margin": 0,
                "flag": "NO-TEXT"}
    # score = best single page (the title page should dominate); earlier
    # pages get a small head start over back-of-book ads/colophons
    best_by_canon = {}
    for cand, canon in CANDIDATES.items():
        sc = max((_score(pg, cand) * (1.0 if i < 3 else 0.9 if i < PAGES_FRONT else 0.8)
                  for i, pg in enumerate(pages)), default=0.0)
        if sc > best_by_canon.get(canon, 0.0):
            best_by_canon[canon] = sc
    scored = sorted(best_by_canon.items(), key=lambda x: -x[1])
    (best, s1), (second, s2) = scored[0], scored[1]
    if s1 < 20:
        return {"school": "", "badge": "", "is_high_school": "", "printed_name": "",
                "title_suggests": "", "score": round(s1, 1), "runner_up": "",
                "margin": 0, "flag": "NO-TEXT"}
    flag = "NEEDS-REVIEW" if (s1 < 75 or s1 - s2 < 8) else ""
    printed = _printed_name(pages)
    title_suggests = ""
    if printed and _printed_disagrees(printed, best):
        # The book's own title page disagrees with the match: either the
        # school is missing from school_types.json (West Hyde, quarterian
        # 1956) or ad/faculty text out-scored the title page (greenlights
        # 1947). Either way a human decides -- but hand them the closest
        # list entry for the PRINTED name so it's a one-glance call.
        # (Never auto-adopt: the 1911 Yackety Yack prints students' prep
        # school "Warrenton High School" on its front pages, which scores a
        # perfect 100 against the list while UNC-CH is the true school.)
        flag = (flag + " " if flag else "") + "NOT-IN-LIST?"
        cand, sc = max(((c, _score(printed, c)) for c in CANDIDATES),
                       key=lambda x: x[1])
        if sc >= 60:
            title_suggests = f"{CANDIDATES[cand]} ({round(sc, 1)})"
    return {"school": best, "badge": SCHOOLS[best],
            "is_high_school": SCHOOLS[best] == "High School",
            "printed_name": printed, "title_suggests": title_suggests,
            "score": round(s1, 1), "runner_up": f"{second} ({round(s2,1)})",
            "margin": round(s1 - s2, 1), "flag": flag}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "cached"
    rows = []
    if mode == "cached":
        out_csv = sys.argv[2] if len(sys.argv) > 2 else "school_identifications.csv"
        for book_dir in sorted((ROOT / "output").iterdir()):
            if not (book_dir / "ocr_json").is_dir():
                continue
            texts = _page_texts_from_cache(book_dir)
            if not texts:
                continue
            rows.append({"filename": book_dir.name + ".pdf", **identify(texts)})
    elif mode == "scan":
        folder = Path(sys.argv[2])
        out_csv = sys.argv[3] if len(sys.argv) > 3 else "school_identifications.csv"
        out_dir = ROOT / "output"
        for pdf in sorted(folder.glob("*.pdf")):
            try:
                texts = _page_texts_from_pdf(pdf, out_dir)
                rows.append({"filename": pdf.name, **identify(texts)})
            except Exception as e:
                rows.append({"filename": pdf.name, "school": "", "badge": "",
                             "is_high_school": "", "printed_name": "",
                             "title_suggests": "", "score": 0, "runner_up": "",
                             "margin": 0, "flag": f"ERROR: {e}"})
            print(rows[-1]["filename"], "->", rows[-1]["school"],
                  rows[-1].get("flag", ""))
    else:
        print(__doc__)
        return
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    flagged = sum(1 for r in rows if r["flag"])
    print(f"wrote {out_csv}: {len(rows)} PDFs, {flagged} flagged for review")


if __name__ == "__main__":
    main()
