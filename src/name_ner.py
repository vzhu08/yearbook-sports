# src/name_ner.py
"""
spaCy NER: extract PERSON name tokens from preprocessed candidate strings.

Responsibilities:
  - spaCy model loading (lazy)
  - GPU/CPU detection (but we force n_process=1 to avoid multiprocessing overhead)
  - merging overlapping/contained PERSON spans within each candidate doc
  - post-filtering name strings
  - dedupe that removes nested token-subparts (e.g., keep "Ralph Wilson", drop "Wilson")

Public API:
  - extract_person_names_per_candidate(candidates, verbose=False) -> List[List[str]]
  - extract_person_names_from_candidates(candidates, verbose=False) -> List[str]
  - dedupe_person_names(names) -> List[str]
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Set, Tuple

# ---------------- CPU oversubscription guard ----------------
# Prevent BLAS thread explosion when using spaCy with n_process>1
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ------------------------------- Config -------------------------------

NAME_STOPWORDS: Set[str] = set()
SUFFIXES = {"Jr", "Jr.", "Sr", "Sr.", "II", "III", "IV", "V"}

# Batch sizing (we force n_process=1 always; no multiprocessing)
NER_BATCH_CPU = 256
NER_BATCH_GPU = 1024

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


# --------------------------- String helpers ---------------------------

def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


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


# --------------------- NER-miss fallback (isolated candidates) ---------------------

# Common non-name words that show up in yearbook rosters as organizational
# labels rather than people (e.g. a stray "OFFICERS" or "CHIEF" isolated by
# the upstream comma/period splitter). Used only to gate the fallback below,
# never applied to names the NER model already found.
_ROSTER_NOISE_WORDS = {
    "chief", "captain", "coach", "coaches", "officers", "members", "member",
    "squad", "team", "club", "class", "committee", "staff",
    "president", "vice", "secretary", "treasurer", "reporter", "editor",
    "sponsor", "advisor", "adviser", "manager", "director", "mascot",
    "assistant", "coordinator", "trainer", "instructor", "scorer",
    "statistician", "statistian",  # incl. common OCR/print misspelling
}

# Function words and caption vocabulary that never occur inside a person's
# name. Title-Case photo captions ("Dogs Still Remain No. 1", "The Number
# One Score-Keeper", "Leaders of a Winning Team") are otherwise exactly the
# 2-4 capitalized tokens the NER-miss fallback below rescues as a name.
_CAPTION_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "but", "in", "on", "at", "to", "for", "by",
    "with", "from", "our", "your", "his", "her", "their", "its", "this",
    "that", "these", "those", "is", "are", "was", "were", "be", "no", "not",
    "one", "two", "three", "four", "five", "still", "all", "big",
    "first", "last", "most", "more", "new", "old", "up", "down", "out",
    "over", "under", "into", "after", "before", "during", "off", "action",
    "team", "teams", "players", "number", "score", "game", "games",
    "season", "year", "night", "time", "home", "away", "life",
    "world", "top", "bottom", "left", "right", "center", "row", "front",
    "back", "above", "below", "who", "what", "when", "where", "how", "why",
    "we", "you", "they", "it", "he", "she", "my", "me", "us", "them",
    "than", "then", "there", "here", "very", "just", "only", "also", "some",
    "any", "each", "every", "both", "few", "many", "much", "such", "own",
    "same", "so", "too", "again", "once", "about", "against", "between",
    "through", "mighty", "great", "super", "proud", "fighting",
    "hard", "hitting", "gifted", "striving", "winning", "leading",
    "leaders", "champions", "champs", "varsity", "spirit", "heart",
    "boys", "girls", "men", "women", "school", "high", "class", "club",
    "candidate", "secondary", "all-american", "director", "coordinator",
}

_FALLBACK_MIN_TOKENS = 2
_FALLBACK_MAX_TOKENS = 4
_GAP_FALLBACK_MAX_TOKENS = 2
_FALLBACK_TOKEN_RE = re.compile(r"[A-Za-z'\-]+")


def _looks_like_probable_name_shape(text: str, max_tokens: int = _FALLBACK_MAX_TOKENS) -> bool:
    """
    Narrow shape check used ONLY as a fallback for text NER didn't tag as a
    PERSON. Requires 2-`max_tokens` capitalized, letters-only(-ish) tokens
    and rejects common roster/organizational words, so obvious non-name
    fragments aren't rescued.

    Two call sites use different `max_tokens`:
      - A candidate where NER found NOTHING at all (e.g. "Dot Fulk", already
        isolated by upstream comma/period splitting, so very likely one
        person) allows up to _FALLBACK_MAX_TOKENS.
      - A gap between spans NER DID find, inside an unstructured multi-name
        run with no delimiters (e.g. "GEORGE HAM NEAL WESTMORELAND" between
        two found names), is capped at _GAP_FALLBACK_MAX_TOKENS (2) — a
        4-token gap there is genuinely ambiguous between "one four-word
        name" and "two two-word names", so we don't guess.
    """
    t = _normalize_space(text)
    if not t:
        return False

    tokens = t.split()
    if not (_FALLBACK_MIN_TOKENS <= len(tokens) <= max_tokens):
        return False

    for tok in tokens:
        bare = tok.strip(".,;:()[]{}'\"")
        if not bare or not _FALLBACK_TOKEN_RE.fullmatch(bare):
            return False
        if not bare[0].isupper():
            return False
        low = bare.lower()
        # Prefix (not just exact) match: OCR often drops the space after a
        # role word, e.g. "COACHPASCO GETTLE" for "Coach Pasco Gettle".
        if any(low == w or low.startswith(w) for w in _ROSTER_NOISE_WORDS):
            return False
        if low in _CAPTION_STOPWORDS:
            return False

    return True


# --------------------------- Span/Token dedupe ---------------------------

def _merge_overlapping_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge overlapping/contained spans into a single span.
    - Containment collapses to the larger one (Ralph Wilson vs Wilson)
    - Partial overlap unions (Colin Vincent) + (Vincent Foster) -> (Colin Vincent Foster)
    """
    if not spans:
        return []

    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = spans[0]

    for s, e in spans[1:]:
        if s < cur_e:  # overlap or containment
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    merged.append((cur_s, cur_e))
    return merged


def _is_token_subsequence(short_tokens: List[str], long_tokens: List[str]) -> bool:
    """True if short_tokens appear as a contiguous slice inside long_tokens."""
    if not short_tokens or len(short_tokens) > len(long_tokens):
        return False
    k = len(short_tokens)
    for i in range(len(long_tokens) - k + 1):
        if long_tokens[i:i + k] == short_tokens:
            return True
    return False


def _dedupe_nested_names(names: List[str]) -> List[str]:
    """
    Remove names that are token-subsequences of a longer extracted name.
    Example: keep "Ralph Wilson", drop "Wilson".
    """
    names = [_normalize_space(n) for n in names if _normalize_space(n)]
    if not names:
        return []

    # Longest first so big names win
    names_sorted = sorted(names, key=lambda n: (len(n.split()), len(n)), reverse=True)

    kept: List[str] = []
    kept_tokens: List[List[str]] = []

    for n in names_sorted:
        toks = n.split()
        if any(_is_token_subsequence(toks, kt) for kt in kept_tokens):
            continue
        kept.append(n)
        kept_tokens.append(toks)

    # Return in first-appearance (document/yearbook) order, not
    # alphabetical, so output can be checked against the physical page.
    first_idx: Dict[str, int] = {}
    for i, n in enumerate(names):
        first_idx.setdefault(n, i)
    return sorted(set(kept), key=lambda n: first_idx[n])


# --------------------------- Public API ---------------------------

def dedupe_person_names(names: List[str]) -> List[str]:
    """Public wrapper: remove nested subnames and dedupe."""
    return _dedupe_nested_names(names)


def extract_person_names_per_candidate(candidates: List[str], verbose: bool = False) -> List[List[str]]:
    """
    Run spaCy NER over *preprocessed* candidates once and return names per candidate.

    - No multiprocessing (n_process=1 always).
    - Within each candidate, merge overlapping PERSON spans before extracting text.
    - Per-candidate output is not nested-deduped across candidates; do that after grouping.
    """
    if not candidates:
        return []

    nlp = _get_spacy_nlp()

    keep = {"ner", "transformer"} & set(nlp.pipe_names)
    disable = [p for p in nlp.pipe_names if p not in keep]

    # Decide GPU vs CPU for batch size (still n_process=1 either way)
    is_gpu = False
    device = None
    try:
        import torch  # type: ignore
        is_gpu = torch.cuda.is_available() and ("transformer" in nlp.pipe_names)
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        is_gpu = False
        device = None

    n_proc = 1
    bsz = NER_BATCH_GPU if is_gpu else NER_BATCH_CPU

    if verbose:
        print(
            f"[NER] model={nlp.meta.get('name')} pipes={nlp.pipe_names} "
            f"is_gpu={is_gpu} device={device} n_process={n_proc} batch_size={bsz} "
            f"candidates={len(candidates)}"
        )

    out: List[List[str]] = []
    with nlp.select_pipes(disable=disable):
        for doc in nlp.pipe(candidates, batch_size=bsz, n_process=n_proc):
            spans = [(ent.start_char, ent.end_char) for ent in doc.ents if ent.label_ == "PERSON"]
            spans = _merge_overlapping_spans(spans)

            names_here: List[str] = []
            for s, e in spans:
                raw = doc.text[s:e]
                cand = _post_filter_name_str(raw)
                if cand:
                    names_here.append(cand)

            # Check text NOT covered by any found span -- the whole string
            # if NER found nothing at all, or the gaps between/around spans
            # otherwise -- for a name NER skipped entirely. E.g. NER finds
            # "DoT McKee" and "Louise Ramsey" in "DoT McKEE MARY MOTTER
            # LOUISE RAMSEY" but misses "MARY MOTTER" in between; the gap
            # itself still passes the narrow name-shape check. Gaps are
            # capped at 2 tokens (stricter than a total NER miss) since a
            # gap sits inside an unstructured, delimiter-free multi-name
            # run where a longer match could actually be two people.
            prev_end = 0
            gaps: List[Tuple[int, int]] = []
            for s, e in spans:
                gaps.append((prev_end, s))
                prev_end = e
            gaps.append((prev_end, len(doc.text)))

            max_gap_tokens = _FALLBACK_MAX_TOKENS if not spans else _GAP_FALLBACK_MAX_TOKENS
            for gs, ge in gaps:
                gap_text = doc.text[gs:ge]
                if _looks_like_probable_name_shape(gap_text, max_tokens=max_gap_tokens):
                    fallback = _post_filter_name_str(gap_text)
                    if fallback:
                        names_here.append(fallback)

            out.append(names_here)

    return out


def extract_person_names_from_candidates(candidates: List[str], verbose: bool = False) -> List[str]:
    """
    Convenience: run NER, flatten, and nested-dedupe globally.
    """
    per = extract_person_names_per_candidate(candidates, verbose=verbose)
    flat: List[str] = []
    for lst in per:
        flat.extend(lst)
    return _dedupe_nested_names(flat)
