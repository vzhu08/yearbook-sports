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

    return sorted(set(kept))


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
