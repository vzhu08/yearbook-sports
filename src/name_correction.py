#!/usr/bin/env python3
"""
Optional name correction utilities.

- Loading is **optional**; if corpora are missing or disabled, use NoOpCorrector.
- Exposes `maybe_load_corrector(...)` which returns an object with:
      .enabled -> bool
      .correct(name: str) -> str   # returns corrected name or original
"""

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Tuple, Protocol

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein


# ---------------------------------------------------------------------------
# Protocol / Interfaces
# ---------------------------------------------------------------------------
class Corrector(Protocol):
    enabled: bool
    def correct(self, text: str) -> str: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _substitution_similarity(a: str, b: str) -> float:
    """
    Custom similarity that penalizes length differences a bit more than plain edit distance.
    Returns 0..100 like RapidFuzz scorers.
    """
    dist = Levenshtein.distance(a, b)
    diff = abs(len(a) - len(b))
    weighted = dist + diff
    min_len = max(1, min(len(a), len(b)))
    sim = (1 - weighted / min_len) * 100
    return max(0.0, sim)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# ---------------------------------------------------------------------------
# Concrete Correctors
# ---------------------------------------------------------------------------
class NoOpCorrector:
    def __init__(self) -> None:
        self.enabled = False

    def correct(self, text: str) -> str:
        return _normalize(text)


class NameCorrector:
    """
    Uses fuzzy matching + frequency/rank priors to map noisy OCR names to likely real names.

    first_map: dict first_name -> frequency (higher is more common)
    last_map:  dict last_name  -> rank (lower number is more common)
    """
    def __init__(
        self,
        first_names: List[str],
        first_map: Dict[str, int],
        last_names: List[str],
        last_map: Dict[str, int],
        nlp,
    ) -> None:
        self.enabled = True
        self.first_names = first_names
        self.first_map = first_map
        self.last_names = last_names
        self.last_map = last_map
        self.nlp = nlp

    def _split_name_via_ner(self, text: str) -> Tuple[List[str], str, str]:
        """Returns (given_tokens, raw_last, suffix)."""
        base = _normalize(text).title()
        suffix = ""

        # Suffix handling
        if "," in base:
            base, suffix = base.split(",", 1)
            suffix = suffix.strip()
        else:
            parts = base.split()
            if parts and parts[-1].rstrip(".").lower() in {"jr", "sr", "ii", "iii", "iv", "v"}:
                suffix = parts[-1]
                base = " ".join(parts[:-1])

        # Prefer PERSON span if spaCy isolates it
        doc = self.nlp(base)
        ents = [e.text for e in doc.ents if e.label_ == "PERSON"]
        name = ents[0] if ents else base

        tokens = name.split()
        if len(tokens) == 0:
            return [], "", suffix

        return tokens[:-1], tokens[-1], suffix

    def correct(self, text: str) -> str:
        raw = _normalize(text)
        if not raw:
            return raw

        given_tokens, raw_last, suffix = self._split_name_via_ner(raw)

        # First names: choose by (similarity, frequency)
        corrected_firsts: List[str] = []
        for tok in given_tokens:
            key = re.sub(r"[^a-z]", "", tok.lower())
            if not key:
                continue
            cands = process.extract(key, self.first_names, scorer=_substitution_similarity, limit=5)
            best = max(cands, key=lambda x: (x[1], self.first_map.get(x[0], 0)))
            corrected_firsts.append(best[0].title())

        # Last name: choose by (similarity, *inverse* rank)
        if raw_last:
            key = re.sub(r"[^a-z]", "", raw_last.lower())
            if key:
                cands = process.extract(key, self.last_names, scorer=_substitution_similarity, limit=5)
                best = max(cands, key=lambda x: (x[1], -self.last_map.get(x[0], 0)))
                corrected_last = best[0].title()
            else:
                corrected_last = raw_last.title()
        else:
            corrected_last = ""

        candidate = " ".join([*corrected_firsts, corrected_last]).strip()
        if suffix:
            candidate = f"{candidate}, {suffix.title()}"

        return candidate or raw


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _pick_decade_key(dct: Dict[str, Any], year: int) -> str:
    decades = sorted(int(k) for k in dct)
    tgt = (year // 10) * 10
    if str(tgt) in dct:
        return str(tgt)
    if tgt < decades[0]:
        return str(decades[0])
    return str(decades[-1])


def _load_corpora(
    first_name_corpus_path: str,
    last_name_corpus_path: str,
    year: int
) -> Tuple[List[str], Dict[str, int], List[str], Dict[str, int]]:
    with open(first_name_corpus_path, encoding="utf-8") as f:
        fn_by_decade = json.load(f)
    with open(last_name_corpus_path, encoding="utf-8") as f:
        ln_by_decade = json.load(f)

    kf = _pick_decade_key(fn_by_decade, year)
    kl = _pick_decade_key(ln_by_decade, year)

    # First names: higher frequency first; Last names: lower rank number first
    first_sorted = sorted(fn_by_decade[kf].items(), key=lambda x: -x[1])
    last_sorted  = sorted(ln_by_decade[kl].items(), key=lambda x: x[1])

    first_names = [n for n, _ in first_sorted]
    last_names  = [n for n, _ in last_sorted]
    return first_names, fn_by_decade[kf], last_names, ln_by_decade[kl]


def maybe_load_corrector(
    *,
    enabled: bool,
    first_name_corpus_path: str,
    last_name_corpus_path: str,
    year: int,
    nlp
) -> Corrector:
    """
    Try to load a NameCorrector if `enabled` and corpora exist; otherwise return NoOpCorrector.
    """
    if not enabled:
        print("[NAMES] Name correction disabled (default).")
        return NoOpCorrector()

    # Graceful checks
    if not (os.path.exists(first_name_corpus_path) and os.path.exists(last_name_corpus_path)):
        print("[NAMES][WARN] Name corpora not found; correction disabled.")
        return NoOpCorrector()

    try:
        fnames, fmap, lnames, lmap = _load_corpora(first_name_corpus_path, last_name_corpus_path, year)
        print(f"[NAMES] Loaded corpora: {len(fnames)} first, {len(lnames)} last.")
        return NameCorrector(fnames, fmap, lnames, lmap, nlp)
    except Exception as e:
        print(f"[NAMES][WARN] Failed to load corpora: {type(e).__name__}: {e} — correction disabled.")
        return NoOpCorrector()
