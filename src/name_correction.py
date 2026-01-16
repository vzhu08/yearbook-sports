#!/usr/bin/env python3
"""
Optional name correction utilities.

- Loading is **optional**; if corpora are missing or disabled, use NoOpCorrector.
- Exposes `maybe_load_corrector(...)` which returns an object with:
      .enabled -> bool
      .correct(name: str) -> str   # returns corrected name or original

Also exposes:
  - run_name_correction(...): per-PDF step that reads/writes:
        <out_dir>/<pdf_stem>/sports_sections.json
    and records applied corrections in JSON under each section.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Protocol, Optional

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
def _substitution_similarity(a: str, b: str, *, processor=None, score_cutoff: float = 0.0, **kwargs) -> float:
    """
    Custom similarity that penalizes length differences a bit more than plain edit distance.
    Returns 0..100 like RapidFuzz scorers and supports RapidFuzz's scorer API:
      - accepts `processor` and `score_cutoff` (and extra **kwargs)
      - returns 0 when the score is below `score_cutoff`
    """
    if processor is not None:
        a = processor(a)
        b = processor(b)

    a = a or ""
    b = b or ""

    dist = Levenshtein.distance(a, b)
    diff = abs(len(a) - len(b))
    weighted = dist + diff
    min_len = max(1, min(len(a), len(b)))
    sim = (1 - weighted / min_len) * 100.0
    sim = max(0.0, sim)

    return sim if sim >= score_cutoff else 0.0


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _alpha_key(tok: str) -> str:
    """Lowercase token key restricted to [a-z]."""
    return re.sub(r"[^a-z]", "", (tok or "").lower())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Concrete Correctors
# ---------------------------------------------------------------------------
class NoOpCorrector:
    def __init__(self) -> None:
        self.enabled = False

    def correct(self, text: str) -> str:
        return _normalize(text)


@dataclass(frozen=True)
class CorrectionRecord:
    original: str
    corrected: str


class NameCorrector:
    """
    Uses fuzzy matching + frequency/rank priors to map noisy OCR names to likely real names.

    IMPORTANT LOGIC CHANGE:
      - Only apply fuzzy correction to a first/last token if that token is NOT present
        in the corresponding dataset map at all.

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

    def _correct_first_token(self, tok: str) -> str:
        """
        Only correct via fuzzy if tok is not present in first_map at all.
        Otherwise preserve it (normalized formatting only).
        """
        key = _alpha_key(tok)
        if not key:
            return ""

        # If it exists in the dataset, DO NOT correct.
        if key in self.first_map:
            # Keep "as name" formatting; not using corpus variant to avoid unexpected replacements.
            return _normalize(tok).title()

        # Otherwise, fuzzy match and pick by (similarity, frequency)
        cands = process.extract(key, self.first_names, scorer=_substitution_similarity, limit=5)
        if not cands:
            return _normalize(tok).title()

        best = max(cands, key=lambda x: (x[1], self.first_map.get(x[0], 0)))
        return best[0].title()

    def _correct_last_token(self, tok: str) -> str:
        """
        Only correct via fuzzy if tok is not present in last_map at all.
        Otherwise preserve it (normalized formatting only).
        """
        key = _alpha_key(tok)
        if not key:
            return _normalize(tok).title()

        # If it exists in the dataset, DO NOT correct.
        if key in self.last_map:
            return _normalize(tok).title()

        # Otherwise, fuzzy match and pick by (similarity, inverse-rank)
        cands = process.extract(key, self.last_names, scorer=_substitution_similarity, limit=5)
        if not cands:
            return _normalize(tok).title()

        best = max(cands, key=lambda x: (x[1], -self.last_map.get(x[0], 0)))
        return best[0].title()

    def correct_with_record(self, text: str) -> Tuple[str, Optional[CorrectionRecord]]:
        """
        Returns (corrected_text, record_if_changed).
        record_if_changed is only returned when an actual change occurred
        (case-insensitive comparison to avoid logging capitalization-only differences).
        """
        raw = _normalize(text)
        if not raw:
            return raw, None

        given_tokens, raw_last, suffix = self._split_name_via_ner(raw)

        corrected_firsts: List[str] = []
        for tok in given_tokens:
            fixed = self._correct_first_token(tok)
            if fixed:
                corrected_firsts.append(fixed)

        corrected_last = self._correct_last_token(raw_last) if raw_last else ""

        candidate = " ".join([*corrected_firsts, corrected_last]).strip()
        if suffix:
            candidate = f"{candidate}, {suffix.title()}"

        corrected = candidate or raw

        # Only record when something truly changed (ignore casing differences)
        if corrected.lower() != raw.lower():
            return corrected, CorrectionRecord(original=raw, corrected=corrected)

        return corrected, None

    def correct(self, text: str) -> str:
        corrected, _rec = self.correct_with_record(text)
        return corrected


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


# ---------------------------------------------------------------------------
# Per-PDF JSON application
# ---------------------------------------------------------------------------
def _iter_sections(root: Any) -> List[Dict[str, Any]]:
    """
    Accepts:
      - {"sections": [ ... ]} or
      - [ ... ] or
      - { ... } (single section)
    Returns a list of dict sections.
    """
    if isinstance(root, dict) and isinstance(root.get("sections"), list):
        return [s for s in root["sections"] if isinstance(s, dict)]
    if isinstance(root, list):
        return [s for s in root if isinstance(s, dict)]
    if isinstance(root, dict):
        return [root]
    return []


def _apply_to_sports_sections_json(root: Any, corrector: Corrector, *, min_tokens: int = 2) -> int:
    """
    Mutates root in-place:
      - For each section that has section["names"] as a list[str], correct each name.
      - If a name changes, append a record to section["name_corrections"].

    Returns number of changed names.
    """
    if not getattr(corrector, "enabled", False):
        return 0

    changed = 0
    sections = _iter_sections(root)

    for sec in sections:
        names = sec.get("names")
        if not isinstance(names, list):
            continue

        new_names: List[str] = []
        for nm in names:
            if not isinstance(nm, str):
                new_names.append(nm)
                continue

            raw = _normalize(nm)
            if len(raw.split()) < min_tokens:
                # Skip correction for non-full names; still normalize whitespace.
                new_names.append(raw)
                continue

            if hasattr(corrector, "correct_with_record"):
                corrected, rec = corrector.correct_with_record(raw)  # type: ignore[attr-defined]
            else:
                corrected, rec = corrector.correct(raw), None

            new_names.append(corrected)

            if rec is not None:
                sec.setdefault("name_corrections", [])
                sec["name_corrections"].append(
                    {
                        "original": rec.original,
                        "corrected": rec.corrected,
                    }
                )
                changed += 1

        sec["names"] = new_names

    # Optional top-level meta stamp (non-breaking)
    if isinstance(root, dict):
        root.setdefault("meta", {})
        if isinstance(root["meta"], dict):
            root["meta"]["name_correction_applied_utc"] = _utc_now_iso()
            root["meta"]["name_correction_changed_count"] = changed

    return changed


def run_name_correction(
    *,
    pdf_path: str,
    out_dir: str,
    year: int = 1980,
    model_name: str = "en_core_web_trf",
    min_tokens: int = 2,
    enable_correction: bool = False,
    first_name_corpus_path: str = "data/first_names_by_decade.json",
    last_name_corpus_path: str = "data/last_names_by_decade.json",
) -> None:
    """
    Per-PDF pipeline step.

    Reads:
      <out_dir>/<pdf_stem>/sports_sections.json

    Writes (in-place overwrite for downstream compatibility):
      <out_dir>/<pdf_stem>/sports_sections.json

    Also writes a one-time backup if it doesn't already exist:
      <out_dir>/<pdf_stem>/sports_sections.uncorrected.json
    """
    pdf = Path(pdf_path)
    out_root = Path(out_dir)
    pdf_out = out_root / pdf.stem
    in_path = pdf_out / "sports_sections.json"

    if not in_path.exists():
        raise FileNotFoundError(f"[NAMES] Missing input: {in_path}")

    # Backup once
    backup_path = pdf_out / "sports_sections.uncorrected.json"
    if not backup_path.exists():
        backup_path.write_bytes(in_path.read_bytes())

    if not enable_correction:
        print("[NAMES] enable_correction=False; leaving sports_sections.json unchanged.")
        return

    # Load spaCy only if needed
    import spacy  # local import to avoid overhead in other steps

    try:
        nlp = spacy.load(model_name)
    except Exception as e:
        raise RuntimeError(f"[NAMES] Failed to load spaCy model '{model_name}': {type(e).__name__}: {e}")

    corrector = maybe_load_corrector(
        enabled=True,
        first_name_corpus_path=first_name_corpus_path,
        last_name_corpus_path=last_name_corpus_path,
        year=year,
        nlp=nlp,
    )

    with open(in_path, "r", encoding="utf-8") as f:
        root = json.load(f)

    changed = _apply_to_sports_sections_json(root, corrector, min_tokens=min_tokens)

    with open(in_path, "w", encoding="utf-8") as f:
        json.dump(root, f, ensure_ascii=False, indent=2)

    print(f"[NAMES] Corrected {changed} names in {in_path}")
