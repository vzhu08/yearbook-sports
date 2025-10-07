# src/name_correction.py

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from src.common.io_utils import ensure_dir, read_json, write_json


# -------------------------- Internal --------------------------

def _load_trf(model_name: str):
    if model_name != "en_core_web_trf":
        raise ValueError("--nc-model must be 'en_core_web_trf' only.")
    try:
        import spacy
        nlp = spacy.load("en_core_web_trf")
    except Exception as e:
        raise RuntimeError(
            "Missing spaCy transformer model 'en_core_web_trf'. "
            "Install: pip install spacy[transformers] && python -m spacy download en_core_web_trf"
        ) from e
    return nlp


def _extract_person_spans(nlp, compiled_clean_path: Path, min_tokens: int) -> List[Dict[str, Any]]:
    data = read_json(compiled_clean_path)
    persons: List[Dict[str, Any]] = []

    for pg_idx, page in enumerate(data.get("pages", [])):
        for item in page.get("items", []):
            txt = item.get("text", "")
            if not txt or len(txt) < 2:
                continue
            doc = nlp(txt)
            for ent in doc.ents:
                if ent.label_ == "PERSON" and len(ent.text.split()) >= min_tokens:
                    persons.append(
                        {
                            "page": pg_idx,
                            "text": ent.text,
                            "start_char": int(ent.start_char),
                            "end_char": int(ent.end_char),
                        }
                    )
    return persons


def _apply_correction(person_spans: List[Dict[str, Any]], year: int, enabled: bool) -> List[Dict[str, str]]:
    if not enabled:
        return [{"orig": p["text"], "corrected": p["text"]} for p in person_spans]

    # TODO: add new corrections.
    corrected: List[Dict[str, str]] = []
    for p in person_spans:
        name = p["text"]
        corrected.append({"orig": name, "corrected": name})
    return corrected


# -------------------------- Public API --------------------------

def run_name_correction(
    pdf_path: str,
    out_dir: str,
    year: int = 1980,
    model_name: str = "en_core_web_trf",
    min_tokens: int = 2,
    enable_correction: bool = False,
) -> None:
    """
    Integrated run function for Step 2. Prints progress to console.
    """
    t0 = time.time()
    book_dir = Path(out_dir) / Path(pdf_path).stem
    ensure_dir(book_dir)

    compiled_clean = book_dir / "compiled_ocr_clean.json"
    if not compiled_clean.exists():
        print(f"[corr] skip: {book_dir.name} missing compiled_ocr_clean.json")
        return

    print(f"[corr] start: {book_dir.name}, model={model_name}, min_tokens={min_tokens}, year={year}")
    t_model = time.time()
    nlp = _load_trf(model_name)
    print(f"[corr] model: loaded time={time.time() - t_model:.2f}s")

    t_span = time.time()
    persons = _extract_person_spans(nlp, compiled_clean, min_tokens=min_tokens)
    print(f"[corr] NER: persons={len(persons)} time={time.time() - t_span:.2f}s")

    t_fix = time.time()
    corrected = _apply_correction(persons, year=year, enabled=enable_correction)
    print(f"[corr] correction: enabled={enable_correction} corrected={len(corrected)} time={time.time() - t_fix:.2f}s")

    write_json({"persons": persons, "year": year, "model": "en_core_web_trf"}, book_dir / "person_spans.json")
    write_json({"corrected": corrected, "year": year}, book_dir / "corrected_names.json")

    # CSV report
    import csv
    with open(book_dir / "correction_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["orig", "corrected"])
        for row in corrected:
            w.writerow([row["orig"], row["corrected"]])

    print(f"[corr] save: person_spans.json, corrected_names.json, correction_report.csv")
    print(f"[corr] done: {book_dir.name} total_time={time.time() - t0:.2f}s")
