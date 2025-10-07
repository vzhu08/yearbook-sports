# src/common/io_utils.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def pdf_iter(pdf: Path | None, pdf_dir: Path | None):
    if pdf and pdf.exists():
        yield pdf
        return
    if pdf_dir and pdf_dir.exists():
        for p in sorted(pdf_dir.glob("*.pdf")):
            yield p
