# src/main.py
# -*- coding: utf-8 -*-
"""
Per-PDF pipeline with function-based subprocesses.

Order (per PDF):
  1) src.text_extraction.extract_text
  2) src.name_correction.run_name_correction
  3) src.name_extraction.extract_names
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------- Levers ----------------
RUN_TEXT_EXTRACTION = True
RUN_NAME_CORRECTION = False
RUN_NAME_EXTRACTION = True

# ---------------- IO ----------------
PDF: Optional[str] = None            # e.g., "pdf_input/book.pdf"
PDF_DIR: Optional[str] = "pdf_input" # or None
OUT_DIR: str = "output"

# Step 1
TX_DPI = 300
TX_FORMAT = "jpg"         # "jpg" | "png"
TX_JPEG_QUALITY = 90
TX_USE_GPU = True
TX_LANG = "en"

# Step 2
NC_YEAR = 1980
NC_MIN_TOKENS = 2
NC_MODEL = "en_core_web_trf"         # enforced in name_correction
NC_ENABLE_CORRECTION = False

# Step 3
NE_HEADER_THRESH = 1.8
NE_MAX_GAP = 0.25


# ---------------- Helpers ----------------
def _norm(p: Optional[str]) -> Optional[Path]:
    return Path(p).resolve() if p else None


def _pdf_iter(pdf: Optional[Path], pdf_dir: Optional[Path]):
    if pdf and pdf.exists():
        yield pdf
        return
    if pdf_dir and pdf_dir.exists():
        for p in sorted(pdf_dir.glob("*.pdf")):
            yield p


def run_func(module: str, func: str, kwargs: Dict[str, Any], env_overrides: Optional[Dict[str, str]] = None) -> None:
    """
    Launch a clean Python interpreter that runs:
      import importlib, json, os
      m = importlib.import_module(MODULE)
      getattr(m, FUNC)(**json.loads(KWARGS))
    Uses env vars to pass parameters. Child logs go to console.
    """
    # Compose child environment
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    # Light isolation to avoid DLL/OpenMP clashes
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    if os.name == "nt":
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Ensure repo root on PYTHONPATH so 'src.*' imports resolve
    repo_root = Path(__file__).resolve().parent.parent
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    # Payload for child
    env["YB_MODULE"] = module
    env["YB_FUNC"] = func
    env["YB_KWARGS"] = json.dumps(kwargs)

    code = (
        "import importlib, json, os; "
        "m = importlib.import_module(os.environ['YB_MODULE']); "
        "getattr(m, os.environ['YB_FUNC'])(**json.loads(os.environ['YB_KWARGS']))"
    )

    # Stream stdout/stderr directly
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


# ---------------- Orchestrator ----------------
def main() -> None:
    out_root = Path(OUT_DIR).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pdf = _norm(PDF)
    pdf_dir = _norm(PDF_DIR)

    for target_pdf in _pdf_iter(pdf, pdf_dir):
        # 1) text extraction
        if RUN_TEXT_EXTRACTION:
            run_func(
                module="src.text_extraction",
                func="extract_text",
                kwargs=dict(
                    pdf_path=str(target_pdf),
                    out_dir=str(out_root),
                    dpi=TX_DPI,
                    fmt=TX_FORMAT,
                    jpeg_quality=TX_JPEG_QUALITY,
                    use_gpu=TX_USE_GPU,
                    lang=TX_LANG,
                ),
            )

        # 2) name correction
        if RUN_NAME_CORRECTION:
            run_func(
                module="src.name_correction",
                func="run_name_correction",
                kwargs=dict(
                    pdf_path=str(target_pdf),
                    out_dir=str(out_root),
                    year=NC_YEAR,
                    model_name=NC_MODEL,
                    min_tokens=NC_MIN_TOKENS,
                    enable_correction=NC_ENABLE_CORRECTION,
                ),
            )

        # 3) name extraction
        if RUN_NAME_EXTRACTION:
            run_func(
                module="src.name_extraction",
                func="extract_names",
                kwargs=dict(
                    pdf_path=str(target_pdf),
                    out_dir=str(out_root),
                    header_thresh=NE_HEADER_THRESH,
                    max_gap=NE_MAX_GAP,
                ),
            )


if __name__ == "__main__":
    main()
