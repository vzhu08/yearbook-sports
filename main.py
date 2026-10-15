# main.py
# -*- coding: utf-8 -*-
"""
Per-PDF pipeline with function-based subprocesses.

Order (per PDF):
  1) src.text_extraction.extract_text      (PP-StructureV3 layout + plain-OCR pass)
  1b) src.text_extraction.extract_text     (layout=False: class-portrait pages for the name index)
  2) src.name_extraction.extract_names
  3) src.name_resolution.resolve_names     (map partial names against the book's own portraits)
  4) src.name_correction.run_name_correction (optional; needs data/ corpora)

Page ranges (TX_PAGE_RANGE / TX_PORTRAIT_RANGE) are for quick local runs; on the
whole-book (HPC) run leave both None: every page is OCR'd with layout and the
name index is built from all non-sports pages.
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ---------------- Levers ----------------
RUN_TEXT_EXTRACTION = True
RUN_PORTRAIT_EXTRACTION = True   # only acts when TX_PORTRAIT_RANGE is set
RUN_NAME_EXTRACTION = True
RUN_NAME_RESOLUTION = True
RUN_NAME_CORRECTION = False      # needs data/first_names_by_decade.json etc.

# ---------------- IO ----------------
# Env vars (for HPC scripts / array jobs) override the constants; unset, the
# local defaults below apply unchanged.
PDF: Optional[str] = os.environ.get("YB_PDF") or None            # e.g., "pdf_input/book.pdf"
PDF_DIR: Optional[str] = os.environ.get("YB_PDF_DIR") or "pdf_input"  # or None
OUT_DIR: str = os.environ.get("YB_OUT_DIR") or "output"

# Step 1
TX_DPI = 400              # 200 loses stylized titles and misreads thin glyphs
TX_USE_GPU = os.environ.get("YB_USE_GPU", "") in ("1", "true", "True")  # set True (or YB_USE_GPU=1) on the HPC/GPU run
TX_PAGE_RANGE: Optional[Tuple[int, int]] = None      # e.g. (95, 146): sports pages, 1-indexed inclusive
TX_AUTO_RANGE = True      # verify/extend TX_PAGE_RANGE with cheap OCR probes (printed-vs-PDF offset, section end)
TX_PORTRAIT_RANGE: Optional[Tuple[int, int]] = None  # e.g. (150, 217): class-portrait pages (plain OCR only)

# Step 2
NC_YEAR = 1980
NC_MIN_TOKENS = 2
NC_MODEL = "en_core_web_trf"         # enforced in name_correction
NC_ENABLE_CORRECTION = True

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
        # 0) sports page ranges: verify/extend the hint, or scan the whole book
        ranges = [list(TX_PAGE_RANGE)] if TX_PAGE_RANGE else []
        if RUN_TEXT_EXTRACTION and TX_AUTO_RANGE:
            from src.text_extraction import resolve_sports_range, find_sports_pages  # cheap probes
            if TX_PAGE_RANGE:
                ranges = [list(resolve_sports_range(str(target_pdf), str(out_root), tuple(TX_PAGE_RANGE)))]
            else:
                ranges = [list(r) for r in find_sports_pages(str(target_pdf), str(out_root))]

        # 1) text extraction (layout + plain-OCR pass) on the sports pages, or
        #    the whole book when nothing was found / auto-range is off
        if RUN_TEXT_EXTRACTION:
            for rng in (ranges or [None]):
                run_func(
                    module="src.text_extraction",
                    func="extract_text",
                    kwargs=dict(
                        pdf_path=str(target_pdf),
                        out_dir=str(out_root),
                        dpi=TX_DPI,
                        use_gpu=TX_USE_GPU,
                        page_range=rng,
                    ),
                )
            # every other page, plain OCR only: the name index for mapping
            if ranges and not TX_PORTRAIT_RANGE:
                run_func(
                    module="src.text_extraction",
                    func="extract_text",
                    kwargs=dict(
                        pdf_path=str(target_pdf),
                        out_dir=str(out_root),
                        dpi=TX_DPI,
                        use_gpu=TX_USE_GPU,
                        page_range=None,
                        layout=False,
                    ),
                )

        # 1b) class-portrait pages, plain OCR only (fast) — feeds the name index
        if RUN_PORTRAIT_EXTRACTION and TX_PORTRAIT_RANGE:
            run_func(
                module="src.text_extraction",
                func="extract_text",
                kwargs=dict(
                    pdf_path=str(target_pdf),
                    out_dir=str(out_root),
                    dpi=TX_DPI,
                    use_gpu=TX_USE_GPU,
                    page_range=list(TX_PORTRAIT_RANGE),
                    layout=False,
                ),
            )

        # 2) name extraction
        if RUN_NAME_EXTRACTION:
            run_func(
                module="src.name_extraction",
                func="extract_names",
                kwargs=dict(
                    pdf_path=str(target_pdf),
                    out_dir=str(out_root),
                    verbose_ner=True,
                ),
            )

        # 3) partial-name resolution against the book's own portraits
        if RUN_NAME_RESOLUTION:
            run_func(
                module="src.name_resolution",
                func="resolve_names",
                kwargs=dict(
                    sports_pdf_path=str(target_pdf),
                    portraits_pdf_path=None,
                    out_dir=str(out_root),
                    portrait_pages=(
                        list(range(TX_PORTRAIT_RANGE[0], TX_PORTRAIT_RANGE[1] + 1))
                        if TX_PORTRAIT_RANGE else None
                    ),
                ),
            )

        # 4) name correction (optional)
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


if __name__ == "__main__":
    main()
