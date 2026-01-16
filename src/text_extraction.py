# src/text_extraction.py
"""
Step 1 — Text Extraction (OCR) with single-file compiled output.

Per PDF:
  1) Render pages to GRAY numpy arrays in memory (no image files saved).
  2) Run PaddleOCR batch on GRAY arrays and save per-page raw JSONs into ocr_json/ (intermediate).
  3) Load per-page JSONs, clean empty text entries, tag page numbers on pages and blocks,
     compute metadata, build headers index, and write ONE file:

     <out_dir>/<pdf_stem>/compiled_ocr.json

Structure:
{
  "meta": {...},
  "headers_index": [...],
  "pages": [...]
}

Notes:
  - Page numbers are 1-indexed and stored both on each page and each block in parsing_res_list.
  - Only a single compiled JSON is written. No separate meta, headers, or “clean” files.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter

import fitz  # PyMuPDF
import numpy as np

import paddle  # noqa: F401  (import to initialize Paddle runtime)
from paddleocr import PPStructureV3

from src.common.io_utils import ensure_dir, write_json, read_json


# ----------------------------- Data -----------------------------

@dataclass
class PageGrayMeta:
    index: int
    width: int
    height: int
    gray: np.ndarray  # uint8, shape (H, W)


# --------------------------- Rendering --------------------------

def _render_pdf_to_gray_arrays(pdf_path: Path, dpi: int) -> List[PageGrayMeta]:
    """
    Render each page to an in-memory grayscale numpy array.
    No image files are saved.

    Returns:
      List[PageGrayMeta] with .gray as uint8 array (H,W).
    """
    t0 = time.time()

    doc = fitz.open(pdf_path)
    scale = dpi / 72.0
    total_pages = len(doc)

    metas: List[PageGrayMeta] = []

    for i in range(total_pages):
        p0 = time.time()
        page = doc.load_page(i)
        mat = fitz.Matrix(scale, scale)

        # GRAY render (single channel)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)

        # pix.samples is a bytes-like buffer of length w*h for GRAY
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

        metas.append(PageGrayMeta(index=i, width=pix.width, height=pix.height, gray=arr))

        dt = time.time() - p0
        print(
            f"[text] page {i+1}/{total_pages}: {pix.width}x{pix.height} (gray in-memory) time={dt:.2f}s",
            flush=True,
        )

    doc.close()
    print(f"[text] render: done pages={len(metas)} time={time.time() - t0:.2f}s", flush=True)
    return metas


# ----------------------------- OCR ------------------------------

def _run_paddleocr_batch(gray_arrays: List[np.ndarray], use_gpu: bool, batch_size: int = 64) -> List[Any]:
    """
    Run PPStructureV3 on a list of grayscale numpy arrays.

    PP-StructureV3 predict() supports numpy.ndarray inputs (and lists of them). :contentReference[oaicite:1]{index=1}
    """
    ocr = PPStructureV3(
        device=("gpu" if use_gpu else "cpu"),
        text_recognition_batch_size=batch_size,
        text_det_limit_side_len=2000,
        text_det_limit_type="max",
        text_det_box_thresh=0.60,
        text_rec_score_thresh=0.70,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        use_seal_recognition=False,
        use_table_recognition=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_region_detection=True,
    )

    # NOTE: pass arrays directly; PaddleOCR will handle internally
    return ocr.predict(gray_arrays)


def _save_paddle_jsons(ocr_json_dir: Path, results: List[Any], page_count: int) -> None:
    """
    Save each OCRResult to a deterministic file path: ocr_json/pageNNNN.json.
    This avoids relying on directory-based naming behavior (which varies by input type).
    """
    ensure_dir(ocr_json_dir)

    created = 0
    for i, res in enumerate(results):
        target = ocr_json_dir / f"page{(i+1):04d}.json"
        try:
            if hasattr(res, "save_to_json"):
                # save_path can be a file path per docs; use deterministic filename
                res.save_to_json(save_path=str(target))
            else:
                write_json({"result": res}, target)
            created += 1
            print(f"[text] ocr_json: saved {target.name}", flush=True)
        except Exception as e:
            print(f"[text] ocr_json: ERROR saving page {i+1}: {e}", flush=True)

    if created != page_count:
        print(f"[text] ocr_json: WARN created/confirmed {created}/{page_count} files", flush=True)


# ------------------------ Compile + Clean ------------------------

def _clean_page_json(page_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove entries with empty text strings from the parallel arrays.
    Supports either {rec_texts, rec_scores, rec_polys, rec_boxes} or {texts, scores, polys, boxes}.
    """
    obj = dict(page_obj)  # shallow copy

    def _filter_parallel(text_key: str, score_key: str, poly_key: str, box_key: str):
        if text_key not in obj or not isinstance(obj[text_key], list):
            return
        texts = obj.get(text_key, [])
        mask = [(t is not None) and (str(t).strip() != "") for t in texts]

        def _apply(key: str):
            arr = obj.get(key, None)
            if isinstance(arr, list):
                new = [arr[i] for i, keep in enumerate(mask) if i < len(arr) and keep]
                obj[key] = new if key == text_key else new

        obj[text_key] = [t for t in texts if (t is not None and str(t).strip() != "")]
        _apply(score_key)
        _apply(poly_key)
        _apply(box_key)

    # Try modern rec_* keys first
    _filter_parallel("rec_texts", "rec_scores", "rec_polys", "rec_boxes")
    # Also support generic keys
    _filter_parallel("texts", "scores", "polys", "boxes")

    return obj


def _compile_clean_from_saved(ocr_json_dir: Path, page_count: int) -> Dict[str, Any]:
    """
    Return {"pages": [cleaned_page_json, ...]} in page order.
    """
    pages: List[Dict[str, Any]] = []
    for i in range(page_count):
        fname = ocr_json_dir / f"page{(i + 1):04d}.json"
        if fname.exists():
            raw = read_json(fname)
            pages.append(_clean_page_json(raw))
        else:
            pages.append({"page_index": i})
    return {"pages": pages, "note": "cleaned (empty texts removed)"}


# ------------------- Tagging + Metadata + Headers -------------------

def _attach_page_numbers(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add page_number to each page and to each block in parsing_res_list if present.
    """
    out = {"note": str(bundle.get("note", ""))}
    new_pages: List[Dict[str, Any]] = []
    for i, pg in enumerate(bundle.get("pages", []), start=1):
        if not isinstance(pg, dict):
            new_pages.append(pg)
            continue
        new_pg = dict(pg)
        new_pg["page_number"] = i

        prl = new_pg.get("parsing_res_list")
        if isinstance(prl, list):
            tagged = []
            for blk in prl:
                if isinstance(blk, dict):
                    nb = dict(blk)
                    nb["page_number"] = i
                    tagged.append(nb)
                else:
                    tagged.append(blk)
            new_pg["parsing_res_list"] = tagged

        new_pages.append(new_pg)

    out["pages"] = new_pages
    return out


def _build_meta(bundle: Dict[str, Any], file_name: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute file-level metadata and attach settings and timestamp.
    """
    pages = bundle.get("pages", [])
    cnt = Counter()
    for pg in pages:
        if isinstance(pg, dict):
            prl = pg.get("parsing_res_list")
            if isinstance(prl, list):
                for blk in prl:
                    if isinstance(blk, dict):
                        cnt[str(blk.get("block_label"))] += 1

    return {
        "file_name": file_name,
        "total_pages": len(pages),
        "counts_by_block_label": dict(cnt),
        "pipeline_settings": settings,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _extract_headers_index(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Collect entries where block_label == "header".
    """
    rows: List[Dict[str, Any]] = []
    for pg in bundle.get("pages", []):
        if not isinstance(pg, dict):
            continue
        page_num = pg.get("page_number")
        prl = pg.get("parsing_res_list")
        if not isinstance(prl, list):
            continue
        for blk in prl:
            if isinstance(blk, dict) and str(blk.get("block_label")) == "header":
                rows.append({
                    "page": int(page_num) if isinstance(page_num, int) else None,
                    "block_content": blk.get("block_content", ""),
                    "block_bbox": blk.get("block_bbox", None),
                })
    return rows


# ------------------------------- Entry Point -------------------------------

def extract_text(
    pdf_path: str,
    out_dir: str,
    dpi: int = 200,
    lang: str = "en",
    use_gpu: bool = True,
    batch_size: int = 64,
) -> None:
    """
    Integrated run for Step 1.
    Writes a single file: compiled_ocr.json

    Changes vs previous:
      - No images are written to disk; pages are rendered to in-memory grayscale arrays.
      - Per-page OCR JSONs are still written to ocr_json/ for debugging and reproducibility.
    """
    t0 = time.time()

    pdf = Path(pdf_path)
    out_root = Path(out_dir)
    book_dir = out_root / pdf.stem
    ocr_json_dir = book_dir / "ocr_json"

    ensure_dir(book_dir)
    ensure_dir(ocr_json_dir)

    with fitz.open(pdf) as _doc:
        page_count = len(_doc)

    print(f"[text] start: {pdf.name} -> {book_dir} (pages={page_count})", flush=True)
    print(f"[text] settings: dpi={dpi}, device={'gpu' if use_gpu else 'cpu'}, batch={batch_size}", flush=True)

    def _expected_stem(i: int) -> str:
        return f"page{(i + 1):04d}"

    # ---- Step 1: render in memory (always) ----
    print("[text] render: generating in-memory grayscale arrays...", flush=True)
    metas = _render_pdf_to_gray_arrays(pdf, dpi=dpi)

    # ---- Step 2: OCR ----
    existing_ocr_pages = sum((ocr_json_dir / f"{_expected_stem(i)}.json").exists() for i in range(page_count))
    if existing_ocr_pages == page_count:
        print(f"[text] ocr: skip (found {page_count} ocr_json pages)", flush=True)
    else:
        print(f"[text] ocr: running Paddle on {len(metas)} gray arrays...", flush=True)
        t_ocr = time.time()
        results = _run_paddleocr_batch([m.gray for m in metas], use_gpu=use_gpu, batch_size=batch_size)
        print(f"[text] ocr: done time={time.time() - t_ocr:.2f}s", flush=True)
        _save_paddle_jsons(ocr_json_dir, results, page_count)

    # ---- Step 3: compile clean + tag + meta + headers (single JSON) ----
    compiled_single_path = book_dir / "compiled_ocr.json"

    print("[text] compile: building cleaned pages from ocr_json...", flush=True)
    cleaned = _compile_clean_from_saved(ocr_json_dir, page_count)

    print("[text] tagging: adding page_number to pages and parsing_res_list blocks...", flush=True)
    cleaned_paged = _attach_page_numbers(cleaned)

    print("[text] report: computing metadata and headers index...", flush=True)
    settings = {
        "dpi": dpi,
        "device": ("gpu" if use_gpu else "cpu"),
        "lang": lang,
        "batch_size": batch_size,
        "rendering": "pymupdf_gray_in_memory",
    }
    meta = _build_meta(cleaned_paged, pdf.name, settings)
    headers = _extract_headers_index(cleaned_paged)

    final_bundle = {
        "meta": meta,
        "headers_index": headers,
        "pages": cleaned_paged.get("pages", []),
    }

    write_json(final_bundle, compiled_single_path)

    def _count_texts(bundle: Dict[str, Any]) -> int:
        total = 0
        for pg in bundle.get("pages", []):
            if isinstance(pg, dict):
                total += len(pg.get("rec_texts", pg.get("texts", [])) or [])
        return total

    total_clean = _count_texts(final_bundle)
    print(f"[text] stats: texts_clean={total_clean}", flush=True)
    print(f"[text] output: {compiled_single_path.name}", flush=True)
    print(f"[text] done: {pdf.name} total_time={time.time() - t0:.2f}s", flush=True)
