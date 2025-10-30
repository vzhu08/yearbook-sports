# src/text_extraction.py
"""
Step 1 — Text Extraction (OCR) with single-file compiled output.

Per PDF:
  1) Render pages to RGB and GRAY images.
  2) Run PaddleOCR batch on GRAY images and save per-page raw JSONs into ocr_json/ (intermediate).
  3) Load per-page JSONs, clean empty text entries, tag page numbers on pages and blocks,
     compute metadata, build headers index, and write ONE file:

     <out_dir>/<pdf_stem>/compiled_ocr.json

     Structure:
     {
       "meta": {
         "file_name": "<pdf name>",
         "total_pages": <int>,
         "counts_by_block_label": {"header": N, "text": M, ...},
         "pipeline_settings": {...},
         "generated_utc": "YYYY-MM-DDTHH:MM:SSZ"
       },
       "headers_index": [
         {"page": 3, "block_content": "SENIOR", "block_bbox": [x1,y1,x2,y2]},
         ...
       ],
       "pages": [
         {
           ...  # cleaned page JSON from Paddle with:
           "page_number": 1,
           "parsing_res_list": [
              {"block_label": "header", "block_content": "...", "block_bbox": [...], "page_number": 1},
              ...
           ],
           # rec_texts/texts arrays already cleaned of empties
         },
         ...
       ]
     }

Notes:
  - Page numbers are 1-indexed and stored both on each page and each block in parsing_res_list.
  - Only a single compiled JSON is written. No separate meta, headers, or “clean” files.
"""

from __future__ import annotations

import time
import json
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter

import fitz  # PyMuPDF
from PIL import Image

from paddleocr import PPStructureV3

from src.common.io_utils import ensure_dir, write_json, read_json


# ----------------------------- Data -----------------------------

@dataclass
class PageImageMeta:
    index: int
    width: int
    height: int
    rgb_path: Path
    gray_path: Path


# --------------------------- Rendering --------------------------

def _save_pil(img: Image.Image, out_path: Path, fmt: str, quality: int) -> None:
    ensure_dir(out_path.parent)
    if fmt.lower() == "jpg":
        img.convert("RGB").save(out_path, "JPEG", quality=quality, optimize=True, progressive=False)
    else:
        img.save(out_path, "PNG", optimize=True)


def _render_pdf_to_images(pdf_path: Path, pages_dir: Path, gray_dir: Path,
                          dpi: int, fmt: str, jpeg_quality: int) -> List[PageImageMeta]:
    import time as _t

    doc = fitz.open(pdf_path)
    scale = dpi / 72.0
    metas: List[PageImageMeta] = []
    total_pages = len(doc)

    for i in range(total_pages):
        t0 = _t.time()
        page = doc.load_page(i)
        mat = fitz.Matrix(scale, scale)

        # RGB render
        rgb = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        img_rgb = Image.frombytes("RGB", [rgb.width, rgb.height], rgb.samples)

        # GRAY render
        gry = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)
        img_gray = Image.frombytes("L", [gry.width, gry.height], gry.samples)

        stem = f"page{(i + 1):04d}"
        rgb_path = pages_dir / f"{stem}.{fmt}"
        gray_path = gray_dir / f"{stem}.{fmt}"

        _save_pil(img_rgb, rgb_path, fmt, jpeg_quality)
        _save_pil(img_gray, gray_path, fmt, jpeg_quality)

        metas.append(PageImageMeta(index=i, width=gry.width, height=gry.height,
                                   rgb_path=rgb_path, gray_path=gray_path))

        dt = _t.time() - t0
        print(f"[text] page {i+1}/{total_pages}: {gry.width}x{gry.height} -> {rgb_path.name}, {gray_path.name} time={dt:.2f}s", flush=True)

    doc.close()
    return metas


# ----------------------------- OCR ------------------------------

def _run_paddleocr_batch(gray_paths: List[Path], use_gpu: bool, batch_size: int = 64) -> List[Any]:
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
    return ocr.predict([str(p) for p in gray_paths])


def _save_paddle_jsons(book_dir: Path, results: List[Any], page_count: int) -> None:
    """
    Use OCRResult.save_to_json(dir). Normalize filenames to pageNNNN.json.
    """
    ocr_dir = book_dir / "ocr_json"
    ensure_dir(ocr_dir)

    def _json_set() -> set[Path]:
        return set(ocr_dir.glob("*.json"))

    before_all = _json_set()
    created = 0

    for i, res in enumerate(results):
        pre = _json_set()
        if hasattr(res, "save_to_json"):
            res.save_to_json(str(ocr_dir))
        else:
            write_json({"result": res}, ocr_dir / f"page{(i+1):04d}.json")

        post = _json_set()
        target = ocr_dir / f"page{(i+1):04d}.json"

        if target in post - pre:
            print(f"[text] ocr_json: saved {target.name}")
            created += 1
        else:
            # Fallback: move/clone the most recent new file
            candidates = sorted(post - before_all, key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                src = candidates[0]
                if src != target:
                    try:
                        src.replace(target)
                    except Exception:
                        obj = read_json(src)
                        write_json(obj, target)
                print(f"[text] ocr_json: normalized {target.name}")
                created += 1
            else:
                print(f"[text] ocr_json: WARN could not determine JSON for page {i+1}")

    if created != page_count:
        print(f"[text] ocr_json: WARN created/confirmed {created}/{page_count} files")


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
    dpi: int = 300,
    fmt: str = "jpg",
    jpeg_quality: int = 95,
    use_gpu: bool = True,
    batch_size: int = 64,
) -> None:
    """
    Integrated run for Step 1.
    Writes a single file: compiled_ocr.json
    """
    t0 = time.time()

    pdf = Path(pdf_path)
    out_root = Path(out_dir)
    book_dir = out_root / pdf.stem
    pages_dir = book_dir / "pages"
    gray_dir = book_dir / "pages_gray"
    ocr_json_dir = book_dir / "ocr_json"

    ensure_dir(book_dir)
    ensure_dir(pages_dir)
    ensure_dir(gray_dir)
    ensure_dir(ocr_json_dir)

    def _expected_stem(i: int) -> str:
        return f"page{(i + 1):04d}"

    with fitz.open(pdf) as _doc:
        page_count = len(_doc)

    print(f"[text] start: {pdf.name} -> {book_dir} (pages={page_count})")
    print(f"[text] settings: dpi={dpi}, fmt={fmt}, quality={jpeg_quality}, device={'gpu' if use_gpu else 'cpu'}, batch={batch_size}")

    # ---- Step 1: render ----
    have_rgb = sum((pages_dir / f"{_expected_stem(i)}.{fmt}").exists() for i in range(page_count))
    have_gray = sum((gray_dir / f"{_expected_stem(i)}.{fmt}").exists() for i in range(page_count))
    if have_rgb == page_count and have_gray == page_count:
        print(f"[text] render: skip (found {page_count} RGB and {page_count} GRAY images)")
        # rebuild metas from existing gray images
        metas: List[PageImageMeta] = []
        for i in range(page_count):
            stem = _expected_stem(i)
            gray_path = gray_dir / f"{stem}.{fmt}"
            with Image.open(gray_path) as im:
                w, h = im.size
            metas.append(PageImageMeta(index=i, width=w, height=h,
                                       rgb_path=pages_dir / f"{stem}.{fmt}",
                                       gray_path=gray_path))
    else:
        print(f"[text] render: generating images...")
        t_render = time.time()
        metas = _render_pdf_to_images(pdf, pages_dir, gray_dir, dpi, fmt, jpeg_quality)
        print(f"[text] render: done pages={len(metas)} time={time.time() - t_render:.2f}s")

    # ---- Step 2: OCR ----
    existing_ocr_pages = sum((ocr_json_dir / f"{_expected_stem(i)}.json").exists() for i in range(page_count))
    if existing_ocr_pages == page_count:
        print(f"[text] ocr: skip (found {page_count} ocr_json pages)")
    else:
        print(f"[text] ocr: running Paddle on {len(metas)} images...")
        t_ocr = time.time()
        results = _run_paddleocr_batch([m.gray_path for m in metas], use_gpu=use_gpu, batch_size=batch_size)
        print(f"[text] ocr: done time={time.time() - t_ocr:.2f}s")
        _save_paddle_jsons(book_dir, results, page_count)

    # ---- Step 3: compile clean + tag + meta + headers (single JSON) ----
    compiled_single_path = book_dir / "compiled_ocr.json"

    # Clean
    print("[text] compile: building cleaned pages from ocr_json...")
    cleaned = _compile_clean_from_saved(ocr_json_dir, page_count)

    # Tag page numbers
    print("[text] tagging: adding page_number to pages and parsing_res_list blocks...")
    cleaned_paged = _attach_page_numbers(cleaned)

    # Meta + headers
    print("[text] report: computing metadata and headers index...")
    settings = {
        "dpi": dpi,
        "image_format": fmt,
        "jpeg_quality": jpeg_quality,
        "device": ("gpu" if use_gpu else "cpu"),
        "lang": lang,
        "batch_size": batch_size,
    }
    meta = _build_meta(cleaned_paged, pdf.name, settings)
    headers = _extract_headers_index(cleaned_paged)

    # Assemble final single JSON
    final_bundle = {
        "meta": meta,
        "headers_index": headers,
        "pages": cleaned_paged.get("pages", []),
    }

    write_json(final_bundle, compiled_single_path)

    # Stats
    def _count_texts(bundle: Dict[str, Any]) -> int:
        total = 0
        for pg in bundle.get("pages", []):
            if isinstance(pg, dict):
                total += len(pg.get("rec_texts", pg.get("texts", [])) or [])
        return total

    total_clean = _count_texts(final_bundle)
    print(f"[text] stats: texts_clean={total_clean}")
    print(f"[text] output: {compiled_single_path.name}")
    print(f"[text] done: {pdf.name} total_time={time.time() - t0:.2f}s")
