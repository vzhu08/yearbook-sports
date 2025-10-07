# src/text_extraction.py
"""
Step 1 — Text Extraction (OCR).

Pipeline per PDF:
  1) Render pages to RGB and GRAY images.
  2) Run PaddleOCR on all GRAY images in one batch (predict) and, for each page,
     call res.save_to_json(...) into ocr_json/.
  3) Compile the saved Paddle JSONs into:
        - compiled_ocr.json  (just concatenated per-page Paddle JSON)
        - compiled_ocr_clean.json  (same, but entries with empty text removed)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

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

        # Per-page printout
        dt = _t.time() - t0
        try:
            rgb_kb = rgb_path.stat().st_size // 1024
            gray_kb = gray_path.stat().st_size // 1024
            size_str = f"{rgb_kb}KB/{gray_kb}KB"
        except Exception:
            size_str = "n/a"

        print(f"[text] page {i+1}/{total_pages}: {gry.width}x{gry.height} -> {rgb_path.name}, {gray_path.name} ({size_str}) time={dt:.2f}s",
              flush=True)

    doc.close()
    return metas


# ----------------------------- OCR ------------------------------

def _run_paddleocr_batch(gray_paths: List[Path], use_gpu: bool, lang: str, batch_size: int = 64) -> List[Any]:
    ocr = PPStructureV3(
        lang=lang,
        device=("gpu" if use_gpu else "cpu"),
        text_recognition_batch_size=batch_size,
        text_det_limit_side_len=3000,
        text_det_limit_type="max",
        text_det_box_thresh=0.60,
        text_rec_score_thresh=0.70,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_seal_recognition=False,
        use_table_recognition=False,
        use_formula_recognition=False,
        use_chart_recognition=False,
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
        new_files = list(post - pre)
        target = ocr_dir / f"page{(i+1):04d}.json"

        if len(new_files) == 1:
            src = new_files[0]
            if src != target:
                try:
                    src.replace(target)
                except Exception:
                    obj = read_json(src)
                    write_json(obj, target)
            print(f"[text] ocr_json: saved {target.name}")
            created += 1
        elif target.exists():
            print(f"[text] ocr_json: found existing {target.name}")
            created += 1
        else:
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

def _compile_from_saved(ocr_json_dir: Path, metas: List[PageImageMeta]) -> Dict[str, Any]:
    """
    compiled_ocr.json = {"pages": [<raw paddle json for page1>, <raw for page2>, ...]}
    """
    pages: List[Dict[str, Any]] = []
    for meta in metas:
        fname = ocr_json_dir / f"page{(meta.index + 1):04d}.json"
        if fname.exists():
            pages.append(read_json(fname))
        else:
            pages.append({"page_index": meta.index})
    return {"pages": pages, "note": "concatenated Paddle JSONs by page"}


def _clean_page_json(page_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Only remove entries with empty text strings. Keep everything else.
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
                obj[key] = new

        obj[text_key] = [t for t in texts if (t is not None and str(t).strip() != "")]
        _apply(score_key)
        _apply(poly_key)
        _apply(box_key)

    # Try modern rec_* keys first
    _filter_parallel("rec_texts", "rec_scores", "rec_polys", "rec_boxes")
    # Also support generic keys
    _filter_parallel("texts", "scores", "polys", "boxes")

    return obj


def _compile_clean_from_saved(ocr_json_dir: Path, metas: List[PageImageMeta]) -> Dict[str, Any]:
    pages: List[Dict[str, Any]] = []
    for meta in metas:
        fname = ocr_json_dir / f"page{(meta.index + 1):04d}.json"
        if fname.exists():
            raw = read_json(fname)
            pages.append(_clean_page_json(raw))
        else:
            pages.append({"page_index": meta.index})
    return {"pages": pages, "note": "concatenated + cleaned (empty texts removed)"}


# -------------------------- Public API --------------------------

def extract_text(
    pdf_path: str,
    out_dir: str,
    dpi: int = 300,
    fmt: str = "jpg",
    jpeg_quality: int = 95,
    use_gpu: bool = True,
    lang: str = "en",
    batch_size: int = 64,
) -> None:
    """
    Integrated run function for Step 1. Minimal cleaning: drop empty text entries only.
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

    def _metas_from_existing(n_pages: int) -> List[PageImageMeta]:
        metas_: List[PageImageMeta] = []
        for i in range(n_pages):
            stem = _expected_stem(i)
            rgb_path = pages_dir / f"{stem}.{fmt}"
            gray_path = gray_dir / f"{stem}.{fmt}"
            with Image.open(gray_path) as im:
                w, h = im.size
            metas_.append(PageImageMeta(index=i, width=w, height=h,
                                        rgb_path=rgb_path, gray_path=gray_path))
        return metas_

    with fitz.open(pdf) as _doc:
        page_count = len(_doc)

    print(f"[text] start: {pdf.name} -> {book_dir} (pages={page_count})")
    print(f"[text] settings: dpi={dpi}, fmt={fmt}, quality={jpeg_quality}, device={'gpu' if use_gpu else 'cpu'}, lang={lang}, batch={batch_size}")

    # ---- Step 1: render ----
    have_rgb = sum((pages_dir / f"{_expected_stem(i)}.{fmt}").exists() for i in range(page_count))
    have_gray = sum((gray_dir / f"{_expected_stem(i)}.{fmt}").exists() for i in range(page_count))
    if have_rgb == page_count and have_gray == page_count:
        print(f"[text] render: skip (found {page_count} RGB and {page_count} GRAY images)")
        metas = _metas_from_existing(page_count)
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
        results = _run_paddleocr_batch([m.gray_path for m in metas], use_gpu=use_gpu, lang=lang, batch_size=batch_size)
        print(f"[text] ocr: done time={time.time() - t_ocr:.2f}s")
        _save_paddle_jsons(book_dir, results, page_count)

    # ---- Step 3: compile ----
    compiled_path = book_dir / "compiled_ocr.json"
    clean_path = book_dir / "compiled_ocr_clean.json"

    print("[text] compile: building compiled_ocr.json from ocr_json...")
    t_save = time.time()
    compiled = _compile_from_saved(ocr_json_dir, metas)
    write_json(compiled, compiled_path)

    print("[text] compile: building compiled_ocr_clean.json (drop empty texts)...")
    cleaned = _compile_clean_from_saved(ocr_json_dir, metas)
    write_json(cleaned, clean_path)

    # Stats: count texts
    def _count_texts(bundle: Dict[str, Any]) -> int:
        total = 0
        for pg in bundle.get("pages", []):
            if isinstance(pg, dict):
                total += len(pg.get("rec_texts", pg.get("texts", [])) or [])
        return total

    total_raw = _count_texts(compiled)
    total_clean = _count_texts(cleaned)
    print(f"[text] stats: texts_raw={total_raw}, texts_clean={total_clean} time={time.time() - t_save:.2f}s")

    print(f"[text] done: {pdf.name} total_time={time.time() - t0:.2f}s")
