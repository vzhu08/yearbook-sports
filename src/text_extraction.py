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

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter

import fitz  # PyMuPDF
import numpy as np

import paddle  # noqa: F401  (import to initialize Paddle runtime)
from paddleocr import PaddleOCR, PPStructureV3

from src.common.io_utils import ensure_dir, write_json, read_json


# --------------------------- Rendering --------------------------

def _render_page_gray(doc: "fitz.Document", page_index: int, dpi: int) -> np.ndarray:
    """
    Render a single page to an in-memory grayscale uint8 array (H, W).
    No image files are saved.
    """
    scale = dpi / 72.0
    page = doc.load_page(page_index)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY, alpha=False)
    # pix.samples is a bytes-like buffer of length w*h for GRAY
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


# ----------------------------- OCR ------------------------------

def _create_ppstructure(use_gpu: bool, batch_size: int) -> PPStructureV3:
    return PPStructureV3(
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


def _create_plain_ocr() -> PaddleOCR:
    """
    Plain det+rec OCR pipeline (same PP-OCRv5 server models as the structure
    pipeline, same thresholds) used for a SECOND pass over every page.

    PP-StructureV3's own OCR of the same page is markedly worse on small
    roster type: on the 1975 Ayantee's football roster it read "Bastick",
    "Thamas", "McKinnev", "Dexrer" where this pipeline reads "Bostick",
    "Thomas", "McKinney", "Dexter"; it silently dropped whole wrestling-table
    rows ("Ike Smith", "George Harris", "Robert Brown") that this pipeline
    returns; and with textline orientation it reads a rotated caption ("Ron
    Johnson") the structure pass gave as "uosuyor uoy". It is also 5-20x
    faster per page. Layout (blocks/labels/bboxes) still comes from
    PP-StructureV3; only the text is taken from here (see _merge_plain_ocr).
    """
    # Detection thresholds are LOW on purpose: at box_thresh 0.60 the
    # detector dropped whole caption lines that it reads perfectly once
    # detected (1949 Black Bear: "First row: Scotty Rhodarmer, ..." missing
    # from a roster). Recognition confidence (0.70) still keeps junk out.
    return PaddleOCR(
        text_det_limit_side_len=2000,
        text_det_limit_type="max",
        text_det_thresh=0.20,
        text_det_box_thresh=0.30,
        text_rec_score_thresh=0.70,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )


_PLAIN_MERGED_KEY = "plain_ocr_merged"
# Blocks whose block_content is rebuilt from the plain-OCR lines inside them.
_REBUILD_LABELS = {
    "text", "figure_title", "paragraph_title", "doc_title", "header", "image",
    "aside_text", "content", "vision_footnote", "abstract", "footer",
}


def _boxes_from_result(res: Dict[str, Any]) -> List[List[float]]:
    boxes = res.get("rec_boxes")
    if boxes is not None and len(boxes):
        return [[float(v) for v in b] for b in boxes]
    out: List[List[float]] = []
    for poly in res.get("rec_polys") or []:
        xs = [float(pt[0]) for pt in poly]
        ys = [float(pt[1]) for pt in poly]
        out.append([min(xs), min(ys), max(xs), max(ys)])
    return out


def _merge_plain_ocr(plain: PaddleOCR, page_json: Path, bgr: "np.ndarray") -> bool:
    """
    Second OCR pass for one saved page: run the plain pipeline on the page,
    then
      - overall_ocr_res := plain lines + structure lines whose box overlaps
        no plain line (the structure pass still wins the odd big display
        title, e.g. an outline-font "BASEBALL", that plain OCR fragments);
      - block_content of every text-bearing block := the merged lines whose
        centre lies inside the block, in reading order (rows top-to-bottom,
        left-to-right); blocks with no line inside keep their old content;
      - structure_ocr_res keeps the structure pass's original lines.
    Idempotent per page via the plain_ocr_merged flag. Returns True if the
    page was (re)written.
    """
    try:
        data = read_json(page_json)
    except Exception:
        return False
    if data.get(_PLAIN_MERGED_KEY): # if already merged, just skip
        return False

    res = list(plain.predict([bgr]))[0]
    try:
        res = res.json["res"] if hasattr(res, "json") else dict(res)
    except Exception:
        res = dict(res)
    new_texts = [str(t) for t in (res.get("rec_texts") or [])]
    new_scores = [float(x) for x in (res.get("rec_scores") or [])]
    new_boxes = _boxes_from_result(res)
    n = min(len(new_texts), len(new_boxes))
    # combine the results
    new_lines = [
        (new_texts[i], new_scores[i] if i < len(new_scores) else 0.0, new_boxes[i])
        for i in range(n)
        if new_texts[i].strip()
    ]

    old = data.get("overall_ocr_res") or {}
    old_texts = old.get("rec_texts") or []
    old_scores = old.get("rec_scores") or []
    old_boxes = old.get("rec_boxes") or []

    def _overlap(a: List[float], b: List[float]) -> float:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        return inter / area if area > 0 else 0.0

    def _covered(ob: List[float]) -> float:
        """Fraction of an old line's box area covered by new lines."""
        area = max((ob[2] - ob[0]) * (ob[3] - ob[1]), 1.0)
        inter = 0.0
        for _, _, nb in new_lines:
            ix = max(0.0, min(ob[2], nb[2]) - max(ob[0], nb[0]))
            iy = max(0.0, min(ob[3], nb[3]) - max(ob[1], nb[1]))
            inter += ix * iy
        return inter / area

    def _degenerate(new_t: str, old_t: str) -> bool:
        """The plain pass mangled this line where the structure pass had a
        clean read ("aic  oke" for "Janice Hooker"): much shorter, or with
        a run of spaces, or hardly any letters."""
        a, b = new_t.strip(), old_t.strip()
        if "  " in a:
            return True
        if len(a) < 0.6 * len(b):
            return True
        letters = sum(ch.isalpha() for ch in a)
        return len(a) >= 4 and letters < 0.6 * len(a)

    merged: List[Tuple[str, float, List[float]]] = []
    dropped_new: Set[int] = set()
    kept_old: List[Tuple[str, float, List[float]]] = []
    for i, t in enumerate(old_texts):
        if not str(t).strip() or i >= len(old_boxes):
            continue
        ob = [float(v) for v in old_boxes[i]]
        overlapping = [k for k, (_, _, nb) in enumerate(new_lines) if _overlap(ob, nb) >= 0.3]
        if overlapping and _covered(ob) >= 0.5:
            # Same box re-read by the plain pass: it wins unless its read
            # is degenerate ("aic  oke") or clearly less confident than the
            # structure pass's ("Eva Parson" 0.80 vs "Eva Patterson" 0.98),
            # in which case the old line stays and the new one goes. The
            # margin (0.05) is deliberate: at 0.98 vs 0.99 the plain read
            # ("McKinney" vs "McKinnev") is the better one.
            old_sc = float(old_scores[i]) if i < len(old_scores) else 0.0
            new_sc = max(new_lines[k][1] for k in overlapping)
            if all(_degenerate(new_lines[k][0], str(t)) for k in overlapping) or old_sc - new_sc >= 0.05:
                dropped_new.update(overlapping)
            else:
                continue
        elif overlapping:
            # A big display title the plain pass only fragmented
            # ("BASEBALL" -> "ASE", "TRACK" -> "TR"): keep the old line and
            # drop the fragments lying inside it.
            for k in overlapping:
                nb = new_lines[k][2]
                n_area = max((nb[2] - nb[0]) * (nb[3] - nb[1]), 1.0)
                ix = max(0.0, min(ob[2], nb[2]) - max(ob[0], nb[0]))
                iy = max(0.0, min(ob[3], nb[3]) - max(ob[1], nb[1]))
                if ix * iy / n_area >= 0.5:
                    dropped_new.add(k)
        kept_old.append((str(t), float(old_scores[i]) if i < len(old_scores) else 0.0, ob))
    merged = [ln for k, ln in enumerate(new_lines) if k not in dropped_new] + kept_old

    data["structure_ocr_res"] = {
        "rec_texts": old_texts, "rec_scores": old_scores, "rec_boxes": old_boxes,
    }
    data["overall_ocr_res"] = {
        "rec_texts": [m[0] for m in merged],
        "rec_scores": [m[1] for m in merged],
        "rec_boxes": [m[2] for m in merged],
    }

    _rebuild_block_contents(data, merged)

    data[_PLAIN_MERGED_KEY] = True
    write_json(data, page_json)
    return True


# Title/caption blocks: PP-StructureV3 joins their printed lines with a space
# ("Hornsby Howell Head Coach"), which downstream role filters rely on;
# everything else keeps one line per printed line.
_SPACE_JOIN_LABELS = {"figure_title", "paragraph_title", "doc_title", "header"}


def _rebuild_block_contents(data: Dict[str, Any], merged: List[Tuple[str, float, List[float]]]) -> None:
    """block_content := merged lines inside the block, in reading order."""
    blocks = data.get("parsing_res_list")
    if not isinstance(blocks, list):
        return
    for blk in blocks:
        if blk.get("block_label") not in _REBUILD_LABELS:
            continue
        bbox = blk.get("block_bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        if "structure_block_content" not in blk:
            blk["structure_block_content"] = blk.get("block_content", "")
        x0, y0, x1, y1 = (float(v) for v in bbox)
        inside = []
        for t, _, b in merged:
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                inside.append((t, b))
        if not inside:
            continue
        # reading order: rows (y-overlap) top-to-bottom, left-to-right. A
        # tall box (a name set vertically down a photo, "CAROLYN") is its
        # own line and never defines a row, or it would swallow every line
        # beside it.
        inside.sort(key=lambda it: (it[1][1], it[1][0]))
        rows: List[List[Any]] = []
        for it in inside:
            b = it[1]
            tall = (b[3] - b[1]) > 2.5 * max(b[2] - b[0], 1.0)
            cy = (b[1] + b[3]) / 2.0
            if not tall:
                for row in rows:
                    if row[0][2]:
                        continue
                    ry0 = min(r[1][1] for r in row)
                    ry1 = max(r[1][3] for r in row)
                    if ry0 <= cy <= ry1:
                        row.append((it[0], it[1], False))
                        break
                else:
                    rows.append([(it[0], it[1], False)])
            else:
                rows.append([(it[0], it[1], True)])
        lines: List[str] = []
        for row in rows:
            row.sort(key=lambda it: it[1][0])
            lines.append(" ".join(t for t, _, _ in row))
        # Join mode: what the structure pass did for this block when known
        # (space-joined caption vs one line per printed line); otherwise
        # title/caption labels and 1-2 row blocks are space-joined.
        orig = blk.get("structure_block_content")
        if isinstance(orig, str) and orig.strip():
            sep = "\n" if "\n" in orig.strip() else " "
        elif blk.get("block_label") in _SPACE_JOIN_LABELS or len(lines) <= 2:
            sep = " "
        else:
            sep = "\n"
        blk["block_content"] = sep.join(lines) + ("\n" if sep == "\n" else " ")


def _plain_only_pages(pdf: Path, ocr_json_dir: Path, missing: List[int], page_count: int, dpi: int) -> None:
    """OCR pages with the plain pipeline only (no layout blocks), one page
    per predict(), resumable; for portrait/index pages."""
    if not missing:
        return
    ensure_dir(ocr_json_dir)
    print(f"[text] plain-only: {len(missing)} page(s)", flush=True)
    plain = _create_plain_ocr()
    doc = fitz.open(pdf)
    try:
        for n, i in enumerate(missing, start=1):
            t_pg = time.time()
            gray = _render_page_gray(doc, i, dpi=dpi)
            bgr = np.repeat(gray[:, :, None], 3, axis=2)
            target = ocr_json_dir / f"page{(i + 1):04d}.json"
            try:
                res = list(plain.predict([bgr]))[0]
                try:
                    res = res.json["res"] if hasattr(res, "json") else dict(res)
                except Exception:
                    res = dict(res)
                texts = [str(t) for t in (res.get("rec_texts") or [])]
                boxes = _boxes_from_result(res)
                scores = [float(x) for x in (res.get("rec_scores") or [])]
                k = min(len(texts), len(boxes))
                write_json({
                    "page_index": i,
                    "parsing_res_list": [],
                    "overall_ocr_res": {
                        "rec_texts": texts[:k], "rec_scores": scores[:k], "rec_boxes": boxes[:k],
                    },
                    _PLAIN_MERGED_KEY: True,
                    "layout": False,
                }, target)
                print(
                    f"[text] plain-only: page {i+1}/{page_count} saved "
                    f"time={time.time() - t_pg:.2f}s ({n}/{len(missing)})",
                    flush=True,
                )
            except Exception as e:
                print(f"[text] plain-only: ERROR page {i+1}: {e}", flush=True)
    finally:
        doc.close()


def _plain_ocr_pass(pdf: Path, ocr_json_dir: Path, page_count: int, dpi: int) -> None:
    """Run/resume the plain-OCR merge over every saved page that lacks it."""
    todo = [
        i for i in range(page_count)
        if (ocr_json_dir / f"page{(i + 1):04d}.json").exists()
        and not (read_json(ocr_json_dir / f"page{(i + 1):04d}.json") or {}).get(_PLAIN_MERGED_KEY)
        and (read_json(ocr_json_dir / f"page{(i + 1):04d}.json") or {}).get("parsing_res_list")
    ]
    if not todo:
        print("[text] plain-ocr: skip (all pages merged)", flush=True)
        return
    print(f"[text] plain-ocr: {len(todo)} page(s) to merge", flush=True)
    plain = _create_plain_ocr()
    doc = fitz.open(pdf)
    try:
        for n, i in enumerate(todo, start=1):
            t_pg = time.time()
            gray = _render_page_gray(doc, i, dpi=dpi)
            bgr = np.repeat(gray[:, :, None], 3, axis=2)
            target = ocr_json_dir / f"page{(i + 1):04d}.json"
            try:
                _merge_plain_ocr(plain, target, bgr)
                print(
                    f"[text] plain-ocr: page {i+1}/{page_count} merged "
                    f"time={time.time() - t_pg:.2f}s ({n}/{len(todo)})",
                    flush=True,
                )
            except Exception as e:
                print(f"[text] plain-ocr: ERROR page {i+1}: {e}", flush=True)
    finally:
        doc.close()


# Text-bearing labels eligible for the empty-block retry, and the minimum
# region size worth retrying (at 400 DPI ~ half an inch of text height).
# "image" is included because a stylized full-page section divider (the 1975
# Ayantee's "SPORTS" page, PDF p. 91) is detected as one image block with no
# text at all; without the retry the divider never becomes a section header and
# every page before the next header is dropped from the book's sections.
_RETRY_LABELS = {"text", "figure_title", "paragraph_title", "doc_title", "image"}
_RETRY_MIN_W, _RETRY_MIN_H = 200, 50


def _repair_empty_blocks(ocr: PPStructureV3, page_json: Path, bgr: "np.ndarray") -> int:
    """
    Second-chance OCR for caption blocks the layout model FOUND but whose
    text came back empty. Each empty text-bearing block's region is cropped, upscaled 2x
    (nearest-neighbor), and re-OCR'd in isolation; recovered text is patched
    into the saved page JSON. Returns the number of blocks repaired.
    """
    try:
        data = read_json(page_json)
    except Exception:
        return 0

    blocks = data.get("parsing_res_list")
    if not isinstance(blocks, list):
        return 0

    repaired = 0
    h, w = bgr.shape[:2]
    for blk in blocks:
        if blk.get("block_label") not in _RETRY_LABELS:
            continue
        # "Empty" includes content with no letters or digits at all: the
        # 1975 Ayantee's full-page "SPORTS" divider (PDF p. 95) came back as
        # a single image block whose content was just "-", so the divider
        # never became a section header.
        if re.search(r"[A-Za-z0-9]", blk.get("block_content") or ""):
            continue
        bbox = blk.get("block_bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        x0, y0, x1, y1 = (int(v) for v in bbox)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < _RETRY_MIN_W or y1 - y0 < _RETRY_MIN_H:
            continue
        crop = bgr[y0:y1, x0:x1]
        crop2x = np.repeat(np.repeat(crop, 2, axis=0), 2, axis=1)
        try:
            res = list(ocr.predict([crop2x]))[0]
            tmp = page_json.with_suffix(".retry.json")
            res.save_to_json(save_path=str(tmp))
            retry = read_json(tmp)
            tmp.unlink(missing_ok=True)
            texts = (retry.get("overall_ocr_res") or {}).get("rec_texts") or []
            recovered = "\n".join(t for t in texts if t and t.strip())
            if recovered:
                blk["block_content"] = recovered
                repaired += 1
        except Exception as e:
            print(f"[text] retry: ERROR on empty block {bbox}: {e}", flush=True)

    if repaired:
        write_json(data, page_json)
    return repaired


def _ocr_missing_pages_incremental(
    ocr: PPStructureV3,
    pdf: Path,
    ocr_json_dir: Path,
    missing: List[int],
    page_count: int,
    dpi: int,
) -> None:
    """
    OCR pages one at a time, saving each page's JSON immediately.

    One page per predict() call, deliberately:
      - Output for a page cannot depend on which other pages happen to be in
        the same run (PP-StructureV3's results were observed to vary with
        batch composition, occasionally dropping whole caption blocks).
      - Memory stays bounded by a single page (rendered array + result
        object) instead of accumulating every page and every heavy result
        for the whole book, which previously grew to multiple GB on long
        runs and destabilized the process.
      - A crash or kill loses at most the in-flight page; rerunning skips
        every page already saved.
    """
    ensure_dir(ocr_json_dir)

    doc = fitz.open(pdf)
    try:
        for n, i in enumerate(missing, start=1):
            t_pg = time.time()
            gray = _render_page_gray(doc, i, dpi=dpi)
            # PaddleX's current preprocessors expect (H, W, C) input; expand
            # single-channel grayscale to 3 channels before predicting.
            bgr = np.repeat(gray[:, :, None], 3, axis=2)

            target = ocr_json_dir / f"page{(i + 1):04d}.json"
            try:
                results = list(ocr.predict([bgr]))
                res = results[0]
                if hasattr(res, "save_to_json"):
                    res.save_to_json(save_path=str(target))
                else:
                    write_json({"result": res}, target)
                fixed = _repair_empty_blocks(ocr, target, bgr)
                print(
                    f"[text] ocr: page {i+1}/{page_count} saved {target.name} "
                    f"time={time.time() - t_pg:.2f}s ({n}/{len(missing)} this run)"
                    + (f" [repaired {fixed} empty block(s)]" if fixed else ""),
                    flush=True,
                )
            except Exception as e:
                print(f"[text] ocr: ERROR page {i+1}: {e}", flush=True)
    finally:
        doc.close()


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


# --------------------------- Page-range resolution ---------------------------

_PROBE_DPI = 200

# neither first nor second pass
def _probe_lines(plain: PaddleOCR, doc: "fitz.Document", i: int, cache_dir: Path) -> Dict[str, Any]:
    """Cheap plain OCR of one page (200 DPI, ~3 s) for range detection;
    cached in <book>/probe_json/pageNNNN.json."""
    ensure_dir(cache_dir)
    target = cache_dir / f"page{(i + 1):04d}.json"
    if target.exists():
        return read_json(target)
    gray = _render_page_gray(doc, i, dpi=_PROBE_DPI)
    bgr = np.repeat(gray[:, :, None], 3, axis=2)
    res = list(plain.predict([bgr]))[0]
    try:
        res = res.json["res"] if hasattr(res, "json") else dict(res)
    except Exception:
        res = dict(res)
    texts = [str(t) for t in (res.get("rec_texts") or [])]
    boxes = _boxes_from_result(res)
    k = min(len(texts), len(boxes))
    out = {"page": i + 1, "height": int(gray.shape[0]), "width": int(gray.shape[1]),
           "lines": [[texts[j], boxes[j]] for j in range(k) if texts[j].strip()]}
    write_json(out, target)
    return out


def _page_signals(probe: Dict[str, Any]) -> Dict[str, Any]:
    import re as _re
    from src.name_extraction import (  # local import: keeps OCR subprocess light
        _SPORTS_SINGLE_WORD, _SPORTS_MULTI_WORD, _MASCOT_WORDS,
        _is_non_sports_divider_title, _is_sports_umbrella_title,
    )

    def _names_sport(t: str) -> bool:
        # whole words only: "Tracy" is not "track", "Golfredo" is not golf
        low = t.lower()
        words = set(_re.findall(r"[a-z]+", low))
        if words & _SPORTS_SINGLE_WORD:
            return True
        # agent/plural forms of a sport word are still that sport
        # ("SENIOR FOOTBALLERS", "GOLFERS") — but only exact stem+suffix,
        # so "Golfredo"/"Tracy" still never match
        for w in words:
            # "FOOTBALLTEAM": smallcaps titles print with no space
            for suf in ("ers", "er", "s", "team", "teams"):
                if w.endswith(suf) and (
                    w[: -len(suf)] in _SPORTS_SINGLE_WORD
                    or w[: -len(suf)] + "ing" in _SPORTS_SINGLE_WORD  # wrestlers, swimmers
                ):
                    return True
        return any(kw in low for kw in _SPORTS_MULTI_WORD)
    H = float(probe.get("height") or 1)
    sports_title = False
    divider = False
    umbrella = False
    other_title = False
    vocab_hits: set = set()
    pos_lines = 0
    big_sports_title = False
    mascot_title = False
    roster_cue = False
    num_roster = 0
    digit_lines = 0
    printed: List[int] = []

    def _is_mascot_word(w: str) -> bool:
        # "braves"; "bravettes" (girls' team form of a mascot)
        if w in _MASCOT_WORDS or w.rstrip("s") in _MASCOT_WORDS:
            return True
        if w.endswith("ettes"):
            stem = w[:-5]
            return stem in _MASCOT_WORDS or stem + "e" in _MASCOT_WORDS
        return False
    for text, box in probe.get("lines", []):
        t = " ".join(str(text).split())
        words = t.split()
        h = float(box[3]) - float(box[1])
        big = h >= 0.018 * H
        if (
            1 <= len(words) <= 4 and _names_sport(t) and not any(ch.isdigit() for ch in t)
            and not _re.match(r"(?i)^(?:mrs?|miss|ms|dr|rev)\.?\s", t)  # "Mrs. Alva Bowling" is a person
        ):
            sports_title = True
            # a BIG lowercase-led line counts only when it is pure
            # lettering (a single word — jersey/floor print reads
            # lowercase); "wrestler, but you know" in big autograph-page
            # handwriting is prose, not a title
            if big and (t[:1].isupper() or len(words) == 1):
                big_sports_title = True
        if _POSITION_LINE_RE.fullmatch(t):
            pos_lines += 1  # "End", "Guard", "Defensive End" under player photos
        if _ROSTER_CUE_PROBE_RE.search(t):
            roster_cue = True  # "Kneeling: Left to right: ..." — a team-photo caption
        if _re.match(r"\d{1,2}[.)]\s*[A-Z][a-z]", t):
            num_roster += 1  # "7. Randy Snelson" — numbered team-photo key
        if _re.fullmatch(r"\d{1,3}", t):
            digit_lines += 1  # jersey numbers / scores
        if (
            big and 1 <= len(words) <= 4 and not any(ch.isdigit() for ch in t)
            and words and _is_mascot_word(_re.sub(r"[^a-z]", "", words[-1].lower()))
        ):
            # a big team-name title ("P. S. C. Braves", "The Bravettes"):
            # only counts as a sports section when corroborated by nearby
            # sports vocabulary — see find_sports_pages
            mascot_title = True
        if _is_sports_umbrella_title(t) and big:
            umbrella = True
        if _is_non_sports_divider_title(t) and big:
            divider = True
        if big and 1 <= len(words) <= 4 and not _names_sport(t) and not _is_sports_umbrella_title(t) \
                and not any(ch.isdigit() for ch in t) and t.upper() == t and sum(ch.isalpha() for ch in t) >= 5:
            other_title = True
        if t.isdigit() and len(t) <= 3:
            cy = (float(box[1]) + float(box[3])) / 2.0
            if cy < 0.12 * H or cy > 0.88 * H:
                printed.append(int(t))
        for m in _SPORTS_VOCAB_RE.finditer(t):
            vocab_hits.add(m.group(0).lower())
    # two DIFFERENT sports words on the page ("Coach", "halftime"): sports
    # coverage; a lone "team" in an advertisement is not
    sports_words = len(vocab_hits) >= 2 or pos_lines >= 3
    # A table of contents lists every division of the book — "SPORTS" there
    # is a pointer, not a section. Detected by the literal phrase (possibly
    # split across lines) or by a sports umbrella AND a non-sports divider
    # title AND bare page-number lines sharing one page, which no real
    # divider page ever has.
    joined = " ".join(" ".join(str(t).split()) for t, _ in probe.get("lines", [])).lower()
    num_lines = sum(
        1 for t, _ in probe.get("lines", [])
        if re.fullmatch(r"\d{1,3}", " ".join(str(t).split()))
    )
    if "table of contents" in joined or (umbrella and divider and num_lines >= 2):
        sports_title = big_sports_title = umbrella = False
    return {"sports_title": sports_title, "divider": divider, "umbrella": umbrella,
            "other_title": other_title, "sports_words": sports_words,
            "big_sports_title": big_sports_title,
            "mascot_title": mascot_title, "vocab_n": len(vocab_hits),
            "pos_lines": pos_lines, "roster_cue": roster_cue,
            "num_roster": num_roster, "digit_lines": digit_lines,
            "printed": printed, "n_lines": len(probe.get("lines", []))}


_ROSTER_CUE_PROBE_RE = re.compile(
    r"(?i)(?:left\s+to\s+right|right\s+to\s+left|kneeling|first\s+row|"
    r"second\s+row|third\s+row|front\s+row|back\s+row|row\s+one|row\s+two)"
)

_POSITION_LINE_RE = re.compile(
    r"(?:(?:defensive|offensive|left|right|center|split|tight|wide|first|second|third|short)\s+)?"
    r"(?:end|guard|tackle|center|forward|halfback|fullback|quarterback|linebacker|back|"
    r"pitcher|catcher|shortstop|infielder|outfielder|fielder|baseman|goalie|goalkeeper|"
    r"wing|safety|cornerback|receiver|kicker|punter)\.?",
    re.IGNORECASE,
)

# Words that mark a page as sports COVERAGE even without a sport title (action
# pages between a team's title page and the next section).
_SPORTS_VOCAB_RE = re.compile(
    r"\b(?:coach(?:es|ing)?|team|season|score[sd]?|victory|defeat|touchdown|halftime|"
    r"tournament|championship|players?|practice|opponents?|game|games|athlet\w*|"
    r"varsity|captain|co-captain|scrimmage|kickoff|inning|quarterback|dribbl\w*|"
    r"rebound\w*|tackle[sd]?|pitch\w*|batt\w*|serve[sd]?|match(?:es)?|meet|relay|"
    r"cheerleaders?|homecoming game)\b",
    re.IGNORECASE,
)


def resolve_sports_range(
    pdf_path: str,
    out_dir: str,
    hint: Tuple[int, int],
    max_extend: int = 40,
) -> Tuple[int, int]:
    """
    Turn a user-supplied page range into the sports section's actual PDF
    range, using cheap OCR probes:
      1. if no page of the hint range shows a sports title, the hint was
         probably PRINTED page numbers — read the printed folios on those
         pages, derive the PDF offset and shift the range;
      2. extend the end forward while sports titles keep appearing (up to
         two photo-only pages in a row are tolerated), stopping at a big
         non-sports divider title ("ADVERTISEMENTS", "CLASSES", ...);
      3. extend the start back over an immediately preceding sports page or
         "SPORTS"/"ATHLETICS" divider.
    """
    pdf = Path(pdf_path)
    cache = Path(out_dir) / pdf.stem / "probe_json"
    plain = _create_plain_ocr()
    doc = fitz.open(pdf)
    try:
        n = len(doc)
        lo, hi = max(1, hint[0]), min(n, hint[1])

        def sig(p: int) -> Dict[str, Any]:
            return _page_signals(_probe_lines(plain, doc, p - 1, cache))

        sigs = {p: sig(p) for p in range(lo, hi + 1)}
        if not any(v["sports_title"] or v["umbrella"] for v in sigs.values()):
            offsets = Counter()
            for p, v in sigs.items():
                for pr in v["printed"]:
                    if 0 <= p - pr <= 30:
                        offsets[p - pr] += 1
            if offsets:
                off, cnt = offsets.most_common(1)[0]
                if off > 0 and cnt >= 2:
                    lo2, hi2 = lo + off, min(n, hi + off)
                    sigs2 = {p: sig(p) for p in range(lo2, hi2 + 1)}
                    if any(v["sports_title"] or v["umbrella"] for v in sigs2.values()):
                        print(f"[range] hint {hint} looks like printed page numbers; "
                              f"PDF offset +{off} -> {lo2}-{hi2}", flush=True)
                        lo, hi, sigs = lo2, hi2, sigs2

        # forward
        p, misses, last_hit = hi + 1, 0, hi
        while p <= n and p - hi <= max_extend and misses < 4:
            v = sig(p)
            if v["divider"]:
                break
            if v["sports_title"] or v["umbrella"] or v.get("sports_words"):
                last_hit, misses = p, 0
            else:
                misses += 1
            p += 1
        if last_hit > hi:
            print(f"[range] sports content continues past {hi}: extending to {last_hit}", flush=True)
            hi = last_hit
        # backward
        for p in (lo - 1, lo - 2):
            if p < 1:
                break
            v = sig(p)
            if v["umbrella"] or v["sports_title"]:
                lo = p
            else:
                break
        print(f"[range] resolved sports pages: {lo}-{hi} (hint {hint})", flush=True)
        _save_sports_pages(out_dir, pdf.stem, [(lo, hi)])
        return (lo, hi)
    finally:
        doc.close()


def _save_sports_pages(out_dir: str, stem: str, ranges: List[Tuple[int, int]]) -> None:
    """Record the detected sports page ranges for name_extraction: every
    section lying inside them is a sports section even when its own title is
    a headline ("Braves Have A Disappointing Season") rather than a sport."""
    book_dir = Path(out_dir) / stem
    ensure_dir(book_dir)
    prev: List[List[int]] = []
    fp = book_dir / "sports_pages.json"
    if fp.exists():
        try:
            prev = read_json(fp).get("ranges", [])
        except Exception:
            prev = []
    merged = sorted({tuple(r) for r in prev} | {tuple(r) for r in ranges})
    write_json({"ranges": [list(r) for r in merged]}, fp)


_BRIDGE_MAX_GAP = 30


def find_sports_pages(pdf_path: str, out_dir: str, gap: int = 2) -> List[Tuple[int, int]]:
    """
    Whole-book scan for sports pages when no range is known (sports may be
    scattered: fall sports up front, basketball mid-book, spring sports at
    the back). Probes EVERY page cheaply and returns the page ranges that
    carry sports titles, merged when at most `gap` non-sports pages
    separate them and widened to include an adjacent "SPORTS"/"ATHLETICS"
    divider. Layout OCR then only runs on those pages.
    """
    pdf = Path(pdf_path)
    cache = Path(out_dir) / pdf.stem / "probe_json"
    plain = _create_plain_ocr()
    doc = fitz.open(pdf)
    try:
        n = len(doc)
        sigs: Dict[int, Dict[str, Any]] = {}
        for p in range(1, n + 1):
            sigs[p] = _page_signals(_probe_lines(plain, doc, p - 1, cache))
    finally:
        doc.close()
    out = _build_sports_ranges(sigs, n, gap)
    if not out:
        # FALLBACK, only when nothing at all was found: a big team-name
        # (mascot) title stands in for a sports title when a page within 2
        # carries sports vocabulary or position labels — the 1954
        # Indianhead's basketball pages are titled only "P. S. C. Braves" /
        # "The Bravettes". Never applied when regular titles exist: mascot
        # titles also head pep/homecoming pages.
        for p, v in sigs.items():
            if v.get("mascot_title") and not v["sports_title"] and any(
                sigs.get(q, {}).get("vocab_n", 0) >= 1 or sigs.get(q, {}).get("pos_lines", 0) >= 1
                for q in range(p - 2, p + 3)
            ):
                v["sports_title"] = True
                v["big_sports_title"] = True
        out = _build_sports_ranges(sigs, n, gap)
    if not out:
        # LAST fallback: a dense cluster of sports-coverage pages whose
        # section titles never name a sport at all (1964 Nuhosca headlines
        # its athletics block "Big Eleven", "Letter Men", "Time Out!") —
        # runs of sports_words pages (gaps of at most 2 pages) at least 4
        # pages long that contain at least one sports_title hit. Senior
        # activity-list pages ("Basketball—1, 2, 3") never qualify: their
        # coverage is scattered singles and their digit-bearing lines
        # yield no sports_title.
        runs: List[List[int]] = []
        for p in range(1, n + 1):
            if sigs[p].get("sports_words"):
                if runs and p - runs[-1][-1] <= 3:
                    runs[-1].append(p)
                else:
                    runs.append([p])
        out = [
            (r[0], r[-1]) for r in runs
            if len(r) >= 4
            and any(sigs[q]["sports_title"] for q in range(r[0], r[-1] + 1))
        ]
    print(f"[range] sports pages found: {out}", flush=True)
    _save_sports_pages(out_dir, pdf.stem, out)
    return out


def _build_sports_ranges(
    sigs: Dict[int, Dict[str, Any]], n: int, gap: int
) -> List[Tuple[int, int]]:
    # a page carrying a big NON-sports division title ("Juniors") is a
    # class/divider page whatever else is printed on it (the 1972
    # Indianhead's junior-class activity calendar prints "Baseball" big) —
    # never a sports hit
    hits = [p for p, v in sigs.items() if v["sports_title"] and not v["divider"]]
    # a lone page whose only signal is a small line with a sport word in it
    # ("Golf" in a staff member's title) is not a sports page
    hits = [
        p for p in hits
        if sigs[p]["sports_words"] or sigs[p]["big_sports_title"] or sigs[p]["umbrella"]
        or (p - 1 in sigs and sigs[p - 1]["sports_title"]) or (p + 1 in sigs and sigs[p + 1]["sports_title"])
    ]
    ranges: List[Tuple[int, int]] = []
    for p in hits:
        if ranges and p - ranges[-1][1] <= gap + 1:
            ranges[-1] = (ranges[-1][0], p)
        else:
            ranges.append((p, p))
    # A range with no BIG sports title and no umbrella page anywhere in it
    # is small-print incidental sports vocabulary, not a sports section (the
    # 1976 Falcon's Senior Directory lists "Org: Basketball Team" under
    # every other portrait) — UNLESS it bridges to a strong range below
    # (headline-style section titles — "HHS Golfers Fall Short of Season
    # Expectations" — never yield a big title, yet sit right beside pages
    # that do). Tag instead of dropping; the bridge step keeps only groups
    # holding at least one strong range.
    strong_ranges = {
        (lo, hi) for lo, hi in ranges
        if any(sigs[p]["big_sports_title"] or sigs[p]["umbrella"] for p in range(lo, hi + 1))
    }
    out: List[Tuple[int, int]] = []
    out_strong: List[bool] = []
    for lo, hi in ranges:
        rng_strong = (lo, hi) in strong_ranges
        # a run of sports titles is followed by its photo/action pages: keep
        # going over pages without any heading (up to two in a row) until a
        # non-sports title or divider shows up
        p, misses = hi + 1, 0
        cue_used = False
        while p <= n and misses < 5:
            v = sigs.get(p)
            if v is None:
                break
            if v["divider"] and v.get("digit_lines", 0) < 3:
                # a digit-heavy "divider" is a scoreboard/jersey spread
                # ("ATOMICS -VS- FACULTY"), not a division page
                break
            if v.get("roster_cue") and p == hi + 1 and not cue_used:
                # a team-photo caption ("Kneeling: Left to right: ...")
                # DIRECTLY continuing the range is the section's roster
                # spilling onto the next page (1972 Indianhead wrestling).
                # ONE page only, and only direct continuation — every club
                # photo in the book has such a caption too, and unlimited
                # cue-chaining would walk straight through the club section
                # (1969 Chowanoka's Y.W.A./B.S.U. pages)
                cue_used = True
                hi, misses = p, 0
                p += 1
                continue
            if v["other_title"] and v.get("digit_lines", 0) < 3 and not (
                v["sports_title"] or v["umbrella"] or v.get("sports_words")
            ):
                # a big non-sports title with NO sports coverage at all —
                # the next division has begun (1974 Highlander: the senior
                # portrait grids right after girls' basketball would
                # otherwise flood the last sports section). Digit-heavy
                # pages are jersey/score spreads, never division starts.
                break
            if (
                v["sports_title"] or v["umbrella"] or v.get("sports_words")
                or v.get("num_roster", 0) >= 3
            ):
                # a numbered team-photo key ("1. Kenny Ball ... 12. Mike
                # Norman", 1977 Mountain Tech basketball) is roster coverage
                hi, misses = p, 0
            else:
                misses += 1
            p += 1
        # walk BACKWARD the same way: section titles that never name the
        # sport ("Eagles Break Jinx" football pages) still carry sports
        # coverage; stop at a divider or a titled non-sports page
        p, misses = lo - 1, 0
        while p >= 1 and misses < 5:
            v = sigs.get(p)
            if v is None:
                break
            if v["divider"] and v.get("digit_lines", 0) < 3:
                break
            if v["other_title"] and v.get("digit_lines", 0) < 3 and not (
                v["sports_title"] or v["umbrella"] or v.get("sports_words")
            ):
                break
            if (
                v["sports_title"] or v["umbrella"] or v.get("sports_words")
                or v.get("num_roster", 0) >= 3
            ):
                lo, misses = p, 0
            else:
                misses += 1
            p -= 1
        while lo - 1 >= 1 and sigs[lo - 1]["umbrella"]:
            lo -= 1
        if out and lo - out[-1][1] <= gap + 1:
            out[-1] = (out[-1][0], max(hi, out[-1][1]))
            out_strong[-1] = out_strong[-1] or rng_strong
        else:
            out.append((lo, hi))
            out_strong.append(rng_strong)
    # Bridge two runs when nothing but sports coverage can lie between them:
    # no divider and no non-sports section title in the gap (photo spreads,
    # scoreboards and jersey lettering carry no titles at all — the 1978
    # ECSU's football/basketball pages split the athletics section into
    # five islands). Jersey caps do fire "other_title", so a gap page only
    # blocks bridging when its big title is a real phrase (two+ words or 8+
    # letters without digits nearby was too fiddly — instead: divider pages
    # and pages whose big title is in the divider vocabulary block, plus a
    # hard cap on the gap).
    merged: List[Tuple[int, int]] = []
    merged_strong: List[bool] = []
    for (lo, hi), rng_strong in zip(out, out_strong):
        # the long bridge is for STRONG islands only (ECSU's five football/
        # basketball islands); a weak range must sit adjacent to a strong
        # one (handled above) or it is incidental vocabulary — the 1947
        # Green Lights' "Sports Editor" line on the annual-staff page must
        # not drag the whole club block into the sports range.
        if merged and rng_strong and merged_strong[-1] and lo - merged[-1][1] - 1 <= _BRIDGE_MAX_GAP and not any(
            # a "divider" page that also carries sports coverage is a
            # sports headline misread ("Seniors and Juniors Compose Team"
            # over a girls' basketball spread) — it doesn't block
            sigs.get(p, {}).get("divider") and not sigs.get(p, {}).get("sports_words")
            for p in range(merged[-1][1] + 1, lo)
        ):
            merged[-1] = (merged[-1][0], hi)
            merged_strong[-1] = merged_strong[-1] or rng_strong
        else:
            merged.append((lo, hi))
            merged_strong.append(rng_strong)
    return [r for r, s in zip(merged, merged_strong) if s]


# ------------------------------- Entry Point -------------------------------

def extract_text(
    pdf_path: str,
    out_dir: str,
    # 400 DPI (up from 200): at 200, PP-StructureV3 missed entire stylized
    # section titles (e.g. "Golf Team" produced no block at all) and misread
    # thin glyphs ("Bouler" -> "Boule1"); 400 fixed both on the same pages,
    # and per-page timing was no worse than at 300.
    dpi: int = 400,
    lang: str = "en",
    use_gpu: bool = True,
    batch_size: int = 64,
    page_range: Optional[Tuple[int, int]] = None,
    layout: bool = True,
) -> None:
    """
    Integrated run for Step 1.
    Writes a single file: compiled_ocr.json

    Pages are rendered and OCR'd ONE AT A TIME (see
    _ocr_missing_pages_incremental for why), each page's raw JSON saved to
    ocr_json/ immediately. Pages whose JSON already exists are skipped, so an
    interrupted run resumes where it left off instead of redoing the book.

    page_range=(first, last) (1-indexed, inclusive) restricts OCR to those
    pages; pages without a saved JSON simply come out empty in
    compiled_ocr.json (the whole book is still compiled).

    layout=False skips PP-StructureV3 entirely and saves only the plain-OCR
    lines (no blocks) — ~10 s/page instead of minutes. Meant for the
    class-portrait pages that only feed the name index
    (src/name_resolution.py), never for the sports pages themselves.
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

    # ---- Steps 1+2: render + OCR missing pages, one page per predict() ----
    wanted = range(page_count)
    if page_range:
        lo, hi = page_range
        wanted = range(max(0, lo - 1), min(page_count, hi))
        print(f"[text] page_range: {lo}-{hi}", flush=True)
    missing = [
        i for i in wanted
        if not (ocr_json_dir / f"page{(i + 1):04d}.json").exists()
    ]
    if not layout:
        _plain_only_pages(pdf, ocr_json_dir, missing, page_count, dpi=dpi)
        missing = []
    if not missing:
        print(f"[text] ocr: skip (found {page_count} ocr_json pages)", flush=True)
    else:
        print(
            f"[text] ocr: {len(missing)}/{page_count} pages to process "
            f"(per-page, resumable)",
            flush=True,
        )
        t_ocr = time.time()
        ocr = _create_ppstructure(use_gpu=use_gpu, batch_size=batch_size)
        _ocr_missing_pages_incremental(
            ocr, pdf, ocr_json_dir, missing, page_count, dpi=dpi
        )
        print(f"[text] ocr: done time={time.time() - t_ocr:.2f}s", flush=True)

    # ---- Step 2b: plain-OCR second pass (better line text; resumable) ----
    _plain_ocr_pass(pdf, ocr_json_dir, page_count, dpi=dpi)

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

    # write as a json file
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
