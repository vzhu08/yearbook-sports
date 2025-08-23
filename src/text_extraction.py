#!/usr/bin/env python3
"""
Yearbook OCR extraction (PaddleOCR via subprocess), header grouping (spillover across pages),
and optional name correction.

Pipeline:
1) Parallel PDF render + preproc (thread pool; JPG I/O).
2) Batched PaddleOCR.predict(...) -> save res JSONs/overlays AFTER inference.
3) Build compiled RAW json (normalized + lowercased).
4) Build compiled CLEAN json (remove non-alnum-only boxes and very short text).
5) Identify headers using book-wide median text height (>150%) with NO keyword filter.
6) Run spaCy PERSON NER over all spans (multi-entity per span, names must have ≥2 words)
   and assign names to the PREVIOUS header globally (spills across pages until the next header).
7) Write final headers_with_names.json.

Notes:
- RAW/CLEAN JSON use ensure_ascii=True so non-ASCII shows as \\uXXXX (safe for viewing).
- Name correction is optional; no seniors/sports split in the final output.
"""

from __future__ import annotations
import os
import re
import json
import time
import unicodedata
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median

import cv2
import fitz  # PyMuPDF
import spacy

from .ocr_helper import run_paddle_ocr
from .name_correction import maybe_load_corrector


# -----------------------
# Geometry / text helpers
# -----------------------

def _normalize_text(s: str) -> str:
    """Unicode-normalize, collapse whitespace, strip; caller lowercases afterward."""
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _to_lower_ascii(s: str) -> str:
    """Lowercase for logic; JSON writing uses ensure_ascii=True so \\uXXXX is fine."""
    return (s or "").lower()

def _box_height_from_poly(poly: List[List[float]]) -> float:
    try:
        ys = [float(p[1]) for p in poly[:4]]
        return max(ys) - min(ys)
    except Exception:
        return 0.0

def _box_center_y_from_poly(poly: List[List[float]]) -> float:
    try:
        ys = [float(p[1]) for p in poly[:4]]
        return min(ys) + (max(ys) - min(ys)) / 2.0
    except Exception:
        return 0.0

def _has_any_alnum(s: str) -> bool:
    return bool(re.search(r"[a-z0-9]", s))

def _two_or_more_words(s: str) -> bool:
    """At least two word tokens that contain letters (handles commas/extra spaces)."""
    tokens = [t for t in re.split(r"\s+", s.strip()) if re.search(r"[A-Za-z]", t)]
    return len(tokens) >= 2

def _extract_persons(text: str, nlp) -> List[str]:
    """Extract multiple PERSON entities from a span; keep only names with ≥2 words."""
    if not text:
        return []
    doc = nlp(text)
    out: List[str] = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            nm = ent.text.strip()
            # require at least 2 words and at least one ASCII letter overall
            if _two_or_more_words(nm) and re.search(r"[A-Za-z]", nm):
                out.append(nm)
    # de-dup preserving order
    seen = set()
    uniq: List[str] = []
    for n in out:
        k = n.lower()
        if k not in seen:
            uniq.append(n)
            seen.add(k)
    return uniq

# position compare helpers (page index + cy within page)
def _pos_lt(a_pi: int, a_cy: float, b_pi: int, b_cy: float) -> bool:
    return (a_pi < b_pi) or (a_pi == b_pi and a_cy < b_cy)

def _pos_le(a_pi: int, a_cy: float, b_pi: int, b_cy: float) -> bool:
    return (a_pi < b_pi) or (a_pi == b_pi and a_cy <= b_cy)


# -----------------------
# Main extraction
# -----------------------

def extract_yearbook_ocr(
    *,
    pdf_path: str,
    output_dir: str,
    year: int,
    dpi: int = 300,
    start_page: int = 1,
    end_page: Optional[int] = None,
    resize_max_side: Optional[int] = 2400,
    use_otsu: bool = True,
    header_thresh = 2,
    fuzz_threshold: int = 85,
    name_correction_enabled: bool = False,
    first_name_corpus_path: str = "name_data/first_names_by_decade.json",
    last_name_corpus_path: str  = "name_data/last_names_by_decade.json",
    paddle_device: str = "gpu",
    paddle_batch_size: int = 64,
    save_overlays: bool = False,
    preproc_workers: Optional[int] = None,   # None -> (cores-2); provided -> capped at (cores-2)
) -> None:
    """
    See module docstring for the full pipeline description.
    """
    # Decide worker count up-front
    hw = os.cpu_count() or 4
    max_allowed = max(1, hw - 2)
    if preproc_workers is None:
        num_workers = max_allowed
    else:
        num_workers = max(1, min(int(preproc_workers), max_allowed))

    print(f"\n========== [START] OCR extract ==========")
    print(f"[CONFIG] pdf='{pdf_path}'  out='{output_dir}'  year={year}")
    print(f"[CONFIG] dpi={dpi}  pages={start_page}..{('end' if end_page is None else end_page)}")
    print(f"[CONFIG] preproc: otsu={use_otsu}  threads={num_workers} (cpu={hw}, cap={max_allowed})")
    print(f"[CONFIG] paddle: device='{paddle_device}'  batch_size={paddle_batch_size}  overlays={save_overlays}")
    print("=========================================\n")

    t0 = time.perf_counter()

    # Dirs
    os.makedirs(output_dir, exist_ok=True)
    pages_dir   = os.path.join(output_dir, "pages")
    pre_dir     = os.path.join(output_dir, "preproc")
    json_dir    = os.path.join(output_dir, "ocr_json")     # Paddle's save_to_json output
    overlay_dir = os.path.join(output_dir, "ocr_images")   # Paddle's save_to_img output
    for d in (pages_dir, pre_dir, json_dir, overlay_dir):
        os.makedirs(d, exist_ok=True)

    compiled_raw_path   = os.path.join(output_dir, "compiled_ocr.json")        # RAW (normalized+lower)
    compiled_clean_path = os.path.join(output_dir, "compiled_ocr_clean.json")  # CLEAN (filtered)
    final_headers_path  = os.path.join(output_dir, "headers_with_names.json")  # FINAL

    # spaCy
    print("[NLP] Loading spaCy models...")
    try:
        nlp = spacy.load("en_core_web_trf")
        print("[NLP] Loaded en_core_web_trf")
    except Exception:
        try:
            from spacy.cli import download
            print("[NLP] Downloading en_core_web_trf...")
            download("en_core_web_trf")
            nlp = spacy.load("en_core_web_trf")
            print("[NLP] Loaded en_core_web_trf after download")
        except Exception as e:
            print(f"[NLP][WARN] transformer failed ({type(e).__name__}: {e}); falling back to en_core_web_sm.")
            nlp = spacy.load("en_core_web_sm")
            print("[NLP] Loaded en_core_web_sm")

    print(f"[NLP] Name correction enabled? {name_correction_enabled}")
    corrector = maybe_load_corrector(
        enabled=name_correction_enabled,
        first_name_corpus_path=first_name_corpus_path,
        last_name_corpus_path=last_name_corpus_path,
        year=year,
        nlp=nlp,
    )

    # Page keys
    print("[PDF] Opening PDF and enumerating pages...")
    doc = fitz.open(pdf_path)
    total = len(doc)
    last_pg = total if end_page is None or end_page > total else end_page
    keys = [f"page{i:03d}" for i in range(start_page, last_pg + 1)]
    print(f"[PDF] Total pages in doc: {total} | Processing keys: {keys[0]}..{keys[-1]} (count={len(keys)})")

    # --------- PREPROC ----------
    print("\n---------- [STAGE] PREPROC ----------")
    t_pre0 = time.perf_counter()

    JPEG_QUALITY = 95  # save both original & preproc as JPG

    def _prep_one(key: str) -> str:
        page_jpg = os.path.join(pages_dir, f"{key}.jpg")
        pre_jpg  = os.path.join(pre_dir,   f"{key}.jpg")

        if not os.path.exists(page_jpg):
            pg_idx = int(key[-3:]) - 1
            # Pixmap.save infers JPEG from .jpg extension
            doc.load_page(pg_idx).get_pixmap(dpi=dpi).save(page_jpg)

        if not os.path.exists(pre_jpg):
            img = cv2.imread(page_jpg, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read {page_jpg}")
            if use_otsu:
                _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                bw = img
            # Downscale if requested
            if resize_max_side and resize_max_side > 0:
                h, w = bw.shape[:2]
                long_side = max(h, w)
                if long_side > resize_max_side:
                    scale = resize_max_side / float(long_side)
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    bw = cv2.resize(bw, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(pre_jpg, bw, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        return pre_jpg

    pre_imgs: List[str] = []
    print(f"[PREPROC] Using {num_workers} worker threads")
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(_prep_one, key): key for key in keys}
        for i, fut in enumerate(as_completed(futs), start=1):
            try:
                pre_imgs.append(fut.result())
            except Exception as e:
                k = futs[fut]
                print(f"[PREPROC][WARN] {k}: {type(e).__name__}: {e}")
            if i % 5 == 0 or i == len(keys):
                print(f"[PREPROC] progress: {i}/{len(keys)} pages ready")

    pre_imgs.sort()  # ensure page order
    print(f"[PREPROC] Completed. Images ready: {len(pre_imgs)} | First: {os.path.basename(pre_imgs[0])}  Last: {os.path.basename(pre_imgs[-1])}")
    t_pre1 = time.perf_counter()
    print(f"[PREPROC][TIMING] {t_pre1 - t_pre0:.2f}s\n")

    # --------- OCR ----------
    print("---------- [STAGE] OCR INFERENCE ----------")

    existing_jsons = set(fn for fn in os.listdir(json_dir) if fn.endswith(".json"))

    def _has_res_json(key: str) -> bool:
        if f"{key}.json" in existing_jsons:
            return True
        for fn in existing_jsons:
            if fn.startswith(key) and fn.endswith(".json"):
                return True
        return False

    todo_pairs = [(k, img) for k, img in zip(keys, pre_imgs) if not _has_res_json(k)]
    todo_imgs = [img for _, img in todo_pairs]

    skipped = len(keys) - len(todo_imgs)
    print(f"[OCR] cached pages: {skipped}  | to_run: {len(todo_imgs)}  | device='{paddle_device}'  batch_size={paddle_batch_size}  overlays={save_overlays}")

    # guard timing vars in case OCR is skipped
    t_ocr0 = t_ocr1 = time.perf_counter()

    if not todo_imgs:
        print("[OCR] All pages already have res JSON; skipping OCR call.\n")
    else:
        t_ocr0 = time.perf_counter()
        srv = run_paddle_ocr(
            todo_imgs,
            device=paddle_device,
            lang="en",
            batch_size=paddle_batch_size,
            save_json_dir=json_dir,
            save_img_dir=(overlay_dir if save_overlays else None),
        )
        if srv.get("errors"):
            print(f"[OCR][WARN] service reported errors ({len(srv['errors'])}): {srv['errors']}")
        res_map = srv.get("results", {}) or {}
        print(f"[OCR] Service returned summaries for {len(res_map)}/{len(todo_imgs)} new images")
        for _, img in todo_pairs:
            base = os.path.basename(img)
            info = res_map.get(img) or {}
            print(f"[OCR][{base}] count={info.get('count')} device={info.get('device')}")
        t_ocr1 = time.perf_counter()
        print(f"[OCR][TIMING] {t_ocr1 - t_ocr0:.2f}s\n")

    # --------- COMPILE (RAW + CLEAN) ----------
    print("---------- [STAGE] COMPILE OCR JSON (RAW & CLEAN) ----------")
    t_cmp0 = time.perf_counter()

    def _choose_polys(d: Dict[str, Any]) -> List[List[List[float]]]:
        for k in ("rec_polys", "dt_polys", "polys", "boxes"):
            v = d.get(k)
            if isinstance(v, list) and v:
                return v
        return []

    # Build RAW (normalized + lower); CLEAN (filtered)
    raw_pages: Dict[str, Dict[str, Any]] = {}
    clean_pages: Dict[str, Dict[str, Any]] = {}

    total_texts_raw = 0
    total_texts_clean = 0
    total_polys_raw = 0
    total_polys_clean = 0
    missing_pages = 0

    for key in keys:
        pj = os.path.join(json_dir, f"{key}.json")
        if not os.path.exists(pj):
            matches = [fn for fn in os.listdir(json_dir) if fn.startswith(key) and fn.endswith(".json")]
            if matches:
                pj = os.path.join(json_dir, sorted(matches)[0])
            else:
                print(f"[COMPILE][WARN] Missing res JSON for {key}; skipping this page.")
                missing_pages += 1
                continue

        try:
            with open(pj, "r", encoding="utf-8") as f:
                jd = json.load(f)
            res = jd.get("res", jd) or {}
        except Exception as e:
            print(f"[COMPILE][WARN] Bad res JSON for {key}: {type(e).__name__}: {e}")
            continue

        rec_texts_src  = res.get("rec_texts") or []
        rec_scores     = [float(s) for s in (res.get("rec_scores") or [])]
        polys          = _choose_polys(res)

        # Normalize + lowercase for RAW
        texts_norm_lower = [_to_lower_ascii(_normalize_text(t)) for t in rec_texts_src]

        raw_pages[key] = {"polys": polys, "rec_texts": texts_norm_lower, "rec_scores": rec_scores}
        total_texts_raw  += len(texts_norm_lower)
        total_polys_raw  += len(polys)

        # CLEAN: remove boxes with no letters/digits OR with <= 2 non-space chars
        texts_clean: List[str] = []
        polys_clean: List[List[List[float]]] = []
        scores_clean: List[float] = []

        for t, p, sc in zip(texts_norm_lower, polys, rec_scores):
            no_space_len = len(t.replace(" ", ""))
            if not _has_any_alnum(t) or no_space_len <= 2:
                continue
            texts_clean.append(t)
            polys_clean.append(p)
            scores_clean.append(sc)

        clean_pages[key] = {"polys": polys_clean, "rec_texts": texts_clean, "rec_scores": scores_clean}
        total_texts_clean += len(texts_clean)
        total_polys_clean += len(polys_clean)

    # Write RAW and CLEAN (ASCII-escaped)
    with open(compiled_raw_path, "w", encoding="utf-8") as f:
        json.dump({"pages": raw_pages}, f, indent=2, ensure_ascii=True)
    with open(compiled_clean_path, "w", encoding="utf-8") as f:
        json.dump({"pages": clean_pages}, f, indent=2, ensure_ascii=True)

    print(f"[WRITE] RAW   -> {compiled_raw_path}   (pages={len(raw_pages)}  texts={total_texts_raw}  polys={total_polys_raw})")
    print(f"[WRITE] CLEAN -> {compiled_clean_path} (pages={len(clean_pages)} texts={total_texts_clean} polys={total_polys_clean})")
    t_cmp1 = time.perf_counter()
    print(f"[COMPILE][TIMING] {t_cmp1 - t_cmp0:.2f}s\n")

    # --------- HEADERS + NAMES (GLOBAL SPILLOVER) ----------
    print("---------- [STAGE] HEADERS & NAMES (spillover) ----------")
    t_post0 = time.perf_counter()

    # 1) Compute book-wide median text height using CLEAN pages
    heights: List[float] = []
    for page in clean_pages.values():
        for poly in page.get("polys", []):
            h = _box_height_from_poly(poly)
            if h > 0:
                heights.append(h)
    med_h = median(heights) if heights else 0.0
    thresh_h = header_thresh * med_h if med_h > 0 else float("inf")
    print(f"[HEADERS] median_h={med_h:.2f}  threshold(>150%)={thresh_h:.2f}")

    # 2) Find headers per page (height > 1.5 * median). No keyword filtering.
    headers_by_page: Dict[str, List[Dict[str, Any]]] = {}
    for key, page in clean_pages.items():
        texts = page.get("rec_texts", []) or []
        polys = page.get("polys", []) or []
        headers: List[Dict[str, Any]] = []
        for t, poly in zip(texts, polys):
            h = _box_height_from_poly(poly)
            if h > thresh_h:
                headers.append({"text": t, "poly": poly, "h": h, "cy": _box_center_y_from_poly(poly)})
        headers.sort(key=lambda x: x["cy"])  # top → bottom
        headers_by_page[key] = headers
        print(f"[HEADERS][{key}] found={len(headers)}")

    # ---- Build GLOBAL header markers (sorted across the whole book) ----
    page_index = {k: i for i, k in enumerate(keys)}

    header_markers: List[Dict[str, Any]] = []
    for k, headers in headers_by_page.items():
        for h in headers:
            header_markers.append({
                "page_key": k,
                "page_index": page_index[k],
                "cy": float(h["cy"]),
                "text": h["text"],
                "poly": h["poly"],  # same object as in clean_pages -> identity works
            })
    header_markers.sort(key=lambda m: (m["page_index"], m["cy"]))

    # Prepare header groups with page-span boundaries
    groups: List[Dict[str, Any]] = []
    if header_markers:
        for i, hm in enumerate(header_markers):
            start_pk, start_pi, start_cy = hm["page_key"], hm["page_index"], hm["cy"]
            if i + 1 < len(header_markers):
                end_hm = header_markers[i + 1]
                end_pk, end_pi, end_cy = end_hm["page_key"], end_hm["page_index"], end_hm["cy"]
            else:
                end_pk, end_pi, end_cy = (keys[-1], page_index[keys[-1]], float("inf"))

            # Human-friendly span label like: "knitting club (page015–page020)"
            label = f"{hm['text']} ({start_pk}\u2013{end_pk})"

            groups.append({
                "header": hm["text"],
                "label": label,                  # includes full span
                "start_page": start_pk,
                "start_index": start_pi,
                "start_cy": start_cy,
                "end_page": end_pk,              # boundary page
                "end_index": end_pi,
                "end_cy": end_cy,
                "poly": hm["poly"],
                "names": [],
            })

        # Walk all spans in reading order and route to current header group
        hdr_i = 0
        first = groups[0]
        first_pi, first_cy = first["start_index"], first["start_cy"]

        for pk in keys:
            page = clean_pages.get(pk) or {}
            texts = page.get("rec_texts", []) or []
            polys = page.get("polys", []) or []

            # sort spans by vertical position within the page
            spans = []
            for t, poly in zip(texts, polys):
                ys = [float(p[1]) for p in poly[:4]] if poly else [0.0, 0.0]
                cy = min(ys) + (max(ys) - min(ys)) / 2.0
                spans.append((cy, t, poly))
            spans.sort(key=lambda x: x[0])

            for cy, t, poly in spans:
                pi = page_index[pk]

                # Skip anything before the very first header start
                if _pos_lt(pi, cy, first_pi, first_cy):
                    continue

                # Advance header pointer while the next header starts before/equal current span
                while hdr_i + 1 < len(groups) and _pos_le(groups[hdr_i + 1]["start_index"],
                                                          groups[hdr_i + 1]["start_cy"],
                                                          pi, cy):
                    hdr_i += 1

                # Skip the header line itself
                if poly is groups[hdr_i]["poly"]:
                    continue

                # Extract persons (multi-entity per span) and require ≥2 words
                persons = _extract_persons(t, nlp)
                if not persons:
                    continue
                if name_correction_enabled:
                    persons = [maybe for maybe in (corrector.correct(p) for p in persons)]
                for p in persons:
                    if p and p not in groups[hdr_i]["names"]:
                        groups[hdr_i]["names"].append(p)

    # 5) Write final JSON (headers list with spans & names)
    final_out = {
        "meta": {
            "median_height": med_h,
            "threshold": thresh_h,
            "header_rule": "height > 150% of book-wide median (no keyword filter)",
            "year": year,
            "pdf": os.path.basename(pdf_path),
            "spillover": "names assigned to previous header until the next header begins",
        },
        "headers": groups
    }
    with open(final_headers_path, "w", encoding="utf-8") as f:
        json.dump(final_out, f, indent=2, ensure_ascii=True)
    print(f"[WRITE] FINAL  -> {final_headers_path}")

    t_post1 = time.perf_counter()
    print(f"[HEADERS+NAMES][TIMING] {t_post1 - t_post0:.2f}s\n")

    # --------- Final timing summary ----------
    t1 = time.perf_counter()
    print("========== [DONE] ==========")
    print(f"[TIMING] preproc={t_pre1 - t_pre0:.2f}s  infer={t_ocr1 - t_ocr0:.2f}s  compile={t_cmp1 - t_cmp0:.2f}s  headers+names={t_post1 - t_post0:.2f}s  total={t1 - t0:.2f}s")
    print("=========================================\n")
