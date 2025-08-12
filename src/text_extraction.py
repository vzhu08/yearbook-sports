#!/usr/bin/env python3
"""
Yearbook OCR extraction with preprocessing, senior class detection (optional),
sports (varsity) roster parsing, transformation, and CSV export.

- Preserves your entrypoint/signature and OCR pipeline.
- Adds step-level caching to avoid reprocessing if outputs already exist.
- Skips senior extraction entirely if a senior header isn't found.
- VERY LENIENT sports subheader detection:
    • any line that is among the BIGGEST text on the page (within 80% of max height)
    • AND contains/fuzzy-matches "varsity"

Defines:
    load_sorted_name_lists(...)
    substitution_similarity(...)
    correct_name(...)
    extract_yearbook_ocr(...)
"""

import os
import re
import json
import csv
import cv2
import fitz  # PyMuPDF
import spacy
from paddleocr import PaddleOCR
from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein
from typing import List, Dict, Optional, Tuple, Any


# -----------------------------------------------------------------------------
# Name corpora utilities
# -----------------------------------------------------------------------------
def load_sorted_name_lists(
    first_name_corpus_path: str,
    last_name_corpus_path: str,
    year: int
) -> Tuple[List[str], Dict[str, int], List[str], Dict[str, int]]:
    print("[LOAD NAMES] Loading name lists...")
    with open(first_name_corpus_path, encoding='utf-8') as f:
        fn_by_decade = json.load(f)
    with open(last_name_corpus_path, encoding='utf-8') as f:
        ln_by_decade = json.load(f)

    def pick_decade_key(dct: dict, year: int) -> str:
        decades = sorted(int(k) for k in dct)
        target = (year // 10) * 10
        if str(target) in dct:
            return str(target)
        elif target < decades[0]:
            return str(decades[0])
        else:
            return str(decades[-1])

    key_first = pick_decade_key(fn_by_decade, year)
    key_last  = pick_decade_key(ln_by_decade, year)

    # first names: higher frequency first; last names: lower rank number first
    sorted_first = sorted(fn_by_decade[key_first].items(), key=lambda x: -x[1])
    sorted_last  = sorted(ln_by_decade[key_last].items(), key=lambda x: x[1])

    first_names = [name for name, _ in sorted_first]
    last_names  = [name for name, _ in sorted_last]
    first_map   = fn_by_decade[key_first]
    last_map    = ln_by_decade[key_last]

    print(f"[NAMES] Loaded {len(first_names)} first names for decade {key_first}")
    print(f"[NAMES] Loaded {len(last_names)} last names for decade {key_last}")

    return first_names, first_map, last_names, last_map


# -----------------------------------------------------------------------------
# Fuzzy string scorer
# -----------------------------------------------------------------------------
def substitution_similarity(a: str, b: str, *args, **kwargs) -> float:
    dist = Levenshtein.distance(a, b)
    diff = abs(len(a) - len(b))
    weighted_dist = dist + diff
    min_len = min(len(a), len(b)) or 1
    sim = (1 - weighted_dist / min_len) * 100
    return max(sim, 0.0)


# -----------------------------------------------------------------------------
# Name correction
# -----------------------------------------------------------------------------
def correct_name(
    text: str,
    first_names: List[str],
    first_map: Dict[str, int],
    last_names: List[str],
    last_map: Dict[str, int],
    nlp
) -> str:
    print(f"[CORRECT] Raw text: '{text}'")

    suffix = ""
    base = text.strip().title()

    # Handle suffixes
    if ',' in base:
        base, suffix = base.split(',', 1)
        suffix = suffix.strip()
    else:
        parts = base.split()
        if parts and parts[-1].rstrip('.').lower() in {'jr','sr','ii','iii','iv','v'}:
            suffix = parts[-1]
            base = " ".join(parts[:-1])

    print(f"[CORRECT] Base: '{base}', Suffix: '{suffix}'")

    # Prefer PERSON span if spaCy isolates it
    doc_sp = nlp(base)
    ents = [ent.text for ent in doc_sp.ents if ent.label_ == 'PERSON']
    name_for_split = ents[0] if ents else base
    print(f"[CORRECT] NER split: '{name_for_split}'")

    tokens = name_for_split.split()
    if not tokens:
        return name_for_split + (f", {suffix}" if suffix else "")

    given_tokens = tokens[:-1]
    raw_last = tokens[-1]

    corrected_firsts: List[str] = []
    for tok in given_tokens:
        clean_tok = re.sub(r'[^a-z]', '', tok.lower())
        if clean_tok:
            candidates = process.extract(clean_tok, first_names, scorer=substitution_similarity, limit=5)
            best = max(candidates, key=lambda x: (x[1], first_map.get(x[0], 0)))
            corrected_firsts.append(best[0].title())
            print(f"[CORRECT] First token '{tok}' -> '{best[0]}'")

    clean_last = re.sub(r'[^a-z]', '', raw_last.lower())
    if clean_last:
        candidates = process.extract(clean_last, last_names, scorer=substitution_similarity, limit=5)
        best = max(candidates, key=lambda x: (x[1], -last_map.get(x[0], 0)))
        corrected_last = best[0].title()
        print(f"[CORRECT] Last token '{raw_last}' -> '{corrected_last}'")
    else:
        corrected_last = raw_last.title()

    candidate = " ".join(corrected_firsts + [corrected_last])
    if suffix:
        candidate = f"{candidate}, {suffix.title()}"

    print(f"[CORRECT] Final: '{candidate}'")
    return candidate.strip()


# -----------------------------------------------------------------------------
# Helpers for layout / headers / sports parsing
# -----------------------------------------------------------------------------
def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _box_height(box: Any) -> float:
    # Expecting [x0,y0,x1,y1]
    try:
        return float(box[3] - box[1])
    except Exception:
        return 0.0


def _box_center_y(box: Any) -> float:
    try:
        return float(box[1] + (box[3] - box[1]) / 2.0)
    except Exception:
        return 0.0


def _percentile(values: List[float], pct: float, default: float) -> float:
    if not values:
        return default
    import numpy as _np
    return float(_np.percentile(_np.array(values, dtype=float), pct))


def _find_senior_start(keys: List[str], compiled_raw: Dict[str, List[Dict]]) -> Optional[int]:
    """
    Find the page index to start senior extraction:
    - Locate the largest-height text containing 'senior' across the range
    - Start from the page AFTER that header
    - Return None if not found (caller should SKIP seniors)
    """
    senior_page = None
    max_h = -1.0

    for key, blocks in compiled_raw.items():
        for blk in blocks:
            for t, box in zip(blk.get('texts', []), blk.get('boxes', [])):
                if 'senior' in (t or '').lower():
                    h = _box_height(box)
                    if h > max_h:
                        max_h = h
                        senior_page = key

    if senior_page:
        print(f"[HEADER] Senior header on {senior_page}")
        return keys.index(senior_page) + 1
    else:
        print("[HEADER] No senior header found; SKIPPING senior extraction")
        return None


_NAME_SPLIT_RE = re.compile(r"[,\.;]+|\u2022|•|·|\s{2,}")


def _extract_person_exact_line(t: str, nlp) -> Optional[str]:
    """
    Match your exact senior rule:
        - exactly one PERSON entity
        - entity text equals the entire line
        - at least two words
    """
    if not t:
        return None
    ents = nlp(t).ents
    if len(ents) == 1 and ents[0].label_ == 'PERSON':
        name = ents[0].text.strip()
        if name == t.strip() and len(name.split()) >= 2:
            return name
    return None


def _extract_person_from_segment(seg: str, nlp) -> Optional[str]:
    """
    For sports list items split by punctuation:
        - accept if there's exactly one PERSON span with >= 2 tokens
        - otherwise, reject
    """
    seg = _normalize_text(seg)
    if not seg:
        return None
    ents = [e for e in nlp(seg).ents if e.label_ == "PERSON"]
    if len(ents) == 1:
        name = ents[0].text.strip()
        if len(name.split()) >= 2:
            return name
    return None


def _gather_page_spans(compiled_raw: Dict[str, List[Dict]], key: str) -> List[Dict[str, Any]]:
    """
    Flatten compiled_raw[key] blocks into linear spans with text + box + height + cy,
    sorted by reading order (y, then x).  Skips empty and 1–2 character texts.
    """
    spans: List[Dict[str, Any]] = []
    for blk in compiled_raw.get(key, []):
        texts = blk.get('texts', []) or []
        boxes = blk.get('boxes', []) or []
        for t, b in zip(texts, boxes):
            tt = _normalize_text(t)
            # DROP empties and very short strings (<= 2 chars) before any header/name work
            if not tt or len(tt.replace(" ", "")) <= 2:
                continue
            spans.append({
                "text": tt,
                "box": b,
                "h": _box_height(b),
                "cy": _box_center_y(b),
                "y": float(b[1]) if isinstance(b, (list, tuple)) and len(b) >= 2 else 0.0,
                "x": float(b[0]) if isinstance(b, (list, tuple)) and len(b) >= 1 else 0.0
            })
    spans.sort(key=lambda s: (s["y"], s["x"]))
    return spans


# -------------------- LENIENT VARSITY HEADER DETECTION -----------------------
def _has_varsity(text: str, fuzz_threshold: int) -> bool:
    """
    True if line clearly indicates a 'varsity' subheader.
    - exact 'varsity' substring (case-insensitive)
    - OR token-level fuzzy >= fuzz_threshold (handles OCR slips like 'yarsity', 'varsitv')
    """
    t = (text or "").lower()
    if "varsity" in t:
        return True
    toks = re.split(r"[^a-z]+", t)
    for tok in toks:
        if not tok:
            continue
        if fuzz.ratio(tok, "varsity") >= fuzz_threshold or fuzz.partial_ratio(tok, "varsity") >= max(90, fuzz_threshold):
            return True
    return False


def _find_varsity_headers(spans: List[Dict[str, Any]], fuzz_threshold: int) -> List[Dict[str, Any]]:
    """
    Pick headers as *the tallest* text box(es) on the page that contain/fuzzy-match 'varsity'.
    We allow a small tolerance so ties at the same font size (multi-column layouts) are included.
    """
    heights = [s["h"] for s in spans if s["h"] > 0]
    if not heights:
        return []

    max_h = max(heights)

    # tolerate near-ties: ±2 px or ±2% of max height, whichever is larger
    tol_px = 2.0
    tol_pct = 0.02 * max_h
    cutoff = max_h - max(tol_px, tol_pct)

    print(f"[SPORTS][DEBUG] page max height = {max_h:.2f}, tie cutoff >= {cutoff:.2f}")

    # Candidates: tallest lines that clearly say "varsity"
    cand = [s for s in spans if s["h"] >= cutoff and _has_varsity(s["text"], fuzz_threshold)]

    # Debug: list all candidates
    for s in cand:
        print(f"[SPORTS][DEBUG] header candidate: h={s['h']:.2f} | '{s['text']}' (y={s['y']:.1f})")

    # Dedup near-duplicates on the same line
    cand.sort(key=lambda s: (s["y"], s["x"]))
    deduped: List[Dict[str, Any]] = []
    for s in cand:
        if not deduped:
            deduped.append(s)
            continue
        prev = deduped[-1]
        same_line = abs(s["y"] - prev["y"]) < 5
        same_txt = _normalize_text(s["text"]) == _normalize_text(prev["text"])
        if same_line and same_txt:
            continue
        deduped.append(s)

    return deduped


def _extract_sports_on_page(key: str, compiled_raw: Dict[str, List[Dict]], nlp, fuzz_threshold: int) -> List[Dict[str, Any]]:
    """
    Find tallest 'Varsity' headers on the page and take *all text below each header*
    until the next tallest 'Varsity' header (or end of page). Then split lines into
    segments and keep PERSON names.
    """
    spans = _gather_page_spans(compiled_raw, key)
    if not spans:
        return []

    varsity_spans = _find_varsity_headers(spans, fuzz_threshold=fuzz_threshold)
    if not varsity_spans:
        return []

    # Process headers in reading order
    varsity_spans.sort(key=lambda s: (s["y"], s["x"]))
    out_sections: List[Dict[str, Any]] = []

    for i, sub in enumerate(varsity_spans):
        y_start = sub["cy"]
        y_end = float("inf")

        # Bound only by the NEXT varsity header (per your instruction)
        if i + 1 < len(varsity_spans):
            y_end = varsity_spans[i + 1]["cy"]

        # Gather region strictly below this header
        region = [s for s in spans if s["cy"] > y_start and s["cy"] <= y_end]

        # Split by commas/periods/bullets; detect PERSON names
        region_text = " ".join(s["text"] for s in region)
        segments = [seg.strip() for seg in _NAME_SPLIT_RE.split(region_text) if seg.strip()]

        names: List[str] = []
        for seg in segments:
            nm = _extract_person_from_segment(seg, nlp)
            if nm:
                names.append(nm)

        # Dedup preserve order
        seen = set()
        uniq = []
        for nm in names:
            k = nm.lower()
            if k not in seen:
                uniq.append(nm)
                seen.add(k)

        print(f"[SPORTS] {key}: header='{sub['text']}' -> {len(uniq)} names (y={sub['y']:.1f}, h={sub['h']:.1f})")
        if uniq:
            out_sections.append({
                "subheader": sub["text"],
                "names": uniq
            })

    return out_sections

# -----------------------------------------------------------------------------
# MAIN ENTRYPOINT (signature preserved)
# -----------------------------------------------------------------------------
def extract_yearbook_ocr(
    pdf_path: str,
    output_dir: str,
    year: int,
    dpi: int = 300,
    start_page: int = 1,
    end_page: Optional[int] = None,
    use_otsu: bool = True,
    fuzz_threshold: int = 85,
    first_name_corpus_path: str = 'name_data/first_names_by_decade.json',
    last_name_corpus_path: str = 'name_data/last_names_by_decade.json',
    name_correction: bool = True
) -> None:
    print(f"[START] OCR extract: {pdf_path}")

    # Directories
    os.makedirs(output_dir, exist_ok=True)
    pages_dir   = os.path.join(output_dir, 'pages')
    pre_dir     = os.path.join(output_dir, 'preproc')
    json_dir    = os.path.join(output_dir, 'ocr_json')
    overlay_dir = os.path.join(output_dir, 'ocr_images')
    for d in (pages_dir, pre_dir, json_dir, overlay_dir):
        os.makedirs(d, exist_ok=True)

    # Initialize models (keep YOUR settings)
    try:
        nlp = spacy.load('en_core_web_trf')
    except OSError:
        from spacy.cli import download
        download('en_core_web_trf')
        nlp = spacy.load('en_core_web_trf')

    # Keep your PaddleOCR initializer and the .predict() contract
    ocr = PaddleOCR(use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    lang='en')

    # Page range
    doc = fitz.open(pdf_path)
    total = len(doc)
    last_pg = total if end_page is None or end_page > total else end_page
    keys = [f"page{i:03d}" for i in range(start_page, last_pg + 1)]

    # Output file paths
    compiled_raw_path = os.path.join(output_dir, 'compiled_raw_ocr.json')
    clean_out_path    = os.path.join(output_dir, 'compiled_names_ocr.json')
    csv_out_path      = os.path.join(output_dir, 'names_list.csv')
    senior_simple     = os.path.join(output_dir, 'senior_names.json')
    sports_json_path  = os.path.join(output_dir, 'sports_names.json')
    sports_csv_path   = os.path.join(output_dir, 'sports_names.csv')

    # -------------------------------------------------------------------------
    # Step A/B: Load compiled_raw if already done; else OCR with page-level cache
    # -------------------------------------------------------------------------
    compiled_raw: Dict[str, List[Dict]] = {}
    compiled_loaded = False

    if os.path.exists(compiled_raw_path):
        try:
            with open(compiled_raw_path, encoding='utf-8') as f:
                possible = json.load(f)
            # Only accept if it covers all pages we need
            if all(k in possible for k in keys):
                compiled_raw = possible
                compiled_loaded = True
                print(f"[SKIP] Using existing {compiled_raw_path}")
        except Exception as e:
            print(f"[WARN] Failed to read {compiled_raw_path}: {e}")

    if not compiled_loaded:
        for key in keys:
            print(f"[PAGE] {key}")
            compiled_raw[key] = []

            page_img = os.path.join(pages_dir, f"{key}.png")
            existing_jsons = sorted(
                fn for fn in os.listdir(json_dir)
                if fn.startswith(f"{key}_") and fn.endswith("_raw.json")
            )

            # If both image and raw JSON blocks exist, reuse them
            if os.path.exists(page_img) and existing_jsons:
                print(f"[SKIP] {key} already OCR'd ({len(existing_jsons)} blocks)")
                for fn in existing_jsons:
                    raw = json.load(open(os.path.join(json_dir, fn), encoding='utf-8'))
                    data  = raw.get('res', raw)
                    texts = [(_normalize_text(t)).title() for t in data.get('rec_texts', [])]
                    boxes  = data.get('rec_boxes', [])
                    scores = data.get('rec_scores', [])
                    compiled_raw[key].append({'texts': texts, 'boxes': boxes, 'scores': scores})
                continue

            # Otherwise, ensure render + preprocess exist before OCR
            pg_idx = int(key[-3:]) - 1

            if not os.path.exists(page_img):
                doc.load_page(pg_idx).get_pixmap(dpi=dpi).save(page_img)

            pre_img = os.path.join(pre_dir, f"{key}.png")
            if not os.path.exists(pre_img):
                gray = cv2.cvtColor(cv2.imread(page_img), cv2.COLOR_BGR2GRAY)
                _, bw = cv2.threshold(
                    gray, 0, 255,
                    cv2.THRESH_BINARY + (cv2.THRESH_OTSU if use_otsu else 0)
                )
                cv2.imwrite(pre_img, bw)

            # Finally OCR unless we already had JSONs (guarded above)
            for i, res in enumerate(ocr.predict(pre_img), start=1):
                print(f"[OCR] {key}_{i:02d}")
                raw_path = os.path.join(json_dir, f"{key}_{i:02d}_raw.json")
                if not os.path.exists(raw_path):
                    res.save_to_json(raw_path)
                    res.save_to_img(overlay_dir)

                # Read from just-saved (or existing) JSON to build compiled_raw
                data = json.load(open(raw_path, encoding='utf-8'))
                data = data.get('res', data)
                texts = [(_normalize_text(t)).title() for t in data.get('rec_texts', [])]
                boxes  = data.get('rec_boxes', [])
                scores = data.get('rec_scores', [])
                compiled_raw[key].append({'texts': texts, 'boxes': boxes, 'scores': scores})

        # Save compiled_raw once after OCR
        with open(compiled_raw_path, 'w', encoding='utf-8') as f:
            json.dump(compiled_raw, f, indent=2)
        print(f"[DONE] Raw OCR compiled for {len(keys)} pages -> {compiled_raw_path}")

    # -------------------------------------------------------------------------
    # Step C: Detect senior header page (largest 'senior' box height)
    #         If not found, SKIP senior extraction entirely.
    # -------------------------------------------------------------------------
    start_idx = _find_senior_start(keys, compiled_raw)

    # -------------------------------------------------------------------------
    # Step D: Senior names — multi-page NER from senior start (if present)
    # -------------------------------------------------------------------------
    senior_seen: List[str] = []

    seniors_already = os.path.exists(senior_simple) and os.path.exists(clean_out_path) and os.path.exists(csv_out_path)
    if seniors_already:
        print("[SKIP] Senior outputs already exist; skipping senior extraction")
    else:
        if start_idx is not None:
            print("[EXTRACT] Performing NER from senior start")
            for key in keys[start_idx:]:
                print(f"[NER PAGE] {key}")
                for blk in compiled_raw.get(key, []):
                    for t in blk.get('texts', []):
                        if not t:
                            continue
                        print(f"[TEXT] {t}")
                        nm = _extract_person_exact_line(t, nlp)
                        if nm:
                            print(f"[FOUND] {nm}")
                            senior_seen.append(nm)
            print(f"[RESULT] Senior names extracted: {len(senior_seen)}")
        else:
            print("[INFO] Senior header not found; skipping senior NER")

    # -------------------------------------------------------------------------
    # Step E: Sports/Athletics — page-local 'Varsity' headers (VERY LENIENT)
    # -------------------------------------------------------------------------
    sports_pages: Dict[str, List[Dict[str, Any]]] = {}
    sports_json_path = os.path.join(output_dir, 'sports_names.json')
    sports_csv_path  = os.path.join(output_dir, 'sports_names.csv')

    sports_json_exists = os.path.exists(sports_json_path)
    if sports_json_exists:
        print(f"[SKIP] Sports JSON exists: {sports_json_path}")
        try:
            with open(sports_json_path, encoding='utf-8') as f:
                sports_pages = json.load(f).get('pages', {})
        except Exception as e:
            print(f"[WARN] Failed to read {sports_json_path}: {e}")
            sports_pages = {}

    if not sports_json_exists:
        print("[EXTRACT] Parsing Sports (Varsity) rosters (lenient headers)")
        for key in keys:
            sections = _extract_sports_on_page(key, compiled_raw, nlp, fuzz_threshold=fuzz_threshold)
            if sections:
                sports_pages[key] = sections
        total_sections = sum(len(v) for v in sports_pages.values())
        print(f"[RESULT] Sports pages: {len(sports_pages)} | Varsity sections: {total_sections}")
        with open(sports_json_path, 'w', encoding='utf-8') as f:
            json.dump({'pages': sports_pages}, f, indent=2)
        print(f"[WRITE] {sports_json_path}")

    # -------------------------------------------------------------------------
    # Step F: Corrections + outputs
    # -------------------------------------------------------------------------
    fnames, fmap, lnames, lmap = load_sorted_name_lists(
        first_name_corpus_path, last_name_corpus_path, year
    )

    # Seniors — write only if missing and we found names
    out_clean = []
    if not seniors_already and senior_seen:
        for name in senior_seen:
            entry = {'text': name}
            if name_correction:
                entry['corrected'] = correct_name(name, fnames, fmap, lnames, lmap, nlp)
            out_clean.append(entry)

    compiled_raw_path = os.path.join(output_dir, 'compiled_raw_ocr.json')
    clean_out_path    = os.path.join(output_dir, 'compiled_names_ocr.json')
    csv_out_path      = os.path.join(output_dir, 'names_list.csv')
    senior_simple     = os.path.join(output_dir, 'senior_names.json')

    # raw compiled always present at this point (either reused or new)
    print(f"[DONE] compiled_raw: {compiled_raw_path}")

    if out_clean:
        if not os.path.exists(senior_simple):
            with open(senior_simple, 'w', encoding='utf-8') as f:
                json.dump({'names': [e['text'] for e in out_clean]}, f, indent=2)
            print(f"[WRITE] {senior_simple}")

        if not os.path.exists(clean_out_path):
            with open(clean_out_path, 'w', encoding='utf-8') as f:
                json.dump({'names': out_clean}, f, indent=2)
            print(f"[WRITE] {clean_out_path}")

        if not os.path.exists(csv_out_path):
            with open(csv_out_path, 'w', newline='', encoding='utf-8') as cf:
                writer = csv.writer(cf)
                writer.writerow(['Name', 'Corrected'])
                for e in out_clean:
                    writer.writerow([e['text'], e.get('corrected', '')])
            print(f"[WRITE] {csv_out_path}")
    else:
        print("[INFO] No senior names to write (either skipped or none found)")

    # Sports CSV (generate if missing, using JSON + correction setting)
    if not os.path.exists(sports_csv_path):
        with open(sports_csv_path, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            writer.writerow(['Subheader', 'Name', 'Corrected', 'Page'])
            for page_key, sections in sports_pages.items():
                for sec in sections:
                    sub = sec.get('subheader', '')
                    for name in sec.get('names', []):
                        corrected = correct_name(name, fnames, fmap, lnames, lmap, nlp) if name_correction else ''
                        writer.writerow([sub, name, corrected, page_key])
        print(f"[WRITE] {sports_csv_path}")

    if os.path.exists(senior_simple):
        print(f"[DONE] senior simple json: {senior_simple}")
    if os.path.exists(clean_out_path):
        print(f"[DONE] senior clean json: {clean_out_path}")
    if os.path.exists(csv_out_path):
        print(f"[DONE] senior csv: {csv_out_path}")
    if os.path.exists(sports_json_path):
        print(f"[DONE] sports json: {sports_json_path}")
    if os.path.exists(sports_csv_path):
        print(f"[DONE] sports csv: {sports_csv_path}")
