#!/usr/bin/env python3
"""
Yearbook OCR extraction with preprocessing, senior class detection, multi-page NER extraction,
transformation, and CSV export.
using PaddleOCR, spaCy (transformer-based), and RapidFuzz.

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
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from typing import List, Dict, Optional, Tuple


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

    sorted_first = sorted(fn_by_decade[key_first].items(), key=lambda x: -x[1])
    sorted_last  = sorted(ln_by_decade[key_last].items(), key=lambda x: x[1])

    first_names = [name for name, _ in sorted_first]
    last_names  = [name for name, _ in sorted_last]
    first_map   = fn_by_decade[key_first]
    last_map    = ln_by_decade[key_last]

    print(f"[NAMES] Loaded {len(first_names)} first names for decade {key_first}")
    print(f"[NAMES] Loaded {len(last_names)} last names for decade {key_last}")

    return first_names, first_map, last_names, last_map


def substitution_similarity(a: str, b: str, *args, **kwargs) -> float:
    dist = Levenshtein.distance(a, b)
    diff = abs(len(a) - len(b))
    weighted_dist = dist + diff
    min_len = min(len(a), len(b)) or 1
    sim = (1 - weighted_dist / min_len) * 100
    return max(sim, 0.0)


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
    if ',' in base:
        base, suffix = base.split(',', 1)
        suffix = suffix.strip()
    else:
        parts = base.split()
        if parts and parts[-1].rstrip('.').lower() in {'jr','sr','ii','iii','iv','v'}:
            suffix = parts[-1]
            base = " ".join(parts[:-1])
    print(f"[CORRECT] Base: '{base}', Suffix: '{suffix}'")

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
    os.makedirs(output_dir, exist_ok=True)
    pages_dir = os.path.join(output_dir, 'pages')
    pre_dir = os.path.join(output_dir, 'preproc')
    json_dir = os.path.join(output_dir, 'ocr_json')
    overlay_dir = os.path.join(output_dir, 'ocr_images')
    for d in (pages_dir, pre_dir, json_dir, overlay_dir): os.makedirs(d, exist_ok=True)

    # Initialize models
    try:
        nlp = spacy.load('en_core_web_trf')
    except OSError:
        from spacy.cli import download; download('en_core_web_trf'); nlp = spacy.load('en_core_web_trf')
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=True, lang='en')

    doc = fitz.open(pdf_path)
    total = len(doc)
    last_pg = total if end_page is None or end_page > total else end_page
    keys = [f"page{i:03d}" for i in range(start_page, last_pg+1)]

    # Step A & B: OCR and compile raw
    compiled_raw: Dict[str, List[Dict]] = {}
    for key in keys:
        print(f"[PAGE] {key}")
        compiled_raw[key] = []
        page_img = os.path.join(pages_dir, f"{key}.png")
        existing_jsons = sorted(fn for fn in os.listdir(json_dir) if fn.startswith(f"{key}_") and fn.endswith("_raw.json"))
        if os.path.exists(page_img) and existing_jsons:
            print(f"[SKIP] {key} already OCR'd ({len(existing_jsons)} blocks)")
            for fn in existing_jsons:
                raw = json.load(open(os.path.join(json_dir, fn), encoding='utf-8'))
                texts = [t.title() for t in raw.get('rec_texts', [])]
                boxes = raw.get('rec_boxes', [])
                scs = raw.get('rec_scores', [])
                compiled_raw[key].append({'texts': texts, 'boxes': boxes, 'scores': scs})
            continue
        pg_idx = int(key[-3:]) - 1
        imgp = os.path.join(pages_dir, f"{key}.png")
        if not os.path.exists(imgp): doc.load_page(pg_idx).get_pixmap(dpi=dpi).save(imgp)
        gray = cv2.cvtColor(cv2.imread(imgp), cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + (cv2.THRESH_OTSU if use_otsu else 0))
        prp = os.path.join(pre_dir, f"{key}.png"); cv2.imwrite(prp, bw)
        for i, res in enumerate(ocr.predict(prp), start=1):
            print(f"[OCR] {key}_{i:02d}")
            raw_path = os.path.join(json_dir, f"{key}_{i:02d}_raw.json")
            res.save_to_json(raw_path)
            res.save_to_img(overlay_dir)
            raw = res.json.get('res', res.json)
            texts = [t.title() for t in raw.get('rec_texts', [])]
            boxes = raw.get('rec_boxes', [])
            scs = raw.get('rec_scores', [])
            compiled_raw[key].append({'texts': texts, 'boxes': boxes, 'scores': scs})
    print(f"[DONE] Raw OCR compiled for {len(keys)} pages")

    # Step C: detect senior header
    senior_page = None
    max_h = -1
    for key, blocks in compiled_raw.items():
        for blk in blocks:
            for t, box in zip(blk['texts'], blk['boxes']):
                if 'senior' in t.lower():
                    if isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
                        h = box[3] - box[1]
                    else:
                        print(f"[WARN] Unexpected box format for header: {box}")
                        continue
                    if h > max_h:
                        max_h = h
                        senior_page = key
    if senior_page:
        start_idx = keys.index(senior_page) + 1
        print(f"[HEADER] Senior header on {senior_page}, extracting from {keys[start_idx]}")
    else:
        start_idx = 0
        print("[HEADER] No senior header found, extracting from first page")

    # Step D: multi-page NER (collect all names)
    print("[EXTRACT] Performing NER from senior start")
    seen = []
    for key in keys[start_idx:]:
        print(f"[NER PAGE] {key}")
        for blk in compiled_raw.get(key, []):
            for t, box in zip(blk['texts'], blk['boxes']):
                print(f"[TEXT] {t}")
                ents = nlp(t).ents
                if len(ents) == 1 and ents[0].label_ == 'PERSON' and ents[0].text == t and len(t.split()) >= 2:
                    print(f"[FOUND] {t}")
                    seen.append(t)
    print(f"[RESULT] Names extracted: {seen}")

    # Step E: corrections + outputs
    print("[OUTPUT] Saving results")
    fnames, fmap, lnames, lmap = load_sorted_name_lists(first_name_corpus_path, last_name_corpus_path, year)
    out_clean = []
    for name in seen:
        entry = {'text': name}
        if name_correction:
            entry['corrected'] = correct_name(name, fnames, fmap, lnames, lmap, nlp)
        out_clean.append(entry)

    raw_out = os.path.join(output_dir, 'compiled_raw_ocr.json')
    clean_out = os.path.join(output_dir, 'compiled_names_ocr.json')
    csv_out = os.path.join(output_dir, 'names_list.csv')
    with open(raw_out, 'w', encoding='utf-8') as f:
        json.dump(compiled_raw, f, indent=2)
    with open(clean_out, 'w', encoding='utf-8') as f:
        json.dump({'names': out_clean}, f, indent=2)
    with open(csv_out, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(['Name', 'Corrected'])
        for e in out_clean:
            writer.writerow([e['text'], e.get('corrected', '')])

    print(f"[DONE] raw: {raw_out}, clean: {clean_out}, csv: {csv_out}")
