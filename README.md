# Yearbook Sports Participation Pipeline

A Python OCR and natural-language-processing pipeline for extracting student names and sports sections from digitized school yearbooks. The goal is to turn scanned pages into structured records that can support research on student sports participation.

**See an example:** [Extracted sections and names](sample_output/sections_with_names.json) · [Source yearbook PDF](sample_output/turkeytail197000glen.pdf)

## How it works

1. **Extract page text and layout.** PyMuPDF renders PDF pages, and PaddleOCR processes them into page-level OCR records and a combined `compiled_ocr.json`.
2. **Build sections.** Header and layout heuristics group text between section boundaries, preserving the associated page numbers.
3. **Extract names and identify sports.** spaCy's transformer-based named-entity recognizer finds candidate person names; normalization and deduplication clean the results. Sports-related headers identify the athletics subset.
4. **Optionally correct OCR errors in names.** RapidFuzz-based matching uses decade-specific name corpora when supplied, recording corrections alongside the output and keeping an uncorrected backup.

The stages run in separate Python processes, and saved OCR artifacts allow later stages to be rerun without repeating the full extraction.

## Outputs

Each input PDF gets a folder beneath `output/<pdf-stem>/`:

| File | Contents |
| --- | --- |
| `compiled_ocr.json` | Page text, layout blocks, metadata, and a header index |
| `sections_with_text.json` | Sections with their source text and page references |
| `sections_with_names.json` | Candidate names grouped by section |
| `sports_sections.json` | The sports-related subset, including corrections when available |
| `sports_sections.uncorrected.json` | Backup created by the name-correction stage |

The checked-in sample is raw extraction output. It illustrates the data structure and contains OCR/NER errors; it is not a manually verified roster or an accuracy benchmark.

## Setup and use

Install PaddlePaddle for the target hardware before installing the remaining packages. For GPU use, follow the [PaddlePaddle installation guide](https://www.paddlepaddle.org.cn/en/install/quick) to select the appropriate CUDA build.

```bash
python -m pip install -r requirements
python -m spacy download en_core_web_trf
```

Then:

1. Place PDFs in `pdf_input/`, or set `PDF` / `PDF_DIR` in [main.py](main.py).
2. Enable `RUN_TEXT_EXTRACTION` for a fresh PDF; it is disabled in the checked-in configuration for reruns using existing OCR.
3. Review the GPU, extraction, and name-correction settings. Set `NC_YEAR` to the yearbook's year when using correction.
4. Run from the repository root:

```bash
python main.py
```

Optional correction expects `data/first_names_by_decade.json` and `data/last_names_by_decade.json`. If the corpora are missing or correction is disabled, the correction module falls back to leaving names uncorrected.

## Code guide

- [text_extraction.py](src/text_extraction.py): page rendering, OCR, caching, and combined output.
- [name_extraction.py](src/name_extraction.py): section boundaries and sports detection.
- [name_ner.py](src/name_ner.py): person-name extraction and deduplication.
- [name_correction.py](src/name_correction.py): optional corpus-based corrections and audit records.

## Status

Research prototype. Scan quality, page layout, and uncommon names affect extraction quality. Review the linked source pages before treating a detected name as evidence of sports participation.
