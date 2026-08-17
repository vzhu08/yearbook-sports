# Follow-up items (OCR / name extraction pipeline)

Found while validating the NER candidate-filtering fixes (commit `6bb9ccf`) against
two real yearbooks: "The Acorn" 1945 (pages 74-81) and 1969 (pages 81-103).

## 1. OCR DPI: 200 vs 400

`src/text_extraction.py` renders pages at `dpi=200` by default. On the 1969 Acorn,
this caused PP-StructureV3 to completely miss the "Golf Team" section title on one
page (no header block produced at all, not just mislabeled), so that page's roster
got silently merged into the preceding "Tennis Team" section. It also contributed to
an OCR misread ("James Bouler" -> "James Boule1", digit instead of "r"), which then
failed NER's name-shape check.

Bumping to `dpi=400` and re-OCR'ing that page in isolation fixed both: the title was
correctly detected as its own header, and "Bouler" was read correctly. Single-page
timing was not meaningfully slower at 400 DPI than at 300 DPI (~109s vs ~111s), so
there's no reason to stop at an intermediate value once bumping.

**Not yet committed** because a full 23-page run at 400 DPI was not reproducibly
clean — see items 2 and 3 below, which surfaced while validating this change end to
end. Revisit once those are understood; the DPI bump itself is validated as correct
in isolation.

## 2. Long single-process OCR runs become unstable

Running OCR + name extraction for all 23 pages in one continuous Python process
(at 400 DPI) produced two symptoms after ~40 minutes of runtime:
- A chunk of roster text on one page (FOOTBALL) was silently dropped mid-block
  (not misread -- entirely missing from the OCR output), even though the same
  page OCR'd perfectly when run alone.
- Immediately after, `spacy.load("en_core_web_trf")` failed for every candidate
  model in `name_ner.py`'s `_SPACY_MODEL_CANDIDATES` list, even though the same
  model loads fine in a fresh process moments later.

Both point to some kind of resource exhaustion/instability accumulating over a
long-running single process (memory pressure, thread/handle leak, or similar),
independent of DPI, though the larger 400-DPI image buffers may make it surface
sooner. Chunking the run into smaller per-process batches (e.g. ~6 pages per
subprocess, then merging the resulting `compiled_ocr.json` files) avoided this,
suggesting a workaround: process large books in bounded-size subprocess batches
rather than one continuous process.

## 3. PaddleOCR/PP-StructureV3 batch-composition-dependent output variance

Even after chunking to avoid the long-run instability above, the same physical
page produced *different* OCR output depending on which other pages were batched
with it: the "Golf Team" roster caption came back complete when that page was
OCR'd alone, but came back as an empty string when OCR'd as part of a 6-page batch
(a different, valid batch grouping than the one used in item 2's failed run).

This means OCR output for a given page is not fully reproducible/deterministic
across runs unless it's run in isolation (`batch_size=1` equivalent) -- a
reliability characteristic of the underlying PaddleOCR/PP-StructureV3 engine, not
something fixable in this repo's Python orchestration code. Worth investigating
separately (e.g. whether `batch_size=1` fixes it at an acceptable speed cost, or
whether the layout model's confidence thresholds are the sensitive part).

## 4. Girls' Basketball section has one-word names that belong to Wrestling

In the 1969 Acorn test, `Girls'Basketball`'s extracted names include several
bare-surname entries (e.g. "Patterson", "Pistone", "Randall", "Savage", "Wilson")
that actually come from a "CENTER TO RIGHT: ..." caption belonging to a different
(Wrestling) photo, mis-attributed to the Girls' Basketball section. These surnames
duplicate people whose full names likely already appear correctly elsewhere (e.g.
"Sam Pistone", "Stan Patterson" under Wrestling).

Decided to leave this as-is for now. Possible future fix: once a section already
contains a name whose last token matches a lone one-word candidate, drop the
one-word candidate as a probable duplicate/partial of an already-captured full
name, rather than keeping it as a separate entry.

## 5. Ayantee 1975 / Belles Memoires 1972 validation (Aug 16, 2026)

Fixes made in `src/name_extraction.py` (+ small `name_ner.py`/`text_extraction.py`
changes) while validating these two books:
- orphan OCR lines (recognized in `overall_ocr_res` but wrapped in no layout
  block) are now synthetic captions — this recovered most "missed" names;
- image-block / orphan sports titles ("WRESTLING", "TRACK", "ATHLETICS") become
  section markers; a title-only "SPORTS"/"ATHLETICS" divider page opens a sports
  zone that lasts until a title-only non-sports divider ("CLASSES", ...);
- roster tables with a NAME/NAMES heading yield plain "First Last" cells;
- dot-leader award lists; trailing position/class words stripped; opponents after
  "against"/"vs" dropped; partial names resolved against full names in the same
  or neighbouring section (nicknames, glued caps, initial+surname).

Still open:
- Ayantee PDF p. 95 "SPORTS" divider: PaddleOCR reads nothing (tried crop /
  inverted / 2x); so no sports zone starts there and pp. 96-99 rely on the
  "OUR CHEERLEADERS" image line. Cheerleader names on p. 98-99 are first names
  only (BETTYE, BERYL, CAROLYN lost to vertical/interleaved lettering).
- Ayantee p. 122: "Ron Johnson" caption is printed rotated ("uosuyor uoy").
- Belles: opponents named in action captions ("Claude Warren", "Reid Pollard",
  "Scott") cannot be told from players; "Charlie Strea King" = "Charlie
  Streaking"; "Rogers" is the coach's signature under a paragraph.
- Regression: acorn1945/acorn1969/archway1970/ashhilife1945/babbles1958 outputs
  under /private/tmp/*_test rerun clean (acorn1969 wrestlers now resolved to full
  names, ashhilife gains its ATHLETICS section; archway/babbles unchanged).

## 6. Plain-OCR second pass (Aug 17, 2026)

`text_extraction.py` now runs plain `PaddleOCR` (det+rec, same PP-OCRv5 models)
over every page after PP-StructureV3 and merges the lines (`_merge_plain_ocr`):
PP-StructureV3's internal OCR misread the italic roster type constantly
("Bastick/Thamas/McKinnev/Dexrer" vs "Bostick/Thomas/McKinney/Dexter"), dropped
whole wrestling-table rows ("Ike Smith", "George Harris", "Robert Brown"), and
could not read a rotated caption ("Ron Johnson"); the plain pass fixes all of
that and runs in ~8-20 s/page. Layout still comes from PP-StructureV3;
block_content is rebuilt from the merged lines (join mode from the original
content when known, `structure_block_content`, else heuristic). Structure lines
survive where the plain pass fragmented a display title ("BASEBALL" -> "ASE") or
mangled a line ("aic  oke" for "Janice Hooker"). Resumable per page via the
`plain_ocr_merged` flag; the ayantee/belles ocr_json were re-merged in place
(their original block contents are lost, so `structure_block_content` is null).

Also new in name_extraction: "FOOTBALL Hornsby Howell BASKETBALL Warren
Reynolds ..." staff blocks split into per-sport markers (these are COACHES —
kept only because the user asked for the names on that page); a<->o vote
correction against the book's own names; token-frequency vote to merge
"Erc Harris"/"Eric Harris"; "James Jeff ries" fragment join; awards lines
without dot leaders; belles OCR extended to PDF pp. 91-95 (athletics runs to
printed p. 90; "Favorites" divider closes the sports zone).

Still open: cheerleaders' full names (first names only in the book's sports
pages — needs the class-portrait index, i.e. OCR of the portrait pages +
src/name_resolution.py); "Jerame Simmons", "Ronald Margan", "Charles Raberts",
"Louis Alstan", "Rachan Fuller", "Clinton Bastick" (roster p. 101) have no a/o
twin elsewhere in the book so stay as read; opponents in belles captions.

## 7. Staff exclusion, portrait-index mapping, fast index OCR (Aug 17, 2026)

- Coaches/managers/sponsors are removed from every sports section
  (`_collect_staff_candidates` / `_drop_staff_names`): windows around role
  words anywhere in the book ("Hornsby Howell Head Coach", "Coach: Mel Groomes",
  "Coaches are X and Y") plus sport-labelled directories ("FOOTBALL Hornsby
  Howell BASKETBALL ...") are NER'd and excluded; single-token names equal to a
  staff surname/first name go too ("Rogers"). "Leading scorer X" is a player.
- `extract_text(page_range=(a,b), layout=False)`: plain-OCR-only pages (~10 s
  each) for the class-portrait section; `src/name_resolution.resolve_names(pdf,
  None, out_dir, portrait_pages=[...])` builds the index from the same book and
  resolves surnames / initial+surname / first names (with nicknames) ->
  `sports_sections.resolved.json`. Ayantee portraits: PDF 150-217; Belles:
  10-48. In a whole-book run pass no portrait_pages (index = all non-sports pages).
- Ayantee cheerleaders after mapping: Bettye Tyler, Beryl Jones, Robin Smith
  resolved; Carolyn/Cassandra/Debbie/Cheryl/Mary ambiguous (candidates kept),
  Reba/Charlene no match.

## 8. Notes (Aug 17, 2026, later)

- a<->o with no evidence in the book: `_vote_correct_ao` now falls back to a
  built-in common-name list (`_COMMON_NAME_TOKENS`) — a token that is not a
  known name but whose single a/o variant is, gets flipped ("Margan"->"Morgan",
  "Raberts"->"Roberts", "Jerame"->"Jerome", "Alstan"->"Alston",
  "Bastick"->"Bostick"). Book evidence always wins over the list.
- Ayantee name index now built from the WHOLE book (all non-sports pages were
  OCR'd plain-only). Cheerleaders: Beryl Jones, Robin Smith resolved; Bettye
  (Melamb / L. Tyler / M. Young), Carolyn (4), Cassandra (Spells / Wright),
  Debbie (4 Deborahs), Cheryl (2), Mary (4) are genuinely ambiguous — the book
  has several students of each first name; Reba and Charlene have no portrait
  match. Candidates are kept in sports_sections.resolved.json.
- Belles: "Claude Warren", "Reid Pollard", "Scott" in action captions are
  most likely opponents (kept as printed; user OK with that); "Donnie" is
  ambiguous (Donnie Kincaid or one of three Donalds) — left as "Donnie".
- Output shape (Aug 17): sports_sections.resolved.json "sections" = plain
  name strings per section + a last "NEEDS MAPPING" section with candidates
  for ambiguous partials; "details" keeps per-name status records.
