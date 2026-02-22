# Continuity Summary — 2026-02-22b

## Solve Rate
- Start of session: 13/29 (44%) on puzzle 31166
- End of session: 16/29 (55%) on puzzle 31166
- New solves: ADRIATIC, ALBERT, +1 other (from data cleanup / numeric fix)

## Work Completed This Session

### a. Data cleanup — synonyms_pairs in cryptic_new.db
Removed 286,002 garbage rows across 4 tiers:
- Tier 1: Empty words (4,543), self-referential (350), duplicates (182,142)
- Tier 2: Enumeration patterns (2,113), fragment words (3,675), trailing comma fix (1,507)
- Tier 3: Long legacy words (10,293), long legacy synonyms (5,709), unicode normalization (10)
- Tier 4: DD truncated fragments (1,344), parenthetical words (39,842), parenthetical synonyms (34,474)
- Table went from 1,628,356 to 1,342,354 rows
- Also deleted "in"→"a" garbage synonym (caused false match in ADRIATIC)

### b. Numeric token fix — compound.py
`lookup_substitution()` and `lookup_indicator()` stripped all non-alpha chars, making "1" become "" and return []. Fix: if alpha stripping produces empty string, try extracting digits instead. Unlocked 24 roman_numeral entries (e.g., "1"→"I").

### c. Cache fix — compound.py
`_substitution_cache` keyed by word only, not by `max_synonym_length`. First caller's max determined what all subsequent callers saw. Fix: cache stores ALL synonyms unfiltered, filtering by max_synonym_length happens on return. Fixed ADRIATIC: ADRIAN (6 chars) was excluded when max=5 was cached first.

### d. ADRIATIC solve — secondary.py
Three fixes to make ADRIATIC fully solved:
1. `attempt_deletion` return now includes `indicator_words` so `_check_fully_solved` consumes the deletion indicator word ("unfinished,")
2. `_check_fully_solved` now consumes indicator_words from handler results
3. `_improve_formula` strips intermediate `= VALUE` to avoid double-equals nonsense in formulas
4. `_build_result` updates breakdown entries for consumed indicator words

### e. ALBERT solve — secondary.py + enrichment
1. Deleted bad `bells→B` (id 28279) and `league→L` (id 28280) from wordplay — these were fake abbreviations inserted by self-learning step_b
2. Fixed step_b in `05_self_learning_enrichment.py`: skip ALL n==1 cases. Single-letter matches should come from reference data or indicator+fodder logic, not manufactured abbreviations
3. Fixed `_find_extraction_target` in secondary.py: now skips over linker words when searching for fodder. "bells at the front" → skips "at", "the" to find "bells" as target for "front" (first_use indicator)
4. `attempt_partial_resolve` now returns `overrides` format so each word gets its specific role (fodder vs indicator) in the breakdown
5. Added operation indicator consumption at partial_resolve call site (same pattern as deletion fix)

### f. Pipeline speed — SELF_LEARN toggle
Added `SELF_LEARN = False` to report.py selection criteria section. Controls whether self-learning enrichment + pipeline re-run executes. User set to False for faster testing. Also available as `--no-self-learn` CLI arg.

### g. Removed ADRIATIC trace code from secondary.py

---

## Design Discussion — V3 Direction

User identified fundamental insight for next version:

### Key principles:
1. **Clue structure reveals mechanism** — Setters follow unconscious patterns. The surface structure of a clue predicts the wordplay type without looking up individual words. 500k clue corpus = training data for pattern classification.

2. **Don't need total word attribution** — For a mobile hint app, user needs only three things:
   - **Definition** — what am I looking for?
   - **Wordplay type** — anagram, container, deletion, etc.
   - **Fodder** — which words produce the letters?

   Indicators and link words are scaffolding the user can infer.

3. **Use techniques suited to AI strengths** — Pattern classification across large datasets, not step-by-step replication of human cryptic reasoning. The current approach requires too much nuanced recall and context.

### Implication:
The Wire Principle (account for every word) becomes unnecessary. The DB becomes training data for a classifier, not a lookup engine. The cascade of stages is replaced by pattern recognition.

---

## Current Code State — All Uncommitted

| File | Changes |
|------|---------|
| stages/compound.py | Numeric token fix + cache fix |
| stages/secondary.py | Deletion indicator consumption, formula double-equals fix, _find_extraction_target linker skip, partial_resolve overrides, operation indicator consumption, breakdown multi-word handling, trace removal |
| stages/unified_explanation.py | 'wordplay connector' → 'unresolved' (from prior session) |
| enrichment/05_self_learning_enrichment.py | Step B skip all n==1 cases |
| enrichment/api_gap_finder.py | Homophone support, stage_secondary scope (from prior session) |
| enrichment/common.py | insert_homophone (from prior session) |
| report.py | SELF_LEARN config toggle |
| data/cryptic_new.db | 286,002 rows cleaned + 3 bad entries deleted |

---

## DB Deletions This Session
- synonyms_pairs: 286,002 garbage rows (4 tiers)
- synonyms_pairs: "in"→"a" (1 row)
- wordplay: bells→B id=28279, league→L id=28280
