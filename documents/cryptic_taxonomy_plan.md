# Cryptic Crossword Pattern Taxonomy System — Detailed Build Plan

## Context & Assets

### What we have:
- `clues_master.db` — 500k high quality clues with answers (limited sources: Times, Telegraph, Guardian)
- `cryptic_new.db` — reference tables:
  - ~2 million synonym pairs
  - 778+ substitution patterns
  - 3000+ indicator words (anagram, reversal, container, deletion etc.)
  - Homophones database
  - Crossword-specific definition/answer pairs
- 162k human explanations (Telegraph, Guardian, Times blog style)
- 26k Times explanations with **structured notation** and **type tags** (growing to 80-118k via active scrape)
- Existing pipeline: `explanation_pipeline.py`, `unified_parse_builder.py`
- Definition/wordplay token separation already implemented

### Key architectural principle:
We always know the answer. We are working **backwards** — explaining how the answer arises from the clue, not generating the answer from the clue. Definition detection is largely solved via synonym database lookup.

---

## Phase 1 — Analyse Existing 26k Times Explanations

**Goal:** Understand what we have before the full scrape arrives. Build analysis tooling that can be immediately re-run on 80-118k dataset.

### Task 1.1 — Database Inspection
Connect to the database containing Times explanations and confirm:
- Exact table name and schema for the 26k Times records
- Which fields contain: clue, answer, explanation text, type tag, notation string
- Sample 50 rows to confirm field mapping matches expected format:
  ```
  id | source | puzzle_id | clue | answer | definition | explanation | type_tag
  ```
- Count records by type_tag to get initial frequency distribution
- Identify records with NULL or missing type tags
- Identify records with NULL or missing explanation/notation

### Task 1.2 — Type Tag Frequency Analysis
Produce a frequency table of all type tags present:
- Count and percentage for each type (charade, anagram, double_definition, container etc.)
- Identify any unexpected or inconsistent type tag values (typos, variants)
- Normalise type tag variants into canonical names
- Identify what percentage of 26k have no type tag at all

Expected output: ranked frequency table saved to `pattern_frequency_26k.csv`

### Task 1.3 — Notation Pattern Analysis
The Times explanations use a consistent formal notation:
- `+` separates charade components: `T + ROUT`
- `[]` indicates deletion/removal: `[pr]EVENT`, `[r]IVE[r]`
- `()` indicates container/insertion: `INTER + C(ED)E`
- Alternate letter extraction: `HE([a]D[d]E[r])ATH`
- Anagram indicated by "Anagram of X"

For each explanation with notation:
- Detect which notation symbols are present
- Extract the component tokens
- Verify the token combination produces the known answer
- Flag records where notation does NOT verify against answer (data quality issues)

Expected output: `notation_analysis_26k.csv` with columns:
```
id | type_tag | notation_symbols_present | tokens_extracted | verification_pass | failure_reason
```

### Task 1.4 — Token Coverage Check
For each wordplay token extracted in Task 1.3:
- Check what percentage exist directly in synonym database
- Check what percentage exist in indicator database
- Check what percentage exist in substitution database
- Identify tokens that appear frequently but are NOT in any database (gaps)

Expected output: `token_coverage_gaps.csv` — tokens appearing 3+ times not in any database

---

## Phase 2 — Build Formal Primitive Definitions

**Goal:** Produce a formal specification document for each primitive wordplay operation, derived from the data in Phase 1.

### The Primitives
Based on Times data, define each of the following formally. More may emerge from data analysis:

1. CHARADE — concatenation of components
2. ANAGRAM — letter permutation of fodder
3. DOUBLE_DEFINITION — two separate definitions
4. CONTAINER — one component inserted inside another
5. REVERSAL — string reversed
6. DELETION — removal of letters from start/end/middle
7. HIDDEN — answer concealed in consecutive clue letters
8. HOMOPHONE — sounds like another word
9. ALTERNATION — every other letter extracted
10. CRYPTIC_DEFINITION — whole clue is oblique definition
11. SUBSTITUTION — one component replaced by another
12. ANDLIT — wordplay and definition are the same text

### For Each Primitive, Document:

```
PRIMITIVE: [NAME]
-----------------
STRUCTURE:
  What are the mandatory components?
  What is their required order?
  
INDICATORS:
  What words/phrases signal this type?
  Extract from indicator database + confirm against Times data
  
FODDER RULES:
  What can the input material be?
  Can it span multiple clue words?
  Can it include wordplay itself (nested)?
  
TRANSFORMATION:
  Exact rule for how fodder becomes answer/component
  
NOTATION:
  How is this represented in Times explanation notation?
  
VERIFICATION:
  How do we confirm result matches known answer?
  
NEGATIVE INDICATORS:
  What rules this type out immediately?
  
FREQUENCY:
  Count from 26k dataset
  Percentage of total clues
  
EXAMPLES:
  5 confirmed examples from Times data with full decomposition
  
COMMON COMPOSITIONS:
  Which other primitives does this frequently combine with?
  e.g. CHARADE where one component is itself an ANAGRAM
```

### Task 2.1 — Generate Primitive Specs from Data
For each primitive type:
- Pull all confirmed examples from 26k dataset (by type tag)
- Extract indicator words that appear in those clues
- Cross-reference with existing indicator database
- Document any indicators present in data but missing from database
- Write formal spec using template above
- Save each spec as `primitives/[name].md`

### Task 2.2 — Identify Composite Patterns
Find the most frequent two-primitive and three-primitive combinations:
- Query records where notation contains both `+` and `[]` (charade + deletion)
- Query records where notation contains `()` and anagram indicator (container + anagram)
- Rank all combinations by frequency
- Document top 20 composite patterns formally

These composites become your ~100 pattern taxonomy entries.

Expected output: `composite_patterns_ranked.csv`

---

## Phase 3 — Re-run on Full Scraped Dataset (80-118k)

**Goal:** When scraper completes, immediately re-run all Phase 1 and Phase 2 analysis on full dataset.

### Task 3.1 — Ingest New Data
- Confirm schema consistency between existing 26k and newly scraped records
- Merge into single analysis table (do NOT modify production databases)
- Re-run Tasks 1.1 through 1.4 on full dataset
- Compare frequency distributions — did rare types become better represented?

### Task 3.2 — Enrich Primitive Specs
- Update each primitive spec with full example counts from larger dataset
- Identify any new primitives or composite patterns not seen in 26k
- Update indicator lists with any new indicators found
- Produce final `pattern_taxonomy_final.csv` with full frequency data

### Task 3.3 — Identify Underrepresented Types
Flag any pattern types with fewer than 50 examples — these will need special handling or may need supplementing from Telegraph/Guardian explanations.

---

## Phase 4 — Build Pattern Matching Engine

**Goal:** Given isolated wordplay tokens and known answer, identify which pattern(s) explain the transformation.

### Architecture

```
INPUT:
  - clue (full text)
  - answer (known)
  - definition_tokens (already identified by existing pipeline)
  - wordplay_tokens (remainder after definition removed)

OUTPUT:
  - matched_pattern (from taxonomy)
  - explanation (human-readable)
  - confidence (HIGH/MEDIUM/LOW)
  - verification_status (CONFIRMED/UNCONFIRMED)
```

### Task 4.1 — Build Pattern Detector Registry
Create a registry of pattern detectors, one per primitive:

```python
class PatternDetector:
    name: str
    detect(wordplay_tokens, answer) -> DetectionResult
    # Returns: match confidence, component mapping, explanation string
```

Each detector should:
- Be self-contained and fast-failing
- Return PASS immediately if required components not found
- Return SOLVED only if transformation verifiably produces answer
- Have a hard token budget — no open-ended searching

### Task 4.2 — Implement Primitive Detectors
Implement detectors in order of frequency (most common first):

**Priority order based on expected frequency:**
1. CHARADE detector — try all splits of wordplay tokens, check if concatenation of synonyms = answer
2. ANAGRAM detector — check if any wordplay token subset is anagram of answer, confirm indicator present
3. DOUBLE_DEFINITION detector — check if two clue words are independently synonyms of answer
4. HIDDEN detector — check if answer appears as substring of clue string
5. CONTAINER detector — find two components A, B where A contains B = answer or B contains A = answer
6. REVERSAL detector — check if answer reversed = synonym of any wordplay token, confirm indicator
7. DELETION detector — check if answer + deleted_letters = known word, confirm position indicator
8. HOMOPHONE detector — check if answer sounds like synonym of wordplay token, confirm indicator
9. ALTERNATION detector — extract alternate letters from wordplay token, check = answer
10. SUBSTITUTION detector — check substitution database for applicable patterns

### Task 4.3 — Build Composite Pattern Matcher
For clues not solved by single primitive detectors:
- Try top 20 composite patterns from Task 2.2
- Each composite pattern specifies which detectors to run in which order
- First composite pattern that verifiably produces answer wins

### Task 4.4 — Triage Dispatcher
Build a dispatcher that:
- Runs detectors in frequency order
- Each detector has a hard timeout/attempt limit
- On PASS, immediately moves to next detector — no retrying with variations
- On SOLVED, returns result immediately
- Logs which detectors were tried and why each passed
- Returns UNSOLVED cleanly if all detectors exhaust

```python
def triage(clue, answer, definition_tokens, wordplay_tokens):
    for detector in DETECTOR_REGISTRY:  # ordered by frequency
        result = detector.detect(wordplay_tokens, answer)
        if result.solved:
            return result  # immediate return
        # else: continue to next detector, no lingering
    return UnsolvedResult(tried=all_detectors)
```

---

## Phase 5 — Validation Against Human Explanations

**Goal:** Verify the pattern matching engine reproduces human explanations correctly.

### Task 5.1 — Holdout Test Set
From the 26k Times explanations, reserve 2,000 records as a holdout test set (not used in building taxonomy). Run the pattern matcher against all 2,000 and measure:
- Correct pattern identified: target >80%
- Correct component decomposition: target >75%
- Explanation matches human explanation: target >70%
- False positives (wrong pattern confidently identified): target <5%

### Task 5.2 — Failure Analysis
For records where matcher fails or gives wrong pattern:
- Categorise failure types
- Identify systematic gaps in primitive definitions
- Identify missing indicators or synonyms
- Feed findings back into Phase 2 primitive specs

### Task 5.3 — Cross-Source Validation
Test against 500 Telegraph and 500 Guardian explanations:
- Do Times-derived pattern definitions generalise?
- Are there source-specific conventions not captured?
- Document any source-specific variations

---

## Phase 6 — Integration with Existing Pipeline

**Goal:** Replace current explanation_pipeline.py cascade with new pattern matching engine.

### Task 6.1 — Integration Points
- New engine receives definition_tokens and wordplay_tokens from existing pipeline (already produced)
- Returns structured result compatible with existing output format
- Preserves existing DD and Hidden stages (already working well)
- Replaces problematic anagram/compound/general stages

### Task 6.2 — Confidence Thresholds
Define confidence levels and how pipeline handles each:
- HIGH confidence (pattern verified, answer confirmed): return explanation
- MEDIUM confidence (pattern likely but unverified): return with caveat
- LOW confidence (best guess): flag for human review
- UNSOLVED: log for analysis, do not return false explanation

### Task 6.3 — Regression Testing
Run new integrated pipeline against existing test suite. Ensure solve rate improves over current ~4/50 baseline. Target: >30/50 on same test set.

---

## Database Additions Required

During build, maintain a log of database gaps found:

### New indicator entries needed:
- Any indicator words found in Times data not in existing indicator database
- Add with source attribution and frequency count

### New synonym pairs needed:
- Any wordplay tokens not resolvable via existing synonym database
- Prioritise high-frequency gaps

### Pattern taxonomy table (new):
Create new table in `cryptic_new.db`:
```sql
CREATE TABLE pattern_taxonomy (
    pattern_id INTEGER PRIMARY KEY,
    pattern_name TEXT,           -- e.g. "charade", "anagram+deletion"
    primitive_count INTEGER,     -- 1 for simple, 2+ for composite
    primitives TEXT,             -- JSON array of primitive names
    frequency_count INTEGER,     -- from Times analysis
    frequency_pct REAL,          -- percentage of total clues
    formal_spec TEXT,            -- path to spec document
    example_clue_ids TEXT        -- JSON array of example record ids
);
```

---

## File Structure

```
cryptic_taxonomy/
├── analysis/
│   ├── phase1_analyse_26k.py        -- Task 1.1-1.4
│   ├── phase2_build_primitives.py   -- Task 2.1-2.2
│   ├── phase3_rerun_full.py         -- Task 3.1-3.3
│   └── phase5_validate.py           -- Task 5.1-5.3
├── primitives/
│   ├── charade.md
│   ├── anagram.md
│   ├── double_definition.md
│   ├── container.md
│   ├── reversal.md
│   ├── deletion.md
│   ├── hidden.md
│   ├── homophone.md
│   ├── alternation.md
│   ├── cryptic_definition.md
│   ├── substitution.md
│   └── andlit.md
├── matchers/
│   ├── base_detector.py             -- Abstract base class
│   ├── charade_detector.py          -- Task 4.2
│   ├── anagram_detector.py
│   ├── double_def_detector.py
│   ├── container_detector.py
│   ├── reversal_detector.py
│   ├── deletion_detector.py
│   ├── hidden_detector.py
│   ├── homophone_detector.py
│   ├── alternation_detector.py
│   ├── substitution_detector.py
│   ├── composite_matcher.py         -- Task 4.3
│   └── triage_dispatcher.py         -- Task 4.4
└── outputs/
    ├── pattern_frequency_26k.csv
    ├── notation_analysis_26k.csv
    ├── token_coverage_gaps.csv
    ├── composite_patterns_ranked.csv
    └── pattern_taxonomy_final.csv
```

---

## Immediate Next Steps (while scraper runs)

1. Run Phase 1 analysis on existing 26k — confirm schema, get frequency distribution
2. Inspect 50 raw explanation records to confirm notation parsing assumptions
3. Build notation parser for `+`, `[]`, `()` symbols
4. Write formal spec for CHARADE (most frequent type) as template for others
5. Be ready to re-run immediately when scrape completes

---

## Success Criteria

- Pattern taxonomy covers >95% of Times clues
- Pattern taxonomy generalises to >85% of Telegraph/Guardian clues  
- Triage dispatcher solves >60% of test clues with correct explanation
- Each detector fast-fails cleanly with no lingering
- Zero false positives at HIGH confidence level
- Full pipeline runs in <2 seconds per clue
