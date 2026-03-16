# Continuity Summary — 2026-03-16
# Next task: Full review of both systems to combine them for best effect

## What We Have: Two Solver Systems

### 1. Signature Solver (`signature_solver/`)
**Mechanical, zero-API, produces verified component-level explanations.**

- Analyses each word's possible roles via DB lookups (synonyms, abbreviations, indicators)
- Matches against catalog of known wordplay patterns (76 base I/F patterns)
- Executes mechanically: concatenation, insertion, reversal, anagram, hidden, positional extraction
- Confidence scoring: base 60 + per-piece verification bonuses, all-verified guarantees 82+
- Entry point: `signature_solver/solver.py` → `solve_clue(clue_text, answer, db)`
- Test: `python -m signature_solver.test_puzzle 31185 telegraph`

**Current scores (mechanical only, zero API):**
| Puzzle | HIGH (80+) | Found | Total |
|--------|-----------|-------|-------|
| 31184 | 12 (43%) | 13 | 28 |
| 31174 | 7 (23%) | 7 | 30 |
| 31185 | 15 (47%) | 15 | 32 |

### 2. Production Solver (`sonnet_pipeline/`)
**API-powered (Claude Sonnet), two-pass reasoning + structuring.**

- Pass 1: Claude reasons about the clue in free text
- Pass 2: Claude structures its reasoning into JSON (definition, wordplay_type, pieces)
- Assembly verification: mechanically checks pieces produce the answer
- Enrichment: DB lookups provide context/hints to the API
- Score 80+ auto-approved; lower scores go to review queue
- Generates DB gap suggestions (pending_enrichments) for human approval
- Entry point: `sonnet_pipeline/run.py`

**Puzzle 31185 production scores:** 8 HIGH, 16 MED, 1 LOW, 7 failed

### 3. Integration Test (`signature_solver/test_integration.py`)
**Proves the combined pipeline concept:**
- Phase 1: Signature solver (mechanical) — 10/32 HIGH
- Phase 2: API call → inject discoveries into cloned RefDB → re-run signature — +3 more
- Phase 3: Still needs pure AI explanation — 19/32
- Coverage: 13/32 (41%) with signature-quality explanations

## Key Insight: They Are Sequential, Not Alternatives

```
Signature solver runs first (free, instant)
    → HIGH? Serve signature explanation. Done.
    → Not HIGH? Production solver runs (API call)
        → Discovers pieces, suggests DB additions
        → Human approves DB additions
        → Signature solver can now solve that pattern forever
```

The production solver is the **discovery engine**. The signature solver is the **explanation engine**. Every approved DB entry permanently expands signature coverage with superior explanation quality.

## What Was Fixed This Session

### Confidence Scoring (`confidence.py`)
- All-verified pieces guarantee score 82+ (was capped at 68-75 for simple patterns)
- Circularity penalty only for exact answer match (was falsely penalizing charade pieces like DIE in DIET)

### New Operation: container_positional
- EX inside SY → SEXY (container where one piece from positional extraction)
- Required: two indicator types (CON_I + POS_I_*) — solved by scanning leftover words for POS_I
- Files: base_catalog.py, base_matcher.py, matcher.py, data/base_catalog.json

### Matcher Improvements
- Indicator words can be "skippable" gap words (not just LNK) — was blocking PIPER
- POS_F extraction tries both use-type and trim-type indicators
- POS_I_TRIM_MIDDLE also tries POS_I_OUTER (because "empty X" = keep outer shell)
- Container-outer synonym check includes container_positional

### Link Words
- Added "wanting", "needing", "requiring" to LINK_WORDS (coexist with DEL_I role)

### DB Inserts (cryptic_new.db)
- "seize" → container indicator (blocked DUPE)
- "bottling" → container indicator (blocked PINOT)
- "one plays" → PIPER definition (definition split was wrong)

## Why Clues Fail: Failure Categories

From analysing puzzle 31185's 17 unsolved clues:

1. **Definition gate (no def in DB)**: ~8 clues — the solver can't find a definition→answer mapping, so it never gets to the wordplay window. Production handles this via AI.

2. **Synonym/abbreviation gaps**: The specific word→letters mapping isn't in the DB. E.g. "complete"→RIPE, "deflated"→LIMP. These are the gaps the production solver discovers and suggests.

3. **Indicator gaps**: Words not tagged as the right indicator type. E.g. "seize" wasn't CON_I. Fixed by manual DB inserts.

4. **Complex multi-step operations**: E.g. ARRAIGN = homophone of A+REIGN (charade inside homophone). Not yet supported.

5. **Operations not in catalog**: Some wordplay types still missing (substitution, some compound types).

## Architecture Quick Reference

### Signature Solver Files
```
signature_solver/
  solver.py          — main entry: solve_clue(), solve(), extract_definition_candidates()
  db.py              — RefDB class (loads cryptic_new.db into memory dicts)
  word_analyzer.py   — analyze_phrases() → WordAnalysis per word + phrases
  tokens.py          — 28 token types, LINK_WORDS, indicator mappings
  base_catalog.py    — BaseEntry, OPERATION_INDICATOR_TYPE, OPERATION_FODDER_TYPES
  base_matcher.py    — match_base() — flexible spans, tries all I/F combos
  positional_matcher.py — match_positional() — fixed spans from JSON catalog
  matcher.py         — match_signatures() + shared: _lookup_slot, _verify_combo
  catalog.py         — hand-curated 247 entries (oldest matcher)
  executor.py        — execute_signature(), extract_positional(), assembly ops
  confidence.py      — score_result() — per-piece verification scoring
  api_solver.py      — API fallback with structured evidence
  evidence.py        — format_evidence() for API prompts
  test_puzzle.py     — full puzzle test harness
  test_integration.py — integration test (signature + API enrichment)
```

### Production Solver Files
```
sonnet_pipeline/
  run.py             — orchestrates pipeline: solve → score → store → report
  solver.py          — two-pass API (reasoning + structuring), assembly, scoring
  enricher.py        — DB lookups for API context, definition validation
  report.py          — generates reports, extracts DB gaps
```

### Key Databases
- `data/cryptic_new.db` — reference tables (indicators, synonyms_pairs, wordplay, homophones, definition_answers_augmented)
- `data/clues_master.db` — all clues, answers, structured_explanations, pending_enrichments

## Next Session Goals

**Full review of both systems to combine them for best effect:**

1. **How to wire signature solver into production pipeline** — run signature first, skip API if HIGH, feed API discoveries back into DB
2. **Replace AI explanations with signature explanations** where signature can solve — superior quality
3. **Systematic gap analysis** — which gap categories have highest impact? Where should we focus DB enrichment?
4. **Definition gate** — biggest single blocker. Options: expand definition DB, try all splits even without DB match, use production solver's AI-identified definitions
5. **Explanation format** — signature gives `pibroch's=P, support=PIER = PIPER` — how should this render in the web app vs what production currently shows?
6. **Test on more puzzles** to validate improvements generalise
