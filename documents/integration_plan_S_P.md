# Integration Plan: Signature Solver (S) into Production Pipeline (P)

## Flow

```
For each clue in puzzle:
  1. S runs (zero cost)
     - HIGH confidence → store S result, skip P
  2. P runs on remaining clues (API cost, once each)
  3. Extract gaps from P → inject into in-memory RefDB clone
  4. S re-runs on failures with enriched DB (zero cost)
     - HIGH now → overwrite with S result
     - Still failed → keep P's result
```

## Implementation: 3 files to create, 2 files to modify

### New: `sonnet_pipeline/sig_adapter.py` (~150 lines)

Maps S's SolveResult to P's DB schema and result dict format.

**Functions:**
- `SIG_OP_TO_TYPE` — maps S operations to P wordplay type strings
- `SIG_TOKEN_TO_MECHANISM` — maps S tokens to P mechanism strings
- `build_assembly_dict(sr)` — builds P-compatible assembly dict from word_roles
- `build_result_dict(sr, row_data)` — builds full result dict for results[]
- `store_signature_result(conn, clue_id, sr, clue_text, answer)` — writes to clues + structured_explanations

**DB field mapping:**
- `has_solution`: 1 (HIGH), 2 (MED), 0 (FAIL)
- `ai_explanation`: sr.result.explanation_parts[0]
- `reviewed`: 1 if HIGH (auto-approved), 0 otherwise
- `wordplay_type`: from SIG_OP_TO_TYPE
- `structured_explanations.components`: JSON with ai_pieces, assembly, wordplay_type
- `structured_explanations.confidence`: sr.confidence / 100.0
- `structured_explanations.model_version`: "signature_solver_v1"

**Token → mechanism mapping:**
- SYN_F → "synonym"
- ABR_F → "abbreviation"
- ANA_F → "anagram_fodder"
- RAW → "literal"
- POS_F → "first_letter"/"last_letter"/etc. (from positional indicator)
- HID_F → "hidden"
- HOM_F → "sound_of"
- Indicators (ANA_I, REV_I, etc.) → skip
- LNK → skip

**Assembly dict shapes (must match what _describe_assembly expects):**
- charade: `{"op": "charade", "order": ["P", "AGE"]}`
- container: `{"op": "container", "inner": "X", "outer": "WORD"}`
- anagram: `{"op": "anagram", "fodder": ["WORD"]}`
- reversal: `{"op": "reversal", "reversed": "DROW"}`
- hidden: `{"op": "hidden", "words": "clue text words"}`

### New: `sonnet_pipeline/sig_enrichment.py` (~40 lines)

- Extract `enrich_refdb()` from test_integration.py (or import directly)
- `collect_gaps_from_results(results)` — pulls synonym/abbreviation pieces from P results

### Modify: `sonnet_pipeline/run.py`

**Phase 1 (before main loop):**
- Load RefDB once (~2s)
- Run S on all clues
- Build sig_solved_ids set and sig_results dict
- Store S HIGH results to DB via sig_adapter
- Append S result dicts to results[]

**Phase 2 (existing loop, modified):**
- Skip clues in sig_solved_ids
- Everything else unchanged

**Phase 3 (after main loop, before DB commit):**
- Collect gaps from P results
- Clone RefDB, inject gaps in-memory
- Re-run S on failures with enriched DB
- Replace results and store if S now HIGH

**Stats updates:**
- Add stats["signature"] and stats["signature_re"] counters
- Update summary printing

### Modify: `sonnet_pipeline/report.py`

**1 line only:**
- Add "Signature" to tier label mapping dict

## What NOT to change

- All files in `stages/`
- `sonnet_pipeline/solver.py` — P's solver logic
- `sonnet_pipeline/enricher.py`
- `signature_solver/` — all files unchanged
- `dashboard/` — all files unchanged (reads same DB tables)
- DB schemas — no changes

## Edge cases

- Cross-references: let P handle (S can't resolve them)
- Single-clue mode: S runs first, P fallback
- Force flag: S runs regardless of cache
- Partials: S can upgrade has_solution=2 to 1
- S medium / P high: P wins (only use S when HIGH or P failed)

## Performance

- Phase 1: ~2s (30 clues, DB only)
- Phase 2: same as today but ~40% fewer API calls
- Phase 3: ~0.5s
- Net: same wall-clock, lower cost, better explanations
