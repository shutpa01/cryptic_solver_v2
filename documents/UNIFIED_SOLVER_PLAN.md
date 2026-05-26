# Unified Solver Plan — Reconciled

Date: 2026-05-24
Authors: Claude Code + Codex (reconciled)

This is the single agreed plan. It supersedes both
`UNIFIED_SOLVER_DESIGN.md` and `legacy_first_evidence_recovery_plan_2026-05-24.md`.

---

## Goal

Restore the system to at least legacy/obase solving strength, then add the new
design elements around it. The legacy solver is the foundation. Nothing new
replaces it until parity is proven.

Minimum success bar:
- Anything the legacy solver could solve must still solve.
- Every solve must retain enough structured evidence to audit and display.
- No new Stage/WFW component may suppress a correct legacy solve.

---

## Non-Negotiable Principles

1. Do no harm. The working solver path is protected. New stages observe,
   record, verify, and enrich. They do not replace working mechanisms.

2. Stage One always runs first. Every clue begins with atomisation and
   definition boundary detection. No solver path bypasses Stage One.

3. Legacy mechanism knowledge is authoritative. Obase/legacy mechanisms
   are the operational solver. The new system wraps them and captures what
   they found.

4. There is one pipeline. Puzzle run, clue rerun, admin rerun, and
   dashboard run all call the same function. They differ only in scope.

5. The database stores current truth. One current row per clue keyed by
   `clue_id`. Append-only artifacts are audit history, not app truth.

6. Failed clues retain evidence. Failure still records definition
   candidates, source candidates, mechanisms tried, partial assemblies,
   unresolved words, and enrichment candidates.

---

## Pipeline

```
Stage One
  - atomise clue and answer
  - POS tagging and grammar phrases
  - definition boundary detection (DB lookup)
  - output: stage_one_context
  - persist stage_one_context to clue_pipeline_state
        |
        v
Legacy/Obase Solver  [solve_clue in solver.py]
  - receives stage_one_context (definition candidates, word annotations)
  - catalog matching, synonym/abbreviation lookup, pattern verification
  - output: SolveResult (word_roles, definition, confidence, assembly)
        |
        v
Evidence Adapter  [new: build_legacy_solve_evidence(sr)]
  - translates SolveResult.word_roles into structured evidence
  - does not invent evidence; marks gaps as gaps
  - output: legacy_solve_evidence dict containing:
      definition_span, wordplay_span, source_spans,
      mechanism_type, indicator_spans, produced_values,
      assembly_order, answer_letter_links, confidence,
      unresolved_clue_spans
        |
        v
Stage Two  [build_stage_two_from_legacy_evidence() — new primary path]
  - consumes stage_one_context + legacy_solve_evidence
  - populates: definition_candidates, source_candidates,
    operation_candidates, assemblies, unresolved_words,
    enrichment_candidates, answer_coverage
  - does NOT re-solve; reads what the solver found
  - fallback: build_stage_two_casefile() for failed/partial solves
        |
        v
Stage Three  [build_stage_three_proof() — existing, unchanged logic]
  - verifies Stage Two casefile
  - statuses: PASS | REVIEW | CONDITIONAL_NEEDS_ENRICHMENT | FAIL_EVIDENCE_ONLY
  - never writes to DB, never invents evidence
        |
        v
Persistence  [clue_pipeline_state — upsert by clue_id]
  - one current row per clue
  - columns: stage_one_json, legacy_solve_evidence_json,
    stage_two_json, stage_three_json, wfw_display_json,
    status, confidence, solver_version, updated_at
        |
        v
WFW Display  [display_from_stage_three_proof()]
  - PASS: show proven word-for-word breakdown
  - REVIEW/CONDITIONAL: show what was found + what is missing
  - reads current pipeline state only; never reads stale wfw_proof_attempts
```

---

## Evidence Adapter Detail

New function: `build_legacy_solve_evidence(sr, clue_text, stage_one_context)`

Lives in `signature_solver/sig_evidence_adapter.py` (new file, clean separation
from sonnet_pipeline so stage_two_casefile.py can import it directly).

Mapping from SolveResult:

```
sr.result.word_roles  ->  per-word structured evidence
  (word, SYN_F, value)     -> source span, mechanism=synonym, value
  (word, ABR_F, value)     -> source span, mechanism=abbreviation, value
  (word, ANA_I, None)      -> indicator span, mechanism=anagram
  (word, CON_I, None)      -> indicator span, mechanism=container
  (word, REV_I, None)      -> indicator span, mechanism=reversal
  (word, DEL_I, None)      -> indicator span, mechanism=deletion
  (word, LNK, None)        -> link word span, no mechanism
  (word, POS_I_*, None)    -> indicator span, mechanism=positional

sr.definition            -> definition_span text
sr.confidence            -> confidence score

words in clue NOT in word_roles and NOT link words
                         -> unresolved_clue_spans
```

Span mapping: use stage_one_context.annotations to get (start, end) for each
word. Fall back to word-order position if word not found in annotations.

Assembly is inferred from the token mix:
- ANA_I present -> anagram
- CON_I present -> container
- REV_I present -> reversal
- DEL_I present -> deletion
- Only SYN_F/ABR_F -> charade
- Mixed -> compound (charade of operations)

This covers all operation types the legacy solver already handles. No new
assembly logic is needed.

---

## Stage Two from Legacy Evidence

New function: `build_stage_two_from_legacy_evidence(clue_text, answer, legacy_evidence, stage_one_context)`

Logic:

```python
if legacy_evidence is None or not legacy_evidence.get("sources"):
    # Fall back to grammar-only path
    return build_stage_two_casefile(clue_text, answer, db, stage_one_context)

assemblies = [legacy_evidence["assembly"]]   # already correct for all op types
definition_candidates = [legacy_evidence["definition_span"]]
source_candidates = legacy_evidence["source_spans"]
operation_candidates = legacy_evidence["indicator_spans"]
unresolved_words = legacy_evidence["unresolved_clue_spans"]
enrichment_candidates = grammar_span_enrichments(
    stage_one_context, unresolved_words, source_candidates)

return StageTwoCaseFile(
    ...,
    assemblies=assemblies,
    status="answer_fit" if high_confidence else "evidence_only"
)
```

---

## Stage Three Statuses

```
PASS
  All words accounted for. Assembly verified. Definition matched.
  Definition and all sources are DB-supported.

REVIEW
  Assembly verified but surface words remain unaccounted for.
  Or definition not in DB. Correct solve, needs enrichment.

CONDITIONAL_NEEDS_ENRICHMENT
  Assembly would work if pending enrichments are accepted.
  Must not be shown as proven.

FAIL_EVIDENCE_ONLY
  No assembly found. Records what partial evidence was gathered.
```

---

## Entry Point Unification

All of these must call the same function:

```
dashboard puzzle run
normal pipeline puzzle run
clue rerun (web route)
admin rerun
batch scripts
```

Single authoritative entry point:

```python
run_authoritative_clue_pipeline(
    conn, clue_id, source, puzzle_number,
    clue_text, answer, ref_db,
    write_db=True, store_solution=True)
```

This replaces `run_signature_clue_pipeline`, `run_clue_pipeline`, and any
direct calls to `sig_solve_clue` in run.py.

---

## Regression Harness

Before claiming recovery, this clue set must be no worse than legacy:

```
SLAVISHLY        - final letter + synonym charade
STIPULATION      - anagram
AINTREE          - WFW review case (definition not in DB)
TIJUANA          - phrase sources (Note=TI, Spanish male=JUAN)
TOTALLY          - charade with phrase sources (TOT+ALLY)
ROC              - deletion (heading off)
HASBEEN          - compound / conditional enrichment
one hidden clue
one double definition
one homophone
```

For each clue the harness records:
- legacy solved? / current pipeline solved? / same answer?
- same or better confidence?
- evidence retained in stage_two_json?
- Stage Three status?
- display honest?

No claim of recovery until every row passes.

---

## What Stops Now

- No more `solve_wfw_unified` in the live pipeline path as a solver.
- No more new solving architecture until parity is restored.
- No more duplicated solver phases across different entry points.
- No more app reading "latest append row" from wfw_proof_attempts as display truth.
- No more `_conditional_suburb_enrichments` hard-code (Phase 4 in handover).

---

## Files Changed

```
signature_solver/sig_evidence_adapter.py     NEW — build_legacy_solve_evidence()
signature_solver/stage_two_casefile.py       ADD build_stage_two_from_legacy_evidence()
signature_solver/solver.py                   _attach_gt2_evidence() calls new path
sonnet_pipeline/clue_pipeline.py             rename/consolidate to run_authoritative_clue_pipeline()
sonnet_pipeline/run.py                       Phase 1 calls run_authoritative_clue_pipeline()
web/routes/clue.py                           rerun calls run_authoritative_clue_pipeline()
```

## Files NOT Changed

```
signature_solver/stage_three_proof.py        logic unchanged
signature_solver/clue_context.py             unchanged
web/templates/clue.html                      unchanged
data schemas                                 clue_pipeline_state columns extended, no drops
```

---

## Implementation Order

1. Audit all current entry points. Map every place that calls solve_clue,
   run_clue_pipeline, run_signature_clue_pipeline.
2. Write `build_legacy_solve_evidence()` in sig_evidence_adapter.py.
3. Write `build_stage_two_from_legacy_evidence()` in stage_two_casefile.py.
4. Wire them into `_attach_gt2_evidence()` in solver.py.
5. Create `run_authoritative_clue_pipeline()` consolidating all entry points.
6. Update run.py Phase 1 and web/routes/clue.py to use it.
7. Run regression harness. Fix until all clues pass.
8. Only then improve parser coverage for partial/failed solves.

---

## Open Questions for Implementation

1. `solve_wfw_unified` is currently called inside `solve_clue`. It must be
   removed from the live path entirely (it was the 65s-per-clue cause).
   Confirm: is there any display or proof path that still needs it, or is
   it fully replaced by the legacy evidence adapter?

2. The existing `stage_three_proof.py` checks assume Stage Two assemblies
   are present. Once Stage Two is fed real assemblies from the legacy solver,
   which checks will still fail for the regression harness clue set, and why?
   This needs a dry run before claiming the architecture is correct.

3. `clue_pipeline_state` currently has columns stage_one_json, stage_two_json,
   stage_three_json, wfw_json. The new column `legacy_solve_evidence_json` needs
   to be added. Is this a schema migration or does the existing infrastructure
   handle new columns automatically (via _ensure_column)?
