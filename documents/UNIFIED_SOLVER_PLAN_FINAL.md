# Unified Solver Plan — Final

Date: 2026-05-24
Authors: Claude Code + Codex
Status: APPROVED by user 2026-05-24 22:06 — agreed without reservation
Phase 0: COMPLETE — audit findings in PHASE_0_AUDIT_2026-05-24.md
Phase 1: READY TO BEGIN — prerequisite confirmed by both parties

This document supersedes:
- UNIFIED_SOLVER_DESIGN.md
- legacy_first_evidence_recovery_plan_2026-05-24.md
- UNIFIED_SOLVER_PLAN.md
- CODEX_REVIEW_OF_UNIFIED_SOLVER_DESIGN_2026-05-24.md

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

7. A high-confidence legacy solve is never erased. Stage Two/Three may
   downgrade proof status to REVIEW because evidence is incomplete, but
   they must not replace or suppress the real solve result.

---

## Pipeline

```
Stage One  [clue_context.py]
  - atomise clue and answer to character atoms
  - POS tagging and grammar phrases
  - definition boundary detection using DB
  - persist: stage_one_context (tokens, spans, atoms, definition candidates,
    wordplay windows, POS spans, DB annotations)
        |
        v
Legacy/Obase Solver  [solve_clue in solver.py]
  - receives stage_one_context
  - catalog matching, synonym/abbreviation lookup, pattern verification
  - output: SolveResult
      sr.result.word_roles   (word, token, value) per wordplay word
      sr.definition          definition phrase found
      sr.confidence          0-100
      sr.analyses            per-word analysis for evidence
        |
        v
Evidence Adapter  [_attach_gt2_evidence in solver.py — modified]
  if sr.high_confidence:
    build pieces and assembly via build_ai_pieces(sr), build_assembly_dict(sr)
    call build_stage_two_from_solve_result(
        clue_text, answer, db, sr,
        stage_one_context=stage_one_context,
        pieces=pieces,
        assembly=assembly)
  else:
    call build_stage_two_casefile(clue_text, answer, db, stage_one_context)
        |
        v
Stage Two  [stage_two_casefile.py]
  High-confidence path: reads from solver evidence (see below)
  Fallback path: grammar-only evidence for partial/failed solves
  Output: StageTwoCaseFile with definition_candidates, source_candidates,
    operation_candidates, assemblies, unresolved_words,
    enrichment_candidates, word_coverage
        |
        v
Stage Three  [stage_three_proof.py — unchanged]
  Consumes StageTwoCaseFile
  Statuses: PASS | REVIEW | CONDITIONAL_NEEDS_ENRICHMENT | FAIL_EVIDENCE_ONLY
  Never writes to DB, never invents evidence
        |
        v
Persistence  [clue_pipeline_state — upsert by clue_id]
  One current row per clue. Columns:
    stage_one_json, stage_two_json, stage_three_json,
    wfw_json, status, confidence, solver_version, updated_at
  Rerun must update this row. App reads current row, not latest append.
        |
        v
WFW Display  [wfw_display_adapter.py]
  PASS:        show proven word-for-word breakdown
  REVIEW:      show what was found + what is missing honestly
  CONDITIONAL: show conditional evidence as candidate, not proof
  Reads current pipeline state only.
  Never reads stale wfw_proof_attempts as display truth.
```

---

## New Function: `build_stage_two_from_solve_result`

Location: `signature_solver/stage_two_casefile.py`

Signature:
```python
def build_stage_two_from_solve_result(
    clue_text,
    answer,
    db,
    solve_result,
    *,
    stage_one_context=None,
    pieces=None,       # from build_ai_pieces(sr) — caller provides
    assembly=None,     # from build_assembly_dict(sr) — caller provides
):
```

Caller provides `pieces` and `assembly` so this function never imports
from `sonnet_pipeline`. Package boundary is preserved.

If `pieces` or `assembly` are None, return evidence_only with a clear gap
recorded. Do not duplicate sig_adapter logic inside this function.

### Mapping from SolveResult to Stage Two

Source candidates — from `pieces` where mechanism is not an indicator:
```
synonym, abbreviation, positional, first_letter, last_letter, etc.
-> source span, mechanism, value
```

Operation candidates — from `pieces` where mechanism is an indicator:
```
anagram, container, reversal, deletion, homophone, positional, etc.
-> indicator span, mechanism
```

Assembly — directly from `assembly` dict. This covers all operation types
the legacy solver handles (charade, container, reversal, anagram, deletion,
and compounds). No new assembly logic needed.

Definition candidates — from `sr.definition`.

Unresolved words — clue words not in `word_roles` and not link words.

### Span Mapping (mandatory, not optional)

For every item in `word_roles`, map to Stage One spans using this priority:

1. Match by: same text, same token, same produced value, unused annotation
   occurrence in stage_one_context.annotations.
2. If no annotation: match by tokenised clue text occurrence, within the
   wordplay window, in word-order.
3. If still ambiguous or not found: record span=None, span_status="ambiguous".
   Stage Three will REVIEW rather than fabricate span certainty.
   Repeated words use occurrence-aware matching.

Spanless evidence is recorded as a gap, not silently dropped.

For every piece, Stage Two retains where possible:
```
clue token span
clue atom ids
answer token/letter span
answer atom ids
mechanism token
produced value
```

---

## Stage Three Statuses

The current code has two states: PASS and REVIEW.

Phase 1 does not add new enum names. It corrects the behaviour so those
two states are honest. The four behaviours below are what matters; the
exact names for conditional and evidence-only states are a Phase 4 cleanup.

```
PASS
  All words accounted for. Assembly verified. Definition matched.
  All sources DB-supported. No unresolved words.

REVIEW
  Covers all non-PASS outcomes in Phase 1:
  - assembly verified but surface words remain unaccounted for
  - definition not in DB
  - assembly would work if enrichments are accepted (conditional)
  - no assembly found (evidence only)
  In each case the output must describe what was found and what is missing.
  It must not suppress the legacy solve result.
```

Phase 4 may introduce CONDITIONAL_NEEDS_ENRICHMENT and FAIL_EVIDENCE_ONLY
as distinct states once the behaviour is verified and stable.

---

## Entry Points — Audit Before Touching

Do not change run.py or any entry point in the first patch.

First, audit every place that calls:
- `solve_clue` / `sig_solve_clue`
- `run_clue_pipeline`
- `run_signature_clue_pipeline`

Confirm whether each already goes through `_attach_gt2_evidence`. If any
entry point bypasses the new Stage-Two-from-SolveResult branch, correct it
after Phase 1-3 are verified.

Goal: one authoritative entry point for dashboard puzzle run, clue rerun,
admin rerun, and batch scripts. Do not force that unification in the first
patch if it risks breaking working paths.

---

## Emergency Changes — Review and Record

Recent emergency debugging changed `solve_clue`:
- disabled WFW assembly in the legacy path
- removed recursive fallback
- capped speculative GT2/indicator retries
- skipped token_parse assembly for unsolved results

Before or during Phase 1, review each of these changes:
- If consistent with the legacy-first design, keep it and record it as a
  deliberate design choice.
- If it was an accidental performance patch that the new architecture makes
  unnecessary, revert it cleanly.

In particular: `assemble=False` in the `solve_wfw_unified` call may be correct
if the new Stage-Two-from-SolveResult path makes WFW native assembly
unnecessary. Record the decision explicitly.

---

## `_conditional_suburb_enrichments` — Do Not Remove Yet

This hard-coded prototype for the HASBEEN clue is ugly but harmless.

Do not remove it in Phase 1. It will be replaced by a general
grammar-span enrichment rule in Phase 4. Mark it as technical debt.

---

## Regression Harness

Before claiming recovery, this clue set must be no worse than legacy:

```
SLAVISHLY        final letter + synonym charade
STIPULATION      anagram
AINTREE          WFW review case (definition not in DB)
TIJUANA          phrase sources (Note=TI, Spanish male=JUAN)
TOTALLY          charade with phrase sources (TOT+ALLY)
ROC              deletion (heading off)
HASBEEN          compound / conditional enrichment
one hidden clue
one double definition
one homophone
```

For each clue the harness records:
```
legacy high confidence?
current pipeline high confidence?
same answer?
Stage Two from SolveResult used?
Stage Three status?
wfw_proof row written?
display honest?
runtime (must not hang)
```

No claim of recovery until all rows pass.

---

## Phase 0 Decisions — Agreed by Both Parties

Recorded here after audit completion.

Emergency change 1 — recursion removed from solve_clue: KEEP. Consistent
with the plan. Removing recursion removed the 65s-per-call blowup.

Emergency change 2 — haiku_tried flag: KEEP. Two haiku calls both gated
correctly. Deliberate and consistent with the plan.

Emergency change 3 — _wfw_already_attempted parameter: REMOVE in cleanup.
Dead code since no caller passes True now that recursion is removed.

Emergency change 4 — solve_wfw_unified called before legacy solver: REMOVE
from solve_clue entirely. This is the prerequisite for Phase 1. It violates
the plan by allowing WFW to short-circuit the legacy solver. It moves to
_attach_gt2_evidence where it runs after the legacy solve, as evidence, not
as a competing solver.

Translation helpers: build_stage_two_from_solve_result translates
word_roles directly in stage_two_casefile.py for Phase 1. This is tightly
scoped and does not duplicate large logic. Shared helpers may be extracted
into signature_solver later if needed, but that is not Phase 1 scope.

---

## Implementation Order

### Phase 0 — Audit (no code changes)

1. Identify all call sites of solve_clue, run_clue_pipeline,
   run_signature_clue_pipeline.
2. Confirm high-confidence SolveResult still exists before Stage Two
   for a representative solved clue.
3. Confirm where build_ai_pieces and build_assembly_dict are called today.
4. Confirm current DB write path for clue_pipeline_state.
5. Review and record emergency solver changes.

### Phase 1 — Build Stage Two from SolveResult

6. Implement `build_stage_two_from_solve_result()` in stage_two_casefile.py.
7. Add the branch in `_attach_gt2_evidence()` in solver.py.
8. Run regression harness. All clues must be no worse than legacy.

### Phase 2 — Persistence and Display

9. Ensure clue_pipeline_state upserts correctly (one row per clue).
10. Ensure wfw display reads current pipeline state, not stale rows.
11. Re-run regression harness. Verify display is honest for all statuses.

### Phase 3 — Entry Point Audit

12. Map all entry points against the authoritative pipeline path.
13. Correct any that bypass it.
14. Re-run regression harness.

### Phase 4 — Enrich and Generalise

15. Replace `_conditional_suburb_enrichments` with general rule.
16. Extend grammar-only Stage Two fallback to cover more operation types.
17. Wire enrichment candidates to pending_enrichments table.
18. Only then improve parser coverage for partial/failed solves.

---

## Files Changed in Phase 1

```
signature_solver/stage_two_casefile.py   add build_stage_two_from_solve_result()
signature_solver/solver.py               branch in _attach_gt2_evidence()
```

## Files NOT Changed in Phase 1

```
signature_solver/stage_three_proof.py    unchanged
signature_solver/clue_context.py         unchanged
sonnet_pipeline/run.py                   unchanged until Phase 3 audit
web/routes/clue.py                       unchanged until Phase 3 audit
web/templates/clue.html                  unchanged
```
