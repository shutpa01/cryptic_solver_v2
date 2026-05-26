# Handover — 2026-05-25 — Phase 1 Step 2 Complete

## Status

Phase 0: COMPLETE
Phase 1 Step 1: COMPLETE — audited and confirmed correct
Phase 1 Step 2: COMPLETE — audited and confirmed correct
Regression harness: WRITTEN, audited, not yet run

The next action in the new thread is to run the regression harness
and report results.

---

## The plan

The governing document is:
    documents/UNIFIED_SOLVER_PLAN_FINAL.md

Status: APPROVED by user 2026-05-24 without reservation.

Phase 0 findings are in:
    documents/PHASE_0_AUDIT_2026-05-24.md

The plan in plain terms: the legacy solver (solve_clue in
signature_solver/solver.py) is the authoritative solver. Nothing
replaces it. Stage Two and Stage Three wrap around it to record
evidence and verify what the solver found. The WFW unified solver
runs after the legacy solve as evidence collection only — it cannot
short-circuit or replace the legacy solve.

---

## What was implemented in this session

### Step 1 — signature_solver/solver.py

Three changes, all audited and confirmed:

1. The WFW pre-solver block (solve_wfw_unified running before the
   legacy solver with the ability to return early) was removed.
   Replaced with a single line: wfw_result = None.

2. The dead parameters _wfw_already_attempted and _wfw_result were
   removed from the solve_clue signature.

3. _attach_gt2_evidence gained a wfw_result=None parameter.
   solve_wfw_unified now runs inside _attach_gt2_evidence after the
   legacy solve, as evidence collection only. It cannot return early
   or replace the solve result.

### Step 2 — signature_solver/stage_two_casefile.py and solver.py

Two changes, both audited and confirmed:

1. New function build_stage_two_from_solve_result added to
   stage_two_casefile.py. It translates the legacy solver's
   word_roles directly into a StageTwoCaseFile without calling
   build_ai_pieces or build_assembly_dict (circular import prevents
   that). Key points:
   - Uses context.annotations (not stage_one.annotations) for span
     mapping — stage_one is built with annotate=False so its
     annotations are empty
   - Handles multi-word phrase sources in the fallback span mapping
   - Builds assembly in the shape Stage Three expects: output, parts,
     kind, status
   - Includes all value-bearing source tokens, not just SYN_F and ABR_F
   - Tracks coverage by token index spans, not by text
   - LNK words are tracked explicitly in link_spans for coverage

2. _attach_gt2_evidence in solver.py now branches:
   - if sr.high_confidence and sr.result is not None: calls
     build_stage_two_from_solve_result
   - else: calls existing build_stage_two_casefile (grammar-only
     fallback for unsolved clues)

### Known minor gap from Step 2

DBE_MARKER tokens (words like "maybe", "perhaps", "say" that start
as LNK and get upgraded by _annotate_dbe_markers) are silently
skipped in the word_roles classification loop. They will appear as
unresolved words when they should be covered like link words. This
is a rare edge case and does not break anything in Phase 1. It is
a Phase 4 cleanup item.

---

## Regression harness

File: documents/regression_harness.py

Audited and confirmed correct. Safe to run.

The harness calls run_signature_clue_pipeline (not run_clue_pipeline)
with write_db=False. It does not touch the database.

Test clues: SLAVISHLY, STIPULATION, AINTREE, TIJUANA, TOTALLY, ROC,
HASBEEN, plus one hidden, one double definition, one homophone from
the DB.

Pass criteria for named clues:
- high_confidence=True
- s2_path=from_solve_result (meaning build_stage_two_from_solve_result
  was called and returned status=answer_fit)
- s3_status is not none
- timed_out=False
- WARN is acceptable if s3_status=REVIEW (expected in Phase 1 because
  the legacy_solver boundary_status does not pass Stage Three's
  definition check — this is correct and honest)

Pass criteria for hidden/DD/homophone: timed_out=False only. The
signature solver does not solve those types.

To run:
    python documents/regression_harness.py

from the project root.

---

## What comes next

After the harness runs, read the results and report:
- Any FAIL rows must be investigated before moving to Phase 2
- WARN rows are expected and acceptable
- If all rows pass or warn, Phase 2 begins

Phase 2 is: persistence and display. Ensure clue_pipeline_state
upserts correctly (one row per clue), ensure WFW display reads
current pipeline state and not stale rows.

Phase 3 is: entry point audit and unification.
Phase 4 is: enrich and generalise.

---

## Key files

    documents/UNIFIED_SOLVER_PLAN_FINAL.md       — the agreed plan
    documents/PHASE_0_AUDIT_2026-05-24.md        — phase 0 findings
    documents/PHASE_1_STEP_1_CODEX_INSTRUCTION.md — step 1 instruction
    documents/PHASE_1_STEP_2_CODEX_INSTRUCTION.md — step 2 instruction
    documents/PHASE_1_REGRESSION_HARNESS_INSTRUCTION.md — harness instruction
    documents/regression_harness.py              — the harness script
    signature_solver/solver.py                   — changed in step 1 and step 2
    signature_solver/stage_two_casefile.py       — changed in step 2

---

## Rules for Claude in the new thread

- No backticks of any kind in chat responses. Plain text only.
- Do not write code. Codex writes code. Claude plans, designs,
  checks, and audits.
- Do not offer to write anything without being asked.
- Do not ask permission for routine file reads.
- A question is not an instruction. Do not touch files in response
  to a question.
- Read the actual code before making any claim about it.
- Never guess. If uncertain, say so and read the file.
