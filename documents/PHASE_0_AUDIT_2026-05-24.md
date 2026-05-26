# Phase 0 Audit

Date: 2026-05-24
Status: Complete — no code changes made

This document records all Phase 0 findings as required by the agreed plan
before any Phase 1 implementation begins.

---

## 1. Entry Points

Every place that calls solve_clue, run_clue_pipeline, or
run_signature_clue_pipeline:

solve_clue (signature_solver/solver.py line 241) is called directly from:

  web/explainer.py line 200 — explain_clue()
  sonnet_pipeline/tftt_pipeline.py line 297 — solve_with_sonnet()
  sonnet_pipeline/tftt_pipeline.py line 784 — same context
  sonnet_pipeline/run.py line 963 — second clue loop (Phase 3 enrichment pass)
  backfill_ai_exp/batch_enrichment.py line 190 — run_batch()
  backfill_ai_exp/batch_offline.py line 341 — batch processing loop
  sonnet_pipeline/sig_mining.py line 195 — mining loop

run_signature_clue_pipeline (sonnet_pipeline/clue_pipeline.py line 25) is
called from:

  sonnet_pipeline/clue_pipeline.py line 85 — inside run_clue_pipeline()
  sonnet_pipeline/run.py line 616 — Phase 1 puzzle solve loop

run_clue_pipeline (sonnet_pipeline/clue_pipeline.py line 74) is called from:

  sonnet_pipeline/run.py line 1185 — Phase 3 enrichment pass
  web/routes/admin.py line 912 — admin rerun
  web/routes/admin.py line 1108 — admin edit/update endpoint

Conclusion: there is no single authoritative entry point today. The puzzle run,
admin rerun, and several batch scripts call different paths. Entry point
unification is confirmed as Phase 3 work.

---

## 2. _attach_gt2_evidence — Insertion Point for Phase 1

Location: signature_solver/solver.py lines 87-125.

Current behaviour:
- Calls build_stage_two_casefile only when stage_one_context is not None.
- Immediately calls build_stage_three_proof on the result.
- Both calls are in a single try/except block.
- No branch exists for high-confidence solves.

This is the single insertion point for Phase 1.
The branch to add: if sr.high_confidence and sr.result, call
build_stage_two_from_solve_result instead of build_stage_two_casefile.

---

## 3. build_ai_pieces and build_assembly_dict — Circular Import Constraint

Both functions live in sonnet_pipeline/sig_adapter.py.

sig_adapter.py imports from signature_solver. Therefore solver.py cannot
import from sig_adapter without creating a circular import.

build_ai_pieces is called from:
  sonnet_pipeline/sig_adapter.py lines 724, 782 (internal)
  backfill_ai_exp/batch_offline.py line 172

build_assembly_dict is called from:
  sonnet_pipeline/sig_adapter.py lines 725, 783 (internal)
  backfill_ai_exp/batch_offline.py line 173

Conclusion: build_stage_two_from_solve_result in stage_two_casefile.py must
translate sr.result.word_roles directly, without calling build_ai_pieces or
build_assembly_dict. The translation logic belongs in stage_two_casefile.py.
Option (b) from the plan (caller passes pieces and assembly) is not viable
because the caller is _attach_gt2_evidence in solver.py, which has the same
circular import constraint. Direct translation is the right approach.

---

## 4. clue_pipeline_state Schema

Primary key: clue_id INTEGER PRIMARY KEY.
Upserts already work via INSERT OR REPLACE.

Columns confirmed present:
  stage_one_json
  annotations_json
  stage_two_json
  stage_three_json
  wfw_json
  status, confidence, solver_version, updated_at

No schema change needed for Phase 1.

---

## 5. Emergency Solver Changes — Assessment

The following changes were made during emergency debugging. Each is assessed
against the agreed plan.

Change 1: Recursion removed from solve_clue.

Assessment: CONSISTENT WITH PLAN. The plan requires legacy solve to remain
authoritative and fast. Removing recursion removed the 65s-per-recursive-call
blowup. Keep this change.

Change 2: haiku_tried flag preventing double Haiku calls.

Assessment: CONSISTENT WITH PLAN. Two haiku calls exist (line 323 when no
candidates, line 757 as second-chance), both gated by haiku_tried. This is
correct and deliberate. Keep this change.

Change 3: _wfw_already_attempted parameter added to solve_clue.

Assessment: NOW DEAD CODE. The parameter exists in the signature but no caller
passes it as True, because recursion was removed. The guard at line 338
(if _wfw_already_attempted: wfw_result = _wfw_result) is never triggered.
This can be removed in a cleanup pass but does no harm in Phase 1.

Change 4: solve_wfw_unified called with assemble=False.

Assessment: INCONSISTENT WITH PLAN. This is the most important finding.

solve_wfw_unified is still called unconditionally in the live solve path at
solver.py lines 343-356, even with assemble=False. It still runs stages 1-4
(atom context, grammar context, candidate graph). If it returns a result with
confidence >= min_confidence, solve_clue returns early and the legacy solver
never runs. This violates the plan principle that legacy mechanism knowledge
is authoritative.

assemble=False skips the assembly step (stage 5), which removes some cost,
but stages 1-4 still run for every clue. This is still adding substantial
work to what is supposed to be a fast mechanical solve path.

The plan states: no more WFW native assembly in the live path as a replacement
solver. The current code still allows solve_wfw_unified to short-circuit the
legacy solver.

Decision required before Phase 1: this call should either be removed from
solve_clue entirely, or moved to _attach_gt2_evidence where it runs after
the legacy solve rather than before it. The plan says the pipeline is:
legacy solver first, then evidence wrapping. solve_wfw_unified belongs in
the evidence wrapping stage, not as a pre-solver.

---

## 6. _conditional_suburb_enrichments

Location: signature_solver/stage_two_casefile.py lines 411-443.

Only fires when both conditions are true:
  - a working_pair exists with source_text "base" producing output "ASBE"
  - both "north london suburb" and "western half" appear as grammar spans

This is HASBEEN-specific. It is harmless and should not be removed in Phase 1.
Marked as technical debt for Phase 4.

---

## Summary — What Phase 1 Must Address

Two files change:

  signature_solver/stage_two_casefile.py
    Add build_stage_two_from_solve_result() translating sr.result.word_roles
    directly.

  signature_solver/solver.py
    In _attach_gt2_evidence: add high-confidence branch calling
    build_stage_two_from_solve_result.

One decision required before Phase 1 starts:

  solve_wfw_unified at solver.py lines 343-356 is inconsistent with the plan.
  It must be removed from its current position as a pre-legacy-solver early
  return path. It should move to _attach_gt2_evidence and run after the legacy
  solve, not before it. This is a prerequisite for Phase 1, not a separate
  phase.
