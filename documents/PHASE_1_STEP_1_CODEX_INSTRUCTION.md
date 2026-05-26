# Phase 1 Step 1 — Codex Instruction

Task: Remove solve_wfw_unified from its position as a pre-legacy solver in
solve_clue, and move it to _attach_gt2_evidence where it runs as evidence
collection after the legacy solve.

File: signature_solver/solver.py

Do not touch any other file in this step.
Do not change the legacy solver logic.
Do not change any return statements in solve_clue other than what is described.

---

## Change 1 — Remove the WFW pre-solver block from solve_clue

Find this block (currently lines 334-356):

    # --- WFW-native authoritative path ---
    # The new solver starts from preserved character atoms and materialised
    # working-out. If it can prove the clue, it wins; the legacy return shape
    # is only an adapter for existing callers.
    if _wfw_already_attempted:
        wfw_result = _wfw_result
    else:
        wfw_result = None
        try:
            from .wfw_unified_solver import solve_wfw_unified
            wfw_result = solve_wfw_unified(
                clue_text, answer_clean, db, assemble=False,
                manual_roles=manual_roles,
                clue_context=clue_context,
                stage_one_context=stage_one_context)
            wfw_sr = _solve_result_from_wfw(wfw_result, db)
            if wfw_sr is not None and wfw_sr.confidence >= min_confidence:
                return _attach_gt2_evidence(
                    wfw_sr, clue_text, answer_clean, db, candidates,
                    getattr(wfw_result, "clue_context", None),
                    stage_one_context)
        except Exception:
            wfw_result = None

Replace the entire block with this single line:

    wfw_result = None

Reason: solve_wfw_unified must not run before the legacy solver. It must not
be able to return early and bypass the legacy solve. wfw_result is still
initialised to None so all existing references to it downstream compile
correctly.

---

## Change 2 — Remove dead parameters from solve_clue signature

Find the solve_clue function signature (currently lines 241-247):

    def solve_clue(clue_text, answer, db, min_confidence=0, extra_catalog=None,
                   extra_synonyms=None, extra_indicators=None,
                   _dbe_already_attempted=False,
                   _span_value_already_attempted=False,
                   manual_roles=None,
                   _wfw_already_attempted=False,
                   _wfw_result=None):

Remove the two dead parameters _wfw_already_attempted and _wfw_result.

Replace with:

    def solve_clue(clue_text, answer, db, min_confidence=0, extra_catalog=None,
                   extra_synonyms=None, extra_indicators=None,
                   _dbe_already_attempted=False,
                   _span_value_already_attempted=False,
                   manual_roles=None):

Reason: _wfw_already_attempted and _wfw_result were guards for recursive calls
that no longer exist. They are dead code. No caller passes them.

---

## Change 3 — Add solve_wfw_unified to _attach_gt2_evidence

Find _attach_gt2_evidence (currently lines 87-125). It currently starts:

    def _attach_gt2_evidence(sr, clue_text, answer_clean, db, candidates,
                             clue_context=None, stage_one_context=None):
        """Attach read-only GT2 evidence bundles to a SolveResult."""
        if sr is None:
            return sr
        if stage_one_context is not None:
            sr.stage_one_context = stage_one_context
            try:
                from .stage_two_casefile import build_stage_two_casefile
                from .stage_three_proof import build_stage_three_proof
                stage_two = build_stage_two_casefile(
                    clue_text, answer_clean, db,
                    stage_one_context=stage_one_context)
                sr.stage_two_casefile = stage_two
                sr.stage_three_proof = build_stage_three_proof(stage_two)
            except Exception:
                pass

Add a new parameter wfw_result=None to the function signature, and attach
it to sr so downstream display code can use it. Replace the function
definition line with:

    def _attach_gt2_evidence(sr, clue_text, answer_clean, db, candidates,
                             clue_context=None, stage_one_context=None,
                             wfw_result=None):

Then immediately after the line:

        if sr is None:
            return sr

Add this block:

        # If no wfw_result was passed in, build it now as evidence only.
        # This runs after the legacy solve — it is evidence collection,
        # not a solver. It must never return early or replace sr.
        if wfw_result is None and stage_one_context is not None:
            try:
                from .wfw_unified_solver import solve_wfw_unified
                wfw_result = solve_wfw_unified(
                    clue_text, answer_clean, db, assemble=False,
                    clue_context=clue_context,
                    stage_one_context=stage_one_context)
            except Exception:
                wfw_result = None
        if wfw_result is not None:
            sr.wfw_unified_result = wfw_result

The rest of _attach_gt2_evidence is unchanged.

---

## After Making the Changes

Paste the following for audit before running anything:

1. The full solve_clue function signature (the def line and its parameters).
2. The complete block where wfw_result = None now sits (10 lines of context
   either side, so the removal is clearly visible).
3. The complete _attach_gt2_evidence function from its def line to its
   return sr line.

Do not run the pipeline or any tests until Claude has audited those three
sections and confirmed the changes are correct.
