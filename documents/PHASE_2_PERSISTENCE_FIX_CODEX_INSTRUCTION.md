# Phase 2 Persistence Fix — Codex Instruction (revised)

## Task

Fix the persistence gap that allows clue_pipeline_state to have stage_three_json
with no corresponding wfw_proof_attempts row.

Three files change. No other files may be touched.

    signature_solver/atomic_parse_store.py
    sonnet_pipeline/clue_pipeline.py
    sonnet_pipeline/sig_adapter.py


---

## Root cause

There are two places in production code that write to clue_pipeline_state.
Neither reliably keeps wfw_proof_attempts in sync.

Place 1: upsert_solve_result_pipeline_state (atomic_parse_store.py)
Called by store_signature_evidence. Writes stage_three_json to
clue_pipeline_state. Does NOT write to wfw_proof_attempts.

Place 2: _mark_current_state_solved (clue_pipeline.py)
Called when a legacy engine (hidden, DD, spoonerism) solves a clue.
Calls artifact_from_solve_result then upsert_pipeline_state directly,
bypassing upsert_solve_result_pipeline_state entirely. Overwrites
clue_pipeline_state with status="solved" and confidence=100. Does NOT
write to wfw_proof_attempts.

The wfw_proof_attempts write currently lives separately in
store_signature_evidence (sig_adapter.py lines 965-978), inside a
try/except Exception: pass block that silently swallows any failure.
Because it is separate, a silent failure leaves clue_pipeline_state
written but wfw_proof_attempts empty — which is the state observed for
clue_id 10068566 (NEUTRAL, DM 17884).

The fix: move the write_wfw_proof_attempt call into
upsert_solve_result_pipeline_state, and make _mark_current_state_solved
use upsert_solve_result_pipeline_state instead of calling
artifact_from_solve_result + upsert_pipeline_state directly. After the
fix, every write to clue_pipeline_state goes through one function, and
that function always also writes wfw_proof_attempts when stage_three is
present.


---

## Change 1 — signature_solver/atomic_parse_store.py

Function: upsert_solve_result_pipeline_state (currently lines 231-237)

Current code:

    def upsert_solve_result_pipeline_state(
            clue_id, source, puzzle_number, clue_text, answer, solve_result,
            conn=None, solver_version=SCHEMA_VERSION):
        """Serialise a SolveResult into the current clue pipeline state."""
        artifact = artifact_from_solve_result(solve_result, solver_version)
        return upsert_pipeline_state(
            clue_id, source, puzzle_number, clue_text, answer, artifact, conn)

Replace with:

    def upsert_solve_result_pipeline_state(
            clue_id, source, puzzle_number, clue_text, answer, solve_result,
            conn=None, solver_version=SCHEMA_VERSION,
            status_override=None, confidence_override=None):
        """Serialise a SolveResult into the current clue pipeline state.

        status_override and confidence_override, when provided, take precedence
        over the values computed by artifact_from_solve_result. Used by
        _mark_current_state_solved to record legacy-engine solves as
        status='solved', confidence=100 while still persisting the stage_three
        evidence produced by the signature solver.

        Also writes to wfw_proof_attempts whenever stage_three_proof is present,
        so the two tables always stay in sync.
        """
        artifact = artifact_from_solve_result(solve_result, solver_version)
        if status_override is not None:
            artifact["status"] = status_override
        if confidence_override is not None:
            artifact["confidence"] = confidence_override
        upsert_pipeline_state(
            clue_id, source, puzzle_number, clue_text, answer, artifact, conn)
        stage_three = (
            getattr(solve_result, "stage_three_proof", None)
            if solve_result is not None else None
        )
        if stage_three is not None:
            from signature_solver.wfw_proof_store import write_wfw_proof_attempt
            proof_dict = stage_three.as_dict()
            proof_dict["status"] = (
                "wfw_proven" if stage_three.status == "PASS" else "wfw_review"
            )
            proof_dict["source"] = "stage_three_pipeline"
            write_wfw_proof_attempt(
                clue_id, source, puzzle_number, proof_dict, conn=conn)
        return clue_id

Key points:
- status_override and confidence_override are keyword-only with default None
  so all existing callers continue to work without change.
- The return is separated from upsert_pipeline_state so work can follow it.
- stage_three is read from solve_result directly (not from the serialised
  artifact) so stage_three.status is the native "PASS"/"REVIEW" attribute and
  stage_three.as_dict() produces schema "stage_three_proof:v1" with status
  still "PASS"/"REVIEW". proof_dict["status"] is then translated to
  "wfw_proven"/"wfw_review" before storage.
- No try/except. Silent failures are what caused the gap in the first place.
- The import is local to avoid circular-import risk at module load time.


---

## Change 2 — sonnet_pipeline/clue_pipeline.py

Function: _mark_current_state_solved (currently lines 473-488)

Current code:

    def _mark_current_state_solved(
            conn, clue_id, source, puzzle_number, clue_text, answer_clean, sr,
            solver_version):
        if sr is None:
            return
        from signature_solver.atomic_parse_store import (
            artifact_from_solve_result,
            upsert_pipeline_state,
        )
        artifact = artifact_from_solve_result(sr, solver_version)
        artifact["status"] = "solved"
        artifact["confidence"] = 100
        artifact["solver_version"] = solver_version
        upsert_pipeline_state(
            clue_id, source, puzzle_number, clue_text, answer_clean, artifact,
            conn=conn)

Replace with:

    def _mark_current_state_solved(
            conn, clue_id, source, puzzle_number, clue_text, answer_clean, sr,
            solver_version):
        if sr is None:
            return
        from signature_solver.atomic_parse_store import (
            upsert_solve_result_pipeline_state,
        )
        upsert_solve_result_pipeline_state(
            clue_id, source, puzzle_number, clue_text, answer_clean, sr,
            conn=conn, solver_version=solver_version,
            status_override="solved", confidence_override=100)

Key points:
- The import changes from artifact_from_solve_result + upsert_pipeline_state
  to upsert_solve_result_pipeline_state only.
- status_override="solved" and confidence_override=100 replicate the previous
  explicit overrides exactly.
- solver_version is passed as a keyword argument and is still applied
  (artifact_from_solve_result uses it internally in upsert_solve_result_pipeline_state).
- sr is the signature SolveResult which carries stage_three_proof from the
  grammar-only path. upsert_solve_result_pipeline_state will now also write
  that stage_three to wfw_proof_attempts.


---

## Change 3 — sonnet_pipeline/sig_adapter.py

Function: store_signature_evidence (lines 943-985)

Remove the wfw_proof_attempts block (currently lines 965-978) in full:

    wfw_attempt_id = None
    stage_three = getattr(sr, "stage_three_proof", None) if sr else None
    if stage_three is not None:
        from signature_solver.wfw_proof_store import write_wfw_proof_attempt
        proof_dict = stage_three.as_dict()
        proof_dict["status"] = (
            "wfw_proven" if stage_three.status == "PASS" else "wfw_review"
        )
        proof_dict["source"] = "stage_three_pipeline"
        try:
            wfw_attempt_id = write_wfw_proof_attempt(
                clue_id, source, puzzle_number, proof_dict, conn=conn)
        except Exception:
            pass

Replace the entire block with a single line:

    wfw_attempt_id = None

The write is now handled by upsert_solve_result_pipeline_state, which is
called from this function two lines earlier. The return dict is unchanged:

    return {
        "stage_context_id": context_id,
        "atomic_artifact_id": artifact_id,
        "pipeline_state_clue_id": clue_id,
        "wfw_attempt_id": wfw_attempt_id,
    }


---

## What to verify before writing

Read all three files in full before making any change. Confirm:

1. atomic_parse_store.py: upsert_solve_result_pipeline_state currently has no
   status_override or confidence_override parameters and no write_wfw_proof_attempt
   call. The function body is exactly two statements.

2. atomic_parse_store.py: artifact_from_solve_result does NOT translate
   stage_three.status from "PASS"/"REVIEW" to "wfw_proven"/"wfw_review".
   The stage_three_proof dict in the artifact still has status "PASS" or "REVIEW".
   Confirm this by reading the function body.

3. clue_pipeline.py: _mark_current_state_solved currently imports
   artifact_from_solve_result and upsert_pipeline_state (not
   upsert_solve_result_pipeline_state). Confirm the import list before replacing.

4. sig_adapter.py: the wfw_proof_attempts block is wrapped in
   try/except Exception: pass. The entire block from "wfw_attempt_id = None"
   through "pass" is what gets replaced with a single "wfw_attempt_id = None".

5. Search for all callers of upsert_pipeline_state in production code (not
   tests). Confirm the only production callers are upsert_solve_result_pipeline_state
   (internal, in atomic_parse_store.py) and _mark_current_state_solved
   (clue_pipeline.py). After this change, _mark_current_state_solved no longer
   calls upsert_pipeline_state directly, making upsert_pipeline_state an
   internal-only function.


---

## What not to do

Do not modify upsert_pipeline_state.
Do not modify artifact_from_solve_result.
Do not modify write_wfw_proof_attempt or wfw_proof_store.py.
Do not modify stage_three_proof.py.
Do not modify run.py.
Do not modify any template or web route.
Do not modify any other file.


---

## After writing

Paste all three modified functions in full:
- upsert_solve_result_pipeline_state from atomic_parse_store.py
- _mark_current_state_solved from clue_pipeline.py
- store_signature_evidence from sig_adapter.py

Claude will audit all three before anything is run.
