# Phase 2 Display Fix — Codex Instruction

## Task

Add one missing schema branch to display_from_wfw_proof_attempt in
signature_solver/wfw_display_adapter.py.

Do not modify any other file. Do not modify any other function in this file.


---

## Background

store_signature_evidence (sonnet_pipeline/sig_adapter.py lines 966-978)
writes stage_three proof dicts to wfw_proof_attempts. Before writing, it
translates the native stage_three status:

    proof_dict["status"] = (
        "wfw_proven" if stage_three.status == "PASS" else "wfw_review"
    )

So proof_json stored in the database has proof["status"] = "wfw_proven"
or "wfw_review", never "PASS" or "REVIEW".

The proof dict also has schema "stage_three_proof:v1" (set by
StageThreeProof.as_dict() at stage_three_proof.py line 50).

wfw_proof_store._row_to_attempt returns:

    {
        "id": ...,
        "status": row[4],          # "wfw_proven" or "wfw_review"
        "proof_source": row[5],
        "proof": _load(row[6]),    # deserialized proof_json
        ...
    }

So attempt_row["status"] is the authoritative translated status.
attempt_row["proof"]["status"] holds the same translated value.

The clue page calls display_from_wfw_proof_attempt(attempt_row).

display_from_wfw_proof_attempt (wfw_display_adapter.py lines 10-68)
currently dispatches on two schemas:

    if proof.get("schema") == "wfw_manual_correction:v1":
        return _display_from_manual_correction(attempt_row, proof)
    if proof.get("schema") == "wfw_unified_proof:v1":
        return _display_from_unified_proof(attempt_row, proof)

There is no case for "stage_three_proof:v1". The stage_three dict falls
through to the old atom-context logic (which reads proof_attempt =
proof.get("proof_attempt"), absent in stage_three dicts) and produces
a broken empty display.

display_from_stage_three_proof (wfw_display_adapter.py lines 71-150)
already exists in the same file and renders "stage_three_proof:v1" dicts
into the clue-page display contract. However, at line 87 it derives
status as:

    status = "wfw_proven" if proof.get("status") == "PASS" else "wfw_review"

This was designed for live (in-memory) use where proof["status"] is still
"PASS" or "REVIEW". A stored proof has proof["status"] = "wfw_proven" or
"wfw_review" (already translated). "PASS" never matches, so the function
always computes status = "wfw_review" regardless of the stored value.

If the new branch simply called display_from_stage_three_proof(proof) and
returned the result, a stored PASS proof (status = "wfw_proven") would be
displayed as "wfw_review". That is wrong.

The fix must call display_from_stage_three_proof(proof) for the block
structure, then restore display["status"] from the authoritative source
in attempt_row before returning.


---

## The change

File: signature_solver/wfw_display_adapter.py
Function: display_from_wfw_proof_attempt

After the "wfw_unified_proof:v1" check at line 25 and before the
fallthrough logic at line 27, insert this block:

    if proof.get("schema") == "stage_three_proof:v1":
        display = display_from_stage_three_proof(proof)
        if display is not None:
            display["status"] = (
                attempt_row.get("status")
                or proof.get("status")
                or display.get("status")
            )
        return display

Explanation of the status line:
- attempt_row.get("status") is the authoritative translated status stored
  in the database column ("wfw_proven" or "wfw_review"). Use this first.
- proof.get("status") is the same translated value stored inside proof_json.
  Used as a fallback in case the column is unexpectedly absent.
- display.get("status") is the status computed by display_from_stage_three_proof,
  which will always be "wfw_review" for stored proofs. Used only as a last
  resort to avoid None.

The result after the insertion is:

    if proof.get("schema") == "wfw_manual_correction:v1":
        return _display_from_manual_correction(attempt_row, proof)
    if proof.get("schema") == "wfw_unified_proof:v1":
        return _display_from_unified_proof(attempt_row, proof)
    if proof.get("schema") == "stage_three_proof:v1":
        display = display_from_stage_three_proof(proof)
        if display is not None:
            display["status"] = (
                attempt_row.get("status")
                or proof.get("status")
                or display.get("status")
            )
        return display

    proposals = proof_attempt.get("proposals") or []

No import is needed. display_from_stage_three_proof is defined earlier in
the same file. The argument is proof (the inner dict, already assigned at
line 18), not attempt_row.


---

## What to verify before writing

Read signature_solver/wfw_display_adapter.py in full before making any
change. Confirm:

1. Lines 22-25 contain exactly the two schema checks described above.
2. Line 27 begins "proposals = proof_attempt.get(...)".
3. display_from_stage_three_proof is defined in the same file, its first
   positional parameter is named proof (not attempt_row), and line 87
   reads: status = "wfw_proven" if proof.get("status") == "PASS" else "wfw_review".
4. display_from_stage_three_proof returns None when
   proof.get("schema") != "stage_three_proof:v1", confirming the
   None-guard in the new branch is necessary.


---

## What not to do

Do not modify display_from_stage_three_proof.
Do not modify _display_from_manual_correction.
Do not modify _display_from_unified_proof.
Do not modify web/routes/clue.py.
Do not modify sonnet_pipeline/sig_adapter.py.
Do not modify signature_solver/stage_three_proof.py.
Do not modify signature_solver/wfw_proof_store.py.
Do not modify any other file.


---

## After writing

Paste the modified function display_from_wfw_proof_attempt in full so
Claude can audit it before it is run.
