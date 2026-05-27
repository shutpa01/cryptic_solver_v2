# Phase 2 Safe Rerun Delete — Codex Instruction

## Task

Remove the pre-emptive DELETE FROM wfw_proof_attempts from _rerun_clue_inner's
upfront clear block. Old WFW proof rows must not be deleted before a replacement
is confirmed. The wfw_proof_attempts table uses latest-row-wins semantics: after
a successful rebuild the new row supersedes old ones by timestamp; after a failed
rebuild (caught by the Phase 0 exception handler) the failure row written by the
previous instruction supersedes old ones. Pre-emptive deletion is unnecessary and
leaves the clue with no WFW authority record when the rebuild fails.

One file changes: web/routes/admin.py.
No other file changes.

Do not change any other part of _rerun_clue_inner. Do not change clue_pipeline.py,
atomic_parse_store.py, stage_three_proof.py, or any other file.


---

## Background

The current upfront clear block (lines 880-894) runs three statements and commits:

  UPDATE clues SET definition = NULL, wordplay_type = NULL,
                   ai_explanation = NULL, reviewed = NULL
  DELETE FROM structured_explanations WHERE clue_id = ?
  DELETE FROM wfw_proof_attempts WHERE clue_id = ?
  COMMIT

The commit is unconditional. If the subsequent rebuild (Phase 0) raises an
exception, the clue is left with:

  wfw_proof_attempts: empty
  clue_pipeline_state: no new row
  clue.definition / ai_explanation: NULL

The previous instruction (PHASE_2_RERUN_FAILURE_PROOF_ROW) addresses the
exception path by writing a failure row. But that failure row is only useful
if there is something to supersede. If old rows were already deleted and the
failure row write also fails (inner except swallows it), the clue has no WFW
record at all.

Removing the DELETE FROM wfw_proof_attempts means:

  On successful rebuild: new rows written by upsert_solve_result_pipeline_state
    and _write_manual_role_stage_three_for_clue are the latest and win.
  On failed rebuild: old rows survive. The failure row from the exception handler
    is then the latest and wins. If the failure row write also fails, the old rows
    still provide historical evidence rather than nothing.

For rerun/rebuild, wfw_proof_attempts should be treated as latest-row-wins
history. Old rows for a clue do not interfere with display; the clue page reads
the latest row via get_latest_wfw_proof_attempt (which orders by created_at DESC,
id DESC). Note: some puzzle-list queries use MAX(id) rather than the same
ordering, but in all cases the intent is "most recently inserted row wins."

The UPDATE clues and DELETE FROM structured_explanations are unchanged. Those
tables do not have latest-row-wins semantics and must be cleared before rebuild.


---

## Change: remove the wfw_proof_attempts DELETE

Location: _rerun_clue_inner, lines 890-893 in the current file.

Current clear block (lines 880-894):

    # Clear previous results
    db.execute(
        "UPDATE clues SET definition = NULL, wordplay_type = NULL, "
        "ai_explanation = NULL, reviewed = NULL WHERE id = ?",
        (clue_id,),
    )
    db.execute(
        "DELETE FROM structured_explanations WHERE clue_id = ?",
        (clue_id,),
    )
    db.execute(
        "DELETE FROM wfw_proof_attempts WHERE clue_id = ?",
        (clue_id,),
    )
    db.commit()

Replace with:

    # Clear previous results
    db.execute(
        "UPDATE clues SET definition = NULL, wordplay_type = NULL, "
        "ai_explanation = NULL, reviewed = NULL WHERE id = ?",
        (clue_id,),
    )
    db.execute(
        "DELETE FROM structured_explanations WHERE clue_id = ?",
        (clue_id,),
    )
    db.commit()

The only removal is the three-line db.execute block for wfw_proof_attempts
(lines 890-893). The comment, the two remaining db.execute calls, and the
db.commit() are all unchanged.


---

## What not to change

Do not remove or modify the UPDATE clues statement.
Do not remove or modify the DELETE FROM structured_explanations statement.
Do not remove or modify the db.commit() call.
Do not change the print statement on line 895.
Do not change any other part of _rerun_clue_inner.
Do not change the Phase 0 exception handler (modified by a separate instruction).


---

## Verification

### Check 1 — syntax only

    C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe
        -m py_compile web\routes\admin.py

Expected: no output, exit 0.


### Check 2 — confirm the changed block

Read lines 879-896 of web/routes/admin.py from the file and paste them.
Confirm:

  a. The comment "# Clear previous results" is present.
  b. The UPDATE clues db.execute call is present and unchanged.
  c. The DELETE FROM structured_explanations db.execute call is present
     and unchanged.
  d. There is no db.execute call referencing wfw_proof_attempts.
  e. db.commit() is present immediately after the structured_explanations
     delete.
  f. The print statement is present on the line after db.commit().


---

## After writing

Paste:
  1. Lines 879-896 of the changed web/routes/admin.py.

Then run Check 1 and paste the output.
