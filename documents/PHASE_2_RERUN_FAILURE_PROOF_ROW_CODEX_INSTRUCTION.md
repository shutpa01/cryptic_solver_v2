# Phase 2 Rerun Failure Proof Row — Codex Instruction

## Task

When _rerun_clue_inner's Phase 0 pipeline run raises an exception, write a
current failure proof row to wfw_proof_attempts before returning the error
div. This ensures the WFW authority invariant: after any rerun attempt, the
latest wfw_proof_attempts row for the clue is current, not a stale row from
a previous session.

One file changes: web/routes/admin.py.
No other file changes.

Do not change sonnet_pipeline/clue_pipeline.py, signature_solver/atomic_parse_store.py,
signature_solver/stage_three_proof.py, or any other file.


---

## Background

_rerun_clue_inner deletes old wfw_proof_attempts rows unconditionally before
running the pipeline (lines 891-894), then commits the deletion. If Phase 0
(run_clue_pipeline) then raises an exception, the clue is left with:

  wfw_proof_attempts: empty (old rows deleted and committed)
  clue_pipeline_state: no new row

The exception handler at lines 947-953 returns an HTML error div but writes
nothing to the database. The clue has no current WFW authority record.

The fix: in the Phase 0 exception handler, before returning the error div,
attempt to write a minimal "wfw_review" proof row to wfw_proof_attempts
recording what failed and why. This row becomes the current authority record
for the clue: it says "a pipeline run was attempted and failed; here is why."

The write attempt is wrapped in its own try/except. If the failure row cannot
be written (because wfw_proof_store is unavailable or the DB is locked), the
original error div is still returned and no second exception propagates.


---

## Background: persistence path when the pipeline succeeds

When run_clue_pipeline completes without exception:

  run_clue_pipeline
    -> run_signature_clue_pipeline (write_db=True)
       -> store_signature_evidence
          -> upsert_solve_result_pipeline_state
             -> writes clue_pipeline_state.stage_two_json
             -> writes clue_pipeline_state.stage_three_json
             -> writes wfw_proof_attempts row (wfw_proven or wfw_review)

This path is correctly wired for all confidence levels, not only
high_confidence. No change to this path is required.

The failure row written by this instruction is only needed when the pipeline
itself raises before reaching that path. It is a fallback record, not a
replacement for the normal persistence.


---

## Change: add failure proof row to Phase 0 exception handler

Location: _rerun_clue_inner (lines ~947-953 in the current file).

The current Phase 0 exception handler:

        except Exception as e:
            import traceback
            print(f"[RERUN WFW] Error: {e}")
            traceback.print_exc()
            return (
                '<div class="mt-2 text-xs text-red-600 bg-red-50 rounded '
                'px-2 py-1">Unified pipeline error: %s</div>' % str(e)
            )

Replace with:

        except Exception as e:
            import traceback
            print(f"[RERUN WFW] Error: {e}")
            traceback.print_exc()
            try:
                from signature_solver.wfw_proof_store import write_wfw_proof_attempt
                write_wfw_proof_attempt(
                    clue_id, source, puzzle_number,
                    {
                        "status": "wfw_review",
                        "source": "stage_three_pipeline_failure",
                        "schema": "stage_three_proof:v1",
                        "clue_text": clue_text,
                        "answer": answer_clean,
                        "error": str(e),
                        "checks": [
                            {
                                "name": "pipeline_failure",
                                "status": "REVIEW",
                                "detail": str(e),
                            }
                        ],
                    },
                    conn=db,
                )
                db.commit()
            except Exception:
                pass
            return (
                '<div class="mt-2 text-xs text-red-600 bg-red-50 rounded '
                'px-2 py-1">Unified pipeline error: %s</div>' % str(e)
            )

The outer try/except that surrounds the failure row write is separate from
the main pipeline exception. It catches any error in writing the failure row
and swallows it silently with pass, so the original error div is always
returned regardless.

The proof record fields:

  status: "wfw_review" — failure rows are never wfw_proven.
  source: "stage_three_pipeline_failure" — distinguishes from normal rows.
  schema: "stage_three_proof:v1" — marks the schema family.
  clue_text, answer: from the outer scope (already set before Phase 0).
  error: the str() representation of the pipeline exception.
  checks: a one-element list containing a single pipeline_failure check.
    The check has name "pipeline_failure", status "REVIEW", and detail
    set to str(e). This gives the Stage Three display adapter a concrete
    named check to render rather than an empty checks list, which could
    produce an oddly blank WFW panel. Other Stage Three fields (blocks,
    word_purposes, atomic_links, etc.) are absent; the adapter must
    tolerate missing fields by treating them as empty collections.

_normalise_proof in wfw_proof_store.py accepts any dict and sets default
status and source if absent, so this dict is sufficient. No StageThreeProof
dataclass is needed.

conn=db is passed to write_wfw_proof_attempt so the write uses the same open
connection. db.commit() after the write commits only the failure row; the
earlier deletion commit (line 894) is already committed.


---

## What not to change

Do not change the deletion block at lines 881-894. That is addressed by a
separate instruction.

Do not change the Phase 1 or Phase 2+ exception handlers in _rerun_clue_inner.
Only the Phase 0 handler (lines 947-953) changes here.

Do not change the normal pipeline path (run_clue_pipeline, store_signature_evidence,
upsert_solve_result_pipeline_state). That path is correctly wired.

Do not change _write_manual_role_stage_three_for_clue or its call at line 963.


---

## Verification

### Check 1 — syntax only

    C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe
        -m py_compile web\routes\admin.py

Expected: no output, exit 0.

Note: admin.py imports Flask; py_compile only checks syntax, not imports,
so the project virtualenv is sufficient for this check.


### Check 2 — confirm the changed block

Read the changed section of web/routes/admin.py (the Phase 0 except block)
and paste it in full. Confirm:

  a. The outer except Exception as e: block is unchanged in signature.
  b. An inner try block imports write_wfw_proof_attempt and calls it with
     the five required fields.
  c. An inner except Exception: pass catches write failures.
  d. The original return statement follows the inner try/except unchanged.


---

## After writing

Paste:
  1. The full changed Phase 0 except block (from "except Exception as e:"
     through the closing "return" line).

Then run Check 1 and paste the output.
Then paste the block read for Check 2 from the actual file.
