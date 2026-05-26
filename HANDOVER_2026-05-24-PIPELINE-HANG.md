# Handover — Pipeline Hang Fix — 2026-05-24 (second session)

## Working model for this thread

Claude instructs and audits. Codex writes all code. Claude never touches files.
No inline backtick formatting. No syntax-highlighted code fences. Both render blue.
The user cannot read blue text. Use plain text and plain ``` blocks only.

---

## What was fixed this session (audited and confirmed)

### Phase 1A — web/templates/clue.html line 123

```
{% set show_wfw = clue.wfw_proven and clue.atomic_wfw %}
```

Was: set show_wfw = clue.atomic_wfw (missing wfw_proven guard). Fixed. Audited correct.

### Phase 1B — web/models.py compute_hint_tier (around lines 472-484)

Codex had hardcoded tier=HIGH for all solved clues. Correct three-tier logic restored:
HIGH >= 70, MEDIUM >= 40, LOW < 40. get_puzzle_clues and get_clue_by_id now LEFT JOIN
wfw_proof_attempts to populate clue.wfw_proven. Audited correct.

### Phase 1C — web/routes/clue.py lines 1164-1169

coverage_warning call restored (Codex had hardcoded False). Audited correct.

### Phase 2 — sonnet_pipeline/sig_adapter.py store_signature_evidence

Stage Three results are now written to wfw_proof_attempts during pipeline runs.
PASS maps to wfw_proven, anything else maps to wfw_review. Audited correct.

---

## The pipeline hang — root cause found, fix NOT yet given to Codex

### Symptom

Running a puzzle from the dashboard hung for over one hour. The dashboard uses
subprocess.run blocking call to run sonnet_pipeline.run --mode 1 --no-review.

### Root cause (confirmed from reading clue_pipeline.py lines 306-337)

Phase 1 in sonnet_pipeline/run.py (lines 616-619) calls run_clue_pipeline.
run_clue_pipeline calls _try_legacy_engines for every clue.
_try_legacy_engines calls _try_v1_engine last.
_try_v1_engine (clue_pipeline.py lines 329-337) makes a Haiku API call
via haiku_definition.find_definition whenever the mechanical definition lookup fails.

That is an API call running inside what the comment labels "zero API cost Phase 1",
for every clue that the mechanical lookup cannot resolve. That is the hang.

Before Codex's revamp, Phase 1 called sig_solve_clue directly — fast, zero API.

### The fix to give Codex

File: sonnet_pipeline/run.py

Step 1 — Find the import near the top of the file:

```
from .clue_pipeline import run_clue_pipeline
```

Change to:

```
from .clue_pipeline import run_clue_pipeline, run_signature_clue_pipeline
```

Step 2 — In Phase 1 (around line 616), replace the run_clue_pipeline call with:

```
pipeline_result = run_signature_clue_pipeline(
    conn, cid, source, puzzle, clue, answer,
    ref_db, write_db=write_db, store_solution=True)
sr = pipeline_result.solve_result
```

The function signature is: conn, clue_id, source, puzzle_number, clue_text, answer,
ref_db, plus keyword args write_db and store_solution. The loop variables are:
cid for clue_id, puzzle for puzzle_number, clue for clue_text.

Step 3 — After that call, the existing code branches on pipeline_result.tier for
"Hidden", "DD", and "Mechanical". run_signature_clue_pipeline only ever returns
tier "Signature" or None. Remove the three dead branches for Hidden, DD, Mechanical.
Keep only the Signature branch.

Step 4 — Find:

```
elif sr.solved:
```

Change to:

```
elif sr is not None and sr.solved:
```

Do NOT touch clue_pipeline.py.

After Codex completes: paste lines 610 to 660 of run.py for audit before running anything.

---

## Secondary concern — None-guard on sr references

After the fix above, lines 656, 666, 681 in run.py also reference sr attributes
(sr.high_confidence, sr.dbe_haiku_candidates, sr.suggested_indicators) without
a None check. Read those lines in context after the fix and add None guards if needed.

---

## Phases still to do (in order, not yet started)

Phase 3 — Make Stage Three run reliably for every clue.
  Stage Three currently returns status=None for most clues. Ensure build_stage_three_proof
  is called for every clue with a Stage Two casefile. Read stage_three_proof.py and
  sig_adapter.py before writing the Codex instruction.

Phase 4 — Remove _conditional_suburb_enrichments hard-code from stage_two_casefile.py.
  Prototype patch that needs to be removed.

Phase 5 — Wire wfw_proof_attempts into compute_hint_tier in web/models.py.
  A wfw_proven clue should be HIGH regardless of confidence score. Not yet done.

Phase 6 — Clean up dead display adapter paths in wfw_display_adapter.py.
  Dead paths left after Phase 1A fix. Read first, list, then instruct Codex.

---

## Key files

```
sonnet_pipeline/run.py                 — hang fix needed here (Phase 1, ~line 616)
sonnet_pipeline/clue_pipeline.py       — run_signature_clue_pipeline is the fast path
                                         _try_v1_engine at line 306 has the Haiku call
sonnet_pipeline/sig_adapter.py         — Phase 2 fix landed here
signature_solver/stage_three_proof.py  — StageThreeProof, build_stage_three_proof
signature_solver/wfw_proof_store.py    — write_wfw_proof_attempt
web/models.py                          — compute_hint_tier
web/routes/clue.py                     — per-clue rerun route
web/templates/clue.html                — show_wfw gate (fixed this session)
```

---

## State before starting next task

wfw_proof_attempts had 215 rows before this session's clean run.
The hang prevented a clean run from completing.
After fixing the hang, run the pipeline on a test puzzle and confirm row count increases.

Design document (describes the project design — read this first):

```
C:\Users\shute\PycharmProjects\cryptic_solver_V2\HANDOVER_STAGE_TWO_2026-05-23.md
```

Previous session recovery handover (what went wrong before this session, what was reverted):

```
C:\Users\shute\PycharmProjects\cryptic_solver_V2\HANDOVER_STAGE_THREE_RECOVERY_2026-05-24.md
```
