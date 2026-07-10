# Nightly post-publish diagnosis (automated, headless — invoked by scripts/nightly_run.py)

You are running unattended at night. Your ONE job: step 4 of the publish-first
process — diagnose the frozen manual solves the user committed since the last
run. Read these two memory files IN FULL first and follow them exactly:
`C:\Users\shute\.claude\projects\C--Users-shute-PycharmProjects-cryptic-solver-V2\memory\postpub_diagnosis_design.md`
`C:\Users\shute\.claude\projects\C--Users-shute-PycharmProjects-cryptic-solver-V2\memory\publish_first_process.md`
The human reading IS the ground truth; you never guess and never re-litigate it.

## Scope
Frozen manual solves (`wfw_frozen` + `wfw_solve.solved_by='manual'`) whose
solve row was created in the last 2 days and that have not already been
diagnosed (check the report logs and engine_worklist notes before re-processing
a clue). If there are none, write a one-line report saying so and stop.

## Classification (evidence = fresh-solve on SNAPSHOT DB copies, never persist)
- Solves fresh now → **data-only** (the commit taught the DB; nothing to build).
- Fails, shape expressible as a catalog signature → **signature gap**: derive the
  candidate from the stored grid assignments via `_cand_from_assignments`
  (never hand-written), rubric-tier it, and FILE IT PENDING-ONLY
  (tier='pending', origin='postpub_diagnosis') — pending-only can only turn
  FAIL→amber, so filing is safe by construction. PASS-tier proposals are NEVER
  filed — note them for the user instead.
- Fails, mechanism no engine can express → **engine gap**: UPSERT the mechanism
  into `engine_worklist` (ON CONFLICT(mechanism) DO UPDATE counts + test clue ids).

## Hard rules
- Writes allowed: pending-only catalog rows, engine_worklist, the report log.
  NOTHING else — never a pass-tier signature, never engine code, never
  wfw_solve, never the reference DB.
- Back up the catalog table before filing signatures (established pattern).
- All verification solves on temp snapshot copies of the DBs.
- Honesty over score. False HIGHs are unacceptable. If uncertain, classify as
  engine gap and say why.
- Finish by writing a plain-English report to `logs/diagnosis_YYYY-MM-DD.md`:
  counts per class, each signature filed (template id + shape), each worklist
  mechanism touched. Short sentences, bad news first.
