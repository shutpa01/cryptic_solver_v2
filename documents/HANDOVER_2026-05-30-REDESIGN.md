# Handover — 2026-05-30 — Pipeline Redesign

## State of the redesign log

`documents/REDESIGN_LOG.md` is in a messy state. It has three tables (current sequence,
a tracking table, a redesigned order) that overlap and confuse each other. The document
needs to be simplified at the start of the next session before any new entries are written.

---

## What is agreed and correct

### The current sequence — what the code does today (verified from run.py)

1. Phase 0a: Hidden word
2. Phase 0b: Spoonerism
3. Phase 0c: Double definition
4. Phase 0.5: Find definition (DB lookup + Haiku fallback — currently unnamed in code)
5. Phase 0.5: Anagram
6. Phase 0.5: Container
7. Phase 0.5: Deletion
8. Phase 0.5: Charade
9. Phase 0.5: Reversal
10. Phase 0.5: Acrostic
11. Phase 0.5: Homophone
12. Phase 1: Signature solver (grammar triage runs inside)
13. Phase 1.5: TFTT (Times only)
14. Phase 1.5b: fifteensquared (Guardian/Independent only)
15. Phase 2: Tier 2 solver (grammar triage runs inside)
16. Phase 3: Enrichment + re-run

### The redesigned sequence — what the code should do

1. Pre-processing: Atomisation (new code)
2. Pre-processing: Definition engine (existing code, moved from inside Phase 0.5)
3. Pre-processing: Grammar triage (existing code, moved from inside Phase 1 + 2)
4. Phase 0a: Hidden word
5. Phase 0b: Spoonerism
6. Phase 0c: Double definition
7. Phase 0.5: Anagram
8. Phase 0.5: Container
9. Phase 0.5: Deletion
10. Phase 0.5: Reversal
11. Phase 0.5: Acrostic
12. Phase 0.5: Homophone
13. Phase 0.5: Charade (moved to last — most processing-intensive)
14. Phase 1: Signature solver
15. Phase 1.5: TFTT (Times only)
16. Phase 1.5b: fifteensquared (Guardian/Independent only)
17. Phase 2: Tier 2 solver
18. Phase 3: Enrichment + re-run

### Key design decisions already agreed

- **Evidence preservation** is a foundational requirement. Every phase must record what
  it found, whether it solved or not. No phase is exempt.
- **Atomisation** runs first. Every character gets a numerical reference. Proximity and
  contiguity checks become trivial index comparisons.
- **Definition engine** runs as pre-processing, not inside Phase 0.5. Output: definition
  atoms tagged, wordplay zone clear.
- **Grammar triage** runs as pre-processing, not embedded in Phase 1/2. Produces a
  shortlist of licensed mechanisms. Only phases on the shortlist run.
- **Charade runs last** among Phase 0.5 solvers. It is the most processing-intensive and
  should only run after all single-operation solvers have failed.
- **Charade is a sub-project.** Do not patch the existing code. Redesign from scratch.

---

## Design log entries written so far

These entries have content in REDESIGN_LOG.md and do not need to be rewritten:

| Phase | Notes |
|-------|-------|
| Pre-processing: Atomisation | New code |
| Pre-processing: Definition engine | Existing, reorganised |
| Pre-processing: Grammar triage | Existing, reorganised |
| Phase 0b: Spoonerism | Written |
| Phase 0c: Double definition | Written |
| Phase 0.5: Anagram | Written |
| Phase 0.5: Deletion | Written |
| Phase 0.5: Charade | Written — sub-project |
| Phase 0.5: Reversal | Written |

## Design log entries still to write

In the order they appear in the current sequence:

1. Phase 0a: Hidden word
2. Phase 0.5: Find definition
3. Phase 0.5: Container
4. Phase 0.5: Acrostic
5. Phase 0.5: Homophone
6. Phase 1: Signature solver
7. Phase 1.5: TFTT
8. Phase 1.5b: fifteensquared
9. Phase 2: Tier 2 solver
10. Phase 3: Enrichment + re-run

---

## What to do at the start of the next session

1. Simplify REDESIGN_LOG.md — replace the three messy tables with exactly two:
   one showing the current sequence, one showing the redesigned sequence.
   No entry numbers, no Done column, no tracking cruft. Keep the written entries as-is.

2. Then resume the walk-through in order, starting with Phase 0a: Hidden word.

---

## Ground rules (do not deviate)

- Walk through phases in the exact order they run in the code today.
- Read the code before saying anything about what it does.
- Do not look at the archive unless there is a specific reason to compare.
- Write up each phase in the log before moving to the next one.
- Do not state anything about the code without verifying it first.
