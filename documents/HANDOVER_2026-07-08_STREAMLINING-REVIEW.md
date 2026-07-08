# HANDOVER 2026-07-08 — Next job: review a STREAMLINING of the process

**START HERE.** The user's verdict on the current system: "brittle, awkward and convoluted —
held together by string." The pieces now WORK (see §2) but the user experiences too many
surfaces, too many steps, and states that don't explain themselves. The next job is a
REVIEW: propose how to streamline the whole nightly flow. Review first, in plain English,
agreed with the user — do NOT start building a redesign unprompted.

## 1. HOW TO WORK (the rules that got enforced the hard way this week)
1. **Plain English. Short sentences. No metaphors, no performance.** The user ended
   sessions over this. Bad news in the first sentence. (memory: feedback-plain-writing)
2. **Verify before claiming; cite file:line.** Never assert from memory.
3. **"All means all."** When the user says fix them all, no self-granted scope cuts. If
   something genuinely should be excluded, say so explicitly and let them veto.
4. **Claude never re-runs a clue on its own initiative** — user runs the solver (buttons
   exist now). Claude diagnoses, suggests, and runs A/B regressions as gatekeeper.
5. Every engine change is A/B-gated: full fresh-solve regression (recreate
   `scratchpad/_regr.py` — corpus = all passes + pendings + 500-fail sample, AI OFF
   including `suggest_hom`), 0 regressions, inspect every new pass by hand.
6. Browser CACHE and the stale-server trap burned the user repeatedly: after server
   restarts tell the user to hard-refresh (Ctrl+F5); verify what the LIVE server serves
   (Invoke-WebRequest) before claiming a change is visible.
7. Env: `.venv\Scripts\python.exe`, `$env:PYTHONPATH=(Get-Location).Path`, server 5099
   (`python -m core.wfw_web`; restart = stop process matching 'core.wfw_web', Start-Process).

## 2. WHAT NOW WORKS (all UNCOMMITTED on `redesign`, HEAD 7242ea33 + ~30 changed files)
**The user's publish flow (their spec, built + live 2026-07-08):**
- Clue matched by a NEW signature → amber PENDING with a message telling the user to check
  the solve and set Status to PASS → puzzle publishable.
- **/triage page**: per-clue **Re-run** button (seconds, resident wiring); **delete row**
  per clue (spurious synonym/abbr/indicator/definition/link; recoverable via
  `deleted_entries`; executed by Apply); editable Accept/Reject suggestion rows;
  compact Reading/Problem/Fix diagnosis blocks (detail collapsed);
  **"Regression-check pending signatures"** button → background thread,
  `ab_signature.try_promote` per pending template, promotes only on clean diff, progress in
  `documents/triage/sigregress_status.json`, shown on the page.
- **/hs**: selection piece role (rule-derived values incl. multi-word runs), pre-populate
  from triage `hs_seed`, prune links for synonyms/abbreviations/indicator typings
  (`/hstypes`), release-tile, etc.
- **Engines**: the single-word disease is FIXED across the cascade (both indicator side —
  19 engines, and fodder/selection side — 7 more, plus the SUBTYPE_RULE alternation licence
  map). Three A/B rounds, final: 0 regressions. (memory: phrase-indicator-fix-batch1 has
  the full list + the two regression lessons.)
- **31285 scoreboard**: 22 pass · 1 pending · 9 fail (was 14/3/15). The pending is
  TERMINAL — signature template 1670 filed by the user's Resolve (FIRST full signature-loop
  closure); awaiting the user's check-and-PASS. The 9 fails: engine escalations (ORION,
  TERRIFIED, LISTENS, GRAND, STATED, TIGER) + 2 data accepts not yet taken (STRUGGLED def,
  ROOT OUT synonym) + BRACES (needs its /hs Resolve to file the second signature).

## 3. THE STREAMLINING REVIEW — raw material
What the user has actually complained about (their words paraphrased, sessions 07-07/08):
- Too many surfaces: triage page, clue page, hand-solver, CLI commands — a fix's effect
  never appears where they are standing.
- States that don't explain themselves (the amber pending-signature message was
  "meaningless"; now improved, but the pattern is the issue).
- Stale verdicts: pages show stored results; nothing re-solves unless explicitly clicked
  (Re-run button mitigates; the model question remains).
- Multiple hour-long serial A/B runs when one combined run would do (fixed once by
  combining; make it the norm).
- The diagnosis text was verbose (fixed: Reading/Problem/Fix); keep that standard.
Open design questions for the review (do NOT pre-decide; bring options):
- One consolidated review surface? (triage + clue page + HS bridge)
- Auto re-solve on page load for listed clues? (cost: seconds per clue — measure first)
- The "Run / refresh diagnosis" button still calls the DEAD parallel classifier
  (`triage_classify.py`) and would overwrite Claude's classified_<pnum>.json — retire or
  repoint it as part of the streamline.
- Where nightly Claude-diagnosis fits: currently a session writes
  `documents/triage/{classified,diagnoses}_<pnum>.json` (+ hs_seed). Cadence/trigger TBD.
- The hardcoded-lists→DB migration (other thread) overlaps; SUBTYPE_RULE etc. still belong
  in the DB long-term.

## 4. LOOSE ENDS (small, carry over)
- User to: check TERMINAL and set PASS; Resolve BRACES in /hs; take/refuse the two data
  accepts; click the sig-regression button when a day's puzzles are done.
- A REAL delete via the new facility not yet exercised.
- `git` checkpoint STRONGLY overdue: ~30 files, all regression-validated. Suggest commit
  as the first act of the new thread (user approves).
- Memory files: phrase-indicator-fix-batch1, delete-facility, triage-pilot-31285,
  hs-selection-role-and-seeding, feedback-plain-writing — all current as of this handover.
