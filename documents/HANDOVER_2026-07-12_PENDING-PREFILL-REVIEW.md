# HANDOVER 2026-07-12 — pending-prefill review flow live; loose ends listed

Cold-start document for the next session. Read this first, then the memory
index. Plain English throughout; verify against the code before acting.

## 1. Where the repo stands

- Branch `redesign`, working tree CLEAN. Three commits today, **NOT pushed**
  (origin/redesign is at e7a9b72d, the 2026-07-11 handover):
  - **177a9443** — split-enumeration span-join (joined answers, continuation
    stubs, in-order-fodder commit guard) + Sat/Sun scrape JSONs.
  - **b6d772c6** — prefill pending-review flow + on-demand prefill + reloader
    fix + unclaimed-row fix + MANUAL-badge fix + puzzle-page strip cleanup.
  - **dad15d23** — per-clue "Re-run engines" button restored on /hs.
- The commit message of dad15d23 says 2026-07-13; the work was 2026-07-12.

## 2. THE PROCESS as of tonight (this changed today — read carefully)

1. Nightly (Task `CrypticSolver_NightlyRun`, 02:00): scrape → cascade →
   claude diagnosis → claude prefill. **The prefill now COMMITS its readings
   as status='pending', solved_by='prefill' via
   core.prefill_commit.file_pending_prefill — never pass, never frozen, no
   reference-DB writes.** Tonight (2026-07-13, Sunday night) is the FIRST
   nightly on the new prompt — review how it went in the morning.
2. Morning review happens on the WFW CLUE PAGE (one scroll, no page
   swapping): each prefill reading shows an amber "PREFILL — awaiting
   review" label + green **Confirm** button. Confirm = re-validate the saved
   payload through the same gate as a hand commit → FROZEN manual pass +
   reference-DB harvest, in one click (user decision). Wrong reading → the
   Hand-solver link, grid pre-seeded, fix and Commit as before.
3. Prize puzzles (answers arrive when the user solves the grid): puzzle-page
   **"Prefill now"** button → background chain (cascade remainder → claude
   prefill) + a self-polling status box (RUNNING with filed-so-far count →
   FINISHED with a full-clutch review link). CLI equivalent:
   `scripts/run_prefill.py --source X --pnum N` (or `--date` for catch-ups —
   the nightly prompts scope to "today", so a missed day NEEDS the override
   this script provides).
4. Data-gap clues (e.g. double definitions): enrich the DB from /hs, then
   click the restored **"↻ Re-run engines"** button (route /hsrerun; frozen
   manual solves refused). This is the ONLY re-run control in the UI and it
   is deliberate: the user clicks, Claude never re-runs.
5. Clues NO engine claims now get an honest fail row from the cascade
   (solved_by='none') — they used to get NO row and fell out of every work
   list (and forced wiring-building re-solves on clue-page loads).

## 3. What the USER still has open (their actions, not yours)

- **telegraph 3377** (Sunday prize, cascaded 31/31, prefilled): review the
  23 pending prefills on the clue page (Confirm/fix). By hand: 16a LIGHTNING
  STRIKE + 24d ELDER (pure DDs — enrich defs then Re-run, or hand-solve; the
  /hs commit gate cannot represent a DD), 7a IMPASSE (suspected I'M PASSE
  homophone, low confidence), 13a ON THE SPOT (defs ALREADY added by the
  user — just needs the Re-run click → DD pass).
- **telegraph 31289**: 27a DINING ROOM manual solve (cross-ref "3 Down" =
  GROOM — WARNING: committing it as a synonym piece would harvest
  "3 down→GROOM" into synonyms_pairs; use a non-harvested role or prune
  after). 1d BALTIS. 25a BACKSTAGE filed as CD, PENDING — confirm via Mark
  verdict.
- Restart the site server if not done since dad15d23 (reloader is OFF now —
  code changes require a manual restart, template changes do not):
  `.venv\Scripts\python.exe web\run_dev.py` (the V2 venv — the user often
  launches with the AI_Solver venv; it works, but is the wrong interpreter).
- Delete the old scheduled task (needs an ADMIN shell; refused elevation):
  `schtasks /Delete /TN "Cryptic Solver Nightly Pipeline" /F` (old-project
  Merriam-Webster lexicon job, fires 14:15 + on-boot catch-up, wanted GONE).

## 4. Faults found + fixed today (know these, they explain behaviour)

- **The "wfw hangs" fault was the dev auto-reloader** killing the worker
  mid-request whenever a render's lazy imports touched the watched tree
  (died at ~15-25s, respawned silently, browser hung forever). Fixed:
  `use_reloader=False` in web/run_dev.py. Any future "every click hangs":
  check the reloader first. First page load after a restart still takes
  ~15s warm-up — it FINISHES, that's normal for now.
- **Nightly ran twice on 2026-07-11** (01:00 + a 21:48 catch-up): the task
  has "run ASAP after missed start" ON. The catch-up's claude steps failed
  (API ConnectionRefused right after machine wake) — 31289 was caught up by
  hand in-session; nothing pending from it.
- **MANUAL badge regression**: manual/prefill parses store operation
  'manual'; the clue-page renderer now derives the real type from indicator
  notes (core/wfw_render._manual_type_label), mirroring the live site's
  web/wfw_read._manual_label. Keep the two in sync BY HAND.
- **"Cascade now" runs synchronously inside the request** (looks frozen for
  minutes; twice confused the user). Backgrounding it like Prefill-now was
  OFFERED, not approved — a candidate next improvement.
- The session's Claude Code instance ran ~10h and was flagged by Windows
  for memory growth — the user saw it as "losing the connection". Long
  sessions: hand over sooner.

## 5. Rules (unchanged, the ones that bit today)

- Claude NEVER re-runs a clue for score; the user clicks. Prefill files
  PENDING only; the user is the only path to pass.
- A word whose letters land in the answer in original order is a LITERAL,
  never anagram fodder (guard now enforces at commit; prefill rule 5).
- Pure DDs and whole-clue CDs cannot be prefilled or hand-committed from
  the grid — CD button (/hscd) for CDs; enrich+Re-run for DDs.
- Every link into /hs or the clue page carries the WHOLE-PUZZLE clutch.
- Plain writing. Verify with file:line before claiming. Temp-DB copies for
  tests (patch wfw_web.DB + store.DEFAULT_DB + admin_db.CRYPTIC_DB), and
  fingerprint the live DBs before/after.

## 6. Next bigger jobs (in order, from the standing plan)

1. Review the first NEW-flow nightly (morning of 2026-07-13) — did the
   prefill file pending commits cleanly unattended?
2. Phase 7: SEO render (crawler-visible clue pages) — carried over from the
   2026-07-11 handover, still the next phase of the standing plan.
3. Candidates raised today, not approved: background Cascade-now; a "last
   run" result line on the puzzle page; DD support in the hand-solver grid.
