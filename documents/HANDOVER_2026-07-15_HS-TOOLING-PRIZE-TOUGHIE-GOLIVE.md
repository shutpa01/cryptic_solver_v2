# HANDOVER 2026-07-15 — /hs tooling, homophones, punctuation, Prize Toughie + GO-LIVE decisions

Cold-start document for the next session. Read this first, then the memory index.
Plain English; verify against the code before acting. Launch target **Friday
2026-07-17**. The user's stated next step: **discuss the go-live decisions in §3.**

## 1. WHERE THE REPO STANDS — read this before anything

- Branch `redesign`. **EVERYTHING built this session is UNCOMMITTED** — it lives in
  the working tree only. Nothing was committed (the user hasn't asked). If the
  session is lost, the work is still on disk but not in git. Modified/untracked:
  - `core/wfw_web.py` — homophone-add route, manual homophone role, double-definition
    fix, punctuation-selectable grid (§2).
  - `web/serving.py` — INVALID-served-as-pass card + SERVED_BROWSE prize-toughie.
  - `web/models.py` — Prize Toughie type wiring.
  - `web/routes/seo.py` — sitemap counts invalid-with-comment.
  - `web/templates/puzzle.html` — admin header fixes.
  - `scripts/ingest_prize_toughie.py` — NEW (untracked).
  - Also modified, NOT from this session, pre-existing uncommitted: `scripts/nightly_run.py`,
    `scripts/run_prefill.py` (the 07-14 billing fix was committed as 04ab93b8 — investigate
    why these show modified again before committing; may be stray edits).
  - `.claude/settings.local.json` (deliberately uncommitted). Scraper JSONs (nightly, normal).
- **RESTART REQUIRED.** The reloader is OFF. None of this session's changes are live
  until the dev server is restarted (`.venv\Scripts\python.exe web\run_dev.py`, the V2
  venv). The user's first "it failed" this session was because the server hadn't been
  restarted — confirm a restart before testing.
- 163 commits ahead of origin/master; unpushed. Push needs explicit user approval.

## 2. WHAT WAS BUILT THIS SESSION (all user-driven, all verified through the real path)

1. **Puzzle-page admin header fixed** (`puzzle.html`). Buttons no longer run off the
   page (flex-wrap, dropped shrink-0); the Cascade/Prefill "working" spinner is now
   visible (it used to render off-screen with the overflowing buttons); the result box
   moved out of the `justify-between` header so the finish message is legible.
2. **INVALID served as pass** (`web/serving.py` `_invalid_card`, `seo.py` sitemap). A
   clue the reviewer marks INVALID **with a comment** now renders a public page (answer +
   "doesn't work by standard cryptic rules" banner + the comment) instead of 410; no
   comment ⇒ still 410. Verified end-to-end on telegraph 31291 1a TAP-DANCING (200).
   Memory: invalid_served_as_pass. NB the completeness rule: "served" = pass OR
   invalid-with-comment — count both when judging launch readiness.
3. **Homophone-add restored** (`/hsaddhomophone` route + box, `wfw_web.py`). Was a
   regression: the old "Add to reference DB" panel is dead code (`_clue_admin_panel`,
   called nowhere); homophone + literal add had no path. Homophone restored; **literal
   still not** (open). Memory: homophone_add_restored.
4. **Manual homophone role in /hs** (`wfw_web.py`). Tick fodder → click answer tiles →
   attach; value from tiles; allowed only when a homophone INDICATOR is tagged (no
   auto sound-check — the human owns the verdict). Solves clues the charade_homophone
   engine can't (it has only 8 single-word-slot templates — memory:
   charade_homophone_singleword_limit). Memory: homophone_role_built.
5. **Double-definition overwrite fixed** (`wfw_web.py`). Tagging two definitions (e.g.
   "perhaps" as definition-by-example) silently overwrote the first and then blamed the
   DROPPED words for having "no role". Now a clear error: "Two definitions tagged: X and
   Y…". DD unaffected (its 2nd def is a synonym piece, not a 2nd definition tag).
6. **Punctuation/symbol atoms selectable in /hs** (`wfw_web.py`). Symbols (…, ?, commas)
   now appear as SELECTABLE grid rows (greyed, appended after words so word indices don't
   shift), so a clue whose definition IS punctuation can be tagged (… = ELLIPSIS). Never
   required (`unexplained_words` skips non-word tokens). Memory: hs_punctuation_selectable.
   NOTE: the ELLIPSIS "…" was NEVER corrupted — it is U+2026 and renders fine; the "�" was
   only the terminal. No data fix was needed.
7. **Telegraph PRIZE TOUGHIE added** (§4). New public type; loaded last week's No 233.

All of 2-7 are verified via the real functions/routes, but only #7's pages and the header
have been browser-loaded; the /hs interactions (homophone role, punctuation tagging) are
verified through `_build_manual_parse` + page render, NOT yet a live tick/click/commit.
Browser-drive those after restart.

## 3. ★ GO-LIVE DECISIONS TO MAKE (the discussion the user wants next)

Launch is Fri 2026-07-17. These are the open calls:

1. **Prize-answer embargo policy.** Prize puzzles (Prize Cryptic, Prize Toughie) have
   embargoed answers. Prize Toughie #233 is now loaded with BLANK answers and listed
   publicly (clue pages 410 until solved). The general policy — do we serve prize answers
   during the ~1-week embargo, or hold each prize puzzle until its embargo lifts? This was
   open from 07-13 and now applies to two prize types. DECISION NOT TAKEN.
2. **Flip the site live.** justcordelia.com currently serves intentional 503s
   (maintenance mode — memory: site_maintenance_mode_2026_05_15, restore command in that
   file). Going live = flipping that off. When, and any final checks first?
3. **Internal-links sweep** — approved on 07-13, NOT started. Ensures no internal link
   points to a clue that 410s (the serving rule requires "never link to a 410"). Should
   run before launch.
4. **Styled 410 page** — everything-else answers 410 Gone; needs a styled page (carried
   from 07-13, not built).
5. **Old-puzzle-page scope** — how far back the served archive goes at launch (week-only
   is settled, but the exact cut and resurrection behaviour want a final word).
6. **Everyman 4155-4159 backlog** — cascade + prefill decision (carried from 07-13).
7. **Push the branch** — 163 commits + this session's uncommitted work. Commit + push
   needs the user's explicit yes.
8. **7-day window confirmed** = the 11th onward (user, this session). The 08/09/10 gaps
   (never-cascaded + heavy-fail puzzles) are OUT of scope and can be ignored for launch.

## 4. TELEGRAPH PRIZE TOUGHIE (built this session, memory: telegraph_prize_toughie)

- We DO download it weekly (scraper/telegraph/telegraph_prize-toughie_*.json). Last
  week's = **Prize Toughie No 233** (id 93057, Zandio, Sunday 12 July 2026), 28 clues.
- Added as a **type under `telegraph`** (NOT a new source — a new source would need
  editing 7+ hardcoded `source IN (...)` lists; telegraph is already in all of them).
  Filed source='telegraph', puzzle_number=233; classify maps telegraph **1-2999 → Prize
  Toughie** (no collision). Public via SERVED_BROWSE.
- Verified: /telegraph/prize-toughie/, /telegraph/prize-toughie/233, /puzzles all 200 to
  a public client; only #233 in the 1-2999 range.
- **Follow-ups:** (a) weekly ingest is manual — run
  `python -m scripts.ingest_prize_toughie --commit` each week (grabs the newest file), or
  wire into the nightly (the scraper's own promote-to-clues is broken for prize puzzles —
  it needs an explanation and drops publication_date). (b) Enter #233's answers via the
  admin prize-grid flow, then cascade/solve to make its clue pages serve.

## 5. LAUNCH-WINDOW STATE (07-11 to 07-17)

- **07-11 to 07-14: essentially complete.** All served except ONE genuine stray:
  **times 5224 21a ELLIPSIS** — a reversal charade (SIS+PILL+E reversed) whose definition
  is the trailing "…"; now hand-solvable via the punctuation-selectable grid (§2.6). The
  telegraph 31291 "1 short" flagged earlier was a COUNTING error on my part (the INVALID
  clue serves fine).
- **07-15 (today): fresh, need the walk.** guardian 30059, telegraph 31292, times 29595 —
  cascaded + prefilled to PENDING; walk them (Commit) to make them serve.
- **07-16, 07-17:** publish before/at launch; same daily walk.

## 6. OTHER OPEN ITEMS (non-launch)

- **&lit frozen-pending has no confirm path** (from 07-14 §5.1) — still open.
- **literal-add** lost with homophone-add (§2.3) — restore if wanted (mirror /hsaddhomophone).
- **9 dead /hs helper functions** (old clue-page hand-solver, superseded) + a **/hs render
  smoke test** — the cleanup that would have caught the homophone-add regression. Offered,
  not done.
- **cp1252 logger crash** in scripts/run_prefill.py log() (from 07-14 §5.2) — one-line fix, not approved.
- **'leap"' junk row** in synonyms_pairs (from 07-14 §5.3) — strip punctuation in filed phrases.
- Engine-worklist: cyclic-selection gap, Guardian "See N Across" stub span-join gap
  (memory), charade_homophone single-word-slot limit (§2.4).

## 7. ENVIRONMENT NOTES

- ONE app on :5000 (`web/run_dev.py`, V2 venv `.venv\Scripts\python.exe` — bare `python`
  is a Store stub); /solver/* = mounted wfw admin app; :5099 fallback. Reloader OFF —
  restart after any change; hard-refresh.
- data/clues_master.db = clues + wfw_* tables; data/cryptic_new.db = reference tables.
  Both gitignored, backed up twice daily.
- Headless billing bills the Max subscription (07-14 fix). NEVER state a billing fact you
  can't see; a billing error is a STOP.
- Sacred principle: NEVER modify a working stage engine for an edge case — build/add.
