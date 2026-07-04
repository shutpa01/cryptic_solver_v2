# Handover — 2026-06-23 — hand-solver reset (read this cold-start)

Branch: `redesign`. This thread ended badly: the hand-solver UI was iterated many times,
the user found it worse with each pass, and trust ran out. A fresh thread should restart
the **hand-solver UI** from a clear spec. The **auto-solver is reliable and committed** and
must be protected. Read also: `force-role-handsolver-direction`,
`dt-31273-worklist-and-caution`, `times-29575-session` (memory), and
`documents/DESIGN_FORCE_ROLE_AND_HANDSOLVER.md`.

---

## 0. THE TONE TO TAKE NEXT TIME (why this thread failed)

1. **Don't build UI before the interaction is nailed.** Each iteration chased the last
   complaint and added complexity. Agree the exact click-by-click flow in words FIRST.
2. **A stale server served the user OLD code for many turns** while I claimed (from curl)
   that it worked — this gaslit the whole conversation. See §4. ALWAYS verify the live
   build before claiming anything.
3. **The user's screen is ground truth.** When the user says "no change" and your test
   says "works", STOP and find the environment mismatch — do not iterate the code.
4. The user's core critique: *"a UI built to serve code, rather than the other way round."*
   Design around how a human fixes a clue, not around the engine's internal mechanisms.

## 1. WHAT IS SOLID AND COMMITTED (do NOT disturb — the reliable auto-solver)

All on `redesign`, each A/B-validated (PYTHONHASHSEED=0 same-session isolation), zero
regressions. Capability engines + fixes built this stretch:
- `43a84384` — 3 engines: selection+reversal charade (TRAIL), located substitution (DRESSES),
  charade+anagram-container (DOGFIGHT).
- `fff1998d` — nested-container engine (VACUUM, FALLENANGEL).
- `5d981615` — charade_named_deletion role fix (PEERGROUP; role of the deleted-letter SOURCE
  word changed indicator->deletion so role_validity stops rejecting named deletions).
- `94562320` — charade+acrostic (DOCTORS) + reversed-outer-container (EMPEROR) engines.
- (earlier `3d24f4a8` — role_validity DB-validity check in ~30 engines + bidirectional synonym
  lookup.)
- DOT fixed by reactivating reversal catalog template 348 (DB change in clues_master.db,
  backed up; not in git — DB is gitignored). OVERPOPULATED via catalog template 1659.

Rule that protects this: **additive-first; never modify a working engine to chase one clue;
A/B every change and prove ZERO losses or it doesn't ship.** Harness: `core/_ab_general.py`
(now reads status from the returned parse — does NOT persist to the store — so it can run
alongside the live server). DB changes live in clues_master.db / cryptic_new.db (gitignored),
backed up as `*.bak-*`.

## 2. WHAT WAS BUILT FOR THE HAND-SOLVER (committed, but the UI is unsatisfactory)

The plumbing is sound; the UI presentation is what failed. Commits `308e22f0`, `4f4fd99b`,
`48e0ec25`, `416b0631`, `fd5183a9`, `1c686631`, `f92e061d`, `4a01b0d7`, `d1d89640`.

GOOD, KEEP (the mechanics work and are tested):
- `core/clue_overrides.py` — `apply_forced_overrides(wiring, clue_id)`: one shared point that
  applies a clue's per-clue overrides (filler / forced definition / forced indicator, incl.
  positional via charade_positional_subtypes) on EVERY solve path. STRICT NO-OP when a clue
  has none. Wired into the page, batch, and `_ab_general`.
- `store.py` — tables `wfw_forced_indicator`, `wfw_frozen`; helpers add/get/clear; FREEZE
  guarantee: `save_parse` refuses to overwrite a frozen clue with a non-PASS and auto-freezes
  a clean PASS that carries an override; `unforce()` clears all overrides+freeze. (Items 1 & 2.)
- The batch apply model in `wfw_web._rolegrid_block` + `/gridapply`: per-word role dropdown
  defaulted to "keep (current role)"; adjacent same-role words auto-form a phrase; synonym
  value reused from a known DB value (no retyping); ONE "Apply & re-solve"; PRG redirect.
  VERIFIED end-to-end on SUNGLASSES 10075119 (some+girls -> "some girls"=LASSES reused ->
  solves SUNG+LASSES). This is the interaction the USER APPROVED.

THE PROBLEM: despite the above working in tests, the user experienced "no change / getting
worse" — largely the stale-server bug (§4). Whether the approved batch grid is actually right
for the user is UNCONFIRMED on their screen. Next thread: get the user to confirm the grid on
a FRESH server (check the build marker) before building anything else. If still wrong, redesign
the interaction with them — do not patch.

OPTION THE USER IS CONSIDERING: reverting the hand-solver commits (308e22f0..d1d89640) to start
the hand-solver UI clean. The auto-solver commits stay regardless. ASK the user; do not revert
unilaterally.

## 3. THE INTERACTION THE USER WANTS (the spec to build to)

- Clue page = clean END-USER view. Hand-solver = the COMPLETE admin workbench; no switching
  back and forth (it must show everything admin needs).
- Each clue word listed with a **role dropdown, default = the role the solver already gave it**.
  Change only what's wrong.
- **Resolve once, at the end** — never re-solve per action.
- **Never retype a value that already exists** — reuse the discovered synonym; to extend it,
  include the adjacent word(s) and the value is reused.
- Per-word dropdown is acceptable. Adjacent same-role words = one phrase.
- Roles: definition, indicator (incl. positional after/before), synonym, link, filler.
- Signature creation (design §6, NOT built): when a grid completes a solve, derive the
  signature, verify it's a catalog gap (re-solve through the real cascade), auto-queue +
  automatic A/B, auto-approve on zero-loss. The clue's freeze is independent and immediate.

## 4. ⚠️ OPERATIONAL TRAPS THAT WASTED THIS THREAD — AVOID

- **Stale server = the #1 time-sink.** Launching the server by backgrounding with `&` inside a
  Bash task that then EXITS orphaned/left an OLD server bound to :5099, so the user hit old
  code for many turns. FIX, ALWAYS: run the server as a TRACKED background process —
  `Bash(run_in_background=true)` running `python -m core.wfw_web` DIRECTLY (no `&`, no wrapper
  loop). Before claiming anything: confirm EXACTLY ONE listener on 5099, and check the visible
  **"hand-solver build HH:MM:SS"** marker on the grid (BOOT_ID in wfw_web, changes each
  restart). If the marker didn't change, you're looking at stale code — fix that first.
- No-cache headers are now sent (`@app.after_request`, commit f92e061d) and all grid actions
  use Post/Redirect/Get (4a01b0d7) — but a page loaded BEFORE those still needs one hard
  refresh.
- A/B and the live server both write the store DB historically; `_ab_general` no longer
  persists, so they can coexist now. Cross-run A/B nondeterminism exists even at
  PYTHONHASHSEED=0 — isolate an engine SAME-SESSION (env-gate it), never vs a stale baseline.
- The STORE (store.load_parse / wfw_filler / forced tags) is the source of truth for "what
  passes", NOT a raw batch solve — a batch that doesn't load overrides under-reports.

## 5. KNOWN REMAINING GAPS (hand-solver)
- "DB has" column is per-single-word, so a multi-word synonym you add doesn't display there
  (the apply still works). Make it phrase-aware for the selected span.
- Signature creation step not built (§3).
- Clue-page tidy-up (move admin into the hand-solver) deliberately NOT done — keep a working
  admin path until the grid is confirmed good.
- Legacy atom hand-solver still at `/handsolve` (parked, relabelled "Atoms"); the role grid is
  `/rolegrid`. Consider retiring /handsolve once the grid is confirmed.

## 6. QUICK COMMANDS
- Server (TRACKED, the only correct way): Bash run_in_background=true -> `.venv/Scripts/python.exe -m core.wfw_web`
  (port 5099). Kill ALL prior listeners first; verify one listener + the build marker.
- Role grid: `http://localhost:5099/rolegrid?id=<clue_id>`. Clue page: `/?id=<clue_id>`.
- Solve one clue in code: `engine_registry.solve_clue_text(clue, ans, wiring, direction=...)`
  with `wiring = engine_registry.db_only(engine_registry.make_db_wiring())`.
- A/B: `python -m core._ab_general logs/out.json 500` then `--compare a b`. Use
  PYTHONHASHSEED=0 and isolate same-session.
- Python is `.venv/Scripts/python.exe` (bare `python` is the Windows stub). Use
  `PYTHONIOENCODING=utf-8` for clue text with curly quotes/arrows.
