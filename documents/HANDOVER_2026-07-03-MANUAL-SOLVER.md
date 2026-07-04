# Handover — 2026-07-03 — Manual Solver (word grid) + render parity + substitution role

Cold start for a NEW thread. Read top-to-bottom.

## TONE / WHAT HAPPENED (front-loaded honesty)
This was a long, painful multi-day session that finally delivered a working **Manual Solver**.
It was built WRONG two or three times first — as a left-to-right "concatenate the pieces to the
answer" tool (couldn't express a reversal like MERGE), then bolted onto the atom-letter tool
(`/handsolve`, incomprehensible, no role tagging) — before landing on the correct model the user
had specified all along. Lessons the new thread MUST honour:
- **Read the user's design before building.** The spec is in `memory/manual_solver_spec_from_user.md`
  (verbatim). Do not reinterpret it.
- **Verify through the REAL path and LOOK at the actual output** before claiming done. Repeatedly
  this session I shipped something without checking the render / the browser flow and the user had
  to catch the breakage. That is backwards.
- **The browser extension is NOT connected here**, so the in-browser click path cannot be driven
  from Claude Code. Backend + render were verified via `test_client`; the user tests the clicks.
- **Server restarts via `&` have been flaky** and served stale pages mid-restart — verify ONE
  listener after every restart, and tell the user to HARD-REFRESH (Ctrl+Shift+R; a plain reload
  serves cache).

## CODE STATE
- Branch `redesign`, **HEAD = 486844ba** (NOT pushed). Working tree clean of MY work (only
  pre-existing untracked scraper JSON / old handovers / `core/atomsig/` / a `scraper/telegraph/
  telegraph_daily.py` modification that is NOT mine — leave them).
- This session's commits on `redesign` (none pushed): `398e43b0` (selection-as-source engines +
  container 'raw' + anagram_deletion label), `6cdc16b4` (reversal_charade multi-word indicator fix,
  0-regression A/B), `16bc9edb` (clue-page selection-indicator add mechanism + first Manual Solver
  attempt), `0d6aecba` (SHELVED the fabricating charade_synonym_multi_deletion engine — deleted),
  **`486844ba`** (this session's Manual Solver + render parity + substitution role).

## THE MANUAL SOLVER — what it is and where
A HUMAN authoring recorder for clues the cascade can't/shouldn't solve. It DERIVES nothing,
VERIFIES nothing, never runs the cascade, writes NOTHING to the reference DB. It lives on the
**word grid at `/hs`** (route `span_surface` / JS `initGrid` in `core/wfw_web.py`), reached from the
clue page via the single teal **"☰ Hand-solver"** link. (The old `/rolegrid` "Old grid" and
`/handsolve` "Atoms" buttons were REMOVED from the clue page.)

**Workflow:** tick word(s) → pick a ROLE from the dropdown → Assign; repeat; then **Commit (manual)**.
- Roles: `definition` / `synonym` / `substitution (abbr/symbol)` / `letters` (literal) /
  `indicator` (+ type, + deletion subtype) / `link` / `filler` / `none`.
- **Piece roles** (need answer TILES): synonym, substitution, letters. In the JS, helpers
  `isValued(r)` (types/picks a value) and `isPiece(r)` (lands on tiles) gate the behaviour.
- For a synonym/substitution: type OR pick a value (candidate list from `/hslookup`), then **click
  the answer tiles** it occupies; Assign colours the piece + those tiles. A forward value whose
  letters sit in the answer in reading order **auto-places** (`locateValue`); reversed/split pieces
  (REM in MERGE, TE split in a container) need manual tile clicks — that placement IS how reversal/
  container are expressed.
- For `letters` (literal): click the tiles; value = the answer letters there.
- indicator/definition/link/filler take NO tiles.
- **Commit** builds a FROZEN manual Parse and REPLACES whatever was stored (any wrong pass/partial/
  fail). **Uncommit** clears the freeze and re-solves via the cascade. Commit redirects to the clue
  page carrying the whole CLUTCH (`DATA.back`), not just the one clue.

**Routes (core/wfw_web.py):** `/hsmanualcommit` (build + persist frozen manual parse), `/hsmanualuncommit`
(clear_frozen + `_resolve_one`). `/hssave` still persists the assignment JSON on every Assign.
Assignment JSON shape: `{idx:[word indices], role, value, pos:[1-based answer tiles], itype, isub}`.
Commit sets each piece's mechanism: letters->`raw`, substitution->`abbreviation`, synonym->`synonym`;
`operation="manual"`, `solved_by="manual"`, `status="pass"`, frozen. Requires EVERY answer tile
covered by exactly one piece or it refuses (loud red message).

**GUARD:** `_resolve_one` skips a clue whose stored parse is `solved_by=="manual"` AND frozen, so the
cascade can never overwrite a committed manual solve (uncommit lifts the freeze first).

## RENDER PARITY (the user's key principle: manual == engine experience)
`core/wfw_render.py`: `operation="manual"` is registered on the engines' OWN `_render_assembly`
(added `"manual"` to its `@renders(...)`), and the parallel manual renderer was DELETED. `_assembly_expr`
builds the whole expression (charade `A + B`, container `OUTER around (INNER)`, nesting) from the
parse LINKS alone; `_piece_label` marks each piece reversed / −deleted / anagram from its VALUE vs
its answer positions — no engine-only fields. So a manual parse renders byte-identical to an engine
compound solve. Verified: RAMBLES `RS around (AMBLE)`, MERGE `REM reversed + EG reversed`, NOEL
`N + OEL`. `_MECH_LABEL["abbreviation"] = "Substitution"`.

## VALUE LIST + SUBSTITUTION + DEDUP
- `/hslookup`: now lists abbreviations/substitutions (wordplay + substitutions tables) FIRST, then
  synonyms, ALL ALPHABETICAL, NO cap (was a 30-item cap that dropped every abbreviation — e.g.
  point's E/N/S/W). "good" returns ~719.
- New `substitution` role -> in the manual commit renders "Substitution"; on `/hsresolve` writes to
  the **wordplay** table via `admin_db.add_substitution` (NOT synonyms_pairs), so abbrevs/symbols
  aren't misfiled.
- `admin_db.add_synonym` and `add_substitution` now dedup **case-insensitively** (capitalised clue
  words like "Good" no longer duplicate "good").

## STORE / DB STATE
- The store is **`data/clues_master.db`** (`wfw_solve`/`wfw_piece`/`wfw_link`/`wfw_frozen`) — durable,
  NOT the regenerated `pipeline_stages.db`. Manual solves persist across restarts and re-solves.
- This session's committed manual solves (all frozen): 10077223 RAMBLES, 10077226 THEME, 10077227
  MAIDEN, 10077234 SCABBIEST, and 10077221 MERGE (10077221 is a TEST ARTIFACT from my verification —
  the user may want to Uncommit it). DB writes are gitignored/local.

## SERVER
`.venv/Scripts/python.exe -m core.wfw_web` on 127.0.0.1:5099 — exactly ONE listener. Restart after
any code change (Python loads modules once). Verify with `netstat | grep 5099` + a curl. Tell the
user to HARD-REFRESH.

## NEXT / CLEANUP (in priority order)
1. **Strip the dead atom-tool MS code** from `core/wfw_web.py`: routes `/handsolvecommit` +
   `/handsolveuncommit`, and the add-piece/commit/uncommit JS in `initHS` + HTML in
   `_handsolve_block` (`hs-addpc`/`hs-commit`/`hs-uncommit`/`hs-pctext`/`hs-pcval`). `/handsolve` is
   now UNREACHABLE from the clue page — it's harmless dead code but confusing. ~14 lines / a few blocks.
2. **DB cleanup (user deferred 2026-07-03)** — wordplay/substitution values leak into `get_synonyms`
   (via `definition_answers_augmented`), so some render as SYNONYM (e.g. LAIRD 10077242 `right->R`).
   Root fix = PRUNE the polluting synonym/def_aug rows. Do NOT relabel `lookup_all` at solve time
   (signatures match ABR_F/SYN_F on the mechanism). See `memory/wordplay_substitution_labelling.md`.
3. `substitution` role's `/hsresolve` path calls `reload_wiring()` (~9s); could add an incremental
   `invalidate("abbreviation")` instead.
4. **Push/merge GATE:** the engine commits (`398e43b0`, `6cdc16b4`) were validated (reversal fix had a
   0-regression A/B; selection engines per-clue + small A/Bs). Prior handovers wanted a full-corpus
   overnight A/B before push — confirm with the user before pushing `redesign`.

## MEMORY POINTERS (auto-loaded — read before touching these areas)
- `manual_solver_spec_from_user.md` — the user's EXACT manual-solver spec + the render-parity
  principle. Do NOT reinterpret.
- `wordplay_substitution_labelling.md` — the deferred DB-cleanup for the synonym/substitution
  mislabelling.
- `feedback_soundness_by_design_not_gates.md` — don't ship gates+happy-path; check your own work.
- `feedback_match_test_size_never_block_user.md` / `feedback_large_tests_at_night.md` — A/B sizing;
  never block the user's server.
