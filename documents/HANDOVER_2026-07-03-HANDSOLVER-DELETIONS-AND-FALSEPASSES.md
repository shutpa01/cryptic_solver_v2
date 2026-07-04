# Handover — 2026-07-03 (PM) — Hand-solver deletions/anagram/hyphen + false-PASS fixes + new engine

Cold start for a NEW thread. Read top-to-bottom. Supersedes HANDOVER_2026-07-03-MANUAL-SOLVER.md.

## CODE STATE
- Branch **`redesign`**, HEAD = **`aa3fd09e`** (committed this session, **NOT pushed**). Prior session
  ended at 486844ba; everything since is this session, now committed in one commit.
- 7 files: `core/wfw_web.py`, `core/wfw_render.py`, `core/charade_deletion_engine.py`,
  `core/anagram_multi_substitution_engine.py`, `core/admin_db.py`, `core/engine_registry.py`, and NEW
  `core/charade_multi_named_deletion_engine.py`.
- Working tree still has pre-existing untracked cruft (scraper JSON, old handovers, `core/atomsig/`,
  a `scraper/telegraph/telegraph_daily.py` change that is **NOT mine**) — LEAVE them.
- **Gitignored/local DB state** (`data/clues_master.db` = the wfw store) is NOT in the commit. Manual
  solves + re-solved verdicts live there only.

## SERVER
`.venv/Scripts/python.exe -m core.wfw_web` on 127.0.0.1:5099 (last PID 36708). Restart after ANY code
change (Python loads modules once). Verify ONE listener (`netstat | grep 5099`) + a curl. Tell the user
to HARD-REFRESH (Ctrl+Shift+R) — plain reload serves cache.

## ⚠️ THE THREE FALSE-PASS FIXES (most important — the user rates these top severity)
A false PASS is the worst bug in this project. Three were found + fixed today:
1. **SUBSCRIBER (10077401)** — `charade_deletion` let a deletion indicator bind a piece it wasn't
   adjacent to, across a container. FIX in `charade_deletion_engine._finalize`: the deletion indicator
   must be reachable from the piece's word-run through link/deletion/location words only (a content or
   non-link foreign indicator blocks). Clean before/after A/B: 2 flips (SUBSCRIBER→fail, AMPS→correct
   container), 0 regressions.
2. **SANATORIA (10077533)** — `anagram_multi_substitution` absorbed one UNUSED "surface indicator" as
   dead glue ("Top" hand-waved). FIX: guard `inert_ind > 1` → **`> 0`** (only genuine LINK words may be
   leftover). A/B: the engine wins ~0 clues in a 400-anagram sample, so the fix only removes the
   false-pass path. **I had PROPOSED this fix earlier and let it slip un-implemented — don't do that;
   build a proposed correctness fix or clearly flag it pending.**
3. **Manual commit passed with clue words UNACCOUNTED (10077531)** — `/hsmanualcommit` only checked
   answer-TILE coverage. FIX: before saving, refuse if `definition is None` OR
   `parse.unexplained_words(ctx)` is non-empty (names the missing words). A manual PASS now requires
   EVERY clue word to have a role.

## NEW ENGINE — charade_multi_named_deletion (sibling, A/B-clean)
Delete a CHARADE of NAMED letters from a real DB value. **DANTE (10077543)** = ANDANTE(slowish) − "a
name"(A+N). Sibling of `charade_named_deletion` (which deletes ONE named value from ONE word) — that one
is UNTOUCHED. Registered in engine_registry after it. Safety: base value real DB; deleted string = exact
concat of ≥2 contiguous namer words' verified abbreviation values (a single-letter word supplies its own
letter, a→A); deletion indicator present + adjacent; every word accounted; exact reconstruction. A/B
(930 clues): wins 1 sampled clue (CAPTAINED = CAP+OBTAINED−OB, a GENUINE gain), 0 fabrications.

## HAND SOLVER (/hs) — new roles & abilities this session
Read `memory/manual_solver_spec_from_user.md` first. All render via the engines' OWN `_render_assembly`
(parity). New this session:
- **− delete** on synonym/substitution: remove a contiguous run from a derivative (ORATION −O → RATION).
- **letter-shift** indicator type (+ subtypes last→front / first→end) + rotation AUTO-PLACE (TERNS→STERN).
- **anagram fodder** role: value = the ticked words' letters, click the tiles it rearranges into
  (permutation-checked). Fodder may EXCEED the tiles → the surplus is a deletion; renders "FODDER −X".
- **deletion** role: assign the WORD that supplies a removed letter (a→A), no tiles → "a → A Deleted".
  (This is how the user wanted to "assign the A" — not the typed − delete box.)
- **split hyphenated words**: `_hs_word_units(ctx)` splits "line-up" → "line" + "up" so each part gets
  its own role. `_word_roles(split_hyphens=True)` for /hs; manual commit uses the same units (indices
  aligned); old /rolegrid untouched. (10077545 WORCESTER = line[synonym] + up[indicator].)
- Sub-type dropdown is DATA-DRIVEN (`fillSub`, from `_IND_SUBTYPES`) — deletion/selection/letter_shift.

## CLUE PAGE
- Inline editable **comment** (below the summary; `/cluecomment`, shares wfw_notes with the /hs note).
- **Abbreviation** add form → wordplay table (`kind=substitution` → `admin_db.add_substitution`).
- **Status** override reinstated (pass/fail/pending/invalid); display-only Definition + Unforce removed
  (`/setdef`, `/unforce` routes are now dead code — candidates for a cleanup sweep).

## RENDER (wfw_render.py)
- Breakdown ROWS now show the piece transform: reversed / −deleted / last→front / first→end / anagram
  (+ "anagram −X" when fodder exceeds tiles) — via shared `_transform_note` + `_piece_label`.
- Indicator rows show the SUB-TYPE ("Selection — first letter(s)", "Deletion — remove first letter") via
  `_indicator_label` + `_SUBTYPE_DETAIL`. Guard: `_transform_note` returns '' when got=='' (a zero-link
  source) — else it emitted bogus "−WHOLEVALUE".

## TESTING GOTCHAS (learned the hard way)
- **role_validity fails CLOSED standalone** — calling an engine directly gives status=fail even for a
  real pass. Verify PASS through the full cascade: `make_db_wiring()` + `solve(ctx, w)` (AI off).
- **Clean A/B = swap ONLY the engine file, same AI-off config, before vs after.** Comparing a re-solve
  to the STORED parse is contaminated (AI/discover/DB drift). Swapping the working file mid-A/B is
  fragile — it left the anagram engine at inert>1 briefly; always re-verify the file after.
- Browser extension NOT connected here — verify backend/render via `test_client`; the user tests clicks.
- After changing an engine, RE-SOLVE affected clues live (`POST /reload` with `id`+`only`) — stored
  parses are stale. SUBSCRIBER/SANATORIA were re-solved to fail; DANTE re-solved to pass.

## OUTSTANDING / NEXT
- **PUSH GATE:** nothing pushed. Confirm with the user before pushing `redesign`. A full-corpus overnight
  A/B was the historical gate; today's changes were validated by targeted A/Bs (all clean).
- `/hsresolve` (Resolve→cascade) was NOT updated to split hyphens — only the manual Commit path was.
  Revisit if Resolve is needed on a hyphenated clue.
- Deferred (pre-existing): wordplay/substitution rows leak into get_synonyms (mislabel); dead atom-tool
  code (`/handsolve*`, initHS) still in wfw_web.py; incremental `invalidate("abbreviation")` (abbrev add
  currently pays a ~9s reload_wiring).
- Keep reviewing failed clues; the user is (rightly) intolerant of false PASSes — prefer an honest FAIL
  and the hand solver over any loose auto-parse.

## MEMORY POINTERS (auto-loaded)
See `memory/MEMORY.md` top entries: manual_commit_and_sanatoria_falsepass, charade_deletion_adjacency_fix,
charade_multi_named_deletion_engine, hand_solver_anagram_deletion, hand_solver_split_hyphenated,
indicator_subtype_in_render, and the base manual_solver_spec_from_user.
