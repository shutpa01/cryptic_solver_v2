# COLD HANDOVER — 2026-08-04

Honest account. Verified facts are labelled; anything I could not verify is called out.
This session I twice stated a guess as fact and the user rightly called it out (see §5).
TRIAGE FIRST, verify the REAL running thing, do not guess-and-assert.

---

## 0. WHAT IS COMMITTED (branch redesign, HEAD = 67c79f13)

- **03b97c5a** — charade_positional direction derived from the assembly, end to end
  (core/wfw_web.py, admin_db.py, run_dev.py). See
  [[positional_direction_derivable_from_assembly]]. Not deployed.
- **67c79f13** — clue-page Cordelia flower alt="" -> alt="Cordelia" (web/templates/clue.html).
  Fixes a Bing "alt attribute missing" flag on every clue page. Not deployed.

## 1. UNCOMMITTED CODE (working tree only — nothing deployed)

- **core/wfw_web.py** (6 lines) — the `/hscd` fix (§3). NEW, uncommitted.
- **scraper/orchestrator/daily_scraper.py** (22 lines) — the HTML-tag strip at ingest
  (`strip_html_tags` in `sync_to_master_clues`). Pre-existing uncommitted (since 2026-07-26).
  **STILL UNCOMMITTED — this is the likely crux of §4.**
- Other modified files NOT touched this session (leave for their own owners):
  container_deletion_engine.py, danword_lookup.py, a telegraph json, about.html,
  puzzles.html, web/wfw_read.py, .claude/settings.local.json.

## 2. HTML-TAG-IN-CLUE-TEXT BACKLOG — CLEARED (local DB), but see §4

Root cause (VERIFIED via `core.diagnose`): literal `<i>`/`<b>` tags in `clue_text`
tokenize into stray `'i'` word units, polluting the wordplay so no engine matches.
See [[everyman_html_tags_in_clue_text]].

Cleaned at the SOURCE (stripped tags from clue_text) — NOT band-aided:
- **10082794 AGHAST** ("American checks out of dilapidated Gasthaus, horrified") — tags
  stripped, assignment rebuilt on clean tokens (American=US deletion, "checks out of"=
  deletion ind, dilapidated=anagram ind, Gasthaus=fodder, horrified=def). Frozen manual pass.
- **8 unsolved backlog clues** — tags stripped from clue_text (no saved state to break).
- **10080278 CADET**, **10082804 SHOOTS THE BREEZE**, **10082809 MAESTRI** — tags stripped,
  assignments rebuilt (dropped the filler 'i' tokens, reindexed), frozen manual passes.
- LEFT ALONE: **10053603** ("< kind of tense") — a genuine cryptic `<`, not markup.

All done via the real `/solver/hsmanualcommit` route, verified by direct DB query.
Backups in the scratchpad (backup_10082794.json, backup_3clues.json, backup_backlog_clean.json).

## 3. CD "will not solve" — FIXED (uncommitted) — clue 10082763

"A posh roll?" -> DEBRETT'S PEERAGE is a cryptic definition. It carried a STALE BOGUS
frozen charade pass (the whole stored parse was just `posh=P` at answer position 9 — 1 of
15 letters). ROOT (VERIFIED by reading save_parse:132 + reproducing on the live route):
the CD engine files a clue 'pending', and `save_parse` never overwrites a FROZEN pass with
a non-pass, so the CD could never land — it "would not solve as a CD" every time.

FIX (core/wfw_web.py `/hscd`, uncommitted): a CD declaration now `clear_frozen`s a frozen
ENGINE pass before re-solving (a frozen MANUAL solve is still refused — Uncommit first).
Verified on the real route: clue now operation='cd', pass, frozen, def "A posh roll?" ->
DEBRETT'S PEERAGE, no bogus pieces. Durable for any clue stuck the same way.
I do NOT know how that bogus parse got written originally (did not investigate — was an
old engine false-pass at some point; the current engines correctly fail it).

## 4. OPEN / NOT DONE — the real recurrence is NOT fully closed

- **NEW Independent clues arrived TAGGED.** After the backlog clean (tags 12 -> 1), two new
  clues came in tagged: **10083006** (independent/12424, `<i>Matilda</i>`) and **10083282**
  (independent/12425, `<i>Turandot</i>`). So HTML tags STILL enter master for at least the
  Independent source. `strip_html_tags` IS wired into `sync_to_master_clues` and Independent
  IS in `PUBLICATION_TABLES` — so WHY these slipped through is unresolved. Candidates to
  check (NOT verified): the strip fix is uncommitted and the nightly that ran may have used
  code without it; or an independent-specific ingest path bypasses the strip; or the
  ON CONFLICT clause let a pre-strip row survive. **My earlier claim "new clues come in
  clean" was PREMATURE — it only held for the sources present at the time (Times/Guardian).**
- **Commit the two fixes** so they stop being fragile: the `/hscd` fix and (critically) the
  `daily_scraper` strip. Uncommitted = one git reset from gone.
- Nothing is deployed. All clue fixes + reference-DB harvests are LOCAL only.

## 5. UNRESOLVED QUESTION — clue 10082787 "no summary" (could NOT reproduce)

User: "why does this clue have no summary?" (TIMES 5227 16 down, "Cycles taken out of
housing development" -> EVOLUTION). VERIFIED it IS that clue, and on the dev server RIGHT
NOW it DOES show a summary ("REVOLUTIONS -> EVOLUTION"; frozen manual deletion solve). I
fetched the real rendered `/solver/?id=10082787` page — the `wfw-build` summary div is
present. So I could not reproduce "no summary"; the user's view holds a different/older
state than the live DB. Left OPEN pending the user telling me the exact surface/URL.
NB port 5001 was repeatedly squatted by the AI_Solver process this session (run_dev.py
self-kill reclaims it) — the user may be on a stale server/cached page.

**Behavioural (why the user pushed back):** I asserted (a) "stale bogus frozen engine
passes from old engine versions can exist on other clues" — no evidence, retracted; and
(b) "manual solve => no admin summary" — EMPIRICALLY FALSE (23 manual TIMES-5227 clues all
have summaries; only the CD lacks one). Verify before asserting. The `_summary` paths:
web/wfw_read.py `_summary` (public, generic, works for manual) vs core/wfw_render.py:312
(`_TYPE_RENDERERS.get(parse.operation, _render_generic_breakdown)`; the badge derives the
real type at :259-260 but this dispatch uses raw operation) — both DO produce a summary for
this clue, so neither is the cause of what the user saw.

## 6. ENV
- Dev: `web/run_dev.py` on :5001 (V2 venv `.venv\Scripts\python.exe`, reloader OFF — full
  restart per .py change; self-kills stale/AI_Solver listeners). Admin via `?admin=<ADMIN_KEY>`
  (default 'dev-admin-key'); the SolverMount gates `/solver/*` on the SITE session, so hit a
  non-/solver route with `?admin=...` FIRST to set the cookie, then reuse the cookie jar.
  Routes used: /solver/hsmanualcommit, /solver/hscd, /solver/hsresolve, /solver/approveall,
  /solver/?id=N (clue page), /solver/hs?id=N (hand-solver).
- data/clues_master.db (clues, wfw_*, pending_enrichments); data/cryptic_new.db (reference).
  Both gitignored, local. Live droplet has its own DBs; code deploys do NOT carry DB changes.
- The user deploys; Claude never deploys. Commit only when the user says so.
