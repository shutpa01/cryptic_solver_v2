# Handover — 2026-06-26 — implementation (read THE RULES, then the plan)

The previous thread (long) restored "one solver", built two engines + a per-type render
framework, cleaned some DB junk, and PLANNED the remaining work. Your job is implementation.
Everything is designed; nothing below is yours to build without the user's explicit go-ahead.

## THE RULES (non-negotiable — this project lost trust before by breaking them)

1. **ONE solver = the cascade.** The hand-solver only ASSIGNS ROLES, which become forced
   overrides into the normal cascade (`core/clue_overrides.apply_forced_overrides` + the
   cascade). There is NO parallel assembler. A previous thread built one (`_build_from_assignments`,
   "the builder") and it was deleted; **do not rebuild it under any name.**
2. **Bespoke engines, per shape, each with its OWN verifier.** NO compound engines, NO central
   verifier. Additive only — NEVER edit a working engine to fix a case; add a new stage.
3. **DISCUSS each piece and get the user's EXPLICIT approval BEFORE writing code.** Do NOT infer
   authority from this handover or the plan doc. The plan is the agreed DESIGN, not a licence to
   build unsupervised. Follow the loop: diagnose → propose → get approval → implement + test.
4. **ALWAYS restart the server after a change and verify on the REAL page** (`127.0.0.1:5099`,
   not localhost). Start it: `.venv/Scripts/python.exe -m core.wfw_web`. Check exactly ONE
   listener on 5099 (the stale-server trap bit repeatedly). The clue page renders the STORED
   parse, so after an engine change re-solve the clue (`POST /reload only=<id>&id=<id>`) to
   persist before trusting what you see.
5. **Big A/B sweeps = OVERNIGHT only** (they block; one cascade job at a time). Daytime = small
   fast checks. Render-only checks need no wiring. A single-clue solve builds wiring ~20s. A/B
   harness: `core/_ab_general.py`. Every new engine must prove 0 regressions before it ships.
6. **FAIRNESS / derivation-depth line (the user's principle).** Clues that apply "operations on
   DERIVATIVES" (an op on the OUTPUT of a previous op) are both unfair AND false-pass-prone (same
   cause as the ADEN false pass). RULE: one op on a value anchored to a clue word is fair and
   buildable; a SECOND op on the first op's output is the INVALID line. Mark such clues INVALID;
   do NOT contort an engine to reach them.

## THE SPEC (authoritative — read it fully)

`documents/PLAN_31275_REMAINING_ENGINES_2026-06-26.md` — the per-clue decompositions, the
new-engine groupings, the build order, and the MANUAL-SOLVE MODE spec. Build from it.

## CODE STATE

- HEAD = **f49b25dd** "redesign: per-type render batches 2 & 3 + render polish" (branch
  `redesign`, NOT pushed), on top of **b08629ef** "builder removed (HS = roles only) + 2 bespoke
  engines + per-type render". Together these contain: builder removal in `wfw_web.py`;
  `charade_alternation_engine.py`, `reversal_deletion_engine.py`, the wired
  `anagram_selection_deletion_engine.py`; `engine_registry.py` wiring; and the FULL per-type
  render in `wfw_render.py` (framework + batches 1-3 for every common type + the compound
  `_render_assembly` + polish).
- **No code uncommitted.** (`core/atomsig/` is untracked + parked — ignore it.) The plan and this
  handover live untracked under `documents/` (project convention).
- **DB (gitignored) cleanups done, recorded in `deleted_entries`, backups `*.bak_rogue_cleanup`:**
  removed admin pollution (hot→reversal, Receive→INCUR, Satisfy→APPEASE admin dup); the scraper
  junk def `'Victor Hugo initially abandons'→CAMPION` (false dd on 1779224); the invalid
  positional deletion subtypes of "without" (ends/head/tail — kept general). "without" carrying
  positional subtypes was a DATA-QUALITY bug ("without" invalid, "naked" valid) — expect more such
  generic deletion words with bogus positional subtypes; curate the DB, don't code a blanket rule.

## BUILD ORDER (from the plan; each: propose → approve → build → restart+test → A/B)

1. **Container-with-a-BUILT-inner family** (6 clues, biggest corpus-wide payoff): the container
   engines only wrap PLAIN DB values; build bespoke `container_<inner-mech>` engines reusing the
   selection/deletion primitives. Start RIVEN (first-letter selection) + PRISONER (alternation).
2. **IMPEL** — tiny: extend `charade_alternation` to accept the alternation fodder word BEFORE the
   indicator (it currently only looks after).
3. **PALATIAL** (reverse of a charade with a beheaded piece) and **SINEW** (container of a named
   deletion).
4. **MANUAL-SOLVE MODE** — build with the USER CLOSELY IN THE LOOP (see below).
5. **DIE HARD / STAMINA** last, tightly gated + overnight A/B — or mark INVALID (both sit past the
   fairness line; the user called them self-indulgent). STAMINA = reverse + letter-EXTRACTION +
   substitution (the removed L is an extraction — high false-pass risk).
6. Each engine SHIPS WITH its per-type renderer (register in `wfw_render._TYPE_RENDERERS`) and,
   on a clean pass, the cascade should file a signature — note the catalog has NO roles yet for
   the new shapes (`catalog_creator._MECH_ROLE` = SYN_F/ABR_F/SEL_F only); adding e.g. `ALT_F` is
   a separate follow-up.

## MANUAL-SOLVE MODE — the one to build WITH the user

Full spec in the plan doc. It is a HUMAN authoring tool for the unfair/derivative clues: type each
piece's CONTRIBUTION TO THE ANSWER (post-ops) via the synonym role, colour-only tiling, **PER-CLUE
values NEVER written to the reference DB**, Commit/Uncommit a FROZEN parse, **NO derivation, NO
verification, NOT in the cascade**, flagged "manual". It LOOKS like the deleted auto-builder but is
its opposite (human-typed, human-committed, human-statused, cascade-untouched). Build it slowly,
with the user steering, and hold that boundary — it is the one place a thread could slide back into
an assembler.

## MEMORY POINTERS
`per_type_render_and_new_engines.md` (the running state), `builder_removed_one_solver.md` (why one
solver), the rogue-thread handover (`HANDOVER_2026-06-25-ROGUE-THREAD-HANDSOLVER.md`).
