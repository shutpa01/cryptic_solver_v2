# Solver Redesign — detailed design (draft, 2026-05-30)

Status: DRAFT for discussion. Not frozen. Supersedes nothing yet.

This is the design for the consolidated, cheap, word-for-word solver agreed in the
2026-05-30 session, built on the understanding captured in AS_IS_PIPELINE.md.

---

## 1. Purpose and end game

The end game is word-for-word (WFW): the user is shown, for every letter of the
answer, where it came from and by what operation. The solver is not a black box that
emits a label; it produces a complete, letter-level account of how the clue makes the
answer.

Two consequences drive the whole design:
1. WFW is a DATA-MODEL decision, not a display feature. It must be built into the
   foundation; it cannot be retrofitted onto a word-level engine.
2. We always know the answer. The solver's job is to EXPLAIN, not to discover. Every
   step is answer-aware.

Secondary goals (all evidenced in AS_IS_PIPELINE.md): far simpler, far cheaper (no
paid API in the core path), and no duplicated engines or catalogs.

---

## 2. Principles (carried from the AS-IS findings)

- Atomise the clue and answer once, reuse everywhere.
- Preserve all letter-contributing evidence, even on failure.
- An operation may only apply to material near its controlling indicator (proximity).
- Model compound clue types from composable operations, not single-type engines.
- Run the best engine first / route, rather than first-to-claim-wins.
- Triage routes to the catalog; it does not solve in parallel.
- One catalog, not three. One confidence model, not four.

---

## 3. The atomised data model (the foundation)

Built once per clue, reused by every stage.

### 3.1 Atoms
The clue is atomised into ordered atoms — normally words, but the atomiser may split
contractions and possessives so cryptic devices are reachable (e.g. "he's" -> "he" +
"'s"; "I'd" -> "I" + "'d"). Each atom carries: surface text, normalised text, original
position. The committed WFW project already has a good atomiser; revive it rather than
rebuild.

### 3.2 Answer as letter-slots
The answer is represented as an ordered array of N letter-slots:

    slot[i] = { letter: 'O', source: null }   # source filled during solving

This array IS the WFW substrate. Solving = filling every slot's source.

### 3.3 Piece
A piece is a contribution from one or more clue atoms:

    Piece = {
      atoms:      [atom indices],     # which clue atoms produced this
      value:      "MAN",              # the letters it contributes
      mechanism:  "synonym",          # synonym | abbreviation | raw | first_letter |
                                      #   last_letter | outer | homophone | ...
      source_text:"Bloke",            # for display
    }

### 3.4 Provenance map (the WFW core contract)
The output of solving is not a boolean and not a flat list — it is a map from each
answer letter-slot to its origin:

    provenance[i] = {
      piece:      <Piece>,            # the piece that produced this letter
      operation:  "container",        # the op that placed it here
      transform:  null | "reversed" | "anagram_of" | ...
    }

Granularity note (honest): for charade / container / reversal / deletion the mapping
is per-letter (each answer letter has a definite source letter). For anagram and
hidden the individual letters are not separately sourced, so those spans carry a
SPAN-level provenance ("answer[2..6] = anagram of HAW+TREE") rather than per-letter.
The data model allows both: a provenance entry may cover one slot or a span.

### 3.5 ParseResult

    ParseResult = {
      definition_atoms: [...],        # the atoms that form the definition
      pieces:           [Piece, ...],
      operations:       [...],        # the op tree (compound-aware)
      provenance:       [per-slot/span],
      confidence:       0-100,
      reasons:          [(text, delta)],
    }

---

## 4. The pipeline shape

    CLUE (answer known)
      |
      v  atomise once -> atoms + answer letter-slots
      |
      +--> Quick checks (free, high-precision):
      |       - Hidden-word scan (letter scan; no indicator needed)
      |       - Double definition
      |     If solved, emit ParseResult with provenance. Done.
      |
      +--> Grammar router:
      |       POS-tag the atoms, map the shape to a RANKED list of catalog
      |       templates worth trying (most likely first).
      |
      +--> Catalog engine (the core):
      |       for each ranked template:
      |         - lay atoms on slots (leftovers must be link/indicator, proximity-gated)
      |         - look up each fodder slot's candidate values from the DB
      |         - VERIFY by reconstruction -> returns a provenance map (or None)
      |         - score the ParseResult
      |       keep the best; accept at threshold.
      |
      +--> If nothing solves:
              - Grammar flags likely Cryptic Definition -> label, do NOT attempt.
              - Otherwise: genuine gap -> Catalog-creation process (section 8).
      |
      v
    Enrichment loop: discoveries feed DB + catalog; next run is stronger.

No paid API in the core path. (An optional, clearly-separated AI assist may exist for
hard leftovers, but it is not part of the core and not required.)

---

## 5. Components

### 5.1 Atomiser
Single source of truth. Produces atoms + the answer letter-slot array. Handles
contraction/possessive splitting. Revive the WFW atomiser.

### 5.2 Quick checks
- Hidden: scan the atom letter-stream for the answer appearing contiguously (forward
  or reversed) inside/spanning atoms. Cheap, reliable, kept standalone (the catalog's
  indicator-gated hidden op is weaker). Emits per-span provenance.
- Double definition: both ends define the answer via the DB. Kept as a dedicated
  engine. Emits provenance (each half -> whole answer, flagged as definition).

### 5.3 Grammar router
The repurposed grammar layer. Input: POS shape of the atoms. Output: a ranked list of
candidate templates (operation + slot pattern). It does NOT solve; it narrows the
search. The 166 learned POS->role patterns become this ranking. A clue shape with no
viable template AND no wordplay indicators present is a candidate for the CD class.

### 5.4 Catalog (single, consolidated)
One catalog. Each entry: a slot pattern (F/I) + an operation + a frequency/priority.
Collapse the current three (base/positional/old) into one, keeping every template that
uniquely earns its place (measured, not assumed — the old catalog currently out-covers
the distilled base, so the merge must be coverage-driven).

### 5.5 Operation verifiers — the provenance contract
THE central rule: every verifier returns a provenance map, not a boolean.

    verify(operation, pieces, answer_slots) -> provenance | None

Each operation knows where its pieces' letters land:
- charade: pieces concatenate; slot i <- the piece covering position i.
- container: outer letters map to the outer piece; the inserted span maps to the
  inner piece(s). (Inner may be multi-piece — see container_multi_inner, already
  built 2026-05-30.)
- reversal: map the flipped positions back to the source piece, transform="reversed".
- deletion/trim: map surviving letters to the source; record the removed letters.
- anagram: span-level provenance "this span = anagram of <fodder>".
- hidden: span-level provenance into the host atoms.
- homophone: whole answer <- "sounds like <source>".
- acrostic/alternate: each answer letter <- the atom it was taken from.
Compounds compose: a container whose inner is an anagram nests provenance.

Getting this contract right BEFORE adding templates means every template added later
supports WFW automatically.

### 5.6 Scorer
One deduction model (start 100, deduct for unconfirmed/nonsense pieces, missing
indicators, circularity), reused everywhere. Now provenance-aware: it can check that
every answer slot has a source and every clue atom has a role (the residue check
becomes a completeness check over the provenance map). One confidence number that
means the same thing across the whole solver.

### 5.7 CD classifier
High-precision. Flags a clue as a cryptic definition only when: no definition+wordplay
split is possible AND no wordplay indicators are present. A CD's ParseResult is the
clue defining the answer as a whole (no wordplay breakdown). Conservative by design —
mislabelling a solvable clue as CD silently stops us solving it.

### 5.8 (see section 8 — Catalog-creation process)

### 5.9 Enrichment
Discoveries from the catalog-creation process (a missing synonym, a new template) feed
back into the DB and catalog. Because the near-miss evidence is preserved (principle),
a clue that fails by one missing piece can name that piece for enrichment — unlike
today, where total failures contribute nothing.

---

## 6. What is removed (versus AS_IS_PIPELINE.md)

- Phase 0b spoonerism (niche; revisit only if measured to matter).
- Phase 0.5 V1 mechanical solvers (anagram/container/deletion/charade/reversal/
  acrostic/homophone) — duplicates of catalog operations. (~1,500 lines.)
- grammar triage's structural tests (duplicates) — the file slims to a router.
- positional + old catalogs as separate engines — merged into one catalog.
- the paid fallbacks (DBE-Haiku, indicator-enrichment, Haiku definition) and the
  paid Tier-2 stage — out of the core path.
- four confidence schemes -> one.
Net: thousands of lines removed; no API in the core; one engine, one catalog.

---

## 7. Build order (strict processing order — the order a clue travels)

Build the pipeline front to back, in the exact order a clue flows through it. Each
stage takes the atomised clue and produces the same word-for-word record (section 3).
Each stage is proved on real clues with output shown, and regression-/coverage-checked,
before the next.

Foundations (shared, built once, not stages a clue "passes"):
- F1. Atomiser — front-end input prep. DONE (core/atomiser.py).
- F2. The provenance record (Piece / Provenance / ParseResult) — the shared output type
  every stage fills in. DONE (core/model.py).

Stages, in clue-flow order:
1. Hidden-word check — the first thing a clue meets. DONE (core/hidden.py).
2. Double definition. DONE (core/dd.py).
3. Definition extraction — split the clue into a definition (at one end) and the
   wordplay words, so the router and catalog work on the wordplay. DB check injected.
   (NEXT.) Needed before stages 4-5 because they operate on the wordplay words.
4. Grammar router — POS shape of the wordplay -> a ranked list of catalog templates
   worth trying.
5. Catalog engine — match -> verify -> score, emitting the record. Reuse the existing
   matcher/verifier logic wrapped to emit records (do not rebuild the big matcher), and
   consolidate the three catalogs into one here, coverage-driven.
6. Cryptic-definition catch — when nothing above solves, classify CD (high precision)
   versus a genuine leftover.

Then migrate and retire (only once the new pipeline covers the clues, measured):
- Retire the Phase 0.5 V1 solvers and grammar-triage's structural tests once stages 4-5
  demonstrably cover their clues (measured per-mechanism, especially homophone/acrostic).
- Remove the paid core fallbacks once coverage is acceptable without them.
- Cut over run.py / solve_clue to the new pipeline; delete the orphaned files
  (section 12 dependency map and cleanup order).

Stop-and-measure before retiring anything (stages 4-5) — that is where coverage could
regress.

---

## 8. The catalog-creation process (run when there is no catalog match)

The structured replacement for the loose leftover process. For each unsolved clue:

1. Hand-derive the intended parse: definition, operation(s), pieces.
2. Diagnose and classify the gap as exactly one of:
   - missing DB entry (synonym / abbreviation / indicator),
   - missing catalog template (structure not representable),
   - mis-classified indicator,
   - cryptic definition (route away).
3. Fix only at the right layer:
   - DB gap -> add the entry (enrichment),
   - structural gap -> create the template (data -> wiring -> verifier -> explanation),
   - indicator gap -> fix its classification,
   - CD -> label and stop.
4. Prove it solves at threshold through the real engine, correct parse + provenance shown.
5. Regression-check existing solves unaffected.
6. Record what was added.

Discipline (what makes it tighter than the old leftover process): classify the gap
BEFORE touching anything, so the fix lands in the right layer; verify through the real
path with output shown; one change at a time with a regression check.

---

## 9. Open questions / risks

- Coverage on retirement: must measure, per mechanism, that catalog+router covers what
  the V1 solvers and grammar-triage tests cover before deleting them. Homophone and
  acrostic are the thin spots.
- Without the paid stage, enrichment becomes human-driven (catalog-creation process).
  The DB grows slower; acceptable, but it changes the cadence.
- Anagram/hidden provenance is span-level, not per-letter — confirm the WFW display
  handles span attribution cleanly.
- CD precision: a conservative classifier risks leaving real CDs in leftovers; an eager
  one risks never attempting solvable clues. Tune toward precision (never skip a
  solvable clue) and let true CDs that slip through land in leftovers.
- The single consolidated catalog must be built coverage-first; the current distilled
  base catalog is the weakest of the three, so naive "keep base, drop the rest" loses
  coverage.

---

## 10. DB and storage changes

### Reference tables (cryptic_new.db) — schema unchanged
The new engine reads exactly the data the catalog engine reads today, so these stay:
synonyms_pairs, the abbreviation/substitution table, indicators, homophones,
definition_answers_augmented. Enrichment still appends rows to them. One data cleanup,
not schema: harmonise the indicators table's PARTS subtypes so each indicator carries
one consistent subtype, removing the per-solver remapping the AS-IS doc found.

### The catalog moves into a table (decided)
Replace base_catalog.json / positional_catalog.json / catalog.py with one table,
catalog_templates:

    catalog_templates(
      id, pattern, operation, priority,
      origin,        -- 'mined' | 'hand_added'
      active,        -- soft-disable without deleting
      version, created_at, notes
    )

- The loader reads it once into memory at run start (as it reads JSON today).
- The catalog-creation process INSERTs a row instead of editing JSON.
- It becomes queryable (which templates exist, which fire) and versioned.
- It pairs with recording which template solved each clue (below), so coverage is
  measurable for the first time.

### Parse / explanation storage (clues_master.db) — changes for WFW
- Today structured_explanations.components holds a flat parse. WFW needs the
  per-letter / per-span provenance map. Add a dedicated table rather than overload the
  JSON:

      clue_provenance(
        clue_id, slot_start, slot_end,   -- the answer letters this entry covers
        piece_text, mechanism,           -- e.g. 'Bloke' / 'synonym'
        value,                           -- the letters produced, e.g. 'MAN'
        operation, transform             -- e.g. 'container' / 'reversed'
      )

  This is the WFW substrate persisted: render straight from it.
- clue_word_roles (per-word) is subsumed by the per-letter provenance; keep it only as
  a derived view if anything still needs word-level.
- Record the solver path on structured_explanations: add solved_by
  ('hidden' | 'dd' | 'catalog' | 'cd') and template_id (FK to catalog_templates when
  the catalog solved it). This fixes the current "can't tell what matched" gap and
  enables per-template coverage measurement.
- model_version stays but is no longer the attribution of record; solved_by +
  template_id are. The redesign must write attribution that a later pass does not
  silently overwrite (the last-writer-wins problem from the AS-IS doc).

### Enrichment tables — unchanged
pending_enrichments / rejected_enrichments stay; the catalog-creation process feeds
pending_enrichments as today.

### Migration note
All of this is additive: build the catalog_templates table and clue_provenance
alongside the existing columns, backfill/derive where possible, and switch the readers
over once verified — matching the incremental build order (section 7, steps 2 and 5).

---

## 11. File map

How the current files map to the redesign. (One path to confirm: the committed WFW
atomiser — locate it before reviving rather than rebuilding.)

### New files
- atomiser (revive the WFW atomiser) — produces the atoms + the answer letter-slot
  array; single source of truth for clue/answer preparation.
- model (small) — the shared data types: Piece, ParseResult, the provenance map.
- cd_classifier — high-precision cryptic-definition flag (section 5.7).
- catalog loader — reads the new catalog_templates table into memory (replaces the
  JSON loaders).

### Kept / refactored
- signature_solver/db.py (RefDB) — kept; reads the same reference tables.
- backfill_ai_exp/backfill_dd_hidden.py — kept; the hidden + DD quick checks; extended
  to emit provenance.
- signature_solver/grammar_triage.py — SLIMMED to a router (POS shape -> ranked
  templates). The structural tests (its own anagram/reversal/container/charade) are
  removed; the POS-catalog ranking is what survives.
- signature_solver/matcher.py — kept; the operation verifiers, _lookup_slot and
  _verify_combo stay and are extended so each verifier returns a provenance map. The
  old-catalog matcher (match_signatures) folds into the single matcher.
- signature_solver/base_matcher.py — kept as THE matcher over the one catalog (its
  flexible-span match_base is closest to the consolidated form). positional_matcher
  merges in.
- signature_solver/confidence.py — kept; one scorer, made provenance-aware (the
  residue check becomes a completeness check over the provenance map).
- signature_solver/executor.py — kept/refactored to build the explanation FROM the
  provenance map.
- signature_solver/solver.py — SLIMMED: solve_clue becomes a thin orchestrator
  (atomise -> quick checks -> router -> catalog -> CD); the recursive paid fallback
  chain is removed.
- sonnet_pipeline/run.py — SLIMMED: the cascade reduces to the new shape; the paid
  phases (1.5, 1.5b, 2) and the Phase 0.5 block go.
- sonnet_pipeline/sig_enrichment.py — kept; the enrichment loop.
- signature_solver/tokens.py — kept.
- signature_solver/word_analyzer.py — REVIEW: its per-word role analysis largely folds
  into the atomiser + slot lookup; likely slimmed, possibly merged, not necessarily
  deleted.

### Deleted
- backfill_ai_exp/batch_v1_solver.py — the Phase 0.5 V1 solvers (duplicates).
- signature_solver/positional_matcher.py + signature_solver/positional_catalog.py +
  data/positional_catalog.json — merged into the one catalog.
- signature_solver/catalog.py — the old catalog, merged.
- data/base_catalog.json + signature_solver/base_catalog.py — replaced by the
  catalog_templates table + loader (base_catalog.py may survive only as the operation
  registration: OPERATION_INDICATOR_TYPE / OPERATION_FODDER_TYPES, until that too moves
  into the table).
- signature_solver/haiku_definition.py, haiku_dbe.py, haiku_indicator.py — paid
  fallbacks, out of core.
- sonnet_pipeline/tier2_solver.py — paid Sonnet stage.
- the blog pipelines (Times / fifteensquared) — out of the core (could live on as a
  separate optional importer, not part of the solver).

### New DB objects (from section 10)
- tables: catalog_templates, clue_provenance.
- columns on structured_explanations: solved_by, template_id.

---

## 12. Dependency map and cleanup order

Measured 2026-05-30 (git grep of imports across tracked .py, excluding worktrees and
prototypes). The mess is concentrated: nothing is a true orphan yet — almost everything
hangs off two hub files, signature_solver/solver.py (the engine) and
sonnet_pipeline/run.py (the cascade). That is the lever for cleanup.

### Who imports each legacy module
- batch_v1_solver        <- run.py, web/routes/admin.py, (test_new_pipeline.py, archive/merge_and_verify.py)
- positional_matcher     <- solver.py only
- positional_catalog     <- solver.py, base_matcher.py (token-sets), positional_matcher.py, scripts/sig_diagnostic.py
- catalog (old)          <- solver.py, matcher.py, base_catalog.py, run.py, sig_enrichment.py, cryptic_taxonomy/...
- base_catalog           <- base_matcher.py, solver.py, scripts/sig_diagnostic.py   (KEPT — op registration)
- grammar_triage         <- solver.py, tier2_solver.py (reuses its helper functions)
- haiku_definition       <- solver.py, run.py, tier2_solver.py, web/routes/admin.py
- haiku_dbe              <- solver.py, run.py
- haiku_indicator        <- solver.py, run.py
- tier2_solver           <- run.py only
- blog pipelines (sonnet_pipeline/tftt_pipeline.py, fifteensquared_pipeline.py) <- run.py / dashboard subprocess

### The lever
Rewrite the two hubs to the new shape — solve_clue becomes the slim orchestrator, run.py
becomes the new cascade — and most legacy files lose their only callers and become
orphans that delete safely.

### Three snags to handle BEFORE deleting (from the map)
1. Shared helpers, not just callers. base_matcher imports token-sets
   (INDICATOR_TOKENS_SET / FODDER_TOKENS_SET) from positional_catalog, and tier2_solver
   reuses helper functions from grammar_triage. Relocate those helpers into a kept
   module first, or deleting the home file breaks a keeper.
2. Non-pipeline callers. web/routes/admin.py uses batch_v1_solver and haiku_definition
   (an admin re-run feature); scripts/sig_diagnostic.py and cryptic_taxonomy reference
   the catalogs. These need updating/removing too — they don't disappear when run.py
   changes.
3. Already dead. backfill_ai_exp/archive/* and the test files are safe early deletes
   once confirmed.

### Cleanup order (each step a commit, each with a regression check)
1. Relocate shared helpers (token-sets, grammar helpers) into kept modules.
2. Rewrite solve_clue (engine) and run.py (cascade) to the new shape.
3. Delete the now-orphaned engine files (positional_matcher, tier2_solver, the haiku
   files, grammar-triage structural tests, the redundant catalogs/matchers).
4. Update the two non-pipeline callers (admin web route, diagnostic script).
5. Sweep the archive/test leftovers.

Rule throughout: disconnect (remove the call site) -> run the pipeline + regression-check
existing solves -> delete -> commit. Never delete a file that is still wired in.

