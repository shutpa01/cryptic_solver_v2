# Universal-form parallel system — frozen design

**Status:** frozen 2026-05-06.
**Authority:** this document is the source of truth. Individual memory
files in `memory/` carry detail; this is the consolidated reference.
**Do not change** anything in this document without explicit user
agreement; subsequent decisions add to the bottom (Decision log) and
are then woven back into the body once they've been agreed.

This document supersedes the partial-coverage design notes in
`CATALOG_DESIGN.md`, `MAPPER_REBUILD.md`, and `TREE_MATCHER_DESIGN.md`
where they conflict. Those files remain useful for narrower detail
(catalog data shape, tree-matcher API).

-----

## 1. Purpose

The universal-form work is a **UX project**.

The deliverable is a help app that, for any cryptic clue, shows the
user exactly what every single word in the clue is doing — with no
broad-brush wordplay labels and nothing left to assume.

Today's tools settle for "this is roughly an anagram" or
"this is a charade". That is dishonest about most clues:

- A charade is *only* a charade when it concatenates two (or more)
  pure synonyms.
- An anagram is *only* an anagram when the fodder is *native* — the
  letters come straight from the clue surface.
- Most real wordplays are compound (anagram-of-a-charade,
  charade-with-deletion, container-around-reversal, etc.).

Calling these by single labels hides the mechanism from the user.

The form tree, by accounting for every word in the clue, lets the UI
present:

- Each leaf — value, source word, mechanism (synonym / abbreviation /
  literal / positional / homophone / raw).
- Each operator — its operation and its indicator word.
- Link words and definition called out explicitly.
- Nothing left as residue, nothing left to be assumed.

Three payoffs of the same underlying decision:

1. **Honest UX** — every word accounted for, no assumed gaps.
2. **Reliable enrichment** — every verifier FAIL names the exact
   missing DB row.
3. **Precise catalog** — templates carry real structure, so cold-clue
   matches are trustworthy.

-----

## 2. Core terms

**Clue** — the input. Surface text, answer, source, puzzle metadata.

**Template** — an abstract structural pattern, e.g.
`charade(deletion[outer](literal), synonym)`. No specific words, no
specific values. The catalog is a set of templates.

**Form** — a specific verified solve for a specific clue. Template +
binding (which clue word filled which slot, what value each leaf
produced) + definition phrase + link words. The form is the **output**
per clue.

**Catalog** — the structural vocabulary. A set of templates, each
carrying example forms and frequency.

**Signature** — synonym for the unique id of a template, written as a
canonical string (the `id` field of a catalog entry).

**Solve** — a stored form (one row in `shadow_db.solves`).

The relationship: clue → template (matched via the catalog) → form
(stored as a solve). One clue gives at most one form (early-exit at
first match).

-----

## 3. Architecture

### 3.1 Parallel-of-production discipline

The universal-form work is a **parallel of the production solver**.

- Same matching machinery as production wherever possible
  (`signature_solver/db.py::RefDB`, `extract_definition_candidates`,
  `word_analyzer`, the production cascade structure).
- One intentional swap: the **clipboard verifier** replaces the old
  verifier as the gatekeeper.
- Everything else identical, so the parallel can swap into production
  with minimum surface-area change.

The discipline is "safe transition." When the parallel proves itself,
it becomes production by replacement, not rewrite.

**Reuse is the default. New code requires justification.** The list
of production components reused is explicit (section 7).

### 3.2 Cascade

The cascade structure mirrors `sonnet_pipeline/run.py`:

| Phase | Mechanism | Production code |
|---|---|---|
| 0a | Hidden words (definition-confirmed) | `backfill_ai_exp.backfill_dd_hidden.try_hidden` |
| 0b | Spoonerisms | `sonnet_pipeline.solver.try_spoonerism_v2` |
| 0c | Double definitions | `backfill_ai_exp.backfill_dd_hidden.generate_dd_hypotheses` |
| 0.5 | V1 mechanical (anagram, charade, container, deletion, reversal, acrostic, homophone) | `backfill_ai_exp.batch_v1_solver.try_*` |
| 1 | Catalog match | **NEW: tree-aware matcher** (replaces production's flat-token signature_solver) |
| 2 | LLM (Sonnet) | wired but skipped in prototype |
| 3 | Re-solve after enrichment | meta-process |

### 3.3 Production engines as DETECTORS, not solvers

Production's mechanical functions are used to **detect the mechanism
class** (hidden / spoonerism / DD / anagram / etc.). On a positive
detection, we route the clue to the matching template subset in our
catalog.

**Our tree-matcher then does the actual word-by-word accounting and
produces the form.** The form goes through the clipboard verifier;
PASS writes to shadow_db; FAIL emits enrichment candidates.

This means:

- One matcher does everything (uniform word-accounting discipline).
- The catalog must contain templates for *every* clue type, including
  hidden / spoonerism / DD / anagram / etc. Adding these is the
  catalog-growth work.
- Detection biases the matcher's template walk order — it is an
  optimisation, not the truth. If detected templates fail, the
  matcher walks the rest of the catalog.
- Per-phase "adapters" become trivial routing hints
  (`mechanism="hidden"`), not form constructors.

### 3.4 Per-clue flow

For each clue:

1. Tokenise (whitespace split + per-token punctuation strip; preserves
   internal apostrophes). See section 5.1.
2. Find definition candidates via production's
   `extract_definition_candidates`.
3. Run production's detectors (try_hidden, try_spoonerism_v2,
   generate_dd_hypotheses, V1 op detectors). If any fires, note its
   mechanism class as a routing hint.
4. Walk catalog templates in order:
   - If a routing hint is set, prefer matching templates of that class
     first.
   - Otherwise walk by frequency (most common first).
   - For each template, the tree-matcher tries to fit the clue.
     Bindings respect production's `word_analyzer` per-word role
     constraints.
5. The first template whose form clipboard-verifies as PASS wins.
   Stop. Write the solve + per-word assignments to shadow_db.
6. If no template PASSes: emit enrichment candidates from each FAIL
   into the shadow vocabulary tables (with provenance).

### 3.5 Early-exit

When a template produces a verified PASS, stop. Do not try further
templates. Multiple-template matches are an edge case (genuine
ambiguity is rare); we record one canonical form per clue.

**For unsolved clues**, try every template — every FAIL contributes
enrichment candidates. **For solved clues**, no further enrichment
candidates are emitted (they would be spurious).

-----

## 4. Data model

### 4.1 Form schema (`prototypes/universal_form_v2/schema.py`)

A Form has:

- **tree** — recursive `Node` describing the wordplay structure
- **definition** — `{phrase, answer}`
- **link_words** — list of clue words that played no leaf/indicator
  role
- **is_and_lit** — whether the whole clue is both definition and
  wordplay
- **flags** — adapter TODO markers

A Node has:

- **operation** — one of `LEAF_OPERATIONS` (literal, synonym,
  abbreviation, positional, homophone, raw) or `NON_LEAF_OPERATIONS`
  (charade, anagram, reversal, container, deletion, hidden,
  double_definition, acrostic, cryptic_definition, homophone,
  substitution, spoonerism, unknown)
- **indicator** — surface clue word(s) that signal the operation
  (None for charade and other NO_INDICATOR_OPS, and for leaves)
- **sources** — list of child nodes (empty for leaves)
- **value, source_word** — leaf-only
- **positional_kind / deletion_kind / acrostic_kind** — op-specific
  sub-discriminators

### 4.2 Catalog (`runs/catalog_v1.json`)

Each catalog entry:

- **id** — canonical signature string (e.g.
  `charade(deletion[outer](literal),synonym)`)
- **structure** — pure-structure tree (op + children + leaf flag +
  kinds, no values, no source words, no indicators)
- **indicator_slots** — paths to nodes that require an indicator
- **leaf_kinds** — list of leaf types in left-to-right order
- **frequency** — count of clipboard-PASS forms with this signature
- **examples** — up to 5 example clue records (answer / clue / blog /
  source / puzzle / clue_id)

The catalog is **derived**, not authored: `extract_catalog.py`
collects clipboard-PASS forms and groups them by signature.

### 4.3 Shadow DB (`prototypes/universal_form_v2/shadow_db.py`)

Six tables in `data/shadow_blog_v0.db`. Live DB is never touched.

**Vocabulary tables** mirror the live DB shape, with **clue_id and
solve_id provenance** so every shadow row can be audited:

- `synonyms_pairs` (word, synonym, source, clue_id, solve_id)
- `wordplay` (indicator, substitution, category, confidence, notes,
  clue_id, solve_id) — for abbreviations
- `indicators` (word, wordplay_type, subtype, confidence, source,
  clue_id, solve_id)
- `definition_answers_augmented` (definition, answer, source,
  clue_id, solve_id)

**Per-clue solve records:**

- `solves` (clue_id, signature, verdict, answer, form_json,
  checks_json, enrichments_json, created_at) — one row per
  (clue_id, signature) attempt. The **form_json blob is the source
  of truth**.

**Per-word role assignments (queryable index over the form):**

- `solve_assignments` (solve_id, clue_id, span_start, span_end,
  surface_phrase, role, mechanism, op, op_kind, value, qualifier,
  db_source, db_table, db_row_id) — one row per word/span in the
  clue.

### 4.4 Retrieval queries the schema supports cheaply

- Everything for clue X: `solves` filtered by clue_id, joined to
  `solve_assignments` by solve_id.
- All clues where word/phrase Y plays role Z:
  `solve_assignments` filtered by surface_phrase + role.
- All clues solved with signature S: `solves` filtered by signature.
- Shadow vocabulary review: each shadow vocabulary table filtered by
  clue_id IS NOT NULL, joined back to `clues` for context.

-----

## 5. Discipline

### 5.1 Tokenisation

Use `prototypes/universal_form_v2/surface.py::tokenize`. Mirrors
`signature_solver/solver.py::_normalize_clue` + whitespace `.split()`
+ per-token punctuation strip.

- Tokens **preserve internal apostrophes** — "bird's" stays as one
  token. "won't" stays as one token. "twenty-one" stays as one token.
- DB lookups handle possessives via
  `signature_solver/db.py::_word_variants` — strips trailing 's after
  apostrophe; tries the bare and possessive forms.
- The verifier compares surface tokens to leaf source_words using the
  same tokeniser so consistency is automatic.
- LINK_WORDS contains only genuine cryptic link words. The 's
  apostrophe-s remnant is **not** a link word — that hack is gone.

LINK_WORDS containing words that *also* double as cryptic indicators
(out, off, up, down, into, with, etc.) is **not** a problem. The
signature dictates which indicators are required, and the matcher
fills them from the indicators DB. Leftover words being on the
allow-list is just an honest description of the remainder. Assembly
is the hard constraint that drops wrong-mechanism forms.

### 5.2 Adapters fully account for every word

Every per-phase adapter (hidden / spoonerism / DD / V1 mechanical /
tree-match) must produce a Form where every clue word is in **exactly
one** of: leaf source / indicator / link word / definition.

If the adapter cannot account for every word — indicator missing from
DB, leftover word not on LINK_WORDS, etc. — it returns **no Form**.
This is preferable to a half-honest form, because:

- The verifier would reject it anyway.
- A no-Form result lets the cascade fall through cleanly.
- It avoids polluting the shadow DB with low-quality rows.

### 5.3 Match the operator to what the indicator says

When two operators yield the same letters, pick the one whose
mechanism matches the indicator's plain-English meaning, **not** the
mechanically simpler one.

- "uncovered" → `deletion[outer]`, not `positional[middle]`. Both
  yield E from TEA, but the indicator says cover-removed.
- "principally" / "head of" → `positional[first]`.
- "endlessly" → `deletion[tail]`.
- "core" / "centre" → `positional[middle]` or `deletion[heart]`
  depending on whether the indicator describes the kept letter or the
  discarded ones.

Don't shortcut to whichever fits the verifier in fewest nodes. The
verifier accepts both; the catalog and explanation should reflect
intent.

### 5.4 Preserve qualifiers on synonym evidence

When the gloss has a qualifier — "associated with X", "a type of X",
"X-related", "kind of X" — the synonym evidence preserves the
qualifier. The clue word stays as the source (e.g. "beer"); the
recorded source phrase becomes the qualified version
("associated with beer"). The synonym record is therefore (beer →
PORTER, qualifier="associated with"), not just (beer, PORTER).

A flat pair lies about the relationship. PORTER isn't synonymous with
beer; it's a *type* of beer. The UX promise — every word honestly
accounted for — extends to honesty about relationship type.

### 5.5 Reuse production components

Whenever the parallel-of-production system needs a step that
production already implements, reuse the production code unchanged.
Do not write a new equivalent, even a "small" one.

Before designing or implementing any step, check
`signature_solver/`, `backfill_ai_exp/`, etc. for an existing
function. If one exists with the right shape, reuse it.

Every design spec should carry an explicit "production reuse" list.

### 5.6 Catalog is derived, not authored

The catalog grows downstream of producer behaviour, full stop. To get
a new template into the catalog, get some producer (the tree-matcher
walking against a corpus) to emit a form with that signature, have it
clipboard-PASS, and re-run `extract_catalog.py`. There is no path to
hand-edit `catalog_v1.json`.

### 5.7 Catalog work is yellow-highlighter alignment

Building the catalog is **tedious, not difficult**. The mental model:
clue + blog + answer side by side, highlight every word until each
one is a leaf-source / indicator / link / definition. The form falls
out; the signature falls out of the form's shape.

Throughput matters more than cleverness. Catalog growth is a function
of how many clues we run the highlighter over, not how sophisticated
the producers are.

### 5.8 Running a signature = matching against clue text

When you say "run the signature on a sample," you mean fit the
template to **clue text**, with the blog set aside. Never reach for
blog regex — that pattern-matches notation, not mechanism.

-----

## 6. Components

### 6.1 What exists today

| File | Purpose | Status |
|---|---|---|
| `schema.py` | Form / Node / factories | Stable |
| `clipboard_verifier.py` | Three-rules verifier | Stable, tokenisation hardened 2026-05-06 |
| `surface.py` | Shared tokeniser | New 2026-05-06 |
| `shadow_db.py` | Schema for solves + assignments + shadow vocabulary | Extended 2026-05-06 |
| `tree_matcher.py` | Tree-aware matcher (slice 1: literal/synonym + charade/deletion) | New 2026-05-06 |
| `extract_catalog.py` | Builds `catalog_v1.json` from clipboard-PASS forms | Stable |
| `runs/catalog_v1.json` | Current catalog (33 templates, 55 forms) | Regeneratable |
| `runs/match_charade_deletion_outer_synonym.py` | Bespoke per-signature probe | Retained for per-signature validation |
| `TREE_MATCHER_DESIGN.md` | Tree-matcher API + algorithm | Reference |

### 6.2 What's missing

| Piece | Notes |
|---|---|
| Tree-matcher coverage of remaining ops | anagram, reversal, container, hidden, acrostic, homophone, double_definition, spoonerism. Each is a small extension following the deletion template. |
| Tree-matcher coverage of remaining leaf types | abbreviation, positional, homophone, raw. |
| Charade link-word slot support | charade splits today require contiguous spans with no leftovers between children. Most real clues have connectors inside the wordplay. |
| `word_analyzer` integration | The matcher does lazy DB lookups today; should use production's `word_analyzer.analyze_words/_phrases` to pre-filter binding candidates. |
| Cascade driver | Single entry point `solve_clue_parallel(clue_id, clue_text, answer, db, ref_db, dd_graph, shadow_conn)` orchestrating phases 0a → 1, with shadow_db writes. |
| Per-phase routing-hint adapters | Tiny functions that call production's detector and return a mechanism-class hint. |
| Spoonerism factory in schema.py | One-line addition. |
| Review surface for shadow_db | CLI or report so the user can scan shadow rows and approve / reject before promotion. |
| Catalog templates for ops production solves at phase 0/0.5 | Hidden, spoonerism, DD, anagram, etc. They will accrue once the matcher can produce forms in those shapes and runs against a corpus. |

-----

## 7. Production reuse — explicit list

The parallel system reuses these production components without
modification:

- `signature_solver/db.py::RefDB` — live DB access (`get_synonyms`,
  `get_abbreviations`, `get_homophones`, `get_indicator_types`,
  `is_definition_of`, `_word_variants`).
- `signature_solver/solver.py::extract_definition_candidates` —
  definition extraction.
- `signature_solver/solver.py::_normalize_clue` — smart-quote /
  diacritic normalisation (mirrored in `surface.py`).
- `signature_solver/word_analyzer.py::analyze_words /
  analyze_phrases` — per-word role pre-analysis (planned, not yet
  wired into tree-matcher).
- `backfill_ai_exp/backfill_dd_hidden.py::try_hidden` — hidden
  detector (Phase 0a).
- `backfill_ai_exp/backfill_dd_hidden.py::generate_dd_hypotheses` —
  DD detector (Phase 0c).
- `backfill_ai_exp/backfill_dd_hidden.py::build_graph` — DD/synonym
  graph builder (pre-req for 0a + 0c).
- `sonnet_pipeline/solver.py::try_spoonerism_v2` — spoonerism
  detector (Phase 0b).
- `backfill_ai_exp/batch_v1_solver.py::try_anagram / try_charade /
  try_container / try_deletion / try_reversal / try_acrostic /
  try_homophone / find_definition / solve_without_definition` —
  Phase 0.5 op detectors.

The parallel system does **not** reuse:

- `signature_solver/solver.py::solve_clue` — this is the flat-token
  catalog-walking matcher; replaced by the tree-aware matcher.
- `signature_solver/matcher.py / base_matcher.py /
  positional_matcher.py` — flat-token matchers; replaced.
- `sonnet_pipeline/verify_explanation.py::ExplanationVerifier` — the
  current Phase-0.5 gate; replaced by clipboard verifier.

The parallel system does **not write to** the live DB. The live DB
is read-only. All writes go to the shadow DB.

-----

## 8. Pieces of related detail in `memory/`

These memory files carry detail that fed into this design. They are
not authoritative; this document is. They remain as the working
record of how each decision was reached.

- `project_purpose_ux.md` — the UX framing
- `framing_catalog_is_highlighting.md` — yellow-highlighter model
- `framing_catalog_not_parser.md` — catalog additions, not parser
  tweaks
- `framing_synonym_qualifiers.md` — preserve qualifiers
- `framing_indicator_semantic_faithfulness.md` — operator matches
  indicator semantics
- `framing_signature_runs_against_clue_text.md` — clue text not blog
- `framing_parallel_of_production.md` — parallel-of-production
  discipline
- `framing_preservation_schema.md` — solves + assignments + shadow
  vocabulary provenance
- `framing_full_cascade_parallel.md` — full cascade, refined to
  detectors-not-solvers
- `framing_adapter_full_word_accounting.md` — adapters account for
  every word
- `feedback_reuse_production_components.md` — never reinvent
- `feedback_tokenisation_lesson.md` — apostrophe-s is not a link word
- `feedback_llm_highlighter_models.md` — Sonnet > Haiku at 5x cost
- `feedback_small_batches.md` — small batches, learn between them

-----

## 9. Decision log

Each entry records a decision taken, when, and why. Newest at the
bottom. Once an entry has been incorporated into the body of this
document, mark it [woven].

- **2026-05-05 [woven]** — clipboard verifier settled with three
  rules + fodder integrity. Strict_verifier and earlier
  enumerator-trial verifiers retired.
- **2026-05-05 [woven]** — v1 catalog built (33 distinct templates
  from 60 clipboard-PASS forms). Templates frozen as the structural
  vocabulary the parallel system walks.
- **2026-05-06 [woven]** — purpose framed as UX; broad-brush
  wordplay labels rejected.
- **2026-05-06 [woven]** — catalog work framed as
  yellow-highlighter alignment; throughput, not cleverness.
- **2026-05-06 [woven]** — shadow_db schema extended with `solves`
  and `solve_assignments` tables, plus clue_id/solve_id provenance
  on existing vocabulary tables.
- **2026-05-06 [woven]** — tree-aware matcher built (slice 1:
  literal/synonym leaves, charade/deletion ops). Replaces production
  flat-token matchers in the parallel system.
- **2026-05-06 [woven]** — tokenisation switched to whitespace-split
  + punctuation strip + apostrophe preservation. "s" removed from
  LINK_WORDS.
- **2026-05-06 [woven]** — cascade architecture: replicate every
  production phase; production engines are detectors not solvers; one
  matcher does the work; verifier gates universally.
- **2026-05-06 [woven]** — adapter discipline: every adapter accounts
  for every word; reject rather than half-honest form.
- **2026-05-06 [woven]** — early-exit: first PASSing template wins
  per clue; no further attempts; for unsolved clues every template
  is tried and FAILs feed enrichment.
