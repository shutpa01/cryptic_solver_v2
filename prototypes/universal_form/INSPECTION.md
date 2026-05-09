# Universal Form — Pre-build Inspection (2026-05-02)

Findings from inspecting the live DBs and code, gathered before any prototype
file is written. Goal: ground the schema/translator/verifier scope in what
actually exists, not what I'm guessing exists.

## 1. The architectural mismatch the form fixes

- Every generator writes a **flat** structured representation into
  `structured_explanations.components` (JSON column, 180,904 non-null rows of
  181,345 total). The shape is roughly:
  ```json
  { "ai_pieces": [...], "assembly": {"op": "...", ...},
    "wordplay_type" or "wordplay_types": "..." }
  ```
- The verifier (`sonnet_pipeline/verify_explanation.py`) **does not read
  `components` at all**. Every check parses prose out of `clue_text`,
  `definition`, `wordplay_type`, and `ai_explanation`. CHECKs 1–8 and 4b–5e are
  all regexes against `ai_explanation`.
- So the contract is broken in exactly the way the universal form plan says:
  generators emit structured data, verifier scrapes prose. The universal form
  closes the loop by giving both sides the same recursive tree.

## 2. Live `structured_explanations` shape

Schema (`data/clues_master.db`):

```
id, clue_id, definition_text, definition_start, definition_end,
wordplay_types TEXT (JSON list-of-strings),
components TEXT (JSON object — see below),
model_version TEXT,
confidence REAL,
created_at, updated_at,
source, puzzle_number, clue_number
```

19 distinct `model_version` values, top:

| model_version | rows |
|---|---|
| mechanical_parse | 72,854 |
| signature_solver_v1 | 52,546 |
| mechanical_v1_batch | 24,384 |
| v1_dd_backfill | 9,725 |
| v1_hidden_backfill | 7,840 |
| mechanical_hidden | 5,813 |
| claude_review | 1,709 |
| parser_fifteensquared_v1 | 1,636 |
| haiku_sonnet_tiered_v1 | 1,519 |
| manual_edit | 711 |
| fifteensquared+haiku | 603 |
| signature_solver_enriched_v1 | 592 |
| mechanical_v1 | 545 |
| manual_approve | 407 |
| tftt+haiku | 350 |
| mechanical_dd, tutorial, reverified, mechanical_spoonerism | <100 each |

`wordplay_types` is a JSON array of strings. Top values are simple
`["charade"]` (50,755), `["anagram"]` (32,935), `["double_definition"]`
(18,149), `["hidden"]` (11,081). Compound combinations like `["charade",
"container"]` (6,068), `["anagram","charade"]` (5,152), `["charade",
"reversal"]` (3,098) are common; ~20 rare combinations exist with
single-digit counts.

26,410 rows have `wordplay_types = NULL` — mostly `mechanical_v1_batch`,
`parser_fifteensquared_v1`, `manual_approve`, `manual_edit`, `reverified`.

## 3. Components shape — the translator's input surface

Top-level keys (count in the wild):

```
ai_pieces      180,904   (always present, sometimes [])
assembly       180,787   (almost always present; missing in the 2 reverified rows etc.)
wordplay_types 152,009   (plural array — the modern key)
sig_explanation 53,135   (sig-solver only — pre-rendered prose)
wordplay_type   28,916   (singular string — the legacy key)
source          3,340    (claude_review / fifteensquared+haiku tag)
```

`assembly.op` distribution (28 distinct values, top in 1.35M total assemblies):

```
charade            68,983
anagram            51,645
double_definition  18,156
container          13,852
hidden             11,392
homophone           3,994
reversal            3,329
hidden_reversed     3,192
deletion            2,458
cryptic_definition  2,188
hidden_in_word        741
acrostic              664
deletion+anagram       66
spoonerism             16
charade, container     10
reversal_container      9
substitution            4
outer_deletion          4
alternate               3
container_reversal      2
linked                  2
hidden_word             2
alternating             2
fail                    2
container_with_deletion 1   (and a few other singletons)
<no_op_or_no_assembly>  205
```

Per-piece `mechanism` values (top, 220k pieces total):

```
synonym        155,963
anagram_fodder  89,763
abbreviation    31,403
hidden          13,440
first_letter    13,000
deletion         3,609
reversal         2,673
last_letter      2,285
hidden_reversed  1,833
literal          1,668
sound_of           923
core_letters       546
indicator          296
alternate_letters  258
even_letters        11
container            7
substitution         7
odd_letters          7
outer_letters        4
cryptic_definition   4
homophone            4
truncation           2
second_letter        2
spoonerism           2
```

Plus ~80 polluted one-off mechanism strings like
`deletion:UREA:remove last letter` and `reversal:EP` — data leakage where the
generator crammed extra fields into the mechanism slot. Tiny in count, signals
the flat form's brittleness.

### Per-generator notes (translator implications)

- **mechanical_parse / mechanical_v1 / mechanical_v1_batch / mechanical_hidden /
  mechanical_dd / mechanical_spoonerism**: clean shape; `assembly.op` always
  matches one of the conventional types; pieces well-formed. Translates
  cleanly when the assembly is single-op.
- **signature_solver_v1 / signature_solver_enriched_v1**: cleanest shape;
  `assembly` carries extra fields (`fodder`, `order`, `inner`, `outer`);
  includes pre-rendered `sig_explanation` prose useful as a reference.
- **v1_dd_backfill / v1_hidden_backfill**: `assembly` carries `left_def`/
  `right_def` (DD) or `words` (hidden) — direct map to definition phrases.
- **parser_fifteensquared_v1**: pieces have weird `clue_word` annotations like
  `"=man, i.e. male subject pronoun"`. Translator must strip leading `=` and
  trailing comma-separated metadata.
- **claude_review**: often has `ai_pieces: []` — relies on `clues.ai_explanation`
  prose. Cannot translate without re-derivation; flag as "unconvertible".
- **haiku_sonnet_tiered_v1**: a few rows have `components: []` (a list, not
  object) — schema-violating; flag as malformed.
- **manual_edit / manual_approve**: components often `NULL`; must flag.
- **fifteensquared+haiku / tftt+haiku**: `assembly` sometimes missing entirely;
  `wordplay_type` (singular) at top with pieces only. Translator must
  synthesise an assembly from the wordplay_type.
- **tutorial**: clean anagram with `gives: "ANSWER"` field.
- **reverified**: confidence-only update, no components.

### What the flat form CAN'T represent (the killer)

The current form has **one assembly op** + **flat pieces**. Compound mechanisms
break the model:

- Compound assembly ops in the wild (`deletion+anagram`, `container_reversal`,
  `outer_deletion`, `container_with_deletion`) are encoded as ad-hoc combined
  strings — there's no shared decomposition rule.
- Pieces with `mechanism: "deletion"` or `mechanism: "reversal"` carry the
  result letters but **not** the source word the deletion/reversal was
  applied to, nor the indicator. The translator either flags them as opaque
  leaves or has to recover the source by reading sig_explanation prose.

The new tree handles all of this natively because each operation is its own
node with its own indicator and its own children.

## 4. RefDB (`data/cryptic_new.db`) — the verifier's bridge inputs

| Table | Rows | Used for |
|---|---|---|
| `definition_answers_augmented` | 643,819 | `(definition, answer)` lookup — the bridge for definition node + &lit case |
| `synonyms_pairs` | 1,351,564 | `(word, synonym)` lookup — synonym leaves; also fallback for definitions and abbreviations |
| `synonyms` | 78,669 | `(word, synonyms_json)` legacy thesaurus source — wider but messier |
| `indicators` | 5,267 | `(word, wordplay_type, subtype, confidence)` — bridge for non-literal node indicators |
| `wordplay` | 2,431 | `(indicator, substitution, category)` — abbreviation-style lookups (e.g. `"north" -> "N"`) |
| `homophones` | 1,113 | `(word, homophone)` bidirectional — homophone bridge |
| `substitutions` | 7 | `(original_word, substitution, context)` — domain abbrev (`airline -> BA`) |

Every check in the new verifier maps to a small set of these:

- **Synonym leaf**: `synonyms_pairs` (both directions) + fallback in `synonyms` JSON
- **Abbreviation leaf**: `wordplay` then `synonyms_pairs`
- **Definition node**: `definition_answers_augmented` then `synonyms_pairs`
- **Indicator on any non-literal node**: `indicators` filtered by op type
- **Homophone leaf or op**: `homophones`
- **Positional leaf** (first/last/middle/etc.): mechanical only — no DB lookup

The verifier already has cached DB lookup helpers (`is_synonym`,
`is_abbreviation`, `is_indicator`, `is_homophone`, `definition_matches`).
Prototype can call into a cleaned-up version of these — same DB, same SQL —
just feed it the tree instead of the prose.

## 5. The current verifier — checks to map / drop

`sonnet_pipeline/verify_explanation.py` (1725 lines).

| Check | What it does | New form replacement |
|---|---|---|
| 1 | `definition_matches(definition, answer)` | Same call, but on `form.definition.phrase` |
| 2 (synonym) | regex piece annotations → `is_synonym` | Bridge, on synonym leaves |
| 2 (abbreviation) | regex piece annotations → `is_abbreviation` | Bridge, on abbreviation leaves |
| 2c | source-word-in-clue check on regex-extracted piece sources | Subsumed by **residue**: every leaf's `source_word` must be in surface words |
| 3 | regex-extracts `LETTERS(...)` pieces, joins, compares to answer | **Assembly** walk |
| 4 (hidden span) | regex `hidden in "X"`, span-in-clue, uppercase letters check | Bridge on hidden node + assembly walk |
| 4b (DD) | regex `double definition: w1 = ANSWER, w2 = ANSWER`; both windows in DB; remainder all link words | DD node bridge: both window definitions in DB; residue check for link words |
| 4c (CD) | clue text or definition matches answer in DB | CD node bridge — same DB call |
| 4d (homophone) | regex `ANSWER sounds like WORD` | Bridge on homophone leaf or op |
| 4e (spoonerism) | regex first 2 piece words, mechanically swap initial consonants | Assembly walk on spoonerism node |
| (anagram) | regex `anagram of X = ANSWER`, sorted-letters check, fodder-in-clue check | Assembly walk on anagram node + residue |
| (reversal) | regex `WORD reversed/reversal`, reversal check (or piece-in-charade fallback) | Assembly walk on reversal node |
| (container) | regex uppercase pieces, try inserting one into the other | Assembly walk on container node |
| 5 (positional) | regex `LETTERS (first/last/etc letters of "X")`, mechanical | Assembly walk on positional leaf |
| 5a2 (reversal source) | regex `LETTERS (reversal of "X")`, DB lookup for reversed synonym/abbr | Assembly walk + bridge on reversal-of-leaf node |
| 5b/5c/5d (deletion variants) | three regex patterns, mechanical deletion checks | Assembly walk on deletion node |
| 5e (silent piece) | regex every `WORD (content)`, content must match a known prefix | Obsolete — the form has no "silent" pieces, every node has a known op |
| 6 (trivial) | regex single `ANSWER (synonym of definition)` with no wordplay | Tree shape check: tree has only one synonym leaf equal to definition |
| 7 (op-level indicator) | per declared wordplay_type, find `[type: "X"]` annotation, validate | Bridge on every non-literal node's indicator |
| 7b (clue-side indicator scan, uncommitted) | scan clue for indicators of any op-type the parse doesn't address | Subsumed by **residue**: an indicator word in the surface that no node claims becomes unaccounted |
| 8 (word coverage) | role-by-role classification of clue words via prose regex | **Residue**: union of every node's source_word + indicator + definition.phrase = surface words minus link words |

**Net effect**: the new verifier replaces the regex zoo (CHECKs 2/3/4/5 plus
their sub-variants) with a single tree walk + a handful of typed DB lookups.
CHECKs 1, 4b, 4c, 4d, 7 carry over conceptually but operate on the form
instead of regex output. CHECKs 5e, 6, 7b, 8 collapse into structural
properties of the form (no more "silent piece" possible if every leaf has a
declared op; coverage falls out of residue).

## 6. Existing renderers — useful reference, not reusable

Two flat-form renderers exist:

- `web/explanation_builder.py:build_positional_explanation` — HTML output,
  per-op special cases (DD, hidden, anagram, container).
- `web/models.py:_build_explanation` — text fallback, also per-op flat
  cases.

Both consume the flat `components` form. Neither walks a tree. **Prototype
needs a fresh renderer**, but the prose conventions in these two files are the
right starting point for what each operation should "read like" in the
universal renderer's output.

## 7. Generators that emit components — write paths

13 distinct call sites that `INSERT INTO structured_explanations`. Schema is
consistent across them: `(clue_id, definition_text, [definition_start/end,]
wordplay_types, components, model_version, confidence, [source/puzzle_number/
clue_number])`. The prototype does not touch these. The translator reads
their output.

## 8. Signature solver catalog — stays internal

The catalog (`signature_solver/base_catalog.py` + 24 sibling files, 8087
lines) is the sig-solver's internal pattern matcher. It encodes compound
operations as flat names (`reversal_charade`, `container_charade`,
`anagram_container`, `container_reversal`) with `(F, I)` slot patterns and
mappings to indicator types. Per the universal form plan it stays as-is and
emits the flat form on the way out — exactly what `signature_solver_v1` and
`signature_solver_enriched_v1` rows already show.

The prototype does **not** need to translate from the catalog. It translates
from `components` (which sig-solver writes alongside everything else).

## 9. Prototype scope — finalised

`prototypes/universal_form/` (no imports from live modules, read-only DB):

1. **`schema.py`** — `Form`, `Node`, `Leaf` dataclasses; JSON
   serialise/deserialise; validator that rejects malformed trees (e.g. a
   non-literal node missing children, an indicator on a literal leaf).
2. **`renderer.py`** — `render(form) -> str`. Produces a human-readable
   explanation. Reference patterns lifted from the two existing flat-form
   renderers but walks the tree.
3. **`verifier.py`** — `verify(form, clue_text, answer) -> dict`. Four
   checks: assembly, bridge, mechanism, residue. Plus the carry-overs
   (definition, &lit-cap-at-LOW). Read-only `cryptic_new.db` connection,
   reuses the SQL patterns from the live verifier's helpers but operates on
   the form, not prose.
4. **`translator.py`** — `flat_to_form(components_dict, definition_text,
   wordplay_types, clue_text, answer) -> Form | UnconvertibleReport`.
   - Single-op assemblies: direct map.
   - Compound `wordplay_types` with single `assembly.op`: nest using a
     priority rule (outer-op first per the schema).
   - Compound `assembly.op` strings (`deletion+anagram`,
     `container_reversal`, etc.): split by separator and nest.
   - Hidden / DD special cases: use `assembly.words` / `left_def, right_def`.
   - Pieces with `mechanism` of `deletion`/`reversal`/etc. but no source:
     emit a leaf with `op` set and `source_word=null`, flag the row as
     `partial_translation`.
   - `claude_review` / `manual_edit` rows with empty pieces: flag as
     `requires_re_derivation`.
   - `parser_fifteensquared_v1` weird annotations: strip leading `=` and
     comma-separated tail.
   - Malformed `components: []`: flag.
5. **`harness.py`** — `run(filter)` where filter is a SQL fragment selecting
   clues to simulate. Pulls each clue + structured_explanations row;
   translates; renders; verifies. Per clue, emits:
   - clue/answer
   - original `ai_explanation` vs new rendered text
   - original verifier score vs new verifier score
   - residue/bridge breakdown (which words unaccounted, which leaves failed
     bridge, etc.)
   - translation status (`clean`, `partial_translation`,
     `requires_re_derivation`, `malformed`)
6. **`run_simulation.py`** — CLI: `python -m prototypes.universal_form.run_simulation
   --source telegraph --puzzle 31229` or `--limit 200 --random` or
   `--full-corpus`. Writes per-clue report to stdout + summary to a JSON
   file in `prototypes/universal_form/runs/`.

## 10. Build order

1. schema.py + tests (round-trip JSON)
2. renderer.py + sample-tree round-trip render
3. verifier.py — assembly walk first (pure function), then bridge/mechanism/
   residue with read-only RefDB
4. translator.py — start with single-op assemblies (~95% of rows), add
   compound and special cases incrementally; every ambiguous row gets a
   flag, never a guess
5. harness.py + run_simulation.py
6. First simulation run: telegraph 31229 (the puzzle in the handoff). Check
   that the form-based verifier scores the BACH and BRIGHTEYED parses
   correctly (the cases the uncommitted check 7b was added to catch).
7. Then expand: full puzzle → puzzle range → full corpus.

## 11. Open questions to confirm with user before building

1. **Translator policy for ambiguous compound operations**: when
   `wordplay_types: ["charade", "container"]` and `assembly.op = "container"`,
   which is outer? Most likely the container wraps a charade-of-pieces, so
   container is outer — but for the few `container_charade` cases the order
   may flip. Propose: trust `assembly.op` as outer when present; flag any
   conflict and treat as `partial_translation`.
2. **Operation name normalisation**: the tree uses 16 op names from the
   plan. The flat form uses ~28 assembly ops + ~25 piece mechanisms.
   Propose mapping (e.g. `hidden_in_word` → `hidden` with the position
   encoded as a sub-leaf? or a distinct op?). Won't decide unilaterally.
3. **Should the prototype also try to translate from `ai_explanation` prose
   for rows where components is unconvertible?** Adds significant scope —
   would replicate parts of the current verifier's regex zoo backwards
   (prose → form). Default: don't. Flag those rows as
   `requires_re_derivation` and exclude from comparison.
4. **Comparison metric**: how do we judge the new verifier "agrees" with
   the old one? Propose: per clue, classify as
   `same_verdict`/`upgraded`/`downgraded`/`untranslatable`. Headline number
   = % `same_verdict` excluding untranslatable. Worth showing a
   distribution of score deltas too.
