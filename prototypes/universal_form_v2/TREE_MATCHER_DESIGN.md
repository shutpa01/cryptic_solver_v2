# Tree-aware matcher — design spec

**Status:** draft for review (2026-05-06)
**Replaces in production:** `signature_solver/solver.py` and its
flat-token catalog. The matcher built per this spec is the future
production Phase 1 matcher.

## Purpose

Given a catalog entry with a tree-shape signature (e.g.
`charade(deletion[outer](literal), synonym)`) and a clue with a known
answer, enumerate every structurally-plausible Form that the tree can
produce against the clue. Hand each Form to the clipboard verifier;
keep the PASSes.

## Public API

```python
def match_signature(
    catalog_entry: dict,
    clue_text: str,
    answer: str,
    db: RefDB,
    shadow_conn: Optional[sqlite3.Connection] = None,
) -> List[Form]:
    """Walk the catalog entry's structure tree against the clue.

    Each returned Form is structurally plausible:
      - every leaf has a clue span and a derived value
      - every required indicator has a clue word with DB authority
      - definition synonymises the answer
      - all clue words are accounted for as leaf-source / indicator /
        link / definition (residue check is left to the verifier)

    The matcher does NOT verify per-rule correctness; that is the
    verifier's job. The matcher's contract is structural plausibility,
    nothing more.
    """
```

`catalog_entry` is one entry from `catalog_v1.json` — its `structure`
field gives the tree, `indicator_slots` gives the paths to nodes that
need an indicator, `leaf_kinds` gives the leaf type sequence.

## Algorithm

1. Tokenise clue into ordered surface words.
2. **Definition candidates from the production engine.** Call
   `signature_solver.solver.extract_definition_candidates(clue_words,
   answer, db, max_def_words=4)`. This is reused unchanged from
   production: it tries 1-4 words from each end of the clue, checks
   `definition_answers_augmented` and `synonyms_pairs` via
   `db.is_definition_of`, and returns ordered `(definition_phrase,
   wordplay_words)` tuples. The matcher does **not** invent its own
   definition extraction.
3. For each `(definition_phrase, wordplay_words)` returned by step
   2, treat the wordplay_words list as the wordplay window and
   **recursively bind** the structure tree:
   - The root binds to the entire window.
   - For an op-node: split the window into ordered contiguous
     sub-spans, one per child, plus one position for the indicator
     (when the op needs one). Recurse into each child with its
     sub-span.
   - For a leaf-node: bind directly to the sub-span; mechanism
     determines the value.
4. For each completed binding, **assemble** the Form (per
   `schema.py` factories). Include the definition phrase as
   `definition.phrase`. Any sub-span containing only allow-listed
   words is recorded as `link_words`.
5. Yield each Form. Cap the total yield at `MAX_BINDINGS = 100` per
   `(clue, catalog_entry)` to avoid runaway combinatorics.

The walk is depth-first with pruning: bind one child, recurse, fail
fast if the child can't be satisfied.

## Mechanism table for leaves

| mechanism    | source-span size | value derivation                   | DB requirement                                             |
|--------------|------------------|------------------------------------|------------------------------------------------------------|
| literal      | one word         | uppercase letters of the word      | none                                                       |
| synonym      | one or more      | DB lookup, value chosen to fit     | synonyms_pairs (live ∪ shadow)                             |
| abbreviation | one or more      | DB lookup                          | wordplay(category=abbreviation) or synonyms_pairs          |
| positional   | one word         | extract by `positional_kind`       | indicators DB carries authority for the kind via `parts`/`acrostic`/`alternating` |
| homophone    | one or more      | DB lookup                          | homophones                                                 |
| raw          | one word         | uppercase letters of the word      | none (treated like `literal` for fodder integrity)         |

## Op handling

| op                | children                     | indicator? | indicator DB type(s)         |
|-------------------|------------------------------|------------|------------------------------|
| charade           | ≥ 2, ordered                 | no         | —                            |
| anagram           | ≥ 1 literal/raw              | yes        | anagram                      |
| reversal          | exactly 1                    | yes        | reversal                     |
| container         | exactly 2 (outer, inner)     | yes        | container or insertion       |
| deletion          | exactly 1                    | yes        | deletion or parts            |
| hidden            | ≥ 1 literal                  | yes        | hidden                       |
| acrostic          | ≥ 2 literal                  | yes        | acrostic or parts            |
| homophone (op)    | exactly 1                    | yes        | homophone                    |
| double_definition | exactly 2 synonym            | no         | —                            |

For ops requiring an indicator, the matcher considers any clue word
within ±1 position of any leaf in the op's subtree. (Looser than
strict adjacency but tighter than "anywhere in the wordplay" — the
verifier's `mechanism.indicators` check is the final word.)

## Performance bounds

- `MAX_BINDINGS = 100` yields per (clue, catalog_entry).
- Per-call memoisation:
  - phrase-lowercase → list of synonym values
  - word-lowercase → list of indicator-type tuples
- No global caches; each call is self-contained.
- Soft-fail on overshoot: if MAX_BINDINGS is hit, return what's been
  yielded and a flag — never raise.

## Integration points

- `prototypes/universal_form_v2/schema.py` — Form / Node factories
- `prototypes/universal_form_v2/clipboard_verifier.py` — caller runs
  this on each yielded Form
- `prototypes/universal_form_v2/shadow_db.py` — caller writes solves
  + assignments + enrichments
- `signature_solver/db.py::RefDB` — the live DB read interface
- `signature_solver/solver.py::extract_definition_candidates` —
  reused as-is for definition extraction; not reimplemented

The matcher itself imports from schema, reads RefDB, and calls
`extract_definition_candidates`. It does not import the verifier or
shadow_db. The driver (separate file) wires the four together.

## Production reuse — explicit list

Per the parallel-of-production discipline, the matcher reuses these
production components without modification:

- `signature_solver.solver.extract_definition_candidates` —
  definition extraction
- `signature_solver.db.RefDB` — live DB access (`get_synonyms`,
  `get_abbreviations`, `get_homophones`, `get_indicator_types`,
  `is_definition_of`)

If production has an equivalent function for a step the matcher
needs, the matcher uses it. New code is for the genuine gaps.

## What the matcher does NOT do

- Does not propose the answer (answer is given).
- Does not judge correctness — verifier does.
- Does not modify any DB row, live or shadow.
- Does not write any solve / assignment / enrichment.
- Does not reinterpret a binding to make it pass — yields what the
  structure allows; verifier filters.
- Does not handle mid-clue definitions, &lit clues, or the
  cryptic_definition op in v1. These are deliberate scope cuts that
  can be lifted in subsequent versions.

## Module structure

| file                                                         | role                                  |
|--------------------------------------------------------------|---------------------------------------|
| `prototypes/universal_form_v2/tree_matcher.py`               | the matcher (this spec)               |
| `prototypes/universal_form_v2/TREE_MATCHER_DESIGN.md`        | this design doc                        |
| `prototypes/universal_form_v2/parallel_runner.py` (later)    | driver: walk catalog_v1, write shadow |
| `prototypes/universal_form_v2/test_tree_matcher.py` (later)  | tests, including ELITE canary          |

## First implementation slice

The smallest end-to-end useful slice:

1. Implement `match_signature` for **literal, synonym** leaves and
   **charade, deletion** ops only (covers the ELITE shape).
2. Test against the canary: catalog entry =
   `charade(deletion[outer](literal),synonym)`, clue = "Cream tea
   uncovered, with fewer calories?", answer = "ELITE". Must produce
   the form we manually built.
3. Wire into a tiny driver that writes to shadow_db and runs the
   verifier.

After that proves out, add ops/mechanisms one at a time.

## Open questions

(Things this spec deliberately leaves to the implementation, but
worth flagging.)

- **Multi-word literals / synonyms / etc.** — the table above allows
  multi-word source spans. The combinatorial blow-up is bounded by
  the wordplay window length and `MAX_BINDINGS`. First version: yes
  multi-word, capped.
- **Indicator placement priors** — strictly ±1 first, but some
  setters use a more remote indicator. Loosening should be a
  deliberate change, not a silent one.
- **Tie-breaks** when the matcher could yield multiple PASSing forms
  for the same clue — record all, let the driver / review surface
  pick.
