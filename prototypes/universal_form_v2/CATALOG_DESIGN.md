# Catalog design — v0 spec for review

## Goal

A **catalog entry** is a structural wordplay template — fully abstract,
no specific values. Given a cold clue (no blog), the production solver
applies each template to attempt a solve.

Catalogs are extracted only from forms that pass the **strict verifier**.
Anything less and the catalog encodes a lie that the solver will repeat
forever.

## Data shape — minimal viable catalog (v0)

```json
{
  "id": "charade(synonym,abbreviation)",
  "structure": {
    "op": "charade",
    "children": [
      {"op": "synonym",      "leaf": true},
      {"op": "abbreviation", "leaf": true}
    ]
  },
  "indicator_slots": [],
  "leaf_kinds": ["synonym", "abbreviation"],
  "frequency": 5,
  "examples": [
    {"answer": "LINGOES",  "clue": "Try piercing wrinkles in tongues"},
    {"answer": "SJAMBOK",  "clue": "..."}
  ]
}
```

### Fields

- **`id`** — canonical signature string (deterministic from `structure`).
  Two forms with the same structure share an id.
- **`structure`** — the full tree shape, no values. Each node has
  `op` and either `children` (for non-leaves) or `leaf: true` (for
  leaves). Positional leaves carry `positional_kind`; deletion nodes
  carry `deletion_kind`; acrostic nodes carry `acrostic_kind`.
- **`indicator_slots`** — for each non-leaf op that requires an
  indicator, a path to it (e.g. `["root"]`, `["root", "children", 0]`).
  Tells the solver: "to instantiate this template, you must find an
  indicator word in the clue for each of these slots."
- **`leaf_kinds`** — flat list of all leaf-mechanism types. Tells the
  solver: "you need this many synonyms / abbreviations / literals / etc."
- **`frequency`** — how many strict-PASS forms produced this signature.
  Higher = more confidence / more common.
- **`examples`** — sample clues + answers that produced this template.
  For human inspection / regression testing.

## How the solver uses a catalog entry on a cold clue

For a clue with `n_words` words:

1. **Tokenise** the clue.
2. **Identify candidate roles** for each word (definition? synonym
   source? indicator? literal-fodder?). Multi-word phrases too.
3. For each catalog entry whose `leaf_kinds` count matches the clue's
   piece count and whose `indicator_slots` have potential anchor words
   in the clue:
   - **Try to instantiate**: assign clue spans to slots.
   - **Attempt to assemble**: run the template's tree mechanics on
     the candidate values; check if any combination produces a
     known-valid answer.
4. Score by: (a) all words accounted for; (b) all leaves DB-justified;
   (c) all indicators DB-justified; (d) assembly produces a valid
   answer.

(This is exactly what the strict verifier checks — the solver applies
the verifier's logic forward, the verifier applies it backward.)

## What's NOT in the catalog (deliberately)

- **Specific values** (LIVE, HARP, etc.) — those come from the live DBs
  at solve time.
- **Specific indicator words** (traumatised, principally) — same. The
  catalog says "anagram needs an anagram indicator", not "anagram is
  always 'traumatised'".
- **Surface fluency hints** — the catalog is structural only.
  Disambiguating multiple-fitting templates is a separate scoring
  problem.

## Out of scope for v0

- Compound shapes the verifier doesn't yet model (letter substitution,
  two-operand deletion).
- Confidence scoring beyond raw frequency.
- Catalog merging across publications. Right now we're Times-only.

## Minimum viable catalog from existing data

The salvage audit found **60 strictly-verified forms** in the existing
runs (57 from random 400, 3 from Times 29534). Collapsing into distinct
signatures gives ~30 templates. The v0 catalog is just those 30
serialised in the shape above.

That's the deliverable for review. If the shape is wrong, we change it
before any code consumes it.
