# Phase 2 Structured Parse Editor - Slice 2 Proposal For Claude

Date: 2026-05-28

## Context

Slice 1 implemented the first human structured parse path using a container
example:

```text
Bird, female, going over fence (7)
SWALLOW
```

The important design survived:

- JSON in `manual_structured_parses` is the authoritative manual parse.
- Answer boxes in the manual structured parse are 1-based.
- Pieces own answer tile colours.
- Operations validate combinations but do not claim answer tile colours.
- Missing DB facts block high scoring and queue/reveal DB review needs.

The container form works for SWALLOW, but it is too narrow to be the next
surface we build on directly. The next slice should not add one bespoke form
per clue type.

## User Goal

The user needs a manual parser that can correct failed clues reliably, using a
shared structured format that a human or future LLM can produce.

The user does not want:

- graph/node/edge language
- one separate editor per clue type
- cosmetic patches that make the page look solved when the evidence is only a
  candidate
- hidden assumptions that bypass cryptic invariants

## Proposed Slice 2

Build a general **N-piece source parse editor**.

This is the base parse capability underneath many clue mechanisms:

- charades
- abbreviation + synonym combinations
- literal-letter pieces
- simple source-only parses
- future operation inputs

This slice should not yet implement reversal/anagram/deletion/homophone
operation semantics. It should make the parser capable of representing any
number of answer-producing pieces and optional link/filler words, then validate
that those pieces account for the answer boxes.

In short:

```text
definition + N coloured source pieces + optional filler/link words
```

No new automatic solver behaviour. No Stage Three changes. No proof promotion.

## Why This Slice Next

Container slice proved the hard split-colour rule, but the UI is still fixed to
exactly two pieces and one container operation.

Before adding more operations, we need the editor to answer basic questions:

- Can the user add three or four pieces?
- Can the user mark clue words as filler/link without pretending they produce
  answer letters?
- Can the display show all source pieces with stable colours?
- Can DB auditing queue/check all source facts from a manual parse?
- Can reverify/rerun respect a manual parse with no operation?

This is a capability layer, not a clue-type layer.

## Example Acceptance Clue

Use a simple charade-style synthetic or real clue already in the local DB.

Suggested synthetic acceptance shape:

```text
Quiet answer (2)
PA
```

Parse:

```text
definition: answer
piece 1: Quiet -> P, boxes [1], relationship abbreviation, colour blue
piece 2: a -> A, boxes [2], relationship literal_letters, colour pink
```

Better real acceptance can be chosen by Claude/Codex if a clean DB clue exists,
but the test should not depend on a volatile live clue unless needed.

Expected display:

```text
answer      Definition
Quiet       P, blue
a           A, pink
```

Answer tiles:

```text
P -> piece_0
A -> piece_1
```

## Structured JSON Shape

Do not create a separate table or schema. Extend the existing
`manual_structured_parses.parse_json` shape.

Recommended parse:

```json
{
  "version": 1,
  "clue_id": 123,
  "answer": "PA",
  "source": "human",
  "confidence": "verified",
  "definition": {
    "id": "def1",
    "clue_text": "answer",
    "clue_word_positions": [1],
    "answer": "PA"
  },
  "pieces": [
    {
      "id": "piece1",
      "clue_text": "Quiet",
      "clue_word_positions": [0],
      "relationship": "abbreviation",
      "letters": "P",
      "answer_boxes": [1],
      "mapping": "positional",
      "colour": "blue"
    },
    {
      "id": "piece2",
      "clue_text": "a",
      "clue_word_positions": [2],
      "relationship": "literal_letters",
      "letters": "A",
      "answer_boxes": [2],
      "mapping": "positional",
      "colour": "pink"
    }
  ],
  "operations": [],
  "filler": [
    {
      "id": "fill1",
      "clue_text": "with",
      "clue_word_positions": [3],
      "role": "link"
    }
  ]
}
```

Notes:

- `operations` may be empty.
- `pieces` may have 1 to 5 entries in this slice.
- Colours should continue to map to `piece_0` ... `piece_4`.
- Do not invent separate charade operation semantics yet. If the pieces cover
  the answer boxes exactly, that is enough for this slice.

## Validation Rules

Extend `_validate_structured_parse` only as needed.

Required invariants:

1. Definition exists and has clue text.
2. Each piece has:
   - id
   - clue text
   - relationship
   - letters
   - answer boxes
   - valid colour
3. Answer boxes are 1-based.
4. A piece's letters must match its claimed answer boxes.
5. No two pieces claim the same answer box.
6. For a source-only parse with no operations, all answer boxes from 1 to
   answer length must be claimed exactly once.
7. Filler/link words never claim answer boxes.
8. Definition and filler/link words never create answer links.

Important:

The current container validation should continue to pass. Do not break SWALLOW.

## DB Fact Audit

Reuse the current DB-fact audit in `web/routes/admin.py`.

For each piece:

- `relationship == "synonym"` -> synonym fact
- `relationship == "abbreviation"` -> abbreviation fact
- `relationship == "literal_letters"` should not require DB enrichment

For definition:

- definition fact remains `definition clue_text -> answer`

For filler/link:

- no DB fact required in this slice

If DB facts are missing:

- parse is saved
- score remains blocked
- pending enrichment/review message appears as it does now

## UI Proposal

Do not expose graph concepts.

Add a second primary form section beside or below the existing container form:

```text
Manual source parse
```

Human fields:

- Definition clue words
- Piece rows:
  - clue words
  - gives letters
  - relationship
  - answer boxes
  - colour
- Optional filler/link words

For Slice 2, keep the UI simple:

- fixed 5 rows maximum
- blank piece rows ignored
- no JavaScript row builder required unless already easy
- answer boxes remain comma-separated 1-based numbers
- server derives clue word positions from clue text

This is deliberately less ambitious than a full interactive editor. It gives
the user a way to represent multi-piece parses without creating graph nodes.

The existing container form should remain for now. Do not delete it in this
slice.

## Routes

Add one new route:

```text
POST /admin/structured-parse/<clue_id>/source
```

The route:

1. Reads definition.
2. Reads up to 5 piece rows.
3. Reads optional filler/link words.
4. Builds the same `parse_dict` shape.
5. Calls `_validate_structured_parse`.
6. Saves with `write_structured_parse`.
7. Runs the existing structured parse DB-fact audit/scoring function.
8. Re-renders the manual evidence partial.

Do not write to legacy graph tables.

## Display

`merge_structured_parse_into_display` should already mostly support pieces.
Confirm it works with:

- no operations
- 3+ pieces
- filler/link blocks

If filler/link blocks are not currently rendered, add display-only blocks:

```python
{
  "kind": "LINK_BLOCK",
  "role": "surface" or "link",
  "text": filler["clue_text"],
  "span": ...
}
```

They must not create answer links.

## Files To Touch

Preferred maximum:

```text
signature_solver/manual_evidence_store.py
web/routes/admin.py
web/templates/clue.html
web/templates/partials/manual_evidence_nodes.html
```

Only touch `web/routes/clue.py` if necessary for prefill/defaults.

Do not touch:

```text
signature_solver/stage_three_proof.py
signature_solver/wfw_display_adapter.py
automatic solver files
proof attempt storage
reference DB files
```

## Tests / Verification

Minimum automated checks:

1. Existing SWALLOW container validation and merge still pass.
2. New N-piece source parse validation passes.
3. Missing answer box fails validation.
4. Duplicate answer box fails validation.
5. Wrong letter in answer box fails validation.
6. Filler/link word does not create answer links.
7. DB audit queues/checks synonym and abbreviation facts but ignores literal
   letters and filler.
8. App templates load.

Manual browser checks:

1. SWALLOW still pre-fills and verifies.
2. A source-only parse can be saved.
3. Answer tiles use different colours for different pieces.
4. Reverify does not score high if required DB facts are missing.
5. Reverify scores high only when required DB facts exist.

## Anti-Patterns To Avoid

- Do not add a separate UI for every clue mechanism.
- Do not expose node/edge/payload/group terminology.
- Do not make source-only parses score high if DB facts are missing.
- Do not use legacy graph tables as the primary storage.
- Do not create answer links from operation/filler/definition blocks.
- Do not infer a middle definition or other automatic proof behaviour here.
- Do not touch Stage Three or WFW display in this slice.

## Question For Claude

Please review whether this is the right next slice after the container editor.

Specifically:

1. Is N-piece source parsing the right capability layer before adding
   reversal/anagram/deletion operations?
2. Are the proposed validation rules strict enough?
3. Should source-only parses require complete answer-box coverage in this
   slice?
4. Should filler/link words be stored now, or deferred?
5. Are any proposed fields too technical for future LLM/human shared output?
6. Is it safe to keep the existing container form separate for one more slice?

