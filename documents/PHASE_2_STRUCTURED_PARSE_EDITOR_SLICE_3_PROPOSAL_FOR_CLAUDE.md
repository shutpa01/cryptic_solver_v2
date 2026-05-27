# Phase 2 Structured Parse Editor: Slice 3 Proposal for Claude

## Goal

Add one reusable operation shape for single-input transformations, without
building one bespoke editor per crossword mechanism.

This slice should cover the next structural capability after charade and
container:

- reversal
- anagram

Only if validation remains small and clean, deletion can be sketched but not
implemented in this slice.

## Current State

Already implemented:

- Manual structured parse storage
- Manual structured parse merge into WFW display
- Container editor and route
- Charade editor and route
- Pieces own answer tile colours
- Operations validate transformations but do not claim answer tiles
- Filler/link words render as structural, with no answer tiles
- 1-based answer boxes at the structured parse boundary

Do not touch:

- Stage Three
- WFW display adapter
- Solver/proof storage
- Existing graph editor
- Existing container behaviour
- Existing charade behaviour

## Proposed Data Shape

Add support for a generic single-input operation.

For reversal:

```json
{
  "id": "op1",
  "type": "reversal",
  "clue_text": "rejected",
  "clue_word_positions": [2],
  "input_piece_id": "piece1",
  "result": "NIK",
  "colour": "pink"
}
```

For anagram:

```json
{
  "id": "op1",
  "type": "anagram",
  "clue_text": "wild",
  "clue_word_positions": [1],
  "input_piece_id": "piece1",
  "result": "TEAM",
  "colour": "blue"
}
```

The input piece keeps its own answer boxes, letters, and colour. The operation
validates that the claimed result is derivable from the input piece's letters.

## Validation Rules

In `_validate_structured_parse`:

### Reversal

- `input_piece_id` must exist.
- `result` must be present.
- `result == input_letters[::-1]`.
- The piece's answer boxes/letters must still match the final answer at those
  boxes.
- Operation colour must be valid if supplied.
- Operation does not claim answer boxes.

### Anagram

- `input_piece_id` must exist.
- `result` must be present.
- `sorted(result) == sorted(input_letters)`.
- `result` length must equal input letters length.
- The piece's answer boxes/letters must still match the final answer at those
  boxes.
- Operation colour must be valid if supplied.
- Operation does not claim answer boxes.

## Open Design Question

Should the piece `letters` for reversal/anagram be the pre-operation fodder or
the post-operation result?

My recommendation:

For this slice, keep `piece.letters` as the letters occupying the final answer
boxes, because merge/display already depends on that. Add `operation.input_letters`
only if Claude thinks we need to distinguish fodder from answer result now.
Avoid changing merge semantics in this slice.

This is the main design fork Claude should review.

## Admin Route

Add:

`POST /admin/structured-parse/<clue_id>/single-operation`

Fields:

- definition clue words
- piece clue words
- relationship
- letters produced / final letters in answer boxes
- answer boxes
- piece colour
- operation type: reversal or anagram
- operation indicator words
- operation colour
- optional filler/link words

Validation should reject incomplete rows clearly.

## UI

Add a compact form in `web/templates/clue.html` labelled:

`Reversal / anagram parse editor`

It should sit below the charade editor and above the container editor.

Do not remove or change the tested charade/container forms.

## Tests / Verification

Required direct checks:

1. Valid reversal:
   - answer contains `NIK`
   - piece final letters `NIK`
   - operation type `reversal`
   - operation validates against the reverse/fodder design chosen above

2. Invalid reversal:
   - claimed result is not reverse of input letters

3. Valid anagram:
   - same letters, different order accepted

4. Invalid anagram:
   - missing/extra letter rejected

5. Existing SWALLOW container test still passes:
   - `answer_links[0] = piece_0`
   - `answer_links[5] = piece_0`
   - `answer_links[1] = piece_1`

6. Existing charade complete-coverage validation still passes.

7. Filler still produces no answer tile claims.

## Anti-patterns To Avoid

- Do not create a separate storage model per clue type.
- Do not make operations own answer tile colours.
- Do not force scoring by bypassing DB-fact checks.
- Do not touch Stage Three or WFW display candidate colouring.
- Do not mutate existing clue data during verification.
- Do not use live DB clue rows for route tests unless state is restored exactly.

