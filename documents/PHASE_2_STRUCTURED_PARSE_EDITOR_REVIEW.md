# Phase 2 Structured Parse Editor — Architecture Review
# 2026-05-27

This is a critical review of PHASE_2_STRUCTURED_PARSE_EDITOR_SPEC_FOR_CLAUDE.md.
It is not an implementation instruction. It amends the spec before Codex implements.

---

## 1. Clue types the schema cannot represent cleanly

**Anagram** is the serious gap. The spec says "pieces own answer colours; operations
validate combinations." That rule breaks for anagrams because the anagram operation
rearranges letters arbitrarily — you cannot assign `answer_boxes` to a fodder piece
positionally. If you try, the validation rule "letters must match the cleaned answer
at those boxes" will always fail for anagram fodder.

Fix: introduce a `mapping` field on pieces. Two values:

- `positional` (default): the nth letter of `letters` must match the answer at the
  nth element of `answer_boxes`. Used by all non-anagram pieces.
- `block`: the piece's colour covers all its claimed answer boxes but the
  letter-by-letter positional check is skipped. The operation handles correctness.
  Used for anagram-fodder pieces.

**Deletion** is a partial gap. The piece being deleted from contributes some letters
to the answer (the survivors) and not others (the deleted portion). The schema has
no way to mark which letters survive. `answer_boxes` on a deletion-source piece
should list only the surviving positions — but then `letters` should also be only
the surviving letters, which severs the connection to the clue word.

Fix: introduce `mapping: "operation_assigned"` for deletion-source pieces. The
piece's `letters` field records the full clue word. The operation determines which
letters survive and fills in `answer_boxes` at render time. The user does not
manually enter answer boxes for deletion-source pieces.

**Acrostic / first-letter extraction** is missing from the relationship list. The
spec lists "literal letters" and "hidden letters" but not initial-letter extraction.

Fix: add `initial_letters` to the relationship list.

**Multi-step operations** cannot be represented in the flat schema at all (see Q7).

**Spoonerism** is not listed as an operation type. Acceptable to exclude from the
first implementation. Flag as a known omission.

---

## 2. Field names still too technical

Most field names are acceptable. Issues:

- `word_indices` appears in the JSON schema. The human-facing term is
  `clue_word_positions`. The spec mixes the two names. Pick one and use it
  consistently everywhere, including admin routes and templates.

- `outer_piece_id` / `inner_piece_id` in the stored JSON are fine as IDs.
  The UI must show dropdown labels from `clue_text`, never the raw IDs.
  The acceptance test wording "outer=female/SOW" is correct for the UI; make
  sure the implementation does not expose IDs in any rendered line.

- `colour` as a free string (`"blue"`, `"pink"`) is not defined. The current
  system uses `group_id` 0–4 mapping to `piece_0`…`piece_4` CSS classes. The
  named colour scheme must enumerate valid values and their CSS class mappings
  explicitly before implementation. Without this, Codex will invent the mapping.

- `relationship: "other"` is a design smell. If "other" enters any clue and gets
  stored, LLM validation and the verifier cannot act on it. Make the relationship
  list exhaustive by adding the missing types rather than using a catch-all.

---

## 3. Storage: graph-only, JSON-only, or hybrid

**Recommendation: Option B (JSON-only as primary), not Option C (hybrid).**

Two sources of truth drift. The spec notes "graph can be regenerated if rules
improve" — but if someone adds a direct graph-node row (as the current UI allows),
the JSON becomes stale immediately. Preventing that requires a write-lock on the
graph tables for any clue that has a structured parse row. That is more complexity,
not less.

Option B is cleaner: add `manual_structured_parses` table as the spec proposes,
store the JSON as authoritative, and generate graph rows from it on page load for
the display merge. The generation function is deterministic — drift is impossible
because the graph rows have no independent write path once a structured parse exists.

The cost: `merge_manual_into_display` must be rewritten to read from JSON rather
than directly from graph rows. It is a single function and its interface (the
answer_links it returns) does not change.

**Migration path**: once a clue has a `manual_structured_parses` row, its graph
rows are derived-only. Old clues without a structured parse row continue to use
graph rows directly. No migration of existing graph data is needed. This gives a
clean cutover.

If Option C (hybrid) is chosen instead, you MUST immediately prevent direct graph
writes for any clue that has a structured parse row. Otherwise drift will be
debugged within days. Option B avoids this problem entirely.

---

## 4. SWALLOW container example

Correct. No errors.

```
S W A L L O W
1 2 3 4 5 6 7
```

- SOW at [1,6,7]: S→box1 ✓, O→box6 ✓, W→box7 ✓
- WALL at [2,3,4,5]: W→box2 ✓, A→box3 ✓, L→box4 ✓, L→box5 ✓
- Container validates: SOW with WALL inserted at position 1 (0-based) gives
  S+WALL+OW = SWALLOW ✓

The validation logic for the operation: the outer piece's `answer_boxes` split into
a prefix group (all boxes before the inner piece's first box) and a suffix group
(all boxes after the inner piece's last box). Inner piece's letters must fill the
gap. This is derivable from `answer_boxes` and is general for any container.

One edge case not addressed in the spec: a container where the outer piece has no
suffix (inner piece at the very end) or no prefix (inner piece at the very start).
The validation must allow these as valid containers. The answer_boxes naturally
represent them (empty prefix or empty suffix group) and no special handling is
needed in the schema, but the validation code must not assume both groups are
non-empty.

---

## 5. "Pieces own answer colours; operations validate" — soundness

**Sound for**: container (proven by SWALLOW), charade (adjacent blocks, each owns
contiguous boxes), reversal (piece claims boxes in reverse order, operation
validates), homophone (piece claims boxes directly, operation validates).

**Not sound for anagram**: see Q1. Amendment: anagram-fodder pieces claim a block
of boxes (`mapping: "block"`), the operation validates all letters present with the
same multiset. No letter-by-letter positional check.

**Partially sound for deletion**: the source piece's full word is known, but only
surviving letters map to answer boxes. Use `mapping: "operation_assigned"` (see Q1).
The principle holds once the operation fills in the answer_boxes.

The core principle is correct and should be stated precisely as:

```
Pieces own answer tile colours.
Operations validate how pieces combine.
For positional pieces, letter-by-letter match is also validated.
For block pieces (anagram fodder), only multiset equality is validated.
For operation-assigned pieces (deletion source), boxes are derived by the operation.
```

---

## 6. How to represent charade

Explicit operation in LLM JSON; optional for human UI.

For humans: if pieces are entered in clue order and no other operation applies,
the system infers a charade without requiring an explicit operation entry. A charade
is concatenation in order. Requiring the user to add a charade operation for every
simple charade is friction.

For LLM output: always emit `{type: "charade", piece_ids: [...]}`. This makes the
parse inspectable — the structure is visible, not inferred from piece order.

Rule: if `operations` is empty and all answer boxes are covered by pieces in order,
the display and validation assume charade. An explicit charade operation is valid
but not required. The validator should emit a warning (not a blocking error) if
pieces cover the answer in non-contiguous or out-of-order positions without an
explicit operation explaining the combination.

---

## 7. Multi-step operations

The flat schema cannot represent these. For "reverse an abbreviation and then
insert it into another piece," the output of the reversal is an intermediate value
that becomes the input to the container. There is no place in the current schema
to store this intermediate.

Minimum viable extension: add `transform_pieces` as a top-level list alongside
`pieces`. A transform piece has:

```json
{
  "id": "tp1",
  "derived_from": "op1",
  "letters": "RD",
  "answer_boxes": [2, 3],
  "colour": "blue",
  "mapping": "positional"
}
```

A subsequent operation can reference a transform piece id as an input just like a
source piece id. The operation that produces the transform references it via
`result_piece_id`.

For the first implementation slice, exclude multi-step operations entirely.
But add `transform_pieces: []` to the schema now as an empty list. This means
the table and JSON are forward-compatible when multi-step is later implemented,
without a breaking schema change.

---

## 8. Answer boxes: 1-based UI and storage

Enforce 1-based throughout — in stored JSON, in the DB, and in the UI. Convert
to 0-based only at the point of Python array indexing (a single `- 1`).

Rationale: having 0-based storage and 1-based UI is a permanent source of
off-by-one bugs. Making storage 1-based means the display layer is identity
and the only conversion is in array access.

Important: the existing `answer_positions` column in `manual_evidence_nodes`
(Slice 1 implementation) is 0-based. When generating graph rows from the
structured parse JSON, add 1 when writing to `answer_positions`. When reading
existing graph rows for display, subtract 1 before passing to the merge function
if that function also expects 0-based. Document the boundary explicitly and do
not allow it to be ambiguous in the implementation instructions.

---

## 9. Can existing manual_evidence_nodes/edges support this?

Partially, but not cleanly. The tables can store pieces and operations with
enough `payload_json` stretching. But there is no:

- Parse-level version field
- Parse-level provenance (source: human/llm, confidence, created_by)
- Definition node type (current schema only has source, operator, transform)
- Filler concept
- Status field at the parse level (suggested, verified, failed)

More critically: reconstructing a full structured parse from graph rows to feed
to an LLM or to re-render is awkward. The LLM contract requires a clean JSON
object, not a set of node+edge rows that must be reassembled.

**Verdict**: the existing schema cannot support the structured parse format cleanly.
Add the `manual_structured_parses` JSON table. Keep the graph tables as a derived
display-layer cache. Do not attempt to make the graph tables the primary storage
for structured parses.

---

## 10. Smallest safe first implementation slice

**The smallest safe slice is: schema + UI language only. No container-specific
forms yet.**

Concretely:

1. Add `manual_structured_parses` table (the spec's Option B schema). Migration-safe;
   no existing data is touched.
2. Fix UI labels: answer boxes become 1-based throughout; "synonym" replaces vague
   labels; result label removed; payload JSON hidden; "clue word positions" replaces
   "word numbers."
3. Enumerate the named colour values and their CSS class mappings.
4. Saved evidence list renders in crossword wording:
   `female → SOW, boxes 1,6,7, blue` / `going over: container, SOW around WALL = SWALLOW`
5. No changes to the container operation form or the merge function.

Why not Slice B (container editor) first? Slice B requires three interdependent
changes: JSON table exists, merge function reads from JSON, old direct graph writes
blocked for clues with a structured parse row. Attempting all three together is
risky. Do the table first, then Slice B as a separate instruction.

The payoff from the language-only slice is immediate: the admin UI becomes
navigable for a human crossword user. Any parse entered under the new labels is
stored in the JSON table and can be migrated forward. No display logic changes,
no merge risk.

---

## Summary of amendments to the spec

| Issue | Amendment |
|---|---|
| Anagram break | Add `mapping: "positional"` (default) / `"block"` flag; skip letter-check for block pieces |
| Deletion gap | Add `mapping: "operation_assigned"`; operation fills answer_boxes at render |
| Missing relationship | Add `initial_letters` to relationship list |
| Colour enumeration | Enumerate valid named colours and CSS class mappings before implementation |
| Storage | Option B (JSON authoritative); graph rows are derived display cache only |
| Charade | Explicit operation in LLM JSON, optional for human; warn not block if absent |
| Multi-step | Add `transform_pieces: []` to schema now; leave empty for first slice |
| Answer boxes storage | 1-based throughout; convert to 0-based only at Python array access |
| First slice | Table + UI labels only; no container-specific form in first slice |
| `other` relationship | Remove; extend relationship list to cover all known types |

---

## Amended schema

The spec's JSON schema amended with the above changes:

```json
{
  "version": 1,
  "clue_id": 123,
  "answer": "SWALLOW",
  "source": "human",
  "confidence": "verified",
  "created_by": "admin",
  "definition": {
    "id": "def1",
    "clue_text": "Bird",
    "clue_word_positions": [0],
    "answer": "SWALLOW"
  },
  "pieces": [
    {
      "id": "piece1",
      "clue_text": "female",
      "clue_word_positions": [1],
      "relationship": "synonym",
      "letters": "SOW",
      "answer_boxes": [1, 6, 7],
      "mapping": "positional",
      "colour": "blue"
    },
    {
      "id": "piece2",
      "clue_text": "fence",
      "clue_word_positions": [4],
      "relationship": "synonym",
      "letters": "WALL",
      "answer_boxes": [2, 3, 4, 5],
      "mapping": "positional",
      "colour": "pink"
    }
  ],
  "transform_pieces": [],
  "operations": [
    {
      "id": "op1",
      "clue_text": "going over",
      "clue_word_positions": [2, 3],
      "type": "container",
      "outer_piece_id": "piece1",
      "inner_piece_id": "piece2",
      "result": "SWALLOW"
    }
  ],
  "filler": []
}
```

Changes from spec's schema:
- `word_indices` renamed to `clue_word_positions` throughout
- `mapping` field added to each piece (default `"positional"`)
- `transform_pieces: []` added at top level
- Top-level provenance fields (`source`, `confidence`, `created_by`) added
- `colour` remains as named string (enumeration to be defined separately)
