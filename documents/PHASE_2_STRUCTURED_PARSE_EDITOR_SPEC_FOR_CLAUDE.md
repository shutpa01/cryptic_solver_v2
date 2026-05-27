# Phase 2 Structured Parse Editor: Human + LLM Format

Date: 2026-05-27

Purpose: ask Claude to review the design before implementation.

This is not a request for a small label change. The current manual evidence
graph is valuable, but the admin UI still exposes too much of the storage
model. The next step should define a crossword-native structured parse format
that a human can fill in and that an LLM can also produce. The existing graph
tables can remain the persistence/display layer, but they should no longer be
the primary mental model presented to the user.

## Request to Claude

Please review this as an architecture and product-design proposal, not as a
code patch.

The main question is:

```text
What structured parse format should both the human admin UI and future LLM
parsing use, so that any failed clue can be corrected, validated, rendered with
colours, preserved across reruns, and eventually promoted into proof evidence?
```

Please be deliberately critical. In particular:

1. Identify any clue type this structure cannot represent cleanly.
2. Identify any field names that are still too technical for a crossword user.
3. Decide whether graph-only, JSON-only, or hybrid persistence is best.
4. Check the SWALLOW container example for correctness and generality.
5. Check whether the rule "pieces own answer colours; operations validate
   combinations" is sound.
6. Recommend the smallest safe first implementation slice.
7. Do not write implementation code yet unless specifically asked. First give
   a reviewed proposal with risks, amendments, and acceptance tests.

The answer should be concrete enough that Codex can then implement faithfully
without improvising the design.

## Current Problem

The current admin manual parse editor has these concepts:

- source node
- operator node
- transform node
- edge
- payload_json
- word_indices
- answer_positions
- group_id

Those are implementation details. They are not how a crossword solver thinks.

The user thinks in terms of:

- definition
- clue phrase
- synonym / abbreviation / literal / foreign word
- answer boxes
- reversal / container / anagram / deletion / homophone
- which clue piece contributes which answer letters
- colour linking between clue blocks and answer tiles

The current UI also asks the user to manually wire graph edges. That is too
abstract for routine correction. It may be useful internally, but it is not an
acceptable primary interface.

## Core Reframing

The manual editor should become a structured parse editor.

This structured parse should be:

1. Human-fillable.
2. LLM-fillable.
3. Stored durably.
4. Rendered as coloured clue blocks and answer tiles.
5. Validated deterministically where possible.
6. Convertible into the existing manual evidence graph for display merge.
7. Preserved across rerun/reverify.

This replaces the old free-form leftover explanation process. Instead of asking
an LLM to write a paragraph and then trying to interpret it, the LLM should be
able to emit the same structured parse object that the human editor uses.

## Running Example

Clue:

```text
Bird, female, going over fence (7)
```

Answer:

```text
SWALLOW
```

Parse:

```text
Bird = definition
female = SOW
fence = WALL
going over = container indicator
SOW contains WALL
S + WALL + OW = SWALLOW
```

Desired human entry:

```yaml
definition:
  clue_words: Bird
  answer: SWALLOW

pieces:
  - clue_words: female
    relationship: synonym
    letters: SOW
    answer_boxes: [1, 6, 7]
    colour: blue

  - clue_words: fence
    relationship: synonym
    letters: WALL
    answer_boxes: [2, 3, 4, 5]
    colour: pink

operations:
  - clue_words: going over
    type: container
    outer: female
    inner: fence
    result: SWALLOW
```

Important: answer boxes in the UI and LLM format should be 1-based, because
that is how humans refer to answer boxes. The storage layer may convert to
0-based internally.

## Required User Model

The user should be able to describe the parse without knowing about nodes or
edges.

The top-level objects should be:

### 1. Definition

Fields:

- clue words
- word positions, preferably auto-filled by clicking clue words
- answer

Example:

```text
Definition: Bird = SWALLOW
```

### 2. Answer Pieces

An answer piece is a clue phrase that contributes letters to the answer.

Fields:

- clue words
- word positions, preferably auto-filled
- relationship
- letters produced
- answer boxes
- colour

Relationship labels should be crossword-specific and clear:

- synonym
- abbreviation
- literal letters
- foreign word
- pronoun / name
- single letter
- hidden letters
- other, only if necessary

Avoid vague labels such as "same meaning" if "synonym" is clearer.

Example:

```text
female -> SOW, boxes 1,6,7, blue
fence -> WALL, boxes 2,3,4,5, pink
```

### 3. Operations

An operation is a crossword instruction that acts on one or more answer pieces.

Fields:

- indicator clue words
- word positions, preferably auto-filled
- operation type
- operand pieces selected from saved answer pieces
- result letters, where needed for validation

Operation labels should be crossword-native:

- reversal
- container
- anagram
- deletion
- homophone
- charade / join

Example:

```text
going over = container
outer piece: female -> SOW
inner piece: fence -> WALL
result: SWALLOW
```

The UI should not ask the user to create a separate "transform node" or
"operation result" unless the operation genuinely needs a displayed result
object. Even then, it should be worded as "result letters" or "letters made",
not "result label".

### 4. Filler / Link Words

Some clue words are surface/link/filler and should be markable so the parse is
complete.

Fields:

- clue words
- word positions
- role: link/filler/surface

This should not colour answer tiles.

## Critical Container Requirement

Container clues must support split colouring.

For:

```text
SOW contains WALL = SWALLOW
```

The answer row should show:

- box 1: `S`, colour of SOW
- boxes 2-5: `WALL`, colour of WALL
- boxes 6-7: `OW`, colour of SOW

Therefore, a container result must not flatten the entire result into one
colour. The operation validates the combined output, but the answer tile colours
come from the source pieces and their assigned answer boxes.

This is a major design principle:

```text
Operations validate how pieces combine.
Pieces own the answer tile colours.
```

## Answer Box Numbering

Human-facing answer boxes should be 1-based.

Example for SWALLOW:

```text
S W A L L O W
1 2 3 4 5 6 7
```

Storage/display internals may remain 0-based, but conversion must happen at
the boundary.

Validation must catch:

- box number less than 1
- box number greater than answer length
- duplicate claimed box unless intentionally handled as conflict
- number of boxes not equal to number of letters produced, except for a piece
  that contributes discontinuously but still has matching count
- letters that do not match the cleaned answer at those boxes

For `female -> SOW` boxes `[1,6,7]`:

- S must match answer box 1
- O must match answer box 6
- W must match answer box 7

For `fence -> WALL` boxes `[2,3,4,5]`:

- W must match box 2
- A must match box 3
- L must match box 4
- L must match box 5

## UI Improvements Needed

These are the concrete UI improvements identified during use:

1. Let the user click clue words to populate clue phrase and word positions.
   Manual word-position entry can remain as a fallback.

2. Rename "Word numbers" to "Clue word positions" only if still shown. Better:
   hide it behind the click interaction.

3. Use crossword-specific labels:
   - "synonym", not "same meaning"
   - "answer boxes", not "answer letters"
   - "letters produced", not "raw letters" or "result letters" where possible
   - "colour group", not "group_id"

4. Remove or auto-fill "Result label". It is not meaningful to the user.

5. Hide JSON/payload fields from normal use. If advanced details are ever
   needed, they should be generated by operation-specific forms.

6. Replace generic graph-edge connection with operation-specific forms:
   - reversal: choose piece, result is reverse(piece)
   - container: choose outer piece, inner piece, result
   - anagram: choose fodder piece(s), result
   - deletion: choose source piece, delete instruction, result
   - homophone: choose heard-as piece, result

7. Make container colour split automatic from selected outer/inner pieces and
   their answer boxes.

8. Saved evidence should be shown as crossword parse lines:
   - `Bird = definition`
   - `female -> SOW, boxes 1,6,7, blue`
   - `fence -> WALL, boxes 2,3,4,5, pink`
   - `going over: container, SOW around WALL = SWALLOW`

9. The editor should show parse completeness:
   - all clue words classified or intentionally ignored
   - all answer boxes covered or intentionally left unresolved
   - operations valid or marked as failed

10. The same structured parse format should be importable from an LLM response.

## Suggested Structured Parse Schema

This schema is intentionally independent of the current node/edge tables.
Claude should review whether this should be stored as JSON directly, translated
into the existing graph tables, or both.

```json
{
  "version": 1,
  "clue_id": 123,
  "answer": "SWALLOW",
  "definition": {
    "id": "def1",
    "clue_text": "Bird",
    "word_indices": [0],
    "answer": "SWALLOW"
  },
  "pieces": [
    {
      "id": "piece1",
      "clue_text": "female",
      "word_indices": [1],
      "relationship": "synonym",
      "letters": "SOW",
      "answer_boxes": [1, 6, 7],
      "colour": "blue"
    },
    {
      "id": "piece2",
      "clue_text": "fence",
      "word_indices": [4],
      "relationship": "synonym",
      "letters": "WALL",
      "answer_boxes": [2, 3, 4, 5],
      "colour": "pink"
    }
  ],
  "operations": [
    {
      "id": "op1",
      "clue_text": "going over",
      "word_indices": [2, 3],
      "type": "container",
      "outer_piece_id": "piece1",
      "inner_piece_id": "piece2",
      "result": "SWALLOW"
    }
  ],
  "filler": []
}
```

## Mapping to Existing Manual Evidence Graph

The existing graph is still useful as a display/merge implementation:

- source nodes can represent answer pieces
- operator nodes can represent operations
- transform nodes can represent operation outputs where needed
- edges can represent input/output relationships

But the user should not directly create graph nodes/edges.

The system should generate graph rows from the structured parse:

```text
piece -> source node
operation -> operator node
operation result -> transform node only when needed
piece used by operation -> input_to edge
result made by operation -> output_of edge
definition -> definition node
filler -> structural node
```

For container colouring, the generated transform must not override the source
piece colours on answer tiles. The transform may validate and display the
operation, but answer links should remain owned by the pieces unless there is
no finer-grained piece mapping.

## LLM Use

The old free-form leftover system should eventually be replaced by this format.

An LLM should be asked to return structured JSON using the same schema:

```text
Given clue text, answer, and any available automatic evidence, produce a
structured parse with definition, pieces, operations, and filler. Do not invent
database evidence. Mark uncertain fields explicitly.
```

The application can then:

1. validate the JSON shape
2. validate clue word spans
3. validate answer boxes
4. validate operation outputs
5. render the parse
6. store it as manual/LLM evidence with provenance

This makes LLM output inspectable and correctable instead of free-form.

## Provenance

Each parse element should record its source:

- human
- llm
- automatic

For the first implementation, focus on human entries. But the schema should not
block LLM entries.

Suggested fields:

```json
{
  "source": "human",
  "confidence": "verified",
  "created_by": "admin"
}
```

LLM entries could use:

```json
{
  "source": "llm",
  "confidence": "suggested",
  "model": "..."
}
```

## Persistence Options for Claude to Review

Claude should recommend one of these:

### Option A: Keep graph tables only

The UI writes structured form fields, server converts immediately to graph
nodes/edges.

Pros:

- smaller DB change
- uses current merge implementation

Cons:

- structured parse is reconstructed from graph, which may be awkward
- LLM JSON import has no natural home

### Option B: Add a structured parse JSON table

New table:

```sql
manual_structured_parses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clue_id INTEGER NOT NULL,
    parse_json TEXT NOT NULL,
    source TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
)
```

The graph rows can be generated from the structured parse for display.

Pros:

- clean human/LLM contract
- easier to edit whole parse
- easier to validate and re-render

Cons:

- new persistence layer
- must keep graph/display generation deterministic

### Option C: Hybrid

Store the structured parse JSON and generated graph rows. The JSON is
authoritative; graph rows are derived display artefacts.

Pros:

- best user/LLM model
- keeps current display merge
- graph can be regenerated if rules improve

Cons:

- must avoid drift between JSON and derived rows

Initial instinct: Option C is probably the best long-term shape, but Claude
should challenge this.

## Implementation Slices

Do not implement everything at once.

### Slice A: UI language and direct piece entry

- answer boxes become 1-based
- synonym label fixed
- result label removed/auto-filled
- no visible payload JSON
- saved evidence list uses crossword wording

### Slice B: Container-specific editor

For a container operation, user enters/selects:

- indicator clue words
- outer piece
- inner piece
- result

The editor validates:

- outer and inner pieces exist
- result equals outer with inner inserted somewhere
- source piece answer boxes colour the answer tiles

Target example:

```text
female -> SOW, boxes 1,6,7, blue
fence -> WALL, boxes 2,3,4,5, pink
going over: SOW around WALL = SWALLOW
```

### Slice C: Structured parse JSON export/import

- export current manual parse as JSON
- import JSON from textarea or LLM result
- validate before saving

### Slice D: Additional operations

- reversal
- anagram
- deletion
- homophone
- charade/join

### Slice E: Promotion/publication

Separate from this review:

- when/if manual or LLM parse becomes public proof
- how it affects wfw_proof_attempts
- how review status is set

## Acceptance Tests

### Test 1: SWALLOW container parse

Given:

```text
Bird, female, going over fence (7)
Answer: SWALLOW
```

Enter:

- definition: Bird
- piece: female -> SOW, boxes 1,6,7, blue
- piece: fence -> WALL, boxes 2,3,4,5, pink
- operation: going over = container, outer=female/SOW, inner=fence/WALL

Expected:

- clue block `Bird` is definition
- clue block `female` is blue source block
- clue block `fence` is pink source block
- clue block `going over` is operation block
- answer tile 1 is blue S
- answer tiles 2-5 are pink WALL
- answer tiles 6-7 are blue OW
- operation validates as `SOW` containing `WALL` gives `SWALLOW`
- rerun does not delete the manual parse

### Test 2: invalid answer boxes

If `female -> SOW` is assigned boxes `[1, 5, 7]`, validation fails because
box 5 is `L`, not `O`.

Expected:

- user sees a clear validation error
- no misleading coloured answer tile is emitted for the failed mapping

### Test 3: duplicate answer box conflict

If two pieces both claim box 2:

Expected:

- conflict is detected
- tile is left plain or marked conflicted
- no silent overwrite

### Test 4: LLM JSON import

Given the JSON parse for SWALLOW:

Expected:

- JSON validates
- same display as human-entered parse
- provenance records source as LLM/suggested until accepted

## Questions for Claude

1. Is the proposed structured parse schema sufficient for common cryptic clue
   types?

2. Should the authoritative storage be graph-only, JSON-only, or hybrid?

3. For container clues, is "pieces own answer colours; operation validates
   combination" the right rule?

4. How should charades be represented: as an operation, as implicit ordering of
   pieces, or both?

5. How should multi-step operations be represented, for example reverse an
   abbreviation and then put it inside another piece?

6. Should answer boxes always be user-facing 1-based and storage-facing 0-based?

7. What validation errors should block saving, and what should be saved as
   failed/suggested evidence?

8. How should LLM-suggested parses be reviewed and accepted by the human?

9. Can the existing manual_evidence_nodes/manual_evidence_edges schema support
   this cleanly, or do we need a structured parse table?

10. What is the smallest safe implementation slice that materially improves the
    user experience without compromising the long-term architecture?

## Non-Goals for This Slice

- Do not replace the automatic solver.
- Do not make manual evidence public yet.
- Do not promote manual parses into `wfw_proof_attempts` yet.
- Do not delete manual evidence on rerun.
- Do not require the user to know graph terminology.
- Do not ask the user to type JSON in normal use.

## Bottom Line

The target is a structured crossword parse format.

The human admin UI and the LLM output should converge on the same model:

```text
definition + answer pieces + operations + filler + validation
```

If we get this right, the system can always be corrected, the corrections can
be rendered faithfully, and future LLM parsing can be inspected instead of
trusted blindly.
