# Phase 2 Manual Evidence Graph - Slice 2 Specification for Claude

Date: 2026-05-26
Status: specification request for Claude review/proposal

---

## Purpose

Slice 1 created a durable admin-only manual evidence layer. It lets an admin
record that clue text directly contributes specific answer letters, and it can
colour clue blocks and answer tiles consistently.

Slice 2 must extend that into a real manual evidence graph for cryptic
operations.

The goal is not to create a manual-only replacement solver. The goal is to let
an admin faithfully model the parse when the automatic solver cannot find it:

- "CAT" gives "TOM"
- "BACKS" reverses that to "MOT"
- "THE SPANISH" gives "EL"
- the combined result gives "MOTEL"

In other words, the admin must be able to preserve both the source evidence and
the transformation evidence, then display the result with correct clue-block and
answer-tile colour coding.

---

## Current State After Slice 1

The following is already implemented and must be treated as the base reality.
Do not redesign it from scratch.

### Existing files

- `signature_solver/manual_evidence_store.py`
- `web/routes/admin.py`
- `web/routes/clue.py`
- `web/templates/clue.html`
- `web/templates/partials/manual_evidence_nodes.html`

### Existing storage

`manual_evidence_nodes` exists with:

- `id`
- `clue_id`
- `node_type`
- `word_indices`
- `word_text`
- `role`
- `raw_letters`
- `answer_positions`
- `group_id`
- `source`
- timestamps

`manual_evidence_edges` exists with:

- `id`
- `clue_id`
- `from_node_id`
- `to_node_id`
- `edge_type`
- timestamp

The edges table is currently created but unused.

### Existing supported node types

- `source`
- `definition`
- `structural`

### Existing display merge

`merge_manual_into_display(wfw_display, manual_nodes, answer)`:

- is admin-only display merge
- does not write to `wfw_proof_attempts`
- suppresses automatic `REVIEW_BLOCK`s and `SOURCE_BLOCK`s where manual nodes
  cover clue words
- suppresses automatic `DEF_BLOCK`s when a manual definition exists
- validates direct source letters against the cleaned answer
- uses `group_id` values 0-4 to produce `piece_0` through `piece_4` roles
- builds `answer_links` from an authoritative position map so the answer row
  always has exactly one tile per cleaned-answer letter
- duplicate manual claims for the same answer position produce a plain tile

This behaviour must not regress.

---

## Scope of Slice 2

Slice 2 should add the smallest useful manual operation graph.

It must support these operation types:

1. `reversal`
2. `deletion`
3. `anagram`
4. `container`

It may support `homophone` if the design is simple, but homophone must not make
the first four weaker or less testable.

Slice 2 must remain admin-only and display-time only. It must still not promote
manual evidence into public `wfw_proof_attempts`. Public proof promotion is a
later slice.

---

## Required User Capability

After Slice 2, an admin must be able to model this kind of invented clue:

> CAT BACKS ANNUAL EXAM with the Spanish hotel. = MOTEL

The admin should be able to record:

- Source node: `CAT` -> `TOM`
- Operator node: `BACKS` means reversal
- Edge: `CAT/TOM` is input to `BACKS`
- Transform result: reversal of `TOM` is `MOT`
- Source node: `THE SPANISH` -> `EL`
- Final answer placements:
  - `MOT` occupies answer positions 0,1,2
  - `EL` occupies answer positions 3,4

The display must show:

- `CAT` as a coloured source block
- `BACKS` as an operation block, not as unused text
- answer tiles `MOT` coloured consistently with the CAT/TOM source group
- `THE SPANISH` as a coloured source block
- answer tiles `EL` coloured consistently with the THE SPANISH source group
- no duplicate answer tiles
- no stale automatic source/review blocks competing with manual evidence over
  the same clue words

---

## Data Model Requirements

### Node types

Add support for at least:

- `operator`
- `transform`

Do not remove or reinterpret the existing Slice 1 types.

Recommended meanings:

- `source`: clue text produces raw letters directly or as input to an operator
- `operator`: clue text names a cryptic operation, such as reversal/deletion
- `transform`: the result of applying an operator to one or more input nodes
- `definition`: clue text defines the answer
- `structural`: link/surface/filler text that should suppress review blocks

### Roles

For `operator` nodes, role should be one of:

- `reversal`
- `deletion`
- `anagram`
- `container`
- optionally `homophone`

For `source` nodes, existing roles such as `synonym`, `abbreviation`, etc. may
continue to be used.

### Transform data

The current `manual_evidence_nodes` table has no generic JSON payload column.
Claude must propose the safest minimal way to store operator-specific data.

Acceptable options:

1. Add a nullable `payload_json` column to `manual_evidence_nodes`.
2. Reuse existing fields only if the design remains clear and testable.

Preferred option: add `payload_json`, because deletion/container/anagram need
structured data and overloading `raw_letters` will become brittle quickly.

If adding `payload_json`, provide migration-safe DDL using `ALTER TABLE` guarded
against duplicate-column errors, and update read/write helpers accordingly.

### Edges

Use `manual_evidence_edges` for graph relationships.

At minimum support:

- `input_to`: source/transform node feeds an operator node
- `output_of`: transform node is output of an operator node
- `placed_as`: source/transform node is placed into answer positions

If a simpler edge vocabulary is proposed, it must still allow the direction of
evidence to be unambiguous.

Edges must be read back and returned by the store layer. Deleting a node must
continue to delete attached edges.

---

## Operation Semantics

The operation evaluation must be deterministic and conservative. A manual
operation should display as failed rather than falsely colour answer tiles.

### Reversal

Input:

- exactly one input source/transform with `raw_letters` or transform output

Output:

- reverse the input letters

Validation:

- output must match claimed transform letters, if claimed
- if answer positions are supplied, output letters must match the cleaned answer
  at those positions

Example:

- source `CAT` has letters `TOM`
- operator `BACKS` role `reversal`
- transform output `MOT`

### Deletion

Input:

- exactly one input source/transform

Payload should specify what is deleted:

- `delete_text`: exact letters to remove, or
- `delete_positions`: zero-based positions in the input letters

Output:

- input with the deleted letters removed

Validation:

- deletion must be possible
- output must match claimed transform letters, if claimed
- answer placements must match cleaned answer

### Anagram

Input:

- one or more source/transform/fodder nodes

Payload:

- may specify output letters

Output validation:

- sorted input letters must equal sorted output letters
- output placements must match cleaned answer

This is manual validation, not automatic anagram solving. The admin is telling
the system what the anagram result is; the system verifies it is possible.

### Container

Inputs:

- one outer/frame source or transform
- one inner/content source or transform

Payload:

- explicit output letters
- optional split point or frame positions if needed

Validation:

- output must contain the inner letters inside the frame letters in the claimed
  order
- output must match claimed transform letters
- output placements must match cleaned answer

Do not attempt clever automatic inference in Slice 2. Require explicit enough
manual data that validation is deterministic.

---

## Display Requirements

The existing `atomic_parse.html` partial already knows how to render:

- `SOURCE_BLOCK`
- `OP_BLOCK`
- `DEF_BLOCK`
- `REVIEW_BLOCK`
- answer links with `source_role`
- colour roles `piece_0` through `piece_4`

Slice 2 should keep using that display contract unless there is a specific,
justified reason to extend it.

### Blocks

Manual source nodes should continue to emit `SOURCE_BLOCK`s.

Manual operator nodes should emit `OP_BLOCK`s with:

- `block_id`: stable manual id, e.g. `manual_op_<id>`
- `kind`: `OP_BLOCK`
- `role`: operator role, e.g. `reversal_indicator`
- `text`: clue text, e.g. `BACKS`
- `value`: concise operation label or output summary
- `span`: clue word span
- `evidence_status`: `manual` or `failed`

Manual transform nodes may emit either:

- a `SOURCE_BLOCK` showing the transformed output, or
- a `RELATION_BLOCK`/`OP_BLOCK` if that better matches current display

Claude must recommend one representation and explain why it fits the existing
template.

### Answer links

Answer links should be built from the transformed output when an operator is
used.

Example:

- `CAT` -> `TOM`
- `BACKS` reverses to `MOT`
- answer positions 0,1,2 are linked to the transform output, but should retain
  the same colour group as the original CAT/TOM source unless explicitly
  overridden.

Colour continuity is essential. The user must be able to see that the same
piece has moved through a transformation.

### Conflict handling

The Slice 1 invariant remains mandatory:

- exactly one answer tile per cleaned-answer position
- no duplicate answer positions
- conflicting manual claims produce plain or failed display, never false colour

Claude should decide whether conflicts should:

- make the answer tile plain, as Slice 1 does, and/or
- mark involved transform/source/operator blocks as failed

The proposal must be explicit.

---

## Admin UI Requirements

The current Slice 1 UI is basic but usable. Slice 2 may keep it basic, but it
must let an admin create the operation graph without editing the database by
hand.

Minimum acceptable UI:

1. Existing node creation remains.
2. Admin can create an operator node with:
   - word text
   - word indices
   - operator role
3. Admin can create a transform node with:
   - output letters
   - answer positions
   - group id
   - payload JSON or structured fields
4. Admin can create edges between existing node ids.
5. The manual evidence list shows enough ids and details that the admin can
   understand and connect nodes.

The UI does not need to be elegant in Slice 2. It does need to be safe,
understandable, and testable.

Do not hide ids if ids are needed to create edges.

---

## Store/API Requirements

Extend `signature_solver/manual_evidence_store.py` with helpers such as:

- `write_edge(clue_id, from_node_id, to_node_id, edge_type, conn=None)`
- `delete_edge(edge_id, conn=None)`
- `get_edges_for_clue(clue_id, conn=None)`
- possibly `get_graph_for_clue(clue_id, conn=None)` returning nodes and edges

If adding `payload_json`, update:

- DDL / migration helper
- `write_node`
- `get_nodes_for_clue`
- route form parsing
- template display

All JSON fields must be decoded safely. Bad JSON from admin input should return
HTTP 400 at the route layer, not be silently accepted.

---

## Non-Goals For Slice 2

Do not do these in Slice 2:

- Do not rewrite the automatic solver.
- Do not promote manual evidence into `wfw_proof_attempts`.
- Do not make manual evidence public.
- Do not remove Slice 1 direct source placement.
- Do not make rerun delete manual evidence.
- Do not require a perfect UI.
- Do not depend on live Flask/browser testing as the only verification.

---

## Acceptance Tests Claude Must Specify

Claude's proposal must include verification that can be run without touching the
live database.

### Check 1 - syntax

Compile all changed Python files.

### Check 2 - temp DB graph round-trip

Using a temporary SQLite file:

1. create tables
2. write source node `CAT` -> `TOM`
3. write operator node `BACKS` role `reversal`
4. write transform node output `MOT`, answer positions `[0,1,2]`, group id `0`
5. write edges connecting source -> operator -> transform
6. read back graph
7. assert nodes and edges round-trip correctly
8. delete a node and assert attached edges are deleted

### Check 3 - reversal display merge

Pure function test. No Flask. No live DB.

Input graph:

- source `CAT` -> `TOM`, group 0
- operator `BACKS` reversal
- transform output `MOT`, positions `[0,1,2]`, group 0
- answer `MOTEL`

Expected:

- answer row has exactly 5 links
- positions 0,1,2 are `piece_0`
- positions 3,4 are plain unless another node claims them
- `CAT` source block appears
- `BACKS` OP_BLOCK appears
- no duplicate answer positions

### Check 4 - two-piece display merge

Input graph:

- CAT/BACKS produces MOT, group 0, positions 0-2
- THE SPANISH produces EL, group 1, positions 3-4
- answer MOTEL

Expected:

- exactly 5 answer links
- positions 0,1,2 have source_role `piece_0`
- positions 3,4 have source_role `piece_1`
- both clue source blocks appear
- operator block appears
- no conflicting auto `SOURCE_BLOCK`/`REVIEW_BLOCK` remains over covered words

### Check 5 - failed operation does not falsely colour

Input graph:

- source `CAT` -> `TOM`
- operator reversal
- transform claims output `XYZ` at positions 0,1,2 in answer `MOTEL`

Expected:

- transform/operator/source evidence is marked failed or review according to the
  chosen design
- positions 0,1,2 are not falsely coloured as `piece_0`
- answer row still has exactly 5 tiles

### Check 6 - rerun safety audit

Read the `_rerun_clue_inner` upfront clear block and confirm:

- it updates `clues`
- it deletes from `structured_explanations`
- it does not delete from `manual_evidence_nodes`
- it does not delete from `manual_evidence_edges`

---

## Questions Claude Must Answer Before Codex Implements

Claude should produce a proposal/instruction document, not just prose. It must
answer these directly:

1. Is `payload_json` needed? If yes, what exact migration code should be used?
2. What exact node/edge shapes represent reversal, deletion, anagram, and
   container?
3. How does the merge function decide whether a transform is valid?
4. How are failed transforms displayed?
5. How does colour flow from source through transform to answer tiles?
6. How are duplicate/conflicting answer claims handled?
7. What exact files change?
8. What exact verification commands prove the slice?

---

## Expected Claude Output

Please produce a Codex implementation instruction document for Slice 2.

It should be explicit enough that Codex can implement it faithfully without
guessing.

It should include:

- exact file list
- exact schema/storage changes
- exact route/template changes
- exact merge/evaluation algorithm
- exact test scripts or commands
- explicit non-goals
- known limitations

Do not weaken the design by turning this into another direct source-placement
patch. The point of Slice 2 is preserved manual operation evidence.
