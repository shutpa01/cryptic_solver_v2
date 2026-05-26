# GT V2 Block Graph Contract

Date: 2026-05-17

This contract defines the representation GT V2 should use for block anatomy candidates.

It is deliberately graph-shaped rather than a single flat role sequence.

## Why A Graph

The experiments show that token labels alone are too weak.

Grammar improves boundary recovery, but it does not produce a trustworthy single parse.

Cryptic attachment is not the same as parser-head attachment.

Therefore GT V2 should represent clue anatomy as candidate graphs:

- nodes are blocks
- edges are relationships
- evidence is attached to both nodes and edges
- unresolved scope remains explicit until answer verification

This lets the system preserve grammar and wordplay evidence without forcing either to erase the other.

## Graph Object

A block graph represents one possible anatomy of one clue.

Required fields:

- `clue_id`
- `clue`
- `answer`
- `candidate_id`
- `nodes`
- `edges`
- `global_evidence`
- `status`
- `notes`

`status` may be:

- `candidate`
- `mechanically_verified`
- `rejected`
- `needs_human_review`

## Node Object

Each node is an exact clue span or an inferred assembly result.

Required fields:

- `node_id`
- `kind`
- `span`
- `span_space`
- `text`
- `normalised_text`
- `source_piece_ids`
- `operation`
- `value`
- `mechanisms`
- `evidence`
- `status`

`span` is token-index based and may be `null` for inferred assembly nodes or for an explicit definition-gap placeholder.

`span_space` records which token stream the span indexes. It should be `full_clue_tokens` for definition blocks, `wordplay_tokens` for current source/residue-derived blocks, and `null` or absent for inferred assembly nodes. This is a temporary but important guardrail while the R&D artifacts still carry both full-clue and wordplay-only coordinates.

If no definition span is preserved by the structured explanation, the graph should still carry an ambiguous `DEF_BLOCK` placeholder with `span = null`, empty `text`, and graph-level `definition_status = missing_from_structured_explanation`. This keeps definition modifiers and later all-in-one hypotheses attached to a definition candidate rather than collapsing them onto the answer.

For `SOURCE_BLOCK`s, `mechanisms` preserves the structured explanation mechanism such as `synonym`, `anagram_fodder`, `hidden`, `abbreviation`, or `last_letter`. This is essential evidence: a source span produced by `last_letter` or another selection mechanism may contain locator/surface material that should not be treated like a simple synonym block.

`kind` may be:

- `DEF_BLOCK`
- `DEF_MODIFIER_BLOCK`
- `SOURCE_BLOCK`
- `OP_BLOCK`
- `LOCATOR_BLOCK`
- `POSITION_BLOCK`
- `RELATION_BLOCK`
- `CONNECTOR_BLOCK`
- `SOURCE_INTERNAL_BLOCK`
- `SCOPE_BLOCK`
- `ASSEMBLY_BLOCK`

`status` may be:

- `observed`
- `inferred`
- `ambiguous`
- `rejected`

## Edge Object

Edges record the relationship between blocks.

Required fields:

- `edge_id`
- `kind`
- `from_node`
- `to_node`
- `evidence`
- `scope_status`
- `confidence`
- `notes`

`kind` may be:

- `DEFINES`
- `MODIFIES_DEFINITION`
- `CONTRIBUTES_TO`
- `OPERATES_ON`
- `LOCATES_WITHIN`
- `ORDERS`
- `CONTAINS`
- `CONTAINED_BY`
- `BELONGS_TO_SOURCE`
- `SURFACE_CONNECTS`
- `AWAITING_SCOPE`
- `ASSEMBLES_WITH`

`scope_status` may be:

- `known`
- `weakly_scoped_from_operation`
- `answer_aware_scope_needed`
- `surface_only_until_grammar_check`
- `candidate_source_absorption`
- `unresolved`

## Evidence Object

Evidence should be explicit and plural.

Examples:

- `structured_explanation`
- `answer_letters`
- `grammar_pos`
- `grammar_dependency`
- `parser_head`
- `residue_lexicon`
- `indicator_table`
- `synonym_table`
- `homophone_table`
- `hidden_string_match`
- `blog_explanation`
- `surface_grammar_objection`

Evidence should record:

- `source`
- `detail`
- `supports`
- `opposes`
- `strength`

The key design point is that evidence can oppose as well as support a relation.

For GOLDEN RETRIEVER, a surface grammar objection opposes treating `on` as a connector.

For definition-by-example leakage, a definition-span objection may oppose treating `say` or `perhaps` as wordplay residue.

## Minimal Example Shape

For a simple anagram clue:

```json
{
  "candidate_id": "example-1",
  "nodes": [
    {
      "node_id": "n_def",
      "kind": "DEF_BLOCK",
      "span": [4, 6],
      "text": "emergency barrier"
    },
    {
      "node_id": "n_src",
      "kind": "SOURCE_BLOCK",
      "span": [0, 2],
      "text": "Follow lad",
      "value": "FOLLOWLAD"
    },
    {
      "node_id": "n_op",
      "kind": "OP_BLOCK",
      "span": [2, 4],
      "text": "after repairing",
      "operation": "anagram"
    },
    {
      "node_id": "n_answer",
      "kind": "ASSEMBLY_BLOCK",
      "span": null,
      "text": "FLOODWALL",
      "value": "FLOODWALL"
    }
  ],
  "edges": [
    {
      "kind": "OPERATES_ON",
      "from_node": "n_op",
      "to_node": "n_src",
      "scope_status": "weakly_scoped_from_operation"
    },
    {
      "kind": "CONTRIBUTES_TO",
      "from_node": "n_src",
      "to_node": "n_answer",
      "scope_status": "known"
    },
    {
      "kind": "DEFINES",
      "from_node": "n_def",
      "to_node": "n_answer",
      "scope_status": "known"
    }
  ],
  "status": "candidate"
}
```

## Required Preservation Rules

Do not assume a residue run is a block.

Example:

- `say with` may need to split into `say` as `DEF_MODIFIER_BLOCK` and `with` as `RELATION_BLOCK`
- the graph must preserve both nodes and record that they came from the same surface residue run

Do not flatten operation and locator.

Example:

- `loses` is an operation
- `heart of X` is a locator/target relation
- the chain must be preserved as edges

Do not consume apparent connectors without a grammar check.

Example:

- `on` may be connector, positional instruction, or source-internal phrase material
- the graph should preserve the alternatives until evidence rejects them

Do not attach reversal/anagram indicators too early.

Example:

- a leading reversal indicator may apply to a local source or a larger assembled span
- the graph should permit an `AWAITING_SCOPE` edge

Do not treat parser heads as cryptic truth.

Parser evidence can support or oppose an edge, but cryptic operation anatomy and answer verification decide the final parse.

Do not force definition modifiers into residue.

Example:

- `Slug, say` may define a category by example
- `say` should be allowed to attach to `DEF_BLOCK` as `DEF_MODIFIER_BLOCK`
- but `did you say` in a homophone clue may remain an `OP_BLOCK`

## Candidate Generation Strategy

The first GT V2 graph generator should be conservative.

For each clean clue:

1. Create `DEF_BLOCK` from the known definition span.
2. Create `SOURCE_BLOCK`s from structured source spans.
3. Create residue-run nodes.
4. Assign weak residue kinds from operation-derived labels.
5. Add relationship edges from the weak label.
6. Attach grammar evidence to nodes and edges.
7. Mark unresolved scope explicitly.
8. Later, verify graph candidates against answer mechanics.

This is a research scaffold, not a production solver.
