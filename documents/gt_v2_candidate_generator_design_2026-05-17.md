# GT V2 Candidate Generator Design

Date: 2026-05-17

Status: design draft

This document defines the stand-alone design for GT V2 candidate generation.

It is not a solver patch and it is not an implementation plan for `stages/`.

The purpose is to define the anatomy object GT V2 should propose from a fresh clue before answer mechanics verifies or rejects it.

## Objective

GT V2 should not try to choose one complete parse immediately.

It should generate plausible block-graph candidates from the raw clue, preserve competing anatomical possibilities, and then let answer mechanics test which candidate graph can actually produce the answer.

The central design claim is:

The solver should receive a reduced, structured operating space rather than a flat clue string.

GT V2 therefore sits between grammar triage and mechanical solving.

Its job is to say:

- these spans may be definitions
- these spans may be sources
- these spans may be operations, locators, positions, relations, or connectors
- these words may belong together despite looking separable
- these scopes are known
- these scopes must remain unresolved until answer verification
- these alternatives are opposed by grammar or answer mechanics

It must not flatten ambiguity into a premature parse.

## Inputs

The candidate generator may use:

- raw clue text
- answer and enumeration
- clue direction, if available
- grammar parse
- tokenisation
- punctuation and clue position
- indicator lexicons
- synonym, abbreviation, homophone, hidden-string, and named-entity evidence
- answer-letter evidence
- source mechanism evidence from prior supervised mining
- surface grammar objections

The answer may be used for verification and ranking.

Using the answer is not cheating in this design phase. The point is to learn which block anatomies can reproduce the known answer while preserving how the clue got there.

The important distinction is:

- answer mechanics may verify a candidate
- answer mechanics must not erase clue-span evidence

## Output

GT V2 emits one or more candidate graphs.

Each graph represents one possible anatomy of the clue.

The graph contains:

- block nodes
- relationship edges
- evidence on nodes and edges
- unresolved scope status
- rejection or ambiguity notes
- a graph-level verification status

A candidate graph may be:

- `candidate`
- `mechanically_verified`
- `rejected`
- `needs_human_review`

The output is not a table of roles. It is a structured explanation candidate.

## Emitted Object Summary

An implementation should emit this shape for each candidate graph:

```json
{
  "candidate_id": "stable candidate id",
  "clue": "raw clue text",
  "answer": "normalised answer",
  "definition_status": "observed | missing_from_structured_explanation | whole_surface_candidate",
  "nodes": [
    {
      "node_id": "stable node id",
      "kind": "DEF_BLOCK | SOURCE_BLOCK | OP_BLOCK | ...",
      "span": [0, 2],
      "span_space": "full_clue_tokens",
      "text": "exact clue text",
      "normalised_text": "normalised clue text",
      "operation": "optional operation",
      "value": "optional candidate value",
      "mechanisms": ["synonym"],
      "evidence": [],
      "status": "candidate | observed | inferred | ambiguous | rejected"
    }
  ],
  "edges": [
    {
      "edge_id": "stable edge id",
      "kind": "DEFINES | CONTRIBUTES_TO | OPERATES_ON | ...",
      "from_node": "node id",
      "to_node": "node id",
      "evidence": [],
      "scope_status": "known | answer_aware_scope_needed | unresolved | ...",
      "confidence": "strong | medium | weak",
      "notes": "short explanation"
    }
  ],
  "verification": {
    "status": "candidate | mechanically_verified | rejected | needs_human_review",
    "tested_forms": [],
    "accepted_form": null,
    "rejection_reasons": [],
    "unresolved_questions": []
  }
}
```

The exact storage format can change later.

The essential requirement is that roles, relationships, evidence, and verification state remain separate.

## Candidate Graph Shape

Every graph should include an answer assembly node.

```json
{
  "candidate_id": "example:candidate_0",
  "clue": "Receptacle former PM brought in without delay",
  "answer": "SPITTOON",
  "definition_status": "observed",
  "nodes": [],
  "edges": [],
  "global_evidence": [],
  "verification": {
    "status": "candidate",
    "answer_value": "SPITTOON",
    "notes": []
  }
}
```

Nodes and edges follow the existing block graph contract.

The generator should prefer exact clue spans. Inferred assembly nodes may have no clue span.

If no definition span can be located, the graph still carries an ambiguous `DEF_BLOCK` placeholder. The model should represent `definition candidate not yet located`, not `no definition`.

## Block Types

`DEF_BLOCK`

The definition candidate.

It may be clue-initial, clue-final, or in special cases implicit across the whole surface.

Evidence can include answer synonym evidence, phrase type, clue edge position, punctuation boundary, grammar phrase type, question marks, definition-by-example markers, and exclusion after verified wordplay has consumed the rest.

Opposing evidence can include a span being mechanically required as source, part of hidden fodder, or grammatically required to complete a source phrase.

`DEF_MODIFIER_BLOCK`

A span that modifies the definition rather than the wordplay.

Examples include `say`, `perhaps`, `maybe`, and `for example`.

This block is usually definition-by-example evidence. It must attach to a definition candidate, not to the answer placeholder. If the definition is missing, it attaches to the ambiguous definition-gap node.

`SOURCE_BLOCK`

A clue span that can contribute letters, a value, or transformed material to the answer.

Source candidates may come from synonym evidence, abbreviation evidence, named-entity evidence, raw anagram fodder, hidden fodder, homophone source, deletion source, reversal source, or letter-selection source.

Every `SOURCE_BLOCK` must preserve candidate mechanisms.

This matters because mechanisms imply different anatomy:

- `synonym` is often a lexical source
- `anagram_fodder` is raw material awaiting operation scope
- `hidden` is contiguous fodder and may already be pre-verified
- `first_letter` and `last_letter` imply operation plus locator, even when mapped as a source span
- `deletion` may require expansion before removal

A source block is therefore not just text. It is text plus mechanism evidence plus possible value.

`OP_BLOCK`

An operation indicator candidate.

Examples include anagram, reversal, hidden, homophone, deletion, and letter-selection indicators.

The same word can support different operations in different contexts. `broadcast`, for example, can indicate an anagram or a homophone. The generator should attach operation evidence, not final truth.

`LOCATOR_BLOCK`

A span that identifies a subpart or position inside a source or target.

Examples include `heart of`, `head of`, `initially`, `finally`, `at intervals`, `outside`, and `endlessly`.

Locators must not be flattened into operations. In deletion and selection clues, operation and locator are separate but chained.

`POSITION_BLOCK`

A span that orders or places source blocks or operation results.

Examples include `after`, `before`, `following`, `on`, `over`, and `under`.

Position words are highly context-sensitive. They may be genuine positional instructions, surface connectors, or source-internal material. The candidate generator should preserve these alternatives until grammar and answer mechanics can disambiguate them.

`RELATION_BLOCK`

A span that establishes a relation between source blocks.

Common cases include container, insertion, surrounding, replacement, and adjacency.

Container direction must remain testable. If `A brought in B` appears, the generator should preserve both candidate orientations until answer mechanics proves which source is outer and which is inner.

`CONNECTOR_BLOCK`

A span that may connect definition and wordplay at surface level.

This is not a simple word-list category. A connector candidate must pass grammar sanity checks.

If a preposition has no following object, the model should question connector status and consider source-internal absorption.

This is the GOLDEN RETRIEVER lesson: `on` could not be treated as a harmless connector when the grammar required it to belong with `information on`.

`SOURCE_INTERNAL_BLOCK`

A word or phrase that looked separate but actually belongs inside a source expression.

Examples include a preposition completing a noun phrase, part of a named expression, or idiomatic material needed for synonym lookup.

This block type protects against false repairs.

`SCOPE_BLOCK`

A block representing the span over which an operation applies.

The scope may be known immediately, weakly scoped by local adjacency, unresolved until answer verification, or opposed by grammar.

Reversal and anagram indicators especially require unresolved scope.

`ASSEMBLY_BLOCK`

An inferred answer-building node.

This node has no clue span. It represents the constructed value being tested against the answer.

## Evidence Rules

Evidence should be plural and allowed to disagree.

A node or edge can be supported by one source and opposed by another.

Useful evidence categories include:

- `grammar_pos`
- `grammar_dependency`
- `surface_grammar_objection`
- `indicator_table`
- `synonym_table`
- `abbreviation_table`
- `homophone_table`
- `hidden_string_match`
- `answer_letters`
- `source_mechanism`
- `punctuation`
- `clue_position`
- `definition_by_example_marker`
- `named_entity`
- `enumeration`

Evidence should answer:

- what span is being proposed
- what role is being proposed
- what relationship is being proposed
- whether the evidence supports or opposes that proposal
- whether the evidence is strong, weak, or merely suggestive

## Generation Pipeline

The generator should proceed in layers.

### 1. Token And Grammar Layer

Tokenise the clue and attach grammar features.

This layer should propose:

- noun phrase spans
- verb phrase spans
- prepositional phrase spans
- dependency heads
- punctuation-separated spans
- likely definition windows at clue edges
- grammar objections such as orphaned prepositions

Grammar is not cryptic truth.

It is boundary and plausibility evidence.

### 2. Pre-Operation Layer

Some operations are mechanically faithful enough to generate early candidates.

Hidden clues are the clearest example.

If the answer appears as a contiguous string inside the clue text, forwards or backwards, the generator should create a hidden candidate before broad wordplay search.

This candidate should identify:

- hidden fodder span
- hidden indicator candidates
- leftover definition candidate
- possible all-in-one or semi-all-in-one status

Double definition and Spoonerism can also be pre-operation suspects.

They do not prove the parse, but they narrow the operating space.

### 3. Definition Candidate Layer

Propose definition candidates.

Usual candidates:

- clue-initial phrase
- clue-final phrase
- punctuation-delimited phrase
- noun phrase matching answer type
- adjective or verb phrase matching answer sense

Special candidates:

- whole-surface definition
- implicit definition gap
- definition plus DBE modifier
- cryptic definition surface

The generator should not remove the definition from memory. It may remove it from wordplay search space for a candidate graph, but the definition remains a preserved node.

### 4. Source Candidate Layer

Propose source blocks from all remaining spans and selected source-internal extensions.

Source candidates may overlap.

For each source, preserve:

- clue span
- candidate value or values
- mechanism
- evidence
- confidence
- whether the source requires an operation

The generator should allow source phrases to absorb adjacent words when grammar requires them.

For example, an apparent connector may become `SOURCE_INTERNAL_BLOCK` if the grammar says it completes the phrase.

### 5. Operation And Locator Layer

Propose operations and locators separately.

Examples:

- `loses` -> deletion operation
- `heart of` -> locator
- `endlessly` -> locator/removal candidate
- `upset` -> reversal, anagram, or synonym source candidate depending context

The generator should create chained structures rather than single flattened labels.

For deletion:

```text
OP_BLOCK loses
LOCATOR_BLOCK heart of X
SOURCE_BLOCK target source
```

For letter selection:

```text
LOCATOR_BLOCK finally
SOURCE_BLOCK watercolour
mechanism candidate: last_letter -> R
```

### 6. Relation And Position Layer

Propose relationships between source blocks.

This layer creates edges such as:

- `CONTAINS`
- `CONTAINED_BY`
- `ORDERS`
- `ASSEMBLES_WITH`
- `AWAITING_SCOPE`

It must avoid committing too early.

For a container clue, both orientations may be possible until answer mechanics tests them.

For a charade, order may be strongly suggested by clue order, but position words can override it.

For down clues, `on`, `over`, and `under` require direction-aware interpretation.

### 7. Graph Assembly Layer

Assemble candidate graphs from compatible blocks.

A graph is compatible when:

- it has at least one definition candidate or an explicit definition gap
- it has one or more source candidates
- operation nodes have possible targets
- locators are chained to a source or operation
- relation nodes have possible participating sources
- connector nodes do not violate grammar
- source-internal alternatives are preserved if grammar supports them

The graph should keep unresolved edges explicit.

Unresolved is a valid state, not a failure.

### 8. Answer Mechanics Layer

Test candidate graphs against the answer.

This layer may:

- verify a candidate
- reject a candidate
- rank a candidate lower
- mark a scope as resolved
- propose a missing source value

It must not destroy the evidence trail.

## Verification Rules

Hidden:

- search for answer as contiguous string in clue text
- search both directions
- identify maximal fodder span
- identify indicator candidates
- preserve definition candidate by exclusion or whole-surface ambiguity

Anagram:

- collect candidate fodder values
- compare letter multiset with answer or answer segment
- allow operation scope over multiple sources
- preserve ambiguous scope until letters fit

Deletion:

- test deletion before and after synonym expansion
- preserve operation and locator separately
- allow deletion target to be a letter, abbreviation, phrase part, or semantic locator

Reversal:

- test local source reversal
- test reversal of assembled source chunks
- preserve leading indicators at clue level until answer resolves scope

Charade:

- test source values in clue order
- test position-instructed order
- test compound outputs from container, deletion, reversal, or anagram subgraphs

Container:

- test both inner/outer orientations unless grammar or answer mechanics rejects one
- preserve relation block independently of container direction
- support nested operations

Homophone:

- treat homophone evidence as suspect, not proof
- check homophone table or phonetic similarity
- reject silly sound-only coincidences without clue support
- respect context where words like `say` are definition modifiers rather than homophone indicators

Spoonerism:

- split answer into word windows
- switch initial sounds or letters
- test whether transformed phrase can be parsed from clue sources
- be conservative because Spoonerisms are often loose

Double Definition:

- propose two definition windows
- allow connector between them
- both windows must independently define the answer
- WFW preservation should account for each definition separately

## Failure Guards

The generator must explicitly guard against known traps.

False connector:

- do not classify by word list alone
- require grammar sanity
- if the connector needs an object, ensure the object exists
- if not, consider source-internal absorption

Premature scope:

- do not attach an anagram or reversal indicator to the nearest token merely because it is adjacent
- create `AWAITING_SCOPE` when the final scope depends on answer mechanics

Residue-run flattening:

- a residue run may contain multiple blocks
- `say with` can split into definition modifier plus relation
- surface contiguity is not anatomical unity

Operation-locator flattening:

- `loses heart of X` is not one undifferentiated indicator
- operation and locator preserve separately but remain chained

Missing definition:

- do not treat absent definition as harmless
- create ambiguous definition gap
- allow later whole-surface, residue-surface, or implicit definition hypotheses

Mechanism blindness:

- do not treat every source as a synonym source
- `last_letter`, `first_letter`, `hidden`, `anagram_fodder`, and `deletion` sources carry different structural implications

Parser-head overtrust:

- parser heads may help suggest surface attachment
- they do not define cryptic attachment

## Worked Example: SPITTOON

Clue:

```text
Receptacle former PM brought in without delay
```

Answer:

```text
SPITTOON
```

Intended anatomy:

- definition: `Receptacle`
- source: `former PM` -> PITT
- relation: `brought in`
- source: `without delay` -> SOON
- assembly: PITT inside SOON -> S(PITT)OON -> SPITTOON

Candidate graph:

```json
{
  "candidate_id": "spittoon:candidate_0",
  "clue": "Receptacle former PM brought in without delay",
  "answer": "SPITTOON",
  "definition_status": "observed",
  "nodes": [
    {
      "node_id": "answer",
      "kind": "ASSEMBLY_BLOCK",
      "span": null,
      "text": "SPITTOON",
      "value": "SPITTOON",
      "status": "inferred"
    },
    {
      "node_id": "def_0",
      "kind": "DEF_BLOCK",
      "span": [0, 1],
      "text": "Receptacle",
      "value": "SPITTOON",
      "evidence": [
        "clue_initial_definition_candidate",
        "answer_synonym_candidate"
      ],
      "status": "candidate"
    },
    {
      "node_id": "src_0",
      "kind": "SOURCE_BLOCK",
      "span": [1, 3],
      "text": "former PM",
      "value": "PITT",
      "mechanisms": ["named_entity", "knowledge"],
      "status": "candidate"
    },
    {
      "node_id": "rel_0",
      "kind": "RELATION_BLOCK",
      "span": [3, 5],
      "text": "brought in",
      "operation": "container",
      "status": "candidate"
    },
    {
      "node_id": "src_1",
      "kind": "SOURCE_BLOCK",
      "span": [5, 7],
      "text": "without delay",
      "value": "SOON",
      "mechanisms": ["synonym"],
      "status": "candidate"
    }
  ],
  "edges": [
    {
      "kind": "DEFINES",
      "from_node": "def_0",
      "to_node": "answer",
      "scope_status": "known"
    },
    {
      "kind": "CONTRIBUTES_TO",
      "from_node": "src_0",
      "to_node": "answer",
      "scope_status": "known"
    },
    {
      "kind": "CONTRIBUTES_TO",
      "from_node": "src_1",
      "to_node": "answer",
      "scope_status": "known"
    },
    {
      "kind": "CONTAINS",
      "from_node": "rel_0",
      "to_node": "src_0",
      "scope_status": "answer_aware_scope_needed",
      "notes": "candidate inner source"
    },
    {
      "kind": "CONTAINS",
      "from_node": "rel_0",
      "to_node": "src_1",
      "scope_status": "answer_aware_scope_needed",
      "notes": "candidate outer source"
    }
  ],
  "verification": {
    "status": "mechanically_verified",
    "tested_forms": [
      "PITT containing SOON -> rejected",
      "SOON containing PITT -> SPITTOON -> accepted"
    ],
    "resolved_edges": [
      "src_1 outer",
      "src_0 inner"
    ]
  }
}
```

Design lesson:

The relation block does not itself know the final container direction.

The graph preserves relation plus two source candidates. Answer mechanics resolves orientation.

Only one synonym unlocks much of the graph: once `without delay -> SOON` is found, the answer shape strongly suggests insertion of `PITT`.

## Worked Example: ARTHUR

Clue:

```text
Legendary ruler in craft endlessly upset
```

Answer:

```text
ARTHUR
```

Intended anatomy:

- definition: `Legendary ruler`
- connector: `in`
- source: `craft` -> ART
- source: `upset` -> HURT
- locator: `endlessly` -> remove final letter
- assembly: ART + HUR[T] -> ARTHUR

The important grammar point is that `in` appears after the definition and before the wordplay.

It begins the wordplay-side surface, but it has no object as a positional indicator inside the wordplay. Therefore it is a plausible connector here, not a position instruction.

Candidate graph:

```json
{
  "candidate_id": "arthur:candidate_0",
  "clue": "Legendary ruler in craft endlessly upset",
  "answer": "ARTHUR",
  "definition_status": "observed",
  "nodes": [
    {
      "node_id": "answer",
      "kind": "ASSEMBLY_BLOCK",
      "span": null,
      "text": "ARTHUR",
      "value": "ARTHUR",
      "status": "inferred"
    },
    {
      "node_id": "def_0",
      "kind": "DEF_BLOCK",
      "span": [0, 2],
      "text": "Legendary ruler",
      "value": "ARTHUR",
      "status": "candidate"
    },
    {
      "node_id": "conn_0",
      "kind": "CONNECTOR_BLOCK",
      "span": [2, 3],
      "text": "in",
      "status": "candidate"
    },
    {
      "node_id": "src_0",
      "kind": "SOURCE_BLOCK",
      "span": [3, 4],
      "text": "craft",
      "value": "ART",
      "mechanisms": ["synonym"],
      "status": "candidate"
    },
    {
      "node_id": "loc_0",
      "kind": "LOCATOR_BLOCK",
      "span": [4, 5],
      "text": "endlessly",
      "operation": "remove_final_letter",
      "status": "candidate"
    },
    {
      "node_id": "src_1",
      "kind": "SOURCE_BLOCK",
      "span": [5, 6],
      "text": "upset",
      "value": "HURT",
      "mechanisms": ["synonym"],
      "status": "candidate"
    }
  ],
  "edges": [
    {
      "kind": "DEFINES",
      "from_node": "def_0",
      "to_node": "answer",
      "scope_status": "known"
    },
    {
      "kind": "SURFACE_CONNECTS",
      "from_node": "conn_0",
      "to_node": "def_0",
      "scope_status": "surface_only_until_grammar_check",
      "notes": "definition-to-wordplay connector candidate"
    },
    {
      "kind": "CONTRIBUTES_TO",
      "from_node": "src_0",
      "to_node": "answer",
      "scope_status": "known"
    },
    {
      "kind": "LOCATES_WITHIN",
      "from_node": "loc_0",
      "to_node": "src_1",
      "scope_status": "weakly_scoped_from_operation",
      "notes": "remove final letter from HURT"
    },
    {
      "kind": "CONTRIBUTES_TO",
      "from_node": "src_1",
      "to_node": "answer",
      "scope_status": "known"
    },
    {
      "kind": "ASSEMBLES_WITH",
      "from_node": "src_0",
      "to_node": "src_1",
      "scope_status": "known",
      "notes": "ART + HUR"
    }
  ],
  "verification": {
    "status": "mechanically_verified",
    "tested_forms": [
      "craft -> ART",
      "upset -> HURT",
      "endlessly(HURT) -> HUR",
      "ART + HUR -> ARTHUR"
    ]
  }
}
```

Design lesson:

The word `in` is not allowed to become a positional indicator just because it is an indicator-like preposition.

The graph carries it as a connector candidate with grammar evidence.

The actual wordplay consists of a straight source plus a modified source.

## Graph Ranking

The generator should rank candidates but not over-prune.

High-ranking evidence:

- answer mechanics exactly reproduces the answer
- definition candidate independently matches the answer
- hidden string exactly matches the answer
- source mechanisms align with indicator evidence
- grammar supports the proposed span boundaries
- relation direction is verified by answer mechanics

Medium-ranking evidence:

- indicator lexicon supports operation
- clue position supports definition
- dependency parse supports source phrase
- punctuation separates definition and wordplay
- clue direction supports position indicator

Negative evidence:

- connector candidate violates grammar
- source candidate requires unsupported synonym leap
- operation has no plausible target
- locator has no source to locate within
- answer mechanics fails
- source-internal phrase was split despite grammar requiring it
- definition marker is treated as wordplay without contextual support

Rejection should be explicit.

A rejected graph remains useful because it records which assumption failed.

## Candidate Generator Deliverable

For each fresh clue, GT V2 should be able to emit:

- a small set of candidate graphs
- all proposed block nodes
- all relationship edges
- evidence supporting and opposing each important decision
- answer-mechanics verification status
- unresolved scope notes
- rejection notes for failed candidates

The ideal output is not necessarily one graph.

For hard clues, the right output may be:

- one verified graph
- two plausible unresolved graphs
- several rejected graphs showing why obvious readings failed

## Implementation Questions For Later

These are not to be implemented in this design pass.

The next engineering design will need to decide:

- how candidate spans are enumerated without exploding combinatorially
- which lexical resources provide synonym and abbreviation candidates
- how named-entity evidence is introduced safely
- how grammar objections are encoded
- how answer mechanics writes evidence back to graph edges
- how graph candidates are stored and displayed
- how WFW preservation consumes the final verified graph
- how human review corrects a graph without losing evidence

## Non-Goals

This document does not:

- modify solver code
- define production database changes
- replace WFW preservation rules
- claim grammar can solve the clue alone
- claim the supervised reports are production truth

The purpose is narrower:

Define the anatomy object GT V2 should generate from a fresh clue.

## Stop Condition

Before implementation begins, this design should be tested against a small set of hand-reviewed clues.

The test is not whether GT V2 solves them.

The test is whether the candidate graphs preserve the distinctions humans care about:

- definition versus wordplay
- source versus source-internal phrase material
- operation versus locator
- relation versus direction
- connector versus grammar requirement
- known scope versus answer-aware scope
- explicit definition versus definition gap

If the graph can preserve those distinctions, the solver has a better search space.

If it cannot, implementation would only make the wrong model faster.
