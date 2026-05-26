# Operation-Derived Attachment Targets

Date: 2026-05-17

The residue attachment experiment showed that surface adjacency is not enough.

The next target must be derived from operation anatomy.

This note defines the attachment labels we should mine from structured explanations.

## Why This Matters

GT V2 is not trying to predict clue type as a headline category.

It is trying to recover block anatomy:

- source blocks
- residue blocks
- operation blocks
- locator blocks
- order/position blocks
- definition blocks

The important question is not merely whether a word is `RESIDUE`.

The question is what the residue does to the source material.

## Proposed Attachment Labels

`OPERATOR_SCOPE`

The residue run licenses an operation over one or more source spans.

Examples:

- anagram indicator over anagram fodder
- hidden indicator over hidden fodder
- homophone indicator over a sound-like source
- reversal indicator over a reversible source

`LOCATOR_SCOPE`

The residue run identifies which part of a source should be used.

Examples:

- `heart of`
- `head of`
- `tail of`
- `outside of`
- `at intervals`

This must be chained to the operation it supports. For deletion, `loses heart of X` is not a single flat indicator; it is an operation plus locator.

`ORDER_OR_POSITION`

The residue run orders or positions source spans without contributing letters.

Examples:

- `after`
- `before`
- `on`
- `under`
- `over`
- `following`

These are especially dangerous because they can also be ordinary grammar or part of a source phrase.

`CONTAINER_RELATION`

The residue run defines containment between source spans.

Examples:

- `in`
- `around`
- `holding`
- `eating`
- `swallowing`
- `interrupting`

This should record container and contained spans separately.

`CONNECTOR_OR_SURFACE`

The residue run exists primarily to make the surface grammatical or connect definition and wordplay.

This must not be identified by word list alone.

The GOLDEN RETRIEVER correction is the warning case: `on` could not be treated as a link because it was grammatically orphaned. It belonged with `information on`.

`SOURCE_INTERNAL`

The apparent residue should probably be absorbed into a source phrase.

Examples:

- `information on` -> `GEN`
- phrases where a preposition completes the ordinary sense of the synonym phrase
- grammar units that wordplay should not split unless the split is justified as a cryptic transformation

`DIRECTION_OR_ORIENTATION`

The residue run gives direction, but scope must be answer-aware.

Examples:

- `up`
- `back`
- `from the east`
- `reversed`
- `returning`

This is not resolved by position alone. In down clues, `up` can be a reversal instruction or a positional instruction.

`DEF_MODIFIER`

The apparent residue belongs with the definition, often as a definition-by-example marker.

Examples:

- `say`
- `perhaps`
- `maybe`
- `for example`

This must be contextual. In homophone clues, `did you say` may be genuine wordplay. In definition phrases such as `Slug, say`, the same word may be a definition modifier.

## Deriving Labels From Existing Structured Data

The structured explanation gives us:

- operation type
- source pieces
- mechanisms
- assembly object
- source token spans
- residue token runs

A first label pass can be weakly supervised:

- anagram operation plus residue near fodder -> `OPERATOR_SCOPE`
- hidden or hidden-reversed operation plus residue near hidden words -> `OPERATOR_SCOPE`
- homophone operation plus residue phrase -> `OPERATOR_SCOPE`
- reversal operation plus direction residue -> `DIRECTION_OR_ORIENTATION`
- container operation plus containment residue -> `CONTAINER_RELATION`
- deletion operation plus locator terms -> `LOCATOR_SCOPE`
- charade operation plus order terms -> `ORDER_OR_POSITION`
- residue phrase whose parser head is source, and whose words complete a known phrase -> `SOURCE_INTERNAL` candidate
- definition-by-example marker adjacent to definition -> `DEF_MODIFIER` candidate

This will be noisy, but that is acceptable for the science phase. The purpose is to create inspectable targets, not production truth.

## What Grammar Should Contribute

Grammar should not decide the operation by itself.

Grammar should help answer narrower questions:

- does a residue phrase syntactically attach to source material?
- does a preposition complete a source phrase?
- does a supposed connector have the required grammatical object?
- does a direction word modify the source span or the whole clue surface?
- does a noun phrase or prepositional phrase cross a proposed wordplay boundary?

Those are the questions that map directly onto the WFW preservation design.

## Next Corpus Artifact

The next artifact should be an operation-attachment training slice.

Each record should contain:

- clue
- answer
- operation
- token labels
- source spans
- residue runs
- proposed attachment label for each residue run
- attached source span ids, if derivable
- grammar evidence for or against that attachment
- raw structured explanation and blog explanation where available

This should be readable one clue at a time.

No tables are needed in the design document. The JSONL can carry the structured detail; the markdown report should carry examples and interpretation.
