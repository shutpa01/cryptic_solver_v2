# GT V2 Block Anatomy Model

Date: 2026-05-17

This note consolidates the current science result into a working model for clue anatomy.

It is not a solver implementation plan yet.

It defines the objects GT V2 should try to recover before mechanical solving.

## Core Claim

A clue should be deconstructed into blocks before it is solved mechanically.

The block model must preserve:

- the clue's surface grammar
- the wordplay operation
- the source material
- the relationship between operation and source
- the definition
- any ambiguity that remains unresolved until answer verification

The grammar signature and the wordplay signature must come from the same words.

If wordplay splits a grammar unit, that split must be visible and justified.

## Block Types

`DEF_BLOCK`

The definition window.

This is removed from the wordplay search space once identified, but it remains preserved as part of the explanation.

`DEF_MODIFIER_BLOCK`

A word or phrase that modifies the definition rather than the wordplay.

Examples:

- `say`
- `perhaps`
- `maybe`
- `for example`

These often mark definition-by-example.

They must not be allowed to leak into residue and become false wordplay structure.

`SOURCE_BLOCK`

A clue span that contributes letters or a transformed value to the answer.

Examples:

- direct synonym source
- abbreviation source
- raw fodder
- hidden fodder span
- homophone source
- deletion source
- reversal source

`OP_BLOCK`

A residue span that licenses an operation.

Examples:

- anagram indicator
- hidden indicator
- reversal indicator
- homophone indicator
- container indicator
- deletion indicator

`LOCATOR_BLOCK`

A residue span that identifies a part or position within a source.

Examples:

- `heart of`
- `head of`
- `at intervals`
- `outside of`
- `last of`

This is not the same as an operation. In `loses heart of X`, `loses` is the operation and `heart of X` is the locator/target relation. They preserve separately but remain chained.

`POSITION_BLOCK`

A residue span that orders or positions source blocks.

Examples:

- `after`
- `before`
- `following`
- `on`
- `over`
- `under`

This block is especially context-sensitive because the same words can be surface glue, positional instruction, or part of a source phrase.

`RELATION_BLOCK`

A residue span that establishes a relationship between two source blocks.

Examples:

- container and contained
- insertion
- surrounding
- replacement

`CONNECTOR_BLOCK`

A span whose role is primarily grammatical or connective.

This is not a word-list category. It must pass a grammar sanity check.

The GOLDEN RETRIEVER correction shows the failure mode: `on` cannot be treated as a connector if the surface grammar leaves it orphaned. In that clue, `information on` belongs together as source phrase evidence for `GEN`.

`SOURCE_INTERNAL_BLOCK`

An apparent residue word that actually belongs inside a source phrase.

This is the most important new block type for avoiding false fixes.

Examples:

- `information on` -> `GEN`
- source phrases completed by prepositions
- named or idiomatic phrases that must not be split without justification

`SCOPE_BLOCK`

A block representing the operation's span of action.

This may be wider than the adjacent word.

Examples:

- anagram over multiple fodder blocks
- reversal over a complete charade
- container operation over two independently sourced blocks
- hidden operation over a contiguous phrase

Scope may remain unresolved until the answer is checked.

## Block Relationships

Blocks are not just labels. The relationships matter.

`OPERATES_ON`

An `OP_BLOCK` acts on one or more `SOURCE_BLOCK`s.

`LOCATES_WITHIN`

A `LOCATOR_BLOCK` identifies a subpart of a source or target.

`ORDERS`

A `POSITION_BLOCK` orders source blocks or operation results.

`CONTAINS`

A `RELATION_BLOCK` establishes container and contained roles.

`BELONGS_TO_SOURCE`

A token that looked like residue is absorbed into a `SOURCE_INTERNAL_BLOCK`.

`AWAITING_SCOPE`

The operation is visible, but its exact source span cannot be resolved until answer verification.

This matters for reversals and anagrams, where early attachment can wrongly make an indicator unavailable at clue level.

## Why Token Labelling Is Not Enough

The first token classifier gave:

- lexical-only: 67%
- lexical-plus-grammar: 72%

The first span experiment gave:

- grammar exact `SOURCE` span recall: 43%
- grammar exact `RESIDUE` span recall: 39%
- grammar boundary-pair accuracy: 50%

That is useful signal, but not a solve.

The system should therefore not ask:

Can grammar produce the one true parse?

It should ask:

Can grammar generate and rank plausible block anatomies so answer mechanics has a much smaller operating space?

## Residue Runs Are Not Blocks

A contiguous residue run is only a surface span.

It may contain more than one anatomical block.

Example:

- `Slug, say, with time in gaps door created`
- the residue run `say with` must split
- `say` is a `DEF_MODIFIER_BLOCK`
- `with` is a relation/position block attached to wordplay

This matters because otherwise the model must choose between two wrong flattenings:

- treat all of `say with` as definition modifier and lose `with`
- treat all of `say with` as wordplay relation and lose the definition-by-example marker

GT V2 must therefore permit residue-run splitting before graph verification.

## Required Ambiguity Preservation

Several ambiguity types must be preserved rather than prematurely solved.

Reversal:

- an indicator at the beginning may apply to the next piece or to a larger assembled span
- scope becomes clear only once the answer confirms it

Anagram:

- fodder may be multiple source blocks
- indicator scope may include a phrase rather than a single word
- compound operations may require collecting pieces first

Deletion:

- operation and locator must be preserved separately
- `loses power` is not necessarily raw-letter deletion from the clue word
- the deletion source may be a synonym expansion before deletion

Hidden:

- if the answer is present as a contiguous bidirectional string, this is a pre-op candidate
- the remaining task is to identify full fodder span and indicator
- definition is already known by exclusion/verification

Homophone:

- homophone evidence is a suspect, not a solve
- source must be plausible and not merely a silly sound coincidence

Connector/source ambiguity:

- apparent connector words must obey surface grammar
- if the connector lacks its required object, it may belong to a source phrase

Definition marker leakage:

- `say`, `perhaps`, `maybe`, and similar markers may belong with the definition
- if they leak into residue they can poison operation attachment
- homophone uses such as `did you say` remain valid wordplay candidates, so the distinction is contextual

Missing definition:

- absence of a labelled definition span is itself a signal
- some hidden or semi-all-in-one clues may define through the whole surface
- some structured explanations map the source/fodder but leave the definition implicit
- definition-by-example markers can be present even when the definition anchor is missing
- letter-selection sources may swallow locator or definition surface if the explanation maps a phrase too coarsely

GT V2 should preserve this as `definition_status`, not quietly collapse it into residue.

Source mechanism preservation:

- a `SOURCE_BLOCK` is not just text
- it must preserve the mechanism that produced answer material
- `synonym`, `abbreviation`, `hidden`, `anagram_fodder`, `first_letter`, `last_letter`, `deletion`, and similar mechanisms carry different anatomical risks
- in particular, a source with `last_letter` is partly an operation-locator construction, even if the structured explanation mapped it as one source span

## Design Consequence

GT V2 should produce block anatomy candidates.

Each candidate should contain:

- exact clue spans
- block type
- block relationship
- source piece ids
- operation type, if known
- scope status
- grammar evidence
- answer-mechanics evidence
- unresolved ambiguity notes

The solver should then verify candidates mechanically.

The design must avoid the trap of turning every new failure into a code patch. The corpus experiments are there to teach the anatomy first.
