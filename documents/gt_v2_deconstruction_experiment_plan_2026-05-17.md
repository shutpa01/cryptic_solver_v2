# GT V2 Deconstruction Experiment Plan

Date: 2026-05-17

This note records the current R&D direction after mining the structured explanation corpus.

## Objective

The objective is not to improve the production solver yet.

The objective is to discover whether grammar can help recover clue block anatomy:

- which clue words form answer-producing source blocks
- which words are definition
- which residue words are operation indicators
- which residue words are positional or grammatical glue
- which residue words must be chained to a source block rather than treated independently

The central test is the user's claim:

Grammar signature and wordplay signature must emanate from the same words.

If wordplay wants to split a grammar unit, that split must be explicit and justified as a cryptic transformation.

## What We Have Established

The existing structured explanations are already a useful supervision source.

They give us:

- clue text
- answer
- definition span, when available
- structured source pieces
- letters contributed by each source piece
- mechanism or assembly operation
- optional blog explanation

This means we do not need blogs as the primary source of truth. Guardian and Independent blogs are still useful because they often explain setter intent more fully, but Daily Mail and Telegraph remain valuable because their structured explanations already map clue words to answer pieces.

## Current Research Slice

The clean boundary slice contains 931 high-confidence records where structured pieces map cleanly back to clue text.

For each record we can mark clue tokens as:

- `DEF`
- `SOURCE`
- `RESIDUE`

This gives us a supervised clue-anatomy surface before we ask the solver to infer anything.

## Residue Baseline

A residue-only baseline has now been measured.

It deliberately ignores grammar, syntax, answer letters, and block mechanics.

It predicts the assembly operation using only residue words left after `DEF` and `SOURCE` tokens are removed.

With a minimum support threshold of 3 training examples:

- coverage: 193/247 test clues, 78%
- overall accuracy: 112/247, 45%
- accuracy when covered: 112/193, 58%
- high-confidence accuracy: 30/35, 86%

This is the control group. Grammar must improve on this, not merely rediscover obvious indicator words.

## What The Baseline Teaches

Residue is strong when it contains a conventional indicator phrase:

- `reportedly`
- `on the radio`
- `we hear`
- `some`
- `partly`
- `section of`
- `to some extent`

Residue is weak when the remaining words are mostly glue:

- `in`
- `with`
- `by`
- `for`
- `of`
- `on`
- `and`
- `to`

This matches the design concern from GOLDEN RETRIEVER: connector-like words cannot be classified by word list alone. They have to obey the grammar of the clue surface.

## Grammar Triage Target

The next grammar experiment should not begin by predicting clue type.

That was the V1 ambition, and it is not enough.

The next experiment should ask whether grammar can protect or recover block boundaries:

- Does the grammar unit align with a `SOURCE` block?
- Does an operation word attach to the preceding source, following source, or whole wordplay span?
- Is a preposition an orphan connector, a positional instruction, or part of a source phrase?
- Is an apparent link word actually required by the source phrase?
- Is a direction word attached to a hidden span, a reversal span, or a down-clue positional instruction?

## First Target Buckets

The error-space report identifies the best first buckets.

`glue-only` and `glue-dominates` are the strongest grammar candidates.

In these cases the residue lexicon mostly sees prepositions and conjunctions. Syntax and attachment should decide whether those words are glue, operation, or part of a phrase.

`indicator-plus-glue` tests whether an operation word can stay chained to the correct source block without nearby glue stealing the decision.

`direction-or-scope` tests reversal, hidden, and reversed-hidden cases where the same residue words can point to different operations depending on scope and answer confirmation.

`low-support-indicator` is mostly a data problem. The operation signal exists, but the clean slice has not seen enough examples yet.

## Existing GT Reading

The active production GT is `signature_solver/grammar_triage.py`.

It already contains useful concepts:

- definition is stripped before wordplay analysis
- POS tags are abstracted into mid-level grammar tags
- role candidates are verified mechanically against the answer
- role sets separate fodder and indicators
- positional indicators are checked separately from coincidental letter extraction

But V1 is still role-sequence oriented.

V2 needs a preceding block-anatomy layer:

1. identify likely grammar spans
2. compare grammar spans with supervised `SOURCE` spans
3. identify where residue words attach
4. only then assign cryptic roles and verify mechanically

## Next Experiment

Build a grammar-feature scaffold over the 931-record clean slice.

For each clue, store:

- token text
- supervised label: `DEF`, `SOURCE`, `RESIDUE`
- source piece index, if any
- residue runs
- source span boundaries
- simple grammar features available without solver changes

If spaCy is available, add:

- fine POS tag
- mid POS tag
- dependency label
- syntactic head
- noun chunks
- verb/preposition attachment

Then measure simple questions before building a model:

- How often do noun chunks align with source blocks?
- How often do prepositional phrases sit wholly inside residue?
- How often does a preposition attach to a source block rather than stand alone?
- In failed residue-baseline cases, does POS/dependency context distinguish operation from glue?
- Do hidden and reversed-hidden clues have distinctive grammar/position patterns once the source span is known?

This keeps the work scientific. We are not writing solver fixes. We are testing whether grammar can actually expose block anatomy at scale.

## First Grammar Results

spaCy 3.8 and `en_core_web_sm` were installed in the project venv so the scaffold could be enriched with parser features.

The enriched scaffold now contains:

- POS tags
- mid-level POS tags
- dependency labels
- parser heads
- supervised `DEF`, `SOURCE`, and `RESIDUE` labels
- source span ids
- residue run ids

The first boundary-signal report found:

- 931 records have grammar features
- 2,235 `SOURCE` tokens
- 1,913 `RESIDUE` tokens
- when a `SOURCE` token's parser head is also `SOURCE`, it stays inside the same supervised source span 883/1,151 times, or 77%

That is useful but not authoritative. Parser structure often respects source spans, but it also crosses cryptic boundaries.

A small classifier then compared two token-boundary models over wordplay tokens only:

- lexical-only: 715/1,060 correct, 67%
- lexical-plus-grammar: 758/1,060 correct, 72%
- gain from grammar: 43 correctly labelled tokens

This is the first hard evidence that grammar adds signal beyond residue/word identity, but the gain is modest.

The correct interpretation is:

- grammar is promising for boundary and attachment triage
- grammar should not be treated as a primary solver or clue-type oracle
- grammar is especially useful where residue glue attaches syntactically to source material
- grammar still needs answer-aware and operation-aware verification

The next experiment should move from token labelling to span labelling:

- predict contiguous `SOURCE` spans
- predict contiguous `RESIDUE` runs
- test whether grammar helps attach residue runs to the left source, right source, or whole wordplay span
- measure this specifically on the residue-baseline failure buckets

## First Span Results

The first span-level experiment has now been run from the enriched scaffold.

It compared lexical-only and lexical-plus-grammar token classifiers, then converted their predictions into contiguous `SOURCE` and `RESIDUE` spans.

Results:

- lexical exact `SOURCE` span recall: 88/288, 31%
- grammar exact `SOURCE` span recall: 125/288, 43%
- lexical exact `RESIDUE` span recall: 59/291, 20%
- grammar exact `RESIDUE` span recall: 114/291, 39%
- lexical boundary-pair accuracy: 367/813, 45%
- grammar boundary-pair accuracy: 404/813, 50%

This is stronger evidence than the token-level result because exact spans are much harder.

The interpretation is still cautious:

- grammar improves every span measure
- exact span recovery remains too weak for a single hard parse
- glue words are still the main failure mode
- V2 should generate a small set of plausible block anatomies rather than committing to one grammar parse

The next useful experiment is residue attachment:

- for each contiguous residue run, predict whether it attaches left, right, both sides, or the whole wordplay span
- test especially on `glue-dominates`, `indicator-plus-glue`, and `direction-or-scope`
- preserve grammar evidence as WFW support, not as a destructive rewrite of the clue

## Residue Attachment Result

The first residue-attachment experiment exposed a design flaw.

If the gold target is defined as nearest adjacent `SOURCE` material, then a nearest-source baseline scores 291/291 by construction. That is not a meaningful model.

A parser-head attachment test was more informative:

- residue runs tested: 291
- parser-head attachment hits against the adjacency proxy: 87/291, 30%

This is a useful negative result.

Parser heads are not cryptic attachment.

The parser often describes the surface sentence correctly, while cryptic attachment follows the wordplay operation.

Therefore the next attachment target must be derived from operation anatomy, not surface adjacency:

- anagram indicator attaches to all anagram fodder in scope
- deletion operation attaches source and deletion target as a chained block
- hidden indicator attaches to hidden fodder span
- reversal direction attaches to a reversible source span, with answer-aware direction confirmation
- charade connectors may indicate order without consuming source material

This keeps the design aligned with the WFW rule: operation and locator may preserve separately, but must remain chained.

## Residue Splitting Result

The operation attachment slice exposed another important rule:

A contiguous residue run is not necessarily one block.

Example:

- `Slug, say, with time in gaps door created`
- surface residue run: `say with`
- `say` is a definition modifier
- `with` is a wordplay relation

The graph builder now splits mixed residue runs into separate nodes while preserving evidence that they came from the same surface run.

This is a strong design confirmation. GT V2 must not flatten residue runs before block anatomy is known.
