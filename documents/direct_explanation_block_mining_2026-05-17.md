# Direct Explanation Block Mining

Date: 2026-05-17

## Purpose

This note follows the first block-anatomy mining report and goes back one layer,
from `word_roles.db` to the human explanations themselves.

The conclusion is strong: the human explanation corpus should be treated as the
primary source for block anatomy. The role table is useful, but it is already a
flattened derivative. The explanations still preserve the relationships we care
about: source phrase, produced letters, operation, indicator, deletion material,
container relation, reversal, anagram fodder, and occasionally the setter's
intended gloss.

This remains design work. No solver behaviour should be changed from this pass.

## What Was Checked

I sampled 5,000 rows from `data/times_explanations.db` and passed the human
explanation text through the existing notation parser and mapper:

- `cryptic_taxonomy/analysis/notation_parser.py`
- `cryptic_taxonomy/analysis/improved_mapper.py`

The parser produces explanation pieces with useful anatomy already attached:

- produced letters
- source type
- source word or source expression
- gloss
- operation type
- sub-operations where the notation exposes them

The mapper then tries to attach those pieces back to clue word spans. That step
is useful, but it is also where a lot of phrase evidence gets weakened.

## Sample Result

From the first 5,000 explanation rows:

- 3,075 parsed as verified
- 1,925 remained unverified
- 1,462 clues had mapped pieces
- 1,379 failed the mapping step
- 5,633 piece records were extracted

The dominant parsed operation families in the sample were charades, containers,
anagrams, synonyms, abbreviations, hidden clues, double definitions, reversals,
homophones, and deletions.

That spread is exactly what we need. It means the existing explanation parser is
not just finding final answers; it is already exposing much of the intermediate
working.

## Why Direct Explanations Matter

The `word_roles.db` layer asks:

What role did this clue word play?

The direct explanation layer can ask a better question:

What block did the human explanation say existed?

That distinction matters.

In the role table, a phrase can be broken into separate word rows too early. Once
that happens, the solver has to rediscover that `that is`, `in charge`, `the
French`, `without delay`, or `small issue of litter` was meant to live as one
block.

In the explanation, that unity is often still visible.

For WFW preservation and Grammar Triage, that is the signal we need most.

## What The Parser Preserves

The parser already preserves several kinds of block anatomy that are hard to
recover later.

For ordinary source blocks, it can preserve produced letters and the human gloss.
That lets us distinguish the answer material from the clue phrase that licensed
it.

For anagrams, it preserves the fodder expression rather than only the resulting
letters. That matters because anagram scope is often only clear once all pieces
are collected.

For deletions, it can preserve both the surviving material and the removed
material. Examples like `{w} ELDER (joiner) [dismissing wide]` and `{h} OLD
(keep) [the first letter to be destroyed]` show exactly the kind of operation
chain the WFW design requires.

For homophones, it preserves the spoken source candidate. That is where the
"suspect, not solve" rule belongs: a table-backed sound match plus a possible
indicator creates a candidate, but grammar and fairness still have to confirm it.

For reversals, it preserves that a source block was reversed. This fits the
settled rule that reversal indicators should not be greedily attached before
scope is known.

For hidden clues, it preserves the hidden source string. Whole hidden clues still
belong in the pre-op layer because the answer match is mechanically faithful,
but the explanation remains useful for identifying full fodder and indicator.

## What The Mapper Loses

The current mapper is useful but too word-level for the design we are now
building.

A block such as `harbour areas` may be reduced to `harbour`. That is not harmless:
the missing word can be part of the grammatical signature, and it can affect
whether nearby words are live source material, qualifiers, or indicators.

Some mappings fail even when the explanation is clear. That is not just a bug
list; it is high-value design evidence.

Examples from the sample include:

- `NOT (never) contained by [entering] S~TY (dirty home)`
- `RUNT (small issue of litter) containing ECO (green)`
- `HAM (amateur) containing RE (respecting)`
- hidden material such as `smitTEN ORC LEFt`

Those failures show the next extractor must be phrase-first. It should not begin
by assigning each clue token a role and then hoping phrase structure emerges.

## SPITTOON Revisited

For the clue:

`Receptacle former PM brought in without delay`

The desired anatomy is:

- `Receptacle` defines the answer
- `former PM` produces `PITT`
- `without delay` produces `SOON`
- `brought in` supplies the container relation
- the assembly puts `PITT` inside `SOON` to make `SPITTOON`

The important point is not merely that this is a container clue. The important
point is that one strong block can unlock the whole signature.

If `former PM` gives `PITT`, then the remaining answer geometry asks where
`PITT` can sit inside the rest. If `without delay` gives `SOON`, the container
shape becomes even more constrained. The solver should be able to move from a
single trustworthy block to the likely anatomy of the clue.

That is why block anatomy must be earlier than full mechanical solving.

## ARTHUR Revisited

For the clue:

`Legendary ruler in craft endlessly upset`

The desired anatomy is:

- `Legendary ruler` defines the answer
- the wordplay space begins with `in`
- because leading `in` is orphaned as a positional or container instruction, it
  can be treated as a genuine connector here
- `craft` produces `ART`
- `upset` produces `HURT`
- `endlessly` removes the final letter, giving `HUR`
- `ART` plus `HUR` gives `ARTHUR`

This is the same design principle as the GOLDEN RETRIEVER correction, but in the
opposite direction.

In GOLDEN RETRIEVER, `information on` must be allowed to live as a phrase. In
ARTHUR, leading `in` cannot sensibly operate on a preceding object because there
is none. So it is available as a connector.

The rule is not "on is a link" or "in is a link". The rule is that small words
must obey the grammar of the block they claim to belong to.

## The Next Extractor

The next useful artefact should be a direct block-candidate extractor from the
human explanations.

It should start from explanation pieces, not from `word_roles.db`.

For each parsed clue, it should preserve:

- the original clue
- the answer
- the extracted definition, if present
- the human explanation
- the parsed operation
- each produced answer fragment
- each source phrase or source expression
- each gloss
- each explicit operation marker
- each candidate clue span
- any mapping objection or ambiguity

This is not yet a final database schema. It is the design shape of the record we
need.

The important discipline is preservation. If the explanation says `RUNT (small
issue of litter) containing ECO (green)`, the extractor must preserve `small
issue of litter` as a candidate source block. It can later decide whether the
clue text supports that full phrase, but it must not prematurely collapse it to a
single word.

## Mapping Rule

The mapping order should become:

First, try exact phrase evidence from the explanation.

Then try explicit gloss-to-clue matching.

Then try known phrase lookups.

Then try answer-fragment geometry.

Only after that should it fall back to single-word role assignment.

This is the key design correction. Word roles are still useful, but they should
be fallback evidence, not the first thing that destroys the phrase.

## Why Failures Are Valuable

Mapping failures should be stored, not discarded.

A failed mapping often means one of three useful things:

- the explanation contains a phrase block the current mapper cannot see
- the clue has an operation chain that needs later binding
- the definition stripping has removed too much or too little, especially in
  &lit and semi-&lit clues

Those are exactly the cases Grammar Triage has to learn from.

So the next corpus should contain both successes and failures. The failures are
where the block-anatomy design will improve fastest.

## Follow-Up Corpus Probe

I then scanned 20,000 explanation rows directly for phrase-shaped pieces.

This confirmed the opportunity, but also exposed a useful warning. A naive phrase
extractor over-counts abbreviation gloss pairs such as `I (one)`, `O (old)`, or
`E (English)`. Those are valid WFW facts, but they are not the phrase-block
problem we are trying to solve.

Once that noise is recognised, the same pass still surfaces the important block
signals. Examples include `in charge` producing `IC`, `at home` producing `IN`,
`on the way` producing `ROUTE`, `hospital department` producing `ENT`, and the
failed-map phrase `small issue of litter` producing `RUNT`.

The failed-map example is especially important:

`RUNT (small issue of litter) containing ECO (green)`

The existing parser can see the container as a whole, but the next design layer
must preserve both internal source blocks:

- `small issue of litter` -> `RUNT`
- `green` -> `ECO`
- `engulfing` / `containing` as the container relation

That is exactly the anatomy we lose if we start from single clue words.

This follow-up also shows why raw evidence must travel with interpreted evidence.
Some explanation notations are clean, such as `IC (in charge)`. Others are prose
heavy or imperfectly parsed. The extractor should therefore preserve the raw
human explanation, the parser's current interpretation, and any mapping
objection together. If later grammar work improves the interpretation, the
original evidence is still there.

## Design Conclusion

The route forward is now clearer.

We should not begin by modifying the solver engine. We should first build a
block-anatomy corpus from human explanations, preserving the explanation's own
structure as much as possible.

That corpus can then be used to design and stress-test Grammar Triage:

- Can it find the definition window?
- Can it keep phrase blocks intact?
- Can it distinguish source blocks from operation blocks?
- Can it leave later-binding indicators unattached until scope is known?
- Can it reject connector-like words when the grammar leaves a dangling object?
- Can it identify when a single strong block unlocks the rest of the clue?

This is the Rosetta stone work. The explanations already contain more anatomy
than the current GT is using. The next job is to extract that anatomy without
flattening it away.

## Extractor Artefact

The first standalone extractor has now been added:

`scripts/extract_explanation_blocks.py`

It reads `data/times_explanations.db`, parses the human explanations, and writes
a reviewable block-candidate corpus without touching the solver pipeline.

The first generated sample is:

- `documents/explanation_block_candidates_2026-05-17.jsonl`
- `documents/explanation_block_candidates_2026-05-17.md`

The sample scanned 20,000 explanation rows with clue text and wrote 16,389
interesting records. It preserved 34,351 extracted blocks, including 7,626
phrase-shaped blocks.

The output is deliberately not treated as final truth. It preserves raw
explanation text, parser interpretation, candidate phrases, exact clue-span
matches, and objections together. That is the important design step: even when
the current parser is wrong or noisy, the evidence needed to improve Grammar
Triage is still present.
