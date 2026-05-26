# WFW Preservation Design - Session Notes

Date: 2026-05-16

## Purpose

This note captures the WFW preservation design discussion so the next session can continue without relying on memory.

The key shift is:

**Do not save only an explanation. Save the working-out.**

Once the parse creates durable relationships between clue text, working letters, mechanisms, and answer letters, the remaining WFW display becomes much more straightforward.

## Main Insight

Previously, a parse was only the beginning, because the system did not preserve enough structure. Later stages could flatten or overwrite useful working-out.

The new aim is different:

**The parse creates all the useful relationships. The UI displays those relationships.**

The UI should not need to solve the clue again.

## Settled Principles

### 1. Preserve The Original Text

The original clue text is never overwritten.

The exact published clue must be preserved, including punctuation, apostrophes, quotation marks, spaces, and hyphens.

Any cleaned, split, expanded, or normalised version is stored separately and linked back to the original text.

### 2. Character Tokens Are The Foundation

Every clue character should have a stable token.

Every answer letter should also have a stable token.

This gives us a permanent lowest-level layer for:

- splitting words
- joining words
- preserving punctuation
- mapping clue material to answer material
- colour-coding answer letters
- verifying placement later

### 3. Original Tokens And WFW Tokens Are Separate

The original clue token remains preserved exactly as written.

The WFW layer may create refined tokens when the solve requires it.

Example:

`president's`

Original token:

- `president's`

WFW tokens, if needed:

- `president`
- `'s`

Both WFW tokens link back to the original token.

### 4. Store Transformations As Data

If the system changes, splits, normalises, expands, excludes, substitutes, deletes, or rearranges something, that transformation must be stored as data.

It must not be hidden inside code.

No one should have to reverse-engineer WFW behaviour from a compound function later.

### 5. Definitions Define The Answer As A Whole

The definition defines the answer as a whole, even when the answer is split across more than one grid entry.

Do not say the definition defines a wordplay segment unless dealing with a genuinely separate clue/answer structure.

### 6. Definition Qualifiers Are Separate

Definition text is stored separately from definition qualifiers.

Examples:

`Perhaps trunk`

- definition text: `trunk`
- definition qualifier: `Perhaps`
- qualifier type: definition by example

`Tarzan?`

- definition text: `Tarzan`
- definition qualifier: `?`
- qualifier token is stored separately from `Tarzan`

Do not store subjective categories such as "loose definition" as core truth. Store the observable qualifier token instead.

### 7. Connectors Sit Between Parts

A connector does not belong to either side by default.

Example:

`wordplay for definition`

Store:

- left part: wordplay-side material
- connector: `for`
- right part: definition-side material

The connector has its own role. It is not swallowed into the definition or the wordplay.

### 8. Punctuation Is Preserved

Question marks are stored as their own tokens.

Example:

`Tarzan?`

- token: `Tarzan`
- token: `?`

The `?` can then act as a definition qualifier or other clue signal.

Exclamation marks are also stored as their own tokens, but their role is not assumed. They may be surface punctuation, emphasis, or a sign of a special clue mode.

Quotation marks are stored as punctuation tokens and linked to the word or phrase they enclose. Their default role is presentation/surface unless there is evidence that they affect the cryptic reading.

Example:

`for "wealth"`

- quote token linked to `wealth`
- `wealth` remains the definition text
- quotes are preserved even if they add no cryptic function

### 9. Apostrophes Are Preserved Until WFW Needs A Split

Do not strip apostrophes before deciding their role.

An apostrophe may be:

- possessive
- contraction
- letter omission
- surface punctuation

Example:

`KING'S`

If the clue says `borders of KING'S` and the intended letters are `KG`, then:

- `KING` is the selected source
- `'s` is preserved as surface possessive ending
- `'s` is excluded from the letter selection

Example:

`president's`

If the answer needs `CHAIRS`, WFW may split:

- `president` -> `CHAIR`
- `'s` -> `S`

If the apostrophe is only surface grammar, it remains attached at WFW level.

### 10. Pieces Stay Low-Level

Do not merge separate mechanisms into one blended piece.

Pieces should remain as low-level as possible.

For a simple charade:

`F` + `RAIL` = `FRAIL`

Store:

- piece: `F`
- piece: `RAIL`
- assembler places them next to each other

Do not create a new merged `FRAIL` piece unless a later clue operation uses `FRAIL` as input.

This keeps colour coding clean.

### 11. Pieces Link To Answer Letter Tokens

Every solved piece fills one or more answer letter tokens.

Example:

`Salesman and consumer in Winchester?`

Answer: `REPEATER`

- `Salesman` -> `REP`
- `REP` fills answer tokens 1, 2, 3
- `consumer` -> `EATER`
- `EATER` fills answer tokens 4, 5, 6, 7, 8

The clue-to-answer relationship is usually block-to-block, but the answer block still contains numbered letter tokens.

### 12. Letter-Level Links Are Also Required

The model must support direct letter-to-letter links.

This is needed for:

- initials
- final letters
- outer letters
- inner letters
- middle letters
- alternate letters
- deletions
- additions
- abbreviations
- containers
- hidden answers

Example:

`daughter` -> `D`

Store:

- clue source: `daughter`
- produced letter: `D`
- answer letter token: position 1 in the answer

### 13. Mechanisms Are Relationships Between Pieces

A mechanism records how pieces relate. It does not always need to create a new merged piece.

Example:

`GEN` adopting `OLD`

Store:

- piece: `GEN`
- piece: `OLD`
- mechanism: container
- controller: `adopting`
- relationship: `OLD` is inserted into `GEN`
- answer placement map:
  - `GEN` -> positions 1, 5, 6
  - `OLD` -> positions 2, 3, 4

This makes containers easy to display.

### 14. Answer Letter Tokens Make Assembly Easy

Example:

Answer: `GOLDEN RETRIEVER`

Letter positions, ignoring the space:

- 1 `G`
- 2 `O`
- 3 `L`
- 4 `D`
- 5 `E`
- 6 `N`
- 7 `R`
- 8 `E`
- 9 `T`
- 10 `R`
- 11 `I`
- 12 `E`
- 13 `V`
- 14 `E`
- 15 `R`

Then:

- `GEN` -> 1, 5, 6
- `OLD` -> 2, 3, 4

Spaces and hyphens are preserved for display, but answer letter numbering counts letters.

### 15. Anagrams Use Fodder Pools And Mappings

For an anagram, preserve the mapping from fodder letter tokens to answer letter tokens.

Example:

Fodder: `TRAPS`

Answer: `PARTS`

Fodder positions:

- 1 `T`
- 2 `R`
- 3 `A`
- 4 `P`
- 5 `S`

Answer positions:

- 1 `P`
- 2 `A`
- 3 `R`
- 4 `T`
- 5 `S`

Mapping:

- answer 1 `P` came from fodder 4
- answer 2 `A` came from fodder 3
- answer 3 `R` came from fodder 2
- answer 4 `T` came from fodder 1
- answer 5 `S` came from fodder 5

Permutation: `4,3,2,1,5`

The anagram controller token, such as `mixing`, controls the rearrangement.

### 16. Fodder Pools Can Have Several Sources

An anagram fodder pool may be built from more than one source.

Example:

- source 1: `TRAPS`
- source 2: `E` from another clue word

Store:

- each source
- each source letter token
- the combined fodder pool
- the controller token
- the final mapping into answer tokens

### 17. Substitutions Create Working Blocks

A synonym or other substitution creates a new working block that later mechanisms can operate on.

Example:

`dog losing power`

Step 1:

- clue block: `dog`
- substitution relationship: synonym
- produced working block: `POODLE`

Step 2:

- clue block: `power`
- substitution relationship: abbreviation
- produced working block: `P`

Step 3:

- controller: `losing`
- mechanism: deletion
- starting block: `POODLE`
- removed block: `P`
- result block: `OODLE`

So in the WFW working layer:

`dog losing power`

becomes:

`POODLE losing P`

Then deletion operates on the produced blocks.

This is essential for compound wordplay without black-box functions.

### 18. No Compound Black-Box Functions

Do not store a single opaque function that hides several steps.

For:

`dog losing power`

Do not store only:

- `dog losing power` -> `OODLE`

Store:

- `dog` -> `POODLE`
- `power` -> `P`
- `losing` -> deletion
- before: `POODLE`
- removed: `P`
- after: `OODLE`

Store the ingredients and the steps, not just the finished dish.

### 19. Assembly And Verification Are Separate

The assembler places pieces into answer tokens.

The verifier later checks whether the placed pieces reconcile with the answer.

Do not make the assembler responsible for verification.

Do not make the verifier merely repeat the solver's reasoning.

Verifier details remain downstream and are not settled in this note.

### 20. Grammar Triage Comes Before Mechanical Solving

Cryptic definitions, &lit clues, semi-&lit clues, punning clues, and non-mechanical clues should be identified before mechanical solving.

The mechanical solver should not be allowed to output `cryptic_definition` as a fallback.

Cryptic definition is a triage classification, not an escape hatch for failed parsing.

Mechanical solving should either produce a normal mechanical parse or fail honestly.

## GOLDEN RETRIEVER Example

Clue:

`Go back upset across island after information on adopting elderly dog (6,9)`

Answer:

`GOLDEN RETRIEVER`

Working pieces:

- `information` -> `GEN`
- `elderly` -> `OLD`
- `adopting` controls container
- `GEN` occupies answer positions 1, 5, 6
- `OLD` occupies answer positions 2, 3, 4
- together they display as `GOLDEN`

Second answer section:

- `Go back` -> `REVERT`
- `island` -> `I`
- `on` -> `RE`
- `upset` controls anagram
- fodder pool: `REVERT + I + RE`
- result fills answer positions 7-15 as `RETRIEVER`

Definition:

- `dog` defines the whole answer

Relationship/ordering:

- `after` helps order the two answer sections
- `across` remains to be classified or marked unresolved; it must not vanish

## REPEATER Example

Clue:

`Salesman and consumer in Winchester?`

Answer:

`REPEATER`

Working links:

- `Salesman` -> `REP`
- `consumer` -> `EATER`
- `REP` fills answer tokens 1-3
- `EATER` fills answer tokens 4-8
- `Winchester` provides definition/context
- `?` is stored as its own token and likely qualifies the definition/context

This shows block-to-block linking.

## Important Design Shift

The parse does not just produce prose.

The parse produces durable relationships:

- clue tokens
- answer tokens
- working blocks
- substitutions
- mechanisms
- controllers
- before/after states
- answer placements

Once these relationships exist, display becomes easy.

Colour-coding comes from the saved relationships.

## Still To Settle

The next session should continue with exact rules for:

- deletions
- selections: first, last, middle, inner, outer, alternate letters
- reversals
- hidden clues
- homophones
- ordering words: after, before, behind, following
- connectors: for, with, and, by, in, about
- nested containers
- final clue state model
- conceptual DB record types

## Current Working Conclusion

The durable record must preserve the lowest useful level of the solve.

Every relationship should be classified.

Every transformation should be stored.

Every produced piece should link to answer tokens.

No rich parse should ever be flattened into empty pieces and a bare operation label.

## Continuation Notes

These notes continue the same design discussion and should be treated as part of
the preservation contract.

### Deletions

A deletion operates on working blocks, not necessarily on the surface clue word.

The settled example is:

- `dog` produces the working block `POODLE`
- `power` produces the working block `P`
- `losing` controls deletion
- deletion operates on `POODLE losing P`
- the result is `OODLE`

Do not treat this as `DOG` minus `P`.

The deletion record must preserve the operation controller, the starting working
block, the removed working block, and the result. The removed material must not
vanish. It remains part of the working-out even though it normally does not fill
answer tokens.

### Operations And Locators

Some clue instructions have two linked parts:

- an operation instruction
- a location identifier

For example, in a phrase such as `loses heart of ...`, `loses` is the operation
instruction and `heart of` is the location identifier.

These parts must be preserved separately, but they must also remain chained
together. The locator identifies the material; the operation uses that identified
material.

A selection/location instruction may therefore either produce answer material
directly, or feed another operation such as deletion, insertion, reversal, or
fodder construction.

### Reversals

Reversal indicators require special care because their scope may not be clear
from adjacency.

If a reversal indicator appears before material, it must not be greedily attached
to the first following piece. It may control one piece, several following pieces,
or a larger assembled block.

The indicator should remain available at clue level until the relevant working
pieces exist. Once the pieces are available, the answer placement can resolve
the true scope.

Only then should the WFW record bind the indicator to the resolved reversed
block or sequence.

### Anagrams

Anagram indicators have the same later-binding issue.

The fodder pool may be assembled from several produced pieces, selected letters,
abbreviations, synonyms, or other working blocks. The indicator must not be
attached too early to a neighbouring word just because it is nearby.

The anagram record should preserve the controller, all fodder sources, the
combined fodder pool, and the mapping from fodder letters to answer letters.

### Grammar Triage And Wordplay Signature

The grammar signature and the wordplay signature must come from the same words.

Grammar triage should help identify which words live together before the solver
explodes the clue into possible wordplay roles. If wordplay wants to split a
grammar unit, that split must be explicit and justified as a transformation.

The role of the pre-operations is to narrow the operating space of the solver,
not to create one-off code fixes.

### Connector-Like Words

Words such as `on`, `for`, `with`, `by`, `in`, and `about` must not be accepted
or rejected by word-list membership alone.

Their role must be licensed by grammar, clue direction, complement structure,
and answer fit.

The GOLDEN RETRIEVER example exposed an important correction. The earlier parse
treated `on` as `RE`, but the grammar suggests that `information on` is a source
phrase producing `GEN`. Treating `on` as a loose connector or separate `RE`
leaves it as a dangling preposition without a following object.

This should not become a hard-coded fix for `on`. In another clue, especially a
down clue, `on` may be a genuine positional instruction. The design must handle
the nuance rather than patching the particular word.

### Hidden Clues

Whole hidden clues are recognised by pre-operation because they are mechanically
faithful.

The pre-op looks for a contiguous bidirectional clue string matching the answer.
At that point the solver already knows it has a hidden clue and already has the
definition.

The remaining WFW work is to identify the full fodder span and the indicator,
then preserve the exact character mapping from clue text to answer letters.

### Homophones

A homophone table hit is a suspect, not a solve.

A possible homophone indicator plus a table-backed answer/source pair creates a
candidate that must still be proven by grammar, source phrase, answer fit, and
fairness.

This matters because homophone data can be noisy or playful. A pair such as
`move on` and `moo von` may be phonetically suggestive, but it still needs a fair
cryptic path before it can become a confirmed WFW relationship.

The WFW record should preserve the indicator, answer-side spelling, source-side
spelling or phrase, and the phonetic relationship.

### Spoonerisms

Spoonerisms are another suspect-generating mechanism.

The practical solver approach is answer-first: split the answer into candidate
word windows, switch the initial letters or sounds of the two words, and try to
parse that switched version from the clue rather than parsing the actual answer
directly.

Because Spoonerisms can be silly or self-indulgent, the generated switched phrase
is only a suspect until the clue grammar, indicator, and answer fit confirm it.

The WFW record should preserve the answer window, the switched candidate phrase,
the Spoonerism indicator, and the clue material that parses the switched phrase.

### Double Definitions

Double definitions are preprocessed.

The pre-op looks for two definition windows, possibly split by a connector, that
both independently define the same answer.

WFW does not need to manufacture wordplay pieces for a confirmed double
definition. It only needs to account for each definition separately, with both
definition windows defining the whole answer.

Any connector between the two definitions is preserved separately.

### Corpus Stress Test

After the design document is complete, it must be challenged against a large clue
corpus.

The goal of that work is not to patch individual code failures. The goal is to
try to break the design: ambiguous connectors, late-binding reversal and anagram
scope, noisy homophones, Spoonerisms, hidden clues, double definitions, and
compound operations should all be tested against the preservation rules.

If the corpus exposes failures, the design should be strengthened before code is
changed.
