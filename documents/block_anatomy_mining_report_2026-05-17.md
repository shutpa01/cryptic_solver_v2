# Block Anatomy Mining Report

Date: 2026-05-17

## Purpose

This note captures the first pass at mining the existing human explanation corpus
for clue anatomy and block anatomy.

The working hypothesis is that the human explanations are the Rosetta stone:
they already contain the relationships between clue words, phrase blocks,
operations, indicators, and answer pieces. Earlier work mined this into word
roles, but stopped short of fully promoting phrase/block anatomy.

The aim is not to change solver code yet. The aim is to understand what the
corpus can teach us before designing the next Grammar Triage layer.

## Existing Assets

The local corpus contains:

- `data/times_explanations.db`
- `data/word_roles.db`
- parsed explanation JSONL files
- scripts in `cryptic_taxonomy/analysis/` that parse explanations, map pieces to
  clue words, build word-role rows, and mine signatures

The most important confirmed counts are:

- `times_explanations.db` has 72,545 clue rows
- 72,459 rows have explanations
- 57,175 rows have both a definition and an explanation
- `word_roles.db` has 13,925 distinct mapped clues
- `word_roles.db` has 65,472 word-role rows

The `word_roles.db` build stats show:

- 29,584 verified parsed explanations
- 12,887 successfully mapped clues
- 20,896 unverified parses
- 10,048 mapping failures
- 1,038 mapped hidden clues
- 1,834 skipped double definitions
- 870 skipped homophones

## What The Existing GT Already Does

The active production GT is `signature_solver/grammar_triage.py`.

It already does useful mechanical triage:

- strips candidate definitions before solving
- tags wordplay words with POS if spaCy is available
- looks up POS signature candidates from `data/grammar_catalog.json`
- verifies candidate role sequences against the known answer
- tries structural tests for anagrams, charades, containers, reversals,
  container-charades, anagram-charades, and container-with-deletion
- gates positional extraction on the presence of a licensing indicator
- handles definition-by-example markers such as `maybe`, `perhaps`, and `say`
- blocks raw-letter use when a word is DBE-marked
- tries limited two- and three-word phrase lookups

This is valuable, but it is not yet the richer clue-anatomy layer we need.

Current GT mostly asks:

What role can this word play?

The new design needs to ask first:

What blocks exist here, and what kind of block is each one?

## First Mining Pass

Using `word_roles.db`, I compressed consecutive word-role rows back into
answer-producing blocks.

The first pass found:

- 25,934 answer-producing blocks
- 2,181 multiword fodder blocks
- 1,067 multiword phrase-source blocks after excluding anagram and hidden fodder

The 2,181 multiword blocks break down as:

- 1,114 anagram fodder blocks
- 969 synonym-source blocks
- 98 abbreviation-source blocks

The 1,067 non-anagram/non-hidden phrase-source blocks are especially important.
They are the clue anatomy signal that current word-level GT does not fully use.

Most of those phrase-source blocks occur in:

- container clues
- charades
- container-charades
- reversal-charades
- reversals

## Repeated Phrase Sources

The corpus already knows many phrase blocks that should be treated as units.

Examples:

- `that is` -> `IE`
- `the French` -> `LE` or `LA`
- `in charge` -> `IC`
- `hospital department` -> `ENT`
- `one of five` -> `QUIN`
- `Welsh girl` -> `SIAN`
- `a French` -> `UN`
- `Greek character` -> `CHI` or `PHI`
- `old man` -> `PA`
- `post office` -> `PO`
- `no longer` -> `EX`
- `before noon` -> `AM`
- `getting on` -> `AGE` or `OLD`
- `Italian team` -> `INTER`
- `Scottish island` -> `TIREE` or `IONA`
- `put off` -> `DETER`
- `stiffly formal` -> `PRIM`
- `young woman` -> `LASS` or `MISS`
- `social worker` -> `ANT`
- `very large` -> `OS`
- `for one` -> `EG`

These are not just synonym lookups. They are phrase blocks. Splitting them too
early creates noise and can destroy the parse.

## SPITTOON Example

The exact user clue was:

`Receptacle former PM brought in without delay`

Answer:

`SPITTOON`

The desired anatomy is:

- definition: `Receptacle`
- wordplay space: `former PM brought in without delay`
- source block: `former PM` -> `PITT`
- instruction block: `brought in`
- source block: `without delay` -> `SOON`
- container relation: insert `PITT` into `SOON`
- answer assembly: `S` + `PITT` + `OON` -> `SPITTOON`

The Times explanation corpus contains a close variant:

`Old hacker's target presently, PM admitted`

Answer:

`SPITTOON`

Human explanation:

`SOON (presently) with PITT (PM) contained [admitted]`

This confirms the anatomy:

- `presently` -> `SOON`
- `PM` -> `PITT`
- `admitted` -> container instruction

The important design lesson is that only one strong phrase synonym may be needed
to unlock the whole signature. Once `PITT` or `SOON` is known, the answer
geometry strongly suggests a container.

## ARTHUR Example

The user clue was:

`Legendary ruler in craft endlessly upset`

Answer:

`ARTHUR`

The desired anatomy is:

- definition: `Legendary ruler`
- wordplay space: `in craft endlessly upset`
- leading `in` is orphaned as a positional/container instruction
- therefore `in` can be a true connector in this clue
- source block: `craft` -> `ART`
- source block: `upset` -> `HURT`
- modifier/locator: `endlessly` removes the final letter from `HURT`
- result: `HUR`
- assembly: `ART` + `HUR` -> `ARTHUR`

This is a useful contrast with the GOLDEN RETRIEVER `on` problem.

In GOLDEN RETRIEVER, `information on` behaves as a phrase and should not be
split casually.

In ARTHUR, the wordplay-space starts with `in`. Because it has no preceding
material and no valid positional structure, it can be treated as a connector.

So connector-like words must be classified by grammar and position, not by a
static list alone.

## Connector-Like Words

The corpus confirms that small connector-like words are genuinely ambiguous.

Examples from `word_roles.db`:

- `in` appears as `LNK`, `CON_I`, `HID_F`, `SYN_F`, `ABR_F`, and `ANA_F`
- `on` appears as `LNK`, `SYN_F`, `ABR_F`, `ANA_F`, and `HID_F`
- `for` appears as `LNK`, `SYN_F`, `HID_F`, `ABR_F`, `ANA_F`, and `DEL_I`
- `with` appears as `LNK`, `ABR_F`, `SYN_F`, `HID_F`, and `ANA_F`
- `by` appears as `LNK`, `SYN_F`, `HID_F`, `ABR_F`, and `ANA_F`
- `after` appears as `LNK`, `ANA_I`, `SYN_F`, `ABR_F`, `HID_F`, and `ANA_F`
- `before` appears as `LNK`, `SYN_F`, and `ABR_F`
- `without` appears as `SYN_F`, `LNK`, `ABR_F`, `DEL_I`, and `HID_F`

This supports the design rule from the WFW preservation notes:

Do not hard-code a word such as `on`, `in`, or `for` as a single role. Treat it
as a suspect whose role must be licensed by grammar, clue direction, complement
structure, phrase membership, and answer fit.

## What The Old Mining Missed

The old pipeline successfully built word-role rows, but it flattened several
things we now care about.

It did not fully preserve:

- phrase blocks as first-class objects
- operation blocks such as `brought in`
- modifier-plus-source chains such as `endlessly upset`
- orphaned connector diagnostics
- operation scope candidates
- relationship chains between blocks
- why a phrase should not be split

This means `word_roles.db` is useful evidence, but not the final anatomy model.

It is especially noisy around container and container-charade clues. Sometimes a
container operation is compressed into one large `SYN_F` span. This is a clue
that the human explanation contains the information we need, but the old mapper
did not preserve it at the right level.

## Block Anatomy Needed

The next GT design should produce block suspects before word-role suspects.

A block suspect may be:

- active source material
- operation instruction
- location identifier
- modifier
- ordering relation
- connector candidate
- definition qualifier
- unresolved residue

For every block, GT should preserve:

- the original tokens
- the phrase span
- the block type suspect
- whether the block is active answer-producing material
- whether it requires a complement
- whether it modifies a neighbouring block
- whether it may control a later-binding scope
- whether grammar objects to a proposed role

The solver should receive a constrained operating field, not a flat list of
words.

## Proposed Next Mining Step

The next mining step should go back to the human explanations directly, not only
to `word_roles.db`.

For each verified explanation, extract:

- source phrase shown by the human explanation
- produced value
- operation indicator phrase
- operation type
- source phrase span in clue text
- whether the source is single-word or multiword
- whether the operation is explicit in the explanation
- whether current `word_roles.db` flattened or mis-grouped it

Then compare those extracted blocks to the clue grammar.

The aim is to build a block-level corpus:

- clue text
- answer
- definition
- wordplay window
- block spans
- block type
- produced values
- operation links
- unresolved or discarded words

This block-level corpus is the missing bridge between human explanations, GT,
the mechanical solver, and WFW preservation.

## Working Conclusion

The corpus does contain the Rosetta-stone signal.

The previous mining reached word roles and signatures, but not full block
anatomy. The design phase should now focus on recovering block anatomy from the
human explanations at scale.

Only after that should solver code be changed.
