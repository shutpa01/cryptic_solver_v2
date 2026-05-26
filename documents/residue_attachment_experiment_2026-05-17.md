# Residue Attachment Experiment

Date: 2026-05-17

This experiment asks whether grammar can attach contiguous `RESIDUE` runs to the relevant `SOURCE` material.

It was run over the enriched grammar scaffold, using the same deterministic 684/247 train/test split as the previous experiments.

## Important Caveat

The first attempted target was deliberately simple:

- `right`: residue immediately before a source block
- `left`: residue immediately after a source block
- `between`: residue between two source blocks
- `loose`: residue not adjacent to source

This target is not yet the true cryptic attachment relation.

It is only a surface adjacency proxy.

Because of that, a nearest-source baseline scores 291/291 by construction. That is not a meaningful success. It simply proves the target was defined by adjacency.

## Parser-Head Attachment

The more interesting test was parser-head attachment:

- if a residue token's syntactic head is a `SOURCE` token to the left, predict `left`
- if its head is a `SOURCE` token to the right, predict `right`
- if it does not head to source material, treat it as `loose`

Against the adjacency proxy:

- residue runs tested: `291`
- parser-head attachment hits: `87/291` (`30%`)

By gold adjacency class:

- `right`: 46 hits, 64 loose
- `left`: 41 hits, 99 loose
- `between`: 18 non-loose directional hits, 23 loose

By operation:

- `anagram`: `27/57`
- `charade`: `12/70`
- `hidden`: `28/59`
- `hidden_reversed`: `8/33`
- `homophone`: `8/35`
- `reversal`: `4/31`
- `container`: `0/4`
- `deletion`: `0/1`
- `deletion+anagram`: `0/1`

## Examples

- `Routes in event of emergency across a deep rupture` (`ESCAPEROADS`)
  Residue: `in event of emergency`
  Gold adjacency: `right`
  Parser-head prediction: `loose`
  Heads: `in->Routes`, `event->in`, `of->event`, `emergency->of`

- `Engineer dares to skirt eastern body of water` (`REDSEA`)
  Residue: `to skirt`
  Gold adjacency: `between`
  Parser-head prediction: `left`
  Heads: `to->skirt`, `skirt->dares`

- `Painter got excited about shift in financial projection` (`OPERATINGBUDGET`)
  Residue: `excited about`
  Gold adjacency: `between`
  Parser-head prediction: `left`
  Heads: `excited->got`, `about->excited`

- `File apt to be shredded in squalid cinema` (`FLEAPIT`)
  Residue: `to be shredded in squalid`
  Gold adjacency: `left`
  Parser-head prediction: `loose`
  Heads: `to->shredded`, `be->shredded`, `shredded->apt`, `in->shredded`, `squalid->cinema`

- `Roger today put in a mess is critical` (`DEROGATORY`)
  Residue: `put in a mess is`
  Gold adjacency: `left`
  Parser-head prediction: `loose`
  Heads: `put->put`, `in->put`, `a->mess`, `mess->put`, `is->put`

## Reading

This is a useful negative result.

Grammar heads are not a direct substitute for cryptic attachment.

The parser often describes the surface sentence correctly, while cryptic attachment follows the wordplay operation.

That does not make grammar useless. It means grammar should be used as evidence for possible attachment, not as the definition of attachment.

The next target must be derived from structured operation anatomy, not from surface adjacency alone.

For example:

- anagram indicator attaches to all anagram fodder in scope
- deletion operation attaches source and deletion target as a chained block
- hidden indicator attaches to the hidden fodder span
- reversal direction attaches to a reversible source span, with answer-aware direction confirmation
- charade connectors may indicate order without consuming source material

This is closer to the design principle already established: the active operation and its locator may preserve separately, but they must still remain chained as a block.

