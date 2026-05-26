# Block Graph Candidates

Date: 2026-05-17

This is the first graph-shaped GT V2 anatomy artifact.
It is generated from weak operation attachment labels and is intended for inspection, not solving.

Graphs: `865`

## First Examples

- `Follow lad after repairing emergency barrier` (FLOODWALL)
  Operation: `anagram`
  Nodes: `5`; edges: `4`
  Residue node: `after` -> `SCOPE_BLOCK` / `ambiguous`
  Residue node: `repairing` -> `OP_BLOCK` / `inferred`
- `Sister, say, in a bolder violet when dressed` (BLOODRELATIVE)
  Operation: `anagram`
  Nodes: `6`; edges: `5`
  Residue node: `in` -> `CONNECTOR_BLOCK` / `inferred`
  Residue node: `when` -> `SCOPE_BLOCK` / `ambiguous`
  Residue node: `dressed` -> `OP_BLOCK` / `inferred`
- `Man is unaccompanied when cycling` (ELON)
  Operation: `anagram`
  Nodes: `6`; edges: `5`
  Residue node: `is` -> `CONNECTOR_BLOCK` / `inferred`
  Residue node: `when` -> `SCOPE_BLOCK` / `ambiguous`
  Residue node: `cycling` -> `OP_BLOCK` / `inferred`
- `Slug, say, with time in gaps door created` (GASTROPOD)
  Operation: `charade`
  Nodes: `8`; edges: `8`
  Residue node: `say` -> `DEF_MODIFIER_BLOCK` / `inferred`
  Residue node: `with` -> `RELATION_BLOCK` / `inferred`
  Residue node: `in` -> `RELATION_BLOCK` / `inferred`
- `Routes in event of emergency across a deep rupture` (ESCAPEROADS)
  Operation: `anagram`
  Nodes: `5`; edges: `4`
  Residue node: `in event of emergency` -> `SCOPE_BLOCK` / `ambiguous`
  Residue node: `rupture` -> `SCOPE_BLOCK` / `ambiguous`
- `Bet Rod's upset those owing money` (DEBTORS)
  Operation: `anagram`
  Nodes: `5`; edges: `4`
  Residue node: `upset` -> `OP_BLOCK` / `inferred`
  Residue node: `those` -> `SCOPE_BLOCK` / `ambiguous`
- `Nautical ode when reviewed having instructive value?` (EDUCATIONAL)
  Operation: `anagram`
  Nodes: `6`; edges: `5`
  Residue node: `when` -> `SCOPE_BLOCK` / `ambiguous`
  Residue node: `reviewed` -> `OP_BLOCK` / `inferred`
  Residue node: `having` -> `SCOPE_BLOCK` / `ambiguous`
- `Hammer, say, broadcast of stunt with nimble turn` (BLUNTINSTRUMENT)
  Operation: `anagram`
  Nodes: `8`; edges: `8`
  Residue node: `say` -> `DEF_MODIFIER_BLOCK` / `inferred`
  Residue node: `broadcast` -> `OP_BLOCK` / `inferred`
  Residue node: `of` -> `CONNECTOR_BLOCK` / `inferred`

## Reading

The useful question is now whether these graph candidates preserve the right distinctions.
In particular, inspect whether `OP_BLOCK`, `LOCATOR_BLOCK`, and `CONNECTOR_BLOCK` nodes should be split, merged, or retyped.
Answer verification should come later; this artifact is still anatomy research.
