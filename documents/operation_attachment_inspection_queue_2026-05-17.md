# Operation Attachment Inspection Queue

Date: 2026-05-17

This is a compact queue for manual design inspection.
The labels are weak and provisional. The task is to decide whether each residue block has the right anatomical relationship to the source material.


## OPERATOR_SCOPE

Available examples: `281`

- `Unravels leading nets in a mess with son` (DISENTANGLES)
  Operation: `anagram`
  Residue: `in a mess with`
  Sources: `leading nets`; `son`
  Relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Parser source heads: `in->nets:prep`
- `Some bloke choosing feature in a chamber?` (ECHO)
  Operation: `hidden`
  Residue: `feature in a chamber`
  Sources: `bloke choosing`
  Relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Needs split: `feature->UNCLASSIFIED_RESIDUE, in->OPERATOR_SCOPE, a->CONNECTOR_OR_SURFACE, chamber->UNCLASSIFIED_RESIDUE`
  Parser source heads: `feature->choosing:dobj`, `in->choosing:prep`
- `Rough path, by the sound of it?` (COARSE)
  Operation: `homophone`
  Residue: `by the sound of it`
  Sources: `path`
  Relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Needs split: `by->CONNECTOR_OR_SURFACE, the->CONNECTOR_OR_SURFACE, sound->OPERATOR_SCOPE, of->CONNECTOR_OR_SURFACE, it->UNCLASSIFIED_RESIDUE`
  Parser source heads: `by->path:prep`
- `Mark in text reflected some spiritual musings` (UMLAUT)
  Operation: `hidden_reversed`
  Residue: `in text reflected some`
  Sources: `spiritual musings`
  Relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Needs split: `in->OPERATOR_SCOPE, text->UNCLASSIFIED_RESIDUE, reflected->UNCLASSIFIED_RESIDUE, some->OPERATOR_SCOPE`
  Parser source heads: `some->musings:det`

## LOCATOR_SCOPE

Available examples: `22`

- `Truncated holiday time? On top of this, caught ailment` (CHOLERA)
  Operation: `charade`
  Residue: `On top of this`
  Sources: `holiday time`; `caught`
  Relationship: `LOCATES_WITHIN`
  Scope status: `weakly_scoped_from_operation`
  Needs split: `On->ORDER_OR_POSITION, top->LOCATOR_SCOPE, of->CONNECTOR_OR_SURFACE, this->UNCLASSIFIED_RESIDUE`
  Parser source heads: `On->time:prep`
- `Shellfish roast arranged on top of white fish` (OSTRACOD)
  Operation: `anagram`
  Residue: `arranged on top of white`
  Sources: `roast`; `fish`
  Relationship: `LOCATES_WITHIN`
  Scope status: `weakly_scoped_from_operation`
  Needs split: `arranged->UNCLASSIFIED_RESIDUE, on->CONNECTOR_OR_SURFACE, top->LOCATOR_SCOPE, of->CONNECTOR_OR_SURFACE, white->UNCLASSIFIED_RESIDUE`
  Parser source heads: `white->fish:amod`
- `Famous, on edge and cautious, covering head` (LEGENDARY)
  Operation: `deletion`
  Residue: `covering head`
  Sources: `edge`; `cautious`
  Relationship: `LOCATES_WITHIN`
  Scope status: `weakly_scoped_from_operation`
  Needs split: `covering->UNCLASSIFIED_RESIDUE, head->LOCATOR_SCOPE`
  Parser source heads: `head->edge:conj`
- `Part of church in steeple's painted towards the top` (APSE)
  Operation: `hidden_reversed`
  Residue: `the top`
  Sources: `steeple's painted towards`
  Relationship: `LOCATES_WITHIN`
  Scope status: `weakly_scoped_from_operation`
  Parser source heads: `none`

## DIRECTION_OR_ORIENTATION

Available examples: `112`

- `One from Lisbon turned up on board debonair e-bike` (Iberian)
  Operation: `hidden_reversed`
  Residue: `turned up on board`
  Sources: `debonair e-bike`
  Relationship: `AWAITING_SCOPE`
  Scope status: `answer_aware_scope_needed`
  Needs split: `turned->UNCLASSIFIED_RESIDUE, up->DIRECTION_OR_ORIENTATION, on->CONNECTOR_OR_SURFACE, board->UNCLASSIFIED_RESIDUE`
  Parser source heads: `board->debonair:compound`
- `Hands from the East in trade` (SWAP)
  Operation: `reversal`
  Residue: `from the East in`
  Sources: `Hands`
  Relationship: `AWAITING_SCOPE`
  Scope status: `answer_aware_scope_needed`
  Parser source heads: `from->Hands:prep`

## ORDER_OR_POSITION

Available examples: `27`

- `Son and family, following attack, hide from nanny perhaps` (GOATSKIN)
  Operation: `charade`
  Residue: `following attack hide from`
  Sources: `Son`; `family`; `nanny`
  Relationship: `ORDERS`
  Scope status: `answer_aware_scope_needed`
  Needs split: `following->ORDER_OR_POSITION, attack->UNCLASSIFIED_RESIDUE, hide->UNCLASSIFIED_RESIDUE, from->CONNECTOR_OR_SURFACE`
  Parser source heads: `following->Son:prep`

## CONTAINER_RELATION

Available examples: `100`

- `Very similar pair also, we hear, round a family in France and Germany` (TWOOFAKIND)
  Operation: `charade`
  Residue: `in France and Germany`
  Sources: `pair also we hear round a family`
  Relationship: `CONTAINS`
  Scope status: `weakly_scoped_from_operation`
  Needs split: `in->CONTAINER_RELATION, France->UNCLASSIFIED_RESIDUE, and->CONNECTOR_OR_SURFACE, Germany->UNCLASSIFIED_RESIDUE`
  Parser source heads: `in->family:prep`
- `Resolve without proposal to keep single` (EXPLAIN)
  Operation: `container`
  Residue: `to keep`
  Sources: `without proposal`; `single`
  Relationship: `CONTAINS`
  Scope status: `weakly_scoped_from_operation`
  Parser source heads: `keep->proposal:acl`

## CONNECTOR_OR_SURFACE

Available examples: `118`

- `Smear limited publicity for a book` (BLUR)
  Operation: `deletion`
  Residue: `for a`
  Sources: `publicity`; `book`
  Relationship: `SURFACE_CONNECTS`
  Scope status: `surface_only_until_grammar_check`
  Parser source heads: `for->publicity:prep`, `a->book:det`
- `Settled hill by a lake shore` (LITTORAL)
  Operation: `charade`
  Residue: `by a`
  Sources: `Settled hill`; `lake`
  Relationship: `SURFACE_CONNECTS`
  Scope status: `surface_only_until_grammar_check`
  Parser source heads: `by->hill:prep`
- `With sun gone, sail by unstable African country` (LIBYA)
  Operation: `deletion+anagram`
  Residue: `With`
  Sources: `sun gone sail by unstable`
  Relationship: `SURFACE_CONNECTS`
  Scope status: `surface_only_until_grammar_check`
  Parser source heads: `With->gone:mark`
- `To an extent, part is terrifying for performer` (ARTISTE)
  Operation: `hidden`
  Residue: `for`
  Sources: `part is terrifying`
  Relationship: `SURFACE_CONNECTS`
  Scope status: `surface_only_until_grammar_check`
  Parser source heads: `for->terrifying:prep`

## UNCLASSIFIED_RESIDUE

Available examples: `480`

- `Types about to run around scenic site` (BEAUTYSPOT)
  Operation: `anagram`
  Residue: `to run around scenic`
  Sources: `Types about`
  Relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Parser source heads: `run->about:xcomp`
- `Turning around, strips – you do it in bed` (SLEEP)
  Operation: `reversal`
  Residue: `you do it in`
  Sources: `strips`
  Relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Parser source heads: `do->strips:relcl`
- `Singer, in key or right to be sacked?` (ALTO)
  Operation: `charade`
  Residue: `right to be sacked`
  Sources: `key or`
  Relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Parser source heads: `right->key:conj`
- `Lairy paparazzo from the south frames old documents` (PAPYRI)
  Operation: `hidden_reversed`
  Residue: `from the south frames`
  Sources: `Lairy paparazzo`
  Relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Parser source heads: `from->paparazzo:prep`

## Inspection Questions

- Is the residue label right, or is it actually source-internal/surface material?
- If it is an operation, what exact source span does it operate on?
- If it is a locator, what operation is it chained to?
- If it is a connector, does the surface grammar permit that reading?
- If parser evidence disagrees with cryptic evidence, which should be preserved as primary?
