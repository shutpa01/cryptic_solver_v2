# Block Graph Quality

Date: 2026-05-17

This report quantifies the first-pass block graph candidate artifact.
It is intended to show where the graph representation is informative and where it remains too weak.

Graphs analysed: `865`

## Status

`candidate`=865

## Node Kinds

`SOURCE_BLOCK`=1084, `ASSEMBLY_BLOCK`=865, `DEF_BLOCK`=865, `SCOPE_BLOCK`=820, `OP_BLOCK`=290, `CONNECTOR_BLOCK`=199, `RELATION_BLOCK`=103, `POSITION_BLOCK`=32, `LOCATOR_BLOCK`=22, `DEF_MODIFIER_BLOCK`=5

## Edge Kinds

`CONTRIBUTES_TO`=1084, `DEFINES`=846, `UNRESOLVED`=814, `OPERATES_ON`=293, `SURFACE_CONNECTS`=235, `CONTAINS`=157, `AWAITING_SCOPE`=113, `ORDERS`=52, `LOCATES_WITHIN`=34, `MODIFIES_DEFINITION`=5

## Scope Statuses

`known`=1930, `unresolved`=814, `weakly_scoped_from_operation`=484, `surface_only_until_grammar_check`=235, `answer_aware_scope_needed`=165, `definition_modifier_candidate`=5

## Common Graph Sizes

`4 nodes/3 edges`=388, `5 nodes/4 edges`=181, `6 nodes/5 edges`=77, `5 nodes/5 edges`=61, `6 nodes/6 edges`=49, `7 nodes/6 edges`=27, `6 nodes/7 edges`=11, `7 nodes/8 edges`=9, `4 nodes/2 edges`=9, `8 nodes/7 edges`=8, `7 nodes/7 edges`=8, `9 nodes/8 edges`=7

## Unresolved Node Kinds By Operation

- `anagram`: `SCOPE_BLOCK`=194
- `charade`: `SCOPE_BLOCK`=180
- `deletion`: `SCOPE_BLOCK`=4
- `deletion+anagram`: `SCOPE_BLOCK`=11
- `hidden`: `SCOPE_BLOCK`=123
- `hidden_reversed`: `SCOPE_BLOCK`=89
- `homophone`: `SCOPE_BLOCK`=15
- `reversal`: `SCOPE_BLOCK`=91

## Unresolved Examples

- `Follow lad after repairing emergency barrier` (FLOODWALL)
  `after` -> `SCOPE_BLOCK` / `ambiguous`
- `Sister, say, in a bolder violet when dressed` (BLOODRELATIVE)
  `when` -> `SCOPE_BLOCK` / `ambiguous`
- `Man is unaccompanied when cycling` (ELON)
  `when` -> `SCOPE_BLOCK` / `ambiguous`
- `Slug, say, with time in gaps door created` (GASTROPOD)
  `created` -> `SCOPE_BLOCK` / `ambiguous`
- `Routes in event of emergency across a deep rupture` (ESCAPEROADS)
  `in event of emergency` -> `SCOPE_BLOCK` / `ambiguous`
  `rupture` -> `SCOPE_BLOCK` / `ambiguous`
- `Bet Rod's upset those owing money` (DEBTORS)
  `those` -> `SCOPE_BLOCK` / `ambiguous`
- `Nautical ode when reviewed having instructive value?` (EDUCATIONAL)
  `when` -> `SCOPE_BLOCK` / `ambiguous`
  `having` -> `SCOPE_BLOCK` / `ambiguous`
- `Volatile sort heard in good or hard bargaining` (HORSETRADING)
  `Volatile` -> `SCOPE_BLOCK` / `ambiguous`
  `or` -> `SCOPE_BLOCK` / `ambiguous`
- `Bad smell began to spread around marsh plant` (BOGBEAN)
  `Bad` -> `SCOPE_BLOCK` / `ambiguous`
  `spread` -> `SCOPE_BLOCK` / `ambiguous`
- `A term's e.g. formulated for one wagering money` (GAMESTER)
  `one` -> `SCOPE_BLOCK` / `ambiguous`
  `wagering` -> `SCOPE_BLOCK` / `ambiguous`

## Connector Examples

- `Sister, say, in a bolder violet when dressed` (BLOODRELATIVE)
  `in` -> connector candidate
- `Man is unaccompanied when cycling` (ELON)
  `is` -> connector candidate
- `Hammer, say, broadcast of stunt with nimble turn` (BLUNTINSTRUMENT)
  `of` -> connector candidate
  `with` -> connector candidate
- `Have a second job looming somehow by empty hut` (MOONLIGHT)
  `by` -> connector candidate
- `Bad smell began to spread around marsh plant` (BOGBEAN)
  `to` -> connector candidate
- `A term's e.g. formulated for one wagering money` (GAMESTER)
  `for` -> connector candidate
- `Thing to ride wrongly takes B-road` (SKATEBOARD)
  `to` -> connector candidate
- `See glum eccentric getting plants grown as a crop` (LEGUMES)
  `as` -> connector candidate
  `a` -> connector candidate
- `Bond hated other criminal in a state of anxiety?` (HOTANDBOTHERED)
  `in` -> connector candidate
  `a` -> connector candidate
  `of` -> connector candidate
- `Renown at intervals for a very long period` (EON)
  `at` -> connector candidate

## Missing Definition Examples

- `Supermarket among restaurants stricken in flood, maybe` (NATURALDISASTER)
  Residue nodes: `among; stricken; in; flood; maybe`
- `It carries passengers in suburb, usually` (BUS)
  Residue nodes: `It carries`
- `Some bloke choosing feature in a chamber?` (ECHO)
  Residue nodes: `Some; feature; in; a; chamber`
- `Fifty plus acres, perhaps?` (LAND)
  Residue nodes: `acres perhaps`
- `Son and family, following attack, hide from nanny perhaps` (GOATSKIN)
  Residue nodes: `and; following; attack; hide; from; perhaps`
- `No serious accident: police initially called` (PRANG)
  Residue nodes: `No serious accident`
- `Horse's leg too short` (PINTO)
  Residue nodes: `Horse's`
- `Some credit to editor is echoed` (DITTOED)
  Residue nodes: `Some`
- `Visual representation in viewing Raphael` (GRAPH)
  Residue nodes: `Visual; representation; in`
- `Target identified among America's troublemakers?` (Castro)
  Residue nodes: `Target; identified; among`
- `Nuts I'd set free?` (NUDIST)
  Residue nodes: `set free`
- `Daft racist lost at sea` (cast adrift)
  Residue nodes: `lost at sea`
- `Those at Greer plays?` (theatre-goers)
  Residue nodes: `plays`
- `Elastic in a closet in bits` (SECTIONAL)
  Residue nodes: `Elastic; in bits`
- `Muddies watercolour finally with paints` (ROILS)
  Residue nodes: `finally; with`
- `Dance music's introduction is right for rattles` (DISCOMFITS)
  Residue nodes: `right for`
- `Design it? I only should cover lighting` (IGNITION)
  Residue nodes: `should; cover; lighting`
- `Musical character article fully embraces` (CLEF)
  Residue nodes: `Musical character; embraces`
- `Poet's quickly captivated by drama, inevitably` (AMAIN)
  Residue nodes: `Poet's quickly captivated by drama`

## Reading

The graph artifact is useful if it makes unresolved anatomy explicit.
The key next design pressure is reducing ambiguous `SCOPE_BLOCK` nodes by deriving operation-specific attachment from structured explanations.
Connector candidates need grammar objections, especially where a preposition is orphaned or source-internal.
