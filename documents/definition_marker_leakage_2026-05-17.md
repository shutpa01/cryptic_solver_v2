# Definition Marker Leakage

Date: 2026-05-17

This report finds definition-by-example markers that appear as `RESIDUE` rather than travelling with the definition.
These are important because they can poison residue attachment labels.

Records with possible leakage: `18`
Leaked markers near a definition boundary: `5`

## Marker Counts

`say`=7, `perhaps`=7, `like`=3, `maybe`=1

## By Operation

`charade`=7, `homophone`=7, `reversal`=2, `anagram`=1, `hidden`=1

## Examples

- `Slug, say, with time in gaps door created` (GASTROPOD)
  Operation: `charade`
  Labels: `D R R S R S S R`
  Marker: `say` with context `DEF<RESIDUE>RESIDUE`
- `Hammer, say, broadcast of stunt with nimble turn` (BLUNTINSTRUMENT)
  Operation: `anagram`
  Labels: `D R R R S R S S`
  Marker: `say` with context `DEF<RESIDUE>RESIDUE`
- `Supermarket among restaurants stricken in flood, maybe` (NATURALDISASTER)
  Operation: `charade`
  Labels: `S R S R R R R`
  Marker: `maybe` with context `RESIDUE<RESIDUE><END>`
- `Deal, say, with receptacle close to one honeysuckle` (WOODBINE)
  Operation: `charade`
  Labels: `S R R S S S S D`
  Marker: `say` with context `SOURCE<RESIDUE>RESIDUE`
- `Port, perhaps, now in electric fences` (WINE)
  Operation: `hidden`
  Labels: `D R S S S S`
  Marker: `perhaps` with context `DEF<RESIDUE>SOURCE`
- `Fifty plus acres, perhaps?` (LAND)
  Operation: `charade`
  Labels: `S S R R`
  Marker: `perhaps` with context `RESIDUE<RESIDUE><END>`
- `Son and family, following attack, hide from nanny perhaps` (GOATSKIN)
  Operation: `charade`
  Labels: `S R S R R R R S R`
  Marker: `perhaps` with context `SOURCE<RESIDUE><END>`
- `Fair chance, did you say?` (FETE)
  Operation: `homophone`
  Labels: `D S R R R`
  Marker: `say` with context `RESIDUE<RESIDUE><END>`
- `Sounds like free money for Federer!` (FRANC)
  Operation: `homophone`
  Labels: `R R S D D D`
  Marker: `like` with context `RESIDUE<RESIDUE>SOURCE`
- `Gong that sounds like fiddle?` (MEDAL)
  Operation: `homophone`
  Labels: `D R R R S`
  Marker: `like` with context `RESIDUE<RESIDUE>SOURCE`
- `Seat hurled, did you say?` (THRONE)
  Operation: `homophone`
  Labels: `D S R R R`
  Marker: `say` with context `RESIDUE<RESIDUE><END>`
- `Table in church to change did you say?` (ALTAR)
  Operation: `homophone`
  Labels: `D D D R S R R R`
  Marker: `say` with context `RESIDUE<RESIDUE><END>`
- `Country Siberian, did you say?` (CHILE)
  Operation: `homophone`
  Labels: `D S R R R`
  Marker: `say` with context `RESIDUE<RESIDUE><END>`
- `Sounds like standard shot at billiards?` (Cannon)
  Operation: `homophone`
  Labels: `R R S D D D`
  Marker: `like` with context `RESIDUE<RESIDUE>SOURCE`
- `Perhaps cobbler's upset getting nervous` (STRESSED)
  Operation: `reversal`
  Labels: `R S R R D`
  Marker: `Perhaps` with context `<START><RESIDUE>SOURCE`
- `Perhaps fool's turned anxious` (STRESSED)
  Operation: `reversal`
  Labels: `R S R D`
  Marker: `Perhaps` with context `<START><RESIDUE>SOURCE`
- `Man perhaps lives large on vacation` (ISLE)
  Operation: `charade`
  Labels: `D R S S R R`
  Marker: `perhaps` with context `DEF<RESIDUE>SOURCE`
- `Cold bird, perhaps chicken` (CRAVEN)
  Operation: `charade`
  Labels: `S S R D`
  Marker: `perhaps` with context `SOURCE<RESIDUE>DEF`

## Reading

DBE marker leakage should be treated as a definition-span quality issue before it is treated as wordplay residue.
A future block graph should allow a `DBE_MARKER` or `DEF_MODIFIER` node attached to `DEF_BLOCK`.
This is another example of why block anatomy must be preserved rather than flattened into source/residue labels.
