# Definition Gap Analysis

Date: 2026-05-17

This report inspects graph candidates where the structured explanation did not preserve an explicit definition span.
These are not automatically bad records. They may be hidden clues, all-in-one surfaces, cryptic definitions, or explanation gaps.

Definition gaps inspected: `19`

## Hypotheses

- `hidden clue with definition omitted by structured explanation`: 5
- `single mapped source with surrounding residue possibly acting as definition`: 4
- `definition-by-example marker present but definition anchor missing`: 3
- `hidden clue with possible all-in-one or cryptic definition surface`: 3
- `question-mark surface may be carrying definition or cryptic definition force`: 2
- `letter-selection source may have swallowed locator or definition surface`: 2

## Examples

- `Supermarket among restaurants stricken in flood, maybe` (NATURALDISASTER)
  Operation: `charade`
  Hypothesis: `definition-by-example marker present but definition anchor missing`
  Source blocks: `Supermarket (synonym; value=ALDI); restaurants (anagram_fodder; value=RESTAURANTS)`
  Residue blocks: `among; stricken; in; flood; maybe`
- `It carries passengers in suburb, usually` (BUS)
  Operation: `hidden`
  Hypothesis: `hidden clue with definition omitted by structured explanation`
  Source blocks: `passengers in suburb usually (hidden; value=PASSENGERSINSUBURBUSUALLY)`
  Residue blocks: `It carries`
- `Some bloke choosing feature in a chamber?` (ECHO)
  Operation: `hidden`
  Hypothesis: `hidden clue with possible all-in-one or cryptic definition surface`
  Source blocks: `bloke choosing (hidden; value=ECHO)`
  Residue blocks: `Some; feature; in; a; chamber`
- `Fifty plus acres, perhaps?` (LAND)
  Operation: `charade`
  Hypothesis: `definition-by-example marker present but definition anchor missing`
  Source blocks: `Fifty plus (abbreviation,synonym; value=LAND)`
  Residue blocks: `acres perhaps`
- `Son and family, following attack, hide from nanny perhaps` (GOATSKIN)
  Operation: `charade`
  Hypothesis: `definition-by-example marker present but definition anchor missing`
  Source blocks: `Son (abbreviation; value=S); family (synonym; value=KIN); nanny (synonym; value=GOAT)`
  Residue blocks: `and; following; attack; hide; from; perhaps`
- `No serious accident: police initially called` (PRANG)
  Operation: `charade`
  Hypothesis: `single mapped source with surrounding residue possibly acting as definition`
  Source blocks: `police initially called (first_letter,synonym; value=PRANG)`
  Residue blocks: `No serious accident`
- `Horse's leg too short` (PINTO)
  Operation: `charade`
  Hypothesis: `single mapped source with surrounding residue possibly acting as definition`
  Source blocks: `leg too short (deletion,synonym; value=PINTO)`
  Residue blocks: `Horse's`
- `Some credit to editor is echoed` (DITTOED)
  Operation: `hidden`
  Hypothesis: `hidden clue with definition omitted by structured explanation`
  Source blocks: `credit to editor is echoed (hidden; value=CREDITTOEDITORISECHOED)`
  Residue blocks: `Some`
- `Visual representation in viewing Raphael` (GRAPH)
  Operation: `hidden`
  Hypothesis: `hidden clue with definition omitted by structured explanation`
  Source blocks: `viewing Raphael (hidden; value=GRAPH)`
  Residue blocks: `Visual; representation; in`
- `Target identified among America's troublemakers?` (Castro)
  Operation: `hidden`
  Hypothesis: `hidden clue with possible all-in-one or cryptic definition surface`
  Source blocks: `America's troublemakers (hidden; value=CASTRO)`
  Residue blocks: `Target; identified; among`
- `Nuts I'd set free?` (NUDIST)
  Operation: `anagram`
  Hypothesis: `question-mark surface may be carrying definition or cryptic definition force`
  Source blocks: `Nuts I'd (anagram_fodder; value=NUTSID)`
  Residue blocks: `set free`
- `Daft racist lost at sea` (cast adrift)
  Operation: `anagram`
  Hypothesis: `single mapped source with surrounding residue possibly acting as definition`
  Source blocks: `Daft racist (anagram_fodder; value=DAFTRACIST)`
  Residue blocks: `lost at sea`
- `Those at Greer plays?` (theatre-goers)
  Operation: `anagram`
  Hypothesis: `question-mark surface may be carrying definition or cryptic definition force`
  Source blocks: `Those at Greer (anagram_fodder; value=THOSEATGREER)`
  Residue blocks: `plays`
- `Elastic in a closet in bits` (SECTIONAL)
  Operation: `anagram`
  Hypothesis: `single mapped source with surrounding residue possibly acting as definition`
  Source blocks: `in a closet (anagram_fodder; value=INACLOSET)`
  Residue blocks: `Elastic; in bits`
- `Muddies watercolour finally with paints` (ROILS)
  Operation: `charade`
  Hypothesis: `letter-selection source may have swallowed locator or definition surface`
  Source blocks: `Muddies watercolour (last_letter; value=R); paints (synonym; value=OILS)`
  Residue blocks: `finally; with`
- `Dance music's introduction is right for rattles` (DISCOMFITS)
  Operation: `charade`
  Hypothesis: `letter-selection source may have swallowed locator or definition surface`
  Source blocks: `Dance music's introduction is (last_letter; value=S); rattles (synonym; value=DISCOMFIT)`
  Residue blocks: `right for`
- `Design it? I only should cover lighting` (IGNITION)
  Operation: `hidden`
  Hypothesis: `hidden clue with possible all-in-one or cryptic definition surface`
  Source blocks: `Design it I only (hidden; value=IGNITION)`
  Residue blocks: `should; cover; lighting`
- `Musical character article fully embraces` (CLEF)
  Operation: `hidden`
  Hypothesis: `hidden clue with definition omitted by structured explanation`
  Source blocks: `article fully (hidden; value=CLEF)`
  Residue blocks: `Musical character; embraces`
- `Poet's quickly captivated by drama, inevitably` (AMAIN)
  Operation: `hidden`
  Hypothesis: `hidden clue with definition omitted by structured explanation`
  Source blocks: `inevitably (hidden; value=AMAIN)`
  Residue blocks: `Poet's quickly captivated by drama`

## Design Consequence

A missing definition must remain explicit in the graph.
The right model is not `no definition`; it is `definition candidate not yet located`.
That allows later grammar and answer-mechanics checks to propose whole-surface, residue-surface, or implicit definition candidates without corrupting the wordplay blocks.
