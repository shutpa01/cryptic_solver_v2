# Grammar Feature Scaffold

Date: 2026-05-17

This scaffold stores supervised token boundaries from structured explanations, with optional spaCy features when available.
It is an R&D artifact for block anatomy, not a solver change.

Records: `931`
spaCy model available: `yes`

## Operation Mix

`charade`=256, `hidden`=203, `anagram`=191, `hidden_reversed`=88, `reversal`=81, `homophone`=80, `container`=21, `deletion+anagram`=7, `deletion`=4

## Source Span Lengths

`1` words=944, `2` words=296, `3` words=175, `4` words=31, `5` words=10

## Residue Run Lengths

`1` words=631, `2` words=329, `3` words=130, `4` words=38, `5` words=14, `6` words=2

## Common Label Contexts

- `DEF`: `<START><DEF>SOURCE`=204, `DEF<DEF><END>`=164, `<START><DEF>RESIDUE`=164, `RESIDUE<DEF><END>`=163, `DEF<DEF>DEF`=149, `<START><DEF>DEF`=132, `RESIDUE<DEF>DEF`=98, `SOURCE<DEF><END>`=85, `DEF<DEF>SOURCE`=67, `SOURCE<DEF>DEF`=66
- `SOURCE`: `SOURCE<SOURCE>RESIDUE`=426, `SOURCE<SOURCE>SOURCE`=407, `RESIDUE<SOURCE>SOURCE`=321, `<START><SOURCE>SOURCE`=206, `SOURCE<SOURCE><END>`=170, `DEF<SOURCE>SOURCE`=151, `RESIDUE<SOURCE>RESIDUE`=144, `DEF<SOURCE>RESIDUE`=120, `SOURCE<SOURCE>DEF`=82, `RESIDUE<SOURCE><END>`=81
- `RESIDUE`: `SOURCE<RESIDUE>RESIDUE`=355, `RESIDUE<RESIDUE>RESIDUE`=256, `RESIDUE<RESIDUE>SOURCE`=244, `RESIDUE<RESIDUE><END>`=146, `SOURCE<RESIDUE>DEF`=138, `DEF<RESIDUE>SOURCE`=137, `SOURCE<RESIDUE>SOURCE`=133, `RESIDUE<RESIDUE>DEF`=123, `SOURCE<RESIDUE><END>`=122, `<START><RESIDUE>SOURCE`=101

## Glue Word Contexts

- `to:SOURCE<RESIDUE>RESIDUE` = 39
- `of:RESIDUE<RESIDUE>SOURCE` = 38
- `in:SOURCE<SOURCE>SOURCE` = 33
- `in:DEF<RESIDUE>SOURCE` = 31
- `the:RESIDUE<RESIDUE>RESIDUE` = 30
- `a:SOURCE<SOURCE>SOURCE` = 27
- `in:RESIDUE<RESIDUE>SOURCE` = 27
- `with:SOURCE<RESIDUE>SOURCE` = 21
- `of:DEF<DEF>DEF` = 20
- `by:RESIDUE<RESIDUE>SOURCE` = 20
- `in:SOURCE<RESIDUE>RESIDUE` = 19
- `for:RESIDUE<RESIDUE>DEF` = 16
- `of:DEF<RESIDUE>SOURCE` = 16
- `on:SOURCE<RESIDUE>RESIDUE` = 15
- `of:SOURCE<SOURCE>SOURCE` = 14
- `for:SOURCE<RESIDUE>RESIDUE` = 14
- `from:DEF<RESIDUE>SOURCE` = 14
- `in:SOURCE<RESIDUE>DEF` = 13
- `to:SOURCE<SOURCE>SOURCE` = 12
- `in:RESIDUE<RESIDUE>RESIDUE` = 12

## Operationish Word Contexts

- `some:<START><RESIDUE>SOURCE` = 20
- `up:RESIDUE<RESIDUE><END>` = 19
- `some:DEF<RESIDUE>SOURCE` = 9
- `up:RESIDUE<RESIDUE>RESIDUE` = 8
- `up:RESIDUE<RESIDUE>SOURCE` = 8
- `reportedly:<START><RESIDUE>SOURCE` = 8
- `about:SOURCE<RESIDUE>SOURCE` = 7
- `picked:SOURCE<RESIDUE>RESIDUE` = 7
- `back:RESIDUE<RESIDUE><END>` = 7
- `broadcast:SOURCE<RESIDUE><END>` = 6
- `some:RESIDUE<RESIDUE>SOURCE` = 6
- `radio:RESIDUE<RESIDUE><END>` = 6
- `back:RESIDUE<RESIDUE>RESIDUE` = 5
- `up:SOURCE<RESIDUE><END>` = 5
- `picked:<START><RESIDUE>RESIDUE` = 4
- `about:SOURCE<RESIDUE>DEF` = 4
- `reported:SOURCE<RESIDUE><END>` = 4
- `excited:SOURCE<RESIDUE>RESIDUE` = 3
- `up:SOURCE<RESIDUE>DEF` = 3
- `up:RESIDUE<RESIDUE>DEF` = 3

## Reading

This creates the shared surface needed for the next experiment.
The immediate question is whether grammar features, when added, improve source-boundary and residue-attachment decisions beyond the residue-only baseline.
