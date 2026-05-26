# Residue Operation Signals

Date: 2026-05-17

This report mines the clean boundary slice after SOURCE and DEF tokens have been labelled.
It asks which RESIDUE phrases correlate with structured assembly operations.

Records analysed: 931

## Residue Length By Operation

- `anagram`: 2 tokens=71, 1 tokens=60, 3 tokens=35, 4 tokens=14, 5 tokens=9, 6 tokens=1, 0 tokens=1
- `charade`: 2 tokens=83, 1 tokens=66, 0 tokens=48, 3 tokens=34, 4 tokens=13, 5 tokens=8, 6 tokens=3, 7 tokens=1
- `container`: 1 tokens=11, 2 tokens=8, 4 tokens=1, 3 tokens=1
- `deletion`: 4 tokens=2, 3 tokens=1, 2 tokens=1
- `deletion+anagram`: 4 tokens=2, 3 tokens=2, 1 tokens=1, 2 tokens=1, 6 tokens=1
- `hidden`: 1 tokens=89, 2 tokens=59, 3 tokens=29, 0 tokens=17, 4 tokens=6, 5 tokens=2, 6 tokens=1
- `hidden_reversed`: 2 tokens=40, 3 tokens=31, 4 tokens=11, 5 tokens=4, 1 tokens=2
- `homophone`: 2 tokens=30, 1 tokens=24, 3 tokens=19, 4 tokens=6, 5 tokens=1
- `reversal`: 3 tokens=24, 2 tokens=20, 4 tokens=18, 1 tokens=12, 6 tokens=4, 5 tokens=3

## Common Residue Runs By Operation

- `anagram`: `of`=7, `in`=6, `with`=5, `to change`=3, `for`=3, `out`=3, `devised for`=2, `moving`=2, `silly`=2, `about`=2, `excited`=2, `off`=2
- `charade`: `with`=21, `in`=20, `and`=11, `for`=7, `by`=6, `is`=6, `about`=5, `from`=5, `on`=5, `over`=4, `given`=3, `to`=3
- `container`: `in`=4, `about`=2, `of`=2, `interrupting`=2, `eating`=2, `on`=1, `through part of`=1, `round`=1, `swallowing`=1, `crossing`=1, `seen in`=1, `with`=1
- `deletion`: `limited`=1, `for a`=1, `losing top`=1, `covering for leader in`=1, `on`=1, `and`=1, `covering head`=1
- `deletion+anagram`: `with`=1, `reg`=1, `excited to catch`=1, `trouble losing pounds is`=1, `mostly playing an`=1, `almost`=1, `play for`=1, `corrupt in`=1, `lisa`=1, `going around capital towards`=1, `west`=1
- `hidden`: `some`=19, `in`=12, `for`=5, `section of`=4, `to an extent`=4, `from`=4, `partly`=3, `houses`=3, `of`=3, `about`=3, `a little`=3, `to some extent`=3
- `hidden_reversed`: `some`=10, `in`=9, `up`=4, `from`=3, `brought back`=2, `rejected`=2, `returned in`=2, `raised`=2, `back in`=2, `turned up on board`=1, `to an extent after retiring`=1, `revolting and`=1
- `homophone`: `on the radio`=9, `reportedly`=9, `picked up`=9, `broadcast`=6, `we hear`=5, `did you say`=4, `reported`=4, `of`=3, `heard`=3, `for the audience`=3, `in`=2, `and`=2
- `reversal`: `in`=5, `in reverse`=2, `from the east`=2, `up`=2, `rising`=2, `on the rise`=2, `put up`=2, `enigmatic sea`=2, `on the way back`=2, `perhaps`=2, `from the east in`=1, `returns`=1

## Distinctive Residue Signals

- `reportedly` -> `homophone` (9/9, 100%)
- `on the radio` -> `homophone` (7/7, 100%)
- `we hear` -> `homophone` (5/5, 100%)
- `broadcast` -> `homophone` (5/5, 100%)
- `section of` -> `hidden` (4/4, 100%)
- `to some extent` -> `hidden` (3/3, 100%)
- `to change` -> `anagram` (3/3, 100%)
- `partly` -> `hidden` (3/3, 100%)
- `over` -> `charade` (3/3, 100%)
- `nurses` -> `hidden` (3/3, 100%)
- `for` -> `charade` (3/3, 100%)
- `did you say` -> `homophone` (3/3, 100%)
- `covers` -> `hidden` (3/3, 100%)
- `some` -> `hidden` (12/13, 92%)
- `with` -> `charade` (10/11, 91%)
- `picked up` -> `homophone` (7/8, 88%)
- `about` -> `hidden` (3/4, 75%)
- `of` -> `hidden` (2/3, 67%)
- `new` -> `anagram` (2/3, 67%)
- `around` -> `anagram` (2/3, 67%)

## Boundary Signatures By Operation

- `anagram`: `S S R R D D`=7, `D S S R`=5, `S S R R R D D`=4, `S S R R D`=4, `S S R R D D D`=4, `D R R S S`=4, `R S S R D`=4, `S S R D`=4, `D D R S S S R R`=3, `S S S R R D D`=3
- `charade`: `D R S R S`=8, `D S S`=7, `D S R S`=6, `D S S S`=5, `S S D`=5, `S S D D D`=5, `D S R R S`=4, `D S S R S`=4, `S S R R S D`=4, `S S R D`=4
- `container`: `D S R S`=4, `D R S R S S`=2, `S R S D`=2, `S S R S D`=2, `D S R S R S`=1, `D R R R S R S S`=1, `S S R S R D`=1, `D D S R S S`=1, `D S R R S`=1, `D R S S R S`=1
- `deletion`: `D R S R R S`=1, `D S R R S`=1, `S R R R R S D`=1, `D R S R S R R`=1
- `deletion+anagram`: `R S S S S S D D`=1, `R S R R R S D`=1, `S S S R R R R D`=1, `S S R R R D`=1, `R S S S R R D`=1, `D S R R S S`=1, `D S S R S R R R R S R`=1
- `hidden`: `D S S R`=14, `D R R S S`=12, `D S S S R`=11, `S S R D`=10, `S S R R D`=8, `S S S R D`=8, `D R R S S S`=7, `D R S S`=6, `D R S S S`=5, `D S S S S`=5
- `hidden_reversed`: `S S R R D`=7, `D R S S S R`=6, `D R R S S`=4, `D R S S R`=4, `D R R R S S`=4, `S S R R D D`=3, `D R R S S S R`=3, `D R S S S R R`=3, `D D R S S S R R`=2, `S S S S R R D`=2
- `homophone`: `D S R R`=12, `D S R R R`=10, `R S D`=8, `D S R`=7, `D D S R R`=4, `R R S D`=4, `D R S R R R`=3, `R S D D`=3, `D R S R R`=2, `R R R S R D`=2
- `reversal`: `D S R R`=6, `D S R R R`=6, `D R S R R`=5, `D S R`=4, `D R R S R R`=4, `S R R R R D`=3, `D S R R R R`=3, `R S R D`=3, `R R S R R R R D`=2, `D R S R R R`=2

## Source Span Shapes By Operation

- `anagram`: `2`=56, `1+1`=37, `3`=36, `1+1+1`=27, `1`=18, `4`=8, `1+1+1+1`=3, `1+2`=2, `1+1+2`=1, `1+1+1+3`=1
- `charade`: `1+1`=109, `1+1+1`=39, `2+1`=20, `1+2`=18, `3+1`=9, `1+3`=6, `1+2+1`=6, `1+1+2`=6, `2+1+1`=5, `1+1+1+1`=5
- `container`: `1+1+1`=10, `1+1`=8, `1+2`=3
- `deletion`: `1+1`=4
- `deletion+anagram`: `1+1+1`=2, `1+1+1+1+1`=1, `1+1`=1, `2`=1, `3`=1, `1+1+1+1`=1
- `hidden`: `2`=102, `3`=70, `4`=16, `5`=9, `1`=5, `2+2`=1
- `hidden_reversed`: `2`=49, `3`=37, `4`=2
- `homophone`: `1`=77, `2`=3
- `reversal`: `1`=76, `1+1+1`=1, `1+1+1+1`=1, `3`=1, `2`=1, `1+1+1+1+3`=1

## Worked Signal Examples

`some`
- `Some jackdaw noticed in start of day` (DAWN)
  Operation: `hidden`
  Residue: `some in start of`
  Labels: `R S S R R R D`
  Assembly: `{'_definition': 'day', 'op': 'hidden', 'words': 'jackdaw noticed'}`
- `Some road engineers in Middle East port` (ADEN)
  Operation: `hidden`
  Residue: `some`
  Labels: `R S S S S S D`
  Assembly: `{'op': 'hidden', 'words': 'road engineers in Middle East'}`
- `Absorb some amazing estimates` (INGEST)
  Operation: `hidden`
  Residue: `some`
  Labels: `D R S S`
  Assembly: `{'op': 'hidden', 'words': 'amazing estimates'}`
`reportedly`
- `Reportedly moderate seminar` (LESSON)
  Operation: `homophone`
  Residue: `reportedly`
  Labels: `R S D`
  Assembly: `{'gives': 'LESSON', 'op': 'homophone', 'sounds_like': 'LESSON'}`
- `Reportedly remained sober` (STAID)
  Operation: `homophone`
  Residue: `reportedly`
  Labels: `R S D`
  Assembly: `{'gives': 'STAID', 'op': 'homophone', 'sounds_like': 'STAID'}`
- `Reportedly single person` (SOUL)
  Operation: `homophone`
  Residue: `reportedly`
  Labels: `R S D`
  Assembly: `{'gives': 'SOUL', 'op': 'homophone', 'sounds_like': 'SOUL'}`
`we hear`
- `Alternative therapy to stop one smoking joint, we hear, son is converted` (HYPNOSIS)
  Operation: `charade`
  Residue: `to stop one smoking we hear converted`
  Labels: `D D R R R R S R R S S R`
  Assembly: `{'op': 'charade', 'order': ['HYP', 'NOSIS']}`
- `Situation understood, we hear` (SCENE)
  Operation: `homophone`
  Residue: `we hear`
  Labels: `D S R R`
  Assembly: `{'gives': 'SCENE', 'op': 'homophone', 'sounds_like': 'SCENE'}`
- `Fixes duty, we hear` (TACKS)
  Operation: `homophone`
  Residue: `we hear`
  Labels: `D S R R`
  Assembly: `{'gives': 'TACKS', 'op': 'homophone', 'sounds_like': 'TACKS'}`
`reconstructed`
- `Oddball reconstructed Roman lab` (ABNORMAL)
  Operation: `anagram`
  Residue: `reconstructed`
  Labels: `D R S S`
  Assembly: `{'fodder': ['ROMAN', 'LAB'], 'gives': 'ABNORMAL', 'op': 'anagram'}`
`kept by`
- `Vegetable kept by Anneka least` (KALE)
  Operation: `hidden`
  Residue: `kept by`
  Labels: `D R R S S`
  Assembly: `{'_definition': 'Vegetable', 'op': 'hidden', 'words': 'Anneka least'}`
`at front of`
- `Poles at front of long grass` (SNITCH)
  Operation: `charade`
  Residue: `at front of`
  Labels: `S R R R S D`
  Assembly: `{'op': 'charade', 'order': ['SN', 'ITCH']}`
`picked up`
- `Picked up perfumes for a little cash` (CENTS)
  Operation: `anagram`
  Residue: `picked up for`
  Labels: `R R S R D D D`
  Assembly: `{'fodder': ['SCENT'], 'gives': 'CENTS', 'op': 'anagram'}`
- `This writer's picked up small packet? Bad luck` (MISFORTUNE)
  Operation: `charade`
  Residue: `picked up`
  Labels: `S S R R S S D D`
  Assembly: `{'op': 'charade', 'order': ['MI', 'S', 'FORTUNE']}`
- `Mistake picked up in anger or resentment` (ERROR)
  Operation: `hidden_reversed`
  Residue: `picked up in`
  Labels: `D R R R S S S`
  Assembly: `{'op': 'hidden_reversed', 'words': 'anger or resentment'}`

## Reading

This gives us the first non-grammar baseline: residue alone often carries strong operation evidence.
The grammar experiment should therefore not ask whether grammar solves everything by itself.
It should ask whether grammar improves block boundary discovery and operation attachment beyond this residue lexicon baseline.
