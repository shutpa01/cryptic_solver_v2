# Supervised Block Anatomy Corpus

Date: 2026-05-17

This is an R&D artefact for GT V2. It uses sanitised explanations as
supervision to mark answer-producing clue blocks, then records the
unmapped residue and the surface grammar over the same wordplay span.

Rows scanned: 10000
Records written: 9219
Complete block mappings: 2895
Complete mappings with compact residue: 1215
Grammar tags available: no

## Operation Mix

- `charade`: 5561
- `abr`: 694
- `synonym`: 553
- `anagram`: 526
- `container`: 489
- `hidden`: 286
- `reversal_charade`: 281
- `container_charade`: 222
- `homophone`: 148
- `reversal`: 137
- `anagram_charade`: 127
- `del`: 111

## Common Compact Residues

- `in`: 33
- `on`: 28
- `for`: 24
- `with`: 21
- `and`: 21
- `is`: 18
- `of`: 14
- `after`: 12
- `to`: 9
- `before`: 6
- `by`: 4
- `supporting`: 4
- `chasing`: 4
- `and before`: 3
- `with and`: 3
- `backing`: 3
- `supported by`: 3
- `led by`: 3

## Anchor Example

`Report on small issue of litter engulfing green` (RECOUNT)

Definition: `Report on`
Source blocks: `small issue of litter` -> `RUNT`, `green` -> `ECO`
Residue: `engulfing`
Boundary labels: `S S S S R S`

This is the target shape for the science project: explanation-supervised source spans, compact residue, and a boundary pattern that can later be tested against grammar.

## Worked Review Seeds

- `I’ll verify accounts of car stuck on hill` (AUDITOR)
  Definition: `I’ll verify accounts`
  Source blocks: `car` -> `AUDI`, `hill` -> `TOR`
  Residue: `of stuck on`
  Labels: `R S R R S`
  Grammar tags: `? ? ? ? ?`
- `English poet martyred expertly by a Pole` (SOUTHWELL)
  Definition: `English poet martyred`
  Source blocks: `a Pole` -> `SOUTH`, `expertly` -> `WELL`
  Residue: `by`
  Labels: `S R S S`
  Grammar tags: `? ? ? ?`
- `One inclined to drive before parking` (RAMP)
  Definition: `One inclined`
  Source blocks: `drive` -> `RAM`, `parking` -> `P`
  Residue: `to before`
  Labels: `R S R S`
  Grammar tags: `? ? ? ?`
- `Way to protect grant for ball game ?` (ROULETTE)
  Definition: `ball game`
  Source blocks: `grant` -> `LET`, `Way` -> `ROUTE`
  Residue: `to protect for`
  Labels: `S R R S R`
  Grammar tags: `? ? ? ? ?`
- `Again do something naughty , on and off, before death` (REOFFEND)
  Definition: `Again do something naughty`
  Source blocks: `on` -> `RE`, `off` -> `OFF`, `death` -> `END`
  Residue: `and before`
  Labels: `S R S R S`
  Grammar tags: `? ? ? ? ?`
- `Frenchman eats fruit in apartment` (PIEDATERRE)
  Definition: `apartment`
  Source blocks: `fruit` -> `DATE`, `Frenchman` -> `PIERRE`
  Residue: `eats in`
  Labels: `S R S R`
  Grammar tags: `? ? ? ?`
- `Knight ’s destiny attached to weapon` (LANCELOT)
  Definition: `Knight`
  Source blocks: `weapon` -> `LANCE`, `destiny` -> `LOT`
  Residue: `s attached to`
  Labels: `R S R R S`
  Grammar tags: `? ? ? ? ?`
- `Able to swallow large fly` (FLIT)
  Definition: `fly`
  Source blocks: `large` -> `L`, `Able` -> `FIT`
  Residue: `to swallow`
  Labels: `S R R S`
  Grammar tags: `? ? ? ?`
- `Area recently allowing in vehicle with choice of fare` (ALACARTE)
  Definition: `with choice of fare`
  Source blocks: `vehicle` -> `CAR`, `Area` -> `A`, `recently` -> `LATE`
  Residue: `allowing in`
  Labels: `S S R R S`
  Grammar tags: `? ? ? ? ?`
- `Remarkable stable state that is difficult to achieve` (TALLORDER)
  Definition: `difficult to achieve`
  Source blocks: `Remarkable` -> `TALL`, `stable state` -> `ORDER`
  Residue: `that is`
  Labels: `S S S R R`
  Grammar tags: `? ? ? ? ?`
- `Ancient ship on lake moving slowly` (LARGO)
  Definition: `moving slowly`
  Source blocks: `lake` -> `L`, `Ancient ship` -> `ARGO`
  Residue: `on`
  Labels: `S S R S`
  Grammar tags: `? ? ? ?`
- `Understand the difference record margin had in cuts` (DISCRIMINATE)
  Definition: `Understand the difference`
  Source blocks: `record` -> `DISC`, `margin` -> `RIM`, `had` -> `ATE`
  Residue: `in cuts`
  Labels: `S S S R R`
  Grammar tags: `? ? ? ? ?`
- `Menace over a hospital means …` (MOOLAH)
  Definition: `means`
  Source blocks: `a` -> `A`, `hospital` -> `H`
  Residue: `Menace over`
  Labels: `R R S S`
  Grammar tags: `? ? ? ?`
- `Pilot quietly longing for somewhere to trade, unofficially` (FLYPITCH)
  Definition: `somewhere to trade, unofficially`
  Source blocks: `Pilot` -> `FLY`, `quietly` -> `P`, `longing` -> `ITCH`
  Residue: `for`
  Labels: `S S S R`
  Grammar tags: `? ? ? ?`
- `One standing in line with old copper and male` (LOCUM)
  Definition: `One standing in`
  Source blocks: `line` -> `L`, `old` -> `O`, `copper` -> `CU`, `male` -> `M`
  Residue: `with and`
  Labels: `S R S S R S`
  Grammar tags: `? ? ? ? ? ?`
- `Long way back after run — that’s very chaotic` (FARRAGO)
  Definition: `that’s very chaotic`
  Source blocks: `Long` -> `FAR`, `run` -> `R`, `way back` -> `AGO`
  Residue: `after`
  Labels: `S S S R S`
  Grammar tags: `? ? ? ? ?`
- `Accessory in relationship cut` (TIECLIP)
  Definition: `Accessory`
  Source blocks: `relationship` -> `TIE`, `cut` -> `CLIP`
  Residue: `in`
  Labels: `R S S`
  Grammar tags: `? ? ?`
- `Justification OK for uprising` (GROUNDSWELL)
  Definition: `uprising`
  Source blocks: `Justification` -> `GROUNDS`, `OK` -> `WELL`
  Residue: `for`
  Labels: `S S R`
  Grammar tags: `? ? ?`
- `Complains tree is cut` (BEEFSTEAK)
  Definition: `cut`
  Source blocks: `Complains` -> `BEEFS`, `tree` -> `TEAK`
  Residue: `is`
  Labels: `S S R`
  Grammar tags: `? ? ?`
- `What was the Left? Rather mean and a bit cowardly?` (YELLOWISH)
  Definition: `a bit cowardly?`
  Source blocks: `the` -> `YE`, `Left` -> `L`, `Rather mean` -> `LOWISH`
  Residue: `What was and`
  Labels: `R R S S S S R`
  Grammar tags: `? ? ? ? ? ? ?`

## Research Reading

The useful records are not final parses. They are supervised examples of
where answer-producing blocks appear in the clue, what residue remains,
and what the ordinary grammar looked like before cryptic interpretation.

The next research question is whether the grammar tags and dependency
shape can predict the same SOURCE/RESIDUE boundaries without seeing the
human explanation.

This run does not include POS/dependency tags because spaCy is not
installed in the active environment. The SOURCE/RESIDUE supervision is
still useful; the grammar layer can be added once the language model is
available.
