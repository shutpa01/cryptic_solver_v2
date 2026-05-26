# Clean Boundary Bucket Analysis

Date: 2026-05-17

This is the first GT V2 boundary-learning slice.
It uses only clean source/type buckets and only records where every structured piece maps back to clue text.

Training slice records: 931

## Bucket Counts

- `dailymail` / `anagram`: 80
- `dailymail` / `charade`: 80
- `telegraph` / `anagram`: 80
- `telegraph` / `charade`: 80
- `telegraph` / `hidden`: 80
- `telegraph` / `hidden_reversed`: 80
- `telegraph` / `homophone`: 80
- `telegraph` / `reversal`: 80
- `telegraph-toughie` / `anagram`: 80
- `telegraph-toughie` / `charade`: 80
- `telegraph-toughie` / `hidden`: 80
- `dailymail` / `hidden`: 51

## Common Boundary Signatures

- `D S S R`: 25
- `S S R R D`: 21
- `D R R S S`: 20
- `D S R R`: 19
- `S S R D`: 18
- `D S R R R`: 16
- `S S S R D`: 14
- `D S S S R`: 14
- `R S D`: 13
- `D S R`: 13
- `S S R R D D`: 11
- `D S R S`: 10
- `D R R S S S`: 10
- `D R S S R`: 10
- `D R S R R`: 9
- `D R S R S`: 9
- `D R S S S R`: 9
- `D R S S S`: 9
- `S S R D D`: 9
- `D R S S`: 8
- `S S S R R D`: 8
- `D S S`: 8
- `D S S R R`: 7
- `D S S S`: 7
- `R S S R D`: 7

## Boundary Signatures By Type

- `anagram`: `S S R R D D`=7, `D S S R`=6, `S S R R D`=5, `S S R R R D D`=4, `S S S R R D D`=4, `S S S R D`=4, `S S R R D D D`=4, `D R R S S`=4
- `charade`: `D S R S`=10, `D R S R S`=7, `D S S`=7, `D S S S`=5, `S S D`=5, `S S D D D`=5, `D S S R S`=4, `D R S S R S`=4
- `hidden`: `D S S R`=14, `D R R S S`=12, `D S S S R`=11, `S S R D`=10, `S S R R D`=8, `S S S R D`=8, `D R R S S S`=7, `D R S S`=6
- `hidden_reversed`: `S S R R D`=7, `D R S S S R`=5, `D R R S S`=4, `D R S S R`=4, `D R R R S S`=4, `D R R S S S R`=3, `D R S S S R R`=3, `D R R R R S S`=2
- `homophone`: `D S R R`=12, `D S R R R`=10, `R S D`=8, `D S R`=7, `D D S R R`=4, `R R S D`=4, `D R S R R R`=3, `R S D D`=3
- `reversal`: `D S R R`=6, `D S R R R`=6, `D R S R R`=5, `D S R`=4, `D R R S R R`=4, `D S R R R R`=3, `R S R D`=3, `R R S R R R R D`=2

## Common Residues

- `in`: 16
- `some`: 13
- `with`: 11
- `reportedly`: 9
- `picked up`: 8
- `on the radio`: 7
- `from`: 5
- `we hear`: 5
- `broadcast`: 5
- `section of`: 4
- `about`: 4
- `to change`: 3
- `of`: 3
- `around`: 3
- `for`: 3
- `over`: 3
- `partly`: 3
- `new`: 3
- `to some extent`: 3
- `did you say`: 3
- `covers`: 3
- `nurses`: 3
- `by`: 2
- `devised for`: 2
- `moving`: 2
- `and`: 2
- `on in`: 2
- `sandwiches`: 2
- `perhaps`: 2
- `hidden by`: 2

## Example Records

`dailymail` / `anagram`
- `Follow lad after repairing emergency barrier` (FLOODWALL)
  Definition: `emergency barrier`
  Sources: Follow -> FOLLOW, lad -> LAD
  Residue: `after repairing`
  Labels: `S S R R D D`
  Assembly: `{'fodder': ['FOLLOW', 'LAD'], 'gives': 'FLOODWALL', 'op': 'anagram'}`
- `Sister, say, in a bolder violet when dressed` (BLOODRELATIVE)
  Definition: `Sister, say`
  Sources: a -> A, bolder -> BOLDER, violet -> VIOLET
  Residue: `in when dressed`
  Labels: `D D R S S S R R`
  Assembly: `{'fodder': ['A', 'BOLDER', 'VIOLET'], 'gives': 'BLOODRELATIVE', 'op': 'anagram'}`
- `Man is unaccompanied when cycling` (ELON)
  Definition: `Man`
  Sources: unaccompanied -> LONE
  Residue: `is when cycling`
  Labels: `D R S R R`
  Assembly: `{'fodder': ['LONE'], 'gives': 'ELON', 'op': 'anagram'}`
`dailymail` / `charade`
- `Complete golf session cancelled` (ROUNDOFF)
  Definition: `Complete`
  Sources: golf session -> ROUND, cancelled -> OFF
  Residue: ``
  Labels: `D S S S`
  Assembly: `{'op': 'charade', 'order': ['ROUND', 'OFF']}`
- `Strong soldiers turning up in front of sculpture` (ROBUST)
  Definition: `Strong`
  Sources: soldiers -> RO, sculpture -> BUST
  Residue: `turning up in front of`
  Labels: `D S R R R R R S`
  Assembly: `{'op': 'charade', 'order': ['RO', 'BUST']}`
- `Eject last of worshippers by bench in church` (SPEW)
  Definition: `Eject`
  Sources: worshippers -> S, bench in church -> PEW
  Residue: `last of by`
  Labels: `D R R S R S S S`
  Assembly: `{'op': 'charade', 'order': ['S', 'PEW']}`
`dailymail` / `hidden`
- `Vegetable kept by Anneka least` (KALE)
  Definition: `Vegetable`
  Sources: Anneka least -> KALE
  Residue: `kept by`
  Labels: `D R R S S`
  Assembly: `{'_definition': 'Vegetable', 'op': 'hidden', 'words': 'Anneka least'}`
- `Walk unevenly entering slim path` (LIMP)
  Definition: `Walk`
  Sources: slim path -> LIMP
  Residue: `unevenly entering`
  Labels: `D R R S S`
  Assembly: `{'_definition': 'Walk', 'op': 'hidden', 'words': 'slim path'}`
- `Some jackdaw noticed in start of day` (DAWN)
  Definition: `day`
  Sources: jackdaw noticed -> DAWN
  Residue: `Some in start of`
  Labels: `R S S R R R D`
  Assembly: `{'_definition': 'day', 'op': 'hidden', 'words': 'jackdaw noticed'}`
`telegraph` / `anagram`
- `Oddball reconstructed Roman lab` (ABNORMAL)
  Definition: `Oddball`
  Sources: Roman -> ROMAN, lab -> LAB
  Residue: `reconstructed`
  Labels: `D R S S`
  Assembly: `{'fodder': ['ROMAN', 'LAB'], 'gives': 'ABNORMAL', 'op': 'anagram'}`
- `Thankless wretch lost garnet south of Italy` (INGRATE)
  Definition: `Thankless wretch`
  Sources: garnet -> GARNET, Italy -> I
  Residue: `lost south of`
  Labels: `D D R S R R S`
  Assembly: `{'fodder': ['GARNET', 'I'], 'gives': 'INGRATE', 'op': 'anagram'}`
- `Fuel from isle Oscar lied about` (diesel oil)
  Definition: `Fuel`
  Sources: isle -> ISLE, Oscar -> O, lied -> LIED
  Residue: `from about`
  Labels: `D R S S S R`
  Assembly: `{'fodder': ['ISLE', 'O', 'LIED'], 'gives': 'DIESELOIL', 'op': 'anagram'}`
`telegraph` / `charade`
- `Poles at front of long grass` (SNITCH)
  Definition: `grass`
  Sources: Poles -> SN, long -> ITCH
  Residue: `at front of`
  Labels: `S R R R S D`
  Assembly: `{'op': 'charade', 'order': ['SN', 'ITCH']}`
- `Old king with large brain?` (OFFAL)
  Definition: `brain`
  Sources: Old king -> OFFA, large -> L
  Residue: `with`
  Labels: `S S R S D`
  Assembly: `{'op': 'charade', 'order': ['OFFA', 'L']}`
- `Thin client periodically cancelled digital protection` (GAUNTLET)
  Definition: `digital protection`
  Sources: Thin -> GAUNT, client -> LET
  Residue: `periodically cancelled`
  Labels: `S S R R D D`
  Assembly: `{'op': 'charade', 'order': ['GAUNT', 'LET']}`
`telegraph` / `hidden`
- `Drive forward as part of grim peloton` (IMPEL)
  Definition: `Drive forward`
  Sources: grim peloton -> IMPEL
  Residue: `as part of`
  Labels: `D D R R R S S`
  Assembly: `{'op': 'hidden', 'words': 'grim peloton'}`
- `Triumphant ombudsman's hiding illusion` (PHANTOM)
  Definition: `illusion`
  Sources: Triumphant ombudsman's -> TRIUMPHANTOMBUDSMANS
  Residue: `hiding`
  Labels: `S S R D`
  Assembly: `{'op': 'hidden', 'words': "Triumphant ombudsman's"}`
- `Often chippy wraps fish` (TENCH)
  Definition: `fish`
  Sources: Often chippy wraps -> OFTENCHIPPYWRAPS
  Residue: ``
  Labels: `S S S D`
  Assembly: `{'op': 'hidden', 'words': 'Often chippy wraps'}`
`telegraph` / `hidden_reversed`
- `Internet message campaign, it's opposing houses going up` (POSTING)
  Definition: `up`
  Sources: campaign it's opposing -> POSTING
  Residue: `Internet message houses going`
  Labels: `R R S S S R R D`
  Assembly: `{'op': 'hidden_reversed', 'words': "campaign, it's opposing"}`
- `Trips over some galvanised iron` (RIDES)
  Definition: `Trips`
  Sources: galvanised iron -> RIDES
  Residue: `over some`
  Labels: `D R R S S`
  Assembly: `{'op': 'hidden_reversed', 'words': 'galvanised iron'}`
- `Wagon in Ayr rolled over` (Lorry)
  Definition: `Wagon`
  Sources: Ayr rolled -> LORRY
  Residue: `in over`
  Labels: `D R S S R`
  Assembly: `{'op': 'hidden_reversed', 'words': 'Ayr rolled'}`
`telegraph` / `homophone`
- `Interfere in award, one hears` (MEDDLE)
  Definition: `Interfere`
  Sources: award -> MEDAL
  Residue: `in one hears`
  Labels: `D R S R R`
  Assembly: `{'gives': 'MEDDLE', 'op': 'homophone', 'order': ['MEDDLE'], 'sounds_like': 'MEDAL'}`
- `Child star in auditorium` (SON)
  Definition: `Child`
  Sources: star -> SON
  Residue: `in auditorium`
  Labels: `D S R R`
  Assembly: `{'gives': 'SON', 'op': 'homophone', 'sounds_like': 'SON'}`
- `On the radio, Gong and Snoop` (MEDDLE)
  Definition: `Snoop`
  Sources: Gong -> MEDDLE
  Residue: `On the radio and`
  Labels: `R R R S R D`
  Assembly: `{'gives': 'MEDDLE', 'op': 'homophone', 'sounds_like': 'MEDDLE'}`
`telegraph` / `reversal`
- `Monster therefore returns` (OGRE)
  Definition: `Monster`
  Sources: therefore -> ERGO
  Residue: `returns`
  Labels: `D S R`
  Assembly: `{'gives': 'OGRE', 'op': 'reversal', 'order': ['OGRE'], 'reversed': 'ERGO'}`
- `Just English left on rising Italian island` (EQUITABLE)
  Definition: `Just`
  Sources: left -> QUIT, island -> ELBA, English -> E
  Residue: `on rising Italian`
  Labels: `D S S R R R S`
  Assembly: `{'gives': 'ABLE', 'op': 'reversal', 'order': ['E', 'QUIT', 'ABLE'], 'reversed': 'ELBA'}`
- `Figure reserve, Mo Salah, finally right to make a comeback` (RHOMBUS)
  Definition: `Figure`
  Sources: reserve -> SUB, Mo -> MO, Salah -> H, right -> R
  Residue: `finally to make a comeback`
  Labels: `D S S S R S R R R R`
  Assembly: `{'gives': 'RHOMBUS', 'op': 'reversal', 'order': ['RHOMBUS'], 'reversed': 'SUBMOHR', 'reversed_parts': ['SUB', 'MO', 'H', 'R']}`
`telegraph-toughie` / `anagram`
- `Due to land after flying, covered in lumps` (NODULATED)
  Definition: `covered in lumps`
  Sources: Due to land -> DUETOLAND
  Residue: `after flying`
  Labels: `S S S R R D D D`
  Assembly: `{'fodder': ['DUETOLAND'], 'op': 'anagram'}`
- `See the relics spread across Irish county` (Leicestershire)
  Definition: `county`
  Sources: See the relics -> SEETHERELICS, Irish -> IR
  Residue: `spread across`
  Labels: `S S S R R S D`
  Assembly: `{'op': 'charade', 'order': ['SEETHERELICS', 'IR']}`
- `Pre-cooked food made early, chopped up` (ready meal)
  Definition: `Pre-cooked food`
  Sources: made early -> MADEEARLY
  Residue: `chopped up`
  Labels: `D D S S R R`
  Assembly: `{'fodder': ['MADEEARLY'], 'op': 'anagram'}`
`telegraph-toughie` / `charade`
- `Love son's suit` (HEARTS)
  Definition: `suit`
  Sources: Love -> HEART, son's -> S
  Residue: ``
  Labels: `S S D`
  Assembly: `{'op': 'charade', 'order': ['HEART', 'S']}`
- `Papal church tale of chivalry` (ROMANCE)
  Definition: `tale of chivalry`
  Sources: Papal -> ROMAN, church -> CE
  Residue: ``
  Labels: `S S D D D`
  Assembly: `{'op': 'charade', 'order': ['ROMAN', 'CE']}`
- `...police officer over in mountains drinks` (ALCOPOPS)
  Definition: `drinks`
  Sources: police -> COP, officer -> O, mountains -> ALPS
  Residue: `over in`
  Labels: `S S R R S D`
  Assembly: `{'op': 'charade', 'order': ['COP', 'O', 'ALPS']}`
`telegraph-toughie` / `hidden`
- `Design it? I only should cover lighting` (IGNITION)
  Definition: `None`
  Sources: Design it I only -> IGNITION
  Residue: `should cover lighting`
  Labels: `S S S S R R R`
  Assembly: `{'op': 'hidden', 'words': 'Design it? I only'}`
- `… epicentre at Yerevan – location for appeal?` (ENTREATY)
  Definition: `appeal?`
  Sources: epicentre at Yerevan -> ENTREATY
  Residue: `location for`
  Labels: `S S S R R D`
  Assembly: `{'op': 'hidden', 'words': 'epicentre at Yerevan'}`
- `Musical character article fully embraces` (CLEF)
  Definition: `None`
  Sources: article fully -> CLEF
  Residue: `Musical character embraces`
  Labels: `R R S S R`
  Assembly: `{'op': 'hidden', 'words': 'article fully'}`

## Reading

This slice is not intended to solve hard clues. It gives the science project a clean baseline:
can a grammar model recover obvious source spans, definition spans, and simple residue before we ask it to handle nested containers, deletions, and reversals?

The next step is to add grammar tags/dependencies to this slice and test whether the grammar features predict these boundary signatures.
