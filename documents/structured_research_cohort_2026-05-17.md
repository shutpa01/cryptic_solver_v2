# Structured Research Cohort

Date: 2026-05-17

This is the first GT V2 research cohort built from existing structured explanations.
The supervision is the structured mapping from clue words to answer pieces, plus assembly.
Blog explanations are retained when present, but they are not required.

Eligible high-confidence records: 159037
Balanced sample records written: 4954

## Source Inventory

- `cordelia`: 340 eligible; 273 complete piece mappings; 281 with blog text; 325 with exact definition tokens; 683/776 pieces mapped
- `dailymail`: 8224 eligible; 8190 complete piece mappings; 0 with blog text; 4406 with exact definition tokens; 19321/19357 pieces mapped
- `guardian`: 73644 eligible; 55529 complete piece mappings; 34645 with blog text; 60689 with exact definition tokens; 111491/138815 pieces mapped
- `independent`: 14575 eligible; 8626 complete piece mappings; 12398 with blog text; 12192 with exact definition tokens; 21543/30909 pieces mapped
- `telegraph`: 33478 eligible; 33231 complete piece mappings; 28802 with blog text; 32456 with exact definition tokens; 67144/67405 pieces mapped
- `telegraph-toughie`: 8197 eligible; 7793 complete piece mappings; 3905 with blog text; 7555 with exact definition tokens; 15803/16366 pieces mapped
- `times`: 20579 eligible; 15083 complete piece mappings; 13883 with blog text; 19524 with exact definition tokens; 34951/43760 pieces mapped

## Operation Mix By Source

- `cordelia`: charade=167, anagram=84, container=29, hidden=27, double_definition=10, deletion=7, reversal=6, homophone=3
- `dailymail`: blank=7794, charade=187, anagram=110, hidden=52, container=39, homophone=10, deletion=10, double_definition=7
- `guardian`: charade=28320, anagram=18135, blank=8620, hidden=5524, container=3963, double_definition=3064, deletion=1379, hidden_reversed=1319
- `independent`: charade=5156, blank=4494, anagram=2466, container=1098, hidden=534, reversal=320, hidden_reversed=198, acrostic=140
- `telegraph`: charade=14864, anagram=10159, hidden=3039, blank=2864, hidden_reversed=806, double_definition=559, acrostic=458, homophone=238
- `telegraph-toughie`: charade=3544, anagram=2128, hidden=938, blank=767, hidden_reversed=364, acrostic=203, homophone=88, reversal=70
- `times`: charade=10406, anagram=4581, container=1540, blank=1478, hidden=923, reversal=548, deletion=317, hidden_reversed=314

## Example Records

`cordelia`
- `Twist of fate's causing blow-out` (FEAST)
  Definition: `blow-out`
  Pieces: fate's -> FEAST (fate's)
  Assembly: `{'op': 'anagram', 'fodder': ['FATES'], 'gives': 'FEAST'}`
  Residue: `Twist of causing`
  Blog: `More food. Lots of it can be found by making an anagram (twist of) of FATE'S`
- `Hide a broken heart` (EARTH)
  Definition: `Hide`
  Pieces: heart -> EARTH (heart)
  Assembly: `{'op': 'anagram', 'fodder': ['HEART'], 'gives': 'EARTH'}`
  Residue: `a broken`
`dailymail`
- `Vegetable kept by Anneka least` (KALE)
  Definition: `Vegetable`
  Pieces: Anneka least -> KALE (Anneka least)
  Assembly: `{'op': 'hidden', 'words': 'Anneka least', '_definition': 'Vegetable'}`
  Residue: `kept by`
- `Walk unevenly entering slim path` (LIMP)
  Definition: `Walk`
  Pieces: slim path -> LIMP (slim path)
  Assembly: `{'op': 'hidden', 'words': 'slim path', '_definition': 'Walk'}`
  Residue: `unevenly entering`
`guardian`
- `See 10` (ELECTRIC)
  Definition: `See 10`
  Pieces: See 10 -> ELECTRIC (unmapped)
  Assembly: `{'op': 'double_definition'}`
  Residue: ``
- `Cover one’s tracks maybe to achieve objective` (TARGET)
  Definition: `objective`
  Pieces: Cover one's tracks -> TAR (unmapped); achieve -> GET (achieve)
  Assembly: `{'op': 'charade', 'order': ['TAR', 'GET']}`
  Residue: `Cover one s tracks maybe to`
`independent`
- `Clog dance` (BALL)
  Definition: `Clog`
  Pieces: dance -> BALL (dance)
  Assembly: `{'op': 'double_definition'}`
  Residue: ``
  Blog: `Double definition – we had to check the first`
- `Marks vehicles on style, at first` (SCARS)
  Definition: `Marks`
  Pieces: vehicles -> CARS (vehicles); style, -> S (style)
  Assembly: `{'op': 'charade', 'order': ['CARS', 'S']}`
  Residue: `on at first`
`telegraph`
- `Poles at front of long grass` (SNITCH)
  Definition: `grass`
  Pieces: Poles -> SN (Poles); long -> ITCH (long)
  Assembly: `{'op': 'charade', 'order': ['SN', 'ITCH']}`
  Residue: `at front of`
- `Shock when quiet retiring gentleman cracks safe` (SURPRISE)
  Definition: `Shock`
  Pieces: safe -> SURE (safe); quiet -> P (quiet); retiring gentleman -> RIS (retiring gentleman)
  Assembly: `{'op': 'container', 'inner': 'PRIS', 'outer': 'SURE', 'pos': 3, 'combined': 'SURPRISE', 'order': ['SURPRISE'], 'merged_inner': 'PRIS'}`
  Residue: `when cracks`
`telegraph-toughie`
- `Love son's suit` (HEARTS)
  Definition: `suit`
  Pieces: Love -> HEART (Love); son's -> S (son's)
  Assembly: `{'op': 'charade', 'order': ['HEART', 'S']}`
  Residue: ``
  Blog: `The imagined place of the origin of love and the abbreviation for Son`
- `Papal church tale of chivalry` (ROMANCE)
  Definition: `tale of chivalry`
  Pieces: Papal -> ROMAN (Papal); church -> CE (church)
  Assembly: `{'op': 'charade', 'order': ['ROMAN', 'CE']}`
  Residue: ``
`times`
- `Portrait of interest to police stolen from among mine` (PHOTOFIT)
  Definition: `Portrait of interest to police`
  Pieces: mine -> PIT (mine); to -> OF (unmapped); stolen -> HOT (stolen)
  Assembly: `{'op': 'container', 'inner': 'HOTOF', 'outer': 'PIT', 'pos': 1, 'combined': 'PHOTOFIT', 'order': ['PHOTOFIT'], 'merged_inner': 'HOTOF'}`
  Residue: `from among`
  Blog: `HOT (stolen from) in (among) PIT (mine) Did not know this registered trademark. Apparently it is an alternative to “identikit”, which I also do not know!`
- `Terms of Endearment remade, so new The Sting?` (SWEETNOTHINGS)
  Definition: `Terms of Endearment`
  Pieces: so new The Sting -> SONEWTHESTING (so new The Sting)
  Assembly: `{'op': 'anagram', 'fodder': ['SONEWTHESTING'], 'gives': 'SWEETNOTHINGS'}`
  Residue: `remade`
  Blog: `anagram of (remade) SO NEW THE STING`

## Research Use

This cohort should be used before any further Times-only mining.
The next science question is whether surface grammar can recover the same
SOURCE/RESIDUE boundaries and operation attachment found in these structured records.

## First Bucket Reading

The cleanest initial buckets are not simply the ones with blog explanations.
They are the buckets where structured pieces already map directly back to clue
text.

Good early training buckets:

- Telegraph charades
- Telegraph anagrams
- Daily Mail hidden clues
- Telegraph hidden clues
- Telegraph simple reversals and homophones
- Telegraph Toughie hidden and charade clues

These are useful for teaching SOURCE/DEF/RESIDUE boundaries because the
structured clue-word mappings are already very clean.

The harder science buckets are:

- Guardian containers
- Independent containers
- Guardian deletions
- Times deletions
- Guardian reversals
- Independent reversals

Those are not bad data. They are harder because the structured piece text often
does not match the clue span directly, or because the operation is nested inside
another operation. These buckets should be treated as the first real stress test
after the clean boundary model exists.

The immediate research path is therefore:

Start with clean structured mappings to learn whether grammar can recover source
boundaries and residue. Then move to the harder container/deletion/reversal
buckets to test whether grammar helps recover nested operation attachment.
