# Residue Baseline Evaluation

Date: 2026-05-17

This is a control experiment for GT V2. It ignores grammar, syntax, answer letters, and block mechanics.
It predicts the structured assembly operation using only residue text left after SOURCE and DEF tokens are removed.

## Result

- Records: `931`
- Train/test split: `684` / `247` using deterministic clue hash
- Minimum residue feature support: `3` training examples
- Test coverage: `193/247` (`78%`)
- Overall accuracy: `112/247` (`45%`)
- Accuracy when covered: `112/193` (`58%`)
- High-confidence accuracy: `30/35` (`86%`)

## Operation Mix

Train: `charade`=190, `anagram`=151, `hidden`=146, `hidden_reversed`=64, `reversal`=58, `homophone`=49, `container`=17, `deletion+anagram`=6, `deletion`=3
Test: `charade`=66, `hidden`=57, `anagram`=40, `homophone`=31, `hidden_reversed`=24, `reversal`=23, `container`=4, `deletion+anagram`=1, `deletion`=1

## Confusion By Gold Operation

- `anagram` (40): `UNKNOWN`=15, `charade`=11, `anagram`=11, `hidden`=1, `reversal`=1, `homophone`=1
- `charade` (66): `charade`=53, `anagram`=5, `UNKNOWN`=3, `reversal`=2, `hidden`=1, `hidden_reversed`=1, `homophone`=1
- `container` (4): `UNKNOWN`=3, `charade`=1
- `deletion` (1): `anagram`=1
- `deletion+anagram` (1): `hidden`=1
- `hidden` (57): `hidden`=19, `UNKNOWN`=18, `charade`=17, `anagram`=2, `homophone`=1
- `hidden_reversed` (24): `hidden_reversed`=9, `charade`=6, `hidden`=5, `UNKNOWN`=3, `homophone`=1
- `homophone` (31): `homophone`=19, `UNKNOWN`=7, `anagram`=2, `charade`=2, `hidden`=1
- `reversal` (23): `hidden_reversed`=8, `UNKNOWN`=5, `homophone`=5, `charade`=2, `anagram`=2, `reversal`=1

## Feature Families Used

- `empty_residue` (23): `charade`=23
- `full_residue` (27): `homophone`=12, `charade`=10, `hidden`=5
- `run` (33): `charade`=20, `hidden`=5, `homophone`=3, `anagram`=3, `hidden_reversed`=1, `reversal`=1
- `token` (110): `charade`=39, `anagram`=20, `hidden`=18, `hidden_reversed`=17, `homophone`=13, `reversal`=3

## Correct Examples

- `Engineer dares to skirt eastern body of water` (REDSEA)
  Gold: `charade`; predicted: `charade`; confidence: `31%` from `token=to`
  Residue: `engineer to skirt`
  Labels: `R S R R S D D D`
- `Painter got excited about shift in financial projection` (OPERATINGBUDGET)
  Gold: `anagram`; predicted: `anagram`; confidence: `67%` from `token=excited`
  Residue: `excited about in`
  Labels: `S S R R S R D D`
- `Vehicle for late figure here as planned` (HEARSE)
  Gold: `anagram`; predicted: `anagram`; confidence: `37%` from `token=for`
  Residue: `for late figure planned`
  Labels: `D R R R S S R`
- `File apt to be shredded in squalid cinema` (FLEAPIT)
  Gold: `anagram`; predicted: `anagram`; confidence: `75%` from `token=be`
  Residue: `to be shredded in squalid`
  Labels: `S S R R R R R D`
- `Loan is devised for female` (ALISON)
  Gold: `anagram`; predicted: `anagram`; confidence: `37%` from `token=for`
  Residue: `devised for`
  Labels: `S S R R D`
- `Supermarket among restaurants stricken in flood, maybe` (NATURALDISASTER)
  Gold: `charade`; predicted: `charade`; confidence: `26%` from `token=in`
  Residue: `among stricken in flood maybe`
  Labels: `S R S R R R R`

## Wrong Examples

- `Routes in event of emergency across a deep rupture` (ESCAPEROADS)
  Gold: `anagram`; predicted: `charade`; confidence: `26%` from `token=in`
  Residue: `in event of emergency rupture`
  Labels: `D R R R R S S S R`
- `Bow to clerk bewildered by tall building` (TOWERBLOCK)
  Gold: `anagram`; predicted: `hidden`; confidence: `38%` from `token=by`
  Residue: `bewildered by`
  Labels: `S S S R R D D`
- `Renown at intervals for a very long period` (EON)
  Gold: `anagram`; predicted: `charade`; confidence: `60%` from `token=at`
  Residue: `renown at intervals`
  Labels: `R R R S D D D D`
- `Unusual logo on bread and citrus fruit` (BLOODORANGE)
  Gold: `anagram`; predicted: `charade`; confidence: `53%` from `token=and`
  Residue: `unusual and citrus`
  Labels: `R S S S R R D`
- `Roger today put in a mess is critical` (DEROGATORY)
  Gold: `anagram`; predicted: `reversal`; confidence: `67%` from `token=put`
  Residue: `put in a mess is`
  Labels: `S S R R R R R D`
- `Burst into tears if in emergency response centre` (FIRESTATION)
  Gold: `anagram`; predicted: `charade`; confidence: `34%` from `run=in`
  Residue: `burst in`
  Labels: `R S S S R D D D`
- `Stops silly boast about runs` (ABORTS)
  Gold: `anagram`; predicted: `charade`; confidence: `45%` from `run=about`
  Residue: `silly about`
  Labels: `D R S R S`
- `Supply container grabbed by swimmer` (FURNISH)
  Gold: `charade`; predicted: `hidden`; confidence: `38%` from `token=by`
  Residue: `grabbed by`
  Labels: `D S R R S`

## Unknown Examples

- `Take a risk having disrupted nice chat` (CHANCEIT)
  Gold: `anagram`; predicted: `UNKNOWN`; confidence: `0%` from `none`
  Residue: `having disrupted`
  Labels: `D D D R R S S`
- `First dame in novel` (MAIDEN)
  Gold: `anagram`; predicted: `UNKNOWN`; confidence: `0%` from `none`
  Residue: `novel`
  Labels: `D S S R`
- `Partly required anger` (IRE)
  Gold: `hidden`; predicted: `UNKNOWN`; confidence: `0%` from `none`
  Residue: `partly`
  Labels: `R S D`
- `Racket heard inside partly` (DIN)
  Gold: `hidden`; predicted: `UNKNOWN`; confidence: `0%` from `none`
  Residue: `partly`
  Labels: `D S S R`
- `Woman featuring in a diary? Not entirely` (NADIA)
  Gold: `hidden`; predicted: `UNKNOWN`; confidence: `0%` from `none`
  Residue: `featuring`
  Labels: `D R S S S S S`
- `Oddball reconstructed Roman lab` (ABNORMAL)
  Gold: `anagram`; predicted: `UNKNOWN`; confidence: `0%` from `none`
  Residue: `reconstructed`
  Labels: `D R S S`

## Ambiguous Residue Examples

- `Bow to clerk bewildered by tall building` (TOWERBLOCK)
  Gold: `anagram`; predicted: `hidden`; confidence: `38%` from `token=by`
  Residue: `bewildered by`
  Labels: `S S S R R D D`
- `Renown at intervals for a very long period` (EON)
  Gold: `anagram`; predicted: `charade`; confidence: `60%` from `token=at`
  Residue: `renown at intervals`
  Labels: `R R R S D D D D`
- `Painter got excited about shift in financial projection` (OPERATINGBUDGET)
  Gold: `anagram`; predicted: `anagram`; confidence: `67%` from `token=excited`
  Residue: `excited about in`
  Labels: `S S R R S R D D`
- `Unusual logo on bread and citrus fruit` (BLOODORANGE)
  Gold: `anagram`; predicted: `charade`; confidence: `53%` from `token=and`
  Residue: `unusual and citrus`
  Labels: `R S S S R R D`
- `Vehicle for late figure here as planned` (HEARSE)
  Gold: `anagram`; predicted: `anagram`; confidence: `37%` from `token=for`
  Residue: `for late figure planned`
  Labels: `D R R R S S R`
- `File apt to be shredded in squalid cinema` (FLEAPIT)
  Gold: `anagram`; predicted: `anagram`; confidence: `75%` from `token=be`
  Residue: `to be shredded in squalid`
  Labels: `S S R R R R R D`
- `Roger today put in a mess is critical` (DEROGATORY)
  Gold: `anagram`; predicted: `reversal`; confidence: `67%` from `token=put`
  Residue: `put in a mess is`
  Labels: `S S R R R R R D`
- `Loan is devised for female` (ALISON)
  Gold: `anagram`; predicted: `anagram`; confidence: `37%` from `token=for`
  Residue: `devised for`
  Labels: `S S R R D`

## Reading

Residue-only prediction is useful but not sufficient. It is strongest when the residue contains a conventional indicator phrase.
It struggles where the same surface phrase can perform several jobs, or where residue is mainly grammatical glue.
That is exactly the space where grammar triage should earn its keep: not by replacing the residue lexicon, but by deciding attachment, scope, and whether a residue phrase is an operation, a connector, or part of a source phrase.
