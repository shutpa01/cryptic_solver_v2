# Grammar Span Boundary Experiment

Date: 2026-05-17

This experiment moves from token labels to span boundaries.
It uses the same token classifier outputs, then asks whether contiguous `SOURCE` and `RESIDUE` spans are recovered exactly.

Train/test records: `684` / `247`
Minimum feature support: `5`

## Result

- Lexical exact SOURCE span recall: `88/288` (`31%`)
- Grammar exact SOURCE span recall: `125/288` (`43%`)
- Lexical exact RESIDUE span recall: `60/291` (`21%`)
- Grammar exact RESIDUE span recall: `114/291` (`39%`)
- Lexical boundary-pair accuracy: `368/813` (`45%`)
- Grammar boundary-pair accuracy: `401/813` (`49%`)

## Boundary Accuracy By Operation

- `anagram`: `62/138` (`45%`)
- `charade`: `128/252` (`51%`)
- `container`: `6/10` (`60%`)
- `deletion`: `1/5` (`20%`)
- `deletion+anagram`: `2/4` (`50%`)
- `hidden`: `82/177` (`46%`)
- `hidden_reversed`: `53/101` (`52%`)
- `homophone`: `34/62` (`55%`)
- `reversal`: `33/64` (`52%`)

## Span Error Examples

- `Routes in event of emergency across a deep rupture` (ESCAPEROADS)
  Operation: `anagram`
  Gold SOURCE: `across a deep`
  Pred SOURCE: `event`; `emergency`; `a deep rupture`
  Gold RESIDUE: `in event of emergency`; `rupture`
  Pred RESIDUE: `in`; `of`; `across`
  Pred labels: `R S R S R S S S`
- `Bow to clerk bewildered by tall building` (TOWERBLOCK)
  Operation: `anagram`
  Gold SOURCE: `Bow to clerk`
  Pred SOURCE: `Bow`; `clerk`
  Gold RESIDUE: `bewildered by`
  Pred RESIDUE: `to`; `bewildered by`
  Pred labels: `S R S R R`
- `Engineer dares to skirt eastern body of water` (REDSEA)
  Operation: `charade`
  Gold SOURCE: `dares`; `eastern`
  Pred SOURCE: `Engineer dares`; `eastern`
  Gold RESIDUE: `Engineer`; `to skirt`
  Pred RESIDUE: `to skirt`
  Pred labels: `S S R R S`
- `Take a risk having disrupted nice chat` (CHANCEIT)
  Operation: `anagram`
  Gold SOURCE: `nice chat`
  Pred SOURCE: `having`; `nice chat`
  Gold RESIDUE: `having disrupted`
  Pred RESIDUE: `disrupted`
  Pred labels: `S R S S`
- `Renown at intervals for a very long period` (EON)
  Operation: `anagram`
  Gold SOURCE: `for`
  Pred SOURCE: `Renown`; `intervals`
  Gold RESIDUE: `Renown at intervals`
  Pred RESIDUE: `at`; `for`
  Pred labels: `S R S R`
- `Painter got excited about shift in financial projection` (OPERATINGBUDGET)
  Operation: `anagram`
  Gold SOURCE: `Painter got`; `shift`
  Pred SOURCE: `Painter`; `excited`; `shift`
  Gold RESIDUE: `excited about`; `in`
  Pred RESIDUE: `got`; `about`; `in`
  Pred labels: `S R S R S R`
- `Unusual logo on bread and citrus fruit` (BLOODORANGE)
  Operation: `anagram`
  Gold SOURCE: `logo on bread`
  Pred SOURCE: `Unusual logo`; `bread`; `citrus`
  Gold RESIDUE: `Unusual`; `and citrus`
  Pred RESIDUE: `on`; `and`
  Pred labels: `S S R S R S`
- `Vehicle for late figure here as planned` (HEARSE)
  Operation: `anagram`
  Gold SOURCE: `here as`
  Pred SOURCE: `late figure here`
  Gold RESIDUE: `for late figure`; `planned`
  Pred RESIDUE: `for`; `as planned`
  Pred labels: `R S S S R R`

## Reading

The span task is much stricter than token classification.
A single misplaced glue word can destroy an otherwise useful source span.
This is why V2 should probably not begin by asking for a single hard parse.
It should generate a small set of plausible block anatomies, then use answer mechanics to verify scope and assembly.
