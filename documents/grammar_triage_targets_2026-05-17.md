# Grammar Triage Targets

Date: 2026-05-17

This report turns the residue-only baseline errors into research targets.
The purpose is to identify where grammar might add information beyond a residue lexicon.

## Buckets

- `glue-only`: `6` cases. Gold: `hidden`=4, `hidden_reversed`=1, `reversal`=1. Predicted: `charade`=6.
- `glue-dominates`: `39` cases. Gold: `anagram`=10, `hidden`=8, `charade`=6, `hidden_reversed`=4, `homophone`=4, `reversal`=4, `deletion+anagram`=1, `deletion`=1, `container`=1. Predicted: `charade`=20, `anagram`=7, `hidden`=4, `homophone`=4, `reversal`=2, `UNKNOWN`=1, `hidden_reversed`=1.
- `indicator-plus-glue`: `5` cases. Gold: `reversal`=2, `anagram`=1, `charade`=1, `hidden_reversed`=1. Predicted: `homophone`=3, `anagram`=1, `charade`=1.
- `direction-or-scope`: `13` cases. Gold: `reversal`=8, `hidden_reversed`=5. Predicted: `hidden_reversed`=8, `hidden`=5.
- `low-support-indicator`: `9` cases. Gold: `anagram`=3, `hidden`=3, `hidden_reversed`=1, `homophone`=1, `reversal`=1. Predicted: `UNKNOWN`=9.
- `empty-residue`: `6` cases. Gold: `hidden`=6. Predicted: `charade`=6.
- `other`: `57` cases. Gold: `hidden`=17, `anagram`=15, `homophone`=7, `charade`=6, `reversal`=6, `hidden_reversed`=3, `container`=3. Predicted: `UNKNOWN`=44, `charade`=6, `anagram`=4, `homophone`=2, `reversal`=1.

## Reading The Buckets

- `glue-only` and `glue-dominates` are the strongest grammar candidates: the residue lexicon is mostly seeing prepositions and conjunctions, so syntax and attachment should matter.
- `indicator-plus-glue` cases test whether we can keep an operation word chained to the right source block without letting nearby glue words steal the decision.
- `direction-or-scope` cases are where reversal, hidden, and reversed-hidden share similar residue and need answer-aware scope/direction checks.
- `low-support-indicator` is mostly a data problem: the operation signal exists, but the clean slice has not seen enough examples yet.

## glue-only

- `Bit of dirt in this mutton` (SMUT)
  Gold/predicted: `hidden` / `charade` from `full_residue=in`
  Residue: `in`
  Labels: `D D D R S S`
  Pieces: this mutton->THISMUTTON (hidden)
- `Main point in meeting is taken` (GIST)
  Gold/predicted: `hidden` / `charade` from `full_residue=in`
  Residue: `in`
  Labels: `D D R S S S`
  Pieces: meeting is taken->GIST (hidden)
- `Somewhat in clover at hers` (Rather)
  Gold/predicted: `hidden` / `charade` from `full_residue=in`
  Residue: `in`
  Labels: `D R S S S`
  Pieces: clover at hers->CLOVERATHERS (hidden)
- `Wagon in Ayr rolled over` (Lorry)
  Gold/predicted: `hidden_reversed` / `charade` from `run=over`
  Residue: `in over`
  Labels: `D R S S R`
  Pieces: Ayr rolled->LORRY (hidden)
- `Merrymaking in bar over` (REVEL)
  Gold/predicted: `reversal` / `charade` from `run=over`
  Residue: `in over`
  Labels: `D R S R`
  Pieces: bar->LEVER (reversal)

## glue-dominates

- `Routes in event of emergency across a deep rupture` (ESCAPEROADS)
  Gold/predicted: `anagram` / `charade` from `token=in`
  Residue: `in event of emergency rupture`
  Labels: `D R R R R S S S R`
  Pieces: across->ACROSS (anagram_fodder); a->A (anagram_fodder); deep->DEEP (anagram_fodder)
- `Bow to clerk bewildered by tall building` (TOWERBLOCK)
  Gold/predicted: `anagram` / `hidden` from `token=by`
  Residue: `bewildered by`
  Labels: `S S S R R D D`
  Pieces: Bow to clerk->BOWTOCLERK (anagram_fodder)
- `Take a risk having disrupted nice chat` (CHANCEIT)
  Gold/predicted: `anagram` / `UNKNOWN` from `none`
  Residue: `having disrupted`
  Labels: `D D D R R S S`
  Pieces: nice->NICE (anagram_fodder); chat->CHAT (anagram_fodder)
- `Renown at intervals for a very long period` (EON)
  Gold/predicted: `anagram` / `charade` from `token=at`
  Residue: `renown at intervals`
  Labels: `R R R S D D D D`
  Pieces: for->OEN (deletion)
- `Unusual logo on bread and citrus fruit` (BLOODORANGE)
  Gold/predicted: `anagram` / `charade` from `token=and`
  Residue: `unusual and citrus`
  Labels: `R S S S R R D`
  Pieces: logo->LOGO (anagram_fodder); on->ON (anagram_fodder); bread->BREAD (anagram_fodder)

## indicator-plus-glue

- `Noble Conservative featured in Ustinov broadcast` (Viscount)
  Gold/predicted: `anagram` / `homophone` from `run=broadcast`
  Residue: `featured in broadcast`
  Labels: `D S R R S R`
  Pieces: Conservative->C (abbreviation); Ustinov->USTINOV (anagram_fodder)
- `The Panel excited by singer – Plant` (elephant grass)
  Gold/predicted: `charade` / `anagram` from `token=excited`
  Residue: `excited by`
  Labels: `S S R R S D`
  Pieces: The Panel->THEPANEL (anagram_fodder); singer->GRASS (synonym)
- `Otherwise included in rising prices ledger` (ELSE)
  Gold/predicted: `hidden_reversed` / `charade` from `token=in`
  Residue: `included in rising`
  Labels: `D R R R S S`
  Pieces: prices ledger->ELSE (hidden_reversed)
- `Damage from sheep on the up` (MAR)
  Gold/predicted: `reversal` / `homophone` from `token=the`
  Residue: `from on the up`
  Labels: `D R S R R R`
  Pieces: sheep->RAM (reversal)
- `Well-provided-for, having arms from the east` (SNUG)
  Gold/predicted: `reversal` / `homophone` from `token=the`
  Residue: `having from the east`
  Labels: `D D R S R R R`
  Pieces: arms->GUNS (reversal)

## direction-or-scope

- `Bill regrets opportunities, to an extent, after retiring` (POSTER)
  Gold/predicted: `hidden_reversed` / `hidden` from `token=extent`
  Residue: `to an extent after retiring`
  Labels: `D S S R R R R R`
  Pieces: regrets opportunities,->REGRETSOPPORTUNITIES (hidden)
- `Photograph part of leg a minx raised` (IMAGE)
  Gold/predicted: `hidden_reversed` / `hidden` from `token=part`
  Residue: `part of raised`
  Labels: `D R R S S S R`
  Pieces: leg a minx->IMAGE (hidden_reversed)
- `Language picked up in Italy, to an extent` (Latin)
  Gold/predicted: `hidden_reversed` / `hidden` from `run=to an extent`
  Residue: `picked up to an extent`
  Labels: `D R R S S R R R`
  Pieces: in Italy,->LATIN (hidden_reversed)
- `Mark in text reflected some spiritual musings` (UMLAUT)
  Gold/predicted: `hidden_reversed` / `hidden` from `token=some`
  Residue: `in text reflected some`
  Labels: `D R R R R S S`
  Pieces: spiritual musings->UMLAUT (hidden_reversed)
- `Some declare bill is regressive and prejudiced` (ILLIBERAL)
  Gold/predicted: `hidden_reversed` / `hidden` from `run=some`
  Residue: `some regressive and`
  Labels: `R S S S R R D`
  Pieces: declare bill is->ILLIBERAL (hidden_reversed)

## low-support-indicator

- `First dame in novel` (MAIDEN)
  Gold/predicted: `anagram` / `UNKNOWN` from `none`
  Residue: `novel`
  Labels: `D S S R`
  Pieces: dame->DAME (anagram_fodder); in->IN (anagram_fodder)
- `Partly required anger` (IRE)
  Gold/predicted: `hidden` / `UNKNOWN` from `none`
  Residue: `partly`
  Labels: `R S D`
  Pieces: required->REQUIRED (hidden)
- `Racket heard inside partly` (DIN)
  Gold/predicted: `hidden` / `UNKNOWN` from `none`
  Residue: `partly`
  Labels: `D S S R`
  Pieces: heard inside->HEARDINSIDE (hidden)
- `Oddball reconstructed Roman lab` (ABNORMAL)
  Gold/predicted: `anagram` / `UNKNOWN` from `none`
  Residue: `reconstructed`
  Labels: `D R S S`
  Pieces: Roman->ROMAN (anagram_fodder); lab->LAB (anagram_fodder)
- `Philosopher's rent decreases, oddly` (Rene Descartes)
  Gold/predicted: `anagram` / `UNKNOWN` from `none`
  Residue: `oddly`
  Labels: `D S S R`
  Pieces: rent decreases,->RENTDECREASES (anagram_fodder)

## empty-residue

- `Whirlpool seen in flooded dyke` (EDDY)
  Gold/predicted: `hidden` / `charade` from `empty_residue=<empty>`
  Residue: `<empty>`
  Labels: `D S S S S`
  Pieces: seen in flooded dyke->SEENINFLOODEDDYKE (hidden)
- `Figure captivated by shindig, I think` (DIGIT)
  Gold/predicted: `hidden` / `charade` from `empty_residue=<empty>`
  Residue: `<empty>`
  Labels: `D S S S S S`
  Pieces: captivated by shindig, I think->CAPTIVATEDBYSHINDIGITHINK (hidden)
- `Periodical storm again? Not entirely` (MAG)
  Gold/predicted: `hidden` / `charade` from `empty_residue=<empty>`
  Residue: `<empty>`
  Labels: `D S S S S`
  Pieces: storm again? Not entirely->STORMAGAINNOTENTIRELY (hidden)
- `Bring home part of gear neatly` (EARN)
  Gold/predicted: `hidden` / `charade` from `empty_residue=<empty>`
  Residue: `<empty>`
  Labels: `D S S S S S`
  Pieces: home part of gear neatly->HOMEPARTOFGEARNEATLY (hidden)
- `Kind seen among pretty people` (TYPE)
  Gold/predicted: `hidden` / `charade` from `empty_residue=<empty>`
  Residue: `<empty>`
  Labels: `D S S S S`
  Pieces: seen among pretty people->SEENAMONGPRETTYPEOPLE (hidden)

## other

- `Stops silly boast about runs` (ABORTS)
  Gold/predicted: `anagram` / `charade` from `run=about`
  Residue: `silly about`
  Labels: `D R S R S`
  Pieces: boast->BOAST (from_clue); runs->R (abbreviation)
- `Woman featuring in a diary? Not entirely` (NADIA)
  Gold/predicted: `hidden` / `UNKNOWN` from `none`
  Residue: `featuring`
  Labels: `D R S S S S S`
  Pieces: in a diary? Not entirely->INADIARYNOTENTIRELY (hidden)
- `Lops head off flowering plant` (ASPHODEL)
  Gold/predicted: `anagram` / `charade` from `token=off`
  Residue: `off`
  Labels: `S S R D D`
  Pieces: head->HEAD (literal); lops->LOPS (literal)
- `Beery yobbo Elgar sozzled, left unconscious` (lager lout)
  Gold/predicted: `anagram` / `UNKNOWN` from `none`
  Residue: `sozzled`
  Labels: `D D S R S S`
  Pieces: Elgar->ELGAR (anagram_fodder); left->L (abbreviation); unconscious->OUT (synonym)
- `Consumes food items vegans reject, except starter` (EATS)
  Gold/predicted: `anagram` / `UNKNOWN` from `none`
  Residue: `food items vegans except starter`
  Labels: `D R R R S R R`
  Pieces: reject->EAST (deletion)

## Research Implication

The next grammar experiment should not start by predicting clue type directly.
It should start by asking whether grammar can protect block boundaries from glue words and attach operation residues to the correct source span.
That keeps the experiment close to the user's core claim: grammar signature and wordplay signature emanate from the same words.
