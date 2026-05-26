# Grammar Boundary Classifier

Date: 2026-05-17

This is a small control experiment over the enriched scaffold.
Given wordplay tokens only, it predicts whether each token is `SOURCE` or `RESIDUE`.
The comparison is lexical-only versus lexical-plus-grammar.

Train/test records: `684` / `247`
Minimum feature support: `5`

## Result

- `lexical` model features: `168`
- `lexical` coverage: `1060/1060` (`100%`)
- `lexical` accuracy: `715/1060` (`67%`)
- `lexical` accuracy when covered: `715/1060` (`67%`)
- `grammar` model features: `511`
- `grammar` coverage: `1060/1060` (`100%`)
- `grammar` accuracy: `758/1060` (`72%`)
- `grammar` accuracy when covered: `758/1060` (`72%`)
- Correct-token gain from grammar: `43`

## Lexical Confusion

- `RESIDUE`: `SOURCE`=303, `RESIDUE`=195
- `SOURCE`: `SOURCE`=520, `RESIDUE`=42

## Grammar Confusion

- `RESIDUE`: `RESIDUE`=299, `SOURCE`=199
- `SOURCE`: `SOURCE`=459, `RESIDUE`=103

## Grammar Result By Operation

- `anagram`: `121/178` (`68%`)
- `charade`: `231/318` (`73%`)
- `container`: `11/14` (`79%`)
- `deletion`: `4/6` (`67%`)
- `deletion+anagram`: `4/5` (`80%`)
- `hidden`: `159/234` (`68%`)
- `hidden_reversed`: `95/125` (`76%`)
- `homophone`: `69/93` (`74%`)
- `reversal`: `64/87` (`74%`)

## Grammar Wrong Examples

- `Routes in event of emergency across a deep rupture` (ESCAPEROADS)
  Token: `event` gold `RESIDUE`, predicted `SOURCE` at `91%` from `mid_pos=N`
  Operation: `anagram`; wordplay tokens: `in event of emergency across a deep rupture`
- `Routes in event of emergency across a deep rupture` (ESCAPEROADS)
  Token: `emergency` gold `RESIDUE`, predicted `SOURCE` at `100%` from `prev_clean=of`
  Operation: `anagram`; wordplay tokens: `in event of emergency across a deep rupture`
- `Routes in event of emergency across a deep rupture` (ESCAPEROADS)
  Token: `across` gold `SOURCE`, predicted `RESIDUE` at `72%` from `mid_context=N<P>D`
  Operation: `anagram`; wordplay tokens: `in event of emergency across a deep rupture`
- `Routes in event of emergency across a deep rupture` (ESCAPEROADS)
  Token: `rupture` gold `RESIDUE`, predicted `SOURCE` at `71%` from `mid_context=J<N><END>`
  Operation: `anagram`; wordplay tokens: `in event of emergency across a deep rupture`
- `Bow to clerk bewildered by tall building` (TOWERBLOCK)
  Token: `to` gold `SOURCE`, predicted `RESIDUE` at `80%` from `clean=to`
  Operation: `anagram`; wordplay tokens: `Bow to clerk bewildered by`
- `Engineer dares to skirt eastern body of water` (REDSEA)
  Token: `Engineer` gold `RESIDUE`, predicted `SOURCE` at `100%` from `mid_context=<START><N>Vi`
  Operation: `charade`; wordplay tokens: `Engineer dares to skirt eastern`
- `Take a risk having disrupted nice chat` (CHANCEIT)
  Token: `having` gold `RESIDUE`, predicted `SOURCE` at `64%` from `dep=aux`
  Operation: `anagram`; wordplay tokens: `having disrupted nice chat`
- `Renown at intervals for a very long period` (EON)
  Token: `Renown` gold `RESIDUE`, predicted `SOURCE` at `100%` from `next_clean=at`
  Operation: `anagram`; wordplay tokens: `Renown at intervals for`

## Grammar Unknown Examples


## Reading

This is intentionally crude, but it answers a useful first question.
If grammar improves token boundary classification, the next experiment should move from token labels to span labels: predicting contiguous SOURCE blocks and RESIDUE attachment.
If it does not improve much, grammar may still be useful as a verifier for ambiguous cases rather than as a primary boundary finder.
