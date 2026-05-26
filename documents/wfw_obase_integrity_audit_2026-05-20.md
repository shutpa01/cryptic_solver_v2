# WFW / obase structural integrity audit

Date: 2026-05-20

Puzzle audited: Daily Mail Cryptic 17882.

## Purpose

This is not a clue-fixing note.  It defines the recovery baseline for WFW.

The design requirement is:

> WFW must be a superset of obase.  There should not be a case where WFW is
> inferior to the old production solver on an existing solving aid.

The first WFW pass does not meet that requirement.

## Current compatibility result

Fresh WFW was compared with stored obase results for the puzzle.

```text
covered: 7
regression: 18
wfw_extra: 0
both_unsolved: 8
```

Breakdown by stored obase type:

```text
anagram: regression=3
anagram, charade, deletion: covered=1
charade: covered=3, regression=4
charade, parts: covered=1, regression=1
charade, reversal: regression=1
container: covered=2, regression=3
deletion: regression=2
double_definition: regression=2
hidden: regression=2
unsolved: both_unsolved=8
```

## Structural failures shown

1. WFW is not currently a faithful carrier of obase capability.

   It proves only a minority of obase-solved clues in this puzzle.  This is
   not acceptable as a replacement path.

2. WFW allowed a one-source whole-answer charade.

   A clue span that directly gives the entire answer is not a charade proof.
   It must be handled by a whole-clue mechanism, usually double definition or
   cryptic definition, or remain in review.

3. WFW gap discovery is underbuilt.

   The system currently does not generate enough useful pending enrichment
   candidates from failed WFW evidence.

4. obase-derived proof states and WFW-native proof states are still mixed.

   Any user-facing WFW proof must come from a WFW-native token parse and
   materialised WFW assembly, not from an obase status row.

## Recovery order

1. Restore obase capability inside WFW.

   For each obase type, make WFW prove at least the same class of clues before
   adding stricter/new methods:

   - double definition
   - hidden
   - anagram
   - charade
   - container
   - deletion
   - reversal and reversal-charade
   - parts/positional clues

2. Build WFW gap collection as a first-class stage.

   A failed WFW proof must explain which evidence is missing:

   - missing synonym/definition pair
   - missing abbreviation
   - missing indicator
   - missing phrase span
   - unsupported transformation

3. Only then tighten scoring.

   Stronger scoring is necessary, but it is not the repair by itself.  Scoring
   can only be meaningful after WFW evidence and gap generation are faithful.

## Tools added

- `scripts/audit_wfw_obase_compatibility.py`
  Compares fresh WFW proof coverage against stored obase results.

- `signature_solver/wfw_gap_collector.py`
  First WFW-native gap collector slice.

- `scripts/audit_wfw_gaps_puzzle.py`
  Prints WFW-native DB gaps without writing to the database.

## First guardrail added

`token_parse_assembler` now rejects one-block whole-answer charades.

This prevents WFW from treating a single full-answer synonym as wordplay proof.
If that relation is legitimate, it must be proved by a proper whole-clue
mechanism.
