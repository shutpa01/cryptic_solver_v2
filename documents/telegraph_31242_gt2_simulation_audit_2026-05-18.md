# Telegraph 31242 GT2 Simulation Audit

Date: 2026-05-18

Purpose: estimate how much GT2/atomic preservation could recover from the gap between the current reviewed result and the optimistic in-run `Signature+Enriched` result.

This is not a design-theory document. It is a clue-by-clue audit of what the current system already had, what the provisional enriched run claimed, and what GT2 would need to preserve for the verifier to recognise the solve.

## Measured States

Observed by user during run/review:

| State | High-confidence result | Meaning |
|---|---:|---|
| Pre-enrichment run output | 3 | Current permanent system before review/apply |
| Provisional in-run report | 23 | `Signature` + temporary `Signature+Enriched` rerun |
| After enrichment review/reverify | 2 | Current reviewed-enrichment path collapses most candidates |

Saved report evidence:

- `documents/puzzle_report_telegraph_31242.txt` reports 30 clues, 23 assembled, 7 failed.
- It splits the 23 assembled clues into 7 `Signature` and 16 `Sig+Enriched`.
- `documents/pending_gaps_telegraph_31242.json` contains 27 suggested gaps.
- Most suggested gaps are definition rows. Two important lexical rows are demonstrably wrong as review rows:
  - `Article -> EL`, should be `Article in Madrid -> EL`.
  - `postgraduate -> AM`, should preserve `postgraduate -> MA` and transformed output `AM`.

## Failure Taxonomy

| Code | Meaning | GT2 relevance |
|---|---|---|
| DEF | Wordplay parse is plausible/valid, but the definition-answer row is missing or too weak. | GT2 should not pretend this is a new parse rule, but should allow structural solve plus separate definition evidence. |
| SPAN | Right value attached to wrong clue span. | Direct GT2 gain: preserve phrase/noun spans as source atoms. |
| XFORM | Source value and transformed output were collapsed into one DB row. | Direct GT2 gain: preserve source atom, operation, and output atom separately. |
| ASSEMBLY | Pieces are mostly right but current explanation/output shape is not verifier-recognisable. | Direct GT2 gain: graph assembly should expose exact operation attachment. |
| BAD | Provisional parse appears materially wrong. | Not a GT2 gain unless a different graph recovers it. |

## Provisional `Sig+Enriched` Audit

| Clue | Answer | Provisional parse | Review issue | Classification | GT2 recoverable? |
|---|---|---|---|---|---|
| 20D | RADICAL | hidden reversed in `placid army`; definition `Militant` | Mainly missing definition row | DEF | Likely, if structural hidden solve can stand with separately verified definition |
| 3D | GLADSTONE | hidden in `sending lads to nearby`; definition `Prime minister` | Mainly missing definition/entity row | DEF | Likely |
| 11D | FREE-SPIRITED | anagram of `I PREFER DIETS`; definition `Unconventional` | Mainly missing definition row | DEF | Likely |
| 13A | MOOR | `ROOM` reversed; definition `Uncultivated land` | Mainly missing definition row | DEF | Likely |
| 15A | OVERWEENING | anagram of `VOW ENGINEER`; definition `Arrogant` | Mainly missing definition row | DEF | Likely |
| 18A | FINGERPRINT | `FINER` / `PRINT` / `G`; definition `digital ID` | Explanation flattens container attachment; likely `FINER` around `G` + `PRINT` | ASSEMBLY/XFORM | Likely |
| 19D | NUDISTS | `DIS` inside `NUTS`; definition `people with no clothes on` | Output is plausible but needs explicit container graph | ASSEMBLY | Likely |
| 21A | CODE | `C` + `ODE`; definition `Set of expectations` | Mainly missing definition row | DEF | Likely |
| 22A | ARCHIMEDES | anagram of `REACHES DIM`; definition `mathematician` | Mainly missing definition row | DEF | Likely |
| 24A | ABSOLUTE | `AB` + `S` + `O` + `LUTE`; definition `Unqualified` | Mostly valid; needs exact first-letter attachment for `front of seamen` | DEF/ASSEMBLY | Likely |
| 25A | STUCCO | `S` + `CO` with reversed `CUT`; definition `plaster` | Needs `CUT -> TUC` as transformed atom and insertion into `CO`/charade order | XFORM/ASSEMBLY | Likely |
| 26A | GO STEADY | anagram of `AGED TOYS`; definition `Regularly see` | Mainly missing definition row | DEF | Likely |
| 27A | PSALMS | `PAL` with `S`, plus `MS`; definition `book` | Needs container attachment: `PAL` penning `S`, then `MS` | ASSEMBLY | Likely |
| 4A | ABSINTHE | anagram of `BATHES IN`; definition `booze` | Mainly missing definition row | DEF | Likely |
| 6D | IMAM | `I` + `M` + `AM`; definition `Religious leader's` | Bad review row: proposed `postgraduate -> AM`; correct source is `postgraduate -> MA`, transformed to `AM` | XFORM | Strong |
| 8D | ELGAR | `EL` + `RAG` reversed; definition `composer` | Bad review row: proposed `Article -> EL`; correct span is `Article in Madrid -> EL` | SPAN | Strong |

Initial read: none of the 16 provisional `Sig+Enriched` parses is obviously worthless. Several are only definition coverage issues; several are direct GT2 representation failures where the current review row is too crude to preserve what the parse actually needs.

## Failed Clue Probe: FORAGE

Clue:

```text
Search advanced into blacksmith's workshop = FORAGE
```

Direct solver evidence:

- Definition candidate exists: `Search -> FORAGE`.
- `advanced` gives `A`.
- `into` is recognised as a container indicator.
- `blacksmith's` alone gives `FORGER`, `WORKER`.
- `workshop` alone gives `STUDIO`, `PLANT`, `SHOP`, `WORK`.
- No exact DB row exists for `blacksmith's workshop -> FORGE`.

With a temporary overlay:

```text
blacksmith's workshop -> FORGE
```

the existing solver verifies:

```text
FORGE containing A = FORAGE
confidence 100
```

GT2 diagnosis:

- Preserve `blacksmith's workshop` as a noun-phrase source span.
- Since answer is known and `A` is known, compute complement `FORAGE - A = FORGE`.
- Ask narrow lexical verifier/Haiku question: does `blacksmith's workshop` clue `FORGE`?
- Existing assembler can then confirm the clue.

Classification: `SPAN` plus complement verification. Strong GT2 candidate.

## What The Collapse Means

The drop from provisional 23 high to reviewed 2 high does not show that the solver is bad. It shows that the current review/enrichment interface is too lossy.

The review layer stores many discoveries as:

```text
word -> letters
```

but the real parse often needs:

```text
source span -> source value -> operation -> output value -> assembly
```

Examples:

```text
Article in Madrid -> EL
postgraduate -> MA -> reversed/supporting -> AM
blacksmith's workshop -> FORGE; answer complement after A removal
PAL contains S -> PSAL; + MS -> PSALMS
CUT -> reversed TUC; placed inside/with S + CO -> STUCCO
FINER contains G; + PRINT -> FINGERPRINT
```

The current pipeline can temporarily use some of this material, but the durable review artifact cannot represent it. That is why reverify collapses.

## Simulation Estimate

Conservative buckets from the current evidence:

| Bucket | Count | Notes |
|---|---:|---|
| Reviewed high after enrichment | 2 | User observation after review/reverify |
| Provisional `Sig+Enriched` with likely valid parse | 16 | Needs per-clue confirmation, but all 16 have plausible mechanics |
| Additional failed clue already shown GT2-recoverable | 1 | `FORAGE` |
| Remaining failed not yet audited | 6 | `KNEADING`, `OWE`, `ESTIMATES`, `GAME SHOWS`, `TRIPOLI`, `RASCAL` |

First-pass GT2 simulated ceiling from audited evidence:

```text
2 reviewed high + 16 likely recoverable provisional + 1 FORAGE = 19/30
```

This is not a claim that GT2 will immediately solve 19. It is the current evidence-backed target: these are clues where the existing system has enough mechanical material, or nearly enough, that a graph-preserving verifier should have a realistic path.

## Immediate GT2 Requirements From This Puzzle

1. Preserve phrase and noun-phrase spans as first-class source blocks.
   - `blacksmith's workshop`
   - `Article in Madrid`

2. Preserve source value separately from transformed output.
   - `postgraduate -> MA`, then `MA -> AM`
   - `CUT -> TUC`

3. Preserve operation attachment.
   - `FINER` around `G`
   - `PAL` penning `S`
   - `DIS` probing `NUTS`

4. Compute answer-constrained complements.
   - `FORAGE` with `A` inserted implies shell `FORGE`.

5. Ask Haiku narrow span-value questions only after the graph has constrained the target.
   - Bad: solve the whole clue.
   - Good: does `blacksmith's workshop` clue `FORGE`?
   - Good: does `Article in Madrid` clue `EL`?

6. Make reverify consume the graph, not a flattened enrichment row.

## Next Audit Slice

The remaining unaudited failed clues are the next test of whether GT2 can go beyond preserving provisional successes:

| Clue | Answer | Initial hypothesis to test |
|---|---|---|
| 10A | KNEADING | Homophone: `needing` / `requiring`; likely phonetic verification issue |
| 14A | OWE | `poet` barely + `W`; likely deletion/container span issue |
| 16D | ESTIMATES | `MATE` in anagram of `SITES`; likely operation attachment issue |
| 17D | GAME SHOWS | `GA` + `?` in `mess hall`/military dining hall; likely phrase/span issue |
| 7D | TRIPOLI | `TRIO`/`L`/`I` with `P`; likely container/order issue |
| 9A | RASCAL | `RA` + `S[till]` + `CALM?`; needs evidence inspection |

Those should be audited one by one using the same method as `FORAGE`: current DB evidence, existing analyzer output, missing span/value, and whether the existing assembler can verify once the missing atom is supplied.
