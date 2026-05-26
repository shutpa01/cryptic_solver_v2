# Explanation Block Candidate Extraction

Date: 2026-05-17

This is a generated mining report from `scripts/extract_explanation_blocks.py`.
It is not a solver change.

Rows scanned: 20000
Interesting records written: 16389
Extracted blocks: 34351
Phrase-shaped blocks: 7626

## Operation Mix

- `charade`: 8830
- `container`: 1559
- `anagram`: 1233
- `abr`: 1015
- `synonym`: 747
- `container_charade`: 599
- `reversal_charade`: 565
- `homophone`: 442
- `hidden`: 399
- `reversal`: 307
- `del`: 283
- `anagram_charade`: 235

## Source Type Mix

- `ABR`: 14908
- `SYN`: 12444
- `CON`: 2171
- `DEL`: 1472
- `ANA`: 1468
- `REV`: 872
- `HOM`: 442
- `HID`: 399
- `CON+REV`: 175

## Mapping Objections

- `no_exact_clue_span_match`: 19463
- `preserve_as_possible_phrase_block`: 7626
- `no_source_phrase`: 4767
- `compound_container_source_needs_internal_blocks`: 1482

## Phrase Examples

- `cookery writer` -> `BEATEN` [SYN] (span found) in `A French cookery writer reportedly never surpassed`
- `a french` -> `UN` [SYN] (span found) in `A French location around one lake without nasty development?`
- `a german` -> `EIN` [SYN] (span found) in `A German turning back to welcome city relative`
- `welsh girl` -> `SIAN` [SYN] (span found) in `A Welsh girl or Chinese?`
- `cut originally` -> `C` [ABR] (span found) in `A certain four sections cut originally in a shorter broadcast`
- `regularly visited` -> `ADE` [SYN] (span found) in `A construction of Naples paddler regularly visited ?`
- `couple of leaves` -> `GO` [SYN] (span found) in `A couple of leaves? Or as many as you like`
- `something shady` -> `HUE` [SYN] (span found) in `A criminal pursuit , something shady with opening of cocaine lines`
- `sermon maybe` -> `ORATION` [ABR] (span found) in `A dean’s original sermon maybe in worship`
- `south america` -> `MB` [SYN] (span found) in `A doctor in South America listening to this?`
- `for example` -> `EG` [ABR] (span found) in `A duck, for example, at no time turns this colour`
- `eastern religion` -> `INTO` [DEL] (span found) in `A fan of eastern religion wanting silence`
- `the bard` -> `SHAK` [DEL] (span found) in `A few characters from the Bard, old hat`
- `a few` -> `SOME` [ABR] (span found) in `A few men returned bearing British headgear`
- `a foreign` -> `UN` [SYN] (span found) in `A foreign writer is confronting text-changer lacking official status`
- `glass object` -> `JAR` [SYN] (span found) in `A glass object allowing some light in?`
- `european union` -> `EU` [SYN] (span found) in `A lady leading the European Union? See you in France!`
- `for each` -> `PER` [ABR] (span found) in `A little extra for each kilo`
- `a little light` -> `STAR` [SYN] (span found) in `A little light covering garden may in the end become inactive`
- `a little mischief` -> `IMP` [SYN] (span found) in `A little mischief by lassoer is unseemly`
- `a long way` -> `FAR` [ABR] (span found) in `A long way round a state for itinerant`
- `a lot` -> `FAR` [ABR] (span found) in `A lot coming with commercial storage unit`
- `us money` -> `BUCK` [SYN] (span found) in `A lot of US money resting on film juvenile collecting Oscar`
- `bad weather` -> `HAI` [DEL] (span found) in `A lot of bad weather has European country recalled in poem`

## Qualitative Review Seeds

- `in charge` appears as `IC` [SYN] (span found) in `A number of sailors in charge of a plant`
- `at home` appears as `IN` [SYN] (span found) in `Hard work at home needing clean after river comes in`

## Design Reading

The extractor deliberately keeps raw explanation text, parsed pieces,
candidate phrases, exact clue-span candidates, and objections together.
That is the preservation rule in executable form.

The important next review is qualitative: inspect the JSONL examples where
phrase blocks have no exact clue span, and where container sources need
internal blocks. Those are likely to teach Grammar Triage the most.
