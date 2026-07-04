# DT 31272 — fail analysis (2026-06-22)

24/32 pass. The 8 fails, each diagnosed grounded (solved through the real cascade +
checked against the DB). Category: DATA = reference-DB gap (enrichment, via queue/approval,
never a direct write); LOOKUP = a lookup bug (code); CAPABILITY = a missing engine ability.

1. **10075102 TRAIL** — CAPABILITY. `T`(end of accoun·t, last-letter selection) + `RAIL`
   (storyteller=LIAR, returning=reversed). Needs a charade combining a SEL_F selection
   piece with a REV_F reversed piece. charade_signature does SEL_F but not reversal;
   reversal_charade does REV_F but not selection. No engine combines them.

2. **10075104 DRESSES** — CAPABILITY. `TRESSES`(hair) with first letter (T) REPLACED by
   `D`(diamonds) = DRESSES. A positional (first-letter) substitution on a synonym base.
   substitution_engine replaces a letter but not at a located position of a synonym base.

3. **10075107 MINIM** — DATA. Palindrome ("that's the same either way"). def "Part of a
   score"→MINIM is in the DB; but `indicator_types("the same either way")` / "same either
   way" is empty — the palindrome indicator phrase is not typed. Enrichment: add the
   palindrome indicator.

4. **10075108 DOGFIGHT** — CAPABILITY. `D`(Day) + anagram of "fog hit" containing `G`
   (Germany's leader, "hiding"=container) = D + OGFIGHT. An anagram-container with a
   charaded abbreviation prefix; the anagram_container engine has no charade prefix.

5. **10075111 CALF** — DATA. Double definition: "Young animal"→CALF and "part of leg"→CALF
   both DB-confirmed; "shows" is the connector but `is_link("shows")` is False, so DD
   leaves it unaccounted and abstains. Enrichment: "shows" as a link/DD connector.

6. **10075116 BLATHER** — LOOKUP (bug). `B`(black) + `LATHER`(bubbles). The pair
   `('lather','bubbles')` IS in synonyms_pairs, but the table isn't symmetric and the
   bidirectional lookup only reverses ON A MISS; `bubbles` has forward synonyms, so the
   reverse (→ lather) never runs and LATHER is missed. FIX: always union forward+reverse
   in core/live_db.get_synonyms / get_synonyms_substring_of (NOCASE index makes it cheap).
   Systemic — affects every reverse-only synonym pair.

7. **10075119 SUNGLASSES** — DATA. `SUNG`(performed) + `LASSES`(girls) = SUNGLASSES, both
   found; but "some" is unaccounted (`is_link("some")` False). Enrichment: "some" as a link
   word, OR "some girls"→LASSES.

8. **10075125 Baghdad** — CAPABILITY/DATA (complex). `BAG`(claim) + `DAD`(father) found, but
   BAGHDAD needs the interior `H`. "father reported" suggests a homophone or an H source;
   needs closer parse. Park until the simpler ones are done.

## Also (from the all-engine DB-validity A/B, not DT)
- 2202543 ASS — honest fail: "missing" IS deletion-typed, but the recorded indicator span
  wasn't DB-backed-as-recorded; the old pass leaned on the location word "introduction"
  beheading (removed in the reclassify). Honest fail now.
- 10056226 SONAR — honest fail: recorded indicator "funny Batty" not a DB indicator
  (fabricated span). Correct.

## Build order (safest/highest-value first)
1. Bidirectional synonym lookup (always union) — LOOKUP, systemic. A/B.
2. Selection+reversal charade (TRAIL) — CAPABILITY, contained.
3. Positional substitution (DRESSES) — CAPABILITY.
4. DOGFIGHT / Baghdad — harder compounds, discuss.
DATA gaps (MINIM, CALF, SUNGLASSES) → enrichment queue / user approval, NOT direct writes.
