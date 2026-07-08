# Triage — Telegraph 31284

**Solved:** 7 pass · 4 pending · 21 fail  (32 clues)

**Why the 25 unsolved clues didn't solve** (a clue may have more than one reason): 17 missing data · 7 missing signature · 1 data quality · 1 unparsed · 1 missing engine

_Read-only diagnostic pass. Claude diagnoses and informs only — nothing here has been written to any database, queued, or re-solved. Suggested enrichments are for a human to accept on the dashboard._

## Summary

| Clue | Answer | Status | Near-miss engine | Diagnosis |
|------|--------|--------|------------------|-----------|
| 10 across | ISSUE | fail | charade | missing data |
| 11 across | ONSET | fail | charade | missing signature |
| 12 across | CONFERRER | pending | anagram | missing data |
| 13 across | TORTOISES | fail | charade | missing data |
| 16 across | ILIAD | fail | charade | missing data |
| 17 across | RIDES | fail | charade | missing signature, data quality |
| 18 across | CHAMPAGNE | fail | anagram | unparsed |
| 20 across | UNNOTICED | fail | anagram | missing data |
| 25 across | ARISE | pending | hidden | missing data |
| 26 across | OFFSPRING | fail | charade | missing data |
| 28 across | SQUARES | fail | anagram | missing data |
| 1 down | BEDPOST | fail | anagram | missing data |
| 2 down | OUSTS | fail | charade | missing data |
| 3 down | SOLUTIONS | fail | charade | missing signature |
| 4 down | MUSIC | fail | charade | missing engine |
| 5 down | INDONESIA | fail | charade | missing data |
| 6 down | PRIME | fail | anagram | missing data, missing signature |
| 7 down | OBSERVING | fail | charade | missing signature |
| 14 down | red ensign | fail | anagram | missing data |
| 15 down | SACKCLOTH | pending | charade_positional | missing data |
| 16 down | IMPROMPTU | fail | charade | missing signature |
| 19 down | ENRAGES | pending | anagram_insert_letter | missing data |
| 21 down | THETA | fail | charade | missing signature |
| 22 down | DOFFS | fail | anagram | missing data |
| 24 down | FLIER | fail | anagram | missing data |

## FAIL clues (21)

### 10 across — ISSUE — "Tear cover off paper magazine"

- status: **fail** · near-miss engine: **charade**
- near-miss detail: ["no charade signature matched this clue (best partial assembly shown \u2014 not a confirmed placement)", "answer letters not explained: I (position 1), S (position 2)", "these clue words are unaccounted for: 'Tear', 'cover', 'off'"]

```
clue:   'Tear cover off paper magazine'
answer: ISSUE  (ISSUE)

DEFINITION candidates (DB-confirmed defines -> answer):
   [end] 'magazine'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'Tear'  indicator:anagram
   'cover'  indicator:container/deletion/parts
   'off'  ->  OFF [literal]  indicator:anagram/deletion
   'paper'  indicator:container
```

**Diagnosis:** missing data
- **Gap:** TISSUE (paper) beheaded — 'cover off' removes the covering first letter → ISSUE; def = magazine. paper→TISSUE is ALREADY present; the only gap is a behead (first-letter deletion) indicator for 'cover off'.
- **Suggested enrichment:** `type=indicator` `word=cover off` `indicator_type=deletion` `subtype=first`
- _Note: The deletion engine handles TISSUE−T once 'cover off' is a first-letter deletion indicator._


### 11 across — ONSET — "Attack in a TV studio?"

- status: **fail** · near-miss engine: **charade**
- near-miss detail: ["no charade signature matched this clue (best partial assembly shown \u2014 not a confirmed placement)", "answer letters not explained: O (position 1), N (position 2)", "these clue words are unaccounted for: 'in', 'a', 'studio'"]

```
clue:   'Attack in a TV studio?'
answer: ONSET  (ONSET)

DEFINITION candidates (DB-confirmed defines -> answer):
   [start] 'Attack'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'Attack'  ->  ONSET [syn] in-answer
   'in'  ->  IN [literal]  indicator:container/hidden/insertion
   'a'  ->  ONE [syn] outer? (answer = value + insertion) | A [literal]
   'TV'  ->  SET [syn] in-answer
```

**Diagnosis:** missing signature
- **Gap:** ON + SET (SET = TV studio, present). 'in a TV studio?' idiomatically = 'on set' — the ON is carried by the whole phrase, not a single word. No per-word data path to ON.
- **Missing signature:** shape `charade where one piece comes from a phrase equivalence ('in a TV studio' = 'on set')` — tier **escalate** — witty semi-&lit; hand-solve or human decision — not a clean per-word lookup
- _Note: Def 'Attack'→ONSET already resolves._


### 13 across — TORTOISES — "Slow movers jog back alongside French river, starting to sweat"

- status: **fail** · near-miss engine: **charade**
- near-miss detail: ["no charade signature matched this clue (best partial assembly shown \u2014 not a confirmed placement)", "answer letters not explained: T (position 1), O (position 2), R (position 3), T (position 4)", "these clue words are unaccounted for: 'jog', 'back', 'alongside', 'French', 'to', 'sweat'"]

```
clue:   'Slow movers jog back alongside French river, starting to sweat'
answer: TORTOISES  (TORTOISES)

DEFINITION candidates (DB-confirmed defines -> answer):
   [start] 'Slow movers'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'jog'  ->  TROT [syn] in-answer(reversed)  indicator:anagram
   'back'  indicator:deletion/parts/reversal
   'river'  ->  R [abbr] in-answer
   'starting'  ->  S [abbr] in-answer  indicator:acrostic/parts
   'to'  ->  TO [literal] in-answer
   'Slow movers'  ->  TORTOISE [syn] in-answer
```

**Diagnosis:** missing data
- **Gap:** reverse(jog = TROT) = TORT + OISE (French river) + S (starting to sweat) → TORTOISES; def = Slow movers. LOIRE is present as a French river, OISE is not.
- **Suggested enrichment:** `type=synonym` `word=French river` `value=OISE` `answer=TORTOISES`
- _Note: reversal_charade assembles once OISE is present._


### 16 across — ILIAD — "Current elected house rejected epic poem"

- status: **fail** · near-miss engine: **charade**
- near-miss detail: ["no charade signature matched this clue (best partial assembly shown \u2014 not a confirmed placement)", "answer letters not explained: L (position 2), I (position 3), A (position 4), D (position 5)", "these clue words are unaccounted for: 'elected', 'house', 'rejected'"]

```
clue:   'Current elected house rejected epic poem'
answer: ILIAD  (ILIAD)

DEFINITION candidates (DB-confirmed defines -> answer):
   [end] 'poem'  <- CONFIRMED
   [end] 'epic poem'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'Current'  ->  I [abbr] in-answer
   'house'  indicator:container/hidden/parts
   'rejected'  indicator:deletion/reversal
   'epic'  ->  ILIAD [syn] in-answer
   'poem'  ->  ILIAD [syn] in-answer
   'epic poem'  ->  ILIAD [syn] in-answer
```

**Diagnosis:** missing data
- **Gap:** I (Current) + reverse(DAIL = elected house) → ILIAD; def = epic poem. 'elected house'→DAIL (the Dáil, Irish lower house) is absent.
- **Suggested enrichment:** `type=synonym` `word=elected house` `value=DAIL` `answer=ILIAD`
- _Note: reversal_charade: I + rev(DAIL)._


### 17 across — RIDES — "Journeys using free diesel essentially"

- status: **fail** · near-miss engine: **charade**
- near-miss detail: ["these clue words are unaccounted for: 'using'"]

```
clue:   'Journeys using free diesel essentially'
answer: RIDES  (RIDES)

DEFINITION candidates (DB-confirmed defines -> answer):
   [start] 'Journeys'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'Journeys'  ->  RIDES [syn] in-answer | RIDE [syn] in-answer
   'using'  indicator:anagram
   'free'  ->  RID [syn] in-answer  indicator:anagram/deletion
   'essentially'  ->  SE [syn] in-answer(reversed)  indicator:deletion/parts/selection
```

**Diagnosis:** missing signature, data quality
- **Gap:** RID (free) + ES (diesel essentially) → RIDES; def = Journeys; 'using' is a joiner. Blocked because 'using' is registered as an ANAGRAM indicator and is left unaccounted.
- **Missing signature:** shape `charade RID+ES with an unindexed joiner ('using')` — tier **escalate** — no DB link words by policy (they overlap with indicators), so 'using' must be tolerated as filler by a signature, or its dubious anagram-indicator row reviewed
- _Note: Also review the 'using' = anagram-indicator row — it misfires as a joiner here._


### 18 across — CHAMPAGNE — "Fake glass delivered for celebratory drink"

- status: **fail** · near-miss engine: **anagram**
- near-miss detail: ["no anagram signature matched this clue (the fodder below is a candidate, not a placement)"]

```
clue:   'Fake glass delivered for celebratory drink'
answer: CHAMPAGNE  (CHAMPAGNE)

DEFINITION candidates (DB-confirmed defines -> answer):
   [end] 'drink'  <- CONFIRMED
   [end] 'celebratory drink'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'Fake'  ->  MAN [syn] outer? (answer = value + insertion)  indicator:anagram
   'delivered'  indicator:hidden
   'for'  ->  FOR [literal]
   'drink'  indicator:anagram/container
   'celebratory drink'  ->  CHAMPAGNE [syn] in-answer
```

**Diagnosis:** unparsed
- **Gap:** def = celebratory drink→CHAMPAGNE (present). The wordplay device is not mechanically clear from the evidence.
- _Note: Honest 'cannot parse wordplay' — flag for a human parse rather than guess._


### 20 across — UNNOTICED — "Criminal continued under the radar"

- status: **fail** · near-miss engine: **anagram**
- near-miss detail: ["no anagram signature matched this clue (the fodder below is a candidate, not a placement)"]

```
clue:   'Criminal continued under the radar'
answer: UNNOTICED  (UNNOTICED)

DEFINITION candidates (DB-confirmed defines -> answer):
   (none — NO confirmed definition; this alone blocks every engine)

PIECE material (value [mechanism] flag), per clue run:
   'Criminal'  ->  C [abbr] in-answer  indicator:anagram/definition
   'under'  indicator:charade_positional
   'the'  ->  THE [literal]
   'radar'  ->  DE [syn] in-answer(reversed)
```

**Diagnosis:** missing data
- **Gap:** anagram of 'continued' (indicator 'Criminal') → UNNOTICED; def = 'under the radar'. The ONLY blocker is the missing definition.
- **Suggested enrichment:** `type=definition` `definition=under the radar` `answer=UNNOTICED`
- _Note: The anagram engine solves once the definition is present._


### 26 across — OFFSPRING — "Children not working well"

- status: **fail** · near-miss engine: **charade**
- near-miss detail: ["these clue words are unaccounted for: 'working'"]

```
clue:   'Children not working well'
answer: OFFSPRING  (OFFSPRING)

DEFINITION candidates (DB-confirmed defines -> answer):
   [start] 'Children'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'Children'  ->  OFFSPRING [syn] in-answer
   'not'  ->  OFF [syn] in-answer | NOT [literal]  indicator:deletion
   'working'  ->  ON [syn] outer? (answer = value + insertion)  indicator:anagram
```

**Diagnosis:** missing data
- **Gap:** OFF (not working) + SPRING (well) → OFFSPRING; def = Children. well→SPRING is already present; only 'not working'→OFF is missing.
- **Suggested enrichment:** `type=synonym` `word=not working` `value=OFF` `answer=OFFSPRING`
- _Note: charade assembles once OFF is present._


### 28 across — SQUARES — "Unfashionable people's particular powers"

- status: **fail** · near-miss engine: **anagram**
- near-miss detail: ["no anagram signature matched this clue (the fodder below is a candidate, not a placement)"]

```
clue:   "Unfashionable people's particular powers"
answer: SQUARES  (SQUARES)

DEFINITION candidates (DB-confirmed defines -> answer):
   [start] 'Unfashionable'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'Unfashionable'  ->  SQUARE [syn] in-answer
   'particular'  indicator:anagram
```

**Diagnosis:** missing data
- **Gap:** double definition: 'Unfashionable people' = SQUARES and 'particular powers' = SQUARES (x-squared). Both full-phrase definitions are absent.
- **Suggested enrichment:** `type=definition` `definition=Unfashionable people` `answer=SQUARES`
- **Suggested enrichment:** `type=definition` `definition=particular powers` `answer=SQUARES`
- _Note: DD engine solves with both definitions._


### 1 down — BEDPOST — "Teaching graduate with job gets support for retirement"

- status: **fail** · near-miss engine: **anagram**
- near-miss detail: ["no anagram signature matched this clue (the fodder below is a candidate, not a placement)"]

```
clue:   'Teaching graduate with job gets support for retirement'
answer: BEDPOST  (BEDPOST)

DEFINITION candidates (DB-confirmed defines -> answer):
   (none — NO confirmed definition; this alone blocks every engine)

PIECE material (value [mechanism] flag), per clue run:
   'gets'  indicator:substitution
   'support'  indicator:charade_positional/parts
   'for'  ->  FOR [literal]
   'retirement'  indicator:reversal
```

**Diagnosis:** missing data
- **Gap:** BED (teaching graduate = B.Ed.) + POST (job) → BEDPOST; def = 'support for retirement' (a bedpost). Needs teaching graduate→BED and the definition.
- **Suggested enrichment:** `type=substitution` `word=teaching graduate` `value=BED` `answer=BEDPOST`
- **Suggested enrichment:** `type=definition` `definition=support for retirement` `answer=BEDPOST`
- _Note: job→POST should be present; charade._


### 2 down — OUSTS — "Expels from old university thoroughfares"

- status: **fail** · near-miss engine: **charade**
- near-miss detail: ["no charade signature matched this clue (best partial assembly shown \u2014 not a confirmed placement)", "answer letters not explained: S (position 3), T (position 4), S (position 5)", "these clue words are unaccounted for: 'from', 'thoroughfares'"]

```
clue:   'Expels from old university thoroughfares'
answer: OUSTS  (OUSTS)

DEFINITION candidates (DB-confirmed defines -> answer):
   [start] 'Expels'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'from'  indicator:anagram/hidden/reversal
   'old'  ->  O [abbr] in-answer
   'university'  ->  U [abbr] in-answer
```

**Diagnosis:** missing data
- **Gap:** O (old) + U (university) + STS (thoroughfares) → OUSTS; def = Expels. thoroughfares→STS (streets) is absent.
- **Suggested enrichment:** `type=substitution` `word=thoroughfares` `value=STS` `answer=OUSTS`
- _Note: charade O+U+STS._


### 3 down — SOLUTIONS — "Explanations lost on us unfortunately, first person overwhelmed"

- status: **fail** · near-miss engine: **charade**
- near-miss detail: ["no charade signature matched this clue (best partial assembly shown \u2014 not a confirmed placement)", "answer letters not explained: L (position 3), U (position 4), T (position 5), O (position 7), N (position 8), S (position 9)", "these clue words are unaccounted for: 'lost', 'us', 'unfortunately', 'overwhelmed'"]

```
clue:   'Explanations lost on us unfortunately, first person overwhelmed'
answer: SOLUTIONS  (SOLUTIONS)

DEFINITION candidates (DB-confirmed defines -> answer):
   [start] 'Explanations'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'Explanations'  ->  SOLUTIONS [syn] in-answer | SOLUTION [syn] in-answer  indicator:homophone
   'lost'  indicator:anagram/deletion
   'on'  ->  ON [literal] in-answer  indicator:charade_positional/container
   'us'  ->  US [abbr] outer? (answer = value + insertion) | US [literal] outer? (answer = value + insertion)
   'unfortunately'  indicator:anagram
   'first'  indicator:acrostic/parts
   'person'  ->  SON [syn] outer? (answer = value + insertion)  indicator:anagram
   'first person'  ->  I [syn] in-answer
```

**Diagnosis:** missing signature
- **Gap:** anagram of ('lost' on us) + I (first person), indicators 'unfortunately'/'overwhelmed' → SOLUTIONS; def = Explanations. Multi-word literal fodder plus one looked-up inserted letter.
- **Missing signature:** shape `anagram(multi-word literal fodder + a synonym-letter, e.g. first person→I)` — tier **pending (tentative — L≈1, borderline)** — the anagram_insert_letter family exists (see 19d); the gap is identifying the scattered fodder
- _Note: Human confirms the shape and tier._


### 4 down — MUSIC — "Leaders of most unions stand in corridors singing, perhaps"

- status: **fail** · near-miss engine: **charade**
- near-miss detail: ["no charade signature matched this clue (best partial assembly shown \u2014 not a confirmed placement)", "answer letters not explained: M (position 1), S (position 3), I (position 4), C (position 5)", "these clue words are unaccounted for: 'Leaders', 'of', 'most', 'stand', 'in', 'corridors', 'singing'"]

```
clue:   'Leaders of most unions stand in corridors singing, perhaps'
answer: MUSIC  (MUSIC)

DEFINITION candidates (DB-confirmed defines -> answer):
   [end] 'perhaps'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'Leaders'  indicator:acrostic/parts
   'of'  ->  OF [literal]
   'most'  indicator:deletion/parts
   'unions'  ->  U [abbr] in-answer
   'in'  ->  IN [literal]  indicator:container/hidden/insertion
   'perhaps'  ->  MUSIC [syn] in-answer  indicator:definition by example/parts
```

**Diagnosis:** missing engine
- **Gap:** acrostic: initial letters of 'Most Unions Stand In Corridors' = MUSIC; 'Leaders of' = acrostic indicator; def = 'singing, perhaps' (definition by example). Needs an initial-letters acrostic solver.
- **Missing signature:** shape `initial-letters acrostic across N words` — tier **escalate** — acrostic engine/family
- _Note: Escalate — engine gap. Also DBE def perhaps→MUSIC._


### 5 down — INDONESIA — "Subcontinent surrounding one's country"

- status: **fail** · near-miss engine: **charade**
- near-miss detail: ["no charade signature matched this clue (best partial assembly shown \u2014 not a confirmed placement)", "answer letters not explained: N (position 2), D (position 3), O (position 4), N (position 5), E (position 6), S (position 7), I (position 8), A (position 9)", "these clue words are unaccounted for: 'Subcontinent', 'surrounding'"]

```
clue:   "Subcontinent surrounding one's country"
answer: INDONESIA  (INDONESIA)

DEFINITION candidates (DB-confirmed defines -> answer):
   [end] 'country'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'Subcontinent'  ->  INDIA [syn] outer? (answer = value + insertion)
   'surrounding'  indicator:container
   "one's"  ->  IS [syn] in-answer(reversed) | I [abbr] in-answer | A [abbr] in-answer | O [abbr] in-answer
```

**Diagnosis:** missing data
- **Gap:** INDIA (Subcontinent) containing ONES (one's) → IND(ONES)IA; def = country. INDIA present; the inner fodder one's→ONES is absent.
- **Suggested enrichment:** `type=synonym` `word=one's` `value=ONES` `answer=INDONESIA`
- _Note: container engine; INDIA around ONES._


### 6 down — PRIME — "Get ready for 11 or 13, say"

- status: **fail** · near-miss engine: **anagram**
- near-miss detail: ["no anagram signature matched this clue (the fodder below is a candidate, not a placement)"]

```
clue:   'Get ready for 11 or 13, say'
answer: PRIME  (PRIME)

DEFINITION candidates (DB-confirmed defines -> answer):
   (none — NO confirmed definition; this alone blocks every engine)

PIECE material (value [mechanism] flag), per clue run:
   'Get'  indicator:substitution
   'for'  ->  FOR [literal]
   'or'  ->  OR [literal]
   'say'  indicator:anagram/definition by example/homophone
```

**Diagnosis:** missing data, missing signature
- **Gap:** double definition: 'Get ready' = PRIME (verb) and '11 or 13, say' = PRIME (a prime number, by example). Needs get ready→PRIME; the numeric DBE side cannot be a DB entry.
- **Suggested enrichment:** `type=synonym` `word=Get ready` `value=PRIME` `answer=PRIME`
- **Missing signature:** shape `DD where one side is a numeric definition-by-example ('11 or 13'→prime)` — tier **pending** — the solver cannot verify the numeric example → verdict pending
- _Note: Half is data (get ready→PRIME), half is an unverifiable numeric DBE._


### 7 down — OBSERVING — "Watching former pupil starting a tennis match"

- status: **fail** · near-miss engine: **charade**
- near-miss detail: ["no charade signature matched this clue (best partial assembly shown \u2014 not a confirmed placement)", "answer letters not explained: E (position 4), R (position 5), V (position 6), I (position 7), N (position 8), G (position 9)", "these clue words are unaccounted for: 'a', 'tennis', 'match'"]

```
clue:   'Watching former pupil starting a tennis match'
answer: OBSERVING  (OBSERVING)

DEFINITION candidates (DB-confirmed defines -> answer):
   [start] 'Watching'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'Watching'  ->  OBSERVING [syn] in-answer
   'starting'  ->  S [abbr] in-answer  indicator:acrostic/parts
   'a'  ->  A [literal]
   'former pupil'  ->  OB [syn] in-answer
```

**Diagnosis:** missing signature
- **Gap:** OB (former pupil = Old Boy) + SERVING → OBSERVING; def = Watching. OB present; SERVING is clued by the whole phrase 'starting a tennis match' (the serve starts play) — a piece defined by a phrase, not a DB lookup.
- **Missing signature:** shape `charade OB + a piece defined by a full phrase` — tier **escalate** — no clean per-word data path to SERVING; hand-solve or human
- _Note: Watching→OBSERVING (def) resolves._


### 14 down — red ensign — "Flag needs ring put around"

- status: **fail** · near-miss engine: **anagram**
- near-miss detail: ["indicator 'put around' is not a DB anagram indicator"]

```
clue:   'Flag needs ring put around'
answer: red ensign  (REDENSIGN)

DEFINITION candidates (DB-confirmed defines -> answer):
   [start] 'Flag'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'Flag'  ->  SIGN [syn] in-answer | RED ENSIGN [syn] in-answer
   'needs'  indicator:anagram
   'ring'  ->  R [abbr] in-answer
   'around'  ->  IN [syn] outer? (answer = value + insertion)  indicator:anagram/container/reversal
```

**Diagnosis:** missing data
- **Gap:** anagram of 'needs ring' (indicator 'put around') → RED ENSIGN; def = Flag. The ONLY blocker: 'put around' is not a DB anagram indicator.
- **Suggested enrichment:** `type=indicator` `word=put around` `indicator_type=anagram`
- _Note: The anagram engine solves once the indicator is present._


### 16 down — IMPROMPTU — "Without preparation, I'm on time, you heard"

- status: **fail** · near-miss engine: **charade**
- near-miss detail: ["these clue words are unaccounted for: 'heard'"]

```
clue:   "Without preparation, I'm on time, you heard"
answer: IMPROMPTU  (IMPROMPTU)

DEFINITION candidates (DB-confirmed defines -> answer):
   [start] 'Without preparation'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'Without'  indicator:deletion
   'preparation'  indicator:anagram
   "I'm"  ->  IM [syn] in-answer
   'on'  ->  ON [literal]  indicator:charade_positional/container
   'time'  ->  T [abbr] in-answer  indicator:anagram
   'you'  ->  U [syn] in-answer | U [abbr] in-answer
   'heard'  indicator:homophone
   'Without preparation'  ->  IMPROMPTU [syn] in-answer
   'on time'  ->  PROMPT [syn] in-answer
```

**Diagnosis:** missing signature
- **Gap:** IM (I'm) + PROMPT (on time) + U (you, heard) → IMPROMPTU; def = 'Without preparation'. All pieces present; U is a homophone ('you' heard = the letter U) and 'heard' is its indicator. The charade engine cannot absorb a homophone-indicated piece.
- **Missing signature:** shape `charade including a homophone-letter piece + its 'heard' indicator` — tier **pass (tentative — deterministic homophone, G=false)** — or a small homophone-in-charade engine
- _Note: Data all present (IM, PROMPT, U); this is a shape/engine gap._


### 21 down — THETA — "Time husband at last wrote thank you letter"

- status: **fail** · near-miss engine: **charade**
- near-miss detail: ["no charade signature matched this clue (best partial assembly shown \u2014 not a confirmed placement)", "answer letters not explained: E (position 3)", "these clue words are unaccounted for: 'at', 'last', 'wrote'"]

```
clue:   'Time husband at last wrote thank you letter'
answer: THETA  (THETA)

DEFINITION candidates (DB-confirmed defines -> answer):
   [end] 'letter'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'Time'  ->  T [abbr] in-answer  indicator:anagram
   'husband'  ->  H [abbr] in-answer
   'at'  ->  AT [literal] in-answer(reversed)
   'last'  indicator:deletion/parts
   'thank'  ->  TA [abbr] in-answer
   'letter'  ->  ETA [syn] in-answer  indicator:anagram
   'at last'  indicator:acrostic/parts/selection
   'thank you'  ->  TA [syn] in-answer
```

**Diagnosis:** missing signature
- **Gap:** T (Time) + H (husband) + E (last letter of 'wrote', via 'at last') + TA (thank you) → THETA; def = letter. 'at last' = last-letter selection is present; the gap is a 4-piece charade that includes a last-letter selection piece.
- **Missing signature:** shape `charade: abbr + abbr + [last-letter selection] + abbr` — tier **pass (tentative — deterministic selection, G=false, L low)**
- _Note: Human confirms shape/tier._


### 22 down — DOFFS — "Removes police officer restraining a third of offenders"

- status: **fail** · near-miss engine: **anagram**
- near-miss detail: ["no anagram signature matched this clue (the fodder below is a candidate, not a placement)"]

```
clue:   'Removes police officer restraining a third of offenders'
answer: DOFFS  (DOFFS)

DEFINITION candidates (DB-confirmed defines -> answer):
   (none — NO confirmed definition; this alone blocks every engine)

PIECE material (value [mechanism] flag), per clue run:
   'Removes'  indicator:deletion
   'officer'  ->  O [abbr] in-answer
   'restraining'  indicator:container
   'a'  ->  A [literal]
   'of'  ->  OF [literal] in-answer
```

**Diagnosis:** missing data
- **Gap:** DS (police officer = Detective Sergeant) containing OFF (a third of 'offenders') → D(OFF)S → DOFFS; def = Removes. Needs police officer→DS (only DI present) and definition Removes→DOFFS.
- **Suggested enrichment:** `type=substitution` `word=police officer` `value=DS` `answer=DOFFS`
- **Suggested enrichment:** `type=definition` `definition=Removes` `answer=DOFFS`
- _Note: container engine; 'a third of offenders' = OFF is a first-third selection._


### 24 down — FLIER — "Martin's leaflet?"

- status: **fail** · near-miss engine: **anagram**
- near-miss detail: ["no anagram signature matched this clue (the fodder below is a candidate, not a placement)"]

```
clue:   "Martin's leaflet?"
answer: FLIER  (FLIER)

DEFINITION candidates (DB-confirmed defines -> answer):
   [end] 'leaflet'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'leaflet'  ->  FLIER [syn] in-answer
```

**Diagnosis:** missing data
- **Gap:** double definition: 'Martin' (a martin is a flier/bird, by example) and 'leaflet' = FLIER (present). Needs Martin→FLIER.
- **Suggested enrichment:** `type=synonym` `word=Martin` `value=FLIER` `answer=FLIER`
- _Note: DD/DBE; leaflet→FLIER already present._


## PENDING clues (4)

### 12 across — CONFERRER — "Giver of dubious corner – ref?"

- status: **pending** · near-miss engine: **anagram**
- near-miss detail: ["the definition is provisional (queued for enrichment)"]

```
clue:   'Giver of dubious corner – ref?'
answer: CONFERRER  (CONFERRER)

DEFINITION candidates (DB-confirmed defines -> answer):
   (none — NO confirmed definition; this alone blocks every engine)

PIECE material (value [mechanism] flag), per clue run:
   'of'  ->  ON [syn] in-answer | OF [literal] outer? (answer = value + insertion)
   'dubious'  indicator:anagram
```

**Diagnosis:** missing data
- **Gap:** anagram of 'corner' + 'ref' (indicator 'dubious') → CONFERRER; def = Giver (provisional). Confirm the definition Giver→CONFERRER.
- **Suggested enrichment:** `type=definition` `definition=Giver` `answer=CONFERRER`
- _Note: Pending ONLY for the provisional definition; the anagram already matched._


### 25 across — ARISE — "Some plagiarisers stand up"

- status: **pending** · near-miss engine: **hidden**
- near-miss detail: ["these clue words are unaccounted for: 'stand'"]

```
clue:   'Some plagiarisers stand up'
answer: ARISE  (ARISE)

DEFINITION candidates (DB-confirmed defines -> answer):
   [end] 'up'  <- CONFIRMED

PIECE material (value [mechanism] flag), per clue run:
   'Some'  ->  S [abbr] in-answer  indicator:hidden
   'up'  ->  UP [literal]  indicator:reversal
```

**Diagnosis:** missing data
- **Gap:** hidden in 'plagiARISErs' ('Some' = hidden indicator); def = 'stand up'. The present 'stand up' definitions lack ARISE.
- **Suggested enrichment:** `type=definition` `definition=stand up` `answer=ARISE`
- _Note: The hidden match is fine; 'stand' stays unaccounted until the 2-word definition is confirmed._


### 15 down — SACKCLOTH — "Dismiss fool over hot itchy fabric"

- status: **pending** · near-miss engine: **charade_positional**
- near-miss detail: ["the definition is provisional (queued for enrichment)"]

```
clue:   'Dismiss fool over hot itchy fabric'
answer: SACKCLOTH  (SACKCLOTH)

DEFINITION candidates (DB-confirmed defines -> answer):
   (none — NO confirmed definition; this alone blocks every engine)

PIECE material (value [mechanism] flag), per clue run:
   'Dismiss'  ->  SACK [syn] in-answer
   'fool'  ->  CLOT [syn] in-answer
   'over'  ->  O [abbr] in-answer  indicator:anagram/charade_positional/reversal
   'hot'  ->  H [abbr] in-answer
   'itchy'  indicator:anagram
   'fabric'  ->  CLOTH [syn] in-answer
```

**Diagnosis:** missing data
- **Gap:** SACK (Dismiss) + CLOT (fool) + H (hot), 'over' = positional → SACKCLOTH; def = 'itchy fabric'. All pieces already resolve; only the definition is missing.
- **Suggested enrichment:** `type=definition` `definition=itchy fabric` `answer=SACKCLOTH`
- _Note: charade_positional matched; pending only for the definition._


### 19 down — ENRAGES — "A genre novel with lead character in Sussex infuriates"

- status: **pending** · near-miss engine: **anagram_insert_letter**
- near-miss detail: ["the definition is provisional (queued for enrichment)"]

```
clue:   'A genre novel with lead character in Sussex infuriates'
answer: ENRAGES  (ENRAGES)

DEFINITION candidates (DB-confirmed defines -> answer):
   (none — NO confirmed definition; this alone blocks every engine)

PIECE material (value [mechanism] flag), per clue run:
   'A'  ->  A [abbr] in-answer | A [literal] in-answer
   'novel'  indicator:anagram
   'lead'  indicator:acrostic/parts
   'character'  indicator:acrostic/anagram/hidden
   'in'  ->  IN [literal]  indicator:container/hidden/insertion
```

**Diagnosis:** missing data
- **Gap:** anagram of 'A genre' + S (lead character of Sussex), 'novel' indicator → ENRAGES; def = infuriates. Present 'infuriates' definitions are IRRITATES/VEXES, not ENRAGES.
- **Suggested enrichment:** `type=definition` `definition=infuriates` `answer=ENRAGES`
- _Note: anagram_insert_letter matched; pending only for the definition._

