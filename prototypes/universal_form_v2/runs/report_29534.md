# Times 29534 — Form Pipeline Report

**Run date**: 2026-05-05  
**Total clues**: 28  
**Pass rate**: 14/28 = 50.0%  
**FAIL**: 2  
**NO_FORM**: 12  

## PASS (14)

### 1a. LIVERWORT

**Clue**: On air row traumatised and exhausted Robert Plant

**Blog**: LIVE (on air), anagram [traumatised] of ROW , then R {ober} T [exhausted]. Apparently Robert Plant was the lead singer of the rock band Led Zeppelin.

**Definition (DB)**: `Plant`

**Pattern**: `anagram`

**Form**: `anagram [traumatised](synonym(LIVE from 'On air'), literal(ROW from 'row'), positional(RT from 'exhausted Robert'))`

**Pieces**: LIVE from `On air`; ROW from `row`; RT from `exhausted Robert`

**Indicators**: `traumatised` [anagram]

- ✓ `assembly`: tree produces LIVERWORT
- ✓ `bridge.leaves`: verified "On air"->LIVE; literal "row"; outer of "exhausted Robert"=RT
- ✓ `bridge.indicators`: verified "traumatised" -> anagram
- ✓ `residue`: 

---

### 6a. SHARP

**Clue**: Piercing, principally shrill instrument

**Blog**: S {hrill} [principally], HARP (instrument)

**Definition (DB)**: `Piercing`

**Pattern**: `charade`

**Form**: `charade(literal(S from 'principally shrill'), synonym(HARP from 'instrument'))`

**Pieces**: S from `principally shrill`; HARP from `instrument`

- ✓ `assembly`: tree produces SHARP
- ✓ `bridge.leaves`: verified literal "principally shrill"; "instrument"->HARP
- ✓ `bridge.indicators`: no indicator-bearing nodes
- ✓ `residue`: 

---

### 10a. MILDEW

**Clue**: Disease from beer we knocked back

**Blog**: MILD (beer), then WE reversed [knocked back]. A plant disease caused by fungus.

**Definition (DB)**: `Disease`

**Pattern**: `reversal_charade`

**Form**: `charade(synonym(MILD from 'beer'), reversal [knocked back](literal(WE from 'we')))`

**Pieces**: MILD from `beer`; WE from `we`

**Indicators**: `knocked back` [reversal]

- ✓ `assembly`: tree produces MILDEW
- ✓ `bridge.leaves`: verified "beer"->MILD; literal "we"
- ✓ `bridge.indicators`: verified "knocked back" -> reversal
- ✓ `residue`: 

---

### 14a. FELL

**Clue**: Knock down chap emitting cry of pain

**Blog**: FELL {ow} (chap) [emitting cry of pain]

**Definition (DB)**: `Knock down`

**Pattern**: `single_piece`

**Form**: `synonym(FELL from 'chap emitting cry of pain')`

**Pieces**: FELL from `chap emitting cry of pain`

- ✓ `assembly`: tree produces FELL
- ✓ `bridge.leaves`: verified "chap emitting cry of pain"->FELL
- ✓ `bridge.indicators`: no indicator-bearing nodes
- ✓ `residue`: 

---

### 17a. PENTAMETER

**Clue**: Write uninspiring incomplete ending for literary work

**Blog**: PEN (write), TAME (uninspiring), TER {m} (ending) [incomplete]

**Definition (DB)**: `literary work`

**Pattern**: `charade`

**Form**: `charade(synonym(PEN from 'Write'), synonym(TAME from 'uninspiring'), synonym(TER from 'incomplete ending'))`

**Pieces**: PEN from `Write`; TAME from `uninspiring`; TER from `incomplete ending`

- ✓ `assembly`: tree produces PENTAMETER
- ✓ `bridge.leaves`: verified "Write"->PEN; "uninspiring"->TAME; "incomplete ending"->TER
- ✓ `bridge.indicators`: no indicator-bearing nodes
- ✓ `residue`: 

---

### 19a. SICILIAN

**Clue**: Italian snail unexpectedly seen around here in France

**Blog**: Anagram [unexpectedly] of SNAIL containing [seen around] ICI (‘here’ in France)

**Definition (DB)**: `Italian`

**Pattern**: `container_anagram`

**Form**: `container [seen around](anagram [unexpectedly](literal(SNAIL from 'snail')), synonym(ICI from 'here in France'))`

**Pieces**: SNAIL from `snail`; ICI from `here in France`

**Indicators**: `unexpectedly` [anagram]; `seen around` [container]

- ✓ `assembly`: tree produces SICILIAN
- ✓ `bridge.leaves`: verified literal "snail"; "here in France"->ICI
- ✓ `bridge.indicators`: verified "seen around" -> container; "unexpectedly" -> anagram
- ✓ `residue`: 

---

### 24a. EMMET

**Clue**: This compiler heading west bumped into tourist

**Blog**: ME (this compiler) reversed [heading west], MET (bumped into). Cornish slang for a holidaymaker. In Devon they’re called Grockles.

**Definition (DB)**: `tourist`

**Pattern**: `reversal_charade`

**Form**: `charade(reversal [heading west](abbreviation(ME from 'This compiler')), synonym(MET from 'bumped into'))`

**Pieces**: ME from `This compiler`; MET from `bumped into`

**Indicators**: `heading west` [reversal]

- ✓ `assembly`: tree produces EMMET
- ✓ `bridge.leaves`: verified "This compiler"->ME (abbr); "bumped into"->MET
- ✓ `bridge.indicators`: verified "heading west" -> reversal
- ✓ `residue`: 

---

### 1d. LOCUM

**Clue**: Substitute officer upset uniform men at the start

**Blog**: COL (officer – Colonel ) reversed [upset], U (uniform), M {en} [at the start]

**Definition (DB)**: `Substitute`

**Pattern**: `reversal_charade`

**Form**: `charade(reversal [upset](synonym(COL from 'officer')), abbreviation(U from 'uniform'), literal(M from 'men at the start'))`

**Pieces**: COL from `officer`; U from `uniform`; M from `men at the start`

**Indicators**: `upset` [reversal]

- ✓ `assembly`: tree produces LOCUM
- ✓ `bridge.leaves`: verified "officer"->COL; "uniform"->U (abbr); literal "men at the start"
- ✓ `bridge.indicators`: verified "upset" -> reversal
- ✓ `residue`: 

---

### 2d. VANILLAICECREAM

**Clue**: Sweet eclair Calvin ordered each month

**Blog**: Anagram [ordered] of ECLAIR CALVIN , then EA (each), M (month)

**Definition (DB)**: `Sweet`

**Pattern**: `anagram`

**Form**: `anagram [ordered](literal(ECLAIRCALVIN from 'eclair Calvin'), abbreviation(EA from 'each'), abbreviation(M from 'month'))`

**Pieces**: ECLAIRCALVIN from `eclair Calvin`; EA from `each`; M from `month`

**Indicators**: `ordered` [anagram]

- ✓ `assembly`: tree produces VANILLAICECREAM
- ✓ `bridge.leaves`: verified literal "eclair Calvin"; "each"->EA (abbr); "month"->M (abbr)
- ✓ `bridge.indicators`: verified "ordered" -> anagram
- ✓ `residue`: 

---

### 4d. OOPS

**Clue**: Served up spirit, just missing king … sorry!

**Blog**: SPOO {k} (spirit) reversed [served up] [just missing king]

**Definition (DB)**: `sorry`

**Pattern**: `reversal`

**Form**: `reversal [Served up](synonym(SPOO from 'spirit just missing king'))`

**Pieces**: SPOO from `spirit just missing king`

**Indicators**: `Served up` [reversal]

- ✓ `assembly`: tree produces OOPS
- ✓ `bridge.leaves`: verified "spirit just missing king"->SPOO
- ✓ `bridge.indicators`: verified "Served up" -> reversal
- ✓ `residue`: 

---

### 5d. TROGLODYTE

**Clue**: Grotty old vagrant last seen in cave?

**Blog**: Anagram [vagrant] of GROTTY OLD , then {cav} E [last seen in…]. A troglodyte was a cave-dweller so I guess the whole clue has to be the definition here.

**Definition (DB)**: `Grotty old vagrant last seen in cave?`

**Pattern**: `anagram`

**Form**: `anagram [vagrant](literal(GROTTYOLD from 'Grotty old'), literal(E from 'last seen in cave'))`

**Pieces**: GROTTYOLD from `Grotty old`; E from `last seen in cave`

**Indicators**: `vagrant` [anagram]

- ✓ `assembly`: tree produces TROGLODYTE
- ✓ `bridge.leaves`: verified literal "Grotty old"; literal "last seen in cave"
- ✓ `bridge.indicators`: verified "vagrant" -> anagram
- ✓ `residue`: 

---

### 12d. GAMEWARDEN

**Clue**: Plucky wife, endlessly zealous officer in reserve

**Blog**: GAME (plucky), W (wife), ARDEN {t} (zealous) [endlessly]

**Definition (DB)**: `officer in reserve`

**Pattern**: `charade`

**Form**: `charade(synonym(GAME from 'Plucky'), abbreviation(W from 'wife'), synonym(ARDEN from 'endlessly zealous'))`

**Pieces**: GAME from `Plucky`; W from `wife`; ARDEN from `endlessly zealous`

- ✓ `assembly`: tree produces GAMEWARDEN
- ✓ `bridge.leaves`: verified "Plucky"->GAME; "wife"->W (abbr); "endlessly zealous"->ARDEN
- ✓ `bridge.indicators`: no indicator-bearing nodes
- ✓ `residue`: 

---

### 13d. GLADSTONE

**Clue**: Young man’s backing good quality statesman

**Blog**: G (good), LAD’S (young man’s), TONE (quality)

**Definition (DB)**: `statesman`

**Pattern**: `charade`

**Form**: `charade(abbreviation(G from 'backing good'), synonym(LAD from 'Young man'), literal(S from 's'), synonym(TONE from 'quality'))`

**Pieces**: LAD from `Young man`; S from `s`; G from `backing good`; TONE from `quality`

- ✓ `assembly`: tree produces GLADSTONE
- ✓ `bridge.leaves`: verified "backing good"->G (abbr); "Young man"->LAD; literal "s"; "quality"->TONE
- ✓ `bridge.indicators`: no indicator-bearing nodes
- ✓ `residue`: 

---

### 21d. ELEGY

**Clue**: Lament English cricket side, the ultimate in ignominy

**Blog**: E (English), LEG (cricket side – as opposed to ‘off’ ), {ignomin} Y [the ultimate in…]

**Definition (DB)**: `Lament`

**Pattern**: `charade`

**Form**: `charade(abbreviation(E from 'English'), synonym(LEG from 'cricket side'), literal(Y from 'the ultimate in ignominy'))`

**Pieces**: E from `English`; LEG from `cricket side`; Y from `the ultimate in ignominy`

- ✓ `assembly`: tree produces ELEGY
- ✓ `bridge.leaves`: verified "English"->E (abbr); "cricket side"->LEG; literal "the ultimate in ignominy"
- ✓ `bridge.indicators`: no indicator-bearing nodes
- ✓ `residue`: 

---

## FAIL (2)

### 16a. ARCH

**Clue**: Chief taking off first of the month

**Blog**: {m} ARCH (month) [taking off first]

**Definition (DB)**: `Chief`

**Pattern**: `single_piece`

**Form**: `synonym(ARCH from 'month')`

**Pieces**: ARCH from `month`

**Indicators**: `taking off first` [container]

- ✓ `assembly`: tree produces ARCH
- ✓ `bridge.leaves`: verified "month"->ARCH
- ✓ `bridge.indicators`: no indicator-bearing nodes
- ✗ `residue`: unaccounted: ['first', 'off', 'taking']

---

### 22d. WHIT

**Clue**: Congress supporting leaders of White House a tiny bit

**Blog**: W {hite} + H {ouse} [leaders of…], IT (congress). A little weak having ‘white’ in the clue although of course it adds to the surface.

**Definition (DB)**: `a tiny bit`

**Pattern**: `charade`

**Form**: `charade(literal(W from 'White'), literal(H from 'House'), abbreviation(IT from 'Congress supporting'))`

**Pieces**: IT from `Congress supporting`; W from `White`; H from `House`

**Indicators**: `leaders of` [acrostic]

- ✓ `assembly`: tree produces WHIT
- ✓ `bridge.leaves`: verified literal "White"; literal "House"; "Congress supporting"->IT (abbr)
- ✓ `bridge.indicators`: no indicator-bearing nodes
- ✗ `residue`: unaccounted: ['leaders', 'of']

---

## NO_FORM (12)

### 9a. CONTEMPTOFCOURT

**Clue**: Smashing racket crime?

**Blog**: Cryptic with reference to tennis

**Definition (DB)**: `Smashing racket crime?`

**Pattern**: `—`


---

### 11a. ALTEREGO

**Clue**: Imperative to get little George a close friend

**Blog**: Imperative here is an instruction, so ‘to get little George ‘ / GEO you ALTER EGO

**Definition (DB)**: `a close friend`

**Pattern**: `—`

**Pieces**: I from `Imperative`; EGO from `to`; GEO from `get`; ALTER from `little`; G from `George`


---

### 13a. GIANTPANDA

**Clue**: Huge letters used to spell “Dad”, being popular in China

**Blog**: GIANT (huge), P AND A (letters used to spell PA – Dad)

**Definition (DB)**: `being popular in China`

**Pattern**: `—`

**Pieces**: GIANT from `Huge`; A from `letters used to spell Dad`


---

### 20a. HOARSE

**Clue**: Husky is a cracking four-legged friend

**Blog**: A contained by [cracking] HO ⁁ RSE ( four-legged friend ). From the days of Uncle Mac and Children’s Favourites on the radio .

**Definition (DB)**: `Husky`

**Pattern**: `—`

**Pieces**: A from `a`; RSE from `four legged friend`

**Indicators**: `is` [container]; `cracking` [anagram]


---

### 23a. OPENANDSHUTCASE

**Clue**: Inspector isn’t overworked by this perfunctory luggage check?

**Blog**: Two meanings

**Definition (DB)**: `Inspector isn’t overworked by this / perfunctory luggage check`

**Pattern**: `—`


---

### 25a. NOTORIETY

**Clue**: Turned on Conservative about regular instances of vilest infamy

**Blog**: ON reversed [turned], TOR ⁁ Y (Conservative) containing [about] {v} I {l} E {s} T [regular instances of…]

**Definition (DB)**: `infamy`

**Pattern**: `—`

**Pieces**: ON from `on`; Y from `Conservative`; IET from `regular instances of vilest`

**Indicators**: `Turned` [reversal]; `about` [reversal]


---

### 3d. REELECTS

**Clue**: Staggers after receiving shock treatment, puts back in power

**Blog**: REEL ⁁ S (staggers) containing [after receiving] ECT (shock treatment – electroconvulsive therapy )

**Definition (DB)**: `puts back in power`

**Pattern**: `—`

**Pieces**: S from `Staggers`; ECT from `shock treatment`

**Indicators**: `after receiving` [container]


---

### 6d. SACHET

**Clue**: Way to accommodate long bag

**Blog**: S ⁁ T (way) containing [to accommodate] ACHE (long)

**Definition (DB)**: `bag`

**Pattern**: `—`

**Pieces**: T from `Way`; ACHE from `long`

**Indicators**: `to accommodate` [container]


---

### 7d. AMUSEMENTARCADE

**Clue**: Cameramen used at broadcast games screened here

**Blog**: Anagram [broadcast] of CAMERAMAN USED AT

**Definition (DB)**: `games screened here`

**Pattern**: `—`

**Pieces**: CAMERAMANUSEDAT from `Cameramen used at`

**Indicators**: `broadcast` [anagram]


---

### 8d. POTBOILER

**Clue**: Post Office worker struggling to pen book, one written to raise cash

**Blog**: PO (Post Office), T ⁁ OILER (worker struggling) containing [to pen] B (book). Not sure where to place ‘struggling’ in this, but ‘toil’ suggests hard work. The definition refers back to ‘book’.

**Definition (DB)**: `one written to raise cash`

**Pattern**: `—`

**Pieces**: PO from `Post Office`; OILER from `worker struggling`; B from `book`

**Indicators**: `to pen` [container]


---

### 15d. IMPOSTER

**Clue**: Deceitful character put up motorway notice

**Blog**: M1 (motorway) reversed [put up], POSTER (notice)

**Definition (DB)**: `Deceitful character`

**Pattern**: `—`

**Pieces**: M from `motorway`; POSTER from `notice`

**Indicators**: `put up` [reversal]


---

### 18d. PLIANT

**Clue**: Plastic factory beginning to implement cuts

**Blog**: I {mplement} [beginning to…] contained by [cuts] PL ⁁ ANT (factory)

**Definition (DB)**: `Plastic`

**Pattern**: `—`

**Pieces**: ANT from `factory`; I from `beginning to implement`

**Indicators**: `cuts` [deletion]


---
