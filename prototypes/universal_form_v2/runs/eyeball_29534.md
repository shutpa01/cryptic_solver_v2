# Times 29534 — clipboard verifier eyeball

## 1a. LIVERWORT

**Counts: {'PASS': 3, 'FAIL': 13, 'NO_FORM': 12}**

- Clue: On air row traumatised and exhausted Robert Plant
- Blog: LIVE (on air), anagram [traumatised] of ROW , then R {ober} T [exhausted]. Apparently Robert Plant was the lead singer of the rock band Led Zeppelin.
- DB definition: `Plant`
- Form: `anagram [traumatised](synonym(LIVE ← 'On air'), literal(ROW ← 'row'), positional[outer](RT ← 'exhausted Robert'))`
- Definition (form): `Plant`
- Link words: ['and']
- **VERDICT: FAIL**
  - ✓ `assembly`: tree produces LIVERWORT
  - ✗ `mechanism.leaves`: positional[outer] of 'exhausted robert': extracted 'ET' != 'RT'
  - ✗ `mechanism.fodder`: anagram child must be literal/raw, got synonym('LIVE'); anagram child must be literal/raw, got positional('RT')
  - ✗ `mechanism.indicators`: indicator 'traumatised' not in DB for op 'anagram'; positional[outer] leaf has no indicator
  - ✓ `residue`: every wordplay word accounted for
  - Enrichment candidates:
    - `indicator` indicator 'traumatised' not in DB for op 'anagram' (need ['anagram'])

## 6a. SHARP

- Clue: Piercing, principally shrill instrument
- Blog: S {hrill} [principally], HARP (instrument)
- DB definition: `Piercing`
- Form: `charade(literal(S ← 'principally shrill'), synonym(HARP ← 'instrument'))`
- Definition (form): `Piercing`
- **VERDICT: FAIL**
  - ✓ `assembly`: tree produces SHARP
  - ✗ `mechanism.leaves`: literal 'S': source letters 'PRINCIPALLYSHRILL' != value
  - ✓ `mechanism.fodder`: anagram/hidden/acrostic children are literal/raw
  - ✓ `mechanism.indicators`: no indicators required
  - ✓ `residue`: every wordplay word accounted for

## 9a. CONTEMPTOFCOURT

- Clue: Smashing racket crime?
- Blog: Cryptic with reference to tennis
- DB definition: `Smashing racket crime?`
- **VERDICT: NO_FORM** (no form built)

## 10a. MILDEW

- Clue: Disease from beer we knocked back
- Blog: MILD (beer), then WE reversed [knocked back]. A plant disease caused by fungus.
- DB definition: `Disease`
- Form: `charade(synonym(MILD ← 'beer'), reversal [knocked back](literal(WE ← 'we')))`
- Definition (form): `Disease from`
- **VERDICT: PASS**
  - ✓ `assembly`: tree produces MILDEW
  - ✓ `mechanism.leaves`: verified synonym(MILD←'beer'); literal(WE←'we')
  - ✓ `mechanism.fodder`: anagram/hidden/acrostic children are literal/raw
  - ✓ `mechanism.indicators`: verified 'knocked back'→reversal
  - ✓ `residue`: every wordplay word accounted for

## 11a. ALTEREGO

- Clue: Imperative to get little George a close friend
- Blog: Imperative here is an instruction, so ‘to get little George ‘ / GEO you ALTER EGO
- DB definition: `a close friend`
- **VERDICT: NO_FORM** (no form built)

## 13a. GIANTPANDA

- Clue: Huge letters used to spell “Dad”, being popular in China
- Blog: GIANT (huge), P AND A (letters used to spell PA – Dad)
- DB definition: `being popular in China`
- **VERDICT: NO_FORM** (no form built)

## 14a. FELL

- Clue: Knock down chap emitting cry of pain
- Blog: FELL {ow} (chap) [emitting cry of pain]
- DB definition: `Knock down`
- Form: `synonym(FELL ← 'chap emitting cry of pain')`
- Definition (form): `Knock down`
- **VERDICT: FAIL**
  - ✓ `assembly`: tree produces FELL
  - ✗ `mechanism.leaves`: synonym 'FELL' not in DB for 'chap emitting cry of pain'
  - ✓ `mechanism.fodder`: anagram/hidden/acrostic children are literal/raw
  - ✓ `mechanism.indicators`: no indicators required
  - ✓ `residue`: every wordplay word accounted for
  - Enrichment candidates:
    - `synonym` synonym 'FELL' not in DB for 'chap emitting cry of pain'

## 16a. ARCH

- Clue: Chief taking off first of the month
- Blog: {m} ARCH (month) [taking off first]
- DB definition: `Chief`
- Form: `synonym(ARCH ← 'month')`
- Definition (form): `Chief`
- Link words: ['of', 'the']
- **VERDICT: FAIL**
  - ✓ `assembly`: tree produces ARCH
  - ✗ `mechanism.leaves`: synonym 'ARCH' not in DB for 'month'
  - ✓ `mechanism.fodder`: anagram/hidden/acrostic children are literal/raw
  - ✓ `mechanism.indicators`: no indicators required
  - ✗ `residue`: unaccounted: 'taking'; unaccounted: 'off'; unaccounted: 'first'
  - Enrichment candidates:
    - `synonym` synonym 'ARCH' not in DB for 'month'

## 17a. PENTAMETER

- Clue: Write uninspiring incomplete ending for literary work
- Blog: PEN (write), TAME (uninspiring), TER {m} (ending) [incomplete]
- DB definition: `literary work`
- Form: `charade(synonym(PEN ← 'Write'), synonym(TAME ← 'uninspiring'), synonym(TER ← 'incomplete ending'))`
- Definition (form): `for literary work`
- **VERDICT: FAIL**
  - ✓ `assembly`: tree produces PENTAMETER
  - ✗ `mechanism.leaves`: synonym 'TER' not in DB for 'incomplete ending'
  - ✓ `mechanism.fodder`: anagram/hidden/acrostic children are literal/raw
  - ✓ `mechanism.indicators`: no indicators required
  - ✓ `residue`: every wordplay word accounted for
  - Enrichment candidates:
    - `synonym` synonym 'TER' not in DB for 'incomplete ending'

## 19a. SICILIAN

- Clue: Italian snail unexpectedly seen around here in France
- Blog: Anagram [unexpectedly] of SNAIL containing [seen around] ICI (‘here’ in France)
- DB definition: `Italian`
- Form: `container [seen around](anagram [unexpectedly](literal(SNAIL ← 'snail')), synonym(ICI ← 'here in France'))`
- Definition (form): `Italian`
- **VERDICT: PASS**
  - ✓ `assembly`: tree produces SICILIAN
  - ✓ `mechanism.leaves`: verified literal(SNAIL←'snail'); synonym(ICI←'here in France')
  - ✓ `mechanism.fodder`: anagram/hidden/acrostic children are literal/raw
  - ✓ `mechanism.indicators`: verified 'seen around'→container; 'unexpectedly'→anagram
  - ✓ `residue`: every wordplay word accounted for

## 20a. HOARSE

- Clue: Husky is a cracking four-legged friend
- Blog: A contained by [cracking] HO ⁁ RSE ( four-legged friend ). From the days of Uncle Mac and Children’s Favourites on the radio .
- DB definition: `Husky`
- **VERDICT: NO_FORM** (no form built)

## 23a. OPENANDSHUTCASE

- Clue: Inspector isn’t overworked by this perfunctory luggage check?
- Blog: Two meanings
- DB definition: `Inspector isn’t overworked by this / perfunctory luggage check`
- **VERDICT: NO_FORM** (no form built)

## 24a. EMMET

- Clue: This compiler heading west bumped into tourist
- Blog: ME (this compiler) reversed [heading west], MET (bumped into). Cornish slang for a holidaymaker. In Devon they’re called Grockles.
- DB definition: `tourist`
- Form: `charade(reversal [heading west](abbreviation(ME ← 'This compiler')), synonym(MET ← 'bumped into'))`
- Definition (form): `tourist`
- **VERDICT: PASS**
  - ✓ `assembly`: tree produces EMMET
  - ✓ `mechanism.leaves`: verified abbreviation(ME←'This compiler'); synonym(MET←'bumped into')
  - ✓ `mechanism.fodder`: anagram/hidden/acrostic children are literal/raw
  - ✓ `mechanism.indicators`: verified 'heading west'→reversal
  - ✓ `residue`: every wordplay word accounted for

## 25a. NOTORIETY

- Clue: Turned on Conservative about regular instances of vilest infamy
- Blog: ON reversed [turned], TOR ⁁ Y (Conservative) containing [about] {v} I {l} E {s} T [regular instances of…]
- DB definition: `infamy`
- **VERDICT: NO_FORM** (no form built)

## 1d. LOCUM

- Clue: Substitute officer upset uniform men at the start
- Blog: COL (officer – Colonel ) reversed [upset], U (uniform), M {en} [at the start]
- DB definition: `Substitute`
- Form: `charade(reversal [upset](synonym(COL ← 'officer')), abbreviation(U ← 'uniform'), literal(M ← 'men at the start'))`
- Definition (form): `Substitute`
- **VERDICT: FAIL**
  - ✓ `assembly`: tree produces LOCUM
  - ✗ `mechanism.leaves`: literal 'M': source letters 'MENATTHESTART' != value
  - ✓ `mechanism.fodder`: anagram/hidden/acrostic children are literal/raw
  - ✓ `mechanism.indicators`: verified 'upset'→reversal
  - ✓ `residue`: every wordplay word accounted for

## 2d. VANILLAICECREAM

- Clue: Sweet eclair Calvin ordered each month
- Blog: Anagram [ordered] of ECLAIR CALVIN , then EA (each), M (month)
- DB definition: `Sweet`
- Form: `anagram [ordered](literal(ECLAIRCALVIN ← 'eclair Calvin'), abbreviation(EA ← 'each'), abbreviation(M ← 'month'))`
- Definition (form): `Sweet`
- **VERDICT: FAIL**
  - ✓ `assembly`: tree produces VANILLAICECREAM
  - ✓ `mechanism.leaves`: verified literal(ECLAIRCALVIN←'eclair Calvin'); abbreviation(EA←'each'); abbreviation(M←'month')
  - ✗ `mechanism.fodder`: anagram child must be literal/raw, got abbreviation('EA'); anagram child must be literal/raw, got abbreviation('M')
  - ✓ `mechanism.indicators`: verified 'ordered'→anagram
  - ✓ `residue`: every wordplay word accounted for

## 3d. REELECTS

- Clue: Staggers after receiving shock treatment, puts back in power
- Blog: REEL ⁁ S (staggers) containing [after receiving] ECT (shock treatment – electroconvulsive therapy )
- DB definition: `puts back in power`
- **VERDICT: NO_FORM** (no form built)

## 4d. OOPS

- Clue: Served up spirit, just missing king … sorry!
- Blog: SPOO {k} (spirit) reversed [served up] [just missing king]
- DB definition: `sorry`
- Form: `reversal [Served up](synonym(SPOO ← 'spirit just missing king'))`
- Definition (form): `sorry`
- **VERDICT: FAIL**
  - ✓ `assembly`: tree produces OOPS
  - ✗ `mechanism.leaves`: synonym 'SPOO' not in DB for 'spirit just missing king'
  - ✓ `mechanism.fodder`: anagram/hidden/acrostic children are literal/raw
  - ✓ `mechanism.indicators`: verified 'Served up'→reversal
  - ✓ `residue`: every wordplay word accounted for
  - Enrichment candidates:
    - `synonym` synonym 'SPOO' not in DB for 'spirit just missing king'

## 5d. TROGLODYTE

- Clue: Grotty old vagrant last seen in cave?
- Blog: Anagram [vagrant] of GROTTY OLD , then {cav} E [last seen in…]. A troglodyte was a cave-dweller so I guess the whole clue has to be the definition here.
- DB definition: `Grotty old vagrant last seen in cave?`
- Form: `anagram [vagrant](literal(GROTTYOLD ← 'Grotty old'), literal(E ← 'last seen in cave'))`
- Definition (form): ``
- **VERDICT: FAIL**
  - ✓ `assembly`: tree produces TROGLODYTE
  - ✗ `mechanism.leaves`: literal 'E': source letters 'LASTSEENINCAVE' != value
  - ✓ `mechanism.fodder`: anagram/hidden/acrostic children are literal/raw
  - ✓ `mechanism.indicators`: verified 'vagrant'→anagram
  - ✗ `residue`: definition phrase '' is not a contiguous span of the clue

## 6d. SACHET

- Clue: Way to accommodate long bag
- Blog: S ⁁ T (way) containing [to accommodate] ACHE (long)
- DB definition: `bag`
- **VERDICT: NO_FORM** (no form built)

## 7d. AMUSEMENTARCADE

- Clue: Cameramen used at broadcast games screened here
- Blog: Anagram [broadcast] of CAMERAMAN USED AT
- DB definition: `games screened here`
- **VERDICT: NO_FORM** (no form built)

## 8d. POTBOILER

- Clue: Post Office worker struggling to pen book, one written to raise cash
- Blog: PO (Post Office), T ⁁ OILER (worker struggling) containing [to pen] B (book). Not sure where to place ‘struggling’ in this, but ‘toil’ suggests hard work. The definition refers back to ‘book’.
- DB definition: `one written to raise cash`
- **VERDICT: NO_FORM** (no form built)

## 12d. GAMEWARDEN

- Clue: Plucky wife, endlessly zealous officer in reserve
- Blog: GAME (plucky), W (wife), ARDEN {t} (zealous) [endlessly]
- DB definition: `officer in reserve`
- Form: `charade(synonym(GAME ← 'Plucky'), abbreviation(W ← 'wife'), synonym(ARDEN ← 'endlessly zealous'))`
- Definition (form): `officer in reserve`
- **VERDICT: FAIL**
  - ✓ `assembly`: tree produces GAMEWARDEN
  - ✗ `mechanism.leaves`: synonym 'ARDEN' not in DB for 'endlessly zealous'
  - ✓ `mechanism.fodder`: anagram/hidden/acrostic children are literal/raw
  - ✓ `mechanism.indicators`: no indicators required
  - ✓ `residue`: every wordplay word accounted for
  - Enrichment candidates:
    - `synonym` synonym 'ARDEN' not in DB for 'endlessly zealous'

## 13d. GLADSTONE

- Clue: Young man’s backing good quality statesman
- Blog: G (good), LAD’S (young man’s), TONE (quality)
- DB definition: `statesman`
- Form: `charade(abbreviation(G ← 'backing good'), synonym(LAD ← 'Young man'), literal(S ← 's'), synonym(TONE ← 'quality'))`
- Definition (form): `statesman`
- **VERDICT: FAIL**
  - ✓ `assembly`: tree produces GLADSTONE
  - ✗ `mechanism.leaves`: abbreviation 'G' not in DB for 'backing good'
  - ✓ `mechanism.fodder`: anagram/hidden/acrostic children are literal/raw
  - ✓ `mechanism.indicators`: no indicators required
  - ✗ `residue`: unaccounted: "man's"
  - Enrichment candidates:
    - `abbreviation` abbreviation 'G' not in DB for 'backing good'

## 15d. IMPOSTER

- Clue: Deceitful character put up motorway notice
- Blog: M1 (motorway) reversed [put up], POSTER (notice)
- DB definition: `Deceitful character`
- **VERDICT: NO_FORM** (no form built)

## 18d. PLIANT

- Clue: Plastic factory beginning to implement cuts
- Blog: I {mplement} [beginning to…] contained by [cuts] PL ⁁ ANT (factory)
- DB definition: `Plastic`
- **VERDICT: NO_FORM** (no form built)

## 21d. ELEGY

- Clue: Lament English cricket side, the ultimate in ignominy
- Blog: E (English), LEG (cricket side – as opposed to ‘off’ ), {ignomin} Y [the ultimate in…]
- DB definition: `Lament`
- Form: `charade(abbreviation(E ← 'English'), synonym(LEG ← 'cricket side'), literal(Y ← 'the ultimate in ignominy'))`
- Definition (form): `Lament`
- **VERDICT: FAIL**
  - ✓ `assembly`: tree produces ELEGY
  - ✗ `mechanism.leaves`: literal 'Y': source letters 'THEULTIMATEINIGNOMINY' != value
  - ✓ `mechanism.fodder`: anagram/hidden/acrostic children are literal/raw
  - ✓ `mechanism.indicators`: no indicators required
  - ✓ `residue`: every wordplay word accounted for

## 22d. WHIT

- Clue: Congress supporting leaders of White House a tiny bit
- Blog: W {hite} + H {ouse} [leaders of…], IT (congress). A little weak having ‘white’ in the clue although of course it adds to the surface.
- DB definition: `a tiny bit`
- Form: `charade(literal(W ← 'White'), literal(H ← 'House'), abbreviation(IT ← 'Congress supporting'))`
- Definition (form): `a tiny bit`
- **VERDICT: FAIL**
  - ✓ `assembly`: tree produces WHIT
  - ✗ `mechanism.leaves`: literal 'W': source letters 'WHITE' != value; literal 'H': source letters 'HOUSE' != value; abbreviation 'IT' not in DB for 'congress supporting'
  - ✓ `mechanism.fodder`: anagram/hidden/acrostic children are literal/raw
  - ✓ `mechanism.indicators`: no indicators required
  - ✗ `residue`: unaccounted: 'leaders'; unaccounted: 'of'
  - Enrichment candidates:
    - `abbreviation` abbreviation 'IT' not in DB for 'congress supporting'
