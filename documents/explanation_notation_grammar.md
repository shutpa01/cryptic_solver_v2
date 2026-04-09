# Times Explanation Notation Grammar

Derived from analysis of 55,000+ Times for the Times blog explanations.

## Notation Symbols

### `+` — Charade separator
Separates pieces that concatenate left-to-right to form the answer.
```
GENT + EEL = GENTEEL
S + O.M. + ME = SOMME
DIRECT + OR + GENERAL = DIRECTOR GENERAL
```

### `()` — Four distinct uses

**1. Container: CAPS(CAPS)CAPS**
Uppercase inside uppercase = insertion. Inner letters go inside outer letters.
```
FA(LUNG ON)G         → F + LUNGON + G → FALUNGONG
A(CHAT)ES            → A + CHAT + ES → ACHATES (CHAT inside AES)
S(TO)IC              → S + TO + IC → STOIC (TO inside SIC)
TR(anagram of TEAM)ENT → TREATMENT
```

**2. Gloss: CAPS(lowercase)**
Lowercase in parens explains the meaning of the preceding CAPS piece. NOT part of the answer.
```
GENT(male)           → GENT means "male"
SABLE(fur)           → SABLE means "fur"
SERVER(acolyte)      → SERVER means "acolyte"
```

**3. Abbreviation source: LETTER(rest-of-word)**
Shows where a single letter or short abbreviation comes from.
```
S(econd)             → S from "second"
V(against)           → V from "versus/against"
E(nglish)            → E from "English"
ER(hesitation)       → ER from "hesitation"
```

**4. Enumeration: (digits)**
Just the letter count of the answer. Ignore.
```
(7), (3,5), (2,4,6)
```

### `[]` — Three distinct uses

**1. Deletion from adjacent word: [lowercase]CAPS or CAPS[lowercase]**
Lowercase letters in brackets were removed from a longer word to get the CAPS piece.
```
[pr]EVENT            → remove PR from PREVENT → EVENT
BIL[ious]            → remove IOUS from BILIOUS → BIL
INDUS[try]           → remove TRY from INDUSTRY → INDUS
A[ppalling]          → A from "appalling" (first letter)
```

**2. Indicator/operation label: [keyword]**
Describes what operation is happening. NOT part of the answer.
```
[messily]            → anagram indicator
[initially]          → first letter indicator
[finally]            → last letter indicator
[reduced]            → truncation indicator
[in]                 → container indicator
```

**3. Hidden word marker: {start}CAPS{end}**
Combined with {} to show hidden word boundaries.
```
Hidden in {churc}H AT E{aster}
Hidden in {tetc}HY BRID{egroom}
```

### `{}` — Partial deletion
Letters in curly braces were removed/excluded. The remaining letters contribute.
```
{r}EEL               → remove R from REEL → EEL
PENT{y}              → remove Y from PENTY → PENT
{dagge}R             → R from "dagger" (last letter)
{c}ODE{s}            → remove C and S from CODES → ODE
{spea}R              → R from "spear" (last letter)
```

### `*` — Anagram marker
Letters before * (usually in parens) are anagrammed to form the answer.
```
(EIGHT NUDES*)       → anagram of EIGHTNUDES = GESUNDHEIT
(NOW SO)*            → anagram of NOWSO = SWOON
(LACK POWER)*        → anagram of LACKPOWER = WORKPLACE
```

### `,` and `.` — Piece separators (informal)
Commas and periods sometimes separate charade pieces instead of +.
```
BRAN DISHES          → BRAN + DISHES (space-separated)
S, COPE              → S + COPE
MO. TIFF             → MO + TIFF
```

### `~` — Partial letter notation
Used to show fragments.
```
R~ A~ GOUT           → R + A + GOUT
~E in FE             → E inside FE
```

## Prose Keywords (no symbol notation)

### Operations
- `reversed` / `rev.` / `backwards` / `upside-down` / `going up` → reversal
- `anagram of` / `Anagram` → anagram
- `hidden in` / `hidden word` → hidden
- `sounds like` / `homophone` / `audible` → homophone
- `double definition` / `DD` / `two meanings` → double definition
- `cryptic` / `cryptic definition` / `CD` → cryptic definition
- `first letters` / `initial letters` / `acrostic` → acrostic
- `final letters` / `last letters` → terminal letters
- `Spoonerism` / `Spooner` → spoonerism
- `alternate letters` / `odd letters` / `even letters` → alternation

### Container prose
- `inside` / `in` / `within` / `contained by` → X inside Y
- `around` / `outside` / `containing` / `surrounding` → Y around X
- `nursing` / `swallowing` / `eating` / `entertaining` → container

### Deletion prose
- `without` / `losing` / `minus` / `dropping` → deletion
- `heartless` / `emptied` → remove middle
- `headless` / `beheaded` → remove first letter
- `endless` / `curtailed` / `briefly` → remove last letter

## Composite Structures

Most clues combine multiple operations. The notation handles this by nesting:
```
PREDISPOSED = PR(Public Relations), then POSED(asked) after SIDE(team) reversed
→ PR + EDIS(=SIDE reversed) + POSED = charade with embedded reversal

SWIMMING POOL = S(WIMMIN)G + POOL
→ container (WIMMIN inside SG) + charade (POOL)

MECHANIC = ME, then I inside CHANC(E)
→ ME + CHA(I)NC{E} = charade with container and deletion
```

## Verification Rule

For any parsed explanation, the extracted letter pieces MUST concatenate
(after applying operations) to produce the known answer. If they don't,
the parse is wrong.
