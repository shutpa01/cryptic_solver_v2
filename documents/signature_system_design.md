# Signature System Design — Cryptic Clue Solver V3

## Core Insight

Every cryptic clue's wordplay window is a left-to-right sequence of functional roles.
Each word plays exactly one role. The role sequence IS the signature.
If you know the signature, solving becomes mechanical execution, not reasoning.

## Token Vocabulary (28 tokens)

### Core Roles
- `SYN_F` — synonym fodder (look up synonym in DB)
- `ABR_F` — abbreviation fodder (look up abbreviation in DB)
- `ANA_F` — anagram fodder (use raw letters, rearrange)
- `RAW` — raw word used as-is (its own letters contribute directly)
- `HID_F` — hidden word fodder (answer spans these words)
- `HOM_F` — homophone fodder (sounds like the answer/component)
- `DEL_F` — deletion fodder (what gets removed)
- `POS_F` — positional fodder (word from which letters are extracted)

### Indicator Tokens
- `ANA_I` — anagram indicator (signals rearrangement)
- `REV_I` — reversal indicator (signals reversal)
- `CON_I` — container indicator (signals insertion)
- `DEL_I` — deletion indicator (signals removal)
- `HID_I` — hidden word indicator (signals lurking answer)
- `HOM_I` — homophone indicator (signals sounds-like)

### Positional Indicators (what survives)
- `POS_I_FIRST` — keep first letter (initially, head of, leader)
- `POS_I_LAST` — keep last letter (finally, ultimately, tail)
- `POS_I_OUTER` — keep outer letters (gutted, emptied)
- `POS_I_MIDDLE` — keep middle letter(s) (heart of, core of)
- `POS_I_ALTERNATE` — keep alternate letters (regularly, oddly, evenly)
- `POS_I_TRIM_FIRST` — remove first letter (beheaded, topless)
- `POS_I_TRIM_LAST` — remove last letter (endlessly, almost, nearly)
- `POS_I_TRIM_MIDDLE` — remove middle (heartless, lose heart)
- `POS_I_TRIM_OUTER` — remove first and last (peeled, edges off)
- `POS_I_HALF` — take half the letters (50% of)

### Structural
- `DEF` — definition
- `LNK` — link word (ignorable connective)
- `DOUBLE_DEFINITION` — whole clue is two definitions
- `CRYPTIC_DEFINITION` — whole clue is a cryptic definition
- `AND_LIT` — whole clue is both wordplay and definition

### Rare
- `SUB` — letter substitution (one letter replaced by another)

## Dominant Signatures (from 12 puzzle analysis, ~360 clues)

### Tier 1 — Very Common (each appears 10+ times)
| Signature | Example | Operation |
|-----------|---------|-----------|
| `ANA_F · ANA_I` | "Fracture below" → ELBOW | Anagram fodder words |
| `ABR_F · SYN_F` | "Former lover mentioned" → EXCITED | Abbreviation + synonym |
| `SYN_F · SYN_F` | "Goat maybe, then run away" → BUTTERFLY | Two synonyms joined |
| `DOUBLE_DEFINITION` | "Train carriage" → COACH | Two meanings |
| `SYN_F · CON_I · SYN_F` | "Fuss about a" → STAIR | Synonym inside synonym |
| `HID_F · HID_I` | "plant sap, covered in" → ANTS | Answer spans clue words |

### Tier 2 — Common (5-9 instances)
| Signature | Example |
|-----------|---------|
| `SYN_F · REV_I · SYN_F` | "innocent revolutionary, little noise" → ERUPTING |
| `ABR_F · ABR_F · SYN_F` | "Quiet answer, Charles perhaps" → SHAKING |
| `ABR_F · CON_I · SYN_F` | "Victor entering only" → SOLVE |
| `ANA_F · ANA_I · ABR_F` | "Large pet damaged, hotel" → TELEGRAPH |
| `HID_F · HID_I · REV_I` | "believe epidemic's, some, over" → PEEVE |
| `HOM_F · HOM_I` | "missing, reported" → MORNING |
| `CRYPTIC_DEFINITION` | "Make no progress, in the main" → TREAD WATER |
| `POS_F · POS_I_ALTERNATE` | "The grasses, regularly" → TERSE |
| `SYN_F · POS_I_TRIM_LAST` | "muffler, end off" → SCAR |
| `SYN_F · ABR_F · DEL_I` | "Swears at, adult, leaving" → BUSES |

### Tier 3 — Occasional (3-4 instances)
| Signature | Example |
|-----------|---------|
| `ABR_F · SYN_F · REV_I` | "Golf, does perhaps, upset" → GREED |
| `SYN_F · POS_I_TRIM_FIRST` | "dull, without leader" → OFTEN |
| `SYN_F · ABR_F · SYN_F` | "Labour, over English, tax" → TOILETRY |
| `ABR_F · ABR_F · CON_I · SYN_F` | "American, quietly, infiltrating group" → SUSPECT |
| `SYN_F · ANA_F · ANA_I` | "greyish brown, Grease, smeared" → DUNGAREES |
| `ABR_F · ANA_F · ANA_I` | "New, treats, prepared" → NATTERS |
| `ANA_F · ANA_I · CON_I · ABR_F` | "cycles, Doctor, about area" → SCARCELY |
| `SYN_F · CON_I · ABR_F` | "See you, accepting, university" → VALUE |
| `ABR_F · SYN_F · CON_I · ABR_F` | "Small, winged insects, covering, old" → SMOOTHS |
| `SYN_F · POS_I_TRIM_LAST · SYN_F` | "Good-looking person, cutting end off, flower" → DISASTER |
| `POS_F · POS_I_OUTER · SYN_F` | "pretzel, hollow, consume" → PLEAT |
| `SYN_F · CON_I · POS_F · POS_I_OUTER` | "Five siblings, consuming, case of choice" → QUINCES |
| `ABR_F · SYN_F · POS_I_TRIM_FIRST` | "singular, snake, heading off" → SAMBA |

### Distribution
- Tier 1 (~6 signatures) covers ~30% of clues
- Tier 1+2 (~16 signatures) covers ~55% of clues
- Tier 1+2+3 (~30 signatures) covers ~75% of clues
- Long tail (~100 signatures) covers remaining ~25%

## Solving Pipeline

### Step 1: Strip Definition
Already implemented. Removes definition, leaving the wordplay window.

### Step 2: Candidate Role Generation
For each word in the wordplay window, generate possible roles:
- Check indicator DB (4,664 entries) → could be `*_I`
- Check abbreviation DB → could be `ABR_F`
- Check synonym DB → could be `SYN_F`
- Check if word is a known link word → could be `LNK`
- Check letter count vs answer length → could be `ANA_F`
- Default: could be `RAW`, `POS_F`, `HID_F`, `HOM_F`

### Step 3: Signature Candidate Filtering
- Word count constrains possible signatures
- Known indicators constrain possible operations
- Answer length constrains fodder combinations
- Prune impossible combinations

### Step 4: Mechanical Execution
For each candidate signature, attempt to execute:
- `SYN_F` → look up synonym, check if it fits
- `ABR_F` → look up abbreviation, check if it fits
- `ANA_F + ANA_I` → anagram the fodder letters, check if they produce answer
- `CON_I` → try inserting inner piece into outer piece
- `REV_I` → reverse the adjacent piece
- `POS_I_*` → extract specified letters
- `HID_F + HID_I` → check if answer is hidden in fodder words
- Assemble result, compare to known answer

### Step 5: API Call (only if needed)
Only called when:
- No candidate signature produces the answer mechanically
- Multiple signatures work and disambiguation is needed
- A word has no DB matches (unknown synonym/abbreviation)

When called, the prompt is highly constrained:
```
Clue: [clue text]
Answer: [answer]
Wordplay window: [words]
Word analysis:
  - "word1": possible roles [ABR_F(→X), SYN_F(?)]
  - "word2": possible roles [ANA_I, SYN_F(?)]
  ...
No mechanical solution found. Which mapping is correct?
```

## Advantages Over Current System
1. AI doesn't reason about cryptic wordplay — it does constrained slot-filling
2. Most clues solved mechanically with zero API calls
3. When API is called, the question is trivially easy
4. Explanations are generated from the signature — perfectly structured
5. Confidence is near-binary: the signature either produces the answer or it doesn't
6. The assembler becomes a simple executor, not a search engine

## Data Assets
- 183k parsed explanations → training data for word-role alignment
- 4,664 indicator DB entries → indicator word recognition
- 1.3M synonym pairs → synonym lookup
- 713k definition-answer pairs → definition identification
- 2,150 wordplay entries → additional reference

## Open Questions
1. How to handle multi-word units ("former lover" = EX, not two separate words)
2. Cockney transformations (H-dropping) — rare but real
3. Cross-reference clues ("17 across" as fodder source)
4. Self-referential setter pronouns ("this writer" = I/ME)
5. Repetition modifiers ("setter repeatedly" = I...I)
6. Where definition ends and wordplay begins — still the hardest subproblem
