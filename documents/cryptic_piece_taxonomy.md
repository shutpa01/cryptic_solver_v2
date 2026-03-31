# Cryptic Clue Piece Taxonomy

## The Anatomy of a Cryptic Answer

Every cryptic clue has a **definition** (a word or phrase meaning the answer) and **wordplay** (instructions for constructing the answer from pieces). The answer is assembled from pieces, and each piece is produced by a specific mechanism.

## Assembly Types (how pieces fit together)

| Assembly | Description | Verification |
|----------|-------------|--------------|
| **charade** | Pieces concatenate left-to-right: A + B + C = ANSWER | `concat(pieces) == answer` |
| **container** | One piece inserted inside another: A contains B = A...B...A | Try each piece as inner, rest as outer, test all insertion points |
| **anagram** | Letters rearranged: sorted(fodder) == sorted(answer) | `sorted(letters) == sorted(answer)` |
| **hidden** | Answer appears contiguously in clue text | `answer in clue_letters` with proper suffix/prefix validation |
| **hidden_reversed** | Answer reversed appears in clue text | `reverse(answer) in clue_letters` |
| **reversal** | Whole answer is a single piece reversed | `reverse(piece) == answer` |
| **double_definition** | Two definitions, no wordplay pieces | Both halves define the answer independently |
| **homophone** | Answer sounds like another word | Phonetic match (hard to verify mechanically) |
| **cryptic_definition** | Entire clue is a misleading definition | No mechanical verification possible |
| **spoonerism** | Initial consonants of two words swapped | Swap initials, check both are real words |

## Piece Mechanisms (how each piece gets its letters)

### Directly Verifiable (DB lookup)

| Mechanism | Description | Example | Verification |
|-----------|-------------|---------|--------------|
| **synonym** | Word means X, X provides letters | "vessel" -> VAT | `is_synonym("vessel", "VAT")` in synonyms_pairs or definition_answers_augmented |
| **abbreviation** | Conventional short form | "quiet" -> P, "street" -> ST | `is_abbreviation("quiet", "P")` in wordplay table |
| **literal** | The clue word itself provides its own letters | "on" -> ON, "a" -> A | Letters match the word exactly (trivial check) |

### Mechanically Verifiable (string operations)

| Mechanism | Description | Example | Verification |
|-----------|-------------|---------|--------------|
| **first_letter** | First letter of a clue word | "Swedish" -> S | `letters == word[0]` |
| **last_letter** | Last letter of a clue word | "corruption" -> N | `letters == word[-1]` |
| **core_letters** | Middle letters (shell removed) | "priest" -> RIES (from pRIESt) | `letters == word[1:-1]` |
| **outer_letters** | First + last letter | "durable" -> DE | `letters == word[0] + word[-1]` |
| **alternate_letters** | Every other letter | "observe" -> BSR or OEV | Check odd/even positions |
| **deletion** | Word with letters removed | "speed" -> SPEE (curtailment) | `letters` is a substring/prefix/suffix of source word |
| **truncation** | Word shortened (first or last removed) | "almost happy" -> HAPP | Prefix or suffix of source |
| **reversal** | A piece (synonym/abbr) spelled backwards | "retiring gentleman" -> RIS (SIR reversed) | `reverse(letters)` is a valid synonym/abbreviation of the clue word |
| **anagram_fodder** | Letters that get rearranged | "PROUST" -> fodder for STUPOR | `sorted(fodder) == sorted(answer or sub-answer)` |
| **sound_of** | Sounds like another word | "flower" sounds like "floe-er" | Phonetic comparison (limited mechanical check) |

### Composite Verification

For a **charade**, each piece is verified independently by its mechanism, then the assembly is verified by concatenation. The verifier should:

1. For each piece, check the mechanism:
   - synonym: DB lookup
   - abbreviation: DB lookup
   - first_letter: `piece == word[0]`
   - deletion: `piece` is subword of source
   - reversal: `reverse(piece)` is synonym/abbreviation of source
   - etc.
2. Check assembly: `concat(all_piece_letters) == answer`
3. Score based on how many pieces verified vs unverifiable vs wrong

## Deletion Subtypes

Deletions are the most varied mechanism. The indicator word tells you WHAT to delete:

| Subtype | Indicator examples | Operation | Example |
|---------|-------------------|-----------|---------|
| **curtailment** (last letter) | "shortly", "almost", "nearly", "endless" | Remove last letter | SPEED -> SPEE |
| **beheading** (first letter) | "headless", "topless", "after the start" | Remove first letter | CHESS -> HESS |
| **gutting** (middle) | "heartless", "emptied", "hollow" | Remove middle letters | TREND -> TD |
| **shelling** (outer) | "skinned", "peeled", "shelled" | Remove first + last | STORE -> TOR |
| **specific letter** | "losing one", "dropping a" | Remove specific letter(s) | CHAIR -> CHAR (remove I) |

## Common Charade Compositions (from 500 production examples)

| Rank | Composition | Count | Example |
|------|-------------|-------|---------|
| 1 | synonym + synonym | 136 | SCREW + DRIVER = SCREWDRIVER |
| 2 | abbreviation + synonym | 99 | ST + OUT = STOUT |
| 3 | first_letter + synonym | 23 | S + TEAK = STEAK |
| 4 | last_letter + synonym | 18 | N + EVER = NEVER |
| 5 | anagram_fodder + synonym | 16 | (anagram) + WORD |
| 6 | deletion + synonym | 12 | SPEE(d) + (c)HESS = SPEECHES |
| 7 | literal + synonym | 11 | ON + TORY = ONTORY |
| 8 | core_letters + synonym | 10 | (p)RIES(t) + synonym |
| 9 | reversal + synonym | 9 | (reversed) + synonym |
| 10 | alternate_letters + synonym | 8 | (alternating) + synonym |

## What This Means for the Verifier

A robust verifier needs:

1. **Piece-level verification**: For each piece, apply the mechanism-specific check
2. **Assembly verification**: Confirmed pieces must concatenate to the answer
3. **Scoring model**:
   - Each verified piece earns points (amount depends on mechanism reliability)
   - DB-verified pieces (synonym, abbreviation) are strong evidence
   - Mechanically verified pieces (first_letter, deletion, reversal) are ironclad
   - Unverifiable pieces (DB gap) are neutral
   - Wrong pieces (proven incorrect) are heavily penalised
   - Verified assembly is the capstone — if all pieces verified AND they assemble, that's HIGH
