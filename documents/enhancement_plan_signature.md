# Signature Solver Enhancement Plan

## Based on: Full analysis of all 32 clues, puzzle 31185

---

## What's Actually Blocking Signature (17 failures, ZERO definition gaps)

Every failure passed the definition gate. All blockers are in matching/execution.

### Fix Type 1: Missing DB Synonyms (5 clues)

| Clue | Missing Entry | Unlocks |
|------|--------------|---------|
| 4D POTTAGE | footman → PAGE | Container: OTT in PAGE |
| 2D SANCTIONS | moves → ACTIONS | Container: S + N inside ACTIONS |
| 27A EGLANTINE | border → LINE | Container: ANT in EG+LINE |
| 20D SALTIRE | cure → SALT | Charade: SALT + IRE |
| 7D BLIMP | deflated → LIMP | Charade: B + LIMP |

### Fix Type 2: Missing Abbreviation (1 clue)

| Clue | Missing Entry | Unlocks |
|------|--------------|---------|
| 3D EXTRA | sweetheart → E | Charade: E + X + TRA |

### Fix Type 3: Missing Indicator + Link Words (3 entries across 3 clues)

| Clue | What's Needed | Why |
|------|--------------|-----|
| 27A EGLANTINE | `fronting` as POS_I_FIRST indicator | Currently has no role, blocks matching |
| 19A LUSTRES | `soft` added to LINK_WORDS | Leftover word rejected by _remaining_are_valid |
| 20D SALTIRE | `getting` added to LINK_WORDS | Same leftover rejection |

### Fix Type 4: Code Bugs (2 clues)

| Clue | Bug | Fix |
|------|-----|-----|
| 28A TRIPE | Possessive `'s` corrupts positional extraction. `"Argument's"` last letter = S not T | Strip `'s` before calling `extract_positional` in `_lookup_slot` |
| 3D EXTRA | Reversal_charade: when no I-slots exist, `assigned_ind_types` = None, so leftover REV_I indicators rejected | Seed `assigned_ind_types = {ind_type_raw}` when ind_type_raw is set but no I-slots |

### Fix Type 5: Missing Catalog Pattern (1 clue)

| Clue | What's Needed |
|------|--------------|
| 5D CURDLED | `positional_charade` F+F+I+F pattern (currently only F+I+F, I+F+F, F+F+I) |

### Fix Type 6: Structural — &lit Support (1 clue)

| Clue | Issue |
|------|-------|
| 22A SWAMI | Acrostic of ALL words including definition: S+W+A+M+I. Def gate consumes "Sage", leaving only WAMI. Need fallback: try full clue as acrostic when normal matching fails. |

### Fix Type 7: Unsupported Operations — Homophone (3 clues)

| Clue | Operation Needed |
|------|-----------------|
| 16D ARRAIGN | Homophone: "a reign" sounds like ARRAIGN |
| 21A LESSENS | Homophone: LESSONS sounds like LESSENS |
| 25D EATEN | Homophone: ETON sounds like EATEN |

### Fix Type 8: Unsolved — Need Human Investigation (2 clues)

| Clue | Status |
|------|--------|
| 11D OPPOSES | Parse unclear — likely needs slang `friends → OPPO`. |
| 30A PERSONALLY | Complex multi-part: PERSON + ALLY with ER embedded. |

---

## Production Failures — How Close Was AI?

| Clue | AI Quality | Notes |
|------|-----------|-------|
| 21A LESSENS | **Perfect** | Homophone LESSONS→LESSENS. Failed on DB definition gap only. |
| 25D EATEN | **Perfect** | Homophone ETON→EATEN. Same — pure DB gap. |
| 20D SALTIRE | **Very close** | Right mechanism, right indicator. Used ANGER instead of IRE (which was in DB). |
| 22A SWAMI | Partial | Found W and I but missed multi-word acrostic. |
| 7D BLIMP | Lost | Never tried B+LIMP. |
| 30A PERSONALLY | Partial | Found ALLY and ER but couldn't assemble. |
| 11D OPPOSES | Lost | Right indicator, wrong letter counts. |

---

## Proposed Execution Order

### Phase 1: DB Inserts (no code, immediate impact)
5 synonyms + 1 abbreviation + 2 link words + 1 indicator.
Unlocks: POTTAGE, SALTIRE, BLIMP, LUSTRES. Partial progress on SANCTIONS, EGLANTINE, EXTRA.

### Phase 2: Bug Fixes (2 small surgical changes)
- Possessive stripping before `extract_positional` → unlocks TRIPE
- Reversal_charade `assigned_ind_types` fix → unlocks EXTRA (with Phase 1 abbreviation)

### Phase 3: Catalog Pattern
- Add positional_charade F+F+I+F → unlocks CURDLED

### Phase 4: &lit Support
- Fallback: try full clue as acrostic when normal matching fails → unlocks SWAMI

### Phase 5: Homophone Operation
- New operation. DB has homophones table already. Need matcher integration.
- Unlocks ARRAIGN, LESSENS, EATEN

### Phase 6: Human Investigation
- 11D OPPOSES, 30A PERSONALLY

---

## Projected Impact

| Phase | New Solves | Cumulative | Rate |
|-------|-----------|------------|------|
| Current | 15 | 15/32 | 47% |
| Phase 1 | +3 to +5 | 18-20/32 | 56-63% |
| Phase 2 | +2 | 20-22/32 | 63-69% |
| Phase 3 | +1 | 21-23/32 | 66-72% |
| Phase 4 | +1 | 22-24/32 | 69-75% |
| Phase 5 | +3 | 25-27/32 | 78-84% |
| Phase 6 | +0 to +2 | 25-29/32 | 78-91% |

---

## Production Gap Detection — Problems Found

### Fixed: Container outer substring check (report.py)
The gap detector used `letters_clean not in answer_clean` (contiguous substring). This silently dropped container outer pieces where the letters are split (e.g. PAGE in P-OTT-AGE). Fixed to letter-count subset check. Now catches footman→PAGE, moves→ACTIONS.

### Remaining: AI fails to decompose when it finds a shortcut
**EGLANTINE example**: The AI was given `say: abbr=EG*` (starred/confirmed), `trapping: ind=container`, `insect: syn=ANT*` (starred). It correctly found EGL(ANT)INE but lumped "fronting border" as opaque anagram fodder `EGLINE` instead of decomposing into EG (say) + LINE (border). It never used the starred `EG` lookup it was given.

**Root cause**: The reasoning prompt (solver.py line 769) tells the AI to "TRY THESE LOOKUPS as letter contributors" but doesn't enforce per-word accounting. The AI can take shortcuts — lumping words together, fabricating intermediate steps — as long as the final answer checks out. When it finds a path to the answer, it stops decomposing.

**Impact**: Pieces the AI never identifies as separate synonyms are never proposed as DB inserts. For EGLANTINE, `border → LINE` was never proposed because the AI never separated it from EG.

**This affects at least 3 of the 4 unproposed DB gaps**:
- border → LINE (EGLANTINE): AI lumped with EG instead of decomposing
- cure → SALT (SALTIRE): AI went down wrong path (anagram), never tried charade
- deflated → LIMP (BLIMP): AI completely lost, never found the pieces

**Potential fix**: Strengthen the reasoning prompt to require explicit per-word letter accounting. Every non-indicator word must have its letter contribution explained individually, using starred lookups where available. This would force decomposition and surface more DB gaps. However, this is a prompting change to P which is being superseded — the better long-term fix is S's mechanical decomposition which does this by design.

### Remaining: core_letters mechanism excluded from gap detection
`sweetheart → E` was returned by AI as mechanism `core_letters`, which is in the skip list (report.py line 351-352). This means positional extractions that happen to be abbreviations are never proposed as DB inserts. Could be fixed by removing `core_letters` from the skip list, but risk is low-value noise since most core_letters extractions are mechanical (not DB-worthy).

---

## Wiring S to Supersede P

S and P already share the same DB tables and answer source. The merge path:

1. **S runs first** (zero cost, instant). If HIGH → serve.
2. **P's API runs only for S failures**. Feed discoveries back as DB suggestions for review.
3. **Report pipeline unified** — S results and P-fallback results flow through same report format for dashboard/review queue.
4. **Scoring**: S's mechanical confidence is primary. P's score only for API-fallback clues, clearly flagged as AI-generated.
5. **Enrichment unification**: S's `word_analyzer.py` and P's `enricher.py` do overlapping work — can be unified later, not a blocker.
