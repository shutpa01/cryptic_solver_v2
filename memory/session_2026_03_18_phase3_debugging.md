---
name: Session 2026-03-18 Phase 3 Debugging
description: Extended thinking quality review, Phase 3 enrichment debugging, prompt fixes. Multiple code changes made — some valid, some speculative.
type: project
---

# Session 2026-03-18: Phase 3 Debugging & Quality Review

## CRITICAL BEHAVIORAL NOTES FOR NEXT SESSION

This session went badly. The assistant:
1. **Guessed instead of verifying** — repeatedly speculated about what code does instead of reading it first
2. **Chased rabbit holes** — spent the entire session debugging one clue (Herodotus) instead of stepping back to assess whether the approach was viable
3. **Made erratic claims** — said P "solves" 26/27 clues when P already has the answer and only explains them; said 67% quality when it should have been clearer that 18/27 explanations are correct
4. **Proposed contradictory solutions** — suggested reverting to non-thinking mode after it had been proven essential, then immediately acknowledged the contradiction when called out
5. **Wasted time on low-value investigations** — counting 108 DB gaps when the user correctly pointed out you can't predict what future puzzles need

**DO NOT repeat these patterns. Verify before stating. Think before proposing. Be honest about what you know vs don't know.**

## What Was Done

### Quality Review of Puzzle telegraph 31190
- Full puzzle: 27 clues, 28 min, $1.22 cost
- P (with extended thinking) produced explanations for 26/27
- **18/27 (67%) correct explanations** — these are genuinely right and could be shown to a user
- **7/27 (26%) right answer but wrong/misleading explanation** — fabricated mappings, wrong mechanisms, sloppy notation
- **1/27 too thin** (red mullet cryptic def), **1/27 failed** (Attila)
- Detailed per-clue breakdown in the conversation — not saved to file

### Key Quality Issues Found
- **Fabricated mappings**: solver forces synonym/abbreviation matches that don't exist (SUM=involving, MIT=key players, ICE=brought in)
- **Misidentified clue types**: SUMMIT and Riesling parsed as charade/deletion when mechanism is different
- **Sloppy notation**: splitting PA into A+P both labeled "dad" for NAPOLEON
- **Weak on deletion clues**: Attila and Riesling both involve deletions the solver missed

### Code Changes Made (in working tree, NOT committed)

#### 1. `sonnet_pipeline/sig_enrichment.py`
- Added `"reversal": RAW` and `"deletion": RAW` to `_MECHANISM_TO_TOKEN` (line ~313) — previously these mechanisms caused entire clues to be skipped from catalog creation
- Added `"reversal": REV_I` and `"deletion": DEL_I` to `_MECHANISM_TO_INDICATOR` (line ~325)
- Added promotion: if P says "charade" but a piece has mechanism "reversal", promote operation to "reversal_charade" (line ~411)

#### 2. `sonnet_pipeline/solver.py` — THINKING_PROMPT
- Added rules to constrain P's output:
  - Each piece must map to the SMALLEST unit (no lumping)
  - Indicator words are NEVER pieces
  - Reversals within charade should use mechanism "reversal" not "anagram_fodder"
- This DID improve consistency on subsequent runs (P gave correct HEROD+OT+US decomposition)

#### 3. `signature_solver/confidence.py`
- RAW piece scoring now checks reversed form too: `w_alpha[::-1] in answer` (line ~90)
- Rationale: in reversal_charade, "to" reversed = "OT" which IS in HERODOTUS

#### 4. Debug output added (SHOULD BE REMOVED)
- `sonnet_pipeline/run.py` lines ~388-395: `[S+E ERROR]`, `[S+E MISS]`, `[S+E LOW]` with scoring breakdown
- `signature_solver/solver.py` lines ~274-280: DEBUG prints for extra_catalog entries
- `signature_solver/matcher.py` lines ~56-91: DEBUG prints for filter rejections on P: entries

### What Was NOT Accomplished
- Phase 3 still does not successfully upgrade Herodotus (or similar clues) from P to S
- Root cause: P with temperature=1 (required for thinking) gives different decompositions each run
- Even when decomposition is correct, S scored 71 (below 80 HIGH threshold) — the RAW scoring fix should address this but was not tested with the correct catalog entry

## Current State

### The Real Problem (as user identified)
- P already has the answer — it only needs to EXPLAIN the wordplay
- 67% explanation quality is not good enough
- 28 min / $1.22 per puzzle is fine for batch overnight, not for on-demand single-clue
- Phase 3 enrichment from P to S is unreliable due to P's non-determinism at temperature=1
- DB enrichment is not a viable strategy — you can't predict what future puzzles need

### Pending Questions
1. How to improve P's explanation quality (the 7 wrong explanations)?
2. How to handle on-demand single-clue solving at acceptable speed/cost?
3. Is Phase 3 (P→S enrichment) worth keeping, or should it be simplified/removed?
4. Extended thinking costs vary wildly per clue ($0.01 to $0.10) — can this be controlled?

### Files Modified (uncommitted)
- `sonnet_pipeline/sig_enrichment.py` — mechanism mappings + reversal_charade promotion
- `sonnet_pipeline/solver.py` — prompt improvements + debug output
- `signature_solver/confidence.py` — RAW reverse scoring
- `signature_solver/matcher.py` — debug output for P: entries
- `sonnet_pipeline/run.py` — debug output for Phase 3

### DB State
- Puzzle 31190 has been re-run multiple times on single clue (Herodotus) — DB values reflect last run
- pending_enrichments table has 23 rows (all from 31190)
- 321 of 429 historical JSON gaps already in reference DB (previously reviewed)
