# Signature Solver — Continuity Summary (2026-03-12)

## What We Built

A `signature_solver/` package — a two-stage cryptic crossword solver:

1. **Stage 1 (Mechanical)**: Zero-cost constraint satisfaction using DB lookups. Tokenizes clue words into roles (synonym, abbreviation, indicator, etc.), tries candidate signatures mechanically.
2. **Stage 2 (API fallback)**: For clues the mechanical solver can't handle with HIGH confidence, calls Claude API with structured DB evidence.

### Key Files Created/Modified

- `signature_solver/solver.py` — Mechanical solver. Returns `SolveResult` with confidence scoring (0-100). Tries all strategies, keeps best. Strategies: charade, container, reversal, anagram, hidden, deletion, trim, mixed_anagram, charade_with_reversal.
- `signature_solver/confidence.py` — Scoring system. HIGH (80+) = serve directly, MED (50-79), LOW (<50). Strategy base scores, indicator quality, synonym verification, circularity checks.
- `signature_solver/evidence.py` — Formats word analyses as structured evidence for API. Shows each word's possible roles from DB (synonyms, abbreviations, indicators).
- `signature_solver/api_solver.py` — Single API call with constrained JSON prompt. Two-layer validation: Layer 1 (letter verification), Layer 2 (DB lookup verification).
- `signature_solver/test_api.py` — Full pipeline test: mechanical first, API fallback for non-HIGH.
- `signature_solver/test_puzzle.py` — Mechanical-only test against full puzzles.
- `signature_solver/test_run.py` — Unit tests against known clue set.
- `signature_solver/word_analyzer.py` — Modified: ALL synonyms ≤4 chars kept (uncapped), longer synonyms capped at 20.
- `signature_solver/db.py` — RefDB class. Methods: `get_synonyms(word, max_len)`, `get_abbreviations(word)`, `get_homophones(word)`, `get_indicators(word)`.

## The Problem We Hit

### Old pipeline (sonnet_pipeline/) vs New approach (signature_solver/)

**Old pipeline on puzzle 31171**: 32/32 solved, 32/32 approved (100%)
**New approach on puzzle 31171**: 6 mechanical + 5 API = 11/31 (35%)

The new approach is dramatically worse despite providing MORE evidence to the API.

### Root Cause Analysis

Three separate failure modes in the new approach:

1. **API produces wrong letter decompositions (10/25 calls)** — The structured evidence format may be *constraining* the AI too much. The old pipeline lets the AI reason freely with full world knowledge and it solves everything. The new approach forces it to select from DB lookups, which limits it.

2. **DB validation layer rejects correct answers (7/25 calls)** — The API correctly identified mappings like `Medic→DOC`, `relative→MA`, `too→TO`, `got→O` but our 1.3M synonym DB doesn't contain these mappings. Layer 2 rejects them as "bogus lookups" when they're actually correct crossword knowledge.

3. **API returns invalid JSON (3/25 calls)** — max_tokens=500 sometimes insufficient for complex clues.

### Test Results With Both Validation Layers

| Puzzle | Mechanical HIGH | API OK | API Failed | Total Solved |
|--------|----------------|--------|------------|-------------|
| 31171  | 6              | 5      | 20         | 11/31 (35%) |
| 31174  | 4              | 7      | 19         | 11/30 (37%) |

### Layer 2 (DB Validation) Impact

Without Layer 2: 10 MECH + 27 API = 37/61 (61%) — but includes false positives
With Layer 2: 10 MECH + 12 API = 22/61 (36%) — but rejects some correct answers

Examples of correct answers killed by Layer 2:
- `Medic→DOC` (valid crossword synonym, not in DB)
- `relative→MA` (mother = MA, not in DB)
- `too→TO` (obvious equivalence, not in DB)
- `Nipper→CRAB` (valid, not in DB)

Examples of genuinely bogus lookups correctly caught:
- `crude→NELEGANT` (not a real word)
- `sitting→SAND` (not a synonym)
- `fashion→N` (not an abbreviation)
- All NAAN lookups (all fabricated)

## The Fundamental Question

The old pipeline already works well (100% on this puzzle). It:
- Gives AI the clue + answer
- Lets it reason freely with full world knowledge
- Gets everything right
- But costs more tokens per call (full reasoning)

The new approach was meant to:
- Reduce cost by pre-computing evidence
- Improve accuracy by constraining to DB-verified options
- But it actually makes results WORSE because:
  - Constraining to DB options limits what the AI can find
  - DB has gaps that kill correct answers
  - The structured JSON format is harder for the AI to get right

**The user's question**: Why are we getting worse results with more evidence? Is the entire approach flawed?

## Architecture Context

- `data/clues_master.db` — All clues (507k), single source of truth
- `data/cryptic_new.db` — Reference tables (synonyms: 1.3M pairs, abbreviations: 2.1k, indicators: 4.6k, homophones: 1k)
- Old pipeline: `sonnet_pipeline/` — Working, don't modify
- New solver: `signature_solver/` — Experimental, this is what we're working on
- API model: `claude-sonnet-4-20250514`, temperature=0, max_tokens=500
- Assistant prefill `{"role": "assistant", "content": "{"}` forces JSON output

## What NOT To Do

- NEVER modify `sonnet_pipeline/` or `stages/` (working pipeline)
- NEVER delete files without confirmation
- NEVER run the old pipeline (costs money, user will do it themselves)
- NEVER suggest serving scraped blog explanations
- Code goes in `signature_solver/`, NOT project root

## Possible Next Directions (Not Yet Decided)

1. **Abandon the constrained approach** — Let the API reason freely like the old pipeline but with evidence as *additional context* rather than *constraints*
2. **Fix DB gaps** — Add missing mappings (Medic→DOC, relative→MA, etc.) to improve Layer 2
3. **Relax Layer 2** — Only reject if the lookup is demonstrably wrong, not just missing from DB
4. **Hybrid** — Use mechanical solver for what it's good at (anagrams, hidden words), fall back to unconstrained API for the rest
5. **Focus on cost reduction** — The real value might be reducing the 6 mechanical HIGH solves' cost to zero, and accept the API handles the rest at similar cost to the old pipeline

The user needs to decide the direction. The data clearly shows the constrained-evidence approach underperforms the free-reasoning approach.
