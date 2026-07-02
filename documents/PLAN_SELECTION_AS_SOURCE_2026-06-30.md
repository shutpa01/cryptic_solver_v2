# Plan — "selection as a SOURCE" (the insertion mirror of deletion) — 2026-06-30

Driven by Times 29582, which we did poorly on. User's insight: *we can take a selected letter
and DELETE it, but we have no symmetric way to take a selected letter and USE it (insert it, or
lay it down as a charade tile).* This plan makes letter-selection a first-class **source**, the
mirror of the selection-**target** machinery we already have.

NOT a single-clue effort: the pattern recurs (2–3× in this one puzzle; pervasive in Times/
Guardian). We will NOT build a per-clue engine.

## Principle
A letter-SELECTION (first / last / outer / middle / alternate of a word — `core/selection.py`)
should be usable as a piece SOURCE in exactly the two ways a synonym/abbreviation already is:
  (a) inserted into an outer (a container INNER), and
  (b) laid down as a charade TILE.
Today selections can only be DELETED (anagram_selection_deletion, container_deletion_selection)
or appear in narrow spots (multi-word acrostic). The symmetry is missing.

## What we reuse (no new primitive needed)
- `core/selection.py` — `select_span(ctx, token, rule)` / `match_span(...)` already produce
  answer-driven, per-letter-sourced selections for first/last/outer/middle/alternate/remove_*.
- `selection_indicators.SUBTYPE_RULE` — maps DB indicator subtypes -> rule (first/last/outer/
  middle/alternate). The selection indicators are DB-typed already.
- `container_deletion_selection_engine.py` — TEMPLATE. It inserts a single-word selection inner
  (RIVEN = RIEN ∋ V[valuables primarily]); we copy its tight, triple-gated, answer-driven design,
  dropping the outer-deletion and adding charade composition.

## Scope — which 29582 fails this targets
Phase 1 (insertion):   COPSHOP, TENDERHEARTED            (selection inserted into a synonym)
Phase 2 (charade tile): BENT, HOTEL, PARLOURS            (single-word selection as a charade tile)
Related (separate):    ARTHURIAN (insertion into a DELETED outer — needs outer-deletion, Group C)
Explicitly OUT of scope here: CICERO/DOBATTLE/MANKINI (deletion-in-charade), TROOP (indirect
deletion from a synonym — we forbid that), ONION (substitution), EDGIEST/STARTUP (anag/data),
CODEINE (constructed homophone), LOCUTORY (container + hollow tile).
Honest expectation: this clears ~5 of 15. The rest are separate decisions.

## Phase 1 — `charade_container_selection_engine` (the priority; your hypothesis)
A charade with ONE container piece whose INNER is a single-word letter-SELECTION and whose OUTER
is a plain DB value; remaining answer covered by ordinary value tiles.
  COPSHOP       = [COSH(truncheon) ∋ P(policeman's, first)] + OP(work)        — &lit def
  TENDERHEARTED = TENDER(offer) + [HEATED(warm) ∋ R(our, last)]
Design (mirror of container_deletion_selection + container_charade):
  - tile the answer left-to-right: value pieces (SYN/ABR/raw) + EXACTLY ONE container piece;
  - container piece = OUTER(DB value of a run) split around INNER = the EXACT letters a
    selection rule takes from ONE disjoint word, the rule licensed by an ADJACENT selection
    indicator; inner strictly interior (true container);
  - require >=1 container/insertion indicator AND the selection indicator in the residue;
    every other word a DB link; answer-driven exact reconstruction; PASS-only; role_validity.

## Phase 2 — single-word selection as a charade TILE
BENT = BE(biddable, outer) + NT(nationalist, outer); HOTEL = HOT(in) + EL(Medellin, middle);
PARLOURS = PA(old man) + R(referencing, first) + LOURS(scowls).
`charade_acrostic` already does MULTI-word first/last tiles but gates out single words (b=a+2),
and the live charade is signature-driven (SEL_F) so it needs a mined signature per role-pattern.
DECISION FOR USER (Phase 2 only):
  Option A — bespoke evidence engine `charade_selection_engine`: free-tiles single-word
    selections (first/last/outer/middle) + DB values, each selection gated by an adjacent DB
    selection indicator, answer-driven, PASS-only. General, no catalog churn. Higher false-pass
    surface (more tiles) -> strict gating + big A/B.
  Option B — add catalog SIGNATURES for the specific SEL_F role-patterns. Lower risk per add,
    but it's the "missing-variant treadmill" (one sig per shape) the handover already flagged.
Recommendation: A, for the same reason we built the other bespoke engines — but only after
Phase 1 ships clean.

## FALSE-PASS RISK — the make-or-break (single-letter inserts are the riskiest thing we do)
A 1-letter selection can land in many positions; ungated, this fabricates. Guards (copied from
container_deletion_selection's discipline, which is the tightest we have):
  1. require the insertion/container indicator AND the matching selection indicator, both DB-typed;
  2. the selection indicator must be ADJACENT to the selected word and its rule must be the one
     the DB licenses (named rule only — no generic widening);
  3. answer-driven: inner = EXACT rule output at its exact answer span; outer = exact DB value
     split around it; whole charade reconstructs the answer exactly;
  4. every non-piece, non-indicator word must be a DB link (no free content words);
  5. PASS-only return (never displaces a simpler engine's pending/fail);
  6. `role_validity` on every recorded role.
Even so: MANDATORY full-corpus A/B before keep. Watch PASS LOST + any new PASS that is a
coincidental tiling. If Phase 1's A/B shows ANY fabricated pass, tighten or shelve before Phase 2.

## Cascade placement
After the existing container-compound engines and after charade_acrostic / container_charade
(more specific, more expensive; cheap pre-gate = both indicators present). PASS-only so order is
safe. Phase 2 engine, if built, sits just after Phase 1.

## Verification
- Per engine: prove the target clues solve through the FULL cascade + persist + render; confirm
  the template clues (RIVEN etc.) still pass (before/after toggle).
- Then the overnight full-corpus A/B (`core/_ab_general.py`): 0 PASS lost, 0 exceptions, and
  hand-verify every PASS GAINED is faithful (letter-exact). This is the keep/rollback gate.

## Build order / sign-off
SIGNED OFF 2026-06-30: Phase 2 = **Option B (signatures)**. Do Phase 1 + Phase 2 now; the
other 29582 buckets afterwards.
1. [in progress] Build Phase 1 engine `charade_container_selection`; verify COPSHOP +
   TENDERHEARTED through full cascade + persist + render; confirm RIVEN/template clues unchanged.
2. Phase 2: add catalog SIGNATURES for BENT (SEL_F outer + SEL_F outer), HOTEL (SYN_F +
   SEL_F middle), PARLOURS (SYN_F + SEL_F first + SYN_F) via catalog_creator (server STOPPED,
   DB Browser CLOSED). Keep only faithful passes.
3. Combined full-corpus A/B over ALL session changes (0 lost, 0 exceptions, gains hand-verified).
4. Then work through the remaining buckets (ARTHURIAN, CICERO/DOBATTLE/MANKINI, TROOP, ONION,
   EDGIEST/STARTUP, CODEINE, LOCUTORY).
