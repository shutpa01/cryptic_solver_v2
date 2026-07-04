# Handover — 2026-06-21 — atom-map solver (atomise.py) + hand-solve

Branch `redesign`, UNCOMMITTED. Read `memory/atom_map_signature_architecture.md`
first — that is the source of truth (user re-confirmed it this session).

## 0. How to behave (the user is right to be wary — read this)
- This session was TOO SLOW and over-claimed repeatedly. The recurring failure:
  saying "fixed/done" without (a) restarting the server, (b) testing through the
  LIVE page the user actually uses. The user now verifies everything himself and
  trusts nothing unverified.
- HARD RULE: after any code change, **restart the wfw_web server** and **curl/observe
  the live page** before claiming anything. Quote the actual served output.
- Do NOT keep bolting operations onto the AUTO-search — it explodes (see §3). The
  agreed split: auto-search does the CLEAN cases; the HAND-SOLVE does the hard ones.
- Agree the design BEFORE coding. Keep it SIMPLE — no special-case pile-ups.

## 1. The agreed architecture (this session)
- ONE atom-map gate: assemble the answer from literals/derivatives, read the
  mechanism off the geometry, confirm with a DB-licensed indicator. No fabrication
  (returns NONE/AMBIGUOUS rather than a guessed parse).
- **Search** = auto-solver: feeds all DB candidates (and a word's own letters as
  literals) through the gate. Limited to clean shapes: charade, container, reversal,
  anagram, double-definition.
- **Hand-solve** = the human links answer atoms to pieces; the gate VALIDATES. This
  is where the harder operations live (selection, awkward/named deletions) — the
  human's click removes the ambiguity so there's no search to explode.
- The cascade is REMOVED from the page (the page now solves via the search).

## 2. BUILT + VERIFIED on the live page this session
All in `core/atomsig/atomise.py` (engine) + `core/wfw_web.py` (page/hand-solve):
- **Double definition** — `search()` DD pre-check: clue splits into two parts that
  both `defines()` the answer, with optional DB link-word(s) in the gap. Verified:
  10074002 "Fight a bit" = SCRAP (and synthetic connector "Fight for a bit").
- **Anagram via literals** — `_viable` keeps equal-length fodder; `run_cands` adds a
  run's raw letters as a `literal` candidate; `_cover` detects anagram (sorted match).
  Verified: 10074325 ATTORNEYGENERAL ("Teeny groan later"), CAFETIERE.
- **Charade / container / reversal** — STARTLING (container, PASS), COMB (charade,
  comes back AMBIGUOUS due to DB noise: doctor=BM + from=reversal), TOPPER.
- **Page integration** — `wfw_web._render_one` calls `_atomsig_card()` (search →
  render). PASS renders the atom-map; NONE/PENDING/AMBIGUOUS render a clean card
  (verdict badge + answer tiles + evidence in a <details>). DD renders a DOUBLE DEF card.
- **Hand-solve** — `/handsolve?id=...` (single id, comma-list, or `A-B` range);
  click answer-atom then piece, re-click to undo (`HANDSOLVE_JS`); `/handsolve/verify`
  runs `verify_placement` and renders the parse. Reached via a "Signature enrichment"
  button on each clue card.
- `verify_placement` (hand-solve gate) handles: identity, reversed, anagram, split
  (container), deletion (behead/curtail/internal), deletion_split (substitution),
  named single-word deletion source. Verified directly on ARCH/SCAR/MARSH/STARTLING/
  TOPPER/UNEATABLE.

## 3. KNOWN PROBLEMS (open)
- **Deletion-overshoot in the AUTO-search EXPLODES.** `_viable` overshoot clause +
  `_cover` charade `for delete in (0,1,2)` + `_confirm_roles` named-source were added
  so LIFER-type clues (FILM→ drop M) could auto-solve. Result: LIFER returns **133
  legitimate maps** (mostly garbage: male=BEER, etc.). Too permissive. Per the agreed
  split, deletion of this kind belongs in HAND-SOLVE, not the auto-search — strongly
  consider REVERTING the auto-search overshoot (the `for delete` loop in `_cover`,
  the overshoot clause in `_viable`, the named-source in `_confirm_roles`) so the
  auto-search stays clean, and rely on hand-solve for these. (verify_placement's
  deletion is separate and fine.)
- **Answer colouring render bug** on solved atomsig cards (tiles not coloured by
  source). Display-only; not yet fixed.
- COMB-style AMBIGUOUS from DB noise (doctor=BM, from=reversal) — disambiguation/
  DB-cleanup question, unsolved.

## 4. NEXT TASK (was being agreed when the thread ended) — SELECTION in hand-solve
Clue 10074429 "Reject fiancée, essentially following incentive" = SPURN
  = `incentive`→SPUR + `fiancée essentially`→N (middle letter of FIANCEE);
  `following` = ordering indicator, `Reject` = definition.

AGREED design (from original spec SEL_F/SEL_I, rules first/last/middle/outer/alternate):
- Selection lives ONLY in the hand-solve.
- **Proposed (was awaiting final yes): ONE TILE PER WORDPLAY WORD** (the word itself,
  not word=value). User links an answer atom to the word; the system DERIVES the value
  + mechanism (DB synonym/abbr, OR the word's own letters under a selection rule).
  Definition detected and shown separately, not a tile. This replaces the current
  word=value tile model in `_handsolve_block`.
- Gate: `_selection_rule(value, seg)` → first/last/middle(odd-length single centre)/
  outer/alternate; require a selection indicator.
- TWO open decisions for the user:
  1. Selection-indicator typing: `essentially/heart/centre/middle` are DB-typed
     `deletion`/`parts`, not `selection`. Accept `parts` as licensing selection (zero
     DB work) OR add a `selection` type and re-type them.
  2. Ordering/join words like `following`: accept ANY DB-indicator leftover word as
     accounted (role=indicator) so it doesn't block the parse. (Recommended yes.)

## 5. Key files / entry points
- `core/atomsig/atomise.py`: `search()` (auto), `verify_placement()` (hand-solve gate),
  `find_maps()`/`_cover()` (tiler), `_viable()`, `_needs()`, `_confirm_roles()`,
  `build_handsolve_parse()`, `_selection_rule` (TO ADD).
- `core/wfw_web.py`: `_atomsig_card()` (page solver), `_render_one()` (cascade removed),
  `/handsolve` + `_handsolve_block()` + `HANDSOLVE_JS`, `/handsolve/verify`.
- Run server: `.venv/Scripts/python.exe -m core.wfw_web` → http://127.0.0.1:5099/
  Page: `/?id=<ids>`  Hand-solve: `/handsolve?id=<ids or A-B>`
- Test clues: 10074002 DD(SCRAP), 10074325 anagram(ATTORNEYGENERAL), 10074321
  deletion(RIO), 10074429 selection(SPURN), 10074272 multi-word(MEDICI), 10074344
  charade(CHEEK).
