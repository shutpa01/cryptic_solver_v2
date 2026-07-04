# Handover — 2026-06-04 — anagram family of engines + shared substrate

## Status: GOOD. Productive session, all committed (redesign branch, NOT pushed).

Branch `redesign`. The true-test page is `python -m core.wfw_web` on
http://127.0.0.1:5099/ (enter a clue id; it re-solves live and persists; each card
shows "CLUE ID n"). debug is off, so RESTART the server after any code change or DB
cleanup to pick it up.

## What's built (core/, the redesign clean engines)
Cascade order in core/engine_registry.solve(): hidden -> anagram -> charade ->
anagram_charade -> anagram_container -> DOUBLE DEFINITION (LAST) -> most-complete fail.
Each stops on pass/pending; a fail falls through.

- hidden, double-definition (dd) — quick checks (pre-existing).
- charade — EVIDENCE-DRIVEN tiler: DB synonym/abbreviation pieces (multi-word phrases
  ok, never raw letters), residue classified last.
- anagram — wordplay-only, fodder by exact letter-match + spaCy POS segmentation;
  interior link-skip ("old and new"); contraction fodder ("Lionel's"->LIONEL).
- anagram_charade — charade with one anagram piece (CUTHBERT, VIOLET, HEGELIAN).
- anagram_container — container with one anagram component, inner OR outer
  (SANDWICHES, PANGOLIN, EXHORT). Needed wiring["lookup_all"] (UNFILTERED — the outer
  value isn't a substring of the answer).
- Each engine has its own thin screen (core/*_screen.py); wfw_web SCREENS dispatch on
  parse.operation.

## Shared substrate (the cross-cutting, single-home pieces)
- core/wordplay.py — POS sets, raw, is_anagram_indicator, is_link_or_glue,
  fodder_letter_forms, adjacent_run, anagram_indicator_source. (Extracted from 4x
  duplication; the engines import these. TYPE-SPECIFIC assembly stays per-engine.)
- core/definition_engine.py — find_definitions (the ONE definition stage; engines get
  the wordplay, never choose the definition); is_dbe peel; dbe_annotation.
- core/grammar.py — spaCy pos_tags (contraction-aligned) + definition-extent.
- core/inflect.py (suffer=suffers), core/contractions.py (Lionel's->Lionel) —
  inflection + contraction matching, combined in engine_registry._match_variants and
  used by the indicator/definition/synonym lookups.
- core/engine_registry.py — make_db_wiring (injected predicates: defines, lookup,
  lookup_all, indicator_types, is_link, is_dbe, ai_is_definition, define_fallback,
  store, the templates; O(1) live indexes + caches); solve() cascade;
  _solve_wordplay_engine (def stage -> hand wordplay to engine -> attach def);
  _most_complete (measured fail selection).
- core/store.py — wfw_solve/wfw_piece/wfw_link substrate; wfw_solve.template_id is the
  clue<->signature cross-reference (design SS10).

## Hard-won PRINCIPLES (do not regress — the user enforced these)
1. NEVER pre-assign link words. Links are RESIDUE, classified LAST from positive
   evidence. Find pieces first (letters/lookup); whatever's left is then classified
   (is_link or POS function/VERB/ADV -> link; else unaccounted -> honest fail).
   [[feedback-never-preassign-links]]
2. Engines work on WORDPLAY only; the definition is decided upstream.
3. Never assign roles on a FAIL / by elimination. [[feedback-no-role-assignment-on-fail]]
4. Inflection + contraction + definition-by-example handled in the shared lookup layer
   so every engine inherits them.
5. No backticks / no blue / no coloured text in chat. Number options. Be brief.
6. Verify through the real path with output shown before claiming done; regression-check.

## This session's commits (redesign, unpushed)
- fb0119d4 catalog-driven charade + anagram + anagram_charade engines (+ signature
  cross-ref wfw_solve.template_id, design SS10)
- 7d5518a3 charade + anagram_charade evidence-driven, no preassigned links
- 82a3388b anagram_container engine
- 0cb96cb6 shared core/wordplay.py + missing-anagram-indicator fallback
- 0b9e87af cascade: run double-definition LAST

## Missing-anagram-indicator fallback (evidence-only)
When an anagram is proven by the letter-match + the definition is found but the
indicator isn't in the DB, the leftover word adjacent to the anagram piece is a
PROVISIONAL indicator (pending), queued via indicator_enrichment. All 3 anagram
engines. Prefer db placements (best-tracking) so provisional never beats confirmed.
Fixed 2076094 (LONG TIME NO SEE), 2076158 (FANCIFUL). [[missing-indicator-fallback]]

## Reference-DB cleanups this session (cryptic_new.db, all reversible w/ backups)
[[db-cleanups-2026-06]]: DBE definition strip (13,609 rows, bak
definition_answers_augmented_bak_dbe); stale DBE anagram tags removed; IN-as-word
synonym pairs deleted (223, bak synonyms_pairs_in_word_bak). Plus DB synonym adds
beach->SANDS, suffering->PAIN (uncommitted, local). The data DBs are gitignored, so
DB changes are NOT in git — they live in the local DBs only.

## Open / next
- More clue types still to build (per [[catalog-13-isolated-engines]]): reversal,
  deletion, homophone, acrostic, base container (no anagram), reversal_charade,
  container_charade, reversal_container. Build evidence-driven, links last.
- The "perhaps/parts" indicator tag (id 7367) still present; "say"/"for example"
  without a comma left in some definitions (conservative strip rule).
- Scratch files still in core/ (_charade_test.py etc.) + bogus pending row — the
  2026-06-03 handover's cleanup list, still not done (needs user OK to delete).
- Per-puzzle render switch (SOLVER_REDESIGN.md SS13) for the live cutover.
- Measurement: run a real sample per clue-type for honest pass rates.

## How to resume
Read MEMORY.md (auto-loaded) — the 2026-06-04 entries cover all the above. Start the
server, run a few of the verified clues (SOSO, INSTANCES, EDISON, NIELLO, VIOLET,
HEGELIAN, SANDWICHES, EXHORT, NEEDLEWORK, FANCIFUL, GARDENIA, INTIMATE), then take the
next clue or clue-type from the user.
