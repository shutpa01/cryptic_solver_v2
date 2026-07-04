# State note — 2026-05-31 — Hidden engine (WFW redesign)

Honest status. BUILT/PROVEN vs NOT BUILT is stated explicitly so nothing is
assumed done that isn't. Branch: redesign. master untouched. Nothing committed
this session — all new core/ files are untracked working changes.

## BUILT AND PROVEN (run against the real DB, output shown)

- core/wfw_atoms.py — character-level atomiser (revived from archived commit
  5b539280). Numbers every character of clue+answer. Test: core/test_wfw_atoms.py
  passes.
- core/wfw_model.py — Source / Link / Annotation / Parse. Parse carries status
  ('pass'/'fail') + warnings + enumeration() + unexplained_words().
- core/definition_engine.py — universal definition finder. DB-confirms an edge
  definition, then grows it to full grammatical EXTENT via core/grammar.py
  (spaCy, extent-only). Proven: ANDEAN grows DB's bare "mountains" ->
  "in the mountains".
- core/grammar.py — narrow spaCy helper, definition-extent only.
- core/hidden_engine.py — the hidden engine, complete:
  - triggered by the hidden RUN (not an indicator);
  - per-character provenance (each answer letter -> exact clue char);
  - definition via the universal engine;
  - indicator identified FIRST (absorbs adjacent both-lists words like "in"
    into "hidden in"); link words classified LAST; gaps surfaced honestly;
  - ENGINE-LEVEL verification (_verify_hidden): 5 rules -> status pass/fail +
    plain warnings. NO grand verifier, NO score.
- core/hidden_screen.py — hidden's screen change: lights host letters in the
  clue line, renders UNCOLOURED on the shared base.
- core/wfw_render.py — the BASE screen (clue line, tiles, per-word breakdown
  with each word's role, verdict). coloured=True/False.
- core/engine_registry.py — orchestrator: runs a clue through every engine that
  exists (only hidden now). make_db_wiring() builds the injected DB predicates.
- core/wfw_web.py — true-test UI, Flask port 5099. Enter a clue id -> runs all
  engines -> shows the screen + PASS/FAIL. Real clue ids: 1710239 BRIE,
  1710238 ANDEAN (PASS), 1710279 TIGER (PASS), 1710251 SARONG (FAIL),
  1710240 PINKO (FAIL).

Proof helpers: core/prove_wfw_model.py, prove_definition.py, prove_hidden.py.

## NOT BUILT (discussed, agreed, but NOT done — do not assume otherwise)

- **Haiku definition fallback — NOT WIRED.** This is why SARONG (1710251) fails
  with "no definition found": the DB has no "piece of cloth -> SARONG" and Haiku
  was never connected. definition_engine has the injected-callable hook
  (ai_define) ready, but nothing calls Haiku. core/ai_definition.py exists from a
  prior session (paid Haiku) but is NOT wired into the new engines.
  When building: ask Haiku for the definition PHRASE ONLY; locate it in the clue
  via the atomiser (must ignore brackets — SARONG's def "piece of cloth" is
  parenthesised at the end); feed it in like any DB definition; grammar-extent +
  verification then run as normal. Haiku IS allowed (user: "I only said no
  sonnet calls").
- **Persistence of the new Parse — NOT wired.** core/store.py exists but predates
  this model; provenance is not being saved for the new engines.
- Data gap: "including" is not listed as a hidden indicator, so BRIE (1710239)
  FAILS ("including" unaccounted + no indicator). Real DB enrichment needed.
- Atomiser nuance: apostrophe-'s' becomes a stray "s" token on some clues
  (e.g. 1710235 IRATE, 1710346 NETHERMOST).
- Only ONE engine exists (hidden). dd / charade / container / anagram / reversal
  / deletion / acrostic / homophone — none built in the new core.

## Principles established (see memory/wfw_redesign_principles.md + feedback_per_type_ui.md)

- Char-level atomiser is the shared substrate; product = per-character map.
- Engine-level verification, not a grand verifier. PASS (no warnings) or FAIL
  (plain warnings, still showing the genuine pieces). No score.
- ONE base screen; each type changes only its specific bit (hidden: lit host
  letters, uncoloured). Not a separate screen per type, not a rigid identical
  format. UI is ugly on purpose; one styling pass later.
- Definition handling is universal (DB + grammar extent + [pending] Haiku).

## Process note for next session

The session's repeated friction: things were discussed and agreed, then left as
"to build" without that being stated plainly, so the user assumed they were done.
ALWAYS state explicitly what is built vs pending. When agreeing to build
something, build it or say clearly you are not.
