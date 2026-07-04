# Handover — 2026-06-02 — core/ redesign: hidden complete, DD engine built

## READ FIRST — how to work with this user
- Be brief. Conclusion first, detail only on request. Verbosity is the main complaint.
- No backticks / no blue or coloured text — the user cannot read it. Write code, paths, commands as plain text.
- Do not preface answers with "honest answer" / "honestly" / "to be honest".
- Do not tell the user a decision is theirs to make. Present options + a recommendation, ask a plain question.
- Do not build or change anything without explicit approval. Diagnose/propose first.
- Do not repurpose data of one kind as another, or invent pieces "from elsewhere". Use the real enrichment process.
- Memory files to honour: feedback_communication_style, feedback_be_brief, feedback_no_blue_text,
  feedback_no_artificial_constraints, wfw_redesign_principles, core_dd_engine, haiku_definition_fallback.

## Branch / git
- Branch: redesign (NOT master).
- Pushed to origin: 7b85ac6b (hidden engine + in-engine verifier + screen + durable persistence),
  a6083ece (removed abandoned core/hidden.py).
- UNCOMMITTED (the DD engine): core/engine_registry.py (M), core/wfw_render.py (M), core/wfw_web.py (M),
  and new core/ai_synonym.py, core/dd_engine.py, core/dd_enrichment.py, core/dd_screen.py.
  core/_scratch_test.py is a pre-existing scratch file (not part of this work).

## Architecture (the new core/ system — separate from the live sonnet_pipeline)
- One entry point: engine_registry.solve / solve_clue_text. The true-test UI (core/wfw_web.py, Flask
  port 5099) calls it. Enter clue id(s); it solves, persists, and renders from the DB.
- Per clue-type engine owns its solve + its in-engine verifier + its screen. Shared substrate:
  wfw_atoms (char atomiser), wfw_model (Source/Link/Annotation/Parse), definition_engine + grammar
  (universal definition + spaCy extent), pending_store (enrichment queue), store (persistence), wfw_render (base screen).
- Cascade: hidden tried first, then DD; a clean PASS short-circuits; otherwise the first engine's FAIL is returned.
- Verification is in-engine, strict, no score: PASS (no warnings) or FAIL (plain warnings, pieces still shown).
- Provisional pieces (AI/structure-derived, not yet DB-verified) carry source='pending', show a "provisional"
  badge, and are queued to pending_enrichments. Human verifies via the dashboard -> reference DB. Rejections are permanent.
- Persistence: every solve (PASS or FAIL) saved to clues_master.db (wfw_solve/wfw_piece/wfw_link); load_parse
  reconstructs it; the UI renders from the DB. Round-trip test: core/test_substrate.py.

## Engines built
1. HIDDEN (complete) — core/hidden_engine.py + core/hidden_screen.py. Trigger = the answer as a contiguous
   clue-letter run. Indicator = the whole grammatically-bound leftover phrase (DB-confirmed or provisional+queued).
   Edge-anchored definition when none in DB. AI definition fallback (ai_definition + definition_fallback) on a DB miss.
2. DOUBLE DEFINITION (built, NOT committed) — core/dd_engine.py. Split clue into two halves covering all words.
   Both halves DB-define -> DD (no AI). One half DB-confirmed + DD-shaped (no wordplay indicator, <=8 words)
   -> narrow Haiku yes/no on the other half (core/ai_synonym.is_definition_of, cap 2 calls) -> DD, other half
   provisional + queued as a definition (core/dd_enrichment). Both halves always shown (core/dd_screen).
   Verified working: both-DB and Haiku paths, screen, persistence round-trip; hidden unaffected. See memory core_dd_engine.
   Caveat: definition_answers_augmented has blog-glosses, not only clean synonyms, so some both-DB splits anchor on
   a loose definition. Data-cleaning follow-up, not an engine fault.

## OPEN DECISIONS (awaiting the user — do not implement unasked)
1. Three-state verdict. Today a fully-parsed clue that depends on a pending enrichment is returned as a PASS
   (just badged provisional) and short-circuits the cascade. Proposed: pass / provisional / fail as distinct
   states; a verified pass beats a provisional; provisional does not short-circuit and does not count as solved
   until accepted. User has not yet said yes/no.
2. Result ranking across engines. When no engine gets a clean PASS, the cascade currently returns the FIRST
   engine's FAIL and discards the others (and only the returned parse is persisted). Competing engine results are
   mutually-exclusive interpretations of one clue — never merge them. Proposed: rank FAILs by completeness, show
   the most-complete, persist the others; optionally list runner-up interpretations. Tied to decision 1.

## Run / test
- Server: python -m core.wfw_web  -> http://127.0.0.1:5099/  (multi-id input; Admin panel adds defs/indicators/synonyms).
- Good clue ids: 1710238 ANDEAN, 1710279 TIGER, 1710251 SARONG, 1710346 NETHERMOST, 1710368 IDLY (hidden);
  for DD use the wfw_web box on any double-definition clue id, or run engine_registry.solve_clue_text directly.
- First request loads RefDB (~1.75M synonyms) + spaCy -> slow once; provisional clues cost one Haiku call.

## Loose ends
- DD work uncommitted (above). Commit when the user asks.
- Scratch analysis harnesses in prototypes/universal_form_v2/runs/: audit_gt_by_type.py, sig_type_correlation.py,
  dd_feature_model.py (the grammar/type-correlation study that grounded the DD design; AUC 0.82). Removable.
- Live (old) DD engine for contrast: backfill_ai_exp/backfill_dd_hidden.py + sonnet_pipeline/run.py Phase 0c.
- A few provisional definitions sit in pending_enrichments from DD testing (cloud->BLUR, Moneymaking->INTEREST,
  "Position for diving for"->PIKE) — normal queue output, the user can accept/reject.

## Likely next steps
Decide the two open questions, then either commit the DD engine, or build the next clue type
(charade/container/anagram/reversal — note grammar_triage's per-token signal is precise for those when it fires).
