# Handover — 2026-06-03 — charade engine built WRONG (ignores the catalog)

## BLUNT ASSESSMENT (read first)
Session largely wasted. The one good, correct artifact is the catalog DB table. The
charade engine that was built is FUNDAMENTALLY WRONG: it free-form brute-tiles the
answer with any synonyms and IGNORES the catalog_templates signatures entirely — the
catalog is the whole point of the redesign. On top of that the assistant repeated the
two cardinal sins it had just been warned about: it discarded evidence (returned None
on a failed tile), and it coded from convenient idioms instead of following the design.
It also proposed queuing every fail-fragment into pending_enrichments (pollution).
Do not trust the charade engine. Rebuild it catalog-driven per design §4.

## THE FUNDAMENTAL ERROR (what the next thread must internalise)
The charade engine must be CATALOG-DRIVEN (design §4 "Catalog engine (the core)"):
  for each charade signature (catalog_templates, in priority/frequency order):
    - lay the clue's words onto the signature's typed role slots,
    - fill each slot per its role (SYN_F -> synonym, ABR_F -> abbreviation, etc.),
    - the definition sits where the signature says (def_pos start/end) — NOT guessed,
    - verify the filled slots reconstruct the known answer,
    - emit the per-letter wfw_model record.
A clue matching NO signature is a catalog-creation gap (design §8), not a free-tile fail.
The engine that was written (core/charade_engine.py) does NONE of this — it never reads
catalog_templates, never uses role types, and finds the definition by Haiku-first. Scrap
its approach.

## WHAT IS GOOD AND CORRECT (keep)
- catalog_templates + catalog_template_slots tables in data/clues_master.db.
  694 templates + 2064 slots, loaded from data/positional_catalog.json (the DETAILED
  positional signatures — the agreed single catalog). Round-trip verified (0 mismatch,
  0 orphans). priority = rank within operation by frequency. THIS is the catalog the
  engine must consume. Schema:
    catalog_templates(id, operation, signature, def_pos, count, priority, origin,
                      active, version, created_at, notes)
    catalog_template_slots(id, template_id, position, role, n_words)
- Design/decision memories (all sound):
  - catalog-13-isolated-engines (13 separate per-type engines, copy not share)
  - catalog-clue-signature-crossref (catalog in clues_master so clue<->signature joins;
    record which signature solved each clue)
  - charade-enrichment-abbreviation-rules (abbr = truncation/initialism of the word's
    OWN letters, not association; ask each word independently)
  - feedback-communication-style (updated: no candour-advertising tics)

## WHAT WAS BUILT WRONG (review / likely rebuild or revert)
- core/charade_engine.py — free-form tiler, ignores the catalog. REBUILD catalog-driven.
- core/charade_screen.py — thin coloured base screen; probably reusable once the engine
  emits a correct wfw_model parse.
- core/engine_registry.py — CHANGED: added lookup(word,answer) to make_db_wiring; wired
  charade into solve() after DD with an order-based "first wins" fail-selection (bad,
  with a comment falsely claiming "most complete"). Review.
- core/definition_engine.py — CHANGED: find_definitions gained optional define_fallback
  (Haiku definition on a DB miss). This is additive and hidden/DD are untouched, but in a
  catalog-driven engine the definition comes from the signature's def_pos, so the role of
  this fallback needs rethinking (it is NOT the primary definition mechanism).
- core/wfw_web.py — CHANGED: imports charade_screen; SCREENS adds "charade"; dispatch now
  keys on parse.operation first then solved_by. This dispatch change is fine and worth keeping.

## POLLUTION / CLEANUP NEEDED
- pending_enrichments has 1 BOGUS row written from a FAILED parse (queuing was not gated
  to pending-only). Remove it:
    DELETE FROM pending_enrichments WHERE type='definition' AND letters='OOPS' AND source='diag';
  Principle the next thread must hold: ONLY a PENDING parse (fully assembled, one piece
  needing DB confirmation) queues. A FAIL shows its evidence on the page but queues NOTHING.
  Showing evidence != queuing it.
- wfw_solve substrate now has charade parses persisted for clue ids 1710380 (SOSO),
  1710439 (BUMPINTO), 2184174 (OOPS) from testing. Re-solve/overwrite once the real engine exists.
- Scratch test files to delete (in core/): _charade_test.py, _charade_fallback_test.py,
  _wire_test.py, _charade_preview.py, _diag_one.py, _diag_fallback.py, _verify_queue.py.
  Also wfw_charade_preview.html in repo root.

## TWO STANDING LESSONS (the assistant broke both this session)
1. NEVER discard evidence. A no-result branch must return an evidence-carrying parse,
   never None, when there is any evidence. Design §2. Check every return in an engine.
2. FOLLOW THE DESIGN as binding rules, not narrative. Before coding an engine, extract the
   rules from the governing sections (charade: §2 principles, §3 model, §4 catalog engine,
   §5.5 verifier contract) and verify the code against them. The catalog-driven shape (§4)
   is non-negotiable — the engine exists to consume catalog_templates.

## STATE
- Branch redesign. NOTHING committed this session. Working tree has the changed/created
  files above. Memories written. catalog tables + the bogus pending row are in clues_master.db.

## LIKELY FIRST STEPS NEXT THREAD
1. Read design §4 (and §3, §5.5) in full. 2. Rebuild core/charade_engine.py to match
   catalog_templates charade signatures (typed slots, def_pos), fill by role, verify
   reconstruction, emit wfw_model, evidence-preserving on fail (no None). 3. Remove the
   bogus pending row. 4. Delete scratch files. 5. Only then re-test on real clues.
