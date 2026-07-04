# Handover — 2026-06-19 — Deletion signature conversion (first reconvergence)

Branch `redesign` (NOT pushed). All work below is UNCOMMITTED. The user controls commits.

## Why this session happened

The redesign is meant to be a SIGNATURE-BASED solver: ONE catalog of recipes (templates)
solved by composable verifiers, with new clue types added as recipes, not code
(SOLVER_REDESIGN.md §2, §5.4, §6). It had drifted badly into ~40 hand-built per-type engine
files in core/. The user was angry — and right. We also found that the "auto signature
creation" we had been claiming as the working mechanism was effectively dormant (7 auto rows
ever, all on 2026-06-05, covering 5 of 9 operations, none for the bespoke types). Telling the
user it was the working mechanism was untrue.

## The audit (see memory/engine_audit_signature_vs_bespoke.md)

Discriminator: a signature-driven engine takes a `templates` list and walks catalog_templates
via core.catalog_loader; a bespoke engine hardcodes one shape and reads no catalog.
- 9 genuinely signature-driven, 8 sanctioned-standalone, **14 bespoke** to reconverge.
- The 14 collapse to ~4-6 new catalog operations, NOT 14 rewrites. None is impossible to
  express as a recipe.

## The plan (agreed)

No new bespoke engines. Convert the 14 one at a time, each via this fixed procedure:
**loader → recipe-driven verifier → re-miner → full-cascade A/B → flip default → retire bespoke.**
Gate to cut over: 0 regressions, signature match-or-beat the bespoke, identical answer letters
on every shared solve. Never delete a working engine before its replacement is proven; keep its
_build/_verify helpers (the signature engine reuses them).

## What was done — DELETION (the template for the rest)

Genuinely recipe-driven, verified. Recipe = base slot (SYN_F/ABR_F) + DEL_I deletion-indicator
slot (+ REM_F removed-letters source for Form B named deletions). The verifier executes: reads
the op (behead/curtail/outer/heartless/empty) from the indicator's DB sub-type and applies it
(Form A), or removes the REM_F value (Form B). A split indicator ("cut at the front") is one
DEL_I slot spanning the phrase; interior glue words become links, matching the bespoke exactly.

Files:
- core/deletion_signature_engine.py — NEW engine. Reuses deletion_engine._build/_verify/_run_values.
- core/catalog_loader.py — load_deletion_templates() (operation='deletion').
- core/_mine_deletion.py — re-miner (dry-run default; --write backs up + replaces). 86 recipes.
- core/engine_registry.py — deletion_templates in make_db_wiring; `deletion_solve` override on
  solve()/solve_clue_text(); cascade DEFAULT now the signature engine.
- core/_ab_deletion_sig.py — full-cascade A/B harness (bespoke vs signature).
- core/deletion_engine.py — RETIRED from cascade, KEPT for shared helpers + A/B override.

A/B (full cascade, DB-only):
- seed 7, n=1500: bespoke 382 / signature 383, 382/382 identical, 0 regressions, +1 correct.
- seed 99, n=1500: bespoke 363 / signature 367, 363/363 identical, 0 regressions, +4 correct.

Rollback: catalog_templates_bak_mine_del / catalog_template_slots_bak_mine_del (pre-conversion).
Cut-over is a one-line default swap in engine_registry.py.

## Honest lessons (do not repeat)

1. An early A/B with the LOOSER engine (before the interior-link fix) reported "44 gains" and I
   claimed they were all correct after eyeballing the CLUES. They were mostly FALSE POSITIVES —
   the loose engine absorbed content words as indicators and "solved" clues whose real base
   synonym isn't even in the DB (FEAST∉syn(banquet), so EAST must FAIL). The proven bespoke
   engine fails them; a faithful signature engine must too. Verify gains through the cascade,
   with the engine's REAL args, never armchair.
2. A flawed isolation test omitted `is_dbe`/`define_fallback` and returned spurious None. Always
   pass the cascade's real args when reproducing a solve.
3. The interior-link constraint (untyped words inside a DEL_I span must be link words) is what
   made the engine honest and byte-identical to the proven engine. Recipe-driven ≠ loose.

## Next session — remaining 4 families (same procedure)

1. substitution (whole-answer single op — sibling of deletion; simplest).
2. anagram-with-modified-fodder: anagram_substitution, anagram_multi_substitution,
   anagram_deletion, anagram_insert_letter (probably ONE operation).
3. charade-with-transformed-piece: charade_deletion, charade_named_deletion, charade_hollow
   (likely folds into existing charade SEL_F 'outer'), charade_positional.
4. container-variants: container_deletion, container_inner_charade, container_acrostic,
   reversal_container (its 2 mined rows are malformed — re-mine from scratch).

Start each by reading core/deletion_signature_engine.py + core/_mine_deletion.py +
core/_ab_deletion_sig.py as the worked template. Python: .venv\Scripts\python.exe.

## Verification still open for the user

The A/B and cut-over were verified through solve_clue_text (the real cascade entry the web
route calls), DB-only. A live HTTP check on :5099 (python -m core.wfw_web) was NOT run this
session — worth an eyeball on a deletion clue's per-letter page before committing.
