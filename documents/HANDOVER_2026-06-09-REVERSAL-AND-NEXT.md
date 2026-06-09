# HANDOVER 2026-06-09 — Reversal family done; next: raw-letters + deletion

Branch: `redesign` (now pushed). Server: `python -m core.wfw_web` on :5099 (restart after any
code change; debug off). Read `CLAUDE.md` and `memory/MEMORY.md` first — this doc assumes them.

---

## 1. TL;DR — where we are

The cascade is now **fully signature-driven** for every built clue type:
charade, anagram, anagram+charade, anagram+container, container, container+charade,
**reversal, reversal+charade** (the last two converted today).

Cascade order (in `core/engine_registry.py::solve`): the catalog/signature engines run
most-specific-first, then **reversal**, then **reversal+charade**, then **double-definition
(DD) LAST**, then AI piece-recovery. DD runs last on purpose (its AI half is soft and would
intercept catalog clues).

Today's commits (all on `redesign`):
- `ee342bb4` container_charade: OOM fix (answer-driven `_container_spans`) — this is the bug
  that **crashed the laptop**; see §6.
- `5f41c733` reversal: plain-reversal **evidence** engine, wired.
- `0c41490f` reversal_charade: **evidence** engine, wired.
- `e5c726f6` reversal family: **converted to catalog-driven signature engines** (the final state).

## 2. Next work (what we do in the new thread)

Two **mechanism gaps** found in a 2026-06-09 Times batch (both real, both pre-existing, neither
caused by the reversal work). Full notes in `memory/worklist_2026_06_07.md` (MECHANISM GAPS).

### A. Raw / literal-letters reading  — clue VERITY (id 2130740)
`VERITY = VER(IT)Y` — "In truth"(def) = VERY("so") around IT("it"), "full"=container indicator,
"of"=link. The container engine gets VERY="so" and "full" is correctly a container indicator;
it misses for ONE reason: **"it"→IT is not in the DB**. IT here is the word's own literal
letters (a *raw* reading), and the lookup layer only returns synonym/abbreviation, never a
word's literal letters. So the clue is one piece short. (The stored synonyms for "it" are
cross-filed junk: CITY, TRADE, SHADIER… IT itself is absent.)

Design:
- Add a **raw/literal-letters** value source: a short function word used as its own uppercased
  letters (it=IT, a=A, in=IN, on=ON, by=BY, no=NO, so=SO, one=ONE/I, …).
- **CARE — false positives:** a blanket "any word → its own letters" rule explodes false
  positives. Gate it to a **curated literal lexicon / short function words** only.
- **Where:** add it as an additive value source in the **shared lookup layer**
  (`core/wordplay.py` / `lookup_all`) so all engines inherit it — do NOT hack the working
  container solve path (sacred rule: never modify a working stage engine for an edge case).
- **Test:** VERITY should then solve via the container engine. Verify through :5099 and an
  A/B (see §5) that no working solves regress (raw letters can create spurious pieces — watch
  container/charade especially).
- Smaller, additive, cross-engine win. Do this FIRST.

### B. Deletion / emptying primitive — clue LIBIDO (id 2130865)
`LIBIDO = LIDO⊃BI` — "Drive"(def), "swimming pool"=LIDO, **"empty Bugatti"=BI** (first+last
letters, B·gatti emptied), "into"=container indicator. Needs an **emptying/deletion
letter-selection** to form BI before the container can run. **No deletion engine exists at all**
(`del`=4 on the worklist). Bigger build than A.

Design (scope it first, evidence-first like the reversal family):
- Deletion sub-types: **emptying** (remove middle: empt(B·gatti)=BI), **beheading** (drop first),
  **curtailment/apocope** (drop last), **endless/heartless**, internal deletion. Indicators:
  empty, without, losing, heartless, endless, topless, … (`indicator_types` already returns
  `deletion` for some — "empty"→{container,deletion,parts}).
- Deletion is usually **compounded** (deletion+container as in LIBIDO, deletion+charade). Plan:
  build a standalone deletion letter-selection primitive + an evidence engine, verify on real
  clues, then mine clean sigs and convert to signature-driven (the proven pattern).
- Bigger; do AFTER A.

## 3. The signature-conversion pattern (proven 4× now: container, container_charade, reversal,
reversal_charade) — reuse it for deletion

1. Build the **evidence engine** `core/<type>_engine.py` (answer-driven; copy a sibling
   evidence engine as scaffold — `container_engine.py` is the simplest, `container_charade_
   engine.py` the charade-tiler one). Wire it after the relevant family, before DD. Verify
   standalone on real clues + a cascade A/B (0 regressions). Commit.
2. **Mine** clean signatures read-only from the evidence engine's verified passes:
   `core/_mine_<type>.py` (copy `_mine_reversal_charade.py`). It reuses the engine's own
   `_assemble`/placement and derives a signature string (roles in clue order + def edge).
3. **Seed** the catalog (reversible): `core/_seed_<type>.py` (copy `_seed_reversal_family.py`)
   — backs up `catalog_templates`/`catalog_template_slots` to `*_bak_<tag>`, **retires the
   noisy catalog_creator sigs via active=0 (NOT delete)**, inserts the mined sigs.
4. Build the **signature engine** `core/<type>_signature_engine.py` (copy
   `reversal_charade_signature_engine.py`): catalog-driven slot placement + answer-driven
   reconstruction; **REUSE the evidence engine's `_build`/`_verify`** so parses are
   byte-identical. Add a `load_<type>_templates` to `core/catalog_loader.py`.
5. **Wire** it to replace the evidence engine in the cascade; A/B signature-vs-evidence
   (target 0 miss; gains OK if verified correct). Commit. Keep the evidence engine for mining.

## 4. File map (reversal family — the template to copy)

- Evidence engines: `core/reversal_engine.py`, `core/reversal_charade_engine.py`
- Signature engines: `core/reversal_signature_engine.py`, `core/reversal_charade_signature_engine.py`
- Miners (read-only): `core/_mine_reversal.py`, `core/_mine_reversal_charade.py`
  (outputs `_mine_rev_out.txt`, `_mine_rcharade_out.txt`)
- Seeder (reversible): `core/_seed_reversal_family.py`
- Loaders: `core/catalog_loader.py::load_reversal_templates / load_reversal_charade_templates`
- Wiring + cascade: `core/engine_registry.py` (`make_db_wiring` loads templates; `solve` runs them)
- Parse model: `core/wfw_model.py` (Source/Link/Annotation/Parse; `Link.transform="reversed"`;
  `is_complete()` only checks every answer pos has one link — the ENGINE verifies the
  reconstruction).
- Definitions/splits: `core/definition_engine.py::find_definitions` (def at start/end; wordplay
  tokens exclude the def). POS tags: `core/grammar.py` (spaCy singleton — see §6 memory note).

Catalog state after today: `reversal` 15 active mined sigs (30 noisy retired), `reversal_charade`
355 active (67 retired). Backups: `catalog_templates_bak_revfam`, `catalog_template_slots_bak_
revfam`. The catalog lives in `data/clues_master.db` (gitignored, backed up twice daily) — the
mine/seed scripts make it reproducible.

## 5. How to test

- **UI:** `python -m core.wfw_web` on http://127.0.0.1:5099 — restart after ANY code change.
- **Direct A/B harness pattern** (copy from how we did it this session; the `_ab_*` scratch
  files were deleted but the pattern is): in ONE process, run `solve()` (or an engine) twice —
  once normal ("after"), once with the new engine monkeypatched to None / the old engine
  ("before") — over a fixed sample including the relevant clue type + the standard types
  (anagram, charade, container, reversal, …). Report per-clue (status, engine) deltas:
  regressions (a working solve changed) vs gains.
- **Standalone engine A/B:** evidence-fn vs signature-fn over N reversal/<type>-tagged clues,
  count pass/miss/gain, then **decode + hand-verify every gain** (false positives hide here).

## 6. Critical lessons from this session (do not relearn the hard way)

- **MEMORY / OOM (this crashed the laptop):** reconstruction must be **answer-driven** — derive
  candidate spans from the answer and check DB membership; NEVER materialise a product of synonym
  values (outer×inner×positions, or value-lists × positions). The container_charade
  `_container_spans` product spiked ~1.5GB on one clue. All new engines: answer-driven only.
- **NEVER run multiple RefDB-loading processes concurrently.** Each loads ~1.7M synonyms (~1GB).
  ~4 at once exhausted RAM and took the machine down. Run ONE heavy process at a time (the idle
  :5099 server is fine alongside one). Prefer single foreground/one background job.
- **DD non-determinism:** `make_db_wiring()` sets `ai_is_definition=True` → the DD engine makes
  live AI calls and is **non-deterministic**. In any cascade A/B, a flip on a DD-solved clue
  (e.g. IMPROMPTU) is AI noise, NOT your engine — verify the clue isn't touched by your engine
  before treating it as a regression.
- **Single-letter "reversal" is a no-op** (rev("S")=S). Today's reversal_charade signature engine
  initially counted a 1-letter REV_F piece as satisfying "≥1 reversed" → false positives
  (TAKINGS, TROVE…). Fixed by skipping reversed pieces where `v[::-1]==v` (matches the evidence
  engine's `rv!=v` guard). Watch for analogous vacuous-op bugs in deletion (e.g. deleting from a
  1-letter piece).
- **Stored `definition` column is unreliable** (e.g. TAFFETA stored as def "cheese", should be
  "Fabric"). Verify defs against the clue, don't trust the column.
- **Seeds are reversible** (backup tables + active=0, never delete). Catalog DBs gitignored.
- **spaCy** is a module singleton in `core/grammar.py::_nlp()` (loaded once, ~575MB resident,
  shared) — that resident memory is expected, not a leak.

## 7. Worklist after A and B (from memory/worklist_2026_06_07.md)
- Positional charade (LEFTOVER, TOPKNOT — "on"/"first"/"at the front" reorder pieces).
- reversal_container (2, tiny); homophone (1); acrostic; `hidden_reversed` (1,599) = a separate
  hidden+reversal engine.
- Systemic: CD/&lit classifier; no-def-floor refinement; DD half-check too loose; suggest_piece
  gaps; synonyms_pairs single-letter cleanup (Phase 2).

## 8. Relevant memory files
`memory/reversal_family_scope.md` (full reversal arc + decisions), `memory/container_charade_
span_memory_fix.md` (the OOM bug), `memory/worklist_2026_06_07.md` (the gaps + worklist),
`memory/catalog_13_isolated_engines.md`, `memory/core_cascade_stop_and_verdict_rules.md`,
`memory/feedback_*` (binding behavioural + cryptic rules).
