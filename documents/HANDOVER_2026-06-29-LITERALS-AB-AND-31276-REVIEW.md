# Handover — 2026-06-29 — FIRST TASK: literals A/B; then DT 31276 review + literals→DB conversion

Cold start for a NEW thread. Read this top-to-bottom. The **first task is a ~20-minute A/B run**
(below). The rest is the in-progress DT 31276 failed-clue review and the design decisions behind it.

---

## ⏱️ FIRST TASK — run the curated-literal A/B (`core/_ab_literals.py`)

**Why.** The literal lexicon is a **hardcoded list** of 36 function words in `core/literals.py`
(`LITERAL_WORDS`): a/an/the/no/our/her, in/to/on/at/of/for/up/out/off, or/as/and/if/so/but,
i/it/me/us/he/one, is/are/be/do/go/was/am/not. Each is usable as a literal piece (its own letters:
`it`→IT, `a`→A). The user's concern (legitimate): these appear in **nearly every clue**, so making
them all available as pieces — especially 1–2 letter ones (`a`→A, `i`→I, `in`→IN, `of`→OF, `is`→IS) —
may cause more false-positive parses than genuine solves. We must MEASURE this, not argue it.

**How.** `core/_ab_literals.py` runs the full cascade over a fixed 510-clue sample (AI disabled,
deterministic) in two modes: `before` (lexicon emptied) and `after` (lexicon active). Run BOTH,
**server STOPPED** (single clean RefDB process; avoids the DB-lock contention that otherwise inflates
timing), sequentially:
```
# stop the server first (kill listeners on 5099)
.venv/Scripts/python.exe -m core._ab_literals before
.venv/Scripts/python.exe -m core._ab_literals after
```
Writes `_ab_lit_before.txt` / `_ab_lit_after.txt` (one line/clue: `id|status|engine|operation|template_id`).
**Measured runtime ≈ 20 min total** (~1.16 s/clue × 1020 + 2 wiring builds; faster with the server down).

**Then diff** the two files and report, per type:
- GAINS: clues `before`=fail/None → `after`=pass (what the list buys).
- HARM: clues `before`=pass → `after`=changed/fail, or a different (spurious) parse (the user's worry).
Identify WHICH words drive any harm. That decides whether the literal list shrinks, grows, or goes.

---

## NEXT — convert `LITERAL_WORDS` to a DB table (user directive: no hardcoded lists)

The user: "these hardcoded lists are amateurish" — literals should be a **DB table, editable via the
clue page**, like synonyms/indicators. This is consistent with the project: `core/selection_indicators.py`
was ALREADY converted hardcoded-list→DB-driven (its docstring calls the hardcoded list "the same
anti-pattern deletion/substitution had before their conversion"). **Follow that pattern.**

Plan (additive, one file at a time, test between each; do AFTER the A/B so the seed is evidence-curated):
1. New DB table for literal words (in `cryptic_new.db`); seed from the current `LITERAL_WORDS` MINUS
   any words the A/B shows are net-negative.
2. `literals.literal_value` reads via a provider the wiring installs (mirror `selection_indicators.set_rules_provider`),
   not the frozenset. Keep the function signature identical so the 6 consumers don't change.
3. Clue-page Add panel: add a "literal" kind (so e.g. `PE` can be added as a literal via the UI).
4. CONSUMERS to keep working (verified this session): `charade_engine`, `charade_alternation_engine`,
   `charade_positional_engine`, `container_charade_signature_engine`, `container_signature_engine`,
   `reversal_charade_engine`, and `engine_registry` `lookup_all` (adds the raw value for literal words).

---

## DT 31276 FAILED-CLUE REVIEW (in progress)

32 clues: 16 pass / 5 pending / 11 fail (after this session's fixes). Triage method: solve all via
`engine_registry.solve(ctx, wiring, clue_id=None)` (no persist), then per fail check data presence
(`db.get_abbreviations`/`get_synonyms`, `wiring["lookup_all"]`) and trace the specific engine.

DONE this session:
- **7d WRESTLING** = W+REST+LING — missing synonym `excess`→REST; user added it → passes. (enrich)
- **12a RAREBIT** = RARE+B+IT — `it` is a LITERAL; failed for lack of the catalog shape; resolved when
  signature 1662 (`SYN_F(3w)+SYN_F+LIT_F charade def:start`) was approved from the auto-discovery queue. (catalog)
- **17a GLINT** = G(`ultimately revealing`, last-letter SELECTION)+LINT — added catalog signature
  **1663** `SEL_F+SYN_F(2w) charade def:start` (via `catalog_creator.add_signature`, server stopped,
  add→re-solve→keep-only-if-faithful). Passes faithfully. (catalog)
- (earlier in the prior thread: NARRATIVE 14a + BEDSPREAD 5d.)

CLASSIFIED / UNRESOLVED:
- **5a BYLAWS** → **Manual Solver** (first MS case): "taking seconds" appears 15× in 596k clues with a
  DIFFERENT meaning each time — no fixed mechanism, not engine-worthy.
- **19a PEDAGOGUE** = (PE+DUE)⟨AGOG⟩ container_charade — UNRESOLVED. `PE` is a **literal** (raw PE),
  NOT an abbreviation (do NOT add a PE→PE abbr row). Adding the LIT_F signature variant (tid 1664)
  FAILED and was ROLLED BACK: `container_charade_engine` fills pieces ONLY from `lookup_all`, which
  offers `raw` only for curated literal words — and `PE` isn't one. **So PEDAGOGUE is downstream of the
  literals→DB work**: once `PE` can be added as a literal (and `lookup_all` offers it), re-add the
  LIT_F signature and it should pass. Don't chase it before the literals work.
- STILL TO TRIAGE: 21a ERRATIC (container ERIC∋RAT), 3d CARDBOARD (spoonerism BARD/CORD),
  4d DAUNT (D+AUNT), 8d OBERON (anag), 17d GO-GETTING (odd-letter selection charade — likely same
  class as GLINT, a SEL signature variant), 18d VEERED (container VED∋ERE), 24d ELUDE (reverse +
  named deletion SCHEDULE−SCH).

## OVERNIGHT EXPERIMENT (separate, bigger — needs a full night, not the 20-min literals A/B)

Systematically fill the missing signature variants so we stop hand-adding them (GLINT/PEDAGOGUE class).
Measured space (charade): 257 sigs / **131 distinct role-patterns**; word-count variants = **3,234**;
full enumeration 4.5k–22.5k (too big). SCOPE: fill **word-count AND role (SYN_F/ABR_F/LIT_F/SEL_F)
variants of the EXISTING 131 patterns**, start with 2–3-piece patterns, charade + container_charade.
**Critical caveat:** a `LIT_F` variant is useless unless the engine free-fills `raw()` for that slot —
the plain charade does, `container_charade` does NOT (it uses `lookup_all`). Track per-engine literal
capability. Add via `catalog_creator` (server stopped), full-corpus A/B, KEEP ONLY IF ~0 regressions.
(Generator not yet written.) This is gated behind the literals decision.

---

## STANDING RULES & METHOD (do not violate)
- **Triage:** data MISSING → enrich DB; data PRESENT + still fails → ENGINE/CATALOG fix; no valid
  mechanism → Manual Solver (last resort). Don't use a manual tool to paper over an engine gap.
- **Engine-grain:** a new engine only for a new STRUCTURAL SHAPE, never for data; the compound smell is
  a branch → fork a bespoke sibling; each engine owns its verifier + render, thin via shared primitives.
- **Faithful labels** matter to the user: a piece is a literal / abbreviation / synonym by what it
  ACTUALLY is (`a`→A and `PE`→PE are literals, NOT abbreviations). I mis-classified twice this thread
  and was corrected — VERIFY via the real engine/page, never assert.
- **No hardcoded lexicons** — DB tables, editable via the clue page (this is why the literals work exists).
- **Catalog adds**: derive the candidate from an existing template (don't hand-type), add via
  `catalog_creator.add_signature` (auto-backs-up), re-solve, KEEP ONLY IF a faithful pass, else roll back
  (delete the template + slots). Catalog lives in `clues_master.db` (GITIGNORED, auto-backed-up).
- **Server**: `.venv/Scripts/python.exe -m core.wfw_web` on 127.0.0.1:5099, exactly ONE listener
  (stale-server trap). STOP it before any DB/catalog write (lock contention). Re-solve a clue to persist:
  `POST /reload` form `id=<id>&only=<id>`.
- **Big A/B = overnight**; small checks (single-clue solves, the 20-min literals A/B) are fine in-session.

## CODE STATE
- Branch `redesign`, **HEAD = cf021385** (NOT pushed): local-pivot charade engine, reversal-charade
  evidence re-wire, faithful piece labels in reversal_charade_engine, clue-page `alternation` type +
  data-driven subtype dropdown. The corpus A/B for these is the remaining gate before any push.
- Catalog (clues_master.db, gitignored): signature **1663** added/kept (GLINT); 1664 rolled back.
- Memory: `manual_solver_and_db_add_split.md` holds the full running state.

## MEMORY POINTERS
`manual_solver_and_db_add_split.md` (running state: triage, engine-grain, the engine fixes, the dropped
pin lever, the MS design, the DT 31276 table, the overnight experiment, the literals A/B + DB-conversion
plan). Prior handover: `HANDOVER_2026-06-28-ENGINE-TRIAGE-AND-MANUAL-SOLVER.md` (MS design in full).
