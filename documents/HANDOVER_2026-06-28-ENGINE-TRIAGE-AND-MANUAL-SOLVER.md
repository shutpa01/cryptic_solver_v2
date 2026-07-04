# Handover — 2026-06-28 — engine-first triage, two engine fixes, and the Manual-Solver design

Cold start for a NEW thread. This session settled the **decision framework** for how we improve the
solver, built/fixed two engines under it, committed the work, and deferred the Manual Solver (MS) by
design. **The next task is to REVIEW FAILED CLUES** and triage each with the framework below.

---

## THE RULES (front-loaded — do not violate)

1. **ONE solver = the cascade.** The hand-solver only ASSIGNS ROLES as forced overrides into the
   normal cascade. There is **no parallel assembler**. A "builder"/"shadow solver" was built and
   deleted before; a per-clue **"pin lever"** was designed this session and then **DROPPED** (see
   below) — do **not** rebuild either.
2. **Bespoke engines, additive.** Never edit a working engine to chase one clue. New shape → new
   engine, with its OWN verifier and OWN clue-page render, kept thin by reusing shared primitives.
3. **VERIFY, NEVER GUESS.** Trace through the real engine / real page and show the actual output
   before naming a cause. This session I twice asserted a wrong diagnosis (a "missing piece", then a
   "missing repetition mechanism") and the user corrected me both times. The cost of guessing is
   trust. Always run it.
4. **Always restart the server and verify on the REAL page** (`127.0.0.1:5099`, exactly ONE
   listener — stale-server trap). Start: `.venv/Scripts/python.exe -m core.wfw_web`. The clue page
   renders the STORED parse, so after an engine change **re-solve the clue** to persist:
   `POST /reload` with form `id=<id>&only=<id>`.
5. **Big A/B sweeps = OVERNIGHT only** (`core/_ab_general.py`); they block and conflict with the
   live server's DB writes. Daytime = small fast checks (isolation diags + single-clue /reload).

## THE DECISION FRAMEWORK (the heart of this session)

When a clue fails, triage by cause — **do not reach for a manual tool to paper over a defect**:

1. **Data MISSING** (a real synonym / abbreviation / indicator / definition the DB lacks)
   → **ENRICH the DB** (global if genuine). This is the only honest "help the solver". It is the
   clue-page "Add to reference DB" panel.
2. **Data PRESENT and it still fails** → an **ENGINE DEFECT** → **fix / add an engine**, however
   long it takes. Diagnostic: confirm every piece + indicator is already in the DB (lookups return
   them); if so, it is the engine. Both clues fixed this session were here.
3. **No valid mechanism / self-indulgent setter trick / unfair derivative** → **Manual Solver**,
   the LAST resort only.

**ENGINE-GRAIN (the balance — many SIMPLE bespoke engines, never few COMPOUND ones):**
- A new engine is justified by a new **structural shape**, NEVER by data. Same shape, different
  words/values → same engine.
- The compound smell is a **branch**: the moment an engine grows `if this shape … elif that …` to
  cover another angle, **fork a bespoke sibling** instead.
- Bespoke ≠ duplicative: own verifier + own render, but thin via shared primitives
  (`selection.select_span`, `deletion`, container-insert, reversal, the `lookup` fns,
  `role_validity`, the render row-builders). Primitives shared; shapes bespoke.
- Decision test (data confirmed present): can an existing engine's single straight path express
  this shape? gap → additive fix; would need a branch → new sibling; only data differs → not an
  engine matter.

## WHY THE PIN LEVER WAS DROPPED (so no thread rebuilds it)

I designed a per-clue "pin a piece value" lever (wrap the wiring's `lookup`/`lookup_all`/
`all_values` via `clue_overrides`) to guide the cascade. The user killed it with the decisive
point: where the data is already present and the clue still fails, **pinning the piece just masks
an engine defect and destroys the diagnostic signal**. Both example clues proved it (data present,
engine at fault). So: no pin lever. Fix the engine instead.

---

## WHAT WAS DONE THIS SESSION — committed `cf021385` on `redesign` (NOT pushed)

| clue | id | answer | fix | result |
|---|---|---|---|---|
| BEDSPREAD | 10075981 | BEDS+P+READ | NEW `charade_positional_local_engine.py` | PASS (real page) |
| NARRATIVE | 10075968 | rev(EVITA+RR+A+N) | re-wired evidence reversal-charade engine | PASS (real page) |

1. **`core/charade_positional_local_engine.py` (new).** A charade where a SWAP positional indicator
   reorders ONLY its adjacent pair, others in clue order ("Cover county show after parking" =
   county + (show after parking → P+READ) = BEDS+P+READ). Bespoke sibling of
   `charade_positional_engine` (global pivot, "School following second-class old" = BO+SCH = BOSCH):
   two distinct shapes — local doesn't solve BOSCH, global doesn't solve BEDSPREAD. Reuses
   `_tile`/`_candidates`/`_find_indicators`/`_verify`; bespoke local-pivot search + render note.
   Wired in `engine_registry.solve` right AFTER the global positional engine.
2. **Reversal-charade re-wire (`engine_registry.py`).** The cascade's reversal-charade slot used the
   catalog `reversal_charade_signature_engine`, which returns None for a WHOLE-charade reversal (the
   signature "never encodes" it). The evidence-driven `reversal_charade_engine.py` (order-free
   tiler) solves it but had been unwired by `e5c726f6` ("reversal family → signature engines"). I
   re-wired it AFTER the signature engine (signature claims what it can; evidence catches the rest).
   Additive — the signature engine is untouched.
3. **Faithful piece labels (`reversal_charade_engine.py`).** It hardcoded `mechanism="synonym"` for
   every piece. Now threads the real mechanism through `_run_values`→`dfs`→`_build` and labels each
   value by priority **raw (literal) > abbreviation > synonym** (so `a`→A = Literal, `new`→N =
   Abbrev., even though the DB lists N as both). Label-only: value list + order unchanged, solving
   untouched.
4. **Clue-page DB-add (`wfw_web.py`).** Added `alternation` to the indicator type dropdown
   (`_IND_TYPES`); made the subtype dropdown data-driven (`window.WFW_SUBTYPES` + `wfwSub`) so any
   type with subtypes shows them (was hardcoded to deletion). NOTE: alternation has NO functional
   subtype (the engines try both odd/even and let the answer decide); a `selection` subtype set is
   the remaining clue-page step (maps to (wordplay_type, subtype) pairs in `selection_indicators.py`).

## OUTSTANDING — the gate before PUSH

**Overnight A/B (`core/_ab_general.py`) has NOT been run.** Run it before pushing, chiefly for the
reversal re-wire (the evidence tiler is ORDER-FREE / more permissive, sitting above the more
specific reversal engines — selection-reversal-charade, reversal-container, reversal-deletion — so
prove it does not over-claim or intercept them). The local-pivot engine should also be in the
sweep. The two label fixes are display-only and need no behavioural A/B. Nothing is pushed.

---

## NEXT TASK — REVIEW FAILED CLUES (start here)

Pull the current failing clues and triage each with the framework above:
- missing data → enrich (clue-page Add panel);
- data present + fails → engine defect (trace it, then build a bespoke sibling or additive fix);
- no valid mechanism → MS candidate.

The two engines fixed this session came straight out of this kind of review (the user supplied
BEDSPREAD and NARRATIVE as examples). Expect more whole-charade-reversed and positional cases, plus
genuinely unfair clues that become the first real MS test cases.

The diagnostic method that worked (reuse it):
- Build the wiring: `engine_registry.make_db_wiring()`; for isolated engine calls also
  `role_validity.set_predicates(w["indicator_types"], w["is_link"])`.
- Check data presence: `db.get_abbreviations(word)`, `db.get_synonyms(word)`, `w["lookup_all"](ph)`.
- Run the specific engine in isolation, then reproduce the FULL cascade with overrides
  (`clue_overrides.apply_forced_overrides(make_db_wiring(), cid)` then `engine_registry.solve`) to
  see what the real path does. Watch for two engines sharing a function name (the NARRATIVE bug:
  `solve_reversal_charade` exists in both `reversal_charade_engine` and
  `reversal_charade_signature_engine`; only the latter was wired).

---

## THE MANUAL SOLVER (MS) — DESIGN (deferred; build only after failed-clue review confirms need)

**Purpose.** A HUMAN authoring tool inside the hand-solver, for the clues triage bucket 3 ONLY:
self-indulgent setter tricks not worth an engine, or clues with NO valid mechanism (second-order /
indirect derivations past the fairness line — e.g. an op applied to the output of another op). It
is **not a solver**: it derives nothing, verifies nothing, never runs in the cascade. It is a dumb
recorder of what the human types, fired only on an explicit Commit.

**Do NOT build it early.** Build the MS before the engine work matures and it becomes a crutch —
people hand-solve clues a missing engine should handle, and the solver rots into "too manual". The
triage framework is precisely what keeps the MS small.

**Workflow (agreed).**
1. Open the hand-solver (`/hs`) for the clue.
2. For each wordplay piece: tick the word(s), and type the piece's CONTRIBUTION TO THE ANSWER in the
   BRINGS column — i.e. the letters it ends up as after the human applies the ops in their head
   (PALATIAL "Indian dish, not starter, served up" → type ATIA, not RAITA).
3. Tag the remaining words: definition / indicator / link / filler.
4. **Tile colouring:** the typed letters are LOCATED in the answer (the order is obvious from the
   answer) and those tiles take the piece's colour. (User settled this: just match the letters; no
   positional drag.)
5. **Commit:** assemble a Parse from EXACTLY what was typed/tagged and persist it FROZEN. No cascade,
   no verification. **Uncommit:** clear it, hand the clue back to the cascade. Human sets the status.

**The boundary (this is where a thread slides back into the banned builder — HOLD IT).** The MS is
mechanically "build a Parse from assignments", which is what the deleted builder did. The ONLY
things that keep it legitimate, and they MUST be enforced in code:
- **Per-clue values ONLY — never written to the reference DB** (the typed values are derivatives
  like ATIA, not real synonyms; writing them manufactures the junk we delete). Build on the existing
  **inert `letters` role** (`wfw_web.py` `/hsresolve`, role "letters" — already per-clue, NO DB
  write), NOT the synonym role (which calls `admin_db.add_synonym`). This is a CORRECTION to the
  older spec, which said "reuse the synonym role + suppress the write" — using `letters` means there
  is no write to suppress.
- **No derivation** (the human types every value), **no verification / no auto-pass** (the human
  sets the status), **never invoked by the cascade** (a separate explicit Commit route — NOT
  `/hsresolve`, which runs the cascade), **flagged "manual"** on the page.

**Build notes (verified infrastructure):**
- `wfw_model.Parse` holds sources / links / annotations / definition. Build sources with
  `clue_atom_ids` (map word idx → atoms via ctx), links tiling the answer letters (locate each
  typed value), definition + indicator/link annotations.
- Render already supports it: `render_parse` falls back to `_render_generic_breakdown` for an
  unknown operation (`wfw_render.py:133`), and a `"manual"` SOURCE provenance already badges
  "manual (not in DB)" (`wfw_render.py:265`). `_MECH_LABEL["raw"]="Literal"`.
- Persistence/freeze caveat: `store.save_parse` (`store.py:132`) only blocks a DOWNGRADE on a frozen
  clue — a PASSING cascade re-solve would still overwrite it. Fine for MS clues (the cascade can't
  solve them), but if true protection is wanted, add an explicit committed-manual guard so
  `_resolve_one`/`save_parse` skip the clue entirely.
- New routes: `/hscommit` (build + persist frozen manual Parse) and `/hsuncommit`. Keep
  `/hsresolve` (the cascade path) untouched.
- Smallest test cases first: PALATIAL (3 clean pieces) then a multi-op one.

---

## MEMORY POINTERS
`manual_solver_and_db_add_split.md` (the running state of all of the above — triage, engine-grain,
the two engine fixes, the dropped pin lever, the MS design), and the prior implementation handover
`HANDOVER_2026-06-27-CONTAINER-FAMILY-AND-MANUAL-SOLVE.md` (the container-with-built-inner family).

HEAD = `cf021385` on `redesign`, not pushed. Server runs the latest code on `127.0.0.1:5099`.
