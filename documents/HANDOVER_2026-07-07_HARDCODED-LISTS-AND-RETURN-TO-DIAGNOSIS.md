# HANDOVER 2026-07-07 (PM) — Finish the hardcoded-subtype problem, then RETURN to failure diagnosis

**START HERE for the new thread.** The previous session (this one) burned ~4 hours and **built
nothing** — no code, no commits, no schema. It produced verified facts and memory notes (below)
and a great deal of discarded guessing. The user ended it because the assistant kept **guessing
and grandstanding instead of checking** — asserting design conclusions it had never traced,
flip-flopping between framings, and "sounding good instead of being good." Do not repeat that.

---

## 0. HOW THE USER NEEDS YOU TO WORK (read before doing anything)

These are non-negotiable and they are WHY the last session failed:

1. **VERIFY BEFORE CLAIMING. Cite file:line.** Never state how the code works from memory or
   inference. Read the actual file. If you have not traced it, say "I have not checked this."
2. **No design opinions until the mechanics are traced end-to-end.** The last session proposed
   table/no-table/no-column/no-storage in successive turns — all ungrounded. Trace first, design second.
3. **Plain English. No jargon, no grandstanding, no performance.** Short answers. If a one-line
   answer is right, give one line.
4. **Claude NEVER re-runs / re-solves a clue. Running the solver is the USER's job.** Claude only
   SUGGESTS. This is the load-bearing anti-cheat rule (see §3).
5. **Small steps, one file at a time, test through the real path.** Confirm understanding with the
   user before writing code in a design-sensitive area.
6. A **question is not an instruction to act** (project CLAUDE.md). "Why…?"/"How…?" = explain, do
   not touch files.

---

## 1. THE TWO WORKSTREAMS

- **A — IMMEDIATE: finish the hardcoded-subtype problem.** The trigger was: the clue-page
  **add-indicator** form does not let the user pick a valid sub-type for `alternation` (and, more
  generally, must offer exactly the sub-types the *engines recognise* for any type that has
  operative sub-parts). This turned into a wider principle (§3) and an audit (§4). It is NOT done.
- **B — THE TWO-DAY MAIN THING (return to this after A): the FAILURE DIAGNOSIS / triage classifier.**
  This is what the user has actually been working on for two days. The design is
  `memory/nightly_triage_process.md`. The built classifier `core/triage_classify.py` is BROKEN
  (§5). Fixing it is the real goal; the hardcoded-subtype work (A) is a detour that surfaced while
  reviewing it.

---

## 2. VERIFIED FACTS — trust these (each was read/queried this session)

**Environment**
- Branch `redesign`, HEAD **7242ea33** (NOT pushed). The triage files
  (`core/triage.py`, `core/triage_classify.py`, `core/ab_signature.py`, `core/signature_reviews.py`)
  are UNTRACKED/uncommitted, as are edits to `catalog_creator.py`, `catalog_loader.py`,
  `engine_registry.py`, `wfw_web.py`. **This session added ZERO code** — only memory files under
  `.claude/.../memory/`. Git state = unchanged from session start.
- Python: `.venv\Scripts\python.exe` (bare `python` is a Store stub). For `core` imports set
  `$env:PYTHONPATH=(Get-Location).Path` (bash: `PYTHONPATH=.`). Server on 5099.
- DBs: `data/clues_master.db` (clues + `wfw_solve` verdicts + `catalog_templates`),
  `data/cryptic_new.db` (reference: `indicators`, `synonyms_pairs`, `literal_words`,
  `link_words`, `definition_answers_augmented`, `wordplay`). Test on TEMP DB COPIES, never the real DB.

**The add-indicator sub-type machinery (workstream A)**
- Dropdown sub-types are a HAND-MAINTAINED dict `_IND_SUBTYPES` (`core/wfw_web.py:70`), types list
  `_IND_TYPES` (`:63`). It is **missing `alternation` entirely** and its own comment is stale.
- The write layer is `admin_db.add_indicator(word, wordplay_type, subtype)` (`core/admin_db.py:209`);
  it validates `selection`/`letter_shift`/`charade_positional` sub-types but NOT deletion/others.
- The engines DO NOT read the raw sub-type string — they map `(wordplay_type, subtype) -> rule/op`
  via HARDCODED dicts, then run a code operation keyed by that rule name:
  - selection: provider `selection_rules` (`engine_registry.py:229`) → `selection_indicators.SUBTYPE_RULE`
    (`core/selection_indicators.py:19`) → rule name → `selection.select_span(ctx, word, rule)`
    (`core/selection.py:99`) → `SPAN_RULES[rule]` lambda (`core/selection.py:86`, e.g.
    `"alternate": la[0::2]/la[1::2]` at `:91` — BOTH alignments, so there is **no even/odd distinction**).
  - deletion: provider `deletion_subtypes` (`engine_registry.py:198`) → `deletion.SUBTYPE_OP`
    (`core/deletion.py:30`, e.g. `head→behead`) → `deletion._OP_FUNCS` lambda (`:85`).
  - charade_positional: provider `charade_positional_subtypes` (`engine_registry.py:213`) →
    subtypes `after/before/after_down/before_down`.
  - `selection_indicators.find_indicators` reads the provider (`selection_indicators.py:57`).
  **This whole path is the thing to TRACE AND CONFIRM before any design (see §6) — the last session
  asserted parts of it without checking and got some wrong.**

**Palindrome vs reversal (settled this session, corrected a wrong earlier claim)**
- The two engines are MUTUALLY EXCLUSIVE by construction: `palindrome_engine.py:29` abstains unless
  `answer == answer[::-1]`; `reversal_engine.py:44-46` abstains WHEN `answer == reverse(answer)` and
  otherwise needs clue fodder that reverses to the answer. They can never fire on the same clue.
- Therefore the earlier "palindrome shares reversal words so it would pollute the shared table" claim
  is WRONG (shared words cannot cross-fire). Palindrome CAN be DB-driven — in its OWN table (its
  `_OPP_PAIRS` "both east AND west" logic just doesn't fit a flat word→type row). See
  `memory/hardcoded_lists_anti_cheat.md` CORRECTION block.

**The alternation bug — CONFIRMED (code + data), still UNFIXED**
- `core/container_inner_alternation_engine.py` finds its alternation indicator only via
  `indicator_types(word) & {"alternation","alternating"}` (`_ALT_TYPES` at `:29`; used `:102`;
  abstains `:103-104`). It has NO `selection_rules` fallback — unlike its sibling
  `charade_alternation_engine._alt_licensed` (`:33`) which also accepts anything `selection_rules`
  maps to `alternate`.
- DATA: of the words meaning "take alternate letters", **24 are invisible to this engine vs 14 it
  can see** — and the missed ones are the common cues: `even, evenly, odd, odds, regular, regularly,
  at regular intervals, cyclically, every other, every second, intermittent(ly), skipping, …`
  (one row `[1]…p[a]r[r]o[t]` is corrupt junk — separate DB-hygiene issue).
- IMPACT: any "container around an alternate-letters inner" clue signalled by one of those words is
  silently abstained on; no other engine covers that mechanism. FIX touches a working stage engine
  → per CLAUDE.md must be A/B-gated and sequenced AFTER the sub-type table work.

**TERMINAL — the worked example of the diagnosis standard (workstream B), verified**
- telegraph 31285, "Last spell in jail oddly rejected" = TERMINAL. Reading: TERM(spell) + IN(literal)
  + AL(alternate letters of "jail", via "oddly [rejected]"). Real solver near-miss = plain `charade`,
  placed only TERM.
- Determination = **missing SIGNATURE, not missing engine.** The capable path exists:
  `charade_signature_engine` `SEL_F` slots (`:139-149`, `:44` doc) do charade+literal+alternation-
  selection and are NOT forward-only; `find_indicators` supplies the indicator (`:177-182`); "oddly"
  is registered `parts/alternate`. But **0 of 1669 `catalog_templates` combine `LIT_F` and `SEL_F`**,
  so no template of shape `def@start + SYN_F + LIT_F + SEL_F` exists. By the tier rubric that missing
  signature is PASS-tier (L=1, G=false, 8-letter answer).
- NOTE: the bespoke `charade_alternation_engine` IS forward-only (`_alt_targets` `:49-71` only pairs
  an indicator with fodder AFTER it) — a real but SECONDARY limitation; the `SEL_F` path is the
  capable route. (An earlier "missing engine capability" call based only on `charade_alternation`
  was wrong and was corrected.)

**31285 solve spread (fresh, un-enriched — the honest test surface):** 32 clues, 14 pass, 3 pending,
15 fail. (31284 is contaminated — the user hand-enriched DB pieces there; use 31285.)

---

## 3. THE ANTI-CHEAT PRINCIPLE (this is the spine of both workstreams)

Full note: `memory/hardcoded_lists_anti_cheat.md`. In short:
- A hardcoded list of ENRICHABLE data can only be enriched by Claude editing code, which BYPASSES the
  user-commit gate and is an ideal place to CHEAT (silently add a word/mapping to turn a fail green).
- Fix = REMOVE THE OPPORTUNITY (soundness by design), not "trust me": enrichable vocabulary AND the
  mappings that WIRE it to a clue → the DB, behind the suggest→user-commit gate.
- The SHARP test for what may stay in code: only an operation IMPLEMENTATION (the letter-manipulation
  lambda) may — and only because it is INERT until a gated DB mapping AND a gated DB word both point
  at it. **Anything that WIRES an existing capability to a clue is DATA and must be in the DB — this
  INCLUDES the `SUBTYPE_RULE`/`SUBTYPE_OP` mappings.** A hardcoded subtype→rule map IS a cheat hole
  (if a word already carries the subtype and only the map entry is missing, adding that one line
  silently unlocks a solve). There is NO safe "leave the subtype lookup in code" option.

---

## 4. HARDCODED-LIST AUDIT RESULT (workstream A context)

- **Genuine holes (enrichable, still hardcoded):** `SUBTYPE_RULE` (`selection_indicators.py:19`),
  `SUBTYPE_OP` (`deletion.py:30`), the three inconsistent alternation type-sets
  (`alternation_engine.py:27`, `charade_alternation_engine.py:30`, `container_inner_alternation_engine.py:29`),
  `dd_engine._DD_CONNECTORS` (`:40`), `palindrome_indicators` (`_SINGLE/_PHRASES/_OPP_PAIRS`),
  `spoonerism_indicators._LEAD`.
- **Already behind the gate:** `literals` (live provider `engine_registry.py:257` reads
  `literal_words`; hardcoded `LITERAL_WORDS` is only a fallback seed; "in" verified in the table);
  all indicator WORDS (via `indicators` table + `get_indicator_types`).
- **Legitimately code (behaviour — leave):** role/mechanism maps, `_VALUE_MECH`, POS-tag sets, the
  operation lambdas (`selection.SPAN_RULES`, `deletion._OP_FUNCS`), tier-rubric op sets.
- Scope caveat: the audit covered module-level constants + the indicator modules; a deep sweep for
  word-lists written INLINE (`if w in {"..."}`) inside functions was NOT done.

---

## 5. WORKSTREAM B — the failure-diagnosis / triage classifier (the real goal)

- **Design:** `memory/nightly_triage_process.md`. Each night after scrape→solve, review every FAIL/
  PENDING clue and classify WHY: **missing data / missing signature / missing engine**. Claude
  DIAGNOSES and INFORMS ONLY — never resolves, commits, iterates, or re-solves. ONE pass, no loop.
  Do NOT chase pass rates; "cannot parse" is acceptable.
- **What's BROKEN:** `core/triage_classify.py` is a PARALLEL SOLVER. `classify()` (takes
  `(clue_text, answer, wiring)` — a solver's inputs) runs `catalog_creator.discover(...)`
  (`triage_classify.py:149`), a from-scratch charade/anagram/container/reversal solver, plus its own
  word-only detectors (`:192-199`, misses phrase gaps). It NEVER reads the real solver's result
  (`wfw_solve.solved_by`/`warnings`). So it "explains" the real failure by re-solving with a
  DIFFERENT solver — useless, and re-solving violates the design. Full note:
  `memory/triage_classifier_recovery.md`.
- **What is FINE (do not rebuild):** `core/triage.py` (evidence/report — reads `wfw_solve`, uses
  `diagnose.pieces` read-only, writes markdown, no DB write, no re-solve) and the `/triage` page.
- **The corrected classifier** must diagnose from the REAL solver's near-miss (`wfw_solve.solved_by` +
  `warnings`) + a phrase-aware DB check via `core.diagnose.pieces` (which IS phrase-aware —
  `diagnose.py:73-108`). It must NEVER call `discover()` and NEVER re-solve.
- **The STANDARD the user demands (raised this session):** for a clue where the MATERIAL is present
  but nothing assembled, "mixed / needs a human eye" is NOT acceptable. Claude must DETERMINE
  signature-vs-engine by READING the engines + cascade wiring (the user cannot judge this). Worked
  example = TERMINAL (§2). Reliable instrument = `python -m core.diagnose <clue_id>` pieces view +
  hand analysis. `diagnose.engines()` is UNRELIABLE (runs a subset without role-validity predicates).

---

## 6. WHAT IS NOT DONE / DISCARD

- **Nothing built or committed.** No schema change.
- **The subtype→rule→operation end-to-end trace was promised and NOT completed.** §2 lists the likely
  path with file:line — the first job of workstream A is to READ that path end-to-end and confirm it,
  producing a verified map (subtype → `SUBTYPE_RULE`/`SUBTYPE_OP` → rule/op name → `SPAN_RULES`/
  `_OP_FUNCS` lambda), THEN design. Do not design before the trace is verified.
- **Discard all of the previous session's storage-design proposals** (new table vs new column vs
  "canonicalise, no storage"). They were ungrounded guesses. The only VERIFIED design constraint is
  §3: the subtype→rule/op MAPPING is data → must be in the DB behind the gate; the operation lambdas
  stay in code. Where exactly in the DB it lives is OPEN and must be decided WITH the user AFTER the
  trace — do not walk in with an architecture.

---

## 7. SUGGESTED FIRST MOVES FOR THE NEW THREAD

1. Read this handover, `memory/nightly_triage_process.md`, `memory/hardcoded_lists_anti_cheat.md`,
   `memory/triage_classifier_recovery.md`.
2. Workstream A: produce the VERIFIED subtype→operation trace (§2/§6), file:line, no design. Show it.
3. With the user, agree where the subtype vocabulary + mapping lives in the DB and the smallest first
   step; build one step; test through the real add-indicator route on a temp DB copy.
4. Confirm/fix the alternation bug (A/B-gated) as a separate step.
5. Then RETURN to workstream B: rebuild the classifier brain to diagnose from the real near-miss +
   `diagnose.pieces`, to the signature-vs-engine standard (§5).

## 8. Memory files written this session (already saved)
- `memory/triage_classifier_recovery.md` — the parallel-solver diagnosis + recovery principles + the
  raised signature-vs-engine standard.
- `memory/hardcoded_lists_anti_cheat.md` — the anti-cheat principle + audit + the sharp
  data-vs-behaviour test + the palindrome/reversal correction.
- `MEMORY.md` index updated with both.
