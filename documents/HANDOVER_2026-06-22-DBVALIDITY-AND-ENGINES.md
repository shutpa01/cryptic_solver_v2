# Handover — 2026-06-22 — DB-validity verifier, bidirectional lookup, and the 4 engines to build

Branch: `redesign`. Read this with `memory/session_2026_06_22_improvements.md` and
`documents/DT_31272_fails.md`. The next session's JOB: build the 4 capability engines in
section 6. Everything else here is context + what's already done.

---

## 1. Git / commit state

- **Committed this session**: `089e1fe2` — ASTRIDE deletion fix, charade fail-evidence
  rewrite, inflection fixes (2a generator + 2b abbreviations-no-inflect), outer-charade
  container engine, auto-signature queue. (engine_registry.py in that commit also carried
  earlier prior-session wiring work — unavoidable, files don't split.)
- **UNCOMMITTED but VALIDATED (recommend committing first thing)**:
  - `core/role_validity.py` (NEW) + the DB-validity check added to ~30 engines' `_verify`.
  - `core/live_db.py` bidirectional synonym lookup (always-union).
  - `core/anagram_signature_engine.py` (inline DB-validity check + fallback tightening).
  These are A/B-clean (see §3). **Suggest: commit them before starting the engine builds**,
  so the new work sits on a stable base.
- **DB changes (data/*.db — gitignored, backed up, NOT in git)**: see §5.

## 2. THE RULE (this is the spine — do not violate)

Every recorded role MUST be DB-backed; a role is NEVER assigned by elimination. Concretely:
indicator → a DB indicator of the right type; link → a DB link word; piece → a DB value by
that engine's mechanism (synonym/abbr/literal/selection; anagram fodder = clue's own
letters); definition → DB-confirmed (else the parse is pending/fail, never a clean pass).
Honesty over score: a fabricated pass is the worst possible outcome. NO central/generic
verifier (the old 1000-line one failed) — each engine verifies in its OWN `_verify`.

## 3. What was built this session (the safe, validated layer)

### 3a. `core/role_validity.py` — the per-engine DB-validity check
- `set_predicates(indicator_types, is_link)` is called ONCE at the top of
  `engine_registry.solve()` (predicates come from the wiring; the frozen ctx can't carry
  them, hence a module holder). Fails CLOSED if unset (never a silent pass).
- `unbacked_roles(parse)` returns the list of roles that are not DB-backed. It maps each
  indicator annotation's NOTE to the expected DB type (anagram/container/insertion/deletion/
  reversal/homophone/hidden/alternation; a "location" note or any unmapped note = "must be
  SOME DB indicator"), and checks links via `is_link`.
- Each engine's `_verify` calls it: `bad = role_validity.unbacked_roles(parse); if bad: parse
  set fail`. Applied to: anagram (inline, stricter), charade_signature (`_verify_charade`),
  charade_positional, charade_named_deletion, charade_hollow, charade_deletion,
  container_deletion, container_signature, container_charade_signature,
  container_inner_charade, container_outer_charade, container_acrostic, reversal_container,
  substitution, anagram_substitution, anagram_charade_signature, anagram_container_signature,
  anagram_multi_substitution, anagram_deletion, anagram_insert_letter, acrostic, alternation,
  homophone, palindrome, spoonerism (all standard), dd (`_verify_dd`, into its warning
  gather), hidden (`_verify_hidden`, bad→pending since hidden never fails), and the reused
  evidence verifiers reversal_engine/reversal_charade_engine/deletion_engine.
- **CONSEQUENCE for any NEW engine you build: its `_verify` MUST call
  `role_validity.unbacked_roles` too**, or it can emit fabricated passes.
- A/B (500 clues) vs pre: 5 PASS LOST, every one a proven fabrication (ASTRONAUTS
  "cooked high", DEAFNESS "prepared to make", Double-breasted "needs new", APERTURE
  "perhaps"=DBE-not-anagram, SONAR "funny Batty"); 0 legit losses. Gains can't be
  fabricated now — the check gates every pass.

### 3b. `core/live_db.py` get_synonyms — bidirectional (always union)
Was reverse-ON-MISS, so a word with ANY forward synonym missed its reverse-only pairs
(`bubbles` has forward synonyms, yet `('lather','bubbles')` is stored one-way → bubbles→
LATHER missed). Now ALWAYS unions forward + reverse (NOCASE value-column index = cheap).
A/B: +5 passes, 0 losses; also fixed ASTRONAUTS PROPERLY (high fliers now a DB def, so the
indicator is correctly just "cooked"). (abbreviations/homophones still reverse-on-miss —
left as-is; revisit only if a clue needs it.)

## 4. The A/B harness — `core/_ab_general.py`
- `python -m core._ab_general logs/out.json 500` solves a deterministic 500-clue spread
  (DB-only, no AI) → JSON of {id:[status,operation]}.
- `python -m core._ab_general --compare base.json new.json` diffs (PASS gained/lost/changed).
- Workflow: capture baseline, make change, capture new, compare, AND audit every lost pass
  to prove it was a fabrication (don't trust the count — prove each loss). Baselines from
  this session are in `logs/ab_*.json` (latest good = `logs/ab_bidi.json`).

## 5. DB changes applied this session (data/cryptic_new.db + clues_master.db)
- Deletion reclassify: removed `deletion/head` from 14 location words + inflected
  `starting`/`fronting`. Backup: `data/cryptic_new.db.bak-incl-dedelete`.
- Catalog templates added (clues_master.db): 1655 (HOUSEBOUND anag+container shape),
  1657 (PESETA `SEL_F+SYN_F+ABR_F` outer-selection charade), 1658 (APPEASES container,
  via the auto-sig queue approve test). Backups: `catalog_*_bak_anacon_manual`,
  `catalog_*_bak_selouter`, `catalog_*_bak_creator`.
- Indicator added by USER: `tips = parts/outer_use` (for PESETA).
- pending_signatures table (auto-sig queue) created in clues_master.db.

## 6. THE BUILD LIST — 4 capability engines (the next session's work)
All must be answer-driven, indicator-gated, links-last, and call
`role_validity.unbacked_roles` in their `_verify`. Build ONE at a time, A/B each, prove
no fabrication. Cascade wiring goes in `engine_registry.solve`.

1. **TRAIL (10075102)** — charade mixing a SELECTION piece and a REVERSAL piece.
   `T` (end of accoun·t; last-letter selection, indicator "end of") + `RAIL` (storyteller=
   LIAR, "returning"=reversed). charade_signature does SEL_F, reversal_charade does REV_F;
   neither combines them. Cleanest: a NEW evidence-driven engine that tiles the answer with
   pieces that may be plain (SYN/ABR), reversed (REV, gated by a reversal indicator), or
   selection (SEL, gated by a selection indicator via core.selection.select_span +
   selection_indicators.find_indicators), requiring ≥1 selection AND ≥1 reversal so it can't
   intercept plain charade/reversal_charade. Model the tiler on
   reversal_charade_signature_engine._reconstruct (add a SEL branch).

2. **DRESSES (10075104)** — first-letter SUBSTITUTION of a synonym base. `TRESSES`(hair)
   with first letter (T) replaced by `D`(diamonds) = DRESSES. substitution_engine replaces a
   letter but not at a LOCATED position of a synonym base. Needs: base = synonym (TRESSES of
   "hair"), a located letter (first, via selection indicator "cut initially") replaced by a
   DB value (D of diamonds). Answer-driven (base with that position swapped == answer).

3. **DOGFIGHT (10075108)** — anagram-container + a charaded abbreviation prefix. `D`(Day) +
   [anag of "fog hit" containing `G`(Germany's leader), "hiding"=container]. anagram_container
   handles the (anag ∋ value) part but has no charade prefix. Either extend the discovery or
   a new compound. Harder; do after 1 & 2.

4. **Baghdad (10075125)** — `BAG`(claim) + interior `H` + `DAD`(father); "reported" hints a
   homophone or H source. Needs a closer parse first; lowest priority.

## 7. Data-gap enrichments (NOT engine builds — user approval, never direct writes)
- MINIM: type "the same either way" / "same either way" as a palindrome indicator.
- CALF: "shows" as a link word (it's the DD connector; currently is_link False → DD abstains).
- (SUNGLASSES already handled by the user's filler tag on "some".)

## 8. Auto-signature QUEUE (built, live)
`core/signature_queue.py` + `catalog_creator.auto_discover_and_queue` + wfw_web banner
(/approvesig, /rejectsig). The per-clue re-run (the ↻ button) discovers + queues a proven
signature; the page banner has Approve/Reject. Page is in QUEUE mode (auto_signature=False).
Discovery covers charade/anagram/reversal_charade/anagram_charade/container(incl SEL_F) — it
does NOT cover the 4 new shapes above, so those need real engines, not auto-sig.

## 9. Operating rules (the user is exacting — honour these)
- Verify on the LIVE page (:5099) before claiming anything. Restart the server after code/DB
  changes (it caches wiring at startup). Start: `python -m core.wfw_web`.
- NEVER state a guess as fact. Read the actual code/DB; cite file:line. The user has zero
  tolerance for guessing ("you must stop guessing").
- NEVER write to the reference DBs directly — enrichment via queue/approval only.
- Prove every A/B loss is a fabrication; never report a raw pass count as validation.
- One engine at a time, A/B-gated. Agree the design before coding if unsure.

## 10. Server / quick commands
- Server: `python -m core.wfw_web` (port 5099). Kill the old PID first (it caches wiring).
- Solve one clue in code: `engine_registry.solve_clue_text(clue, answer, wiring, direction=...)`
  with `wiring = engine_registry.db_only(engine_registry.make_db_wiring())`.
- DT puzzle: telegraph 31272 (clue ids ~10075102–10075132); 24/32 pass.
