# Handover — 2026-06-02 (evening) — catalog stage STALLED, session largely wasted

## BLUNT ASSESSMENT (read this first)
This session achieved very little against its actual goal (build the catalog engine).
A handful of small fixes landed, but the catalog — the whole point — was not started.
The time went to the assistant repeatedly answering from a skim, from memory, or from
whichever code file was convenient, instead of reading the design document first. The
user had to catch and correct the same failure over and over. Hours were burned to save
seconds of reading. Do not repeat this. READ documents/SOLVER_REDESIGN.md in full and
cite it before answering or acting. See memory feedback_read_design_fully.

## HOW TO WORK WITH THIS USER (non-negotiable)
- Read the design (documents/SOLVER_REDESIGN.md) FULLY and cite section numbers BEFORE
  answering any design question or doing any work. Never substitute memory, a partial
  read, or a convenient code file. When sources conflict, read the doc and surface the
  conflict out loud. (memory: feedback_read_design_fully)
- Be brief. Conclusion first. No essays, no options-dialogs unless asked.
- No backticks / no coloured text (user can't read it). Plain text for code/paths.
- No "honest answer"/"honestly" prefaces. Don't tell the user a decision is theirs.
- Diagnose/propose, do not build, until explicitly told. Commit/push only when asked.
- Quality not speed. Skimming to go faster is the slowest path here.

## GIT STATE (verified, not from memory)
- Branch: redesign, IN SYNC with origin/redesign. All session commits pushed.
- This session's commits (all pushed):
  - 1ec7cc4b — core/ DD engine + three-state verdict (pass/pending/fail)
  - 2f66c3eb — Design: add §13 per-puzzle render switch (transition rule)
  - c024e7f8 — Preserve the atomisation in the WFW substrate
- Working tree clean except: .claude/settings.local.json, a worktree, and untracked
  scratch/handover docs (core/_scratch_test.py, documents/HANDOVER_*.md). Not part of work.

## WHAT ACTUALLY LANDED THIS SESSION (the small wins)
1. Three-state verdict pass/pending/fail across hidden + DD, with cascade stop rules
   (memory: core_cascade_stop_and_verdict_rules):
   - hidden is terminal whenever the contiguous run is found (pass or pending), never fails;
   - DD: both-DB -> pass; one DB half + Haiku-confirmed other -> pending; one half confirmed,
     second a definite NO -> fail; pass/pending stop the cascade.
   - pending/fail line = is the gap a queueable enrichment candidate. STOP decided per stage.
   - store + render carry 'pending' (amber badge/banner).
2. DD shape-gate fix: only an OPERATIVE indicator in the LINK REGION between halves
   disqualifies a DD (not any indicator anywhere). Fixed SHORTENING "Reducing fat in pastry".
   Removed the arbitrary MAX_WORDS=8 cap; raised definition window 5 -> 8.
3. ai_synonym.is_definition_of now returns True/False/None; DD records FAIL only on a
   definite NO, abstains (None) on a Haiku error/unknown — so a transient Haiku failure
   never persists a durable FAIL. (This was a real bug: a FAIL had been written for 9940875.)
4. Design §13 added: per-PUZZLE render switch (legacy vs new page) via a puzzle_render flag;
   coverage-gate/sign-off; no per-clue fallback in a flagged puzzle. (memory: transition_per_puzzle_render_switch)
5. Atomisation now PRESERVED: the full WFWAtomContext is serialised to wfw_solve.atoms at
   save and reconstructed at render (store.load_atoms); re-atomise only as fallback for old
   rows. Additive, non-destructive migration. (wfw_atoms.context_from_dict; store.py; engine_registry; wfw_web)

## STORAGE / SHADOW-DB NOTE (the user flagged this hard)
The new WFW substrate (wfw_solve/wfw_piece/wfw_link) is written into the PRODUCTION
data/clues_master.db. The standing project principle is "live DB never touched, shadow
only" — so this is a known violation the user accepted proceeding with FOR NOW, pending a
shadow DB later. Verified: no core/ code does UPDATE/DELETE/DROP/ALTER on any pre-existing
table; the only touch of an existing table is append-only INSERT OR IGNORE into
pending_enrichments (currently 0 rows). Reference writes (accepted enrichments) go to
cryptic_new.db legitimately. Atoms/flag live wherever the substrate lives.

## THE CATALOG (stage 4) — NOT STARTED. Established facts so the next thread doesn't re-derive:
- The DESIGN is what to follow. But parts of SOLVER_REDESIGN.md are STALE and were never
  corrected: §4/§5.4/§7/§10/§11 describe a TEMPLATE catalog reusing the big matcher
  (base_matcher) + a catalog_templates table. The build-progress memory (2026-05-31,
  core_redesign_build_progress) records this was SUPERSEDED by a decision to use PER-
  OPERATION ASSEMBLERS, and says "doc section 10 needs that correction." It was never done.
- TWO provenance models exist in core/:
  - core/model.py (Piece/Provenance/ParseResult — the literal §3 design model). core/catalog.py
    (a COMPLETE per-op assembler set: charade, container, anagram, reversal, deletion,
    homophone, acrostic) and core/dd.py are built on it. NOT wired into anything.
  - core/wfw_model.py (Source/Link/Annotation/Parse). The LIVE system — hidden_engine,
    dd_engine, store, screens, three-state verdict, atom preservation — is all on this.
  These two do not connect. catalog.py cannot just be switched on.
- USER'S STATED PLAN (confirmed this session): copy the matching pieces FROM PRODUCTION
  (signature_solver) into the redesign, and TRANSLATE their output into the new word-for-word
  record so it renders the new page. This matches design §7 ("reuse the matcher, do not
  rebuild it"). NOTE the unresolved tension: the user wants to copy from production, while
  core/catalog.py is a from-scratch per-op rewrite — reconcile this explicitly, do not assume.
- Production matcher, how it works (READ THE FILES, don't trust this summary):
  solver.py (def extraction) -> word_analyzer.py (per-word roles from RefDB) ->
  base_matcher.match_base (F/I patterns, span placement, _lookup_slot, _verify_combo +
  per-op checkers in matcher.py) -> confidence.py. Output = (entry, assignment) where
  assignment = {fodder_order:[(word,role,value)], indicator_indices, lnk_indices}.
  CRITICAL: production confirms the pieces ASSEMBLE but emits NO per-letter map. The new
  page needs per-letter/per-span provenance, which must be COMPUTED per operation from the
  ordered values. catalog.py already contains that per-letter logic.
- Pieces to copy: base_matcher.py, matcher.py, base_catalog.py + data catalog json,
  word_analyzer.py, tokens.py. RefDB (db.py) is NOT reloaded — it's loaded once in
  engine_registry.make_db_wiring and injected (shared by hidden/DD); pass the same instance.
  Baggage to leave: grammar_triage, haiku_*, api_solver, tests.
- CATALOG FREQUENCY: do NOT use base_catalog.json to judge frequency/coverage — design
  §5.4 and §9 say the distilled BASE catalog is the WEAKEST of the three and the old/
  positional catalog OUT-COVERS it. The fuller record is data/positional_catalog.json
  (218KB); also data/grammar_catalog.json. Consolidation must be COVERAGE-DRIVEN and
  MEASURED across all three, per the design — not a count off the weak catalog.
- BUILD UNIT (corrected this session): each operation is its own clue type, like hidden/DD.
  The verifier and the screen are OPERATION-SPECIFIC. So work vertically: ONE operation done
  fully end-to-end (find parse -> translate to wfw_model per-letter record -> operation-
  specific in-engine verifier + three-state verdict -> bespoke screen -> wire into
  engine_registry after DD -> persist with atoms -> prove on real clues), THEN the next.
  There is no generic "add a verifier + screen" step before the operations.
- Design gives RUNTIME order = priority by frequency (most common templates first) and one
  precedence rule that matters: an exact reversal must be tried BEFORE anagram (exact reverse
  is a reversal by definition, not an anagram). The design does NOT prescribe a per-operation
  BUILD order — that's left to us; by frequency, charade is overwhelmingly most common.

## OPEN DECISIONS (still unresolved)
1. The catalog engine's cascade STOP rule (decided per stage) — settle when wiring it in.
2. Cross-engine FAIL ranking: when no engine gets a clean PASS, how to rank competing FAILs
   (today a DD fail is just returned as fallback). Becomes live once a 3rd engine exists.
3. Which provenance model is canonical (wfw_model is the de-facto live one; model.py+catalog.py
   are stranded). The catalog must emit wfw_model to render. Confirm before building.
4. The stale design sections (§4/§5.4/§7/§10/§11 template-catalog text) need correcting to the
   per-op-assembler decision BEFORE building, so the doc stays the authority.

## RUN / TEST
- Redesign true-test UI: python -m core.wfw_web -> http://127.0.0.1:5099/ . Enter clue id(s).
  First request loads RefDB (~1.75M synonyms) + spaCy (slow once). NOT production.
- Production system (the real solver, separate): sonnet_pipeline/run.py + signature_solver.
- Good ids: 1710238 ANDEAN, 1710279 TIGER, 1710251 SARONG, 1710368 IDLY (hidden, pass);
  9940875 SHORTENING, 9942219 DARTS (DD, pass).

## LIKELY NEXT STEP
Correct the stale catalog sections of the design to the agreed per-op-assembler + wfw_model
approach (so the doc is authoritative), settle the canonical-model question, then build the
FIRST operation (likely charade) as a full vertical slice. Read the design first.
