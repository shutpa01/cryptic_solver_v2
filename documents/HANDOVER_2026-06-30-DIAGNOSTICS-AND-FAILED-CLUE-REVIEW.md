# Handover — 2026-06-30 — diagnostics + failed-clue review (resume here)

Cold start for a NEW thread. Read top-to-bottom. We are **reviewing FAILED clues** from recent
DT/Times puzzles and triaging each. The solver is a reliable basis — most "failures" are
**missing data or a missing signature variant, NOT engine bugs** (proven repeatedly this
session — I twice wrongly guessed "engine gap" and was wrong both times).

---

## CODE STATE
- Branch `redesign`, **HEAD = d697d633** (NOT pushed). Working tree clean except the
  pre-existing untracked `core/atomsig/` (leave it).
- Commits this session: **a6c8ee28** (literals→DB, charade_multi_deletion + reverse_charade
  engines, anagram-degeneracy + slash-definition false-PASS guards, specific engine-name
  recording, `core/diagnose.py`, per-clue signature-suggestion UX, lifted discoverer cap,
  clue-page literal/homophone adds) and **d697d633** (reversal_charade_signature crash fix).
- **Gitignored / LOCAL ONLY (not in any commit):** all DB state — catalog signatures (DAUNT
  1664, VICARIOUS 1665, ENID activation of template 107), the `literal_words` table, and the
  me/at synonym deletions. A fresh checkout won't have these.
- **GATE before push/merge: overnight full-corpus A/B (`core/_ab_general.py`) — NOT yet run.**
  Commits are checkpoints, not blessed-for-merge.

## THE KEY NEW TOOL — use it FIRST on every failed clue
`python -m core.diagnose <clue_id>` prints two views (READ-ONLY, touches no engine):
1. **pieces** — definition candidates (DB-confirmed) + what each clue word/phrase resolves to
   (synonyms/abbreviations/literal/indicators), each flagged `in-answer` (charade/inner),
   `in-answer(reversed)` (reversal), or `outer? (answer = value + insertion)` (container outer).
   → tells you instantly whether the raw MATERIAL to reach the answer is present.
2. **engines** — runs each major engine and reports `status | pieces placed | unaccounted | reason`.
Combine them: material present + engine says "no signature matched" ⇒ **missing signature
variant**. A word with no value ⇒ **missing data**.

## TRIAGE METHOD (what actually worked this session)
1. `diagnose` the clue. Work out the intended parse by hand.
2. If a **piece's value is missing** → enrich DB (add synonym/abbreviation; literals via the
   `literal_words` table / clue-page "Literal" add). e.g. INTENDED needed `Set of books`→NT
   (user added → solved); HANDSOME needed `me`→ME (literal, present) + the signature.
3. If **all pieces present but it fails** → missing **signature variant**. Add via
   `catalog_creator.add_signature` (server STOPPED, DB Browser CLOSED). **CHECK IF IT EXISTS
   INACTIVE FIRST** — many needed sigs are in the catalog with `active=0`; ACTIVATE
   (`UPDATE catalog_templates SET active=1 WHERE id=…`, backup first) rather than re-add
   (ENID = template 107). add_signature dedups by signature string regardless of active.
4. If a **leftover word has no role** (it's padding) → it must be a link or **surface filler**;
   the engine CORRECTLY refuses to leave a content word unaccounted (not a bug). e.g. INTENDED
   `Set`, HANDSOME `little` ("little old me" padding).
5. Only a genuinely NEW structural shape → a bespoke engine (this session: charade_multi_deletion
   = OBERON/ALLURE; reverse_charade = whole-charade reversal = TRAIN/SAGA/SIGNALS, 19 gains).

## RECURRING FAILURE PATTERNS (catalogue)
- **Missing signature word-count/role/position variant** — DAUNT (ABR_F+SYN_F(5w)), GLINT
  (SEL_F+SYN_F(2w)), ENID (SYN_F+REV_I(2w) def:start — was INACTIVE), VICARIOUS (SYN_F(2w)+SYN_F(3w)),
  HANDSOME (SYN_F+ABR_F+LIT_F). The catalog has neighbours but not the exact slot shape.
- **Reverse / reorder charades** — pieces in opposite order to the clue. Need a positional
  indicator: `after` works (INTENDED, via charade_positional). `by` is NOT yet a positional
  indicator (user believes it should be; HARMLESS = H+ARMLESS needs it). Adding `by` has a
  **4.2%-of-clues blast radius** → A/B required; subtype would be `after` (trailing piece leads).
- **False-PASS via INDIRECT operations** (deleting/anagramming/reversing a SYNONYM not the
  literal) — the fabrication theme. Guards added this session: anagram-degeneracy (reject
  fodder == answer forward/reversed), slash-definition artifact (`live_db.get_synonyms` drops
  def rows with `/` for multi-word queries), literal-only in charade_multi_deletion. WATCH FOR
  MORE; a "clean pass" does NOT prove faithfulness (a coincidental tiling passes by construction).
- **Stale stored parses** — the page renders from the STORE (cache), not a live re-solve.
  Deleting/correcting reference data does NOT auto-refresh already-solved clues; they show stale
  pieces (even a synonym you've deleted) until re-solved. Fix: `POST /reload` form `id=<id>&only=<id>`.
  (This caused the "old→ME synonym not in the DB" confusion — stale fail-evidence from the
  deleted `me`→old row.) Consider: invalidate stored parses that used deleted rows.

## DB CLEANUP CAPABILITY (done this session; more to do)
Method: verify the ids match (guard on word), DUMP rows to a recovery .tsv, then guarded DELETE.
Deleted 357 corrupt synonyms (all `word='me'` + `word='at'`, e.g. me→toadies, at→sailing).
Recovery file: `data/deleted_me_at_synonyms_20260630.tsv` (re-insertable). The legit
self-reference (setter/compiler/I ↔ me) survived (it lives in `word='setter'/'I'` rows +
reverse lookup). Likely MORE pollution clusters exist (e.g. `old`→HAND from a `hand`→old row;
"old hand" junk). The bidirectional lookup means a junk `X→Y` row also yields `Y→X`.

## SUGGESTION / APPROVAL WORKFLOW (built this session)
Per-clue re-run runs `discover()` → verifies → QUEUES candidate signatures. They now render
**inside the clue card** (`_clue_signature_suggestions` in wfw_web) with a one-click **Approve &
re-solve** that KEEPS the clutch and re-solves in place (the old global-banner + blank-screen bug
is fixed). The discoverer's 4-word cap is LIFTED (finds long-phrase pieces like DAUNT's 5-word
synonym). NOTE: discover is answer-driven free-tiling → suggestions are HUMAN-vetted via Approve;
NEVER auto-keep (auto-keep = the evidence-system fabrication the user explicitly rejected).
"Enumeration generator" idea was discussed and PARKED for this reason.

## STANDING RULES & GOTCHAS
- **Server**: `.venv/Scripts/python.exe -m core.wfw_web` on 127.0.0.1:5099, exactly ONE
  listener (stale-server trap — always restart + verify after a code change). STOP it before
  any catalog/DB write (lock contention).
- **DB Browser** open on cryptic_new.db / clues_master.db LOCKS writes — must be closed before
  add_signature / synonym writes.
- **The LIVE charade engine is `charade_signature_engine` (catalog-driven), NOT
  `charade_engine.py`** (evidence, sidelined). Don't edit the evidence engine to fix a live clue.
- **Catalog adds**: derive from an existing template (don't hand-type), add via
  `catalog_creator.add_signature` (auto-backup), re-solve, KEEP ONLY IF a faithful pass, else
  roll back. Catalog lives in clues_master.db (gitignored, auto-backed-up).
- **Faithful labels** matter: a piece is literal / abbreviation / synonym by what it ACTUALLY
  is (`me`→ME and `PE`→PE are literals, NOT abbreviations).
- Don't modify working engines for edge cases; the fix is usually data/signature. Don't claim
  "engine gap" without tracing (I was wrong twice).

## MEMORY POINTERS (auto-loaded)
`diagnose_tool.md` (the tool + suggestion UX + lifted cap), `engine_recording_and_anagram_degeneracy.md`,
`reverse_charade_engine.md`, `charade_multi_deletion_and_indirect_deletion.md`,
`slash_definition_artifact.md`, `literals_db_conversion.md`, `charade_live_engine_is_signature.md`.

## NEXT
Resume reviewing failed clues (the user is going through recent DT/Times puzzles, e.g. the
10076xxx range). For each: `diagnose` → triage (data / signature / filler / reorder) → fix the
right layer → re-solve to verify. Keep an eye out for more false-PASS fabrication and stale
stored parses. Run the overnight A/B before any push.
