# HANDOVER 2026-05-30 — AS-IS doc, confidence incident, 31251 audit

Three live workstreams from 2026-05-29, all captured in files. Read this plus the
linked memory files. Conventions: NO backticks / no blue text; present-discuss-
THEN-write, one question at a time; read the ACTUAL implementation (never describe
behaviour from run.py labels/docstrings); don't over-generalise.

## 1. Confidence mass-downgrade incident — RESOLVED
- structured_explanations.confidence was bulk-set to ~0.6 between 2026-05-18 and
  05-21 (181,699 HIGH -> ~410). RESTORED 05-29 from the 18 May baseline (181,574
  rows). Pre-restore backup: documents/recovery/2026-05-29-before-confidence-restore.db.
- It was a confidence-only UPDATE (timestamps untouched). Cause NOT definitively
  pinned — likely an ad-hoc bulk reverify during the 05-21 recovery (the reverify
  route rewrites confidence by clue_id, preserving timestamps; protects only
  manual_approve). Reverify worked fine for the user on 05-29, so it's not obviously
  dangerous now. WATCH the DB-wide HIGH count after any bulk operation.
- Memory: incident_confidence_mass_downgrade.md.

## 2. AS-IS pipeline doc — IN PROGRESS
- File: documents/AS_IS_PIPELINE.md. Format: factual summary + DESIGN CONSIDERATION
  (problem + potential solution) per item. Frame findings as divergences to
  harmonise, NOT "bugs".
- DONE: pipeline.py; run.py (shared resources, clue loading/selection); Phase 0
  (hidden), 0b (spoonerism), 0c (double-definition); Phase 0.5 — overview,
  definition stage, all 7 solvers (anagram, container, deletion, charade in 6
  pieces, reversal, acrostic, homophone) and the wordplay-first fallback; a
  "Big picture: cascade and control flow" section; and the General design
  considerations.
- GENERAL design considerations recorded: (a) atomise clue+answer once, reuse
  everywhere (revive WFW version); (b) preserve all letter-contributing evidence
  even on failure; (c) harmonise PARTS subtype classes in the indicator table; (d)
  an operation may only apply to material near its controlling indicator (proximity,
  ALL operators incl container); (e) single-type engines vs compound reality (model
  deeper compound types); (f) cascade order is backwards — best engine (signature)
  runs LAST; run best-first or compete-and-keep-highest-confidence.
- NEXT: write the Phase 1 (signature solver) section, then Phase 1.5/1.5b (blog +
  Haiku), Phase 2 (Tier 2 Sonnet), Phase 3 (enrichment re-solve), post-processing
  (report + gaps -> pending_enrichments). Signature big-picture already understood:
  definition extraction -> grammar triage -> catalog match (base/positional/old) ->
  executor -> confidence; VERIFICATION IS BY RECONSTRUCTION in matcher._verify_combo
  (rebuild answer from candidate pieces per the operation, must equal known answer);
  explanation built by sig_adapter.sig_explain and written by store_signature_result.
- Memory: as_is_pipeline_doc.md.

## 3. Telegraph 31251 clue-by-clue audit — IN PROGRESS
- File: documents/31251_clue_audit.md (CONCLUSION at top). Cold-run puzzle.
- Audited: 8d, 1a, 5a, 10a, 11a, 12a, 13a, 15a, 18a, 20a, 23a, 25a, 26a.
- CONCLUSION (user's, evidence-backed): the engine normally finds the MAJORITY of
  the pieces; it is let down at ASSEMBLY and VERIFICATION. Fixing only those —
  without touching the signature engine — would massively raise the score, and
  several fixes are simple and broad.
- Simple/broad fixes identified:
  - Name the indicator that produced a piece in the explanation (5a reversal "Returned",
    18a alternation "regularly" — both DETECTED but not named -> word_coverage fail).
  - Verifier: normalise spaces in the definition-answer match (12a/25a/26a: DB HAD the
    def, verifier said "not in DB"); combine MULTI-WORD anagram fodder (26a only checks
    first fodder word); treat link words like "and" as links (26a).
  - Clean polluted synonyms_pairs ("first" has 187 junk synonyms incl. first->big;
    junk synonyms cause wrong parses that still VERIFY). 25a PROVED: deleting one junk
    synonym + re-solve self-corrected the parse.
  - Prompt truncation: Tier 2 Sonnet prompt and haiku_definition return single words
    where multi-word is needed (only INDICATORS are told to capture full phrases). Caused
    8d, 11a, and 12a ("Have Spellbound" -> "Spellbound", then orphan rule binned it).
  - Apostrophe tokenisation ("I'm" -> i+m unaccounted, 25a).
- Other confirmed facts: Sonnet STILL ACTIVE (tier2_solver.py + solver.py =
  claude-sonnet-4-6); failed parses stored at 0.0; re-verify != re-solve (enrichment
  change needs a full re-solve); model_version is last-writer only (later phases
  overwrite earlier solves — corrupts attribution).
- NEXT: optionally continue the audit (27a THORNY is the only HIGH/1.0; 28a; the
  down clues), OR start acting on the assembly/verification fixes.
- Memory: findings_prompt_truncation_and_resolve.md.

## Suggested next-thread options
A. Act on the simple, broad assembly/verification fixes (highest ROI per the
   conclusion). NOTE: the verifier is under the standing "don't touch without
   explicit permission" rule — confirm scope first.
B. Continue the AS-IS doc at Phase 1 (signature solver).
C. Continue the 31251 audit.
