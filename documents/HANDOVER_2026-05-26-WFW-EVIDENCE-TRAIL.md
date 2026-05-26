# Handover — WFW Evidence Trail Work
# Date: 2026-05-26

## Process rules for this work (must be honoured in the next thread)

1. Never spawn a Codex agent without the user explicitly saying "send to Codex."
2. No backticks in chat responses (plain text only).
3. Do not write code — Codex writes code, Claude plans, designs, checks, audits.
4. A question is not an instruction to touch files.
5. Verify before claiming — read actual files before making factual assertions.


---

## Governing documents

Two documents drive all WFW work. Read both before proceeding.

documents/WFW_EVIDENCE_TRAIL_AS_IS_AS_SHOULD_BE_2026-05-25.md
  The authoritative description of the current evidence trail, every failure
  point, and the eight-step implementation plan. This supersedes all earlier
  narrower WFW documents for planning purposes.

documents/WFW_MANUAL_ROLES_PROOF_INTEGRATION_PLAN_2026-05-25.md
  An earlier plan covering 12 issues and 6 phases. It is still accurate but
  has been superseded by the evidence trail document for sequencing.


---

## What is already implemented (do not re-implement)

All of the following are confirmed in the codebase as of this session.

signature_solver/stage_three_proof.py:
  _manual_roles_by_index helper — maps word index to manual role entry
  _purpose_for_manual_role helper — maps role string to Stage Three purpose
  _word_purposes — accepts manual_roles parameter, applies as final fallback
  _definition_check — accepts extra_candidates parameter
    detects manual_definition_gap and emits precise gap message
  _accepted_definition_candidate — accepts "manual_definition" boundary_status
  _purpose_requests — skips status "manual" alongside "verified"

web/routes/admin.py:
  _build_manual_definition_candidates — groups consecutive definition roles,
    checks RefDB, returns manual_definition or manual_definition_gap candidates
  _manual_roles_for_clue — queries clue_word_roles for a clue
  _casefile_from_stage_two_json — builds SimpleNamespace from stored stage_two_json
  _write_manual_role_stage_three_for_clue — the main Stage Three writer;
    reads stage_two_json, attaches manual roles, calls build_stage_three_proof,
    writes to wfw_proof_attempts and clue_pipeline_state.stage_three_json;
    called from three places:
      reverify_clue (line 780) — runs after ExplanationVerifier, alongside it
      _rerun_clue_inner (line 963) — runs after signature pipeline result
      atomic_reverify_puzzle (line 1971) — loops over all clues in puzzle

signature_solver/wfw_display_adapter.py:
  Bug 1 fix — _stage_three_missing_word_blocks is now called unconditionally
    (the incorrect "if not word_purpose_by_index:" guard has been removed)
  Bug 2 fix — word_seq mapping corrects for punctuation index drift
  _dedupe_stage_three_display_blocks — dedup on span with priority system


---

## What is NOT done — the evidence trail 8-step plan

Step 1 (extend proof schema) — PENDING
Step 2 (preserve candidate source evidence) — PENDING
Step 3 (preserve failed assembly attempts) — PENDING, out of scope for next step
Step 4 (make checks reference evidence) — PARTIALLY PENDING (source_check improvement is in the next instruction)
Step 5 (manual roles into proof input) — DONE (see above)
Step 6 (one WFW writer) — NOT DONE
Step 7 (update display adapter) — NOT DONE (the current display adapter
  renders proof blocks; candidate blocks are a follow-up instruction)
Step 8 (route-level acceptance tests) — NOT DONE

The ENDEARED clue (10069320) is the canonical test for Steps 1-2-4.
Its stored stage_two_json contains these source candidates (verified from DB):
  expensive -> DEAR, SYN_F, span [3,4], evidence_status verified
  energy -> EN, POS_F, span [4,5], evidence_status failed
  is -> DE, SYN_F, span [5,6], evidence_status verified
Those three candidates currently vanish from the Stage Three proof because no
answer_fit assembly was selected. Steps 1-2 will preserve them.


---

## Instruction ready to send to Codex

documents/PHASE_2_CANDIDATE_EVIDENCE_PRESERVATION_CODEX_INSTRUCTION.md

This instruction implements Steps 1, 2, and the source_evidence message part
of Step 4. One file changes: signature_solver/stage_three_proof.py.

Summary of changes:
  1. StageThreeProof dataclass — two new fields with empty-tuple defaults at end:
       source_candidates: tuple[dict, ...] = ()
       operation_candidates: tuple[dict, ...] = ()
  2. as_dict — both new fields added to the returned dict
  3. build_stage_three_proof — populates source_candidates with assembly_status
     annotation ("used" or "candidate") using span + cleaned value + token matching;
     defensive getattr on casefile.source_candidates and casefile.operation_candidates
  4. _source_check — new branch when no assembly: if source_candidates exist,
     produces "source candidates exist but were not accepted as a complete assembly:
     expensive -> DEAR [verified]; energy -> EN [failed]; is -> DE [verified]"

Key design decisions confirmed before sending:
  a. Schema stays at stage_three_proof:v1 — new fields are absent in old rows,
     consumers must treat absent/null as empty list.
  b. assembly_status "used" matching rule: span + cleaned value required;
     if BOTH candidate and assembly part carry non-null token, token must also
     match. Assembly parts do carry token in current stored rows (confirmed by
     _blocks at line 882 which reads part.get("token")).
  c. Defensive getattr only protects the new preservation block; _blocks still
     directly accesses operation_candidates (line ~916) and other casefile
     fields. That pre-existing gap is out of scope.
  d. This instruction does NOT change wfw_display_adapter.py — candidate
     evidence will be in the proof JSON but will not yet appear as display tiles.

Verification target (three clues, stored stage_two_json):
  ENDEARED: three candidates preserved, all assembly_status "candidate",
    source_evidence detail names them with evidence_status labels
  UNDEMOCRATIC: courted, man, in charge have assembly_status "used"
  MAUI: answer_assembly and word_purpose_coverage still PASS, no regression


---

## Instructions written but not yet sent

The following documents exist but are superseded or pending sequencing
decisions. They are accurate as technical descriptions but should not be
sent without review:

documents/PHASE_2_MISSING_WORDS_CODEX_INSTRUCTION.md
  Both missing-words bugs are ALREADY FIXED in the codebase. This document
  is now a record only. Do not send.

documents/PHASE_2_MANUAL_ROLES_PROOF_CODEX_INSTRUCTION.md
  Also already implemented. Record only. Do not send.

documents/PHASE_2_DISPLAY_DEDUP_CODEX_INSTRUCTION.md
  Also already implemented. Record only. Do not send.


---

## Known facts about stored data

ENDEARED (10069320):
  stage_two_json source_candidates: expensive->DEAR, energy->EN, is->DE
  stage_two_json assembly: status "evidence_only", kind "positional"
    (this is from an older solver version — not "answer_fit", so
    _best_answer_fit_assembly returns None; all source candidates will be "candidate")
  stored proof source: "stage_three_manual_roles"
  manual_roles in clue_word_roles: expensive->answer_source, energy->answer_source,
    is->structural_separator (from evidence trail document)

UNDEMOCRATIC (10069315):
  stage_two_json source_candidates: courted->COURTED (ANA_F), man->MAN (ANA_F),
    in charge->IC (ABR_F)
  assembly: answer_fit charade
  stored proof: wfw_proven, source "stage_three_manual_roles"

MAUI (10069319):
  stage_two_json source_candidates: Graduate->MA, uniform->U, island,->I
  assembly: answer_fit charade
  stored proof: wfw_proven, source "stage_three_manual_roles"
  manual_roles: with=link, by=link, one/in/Hawaii=definition

LEGSPIN (10069316): stored wfw_review, real solver/evidence gap, not display failure
NASCENT (10069317): stored wfw_review, definition found, assembly missing,
  non-definition words unresolved


---

## Key architectural facts

stage_two_json in clue_pipeline_state is the StageTwoCaseFile.as_dict() output,
  written via upsert_pipeline_state → _dump(row["stage_two_casefile"]).
  It DOES contain source_candidates and operation_candidates.

_casefile_from_stage_two_json reads: clue_text, answer, definition_candidates,
  grammar_phrases, source_candidates, operation_candidates, working_pairs,
  assemblies, enrichment_candidates, unresolved_words, status.

_best_answer_fit_assembly selects only assemblies with status "answer_fit".
  The ENDEARED stored assembly has status "evidence_only" and will not be selected.

_blocks directly accesses casefile.operation_candidates (line ~916),
  definition_candidates, working_pairs, and unresolved_words without getattr.
  Existing test_stage_three_proof.py failure on hand-built casefiles with
  missing operation_candidates is pre-existing and pre-dates this work.

wfw_display_adapter.py is untracked in git. It must be committed/tracked
  before any further Codex display adapter changes so changes appear in review.


---

## Follow-up instructions needed (not yet written)

Step 7 partial: display adapter to render proof["source_candidates"] as
  candidate display blocks. Candidate blocks are not yet rendered.

Step 3: record why _first_charade failed (stage_two_casefile.py change —
  separate file, separate instruction). Requires _first_charade to return
  a rejection reason rather than None.

Step 6: one WFW writer — all of Re-run, clue Re-verify, puzzle Re-verify,
  and batch generation must call the same proof-generation function. The
  clue Re-verify route (admin.py:757) currently uses ExplanationVerifier,
  not Stage Three at all. This is the hardest step.

Step 8: route-level acceptance tests.

Phase 0 (from the manual roles plan): admin staleness warning when
  clue_word_roles has rows newer than latest wfw_proof_attempts created_at.
  Still not written.


---

## Grammar_triage.py IC bug

An IC (in charge) indicator classification bug in grammar_triage.py was
mentioned in the previous session handover but was never diagnosed in this
session. The user did not provide the clue text or word_roles output needed
to diagnose it. Ask the user if they still want this investigated.


---

## Git state

No commits were made in this session. All changes are in working tree
from previous sessions. Last commit: 3cf53890 "Add answer-constrained
span value recovery."
