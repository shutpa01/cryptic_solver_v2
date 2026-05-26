# Unified Solver Design

Date: 2026-05-24
Author: Claude Code (for review by Codex)

This document describes the intended architecture for the unified solver.
Codex should read this, critique it, and add his own assessment.
The two reviews will be reconciled into a final implementation plan.

---

## Goal

Improve on the legacy solver, not replace it.

The legacy signature solver works. It correctly solves the majority of clues,
identifies every word's role, and produces a high-confidence result for roughly
60-70% of clues in a puzzle. That is the foundation.

The new design adds three things around it:

1. A Stage Two evidence package — honest reporting of what was found and what
   was not found.
2. A Stage Three verifier — confirms the solve is complete and every word has a
   role.
3. A WFW display — shows the user a word-for-word account of how the clue works,
   derived from the solver's own evidence.

The unified system must be at least as good as the legacy solver on every metric
that existed before. Any change that makes results worse is wrong by definition.

---

## What the Legacy Solver Produces

The core output is a `SolveResult`. When `sr.high_confidence` is True:

- `sr.result.word_roles` — list of `(word, token, value)` tuples.
  Every wordplay word has a token (SYN_F, ABR_F, ANA_I, CON_I, REV_I, LNK, etc.)
  and the letters it contributes. This is the complete mechanical evidence.
- `sr.definition` — the definition phrase (e.g. "group of words").
- `sr.confidence` — 0-100 score.
- `sr.analyses` — per-word analyses used for API evidence.

`build_ai_pieces(sr)` already translates `word_roles` into a flat piece list.
`build_assembly_dict(sr)` already translates `word_roles` into an operation dict
(charade, container, reversal, anagram, etc.).

This means the legacy solver already has everything Stage Two needs.
Stage Two should read it out, not redo the work.

---

## Architecture

```
clue + answer
      |
      v
[Stage One]  clue_context.py
  - tokenise clue and answer
  - POS tagging and grammar phrases
  - definition boundary detection using DB
  - output: stage_one_context (definition candidates, word spans, annotations)
      |
      v
[Legacy Solver]  solver.py : solve_clue()
  - fast, mechanical, zero API cost
  - uses stage_one_context if available (definition candidates, word annotations)
  - catalog matching, synonym/abbreviation lookup, verification
  - output: SolveResult (word_roles, definition, confidence, assembly)
      |
      v
[Stage Two]  stage_two_casefile.py : build_stage_two_from_solve_result()  [NEW PATH]
  - translates SolveResult into evidence package
  - sources: from sr.result.word_roles (SYN_F, ABR_F pieces)
  - operations: from sr.result.word_roles (ANA_I, CON_I, REV_I, etc.)
  - assemblies: from build_assembly_dict(sr) — already works for all op types
  - definition: from sr.definition
  - unresolved_words: clue words not in word_roles and not link words
  - enrichment_candidates: from unresolved words + grammar phrases
  - status: "answer_fit" if high_confidence, "evidence_only" if partial
  |
  | [fallback path when solver has no high-confidence result]
  |
  v
[Stage Two fallback]  stage_two_casefile.py : build_stage_two_casefile()  [EXISTING]
  - grammar-only evidence gathering for partial/failed solves
  - currently only handles anagram + trim_first — needs extension but that
    is lower priority; high-confidence solves cover the majority of clues
      |
      v
[Stage Three]  stage_three_proof.py : build_stage_three_proof()  [EXISTING, UNCHANGED]
  - consumes StageTwoCaseFile
  - verifies answer construction
  - PASS: all words accounted for, assembly verified, definition matched
  - REVIEW: unresolved words remain, or definition not in DB, or enrichment needed
  - never writes to DB, never promotes conditional evidence to proven
      |
      v
[WFW Display]  wfw_display_adapter.py : display_from_stage_three_proof()
  - renders Stage Three evidence to the clue page
  - PASS result → show proven word-for-word breakdown
  - REVIEW result → show what was found + what needs review
  - never shows stale old proof_attempts over fresh evidence
      |
      v
[wfw_proof_attempts]  wfw_proof_store.py
  - written by sig_adapter.store_signature_evidence()
  - status = "wfw_proven" if stage_three.status == PASS
  - status = "wfw_review" otherwise
```

---

## The Key New Function

`build_stage_two_from_solve_result(sr, clue_text, answer_clean, stage_one_context)`

Lives in `stage_two_casefile.py` alongside the existing builder.

Logic:

```
if sr is None or not sr.result:
    return build_stage_two_casefile(clue_text, answer_clean, db, stage_one_context)

pieces = build_ai_pieces(sr)          # already exists in sig_adapter
assembly = build_assembly_dict(sr)    # already exists in sig_adapter

source_candidates = [
    {"text": p["clue_word"], "value": p["letters"], "token": ..., "span": ...}
    for p in pieces if p["mechanism"] not in INDICATOR_MECHANISMS
]

operation_candidates = [
    {"text": p["clue_word"], "token": ..., "span": ...}
    for p in pieces if p["mechanism"] in INDICATOR_MECHANISMS
]

assemblies = [assembly_to_stage_two_format(assembly, answer_clean)]

definition_candidates = [{"text": sr.definition, ...}] if sr.definition else []

unresolved_words = words_in_clue_not_in_word_roles(clue_text, sr.result.word_roles)

enrichment_candidates = grammar_span_enrichments_for_unresolved(
    stage_one_context, unresolved_words, source_candidates)

return StageTwoCaseFile(
    ...,
    assemblies=assemblies,
    status="answer_fit" if sr.high_confidence else "evidence_only"
)
```

The important point: `build_assembly_dict` already handles every operation type
the legacy solver supports — charade, container, reversal, anagram, deletion,
positional, and combinations. By reading from it, Stage Two gets correct
assemblies for all these types immediately. No new assembly logic is needed.

---

## What Stage Three Gets

When fed a StageTwoCaseFile built from a high-confidence SolveResult:

- `definition_candidates` is populated → `definition_evidence` check PASSES
- `assemblies` is populated and matches the answer → `answer_assembly` PASSES
- `source_candidates` are populated → `source_evidence` PASSES
- `assembly_order` follows from the operation → PASSES
- `operation_evidence` is populated → PASSES
- `atomic_coverage` is covered by the assembly → PASSES

The only checks that may still REVIEW are:
- `word_purpose_coverage` — if surface words like "taken" have no role in the clue
- `word_purpose_candidates` — if those surface words have no DB evidence either

This is correct behaviour. A clue where one surface word is genuinely unaccounted
for should be REVIEW, not PASS. Stage Three is being honest.

---

## What This Does Not Change

- `solve_clue` signature and return type — unchanged
- Stage Three code — unchanged
- `wfw_proof_store` — unchanged
- The pipeline (run.py) — unchanged
- The clue page template — unchanged

The only new code is `build_stage_two_from_solve_result` in stage_two_casefile.py
and the branch in `_attach_gt2_evidence` in solver.py that calls it when
`sr.high_confidence` is True.

---

## What This Should Achieve

- High-confidence solves (60-70% of clues) should produce wfw_proven or at worst
  wfw_review-with-useful-evidence, not wfw_review-with-empty-assemblies.
- Stage Three PASS rate should reach the levels the old WFW format achieved
  (37/75 for DT 31243, 12/14 for DM 17883) or better.
- The clue page WFW display should show the actual solve evidence, not a
  placeholder or a stale old proof.
- Partial solves and failed solves still get the grammar-based Stage Two fallback,
  which shows what partial evidence was found.

---

## Outstanding Questions for Codex

1. `build_ai_pieces` and `build_assembly_dict` live in `sig_adapter.py` which is
   in the `sonnet_pipeline` package. `stage_two_casefile.py` is in
   `signature_solver`. The new function needs access to them. Options:
   a. Move `build_ai_pieces` and `build_assembly_dict` to `sig_adapter_utils.py`
      in `signature_solver`, import from both places.
   b. Pass the already-built pieces and assembly as arguments to
      `build_stage_two_from_solve_result` — caller builds them, function just
      translates.
   c. Duplicate the logic (bad).
   Option (b) is cleanest — no new imports, no moving code.

2. The span mapping. `word_roles` has `(word, token, value)` but not a span
   (start, end index in the clue). Stage Two needs spans for enrichment logic.
   `stage_one_context.annotations` has spans. The mapping from word to span
   should use the stage_one_context annotations, falling back to word-order
   index if a word is not found.

3. `_conditional_suburb_enrichments` in stage_two_casefile.py is a hard-coded
   prototype for one specific clue. It should be removed and replaced by the
   general grammar-span enrichment logic when unresolved words are present.
   This is Phase 4 in the handover. Codex should confirm this is safe to remove.

---

## Files to Change

```
signature_solver/stage_two_casefile.py   add build_stage_two_from_solve_result()
signature_solver/solver.py               branch in _attach_gt2_evidence()
```

No other files need changes to implement the core fix.

---

## Files NOT to Touch

```
signature_solver/stage_three_proof.py    unchanged
signature_solver/solver.py solve_clue()  logic unchanged, only _attach_gt2_evidence
sonnet_pipeline/run.py                   unchanged
web/templates/clue.html                  unchanged
web/routes/clue.py                       unchanged
```
