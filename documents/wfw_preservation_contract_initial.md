# WFW Preservation Contract - Initial Pass

Generated: 2026-05-16

## Why This Exists

The current solver can preserve a final clue explanation, but it does not reliably preserve the word-by-word proof needed for a user-facing cryptic explanation. The durable unit must be every word's role in the clue, tied to answer spans and nested wordplay operations.

The central mistake in the current version is preserving final snapshots instead of preserving evidence. A clue can be answer-solved, partly structurally solved, or fully word-by-word solved. Those are different states and must not overwrite each other.

## Exemplar: GOLDEN RETRIEVER

Clue: `Go back upset across island after information on adopting elderly dog (6,9)`

Answer: `GOLDEN RETRIEVER`

Definition: `dog`

### User-Facing Word-by-Word Explanation

| Clue word(s) | Role | Contribution | User explanation |
|---|---|---:|---|
| Go back | fodder / synonym | REVERT | "Go back" can mean "revert". |
| upset | anagram indicator | - | Tells us to rearrange the letters for RETRIEVER. |
| across | structural link / scope word | - | Connects the anagram instruction across the following fodder. |
| island | abbreviation fodder | I | I is a standard abbreviation for island. |
| after | ordering indicator | - | The RETRIEVER part comes after GOLDEN. |
| information | synonym fodder | GEN | GEN means information. |
| on | synonym/abbreviation fodder | RE | RE means on/concerning/about. |
| adopting | container indicator | - | GEN adopts OLD. |
| elderly | synonym fodder | OLD | OLD means elderly. |
| dog | definition | GOLDEN RETRIEVER | A golden retriever is a dog. |

### Answer-Span Proof

`GEN` around `OLD` = `GOLDEN`

Anagram of `REVERT + I + RE` = `RETRIEVER`

`GOLDEN + RETRIEVER` = `GOLDEN RETRIEVER`

## Preservation Requirements Learned

1. Preserve answer segmentation, not only whole-answer type.
2. Preserve nested assembly trees, not only `assembly.op`.
3. Preserve every clue token with a user-facing role.
4. Preserve non-letter structural words such as indicators, ordering words, scope/link words, and DBE markers.
5. Preserve unresolved answer spans explicitly; do not collapse them into absent pieces.
6. Preserve candidate residual diagnostics: unaccounted clue words, residual answer letters, possible indicators, and possible abbreviations.
7. Preserve both the machine proof and the user-facing explanation text.
8. Preserve manual corrections as append-only decisions, not destructive replacement.

## Required Durable Record Shape

`solve_runs`
- One row per pipeline/admin/manual run.
- Must preserve source, puzzle, trigger, command/mode, code version if available, started_at, completed_at, and status.

`solve_attempts`
- One row per clue per stage attempt.
- Must preserve run_id, clue_id, stage, model/source, status, confidence, verdict, raw explanation text, normalized explanation text, and whether it is answer-solved, structurally-solved, and WFW-solved.

`solve_attempt_answer_segments`
- One row per answer segment, for examples like `GOLDEN` and `RETRIEVER`.
- Must preserve segment_text, answer_start, answer_end, status, and segment-level assembly.

`solve_attempt_pieces`
- One row per letter-contributing piece.
- Must preserve clue phrase, clue word span, mechanism, letters, source table/lookup if any, indicator phrase, answer segment, and verification status.

`solve_attempt_word_roles`
- One row per clue token per attempt.
- Must preserve word index, word text, role, role source, letters if any, piece id if any, and user-facing explanation.

`solve_attempt_assemblies`
- One row or JSON tree per nested operation.
- Must preserve operation, operands, result letters, indicator, scope, and parent operation.

`solve_attempt_checks`
- One row per verifier check.
- Must preserve check name, status, detail, score contribution/penalty if available, and the related piece/role/segment.

`solve_attempt_gaps`
- One row per proposed gap.
- Must preserve gap type, word, letters, table target, originating check, originating attempt, clue context, and later decision state.

`manual_parse_decisions`
- One row per human/admin correction.
- Must preserve old value, new value, reason, affected attempt/piece/role, created_at, and whether it protects future snapshot overwrite.

`enrichment_decisions`
- One row per accepted/rejected enrichment.
- Must preserve source attempt/gap, target reference table, inserted row if accepted, rejection reason if rejected, and created_at.

## Product Snapshot Rule

`structured_explanations` can remain the current display snapshot, but it must not be the evidence store.

Writers may update the snapshot only after appending the full attempt. A poorer or flatter snapshot may never erase a richer attempt. It may only become a later attempt with lower structural completeness.

## Corpus Work Needed

The next research task is to run this WFW exercise across a large corpus:

1. Generate automatic token-role drafts from existing structured corpora.
2. Sample high, medium, low, and failed cases.
3. For each clue, require every clue word to receive a role.
4. Record what the current data shape cannot express.
5. Update the preservation schema from those failures before changing solver intelligence.

The target is not to make the solver clever first. The target is to learn, from many clues, what evidence has to survive.
