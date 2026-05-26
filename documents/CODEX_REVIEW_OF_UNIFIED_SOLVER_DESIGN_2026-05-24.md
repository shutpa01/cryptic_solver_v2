# Codex Review of `UNIFIED_SOLVER_DESIGN.md` - 2026-05-24

## Verdict

I agree with the direction of Claude's design.

It is materially better than the implementation path that caused the current
regression, because it restores the correct priority:

```text
legacy solver solves
Stage One preserves atoms/spans
Stage Two reads the actual solver evidence
Stage Three verifies
WFW displays
```

The strongest point in Claude's document is this:

```text
Stage Two should read it out, not redo the work.
```

That is the right correction. The recent failure came from letting WFW/Stage
machinery rediscover the solve instead of retaining and verifying the proven
legacy solve.

## Agreement Points

### 1. The Goal Is Correct

The goal must be:

```text
Improve on the legacy solver, not replace it.
```

This matches the existing design handovers and the WFW preservation notes.

The legacy/obase solver contains hard-won mechanism knowledge. The new system
should preserve and structure that knowledge, not compete with it.

### 2. Stage Two Should Consume `SolveResult`

I agree that the missing core path is:

```text
SolveResult -> structured Stage Two casefile
```

For high-confidence solves, Stage Two should not start from grammar-only
evidence and try to reconstruct the solution. It should translate:

```text
sr.result.word_roles
sr.definition
sr.confidence
build_ai_pieces(sr)
build_assembly_dict(sr)
```

into the Stage Two schema.

This is the shortest route back to legacy parity plus retained evidence.

### 3. Stage Three Should Be Left Mostly Alone Initially

I agree that Stage Three should not be redesigned first.

If Stage Two is empty or malformed, Stage Three can only produce empty or
misleading review output. The first repair is to feed Stage Three the evidence
the legacy solver already has.

### 4. The Core Implementation Scope Is Correct

The core change should be small:

```text
signature_solver/stage_two_casefile.py
signature_solver/solver.py::_attach_gt2_evidence
```

That is the right target for a first recovery step.

## Required Corrections / Clarifications

### 1. Stage One Must Remain Explicitly First-Class

Claude's design includes Stage One in the architecture diagram, but the
implementation section risks underweighting it.

The final plan should explicitly state:

```text
Every run starts by building/preserving Stage One context.
The SolveResult-to-Stage-Two adapter must map every legacy word role back to
Stage One spans and atoms where possible.
```

The evidence adapter must not produce spanless evidence unless mapping genuinely
fails.

For every piece, Stage Two should try to retain:

```text
clue token span
clue atom ids
answer token/letter span
answer atom ids
mechanism token
produced value
```

This is essential. Without it, we merely recreate the old flattened evidence
problem.

### 2. `SolveResult.word_roles` Is Not Quite Enough On Its Own

Claude says:

```text
sr.result.word_roles ... is the complete mechanical evidence.
```

I mostly agree, but it needs qualification.

It is enough to recover many high-confidence solves, but it is not always
complete WFW evidence because it may not preserve:

```text
multi-word source span boundaries
answer-letter placement
operation nesting
surface/link words outside the wordplay window
distinction between source words and controller words when text repeats
```

Therefore the adapter should treat `word_roles` plus `build_assembly_dict(sr)`
plus Stage One context as the evidence source, not `word_roles` alone.

### 3. Option (b) Is Best For `build_ai_pieces` / `build_assembly_dict`

I agree with Claude that option (b) is cleanest:

```text
caller builds pieces and assembly
build_stage_two_from_solve_result receives them as arguments
```

This avoids importing `sonnet_pipeline` from `signature_solver`, which would
blur package boundaries.

Suggested signature:

```python
def build_stage_two_from_solve_result(
    clue_text,
    answer,
    db,
    solve_result,
    *,
    stage_one_context=None,
    pieces=None,
    assembly=None,
):
```

If `pieces` or `assembly` are missing, the function should either:

```text
return evidence_only with a clear gap
```

or the caller should be required to pass them. It should not duplicate
`sig_adapter` logic.

### 4. Span Mapping Needs A Deterministic Rule

Claude identifies span mapping as an open question. This must be treated as a
core requirement, not a detail.

Proposed mapping order:

1. Match `word_roles` entries to Stage One annotations by:

```text
same text
same token
same produced value
unused annotation occurrence
```

2. If no annotation exists, match by tokenised clue text occurrence in wordplay
window order.

3. If still ambiguous, record:

```text
span = None
span_status = "ambiguous"
```

and Stage Three should REVIEW rather than fabricate span certainty.

Repeated words must use occurrence-aware matching.

### 5. The Design Should Explicitly Protect The Legacy Solve Result

Add a hard rule:

```text
If sr.high_confidence is true, Stage Two/Three may downgrade proof status to
REVIEW because evidence is incomplete, but they must not erase or replace the
legacy solve.
```

That avoids the current failure mode where the clue page shows a weak candidate
fragment instead of the real solve/failure state.

### 6. "Pipeline Unchanged" Is Too Strong

Claude says:

```text
The pipeline (run.py) - unchanged
```

As an initial implementation target, fine. But the final plan still needs an
entry-point audit.

The design goal requires one authoritative pipeline for:

```text
dashboard puzzle run
clue rerun
admin rerun
batch scripts
```

If `run.py` already calls the right path, leave it. If any entry point bypasses
the new Stage-Two-from-SolveResult branch, it must be corrected.

So I would phrase this as:

```text
Do not touch run.py for the first implementation unless audit shows it bypasses
the authoritative path.
```

### 7. `_conditional_suburb_enrichments` Should Not Be Removed In The First Patch

Claude asks whether it is safe to remove the hard-coded suburb prototype.

My answer: not in the core recovery patch.

It is ugly and should be replaced, but removing it now creates unnecessary
behaviour churn. The immediate recovery target is:

```text
high-confidence legacy solves produce useful Stage Two/Three/WFW evidence
```

After that is verified, remove or replace `_conditional_suburb_enrichments`
with a general enrichment rule.

So:

```text
Do not remove it in Phase 1.
Mark it as technical debt for Phase 4.
```

### 8. Current Temporary Solver Changes Need Review/Reversal

During emergency debugging, `solve_clue()` was changed to:

```text
disable WFW assembly in the legacy path
avoid recursive fallback
cap speculative GT2/indicator retries
skip token_parse assembly for unsolved results
```

Those changes helped performance, but they are not the final design.

The unified plan should include an explicit cleanup step:

```text
review recent emergency changes and keep only those consistent with the
legacy-first design
```

In particular, if the new Stage-Two-from-SolveResult path makes WFW native
assembly unnecessary in the live legacy path, `assemble=False` may be correct.
But it must be recorded as a deliberate design choice, not an accidental
performance patch.

## Proposed Reconciled Plan

### Phase 0: Baseline Audit

Before coding further:

```text
identify current authoritative solve entry point
confirm where solve_clue is called from dashboard/rerun/admin
confirm whether high-confidence SolveResult still exists before Stage Two
confirm where build_ai_pieces and build_assembly_dict are currently called
confirm current DB write path for clue_pipeline_state and wfw_proof_attempts
```

### Phase 1: Build Stage Two From High-Confidence SolveResult

Implement:

```python
build_stage_two_from_solve_result(...)
```

Inputs:

```text
clue_text
answer
db
solve_result
stage_one_context
pieces
assembly
```

Output:

```text
StageTwoCaseFile
```

Rules:

```text
use sr.definition for definition candidate
use pieces for source/operation candidates
use assembly for answer assembly
map every item to Stage One spans where possible
preserve unresolved words
mark ambiguous spans honestly
do not invent enrichment facts
```

### Phase 2: Branch In `_attach_gt2_evidence`

In `solver.py::_attach_gt2_evidence`:

```text
if sr.high_confidence and sr.result:
    build pieces/assembly
    build Stage Two from SolveResult
else:
    use existing grammar-only Stage Two fallback
```

Then:

```text
sr.stage_two_casefile = stage_two
sr.stage_three_proof = build_stage_three_proof(stage_two)
```

### Phase 3: Regression Harness

Run the fixed set:

```text
SLAVISHLY
STIPULATION
AINTREE
TIJUANA
TOTALLY
ROC
HASBEEN
hidden clue
double definition
homophone
```

For each record:

```text
legacy high confidence?
Stage Two from SolveResult used?
Stage Three status?
WFW proof/review row written?
runtime?
display honest?
```

### Phase 4: Generalise Enrichment

Only after Phase 1-3 pass:

```text
replace hard-coded enrichment prototypes
generalise unresolved grammar-span enrichments
review pending_enrichment flow
```

## Final Position

Claude's document is the right practical recovery direction.

My requested changes are:

1. Make Stage One atom/span mapping explicit and mandatory.
2. Treat `word_roles` as necessary but not sufficient without Stage One mapping.
3. Use option (b): caller passes pieces and assembly.
4. Do not remove `_conditional_suburb_enrichments` in the first patch.
5. Do not promise pipeline files are unchanged until entry points are audited.
6. Add an explicit cleanup review of emergency performance edits.

With those changes, I would accept Claude's design as the basis for the final
implementation plan.
