# Legacy-First Evidence Recovery Plan - 2026-05-24

## Objective

Restore the system to at least legacy/obase solving strength, while adding the
new design's real purpose: atom-level retention, structured evidence,
verification, honest display, and reviewable enrichment.

Minimum success:

```text
Anything the legacy solver could solve must still solve.
Every solve must retain enough structured evidence to audit and display.
No new Stage/WFW component may suppress a correct legacy solve.
```

## Non-Negotiable Principles

1. Do no harm.
   The working solver path is protected. New stages may observe, record,
   verify, and enrich, but must not replace working mechanisms until parity is
   proven.

2. Stage One always runs first.
   Every clue run begins by atomising and retaining the published clue and
   answer. No solver path is allowed to bypass Stage One.

3. Legacy mechanism knowledge is authoritative initially.
   Obase/legacy mechanisms are the operational solver. The new system wraps
   them and captures what they did.

4. There is one pipeline.
   Puzzle run, clue rerun, admin rerun, and dashboard run all call the same
   pipeline. They differ only in scope.

5. The database stores current truth.
   There must be one current per-clue pipeline state keyed by `clue_id`.
   Append-only artifacts are history, not app truth.

6. Failed clues retain evidence.
   Failure must still record definition candidates, source candidates,
   mechanisms tried, partial assemblies, unresolved words, and enrichment
   candidates.

## Target Pipeline

The recovered pipeline should be:

```text
Stage One atomisation
-> legacy/obase solve using Stage One context
-> structured evidence adapter
-> Stage Two casefile
-> Stage Three verifier
-> current pipeline-state DB write
-> WFW/display adapter
-> optional pending enrichment review
```

Not:

```text
new WFW machinery tries to solve independently
legacy solver runs separately
display chooses whichever stale row looks plausible
```

## Stage One Contract

Stage One must run for every clue and persist:

```text
original clue text
answer text
clue character atoms
answer character atoms
punctuation atoms
space/hyphen atoms
clue tokens
answer tokens
token-to-character links
all candidate spans
definition candidates
wordplay windows
POS spans
DB annotations
normalised text
solver version
timestamp
```

Stage One must not decide final wordplay. It prepares the retained surface map.

For `SLAVISHLY`, Stage One should preserve that the clue contains:

```text
End | to | drinks | in | an | extravagant | way | or | in | a | conformist | one | ?
```

and all token/character positions must remain available downstream.

## Legacy/Obase Solver Contract

The legacy solver remains the solver of record.

It must receive either:

```text
Stage One context directly
```

or, if it still operates on cleaned word lists:

```text
a mapping from every cleaned word/span back to Stage One atoms
```

Its output must no longer be only prose or a flattened `SolveResult`. It must
emit structured solve evidence:

```text
definition span
wordplay span
source spans
mechanism type
indicator/controller spans
produced values
operation inputs
operation outputs
assembly order
answer-letter links
confidence
unresolved clue spans
```

For `SLAVISHLY`, the expected retained evidence is:

```text
definition: "in a conformist way" -> SLAVISHLY
source: "drinks" -> S
mechanism: final letter, controller "End to"
source: "extravagant way" -> LAVISHLY
mechanism: synonym
assembly: S + LAVISHLY = SLAVISHLY
unresolved/surface: "or", "one", "?"
```

If the legacy solver currently solves this but does not expose that structure,
the repair is to instrument/adapt the legacy result, not replace the solver.

## Evidence Adapter

Add a strict adapter layer between the legacy solver and Stage Two.

Its job:

```text
legacy result -> structured retained evidence
```

It must not invent evidence. It may only translate:

```text
what source was used
what mechanism was used
what value was produced
where it maps in the answer
what was unresolved
```

If a legacy result lacks enough detail, the adapter marks the missing fields as
gaps rather than fabricating them.

## Stage Two Contract

Stage Two is an internal casefile, not a solver and not a public explanation.

It consumes:

```text
Stage One context
legacy structured evidence
failed/partial evidence
```

It returns:

```text
definition evidence
source evidence
operation evidence
assembly candidates
answer coverage
word coverage
unresolved words
conditional enrichments
review reasons
```

Stage Two must preserve:

```text
found evidence
missing evidence
conditional evidence
failed mechanism attempts
leftover answer letters
leftover clue spans
```

Stage Two must not:

```text
write directly to reference DB
publish final proof
turn conditional evidence into proof
hide useful partial evidence
rerun a separate solver truth path
```

## Stage Three Contract

Stage Three is a verifier/proof gate.

It consumes Stage Two only. It verifies:

```text
definition defines whole answer
sources are DB-supported or accepted
indicators license mechanisms
operations produce stated outputs
assembly equals answer
answer letters are covered
source spans are valid
word roles are accounted for
unresolved words are honestly classified
```

Possible statuses:

```text
PASS
REVIEW
CONDITIONAL_NEEDS_ENRICHMENT
FAIL_EVIDENCE_ONLY
```

Stage Three must never create a polished proof from failed evidence.

## Persistence Contract

Every run writes one current row per clue, keyed by `clue_id`, containing:

```text
stage_one_context_json
legacy_solve_evidence_json
stage_two_casefile_json
stage_three_proof_json
wfw_display_json
status
confidence
solver_authority
solver_version
updated_at
```

Append-only tables may remain for audit/history, but the app reads current
state first.

Rerun must update the current clue row. It must not rely on "latest append row"
as truth.

## Display Contract

The clue page displays from current verified state.

If Stage Three is `PASS`:

```text
show WFW proof
show answer pieces
show definition
show mechanisms
```

If Stage Three is `REVIEW` or failed:

```text
show candidate evidence as candidate evidence
do not show a green solved-looking definition box
show what was found and what is missing
```

Stale `wfw_proof_attempts` must not override current pipeline state.

## Enrichment Contract

Missing facts become pending enrichments only.

Examples:

```text
definition_gap
synonym_gap
source_phrase_widening
indicator_gap
homophone_gap
```

They must include:

```text
clue_id
source
puzzle_number
clue_text
answer
candidate phrase
candidate value
evidence reason
stage
status=pending
```

No speculative enrichment writes directly to `cryptic_new.db`.

Accepted enrichment flow:

```text
human accepts
reference DB updates
clue reruns through same pipeline
Stage Three re-verifies
display updates
```

## Entry Point Unification

Audit and then enforce one call path for:

```text
dashboard puzzle run
normal puzzle run
clue rerun
admin rerun
batch scripts
```

Each must call:

```text
run_authoritative_clue_pipeline(...)
```

Puzzle-level code loops over clues. Single-clue code passes one clue. No
duplicated solver phases.

## Legacy Parity Harness

Before any further feature work, create a regression harness.

For each clue, record:

```text
legacy solved?
current pipeline solved?
same answer?
same or better confidence?
evidence retained?
Stage Three status?
display honest?
runtime?
```

Initial required clues:

```text
SLAVISHLY - final letter + synonym charade
STIPULATION - anagram
AINTREE - WFW display/review case
TIJUANA - phrase sources
TOTALLY - charade with phrase sources
ROC - deletion/heading off
HASBEEN - compound/conditional enrichment
one hidden clue
one double definition
one homophone
```

No claim of recovery until this set is no worse than legacy.

## Implementation Order

1. Audit all current entry points and solver calls.
2. Identify the strongest legacy/obase solve path.
3. Restore that as the single authoritative operational path.
4. Ensure Stage One runs and persists before every solve.
5. Add/repair legacy structured evidence emission.
6. Build the adapter from legacy evidence to Stage Two.
7. Make Stage Three verify Stage Two, not rediscover the solve.
8. Make display read current pipeline state honestly.
9. Add regression harness.
10. Only then improve parser coverage.

## What Must Stop Immediately

No more new solving architecture until parity is restored.

No more "WFW native assembly" in the live path as a replacement solver.

No more clues marked or displayed as useful based on malformed fragments like:

```text
in a conformist one
```

No more hidden duplicated puzzle/admin/rerun paths.

## Success Definition

We are back on track only when:

```text
legacy-easy clues solve again
Stage One atoms are retained
actual solve evidence is retained
Stage Two shows real evidence
Stage Three verifies or honestly rejects
the clue page does not mislead
rerun and puzzle run use the same code
```
