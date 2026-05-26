# WFW Authoritative Evidence Stabilization Proposal

Date: 2026-05-26

Purpose: explain the structural WFW failure where all the evidence needed to prove a clue exists somewhere in the system, but rerun/reverify/display do not turn it into one current authoritative proof state.

This document is for Claude review before implementation. It is not a clue-specific hack request.

## Principles

This proposal is written under four rules:

1. Do not claim what is not known.
2. Do not hide uncertainty.
3. Do not imply something was checked when it was not.
4. Do not make the user pay for assumptions.

## Goal

Fix the class of WFW failures where a clue has enough evidence somewhere in the system, but current WFW remains stale, missing, review-only, or visually misleading.

The target is not to solve one clue. The target is to make WFW evidence authoritative and stable for all clues with the same symptoms.

## Example Symptom: ENDEARMENT

Clue id:

`10067996`

Clue:

`Unusually tender name for sweetheart?`

Answer:

`ENDEARMENT`

Expected cryptic structure:

```text
Unusually = anagram indicator
tender name = fodder
TENDERNAME* = ENDEARMENT
sweetheart? = definition
```

This is not a speculative parse. The old WFW proof already represented this structure.

## Observed Current DB State

For clue `10067996`, current database inspection showed:

```text
clue_pipeline_state: None
stage_one_json: None
stage_two_json: None
stage_three_json: None
```

Latest WFW proof attempts:

```text
id 184: wfw_review, source obase_structured, created 2026-05-21 04:40:36
id 169: wfw_proven, source wfw_unified_solver, schema wfw_unified_proof:v1, created 2026-05-21 04:38:23
id 139: wfw_proven, source obase_structured, created 2026-05-21 04:11:49
```

So the latest row is review, but an older row is proven.

That matters because "latest row wins" can hide a valid older proof, while still not giving the user a current Stage Three proof.

## Evidence That Already Exists

### 1. Old WFW Unified Proof

Proof row `169` is a `wfw_unified_proof:v1` row and is `wfw_proven`.

It contains:

```text
source block:
  text: tender name
  value: TENDERNAME
  mechanism: ANA_F

operation:
  anagram
  detail: anagram of TENDERNAME = ENDEARMENT

placements:
  answer letters mapped from TENDERNAME to ENDEARMENT
```

This old proof contains a phrase-level raw fodder block: tender name -> TENDERNAME. Current solve_clue does not reproduce that exact phrase-level block; it produces equivalent split fodder evidence (tender -> TENDER and name -> NAME) and an answer_fit anagram assembly. The old proof is therefore not needed to prove the clue now, but it is useful evidence that an earlier system represented the fodder more naturally as a single phrase.

### 2. Structured Explanation

The clue row has:

```text
definition: sweetheart?
wordplay_type: anagram
ai_explanation: anagram of TENDERNAME [anagram: "Unusually"] = TENDERNAME + ENDEARMENT ("sweetheart?"); definition: "sweetheart?"
```

The `structured_explanations.components` row contains useful evidence:

```json
{
  "ai_pieces": [
    {
      "mechanism": "anagram_fodder",
      "clue_word": "tender name",
      "letters": "TENDERNAME"
    }
  ],
  "assembly": {
    "op": "charade",
    "order": ["ENDEARMENT", "TENDERNAME"]
  },
  "wordplay_types": ["anagram"]
}
```

Important caveat:

The `ai_pieces` evidence is useful. The `assembly` field is not clean; it says `op: charade` and has an odd order. So structured explanation evidence must be used carefully, not blindly.

### 3. Live Solve Result

A direct call to `solve_clue("Unusually tender name for sweetheart?", "ENDEARMENT", RefDB())` shows:

```text
high_confidence: True
solved: True
Stage Two status: answer_fit
```

Stage Two has an answer_fit anagram assembly. Source candidates include:

```text
tender -> TENDER, ANA_F, verified/used
name -> NAME, ANA_F, verified/used
```

Stage Three check results:

```text
answer_assembly: PASS (TENDERNAME anagram = ENDEARMENT)
source_evidence: PASS
atomic_coverage: PASS
definition_evidence: PASS (sweetheart? -> ENDEARMENT)
word_purpose_coverage: REVIEW
word_purpose_candidates: REVIEW
```

The word_purpose failures are:

```text
Unusually: operation_indicator_candidate (not verified)
for: unresolved_purpose
```

So the solver finds and proves the anagram structure. Stage Three is REVIEW not because the assembly is missing but because the anagram indicator word and the separator word do not yet have verified grammatical purposes in the proof.

The central failure is not Stage Two. It is that this solved, answer_fit result is not being persisted to clue_pipeline_state and wfw_proof_attempts for clue 10067996.

## What The Earlier Candidate-Preservation Fix Did

Recent changes to `signature_solver/stage_three_proof.py` did this:

- added top-level `source_candidates` and `operation_candidates` to `StageThreeProof`
- serialized them in `as_dict`
- annotated source candidates as `used` or `candidate`
- improved `_source_check` when candidates exist but no assembly exists
- fixed `_blocks` defensive access for hand-built test fixtures
- fixed candidate operation blocks so they do not falsely verify word purposes

Those changes passed:

```text
signature_solver/test_stage_three_proof.py
focused ENDEARED / UNDEMOCRATIC / MAUI verification
```

But that fix only applies when Stage Two evidence is already present in the current proof-building input.

It does not:

- create missing `clue_pipeline_state`
- rebuild Stage Two on rerun if persistence fails
- convert old unified proof evidence into Stage Three proof
- make preserved top-level `source_candidates` visible as WFW tiles
- verify the selected anagram indicator as a proven operation purpose
- resolve structural separator words like "for" that sit between wordplay and definition

## The Structural Failure

There are four connected failures.

### Failure 1: Rerun Is Not Authoritative

The user reports pressing rerun. The current DB state after that report still contains no clue_pipeline_state row and only old WFW proof rows (ids 184, 169, 139). Note: if _rerun_clue_inner had actually executed the delete-and-commit block for this clue, those old rows should have been deleted. That they remain is evidence that the delete block may not have fired, or that a different code path ran. The exact reason requires route/request verification.

What is certain is that the user-visible rerun action did not result in current authoritative Stage Two/Stage Three state for this clue:

```text
stage_two_json: None
stage_three_json: None
```

Rerun must either produce a current WFW evidence state or explicitly store a current failure state. Silent absence is not acceptable.

### Failure 2: Rerun Deletes Before Replacement Is Guaranteed

In `web/routes/admin.py`, `_rerun_clue_inner` clears previous outputs early:

```text
UPDATE clues SET definition = NULL, wordplay_type = NULL, ai_explanation = NULL, reviewed = NULL
DELETE FROM structured_explanations WHERE clue_id = ?
DELETE FROM wfw_proof_attempts WHERE clue_id = ?
COMMIT
```

Then it attempts to rebuild.

This is dangerous. If rebuild does not produce current Stage Two/Stage Three proof state, the clue can be left with missing or stale WFW authority.

Confirmed from code (admin.py lines 881-894 and 947): the three deletions and db.commit() occur unconditionally before any rebuild attempt. The exception handler at line 947 returns an HTML error div to the caller but writes nothing to the database. So when run_clue_pipeline raises an exception, the clue is left with no current WFW state and no record of why. The phrasing above "may leave missing or stale WFW authority" understates the reality: an exception always leaves no current state, not sometimes.

The safe behavior is:

- build replacement evidence first, or
- perform deletion/replacement transactionally, or
- write an explicit current failure proof row if replacement fails.

Do not leave no current WFW state.

### Failure 3: `_write_manual_role_stage_three_for_clue` Cannot Help Without Stage Two

`_write_manual_role_stage_three_for_clue` reads `clue_pipeline_state.stage_two_json`.

If `stage_two_json` is missing, it returns:

```text
{"status": "missing_stage_two"}
```

So it cannot repair a clue like ENDEARMENT unless Stage Two has first been persisted.

This matters because rerun currently calls the manual-role-aware Stage Three writer after the pipeline attempt. If the pipeline did not persist Stage Two, the manual-role writer has no evidence package to use.

### Failure 4: Persistence Gap And Word Purpose Gaps

The solver does produce an answer_fit result for ENDEARMENT. The Stage Two anagram assembly and Stage Three answer_assembly PASS are real. However, the user-visible rerun action did not result in persisted clue_pipeline_state or a current Stage Three WFW row for clue 10067996. It is not yet proven whether the persistence pipeline ran and failed, whether the rerun action did not reach that code, whether a different action was triggered, or whether an exception occurred before persistence. Route/request verification is required to determine the exact cause.

Two additional Stage Three gaps remain even when the result is correctly persisted:

First: when an anagram assembly is selected, the anagram indicator word (Unusually) is left as operation_indicator_candidate rather than being promoted to a verified operation purpose. Stage Three currently verifies the assembly PASS from the parts, but does not trace back to confirm which clue word licensed the anagram operation.

Second: the separator word "for" (which sits between the anagram fodder and the definition) remains unresolved_purpose. Stage Three has no rule that treats a word between verified wordplay and a verified definition as a structural separator when the mechanism does not need it as a source.

## Required Structural Fix

### 1. Define A WFW Authority Invariant

After every admin rerun or reverify, the clue must have one of these:

```text
current stage_two_json
current stage_three_json
latest stage_three_proof:v1 WFW proof row
```

or:

```text
explicit current WFW failure proof row explaining why current evidence could not be built
```

The system must not leave:

```text
stage_two_json: None
stage_three_json: None
old proof row as apparent truth
```

### 2. Make Rerun Persist Evidence Even On REVIEW

Rerun should persist Stage Two and Stage Three evidence regardless of whether the clue is solved/high-confidence.

This is crucial. WFW review evidence is still evidence.

If the solver cannot prove the clue, the user should still get:

```text
stage_two_json with candidates
stage_three_json with REVIEW checks
latest WFW proof row with source stage_three_pipeline or stage_three_manual_roles
```

### 3. Stop Destructive Rerun From Leaving No Replacement

Change rerun so it does not delete old WFW rows and commit before proving it can write a replacement.

Acceptable approaches:

1. Build new evidence first, then replace old proof rows.
2. Wrap clear/build/write in one transaction.
3. If rebuild fails, write a current `wfw_review` failure row with a precise reason.

The user-visible invariant is:

```text
after rerun, latest WFW state is current
```

not:

```text
after rerun, old state may be gone and new state may be missing
```

### 4. Verify Selected Anagram Indicator As Proven Operation Purpose

When Stage Three selects an answer_fit anagram assembly, it must trace back to which clue word licensed the anagram operation and mark that word as a verified operation indicator.

Currently, an anagram indicator word remains operation_indicator_candidate even when the assembly that it licensed is selected and passes. The presence of a passing anagram assembly is sufficient proof that the licensing indicator did its job. Stage Three must promote the indicator word from candidate to verified in that case.

This is a Stage Three word_purposes rule change, not a Stage Two change.

### 5. Resolve Structural Separator Words Between Verified Wordplay And Definition

When a word sits between a verified wordplay span and a verified definition span, and the mechanism does not consume it as a source or indicator, Stage Three should classify it as a structural separator rather than unresolved_purpose.

For ENDEARMENT, "for" sits at span [3,4] between fodder "tender name" [1,3] and definition "sweetheart?" [4,5]. It is not a source word and not an indicator. It is a grammatical joiner between wordplay and definition.

Stage Three must have an explicit rule for this: if a word is positionally between verified wordplay evidence and verified definition evidence, and no source or indicator claim covers it, classify it as structural_separator.

### 6. Treat Legacy Proofs As Migration/Diagnostic Input, Not Final Authority

There are two proof schemas in play:

```text
wfw_unified_proof:v1
stage_three_proof:v1
```

The old unified proof can be useful as historical context. It does not contain evidence that is missing from current Stage Two — the solver now produces the correct anagram assembly independently.

But the target should be:

```text
current stage_three_proof:v1
```

The system needs a policy:

- old proof can be displayed as legacy/stale if no current proof exists
- old proof can be used as migration evidence if explicitly converted
- old proof must not silently masquerade as current Stage Three proof

### 7. Use Structured Explanation Carefully

Structured explanation can provide useful retained evidence:

```text
anagram_fodder tender name letters TENDERNAME
```

But structured explanation may contain malformed assembly metadata, as in this clue:

```text
assembly op: charade
order: ["ENDEARMENT", "TENDERNAME"]
```

Therefore, use `ai_pieces` and explicit mechanism fields as hints/evidence, but verify mechanically before creating Stage Three proof.

### 8. Render Preserved Candidates Or Explain Why Not

Current `wfw_display_adapter.py` renders `proof["blocks"]`.

It does not render the newly preserved top-level:

```text
proof["source_candidates"]
proof["operation_candidates"]
```

So candidate-preservation can be correct in JSON and still make no visible difference.

Display behavior must be explicit:

- verified proof blocks appear as proof tiles
- candidate source/operation evidence appears as candidate/review evidence
- missing current proof appears as stale/missing state
- visual overlays must not imply proof has passed

## Proposed Implementation Order

### Step 1: Add A Current-WFW-State Invariant Test

Before changing more logic, write a failing test around the invariant:

After rerun/rebuild of a clue, the system must produce either:

- current `stage_two_json`
- current `stage_three_json`
- current `stage_three_proof:v1` WFW row

or an explicit current failure row.

For unit-level testing, this can be a lower-level helper test. For full confidence, add a route-level/admin-action test.

### Step 2: Ensure Rerun Persists Stage Two/Three On Review

Audit `run_clue_pipeline`, `run_signature_clue_pipeline`, `store_signature_evidence`, and `upsert_solve_result_pipeline_state`.

The goal:

```text
every rerun writes Stage Two and Stage Three evidence when those can be built
```

not only solved/high-confidence cases.

If `solve_clue` swallows Stage Two/Three build errors, surface them into a failure proof row or diagnostic field.

### Step 3: Make Rerun Replacement Safe

Fix `_rerun_clue_inner` so deletion of old WFW rows is not committed before replacement/failure proof is ready.

At minimum:

- write new current proof/failure row after rebuild
- then remove or supersede old rows if needed

Better:

- avoid deleting old rows at all
- rely on latest current row with explicit schema/source/status
- mark old proof as legacy/stale in display if current proof exists

### Step 4: Verify Anagram Indicator Purpose From Selected Assembly

In Stage Three, when an answer_fit anagram assembly is selected, find the clue word that licensed the anagram operation and mark it as a verified operation indicator in word_purposes.

The rule: if assembly.kind == "anagram" and assembly.status == "answer_fit", look for a working_pair of kind anagram_pair whose indicator_span matches a known operation candidate. Mark that word verified. If no working_pair links the indicator, look for an operation_candidate with token ANA_I adjacent to the fodder span and mark it verified.

Do not promote an indicator word as verified unless a passing anagram assembly is present. The assembly being selected is the proof.

### Step 5: Resolve Positional Separator Words

In Stage Three _word_purposes, add a rule: if a word is positionally between the last verified wordplay span and the first verified definition span, and no source, indicator, or enrichment claim covers it, classify it as structural_separator with status verified.

This is a narrow positional rule. It must not absorb words that belong to either the wordplay or the definition.

For ENDEARMENT:

```text
wordplay ends at span [1,3] (tender name)
definition starts at span [4,5] (sweetheart?)
for at span [3,4] is the gap word -> structural_separator
```

This rule generalises to any clue where a single uncovered word sits in the gap between a complete wordplay region and the definition.

### Step 6: Convert Or Label Legacy Proofs

Add a compatibility stance:

- If old `wfw_unified_proof:v1` is present and no current Stage Three proof exists, UI should say legacy proof exists but current proof missing/stale.
- Optionally provide a migration helper that converts old unified proof evidence into Stage Two/Stage Three input.
- Do not silently treat old proof as current authority.

### Step 7: Display Candidate Evidence Honestly

Prerequisite: wfw_display_adapter.py is currently untracked in git. It must be
committed before any Codex instruction targeting this file is sent. If it is not
tracked, Codex changes to it cannot be reviewed in a diff and the implementation
cannot be audited. Run git add signature_solver/wfw_display_adapter.py and commit
before proceeding with this step.

Teach `wfw_display_adapter.py` to display top-level preserved candidates as candidate evidence when there is no verified block.

This must be visually distinct from proven blocks.

Example wording:

```text
candidate source: tender name -> TENDERNAME
candidate indicator: Unusually
not yet accepted as complete assembly
```

Do not let candidate tiles look like proof tiles.

## Acceptance Tests

### Test 1: ENDEARMENT Proof After Rerun

After rerun of clue 10067996:

```text
clue_pipeline_state.stage_two_json is not null
clue_pipeline_state.stage_three_json is not null
latest wfw_proof_attempt schema is stage_three_proof:v1 or explicit current failure schema
```

Stage Three checks:

```text
answer_assembly: PASS (anagram of TENDERNAME = ENDEARMENT)
definition_evidence: PASS (sweetheart? -> ENDEARMENT)
source_evidence: PASS
atomic_coverage: PASS
Unusually: verified operation_indicator (not candidate)
for: structural_separator (not unresolved_purpose)
word_purpose_coverage: PASS
```

If word_purpose_coverage remains REVIEW, the reason must be current and precise,
not an absence of stored evidence.

### Test 2: Rerun Persistence

After rerun of clue `10067996`:

```text
clue_pipeline_state.stage_two_json is not null
clue_pipeline_state.stage_three_json is not null
latest wfw_proof_attempt is current
latest proof schema is stage_three_proof:v1 or explicit current failure schema
```

No silent missing state.

### Test 3: Missing Stage Two Failure Row

If Stage Two cannot be built:

```text
latest WFW row is wfw_review
proof_json explains missing_stage_two or stage_two_build_failed
UI displays current failure, not stale proof
```

### Test 4: Candidate Preservation Still Works

ENDEARED `10069320` must still preserve:

```text
expensive -> DEAR [verified] candidate
energy -> EN [failed] candidate
is -> DE [verified] candidate
```

### Test 5: Existing Proven Examples Do Not Regress

UNDEMOCRATIC must still preserve/prove:

```text
courted
man
in charge
Rogue
definition
```

MAUI manual-role case must still pass when:

```text
manual roles present
definition fact exists
Stage Two assembly exists
```

### Test 6: Display Is Honest

For a current REVIEW proof with candidates:

- candidate evidence is visible
- status remains review
- review reason is shown
- proof tiles and candidate tiles are visually/semantically distinct

For a stale/legacy proof:

- display labels it as stale/legacy or missing current proof
- it does not imply current Stage Three proof passed

## Non-Goals

Do not hard-code ENDEARMENT.

Do not make every anagram-looking phrase a proof.

Do not treat old unified proof as current Stage Three proof without conversion.

Do not let display overlays pass proof checks.

Do not delete old proof rows unless replacement/failure state is already safely written.

## Professional Definition Of Done

The work is done only when:

1. rerun always leaves current WFW evidence state or explicit current failure state
2. Stage Two/Stage Three can represent an accepted anagram assembly with attached indicator evidence, whether the fodder is phrase-level or split across adjacent source parts
3. Stage Three can use the resulting answer-fit assembly while also accounting for the licensing indicator and separator words
4. display reads the current proof and distinguishes proven from candidate/stale evidence
5. route-level or stored-row tests prove the behavior, not just helper scripts

For ENDEARMENT specifically, after rerun the acceptable outcomes are:

```text
current stage_three_proof:v1 proven/review row with anagram assembly evidence
```

or:

```text
current explicit failure row explaining why Stage Two/Stage Three could not be built
```

The unacceptable outcome is the current one:

```text
no stage_two_json
no stage_three_json
old proof rows in conflict
no current authoritative WFW state
```
