# Handover: After Instruction 1, Before Instruction 2

Date: 2026-05-26

Project: `C:\Users\shute\PycharmProjects\cryptic_solver_V2`

Purpose: give the next thread a precise, honest state record before starting Instruction 2 of the WFW authoritative evidence stabilization work.

This handover is written under the working principles:

1. Do not claim what is not known.
2. Do not hide uncertainty.
3. Do not imply something was checked when it was not.
4. Do not make the user pay for assumptions.

## Current Objective

We are working through the WFW stabilization work described in:

`documents/WFW_AUTHORITATIVE_EVIDENCE_STABILIZATION_PROPOSAL_2026-05-26.md`

Instruction 1 has now been implemented and verified. The next thread is expected to work on Instruction 2.

Instruction 1 fixed Stage Three word-purpose classification for the ENDEARMENT symptom:

```text
Unusually tender name for sweetheart?
ENDEARMENT
```

Before Instruction 1, a live solve found the anagram assembly, but Stage Three still reviewed the clue because:

```text
Unusually -> operation_indicator_candidate
for       -> unresolved_purpose
```

After Instruction 1:

```text
Unusually -> operation_indicator, verified
for       -> structural_separator, verified
```

The live solve now has `word_purpose_coverage PASS` and `word_purpose_candidates PASS`.

## Critical Warning For The Next Thread

The worktree is dirty and contains many unrelated modified and untracked files.

Do not revert anything.

Do not assume unrelated modified files are Codex changes.

Do not use `git reset --hard`, `git checkout --`, or any destructive cleanup.

Important observed git state after Instruction 1:

```text
signature_solver/stage_three_proof.py is untracked in this checkout.
```

That means normal `git diff` does not show a concise patch for the Stage Three work. The file exists and the code runs, but from Git's point of view it is an untracked file, not a tracked modified file.

This must be handled deliberately before any future commit/review workflow. Do not tell the user that the diff is clean proof of no changes to this file.

## Documents In Play

Main stabilization proposal:

```text
documents/WFW_AUTHORITATIVE_EVIDENCE_STABILIZATION_PROPOSAL_2026-05-26.md
```

Instruction 1, now implemented:

```text
documents/PHASE_2_STAGE_THREE_WORD_PURPOSE_FIXES_CODEX_INSTRUCTION.md
```

Recent supporting instructions already implemented before Instruction 1:

```text
documents/PHASE_2_CANDIDATE_EVIDENCE_PRESERVATION_CODEX_INSTRUCTION.md
documents/PHASE_2_BLOCKS_DEFENSIVE_ACCESS_CODEX_INSTRUCTION.md
documents/PHASE_2_MANUAL_ROLES_PROOF_CODEX_INSTRUCTION.md
```

Continuity/background documents:

```text
documents/HANDOVER_CONTINUITY_2026-05-25_NEW_THREAD.md
documents/HANDOVER_2026-05-26-WFW-EVIDENCE-TRAIL.md
```

## What Instruction 1 Changed

File changed:

```text
signature_solver/stage_three_proof.py
```

No other file was intentionally changed for Instruction 1.

### Change A: Verify The Selected Anagram Indicator

Location:

```text
signature_solver/stage_three_proof.py
_blocks()
around line 977 in the current file
```

The operation-candidates loop now computes whether the selected assembly is an answer-fit anagram:

```python
_anagram_assembly = (
    assembly is not None
    and assembly.get("kind") == "anagram"
    and assembly.get("status") == "answer_fit"
)
```

If so, it computes the selected source span extent from the selected assembly parts:

```python
_selected_source_start = min(...)
_selected_source_end = max(...)
```

Then, for each operation candidate, it verifies only the `ANA_I` candidate whose span touches the selected source boundary:

```python
_op_is_verified_ana_i = (
    _anagram_assembly
    and operation.get("token") == "ANA_I"
    and span is not None
    and len(span) == 2
    and _selected_source_start is not None
    and (
        span[1] == _selected_source_start
        or span[0] == _selected_source_end
    )
)
```

If this is true, the yielded `OP_BLOCK` gets:

```python
"status": "verified"
```

Otherwise, it keeps the previous behavior:

```python
operation.get("span_status") or "candidate"
```

This was deliberately implemented narrowly. It does not verify every `ANA_I` candidate merely because an anagram assembly exists. The `ANA_I` candidate must be adjacent to the selected fodder span.

For ENDEARMENT:

```text
Unusually [0,1]
tender    [1,2]
name      [2,3]
```

The selected fodder span is `[1,3]`, so `Unusually [0,1]` touches the start boundary and is verified.

### Change B: Verify Positional Separator Words

Location:

```text
signature_solver/stage_three_proof.py
_word_purposes()
around line 1179 and line 1273 in the current file
```

Before the word loop, the function now computes:

```python
_source_block_spans = [
    tuple(block["span"])
    for block in blocks
    if block.get("kind") == "SOURCE_BLOCK"
    and block.get("span")
    and len(block.get("span")) == 2
]
_wordplay_end = (
    max(s[1] for s in _source_block_spans)
    if _source_block_spans else None
)
_definition_start = (
    min(s[0] for s in definition_spans) if definition_spans else None
)
```

Just before the unresolved fallback, the function now adds:

```python
if (
    role is None
    and _wordplay_end is not None
    and _definition_start is not None
    and _wordplay_end <= index < _definition_start
):
    role = "structural_separator"
    status = "verified"
```

This is deliberately conservative.

It fires only when `role is None`. It does not override:

```text
verified source words
verified definition words
verified operation indicators
candidate operation evidence
manual roles
REVIEW_BLOCK unresolved_purpose
```

For ENDEARMENT:

```text
tender name = source coverage ending at index 3
sweetheart? = definition starting at index 4
for         = index 3, in the gap
```

So `for` becomes:

```text
structural_separator, verified
```

## Verification Actually Run After Instruction 1

All checks below were run from:

```text
C:\Users\shute\PycharmProjects\cryptic_solver_V2
```

Python used:

```text
C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe
```

### Check 1: Syntax

Command:

```powershell
& 'C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe' -m py_compile signature_solver\stage_three_proof.py
```

Result:

```text
passed
no output
exit 0
```

### Check 2: Stage Three Contract Suite

Command:

```powershell
& 'C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe' signature_solver\test_stage_three_proof.py
```

Output:

```text
RefDB loaded: 5406 indicators, 1570 abbreviations, 1749997 synonyms (397651 from def_answers), 1126 homophones, 431,605 wordlist entries
Stage Three proof contract passed
```

Result:

```text
passed
exit 0
```

### Check 3: Focused ENDEARMENT Live Solve

Script run through stdin, not saved to repo:

```python
import sys
sys.path.insert(0, r"C:\Users\shute\PycharmProjects\cryptic_solver_V2")
from signature_solver.db import RefDB
from signature_solver.solver import solve_clue

db = RefDB()
sr = solve_clue(
    "Unusually tender name for sweetheart?",
    "ENDEARMENT",
    db,
)
if sr.stage_three_proof is None:
    print("stage_three_proof is None")
    raise SystemExit(1)
proof = sr.stage_three_proof.as_dict()
print("WORD_PURPOSES")
for wp in proof["word_purposes"]:
    print(wp["index"], wp["text"], wp["purpose"], wp["status"])
print("CHECKS")
for check in proof["checks"]:
    print(check["name"], check["status"], check.get("detail", ""))
```

Output:

```text
RefDB loaded: 5406 indicators, 1570 abbreviations, 1749997 synonyms (397651 from def_answers), 1126 homophones, 431,605 wordlist entries
WORD_PURPOSES
0 Unusually operation_indicator verified
1 tender answer_source verified
2 name answer_source verified
3 for structural_separator verified
4 sweetheart? definition_phrase_member verified
CHECKS
definition_evidence PASS accepted definition-answer evidence found
answer_assembly PASS TENDERNAME (anagram) = ENDEARMENT
source_evidence PASS all assembly parts are backed by Stage Two source evidence
assembly_order PASS assembly does not require left-to-right charade order
operation_evidence PASS no operation pair required by the selected evidence
operation_attachment PASS no transformed working pair requires attachment verification
mechanism_rules PASS no transformed working pair requires mechanism verification
atomic_coverage PASS all answer letters are placed from preserved evidence
span_integrity PASS definition and source spans are stable and non-overlapping
word_purpose_coverage PASS every clue word has a recorded purpose
word_purpose_candidates PASS no clue word purpose depends on candidate evidence
conditional_facts PASS no conditional facts are required
```

Result:

```text
passed
exit 0
```

## What Instruction 1 Did Not Fix

Instruction 1 did not fix persistence, rerun authority, stale display, or database state.

It only fixed Stage Three proof classification when current Stage Two and Stage Three evidence is already being built in memory.

Specifically, it did not:

```text
create clue_pipeline_state rows
persist stage_two_json
persist stage_three_json
write a new wfw_proof_attempt row
repair admin rerun behavior
change web/routes/admin.py
change display behavior
convert old wfw_unified_proof:v1 rows
make stale proof rows honest in the UI
```

This matters because the original ENDEARMENT user-visible failure involved persistence/current-state failure, not only proof classification.

The live solver can now prove ENDEARMENT correctly, but the admin rerun action still must be checked/fixed so that the proof becomes current stored state.

## Known ENDEARMENT State Before Instruction 1

Earlier inspection found for clue id:

```text
10067996
```

Clue:

```text
Unusually tender name for sweetheart?
```

Answer:

```text
ENDEARMENT
```

Current database state observed before Instruction 1:

```text
clue_pipeline_state: None
stage_one_json: None
stage_two_json: None
stage_three_json: None
```

Known WFW proof rows then:

```text
id 184: wfw_review, source obase_structured, created 2026-05-21 04:40:36
id 169: wfw_proven, source wfw_unified_solver, schema wfw_unified_proof:v1, created 2026-05-21 04:38:23
id 139: wfw_proven, source obase_structured, created 2026-05-21 04:11:49
```

Important caution:

The user reported pressing rerun, but the old rows were still present and no `clue_pipeline_state` row existed. That does not prove exactly which request handler ran or failed. It proves only that the user-visible action did not leave a current authoritative Stage Two/Stage Three WFW state.

Do not overclaim the cause until the route/action is verified.

## What The Stabilization Proposal Says Instruction 2 Should Address

The next work should move from proof classification to current-state authority.

The proposal's required structural work includes:

1. Define a WFW authority invariant.
2. Make rerun persist Stage Two/Three evidence even on REVIEW.
3. Stop destructive rerun from leaving no replacement.
4. Verify anagram indicator purpose from selected assembly. Done by Instruction 1.
5. Resolve positional separator words. Done by Instruction 1.
6. Convert or label legacy proofs.
7. Display candidate evidence honestly.

Since Instruction 1 completed items 4 and 5, Instruction 2 should likely address items 1, 2, and 3:

```text
current-state invariant
rerun persistence
safe replacement/failure row behavior
```

However, do not assume the exact Instruction 2 scope until the actual Instruction 2 document is read.

## Recommended First Action In The New Thread

Do not begin by editing code.

First read:

```text
documents/WFW_AUTHORITATIVE_EVIDENCE_STABILIZATION_PROPOSAL_2026-05-26.md
documents/PHASE_2_STAGE_THREE_WORD_PURPOSE_FIXES_CODEX_INSTRUCTION.md
```

Then inspect the actual admin rerun code, especially:

```text
web/routes/admin.py
_rerun_clue_inner
atomic_reverify_puzzle
_write_manual_role_stage_three_for_clue
any helper that stores clue_pipeline_state or wfw_proof_attempts
```

Then inspect the persistence helpers:

```text
signature_solver/solver.py
signature_solver/stage_three_write_layer.py
signature_solver/stage_context_store.py
signature_solver/atomic_parse_store.py
```

Names above are based on files present in the worktree. The actual call graph must be confirmed from code, not assumed.

## What Instruction 2 Must Prove Before Implementation

The next instruction should answer these questions with code evidence:

1. Which exact admin action does the user's "Rerun" button call?
2. Does that action call `_rerun_clue_inner`, `atomic_reverify_puzzle`, or another function?
3. Does the action build a live `solve_clue` result?
4. Where is Stage Two persisted?
5. Where is Stage Three persisted?
6. Where is the latest WFW proof row written?
7. What happens if the rebuild errors after old rows are deleted?
8. What happens when the solve result is REVIEW rather than proven?
9. What does the UI display when no current Stage Three row exists but old WFW rows exist?
10. How can a test prove that after rerun there is either current evidence or a current explicit failure row?

Do not rely on button names. Two buttons with the same user-facing label must be traced to their actual handler/action.

## Known Risk: Two "Rerun" Semantics

The user has correctly objected to ambiguous "Rerun" behavior.

If the UI has multiple buttons labelled similarly, the implementation must make their behavior consistent in outcome:

```text
same label means same contract
only scope may differ
```

For example:

```text
clue-level rerun
puzzle-level rerun
reverify
atomic reverify
```

These may call different functions, but from the user's perspective they must all leave the same kind of current WFW authority state for their scope.

Instruction 2 should not create another special case. It should define the shared write contract and reuse it.

## What Must Not Be Repeated

Do not frame Instruction 2 as "fix ENDEARMENT".

ENDEARMENT is the proving example, not the special case.

The structural problem is:

```text
live solver can build evidence
but admin action does not leave current authoritative stored WFW state
```

Do not solve that by adding another display overlay.

Do not solve that by reading stale rows and pretending they are current.

Do not solve that by deleting old rows first and hoping replacement succeeds.

Do not solve that by making `word_purposes` more permissive. Instruction 1 already fixed the known Stage Three purpose gap for this case.

## Suggested Shape Of Instruction 2

Instruction 2 should probably be one of these, depending on what route inspection proves:

### Option A: Shared Current-WFW Writer

Create or repair a single helper that takes the latest solve/proof result and writes:

```text
clue_pipeline_state.stage_two_json
clue_pipeline_state.stage_three_json
latest wfw_proof_attempt proof_json
```

It must write REVIEW evidence too, not only PASS/proven evidence.

All admin rerun/reverify actions should call it.

### Option B: Safe Rerun Transaction

Change `_rerun_clue_inner` so it does not delete/commit old proof state before replacement/current failure state is ready.

Acceptable results after rerun:

```text
current PASS proof
current REVIEW proof
current explicit failure row
```

Unacceptable:

```text
no current row
old row silently shown
missing clue_pipeline_state
```

### Option C: Route-Level Invariant Test First

Before changing behavior, add a test that simulates the admin rerun action and asserts:

```text
current stage_two_json exists
current stage_three_json exists
or current explicit failure row exists
```

This is preferred if the code has enough test scaffolding.

## Acceptance Tests For Instruction 2

At minimum, after Instruction 2, the new thread should verify:

### ENDEARMENT Current State

After the actual user-facing rerun action for clue `10067996`:

```text
clue_pipeline_state.stage_two_json is not null
clue_pipeline_state.stage_three_json is not null
latest current proof row is stage_three_proof:v1 or explicit current failure
```

If the live solver succeeds as it did in Instruction 1, expected Stage Three checks include:

```text
answer_assembly PASS
source_evidence PASS
definition_evidence PASS
atomic_coverage PASS
word_purpose_coverage PASS
word_purpose_candidates PASS
```

Expected word purposes:

```text
Unusually operation_indicator verified
tender answer_source verified
name answer_source verified
for structural_separator verified
sweetheart? definition_phrase_member verified
```

### REVIEW Evidence Persists

Pick or construct a clue where Stage Three is REVIEW but evidence exists.

After rerun:

```text
stage_two_json persists
stage_three_json persists
latest WFW row exists
review details are current and precise
```

### Failure Row Exists On Build Failure

Simulate or force a rebuild failure if possible.

After rerun:

```text
latest WFW row exists
status is review/failure
proof_json names the failure
old proof rows are not silently presented as current
```

### Existing Regression Cases

Do not regress:

```text
ENDEARED candidate preservation
UNDEMOCRATIC used source candidates
MAUI manual-role proof behavior
test_stage_three_proof.py
```

## Commands Known To Pass After Instruction 1

These passed immediately after the implementation:

```powershell
& 'C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe' -m py_compile signature_solver\stage_three_proof.py
```

```powershell
& 'C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe' signature_solver\test_stage_three_proof.py
```

Focused live solve:

```powershell
@'
import sys
sys.path.insert(0, r"C:\Users\shute\PycharmProjects\cryptic_solver_V2")
from signature_solver.db import RefDB
from signature_solver.solver import solve_clue

db = RefDB()
sr = solve_clue(
    "Unusually tender name for sweetheart?",
    "ENDEARMENT",
    db,
)
if sr.stage_three_proof is None:
    print("stage_three_proof is None")
    raise SystemExit(1)
proof = sr.stage_three_proof.as_dict()
print("WORD_PURPOSES")
for wp in proof["word_purposes"]:
    print(wp["index"], wp["text"], wp["purpose"], wp["status"])
print("CHECKS")
for check in proof["checks"]:
    print(check["name"], check["status"], check.get("detail", ""))
'@ | & 'C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe' -
```

## Professional Notes For The Next Thread

Be precise with language.

Do not say "rerun failed" unless the request handler has been traced.

Safer phrasing until verified:

```text
The user-visible rerun action did not leave current authoritative WFW state.
The exact handler/cause must be verified from code and, ideally, request logs or a route-level reproduction.
```

Do not say "all pieces are missing"; they are not.

The live solver currently has the key pieces:

```text
source: tender -> TENDER
source: name -> NAME
assembly: TENDERNAME anagram = ENDEARMENT
definition: sweetheart? -> ENDEARMENT
indicator purpose: Unusually verified after Instruction 1
separator purpose: for verified after Instruction 1
```

The remaining major failure is current authoritative storage and display of that evidence after the actual admin action.

## Bottom Line

Instruction 1 is implemented and verified.

The next thread should work on Instruction 2: making rerun/reverify produce a current authoritative WFW evidence state or a current explicit failure state.

Do not edit proof classification again unless a new verified defect is found. The next likely work is in admin route/write-layer/persistence behavior, not Stage Three word-purpose logic.
