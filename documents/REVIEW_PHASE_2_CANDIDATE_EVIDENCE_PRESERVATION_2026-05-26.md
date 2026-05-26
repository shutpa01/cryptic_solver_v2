# Review: PHASE_2_CANDIDATE_EVIDENCE_PRESERVATION_CODEX_INSTRUCTION.md

Date: 2026-05-26

Reviewed according to these principles:

1. Do not claim what I do not know.
2. Do not hide uncertainty.
3. Do not imply I checked something when I did not.
4. Do not make the user pay for my assumptions.

## Verdict

Do not send `PHASE_2_CANDIDATE_EVIDENCE_PRESERVATION_CODEX_INSTRUCTION.md` as-is.

The instruction is directionally useful, but it contains material errors that would cause another false implementation/verification result.

The main problem is that its ENDEARED verification expectation does not match the current stored `stage_two_json` that the instruction itself says to load.

The second major problem is that the proposed `"used"` / `"candidate"` annotation uses span-only matching, which can mark the wrong candidate as used.

The third problem is that the proposed source-evidence message can list failed source candidates without saying they are failed, which would create a new misleading WFW message.

## What I Checked

I checked the instruction document at:

`documents/PHASE_2_CANDIDATE_EVIDENCE_PRESERVATION_CODEX_INSTRUCTION.md`

I checked the current code locations in:

`signature_solver/stage_three_proof.py`

Relevant observed locations:

- `StageThreeCheck`
- `StageThreeProof`
- `StageThreeProof.as_dict`
- `build_stage_three_proof`
- `_source_check`
- `_blocks`

I checked the current Stage Two shape in:

`signature_solver/stage_two_casefile.py`

Relevant observed locations:

- `StageTwoCaseFile`
- `StageTwoCaseFile.as_dict`
- `_source_candidates`
- `_operation_candidates`
- `build_stage_two_from_solve_result`

I inspected current stored `stage_two_json` in:

`data/clues_master.db`

for these clue ids:

- `10069320` ENDEARED
- `10069315` UNDEMOCRATIC
- `10069319` MAUI

I also searched current tests and consumers for:

- `source_candidates`
- `operation_candidates`
- `display_from_stage_three_proof`
- `stage_three_json`
- `wfw_proof_attempts`

## Observed In Code

The proposed one-file implementation target is structurally plausible.

`signature_solver/stage_three_proof.py` currently contains the relevant objects and functions:

- `StageThreeProof`
- `StageThreeProof.as_dict`
- `build_stage_three_proof`
- `_source_check`

`StageThreeCheck` currently has this shape:

```python
@dataclass(frozen=True)
class StageThreeCheck:
    name: str
    status: str
    detail: str
    evidence: object | None = None
```

Therefore, the instruction's proposed fourth argument to `StageThreeCheck(...)` is syntactically compatible.

`StageThreeProof` currently does not contain top-level `source_candidates` or `operation_candidates`.

Adding those fields with empty-tuple defaults at the end of the dataclass is compatible with Python dataclass rules.

Adding those fields to `as_dict` is also structurally compatible.

## Observed In Stored Data

The instruction's ENDEARED verification expectation is not true for the current stored row.

The instruction says verification should load `stage_two_json` from `clue_pipeline_state` for ENDEARED clue id `10069320`.

I inspected that stored `stage_two_json`.

Current stored ENDEARED source candidates are:

```text
expensive -> DEAR, token SYN_F, span [3, 4], evidence_status verified
energy -> EN, token POS_F, span [4, 5], evidence_status failed
is -> DE, token SYN_F, span [5, 6], evidence_status verified
```

Current stored ENDEARED operation candidates are:

```text
Aim, token POS_I_FIRST, span [0, 1]
```

Current stored ENDEARED assemblies are:

```text
status: evidence_only
kind: positional
output: ENDEARED
parts:
  expensive -> DEAR
  energy -> EN
  is -> DE
```

The stored ENDEARED row does not currently contain these source candidates:

```text
Aim -> END
limit -> END
energy -> E
```

Therefore this expected result in the instruction is false for the verification method it specifies:

```text
At minimum: Aim->END (candidate), limit->END (candidate),
expensive->DEAR (candidate), energy->E (candidate).
```

## Consequence Of The ENDEARED Mismatch

If Codex implements the instruction exactly and then runs the instructed verification against current stored `stage_two_json`, ENDEARED will not show the four expected pieces.

That would not necessarily mean the one-file preservation code failed.

It would mean the instruction's verification target was wrong.

This is exactly the kind of false failure / false success setup that must be avoided.

## UNDEMOCRATIC Stored Data

For clue id `10069315`, the stored Stage Two data is consistent with the instruction's intended happy path.

Current stored source candidates:

```text
courted -> COURTED, token ANA_F, span [1, 2]
man -> MAN, token ANA_F, span [2, 3]
in charge -> IC, token ABR_F, span [3, 5]
```

Current stored operation candidates:

```text
Rogue, token ANA_I, span [0, 1]
```

Current stored assembly:

```text
status: answer_fit
output: UNDEMOCRATIC
parts:
  courted -> COURTED
  man -> MAN
  in charge -> IC
```

This is a valid case for checking that accepted assembly candidates become `"used"`.

## MAUI Stored Data

For clue id `10069319`, the stored Stage Two data is also consistent with the instruction's intended happy path.

Current stored source candidates:

```text
Graduate -> MA, token SYN_F, span [0, 1]
uniform -> U, token SYN_F, span [2, 3]
island, -> I, token SYN_F, span [4, 5]
```

Current stored operation candidates:

```text
none
```

Current stored assembly:

```text
status: answer_fit
output: MAUI
parts:
  Graduate -> MA
  uniform -> U
  island, -> I
```

This is a valid case for checking that the manual-role-aware MAUI proof does not regress and that selected candidates are marked `"used"`.

## Defect 1: ENDEARED Verification Target Is Wrong

The instruction currently mixes two different claims:

1. The current stored Stage Two row contains the desired four ENDEARED pieces.
2. Fresh evidence gathering may be able to find some or all of those pieces.

Only the first claim matters for the proposed verification, because the verification says to load stored `stage_two_json`.

The first claim is false in the current database.

The instruction must choose one of two corrected approaches.

### Corrected Approach A: Verify Stored Stage Two Preservation

If the point is to preserve what is already in `stage_two_json`, then ENDEARED expected output must be changed to the current stored candidates:

```text
expensive -> DEAR
energy -> EN
is -> DE
```

It must also preserve evidence status:

```text
expensive -> DEAR: verified
energy -> EN: failed, unlicensed_partial
is -> DE: verified
```

Under this approach, the change proves:

`Stage Three now preserves stored Stage Two candidates even when no complete answer-fit assembly is selected.`

It does not prove:

`Stage Two currently finds Aim -> END, limit -> END, and energy -> E in the stored ENDEARED row.`

### Corrected Approach B: Verify Fresh Stage Two Evidence

If the point is to prove that fresh Stage Two can find:

```text
Aim -> END
limit -> END
expensive -> DEAR
energy -> E
```

then the verification cannot load current stored `stage_two_json`.

It must rebuild Stage Two from the clue text, answer, and current RefDB, or otherwise explicitly regenerate the Stage Two casefile.

That is a different verification target and may require accepting that stored `stage_two_json` is stale or was produced by a different path.

## Defect 2: Span-Only `"used"` Matching Is Unsafe

The instruction proposes this logic:

```python
"used"
if (sc.get("span") and len(sc.get("span", [])) == 2
    and (sc["span"][0], sc["span"][1]) in _selected_spans)
else "candidate"
```

This uses only the source candidate span.

That is not safe.

If two candidates share the same clue span but have different values or token types, this will mark both as `"used"` even if the selected assembly used only one of them.

The `"used"` test should match selected assembly part identity, not only location.

Minimum safer key:

```text
span + cleaned value
```

Preferred safer key:

```text
span + cleaned value + token
```

The instruction should say:

```text
A source candidate is "used" only if its span and cleaned value match a selected assembly part. If token is present on both the candidate and the assembly part, token should also match. Span alone is not sufficient.
```

## Defect 3: `_source_check` Summary Can Mislead

The proposed `_source_check` branch summarizes candidates like this:

```python
"%s -> %s" % (c.get("text", "?"), c.get("value", "?"))
```

That loses the candidate's evidence status.

For current ENDEARED, this would list:

```text
energy -> EN
```

without saying that it has:

```text
evidence_status: failed
evidence_reason: value is not the full source text and no operation licence is attached
```

That would create a new misleading message: it would imply all listed source pieces are equally usable candidate evidence.

The detail should either:

1. include evidence status/reason in the summary, or
2. summarize only verified candidates and put all candidates in the evidence payload, or
3. explicitly say "candidate/failed source pieces" rather than "candidate source pieces".

Recommended wording:

```text
no complete assembly found; source candidates exist but were not accepted as a complete assembly: expensive -> DEAR [verified]; energy -> EN [failed]; is -> DE [verified]
```

## Defect 4: This Is Not A UI Display Fix

The instruction explicitly says:

```text
Do not change wfw_display_adapter.py.
```

Therefore this change will not, by itself, make candidate source tiles appear in the WFW clue breakdown UI.

It will preserve candidates in the Stage Three proof JSON and may improve the `source_evidence` check detail.

The instruction must say that plainly.

Correct wording:

```text
This is a stored-proof preservation change. It does not change the WFW display adapter and does not guarantee that candidate evidence appears as clue-breakdown tiles. UI rendering of these preserved candidates is a separate follow-up.
```

## Defect 5: Robustness Against Missing Casefile Attributes

The proposed code uses:

```python
for sc in casefile.source_candidates
```

and:

```python
_operation_candidates_out = tuple(casefile.operation_candidates)
```

Real Stage Two casefiles have those attributes.

However, existing tests and some hand-built `SimpleNamespace` casefiles may omit `operation_candidates` or `source_candidates`.

Current `test_stage_three_proof.py` already fails on a missing `operation_candidates` attribute in one hand-built casefile.

That existing failure is not caused by this instruction, but the new code should not add another direct-attribute failure.

Safer implementation:

```python
_source_candidates_in = tuple(
    getattr(casefile, "source_candidates", ()) or ())
_operation_candidates_out = tuple(
    getattr(casefile, "operation_candidates", ()) or ())
```

This matches existing defensive style elsewhere in `stage_three_proof.py`.

## What Is Sound In The Instruction

The following parts are sound or directionally sound:

- Adding `source_candidates` and `operation_candidates` to `StageThreeProof`.
- Giving them empty-tuple defaults at the end of the dataclass.
- Adding both to `as_dict`.
- Preserving source candidates in the proof object even when no assembly is selected.
- Preserving operation candidates in the proof object.
- Improving `_source_check` so it distinguishes "no candidates exist" from "candidates exist but no complete assembly was accepted."
- Excluding Stage Two failed-assembly-reason work from this instruction.
- Keeping the schema string as `stage_three_proof:v1` is technically compatible with absent fields in older rows, provided consumers use `.get(..., [])` style access.

## Required Corrections Before Sending

The instruction should be revised before being sent to Codex.

Required correction 1:

Replace the ENDEARED expected output if verification loads stored `stage_two_json`.

Correct expected stored candidates:

```text
expensive -> DEAR, candidate or used depending on selected assembly policy, evidence_status verified
energy -> EN, candidate, evidence_status failed
is -> DE, candidate, evidence_status verified
```

Because ENDEARED currently has no selected `answer_fit` assembly, all preserved candidates should normally have:

```text
assembly_status: candidate
```

Required correction 2:

If the desired ENDEARED target remains:

```text
Aim -> END
limit -> END
expensive -> DEAR
energy -> E
```

then the verification must rebuild Stage Two fresh and must not claim those pieces are present in current stored `stage_two_json`.

Required correction 3:

Change `"used"` detection from span-only to selected assembly part matching by span and value, preferably also token.

Required correction 4:

Make `_source_check` detail preserve evidence status or avoid implying failed candidates are usable.

Required correction 5:

Add a clear note that this change does not update WFW display tiles because `wfw_display_adapter.py` is out of scope.

Required correction 6:

Use defensive `getattr` for `source_candidates` and `operation_candidates` in `build_stage_three_proof`.

## Proposed Corrected Summary For Claude/Codex

The safe version of the instruction should say:

```text
Preserve Stage Two source_candidates and operation_candidates in the Stage Three proof JSON.

This is a stored-proof preservation change only. It does not alter the WFW display adapter, so candidate source evidence will not necessarily appear as clue-breakdown tiles until a later display change.

For each source candidate, add assembly_status:
  - "used" only when the candidate matches a selected answer-fit assembly part by span and cleaned value, and token when available
  - "candidate" otherwise

When no complete assembly is selected, _source_check should distinguish:
  - no source candidates exist
  - source candidates exist but no complete assembly was accepted

The source_evidence message must not hide evidence_status. If candidates include failed evidence, the detail should say so or include status labels.

Verification against current stored stage_two_json should expect ENDEARED to preserve the candidates actually present in that JSON:
  - expensive -> DEAR, verified
  - energy -> EN, failed/unlicensed_partial
  - is -> DE, verified

Do not claim that stored ENDEARED stage_two_json contains Aim -> END, limit -> END, or energy -> E unless the verification rebuilds Stage Two fresh and proves those candidates are present.
```

## Final Recommendation

Do not send the current instruction as-is.

Send a corrected version only after the ENDEARED verification target is fixed and the `"used"` matching logic is tightened.

The proposal is useful as a narrow stored-proof preservation step, but it must not be represented as a full WFW display fix or as proof that ENDEARED already contains the four desired source pieces in stored Stage Two data.
