# Proposal: Stop Review-State Candidate Blocks Looking Like Verified Roles

Date: 2026-05-27

## Problem

The clue page is showing clue words in strong semantic colours even when those
words do not yet have a verified WFW role.

Example:

- Telegraph Cryptic #31246, 22 Across
- clue id: `10068296`
- clue: `Cartoon character, old and troubled by love`
- answer: `Donald Duck`
- WFW attempt status: `wfw_review`

Current display blocks produced by `display_from_wfw_proof_attempt`:

```text
[0,1] REVIEW_BLOCK unaccounted        Cartoon     No role found
[1,2] DEF_BLOCK    definition         character,  DONALDDUCK
[2,3] REVIEW_BLOCK unaccounted        old         No role found
[3,4] OP_BLOCK     joiner_indicator   and
[4,5] OP_BLOCK     anagram_indicator  troubled
[5,6] OP_BLOCK     joiner_indicator   by
[6,7] REVIEW_BLOCK unaccounted        love        No role found
```

The problem is not merely CSS. The display adapter maps unverified Stage Three
candidate blocks to final-looking semantic roles:

- `OP_BLOCK` + `token=ANA_I` becomes `anagram_indicator`
- `OP_BLOCK` + `token=LNK` becomes `joiner_indicator`

The template then colours these as real indicator/link roles. To a human user,
this reads as "the system knows what this word does", even though the proof is
review-status and the candidate still needs evidence.

This is why the colours feel random: they are candidate guesses being rendered
with verified-role visual language.

## Required Product Invariant

In a `wfw_review` proof, no clue word should receive a strong semantic role
colour unless that role is actually verified as part of a valid parse/assembly.

Candidate words may be displayed, but they must look like candidates/review
items, not solved roles.

More precise invariant:

1. `status == wfw_proven`
   - verified blocks may render with semantic colours:
     - definition green
     - source piece colours
     - operation/indicator colours when attached to the accepted assembly

2. `status == wfw_review`
   - unresolved words render as review/unaccounted
   - unattached OP candidates render as candidate/review, not as final semantic
     indicator roles
   - the underlying candidate type may be shown textually, e.g.
     `candidate anagram indicator`, but not with the same colour treatment as a
     verified role
   - candidate metadata must remain available for admin diagnosis and
     `missing_enrichments`

## Proposed Structural Fix

Do not fix this in CSS alone.

The adapter should preserve the distinction between:

- verified display role
- candidate hypothesis

### Option A: Add candidate display roles

In `signature_solver/wfw_display_adapter.py`, when adapting Stage Three
`OP_BLOCK`s:

If the Stage Three block is candidate/review-only, emit:

```python
{
    "kind": "REVIEW_BLOCK",
    "role": "candidate_indicator",
    "candidate_role": "anagram_indicator",
    "text": "troubled",
    "value": "Candidate: anagram indicator",
    "span": [4, 5],
}
```

For `LNK`/joiner candidates:

```python
{
    "kind": "REVIEW_BLOCK",
    "role": "candidate_separator",
    "candidate_role": "joiner_indicator",
    "text": "by",
    "value": "Candidate: separator",
    "span": [5, 6],
}
```

Then update `partials/atomic_parse.html` so:

- `candidate_indicator` and `candidate_separator` use neutral/review styling
- they do not match `(block.role or '').endswith('indicator')`
- the candidate type is visible in small text but does not claim proof

This keeps review candidates visible without giving them solved-role colours.

### Option B: Keep OP_BLOCK kind but use non-semantic candidate roles

Alternative:

```python
kind = "OP_BLOCK"
role = "candidate_indicator"
candidate_role = "anagram_indicator"
```

This is less clean because the template already treats `OP_BLOCK` as semantic
operation styling in several places. It is more likely to require defensive CSS
exceptions.

Recommendation: Option A.

## What Not To Do

Do not simply make all OP blocks grey in the template.

That would hide the real distinction between:

- verified operation blocks in a proven parse
- candidate operation blocks in a review parse

Do not special-case Telegraph #31246 22 Across, `troubled`, `by`, or `Donald Duck`.

Do not remove candidate data from the proof. The evidence gaps are useful. The
fix is only about visible role semantics.

## Expected Display For #31246 22 Across

For:

```text
Cartoon character, old and troubled by love
```

Until a valid assembly exists, the visible breakdown should read roughly:

```text
Cartoon      No role found
character,   Definition candidate/definition needs boundary review
old          No role found
and          Candidate separator
troubled     Candidate anagram indicator
by           Candidate separator
love         No role found
```

The exact wording can vary, but the key rule is:

- `and`, `troubled`, and `by` must not receive the same semantic colours as
  verified indicators/links in solved clues.

## Verification Plan

Add a regression test for the display adapter using either the real stored
proof for clue `10068296` or a compact synthetic Stage Three proof.

Assertions:

1. The display remains `wfw_review`.
2. `troubled` is present.
3. `troubled` is not rendered with role `anagram_indicator`.
4. `troubled` carries candidate metadata, e.g.
   `candidate_role == "anagram_indicator"`.
5. `and`/`by` are not rendered with role `joiner_indicator`.
6. `missing_enrichments` still contains the relevant purpose requests.
7. Proven WFW clues still render verified operation roles with semantic colours.

Then run:

```text
python -m py_compile signature_solver/wfw_display_adapter.py
python signature_solver/test_wfw_display_adapter.py
python web/test_clue_wfw_render_contract.py
```

Manual browser check:

```text
http://127.0.0.1:5005/clue/10068296?admin=dev-admin-key
```

## Related But Separate Problem

This proposal addresses random semantic colours for unverified candidate roles.

It does not address the separate definition-boundary problem, e.g. definitions
starting in the middle of a clue. That needs a second proposal/invariant:

- a cryptic definition should normally be at the beginning or end unless there
  is explicit proof that an internal definition is valid
- review-status internal definitions must not be rendered as verified

Do not mix the two fixes in one patch.

