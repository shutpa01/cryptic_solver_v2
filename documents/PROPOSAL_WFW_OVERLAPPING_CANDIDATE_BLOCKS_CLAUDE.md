# Proposal: Fix Overlapping WFW Candidate Blocks

## Problem

The WFW clue breakdown still shows the same clue words multiple times when
Stage Three emits overlapping candidate blocks.

Recent visible example:

`What UK constituencies have to call into question?`

The clue breakdown shows overlapping blocks:

- `to` as candidate link
- `to call into question?` as definition
- `call into question?` as definition
- `into` as candidate anagram indicator
- `question?` as definition

This is not merely duplicate text. The existing display dedupe only handles:

- exact same span
- exact same normalized text

The failing case is interval overlap: several blocks cover intersecting spans
but are not exact duplicates.

## Goal

At display-adapter level, render a non-overlapping set of blocks for review
status clues. Each clue word should appear in at most one displayed block.

This is display-only. It must not change:

- Stage Three proof generation
- stored proof rows
- solver confidence
- missing enrichment extraction
- manual parser data
- DB facts

## Current Code

The relevant adapter is:

`signature_solver/wfw_display_adapter.py`

Current path:

```python
blocks = []
for idx, block in enumerate(proof.get("blocks") or []):
    blocks.append(_stage_three_display_block(...))
blocks.extend(_stage_three_missing_word_blocks(...))
...
_normalise_anagram_display_roles(blocks, answer_links)
_dedupe_stage_three_display_blocks(blocks)
...
"blocks": sorted(blocks, key=_block_sort_key)
```

Current helper:

```python
def _dedupe_stage_three_display_blocks(blocks):
    ...
```

It dedupes exact spans and exact word keys, but does not resolve overlapping
intervals.

## Proposed Fix

Replace or extend `_dedupe_stage_three_display_blocks` so it also resolves
overlapping spans.

The rule should be:

1. Blocks with no valid span are kept unless their normalized text duplicates a
   kept block.
2. Blocks with spans are sorted by priority score (descending).
3. **Greedy sweep**: iterate blocks in priority order. Keep a block if its span
   does not overlap any already-kept span. Skip it otherwise.
4. If priority ties, prefer:
   - shorter span length (fewer words wins)
   - fewer punctuation marks
   - shorter display text
5. After interval selection, merge no-span blocks back in, then collapse
   duplicate normalized text across the combined set as the current function
   does.
6. Sort the final list by `_block_sort_key`.

This turns a set of overlapping candidate definitions into one chosen displayed
block instead of several repeated fragments.

## Priority

Use the following priority scheme. Among candidate blocks, **shorter span
length wins** — this is the primary tiebreaker and directly fixes the
repeated-green problem without needing a fine-grained role taxonomy.

Priority scores (higher = kept first in the greedy sweep):

1. `SOURCE_BLOCK` with manual/evidence-backed source role → 100
2. `SOURCE_BLOCK` (other) → 90
3. `OP_BLOCK` → 80
4. candidate operation/link blocks:
   - `role == "op_candidate"` → 70
   - `role == "link_candidate"` → 70
   - other `*_indicator` roles → 70
5. `DEF_BLOCK` where `role != "inferred_definition"` → 60
6. `DEF_BLOCK` where `role == "inferred_definition"` → 40
7. unresolved/review blocks → 20
8. link/surface filler → 10

**Critical:** the existing `_priority()` function inside
`_dedupe_stage_three_display_blocks` gives ALL `DEF_BLOCK`s a score of 100.
That is the root cause of the bug. It must be updated so that
`DEF_BLOCK` with `role == "inferred_definition"` scores 40 (below `OP_BLOCK`
and candidate indicator roles). Non-inferred `DEF_BLOCK` stays high (60 is
fine; adjust to taste).

Among blocks with equal priority score, break ties by span length: shorter
span wins. Then fewer punctuation marks. Then shorter display text.

## Specific Helper Shape

Add helpers inside `wfw_display_adapter.py` near `_dedupe_stage_three_display_blocks`:

```python
def _span_tuple(block):
    span = block.get("span")
    if not isinstance(span, list) or len(span) != 2:
        return None
    start, end = span
    if start is None or end is None or end <= start:
        return None
    return (start, end)


def _spans_overlap(left, right):
    return left[0] < right[1] and right[0] < left[1]
```

Then adjust `_dedupe_stage_three_display_blocks` to:

1. Separate span-blocks from no-span blocks.
2. Collapse exact same spans using `_better_block` (as now).
3. Sort remaining span-blocks descending by priority score; break ties by span
   length ascending (shorter wins), then punctuation count ascending, then
   display text length ascending.
4. Greedy sweep: iterate in that order, keep each block if its span does not
   overlap any already-kept span.
5. Merge in no-span blocks.
6. Collapse duplicate normalized text across the combined set using `_better_block`
   (as the current word-key dedupe does).

Also update `_priority()` as described in the Priority section above.

## Test Requirement

Add or update a direct display-adapter test in:

`signature_solver/test_wfw_display_adapter.py`

Use a synthetic Stage Three proof, not a live DB clue.

Test case:

Blocks before adapter:

- `REVIEW_BLOCK`, role `unaccounted`, text `What`, span `[0,1]`
- `REVIEW_BLOCK`, role `unaccounted`, text `UK`, span `[1,2]`
- `REVIEW_BLOCK`, role `unaccounted`, text `constituencies`, span `[2,3]`
- `REVIEW_BLOCK`, role `unaccounted`, text `have`, span `[3,4]`
- `LINK_BLOCK` or `REVIEW_BLOCK`, role `link_candidate`, text `to`, span `[4,5]`
- `DEF_BLOCK`, role `inferred_definition`, text `to call into question?`, span `[4,8]`
- `DEF_BLOCK`, role `inferred_definition`, text `call into question?`, span `[5,8]`
- `OP_BLOCK` or `REVIEW_BLOCK`, role `op_candidate`, candidate_role `anagram_indicator`, text `into`, span `[6,7]`
- `DEF_BLOCK`, role `inferred_definition`, text `question?`, span `[7,8]`

Expected display:

- No two returned blocks have overlapping spans.
- The words `to`, `call`, `into`, `question` do not appear in multiple displayed
  blocks.
- At most one definition candidate survives in that overlapping region.
- Candidate link/op blocks are not duplicated by larger definition candidates.

Do not assert the exact final set unless the priority rules make that stable.
The key invariant is non-overlap.

Also keep the existing TIJUANA/WREN/Donald Duck tests passing.

## Verification

Run:

```powershell
.\.venv\Scripts\python.exe signature_solver\test_wfw_display_adapter.py
.\.venv\Scripts\python.exe web\test_clue_wfw_render_contract.py
```

No DB writes are required.

## Anti-patterns

- Do not fix one live clue by editing stored proof data.
- Do not query the live DB in the regression test.
- Do not remove missing_enrichments.
- Do not change Stage Three candidate generation.
- Do not hide all review candidates; keep a non-overlapping diagnostic view.
- Do not make definitions from non-edge positions valid again.

