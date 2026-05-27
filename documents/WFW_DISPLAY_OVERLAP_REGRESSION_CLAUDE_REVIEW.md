# WFW Display Overlap Regression - Claude Review

Date: 2026-05-27

## User-visible regression

Clue page:

- Telegraph Cryptic #31246
- 12 Across
- clue id: 10068292
- clue text: `Latest about caging rook and another bird ...`
- answer: `WREN`

The WFW clue breakdown was displaying overlapping surface text:

- `bird`
- `bird ...`
- `...`

This violated the previous fix: each clue word should appear once in the visible WFW breakdown. Punctuation-only operation candidates should not be displayed as clue-word blocks.

## Stored proof source

Latest WFW proof attempts for clue 10068292 are `wfw_review` rows using `stage_three_proof:v1`.

The stored Stage Three proof contains overlapping blocks:

```python
{'kind': 'DEF_BLOCK', 'span': [6, 8], 'text': 'bird ...', 'value': 'WREN'}
{'kind': 'DEF_BLOCK', 'span': [6, 7], 'text': 'bird', 'value': 'WREN'}
{'kind': 'OP_BLOCK', 'span': [7, 8], 'text': '...', 'token': 'ANA_I'}
```

The display adapter was only deduping exact same-span blocks, so nested/same-word duplicates leaked to the page.

## Codex patch applied

File changed:

- `signature_solver/wfw_display_adapter.py`

Function changed:

- `_dedupe_stage_three_display_blocks`

Behavior now:

- drop punctuation-only blocks from the Stage Three display
- collapse blocks with the same alphanumeric word key, preferring:
  - higher display priority
  - less punctuation
  - shorter visible text

This is display-only. It does not change stored proof JSON, solver status, scoring, or DB facts.

## Verification performed

Direct live attempt display for clue 10068292 now yields:

```python
['Latest', 'about', 'caging', 'rook', 'and', 'another', 'bird']
```

No `bird ...`; no standalone `...`.

Checks run:

```text
py_compile signature_solver/wfw_display_adapter.py signature_solver/test_wfw_display_adapter.py
web/test_clue_wfw_render_contract.py passed
WREN-specific synthetic overlap check passed
live attempt display check passed
```

## Important unresolved point

`signature_solver/test_wfw_display_adapter.py` currently fails at an older TIJUANA assertion expecting `taken` to render as an unaccounted `REVIEW_BLOCK`.

Observed current display for TIJUANA:

```python
(3, 4) OP_BLOCK container_indicator 'taken'
```

This appears unrelated to the WREN duplicate-word regression because the stored Stage Three proof itself marks `taken` as an OP candidate. Please review whether that older assertion is stale or whether Stage Three review display should demote unattached OP candidates back to review blocks.

Do not paper over this by weakening WREN display behavior. The user is specifically concerned that old "words displayed more than once" work has regressed.

