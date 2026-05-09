# Phase A — SolveResult → Form mapping (v0)

Purpose: list, for every field of the universal form, where in `SolveResult`
the data comes from. Where data is missing or only recoverable by heuristic,
this document marks it as a **WORKLIST** entry — a TODO for the production
solver, since the data exists at compute-time but isn't surfaced.

## Inputs the adapter has

```python
result: SolveResult = solve_clue(clue_text, answer, db)
# .result          — SignatureResult or None
# .confidence      — 0..100
# .confidence_reasons — [(reason, delta), ...]
# .analyses        — list[WordAnalysis]
# .phrases         — dict[(i,j), WordAnalysis]
# .definition      — str (set by solve_clue)
# .dbe_haiku_candidates — optional dict (set when fired)
# .suggested_indicators — optional list (set when fired)
```

`SignatureResult.word_roles` is the per-word role list:
- 3-tuple `(word, token, value)`
- 4-tuple `(word, token, value, meta)` — only from `grammar_triage` path
- token is one of the constants in `signature_solver/tokens.py`
  (SYN_F, ABR_F, ANA_F, RAW, HID_F, HOM_F, POS_F, DEL_F,
   ANA_I, REV_I, CON_I, DEL_I, HID_I, HOM_I,
   POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE, POS_I_ALTERNATE,
   POS_I_HALF, POS_I_TRIM_FIRST/LAST/MIDDLE/OUTER, LNK, DBE_MARKER)

## Form fields → source

### Form.definition

| Form field            | Source                                  | Notes                          |
|-----------------------|-----------------------------------------|--------------------------------|
| `definition.phrase`   | `solve_result.definition`               | Set by `solve_clue` from winning candidate |
| `definition.answer`   | passed in                               | Always available               |

Clean. No worklist entry.

### Form.tree (root operation)

| Form field         | Source                                                | Status |
|--------------------|-------------------------------------------------------|--------|
| `tree.operation`   | grammar_triage path: `confidence_reasons[0][0]`       | Fragile (smuggling) |
| `tree.operation`   | catalog matchers path: **NOT PRESERVED**              | **WORKLIST #1** |
| `tree.indicator`   | `word_roles` token in `INDICATOR_TOKENS`              | Available, but typed match required |
| `tree.sources`     | `word_roles` fodder tokens, in word_roles order       | Order is engine-dependent — see below |

**WORKLIST #1**: `solver.py:_process_match` discards `entry.operation` when building `SignatureResult`. Should attach `entry.operation` to the result so the adapter doesn't have to re-infer.

**Engine-dependent order**:
- Catalog matchers: `word_roles` is `[fodder pieces in assembly order, indicators, LNKs]`.
- Grammar triage: `word_roles` is in clue order (each clue word appears once; LNK / indicator / fodder tags assigned in place).

**WORKLIST #2**: word_roles ordering should be canonicalised — clue order with role tags is more useful than assembly-order shuffles.

### Form.tree.sources — leaves

For leaf operation (`literal`/`synonym`/`abbreviation`/`positional`/`hidden`/`homophone`):

| Token            | Leaf op       | source_word       | value            |
|------------------|---------------|-------------------|------------------|
| `SYN_F`          | synonym       | `word_roles[i][0]` | `word_roles[i][2]` |
| `ABR_F`          | abbreviation  | same              | same             |
| `RAW`            | literal       | same              | same             |
| `ANA_F`          | literal *(child of anagram parent)* | same | letters |
| `HID_F`          | literal *(child of hidden parent)*   | same | letters |
| `HOM_F`          | homophone     | same              | same             |
| `POS_F`          | positional + kind | same          | same             |
| `DEL_F`          | literal *(child of deletion parent)* | same | letters |

Available, but the **kind** (positional kind, deletion subtype) needs the
indicator-token-in-context to determine — see below.

### Form.tree (compound / nested operations)

Token mix in `word_roles` tells us which operation type fired:

- `ANA_F + ANA_I` → anagram of fodder
- `HID_F + HID_I` → hidden in fodder
- `(SYN_F | ABR_F | RAW) × N` only → charade / DD
- `(SYN_F | ABR_F) × 2` + `CON_I` → container
- any of above + `REV_I` → reversal wrapping
- `SYN_F + DEL_I` or `SYN_F + ABR_F + DEL_I` → deletion
- `POS_I_*` indicates positional kind / deletion subtype

**Compound ops** (e.g. `container_reversal`, `anagram_charade`, `container_with_deletion`):
- The exact nesting (which op wraps which) is **lost** in word_roles.
- The `meta` dict on grammar_triage's word_roles 4-tuples has SOME nesting info
  (`transform=reversed`, `transform=deletion`, `derived`, `subtype`) — but only
  for the grammar_triage path.
- For the catalog path: nesting is lost. Can re-infer from token mix + answer
  letter assembly, but the matcher had explicit `entry.operation =
  'container_reversal'` etc.

**WORKLIST #3**: Catalog matchers should attach the structured assembly description (which child is outer/inner, which is the anagram fodder, etc.) to the result. This is the biggest single gap — every container/compound clue depends on it.

### Form.tree (container outer/inner)

`_verify_container_combo` and friends try both orderings during verification but discard which one worked. By the time the result is returned, only the original `combo` tuple survives — order is not recoverable.

**WORKLIST #4**: `_verify_container_combo`, `_verify_container_reversal_combo`, `_verify_container_charade_combo` should return `(outer_idx, inner_idx)` alongside the verified combo so the adapter knows the role.

### Form.link_words

| Source                              | Notes                              |
|-------------------------------------|------------------------------------|
| `word_roles` entries with token=LNK | Available. Catalog and grammar triage both tag LNK words. |

Clean.

### Form.is_and_lit

Not computed anywhere in the solver.

**WORKLIST #5**: requires DB lookup against `definition_answers_augmented`
and `synonyms_pairs` for the full clue string. Adapter can do this directly
(no production change required) — small cost.

## Per-engine summary

### grammar_triage path — what the adapter gets

- `result.confidence_reasons[0][0]` = op name
- `result.result.word_roles` = clue-ordered, with optional `meta` 4th element
- meta carries: `transform` (reversed/deletion), `derived`, `subtype`,
  `reversed_to`, `reversal_indicator`, `source` (dbe)
- Indicator tokens in word_roles point to the indicator clue-words
- Operations grammar_triage emits: `anagram`, `charade`, `container`,
  `container_charade`, `reversal`, `anagram_charade` (per the return-statement grep)

### Catalog matchers path — what the adapter gets

- `result.confidence_reasons` is scoring deltas (no op name)
- `result.result.word_roles` is `[fodder, indicators, LNKs]` — NOT clue order
- No meta dict on word_roles
- Operation must be re-inferred from token mix (heuristic)

### DD pre-pass / hidden pre-pass

The production `solve_clue` does NOT use these. `sig_to_form.py` did but we're
ignoring that. So actually for the v0 adapter we just need to handle:
1. Grammar triage hits (some clues)
2. Catalog matcher hits (some clues)
3. None of the above (NO_FORM)

## v0 adapter scope

Build a function `solve_result_to_form(SolveResult, clue_text, answer)` that:

1. Returns `None, [reason]` if `result.result is None` (clue unsolved).
2. Determines op from `confidence_reasons[0][0]` if it's an op-name string;
   else from token-mix heuristic (mark as **HEURISTIC** flag).
3. Walks `word_roles`:
   - Categorises each entry (fodder / indicator / LNK / DBE_MARKER / DEF).
   - Builds leaves from fodder tokens.
   - Identifies indicator words by token type (ANA_I → anagram indicator, etc.).
4. Wraps leaves in op-tree according to detected op.
5. For compound ops, returns the form with a **flag** indicating the nesting was
   guessed.
6. Sets link_words from LNK entries.
7. Sets is_and_lit by DB lookup (cheap, no production change).

What v0 explicitly **does not** do:
- No surface-text inference (no `_maybe_infer_container`,
  `_attach_op_indicators`, etc.).
- No reordering of pieces.
- No "swap container roles if assembly fails" — flag instead.
- No outer-op promotion via wordplay_types.

The point is to expose where the solver fails to surface structure — by being
honest about what's missing rather than papering over it.
