# Phase C — Worklist for production solver

The v0 adapter (`adapter.py`) wraps `signature_solver.solver.solve_clue` and
emits the universal form. Where the solver doesn't surface structure that
the adapter needs, the adapter raises a flag. This document turns those
flags into a list of TODOs for the production code.

**The wrapper is scaffolding.** Each entry below is structure the solver
already knows internally but doesn't make available on `SolveResult`.

## Baseline measurements

Test bed: DT 31132, 31138, 31150 (92 clues total — unprocessed or
mechanical-only).

| Source | PASS | FAIL | NO_FORM |
|---|---|---|---|
| `sig_to_form.py` (previous thread) | 8 (9%) | 7 (8%) | 77 (84%) |
| `universal_form_v2` v0 | 10 (11%) | 36 (39%) | 46 (50%) |

The shift is the headline: NO_FORM dropped from 84% → 50% because v0 calls
the full `solve_clue` (including grammar_triage and Haiku enrichment)
instead of just catalog matchers. Most of the new-FAIL cases are clues
that the previous wrapper silently dropped.

## Flag tallies (combined across the three puzzles)

| Count | Flag |
|---|---|
| 46 | `solver_no_result` |
|  6 | `container_outer_inner_not_preserved` |
|  5 | `op_name_not_preserved_in_solveresult` |
|  5 | `op_inferred_from_token_mix_not_provided` |
|  3 | `container_charade_structure_not_preserved` |
|  2 | `hidden_indicator_not_tagged_by_solver` |
|  1 | `definition_phrase_missing` |

Plus a higher-order finding (every-other-clue level): the `residue` check
on FAILs frequently shows clue words tagged LNK that aren't link words.
Sig-solver's LNK includes "redundant indicators" and a broader DB-driven
allow-list. The form faithfully reflects this; the verifier (rightly)
rejects them.

## Worklist entries

### W-1. `solve_clue` returns no result for ~50% of clean-puzzle clues

**Flag**: `solver_no_result` (46 of 92 clues).

**Evidence**: even with grammar_triage + DBE-Haiku + indicator-enrichment
+ Haiku-def fallback, the production solver fails to return a
`SignatureResult` for half of unprocessed test puzzles.

**Implication**: The form-emitting wrapper isn't the bottleneck for
coverage. **Out of scope for the wrapper.** Addressed by the leftover
processing layer once we get there. Records the headline gap in coverage.

**Action**: Not adapter work. Confirm that leftover-processing layer is
where we'll close this gap.

### W-2. Catalog matchers don't preserve `entry.operation`

**Flag**: `op_name_not_preserved_in_solveresult` (5 cases) plus
`op_inferred_from_token_mix_not_provided` (paired, same root cause).

**Evidence**: `signature_solver/solver.py:_process_match` builds a
`SignatureResult` from `(entry, assignment, words, answer)` but **doesn't
attach `entry.operation`** to it. Grammar triage smuggles the op name
via `confidence_reasons[0][0]` (e.g. `[('anagram', 0)]`) — fragile but
present. The catalog path returns scoring deltas in `confidence_reasons`,
no op name anywhere.

**Production-code change** (when we get there):

- `signature_solver/solver.py:_process_match` line ~586:
  `sig_result = SignatureResult(sig_tokens, word_roles, [explanation])`
  → also attach `sig_result.operation = entry.operation`.
- `SignatureResult.__init__` should accept and store `operation`.
- `grammar_triage` paths should set the same field directly, not via
  `confidence_reasons`.

**Adapter plug**: v0 already infers from token mix. Acceptable workaround,
but each inference is a guess.

### W-3. Container outer/inner role lost

**Flag**: `container_outer_inner_not_preserved` (6 cases).

**Evidence**: `_verify_container_combo`, `_verify_container_reversal_combo`,
`_verify_container_charade_combo` each try both orderings and return only
the verified `combo` tuple. Which element was outer is lost.

**Production-code change**:

- `signature_solver/matcher.py:_verify_container_combo` (and the two
  related functions) should return `(combo, outer_idx, inner_idx)` or
  similar tagged structure.
- `SignatureResult` needs a way to carry per-piece role tags
  (`role: outer | inner | base | removed | fodder`).
- Or simpler: emit the structured form right there. Skip word_roles
  collapsing.

**Adapter plug**: v0's verifier swaps outer/inner on its own (`_produces`
for container tries both orderings). This works for assembly but means
the form's recorded `[outer, inner]` ordering is unreliable downstream.

### W-4. Compound op nesting lost

**Flag**: `container_charade_structure_not_preserved` (3 cases),
`reversal_inner_assumed_charade` (potential).

**Evidence**: When `entry.operation == "container_charade"`, the catalog
matcher knows that some pieces form the inner charade and one is the
outer container. The verification function (`_verify_container_charade_combo`)
exhaustively tries pairs of pieces as outer/inner with the rest charade'd.
On success it returns the combo only — not the (outer, inner, rest_idx_list)
shape that was verified.

Same pattern for `anagram_charade`, `anagram_container`, `container_reversal`,
`container_with_deletion`.

**Production-code change**:

- Each `_verify_*_compound_combo` should return the structured assembly
  description (which piece plays which role, including the operation tree).
- The matcher should attach this tree to the `assignment` dict.
- The wrapper then walks it directly — no inference needed.

**Adapter plug**: v0 falls back to heuristic ("first piece outer, rest
inner-charade"). Wrong sometimes, flagged when used.

### W-5. Hidden indicator not tagged when the clue is &lit-style

**Flag**: `hidden_indicator_not_tagged_by_solver` (2 cases).

**Evidence**: Some hidden clues genuinely have no surface "in"/"hidden in"
indicator — the whole fodder phrase reads as the &lit indicator. E.g.
DT 31150 21d "Veteran undersold sterling pounds → OLDSTER".

**Two questions** the user needs to settle:

a) **Spec question**: should the universal form schema permit `hidden`
   nodes without indicators when the clue is genuinely &lit-style?
   Currently the bridge check fails them. The spec says every non-literal
   node has an indicator.

b) **Solver question**: should `solve_clue` flag &lit-hidden cases
   explicitly (e.g. set `is_and_lit=True` on the result)?

If (a) is "yes, allow no-indicator hidden when is_and_lit=True", then (b)
follows: the solver should set `is_and_lit` and the bridge check should
exempt that case.

### W-6. LNK bucket includes non-link content words

**Flag**: not raised by the adapter directly; surfaces as
`residue: non-link words declared as link_words` in verifier output.

**Evidence**: For ~half the FAIL cases, sig_solver's `lnk_indices` includes
words like "Neckwear", "Trump", "Donald", "Dexter's", "lead",
"character" — clearly content words. Examining `solver.py:_process_match`
and `_remaining_are_valid`:

- `_remaining_are_valid` accepts a remaining word as link if
  `db.is_link_word(w)` OR if it's a "redundant indicator" (an indicator
  of a type already assigned).
- `grammar_triage` is more aggressive — every word not assigned to fodder/
  indicator/def is dumped into LNK.

So the LNK bucket has at least three meanings mixed in:
1. Genuine connective words (a, of, in, etc.)
2. Redundant indicators (the second hidden indicator when one is already placed)
3. Words grammar_triage couldn't classify but kept the parse alive

**Production-code change**:

- Distinguish these categories on the SignatureResult (`lnk_genuine`,
  `lnk_redundant_indicator`, `lnk_unclassified`).
- Better: don't accept a parse where words fall into category 3 — that's
  a parse that doesn't account for those words.

**Adapter plug**: the wrapper currently dumps everything into
`form.link_words` and the verifier rejects it. Could split based on a
rule (against the LINK_WORDS allow-list) but that's a bandaid for a
solver-level issue.

### W-7. Definition phrase sometimes empty

**Flag**: `definition_phrase_missing` (1 case).

**Evidence**: some `solve_clue` paths return without setting
`sr.definition`, particularly when no candidate from
`extract_definition_candidates` won and no Haiku fallback fired.

**Production-code change**:

- Make `solve_clue` always set `sr.definition` to something — even if it's
  just the original candidate it tried first or `None` explicitly.
- Decide what the wrapper does when no definition is available.

**Adapter plug**: empty string today; flag raised.

## Summary worklist

In rough priority order:

| # | Flag | Cost to fix in production | Coverage |
|---|---|---|---|
| W-1 | `solver_no_result` | Out of scope (leftover layer) | 50% of clues |
| W-2 | op name not preserved | Tiny — one attribute | 5+ cases |
| W-3 | container outer/inner lost | Small — change 3 verify functions | 6 cases |
| W-4 | compound op nesting lost | Medium — emit structured assembly from each compound verifier | 3+ cases |
| W-5 | hidden &lit indicator | Spec question + small solver flag | 2 cases |
| W-6 | LNK semantics | Small — split bucket | many cases (residue fails) |
| W-7 | empty definition | Tiny — always set | 1 case |

**Net**: If W-2 through W-7 are addressed in production, the wrapper
shrinks dramatically and the FAIL count drops because the verifier no
longer trips on lost structure. W-1 remains the headline coverage problem
and is what the leftover layer must address.

## Next decisions for the user

1. **W-5 spec question**: allow `hidden` without indicator when &lit?
2. **W-1 / leftover format**: is the leftover-processing layer next? Or
   do we want to first try a different strategy for the 46 NO_FORM cases?
   (e.g. give up and let leftover handle them, vs. try widening the
   solver's coverage some other way.)
3. **Production-code worklist**: do W-2 through W-7 go on a separate
   branch as production-code changes? Or stay simulation-only until the
   wrapper plateaus on test-bed expansion?
