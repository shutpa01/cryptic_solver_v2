# WFW Manual Roles Proof Integration Plan

Date: 2026-05-25

This document is for Claude/Codex review before implementation.

## Summary

The current WFW review system mixes three different concepts:

1. **Proof state**: what Stage Three has actually verified.
2. **Reviewer annotations**: what a human has marked in the admin UI.
3. **Display state**: what the clue page visually shows.

Those concepts are not consistently connected. As a result, the page can show every clue word visually accounted for while the stored Stage Three proof still fails. This is not a cosmetic bug. It is a proof-pipeline contract failure.

The fix is not to make a single clue look green. The fix is to make reviewer edits either real verifier input or clearly non-authoritative annotation. For WFW review, manual roles should become verifier input.

## Non-Negotiable Rules

1. No display-only change may be presented as proof repair.
2. Manual roles must either feed the verifier or be labelled as non-authoritative annotation.
3. Stage Three pass/fail must be explainable from stored proof data.
4. Reverify must state exactly what it consumes:
   - retained Stage Two evidence,
   - manual roles,
   - current reference DB facts,
   - or a fresh solver rebuild.
5. All token spans must declare or clearly imply their index space.
6. Missing proof must not be hidden by repainting tiles.

## Current Observed Failure: MAUI

Clue:

```text
Graduate with uniform by island, one in Hawaii
```

Answer:

```text
MAUI
```

The page can show:

```text
Graduate -> MA
with -> Link
uniform -> U
by -> Link
island -> I
one -> Definition
in -> Definition
Hawaii -> Definition
```

But the proof banner can still say:

```text
WFW needs review
MA + U + I = MAUI
The definition still needs accepted evidence.
These clue words need a clear purpose before publication: with, by, one, in
```

This contradiction happens because manual roles are saved and overlaid onto display tiles, but Stage Three still reads stale stored proof data and does not consume those roles as proof evidence.

## Known Issues To Address

### 1. Manual Roles Save But Do Not Prove

Current behavior:

- Admin dropdown writes to `clue_word_roles`.
- Clue page overlays those roles onto WFW tiles.
- Stage Three still uses stored `wfw_proof_attempts.proof_json.word_purposes`.
- Review messages still come from old Stage Three checks.

Required fix:

Manual roles must become verifier input.

Implementation design:

- Add manual roles to the Stage Two / Stage Three casefile path.
- Load `clue_word_roles` during clue reverify and puzzle reverify.
- Store manual roles in Stage Two JSON so retained reverify can reproduce the result.
- Stage Three `_word_purposes` must consume manual roles before declaring a word unresolved.

Manual role schema:

```json
{
  "index": 5,
  "text": "one",
  "role": "definition",
  "letters": null,
  "source": "manual"
}
```

Acceptance:

- If `with = link`, Stage Three no longer reports `with` as missing purpose.
- If `by = link`, Stage Three no longer reports `by` as missing purpose.
- If `one/in/Hawaii = definition`, Stage Three no longer reports those words as missing purpose.

### 2. Display Overlay Creates False Confidence

Current behavior:

- Tiles can show corrected roles.
- Banner can still say WFW review.
- User sees a contradiction.

Required fix:

Separate visual overlay from proof display.

Short-term plan:

- Show a warning when manual roles exist but the stored proof has not been regenerated:

```text
Manual review roles have changed. Reverify to update proof status.
```

Medium-term plan:

- After reverify consumes manual roles, reduce or remove display-only overlay.

Long-term plan:

- Display should render from regenerated proof rows, not from runtime overlay patches.

Acceptance:

- If manual roles differ from stored proof, the page explicitly says so.
- After successful reverify, no warning remains because proof and display agree.

### 3. Reverify Is Ambiguous

Current behavior:

- Puzzle reverify uses retained Stage Two evidence.
- Clue reverify behavior is not clearly aligned with WFW manual roles.
- User expects `Re-verify` to use their edits.

Required fix:

Split actions or label them clearly.

Recommended actions:

1. **Reverify Stored Evidence**
   Uses retained Stage Two JSON plus manual roles plus current DB facts.
2. **Rebuild Evidence**
   Reruns enrichment / Stage Two from current DB and manual roles.
3. **Re-solve**
   Runs solver pipeline from scratch.

Acceptance:

- Button labels describe actual behavior.
- Reverify consumes manual `clue_word_roles`.
- If retained Stage Two is missing or stale, UI says so.

### 4. Token Index Spaces Are Mixed

Known index spaces:

- Stage One tokens: whitespace tokens, punctuation attached to words, e.g. `island,`.
- WFW atom context tokens: punctuation can be separate token, e.g. `island`, `,`.
- Stage Three proof spans: currently Stage One word-token indexing.
- Admin WFW role rows: word-only indices from WFW correction tokens.

Required fix:

Every span-bearing object should declare its span space in new artifacts.

Example:

```json
{
  "span": [5, 8],
  "span_space": "stage_one_words"
}
```

Compatibility:

- Old rows without `span_space` should be treated as `stage_one_words`.

Required conversion helpers:

- `stage_one_word_index -> wfw_word_index`
- `wfw_atom_token_index -> word_index`
- `word_index -> display token`

Acceptance:

- Mid-clue punctuation examples pass:
  - `NASCENT`: comma after `turn`
  - `MAUI`: comma after `island`
- Manual role for `one` never applies to `in`.
- Manual role for `by` never applies to `island`.

### 5. Definition Purpose Is Not Definition Evidence

Current behavior:

- Marking words as `definition` gives them a visual role.
- It does not prove `definition -> answer`.

Required fix:

Stage Three must distinguish:

- **definition purpose**: these words function as the definition span.
- **definition evidence**: DB or accepted evidence proves the phrase defines the answer.

For `MAUI`, manual definition span:

```text
one in Hawaii
```

requires evidence:

```text
one in Hawaii -> MAUI
```

Plan:

- Group consecutive manual `definition` roles into phrase spans.
- Add those spans as manual definition candidates.
- Run definition evidence check against the DB.
- If missing, emit a precise enrichment request:

```text
definition: one in Hawaii -> MAUI
```

Acceptance:

- If DB lacks `one in Hawaii -> MAUI`, only definition evidence remains failing.
- If DB contains `one in Hawaii -> MAUI`, definition check passes.

### 6. Structural Roles Should Satisfy Purpose Without DB

Current issue:

Structural words like `with`, `by`, `of`, `in` can be human-classified as links/joiners but still appear in missing-purpose checks.

Required rule:

Manual structural roles are purpose evidence.

Mapping:

```text
link -> structural_separator, status manual
surface -> surface_padding, status manual
charade_joiner -> structural_separator, status manual
indicator -> operation_indicator, status manual
*_indicator -> operation_indicator, status manual
definition -> definition_phrase_member, status manual
```

Acceptance:

- `with = link` clears purpose failure.
- `by = link` clears purpose failure.
- No DB fact is required for `link`, `surface`, or `charade_joiner`.

### 7. Source Roles Need Value Semantics

Current problem:

A reviewer can mark a source role without providing letters. That may look useful but cannot assemble an answer.

Required rule:

Source-like manual roles must have clear value behavior.

Categories:

- Source role with letters: usable assembly evidence.
- Source role without letters: purpose evidence only, not assembly evidence.
- Existing proof source block with value: manual role can reclassify purpose without changing value.

For `MAUI`:

```text
Graduate -> MA
uniform -> U
island -> I
```

These are assembly pieces because values exist.

Plan:

- Add validation:
  - source role + no value + no existing value = incomplete source evidence.
  - source role + value = candidate source evidence.
- UI should make missing source values visible for source roles.

Acceptance:

- A source role without letters does not falsely pass assembly.
- Existing source blocks keep their values unless a manual value intentionally overrides them.

### 8. Stored Proof Rows Differ From Fresh Rebuilds

Current issue:

Stored proof rows and live rebuilds have been observed to differ for clues such as `NASCENT` and `CHARLIE`.

Required fix:

The UI and tests must identify which artifact is being inspected.

Plan:

- On admin pages, show:
  - proof row id,
  - proof source,
  - created time,
  - whether proof came from stored Stage Two or fresh rebuild.
- Reverify output must report whether it wrote a new proof row.
- Tests must explicitly choose one of:
  - stored-row rendering test,
  - fresh rebuild test,
  - full pipeline test.

Acceptance:

- No test says "expected blocks" without saying stored proof or fresh rebuild.
- User can see whether the page is stale.

### 9. Tests Are Stale

Known example:

- `test_wfw_display_adapter.py` expects `TIJUANA` to be review.
- Current DB/code proves it.

Required fix:

Do not rely on stale tests as confidence signals.

Plan:

- Split tests into:
  - deterministic unit tests with synthetic proof JSON,
  - DB-dependent integration tests,
  - current corpus smoke tests.
- Update or quarantine DB-dependent assertions that changed because DB facts changed.
- Add explicit fixtures for bugs already found:
  - missing-word fallback,
  - punctuation index drift,
  - manual role overlay,
  - manual role proof consumption,
  - stale display warning.

Acceptance:

- A failing test points to a real contract violation, not a changed DB fact.
- `TIJUANA` expectation is updated or moved to a historical fixture.

### 10. `wfw_display_adapter.py` Is Untracked

Current issue:

An important active code file is untracked, so normal git review can miss changes.

Required fix:

Bring file ownership under control.

Plan:

- Confirm whether `signature_solver/wfw_display_adapter.py` should be tracked.
- If yes, add it deliberately in a reviewed change.
- If no, document why an active code file is untracked, because that is dangerous.

Acceptance:

- Important WFW display logic is visible in normal review.
- Future changes to display adapter are not hidden in untracked status noise.

### 11. Review Messages Are Not Updated From Manual Roles

Current issue:

Banner can say:

```text
These clue words need a clear purpose before publication: with, by, one, in
```

even after manual roles exist.

Required fix:

Review messages must be regenerated from the same proof data as display.

Plan:

- Do not patch messages directly.
- Regenerate Stage Three proof after consuming manual roles.
- `word_purpose_coverage` should then pass or list only truly unresolved words.

Acceptance:

- Manual `link` and `definition` roles disappear from missing-purpose message after reverify.
- If proof still fails, the remaining message is specific and true.

### 12. Admin UI Does Not Explain What Actions Mean

Current issue:

The UI implies dropdown edits repair WFW, but currently they do not complete proof repair.

Required fix:

Make admin UX honest.

Plan:

Add visible admin state indicators:

```text
Manual roles saved. Proof not yet regenerated.
```

After reverify:

```text
Proof regenerated from manual roles.
```

If definition DB fact is missing:

```text
Definition fact needed: one in Hawaii -> MAUI
```

Acceptance:

- User knows whether they are editing annotations, adding DB facts, or regenerating proof.
- There is no hidden "you did the work but verifier ignored it" state.

## Implementation Phases

### Phase 0: Stop The Bleeding

No solver changes.

Tasks:

- Add admin warning when manual roles exist newer than proof row.
- Show proof row id/source/time.
- Show whether manual roles were consumed by the proof.

Acceptance:

- User cannot mistake display overlay for verified proof.

### Phase 1: Manual Roles Into Reverify

Likely files involved:

- `web/routes/admin.py`
- `web/routes/clue.py`
- `sonnet_pipeline/word_roles_store.py`
- `signature_solver/stage_two_casefile.py`
- `signature_solver/stage_three_proof.py`

Tasks:

- Load manual roles for the clue.
- Pass manual roles into Stage Two casefile.
- Serialize manual roles into retained Stage Two JSON.
- Rebuild Stage Three using those roles.

Acceptance using `MAUI`:

- `with`, `by`, `one`, `in`, and `Hawaii` no longer appear as missing purpose after reverify.

### Phase 2: Manual Definition Spans

Tasks:

- Group consecutive manual `definition` roles.
- Create manual definition candidates.
- Check phrase against DB.
- Emit exact definition gap if missing.

Acceptance:

- `one in Hawaii -> MAUI` is the only definition evidence request if missing.
- Adding that DB fact and reverifying clears definition evidence.

### Phase 3: Source Value Rules

Tasks:

- Define which manual roles require letters.
- Treat existing source block values as usable.
- Flag manual source roles with no value as incomplete source evidence.

Acceptance:

- No source-looking tile silently fails assembly because a value is missing.
- Reviewer sees exactly which value is needed.

### Phase 4: Span Space Audit

Tasks:

- Audit all span-bearing fields in:
  - Stage Two casefile,
  - Stage Three proof,
  - WFW display adapter,
  - clue page manual role overlay,
  - admin reverify.
- Add `span_space` to new artifacts.
- Add conversion helpers.
- Add punctuation tests.

Acceptance:

- `MAUI`, `NASCENT`, and any clue with mid-clue punctuation remain stable.

### Phase 5: Remove Display Deception

Tasks:

- Either remove runtime overlay after proof consumption works,
- or keep it only as a clearly labelled "pending manual edits" state.

Acceptance:

- WFW display and WFW status cannot silently contradict each other.

### Phase 6: Regression Harness

Test matrix:

1. Stored proof rendering:
   - `NASCENT`
   - `CHARLIE`
   - `MAUI`
2. Manual role proof integration:
   - `with/by = link`
   - `one/in/Hawaii = definition`
3. Definition DB fact:
   - missing fact produces exact gap
   - accepted fact passes
4. Punctuation:
   - comma before manual roles
   - trailing punctuation
   - punctuation inside definition phrase
5. Stale proof:
   - manual role newer than proof row triggers warning
6. Source roles:
   - source with letters passes purpose
   - source without letters is incomplete source evidence

## MAUI End-State Contract

Given:

```text
Graduate with uniform by island, one in Hawaii
MAUI
```

Manual roles:

```text
Graduate = source, MA
with = link
uniform = source, U
by = link
island = source, I
one = definition
in = definition
Hawaii = definition
```

Expected after reverify with missing definition fact:

```text
MA + U + I = MAUI
word_purpose_coverage = PASS
definition_evidence = REVIEW
required enrichment = definition: one in Hawaii -> MAUI
status = wfw_review
```

Expected after adding DB fact `one in Hawaii -> MAUI`:

```text
MA + U + I = MAUI
word_purpose_coverage = PASS
definition_evidence = PASS
answer_assembly = PASS
status = wfw_proven
```

## What Must Not Happen

- Do not hide failed checks.
- Do not clear review messages just because tiles look good.
- Do not treat a definition role as a definition fact.
- Do not treat a source role without value as assembly evidence.
- Do not let punctuation shift manual roles.
- Do not make reverify ignore manual roles.
- Do not leave the user guessing whether they edited display or proof.

## Review Checklist Before Implementation

Before code changes, Claude/Codex should answer these questions:

1. Which route does clue-level `Re-verify` currently call?
2. Does it load `clue_word_roles`?
3. Does it rebuild Stage Two or only Stage Three?
4. Where should manual roles live in Stage Two JSON?
5. How will old Stage Two JSON without manual roles behave?
6. What is the exact `word_purposes` mapping for every manual role choice?
7. How are consecutive manual definition words grouped?
8. What DB table proves `one in Hawaii -> MAUI`?
9. What happens if the definition fact is absent?
10. Which tests prove punctuation does not shift roles?

## Bottom Line

Manual WFW roles must become verifier evidence. The current system saves them, but only the page renderer reliably listens. That is the root bug.

Until manual roles are consumed by reverify and Stage Three, the WFW UI is not a trustworthy review tool.
