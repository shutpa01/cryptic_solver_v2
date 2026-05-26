# Stage Three Recovery Handover - 2026-05-24

This handover is for a new thread after the previous thread lost trust and
failed to deliver a usable end-to-end Stage Three result.

## User Position

The user is rightly frustrated. Do not reassure with vague claims. Do not send
web previews. Do not invent routes. Do not ask the user to use admin-key URL
shortcuts as if they are the normal workflow.

The user wants the normal Cordelia app flow:

1. Start Flask normally.
2. Open the Cordelia home page normally.
3. Use the site navigation to go to Puzzles.
4. Choose Daily Mail Cryptic.
5. Open puzzle 17884 or another normal puzzle.
6. Inspect clue pages through the normal UI.

## Critical Communication Rule

Do not write standalone local route strings in chat if the client turns them
into unusable web previews. If a route must be discussed, describe it in plain
English or put it in prose only after confirming the user wants it.

## What Went Wrong

The previous assistant built internal Stage Three components but did not prove
the normal user-facing flow. It repeatedly confused internal tests, dev route
shortcuts, and the actual app workflow.

The user reported:

- They could not open the puzzle in their running app.
- The puzzle list showed zero definitions.
- This looked like another failed rewrite with no usable result.

The assistant then made the situation worse by:

- Sending route-like web previews after being told not to.
- Presenting an admin-key query parameter as if it were a proper route.
- Changing the visible puzzle list before first stabilising the existing app.

That visible puzzle-list change was reverted before this handover.

## Current Route/Test Reality

Flask test client checks, after the revert, reported:

- normal puzzles page: HTTP 200
- normal Daily Mail Cryptic list: HTTP 200
- normal Daily Mail puzzle 17884 page: HTTP 200
- clue WFW render contract: passed
- compileall over web, signature_solver, and scripts: passed

These are only local test-client checks. They do not prove the user's currently
running Flask process is healthy. The first job in the new thread is to verify
the real running app carefully.

## Database Reality

The app uses:

- `data/clues_master.db` for clues and app-facing clue records.
- `data/cryptic_new.db` for reference/enrichment data.

Daily Mail 17884 exists in `data/clues_master.db` with 32 clues.

For Daily Mail 17884, all 32 `clues.definition` fields are empty. This is why
the legacy puzzle list can show zero definitions. That does not necessarily
mean Stage Three has no candidate definitions; it means Stage Three is not
writing user-facing definitions into `clues.definition`.

This is an integration/design issue, not something to hand-wave away.

## Recent Stage Three Write History

The Stage Three write layer was run for Daily Mail 17884.

Run 7 wrote:

- 32 artifacts
- 34 review items
- 2 pending enrichments

One pending enrichment was bad:

- `Sunlit patio needing repair?` -> `STIPULATION`

The user correctly identified this as nonsense because the phrase is anagram
fodder plus indicator context, not a definition.

That bad pending row was deleted.

After code changes, Run 8 wrote:

- 32 artifacts
- 0 new review items, because existing duplicates were skipped
- 0 new pending enrichments, because the only remaining one already existed

The remaining pending enrichment for Daily Mail 17884 is:

- type: synonym
- word: `marine creature`
- letters: `WHALE`
- answer: `GREYWHALE`

Do not treat this as proof that Stage Three is good. It only means that one
specific bad enrichment class was patched.

## Important Files Created or Changed

Stage Two/Three internals:

- `signature_solver/stage_two_casefile.py`
- `signature_solver/stage_three_proof.py`
- `signature_solver/stage_three_review_queue.py`
- `signature_solver/stage_three_write_layer.py`
- `scripts/run_stage_three_puzzle.py`

Display/contract files involved:

- `signature_solver/wfw_display_adapter.py`
- `web/routes/clue.py`
- `web/templates/clue.html`
- `web/templates/partials/atomic_parse.html`
- `web/test_clue_wfw_render_contract.py`

Tests added/updated:

- `signature_solver/test_stage_two_casefile.py`
- `signature_solver/test_stage_three_proof.py`
- `signature_solver/test_stage_three_review_queue.py`
- `signature_solver/test_stage_three_write_layer.py`
- `scripts/test_stage_three_puzzle_runner.py`
- `signature_solver/test_wfw_display_adapter.py`
- `web/test_clue_wfw_render_contract.py`

Progress note:

- `documents/stage_three_dry_run_progress_2026-05-23.md`

Original Stage Two handover:

- `HANDOVER_STAGE_TWO_2026-05-23.md`

## What Was Reverted Immediately Before This Handover

The previous assistant had changed the puzzle list by removing the Definitions
column and adding an admin-only Stage Three column. That was reverted.

Reverted files:

- `web/models.py`
- `web/templates/puzzle_list.html`

Therefore the visible puzzle list should be back to its prior structure.

## Agreed Design Principles

Stage Two:

- Internal evidence package only.
- It must preserve atomic evidence and unresolved gaps.

Stage Three:

- Strong verifier and proof gate.
- Must not become a separate user-facing page.
- Must not output internal handover/debug material to the user.
- Must feed the agreed WFW display.
- User-facing format has already been agreed and should not be redesigned.

WFW display:

- All clue words must be visible.
- WFW section appears in the agreed format.
- Edit/admin section remains below it.
- Admin controls are not the primary user-facing explanation.

Enrichment:

- Claude, blogs, and human review must not override the result.
- They can only suggest missing evidence to feed earlier stages.
- There is no override except special clue types such as cryptic definitions
  and &lit, which are rare and require human judgement.

Verifier:

- Must prove that answer pieces come from genuine mechanisms.
- Sources, indicators, and synonyms/definitions must be DB-supported where
  appropriate.
- Assembly must be justified left to right or by explicit positional evidence.
- Once mechanical proof is strong, surface/link/definition-separator words can
  be treated more generously, but this should be grammar/POS informed, not a
  crude link-word list.

## Known Technical Concern

The Stage Three write artifact currently stores only a summary in
`atomic_parse_artifacts.wfw_json`, not necessarily the full Stage Three proof
shape needed by the WFW display adapter. However, clue pages in admin mode also
build Stage Three dynamically if no WFW proof display exists.

This needs careful verification before any claim that Stage Three is integrated.

## First Task For New Thread

Do not start by extending the solver.

Start with a recovery check:

1. Inspect current git status.
2. Confirm the Flask app starts.
3. Confirm the user's normal navigation path works in the running app.
4. Confirm Daily Mail Cryptic 17884 opens from the normal UI.
5. Confirm clue 1 across opens from the normal UI.
6. Confirm clue 1 across shows:
   - the agreed WFW section,
   - all six clue words,
   - the edit/admin section below it.
7. Only then inspect Stage Three value.

If any of these fail, fix that visible flow before touching solver logic.

## Suggested Opening For New Thread

We are recovering from a failed Stage Three integration thread. Please begin by
reading `HANDOVER_STAGE_TWO_2026-05-23.md` and
`HANDOVER_STAGE_THREE_RECOVERY_2026-05-24.md`.

Do not redesign the WFW output. The agreed clue-page format is fixed. First
verify the normal Cordelia app route through the UI, not dev shortcuts:
home page, Puzzles, Daily Mail Cryptic, puzzle 17884, clue 1 across.

The first deliverable is a factual status report:

- whether the app starts,
- whether the normal puzzle route opens,
- whether clue 1 across shows the agreed WFW section with all words,
- whether the edit/admin section is present below it,
- and what, if anything, is broken.

Only after that should you work on Stage Three proof/enrichment logic.
