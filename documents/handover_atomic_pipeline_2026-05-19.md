# Handover: Atomic Parser / Pipeline Prototype

Date: 2026-05-19

This handover is for continuing in a new thread. The key correction is that there are two separate surfaces:

- **Streamlit dashboard pipeline**: `dashboard/pages/pipeline.py`. This is the existing production runner the user means by “dashboard”.
- **Flask puzzle/clue pages**: `web/templates/puzzle.html`, `web/routes/admin.py`, `web/templates/clue.html`. These display puzzles/clues and admin clue actions. They are not the production pipeline runner.

Do not conflate them again.

## Standing Rules

- Do not touch `stages/`.
- Evidence-first, grounded in existing code/DB.
- Do not invent a parallel generic solver.
- Do not solve isolated clues unless they represent a wider parser issue.
- Atomic/GT2 work must preserve evidence, not discard it.
- Existing DB-backed generation remains primary.
- Existing grammar/catalog/verifier is no longer enough for the new atomic format; strict assembly plus review is the current gate.
- Do not rewrite or demote old manually verified scores.

## Current Architecture

The intended live flow is now:

1. Existing Streamlit dashboard runs `sonnet_pipeline.run`.
2. The pipeline writes normal clue/explanation/enrichment data as before.
3. If `--atomic` is supplied, `sonnet_pipeline.run` calls the atomic post-pass.
4. Atomic post-pass writes additive artifacts to `atomic_parse_artifacts`.
5. If a clue has no complete atomic WFW parse, it writes a row to `atomic_parse_review_items`.
6. Review items can be exported into the existing `pending_gaps_*.json` format for enrichment review.
7. The clue page renders the new atomic WFW view when an artifact contains WFW.

## Important Files

New/changed atomic core:

- `signature_solver/clue_context.py`
  - Stage-one canonical token/span context.
  - Includes `pos_spans` and `pos_model_status`.
  - Builds definition candidates and span annotations.

- `signature_solver/pos_span_model.py`
  - spaCy-backed POS span proposals.
  - Uses project venv model `en_core_web_sm`.

- `signature_solver/gt2_candidate_generator.py`
  - GT2 candidate/evidence bundles.
  - Used as evidence-preserving candidate generation, not replacement solving.

- `signature_solver/token_parse_assembler.py`
  - Strict atomic assembler.
  - Requires all clue words to be accounted for as definition/source/operation/link.
  - Handles many operation classes already: charade, anagram, hidden, container, container_charade, homophone, reversal, deletion, substitution, positional charade, DD.

- `signature_solver/wfw_formatter.py`
  - Converts token parses to WFW/display data.

- `signature_solver/atomic_parse_store.py`
  - Additive persistence.
  - Existing `atomic_parse_artifacts`.
  - New `atomic_parse_runs`.
  - New `atomic_parse_review_items`.

New scripts:

- `scripts/run_atomic_parse_puzzle.py`
  - Runs atomic post-pass for one puzzle.
  - Appends one artifact per clue.
  - Appends review rows for non-WFW clues.

- `scripts/export_atomic_review_gaps.py`
  - Exports open atomic review rows into the existing gap-review JSON format.
  - Starts rows as type `review`; user can edit to synonym/abbreviation/definition/indicator/homophone.

Pipeline/dashboard:

- `sonnet_pipeline/run.py`
  - Added `--atomic`.
  - After mode 1, calls atomic post-pass and export.

- `dashboard/pages/pipeline.py`
  - Correct dashboard surface.
  - Added `Atomic artifacts` checkbox to “Run by puzzle”.
  - Added `Atomic artifacts` checkbox to selected batch runs.
  - When checked, dashboard appends `--atomic` to the existing command.

Review tooling:

- `sonnet_pipeline/review_gaps.py`
  - Existing command-line enrichment review.
  - Now supports edit-before-accept.
  - Now supports `indicator` and `homophone` inserts as well as synonym/abbreviation/definition.

Flask display:

- `web/routes/clue.py`
  - Reads latest atomic artifact.
  - Sets `clue.atomic_wfw` if WFW exists.

- `web/templates/clue.html`
  - Chooses atomic display path when `clue.atomic_wfw` exists.

- `web/templates/partials/atomic_parse.html`
  - New atomic UI partial.

## Explicit Mistake And Correction

I incorrectly wired a `Run pipeline + atomic` button into the Flask puzzle page and `web/routes/admin.py`. This was wrong because the user’s “dashboard pipeline” is Streamlit, not Flask.

That mistaken Flask route/button was removed:

- Removed `/admin/run-pipeline-atomic/...`.
- Removed puzzle-page `Run pipeline + atomic` button.
- Restored Flask puzzle page behavior.

The correct wiring is now in `dashboard/pages/pipeline.py`.

## Environment

The `.env` file at:

`C:\Users\shute\PycharmProjects\cryptic_solver_V2\.env`

contained the Anthropic key on line 1 but had stray junk text lower down (`where we`) causing `python-dotenv` parse warnings. That junk was removed.

Verified:

```text
load_dotenv('.env') -> True
ANTHROPIC_API_KEY set -> True
```

The project venv was missing `anthropic`; it was installed:

```powershell
.\.venv\Scripts\python.exe -m pip install anthropic
```

## Current DB State

Atomic runs currently in `data/clues_master.db`:

```text
run_id=1 telegraph 31242 complete: 30 clues, 27 complete WFW, 3 review
run_id=2 telegraph 31243 complete: 32 clues, 12 complete WFW, 20 review
```

Important: run `2` was created during the confused dashboard attempt. Even though the web request timed out, the underlying command appears to have continued long enough to write atomic artifacts. However, no `pending_gaps_atomic_telegraph_31243_run2.json` file was found at handover time.

Existing exported atomic review file:

```text
documents/pending_gaps_atomic_telegraph_31242_run1.json
```

No 31243 exported atomic review file currently exists.

## Telegraph 31242 Status

Baseline:

- Old clean baseline was 23/30 assembled.
- Narrow FORAGE probe moved to 24/30.
- Atomic run now gives 27/30 complete WFW.

Remaining 31242 review items:

```text
RASCAL  solver_no_candidate
OWE     solver_no_candidate
IMAM    old_solved_no_atomic_parse
```

Those were useful architecture gaps, but the user does not want to keep focusing on that old puzzle.

## Telegraph 31243 Status

Atomic run `2`:

```text
Complete WFW: 12/32
Review queue: 20/32
```

Open review items for run `2`:

```text
1A  ORIGINATED   old_solved_no_atomic_parse | Started to grin, with idea developing
12A PRESENTS     missing_definition_span    | Squeeze sandwiches hospital department gives out
17A DECLARE      solver_no_candidate        | State education rejected by Bordeaux, nearly
19A NATIVES      solver_no_candidate        | Residents name a town in Cornwall? Not at first
21A RELEASE      solver_no_candidate        | Let off religious education student? No problem
27A OFFENSIVE    old_solved_no_atomic_parse | Loathsome, rotten, broody leader is dismissed
30A INTERESTED   missing_definition_span    | Fascinated teens tried dancing
1D  Oman         missing_definition_span    | Caesar perhaps heading off for monarchy in the Middle East
2D  IMPORTANT    old_solved_no_atomic_parse | Vital ship in an entrepot, finally
3D  IDEAS        old_solved_no_atomic_parse | Pyramid easily restricts views
4D  ABSENCE      solver_no_candidate        | Want stomach muscles, Lawrence? Not half!
8D  DISCOVERED   solver_no_candidate        | Found out Detective Inspector's hidden
9D  CRITICAL     old_solved_no_atomic_parse | Nit-picking of great consequence
14D French loaf  missing_definition_span    | Nice bread or flan? Chef's confused
16D INVITING     missing_definition_span    | Arousing Victor instead of Charlie is tempting
18D ABASEMENT    solver_no_candidate        | See Batman's awful humiliation
20D SESSION      solver_no_candidate        | Sounds picked up around end of business meeting
23D AWFUL        old_solved_no_atomic_parse | Just removing large pants
25D CLOSE        old_solved_no_atomic_parse | Wardrobe almost shut
26D STUD         old_solved_no_atomic_parse | Physically attractive male boss
```

## Commands

Run full pipeline from command line with atomic:

```powershell
.\.venv\Scripts\python.exe -m sonnet_pipeline.run 31243 --source telegraph --write-db --mode 1 --no-review --atomic
```

Run atomic post-pass only:

```powershell
.\.venv\Scripts\python.exe scripts\run_atomic_parse_puzzle.py --source telegraph --puzzle-number 31243
```

Export atomic review items for run 2:

```powershell
.\.venv\Scripts\python.exe scripts\export_atomic_review_gaps.py --run-id 2
```

Review exported gaps:

```powershell
.\.venv\Scripts\python.exe -m sonnet_pipeline.review_gaps documents\pending_gaps_atomic_telegraph_31243_run2.json
```

Verify key tests:

```powershell
.\.venv\Scripts\python.exe -m py_compile dashboard\pages\pipeline.py sonnet_pipeline\run.py sonnet_pipeline\review_gaps.py signature_solver\atomic_parse_store.py scripts\run_atomic_parse_puzzle.py scripts\export_atomic_review_gaps.py
.\.venv\Scripts\python.exe signature_solver\test_atomic_parse_store.py
.\.venv\Scripts\python.exe signature_solver\test_token_parse_assembler.py
```

## Dashboard Usage

Restart **Streamlit** after code changes to `dashboard/pages/pipeline.py`.

In Streamlit dashboard:

- Go to Pipeline Runner.
- Use the existing “Run by puzzle” or “Puzzles awaiting pipeline” workflow.
- Leave `Atomic artifacts` checked.
- Run as usual.

This uses the existing dashboard runner and simply appends `--atomic`.

Do **not** restart/use Flask for the dashboard pipeline runner. Flask is only relevant for clue/puzzle display pages.

## Next Best Step

Do this first in the new thread:

1. Confirm Streamlit sees the new `Atomic artifacts` checkbox.
2. Export run 2 review items:

   ```powershell
   .\.venv\Scripts\python.exe scripts\export_atomic_review_gaps.py --run-id 2
   ```

3. Use the existing review tool/dashboard review process to inspect those items with clue context.
4. Decide whether the review export format is good enough or whether Streamlit review page should read `atomic_parse_review_items` directly.

After that, refine the parser by bottleneck class, not by one-off clue:

- `missing_definition_span`
- `solver_no_candidate`
- `old_solved_no_atomic_parse`
- direction/grid-aware positional assembly
- explicit deletion / trim / substitution / near-complete review queue

## Known Risks / Cautions

- There are many unrelated dirty/untracked files in the repo. Do not revert broad changes.
- `data/cryptic_new.sqbpro` and `data/clues_master.db` may have local DB state changes from runs. Treat them carefully.
- The user is understandably frustrated by careless dashboard changes. Before modifying UI runner behavior, verify whether the surface is Streamlit or Flask.
- Do not add new buttons/routes to Flask for running the full pipeline unless explicitly asked.
- Avoid timeouts or background-process changes unless they match the existing dashboard pattern.

## Current Git Notes

Relevant intended new/changed files from this work:

```text
dashboard/pages/pipeline.py
sonnet_pipeline/run.py
sonnet_pipeline/review_gaps.py
signature_solver/atomic_parse_store.py
scripts/run_atomic_parse_puzzle.py
scripts/export_atomic_review_gaps.py
signature_solver/clue_context.py
signature_solver/pos_span_model.py
signature_solver/gt2_candidate_generator.py
signature_solver/token_parse_assembler.py
signature_solver/wfw_formatter.py
web/routes/clue.py
web/templates/clue.html
web/templates/partials/atomic_parse.html
```

Relevant accidental Flask runner changes were removed from:

```text
web/routes/admin.py
web/templates/puzzle.html
```

They may still show as modified because of earlier unrelated work in this repo; inspect diffs before touching further.

