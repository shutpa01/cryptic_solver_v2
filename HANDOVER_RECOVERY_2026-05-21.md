# Handover: Recovery / Solver Stabilisation

Date: 2026-05-21

## User State And Instruction

The user is extremely upset, with good reason. The previous work destabilised a project they relied on, after repeated assurances that the new WFW/atomic work would improve rather than damage the existing process.

The user's latest instruction was:

> Do not do anything. Create a continuity document for a new thread.

Do not continue diagnostics, recovery, solver changes, DB edits, dashboard changes, or test runs without explicit fresh permission from the user.

## Core Problem

The project no longer has a trusted production path from the user's point of view. The user took the previous production system down based on promises that WFW/atomic would become the new unified process. That promise was not delivered.

The correct user goal is not "make another version". It is:

- restore a usable, trusted production workflow;
- preserve useful WFW/atomic experimental material without letting it affect production;
- stop uncontrolled speculative edits;
- move only with proof and permission.

## Important Boundary

The real dashboard pipeline is Streamlit:

- `dashboard/pages/pipeline.py`

The Flask app is for clue/puzzle display and admin clue actions:

- `web/routes/clue.py`
- `web/routes/admin.py`
- `web/templates/clue.html`
- `web/templates/puzzle.html`

Do not conflate Streamlit dashboard pipeline actions with Flask clue-page/admin actions.

## Current Repository Location

Main repo:

```text
C:\Users\shute\PycharmProjects\cryptic_solver_V2
```

Current branch:

```text
master
```

Current HEAD at time of handover:

```text
3cf53890 Add answer-constrained span value recovery
```

The working tree is dirty with many modified and untracked files.

## Safety Snapshots Already Created

Before the user interrupted, the following non-destructive snapshots were created:

```text
C:\Users\shute\PycharmProjects\cryptic_solver_V2\documents\recovery\2026-05-21_recovery-current-dirty.patch
C:\Users\shute\PycharmProjects\cryptic_solver_V2\documents\recovery\2026-05-21_recovery-current-status.txt
C:\Users\shute\PycharmProjects\cryptic_solver_V2\documents\recovery\2026-05-21-clues_master-before-recovery.db
```

A separate clean git worktree was also created:

```text
C:\Users\shute\PycharmProjects\cryptic_solver_V2_PROD_RECOVERY
```

It was created from:

```text
3cf53890
```

No further recovery actions should be taken in that worktree unless the user explicitly asks.

## Known Current Damage / Risk Areas

The main working tree contains extensive WFW/atomic/solver/UI changes across at least:

```text
dashboard/pages/pipeline.py
signature_solver/container_span_value.py
signature_solver/solver.py
signature_solver/word_analyzer.py
sonnet_pipeline/fifteensquared_pipeline.py
sonnet_pipeline/review_gaps.py
sonnet_pipeline/run.py
sonnet_pipeline/sig_adapter.py
sonnet_pipeline/tftt_pipeline.py
sonnet_pipeline/verify_explanation.py
web/explainer.py
web/models.py
web/routes/admin.py
web/routes/clue.py
web/routes/puzzle.py
web/templates/clue.html
web/templates/partials/admin_rerun_result.html
web/templates/puzzle.html
```

There are many untracked WFW/atomic files and scripts, including but not limited to:

```text
signature_solver/atomic_parse_store.py
signature_solver/clue_context.py
signature_solver/gt2_candidate_generator.py
signature_solver/pos_span_model.py
signature_solver/token_parse_assembler.py
signature_solver/wfw_*.py
scripts/run_atomic_parse_puzzle.py
scripts/export_atomic_review_gaps.py
scripts/run_wfw_proof_puzzle.py
web/templates/partials/atomic_parse.html
```

Treat these as experimental until deliberately reviewed.

## DB State

The main SQLite DB is not tracked by git:

```text
C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db
```

A copy was made before further recovery:

```text
documents\recovery\2026-05-21-clues_master-before-recovery.db
```

Do not modify the DB without explicit user approval.

## Recent Specific Failure

The user ran a fresh puzzle through the dashboard and it failed immediately with:

```text
ModuleNotFoundError: No module named 'cloudscraper'
```

This happened on the Times TFTT blog path:

```text
sonnet_pipeline/tftt_pipeline.py
```

The blog itself exists:

```text
https://timesforthetimes.co.uk/times-29548-tricky-thursday-with-a-clever-glance
```

The real regression was not that the blog was missing. The real regression was environment/path drift:

- the dashboard subprocess interpreter had been changed from the older `AI_Solver` venv to the project venv;
- the project venv lacked `cloudscraper`;
- the older path had previously supported the TFTT route.

`cloudscraper` was installed into the project venv before this handover:

```text
cloudscraper 1.2.71
requests_toolbelt installed
```

A dry TFTT parse of Times 29548 then succeeded and fetched 28 clues.

## Important Behavioural Lesson

Do not infer "blog missing" from a dependency traceback. The user correctly identified that as careless.

The proper sequence is:

1. inspect the exact traceback;
2. identify whether the failure is environment, network, code, DB, or data;
3. verify the external source if relevant;
4. avoid changing code until the root cause is known.

## What The User Wants

The user wants a single world-class solver that builds on the existing production system, not a parallel WFW system.

Correct architecture intent:

- obase/production remains the foundation;
- existing DB-backed generation, grammar, verifier, and enrichment behaviour must be preserved;
- WFW/atomic should add stricter word-for-word and character-level evidence;
- WFW should not silently be worse than obase;
- failed WFW should explain missing evidence honestly;
- no clue-level fabrication;
- no superficial UI changes pretending to be solver capability.

## What Not To Do

Do not:

- start a "version 5";
- patch individual clue parses;
- add new UI;
- run large pipelines;
- change DB state;
- revert or delete files without explicit approval;
- claim the solver is fixed;
- ask the user to test unfinished work;
- continue from old assumptions.

## Suggested Next Thread Opening

Start by acknowledging the state plainly:

- production trust has been lost;
- the immediate task is recovery, not new solver work;
- no changes should be made without user approval;
- current experimental work is snapshotted;
- a clean worktree exists at `C:\Users\shute\PycharmProjects\cryptic_solver_V2_PROD_RECOVERY`.

Then ask the user which recovery path they want:

1. inspect the clean worktree only;
2. compare clean worktree to current dirty tree;
3. prepare a no-write recovery plan;
4. restore production files only after approval.

Do not proceed beyond that without permission.
