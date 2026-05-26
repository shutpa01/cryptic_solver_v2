# Codex Unsupervised Working Rules

Date: 2026-05-17

These rules define the operating envelope for long GT V2 / WFW preservation R&D sessions where Codex may need to work without repeated approval prompts.

## Scope

Codex may work unattended only on design, research, and report-generation tasks for GT V2 clue anatomy and WFW preservation.

Allowed write areas:

- `documents/`
- R&D-only scripts in `scripts/`

Allowed generated outputs:

- Markdown research reports in `documents/`
- JSONL research slices in `documents/`
- Design/runbook documents in `documents/`

## Explicitly Out Of Scope

Codex must not modify these without fresh explicit permission:

- `stages/`
- solver engine behaviour
- database contents
- scraper state
- web app behaviour
- production verifier behaviour
- existing user-authored design documents except by direct request

Codex must not run destructive git commands.

No unattended use of:

- `git reset`
- `git checkout`
- file deletion commands
- database mutation commands
- broad arbitrary Python execution

## Approved Command Principle

Persistent approvals should be narrow command prefixes, not broad tools.

Good approval shape:

```powershell
C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe scripts\build_block_graph_candidates.py
```

Bad approval shape:

```powershell
python
```

## R&D Commands To Approve

These are safe repeatable R&D artifact builders/analyzers under the current envelope:

```powershell
C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe scripts\build_operation_attachment_slice.py
C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe scripts\build_attachment_inspection_queue.py
C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe scripts\build_block_graph_candidates.py
C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe scripts\analyze_block_graph_candidates.py
C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe scripts\build_block_type_inspection_report.py
C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe scripts\analyze_definition_gaps.py
```

## Behavioural Rules

Codex should carry on autonomously inside this envelope.

If a task requires leaving the envelope, Codex must stop and explain why.

If a branch reaches diminishing returns, Codex should say so and stop polishing.

If a command approval is needed, Codex should request it with the narrowest useful prefix rule.

## Current Stop Point

The supervised graph-artifact mining branch is at diminishing returns.

Next useful work:

- design GT V2 candidate generation for fresh clues
- define how block candidates are proposed without structured human explanations
- use the supervised artifacts as design evidence rather than continuing to polish them
