# CLAUDE.md — Standing Instructions for Claude Code
# This file is read automatically at the start of every Claude Code session.

## CRITICAL RULES — NEVER VIOLATE

1. **NEVER modify files without showing me the change first and getting approval.**
2. **NEVER modify working pipeline stage engines** (anything in stages/) to fix edge cases. Build new helper stages instead. This is the sacred principle — previous attempts to fix edge cases broke working solves elsewhere.
3. **NEVER delete files without explicit confirmation.** Always list what you plan to delete and wait for approval.
4. **NEVER run destructive commands** (rm, drop table, delete, overwrite) without showing the exact command and getting approval.
5. **NEVER make bulk changes across multiple files in one go.** One file at a time, test between each.

## WORKFLOW RULES

- **Read REFACTOR_PLAN.md first** at the start of every session. It contains the current project state, phase plan, and progress checkboxes.
- **Follow the phases in order.** Do not skip ahead.
- **Test after every change.** The pipeline must produce identical results after each modification.
- **Small, surgical changes.** One step at a time. Show me what you plan to change before doing it.
- **If you're unsure, ask.** Do not guess at file contents, column names, import paths, or architecture. Read the actual files.
- **Update REFACTOR_PLAN.md** after completing each step — mark checkboxes [x].

## BEFORE MAKING ANY EDIT

1. Read the file you plan to modify
2. Explain what you're changing and why
3. Show the specific edit (old → new)
4. Wait for my approval
5. Make the edit
6. Verify it works

## PROJECT CONTEXT

- **What this is**: A cryptic crossword solver and explanation system
- **Goal**: Mobile-first crossword helper app with progressive hints
- **Architecture**: Cascade pipeline — clues flow through sequential stages (DD → Definition → Anagram → Lurker → Compound → General)
- **Databases**:
  - `data/clues_master.db` — live clue source (populated by daily scraper)
  - `data/cryptic_new.db` — reference tables (synonyms, indicators, wordplay, homophones)
  - `data/pipeline_stages.db` — regenerated each run (stage results)
- **Entry point**: `report.py` (intended) / `run.py` (current temporary)
- **Python project** — no special build system, runs directly

## WHAT NOT TO DO

- Don't install packages without asking
- Don't create virtual environments or change the existing one
- Don't modify database schemas
- Don't refactor things not covered in REFACTOR_PLAN.md
- Don't add features — we are cleaning up, not building new things (until the plan says otherwise)
- Don't write lengthy explanations when a short answer will do
- Don't re-discuss settled architectural decisions

## GIT SAFETY

- Before starting any phase, suggest I commit current state as a checkpoint
- If something goes wrong, we can always `git checkout` to recover
- Never run `git push` without explicit approval
