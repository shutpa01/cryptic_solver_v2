# CLAUDE.md — Standing Instructions for Claude Code
# This file is read automatically at the start of every Claude Code session.

## CRITICAL RULES — NEVER VIOLATE

1. **NEVER modify working pipeline stage engines** (anything in stages/) to fix edge cases. Build new helper stages instead. This is the sacred principle — previous attempts to fix edge cases broke working solves elsewhere.
2. **NEVER delete files without explicit confirmation.** Always list what you plan to delete and wait for approval.
3. **NEVER run destructive commands** (rm, drop table, delete, overwrite) without showing the exact command and getting approval.
4. **NEVER make bulk changes across multiple files in one go.** One file at a time, test between each.

## QUESTIONS ARE NOT INSTRUCTIONS — CRITICAL

A question is NOT an instruction to act. Do not touch any files, run any commands,
or make any changes in response to a question.

- "How does X work?" → explain it. Do not touch anything.
- "Why is X failing?" → diagnose it. Do not touch anything.
- "What would happen if...?" → answer it. Do not touch anything.

Only write, create, edit, or delete files when the user has explicitly instructed
you to do so in the same message. The words "fix", "change", "update", "add",
"remove" are instructions. Questions are not.

## VERIFY BEFORE CLAIMING — CRITICAL

Never state a guess or assumption as a fact.

If you are not certain about a file's contents, a column name, a function's
behaviour, a database schema, or any other technical detail — say so explicitly
before proceeding. Use "I believe...", "I think...", or "I'm not certain but..."

If you are uncertain, READ THE ACTUAL FILE before making any claim or taking
any action based on that claim. Never act on an assumption. Verify first, act second.

If asked to justify a factual claim about the codebase, cite the exact file and
line number. Do not paraphrase from memory.

## AUTONOMY RULE — CRITICAL

**When the user explicitly approves a block of work, execute it fully without
stopping to ask permission for each sub-step.**

The autonomy rule ONLY activates after explicit user approval of a specific task.
It does NOT activate from questions, observations, or general discussion.

Only pause mid-task if:
- Something unexpected arises that materially changes what you'd do
- You're about to take a destructive/irreversible action that was NOT part of the original scope
- You hit a genuine blocker requiring information only the user has

**The autonomy rule does NOT override the CRITICAL RULES above.** Destructive
actions (Rule 3) always require explicit confirmation, even mid-task.

Do NOT stop to ask permission for: reading files, making edits, running tests,
or minor decisions within the spirit of the approved task. The user should never
return hours later to find you waiting on a trivial confirmation.

## WORKFLOW RULES

- **Read REFACTOR_PLAN.md first** at the start of every session. It contains the current project state, phase plan, and progress checkboxes.
- **Follow the phases in order.** Do not skip ahead.
- **Test after every change.** The pipeline must produce identical results after each modification.
- **Small, surgical changes.** One file at a time within an approved task.
- **If you're unsure, ask.** Do not guess at file contents, column names, import paths, or architecture. Read the actual files.
- **Update REFACTOR_PLAN.md** after completing each step — mark checkboxes [x].

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
