# Handover — Phase 2 Pipeline Quality — 2026-05-25

## Critical process rules for this thread (enforce from the first message)

1. Never spawn the Codex agent without the user explicitly saying "send to Codex."
   The process is: Claude writes the instruction document → user reviews it →
   user sends it to Codex. Claude violated this once this session.
2. No backticks in chat responses. Plain text only.
3. Do not write code. Codex writes code. Claude plans, designs, checks, audits.
4. A question is not an instruction to touch files.
5. Verify before claiming. Read actual files before making any factual assertion.


---

## What Phase 2 is

A set of targeted fixes to improve solver output quality and display correctness.
The new pipeline files (stage_two_casefile.py, stage_three_proof.py,
wfw_display_adapter.py and related) are complete enough to run but have not yet
been committed to master — they are all untracked (??).

Grammar_triage.py is tracked and modified in the master working tree.
It also has a separate set of changes in worktree agent-ac562c46d52c5dbec.


---

## Instruction documents and their status

All documents are in the documents/ folder.

### PHASE_2_ANAGRAM_STAGE_THREE_CODEX_INSTRUCTION.md
Covers _assembly_check (sorted-letter comparison for anagram) and _atomic_links
(pool-based letter matching for anagram) in signature_solver/stage_three_proof.py.
Status: written. Whether Codex implemented this is not confirmed in this session.
The stage_three_proof.py file exists and is untracked — check whether these
functions already contain the anagram handling before sending this instruction.

### PHASE_2_ANAGRAM_SOLVER_FIX_CODEX_INSTRUCTION.md
Covers _build_anagram_result (word_overrides parameter, ABR_F/SYN_F tagging)
and _try_anagram substitution section (phrase substitution priority 2, single-word
priority 1, two-word priority 0) in signature_solver/grammar_triage.py.
Status: sent to Codex and implemented (process violation in previous context).

KNOWN BUG: the phrase substitution is attaching "IC" as an unrelated piece in at
least one test clue. The user said: "it attached an unrelated piece, IC that is
not part of the anagram." The root cause was not diagnosed before the session
ended — the user was asked for the clue text and actual word_roles output but the
session ended before that was provided.

The next thread must NOT touch grammar_triage.py until the root cause is clear.
Get the clue text and actual word_roles output from the user first.

### PHASE_2_ANAGRAM_DISPLAY_ROLES_CODEX_INSTRUCTION.md
Covers _normalise_anagram_display_roles in signature_solver/wfw_display_adapter.py.
Normalises SOURCE_BLOCK piece_N roles to anagram_fodder for anagram proofs so all
fodder tiles show the same colour. Also updates answer_links by span.
Status: written, and then implemented by Codex as a side-effect of a process
violation (Claude sent the display roles instruction without user approval).
The function _normalise_anagram_display_roles is already present in
wfw_display_adapter.py at line 80, and is called from display_from_stage_three_proof.
The user has not explicitly reviewed or approved this implementation.
Before considering this done, ask the user to audit the function body against the
instruction document.

### PHASE_2_DISPLAY_DEDUP_CODEX_INSTRUCTION.md
Covers _dedupe_stage_three_display_blocks in signature_solver/wfw_display_adapter.py.
Removes duplicate display tiles for the same token span (e.g. "Retired" appearing
as both "Reversal indicator" and "Needs WFW role").
Status: written this session. Not yet reviewed by user. Not yet sent to Codex.
This is the natural next instruction to send once the user has reviewed it.

Priority list (with LINK_BLOCK at 30 included):
  DEF_BLOCK: 100
  SOURCE_BLOCK role=source_review: 90
  SOURCE_BLOCK: 80
  OP_BLOCK: 70
  role ending _indicator: 70
  review_indicator_candidate: 60
  review_separator_candidate: 50
  LINK_BLOCK: 30
  other: 20
  REVIEW_BLOCK / unaccounted: 10

Call site: after _normalise_anagram_display_roles, before operation_detail.


---

## Worktree state

Three worktrees exist. Two are locked.

worktree agent-a0506936
  Branch: worktree-agent-a0506936
  Modified: .claude/settings.local.json and CLAUDE.md only.
  Not relevant to Phase 2.

worktree agent-ac562c46d52c5dbec (locked)
  Branch: worktree-agent-ac562c46d52c5dbec
  Modified: signature_solver/grammar_triage.py
  This is the anagram solver fix. Contains the phrase substitution code that
  has the known IC bug. Do not merge until the bug is diagnosed and fixed.

worktree agent-a52bf5fe4648df601 (locked)
  Branch: worktree-agent-a52bf5fe4648df601
  No relevant code changes (only settings.local.json).
  The wfw_display_adapter.py changes from the process-violation Codex run are NOT
  in this worktree — they ended up in the main working tree as untracked files.

Master working tree (untracked, not committed):
  signature_solver/wfw_display_adapter.py — contains _normalise_anagram_display_roles
  signature_solver/stage_three_proof.py
  signature_solver/stage_two_casefile.py
  signature_solver/wfw_atoms.py and many other new pipeline files
  All new, all untracked.

Master working tree (modified, tracked):
  signature_solver/grammar_triage.py — contains the anagram solver fix including
    the buggy phrase substitution. Modified in both the main working tree and
    the locked worktree. The main working tree version is what the pipeline runs.


---

## Known bugs / open issues

1. Grammar_triage.py phrase substitution bug
   Symptom: IC is attached as an unrelated piece in an anagram parse.
   Diagnosis needed: get the clue text and the actual word_roles output.
   Do not redesign or touch grammar_triage.py before the root cause is established.

2. review_separator_candidate label
   _review_block_hint returns "Separator?" for structural_separator_candidate
   and "Def separator?" for definition_separator_candidate.
   The user noted this should perhaps say "Joiner/Separator" for display
   consistency. This is a small follow-up change to _review_block_hint in
   wfw_display_adapter.py, separate from the dedup fix. It was explicitly noted
   as out of scope for the dedup instruction.

3. _normalise_anagram_display_roles not formally user-approved
   The function exists in wfw_display_adapter.py but the user did not review
   the implementation before it was written. The instruction document
   PHASE_2_ANAGRAM_DISPLAY_ROLES_CODEX_INSTRUCTION.md describes the intended
   behaviour. The new thread should ask the user to verify the implementation
   matches the instruction before relying on it.


---

## What the next thread should do first

1. Ask the user: for the grammar_triage.py IC bug, what is the clue text and
   what word_roles list did the pipeline produce? Do not touch grammar_triage.py
   until this is known.

2. Once the root cause is clear, design a corrected version of the phrase
   substitution and write a new instruction document for the user to review
   before sending to Codex.

3. Separately: ask the user whether they want to review and send
   PHASE_2_DISPLAY_DEDUP_CODEX_INSTRUCTION.md to Codex. This fix is independent
   of the grammar_triage.py bug and can proceed in parallel.

4. Once the IC bug is fixed and the dedup is in, verify UNDEMOCRATIC
   (clue id 10069315) and STIPULATION (clue id 10068568) through the actual
   pipeline to confirm the anagram fixes work end to end.
