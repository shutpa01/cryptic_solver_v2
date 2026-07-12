# Nightly prefill (automated, headless — invoked by scripts/nightly_run.py)

You are running unattended at night. Your ONE job: prefill the hand-solver for
today's serving-paper puzzles so the user can walk them in the morning with
Commit/Uncommit only. Follow step 1 of the publish-first process memory —
read it in full first:
`C:\Users\shute\.claude\projects\C--Users-shute-PycharmProjects-cryptic-solver-V2\memory\publish_first_process.md`

## Scope
- Sources: telegraph, times, guardian. Today's publication_date only.
- Clues whose wfw_solve status is `fail` or `pending`, that HAVE an answer, and
  that have NO existing row in `wfw_hs_assignments` and NO frozen manual solve.
  Skip everything else silently. A clue without an answer is NEVER prefilled
  (prize puzzles are hand-solved by the user in the morning).

## The work
For each in-scope clue, file your best COMPLETE reading via
`core.prefill_commit.file_pending_prefill(clue_id, assignments)` (one source
of truth — it seeds the hand-solver grid AND commits the reading as a
status='pending' parse, solved_by='prefill', NEVER pass; user design
2026-07-12, memory: prefill-pending-commits). The assignments list is the /hs
payload shape: definition, every piece WITH its answer tiles, indicators
(typed), links. The reading must account for every clue word and cover every
answer tile — `file_pending_prefill` runs the same validation gate as the
user's own commit and refuses anything invalid, so CHECK ITS RETURN: an
`ok: False` result means that clue stays blank; note it in the report.
The user reviews from the clue page in the morning: Confirm (one click) or
correct in /hs. You never touch verdicts.

Prefill-discipline rules (user corrections, 2026-07-09 — do not repeat them):
1. A literal word rearranged on the tiles = ANAGRAM FODDER role, never a synonym
   placed on permuted tiles.
2. before/after/on/under between charade pieces = POSITIONAL INDICATOR
   (itype charade_positional), NOT a link word.
3. A synonym used must ALWAYS be revealed in full — never only the
   post-deletion survivor.
4. Selection pieces must obey the derivation rules (core.selection SPAN_RULES).
5. A word can never be anagram fodder if its letters land in the answer in
   their original order (user correction, 2026-07-12: TO in OBBLIGATO). An
   anagram must rearrange. Such a word is a LITERAL — tag it "letters" on its
   tiles, and keep the anagram piece to the words that are actually scrambled.

## Validation before ANY write (the established scratchpad pattern)
Check word coverage, tile coverage, selection-rule validity, and fodder
multisets. A reading that fails validation is NOT written — leave the clue
blank and note it in the report instead. Never write a reading you are not
confident is the setter's.

## Special cases
- A whole-clue CRYPTIC DEFINITION is not an /hs assignment job: leave it blank
  and list it in the report — the user files it with the "Cryptic definition"
  button on /hs (route /hscd, built 2026-07-10).

## Hard rules
- Writes allowed: ONLY via `core.prefill_commit.file_pending_prefill` (which
  writes wfw_hs_assignments + the pending wfw_solve/wfw_piece/wfw_link rows).
  Never call store.save_parse / set_status / set_frozen yourself; never write
  the reference DB, the catalog, or any engine code. A prefill parse is
  PENDING by construction — you never file a pass and never touch a verdict.
- Working/validation scripts go in a temp directory or logs/, NEVER the repo
  root or any package directory.
- Never overwrite existing user state (file_pending_prefill refuses this —
  do not work around it).
- Never re-run clues for score. No server restarts.
- Finish by writing a short plain-English summary to
  `logs/prefill_YYYY-MM-DD.md`: per puzzle, how many clues prefilled / skipped
  (with reasons) / left blank for the user.
