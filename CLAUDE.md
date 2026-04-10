# CLAUDE.md — Standing Instructions for Claude Code
# This file is read automatically at the start of every Claude Code session.

## CRITICAL RULES — NEVER VIOLATE

1. **NEVER modify working pipeline stage engines** (anything in stages/) to fix edge cases. Build new helper stages instead. This is the sacred principle — previous attempts to fix edge cases broke working solves elsewhere.
2. **NEVER delete files without explicit confirmation.** Always list what you plan to delete and wait for approval.
3. **NEVER run destructive commands** (rm, drop table, delete, overwrite) without showing the exact command and getting approval.
4. **NEVER make bulk changes across multiple files in one go.** One file at a time, test between each.

## QUALITY NOT SPEED — CRITICAL

The user is in pursuit of quality, not speed. Speed does not impress.
Rushing produces broken code that wastes the user's time and erodes trust.

**Never claim a fix is done until you have:**
1. Tested it through the actual code path the user will use (web UI route, not a standalone script)
2. Shown the actual output as proof (not a paraphrase, not "it should work")
3. Checked that existing working features still work

If you cannot test through the UI, say so explicitly. "I've tested this in
isolation but I haven't verified the web path" is honest. "It's done" is a
claim that requires proof.

**When implementing a feature that must work in multiple places** (e.g. both
per-clue Re-run and puzzle-level Re-verify), verify BOTH paths produce
identical results before claiming it's done.

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

## SIGNATURE SOLVER (S) — HOW IT WORKS AND WHERE IT BREAKS

The S solver is a zero-cost mechanical solver. Understanding its pipeline is critical
for debugging why clues fail. The chain is:

1. **Definition extraction** (`solver.py:extract_definition_candidates`) — tries 1-4 words
   from each end of the clue against `definition_answers_augmented` + `synonyms_pairs`
2. **Word analysis** (`word_analyzer.py:analyze_phrases`) — for each wordplay word, looks up
   all possible roles (SYN_F, ABR_F, REV_I, etc.) from RefDB
3. **Catalog matching** (`base_matcher.py:match_base`) — tries each catalog entry pattern
   (e.g. 3F+0I reversal_charade) against the words
4. **Placement** (`_place_spans`) — assigns words to slots, leaving gaps for indicators/links
5. **Slot lookup** (`matcher.py:_lookup_slot`) — for each F slot, finds possible values
   (synonyms, abbreviations) that appear in the answer (or reversed in answer)
6. **Combo verification** (`_verify_combo` → `_verify_reversal_combo`) — tries assembling
   the values to produce the answer, including permutations and reversals
7. **Confidence scoring** (`confidence.py:score_result`)

### Known failure points (debugged 2026-04-06):

- **Synonym cap**: `word_analyzer.py` caps long synonyms (>4 chars) at 20 per word.
  If the needed synonym is 47th in the list, S never sees it. Fix: also include
  any synonym whose reverse appears in the answer (bypasses the cap).
- **Indicator skipping**: When a catalog entry has 0 indicator slots (e.g. 3F+0I
  reversal_charade), `_place_spans` couldn't skip the indicator word because
  `assigned_ind` was None. Fix: set `assigned_ind` from `OPERATION_INDICATOR_TYPE`
  even when `n_indicator=0`.
- **Permutations in reversal_charade**: `_verify_reversal_combo` didn't try
  permutations of pieces, so (NIT,LOVER,G) never became (LOVER,NIT,G) which
  reverses to REVOLTING. Fix: add permutations (up to 4 pieces) matching charade verifier.

### Key files:
- `signature_solver/solver.py` — entry point, definition extraction
- `signature_solver/word_analyzer.py` — builds per-word role analysis
- `signature_solver/base_matcher.py` — placement + verification for base catalog
- `signature_solver/matcher.py` — `_lookup_slot`, `_verify_combo`, `_verify_reversal_combo`
- `signature_solver/base_catalog.py` — catalog entries + `OPERATION_INDICATOR_TYPE` map
- `signature_solver/confidence.py` — scoring
- `signature_solver/db.py` — RefDB (synonyms, abbreviations, indicators, definitions)
- `backfill_ai_exp/backfill_dd_hidden.py` — mechanical hidden word + DD solvers

### Debugging approach:
1. Check definition found: `extract_definition_candidates()`
2. Check word analysis: are the needed synonyms/abbreviations in `wa.roles[SYN_F]`?
3. Check placement: does `_place_spans` generate a placement leaving the indicator as leftover?
4. Check slot lookup: does `_lookup_slot` return the right values for each F slot?
5. Check combo: does `_verify_reversal_combo` / `_verify_combo` assemble the answer?
6. At each step, the answer is either "yes, move to next" or "no, this is the bug."

## SOLVING LEFTOVERS — "solve the DM/DT leftovers"

When the user says "solve the DM leftovers" or "solve the DT leftovers" (or names
any source/puzzle), follow this EXACT workflow:

### Prerequisites (user must have done these BEFORE asking you to solve)
1. The pipeline has already run for the puzzle
2. The user has already verified enrichments in the dashboard

### Steps
1. **Collect**: Run `python scripts/collect_for_review.py --source <source> --puzzle <number>`
2. **Read**: Read the generated file at `data/claude_review_YYYY-MM-DD.txt`
3. **Solve**: Work through every clue. For each one, determine the wordplay type,
   definition, and explanation.
4. **Write response**: Write solutions to `data/claude_review_YYYY-MM-DD_response.txt`
5. **Dry run first**: Run `python scripts/ingest_claude_review.py <response_file> --dry-run`
   to show verifier scores before committing
6. **Ingest**: Run without `--dry-run` to write to the database
7. **Report**: Tell the user the results — how many HIGH/MEDIUM/LOW/FAIL

### Explanation format (CRITICAL — must match verifier expectations)
Use UPPERCASE for letter pieces. Always end with `; definition: "..."`.
Piece sources: `(synonym="word")`, `(abbreviation="word")`, `(first letter of "word")`.

- **Charade**: `PIECE (synonym="x") + PIECE (abbreviation="y") = ANSWER; definition: "def"`
- **Anagram**: `anagram of FODDER + FODDER = ANSWER; definition: "def"`
- **Hidden**: `hidden in "teXT HEREin"; definition: "def"`
- **Double def**: `Double definition: meaning1 = ANSWER, meaning2 = ANSWER`
- **Container**: `OUTER (synonym="x") containing INNER (synonym="y") = ANSWER; definition: "def"`
- **Homophone**: `ANSWER sounds like "word" (indicator); definition: "def"`
- **Deletion**: `SOURCE (synonym="x") minus LETTER = ANSWER; definition: "def"`

### Quality standards
- Every explanation must be HONEST. If you can't parse a clue, say so — don't guess.
- The verifier will score independently. Do NOT try to game the format to get higher scores.
- FAILs and LOWs are fine — the user will review them. False HIGHs are not fine.
- Check letter counts: anagram fodder must have exactly the right number of letters.
- Check assembly: pieces must concatenate/combine to give the exact answer.
- **USE THE BLOG EXPLANATION** — the report includes blog explanations from TFTT/FifteenSquared
  when available. These are human-written breakdowns of the wordplay. Use them as your primary
  reference, then reformat into verifier-compatible format. Don't ignore this free information.

## GIT SAFETY

- Before starting any phase, suggest I commit current state as a checkpoint
- If something goes wrong, we can always `git checkout` to recover
- Never run `git push` without explicit approval
