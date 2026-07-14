# HANDOVER 2026-07-14 — API-credit leak found+fixed; /hs gains spoonerism pairs, double duty, DB-delete flow

Cold-start document for the next session. Read this first, then the memory
index. Plain English; verify against the code before acting.

## 1. Where the repo stands

- Branch `redesign`, working tree clean except `.claude/settings.local.json`
  (deliberately uncommitted) and four untracked scraper JSONs from the
  2026-07-14 nightly (normal). **13 commits ahead of origin, NOT pushed**
  (origin still at e7a9b72d, 2026-07-11). New since the 07-13 handover:
  - **04ab93b8** — billing: strip ANTHROPIC_API_KEY from headless claude env
    (see §2 — the expensive one).
  - **5c7c2ac8** — spoonerism pairs table + /hs spoonerism role (user design).
  - **08b7524a** — double-duty words: definition may overlap a wordplay role
    (user design).
  - **38e15b38** — standalone DB-delete flow on /hs (user design).
  - **ee296676** — widen a missed pair search (the one/is case).

## 2. THE BILLING LEAK (found 2026-07-13, fixed, verified)

**What happened:** the project .env holds a Console ANTHROPIC_API_KEY
(legitimately used by core/ai_piece.py, ai_synonym.py, ai_definition.py,
web/explainer.py). web/explainer.py load_dotenv() puts it into the WEB
SERVER's environment, so every claude -p the server spawned — every
Prefill-now button click — inherited it, and claude.exe prefers an API key
over the claude.ai Max login. Result: button prefills silently billed the
user's prepaid Console API credits (bought repeatedly) until the balance hit
zero at 14:41 on 07-13 ("Credit balance is too low", 400 billing_error).
Nightly (Task Scheduler, clean env) and shell runs were always fine.

**Fix (04ab93b8):** both claude -p call sites (scripts/run_prefill.py,
scripts/nightly_run.py run_claude) strip ANTHROPIC_API_KEY from the child
env — headless runs now ALWAYS bill the Max subscription. Verified by
running the full 4160 prefill with the key deliberately injected: ran clean
on the subscription. Do NOT delete the key from .env — other code needs it.

**Diagnostic lessons (memory: feedback_billing_error_is_a_stop):** read the
failed run's transcript jsonl in ~/.claude/projects/<project>/ FIRST (it
shows the exact API error + billing class); ripgrep skips gitignored files
so `rg ANTHROPIC` misses .env — use ls -Force; check the SPAWNING process's
environment, not just User/Machine env; a billing error is a STOP, never a
retry. I asserted three wrong billing theories before finding this —
violation log updated. The June-15 "Agent SDK billing split" was PAUSED by
Anthropic; claude -p bills the subscription when no key is present.

**User follow-ups possibly still open:** sizing the historical drain on
console.anthropic.com; the claude-fable-5[1m] model setting in global
settings (open Claude Code bugs 45390/64398/64970/65514 about 1M-context
demanding credits — not reproduced here, watch for it).

## 3. THE PUZZLES (state at handover)

- **guardian 4160 (Everyman): 28/28 pass**, 23 frozen — prefilled (20
  readings, report logs/prefill_ondemand_guardian_4160_20260713_1535.md),
  user-reviewed, including 8d THE YEAR DOT via the new spoonerism pair.
- **times 29592: 28/28 pass**, 25 frozen — user solved the prize grid, the
  nightly cascaded it, tricky 12a ISLE + 13d SOUNDBITE hand-committed
  (both exercised the new tooling; 13d = spoonerism pair BOUND SITE).
- **times 29593: 28/28 pass.**
- Today's dailies mid-review: telegraph 31291 (10 pass / 21 pending / 1
  fail), guardian 30058 (3 / 22 / 5). times 29594 scraped.
- spoonerisms table holds 2 vetted pairs: the dear yacht → theyeardot,
  bound site → soundbite.

## 4. NEW /hs TOOLING (all user-designed, all committed, restart to serve)

1. **Spoonerism pairs (5c7c2ac8)** — memory: spoonerism_pairs_table. A
   vetted source→answer pair in the spoonerisms table (cryptic_new.db, the
   homophones pattern) justifies a manual spoonerism piece covering the
   WHOLE answer (sound has no per-letter provenance). /hs role "spoonerism
   (source phrase)": tick source words, type the FULL phrase, Assign files
   the pair + auto-claims all tiles. Manual-only; engines untouched; a
   future cascade stage is possible if pairs accumulate.
2. **Double duty (08b7524a)** — memory: hs_double_duty_words. A definition
   may overlap wordplay roles (setters' boundary-word trick): assigning a
   definition KEEPS overlapping wordplay tags and vice versa; wordplay roles
   stay mutually exclusive with each other. Harvested definition includes
   the double-duty word. **THE GOTCHA (bit the user on 29592 12a):**
   assigning over a word NO LONGER clears its other tag — removal is ONLY
   the × in the tag list, clicking the piece's coloured tile (release), or
   role "none (clear)". Expect this to surprise again.
3. **DB-delete flow (38e15b38 + ee296676)** — memory: hs_db_delete_flow.
   "Delete a DB entry" panel on /hs: type word and/or partner, Search DB
   lists matching rows across ALL reference tables (synonyms, wordplay,
   indicators, definitions, homophones, spoonerisms, link words), click ×
   to delete (browser confirm; recoverable via deleted_entries; re-solves
   the clue). A missed pair search WIDENS: stem-related words first
   ("stored under a related word" — catches ones→is when you typed one/is),
   then partner-only, then word-only, each labelled. New:
   delete_homophone (both direction rows), delete_spoonerism. The old
   contextual "rogue? prune" links remain. NOTE: the "− delete" input in
   the assign bar is LETTER deletion (ORATION−O), not DB deletion — naming
   trap, twice now.

**Principle re-learned (12a/one=I):** deleting valid vocabulary to steer one
clue is the WRONG lever — the override for "right entry, wrong clue" is the
manual commit (frozen beats every engine). The delete flow is for pollution
only. ('ones'→'is' was deleted by the user on 07-14 — recoverable from
deleted_entries if it turns out legit elsewhere.)

## 5. OPEN ITEMS

1. **&lit commits land frozen-PENDING with NO confirm path** — /hsmanualcommit
   line ~4624 writes pending for andlit; /prefillconfirm refuses solved_by
   'manual'. The user hit this, then solved their clue another way and said
   "no changes required" — the structural gap remains if an &lit is ever the
   only honest reading. Two sketched fixes (user's own &lit commit = pass;
   or a confirm button for frozen-pending manual commits) — decision NOT
   taken.
2. **cp1252 logger crash** in scripts/run_prefill.py log() — a ⚠ in a
   message still kills the logger (masked the billing error twice). One-line
   fix (UTF-8 reconfigure), offered, not yet approved.
3. **Punctuation in filed phrases**: /hs Assign filed 'leap”'→BOUND (curly
   quote stuck to the word) on 29592 13d. Junk row still in synonyms_pairs
   unless the user deleted it (find via partner-only search: BOUND).
   Root fix = strip punctuation in _reusable_db_adds phrases — not built.
4. **From the 07-14 nightly diagnosis (memory entries, not this session):**
   Everyman 4160 commit-save anomaly (six commits never filed pieces to the
   reference DB — cause unknown); Guardian "See N Across" stubs never become
   continuation (span_join arithmetic never fires when the primary holds the
   full answer) — both need a look.
5. **Carried from 07-13 handover (relaunch, target Fri 2026-07-17):**
   internal-links sweep (approved, not started); prize-embargo serving
   decision; old-puzzle-page scope; styled 410; Everyman 4155-4159 backlog
   cascade+prefill decision.
6. 13 commits unpushed; push needs explicit user approval.

## 6. Environment notes

- Restart the dev server to serve the new /hs tooling
  (.venv\Scripts\python.exe web\run_dev.py — the V2 venv; the running
  instance was seen on the AI_Solver venv again, works but wrong).
- Headless billing is now guaranteed-subscription (§2). The claude.ai
  usage page is the only authoritative view of what any run billed. NEVER
  state a billing fact you cannot see; never retry a billable call after a
  billing error without the user's explicit yes.
- Old scheduled task "Cryptic Solver Nightly Pipeline" still needs deleting
  from an ADMIN shell:
  schtasks /Delete /TN "Cryptic Solver Nightly Pipeline" /F

## 7. Rules that bit today (full versions in CLAUDE.md + memory)

- Evidence before theory, HARD mode: three wrong billing explanations were
  each corrected by the user before the transcript+.env evidence settled it.
  The evidence was on disk from the start.
- rg respects .gitignore — a "searched the whole repo" claim that missed
  .env. Use Get-ChildItem -Force / ls -Force for dotfiles.
- Questions are not instructions; user-designed features (all four builds
  this session were the user's designs) beat my more complicated proposals.
- Deleting DB rows ≠ overriding the cascade: manual commit + freeze is the
  override. Don't send the user to the delete flow for a selection error.
- The user's time is the scarce resource: an hour lost on one clue (13d)
  to a seeded tile-less piece + a misleading error + my delete detour.
