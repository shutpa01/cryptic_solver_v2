# Handover — 2026-06-25 — Hand-solver: the ONE core problem to fix

Cold-start doc for the next thread. The previous thread built a lot but ended frustrated
because it patched symptoms for ~2 hours without fixing the core issue, and without
discussing before coding. **Read §1 and §7 before touching anything.**

Branch: `redesign`. Everything below is UNCOMMITTED working-tree changes (4 files:
`core/wfw_web.py`, `core/store.py`, `core/admin_db.py`, plus 2 new signatures in the
gitignored `clues_master.db`). Last commit: `f95d94b1` (the definition-floor work).

---

## 1. THE CORE PROBLEM (fix THIS, not the symptoms)

The hand-solver (`/hs` span-assignment grid) lets the user assign roles to clue words,
then click **Resolve**. **Resolve currently throws the assignments at the normal cascade
and lets it solve the clue FROM SCRATCH.** So:
- it IGNORES the user's interpretation and shows the cascade's own (original) parse —
  "it reverts to the original solve";
- it is SLOW on a clue that doesn't pass, because a failing free-solve walks every engine
  (measured ~25s on clue 10075566);
- "blank / clear a wrong role" is pointless while the solver re-decides everything;
- it can't "work out the fodder" because the free solver isn't using the OTHER assignments
  to deduce the leftover.

**Agreed direction (user, end of thread — CONFIRM before building):** Resolve should
BUILD the answer FROM the assignments, not re-solve. The user supplies the definition, the
indicator (=> the operation), and the pieces they know; the engine fills the ONE gap — the
leftover content words are the fodder — verifies it spells the answer, and shows THAT parse.
"Blank" then simply means "ignore this word." It is fast because it does not run the whole
cascade. This was proposed and the user was about to confirm when the thread ended — so the
FIRST action next thread is to confirm/refine this design WITH THE USER, then build it.

Worked example the user was testing — clue 10075566 "Boozer in Britain fuddled with ecstasy
twice ..." = INEBRIATE = anagram(BRITAIN + EE). Assignments: Boozer=def (already recognised),
fuddled=anagram indicator, "ecstasy twice"=EE (synonym), in/with=filler. The engine should
DERIVE that the leftover word "Britain" is the fodder, and that anagram(BRITAIN+EE)=INEBRIATE.
There is NO "fodder" role in the grid by design — the user does not want to tag fodder; the
engine must deduce it.

## 2. Two concrete asks still OUTSTANDING (the user repeated these)
- **BLANK / NONE role** — the grid dropdown only ADDS roles (definition/synonym/indicator/
  link/filler). The user needs a "none (clear)" option to clear a wrong role so a word is
  free. Requested multiple times, NOT delivered. (store has clear_forced_definition,
  clear_clue_filler, clear_forced_indicator(conn,clue_id,phrase) to build on.)
- **PREV / NEXT clue navigation in /hs** — the OLD role grid had prev/next arrows through
  the clutch; `/hs` lost them. The clutch is in the `from`/`back` id-string already.

## 3. What ACTUALLY works (verified) — protect it
- The vertical grid UI (`/hs?id=<clue>`): tick words → role → value (DB candidates) → Assign
  (in-memory) → Resolve. Persistence (save-on-assign + on-resolve, restore on load).
- Per-span DELETE of rogue DB rows (recoverable in `deleted_entries`).
- Whole-clue=definition → cryptic definition (pending).
- DELETION clues solve well, incl. signature creation: LORIS (10075533, positional delete)
  and AILMENT (10075561, NAMED delete DERAILMENT-DER) both PASS. The signature-creation +
  named-deletion (REM_F) machinery is sound for deletion.
- Speed of a PASSING resolve is ~0.6s (warm). The reload_wiring() bottleneck was removed
  from the hot path (uses incremental apply_add_to_wiring). FIRST action after a server
  start is a one-time ~15s wiring build.

## 4. What does NOT work
- ANY clue that doesn't solve cleanly from the committed pieces reverts to the cascade's
  free-solve (see §1). Anagram (esp. anagram-with-substitution like INEBRIATE) is not
  honored — the operation is not forced and the fodder is not derived.
- No blank role; no prev/next (§2). Failing-clue resolve is slow (~4-25s, full cascade).

## 5. State / how to run
- Server: `.venv/Scripts/python.exe -m core.wfw_web` (port 5099). Open `/?id=<space-joined
  clue ids>` to pick a clutch; each clue card has a teal "Hand-solver" link to `/hs` that
  carries the clutch as `from=`; the `/hs` "back to clue page" link returns to `/?id=<clutch>`.
- Routes added this redesign: /hs, /hslookup, /hsdelete, /hsinfer, /hssave, /hsresolve.
  Key fns in core/wfw_web.py: `_span_surface` (grid render), `hsresolve_route` (THE place to
  change for §1), `_cand_from_assignments` + `_try_create_signature` (deletion signature
  creation — the pattern to GENERALISE into the assignment-driven solve).
- Memory: `handsolver-redesign-direction` + `definition-floor-redesign` are current.

## 6. Architecture pointers for the §1 build
- `_cand_from_assignments(assigns, n_total, answer)` already maps assignments -> a catalog
  signature shape for DELETION (incl. named-delete REM_F via deletion.removed_runs). The
  assignment-driven SOLVE is the same idea extended: read def/indicator/pieces/fillers from
  the assignments, treat the UNASSIGNED content words as the fodder/operand, assemble per the
  operation, verify == answer, build the Parse directly (don't run the cascade). For anagram,
  reuse the anagram-substitution residual logic (memory: anagram_substitution_engine —
  "answer minus the raw bulk = what the held-out piece supplies").
- Operations to cover by indicator type: anagram (fodder = leftover raw + assigned pieces),
  deletion (already done via signatures), charade (pieces in order, leftover = ?), container,
  reversal. Start with anagram + the deletion path already working.

## 7. HOW TO WORK (the thread failed on this — read it)
- DISCUSS before building. The user said: "you just rush off doing something we have not
  discussed." Agree the design of each step, in plain English, BEFORE writing code.
- Do NOT patch symptoms. The §1 core problem is the only thing that matters; the blank role,
  fodder derivation, "reverts", and speed are ALL the same root cause.
- Verify on the REAL page; measure on 127.0.0.1; run the server as a tracked background
  process and restart it after edits (code is not hot-reloaded).
- Keep the working deletion path intact while building the assignment-driven solve.
