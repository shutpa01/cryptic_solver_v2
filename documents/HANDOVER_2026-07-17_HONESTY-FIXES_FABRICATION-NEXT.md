# HANDOVER 2026-07-17 (evening) — HONESTY FIXES DONE (local, undeployed); NEXT = FABRICATION AUDIT

Cold-start document. Read this, then MEMORY.md. Plain English; verify against the
code before acting. This session did four connected pieces of work, all LOCAL and
UNDEPLOYED, plus data fixes to today's clues. The next big subject is the FABRICATION
AUDIT — the prefill/cascade builder still manufactures wordplay the DB does not
support. That is the thing the user cares about most: **honesty over passes.**

---

## 0. ★ THE NEXT SUBJECT — the fabrication audit (NOT started)

The prefill/cascade **builder** invents pieces that are not sourced from the reference
DB, to reach the answer. Proven examples from today's puzzles:
- **take=R** (clue 10079981 RIA): the DB only has `take → T` (wordplay table, low
  confidence). The prefill asserted `take → R` (the Latin "recipe" device) — NOT in
  our data. Unsourced = fabrication (violates the abbreviation-strict rule).
- **main road=A1 placed on the IA tiles** (same clue): the value A1 cannot even spell
  the tiles it lands on.
- **Every homophone prefill** tagged the CLUE PHRASE as the spoken word instead of the
  real sound-alike word (single→SOUL skipping SOLE; delivery-of-half-dozen→OVA skipping
  OVER; committed-a-further-offence→RESCIND skipping re-sinned; etc.).

The gates built this session (see §1) STOP these being COMMITTED going forward and make
them fail honestly, but the BUILDER itself still produces them. The audit = find where
the builder gets unsourced letters (hardcoded lists? AI? fabrication?) and make it
FAIL honestly instead of manufacturing a piece. **A true FAIL beats a false solve.**
Do NOT "fix" this by adding the missing pieces to the DB to make clues pass — that is
chasing passes, the exact thing the project exists to refuse (the assistant slipped
into this once today by offering to add take=R; corrected).

---

## 1. WHAT WAS DONE THIS SESSION (all LOCAL / dev; NOT deployed)

### A. Full clue-type on EVERY user-facing place (memory: full_clue_type_labelling)
The clue-type label now names EVERY mechanism ("Container + charade + selection"), not
just the innermost atom, and reads IDENTICALLY on the clue page, the puzzle page, and
the admin badge. Verified: 0 mismatches across all 1763 passed clues; clue==puzzle==admin.
- Canonical deriver `_wordplay_label` (web/wfw_read.py) + `_manual_type_label` /
  `_engine_type_label` (core/wfw_render.py), sharing hand-synced helpers: `_note_mech`
  (aliases: insertion→container, first-letter→selection, deleted→deletion), `_note_mechs`,
  `_has_charade`, `_order_mechs`, `_placed_pieces`, plus `_CHARADE_SUPPRESS`,
  `_ATOMIC_OPS`, `_MECH_ORDER`.
- Rules that make it CORRECT: charade inferred by join-count over PLACED pieces (a
  removed/deleted source places nothing); `_CHARADE_SUPPRESS` (anagram/acrostic/
  alternation/spoonerism/homophone/hidden/palindrome/cycling/substitution/replacement)
  suppresses a FALSE charade; atomic ops (dd/cd/andlit/hidden/…) keep curated labels;
  composites always render in canonical order.
- web/routes/clue.py: REMOVED the old `_mechanism_label` override (the pre-canonical
  outlier that made the clue page disagree with the puzzle page). Clue page now uses
  `wfw_hint` → `_wordplay_label`.
- KNOWN LIMIT (deliberate): a homophone that is ONE piece of a charade (SOUL MAN =
  SOUL[homophone] + MAN) reads "Homophone", not "Charade + homophone", because
  `_CHARADE_SUPPRESS` drops charade when any gather-op is present. Consistent
  everywhere; revisit only if the user wants it.

### B. Homophones must go through the sound-alike word (memory: feedback_homophone_must_go_through_table)
Never tag a homophone unless the recorded SPOKEN word is a genuine homophone of the
answer letters. Tentative→approve gate (user's model: homophones are infinite, gate on
APPROVAL not pre-population):
- admin_db.py: `has_homophone(spoken, answer)`, `queue_homophone(...)` (writes a
  TENTATIVE pair to pending_enrichments, type='homophone').
- core/wfw_web.py `_build_manual_parse` homophone branch: captures the spoken word
  (new assign field `spoken`, blank ⇒ the clue word is it); gates on has_homophone; a
  user-named unsanctioned pair is queued tentative and ACCEPTED PROVISIONALLY
  (source='pending' → the existing "provisional" badge), sanctioned only when the human
  Approves. A blank spoken word whose clue word doesn't itself sound alike is refused.
- **Synonym-disguise guard** (same branch, synonym/substitution/letters/replacement):
  a letter-placing piece's tiles must be a letter-SUBSET of its value (identity/reversal/
  deletion preserve letters; a sound homophone does not). Catches "few will"=FEW WILL on
  FUEL. Blast test: 9 caught, ALL genuine disguised homophones, 0 valid solves broken.
- _enrich_row: Approve/Reject branch for homophone pending rows (Approve → _do_add →
  add_homophone). /hsaddhomophone now QUEUES tentative (was a direct write).
- JS homophone branch reads the add box into `a.spoken`.

### C. Resolve rebuilds from the human's assignment (memory: resolve_from_assignment_authoritative)
The engines must NEVER re-guess over a human hand-solver assignment.
- core/wfw_web.py `_resolve_from_assignment(clue_id)`, wired into `_render_one` and
  `_resolve_one`; the cascade runs ONLY when there is neither an assignment nor a
  committed frozen-manual pass.
  - confirmed PASS → untouched; assembles → persist as PENDING prefill (so a piece whose
    homophone pair was just Approved flips source 'pending'→'db' and Confirm can pass it);
    does NOT assemble (e.g. wordplay marked 'none' = unsolvable) → `store.delete_parse`
    clears the stale solve so no wrong answer shows; a stale freeze is lifted first.
- store.py: new `delete_parse(conn, clue_id)` (removes wfw_solve/piece/link; keeps
  assignment/notes/overrides).
- Fixes: Approve now upgrades provisional→solid; a re-run no longer clobbers a good
  reading with a worse engine guess (this had turned RONDEAU into a broken fail).

### D. Invalid clues never show the rejected parse (memory: resolve_from_assignment_authoritative)
The PUBLIC serving was already correct (stored_card guards status!='pass' →
_invalid_card serves answer + reviewer comment). The stale surface was the ADMIN review
card: core/wfw_render.render_parse rendered the full wrong breakdown with just an
INVALID badge. Fix: when `parse.status == 'invalid'`, suppress the breakdown ("Marked
INVALID — the stored wordplay is unsound, so it is not shown; see the comment") and
badge the type "UNSOUND". ONE place → covers all 12 invalid clues. Pending/pass
unaffected.

---

## 2. FILES CHANGED (local only)

- core/wfw_render.py — clue-type helpers + `_engine_type_label`; invalid-clue render.
- web/wfw_read.py — clue-type helpers + `_wordplay_label` engine branch.
- core/wfw_web.py — homophone gate + synonym-disguise guard in `_build_manual_parse`;
  `_resolve_from_assignment` wired into `_render_one`/`_resolve_one`; `_enrich_row`
  homophone branch; `/hsaddhomophone` → tentative; JS `a.spoken`.
- core/admin_db.py — `has_homophone`, `queue_homophone`.
- core/store.py — `delete_parse`.
- web/routes/clue.py — removed the `_mechanism_label` display override.

All py_compile-clean. Tested via direct function calls; the USER tested each change in
the dev browser (restart-and-look) as we went — the LAST batch (C, D, and the clue.py
part of A) landed after the user's most recent restart, so one more dev restart covers
them the same way.

## 2b. DATA CHANGES to clues_master.db (today's review)

- 6 homophone clues re-recorded honestly (transform → real spoken word; provisional
  unless the pair is sanctioned; `spoken` added to the draft payload):
  - 10079865 SOUL MAN → sole (PASS, user-committed) · 10079980 DESSERT WINE → whine
    (PASS, user-committed) · 10079853 BIOFUEL → few will (pending, solid) ·
    10079970 OVA → over (pending, provisional) · 10079973 RESCIND → re-sinned
    (pending, provisional) · 10079982 RONDEAU → dough (pending, provisional).
- 3 TENTATIVE homophone pairs queued in pending_enrichments awaiting the user's
  Approve: **over~ova, re-sinned~rescind, dough~deau**. (whine~wine, sole~soul,
  few-will~fuel were already sanctioned.)
- 10079981 RIA — cleared (delete_parse): unsolved, unfrozen, fabricated take=R/A1
  pieces gone. take=R stays a FAIL (not in the DB).
- cryptic_new.db (reference) UNCHANGED by this work (over~ova was added then removed
  during a test).

---

## 3. NOT DEPLOYED — how to go live (user drives)

Everything above is on the dev box only. To publish:
1. **Back up both droplet DBs first** (no undo on an overwrite):
   `ssh root@165.232.46.255 "cp /opt/cordelia/data/clues_master.db{,.bak-20260717} && cp /opt/cordelia/data/cryptic_new.db{,.bak-20260717}"` (verify the actual data path first).
2. Deploy the CODE via the dashboard **DEPLOY page** (the sanctioned path; the assistant
   cannot click it).
3. Upload the DBs via the DEPLOY page, then `systemctl restart cordelia` (prod caches
   templates + code — restart to pick up).
Note: uploading clues_master.db makes ALL of today's content live, including clues in
those puzzles that were NOT reviewed — confirm each puzzle is in the state you want
public before uploading (puzzle-level display rule: a puzzle shows only when ALL its
clues are served).

---

## 4. TO FINISH TODAY (none require fabrication)

- Restart dev; click through: Approve a tentative pair (over~ova / re-sinned~rescind /
  dough~deau) → the clue goes solid and Confirm-able; open an invalid clue (10079943
  CENTRE OF GRAVITY, 10079976 ABLE SEAMAN) → no wrong parse; a clue page vs its puzzle
  page → same clue-type.
- Approve or reject the 3 tentative pairs. RIA stays a fail.

---

## 5. PRINCIPLES RE-AFFIRMED (the session was hard on trust — read these)

- **Honesty over passes.** Never fabricate a piece to reach an answer. A true FAIL is
  worth more than a false solve. Do not add missing vocab to the DB to force a pass.
- **Don't fix clue-by-clue when it's a class.** The systemic fixes (C, D, A) address
  classes, not instances — do that.
- **"Real server" = the running site**, not a direct function call. But the user HAS
  been restarting dev and testing in the browser throughout — that IS real-server
  verification; don't over-lecture about it.
- Reproduce the real path before diagnosing; the user's stated facts are facts.
