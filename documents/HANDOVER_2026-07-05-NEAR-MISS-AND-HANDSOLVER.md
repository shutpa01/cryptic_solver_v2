# HANDOVER 2026-07-05 — Near-miss fail reporting + hand-solver upgrades + triage-process design

**START HERE for the new thread.** Long session; a lot landed. Everything below is on branch
`redesign`, **HEAD = 527f1bac**, and **ALL of this session's work is UNCOMMITTED** (13 changed
`core/` files — see §7). Server: `python -m core.wfw_web` → http://127.0.0.1:5099/ (restart after
ANY code change). Git rule: **never commit/push without explicit approval.**

The memory index (MEMORY.md) has one-line pointers to every item below; the detailed memory
files are the source of truth. This doc is the cold-start map.

---

## TL;DR — what was delivered

1. **Near-miss fail reporting (the big one, REGRESSION-VALIDATED).** Failing clues now show what
   the engine HAD — the complete assembly it built and the exact words blocking it — instead of
   silently abstaining. DILEMMAS now renders `FAIL, all letters lit, [having, offering]
   outstanding`. Two changes; 8 engines converted; **0 pass regressions / 0 new passes / 0 errors**
   over 1,598 clues. [[near-miss-fail-reporting]]
2. **Hand-solver upgrades:** release-tile-to-re-edit; three new clue-type controls (DBE,
   positional, All-in-one/&lit); and **/hs now pre-populates editable roles from the stored parse**
   (so re-tagging one word no longer means rebuilding the whole clue).
3. **Process design (NOT built yet):** the nightly triage process — Claude DIAGNOSES ONLY, the
   user commits. Plus a per-puzzle dashboard design and the DT 31283 discovery findings.

---

## §1 Near-miss fail reporting  [[near-miss-fail-reporting]]  ← MOST IMPORTANT

**Why:** conservative compound engines built a COMPLETE assembly then `return None` at the
is_link residue gate the moment one word wasn't a known link — discarding the near-miss. And the
cascade's `_most_complete` best-fail selector only considered ~10 hardcoded engines.

**Change B** (`core/engine_registry.py` ~line 1389): the `_most_complete` candidate list now
includes **ALL** engines' fail parses (excludes `pcd` — that variable is REUSED for
cryptic_definition at ~:1363). Safe because the free-tiling guesser was removed long ago (~:1248),
so every remaining engine is answer/signature-driven and its fail is an honest partial.

**Change A** (8 PASS-only compound engines): stop discarding the near-miss — a non-link residue
word is left UNACCOUNTED (not `return None`), so the shared `_verify` NAMES it and marks a FAIL.
`solve_`/enumerator return the best near-miss (fewest unaccounted) via new
`core.engine_common.better_near_miss`. Engines converted:
- Uniform `_build` pattern: `reversed_outer_container`, `container_inner_deletion`,
  `container_inner_alternation`, `container_deletion_selection`.
- Bespoke (different residue logic): `reversal_deletion` (link_others), `reversal_container`
  (best near-miss PLACEMENT dict), `nested_container` (bounded near-miss search in deep nesting),
  `charade_container_selection` (a DFS — near-misses recorded via a **side-channel accumulator**
  `near=[None]`; `_finalize`'s return protocol kept byte-identical so a later passing tiling is
  never pre-empted → no false pass).

**Why it's safe (pass-invariant BY CONSTRUCTION):** the cascade only short-circuits on
`status in ("pass","pending")`, so a `return None`→`return fail` can NEVER turn a pass into a
fail; and the shared `_verify` fails on any unaccounted word, so no false pass. **Proven**, not
just argued — see §5.

## §2 Hand-solver: release tile  [[hand-solver-release-tile]]
Clicking a COLOURED (owned) answer tile on `/hs` now calls `releasePiece(k)` — pops that piece
back into edit (re-ticks its words, restores role/value/cut, tiles → blue selection) instead of
being locked. Was: only the `×` on the assignment tag could undo, and it deleted the whole piece.

## §3 Hand-solver: DBE / positional / All-in-one  [[handsolver-dbe-positional-aio]]
From the 31283 discovery. All done + verified:
- **DBE**: `definition by example` added to `_IND_TYPES` (wfw_web.py) — tag "perhaps"/"possibly"
  correctly instead of LINK. Available in /hs AND the clue-page /admin add (→ DB).
- **Positional**: `charade_positional` (after/before) added to `_IND_TYPES`/`_IND_SUBTYPES` +
  a subtype-required guard in `admin_db.add_indicator`.
- **All-in-one (&lit)**: a `&lit (all-in-one)` checkbox next to Commit on /hs. When ticked the
  commit auto-builds the whole-clue definition (words used TWICE), `operation="andlit"`, verdict
  PENDING (never auto-confirmed — set via `set_status` to dodge the frozen-downgrade guard,
  `store.py:132`). Renders "ALL-IN-ONE (&LIT)". Verified on WOMANISER.

## §4 Hand-solver: pre-populate roles from the parse  [[hs-prepopulate-from-parse]]
`/hs` now seeds EDITABLE assignments from the stored Parse (pieces+their answer tiles,
definition, indicators with recovered itype, links) instead of a blank grid.
`_assignments_from_parse` + `_itype_from_note` in wfw_web.py; seeds ONLY when there's no prior
hand-solve (in-progress work never overwritten). Verified across charade/container/anagram and
the DILEMMAS near-miss.

## §5 Validation / PUSH GATE
- Near-miss work: regression harness `_regr.py` (fresh non-persist solve over ALL cascade-passing
  clues + 500 fails = 1598 clues) + `_diff.py`, both in this session's SCRATCHPAD (ephemeral —
  the new thread must re-create them if needed; the corpus query is in the memory file). Final
  result: **0 regressions, 0 new passes, 0 errors; 26 fails now show a better engine.**
- **BEFORE PUSH/MERGE:** the standard **overnight full-corpus A/B** has NOT been run this session.
  Also DB-state (catalog sigs, literal_words, me/at deletions) is gitignored/local — not in any
  commit.
- Hand-solver items verified through the real /hs and /admin routes (shown, not paraphrased).

## §6 Design work — NOT built, decisions captured

### Nightly triage process  [[nightly-triage-process]]  ← read before doing any triage
Each night after scraper→solver, review FAIL/PENDING clues and classify WHY (missing DATA /
SIGNATURE / ENGINE, one+). **Hard rules (user-set, load-bearing):**
- **Claude DIAGNOSES and INFORMS ONLY — never resolves, commits, or iterates.** The user commits
  via the dashboard. The wall between diagnosis and resolution is what stops pass-rate chasing.
- **ONE diagnostic pass, no loop.** Do NOT re-run the solver to see if a fix "worked".
- **Signatures are NOT added by Claude** (earlier "add directly" was REVOKED) — report the exact
  signature needed; the user approves + commits.
- Missing DATA → queue to the dashboard enrichment queue (`pending_enrichments`). Missing ENGINE →
  escalate. **DO NOT CHASE PASS RATES**; "cannot parse" is an acceptable, honest outcome.
- Instrument: `python -m core.diagnose <clue_id>` (pieces view + per-engine view) — BUT note its
  engine view runs only a SUBSET (it did not run reversed_outer_container, which caused a
  misdiagnosis of DILEMMAS). The new near-miss reporting on the CLUE PAGE is now a better signal.

### Per-puzzle dashboard (design only)
Authoritative verdict = `wfw_solve.status` (pass/pending/fail) + `solved_by`, keyed by
`clue_id = clues.id`, in `data/clues_master.db`. Group by `clues.(source, puzzle_number)` (both
TEXT); `LEFT JOIN wfw_solve`. **`leftover.py` (shadow_blog_v0.db) and `pipeline.py`
(has_solution/structured_explanations) are LEGACY/prototype substrates — do NOT build on them.**
`review.py` enrichment queue is on the right DB but filters by source only (needs a per-puzzle
filter — `pending_enrichments.puzzle_number` is INTEGER vs `clues.puzzle_number` TEXT, cast on
join). User's envisioned loop: pick puzzle → see enrichment proposals → accept/reject → re-run
the puzzle (must call the REAL WFW solver, not pipeline.py's legacy one).

### DT 31283 discovery findings (7 fails)
Key lesson: after easy enrichment, the residual tail is **signatures/engines, NOT data**. And
**adding those signatures is NOT cheap** — the safe signature-creation tooling
(`_cand_from_assignments`, auto-discovery) is charade-family only; anagram/anagram-deletion/
container signatures can't be added mechanically (would need hand-authored catalog rows = risky).
Per-clue: PRINCECONSORT (anagram+DBE — signature), BELLINGHAM (anagram-deletion — signature),
DILEMMAS (reversed-outer container — turned out engine already exists; blocker was "offering" not
a link → the whole near-miss project), TRUNCATE/WOMANISER (harder/&lit), HASTE (indirect
synonym-deletion — deliberately unsolved), EPIC (unresolved). ALWAYS resolve a DB-checkable fact
with a lookup before reporting "possible" (user corrected me on this).

## §7 Files changed this session (all UNCOMMITTED, branch redesign)
- `core/engine_registry.py` — Change B (best-fail = all engines).
- `core/engine_common.py` — `better_near_miss` helper.
- 8 engines (Change A): `reversed_outer_container`, `container_inner_deletion`,
  `container_inner_alternation`, `container_deletion_selection`, `reversal_deletion`,
  `reversal_container`, `nested_container`, `charade_container_selection`.
- `core/wfw_render.py` — `"andlit"` label.
- `core/admin_db.py` — positional subtype guard in `add_indicator`.
- `core/wfw_web.py` — release-tile, DBE/positional/&lit clue types, HS pre-populate from parse.

## §8 Suggested next steps
1. **Commit this session's work** (get user approval; one logical commit or a few). Run the
   overnight full-corpus A/B first if pushing.
2. Build the per-puzzle triage dashboard (§6) on `wfw_solve`, honoring the DIAGNOSE-ONLY rules.
3. Run the near-miss-enriched discovery pass across more DB-backlog puzzles to get the
   reason-class distribution (data/signature/engine) — the real go-live metric.
4. (Deferred) anagram/container signature-creation tooling, so "missing signature" clues become
   addable safely (cascade-verified + A/B), rather than escalated.
