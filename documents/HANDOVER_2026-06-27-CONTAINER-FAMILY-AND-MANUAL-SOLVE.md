# Handover — 2026-06-27 — container-with-built-inner family DONE; NEXT = manual-solve mode

Cold start for a NEW thread. The previous thread implemented most of the "container-with-a-
BUILT-inner" family from the 31275 plan, added a review-gate, and committed everything. The
NEXT job is: **review the MANUAL-SOLVE MODE design, agree it, then build it.** Do NOT start
coding the manual solver before the user has agreed the design.

## THE RULES (non-negotiable — unchanged; the project lost trust before by breaking them)
1. **ONE solver = the cascade.** The hand-solver only ASSIGNS ROLES (forced overrides into the
   cascade). NO parallel assembler. A previous thread built one ("the builder") and it was
   deleted; do NOT rebuild it under any name. **MANUAL-SOLVE MODE looks superficially like that
   builder but is its OPPOSITE — see the boundary section below; hold that line.**
2. **Bespoke engines per shape, each with its OWN verifier.** No central verifier, no compound
   engines. Additive only — NEVER edit a working engine to fix a case; add a new stage.
3. **DISCUSS each piece and get EXPLICIT approval BEFORE writing code.** The plan/handover is
   the agreed DESIGN, not a licence to build unsupervised. Loop: diagnose → propose → approve →
   implement + test.
4. **ALWAYS restart the server after a change and verify on the REAL page** (`127.0.0.1:5099`,
   not localhost). Start: `.venv/Scripts/python.exe -m core.wfw_web`. Check exactly ONE listener
   on 5099 (stale-server trap). The clue page renders the STORED parse, so after an engine
   change re-solve the clue (`POST /reload` with form `id=<id>&only=<id>`) to persist.
5. **Big A/B sweeps = OVERNIGHT only** (they block). Daytime = small fast checks. Harness:
   `core/_ab_general.py`. Every new engine must prove 0 regressions before it ships/pushes.
6. **FAIRNESS / derivation-depth line.** One op on a clue-anchored value is fair; a SECOND op on
   the first op's output is INVALID — mark such clues INVALID or route them to manual-solve; do
   NOT contort an engine to reach them.

## METHOD THAT WORKED THIS SESSION (keep doing it)
- **Audit the real code + DB BEFORE believing the plan.** The 31275 plan's premise ("the
  container family only wraps plain DB values") was FALSE — confirmed by reading all ten
  container engines. The user explicitly flagged the plan as under-researched; they were right.
- **DB-readiness gates the build.** For each target clue, verify EVERY piece + indicator is in
  the DB (use a throwaway check with `engine_registry.make_db_wiring()`) BEFORE coding — else a
  correct engine still fails and looks like a code bug. This caught PRISONER (needed DB adds)
  and saved hours.
- **Prove a solve via a ZERO-OVERRIDE diag** (`engine_registry.solve(ctx, wd)` with NO clue_id)
  — that is the honest "engine stands alone" proof, independent of any freeze/forced state.

## CODE STATE
- Branch `redesign`, **HEAD = 1e4da62c** (NOT pushed). On top of `5bee1ca4`, on `f49b25dd`.
- **Commit 5bee1ca4**: `container_inner_deletion_engine.py` (ASPIC/LIMBURGER) +
  `container_inner_alternation_engine.py` (PRISONER) + wiring + render registration.
- **Commit 1e4da62c**: `container_deletion_selection_engine.py` (RIVEN) + `review_gate.py` +
  wiring + a bespoke renderer + purple REVIEW verdict badge.
- All new engines: PASS-only, own `_verify` calling `role_validity`, wired in
  `engine_registry.solve` after `container_acrostic` (in this order: inner_deletion,
  inner_alternation, deletion_selection), each renders via `wfw_render` (the first two via the
  generic `_render_assembly`; RIVEN via bespoke `_render_container_built`).
- Untracked throwaway: `core/_diag_container_inner.py` (read-only diag over the 6 target clues).
- `core/atomsig/` is untracked + parked — ignore.

## WHAT WAS ACHIEVED (DT 31275 container-with-built-inner family) — 5/6 handled
| clue | id | answer | status | engine |
|---|---|---|---|---|
| LIMBURGER | 10075753 | LIMBER ∋ URGE-curtail | PASS | container_inner_deletion |
| ASPIC | 10075763 | AC ∋ SPIN-curtail | PASS | container_inner_deletion |
| PRISONER | 10075776 | PRIER ∋ SON(alt) | PASS | container_inner_alternation |
| RIVEN | 10075752 | RIEN(=FRIEND-ends) ∋ V(first) | **REVIEW-pending** (gated) | container_deletion_selection |
| SINEW | 10075780 | SEW ∋ (PAIN−PA) | NOT BUILT — **manual-solve** (indirect: raw "pain" minus a synonym; user called it "stupid, indirect nonsense") | — |
| DREDGE UP | 10075759 | DUP ∋ (RED+GE) | NOT BUILT — DB-blocked + indirect ("George vacuously"=GE) | — |

- All four built clues VERIFIED passing via the zero-override diag. On the PAGE they show a
  "FROZEN (forced pass)" badge because the user hand-solver-tested + froze them during the
  session (22 clues frozen total — cosmetic; the engines solve them cleanly regardless).
- The **review-gate** (RIVEN-only): a clean RIVEN pass is downgraded to a `pending` carrying a
  `REVIEW:`-prefixed warning (persists via store warnings), shown as a purple **⚑ REVIEW** badge,
  so a high-risk full solve always surfaces for human confirmation. Confirm via the existing
  Set-status → pass. Reusable for any future risky engine (`review_gate.gate(parse, label)`).

## DB ADDITIONS THE USER MADE (in clues_master.db — GITIGNORED, so NOT in any commit)
- PRISONER: "peeping Tom"→PRIER (DB had only PRYER), "at intervals"→alternation indicator,
  "filled with"→container indicator.
- RIVEN/LIMBURGER/ASPIC: needed none (already backed).
- SINEW (if ever engined): still NEEDS "stitch"→SEW (synonym) + "overlooked"→deletion indicator
  with subtype **general** (a named-substring removal — NOT a positional subtype). User decided
  SINEW is a hand-solve clue, so this was not added.

## OUTSTANDING / OPEN ITEMS
1. **Corpus A/B (0-regression) NOT yet run** — required before ANY push. Run overnight via
   `core/_ab_general.py`, covering all of today's engines (RIVEN especially — it is the riskiest;
   the review-gate means its passes surface as REVIEW, but it can still claim other clues as
   review-pending, so prove no regressions). The machine was OFF during the user's travel, so
   this could not run yet.
2. **RIVEN still FROZEN** on the page (user's test freeze). To SEE the REVIEW tag, "Unforce &
   re-solve" RIVEN (route `/unforce`, form `only=10075752`) — the user had not yet confirmed
   this when the session ended. Ask first.
3. **Nothing pushed.** Two local commits (5bee1ca4, 1e4da62c) on `redesign`.

## NEXT TASK — MANUAL-SOLVE MODE (review design, agree, THEN build)
Authoritative spec: **`documents/PLAN_31275_REMAINING_ENGINES_2026-06-26.md`** — the
"MANUAL-SOLVE MODE — spec" section at the bottom. Read it fully. Summary + the boundary:

It is a HUMAN authoring tool for the unfair/derivative clues the cascade can't/shouldn't solve
(SINEW, DREDGE UP, and any one-off past the fairness line). For each wordplay piece the human
ticks the word(s), sets role = synonym, and TYPES the piece's CONTRIBUTION TO THE ANSWER (the
letters AFTER the human applies the ops in their head — e.g. PALATIAL "Indian dish, not starter,
served up" → type ATIA, not RAITA). Tag the rest (definition/indicator/link/filler). Auto-colour
tiles by commit order. Commit → persist a FROZEN Parse built from exactly what was typed.
Uncommit → hand back to the cascade. Human sets the status.

**THE BOUNDARY (this is the one place a thread can slide back into the banned builder — hold it):**
- **Per-clue values ONLY — NEVER written to the reference DB** (suppress the synonym role's
  normal DB-enrichment write). The typed values are derivatives (ATIA), not real synonyms.
- **No derivation** — the system computes nothing; the human types every value.
- **No verification / no auto-pass** — the human sets the status by hand.
- **Never invoked by the cascade** — a committed clue is frozen so a cascade re-run never
  overwrites it.
- **Flagged "manual"** on the page so it is never mistaken for a derived solve.
Why it is NOT the deleted builder: the builder DERIVED values, SELF-VERIFIED, and ran SILENTLY in
the resolve path. Manual mode is human-typed, human-committed, human-statused, cascade-untouched,
clearly labelled. Build it SLOWLY, with the user steering each step. Smallest test cases first:
PALATIAL (3 clean pieces) then a 3-op one. Reuse the HS synonym role + add a manual-mode flag
that suppresses the DB write and tiles/colours by commit order; add Commit/Uncommit; render via
the existing source-colour path + a "manual" badge.

## MEMORY POINTERS
`container_inner_deletion_engine.md` (running state of this family), the 31275 plan (spec),
`per_type_render_and_new_engines.md`, the prior implementation handover
(`HANDOVER_2026-06-26-IMPLEMENTATION.md`).
