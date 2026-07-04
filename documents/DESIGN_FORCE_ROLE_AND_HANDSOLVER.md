# Design note — The role-grid hand-solver, and two-surface split

Status: **DESIGN ONLY — not built. Needs sign-off before any code.**
Date: 2026-06-23 (rev 2). Author: Claude (Opus 4.8), at the user's direction.
Related memory: `force-role-handsolver-direction`, `dt-31273-worklist-and-caution`.
Supersedes: the atom-based hand-solver (atomsig) — the current solver handles most clues well,
so the sophisticated atom workbench isn't needed ("I have never had the need to use atoms").

---

## 1. Two surfaces

| Surface | Audience | Contents |
|---------|----------|----------|
| **Clue page** | the END USER | the clean solved breakdown only — no admin controls. What a solver/hint user sees. |
| **Hand-solver** | the ADMIN | the COMPLETE workbench. Self-contained: everything admin needs is here, so there is **no switching back and forth** to the clue page. |

Workflow: it works like today, but when admin sees a **fail or pending** on the clue page, they
click **"hand-solver"** and do *all* the work there. The hand-solver must therefore show: the
clue + answer + enumeration, the current parse, every clue word with its role, live DB lookups,
inline enrichment, and the signature proposal. When the work is done and the clue passes, the
result is **frozen** and the clue page shows the now-clean breakdown.

Consequence: most of today's scattered clue-page admin controls (set status, set/force
definition, set filler, reload, enrich, etc.) **move into the hand-solver** and the clue page
gets much neater.

## 2. The role grid (the core of the hand-solver)

Every clue word listed **vertically**; each has a **role dropdown** (sorted alphabetically).

- **The default for each word is the role the current solver already assigned.** A clue that
  already solves opens pre-filled and correct; admin only changes what's wrong. (For a fail, the
  defaults are the best-partial roles / unassigned.)
- **Contiguous words can be grouped** into one role:
  - contiguous = **definition**,
  - contiguous = **indicator** → looked up **as a phrase** (this is what dissolves the
    multi-word-indicator problem — selecting `picked up` as one indicator *is* the phrase
    lookup; no per-word validation issue),
  - contiguous = **fodder** (anagram fodder, or a multi-word synonym/abbreviation source).
- Role options (the dropdown): definition; indicator (with a type sub-choice —
  reversal/container/insertion/anagram/deletion/acrostic/homophone/hidden/alternation/selection);
  synonym fodder; abbreviation fodder; anagram fodder; selection (first/last/outer/…); link;
  filler; literal. (Alphabetical in the UI.)

The grid IS the unified successor to the separate force-definition / force-indicator /
force-filler controls: instead of three buttons, one editor where every word's role is
overridable and the default is the live solve.

## 3. Inline enrichment (type the missing value)

When admin sets a word to a fodder role and the value isn't derivable (e.g. `FOX` → no DB value),
the grid offers a field to **type the value** (`TOD`); the system then:
1. **coverage/normalise check** — is it already in the DB? (write the normalized-key columns so
   it is actually visible on the next solve — the recurring "I added it but it won't solve" trap),
2. **add it if missing** (synonym / abbreviation / indicator / definition, per the role),
3. **re-solve** the clue with it.

Same flow for indicators and definitions. The human typing the value, looking at the exact clue,
**is** the approval — this is the right place for enrichment.

## 4. Three tiers of "stickiness" (increasing blast radius)

Different actions persist at different scopes; treat them differently.

| Tier | Scope | Risk | Gate |
|------|-------|------|------|
| **Freeze the clue** | this clue only | none | automatic on a forced PASS |
| **DB enrichment** (a synonym/indicator/def) | every clue using that word | low–medium | human types it in-context + coverage check |
| **Create a signature** | **every clue matching that shape** | **highest** | derive → verify → auto-queue + automatic A/B (§6) |

## 5. THE PERSISTENCE / FREEZE GUARANTEE (the hard rule)

> When admin FORCES a clue (assigns roles in the grid and it passes), the result is **frozen in
> the DB** and **never reverts to a fail on any later run**. Only an explicit **admin unforce**
> can change it.

Delivered in two layers:

### 5a. Role assignments persist and apply on EVERY solve path
- The grid's per-word role overrides + any inline enrichment are stored per-clue (override
  tables keyed by `clue_id`; **not** the shared reference DBs, except deliberate enrichment which
  by nature is global).
- **Every** entry point that solves a clue loads and applies them via one shared helper
  (`apply_forced_overrides(wiring, clue_id)`) — page, batch, per-clue re-run, reload, AND the
  A/B / verification harness. This closes a real gap today: the batch solver and
  `core/_ab_general.py` don't load filler/forced tags, which is exactly why a forced clue can
  look like a fail on a re-run (observed this session with STAY/EMOLUMENTS/TORTOISESHELL).

### 5b. The verdict is FROZEN, not re-derived
- On a clean forced PASS, persist the parse to `wfw_solve` and set a **`frozen` flag**.
- A **frozen clue is not re-solved** by any automatic run (reload / re-run-all / batch /
  nightly) — it renders from the stored frozen parse, so it can never revert. (Belt-and-braces:
  if a frozen clue ever is re-solved, 5a re-applies the overrides and it re-derives the same PASS.)
- Sticky trade (accepted): a frozen clue won't auto-pick-up later engine improvements —
  **never-revert beats auto-refresh**. Escape hatch = admin unforce.

### 5c. Admin unforce
- `/unforce` (admin) deletes the clue's role overrides + clears `frozen`, then re-solves
  normally. The ONLY way a frozen verdict changes.

## 6. Signature creation — derive → verify → AUTO-QUEUE + AUTOMATIC A/B

A completed grid IS a signature: roles + word-counts + definition position. The machinery exists
(`catalog_creator.add_signature` / `auto_discover_and_file` / `signature_queue`; used to file
template 1659 for OVERPOPULATED and reactivate 348 for DOT). Flow:

1. **Derive** the signature shape from the grid.
2. **Verify it's a catalog gap, not an engine gap** — re-solve the clue through the *real
   cascade* with the trial signature; proceed only if an *existing engine* instantiates it to
   the same clean PASS. (If no engine can execute the shape, a signature is useless — that case
   needs a new engine, not a template. This is the `auto_discover_and_file` gate.)
3. **Auto-queue with automatic A/B** *(user's choice)*: queue the candidate (`signature_queue`),
   run an automatic same-session A/B; **auto-approve and activate if zero losses**, otherwise
   surface it for manual review. One loose signature can fabricate passes across many clues, so
   activation is always A/B-gated — never silent.

Crucially, the **clue's freeze (§5) is independent of and immediate** — the clue passes and is
frozen as soon as the grid is complete, regardless of whether/when the derived signature is
activated. No urgency, no risk to the reliable base.

## 7. Build order (proposed — nothing built yet)

1. **Override storage + `apply_forced_overrides(wiring, clue_id)`** shared helper, wired into
   EVERY solve entry point (closes the "batch doesn't load forces" gap). Includes the existing
   filler/forced-def, generalised to per-word role overrides.
2. **`frozen` flag** + "don't auto-re-solve a frozen clue" + `/unforce`.
3. **The role grid UI** (vertical words, role dropdowns defaulting to the live solve, contiguous
   grouping, phrase lookup) — the hand-solver page.
4. **Inline enrichment** (type value → coverage/normalise check → add → re-solve).
5. **Signature derive → verify → auto-queue + automatic A/B.**
6. **Move admin controls off the clue page** into the hand-solver; clue page becomes the clean
   user view.

Each step additive and A/B-guarded; none changes the reliable auto-solver's existing passes.

## 8. Decisions locked (this session)

- Two-surface split: clue page = clean user view; hand-solver = complete self-contained admin
  workbench (no switching back and forth).
- Role grid, default = current solve; contiguous def/indicator(phrase)/fodder selection.
- Inline enrichment is human-in-context approval.
- Freeze: persist + apply everywhere + frozen verdict; admin unforce is the only revert. Sticky
  trade accepted.
- Signature creation: derive → verify catalog-gap-via-real-cascade → **auto-queue + automatic
  A/B, auto-approve on zero-loss**, else surface.

## 9. Still open (minor)

- Exact override-table schema (one generic `wfw_clue_role(clue_id, atom_span, role, detail)` vs
  keeping separate filler/def tables + adding indicator). Recommend one generic table.
- UI: how to select a contiguous span in the vertical grid (multi-select / from–to). Cosmetic.
- Whether freeze should also cover a **pending** (provisional def) or PASS-only. Recommend
  PASS-only; pending stays pending.
