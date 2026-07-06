# BUILD PLAN — Two-tier signature catalog (PASS-tier / PENDING-only) + triage integration

**Status:** DESIGN COMPLETE — all 6 policy knobs SETTLED (§9), NOT BUILT. Written 2026-07-05,
knobs settled 2026-07-06. Branch `redesign`.
**Owner rule:** Claude may build this ONLY on explicit per-step approval. Test between every step
(regression 0/0/0 on the corpus). No working engine is modified — every change is additive
(new column, loader filter, one new cascade stage, one guard). Honours CLAUDE.md Rule 1.

---

## 1. Purpose

Unlock the risky signature families (anagram / container / indirect-synonym) that are currently
"escalate — can't add mechanically", WITHOUT ever risking the one unforgivable outcome: a silent
false PASS.

The mechanism: signatures live in **two classes**.
- **PASS-tier** — low fabrication risk. Added after a light A/B; can produce a PASS. Because we
  do not re-run old puzzles, it immediately solves the clue that triggered its creation.
- **PENDING-only** — higher risk. A DIFFERENT class in the catalog. When it fires it can NEVER
  produce a PASS — only ever PENDING — so it always gets a (quick) human review. A fabrication
  therefore degrades from "catastrophic false pass" to "a plausible PENDING that gets rejected".

This is soundness **by construction**, not by gate: we do not try to pre-certify the risky ones
safe; we structurally forbid them from passing.

---

## 2. The safety guarantee (why it is airtight)

The cascade short-circuits on `status in ("pass","pending")` and returns at the site. So a cap
alone is not enough — a PENDING-only signature firing EARLY would still short-circuit and block a
later real PASS. The guarantee comes from **ordering**:

> PENDING-only signatures run in a NEW stage placed AFTER every PASS-capable engine. That stage is
> only reached when everything above already failed (the cascade returned on any earlier
> pass/pending). Therefore a PENDING-only signature can only ever convert **FAIL → PENDING**. It
> can never touch an existing pass, never pre-empt a better engine, and never mint a false pass.

Same "pass-invariant by construction" shape as the near-miss work.

---

## 3. Which class a signature goes into — the risk rubric

Assigned mechanically from the hand-solved mechanism (no case-by-case judgement):

- **L = indirection load** = number of slots valued by a *synonym lookup*. Anchored slots do NOT
  count: clue-literal letters, a fixed DB abbreviation, a selection of clue letters, or a
  deterministic curtail/behead of a literal.
- **G = free choice over indirect material** = TRUE iff the operation adds an
  arrangement/position/lookup degree of freedom that ranges over *synonym* material.
  Literal-fodder anagram → G=FALSE. Charade / reversal / homophone → G=FALSE.

Routing — **G-primary** (SETTLED 2026-07-06). Fabrication comes from operation *freedom* over
synonyms, not from synonym *count*; fixed-order exact reconstruction (charade/reversal/homophone)
is the safe workhorse, so G gates and L only caps:
- **PASS-tier** iff `G = FALSE` **and** `L ≤ 3` **and** NOT short-answer-amplified.
- **PENDING-only** iff `G = TRUE`, **or** `L ≥ 4`, **or** the short-answer amplifier fires.
- **Short-answer amplifier (SETTLED):** answer ≤ 4 letters AND `L ≥ 2` → force PENDING-only.
  (Single-synonym short answers stay PASS-tier — constrained. Can tighten later to add
  "answer = 3 AND L ≥ 1" if review surfaces length-3 single-synonym fabrications.)
- **Third triage outcome (not a catalog class):** a signature so loose it would fire constantly
  and fabricate en masse → **no-signature — escalate / hand-solve** (do not file it at all).

Worked examples (telegraph 31267): EASED (L=0) → PASS-tier. NAPOLEON (L=1,G=false) → PASS-tier.
INHERITED (L=0, literal fodder anagram) → PASS-tier. **SECT+ION charade (L=2, G=false) → PASS-tier**
(the correction from G-primary). PREY (L=1, roaming insertion over a synonym, G=true) →
PENDING-only.

---

## 4. Architecture findings (verified read-only, 2026-07-05)

- **Single final choke point:** `_finish(parse, name, …)` — `engine_registry.py:1505`. Every
  cascade site returns via `return _finish(…)` (pattern confirmed at `:626`, `:638`, …).
- **Tier is knowable there:** signature engines stamp `parse.template_id = template.id`
  (`charade_signature_engine.py:250`, `container_signature_engine.py:263`, …); base engines set
  `None`. Per-template signal, not per-engine.
- **Template injection point:** templates load once via `load_*_templates()`
  (`engine_registry.py:476-485`) into the `wiring` dict (`:541-550`); signature engines read
  `wiring.get("…_templates")`.
- **Schema has room + precedent:** `catalog_templates` already carries
  `active, origin, version, notes` (`catalog_creator.py:521-525`); idempotent
  `ALTER TABLE … ADD COLUMN` precedent in `_migrate_deletion_structure.py:33`; the loader reads
  optional columns only if present (`catalog_loader.py:64`).
- **Cap-alone caveat:** the cascade short-circuits on PENDING too, so ordering (§2) is mandatory —
  a cap without the new last-stage does NOT give the guarantee.

---

## 5. Build steps (ordered, additive, each independently testable)

**Dependency note:** DO NOT insert any `tier='pending'` template into the catalog until Steps 1–4
are complete. Until then, a pending template would be loaded by the main cascade (Step 3 not yet
filtering) and could pass uncapped.

### Step 1 — Schema migration: `tier` column
- New script `core/_migrate_signature_tier.py` mirroring `_migrate_deletion_structure.py`:
  idempotent `ALTER TABLE catalog_templates ADD COLUMN tier TEXT DEFAULT 'pass'` (guarded by a
  PRAGMA check). **Default `'pass'` → every existing template stays pass-capable: zero behaviour
  change.**
- Test: run migration; column exists; every existing row `tier='pass'`; loader still works.

### Step 2 — `Template` dataclass + loader filter
- Add `tier: str = 'pass'` to `Template` (`catalog_loader.py`); read the column only if present
  (graceful, like `assembly`/`structure`).
- Add `tier=None` param to `load_templates`; when set, `WHERE tier = ?`; when None, load all
  (current behaviour).
- Test: `load_templates()` returns all with a `.tier`; `load_templates(tier='pass')` excludes a
  hand-inserted `tier='pending'` row. No engine behaviour change yet.

### Step 3 — Main cascade loads PASS-tier only
- In `engine_registry.py:476-485`, pass `tier='pass'` through the `load_*_templates()` helpers
  (add a `tier` param to each helper forwarding to `load_templates`).
- Test: **corpus regression must be 0/0/0** — trivially true because every existing template is
  already `tier='pass'`. This is the key gate for Step 3.

### Step 4 — New final PENDING-only signature stage + cap
- Load pending templates into new wiring keys (e.g. `…_templates_pending` via
  `load_*_templates(tier='pending')`), or load them inside the new stage.
- Add ONE new stage in the cascade AFTER the last current engine and BEFORE the cryptic-definition
  fallback (~`engine_registry.py:1360`): run the existing signature engines against the pending
  templates; **force `status='pending'`** on any resulting parse and add a warning
  ("provisional — high-tier signature, needs human confirmation"). Reached only when everything
  above failed.
- Test: with NO pending templates → stage is a no-op → corpus 0/0/0. Then insert ONE pending
  template of a known shape → a previously-FAILING clue yields PENDING (not pass), renders its
  breakdown, and NO existing pass is converted.

### Step 5 — Defence-in-depth cap in `_finish` (recommended)
- Provide a `template_id → tier` map in `wiring` (built at load time). In `_finish`, if the
  parse's template is `tier='pending'` and `status=='pass'`, force `'pending'`. Guarantees the cap
  even if a pending template ever reaches the main cascade by mistake.
- Test: unit test — a pass parse stamped with a pending template_id comes out pending.

### Step 6 — Creation tooling sets the tier
- `catalog_creator.create_signature` gains a `tier` param (caller supplies it from the §3 rubric).
  So a triaged signature is filed into the right class from the start.
- Test: create one PASS-tier and one PENDING-only signature; verify the stored `tier`.

### Step 7 — Promotion (eligibility + one-click)
- **Eligibility (SETTLED):** a PENDING-only signature becomes eligible when it has **≥ 10
  confirmations, 0 rejections, across ≥ 3 distinct puzzles** (computed from the `signature_reviews`
  log, Step 8). Any rejection keeps it PENDING-only (still works, just always reviewed).
- **Promotion = one-click, human-committed** (never auto): `UPDATE catalog_templates SET
  tier='pass'`. The track record REPLACES the up-front faithfulness A/B (the fires WERE the
  inspection), BUT promotion moves the signature into the main pass-capable cascade (ordering
  change), so it **triggers the automated regression hunt** — promote only if 0 regressions.
- Build: an "eligible for promotion" query over `signature_reviews`; the one-click action; the
  regression-hunt call. No new catalog machinery — `active`/`version` already exist.

### Step 8 — Review surface + review-capture log (SETTLED: interim, dashboard later)
- **`signature_reviews` log (NEW, required):** `(template_id, clue_id, verdict, reviewed_at)`,
  written whenever a human confirms (→pass) or rejects (→fail/hand-solve) a PENDING-only fire.
  This is what makes Step-7 promotion counts computable — without it promotion is un-computable.
- **Interim surfaces (while testing):** (a) the per-puzzle **markdown triage summary** (my FAIL
  diagnosis); (b) the existing **PENDING clue page** — already renders the breakdown and
  pre-populates the hand-solver, so review is confirm-or-nudge. The clue-page confirm/reject writes
  a `signature_reviews` row.
- **Dashboard = LATER, one shot** (deliberately deferred so real use refines the requirements):
  consolidates per-puzzle triage panels + the pending-only review queue + the computed
  "eligible for promotion" list.

---

## 6. A/B / verification protocol per class (SETTLED 2026-07-06)

- **PASS-tier:** **automated** before/after A/B, reusing the existing ~1,600-clue regression
  harness (all currently-passing clues + a fail sample) — it checks regressions AND surfaces every
  clue the new signature fires on. **Gate: 0 regressions AND every new pass faithful.** Inspect
  **ALL** new passes (they are few — a spot-check could wave through the one fabrication, which is
  unforgivable). **Red-flag heuristic:** > ~10 new passes means the signature is too general for
  PASS-tier → re-examine the tier.
- **PENDING-only:** **no up-front A/B** — regression-proof by construction (§2, only touches
  otherwise-failing clues). Its faithful check is the per-fire human review it was always going to
  get; accumulated confirmations become the promotion evidence.

**Claude's role in the loop (SETTLED): REGRESSION HUNTER, not pass-striver.** The A/B is automated
and Claude analyses its output — but Claude's job is to find what the change *broke* (existing
passes dropping) and to flag any new pass that looks *fabricated* (suppressing a fabrication is the
opposite of striving for passes). Claude NEVER advocates for the green. The user commits.

**Wall consequence:** an A/B needs the signature to exist to measure "after", and Claude must not
add signatures. So filing + committing stay the user's step; Claude reports the spec + tier +
protocol and, post-run, hunts the diff for regressions/fabrications. (PENDING-only needs no A/B —
the user just files it at `tier='pending'`.)

---

## 7. Triage-process integration

For each clue classified **MISSING SIGNATURE** in the fixed triage pipeline:
1. Compute the class from the §3 rubric (mechanical): **PASS-tier**, **PENDING-only**, or
   **no-signature — escalate/hand-solve** (the noisy-fabricator outcome, §3).
2. Report the exact **shape** (operation, def_pos, slot roles + widths) + the **class**.
3. Route: PASS-tier → "propose as PASS-tier; user files → automated regression-hunt A/B → commit if
   clean"; PENDING-only → "propose as PENDING-only, safe by construction, human reviews each fire";
   escalate → "no safe signature — hand-solve / build an engine".
4. Claude commits/queues/adds NOTHING. The user files the signature (with the reported tier). One
   diagnostic pass, no loop.

This is what makes the "fear of signatures" tractable: the risky ones are no longer gated behind an
adversarial proof — they enter as PENDING-only, harmless by construction, and earn PASS status
through use.

---

## 8. Invariants (never violate)

- No working engine is modified. Additive only: one column, one loader filter, one new stage, one
  guard, one creation-tooling param.
- PENDING-only signatures run LAST (after all PASS-capable engines) — the ordering IS the safety.
- Claude never resolves, commits, adds signatures, re-solves to check, or chases pass rates.
- Regression 0/0/0 between every step.

---

## 9. Policy decisions — ALL SETTLED 2026-07-06

1. **Two catalog classes** — `PASS-tier` / `PENDING-only` (3-tier gauntlet dropped); plus a
   triage-level **"no-signature — escalate"** outcome for noisy fabricators (not a catalog class).
2. **G-primary rubric** — PASS-tier iff `G=false` AND `L ≤ 3` AND not short-amplified; else
   PENDING-only. SECT+ION (L=2, G=false) → PASS-tier.
3. **Short-answer amplifier** — `answer ≤ 4 AND L ≥ 2` → PENDING-only. May tighten later to add
   `answer = 3 AND L ≥ 1`.
4. **PASS-tier A/B** — automated, reuse ~1,600 harness; gate = 0 regressions + ALL new passes
   faithful (full inspection); > ~10 new passes = re-examine tier. Claude in the loop as a
   **regression hunter** (hunt damage, never advocate green); user commits.
5. **Promotion** — eligibility = ≥ 10 confirmations / 0 rejections / ≥ 3 distinct puzzles →
   one-click manual promotion + automated regression hunt on promote. Track record replaces the
   up-front faithfulness A/B.
6. **Review surface** — interim: markdown triage summary + existing PENDING clue page + a new
   `signature_reviews` capture log. Dashboard deferred to ONE shot once requirements are refined by
   real use.

---

## 10. Suggested build order when approved
Step 1 → 2 → 3 (regression gate) → 4 (the core) → 5 → 6 → 7/8 (process). The §3 rubric + §9
decisions are settled — mirror them into `nightly_triage_process.md` before filing any real
signature.
