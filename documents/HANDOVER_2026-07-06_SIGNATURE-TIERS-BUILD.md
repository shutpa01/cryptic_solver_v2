# HANDOVER 2026-07-06 — Build the two-tier signature catalog (PASS-tier / PENDING-only)

**START HERE for the new (build) thread.** The design is complete and settled; this thread's job
is to IMPLEMENT it. Everything you need is in the build plan — this doc is the cold-start map.

## The one document to read first
**`documents/BUILD_PLAN_2026-07-05_SIGNATURE-TIERS.md`** — the full spec. It has: the safety
argument (§2), the risk rubric (§3), the verified architecture choke-points with file:line (§4),
the ordered build steps with per-step tests (§5), the A/B protocol (§6), triage integration (§7),
invariants (§8), and all 6 settled policy decisions (§9). Do not re-derive any of it — it's fixed.

## Git state
- Branch `redesign`, **HEAD = 7242ea33** ("checkpoint: near-miss fail reporting + hand-solver DBE +
  signature-tiers plan"). Committed, **NOT pushed**.
- That checkpoint bundled the prior near-miss work + this session's DBE rework + the planning docs.
- Untracked and deliberately NOT committed: 5 scraper JSON files (incidental data — a separate
  `data:` commit if wanted).
- **Push gate (unchanged):** the standard overnight full-corpus A/B has NOT been run. Do not push.
- DB state (catalog rows, literal_words, me/at deletions) is gitignored/local — not in any commit.

## What this build is
Signatures get TWO catalog classes so the risky families (anagram/container/indirect) can be added
without ever risking a silent false PASS:
- **PASS-tier** — low risk (G=false, L≤3, not short-answer-amplified); can PASS; added after an
  automated regression-hunt A/B.
- **PENDING-only** — a DIFFERENT class; when it fires it can NEVER pass, only ever PENDING → always
  a quick human review. Safe by CONSTRUCTION: it runs in a NEW cascade stage placed AFTER every
  PASS-capable engine, so it can only ever convert FAIL → PENDING (never touch a pass, never
  false-pass). See build plan §2/§3 for the rubric that assigns the class.

## Build order (build plan §5 — all additive, NO working engine modified)
1. `tier` column on `catalog_templates` (default `'pass'` = zero behaviour change). New migration
   `core/_migrate_signature_tier.py`, mirror `_migrate_deletion_structure.py`.
2. `tier` field on `catalog_loader.Template` + a `tier=` filter arg on `load_templates`.
3. Main cascade (`engine_registry.py:476-485`) loads `tier='pass'` only. **Regression gate: 0/0/0**
   (trivially true — all existing templates default to 'pass').
4. NEW final PENDING-only signature stage (~`engine_registry.py:1360`, after the last engine,
   before the cryptic-definition fallback) that caps any match to `status='pending'`. THE CORE.
5. Defence-in-depth cap in `_finish` (`engine_registry.py:1505`) keyed on `parse.template_id`→tier.
6. `catalog_creator.create_signature` gains a `tier` param.
7. Promotion: eligibility query (≥10 confirmations / 0 rejections / ≥3 puzzles) + one-click
   `UPDATE tier='pass'` + regression-hunt on promote.
8. Review surface: NEW `signature_reviews(template_id, clue_id, verdict, reviewed_at)` log written
   on confirm/reject; interim = markdown triage summary + existing PENDING clue page. Dashboard
   deferred to one shot later.

**HARD dependency:** do NOT insert any `tier='pending'` template into the catalog until Steps 1–4
are complete (before Step 3 filters, the main cascade would load it and pass it uncapped).

## Invariants (build plan §8 — never violate)
- Additive only. No working engine is modified (honours CLAUDE.md Rule 1).
- PENDING-only signatures run LAST — the ordering IS the safety.
- Regression **0/0/0 between every step** (fresh non-persist solve over the cascade-passing clues +
  a fail sample; the near-miss session used ~1598 clues — re-create the harness in scratchpad, the
  corpus query is in memory `near_near_miss_fail_reporting` / prior handovers).
- Triage stays DIAGNOSE-ONLY: Claude reports signatures + tiers; the USER files/commits. Claude is
  the A/B **regression hunter** (find damage, never advocate the green).

## Settled policy (build plan §9 — do not re-open)
1. Two catalog classes + a triage "no-signature — escalate" outcome for noisy fabricators.
2. G-primary rubric: PASS-tier iff `G=false` AND `L≤3` AND not short-amplified; else PENDING-only.
   (SECT+ION L=2,G=false → PASS-tier. Indirection, not operation type, is the risk driver.)
3. Short-answer amplifier: `answer ≤ 4 AND L ≥ 2` → PENDING-only.
4. PASS-tier A/B: automated, reuse ~1600 harness; 0 regressions + ALL new passes faithful (full
   inspect); >~10 new passes = re-examine tier.
5. Promotion: ≥10 confirmations / 0 rejections / ≥3 puzzles → one-click manual + regression hunt.
6. Review: interim markdown + PENDING clue page + `signature_reviews` log; dashboard = later, once.

## Environment (verified this session)
- **Python:** `.venv\Scripts\python.exe`. The bare `python` on PATH is a Windows-Store stub — do
  NOT use it. For scripts importing `core`, set `$env:PYTHONPATH = (Get-Location).Path` first.
- **Server:** `.venv\Scripts\python.exe -m core.wfw_web` → http://127.0.0.1:5099/ . Restart after
  ANY code change. (This session it was launched via Start-Process, hidden, logs in `$env:TEMP`.)
- **Diagnose:** `python -m core.diagnose <clue_id>` — pieces view + per-engine view. CAVEAT: it
  does NOT initialise the role_validity predicates, so its engine-view is a subset and shows
  "role-validity predicates not initialised (failing closed)" instead of the real verdict; and it
  does not run every engine. Use it for the pieces (material) view; use the clue page for the real
  cascade result.
- **DBs:** `data/clues_master.db` — `clues`, `wfw_solve` (authoritative verdict: status +
  solved_by, keyed clue_id=clues.id), `catalog_templates` + `catalog_template_slots` (the
  signatures you're about to two-tier).

## First action for the build thread
Read the build plan, then implement **Step 1** (the `tier` column migration — zero behaviour
change) and verify it in isolation before anything else. Get the checkpoint diff regression-clean
at Step 3 before touching Step 4 (the core). Commit per step (with user approval), never push.

## Loose ends / not in scope for the build (noted for completeness)
- The OLD indicator-type "definition by example" in `_IND_TYPES` + the `_annotation_row`
  "By example" branch (wfw_render.py) is now redundant with the DBE-as-definition rework — the user
  was asked whether to remove it; NOT yet removed.
- `nightly_triage_process.md` should be updated to mirror the settled rubric/decisions before any
  real signature is filed (build plan §10) — do it alongside Step 7/8.
