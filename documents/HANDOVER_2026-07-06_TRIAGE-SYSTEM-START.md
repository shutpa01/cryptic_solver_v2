# HANDOVER 2026-07-06 — Start the triage system (read the design FIRST)

**START HERE for the new thread.** The previous thread finished a good piece of work (the
two-tier signature build, below) and then went off the rails when asked to start the triage
system: it did NOT read the design, improvised a made-up system out of its own head, invented an
architecture "decision" the design had already settled, and pushed that decision onto the user
twice — all while burying it in jargon. The user rightly lost confidence and asked for a clean
restart. Do not repeat that failure. This handover exists to point you straight.

---

## 0. THE ONE RULE FOR THIS WORK (do not skip)

**Read `memory/nightly_triage_process.md` IN FULL before you do anything, or say anything, about
the triage system.** It is THE design. Then read the code it touches (below) before proposing a
build. There is a standing memory `feedback_read_design_fully` about exactly this failure mode —
honour it.

**How the user wants to work (they stated this explicitly):**
1. Read the design AND the real code, then write back — **in plain English, no jargon** — your
   understanding of (a) what the triage system is, (b) what already exists, (c) what needs
   building.
2. **The user checks and corrects that understanding BEFORE you write a single line of code.**
3. Only once they confirm you understand it do you agree the SMALLEST first piece, build that one
   piece, and show them.
- Do NOT invent decisions and hand them to the user. If the design already settles something,
  follow it — don't re-open it. If something is genuinely open, present it plainly, once, with a
  recommendation — don't make the user choose an architecture.
- No jargon. The user found the previous thread's language very hard to follow. Plain words.

---

## 1. What the triage system IS (from the design — verify by reading it yourself)

`nightly_triage_process.md` (the design), key points, quoted:
- After the nightly scrape → solve on the complete new puzzle, review **every FAIL and PENDING
  clue** and classify **WHY** it failed: **missing data / missing signature / missing engine**
  (one or more per clue).
- **Claude DIAGNOSES and INFORMS ONLY — never resolves, never commits, never iterates.** The user
  commits via the dashboard. This wall is what stops pass-rate chasing.
- **ONE diagnostic pass, no loop.** Do NOT re-run the solver to check a fix "worked".
- **Missing data** → queue the specific missing entries to the dashboard enrichment queue.
  **Missing signature** → report the exact signature needed (now WITH its tier — see §2); the user
  files it. **Missing engine** → escalate.
- **DO NOT CHASE PASS RATES.** "cannot parse" is an acceptable, honest outcome.
- The design's last line: **"Still designing output surface"** — i.e. the actual **dashboard**
  (where the user sees the diagnosis and accepts/rejects/re-runs) is the genuinely UNBUILT part,
  and its shape was left OPEN. Do not assume its architecture — work it out with the user.

The prior thread's mistake was treating "who diagnoses?" as an open question. It is NOT open: the
design says **Claude diagnoses.** The open part is only the output surface (the dashboard).

---

## 2. What WAS built and validated this session — the two-tier signature system (KEEP)

This is real, finished, and regression-clean. It is the safety machinery that lets risky solving
recipes exist without ever silently declaring a clue "solved". Full spec:
`documents/BUILD_PLAN_2026-07-05_SIGNATURE-TIERS.md`. Progress notes: memory
`signature_tiers_built.md`. In plain terms: every recipe is now labelled **pass** (trusted, may
declare solved) or **pending** (needs-checking — can only ever offer a suggestion a human
approves; it runs in a new final stage after all trusted recipes, so it can only ever turn a
FAIL into a suggestion, never mint a false solve).

All 8 steps done + verified in isolation; corpus regression **0/0/0** at steps 3+4 and step 5
(1617 clues, pass set byte-identical). Files:
- NEW `core/_migrate_signature_tier.py` — added the `tier` column to `catalog_templates` (RAN on
  the real DB; all 1669 templates default `'pass'` → zero behaviour change).
- `core/catalog_loader.py` — `Template.tier` field, `tier=` filter on `load_templates`, `tier`
  forwarded through all 10 `load_*_templates` helpers, new `load_template_tiers()`.
- `core/engine_registry.py` — main cascade loads `tier='pass'` only; NEW final stage
  `_solve_pending_signatures` (caps pending-tier matches to PENDING); defence-in-depth cap in
  `_finish`; pending templates + `template_tier` map added to the wiring.
- `core/catalog_creator.py` — `add_signature(..., tier='pass')` param.
- NEW `core/signature_reviews.py` — review-capture log + eligibility + one-click `promote()`
  (≥10 confirms / 0 rejects / ≥3 puzzles → promote pending→pass; regression hunt is a separate
  human step).
- `core/wfw_web.py` — `/setstatus` logs a confirm/reject review when a pending-tier clue's verdict
  is overridden.
- `memory/nightly_triage_process.md` — updated with the tier rubric/routing (the classify step now
  also assigns a tier to a missing-signature clue).

This connects to the triage work: when triage finds a **missing signature**, the proposal now
carries a tier (trusted vs needs-checking), decided by the rubric in `nightly_triage_process.md`
(indirection load L + free-choice-over-synonyms G; short-answer amplifier).

---

## 3. Building blocks that already exist for the dashboard (verified read-only this session)

Do NOT rebuild these; the dashboard ties them together.
- **Verdicts:** `wfw_solve` in `data/clues_master.db` — `status` (pass/pending/fail), `solved_by`,
  `template_id`, keyed `clue_id = clues.id`. Group a puzzle by `clues.(source, puzzle_number)`.
- **Auto-queued proposals:** `pending_enrichments` table (11,415 rows; cols id, type, word,
  letters, answer, clue_text, source, puzzle_number, created_at). NOTE it is SPARSE — for today's
  DT puzzle 31284 it held only **3** rows (definitions for the 3 PENDING clues), and NOTHING for
  the 21 fails. So the solver's own auto-queue does NOT capture most gaps — the rich diagnosis is
  Claude's job (per the design).
- **Routes (in `core/wfw_web.py`):** `/enrich` (accept an enrichment), `/reject`, `/reload`
  (re-solve a clue/puzzle — the REAL solver), `/approvesig` + `/rejectsig` (file/reject a
  signature), `/setstatus` (verdict override + now review-capture).
- **Diagnosis instrument:** `core/diagnose.py` — `pieces()` (definition candidates + what each
  clue word/phrase resolves to, flagged if it appears in the answer) and `engines()`. CLI:
  `python -m core.diagnose <clue_id>`. CAVEAT: `engines()` does NOT initialise the role-validity
  predicates and runs only a SUBSET, so its per-engine verdicts are unreliable — trust the
  `pieces()` view and the real `wfw_solve` verdict, not `engines()`.
- **DO NOT build on the legacy substrates** (per HANDOVER_2026-07-05 §6): `leftover.py` /
  `shadow_blog_v0.db`, `pipeline.py` (has_solution/structured_explanations), the Streamlit
  `dashboard/` dir, and the public `web/` site. The authoritative substrate is `wfw_solve` +
  `pending_enrichments` in `clues_master.db`.

---

## 4. Today's test puzzle (context, not a task) — DT cryptic 31284 (2026-07-06)

Freshly solved on the new code: **7 pass, 4 pending, 21 fail** (32 clues). The prior thread hand-
triaged the 21 fails (raw output: gone with its scratchpad; re-runnable via
`python -m core.diagnose <id>`). Headline finding, honest: on a brand-new un-enriched puzzle the
failures are **overwhelmingly missing DATA** (a synonym/abbreviation/definition not yet in the DB),
e.g. OUSTS needs `thoroughfares→STS`, OFFSPRING needs `not working→OFF`, ILIAD needs
`elected house→DAIL`. Only ~1–2 are genuinely "material present but recipe-shape missing". This
matters for expectations: the triage system's day-one value is mostly surfacing DATA gaps, not the
needs-checking recipes.

Fail clue_ids: 10077822, 10077823, 10077825, 10077826, 10077827, 10077828, 10077829, 10077832,
10077834, 10077835, 10077836, 10077837, 10077838, 10077839, 10077840, 10077841, 10077843,
10077845, 10077848, 10077849, 10077850. Pending: 10077824, 10077831, 10077844, 10077847.

---

## 5. Git / environment state

- Branch `redesign`, **HEAD = 7242ea33** (committed, NOT pushed). The two-tier signature work
  above is **UNCOMMITTED** on top of it: modified `core/catalog_creator.py`,
  `core/catalog_loader.py`, `core/engine_registry.py`, `core/wfw_web.py`; new
  `core/_migrate_signature_tier.py`, `core/signature_reviews.py`. Get the user's approval before
  committing; push gate (unchanged) = the overnight full-corpus A/B has not been run.
- DB state (the `tier` column, catalog rows) is gitignored/local — not in any commit. The
  migration has already been run.
- **Python:** `.venv\Scripts\python.exe` (bare `python` is a Windows-Store stub). For scripts
  importing `core`, set `$env:PYTHONPATH = (Get-Location).Path` first.
- **Server:** running on the NEW code, http://127.0.0.1:5099/ (PID was 14188). Started via
  `Start-Process -WindowStyle Hidden .venv\Scripts\python.exe -m core.wfw_web`. RESTART after any
  code change.
- **Regression harness:** was rebuilt in the prior thread's scratchpad (now gone). To recreate:
  fresh non-persist solve via `solve_clue_text(clue_text, answer, db_only(make_db_wiring()),
  clue_id=None, direction=...)` over the corpus work-list = all `wfw_solve.status='pass'` clue_ids
  + a 500-fail sample; record (status, solved_by); diff vs baseline. Runs ~15 min (slow deep-DFS
  fail clues). `solve_clue_text` returns `(ctx, parse, name)` — THREE values.

---

## 6. Suggested first move for the new thread

1. Read `memory/nightly_triage_process.md` in full. Then read `core/diagnose.py`,
   `core/wfw_web.py` routes `/enrich` `/reject` `/reload`, and the `pending_enrichments` schema.
2. Write back to the user, in plain English, your understanding of the triage system + what exists
   + what's unbuilt (the dashboard/output surface). **Wait for the user to confirm/correct it.**
3. Only then, with the user, agree the smallest first piece of the dashboard and build it.
Do not skip step 2. Do not improvise the output surface. Do not use jargon.
