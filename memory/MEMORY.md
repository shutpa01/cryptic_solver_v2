# Project Memory

## Session 2026-04-09: UX overhaul, tutorial puzzle, learn zone redesign
- See `memory/session_2026_04_09.md` — **READ THIS FIRST FOR NEW THREAD**
- **UX fixes**: hidden word display, explanation delivery, solve mode race fix, continue solving, tab persistence, anagram clear
- **Learn zone redesign**: spec in `documents/learn_zone_redesign_spec.md`, tutorial = real partially-filled puzzle, not custom 5x5
- **NEXT**: User picks tutorial puzzle, wire up pre-filled solve state + guided Cordelia tips
- **5x5 grid in DB**: source=cordelia puzzle_number=1 — may be replaced

## Session 2026-04-02 to 2026-04-09: Mechanical solvers, Cordelia deployment, GEO — MASSIVE
- See `memory/session_2026_04_02_09.md` — was referenced but file missing from prior session

## Coverage Reconciliation — SINGLE SOURCE OF TRUTH
- See `memory/coverage_reconciliation.md` — **KEEP UPDATED after every batch/upload**

## Session 2026-03-30/31: Backfill system, piece verifier, pipeline enhancement — MAJOR
- See `memory/session_2026_03_30_31.md`

## Session 2026-03-29: Backfill DD, Hidden, V1 Mechanical, Explanation Parser — 92k new solves
- See `memory/session_2026_03_29.md`

## Session 2026-03-28/29: Batch solvers, parser, scraper fixes, verifier — MAJOR BUT FLAWED
- See `memory/session_2026_03_28_29.md` — **READ NEXT STEPS FIRST**

## Session 2026-03-27/28: Verifier fixes, re-run unification, Silly Award
- See `memory/session_2026_03_27_28.md`

## Session 2026-03-26/27: Landing page, tips, re-verify, gap collection fix — MANY ISSUES
- See `memory/session_2026_03_26_27.md` — **READ CRITICAL FEEDBACK FIRST**

## Session 2026-03-25: Batch enrichment pipeline, 60k clues, solve rate 18%->27%
- See `memory/session_2026_03_25.md`

## Session 2026-03-24: Guardian fix, Chambers enrichment, Phase 0 solvers — MASSIVE SESSION
- See `memory/session_2026_03_24.md` — **READ THIS FIRST**

## Session 2026-03-21: Interactive Solving Workspace + Honeypot — MASSIVE SESSION
- See `memory/session_2026_03_21.md` — **READ THIS FIRST**

## Session 2026-03-20c: Live Site Features + Solver Tools — MAJOR SESSION
- See `memory/session_2026_03_20c.md`

## Session 2026-03-20: fifteensquared Pipeline + Flask + SEO
- See `memory/session_2026_03_20_fifteensquared.md`

## Session 2026-03-20b: Compound Clues, Independent Backfill, Batch Runner
- See `memory/session_2026_03_20b.md`

## Session 2026-03-19: TFTT Pipeline + Admin Page
- See `memory/session_2026_03_19_tftt_pipeline.md`

## SIGNATURE SYSTEM — V3 PARADIGM SHIFT (2026-03-12)
- See `documents/signature_system_design.md` — full design

## Live Site Build Plan
- See `memory/live_site_build_plan.md` — 7-phase plan

## Infrastructure (SOLID)
- **Web app**: Flask + HTMX + Tailwind — See `memory/product_design.md`
- **Dashboard**: Streamlit — See `memory/dashboard_review.md`
- **Nightly pipeline**: See `memory/nightly_pipeline_design.md`

## Key Architecture Notes
- **Sacred principle**: NEVER modify working pipeline stage engines in stages/
- `.env` must have `DB_PATH=C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db`
- `data/cryptic_new.db` = reference tables; `data/clues_master.db` = all clues (509k)

## Database State (2026-03-21)
- **clues_master.db**: 509k clues, 508k with answers
- **cryptic_new.db**: indicators (5,211), synonyms_pairs (~1.3M), definition_answers_augmented (~713k)

## CRITICAL USER RULES
- **NEVER suggest serving scraped blog explanations** — training data only (UPDATED: now served behind hint button with attribution)
- **NEVER suggest the 183k parsed explanations as a product path**
- **Use Times/Telegraph for testing, NOT Guardian**

## Working Feedback
- See `memory/feedback_never_guess_inputs.md` — if lookup returns no result, say so, don't substitute
- See `memory/feedback_commit_and_extract.md` — COMMIT EVERY FIX immediately
- See `memory/feedback_follow_agreements.md` — don't rush off, faithfully implement agreements
- See `memory/feedback_facts_only.md` — NEVER present assumptions as fact
- See `memory/feedback_stay_focused.md` — no flip-flopping, stay focused
- See `memory/feedback_no_guessing.md` — verify before stating
- See `memory/feedback_no_guessing_critical.md` — NEVER state anything without evidence
- See `memory/feedback_quality_not_speed.md` — QUALITY NOT SPEED: never rush, always verify through actual UI path

## Pending Work
- Tutorial puzzle: user picking a real puzzle to use as partially-filled walkthrough
- Learn zone page rework per spec in documents/learn_zone_redesign_spec.md
- Learn practice page: needs removal or full rework once tutorial finalised
