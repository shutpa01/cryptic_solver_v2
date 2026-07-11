# HANDOVER 2026-07-11 — The live-site plumbing is BUILT and RUNNING; next = phase 7 (SEO) + first-nightly review

**START HERE.** Phases 1–6 of the live-site plumbing were built, tested and committed
on 2026-07-10 in one session. The publish-first process now runs END TO END with no
manual invocation. Read `memory/MEMORY.md`'s 2026-07-10 entries + this doc before code.

## 1. STATE (branch `redesign`, clean tree, NOT pushed — user approves pushes)
Six commits, in order:
- 16945013 pre-wiring cleanup: WFW clue page VIEW-ONLY + /hs Cryptic-definition button
  (/hscd) + /hsstatus now captures signature_reviews (the gap was real).
- e36b1a2c phase 1: WFW-fed hints — web/wfw_read.py (raw SQL, NO core imports) + 3
  additive models.py hooks; listing coverage counts wfw passes.
- 5fe1f51a phase 2: full breakdown as same-page OVERLAY (POST /wfwfull, token-gated) +
  PERMANENT page-shape test web/test_wfw_overlay_contract.py (63/63).
- 0671f731 phase 3: FREEZE RULE (/admin/rerun refuses wfw-solved clues, force cannot
  override) + puzzle.html admin panels stripped (routes kept).
- 9ac8173f phase 5: ONE APP — web/solver_mount.py mounts core/wfw_web.py (UNTOUCHED)
  at /solver, admin-gated, url_map-driven URL rewrite; WFW_ADMIN_BASE="/solver"
  (env-overridable back to 5099). + CLUTCH FIX (3rd strike — see memory
  clutch-loss-fault: every link into /hs / clue page carries the WHOLE-PUZZLE clutch).
- 627956ff phase 6: nightly chain — scrape all 5 → scripts/nightly_cascade.py (serving
  papers telegraph/times/guardian, answer required, never frozen) → headless
  `claude -p` diagnosis → headless prefill (prompts in scripts/prompts/, write-scope
  hard rules inside). Danword now truly opt-in; blog scrapers (TFTT/15sq) retired.
  "Cascade now" admin button on the puzzle page (prize-morning flow).

First supervised run 2026-07-10: diagnosis = correct no-op; prefill = 70/75 readings
validated+written, 5 deliberate blanks (reasons in logs/prefill_2026-07-10.md).

## 2. THE DAILY PROCESS (as of now)
2am Task Scheduler (CrypticSolver_NightlyRun → scripts/nightly_run.bat → .py):
scrape all → cascade serving papers → claude diagnosis (yesterday's commits →
pending-only sigs + engine_worklist) → claude prefill (today's fails → hand-solver).
Morning (user): puzzle list → /solver/hs per puzzle → Commit/Uncommit → publish.
Prize puzzles: hand-solve grid → save answers → "Cascade now" button → walk the list.
Admin on the site: any page + `?admin=dev-admin-key` (param `admin`; 30-day cookie).

## 3. ✅ FIRST UNATTENDED NIGHTLY (2026-07-11 01:00 UK) — CLEAN, chain proven
Reviewed in-session before shutdown: scraped telegraph 31289 + times 29592 (both
Saturday PRIZE puzzles, answerless) + independent 12405; no Guardian (Saturday —
correct). Danword skipped with the opt-in notice. Cascade: 0 to solve / 60 answerless
skipped (correct). Diagnosis: correct no-op (TIGER already diagnosed; report only;
deleted its own scratch script). Prefill: ZERO writes, correctly reasoned (prizes
answerless; Independent not a serving paper). Every guard held unattended.
Morning flow for 31289/29592: solve grid → save answers → "Cascade now" → walk list.

## 3b. ⚠️ TRAVEL GAP (machine off after 2026-07-11)
The nightly runs ON THIS MACHINE. While it is off: no scrape, no cascade, no prefill.
Catch-up on return: `python scripts/nightly_run.py --date YYYY-MM-DD` per missed day
(scrapers fetch what is still available; gapfill scripts exist per paper for older
misses), or scrape-only then `scripts/nightly_cascade.py --date` + the claude steps
via `nightly_run.py --skip-scraper --skip-cascade --date`. 2026-07-10's three puzzles
are already cascaded + prefilled and waiting in /solver/hs.

## 4. NEXT JOBS (in order)
1. (done — see §3) first unattended nightly reviewed clean.
2. **Phase 7: SEO/crawler-visible clue render** (LAST by design — indexing history).
   Precedent material: web/routes/clue_seo.py (schemas), the view-only clue page.
3. Engine worklist (non-critical time): replacement_letter first (101 back-test
   candidates), then word_cycling (9). A/B-gated, frozen-snapshot _regr.py pattern.
4. Open details: unforce UI still only on /rolegrid (fine until /rolegrid dies);
   templates 1672-1674 await user promotion via sig-regress.

## 5. RULES (unchanged, hard-won — full versions in CLAUDE.md + memory)
- Plain English, short sentences, bad news first. Verify before claiming (file:line).
- Questions are not instructions. All means all. One file at a time, test between.
- NEVER modify stages/ engines for edge cases; engine changes A/B-gated.
- Claude never re-runs a clue for score; test on temp DB copies (patch
  store.DEFAULT_DB + signature_reviews._CLUES_DB + admin_db paths +
  catalog_loader._default_db_path — unpatched constants burned a run once).
- EVERY new link into /hs or the clue page carries the whole-puzzle clutch.
- Server 5000 = web/run_dev.py (debug auto-reload; hard-refresh Ctrl+F5). 5099
  fallback = python -m core.wfw_web. Env: .venv\Scripts\python.exe;
  $env:PYTHONPATH=(Get-Location).Path.
- Git: push only with explicit approval.
