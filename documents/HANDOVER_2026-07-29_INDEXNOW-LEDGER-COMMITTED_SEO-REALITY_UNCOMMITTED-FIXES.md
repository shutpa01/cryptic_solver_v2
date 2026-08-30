# COLD HANDOVER — 2026-07-29

**One line:** IndexNow resubmission fixed with a per-puzzle ledger (COMMITTED, live, verified in steady state). Two verified-but-UNCOMMITTED fixes from this session await a commit decision. Confirmed the Google catastrophe's root cause and read the real Bing/GSC numbers — forward path is STABILITY only.

Read this with the memory index (MEMORY.md). Everything below is labelled VERIFIED vs not.

---

## 0. BEHAVIOURAL NOTE FOR THE NEXT SESSION (read first)

This was a long, rocky session. The user (rightly) has low trust in Claude's advice, especially anything touching SEO. Hold to these, hard:

- **Work from facts, verify against the REAL running thing.** An internal `test_client`/dry-run that passes proves nothing about the actual served page or the actual process answering the port. Reproduce the exact URL / the actual server. (This session: a fix "did nothing" because a 2-day-old stale process still owned :5001 — an internal test had "passed".)
- **No guessing dressed as fact.** When an exact answer is available (the DB, today's date, BWT, the real HTTP response), go get it. Don't estimate.
- **Own mistakes without deflecting.** Do NOT say "a previous instance did that." All Claude work is one responsibility.
- **NEVER propose anything that changes the shape of the site to search engines** (URL counts, sitemap size, removals/reinstatements). See §2 — this is what caused the catastrophe.

---

## 1. MAIN WORK — IndexNow puzzle-ledger (COMMITTED e972194c) ✅

**Problem (VERIFIED in BWT):** Bing showed 771 submissions but only ~284 unique URLs — we were re-announcing the same URLs. Root cause: the notifier decided "new" by `wfw_solve.created_at`, which resets whenever a clue is re-saved, so old clues (e.g. a Jan-19 puzzle re-stamped Jul-26) got re-sent. A URL is permanent, so a timestamp is the wrong signal.

**Fix (committed, `scripts/indexnow_notify.py` rewritten + `web/indexnow.py`):** a permanent per-puzzle sent-ledger.
- Ledger = sqlite at `logs/indexnow_state.db`, table `sent_puzzle` PRIMARY KEY `(source, puzzle_number)`. **gitignored, machine-LOCAL, NOT deployed** (like the old watermark). Regenerable.
- A normal run sends only puzzles NOT in the ledger — the puzzle page URL + each served clue URL — and records a puzzle ONLY after Bing accepts all its URLs (per-puzzle, so a partial/interrupted run resumes).
- Puzzle-level is safe because the user only deploys a puzzle once EVERY clue is solved. Keyed on source+number together (Times 29598 ≠ Guardian 29598).
- Modes: default (send+record); `--dry-run`; `--seed-before YYYY-MM-DD`; `--seed-all`; `--force`. **Guard:** refuses if the ledger is empty AND >20 puzzles look new (stops an un-seeded mass-send).
- Internal no-silent-fail fix: every served puzzle's page URL comes straight from `served_puzzle_numbers()`, so a puzzle can't be dropped just because its clues yielded no URL; a served puzzle producing 0 URLs prints a loud WARN. VERIFIED: 57 served puzzles, 0 with zero URLs, dry-run counts unchanged by the fix.
- `deploy.py` still calls it no-args = ledger mode.

**DO NOT TOUCH the submission METHOD.** `web/indexnow.py` sends one URL per request (streaming), changed 2026-07-22 for a Bing reason we can NO LONGER RECALL. **Do not revert to batch** — that blind flip-flop is exactly what damaged Google. It's committed as-is precisely so it can't be accidentally reverted. The `deploy.py` 120s subprocess cap was deliberately left untouched (rarely bites once volume drops; the per-request 15s timeout already prevents a real hang).

**State (VERIFIED):**
- Ledger SEEDED: 45 puzzles published on/before 2026-07-23 marked sent.
- Catch-up SENT 2026-07-27 18:38 — 12 puzzles / 358 URLs, **12/12 OK, 0 failed** (guardian 30067/30069/4162, telegraph 235/31300/31301/31302/3379, times 29603/29604/29605/5226).
- 2026-07-28 05:20 a deploy sent 3 NEW puzzles (guardian 30070, telegraph 31303, times 29606) and recorded them → ledger now **60 puzzles**. Steady state confirmed: only the new day's puzzles go out, once each. Bing's live "97 submitted in last 5 hours" matched our ledger exactly.
- NOTE: Bing's IndexNow UI (chart/total) lags ~1–2 days; the "Submitted URLs list" is de-duplicated. Don't read "one URL today" as a fault — check the LOCAL ledger (`logs/indexnow_state.db`) for truth.

**Operational note:** if the dev machine / ledger DB is ever lost, re-seed with `--seed-all` (marks all currently-served puzzles as sent, sends nothing) to avoid re-blasting Bing.

---

## 2. SEO REALITY — the catastrophe's cause + the real numbers (VERIFIED)

**Root cause of the Google collapse (confirmed by the user this session):** Claude advised massively REDUCING the site's URLs, then the very next day said that was a mistake and to REINSTATE them all. That URL flip-flop is the erratic signal that made Google distrust the site. Traffic "died instantly." **This is the binding lesson: never propose URL/sitemap-shape changes; recovery is stability + time only, no shortcut.**

**GSC (Google, 16-month view, VERIFIED 2026-07-27):** 34K impressions, 1.01K clicks, avg position 4.2 — but the shape is one spike (~a year ago) then a cliff to ≈0, flat ever since incl. now. So Google traffic is effectively dead; the content DID rank (~pos 4) during the spike, so the block is trust/indexing, not quality.

**Bing (BWT, VERIFIED 2026-07-27):**
- ~1,500 pages indexed; ~2,200 known; 516 excluded; 138 error.
- IndexNow submissions working daily. A clue submitted 21 Jul was indexed by 27 Jul (≤6-day lag).
- Search Performance (3mo): 11 clicks / 683 impressions, avg position ~3–4. Impressions are clue-text searches (market 1). **All 11 clicks are consistent with being the user** (10 UK + 1 US likely via VPN) — no confirmed external click yet.
- Indexing lag ~1 week vs a daily puzzle's ~3-day search life → dailies largely miss their window; **prize puzzles (longer life) are the forgiving case.**

---

## 3. VERIFIED but UNCOMMITTED — this session's other fixes (commit decision needed)

Both tested through the real path; NOT committed (user only asked to commit the indexing work):

1. **Prize-toughie grid showed no answers** — `scraper/danword/danword_lookup.py` `find_puzzle_json`: was a loose substring title match ("235" matched "No 31235"), returning the wrong grid so no answers placed. Fixed to whole-number match + utf-8 read. VERIFIED via the real `/grid` + `/grid-progress` routes (131 letters placed; user-typed embargoed answers place too). See memory `prize_toughie_grid_json_substring_mismatch`.
2. **HS: CD auto-pass + INVALID-needs-comment** — `core/wfw_web.py`: the Cryptic-definition button now passes+freezes in one click; the INVALID verdict now requires a comment, saved in the same action (server refuses without). VERIFIED via `/solver` on 5008. See memory `hs_cd_autopass_invalid_requires_comment`. **NB `core/wfw_web.py` also carries pre-existing uncommitted changes from earlier sessions — do not blind-commit the whole file; review the diff.**

---

## 4. OTHER pre-existing UNCOMMITTED changes (NOT this session — don't accidentally bundle)

`git status` tracked-modified at handover (HEAD = e972194c):
- `.claude/settings.local.json`
- `core/admin_db.py` — prefill definition honesty gap (07-24)
- `core/container_deletion_engine.py` — CD false-pass guard (07-22, NOT deployed)
- `core/wfw_web.py` — mix: HS fix (this session) + earlier work
- `scraper/danword/danword_lookup.py` — prize-toughie grid fix (this session)
- `scraper/orchestrator/daily_scraper.py` — HTML-tag strip at sync chokepoint (07-26)
- `web/run_dev.py` — port 5001 (07-24)
- `web/templates/about.html`, `web/templates/puzzles.html` — title SEO (07-22, live)
- `web/wfw_read.py` — OMELETTE summary fix (07-24)

---

## 5. ENVIRONMENT / traps

- **Port 5001 is contested** between this project (V2) and the SEPARATE AI_Solver project — both `web/run_dev.py` hardcode 5001, so a stale/wrong process can answer and make edits "appear to do nothing." After any dev restart, VERIFY exactly ONE listener owns 5001 and it's the V2 venv (`Get-NetTCPConnection -LocalPort 5001`). The V2 dev server launched this session was killed 2026-07-29 — none currently running. Permanent fix (move one project's port) NOT done; user aware.
- Reloader is OFF — any .py change needs a FULL dev restart. Launch with ABSOLUTE venv path.
- The notifier / deploy runs on the DEV machine (dashboard Deploy → subprocess), reading the dev repo + dev DB.

---

## 6. LOOSE ENDS

- **Microsoft Clarity email** (2026-07-28): asks to install a tracking snippet on the site. User suspects the "Cordelia" Clarity project may be under the OTHER login. **Do NOT install** until the account + configured target URL (should be justcordelia.com) are confirmed. It's SEO-safe (async script, no URL change) but adds MS tracking/cookies — a UK-GDPR consent consideration. Snippet would go in `base.html` <head>.
- Automated nightly diagnosis ran 2026-07-29 (separate from this session) — cleared a 584-solve backlog in FOREGROUND; report `logs/diagnosis_2026-07-29.md`; 24 pending templates 1684-1707; worklist +5 new / +31 bumps. See memory `nightly_diagnosis_no_background_wait`.

---

## 7. SUGGESTED NEXT STEPS (user's call)

- Decide whether to commit the two verified fixes in §3 (grid fix is cleanly isolated in `danword_lookup.py`; HS fix shares `wfw_web.py` with older changes — review diff).
- Otherwise indexing is done and self-running; nothing is blocked. Recovery is a waiting game on trust — hold stability, watch the Bing indexed count and GSC curve over weeks.
