# HANDOVER 2026-07-18 EVENING — UX fixes + grid-deploy fix live; DB deploy is NEXT

Cold-start document. Read this, then MEMORY.md. Plain English; **verify against the code
before acting** — memory/handover reflect what was true when written. This session's code
is committed to **redesign @ b37e8877** (NOT pushed) unless stated otherwise.

---

## 0. ★ STATE OF PLAY — LIVE vs COMMITTED vs OPEN

**Git:** branch `redesign`, HEAD = **b37e8877** (one commit this session), **ahead of
origin/redesign by 1 — NOT pushed.** The commit bundles five files:
core/wfw_web.py, core/pending_store.py, web/static/js/puzzle2.js,
dashboard/pages/deploy.py, scraper/orchestrator/puzzle_scraper.py.

**LIVE on the droplet** (deployed by hand this session):
- **puzzle2.js** (the tools-overlay UX fix) — scp'd to /opt/cordelia/web/static/js/;
  backup puzzle2.js.bak-20260718; verified served via justcordelia.com/static/js/puzzle2.js.
- **Scraper grid JSONs** — rsynced scraper/{times,telegraph,guardian} to the droplet with
  the corrected command; today's grids (times_cryptic_29598.json,
  telegraph_prize-toughie_93057.json) confirmed present. This fixed live solve mode for
  today's two puzzles (verified: times/29598 hasGrid=true, solve mode enters).

**COMMITTED (b37e8877) but NOT pushed, and NOT yet on the droplet as a code deploy:**
the deploy.py + puzzle_scraper.py rsync fixes (they RAN by hand this session; the code
change just makes future dashboard deploys do it automatically). Deploying the code is
optional — the manual sync already fixed today.

**LOCAL DB fixes — saved + backed up, NOT deployed** (go live via the dashboard DB deploy):
- puzzle_grids rows added: **times/29598, telegraph/233** (were missing → "No grid
  available" on the droplet).
- Enrichment healing: added **Very big→OS, psychic ability→ESP, volunteer→VOL** to the
  wordplay table; removed the 3 mis-filed synonyms; cleared 5 stale pending rows.
- Today's puzzle solves: 10080055 (CD APOSTROPHE), 10080052 (PIANO), and
  OSCAR/ESPY/VOLCANO/SPAM confirmed to frozen passes.
- Backups: **data/*.db.bak-20260718-datafixes** (full end-of-day snapshot) + earlier
  -enrichfix.

**NEXT, in priority order:**
1. **DB deploy** (dashboard DEPLOY page — your action) carries all of today's DB fixes live
   AND the still-open prior-session data (§3). Back up the droplet DBs first.
2. (Optional) **push b37e8877** to origin/redesign, and/or code-deploy the deploy.py fix.
3. SEO is still recovery = stability + time; IndexNow is confirmed working (§4).

---

## 1. WHAT WAS DONE THIS SESSION

### A. Hand-solver / enrichment honesty fixes (committed b37e8877)
- **CD button clears assignments** (core/wfw_web.py `/hscd`): pressing "Cryptic definition"
  now wipes any leftover per-word hand-solver assignment before resolving. Without it,
  `_resolve_from_assignment` (authoritative since 07-17) rebuilt from the stale assignment,
  found no pieces, deleted the parse, and the CD engine never ran ("got no parse").
  Fixed clue 10080055 (Telegraph 31295 14d, "Mark from King's Cross?" = APOSTROPHE).
  Memory: [[cd-button-clears-all-assignments]].
- **Inline enrichment Approve on the WFW clue page** (core/wfw_web.py `_render_one`):
  a pending-prefill clue now shows its queued enrichment pieces (Approve/Reject) inline,
  so you approve an AI-proposed piece + Confirm without a detour to /hs. Revives the
  orphaned `_enrichment_block`/`_enrich_row`. Memory: [[wfw-page-inline-enrichment-approve]].
- **Substitution queue routing** (core/pending_store.py + `_build_manual_parse` +
  `_enrich_row`): the "sometimes works, sometimes not" enrichment bug. The gate queued
  EVERY provisional piece as type='synonym', even abbreviations — so Approve wrote to
  synonyms_pairs while Confirm checks has_substitution (wordplay table) → never
  reconciled. Fixed: substitution-role pieces queue as type='substitution' →
  add_substitution → wordplay. Healed the 5 stuck pieces (§0). Memory:
  [[wfw-page-inline-enrichment-approve]] (substitution section).

### B. Public solve-mode Tools overlay (committed + DEPLOYED)
puzzle2.js: picking a solver result keeps the overlay OPEN and fills the answer box (was
auto-closing and dumping you back to hunt for the clue); "Add to grid" commits + stays
open with feedback. Browser-verified on the live droplet. Memory:
[[tools-overlay-add-to-grid-inplace]]. (puzzle.js is DEAD — cache-bust rename; edit
puzzle2.js only.)

### C. Solve-mode grid JSON deploy fix (committed; ran by hand + DEPLOYED)
Root cause of "works local, no grid on droplet": solve mode's `has_grid` needs a
puzzle_grids DB row OR the scraper JSON on disk. The scraper JSONs are LOCAL-only and were
supposed to sync to the droplet, but `_rsync_json_dir` (puzzle_scraper.py) was **missing
`-r`** → rsync said "skipping directory ." and shipped NOTHING. So the droplet never got
new grid JSONs. Fixed `-r` in puzzle_scraper.py AND added the JSON-dir sync to the
dashboard deploy.py (it never did it). Ran the corrected sync by hand → today's grids
live. Memory: [[solve-mode-grid-json-deploy]].

### D. IndexNow confirmed working (no code change)
Today's two puzzles were deployed; the green "IndexNow notify: Watermark advanced" box +
dry-run showing 0 pending = submitted OK. Verified the key file is reachable through
Cloudflare (curl -A IndexNow-Verifier → HTTP 200 + the key), so Cloudflare is NOT blocking
the verifier. No BWT check needed day to day. Memory: [[indexnow-automation]].

### E. Prize-puzzle answer investigation → DECISION (no code)
Long investigation into how Danword/Wordplays get prize answers "instantly". Conclusion,
evidence-backed: **the answers are genuinely NOT in the Telegraph API/player at
publication** (empty grid + one whole-grid MD5 hash + reveal features off; the live
logged-in player fetches no answer data, auto-check fires nothing). The "instant" sites
use **online solvers / frequency-ranked DB lookup**, not a feed — proven because Wordplays
returned FARRINGDON (wrong) for the CD "Mark from King's Cross?". **DECISION: keep
hand-solving prizes with our tools + occasional Danword; do NOT build a fetcher and do NOT
re-investigate.** Memory: [[prize-puzzle-answers-not-in-api]].

---

## 2. THE DAILY FLOW GAP THIS EXPOSED (watch for recurrence)

Solve mode breaks for a newly-served puzzle until its grid reaches the droplet. Two paths
now cover it: (a) the puzzle_grids DB row (ships with the DB deploy), (b) the scraper JSON
(ships via the now-fixed rsync in scraper + dashboard deploy). Today's two puzzles slipped
through BOTH (no row built yet + the rsync bug), so they were hand-fixed. With the `-r`
fix, future dashboard/scraper deploys sync the JSONs automatically. If "No grid available"
recurs, check: does the served puzzle have a puzzle_grids row OR its JSON on the droplet?

---

## 3. STILL-OPEN DATA TO-DOs (carried from prior sessions; deploy with the DB)

- **3 tentative homophone pairs to APPROVE** in pending_enrichments: over~ova,
  re-sinned~rescind, dough~deau. Provisional until approved.
- Invalid clues to eyeball (10079943 CENTRE OF GRAVITY, 10079976 ABLE SEAMAN).
- RIA (10079981) stays a frozen INVALID (take=R is fabrication) — leave it.
- The honesty batch from eb72c798 (prior session) is still committed/undeployed; the DB
  deploy carries its data effects. Code deploy of it remains optional/cosmetic.

---

## 4. SEO / IndexNow (unchanged; recovery posture)

Site was deindexed (thousands→~36) by URL churn + a ~2-month 503. Recovery = stability +
time. **Never churn URLs / sitemap size.** IndexNow auto-fires on DB deploy (incremental,
watermark) and is confirmed working end-to-end including the Cloudflare key-file fetch.
The green box on the DEPLOY page is the daily signal; no Bing Webmaster Tools check needed.

---

## 5. PRINCIPLES RE-AFFIRMED THIS SESSION

- **Verify on the REAL target, through the real path.** The grid bug only showed on the
  droplet; the fix was proven by fetching the live file + driving the live browser.
- **No bluffing; read the user's own system as evidence.** The prize-answer thread went
  badly at first because I invented mechanisms ("answers are in the digital payload")
  instead of testing. The user's own design (hand-solving prizes) was the evidence I
  ignored. Logged in violation_log.md. When you don't know: say so and gather evidence.
- **Deploy = dashboard DB deploy** (or scp + restart for a one-off). Only the databases
  (and now the scraper JSONs) reach the droplet; local scraper JSONs are otherwise never
  deployed, so anything depending on a JSON file works local / fails droplet.
