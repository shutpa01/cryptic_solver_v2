# HANDOVER 2026-07-15 (evening) — puzzle-level gate + sitemaps + dashboard prune; NEXT = SEO review

Cold-start document. Read this first, then the memory index. Plain English;
verify against the code before acting. Launch target **Friday 2026-07-17**.

## 0. ★ THE NEXT SUBJECT: SEO REVIEW

The user's next topic is an **SEO review** for the relaunch. Before starting it,
READ these, in order:
1. **`documents/HANDOVER_2026-07-13_RELAUNCH-PREP-WEEK-ONLY.md`** — THE relaunch doc
   ("written a few days ago"). Week-only decision, launch shape, the OPEN SEO
   decisions, and §7's binding scar-tissue rule ("sitemap and page gate share ONE
   rule; never list a URL that doesn't 200").
2. **memory `relaunch_discussion_2026_07_04`** — the strategic framing: the funnel
   (clue-text search -> clue page is the channel at volume), the GSC snapshot, the
   four markets, "freshness machine not archive." Do NOT re-litigate these; do NOT
   state cause-of-freeze theories as fact.
3. **`documents/seo_features.md`** — the SEO feature inventory (STALE, dated March;
   describes the pre-week-only 500k world — treat as history, not current truth).
4. The SEO scar-tissue memories (binding): no-sitemap-size-flipflop,
   never-propose-url-reduction, dont-catastrophise-seo, never-propose-publishing-db.

### What this session already established about SEO (carry into the review)
- **Nothing is indexed.** Site has served 503 since 2026-05-15; Google largely
  forgot it. Last GSC read (early July, user's own check) ~**37 indexed** — and
  those are LEGACY pages that 410 at launch (legacy corpus removed from serving).
  So the WFW index starts from **zero**. Get a FRESH GSC read before Friday.
- **Launch footprint ~1,450 pages** (~1,413 clue + ~18 puzzle + a few static),
  growing ~85-90/day forever. Small-but-dense is the SETTLED strategy — not a bug.
- **Content is short-life news** -> normal crawl indexing is too slow for same-day
  value; **Google News is the primary channel.**
- **Google News / Publisher Center (checked live 2026-07-15):** publication
  "Cordelia" IS registered. The manual approval the user once applied for was
  RETIRED (March 2025 — "Google News now automatically generates publication
  pages"). So there is nothing to "hear back" on. The real test only works once
  live: search news.google.com for the brand / a recent puzzle title, and watch
  for a "News" report appearing in GSC.
- **Request-indexing plan (all in the user's GSC — Claude can't click it):** flip
  live -> resubmit /sitemap.xml (currently "Couldn't fetch" due to the 503) ->
  URL-Inspect -> Request Indexing for the top ~10 (home, /puzzles, browse lists,
  2-3 flagship clue pages; ~10/day quota) -> monitor Pages / Crawl Stats / Sitemaps.
- Do NOT rely on: Google's Indexing API (officially job-postings/livestreams only);
  the sitemap "ping" URL (deprecated 2023).

## 1. REPO STATE

- Branch `redesign`. **This session's work is COMMITTED AND PUSHED** (commit
  `f3280a6f`); `origin/redesign` is in sync (0 ahead / 0 behind). This is the big
  change from the morning handover, which said everything was uncommitted.
- Only `.claude/settings.local.json` stays intentionally uncommitted (local
  settings; tracked-but-modified).
- Dev server: reloader OFF — restart after any change
  (`.venv\Scripts\python.exe web\run_dev.py`, V2 venv). I killed two stale :5000
  servers this session and left ONE clean instance running with all changes live.

## 2. WHAT WAS BUILT THIS SESSION (all committed, all verified via the real routes)

1. **Embargo policy = deleted.** A prior session invented a "prize-answer embargo";
   the user confirmed prize puzzles serve exactly like everything else. There is NO
   embargo. (Memory/handover language scrubbed where it mattered; go-live decision
   list lost that item.)
2. **Puzzle-level display rule (SETTLED by user + BUILT).** A PUZZLE page exists for
   the public only when EVERY clue is served; clues still serve one by one. "Served"
   = a WFW pass OR an INVALID-with-comment; completeness keys off STATUS, so passing
   "See N" stubs count. `web/serving.py` gained `puzzle_is_served` and
   `served_puzzle_numbers`. Applied to: puzzle page (410 public / admin bypass),
   browse list (`web/models.py`), search + typeahead (`web/routes/browse.py`), and
   the puzzle + news sitemaps (`web/routes/seo.py`). Every sitemap URL verified 200;
   in-progress puzzles excluded. Memory: `puzzle-level-display-rule`.
3. **Dashboard pruned** (memory `dashboard-deploy-and-pruning`): deleted pages
   review/indexing/leftover; renamed scraper -> **deploy** (Deploy-to-Cordelia panel
   only). **Deploy code manifest now includes `core/`** (+ core/atomsig) — it was
   missing, and the public clue-page card imports core, so a code deploy without it
   would 410 every clue page. Pipeline page gained a **"Not yet published (WFW status
   missing)"** report (clue count + O/S since a cutoff, default 2026-07-11); removed
   Puzzles-awaiting-pipeline and FifteenSquared catch-up.
4. **Scripts:** headless CLI model pinned to `claude-fable-5` (nightly_run,
   run_prefill) — deliberate, from the 07-14 billing work.

## 3. DEPLOYMENT — keep these straight (memory dashboard-deploy-and-pruning)
- The dashboard **DEPLOY page is the ONLY path to the droplet.** The nightly does
  NOT push to justcordelia.com. So the droplet is only as fresh as the last manual
  Deploy click, and **the DBs must be uploaded whenever the latest puzzles need to
  be live** (the served set + enrichments live in the local DBs).
- The site is almost certainly running PRE-relaunch code (503 since May). Going live
  needs a full code+DB deploy (now that the manifest includes core/). Deploy flow:
  upload -> perm fix -> `systemctl restart cordelia`. The user will click Deploy.

## 4. LAUNCH-WINDOW STATE (11-15 July)
- Use the Pipeline page's new report (or the SQL) to see what's unpublished. As of
  end of session, TWO puzzles are short:
  - **times 5224** — O/S 1 (the trailing-"..." ELLIPSIS reversal charade; hand-
    solvable via the punctuation-selectable /hs grid).
  - **guardian 30059** — O/S 1 (**16a "9 on and on and on?"** is status=invalid
    with NO comment -> 410. Add a comment in /hs to serve it, or resolve it).
- Everything else 11-15 July is fully served. Prize Toughie 233 completed to 28/28
  during the session (resurrection verified — went 410 -> 200 and into the sitemap).

## 5. OPEN GO-LIVE ITEMS (see documents/LAUNCH_CHECKLIST_2026-07-17.md)
- Styled 410 page (Flask default now) — SEO-adjacent, worth doing in the review.
- Fresh GSC read before Friday.
- Everyman 4155-4159 backlog — cascade+prefill as extra stock, or leave out.
- Flip the 503 -> live (restore command in memory site-maintenance-mode-2026-05-15;
  ssh root@165.232.46.255).
- Finish walking times 5224 + guardian 30059 (§4).
- Internal-links sweep is largely DONE (browse/search/cross-links gated;
  puzzle->clue links are admin-only) — confirm nothing else links to a 410.

## 6. ENVIRONMENT
- ONE app on :5000 (`web/run_dev.py`, V2 venv). /solver mount = core.wfw_web.
- data/clues_master.db = clues + wfw_* tables; data/cryptic_new.db = reference.
  Both gitignored, backed up twice daily. Scraper JSONs ARE tracked in git.
- Sacred principle: never modify a working stage engine for an edge case — add.
