# LAUNCH CHECKLIST — go-live Friday 2026-07-17 (justcordelia.com)

Consolidated from HANDOVER_2026-07-15 §3. The prize-embargo item is REMOVED —
it was never real (prize puzzles serve exactly like everything else; see the
07-15 session). Tick as done.

## Serving rule — DONE (all one rule: served source + WFW pass OR invalid-with-comment)
- [x] Clue page 410 gate (committed cac43a12).
- [x] Puzzle-level display rule — a puzzle shows only when EVERY clue is served;
      built + verified on the puzzle page (410 public / admin bypass), browse
      list, and search (2026-07-15, working tree — see memory
      puzzle-level-display-rule).
- [x] Puzzle + news sitemaps gated to fully-served puzzles; every listed URL
      verified 200 (2026-07-15, working tree).

## Pre-flip — still open
- [ ] Styled 410 page (Flask default now) — cosmetic but user-facing.
- [ ] Internal-links final sweep — browse/search/cross-links now gated;
      puzzle->clue links are admin-only. Confirm nothing else links to a 410.
- [ ] Everyman 4155–4159 backlog — cascade + prefill as extra launch stock, or leave out.
- [ ] Fresh GSC read — current indexed count (last known ~37, early July; those
      are LEGACY pages that will 410 at launch, not WFW pages — WFW index starts at 0).
- [ ] Confirm `X-Robots-Tag: noindex` gone (removed 2026-07-04) and robots.txt
      allows crawl, at the moment of flip.
- [ ] Commit + push this session's work (needs explicit user yes).

## At flip (Friday)
- [ ] Turn off the 503 maintenance mode (restore command in memory
      site-maintenance-mode-2026-05-15; ssh root@165.232.46.255).
- [ ] Verify the live site returns 200 (curl the home + a served clue + a served puzzle).
- [ ] GSC: resubmit /sitemap.xml (currently "Couldn't fetch" due to the 503).
- [ ] GSC: URL Inspection -> Request Indexing for the top ~10 pages
      (home, /puzzles, each browse list, 2–3 flagship clue pages). ~10/day quota.

## After flip — monitoring
- [ ] **Check Google News appearance.** Publisher Center: publication "Cordelia"
      is REGISTERED; the manual approval was retired (March 2025 — Google News now
      auto-generates publication pages), so there is no approval to wait on. The
      real test only works once live: search news.google.com for the brand / a
      recent puzzle title, and look for a "News" performance report appearing in
      GSC. Its presence = we are being included. (Requested this check 2026-07-15.)
- [ ] Watch GSC: Pages (indexed vs crawled-not-indexed), Crawl Stats (is Googlebot
      back?), Sitemaps (fetched? URLs discovered?).
- [ ] Daily: request-index the day's key new pages if nudging; change one variable
      at a time; no timescale promises (rebuilding from a forgotten domain).

## Context (do not re-litigate)
- Week-only, no legacy: launch floor ~1,450 pages (≈1,413 clue + ~18 puzzle),
  grows ~85–90/day forever. Small-but-dense is the settled strategy.
- Content is short-life news — same-day indexing matters, which is why Google
  News is the primary channel. Normal crawl indexing is too slow for daily value.
