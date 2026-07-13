# HANDOVER 2026-07-13 — relaunch prep started: week-only serving BUILT; Everyman recovered

Cold-start document for the next session. Read this first, then the memory
index. Plain English throughout; verify against the code before acting.

## 1. Where the repo stands

- Branch `redesign`, working tree CLEAN except `.claude/settings.local.json`
  (local Claude settings, deliberately uncommitted). SEVEN commits ahead of
  origin, **NOT pushed** (origin is at e7a9b72d, 2026-07-11):
  - 177a9443, b6d772c6, dad15d23, 4f3c13f2 — the 2026-07-12 batch (span-join,
    prefill-pending flow, Re-run restore, handover doc).
  - **1adf2a90** — /hs Assign now files reusable pieces (definition / synonym /
    abbreviation / indicator) to the reference DB on EVERY Assign click, with
    a green "Added ..." ack next to the button as proof of arrival; plus the
    manual/prefill hidden-clue highlight restored (core/wfw_card
    _manual_hidden_line — see §4).
  - **cac43a12** — the relaunch site batch (see §3, the big one).
  - **c2a59080** — Everyman scraper fix + catch-up 4155–4160 (see §5).

## 2. THE DAILY PROCESS (unchanged, validated)

Nightly at 02:00 (CrypticSolver_NightlyRun): scrape → cascade → diagnosis →
prefill files PENDING commits. Morning: user reviews on the clue page
(Confirm / hand-fix), publishes. The FIRST unattended new-flow nightly ran
clean on 2026-07-13; the user reviewed all three dailies in ~45 careful
minutes and is happy ("excellent job on the prefills").

New since 07-12, both proven live:
- **Assign = DB write.** Every /hs Assign files the reusable pieces
  immediately (route /hssave). No green message next to Assign = the save
  never reached the server — that is the debugging tell. Wrong values are
  pruned via the delete facility. Commit unchanged (harvests + freezes).
- **Pure-DD flow works**: assign both definitions → they file → user clicks
  Re-run → DD engine passes (16a LIGHTNING STRIKE proven). Re-run works only
  on UNFROZEN clues — committed clues stay frozen, per the user's rule.

## 3. RELAUNCH PREP (started 2026-07-13 — the main event)

**Settled decisions (user, final):**
- **Week-only, no legacy**: a public clue page exists IFF the clue has a WFW
  pass parse AND source in (telegraph, times, guardian). Everything else
  answers **410 Gone** at its stable URL and RETURNS at the same URL when
  solved (resurrection). Legacy corpus stays in the DB, never served.
- The week is the LAUNCH FLOOR, not a rolling window: target launch
  **Friday 2026-07-17** with seven days of 100%-WFW puzzles; the archive
  grows daily from then on. Nothing ages out.

**Built + committed (cac43a12), all contract-tested:**
- web/serving.py — is_served() = THE rule (source allowlist + the card
  renders); used by the clue route, the sitemap, cross-links. SERVED_BROWSE =
  the six public browse options.
- core/wfw_card.py — stored_card(clue_id): the public clue page serves the
  IDENTICAL card the solver renders (user rejected a pastel re-implementation:
  "muddled and faint... base it on what we have"). Light imports, no engine
  wiring. SCREENS + _manual_hidden_line live here now; wfw_web imports back.
- Clue page: real card embedded in crawlable HTML; answer/definition/type/
  explanation boxes REMOVED (the card carries all four); card's clue line
  suppressed except lit hidden runs; PASS/engine/prov chips stripped
  server-side; tier flowers gone (also gone from the puzzle page).
- Sitemap: clue sitemaps list EXACTLY the pages that exist — 1127 URLs, every
  one verified 200; the 2 junk pre-schema pass rows verified 410. This is one
  shared rule with the page gate, NOT a repeat of the April cutoff mismatch.
- Home/puzzles pages (public): only DT, DT prize, Times, Times Sunday,
  Guardian, Observer Everyman. Admin sees everything.

**Launch-window state (measured 2026-07-13):** Jul 13 all three dailies READY
(guardian 30057, telegraph 31290, times 29593). telegraph 3377 READY. Gaps:
31289 needs 25a BACKSTAGE (user's Mark verdict) + 1d BALTIS (marked invalid);
times 29592 + 5224 are answerless prize puzzles (user solves or they wait).

**OPEN DECISIONS (the user's, raised and waiting):**
1. **Prize embargo**: as built, a solved prize puzzle's clue pages (3377) would
   be live WITH answers during the competition window. Standing rule 5 says
   prize puzzles in embargo are NOT commodity. Does the serving rule need an
   embargo hold?
2. **Puzzle-page scope**: do OLD puzzle pages stay served? (Puzzle + news
   sitemaps untouched pending this.)
3. Styled 410 page (Flask default now) — cosmetic.
4. Cascade+prefill the five answered Everyman weeks (4155–4159) as extra
   launch stock? ~5 morning reviews of 28 clues.

**NEXT BUILD ITEM (approved direction, not started): internal-links sweep** —
browse pages and puzzle pages still emit links to unserved clue URLs (now
410s). Filter them through web/serving.is_served like everything else.

## 4. Faults found + fixed today

- **16a "nothing works" mystery**: the user's Assign clicks never reached the
  server (cause unknown, browser-side; the identical request succeeded when
  replayed). The green-ack message added to Assign exists to catch this
  instantly next time. The DD then passed on the user's Re-run click.
- **Hidden highlight regression**: prefill/manual hidden parses (operation
  'manual', host phrase as one raw piece) bypassed the hidden screen, so the
  lit letters (de[BRIE]fs) vanished. Fixed renderer-side: _manual_hidden_line
  derives the run (forward or reversed, strictly inside the host piece's own
  letters, only when a hidden indicator is present). Also fixed: raw
  "DOUBLE_DEFINITION" badge label.
- **Everyman scraper stalled six weeks** — see §5.

## 5. Everyman (recovered, committed c2a59080)

The scraper always existed (scraper/guardian/guardian_all.py, Observer/
slowdownwiseup API) and runs EVERY night inside the guardian scrape step. It
stalled at #4155 on 2026-06-07: the Observer began comma-formatting titles
("Everyman 4,155") → number regex parsed 0 → mismatch guard → two misses stop
the loop. Silent in the nightly logs (the log filter drops "Trying #" lines).
Fixed: comma-strip in parse_everyman_puzzle + precise escaped-JSON uuid
extraction ("type":"puzzle","uuid") with the old probe as fallback.

Catch-up done: 4155–4159 in the DB with ALL answers (embargo strips only the
live week; previous_solution backfill healed the chain) + grids. **4160
(yesterday's, competition closes 19 July) is in answerless with its grid —
the user solves it manually like the other prizes.** From Sunday 4161 onward:
automatic. The user is popular-demand-motivated: "everyman crossword 4148"
was a proven GSC click winner.

## 6. Site server / environment notes

- Dev reloader is OFF: code changes need a manual restart
  (.venv\Scripts\python.exe web\run_dev.py — the V2 venv; the user often
  launches with the AI_Solver venv, which works but is the wrong interpreter).
- The public site now lazily imports core (light render chain only) for the
  clue-page card — the /solver mount precedent. web/wfw_read (hints path)
  stays core-free.
- Old scheduled task "Cryptic Solver Nightly Pipeline" still needs deleting
  from an ADMIN shell: schtasks /Delete /TN "Cryptic Solver Nightly Pipeline" /F

## 7. Rules that bit today (full versions in CLAUDE.md + memory)

- "Base it on what we have" — don't re-implement a renderer the user already
  likes; serve the real one.
- The sitemap scar tissue: sitemap and page gate must share ONE rule; never
  list a URL that doesn't 200.
- Evidence before theory: the Everyman stall and the 16a mystery were both
  solved by reproducing with the real functions/requests before touching code.
- Plain writing, short sentences, bad news first. Questions are not
  instructions. The user clicks Re-run; Claude never re-runs for score.
