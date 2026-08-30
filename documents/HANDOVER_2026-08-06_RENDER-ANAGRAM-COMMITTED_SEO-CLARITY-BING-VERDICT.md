# COLD HANDOVER — 2026-08-06

Honest account. Verified facts are labelled; anything unverified is called out.
This session I again asserted a guess as fact once (said the public site was
unaffected by the render bug — WRONG, it uses the same renderer) and the user caught
it. Verify the ACTUAL served path before claiming a surface is/ isn't affected.

---

## 0. WHAT IS COMMITTED THIS SESSION (branch redesign, HEAD = 756371ab)

- **756371ab** — render: show "anagram" from the anagram indicator, not a guessed
  shift/reversal. core/wfw_render.py + web/wfw_read.py. **NOT deployed.**

Prior committed context still not deployed: 03b97c5a (positional direction from
assembly), 0b0a59d0 (indicator gate), e972194c (indexnow ledger), 67c79f13 (clue alt).

## 1. THE RENDER ANAGRAM FIX (committed 756371ab) — detail

Trigger: clue **10083585** (TIMES 29614, 13ac, "What helps broadcast short advert
round state?" -> RADIO WAVE). User assigned last-letter deletion + container and made
"broadcast" double-duty def + anagram indicator; the card summary showed "letter shift"
(last->front) on advert->ADVER.

ROOT CAUSE (verified end to end):
- The type BADGE reads indicator NOTES, so it correctly showed "Container + anagram +
  deletion".
- The PER-PIECE line keys anagram off the SOURCE MECHANISM only. The fodder source
  advert->ADVER was stored mechanism 'selection' (the remove_last the user assigned),
  NOT 'anagram_fodder'. So the render fell through to `_transform_note`, which
  pattern-guesses the transform from the incidental letter order: ADVER's answer
  letters are RADVE = ADVER rotated (last->front), so it printed a letter shift.
- An anagram INDICATOR alone never changes the fodder source's mechanism, so the
  per-piece line and the badge disagreed.
- Mathematically the placement DOES require re-ordering: IOWA (state) only fits in
  order at answer positions 4-7, forcing ADVER's letters to 1,2,3,8,9 = RADVE. A clean
  "IOWA inside ADVER" cannot spell RADIOWAVE — hence RADVE is a genuine anagram of
  ADVER, which is what the user's "broadcast" anagram indicator declares.

FIX (render-only; NO stored-data or verifier change) — chose render-layer (Option A)
over save-layer marking (Option B) with the user's approval:
- Added `_anagram_note(parse, v, got)` (core/wfw_render.py) + `_anagram_desc(...)`
  (web/wfw_read.py). Shows "anagram [less X]" when: the clue names an anagram indicator
  (via _note_mechs) AND `got` is a sub-multiset of the value AND got != value AND got
  is NOT an in-order subsequence of value (an in-order survivor is a plain deletion) —
  and a real reversal (reversal indicator + exact reverse) keeps priority.
- core/wfw_render.py: wired into `_source_row` and `_piece_label`. **This renderer IS
  the clue-page card on BOTH the admin solver AND the public site** (public via
  core/wfw_card.stored_card; see web/serving.py + web/routes/clue.py:288). One fix,
  both surfaces.
- web/wfw_read.py: wired into `_describe` via `_segments` (new has_ana/has_rev computed
  once from parse["indicators"]). This is the SEPARATE lightweight hint-ladder
  one-liner (no core import; probes every clue on the puzzle page).

VERIFICATION (swept all 3603 status='pass' clues, both surfaces):
- Card: 6 clues change — target fixed; 5 improve (2-letter abbreviations CO/AG/MO/DR +
  LIAR that feed COMBINED anagrams, e.g. ECONOMIC = anagram of INCOME+CO, now read
  "anagram" instead of a coincidental "reversed"/blank). The 4 in-order deletions I
  first over-fired on (AND/CADDY/TALK/TURPIN) correctly stayed deletions after the
  subsequence guard. 0 regressions.
- Hint line: only 2 change (target + LIAR); the other 4 combined anagrams are already
  grouped by the public _segments ("anagram of …") so never reach _describe. 0 errors.
- Both surfaces now agree for the target: card + hint show "…ADVER anagram around
  (IOWA)". Confirmed via core.wfw_card.stored_card and web.wfw_read.load_breakdown.

Two design forks discussed and why A was chosen: A keeps the verifier untouched and
needs no re-save; B (mark the fodder anagram_fodder at save time) is cleaner and fixes
verifier soundness but needs the HS save to bind the anagram indicator to its fodder
(the payload doesn't link them today) and re-saving affected clues. If a future clue
needs the verifier to STOP passing a rotated container placement, revisit B.
[[render_anagram_indicator_per_piece_note]]

## 2. SEO / TRAFFIC INVESTIGATION (analysis only, no code)

User asked why the site gets Safari traffic but doesn't rank on Bing. Findings, all
verified live (Microsoft Clarity + GSC + Bing WMT numbers the user pulled):
- "Safari" traffic = MobileSafari (iPhone/iPad); 24 sessions / ~19 "unique users" of
  117 over 3 days. NOT the user (user is on Chrome/Windows).
- The traffic is NOT from Google search: GSC = 2 clicks total, both the user. It is
  Direct/Other/Referral. **Clarity "User ID" is an anonymous client-side token — Safari
  ITP wipes it (~7d), so "unique users" OVER-counts and per-person journeys can't be
  stitched. Don't trust the unique-user count.**
- FIRST LIVE PROOF the long-tail channel works: a recording's referrer was a **Bing
  search for an exact clue** ("bird-not-up-for-courtship-dancing") landing an engaged
  2-min session. Validates the puzzle-pivot keystone. [[bing_longtail_clue_search_confirmed_working]]
- Bing indexing (user's daily Submitted/Crawled/Indexed table, 7/18–8/3): **IndexedUrls
  == CrawledUrls every row** — every crawled page gets indexed, so indexing QUALITY is
  NOT the bottleneck; CRAWLING is (~16% of submitted). Trend is UP: ~9 indexed/day early
  July -> ~36/day late July/Aug (~4x). The scary "89/34/15" is a burst + regression +
  recent-day lag, NOT decline. Watch cumulative, not daily. Nothing to change.
  [[bing_indexnow_list_vs_chart_and_index_reality]]

## 3. STILL-OPEN / UNCOMMITTED (carried from 2026-08-04, NOT touched this session)

Working tree still holds (leave for their owners / a focused session):
- **core/wfw_web.py** — the /hscd frozen-engine-pass fix (08-04 §3). UNCOMMITTED.
- **scraper/orchestrator/daily_scraper.py** — HTML-tag strip at ingest. UNCOMMITTED and
  still the likely crux of the Independent tagged-clue recurrence (10083006/10083282).
- container_deletion_engine.py, danword_lookup.py, a telegraph json, about.html,
  puzzles.html, .claude/settings.local.json — all pre-existing, not mine.
See the 2026-08-04 handover for the full open list (CD false-pass guard, etc.).

## 4. ENV (unchanged)
- Dev: web/run_dev.py on :5001 (V2 venv .venv\Scripts\python.exe; reloader OFF, full
  restart per .py change; verify ONE listener on 5001). Admin card = core/wfw_render via
  core/wfw_card.stored_card; public clue page uses the SAME card (web/routes/clue.py) +
  a separate lightweight hint one-liner (web/wfw_read._summary). Two render surfaces,
  label logic hand-mirrored across core/wfw_render.py and web/wfw_read.py — keep in sync.
- Regression method used here (reusable): monkeypatch the new helper to a no-op to get
  the "before" render, diff against "after" across all pass clues. Caught the 4
  over-fires immediately.
- data DBs gitignored/local; user deploys, Claude never deploys/pushes.
