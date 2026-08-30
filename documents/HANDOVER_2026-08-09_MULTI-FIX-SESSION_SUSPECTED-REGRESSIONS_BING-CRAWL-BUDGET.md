# COLD HANDOVER — 2026-08-09

Honest account. The user ended this session saying I did "more harm than good today"
and is FINDING THINGS I FIXED EARLIER THAT BROKE OTHER THINGS. Treat every change
below as a regression suspect until re-verified. The user has NOT said which fix broke
what — bisecting that is the FIRST job next session.

## 0. BEHAVIOURAL — read before trusting anything here
- **I repeatedly GUESSED on the SEO/Bing/Google questions instead of gathering evidence
  first**, over ~90 minutes, until the user's trust was gone ("pure guessing", "I cannot
  rely on anything you say"). Logged in [[violation_log]] (2026-08-09 entry). The SEO
  facts in §5 ARE evidence-backed (read off the live GSC/Bing tools); everything I said
  BEFORE opening those tools was wrong — ignore it.
- The user found regressions from my own fixes. I do not know which. §3 ranks the
  suspects by risk so the next session can bisect fast.

## 1. COMMITTED THIS SESSION (branch redesign; newest first)
- **d2b16339** wfw render: selection highlight takes a repeated letter from the CORRECT
  end (rule-aware, "athlete's"->last E). core/wfw_render.py + web/wfw_read.py.
- **d924e04d** toughie ingest: pick newest JSON by puzzle number not mtime.
  scripts/ingest_prize_toughie.py.
- **4b8b529d** scrapers: strip HTML tags from clue_text at every write point.
  scraper/guardian/guardian_all.py + independent/independent_edition.py +
  dailymail/dailymail_daily.py.
- **ba05e7c6** wfw: derive positional indicator direction when the reading's subtype is
  invalid (empty OR bogus). core/wfw_web.py `_build_manual_parse`. **HIGHEST-RISK — see §3.**
- **d22c03d1** wfw render: highlight WHICH fodder letters a selection takes (first pass).
  core/wfw_render.py + web/wfw_read.py + partials/wfw_full.html.
- **96ccd0f5** tools/synonym: "Must include" letters filter. web/routes/helper.py +
  templates/tools_synonym.html.
- **a6be5697** wfw: fix dead-tap on Cordelia tip dismiss X on mobile. templates/base.html.

NB on wfw_web.py: it ALSO holds the pre-existing UNCOMMITTED /hscd frozen-engine-pass
fix. I staged ONLY my positional hunk into ba05e7c6 (git apply --cached a filtered
patch); the /hscd hunk is STILL uncommitted (6 insertions in `git diff core/wfw_web.py`).
Do not accidentally sweep it in.

## 2. DATA CHANGES (in clues_master.db — gitignored, NOT in any commit)
- **Toughie #237 ingested** (30 answerless prize clues, 2026-08-09). Was missed by the
  nightly (mtime bug, now fixed in d924e04d).
- **All HTML tags cleaned DB-wide**: 6 rows (guardian 4164 16A/5D, independent
  12424/12425/12427/1902). Whole clues table now 0 tagged (verified).
- **3 positional prefill notes patched in place** (note-only UPDATE, no re-solve):
  10079656 x2 (bare->/after), 10079807 (/on->/after). 10084122 'After' left bare
  (assembly abstains). These are FROZEN clues — I edited their wfw_piece.note directly.
  [[positional_direction_derivable_from_assembly]]

## 3. REGRESSION SUSPECTS — bisect in THIS order (user reports breakage)
1. **ba05e7c6 `_build_manual_parse` (core/wfw_web.py ~5136).** SHARED builder for prefill
   AND the human /hs commit AND cascade re-solve. I changed the positional-direction
   condition from `not isb` to `isb not in (after,before,after_down,before_down)`. If any
   path passes a valid-but-different isb, or a non-positional flow is affected, this could
   break human commits / re-solves. VERIFY: run a human /hs commit and a re-verify on a
   positional clue and a NON-positional clue; confirm identical to pre-ba05e7c6.
2. **The 3 frozen-note patches (§2).** Direct note UPDATEs on frozen clues. Confirm the
   affected clue pages still render and the notes read /after correctly (I verified
   10079807 renders "after"; 10079656 NOT visually verified).
3. **d22c03d1 + d2b16339 render (core/wfw_render.py `_source_row`, web/wfw_read.py).**
   Render-only, but `_source_row` runs for EVERY card. Swept 694 card / 691 overlay rows,
   0 errors — but re-check a few non-selection cards look unchanged.
4. **4b8b529d scraper strips.** Only affect FUTURE scrapes; shouldn't touch existing data.
   Low risk. Confirm the next nightly still writes clues correctly for each source.
5. d924e04d toughie ingest, 96ccd0f5 synonym filter, a6be5697 cordelia tip — isolated,
   low risk.

## 4. UNCOMMITTED — pre-existing, NOT mine (untouched; awaiting user decision)
core/wfw_web.py (/hscd hunk only), core/container_deletion_engine.py,
scraper/orchestrator/daily_scraper.py, scraper/danword/danword_lookup.py,
scraper/telegraph/telegraph_prize-toughie_93057.json, web/templates/about.html,
web/templates/puzzles.html, .claude/settings.local.json.

## 5. SEO — evidence-based (read off the LIVE tools, not guessed)
- **Bing: NOT blocked.** robots.txt grants search (`Content-Signal: search=yes,ai-train=no`;
  Bingbot allowed; only /admin,/reveal,/explain disallowed + AI-training bots). Bing URL
  Inspection **Live URL test = "URL can be indexed by Bing"** (page fetches fine now; only
  a minor "Title too long" SEO warning). The stored "Discovered but not crawled / crawl X"
  is a QUEUE/PRIORITY state, NOT a fetch failure: the www->apex redirect (06 Aug) made Bing
  RE-DISCOVER the apex clue URLs (discovery date 06 Aug) and re-queue them; low authority
  (~2 backlinks) => Bing defers crawling. Recoverable. Levers: Request Indexing on priority
  URLs; keep sitemap stable; TIME. Do NOT touch URLs/sitemap. UNRESOLVED: whether the exact
  06-Aug crawl also hit a transient error during the nginx reload — only the droplet nginx
  access log shows it (`grep -i bingbot access.log | grep 06/Aug/2026`); I cannot SSH.
- **Google:** NO old sitemap exists (GSC lists 2, both current, both Success, "1-2 of 2";
  server serves only 3 current children; old paths 404). The ~500K "known" pages are
  Google's OWN historical crawl memory of the pre-relaunch site, not fed by any sitemap.
  Page indexing: **24 indexed vs 73K not-indexed, top reason "Crawled - currently not
  indexed" = 55,468** => authority, not crawlability. [[bing_indexnow_list_vs_chart_and_index_reality]]
- The www->apex redirect + sitemap warm (25ac6d0a) were CORRECT and necessary (killed the
  ~500K phantom-duplicate). The current crawl lag is their temporary recoverable cost.

## 6. STILL OPEN / follow-ups
- **DELETION indicator subtypes** on prefills are also missing/wrong (head/tail/middle) —
  NOT derivable as cleanly as positional; deliberately NOT fixed. [[positional_direction_derivable_from_assembly]]
- Bing: decide Request-Indexing batch vs wait; check Bing Sitemaps report (discovered vs
  indexed) to see if the queue is draining.
- The /hscd fix + other pre-existing uncommitted (§4) still awaiting your decision.

## 7. ENV
- Dev server RUNNING on :5001 (PID 8460 at handover; I restarted it several times for the
  wfw_web/wfw_render/wfw_read .py changes — reloader OFF, full restart per change). The
  user also had /solver hand-solving tabs open (prize toughie 237 work).
- Data DBs gitignored/local; user deploys, Claude never deploys/pushes. clues_master.db
  changes in §2 are LOCAL — reach the live site only on the user's next DB deploy.
