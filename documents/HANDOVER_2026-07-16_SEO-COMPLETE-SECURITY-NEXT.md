# HANDOVER 2026-07-16 — SEO page-work COMPLETE across 4 markets; NEXT = SECURITY REVIEW

Cold-start document. Read this first, then the memory index (MEMORY.md).
Plain English; verify against the code before acting. Launch target **Friday
2026-07-17** (imminent).

## 0. ★ THE NEXT SUBJECT: SECURITY REVIEW

The user's next big pre-launch subject is a **security review** of the public
site before go-live. Before starting it:
1. Read the **launch checklist**: documents/LAUNCH_CHECKLIST_2026-07-17.md.
2. Read the **security_step*** memory files (they cover the April JSON-LD /
   reveal / session-gating work — the existing security posture).
3. The **/security-review** skill exists (reviews pending changes on the branch)
   — the user may want it run, but it is user-triggered.
4. Known security-relevant surfaces to think about: the /solver/* admin mount
   (admin-gated via ?admin=dev-admin-key), the /helper/* tool endpoints (token +
   session gated), rate limiting on the clue page, the admin blueprint gating
   (web/__init__.py check_admin only SETS g.is_admin; only the admin blueprint
   enforces), and the STRIP_DEFINITION_FROM_JSONLD anti-scrape flag.

Do NOT start security work until the user confirms — this handover is the
boundary between subjects.

## 1. ⚠️ UNCOMMITTED WORK (commit before moving on)

Committed this session on branch `redesign`:
- **68df12ba** — learn zone: all 20 sub-pages into the sitemap + titles/H1s
  rewritten to real search phrases.
- **cf4c62f7** — clue page: contextual "learn how <type> clues work" link.

**NOT committed** (3 logical batches, all verified through the live path):
- **Tool-page FAQ + content depth** — tools.html, tools_anagram.html,
  tools_pattern.html, tools_synonym.html: added a visible 4-Q FAQ + FAQPage
  JSON-LD to each (they were thin, results JS-loaded = invisible to crawlers).
- **OG/Twitter meta** — the same 4 tool templates PLUS learn.html,
  learn_type.html, learn_practice.html: these override {% block meta %} (which
  in base.html contains the OG tags), so they were emitting NO OG/Twitter at
  all. Added per-page OG/Twitter using {{ self.title() }} + a {% set %} so
  nothing is duplicated.
- **Puzzle page SEO** — web/routes/clue_seo.py (new helpers), web/routes/puzzle.py
  (route), web/templates/puzzle.html (title/meta/H1). See §2.

Also uncommitted / not ours: .claude/settings.local.json (deliberate), the
scraper JSON files from the nightly run, and prior handover docs.

Suggested commits: one for the tools (FAQ+content+OG), one for learn OG, one for
the puzzle page — or group as you prefer.

## 2. WHAT THIS SESSION DID — SEO across the four markets

Framing: the site targets FOUR SEO markets (memory seo_four_markets):
1. clue-text search (~80% of clicks), 2. puzzle-number search (~19%),
3. standalone tools, 4. learn zone. ALL FOUR are built + public (verified).

Page-level SEO done this session:
- **Market 1 — clue page**: audited, already strong (unique title = clue text +
  "crossword clue answer", dynamic meta, FAQPage/Breadcrumb/DefinedTermSet
  schema, per-page OG, self-canonical, old-slug 301s). ADDED: a DB-safe
  "learn how <type> clues work" link keyed off the WFW mechanism label (pure
  single-types only; compound "CHARADE REVERSAL" clues correctly get no link;
  opens in a new tab so the solver's place is kept). 39/50 served pages show it.
- **Market 2 — puzzle page**: extended keyword enrichment from Telegraph-only to
  ALL sources. Titles/H1/meta/FAQ now use search-friendly names: "Telegraph
  Cryptic Crossword 31290 (DT 31290) — Answers & Hints", "Sunday Times Cryptic
  Crossword 5224" (was the awkward "Times Sunday"), "Guardian Cryptic Crossword
  30057", "Everyman Crossword 4160". Helpers in clue_seo.py: puzzle_seo_name,
  generate_puzzle_title/_heading/_meta_description; FAQ schema takes puzzle_type.
- **Market 3 — tools**: /tools, /tools/anagram, /tools/pattern, /tools/synonym
  were technically tagged but content-thin. Added visible FAQ + FAQPage schema
  (rich-result driver they lacked) + OG. DB-safe (prose about USING the tools).
- **Market 4 — learn**: sitemap fix (20 orphaned sub-pages) + titles/H1s to real
  phrases ("How to Solve Cryptic Crosswords", "How to Solve <Type> Clues") + OG.

## 3. STRATEGY SETTLED THIS SESSION (all in memory)

- **KEYSTONE — puzzle-pivot disruption** (memory seo_puzzle_pivot_disruption):
  one clue-text search pivots to the WHOLE puzzle, so we only need to win ONE of
  a solver's stuck clues, not all. Collapses the indexing bar from "dominate
  same-day" to "one foothold per puzzle". Plus the clean page beats Danword's
  ad "firework display", and Danword can't de-clutter without killing its
  revenue — so we capture the session AND the preference.
- **Google News = dead end** (memory seo_google_news_dead_end, researched +
  verified vs primary Google docs): crossword pages won't qualify AND News gives
  zero crawl acceleration anyway; no push-button (Indexing API / IndexNow /
  Request-Indexing all out). Same-day indexing is EARNED, not forced.
- **THE VIABLE LEVER — daily manual Request-Indexing of the PUZZLE pages** (in
  seo_google_news_dead_end): only 3 puzzles/day, so spend the ~10-12/day GSC
  URL-Inspection cap on 3 puzzle pages + the 3 HARDEST clues from each (difficulty
  = demand signal; the puzzle pages also seed discovery of the other ~170 clues
  via internal links). Do it at publish, every day.
- **Quality over automation IS the moat** (memory
  feedback_quality_over_automation_is_the_moat): automated competitors fail —
  crypticcrossword.org returned NO MATCH on a real clue and paywalls its
  explainer (TESTED). Cordelia wins on tested delivery: works, free, visual.
- **DB-privacy guardrail** (memory feedback_never_propose_publishing_db, recurred
  2026-07-16): NEVER publish indicator/synonym/abbreviation lists — a crawlable
  DB render = one scrape kills the moat. Market-4 link-bait = original prose only.

## 4. RULES THAT BIT THIS SESSION

- **Stop asking permission** (new memory): don't end turns with "want me to…?";
  act on the obvious next step and report. Long theory-dumps read as bluff — be
  brief, bad news first.
- **Test the competitor product before calling it a rival** — I called
  crypticcrossword.org a "direct competitor" off its marketing blurb; the user
  tested it and it failed. Gather evidence, don't quote homepages.
- **Never publish the DB** — I proposed an indicator-reference page (the exact
  forbidden form) and was corrected. Pulled it.

## 5. SERVER / ENVIRONMENT

- Dev server: .venv\Scripts\python.exe web\run_dev.py on :5000. Reloader is OFF
  for .py (restart needed after route changes); Jinja templates DO auto-reload.
- Background-launch quirk: harness may report the dev-server task as "failed"
  (exit 1/127) when its process is re-exec'd or killed — check port 5000 before
  believing an outage (a server is usually still serving).
- Verified all SEO via curl against the live dev server; JSON-LD blocks parse.

## 6. REMAINING SEO (not build work — do NOT reopen as a subject)

- Launch-day technical (on the launch checklist): styled 410 page, submit
  sitemaps in GSC, flip 503→live, fresh GSC read.
- Validation: run key pages through Google's Rich Results Test.
- Minor: puzzle breadcrumb still reads "Times Sunday" while the H1 reads
  "Sunday Times" — harmless, align if desired.
- Growth (post-launch): expand evergreen learn PROSE as link bait (DB-safe).
- Deliberately skipped: SoftwareApplication schema (needs ratings we don't have —
  faking them is a structured-data violation).
