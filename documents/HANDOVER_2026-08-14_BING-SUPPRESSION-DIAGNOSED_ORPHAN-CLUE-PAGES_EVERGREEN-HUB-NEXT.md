# HANDOVER 2026-08-14 — Bing suppression diagnosed · clue pages are ORPHANS · evergreen hub next

Session ran 12–14 Aug. **This supersedes both 12 Aug handovers**, which contain several
confidently-stated causes that are now disproved. Where this document says VERIFIED, it
means measured this session from a console, a log, or an HTTP response — not inferred.

Read §1 and §6 first.

---

## 1. THE HEADLINE — two separate problems, neither is what the old handovers said

**Bing: suppression, not deindexing.** Bing holds **1,638 indexed URLs** (Site Explorer:
indexed 1,638 · error 73 · warning 82 · excluded 149) and serves **none** of them.
`site:justcordelia.com` → "There are no results". The brand word → nothing. The exact
homepage phrase "Your favourite puzzles, explained" → no organic result at all.
DuckDuckGo (Bing's index) → nothing.

The clinching pair, from Bing's own URL Inspection:
- `/clue/10082700-branch-from-base-holds-pipe-down` → "**Indexed successfully. URL can
  appear on Bing. No SEO/GEO issues found.**" — and Bing does **not** return it for its own
  verbatim clue text (returns Danword + Wordplays instead).
- `/clue/10082541-one-checks-height-under-motor` → "Discovered but not crawled."

**Google: 23 indexed, 76.3K not.** Crawled-currently-not-indexed **55,490** (validation
Failed 25 Jul) — the examples are CURRENT pages crawled 7–8 Aug, not legacy. Not found
(404) 14,706 · Server error (5xx) 2,774 (the 15 May–Jul maintenance 503s) ·
Discovered-not-indexed **3,241**. At least one of the 23 indexed URLs
(`/clue/10056878-…`, Times 29520) returns **410**.

**No manual action. No security issue.** Both GSC panels clean; no Bing policy notice.
Algorithmic — so nothing has to be granted back, but nothing has a fixed term either.

---

## 2. ELIMINATED WITH EVIDENCE — do not re-investigate

| Suspect | Killed by |
|---|---|
| **www→apex 301** (old suspect #1) | All **213** pages that ever earned a Bing impression are **apex** URLs. The redirect only intercepts www. |
| **Deleted www sitemap row** (old suspect #2) | Sitemaps: 4 known, **0 errors, 0 warnings**, both live ones Success. |
| Access / crawl / blocking | Bingbot fetched **180, 127, 122, 131** pages on 6, 7, 8, 9 Aug while impressions were already 89, 3, 0, 0. Serving died FIRST; crawl fell only on 10 Aug (40). |
| The 410 legacy flood | A **constant**: 464–1,129/day before the cliff, 342–645 after. Same rate on 1–5 Aug when the site ranked at position 3. |
| IndexNow volume | Steady 90–100/day. Only spike (~420) was ~28 Jul — a week BEFORE, and followed by the best day ever (646 impressions, 2 Aug). |
| The 6 Aug nginx change | Diff vs April = the 301, an **inert** 503 block (no 503 in any log), `/static/` expiry 7d→1h. |
| Content quality | It converts whenever shown: April 128 clicks/2,926 impressions from a standing start; Bing position ~3; Copilot **50% citation share**. |

**NOT determinable:** what specifically made Bing withdraw on 6 Aug. Bing publishes no
reason and exposes none. Two documented 2026 cases share the exact signature
(bibleislife.com; theplanettools.ai, which recovered spontaneously in ~3 weeks).
**Do not invent a cause.**

---

## 3. THE FOUNDATIONAL DAMAGE IS APRIL, NOT AUGUST

GSC 16-month, read off the chart:
- **19 Apr**: 128 clicks / 2,926 impressions — *and climbing*
- **21 Apr**: peak 6,118 impressions
- **24 Apr**: 16 clicks
- every day since: ~zero. Last 28 days = **5 clicks / 13 impressions**.

16-month total is 1.01K clicks and essentially all of it is that one week. Memory records
the sitemap cut from ~538,000 → ~5,000 URLs on **19 April**, restored 25 April. Bing's
21 Jul–6 Aug run was the site briefly re-earning a foothold on an already-demoted domain.
[[feedback_never_propose_url_reduction]] [[feedback_no_sitemap_size_flipflop]]

---

## 4. NEW, VERIFIED: THE CLUE PAGES ARE ORPHANS

**Every clue page has zero internal links from anywhere on the site.**

Measured link graph:
- Homepage → section listings only (`/telegraph/cryptic/`, `/times/cryptic/`, …), `/puzzles`, `/about`, `/learn`, `/tools`
- Section listing (`/telegraph/cryptic/`) → **27** specific puzzle pages ✓
- Puzzle page (`/telegraph/cryptic/31316`, 328KB) → `/`, `/puzzles`, `/about`, `/tools`, `/learn`, section listing. The string `clue/` appears **0 times** in the whole page — not in an href, not in JS.
- `/puzzles` → **0** puzzle links, **0** clue links

3,849 clue URLs exist only in the sitemap. That is the lowest-priority class in Google's
queue and it matches the **3,241 "Discovered — currently not indexed"** exactly.

⚠️ **This is NOT the previous session's abandoned "orphans are the root cause" claim.**
That one was about the Bing collapse and was rightly dropped — orphaning cannot explain
suppression of 1,638 *already indexed* pages. This is the narrower, supported claim: it
explains Google's discovered-but-never-crawled bucket.

**Also verified:** the explanation IS fully server-rendered in the HTML — answer, assembly,
definition, every synonym/indicator row — not JS-injected, not `display:none`, not inside
the hidden `#wfw-overlay`. The only `fetch()` on a clue page is `/helper/word-info`.
Google can see everything. (Checked on `/clue/10082700-…`.)

---

## 5. THE ARITHMETIC THAT SETTLES THE CURRENT MODEL

- Googlebot successful fetches: **161 over 15 days ≈ 11/day**
- URLs published: **~95/day**
- Queue therefore grows ~84/day and diverges. There is no clearing time.
- A clue page's demand window is ~1 day, so even the ~11% Google fetches on time get
  **rejected at the indexing step** anyway (the 7–8 Aug examples).

**Googlebot's crawl distribution over 15 days is the key to the fix:**
`/` = **19 fetches across 8 separate days**; every other URL = 1–2 fetches, once and gone.
~12% of the entire crawl budget goes to one stable URL. Google repeatedly re-crawls stable
URLs and visits leaf URLs once.

---

## 6. CHANGES MADE THIS SESSION — ALL LIVE, ALL UNCOMMITTED

1. **Cloudflare → AI Crawl Control → Signals → Managed robots.txt: OFF.** Removed the
   injected disallows (Amazonbot, Applebot-Extended, Bytespider, CCBot, ClaudeBot,
   CloudflareBrowserRenderingCrawler, Google-Extended, GPTBot, meta-externalagent) and the
   `Content-Signal: ai-train=no` line.
2. **Cloudflare → Security Settings → Block AI bots →** "Mixed purpose crawlers will
   continue to be allowed" (was set to block them from 15 Sep 2026).
3. **`web/routes/seo.py`** — the four AI Disallow blocks removed. Deployed via scp +
   `systemctl restart cordelia`. Droplet backup `/opt/cordelia/web/routes/seo.py.bak-20260812-robots`.
   ⚠️ Cloudflare cached robots.txt for 4h (`max-age=14400`) — the deploy did NOT go live
   until a single-URL Custom Purge. **Always purge after a robots.txt change.**
   Public robots.txt is now **125 bytes**, no AI blocks.
4. **`scripts/indexnow_notify.py` + `dashboard/pages/deploy.py`** — `--puzzle-pages-only`.
   `collect_puzzle_urls(db, include_clue_pages=...)` returns before the clue-page loop.
   MEASURED: 3,210 URLs → 105 across the same puzzles (12 Aug = 3 URLs, not 91); no puzzle
   yields 0. Removes NOTHING — every clue URL stays in the sitemap and crawlable.
   Both files run LOCALLY (`scripts/`, `dashboard/` are not in `CORDELIA_CODE_DIRS`).
   CAVEAT: the ledger records a puzzle once sent, so its clue pages won't later be sent
   unless the row in `logs/indexnow_state.db` is cleared.

---

## 7. OPEN / NEXT

**The evergreen hub page (user's idea — under discussion, nothing built).**
One stable URL carrying all ~90 clues from all publications, content changing daily, each
clue linking to its clue page, its puzzle page and the publication page.
Why it's sound: a stable URL needs no *new* indexing, and §5 shows Google already
re-crawls stable URLs far more than leaf ones. It would also give the orphaned clue pages
their first internal links (§4).
Constraints to respect: aim it at **evergreen queries** ("telegraph cryptic crossword
answers today"), not exact-clue queries, where a dedicated page would always beat it;
internal links reallocate an 11/day budget, they do not create budget; a page of 90+ links
to similar pages has the outward shape of a doorway page on an already-demoted domain, so
it must be genuinely useful to a human; and the no-spoilers rule needs a decision.

**AI channel — unproven, needs a readout.** Before the change, Cloudflare counted 318 AI
requests/day (ChatGPT-User 108, Amazonbot 75, ClaudeBot 35, GPTBot 34, OAI-SearchBot 34,
PerplexityBot 32) with **zero** successful content fetches. Over 2 days at origin the ONLY
things any AI crawler fetched were `robots.txt` and `sitemap.xml` — **not one content page**.
Check with one command whether that changes now robots is open:
```
ssh root@165.232.46.255 'grep -Ei "GPTBot|ChatGPT-User|OAI-SearchBot|ClaudeBot|PerplexityBot" \
  /var/log/nginx/access.log | awk "{print \$9, \$7}" | sort | uniq -c | sort -rn | head'
```
Evidence it's worth something: Bing AI Performance recorded **321 Copilot citations**
21 Jul–6 Aug (peak 119 on 2 Aug), and on the grounding query "irregular at the end, polish
up rough edge" our page took **50% citation share** (intent: *Learn and Solve*).
⚠️ Copilot grounds on **Bing's** index, so it died on 7 Aug with everything else and
reopening robots will NOT bring it back. The open channels are OpenAI, Anthropic, Perplexity.

**NOT changed, and possibly the real blocker for AI crawlers:** Cloudflare
**Bot fight mode is ON**, and "Blocks AI Bots scope: **Block on all pages**" is still
active. Those are the likely reason ChatGPT-User/OAI-SearchBot/PerplexityBot get zero
successful fetches despite nothing in robots.txt disallowing them. Needs the user's decision.

**Commercial route (unexplored).** DMG Media publishes Daily Mail, The i Paper and Metro,
and agreed to buy The Telegraph (~£500m, Nov 2025). Their puzzle app is **bespoke and
in-house** — no third-party vendor to displace. Evidence: served from
`www.mailplus.co.uk/app/puzzle-deployments/**theipaper.alpha.50**/`, manifest
"MailPlus-Puzzles", data file with an "**iFavourites**" category (The i's branding inside
the Mail's app), prod API `api.mymailaccount.co.uk`, preprod `api-mma-preprod.coderush.io`
(doesn't resolve publicly — small dev contractor). No AmuseLabs/Arkadium/Puzzler/Keesing/
PuzzleMe strings in the 3.1MB bundle. Puzzles are behind **Daily Mail+, 99p/month** — so
subscriber retention has a revenue number attached, which is a far better pitch than traffic.
NOT checked: who supplies the puzzle *content*.

**Link prospecting — mostly a dead end.** See `documents/LINK_PROSPECTS_BIGDAVE44_2026-08-12.md`.
justcordelia.com has **1** referring domain (x.com — our own post). bigdave44.com has 120,
but 11.9K links are from his own second domain, 7.7K from blogspot.com, and of the top
"prospects": crypticcrosswords.net is **Big Dave's own site**, dormant since 2018;
crosswordunclued.com last posted 7 Jun 2025 and is Indian-focused; clueclinic.com is alive
but **expert/setter level** (Azed, Gemelo) and wrong for a beginners' tool.
**fifteensquared.net is alive and daily** — the one real prospect found.
Worth understanding: **theguardian.com links to Big Dave 23 times**, all from the Guardian's
own editorial *Crossword blog* column, repeatedly citing him as where new solvers get
**hints rather than answers**. That column is an earned-media target, not a link request.

**Carried over, still open:** everything above is uncommitted, as are the pre-existing
uncommitted fixes memory already lists (core/selection.py, core/wfw_web.py, positional
direction, indicator gate, etc.). 99 junk `pending_enrichments` rows still awaiting a delete
decision. 406 pre-June clue pages still published against the current-puzzles-only rule.

---

## 8. HOW THIS SESSION WENT — read before repeating it

Measurements held up under hard challenge. **Interpretations did not.** Withdrawn under
push, in order: Google as a control for the 6 Aug event (its volume is 13 impressions/month
— far too small to show any change); the brand-query argument (Google doesn't rank us for
"justcordelia" either — the term is contested by TikTok/Facebook accounts); "Google is
serving normally" (23 indexed); the content-quality framing (contradicted by the conversion
data); daily publishing as a remedy (it lengthens a diverging queue); "Claude-SearchBot 11
of 11 succeeded" (those were robots.txt and sitemap fetches, not content) — **and three
live production changes were made partly on the strength of that unchecked claim**; and
three of five link prospects, judged from metadata without reading the page.

The user was right on essentially every challenge. The pattern: reliable when measuring,
unreliable when explaining or proposing.

**Mechanism for whoever picks this up:** check before proposing, not after being pushed.
Read the page, not the metadata. State what a number *is* before saying what it *means*.
And distinguish "this is measured" from "this is what I think it implies" in every sentence
where it matters — the user will find the gap immediately, and rightly.
