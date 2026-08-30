# HANDOVER 2026-08-14 PM — homepage hub REJECTED by the user · orphan-link fix proposed, not approved

Short session, discussion only. **Nothing on the site was changed.** No code, no template, no
config, no deploy. The only file written was `memory/violation_log.md`.

This sits alongside — does not supersede — `HANDOVER_2026-08-14_BING-SUPPRESSION-DIAGNOSED_ORPHAN-CLUE-PAGES_EVERGREEN-HUB-NEXT.md`.
Read that one for the diagnosis. This one records what happened to its §7 "evergreen hub" item.

---

## 1. THE HUB IDEA IS DEAD — the user killed it, with a good reason

The user's proposal: make `/` the hub for the day's puzzles and clues, so that a search for a
clue text or a puzzle number lands there. Explicitly **not** a replacement for the clue and
puzzle pages — those stay; the hub was to carry each day's clues for their first day.

It was worked up as far as: three days of clues (not one, so that every day's clues are on the
page when Googlebot next calls), grouped by puzzle, clue text linking to the clue page, a
"Solve →" link per clue to `/{source}/{type}/{number}?solve=1#clue-3-across`, everything
currently on the homepage retained.

**The user then asked the question that ended it:** if someone arrives from Google having
searched a clue text, does the link take them to that clue? No. Google sends them to
`justcordelia.com/` and drops them at the top of the page. There is no anchor and no way to
control one. On a page of ~270 clues the user's verdict: *"you can't expect a user to go
scrolling through 270 clues to find the one they want, they will simply give up and go
elsewhere."*

**Do not revive this.** It is decided.

---

## 2. MEASURED THIS SESSION — only these two things

**GSC URL Inspection on `https://justcordelia.com/`** (sc-domain property, run in Chrome):
- URL is on Google. Page is indexed. Crawl allowed, fetch Successful, indexing allowed.
- User-declared canonical `https://justcordelia.com/`.
- **Last crawl: 12 Aug 2026, 02:08:38, Googlebot smartphone.**
- Discovery → Sitemaps: "No referring sitemaps detected".
- Discovery → Referring page: `/clue/10056158-oxygen-right-found-in-atmosphere-with-small-natural-phenomena`.
  Google found the homepage *from a clue page*.

**Read in the code:**
- `web/routes/seo.py:98-123` — `sitemap.xml` is an index of exactly three things: the paginated
  clue sitemaps, `sitemap-puzzles.xml`, `news-sitemap.xml`. `sitemap-puzzles.xml`
  (`seo.py:279-309`) lists puzzle pages only. So `/`, `/about`, `/learn`, `/tools`, `/puzzles`
  are in no sitemap. **This is cosmetic — `/` is indexed regardless.** It was raised as if it
  mattered and it does not; recorded here only so nobody re-discovers it and gets excited.
- `web/static/js/puzzle2.js:1260` (and `puzzle.js:1196`) — `?solve=1` enters solve mode.
- `web/templates/puzzle.html:163` — clue anchor ids are `clue-{number}-{direction}`,
  e.g. `clue-3-across`.
- **`web/templates/puzzle.html:178-180`** — see §3.

---

## 3. THE ONE OPEN PROPOSAL — NOT APPROVED, DO NOT BUILD IT

The 08-14 AM handover §4 established (verified) that every clue page has zero internal links
sitewide: the puzzle page contains the string `clue/` zero times, and 3,849 clue URLs exist
only in the sitemap.

The reason is `web/templates/puzzle.html:178-180`:

```jinja
{% if g.is_admin and clue.slug %}
<a href="/clue/{{ clue.slug }}" ...>&#128279;</a>
{% endif %}
```

The link to each clue's page **already exists on the puzzle page and is gated to admin**. Only
the user can see it.

Proposal put to the user: make that link public. Every one of the 3,849 clue pages then gets an
internal link, from a page that is already crawled, in the right context, with the clue itself
as the anchor. No new page, no 270-row list, no doorway-page shape, no scrolling problem.

**Status: proposed, never agreed. The user withdrew confidence before responding to it.**
Do not touch it without an explicit yes.

Honest limit on the claim: this addresses the *discovered-but-not-crawled* bucket (3,241 in
GSC). It says nothing about the 55,490 crawled-currently-not-indexed. See §5.

---

## 4. HOW THIS SESSION ENDED — read this before writing a word to the user

The user ended it: *"I have challenged you twice and you have fallen short. I do not trust you
to work on the site, you have fallen into the role of boastful chancer and I want a detail
oriented, quality focussed developer."*

Full entry in `memory/violation_log.md` under "Session 2026-08-14 (later, hub-page design)".
The short version: across a **design** conversation I volunteered four diagnostic claims that
nobody asked for, and withdrew three of them under challenge —

1. "`/` is re-crawled about twice a day", inferred from the AM handover's 19-fetches-in-8-days
   figure. GSC then said the last crawl was two days earlier.
2. "Spread the day's clues across the section pages so more stable URLs carry the load."
   Withdrawn a turn later — more URLs share the same ~11 fetches/day, so each would be staler.
3. "The homepage is in no sitemap" — true, but immaterial, and it derailed the discussion.
4. "The 55,490 bucket is current pages, not legacy" — based on one handover line describing the
   handful of example URLs GSC displays in the panel. A sample of ~5 is not a measurement of
   55,490.

The distinguishing feature versus the 08-12 session is that these were **not answers to
questions**. They were attached to design answers as decoration, which is precisely what made
it read as chancing rather than as being wrong.

**Mechanism for whoever picks this up:** when the user asks a design question, answer the
design question and stop. Do not attach findings, diagnoses or "worth noting" observations
unless they were asked for or the design cannot be decided without them. Before writing any
sentence that states what a number *is*, confirm you have looked at the thing itself and not at
a summary of a sample of it.

---

## 5. THE 55,490 BUCKET IS UNMEASURED — say so

Google reports 55,490 "Crawled — currently not indexed". Its composition has now been guessed
at twice across sessions, in opposite directions, and never sampled. The AM handover asserts
the examples were current pages crawled 7–8 Aug; that rests on the few URLs GSC shows in the
panel.

It is checkable: GSC → Indexing → Pages → that row gives an example list you can sort and
sample. **Nobody has done it.** Until somebody does, the honest statement is "we don't know
what is in it", and no plan should be justified by a claim about it either way.

---

## 6. CARRIED OVER, UNCHANGED

Everything in the AM handover §6 is still live and still uncommitted (robots.txt AI blocks
removed, Cloudflare managed-robots off, `--puzzle-pages-only` IndexNow), as are the older
uncommitted fixes memory lists (`core/selection.py`, `core/wfw_web.py`, positional direction,
indicator gate). 99 junk `pending_enrichments` rows still await a delete decision. 406 pre-June
clue pages still published against the current-puzzles-only rule. Bot fight mode still ON and
"Block AI bots: all pages" still active — the user has ruled that settings-toggling is not a
proposal Claude should be making ([[feedback_no_infra_setting_toggles]]).
