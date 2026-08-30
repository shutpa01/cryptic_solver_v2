# HANDOVER 2026-07-20 — hidden-reversal fix + GSC sitemap explained; ENGINE/SIGNATURE WORKLIST is NEXT

Cold-start document. Read this, then MEMORY.md. Plain English; **verify against the code
before acting** — memory/handover reflect what was true when written. This session made
one code fix (core/diagnose.py) and DB data changes only.

---

## 0. ★ STATE OF PLAY — LIVE vs COMMITTED vs OPEN

**Git:** branch `redesign`, HEAD = **4c63564e**, **ahead of origin/redesign by 3 — NOT
pushed** (b37e8877, 5082587e, 4c63564e).

**Committed this session (4c63564e)** — 5 files that had been sitting uncommitted:
- `core/diagnose.py` — fixed swapped positional args in the hidden-engine call (see §1).
- `core/wfw_web.py` + `scripts/prompts/nightly_prefill.md` — spoonerism provisional build
  path: the verify_db (AI/prefill) path accepts an un-vetted spoonerism pair as
  source='pending' (like an unsanctioned homophone); human commit path unchanged.
- `web/routes/admin.py` + `web/static/js/puzzle2.js` — solve-mode delete: `/admin/clear-answer/<id>`
  route (clears answer + frozen + parse) + puzzle2.js tombstone / add-to-grid-stays-open.

(`.claude/settings.local.json` left uncommitted — local config noise. Handover docs and
scraper JSONs are untracked by convention.)

**DB data changes THIS session (local, gitignored — go live only via a DB deploy):**
- `data/cryptic_new.db` → `indicators` table: added **"extracted from"** as a `hidden`
  indicator (norm_word 'extracted from', source 'paul'). This is the fix that made clue
  10080421 pass (§1).
- `data/clues_master.db` → clue **10080421** (Times 29599 15a PORTFOLIO) re-solved to a
  clean PASS (hidden_reversed). Its stale wrong prefill assignment + parse were cleared.

**Carried-forward, NOT yet deployed (from the 2026-07-18 EVENING handover — CHECK whether
the DB deploy has since run):** puzzle_grids rows (times/29598, telegraph/233), enrichment
healing (OS/ESP/VOL), today-of-that-day solves, and **3 tentative homophone pairs to
approve** (over~ova, re-sinned~rescind, dough~deau). If the DB deploy has NOT run, all of
that plus this session's two DB changes ship together. Verify against the droplet before
assuming anything is live.

**NEXT (the user's stated next task): review the ENGINE / SIGNATURE WORKLIST.** See §3.

---

## 1. WHAT WAS DONE THIS SESSION

### A. Hidden-reversal clue 10080421 fixed — and the real cause found
Clue: Times 29599 15a, "Collection extracted from soil of tropics to the west" = PORTFOLIO
(hidden reversal: s(OILOFTROP)ics reversed). The user saw it mis-labelled "hidden plus
literal" and feared the recurring three-word-span problem.

**It was NOT the span and NOT an engine bug.** Verified by running the real cascade
(`core.engine_registry.solve_clue_text`, keyword args): the hidden engine produces the
correct hidden-reversal reading but only at **pending**, with the warning
`indicator 'extracted from' is not a DB indicator`. The `indicators` table had the single
word "extracted" but not the phrase "extracted from". A hidden clue only PASSES with a
confirmed indicator, so it stayed non-terminal — and the nightly AI prefill then filed its
own wrong "hidden + 3 literals" reading over the top. Same phrase-vs-word class as
[[container-engine-phrase-blind-indicators]] / [[phrase-indicator-fix-batch1]].

**Fix (sanctioned enrichable-vocab route, NO engine change):** added "extracted from" as a
hidden indicator; cleared the stale prefill (`set_hs_assignments(c,cid,'')` +
`delete_parse`); re-solved via `wfw_web._resolve_one(cid)`. Now stored as a clean PASS
(hidden_reversed, "soil of tropics"→PORTFOLIO, def "Collection", no warnings). Memory:
[[hidden-reversal-phrase-indicator-gap]].

**RED HERRING fixed:** `core/diagnose.py:140` called `solve_hidden` with `is_link` and
`indicator_types` swapped (positional), producing a fake `TypeError: 'bool' object is not
iterable` that looked like an engine crash. Production uses keyword args and never crashed.
Fixed diagnose to keyword args. **Lesson: read parse WARNINGS from the REAL cascade, not the
diagnose tool's crash line.** (Diagnose still shows hidden as pending with "role-validity
predicates not initialised" — that's a diagnose-harness limitation, not production.)

### B. GSC sitemap "still shows the old one" — explained, nothing to fix
Fetched the live sitemap (browser UA — Cloudflare Bot Fight Mode 403s the plain fetcher):
`https://justcordelia.com/sitemap.xml` is a healthy sitemap INDEX, HTTP 200,
cf-cache-status DYNAMIC, three children (clues-1, puzzles, news) all lastmod today. Our end
is fresh. GSC's **Last read = 5/15/26, 99,966 discovered pages** — that date is exactly when
the site went into maintenance/503 ([[site-maintenance-mode-2026-05-15]]); the 503 returned
for ~2 months so Google froze at that snapshot and stopped re-reading. The 503 lifted
2026-07-16 (4 days ago); Google simply hasn't re-crawled the sitemap yet. **Recovery =
stability + time; do NOT churn URLs or delete/re-add the sitemap.** [[sitemap-slow-generation-fix]]

---

## 2. HOW TO RE-CLASSIFY A MIS-SOLVED HIDDEN/OTHER CLUE (the reusable recipe)

When a clue looks mis-classified (esp. "hidden plus literal"):
1. Run the REAL cascade, keyword args:
   `engine_registry.solve_clue_text(clue_text, enum_space(answer,enum), wiring, source=..., clue_id=None, direction=...)`
   and read `parse.warnings`. Do NOT trust `core.diagnose`'s crash lines.
2. A "not a DB indicator" warning ⇒ the indicator PHRASE is missing. Add it to
   `indicators` (cryptic_new.db) with the right `wordplay_type` — enrichable-vocab route,
   never an engine edit.
3. Clear the stale prefill so resolve-from-assignment doesn't rebuild the wrong reading:
   `store.set_hs_assignments(c,cid,'')` + `store.delete_parse(c,cid)`.
4. Re-solve via `wfw_web._resolve_one(cid)`; confirm `status='pass'`, `warnings=[]`.
5. DB changes are local — they reach live only on a DB deploy.

---

## 3. NEXT TASK — the ENGINE / SIGNATURE WORKLIST

The user's next focus is to **go through the worklist of engines/signatures.** Starting
material (verify each against current core/ before acting — some are dated):
- [[container-engine-phrase-blind-indicators]] — PROVEN: the container signature engine
  never consults phrase-level indicators (per-word gate). Its topic file holds a worklist.
- [[hidden-reversal-phrase-indicator-gap]] — this session; phrase-vs-word indicator gap in
  the hidden path. Likely the same failure family recurs across engines.
- [[phrase-indicator-fix-batch1]] — earlier batch of phrase-indicator adds.
- [[signature-tiers-built]] — PASS-tier (A/B-gated) vs PENDING-only tiers.
- [[diagnose-tool]] — `python -m core.diagnose <clue_id>` (now with the arg-order fix); use
  as a first look but confirm findings against the real cascade (§1 lesson).
- Open engine gaps in MEMORY.md: cyclic selection (no wrap in SPAN_RULES), Guardian stub
  span-join, clutch-loss fault.

**Sacred rule still binds:** never modify a working stage engine to fix an edge case
(CLAUDE.md rule 1). The phrase-indicator class is fixed by ADDING DB vocab, not by editing
engines. Where a genuine engine BUG exists (like the diagnose swapped-arg, or a crash),
treat it as a bug not an edge-case tweak, and proceed carefully.

To build a fresh worklist, the honest source is the live fail/pending query (see
[[feedback-leftover-process]] for the SQL pattern), grouped by engine/operation, not the
stale intermediate files.

---

## 4. PRINCIPLES RE-AFFIRMED THIS SESSION
- **Verify on the real path.** The "crash" was a diagnose-tool artifact; the truth came from
  running the production cascade and reading the parse warnings.
- **No SEO grandstanding.** The GSC answer was grounded in our own outage history (Last read
  = the 503 start date), not speculation. Recovery is stability + time.
- **Enrichable vocab in the DB is the fix, not engine surgery** — the moat is the manual
  vocab work behind the commit gate. [[quality-over-automation-is-the-moat]]
