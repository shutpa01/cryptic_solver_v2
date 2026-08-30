# HANDOVER 2026-08-12 — selection middle rule widened, selection false-pass closed, Bing "blocked" verdict found

Session ran 11 Aug ~06:00 through 12 Aug. Everything below is verified through the
real code path or read off the live Bing UI; where something is unproven it says so.

---

## 1. OPEN — needs your decision before anything else

### 1a. 99 junk rows in pending_enrichments (awaiting your yes to delete)
My before/after engine sweep (§3) wrote 99 rows into `pending_enrichments` —
98 `type='definition'` + 1 `type='indicator'`, all `source IS NULL`, created
2026-08-11 07:22:41–07:52:36 UTC, every one traceable to a sweep-corpus clue.
I had told you the sweep would write nothing; that was wrong. I neutered
`store.persist`, `piece_fallback.finalize_pieces`, `finalize_homophone_pieces` and
`dd_enrichment.finalize_dd`, but the wiring's AI definition fallback
(core/engine_registry.py:456 `definition_fallback.make_fallback(ai_definition.define, store)`)
queues definitions through `PendingStore` independently and bypassed all of it.

Proposed cleanup, NOT yet run:

    DELETE FROM pending_enrichments WHERE created_at >= '2026-08-11 07:20' AND source IS NULL;

Confirmed to match exactly those 99 rows (every row in that window is source-NULL,
every one matches a sweep clue). Both DBs backed up 2026-08-11 08:20:08 beforehand.

**Lesson for any future cold sweep:** neuter the definition fallback too, or solve
with a wiring that has no `store`.

### 1b. Uncommitted code (all verified, none committed)
- `core/selection.py` — `_middle` widened (§3)
- `core/wfw_web.py` — JS mirror `selCands` case 'middle' at ~line 2349 (§3), and the
  selection tile-check in `_build_manual_parse` (§2)

---

## 2. Selection false pass — INTROIT (FIXED, uncommitted)

Guardian 30082 13a "Sacred music at heart of saintly shroud rite" = INTROIT
(clue 10084568) had been committed as a **frozen manual PASS** on a reading where a
single letter S made all seven answer letters: the three fodder words were tagged as
ONE selection, rule 'first', value 'S', pos [1..7].

**Root cause.** `_build_manual_parse` validated a selection piece against the RULE
only (core/wfw_web.py:4893 — is the value in `_selection_candidates(phrase, rule)`?
'first' of SAINTLYSHROUDRITE = S, so it passed). The "must spell the tiles it lands
on" check at core/wfw_web.py:4903 covers only synonym/substitution/letters/replacement
— selection was never in the list. Tile coverage and word coverage were both satisfied,
so nothing objected.

**Fix.** New block in `_build_manual_parse` before the anagram check: a selection's
value minus its declared `cut` must equal, as a multiset, the tiles it claims. It sits
in the shared builder, so it guards /hsmanualcommit, core.prefill_commit and
/prefillconfirm alike. Verified by POSTing the exact payload to the real
/hsmanualcommit route (refused with the new message) and by re-validating all 381
stored selection payloads — exactly one newly refused (this clue).

**Second gap found, NOT fixed.** `/hsmanualuncommit` lifts the freeze but does NOT
clear a false pass: `_resolve_one` → `_resolve_from_assignment` returns early at
core/wfw_web.py:2230 ("a confirmed PASS is the human's verdict — never rebuild or
downgrade it") whenever an assignment exists. Clearing it took `store.delete_parse`
plus a second `_resolve_one`. **If a bad manual pass ever needs undoing, uncommit
alone is not enough.**

**Pre-existing, unactioned:** two other frozen passes carry selection payloads that
no longer validate against the rule check — 10084126 EPISODE ('final' (last) = E, rule
derives L) and 10084137 LASAGNE ('an' (first) = L, rule derives A), both committed
2026-08-09. Neither was touched.

---

## 3. core.selection `_middle` widened — SWEPT, 0 regressions (uncommitted)

**The problem.** `_middle` returned exactly ONE candidate — the single centre letter
(odd length) or central two (even). So "at heart of saintly" = INT was underivable,
by hand or by engine, and INTROIT was unfilable in /hs at all. You called this out as
an arbitrary word-length rule, and you were right.

**Why widening is safe (your question, and it's the crux).** Every consumer is
ANSWER-DRIVEN: core/charade_signature_engine.py:145-146 keeps a candidate only if
`answer.startswith(value, pos)`; core/container_signature_engine.py:95-96 only if it
completes the insertion to the exact answer. The slot is also indicator-gated
(charade_signature_engine.py:141). core/selection.py:58-60 already states this
contract — "Each rule yields one or more CANDIDATE selections... so a loose rule
cannot fabricate". Extra candidates can only be discarded.

**The change.** `_middle` now returns every centred run bar the whole word, shortest
first (so the previous sole candidate is still tried first):
SAINTLY → N, INT, AINTL · SHROUD → RO, HROU · RITE → IT · BIRD → IR (the docstring's
own example, unchanged). JS mirror `selCands` updated to match.

**The sweep.** 1,077 clues over 180 days: all 477 carrying one of the 20
middle-subtype indicators, plus 600 random controls carrying none. Two cold passes
(clue_id=None, no persist), the widened one patching `SPAN_RULES['middle']`
in-process so both could run side by side. Script + jsonl kept in the session
scratchpad (middle_sweep.py, before.jsonl, after.jsonl).

Result: **4 changed, all fail→pass, 0 regressions, 0 movement in the controls.**

| clue | before | after |
|---|---|---|
| TIGON g29982 "Leave in middle of stint, being cross" | fail | pass — GO inside TIN (STINT centred 3) |
| AT FIRST t3369 "Trees in centre of Seattle, initially" | fail | pass — FIRS inside ATT (SEATTLE centred 3) |
| ANTHROPOLOGY t31308 "…exposed trope in collection of writings" | fail | pass — ANTHOLOGY around ROP |
| INTROIT g30082 | fail | pending (engine now reaches it too) |

All three new passes are correct readings on inspection. Every change sits in the
middle-indicator group.

**INTROIT is now committed correctly** through the real route: saintly=INT,
shroud=RO, rite=IT (all mechanism selection), "at heart" the selection/middle
indicator, "of" the link, "Sacred music"=INTROIT the definition. Status pass,
solved_by manual, frozen. Commit reported "2 already in DB" — no new reference-DB
material.

---

## 4. Nightly 11 Aug — ran late, by hand, completed

The 02:00 task did not fire (machine off/asleep), caught up at 06:05:05 and was killed
within seconds — Task Scheduler `LastTaskResult` 3221225786 (0xC000013A, console/Ctrl-C
termination), leaving a 0-byte log. Re-ran scripts/nightly_run.bat manually 06:10–07:00,
exit 0.

- Scraped: telegraph 31315 (32), toughie 3737 (32), times 29618 (30), guardian 30082 (28),
  independent 12431 (30), dailymail 17941 (26). Prize Toughie ingest a no-op (#237 present).
- Cascade (90 serving-paper clues): pass 14, pending 16, fail 60.
- Prefill: 70 of 76 filed PENDING, all through `file_pending_prefill`, all passing the
  gate first time. 6 left blank with reasons: FOOTFAULT (split-word device), BLUFF
  (triple definition), INTROIT (the middle-rule gap — now fixed, §3),
  FIFTH COLUMNIST (CD), NEW ORLEANS (reverse anagram), OZONE (source of ONE
  unestablished).
- Diagnosis: 42/63 pass cold (67%), 19 fails all classified. **One-click backlog is now
  25 rows / 27 clues and has never been actioned**; telegraph 3381 23d still has an
  EMPTY answer in the clues table. WORSTED's derived signature fired on the snapshot
  and awaits your click. Detail: logs/diagnosis_2026-08-11.md, logs/prefill_2026-08-11.md.

---

## 5. &lit in the hand solver (answered, no code change)

There IS an &lit checkbox in /hs, immediately left of Commit (manual):
core/wfw_web.py:3038. Tag the wordplay as normal, do NOT tick a separate definition
(the whole clue becomes the definition, core/wfw_web.py:5235-5242), tick "&lit
(all-in-one)", Commit. Files operation 'andlit', frozen, verdict forced to **pending**
(core/wfw_web.py:5295).

Two standing gaps: the nightly prefill cannot file an &lit at all (core/prefill_commit.py
never passes the flag), and **an &lit has no Confirm path** — the Confirm control only
renders for `solved_by='prefill'` (core/wfw_web.py:5630-5631) and an &lit commits as
'manual', so it sits at pending indefinitely.

---

## 6. Bing — the IndexNow counter is stuck, and one URL reads "blocked"

Read off the live Bing Webmaster Tools UI on 12 Aug.

- **IndexNow → Indexing Insights:** Submitted 2.5K, Crawled 380, Indexed **320**.
  Bing's own tooltip: "The number of URLs submitted to IndexNow and indexed in Bing" —
  i.e. a per-submission subset, NOT the site's index size. You observed 320 has not
  moved for days; the chart agrees (the indexed line falls to ~0 after 2 Aug).
- **Sitemap index coverage (sitemap-clues-1.xml), last data point 7 Aug** — the report
  is 5 days stale, nothing after the 7th. 4 Aug: indexed 1.5K. 5 Aug: 1.8K. 7 Aug: 1.7K.
  So the rise happened ON 5 Aug and predates everything we shipped (6 Aug sitemap
  cache warm 25ac6d0a, 6 Aug www→apex 301, 7 Aug render fixes) — **not attributable to
  our changes.** Figures are rounded to 0.1K; exact data is behind Download all.
- **Why not indexed (same report):** Discovered but not in index **1.3K**;
  Content quality **62**; Not yet crawled **38**; Cannot crawl the content (403, 5xx) **2**.
- **URL Inspection, justcordelia.com/guardian/cryptic/30082** (IndexNow-submitted
  11 Aug 08:56):
  - Bing Index tab: **"Blocked — URL cannot appear on Bing. The inspected URL is known
    to Bing but has some issues which are preventing us from serving it to our users."**
  - Live URL tab (tested 12 Aug): **"URL can be indexed by Bing", "No SEO/GEO issues
    found"**.
  This CORRECTS the 08-09 note that Bing is "NOT blocked" — for this URL the stored
  verdict is blocked while the live page is clean.
- Site Explorer snapshot: Indexed 1.6K, Error 73, Warning 82, Excluded 149, URLs 1.9K.
  (The 07-30 finding still stands: this count includes old pages that now 410, so only
  the direction is meaningful.)
- Sitemap last crawled **17 July** — nearly four weeks — while IndexNow pings continue
  at ~90–100/day.

**Only two genuinely actionable buckets**, both behind the Download all button on the
sitemap coverage page (a file download — ask first):
1. the 2 URLs returning 403/5xx — nginx allowlists Cloudflare only, so a Bingbot fetch
   that bypasses Cloudflare would produce exactly this. Worth identifying.
2. the 62 under Content quality — the one bucket where a page change could plausibly move.

Everything else is Bing declining to serve pages it already has.

---

## 7. Behavioural notes from this session

- I claimed the sweep "writes nothing to the database". It wrote 99 rows (§1a). Verify
  the claim, don't assert the intent.
- I appended the backlinks/authority count to three consecutive SEO answers unasked.
  User: "Why do you keep going on about backlinks, you know we can't get backlinks to a
  site that does not exist to users." New memory: feedback_stop_repeating_backlinks.
- A server restart does not re-solve anything: after clearing a bad parse, the OLD
  still-running dev process rebuilt it from the assignment one minute before the user
  restarted, so the same wrong reading was still on screen. Check what the process was
  running, and when it started, before explaining a stale screen.

---

## 8. Suggested next steps

1. Say yes/no on the 99-row delete (§1a).
2. Decide whether to commit core/selection.py + core/wfw_web.py (§1b) — sweep evidence
   is in §3; nothing else in the tree depends on them.
3. The one-click backlog (25 rows / 27 clues) has never been actioned and grows nightly.
4. Optional: pull the 403/5xx and content-quality URL lists from Bing (§6).
5. Two 09-Aug frozen passes with invalid selection payloads (§2, EPISODE + LASAGNE).
