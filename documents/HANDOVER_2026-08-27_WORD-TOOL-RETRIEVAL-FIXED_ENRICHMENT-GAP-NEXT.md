# Handover — 2026-08-27

## The word tool now finds what the solver used · the ENRICHMENT gap is the next job

Cold-start document for the **publisher / word-tool thread**. Supersedes
`HANDOVER_2026-08-20_PM_WIDGET-FIXES-PUSHED_MOBILE-NEXT.md` for this thread.

The WFW/solver thread and the YouTube thread have their own cold starts. Do not
mix them.

---

## 1. Read this before you measure anything

**Three times in one session I asked the database a narrower question than the
solver asks, and three times a fact that exists looked missing.** Each time I
reported a wrong conclusion to the user before catching it. The order was:

1. Hand-written SQL on `norm_word` — missed every fact reached through an
   inflection.
2. Added inflections — still missed the **backwards** search: `core/live_db.py:92`
   also matches the clue word against the VALUE column and returns the key, so
   "marry means WED" comes from a row that reads *wed → marry*.
3. Used `LiveDB.get_synonyms` directly — still too narrow, because the engines
   wrap it in a second inflection loop of their own.

**The only honest oracle is the engines' own wiring.** Use it:

```python
from core.engine_registry import make_db_wiring
wiring = make_db_wiring()
wiring["lookup_all"](word)        # every (value, mechanism) the solver can use
wiring["defines"](phrase, answer) # does this phrase define this answer
wiring["indicator_types"](word)   # what roles the word can play
```

Anything those return is in the database by definition. Anything they do not is
a genuine gap. **Do not hand-write the query.** If a number in this document
disagrees with those functions, the functions are right.

---

## 2. What the user actually wants — the standing requirement

> "A word used in the explanation must always be in the DB and retrievable by
> the tools."

Two halves, and they fail separately:

- **Retrievable** — the tool must find a fact the solver used. Fixed for
  indicators this session; still broken for meanings (194 cases, §4).
- **In the DB** — when the solver works something out that is NOT in the DB, it
  must be queued for enrichment. **This is the open job.** 99 cases (§5).

---

## 3. What was fixed and verified this session

**The bug the user found**: Telegraph 31323, 1 across, "Trust rotter to maintain
right payment method" = CREDIT CARD. The card shows *maintain* as the container
indicator; the word tool's "Can indicate" said nothing.

**Cause**: the table holds *maintains* and *maintaining*, not *maintain*. The
solver matches inflections (`core/engine_registry.py:185`); both copies of the
word tool matched the exact key only.

**Fix**: both lookups now try the same variants and name the entry that matched,
so the line reads "Container as maintains" rather than implying a row that does
not exist. `core.inflect` and `core.contractions` are IMPORTED in both (leaf
modules, no imports of their own; the standing rule names only `web/`).

Files changed, all **uncommitted**:
- `web/routes/helper.py` — `_match_variants`, `_indicator_rows`
- `web/templates/partials/helper_results.html` — prints "as <entry>"
- `publisher/reference.py` — `match_variants`, `indicator_roles`
- `publisher/static/js/engine.js` — prints "as <entry>"

**Verified through both real paths, not in isolation**: on the site at :5001,
clicking *maintain* on the puzzle page gives "Could be an indicator for:
Container as maintains"; in the widget at :5003 the same click gives "Container
as maintains". A word with exact entries (*right*) still shows its four roles
with no "as" and no duplicates. 45 publisher tests pass; the overlay contract
test passes 64 clues, 0 failures.

**Also uncommitted from 08-25**, same thread: the full explanation now opens as
a panel over the whole widget with an ✕ and a Back button, pinned to the visible
viewport (`publisher/static/js/engine.js` `Sheet`, `.cg-sheet*` in
`publisher/static/css/engine.css`). The user confirmed it works on their phone.

**Not mine, do not claim or commit them**: `web/routes/browse.py`,
`web/templates/base.html`, `web/wfw_read.py`, `web/templates/privacy.html`,
`publisher/presentation/` were already modified/untracked from other threads.

---

## 4. The retrieval gap — 194 meanings the tool cannot show

Measured over every piece on a PASS clue whose mechanism is synonym,
definition, definition_by_example or abbreviation — 12,816 of them:

| | count |
|---|---|
| the tool finds it | 12,521 |
| in the DB, tool cannot reach it | **194** |
| not in the DB at all | **99** |

The 194 fail for the two reasons in §1: the word needs inflecting, or the pair
is stored the other way round and only the backwards search finds it.

**Proposed, NOT done**: extend the same fix to "Could mean" — inflections and
the reverse direction. Note the widget's Synonym tool has its own weaker variant
helper (`publisher/reference._variants`, plural and possessive only), left alone
on purpose so the two changes stay separable.

**Earlier indicator audit, for context** — 5,316 indicator pieces on PASS
clues: 5,163 exact, 42 reachable only by inflection (now shown), 89 multi-word
spans licensed by a shorter run inside them, **21 with no backing in any form**.
That list of 21 is in this session's transcript; two of them are not indicator
claims at all ("hard" recorded as the deleted letter H).

---

## 5. THE NEXT JOB — 99 values used in explanations that are in no table

The user's framing, and the right one: the solver derived these and **failed to
submit them for enrichment**, so they never entered the DB and no tool can ever
show them.

Examples: Child → SHRIMP · areas → BOROUGHS · Nastiest → SCABBIEST · Hook's
first mate → SMEE · much-loved bear → RUPERT · in place of circus → TENT ·
Pancake from → TORTILLA.

Three shapes are mixed in there and should be separated before any fix:
1. **Genuine missing facts** — Child → SHRIMP. These should have been queued.
2. **Span-shaped** — "Pancake from" → TORTILLA. The fact (*pancake*) exists; the
   recorded span carried an extra word. An authoring/rendering question.
3. **Mislabelled mechanism** — "close to heater" → R is a first letter recorded
   as an abbreviation; "King's Head" → K likewise. Not meaning claims at all.

**What is known about queueing**: `core/pending_store.py` is the only queue
module I read, and it is explicitly *only* for the Haiku definition fallback —
it writes `pending_enrichments` with `type='definition'`. `core/wfw_web.py` and
`web/routes/admin.py` also write that table. **I did not trace whether any path
queues a solver-derived VALUE.** That trace is step one.

`pending_enrichments` currently holds 11,614 rows; its columns are id, type,
word, letters, answer, clue_text, source, puzzle_number, created_at. Nothing was
written to any database this session.

To reproduce the 99, use the oracle in §1 against every synonym / definition /
definition_by_example / abbreviation piece on a PASS clue, comparing values with
all non-alphanumerics stripped from both sides.

---

## 6. Open, needing the user

- Commit. Nothing is committed. **Do not raise deploying.** The publisher is a
  dev prototype with no live target, and getting a `web/` change onto
  justcordelia is the user's own act through the dashboard — never propose it,
  never do it.
- The 99, and the 21 indicators — no DB write happens without approval.
- Whether to extend the fix to "Could mean" (194).

---

## 7. The demo — the user's own thread, not ours to drive

They are building it themselves in PowerPoint with recorded video, having tried
Supademo and rejected it ("if they cannot create decent documentation or demos
for their own product..."). Their method: review the puzzle in detail and pick
the clue that best showcases each feature. `publisher/presentation/` is theirs.

Answered along the way, so it is not re-litigated:
- Supademo does have text slides — they are called Chapters and can be inserted
  between any two steps. The naming is why it looked absent.
- The tools ARE configurable enough to claim it: the shell's More menu already
  persists per-solver options, `MATCH_OPTIONS` (config.py:84) is the "show
  possible answers at N or fewer" dial and is read in two places, and each
  publisher key already carries its own config block. What is missing is that
  the tool list is hard-coded twice (`engine.js` TABS, `telegraph.html`
  dropdown), and a publisher-set limit must be enforced server-side or a solver
  could switch on a tool the title has not paid for.
- The publisher ingest and the date gate are deliberately NOT being built until
  there is interest. The trigger is the first licensed puzzle file arriving, not
  the first pitch.

---

## 8. How the user wants to be worked with

- **Brief.** Short sentences, plain words, small numbers.
- **No invented vocabulary.** "You have started talking in immensely complex
  language using terms never before uttered." Do not name categories that only
  exist in your own analysis.
- A question is not an instruction. Diagnose, then ask.
- Verify through the real path and say which part was not verified.
- Dev servers: `web/run_dev.py` on :5001, `publisher/run_dev.py` on :5003, both
  with the reloader OFF — restart after any .py change, and check the process
  START TIME, not just that a listener exists. Their launchers kill stale
  listeners themselves, so just start them again.
