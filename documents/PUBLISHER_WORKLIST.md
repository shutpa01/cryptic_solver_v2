# Publisher widget — worklist

Deferred items, newest first. Opened 2026-08-18. The rule above all of these:
**the live system is the specification** (CLAUDE.md, top). Every entry below
either replicates something the live site already does, or fixes something the
widget shows that the live site would not.

---

## 1. Corpus junk — spurious entries in the match lists

Raised 2026-08-18 after `SPRAYER} –`, `IDING`, `SOOGIE`, `SPIRIC` appeared in
pattern results and `(rum?)`, `a black`, `and the` in the ⓘ lookup.

Evidence gathered (read-only, nothing changed). Both source tables in
`cryptic_new.db` carry a `source` column, and it separates two different
problems:

- **Encoding damage, from the blog scrapes.** 3,099 rows of
  `definition_answers_augmented` hold non-letter characters in the answer:
  fifteensquared 1,305, telegraph 470, guardian 282, times 276. They read
  `WET THE BABY?S HEAD`, `QU?BEC`, `OBJETS D?ART`, and one is a raw HTML SPAN
  around SECTIONS. **Repairs, not deletions** — the answer is recoverable.
- **Fragments, on the synonym side.** 9,567 rows of `synonyms_pairs` hold junk
  characters, almost all `api_mw_mesh` (6,274) and the 510,426 rows whose
  source is NULL: `in france?`, `play group?`, `, on the whole`,
  `airship&nbsp;`. A further 17,385 have a word side starting with a function
  word. `a black` and `and the` are both NULL-source rows; `(rum?)` is one
  definition row from `clues_master_sync`.
- **Bogus words** (`IDING`, `SOOGIE`, `SPIRIC`, `GLENERANGES`) are single rows
  from a single blog source each. That rule yields 34,189 candidates out of
  131,198 distinct answers — a review queue, not a delete list.

Proposed order, and the reason for it:

1. **Display hygiene in `publisher/corpus.py`.** Most of what shows in the
   widget is a display fault, not membership: the corpus matches on the
   letters-only form and then prints the raw string, so `SPRAYER} –` is really
   SPRAYER wearing scrape residue. Choosing the cleanest display per word fixes
   the visible class with no DB write and nothing else affected.
2. **A source-scoped repair pass** for the mojibake in `definition_answers_augmented`.
   Needs approval before any write.
3. **Leave the NULL-source synonym block alone** until 1 and 2 are done.

## 2. DONE 2026-08-18 — the live helper looked up by LOWER(word)

Found 2026-08-18 chasing "Jill's companion" -> JACK, which the widget could not
find. The reference tables are keyed for lookup by `norm_word` / `norm_def` /
`norm_ind`, written by `signature_solver.db._normalize_key`; our solver reads
them that way (`core/live_db.py:79`). The helper endpoints do not:
`web/routes/helper.py:159`, `:184`, `:196-201` (lookup), the `/helper/synonym`
queries, and `/helper/meanings`.

Effect: every row whose display form differs from its key is invisible to a
click — 39,265 in `synonyms_pairs` alone. Possessives, hyphenations and any
definition with a trailing "?".

Fixed in `publisher/reference.py`, then on the live site at the user's request,
both on 2026-08-18. Live files: `web/routes/helper.py` (every forward lookup
now keys on the normalised column, input through the solver's own
`_normalize_key`), `web/__init__.py` (the `clickable_words` filter emits
`data-plain` beside `data-clean`), `web/static/js/puzzle2.js`,
`web/templates/clue.html`, `learn_practice.html`, `learn_type.html` (send the
phrase as written; `clean` still feeds anagram fodder and the tutorial hooks).

`web/static/js/puzzle.js` holds an older copy of the same code and was NOT
changed — no template loads it (only `puzzle2.js` is referenced, at
`puzzle.html:415`). If it is still live somewhere, it needs the same edit.

**Verified, not deployed.** Through the real page and endpoint on :5001:
`data-plain="Jill&#39;s"` renders in the clue span, and `/helper/lookup`
returns JACK for both "Jill's companion" and the old stripped "jills
companion". Ordinary lookups unchanged (spirit 62 picks, vessel 60, about 61
with indicators), and `/helper/synonym` now answers "Jill's companion" -> JACK.

## 3. Pattern tool — enumeration toggle missing

The live pattern tool carries a `match (4,6)` button that switches the
enumeration filter on and off (`web/templates/puzzle.html:376-378`,
`_togglePatternEnum` in `base.html:197`). The widget applies the entry's
enumeration silently with no way to relax it. Same family of gap as the
missing "Must include" field, which was built on 2026-08-18.

## 4. Automated pre-publishing test — the browser layer has no cover

Raised 2026-08-20 by the user, discussing how hard UAT is with one tester. To
be built as an automated check that runs BEFORE anything is published, not as
something a person remembers to run.

The gap: `publisher/test_publisher.py` holds 43 checks and they are all
server-side — puzzle scoping, answer leakage, embargo, match counts, hint
rungs. Nothing loads a page. The whole of `publisher/static/js/engine.js` —
selection, cursor, typing, Backspace, Clear, the tools panel — has no test at
all. The three fixes of 2026-08-20 (cursor starts at square 1; roles above
synonyms; a completed crossing locks its square) live entirely in that layer.
The 43 passed before and after, because they never look there. The only proof
is that a browser was driven by hand and watched.

Feasibility, checked not assumed: Node 22 is on the machine, and `Engine` is
DOM-free by design — its own header says "model and state. No DOM."
(`engine.js:29`). The file closes `}(window))`, so a harness must hand it a
fake `window` carrying a `localStorage` stub, then take `Cordelia.Engine`.
About thirty lines of scaffolding. The view classes (`GridView`, `ToolsView`)
do touch the DOM and stay out of scope, except `ToolsView.fill`, which is worth
reaching because it holds the crossing-conflict refusal.

First tests to port, each already proven by hand and so a known-good baseline:
select lands on square 1 even when a crossing filled it; a conflicting
keystroke is refused and the cursor moves on; the same letter passes through;
Backspace and Delete are refused on a locked square; Clear keeps the locked
square and clears the rest; clearing the finished clue unlocks it; the tool
refusal returns the site's wording.

Two open questions for whoever picks this up: (a) where the gate hangs — the
dashboard DEPLOY page is the obvious hook, and `dashboard/pages/deploy.py` has
no test step today; (b) what stays human. Layout, mobile touch, wording and
feel cannot be automated, so this item should also produce a short fixed
manual script — ten or so steps in a set order — so a pass is repeatable
instead of ad hoc.

## 5. Synonym tool — the count line can read "Showing 60 of 60"

`runSynonym` passes `data.synonyms.length` as the total, so a capped result
reports itself as complete at its own cap. The live synonym partial prints a
plain result count and offers no expansion, so the honest fix is probably to
print the count, not to invent an expansion the site does not have.
