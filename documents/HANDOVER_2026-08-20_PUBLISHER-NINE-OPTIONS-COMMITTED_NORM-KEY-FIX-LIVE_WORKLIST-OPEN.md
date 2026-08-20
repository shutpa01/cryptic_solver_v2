# Handover — 2026-08-20
## Publisher: nine options committed · norm-key lookup fixed on the live site too

Cold-start document for the **publisher thread**. Supersedes
`HANDOVER_2026-08-18_EVENING_PUBLISHER-COUNT-ALL-ENTRIES_LIVE-REPLICATION-MANDATE.md`
as the cold start; that one is still the record of how the match count was
rebuilt and why.

The WFW/solver thread has its own cold start:
`HANDOVER_2026-08-19_NAMED-LETTER-SHIFT-BUILT_TRANSFORM-DECISION-STILL-PENDING.md`.
Do not mix the two — see §5.

---

## 1. THE MANDATE — read before touching `publisher/`

**The widget re-formats the live solving system into an embeddable,
paper-shaped shell. The live implementation is the specification.**

Read and cite the live path before changing, explaining or defending any widget
behaviour: `web/templates/puzzle.html`, `web/static/js/puzzle2.js` (and
`puzzle.js`), `web/routes/helper.py`, `web/templates/partials/*.html`.

Two things this session proved the mandate is worth:

- Every gap the user reported was the same gap — *we are not replicating what
  we already have*. Must-include, the ⓘ on results, the working "+N more": all
  three existed on the live site and were simply missing here.
- The one place the widget now goes BEYOND the live site is deliberate and was
  approved by the user in the same breath (the nine-option cap, §2).

**Format changes ARE allowed** — the user's words: "we are allowed to make
changes to allow for the format change". The rule is that you replicate unless
the format forces a change, and you say which you are doing.

---

## 2. Committed this session (branch `redesign`; on origin, NOT deployed — see §5)

### `90a7ef10` — publisher: nine options with the answer among them

The user's brief: cap the green numbers at 9, cheat by ALWAYS displaying the
answer, options alphabetical, across numbers top-right of the last cell and
down bottom-left.

The cap has two arms, settled after one clarifying question (the user had not
thought it through, and the wrong reading would have put a nine on every entry
on the board):

- A chip appears only when **9 or fewer words really fit** — so a number means
  "this one has come down to a handful", and the board thins as it is solved.
- **Unless every crossing letter of the entry is already in.** That entry can
  never narrow again from the grid, so it is offered a shortlist of nine
  anyway. The browser computes it (`Engine.crossingsFilled`) and sends
  `crossed` to `/api/match-counts`; the server decides what to do about it.
- The chip number **is** the length of the list the click opens. Same
  computation both sides, so they cannot contradict each other.
- The shortlist always holds the answer; the other eight are sampled by md5 of
  pattern+word, so it is spread rather than the alphabetically first nine.

Honesty limits kept, and they matter:

- An answer that does not fit the letters in the grid is **not** offered — a
  wrong letter must not buy a free Check.
- An embargoed prize puzzle has no answer here, so nothing is injected.
- The panel still prints the true total ("Showing 9 of 22"). An earlier version
  bounded the scan at 200 and printed "9 of 200" — a number the widget cannot
  stand behind. The bound is gone.

Also in this commit, each replicating a live feature and citing it in the code:

- **"Must include"** in the pattern tool (`puzzle.html:368`), searching as you
  type, no second Search button.
- **The ⓘ** on pattern and anagram results (`partials/pattern_results.html:22`,
  `anagram_results.html:21`) behind a new `/api/tools/word-info`.
- **"+N more" as a control**, not a caption (`partials/helper_results.html:59`).
  The lookup endpoint already accepted `letters`; only the browser never asked.
- **Chip placement**: across top-right, down bottom-left, clear of the clue
  number at top-left.
- **Switching tool by hand clears the picked clue words**, so a highlight
  always means the tool is using that word (user-reported UX fault).
- **Corpus buckets built sorted.** They were built from Python sets, whose
  iteration order is randomised per process, so the "unchanging" nine changed
  at every server restart. Three separate processes now return the same nine.

### `de468ac7` — web: look reference words up by their normalised key

**The most important finding of the session, and it applies far beyond the
widget.** The user's framing, which is the right one:

> we know ALL the synonyms are in the DB because we already pushed it through
> our solver

Clicking "Jill's companion" (31320 1 Down, JACKPOT) returned nothing. The row
was there all along: `synonyms_pairs` word "Jill's companion", synonym JACK,
source admin, **norm_word 'jills companion'**.

The reference tables are keyed for lookup by `norm_word` / `norm_def` /
`norm_ind`, written by `signature_solver.db._normalize_key` and reconciled at
every wiring build by `core/norm_backfill.py`. **Our solver reads them that
way** (`core/live_db.py:79`). The helper UI never did — it matched
`LOWER(word)`, which cannot see any row whose display form differs from its
key: **39,265 rows in `synonyms_pairs` alone** (possessives, hyphenations, any
definition with a trailing "?").

The browser made it unfixable server-side: a clicked word was stripped to
letters before it was sent. Clue words now carry `data-plain` (as written)
beside `data-clean` (letters only); the phrase is built from `data-plain`, and
`data-clean` still feeds the anagram box, the admin coverage underline and the
tutorial hooks, so none of those changed.

Files: `web/routes/helper.py` (every forward lookup; the reverse queries that
match the value column are untouched), `web/__init__.py` (`clickable_words`),
`web/static/js/puzzle2.js`, `web/templates/clue.html`, `learn_practice.html`,
`learn_type.html`. `web/static/js/puzzle.js` holds an older copy no template
loads (`puzzle.html:415` loads puzzle2.js) and was left alone.

**A lookup that cannot find what the solver used is a lookup bug, never an
absent fact.** Apply that test to any similar report.

---

## 3. Verification (what was actually proven)

- **43 publisher tests pass.** New ones cover: the cap, the crossings-full
  exception, chip-number equals list-length, wrong letters get no shortlist,
  an embargoed answer is never injected, the shortlist does not reshuffle, the
  corpus is built in a fixed order, must-include narrows and also governs the
  answer, "+N more" fetches the rest, word-info, and the Jill's-companion
  phrase in three spellings.
- **Widget, in the browser**: on 31268, 19 Down `?O?G?E` — 22 words really fit,
  every crossing letter in, chip reads 9, click lists nine alphabetically with
  MORGUE among them. On 31320, clicking Jill's then companion returns JACK;
  switching tool clears the highlight. Must-include: W?????? gave 9 of 510,
  typing SH gave 9 of 62 with WARSHIP still there.
- **Live site, on :5001**: the clue span renders
  `data-clean="jills" data-plain="Jill&#39;s"`, `/helper/lookup` and
  `/helper/synonym` both answer JACK, and the old stripped spelling still works
  so cached pages resolve. Ordinary lookups unchanged (spirit 62, vessel 60,
  about 61 with indicators). Then confirmed by clicking on the real page:
  "Could mean: (4) JACK".
- **NOT deployed.** The live fix is committed and on origin/redesign (carried
  there by another thread's push, §5), but it is not on the server.

---

## 4. Open — `documents/PUBLISHER_WORKLIST.md`

The user asked for deferred items to go on a worklist rather than be chased.
It currently holds, in their priority order:

1. **Corpus junk** (`SPRAYER} –`, `IDING`, `SOOGIE`, `(rum?)`, `a black`). The
   source evidence is already gathered in the doc: 3,099 mojibake answer rows
   concentrated in the blog scrapes (fifteensquared 1,305, telegraph 470), and
   9,567 junk synonyms almost all from `api_mw_mesh` and the 510,426
   NULL-source rows. Proposed order: display hygiene in `publisher/corpus.py`
   first (no DB write), then a source-scoped repair of the mojibake, and leave
   the NULL-source block alone. **Nothing has been changed. No DB writes.**
   Note while you are there: the solver deliberately EXCLUDES `api_mw_mesh`
   from its own synonym lookup (`core/live_db.py`, comment dated 2026-07-23) —
   the widget's corpus does not.
2. (done) the norm-key fix.
3. **Pattern tool: the enumeration toggle is missing** — live has a
   `match (4,6)` button (`puzzle.html:376`, `_togglePatternEnum` in
   `base.html:197`). Same family as the missing Must-include.
4. **Synonym tool count line can read "Showing 60 of 60"** when capped.

---

## 5. State of the tree

Branch `redesign`. This thread's two commits — `90a7ef10`, `de468ac7` — are
**on origin/redesign, and NOT deployed.**

They were not pushed by this thread. The WFW/solver thread committed `5dbc1597`
on top of them and pushed that, which carried these two up with it
(`git branch -r --contains` confirms both are on the remote). Worth knowing
rather than assuming: on a shared branch, "I did not push" does not mean "it is
not on the remote".

`5dbc1597` carries the 08-18 transform work, the 08-19 named letter shift and
08-20's reverse-anagram work. **The 08-18 keep-or-revert decision is still
unmade and nothing is deployed** — read the 08-19 handover before touching any
WFW render or gate code. None of that is the publisher thread's to act on.

Dev servers, both restarted after every .py change, logs in the session
scratchpad: `publisher/run_dev.py` on :5003, `web/run_dev.py` on :5001.
**No reloader.** Verify ONE listener and check the process START TIME. On this
machine the listener can take 10-15 seconds to appear after a restart — an
immediate check reports "no listener" and is wrong.

---

## 6. Testing notes that cost time — do not re-learn

- **A backgrounded tab receives no keystrokes.** Clicks land, keys do not, and
  it looks exactly like a broken handler. Check `document.visibilityState`
  first. Where the tab could not be brought forward, the same handlers were
  driven with dispatched `KeyboardEvent`s and `element.click()`, and the report
  said so — that is honest; claiming "typed it" would not be.
- **The extension resizes the window between calls**, so a coordinate measured
  in one call can be stale in the next. Measure with `getBoundingClientRect()`,
  scale by `1568 / window.innerWidth`, and confirm the hit with a capture-phase
  mousedown probe recording `clientX/clientY` and `target.className`.
- A JS call against a tab still sitting on `chrome://newtab` is refused with
  "Permission denied for JavaScript execution on this domain". That is the new
  tab page, **not** the port — load a page on the origin first. I got this
  wrong once and told the user :5001 needed a permission; it did not.
- Clearing the widget's saved progress from `localStorage` with a `cordelia_pub`
  prefix also wipes the stored option preferences (`cordelia_pub_opt_*`). Match
  `^cordelia_pub_telegraph_` instead.

---

## 7. How the user wants to be worked with

Restated because it was said sharply this session and it applies to every reply:

- **Be concise.** "It is pointless sending back a sea of words, it defeats the
  object of good communication. I have read none of it." Numbered points, no
  preamble, no restating the question.
- Questions are not instructions. A question about a feature gets an answer,
  not an edit.
- Verify through the real path before claiming anything, and say plainly which
  part was not verified.
- The mandate above outranks any design instinct.
