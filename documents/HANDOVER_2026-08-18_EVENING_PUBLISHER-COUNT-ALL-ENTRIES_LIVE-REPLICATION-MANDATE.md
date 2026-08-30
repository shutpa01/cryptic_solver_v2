# Handover — 2026-08-18 evening
## Publisher match count rebuilt · live-replication MANDATE

Cold-start document for the next publisher thread. Read §1 before anything else.

---

## 1. THE MANDATE — read this first

**The widget in `publisher/` re-formats the live solving system into an
embeddable, paper-shaped shell so the Telegraph, Times and Guardian can license
it. The live implementation is the specification.**

Read the live path before changing, explaining or defending any widget
behaviour, and cite it: `web/templates/puzzle.html`,
`web/static/js/puzzle2.js` (and `puzzle.js`), `web/routes/helper.py`.

- Never answer "what is this feature for?" from `publisher/`'s own code, from a
  memory file, or from a design summary — second-hand, and they have been wrong.
- If the live system has no such behaviour, say so and ask. Do not invent it.
- A bug report is about the feature's PURPOSE, not only its mechanism.

Now recorded in three places: `CLAUDE.md` (top), `MEMORY.md` (READ FIRST), and
`memory/feedback_replicate_live_never_design.md`.

**What it cost today.** Three hours, one small change. Asked what the match
count is for, I answered from the widget's own code and a memory note, and was
wrong twice: I made it inert when the live site makes it a doorway, and I left
it on the active entry alone when its purpose is scanning the whole board. Both
answers were in the live code in plain sight.

---

## 2. What the match count now does — and why

Live behaviour, which is what we are replicating:
`puzzle.html:226-228` — the count is a clickable span, title "Open in pattern
finder", tip at :979 "Click the pattern to search for matching words".
`puzzle2.js:1179-1221` — the click opens the tools overlay, switches to the
pattern tab, fills the pattern, runs the search. Clicking a result enters it.

The journey is: **how many words fit → which ones → click one, it goes in the
grid.** The user's framing: the solver scans the grid, sees which clues have
only a few options, and concentrates on those.

Built in the widget today:

- **Every entry shows its count at once**, painted on the last square of the
  entry. One request for the whole grid — `/api/match-counts` already accepted
  up to 100 patterns, only the front end was asking one at a time. Recomputed on
  every letter change, not on selection.
- **Two chip slots per square**, since a square can end an across and a down
  entry: across bottom-right (`.cg-count-a`), down bottom-left (`.cg-count-d`).
- **Each chip carries `dataset.entry`** — clicking it selects THAT clue, not the
  selected one, and opens the pattern finder on it via
  `GridView.onCountClick` (a hook, so each shell places it) →
  `ToolsView.openPattern()` (show + search in one).
- **Hit area fixed.** The chip is drawn proud of its square; with
  `pointer-events:none` its overhanging edge was hit-tested against the squares
  behind it, so clicking the chip on 11 Across selected the square below and the
  count vanished with the selection — reported as "a dead click and then the
  number disappeared".

Verified on 31320: 11 chips paint on load with no clicks (9a=67, 11a=21, 13a=9,
eight at 99+). Clicking the 9 while 1 Across was selected switched to 13 Across
and listed exactly 9 words for `T???V??`. On 31321: empty grid paints nothing,
typing SPRAINS into 1 Across immediately produced counts on the four crossing
down clues, and clicking a listed word entered it in the grid.
`/api/match-counts` and `/api/tools/pattern` agree (21 and 21), so the chip
cannot promise a number the list contradicts. **29 tests pass.**

Two deliberate silences, both by design: an entry with **no letters** has no
meaningful count (counting every 7-letter word is not information), and a
**full** entry is refused a count server-side so it can never be a free Check.

---

## 3. State of the tree

Branch `redesign`, nothing pushed. Committed earlier: `a0f2d69c`, `2022f8f1`,
`e981ddbd`.

**Uncommitted, mine, ready for review:**
`publisher/static/js/engine.js`, `publisher/static/js/shells/telegraph.js`,
`publisher/static/css/engine.css`, plus `CLAUDE.md` (the mandate).

**Uncommitted, NOT mine — do not commit, extend or deploy:**
`core/piece_transform.py`, `core/store.py`, `core/wfw_render.py`,
`core/wfw_web.py`, `web/wfw_read.py` — the transform-overreach fix whose
keep-or-revert decision is still the user's. See
`HANDOVER_2026-08-18_TRANSFORM-RECORD-OVERREACH_REVERT-DECISION-PENDING.md`.

Dev servers were restarted this morning and run detached with logs in the
session scratchpad: `web/run_dev.py` :5001, `publisher/run_dev.py` :5003,
`core.wfw_web` :5099. **No reloader — restart after any .py change.** Static
JS/CSS needs a browser reload, not a server restart.

---

## 4. Open

1. **Corpus junk.** `SPR????` returns `SPRAYER} –`, `SPREADA`, `SPRINGA`,
   `SPREAGH` among 21. The count is the product's trust signal; junk in the
   corpus is junk in the count. Not investigated, nothing changed.
2. **Chip adjacency.** Where an across chip and a down chip from neighbouring
   squares meet, two numbers sit side by side; the hover title names the clue,
   but the placement has not been judged by the user on a real screen.
3. Unchanged from this morning's handover: Times and Guardian shells, the
   publisher JSON feed, the general date gate, and the corpus-coverage gate
   (14 answers missing across 6 puzzles, user said not a priority).

---

## 5. Testing notes that cost time today — do not re-learn

- The extension screenshot is scaled **1.103×** against CSS pixels. Eyeballing
  coordinates off the image clicks the wrong square. Read
  `getBoundingClientRect()`, multiply, and confirm the hit with a capture-phase
  mousedown probe recording `clientX/clientY` and `target.className`.
- **A background tab receives no clicks at all** (`visibilityState: "hidden"`),
  and the first click after a reload or a JS eval is swallowed. Both look
  exactly like a broken handler. Check visibility and the probe before
  concluding anything about the code.
