# Handover — 2026-08-20 (PM)
## Publisher widget: solve-side fixes pushed, ONE renderer for explanations · MOBILE IS NEXT

Cold-start document for the **publisher thread**. Supersedes
`HANDOVER_2026-08-20_PUBLISHER-NINE-OPTIONS-COMMITTED_NORM-KEY-FIX-LIVE_WORKLIST-OPEN.md`.

The WFW/solver thread has its own cold start:
`HANDOVER_2026-08-19_NAMED-LETTER-SHIFT-BUILT_TRANSFORM-DECISION-STILL-PENDING.md`.
Do not mix the two.

---

## 1. THE MANDATE — read before touching `publisher/`

**The widget re-formats the live solving system into an embeddable,
paper-shaped shell. The live implementation is the specification.**

Cite the live path before changing, explaining or defending any widget
behaviour: `web/templates/puzzle.html`, `web/static/js/puzzle2.js`,
`web/routes/helper.py`, `web/templates/partials/*.html`.

Format changes ARE allowed — the user's words: "we are allowed to make changes
to allow for the format change". The rule is that you replicate unless the
format forces a change, and you **say which you are doing**.

Two corrections from today, both worth keeping:

- **`puzzle2.js` is much more than the clue-page helper.** It carries a whole
  solve mode — grid, crossings, Add to grid — including a crossing-conflict
  refusal at `solveAddToGrid` (line ~839, message at 868). I twice told the
  user "the live site has no such feature" from reading `puzzle.html` and the
  clue page alone. Both times it did.
- `web/static/js/puzzle.js` IS dead (only `puzzle2.js` is loaded,
  `puzzle.html:415`). `/wfwfull` is dead too — retired 2026-08-14, noted in
  `partials/hint_step.html`, nothing posts to it.

---

## 2. START HERE — the mobile version is the next job

Nothing has been done on mobile yet. This section is the whole running start.

### Reaching the widget from a phone (verified today, both HTTP 200)

- `http://192.168.68.105:5003/` — the dev index, lists every puzzle so you can
  tap through instead of typing a URL.
- `http://new-lda31ifu5b9.local:5003/` — mDNS name, **use this for a saved
  shortcut**: it follows the machine, the IP does not.
- Straight to today's puzzle:
  `http://192.168.68.105:5003/embed/telegraph/94641?k=demo`

The demo key sets `frame_ancestors: ["*"]` (`publisher/config.py:32`), so a
direct hit from the phone is allowed and will not 403.

**If the phone cannot reach it, diagnose in this order — the first two take
seconds** (from `memory/mobile_dev_server_captive_portal.md`):
1. Load any website on the phone. Android will not route to a wifi it has not
   validated; the symptom is instant refusal with **no trace at the PC**.
2. Re-check the PC's wifi IP — it changes with the router.
3. Only then look at the PC. Windows Firewall, client isolation and NordVPN
   were all ruled out on 2026-08-15 with evidence. **Do not re-check them.**

### What mobile support exists today (checked, not assumed)

- **One breakpoint**, `@media (max-width: 900px)` in
  `publisher/static/css/shells/telegraph.css:223`: the two-column main becomes
  one column and scrolls, the clue list goes single-column, the plaque shrinks,
  and toolbar labels are hidden leaving icons at `min-width: 2.6rem`.
- **`engine.css` has NO media queries at all.** The grid, the tools panel, the
  hint ladder and the word lists have had no mobile pass.
- **There IS a keyboard capture input** — `engine.js:768`, an offscreen input
  with `autocapitalize=characters`, `autocorrect=off`, `autocomplete=off`. A tap
  on a square focuses it to raise the keyboard; keys are taken at the document
  (`engine.js:828`), not at the input.
- **No touch, pointer, or `visualViewport` handling anywhere** in `engine.js` or
  `telegraph.js`. Nothing reacts to the on-screen keyboard covering the grid,
  which is the usual first thing to break.

### The obvious first questions for mobile

Not decided — ask the user rather than guessing:
1. Does the tools panel become a sheet over the grid, or keep pushing it aside?
2. When the keyboard is up, must the active entry stay visible?
3. Phone-portrait target width, and which device is the reference?

---

## 3. State of the tree

Branch `redesign`. **`35c40f80` is committed AND pushed to origin/redesign.**
Nothing is deployed.

`35c40f80` touches `web/wfw_read.py`, so a deploy would carry it — but the
effect on the live site is **nothing a visitor sees**: `load_breakdown` is
reached only from POST `/wfwfull`, which nothing calls. `core/wfw_render`, the
clue page, serving rules, sitemap and URLs are untouched.

The 08-18 transform keep-or-revert decision is **still unmade**. Not this
thread's to act on.

---

## 4. What today changed (so it is not re-trodden)

Four solve-side faults the user hit, all in `publisher/static/js/engine.js`:

- Picking a clue landed the cursor on the first EMPTY square, so an entry whose
  opener was already crossed started at position 2 — and a tools component then
  went in one square late. Now square 1, by every route in.
- The word tool listed synonyms first with roles buried underneath. Now
  indicators, then abbreviations, then meanings — the site's own order
  (`partials/helper_results.html`).
- A letter belonging to a COMPLETE crossing entry could be typed over, deleted
  or cleared. Now it cannot. Same rule the site applies, same wording. Typing
  the letter already there still passes. Clear the finished clue to release it.
- The Hints ladder showed the answer and made the solver type it back. Now a
  button; one click places it, through Reveal's path so the lock cannot block
  the published answer.

**And the structural one.** `web/wfw_read.load_breakdown` was a hand-kept
parallel of the admin card and kept losing what the card shows — three
divergences in one day (deletion rows dropped, homophone partner never printed,
a replacement letter labelled "unclued" while its clue words were discarded).
All three fixed, then the cure: **the widget's full explanation IS the site's
card now.** `publisher/explanations.card_html` serves what
`web.serving.get_card` returns; `CARD_CSS` is inlined in the shell;
`renderBreakdown` is **deleted**. A test asserts byte equality.

`web/test_wfw_overlay_contract.py` claimed the breakdown must show every clue
word and passed through all three faults — it searched the whole page, and the
page prints the clue at the top. It now measures the ROWS against the stored
parse, and it found the third fault itself.

Tests: **45 publisher tests**, contract test **64 clues, 0 failures**.

---

## 5. Open

- **Mobile.** §2. Nothing started.
- `documents/PUBLISHER_WORKLIST.md`, in the user's priority order: corpus junk
  (evidence gathered, no DB writes made) · pattern-tool enumeration toggle ·
  **automated pre-publishing test** (item 4 — the browser layer has no cover;
  feasibility already checked, first tests listed) · synonym count line.
- Two clues in the contract-test skeleton carry a word no piece claims. That is
  **authoring, not rendering** — counted and printed by the test, never failed.
- `load_breakdown` survives only for the clue-type label.

---

## 6. Testing notes that cost time — do not re-learn

- **Dev servers run with the reloader OFF.** Full restart after ANY `.py`
  change; static `.js`/`.css` are read per request and need only a reload. The
  listener can take 10-15 seconds to appear — an immediate check says "no
  listener" and is wrong. Verify ONE listener and check the process START TIME:
  today the server had been up since 18 August and was serving stale Python.
- **The extension resizes the window between calls**, so a coordinate measured
  in one call is often stale in the next. Measure with `getBoundingClientRect()`,
  scale by `1568 / window.innerWidth`, and confirm with a capture-phase
  mousedown probe recording `clientX/clientY` and `target.className`.
- **Element-reference clicks silently fail** on the tools panel and clue rows —
  they report success and no handler runs. Coordinates with a probe, or
  `element.click()`; either way say which was used.
- **A backgrounded tab receives no keystrokes.** Check
  `document.visibilityState` before blaming a handler.
- **Do not wipe the user's saved progress.** Keys are
  `cordelia_pub_telegraph_<id>`; the option keys are `cordelia_pub_opt_*` and a
  `cordelia_pub` prefix match destroys both. Snapshot before, restore after.

---

## 7. How the user wants to be worked with

Said sharply, more than once today:

- **Be brief.** "You are sending back too many words, many of which are not
  relevant." Numbered points, no preamble, no restating the question.
- **Do not ask questions with an obvious answer.** Asking whether they wanted a
  wrong explanation fixed was called strange, and rightly.
- Questions are not instructions. A question about a feature gets an answer.
- Verify through the real path, and say plainly which part was not verified.
- The mandate outranks any design instinct.
