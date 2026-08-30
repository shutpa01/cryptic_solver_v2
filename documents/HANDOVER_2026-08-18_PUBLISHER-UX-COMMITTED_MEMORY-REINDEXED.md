# Handover — 2026-08-18
## Publisher widget UX committed · memory re-indexed

Cold-start note for the next thread. Memory is current, so `MEMORY.md` and the
publisher topic files load automatically — this document is only the "where we
are and what is open" layer.

---

## 1. State of the tree

Branch `redesign`, **nothing pushed**.

| Commit | What |
|---|---|
| `a0f2d69c` | publisher package — Telegraph shell, grid engine, tools mode |
| `2022f8f1` | the real WFW explanation + six UX fixes from testing |
| `e981ddbd` | legible match count, clue-number jump, two clear controls |

**29 tests pass**: `.venv\Scripts\python.exe -m unittest publisher.test_publisher`
Dev server: `.venv\Scripts\python.exe publisher\run_dev.py` → http://127.0.0.1:5003/
(the index labels which puzzles demo everything).

### Modified but NOT committed — and NOT mine to commit
`core/piece_transform.py`, `core/store.py`, `core/wfw_render.py`,
`core/wfw_web.py`, `web/wfw_read.py`

These are the transform-overreach fix from the other thread. **The keep-or-revert
decision is still the user's.** See
`HANDOVER_2026-08-18_TRANSFORM-RECORD-OVERREACH_REVERT-DECISION-PENDING.md`.
Do not commit, extend or deploy them.

---

## 2. What was done today

All of it in `publisher/`, all browser-verified, all user-reported:

- **Full explanation is now the site's real WFW breakdown**, not a prose format I
  invented. This imports `web.wfw_read.load_breakdown` — the one deliberate
  crossing of the "publisher imports nothing from web" rule, recorded in
  `publisher_build_decisions`. Side effect worth keeping: the clue type now reads
  "Container + acrostic" (every mechanism named) instead of a flat "Container".
- **Clue words clickable on the active row of the clue list**, not only the bar.
  Selection state moved onto the engine because it now shows in three places.
- **Match count made legible.** It was never broken — it was 8px. Reported three
  times before I checked legibility rather than correctness.
- **Clue numbers in the grid jump to that clue** (the entry that STARTS there,
  which is not what clicking the square does).
- **Two clear controls**: erase the current answer (beside the active clue on all
  three surfaces), and reset the current tool (foot of every tool panel).
- Also: "99+" when the count exceeds the ceiling (262 entries showed nothing at
  all before), clickable word-lookup options, tool panel resets on clue change,
  right-click no longer runs the left-click path.

---

## 3. Open — nothing here is decided

1. **Transform-overreach revert** (above). Blocks nothing in `publisher/`, but the
   widget renders that breakdown so the label shows up either way.
2. **Corpus-coverage gate** — user said *not a priority for now*. Measured: 8,630
   answers checked, 99.84% already in the corpus, **14 missing across 6 puzzles**
   (mostly Prize Toughies and long multi-word answers). On those clues the red
   zero lies and the anagram tool returns nothing. The design already says this
   should become a hard gate but not in phase one.
3. **Not built**: Times and Guardian shells, the publisher JSON feed, the general
   date gate.
4. **41 of 66 session logs** in `memory/` are referenced by nothing, including
   `sessions_archive.md`. March–June. Left alone deliberately.

---

## 4. Corrections I owe the record

- I showed an anagram breakdown containing "ARREST (not accounted for)" and told
  the user it was a real data fault the renderer had been hiding. Per the 08-18
  finding that was the overreach bug firing on a good row. **Do not re-assert it.**
- I claimed the widget beat everything available before testing any competitor.
  The defensible claim is narrower and better: for solving *a specific published
  puzzle*, nothing known combines grid-aware tools with a hand-verified
  explanation of that exact clue. General word engines know words, not the clue
  or the grid. The user's own point is the strongest part — every licensed puzzle
  goes through our solver, so we KNOW the answer is in the corpus, which no
  general engine can say. That guarantee is what item 3.2 above protects.

---

## 5. Memory re-indexed

`MEMORY.md` 19.9 KB → 15.7 KB and restructured as a pure index. Two new
second-hop catalogues carry what used to be orphaned:

- `memory/index_rules.md` — all 75 previously-unindexed standing rules
- `memory/index_topics.md` — ~155 engine / architecture / history / incident files

**0 non-session files are now orphaned** (was 296). Two that were orphaned and
should not have been: `feedback_leftover_process` (CLAUDE.md calls it the single
source of truth) and `feedback_no_faint_text` (which is exactly the 8px bug).
