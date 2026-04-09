# Session Summary — 2026-03-06

## What Was Accomplished

### 1. Grid Reconstruction: 53/53 Telegraph 2026 Puzzles Working
- **Problem**: Grid reconstruction from clue data failed for ~13% of Telegraph puzzles (barred grids, themed prize puzzles, algorithm limitations)
- **Solution**: Use the Telegraph API's `solution` string (225 chars for 15×15) directly, bypassing algorithmic reconstruction
- **New table** `puzzle_grids` in `clues_master.db`:
  - Columns: source, puzzle_number, solution, grid_rows, grid_cols, api_folder, api_type, api_id
  - 15 rows stored (all Telegraph), all with grid solutions, 10 with API coordinates
- **New function** `parse_grid_solution()` in `web/grid.py` — converts flat solution string to cells with derived clue numbers
- **New function** `get_puzzle_grid_solution()` in `web/models.py` — queries puzzle_grids table
- **Modified route** `puzzle_grid()` in `web/routes/puzzle.py` — tries stored grid first (Path 1), falls back to `reconstruct_grid()` (Path 2)
- **Modified scraper** `telegraph_daily.py`:
  - `fetch_and_save()` now stores grid solution + API coordinates in puzzle_grids on every fetch
  - New `backfill_grid_solutions()` — re-fetches grid solution strings (not answers) for puzzles where solution was NULL; runs after every daily scrape; no browser needed (direct HTTP)

### 2. Restored danword_lookup.py from Archive
- Was accidentally moved to `cryptic_solver_archive` during a previous archiving session
- Restored to `scraper/danword/danword_lookup.py`
- Fully working script: searches danword.com via Google CSE, validates answers against grid, writes to DB
- Orchestrator (`puzzle_scraper.py`) already wired to call it

### 3. Fixed Missing Clues for Several Puzzles
- Manually inserted missing/blank clues from Telegraph API for puzzles: 31127, 31128, 31139, 31145, 31151, 31161, 31163, 31169

---

## Current State of Uncommitted Changes

**Tracked files modified** (850 insertions, 125 deletions):
- `scraper/telegraph/telegraph_daily.py` — +365 lines (grid storage, backfill_grid_solutions, self-healing logic)
- `sonnet_pipeline/run.py` — +178 lines
- `sonnet_pipeline/review_gaps.py` — +181 lines
- `sonnet_pipeline/solver.py` — +120 lines
- `sonnet_pipeline/report.py` — +50 lines
- `sonnet_pipeline/enricher.py` — minor fix
- `scraper/orchestrator/puzzle_scraper.py` — minor fix
- `scraper/times/times_all.py` — +11 lines
- `.gitignore`, `.claude/settings.local.json`, `.idea/workspace.xml`

**Untracked directories** (new, not yet committed):
- `web/` — entire Flask web app (routes, models, templates, grid module)
- `scraper/danword/` — restored danword_lookup.py
- `data/` — databases (should stay in .gitignore)
- `documents/` — puzzle reports

---

## Outstanding Problem: Prize Puzzle Answers

### 17 answerless puzzles in the database

**Telegraph** (5 — all Sunday prize, series 0-999):
- #208 (2026-01-18), #209 (2026-01-25), #212 (2026-02-15), #213 (2026-02-22), #214 (2026-03-01)

**Times** (10 — mix of Saturday prize and Sunday):
- Saturday: #27972 (2021-05-08), #28002 (2021-06-12), #28824 (2024-01-27), #29448 (2026-01-24), #29454 (2026-01-31), #29466 (2026-02-14), #29472 (2026-02-21)
- Sunday: #4960 (2021-06-20), #5200 (2026-01-25), #5205 (2026-03-01)

**Guardian** (2):
- #29924 (2026-02-07), #29942 (2026-02-28)

### How to backfill answers
- Both Telegraph and Times APIs provide answers once competitions close
- The existing scrapers know how to discover API URLs (browser navigates to puzzle page, finds the API endpoint)
- **Telegraph**: `telegraph_daily.py` navigates to puzzles page, finds links matching `#crossword/{folder}/{type}-{api_id}`, fetches from `puzzlesdata.telegraph.co.uk/puzzles/{folder}/{type}-{api_id}.json`
- **Times**: `times_all.py` navigates to puzzle page, finds iframe pointing to `feeds.thetimes.com/puzzles/sp/{feed_type}/{date}/{id}/data.json`, fetches the JSON
- Times has a `--force` flag that deletes and re-fetches a puzzle
- Telegraph uses `INSERT OR IGNORE` so re-fetching won't update blank answers — needs modification to UPDATE existing rows
- **Neither scraper currently has automatic answer backfill for old prize puzzles**
- `danword_lookup.py` exists as a fallback (searches danword.com) but the primary approach should be re-fetching from source APIs
- The `find_answerless_puzzles()` function in `puzzle_scraper.py` currently only checks today's date — won't catch old puzzles

### What needs building
- A mechanism to re-run the scrapers for specific past prize puzzles once their competitions have closed and answers are published
- This requires the browser (to discover API URLs) — there is no stored API URL for most puzzles
- DO NOT touch the login/browser code in the scrapers — it took a long time to get working

---

## Scraper Architecture Reference

### Telegraph (`scraper/telegraph/telegraph_daily.py`)
- Browser logs in via persistent Chrome profile
- Navigates to `telegraph.co.uk/puzzles/`, harvests puzzle links
- Link pattern: `#crossword/{folder}/{type}-{api_id}`
- API: `https://puzzlesdata.telegraph.co.uk/puzzles/{folder}/{type}-{api_id}.json`
- TYPE_MAP: cryptic-crossword, toughie-crossword, prize-cryptic, prize-toughie
- Prize puzzles: date check skipped (link shows closing date), raw JSON saved to disk
- Self-healing: `INSERT OR IGNORE` adds missing clues without duplicating
- `backfill_grid_solutions()`: re-fetches grid solution string only (no browser needed)

### Times (`scraper/times/times_all.py`)
- Browser logs in via persistent Chrome profile
- Navigates to puzzle page, finds iframe with `feeds.thetimes` in src
- API: `https://feeds.thetimes.com/puzzles/sp/{feed_type}/{date}/{id}/data.json`
- Puzzle types: cryptic (Mon-Sat), sunday-cryptic (Sun)
- Competition puzzles: `competitioncrossword: 1` flag, word solutions empty
- `--force` flag: deletes existing puzzle and re-fetches
- JSON backup saved to `scraper/times/` directory

### Orchestrator (`scraper/orchestrator/puzzle_scraper.py`)
- Runs all scrapers, reconciles against expected schedule, sends email report
- `find_answerless_puzzles()`: finds today's puzzles with all blank answers
- `run_danword_backfill()`: calls `scraper/danword/danword_lookup.py` for each
- DANWORD is a fallback — source APIs should be primary

### DANWORD (`scraper/danword/danword_lookup.py`)
- Searches danword.com via Google Custom Search widget (Selenium)
- Validates answers against grid crossings using saved prize JSON
- Writes validated answers to clues table
- Called by orchestrator but can also run standalone: `python danword_lookup.py --source telegraph --puzzle 208`

---

## Database State

### `clues_master.db`
- **clues table**: ~507k rows across guardian (252k), telegraph (156k), times (67k), independent (32k)
- **puzzle_grids table**: 15 rows (all Telegraph), created last session
- **structured_explanations table**: 199 rows (Sonnet pipeline results)

### `cryptic_new.db`
- Reference tables only: synonyms_pairs (1.3M), definition_answers_augmented (713k), indicators (4.6k), wordplay (2.1k)

---

## Project Documentation — Read These First

The new thread should read these files to get full context:

### Core instructions
- **`CLAUDE.md`** — standing instructions, critical rules, project context, autonomy rules, git safety. Read this first always.
- **`REFACTOR_PLAN.md`** — the original refactor plan from 2026-02-12. Phases 0-5 (pipeline cleanup) are largely superseded by the V3 direction and sonnet pipeline. Phase 6 (scraper integration) is done. Useful for understanding the old pipeline architecture but not the current focus.

### Auto-memory (persists across sessions)
- **`~/.claude/projects/.../memory/MEMORY.md`** — master memory file. Contains V3 design direction, roadmap, session state, classifier status, structured hint parser, sonnet pipeline architecture, database state, lessons learned. This is the most important reference after CLAUDE.md.
- **`~/.claude/projects/.../memory/nightly_pipeline_design.md`** — nightly pipeline design (scrape → solve → score → email). Includes best-of-N rejection (tested and rejected), accuracy assessment, report format.
- **`~/.claude/projects/.../memory/product_design.md`** — product design for the web app. Launch scope (Telegraph + Times only), progressive hints, SEO strategy, security architecture, tech stack (Flask + HTMX + Tailwind + SQLite), hint tier mapping.

### Puzzle number series (Telegraph)
- **31xxx** (e.g. 31100-31200): Daily cryptic (Mon-Sat) — Saturday prize cryptic is in this same series
- **3xxx** (4-digit, e.g. 3000-3999): Sunday prize cryptic
- **1000-9999**: Toughie + prize toughie

### Puzzle number series (Times)
- **5000-9999**: Sunday cryptic
- **26000+**: Daily cryptic (Mon-Sat, Saturday is often a competition/prize)

---

## Key Rules (from painful experience)
- NEVER modify working scraper login/browser code
- NEVER run git filter-branch or git gc --prune=now
- NEVER stash database files
- Use Times and Telegraph for testing, NOT Guardian
- Prompt is not the bottleneck — DB enrichment coverage is
- Quality over quantity for fine-tuning
- DO NOT put files in project root — use appropriate subdirectories
