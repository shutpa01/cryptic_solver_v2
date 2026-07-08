# HANDOVER 2026-07-07 — Triage page review

**START HERE for the new thread.** The triage system (diagnose → review → fix → completion) is
built and running. The next job is small and user-led: **the user will open the `/triage` page,
use it, and report back observations.** Your job is to act on those observations — fix, adjust,
polish. Do NOT go build new things until the user has reported what they saw.

---

## 0. HOW TO WORK (the user has been explicit — honour it)

- **Plain English, no jargon.** The user found earlier jargon-heavy explanations hard to follow and
  said so more than once. Short, plain answers. Explain, don't dump.
- **Questions are not instructions** (CLAUDE.md). "How does X work?" → explain, touch nothing.
- **Read the design / the actual code before acting.** A prior thread lost the user's confidence by
  improvising the signature flow instead of reading it. In THIS session I mis-stated the signature
  spec once and had to correct it — the user rightly pushed back ("I thought we already built b?").
  So: read the real code, verify before claiming, cite files.
- **Don't improvise the signature area.** It is the one place a mistake is unforgivable (a false
  green). Read `documents/BUILD_PLAN_2026-07-05_SIGNATURE-TIERS.md` and
  `memory/nightly_triage_process.md` before touching it.
- **Test through the real path on TEMP DB copies** so the real DBs are never mutated (see §10).
- The user drives. Check in at natural points; don't run off building.

---

## 1. HOW TO RUN + ACCESS

- **Python:** `.venv\Scripts\python.exe` (bare `python` is a Windows-Store stub). For scripts that
  import `core`, set `$env:PYTHONPATH = (Get-Location).Path` first.
- **Server (Flask):** http://127.0.0.1:5099/ . Start it:
  ```
  $env:PYTHONPATH = (Get-Location).Path
  Start-Process -WindowStyle Hidden .venv\Scripts\python.exe -ArgumentList '-m','core.wfw_web'
  ```
  **Restart after ANY code change** (it loads modules at start). To restart, kill the running
  `python.exe` whose command line contains `wfw_web`, then start again.
- **The triage page:** `http://127.0.0.1:5099/triage?src=<source>&pnum=<number>`
  e.g. `.../triage?src=telegraph&pnum=31284`. For a fresh puzzle, click **"Run / refresh
  diagnosis"** once (it classifies + caches). 31284 is already classified (cache written).

---

## 2. THE WHOLE PROCESS, SCRAPE → COMPLETION (plain)

1. **Scrape (nightly-ish):** the scraper pulls new puzzles into `data/clues_master.db` (`clues`
   table). *(Trigger mechanism — cron vs manual — not confirmed; likely manual today.)*
2. **Solve:** every clue runs the solver cascade → **green (pass)** / **amber (pending)** /
   **red (fail)**, stored in `wfw_solve` (status, solved_by, template_id).
3. **Triage — diagnose (Claude/tool):** for every amber+red clue, a DETERMINISTIC classifier
   (§4) decides WHY: missing_data / missing_signature / missing_engine / solves_now. This is the
   `/triage` page. Claude only diagnoses + suggests; the human commits.
4. **Fix — the `/triage` page (user):** for a data gap, edit + Accept the suggested value → it's
   written to `data/cryptic_new.db` and the clue is re-solved on the spot (red→green). For a shape
   gap, click **"Open in hand-solver"** → hand-solve → a signature is filed at its risk tier (§5).
5. **Completion:** every clue ends green (trusted), amber (needs a human tick), or an honest red
   ("can't parse" — acceptable). Never a false green, by construction.

**Seams that are still MANUAL / open** (candidate future work, NOT yet built):
- No scheduler ties scrape→solve→triage into one nightly job.
- Safe-signature promotion is a ~30-min CLI command, not a button (§5).
- The consolidated dashboard (per-puzzle triage + pending-review queue + promotions on one screen)
  is deliberately deferred.

---

## 3. THE `/triage` PAGE (what the user is about to review)

Route `/triage?src=&pnum=` in `core/wfw_web.py` (render `_triage_surface`). Per unsolved clue it shows:
- **Deterministic diagnosis** — a category badge (missing data / missing signature / missing engine
  / should solve now) + the mechanical evidence (e.g. "no known value for: French, sweat"; "new
  shape needed: SIG (tier)"; "reads as a double definition"). From the classifier cache.
- **Suggested values** — a SEPARATE block, clearly labelled *"human/AI — not mechanical"*, with the
  specific value guesses (OISE, BED…), each **editable** then **Accept/Reject** (radio), applied in
  one batch by **"Apply queued & re-solve"** at the foot.
- **"Open in hand-solver →"** link (prominent on missing-signature clues) → `/hs?id=<cid>`.

Supporting routes (all in wfw_web.py):
- `/triageclassify` (POST, "Run / refresh diagnosis"): runs `triage_classify.classify` over every
  fail/pending clue, caches to `documents/triage/classified_<pnum>.json`, and re-solves every
  `solves_now` clue once (green → drops off; still failing → the rare genuine oddity).
- `/triageapply` (POST): applies the queued Accepts/Rejects — writes each via the dashboard's
  `admin_db.add_*`, reloads the wiring ONCE, re-solves each affected clue ONCE, warns if a clue
  passes with words still unaccounted, lands back at the first affected clue (not the top).

Data files (in `documents/triage/`):
- `classified_<pnum>.json` — the DETERMINISTIC categories (owns the classification).
- `diagnoses_<pnum>.json` — the human/AI VALUE suggestions ONLY (its old `reasons`/`gaps` are now
  ignored; the classifier owns the category). For 31284 this is my hand-written value list.
- `rejected_<pnum>.json` — rejected suggestion keys.
- `TRIAGE_<src>_<pnum>.md` — a standalone markdown report (`python -m core.triage <src> <pnum> [diag.json]`).

---

## 4. THE DETERMINISTIC CLASSIFIER (the "rigid diagnosis" the user demanded)

`core/triage_classify.py` → `classify(clue_text, answer, wiring)` → `{reasons, detail}`. **Same
clue + same DB state → same category every time** (verified repeatable, ~0-2s/clue). The user's
whole point: the classification must be a fixed method, not Claude's per-clue judgment. Only the
specific missing VALUE stays a labelled human/AI guess.

Rules (fixed): no confirmed definition or an unresolved content word → **missing_data**; `discover`
finds an assembling shape NOT in the catalog → **missing_signature** (+ rubric tier); mechanical
detectors for shapes `discover` doesn't try — **double definition** (both edges define), **acrostic**
(initials spell the answer), **hidden** (answer inside clue letters), **homophone** (homophone
indicator present); pieces+shape+def all present → **solves_now** (re-run to confirm); recognised
mechanism with nothing assembling → **missing_engine**.

**31284 diagnosis (ran live):** 13 clues remain (rest already fixed) — **12 missing_data** (each
with the exact unresolved word) + **1 solves_now (MUSIC)**.

**TWO KNOWN ROUGH EDGES (user said LEAVE AS-IS for now — do not fix unless asked):**
- **MUSIC (10077838):** the acrostic *pattern* is detected (initials spell MUSIC) so it says
  "solves_now", but the re-solve still fails — the acrostic engine needs the indicator/def. The
  acrostic rule is a touch optimistic; should be "acrostic present but indicator/def missing →
  missing_data".
- **ISSUE (10077822):** flagged "possible double definition" but it's really TISSUE − first letter
  (a beheadment). The one-defining-edge + short-clue DD guess misfired.

---

## 5. THE SIGNATURE FLOW (built, safe-by-construction — read before touching)

When a hand-solve implies a signature the catalog lacks (`_try_create_signature` in wfw_web.py, on
the `/hs` Resolve path), the rubric sorts it and files it:
- `catalog_creator.rubric_tier(cand, answer)` — deterministic §3/§9 rubric: **L**=count of synonym
  slots (SYN_F/REM_F); **G**=a free-choice op (anagram/container/insertion/deletion/positional)
  over ≥1 synonym (charade/reversal/homophone = fixed → G=false); short-amp = answer≤4 AND L≥2.
  PASS iff not-G AND L≤3 AND not short-amp, else PENDING. (Tested 12 cases incl. all design examples.)
- **Risky → PENDING-only** (`_create_pending_signature`): filed directly; the clue turns **amber**.
  Safe by construction — it runs in the final cascade stage, so it can only ever turn FAIL→amber,
  never a green pass. Kept iff it fires; else rolled back.
- **Safe → PASS-tier eligible** (`_create_pass_signature`): filed as PENDING first (amber — never
  green off nothing), promoted to green ONLY after the automatic before/after A/B is clean.
- **A/B harness** `core/ab_signature.py`: `run_ab(file_fn)` solves the corpus work-list TWICE (once
  before, once after — two FRESH solves, NOT vs the stored verdicts, which drift). `try_promote(tid,
  trigger_ids)` flips pending→pass, runs the A/B, keeps `pass` ONLY if clean (0 regressions, no
  stray new passes), else reverts to pending. **CLI:** `python -m core.ab_signature promote <tid>
  <clue_id>`. Full run ~30 min (two corpus passes) → it's a CLI/background step, not a click.

**BEHAVIOUR CHANGE:** hand-solving a clue whose shape is a novel *pass-tier* signature no longer
turns it green immediately — it goes amber until the A/B promotes it. The KEEP→amber happy-path is
verified by composition (the pending stage firing was proven in the signature-tiers build); a live
novel-pending hand-solve has NOT been run (novel pending shapes are rare — common shapes already
exist → "already in catalog").

---

## 6. HAND-SOLVER COMMIT NOW ENRICHES THE DB

`/hsmanualcommit` (wfw_web.py) — the **Commit (manual)** button on `/hs` — now saves its reusable
pieces to `cryptic_new.db` on a SUCCESSFUL commit (dedup built in): synonym→`synonyms_pairs`,
definition→`definition_answers_augmented`, substitution/abbr→`wordplay`, indicator→`indicators`.
NOT saved: **link words** (user rule — don't let them overlap with indicators), letters, anagram
fodder, deletion, filler. Routing is ROLE-DRIVEN: abbreviations reach `wordplay` only when tagged
*substitution*, never `synonyms_pairs` (user was firm on this). Before, manual solves taught the
system nothing.

---

## 7. KEY FILES

- `core/triage.py` — NEW. Puzzle collection, markdown report, JSON caches (classified / diagnoses /
  rejected load-save), enrichment apply/present/reject helpers.
- `core/triage_classify.py` — NEW. The deterministic classifier (§4).
- `core/ab_signature.py` — NEW. The before/after A/B harness + `try_promote` (§5).
- `core/catalog_creator.py` — MODIFIED. Added `rubric_tier`; `add_signature(..., tier=)` (from the
  tiers build); `discover` (the decomposition-finder the classifier uses); `_signature_str`.
- `core/wfw_web.py` — MODIFIED (big). The `/triage` page + routes; `_try_create_signature` reworked
  (rubric routing); `_create_pending_signature` / `_create_pass_signature`; `/hsmanualcommit` DB save.
- `core/diagnose.py` — the per-clue evidence tool the classifier reuses (`pieces`, `_hit`).
- `core/engine_registry.py`, `core/catalog_loader.py`, `core/signature_reviews.py` — the
  signature-tiers machinery (built before this session; still uncommitted).

---

## 8. STATE (git / server)

- Branch `redesign`. **HEAD = 7242ea33** (committed). **EVERYTHING below is UNCOMMITTED:**
  - Modified: `core/catalog_creator.py`, `core/catalog_loader.py`, `core/engine_registry.py`,
    `core/wfw_web.py`.
  - New: `core/ab_signature.py`, `core/signature_reviews.py`, `core/triage.py`,
    `core/triage_classify.py`, `documents/triage/`, the two 2026-07-06 handover docs, scraper JSONs.
- **Nothing committed or pushed this session.** Push gate (unchanged): the overnight full-corpus A/B
  has never been run. Get the user's OK before committing; NEVER push without explicit approval.
- **DB state is local/gitignored** (the reference DB `cryptic_new.db`, the catalog `tier` column,
  `wfw_solve`). The user has ALREADY accepted many 31284 data enrichments on the real DB — so most
  31284 clues now pass and only 13 remain (see §4).
- Server 5099 is running the current code.
- **Memory updated:** `memory/triage_system_built.md` (full detail), `memory/handsolver_commit_saves_to_db.md`,
  and the `MEMORY.md` index top entry. Read `triage_system_built.md` first.

---

## 9. OPEN ITEMS / ROUGH EDGES (do not action unless the user asks)

- The two classifier rough edges (MUSIC acrostic-too-optimistic, ISSUE possible-DD misfire) — user
  said LEAVE AS-IS for now.
- Safe-signature promotion is a CLI, not a page button (could add a background "run check & promote").
- No nightly scrape→solve→triage automation.
- Consolidated dashboard deferred.
- A live novel-pending-signature hand-solve hasn't been exercised end-to-end.
- `core/acrostic_engine.py` EXISTS — so any earlier note that "MUSIC = missing engine" is wrong.

---

## 10. TESTING GOTCHAS (how to test without touching the real DBs)

Every test this session ran fully isolated:
- Copy DBs with the sqlite **backup API** (WAL-safe): `sqlite3.connect(src).backup(sqlite3.connect(dst))`
  — a plain file copy MISSES data in the `-wal` sidecar (wfw_solve verdicts live there) and gives
  false results.
- Redirect the solver's reference-DB reads to the temp copy: `import core.live_db as live_db;
  live_db._default_path = lambda: <temp cryptic_new>`. Then patch `wfw_web.DB`, `store.DEFAULT_DB`,
  `admin_db.CRYPTIC_DB`, `admin_db.MASTER_DB`, `triage.MASTER_DB`, `triage.TRIAGE_DIR`,
  `catalog_creator._CLUES_DB` as needed.
- Drive the REAL routes via `wfw_web.app.test_client()` — this exercises the actual page/route code.
- `catalog_creator.discover` is SLOW on some clues (a 120-clue scan timed out at 2 min). Bound it;
  ~13-25 clues is fine (~1-2 min). The classifier caches results precisely because discover is slow.
- `solve_clue_text` returns **(ctx, parse, name)** — three values. Non-persist when `clue_id=None`.
  Set `wiring["auto_signature"]=False` in any A/B to stop the solve filing signatures.

---

## 11. SUGGESTED FIRST MOVE FOR THE NEW THREAD

1. Read `memory/triage_system_built.md` (has everything, links out).
2. Confirm the server is up and open `http://127.0.0.1:5099/triage?src=telegraph&pnum=31284`.
3. Wait for the user's observations of the page, then act on them — small, tested changes; plain
   English; don't improvise the signature area.
