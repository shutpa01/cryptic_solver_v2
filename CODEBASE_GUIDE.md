# cryptic_solver_V2 — Comprehensive Codebase Guide

**Purpose of this document:** Full reference for a new AI assistant (or developer) taking
over this project. Read every section before touching code. Treat this alongside CLAUDE.md
(standing rules) and the session handoff files in `memory/`.

---

## 1. What This Project Is

**justcordelia.com** is a cryptic crossword hint and explanation service. It serves roughly
500,000 individual clue pages, each reachable via a unique SEO URL. Users can progressively
reveal hints (definition → wordplay type → explanation → answer) without spoiling the answer
up front. The site is mobile-first.

**The system does three things:**
1. **Scrapes** daily cryptic crossword puzzles from Telegraph, Daily Mail, Times, Guardian,
   Independent and others.
2. **Solves / explains** each clue using a cascade of solvers (mechanical → AI).
3. **Serves** the explained clues via a Flask web app.

---

## 2. Project Root Layout

```
cryptic_solver_V2/
├── data/                    # All SQLite databases
├── signature_solver/        # Zero-cost mechanical solver
├── sonnet_pipeline/         # Claude Sonnet AI solver + pipeline runner
├── backfill_ai_exp/         # Batch processing + legacy V1 solvers
├── enrichment/              # Scripts that populate cryptic_new.db
├── web/                     # Flask web application (main UI)
├── honeypot/                # Legacy SEO site (separate Flask app)
├── scraper/                 # Puzzle scrapers (one sub-dir per source)
├── scripts/                 # One-off utilities + nightly runner
├── prototypes/              # Experimental / parallel-development code
├── memory/                  # Session handoff notes and feedback rules
├── logs/                    # Application logs
├── documents/               # Pipeline run reports
├── runs/                    # Per-puzzle run data
├── .env                     # ALL secrets (API keys, logins)
├── CLAUDE.md                # Standing instructions for Claude Code
├── AGENTS.md                # Standing instructions for Codex agents
└── REFACTOR_PLAN.md         # Current project phase plan + checkboxes
```

---

## 3. Databases

### 3.1 `data/clues_master.db` — Live Puzzle Data (~383 MB)

The central database. Written by scrapers; read and written by the pipeline and web app.

#### Table: `clues`
The master record for every clue ever scraped.

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | Auto-increment |
| source | TEXT | 'telegraph', 'dailymail', 'times', 'guardian', 'independent' |
| puzzle_number | INTEGER | Puzzle identifier within source |
| publication_date | TEXT | ISO date string |
| clue_number | TEXT | e.g. '1a', '14d' |
| direction | TEXT | 'across' or 'down' |
| clue_text | TEXT | Full clue as printed |
| enumeration | TEXT | Letter count, e.g. '(9)' or '(5,6)' |
| answer | TEXT | Full answer in uppercase, no spaces |
| definition | TEXT | Extracted definition phrase |
| explanation | TEXT | Human-readable explanation (legacy field) |
| ai_explanation | TEXT | **Primary explanation field.** Structured text produced by the pipeline. Format: `PIECE (synonym="phrase") [indicator: "word"] + PIECE = ANSWER; definition: "phrase"` |
| wordplay_type | TEXT | e.g. 'charade', 'anagram', 'homophone', 'hidden', 'container', 'double_definition', 'cryptic_definition', 'deletion', 'reversal', 'unparsed' |
| original_db | TEXT | Source DB path (legacy) |
| original_id | INTEGER | Source row ID (legacy) |
| has_solution | INTEGER | 1 if fully explained, 0 otherwise |
| reviewed | INTEGER | 1 if manually reviewed |
| silly_award | TEXT | Optional silly award label |

#### Table: `structured_explanations`
One row per clue; stores the solver's output in structured form.

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| clue_id | INTEGER FK | → clues.id |
| definition_text | TEXT | The definition phrase |
| definition_start | INTEGER | Token index where definition starts |
| definition_end | INTEGER | Token index where definition ends |
| wordplay_types | TEXT | JSON array of wordplay type strings |
| components | TEXT | JSON — the full parse tree. See §9 |
| model_version | TEXT | 'signature_solver_v1', 'manual_edit', 'manual_approve', 'claude_review', 'haiku_v1' |
| confidence | REAL | 0.0 – 1.0. Multiply by 100 for percentage. |
| created_at | TIMESTAMP | |
| updated_at | TIMESTAMP | |
| source | TEXT | Denormalised: source name |
| puzzle_number | INTEGER | Denormalised: puzzle number |
| clue_number | TEXT | Denormalised: clue position |

**Confidence thresholds (×100):**
- HIGH: ≥ 70
- MEDIUM: 40 – 69
- LOW: < 40
- FAIL: verifier failed to parse
- PENDING: not yet processed

**Model version semantics:**
- `manual_edit` / `manual_approve` — human-edited. **Protected: never auto-overwrite.**
- `signature_solver_v1` — mechanical solver output
- `claude_review` — Sonnet pipeline output
- `haiku_v1` — Haiku-derived definition

#### Table: `clue_word_roles`
Per-word role analysis for the W-F-W (Word-For-Word) admin display. Written by the
verifier after Re-verify; can be overridden by admin.

| Column | Notes |
|--------|-------|
| clue_id | FK → clues.id |
| word_index | 0-based token position in clue_text |
| word_text | The actual clue word |
| role | One of the ~50 role labels (see §7.4 for full list) |
| source | 'auto' (verifier-written) or 'manual' (admin override) |
| letters | Uppercase letters this piece contributes, e.g. 'STAIR' |
| piece_key | Integer grouping pieces (1, 2, 3...) into wordplay slots |
| updated_at | Timestamp |

**Primary key: (clue_id, word_index).**
`write_auto_roles()` never overwrites `source='manual'` rows.
`write_manual_role()` always overwrites, sets `source='manual'`.

#### Table: `pending_enrichments`
Queue of gaps discovered during solving. Processed via clue page Accept buttons or
`apply_candidates.py`. Writing to this table is NOT the same as writing to `cryptic_new.db`
— it is a staging queue only.

| Column | Notes |
|--------|-------|
| id | PK |
| type | 'synonym', 'abbreviation', 'definition', 'indicator' |
| word | The source phrase |
| letters | The target value (uppercase) |
| answer | The clue answer (context) |
| clue_text | The full clue text (context) |
| source | Puzzle source |
| puzzle_number | Puzzle number |
| created_at | Timestamp |

#### Table: `rejected_enrichments`
Blacklist of false enrichments that must not be re-queued.

#### Table: `api_explanations`
Raw Claude API call output (prior to structured storage). Stores full token counts,
timings, model used, raw text, and review status.

#### Table: `puzzle_grids`
Grid solutions (letter arrays) for puzzles where a grid was recovered.

| Column | Notes |
|--------|-------|
| solution | Space-separated rows of letters |
| grid_rows / grid_cols | Grid dimensions |
| api_folder / api_type / api_id | Source identifiers for grid retrieval |

#### Table: `indexing_submissions`
Log of when each puzzle URL was submitted to Google Indexing API.

---

### 3.2 `data/cryptic_new.db` — Reference Data (~324 MB)

**READ-ONLY from the application's perspective.** Written only by:
- Admin Accept buttons on the clue page (source='admin_clue_page')
- `enrichment/` scripts
- `apply_candidates.py`

**Never write to this DB directly in code without explicit user permission.**

#### Table: `synonyms_pairs`
Maps a phrase to a word/answer it can mean.

| Column | Notes |
|--------|-------|
| id | PK |
| word | Lowercase source phrase, e.g. 'look that\'s fixed' |
| synonym | UPPERCASE target, e.g. 'STARE' |
| source | Provenance, e.g. 'admin_clue_page', 'auto_mined' |

Query pattern: `SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(synonym)=?`

#### Table: `indicators`
Maps indicator words/phrases to their wordplay operation type.

| Column | Notes |
|--------|-------|
| id | PK |
| word | Lowercase indicator phrase |
| wordplay_type | 'anagram', 'reversal', 'hidden', 'homophone', 'deletion', 'container', etc. |
| subtype | Refinement (optional) |
| confidence | 'high', 'medium', 'low' |
| frequency | Usage count (optional) |
| source | Provenance |

#### Table: `wordplay`
Abbreviations and substitutions. `indicator` → `substitution`.

| Column | Notes |
|--------|-------|
| id | PK |
| indicator | Lowercase source word, e.g. 'son' |
| substitution | UPPERCASE result, e.g. 'S' |
| category | NULL for abbreviations; 'dbe' for definition-by-example markers (NOT loaded as abbrevs) |
| confidence | |

#### Table: `definition_answers_augmented`
Definition phrase → answer mappings.

| Column | Notes |
|--------|-------|
| definition | Lowercase phrase |
| answer | UPPERCASE answer |
| source | Provenance |

#### Table: `homophones`
Sound-alike word pairs.

| Column | Notes |
|--------|-------|
| id | PK |
| word | Lowercase |
| homophone | Lowercase sound-alike |

Query: both directions needed —
`(LOWER(word)=? AND LOWER(homophone)=?) OR (LOWER(word)=? AND LOWER(homophone)=?)`

---

### 3.3 `data/word_roles.db` (~12 MB)
Pre-computed token role assignments for the signature solver. Cache only — can be rebuilt.

### 3.4 `data/times_explanations.db` (~11 MB)
Times for the Times blog archive. Contains scraped explanations used to enrich
`cryptic_new.db`.

### 3.5 `data/rate_limits.db` (~12 KB)
Per-IP rate limit state for the Flask app.

### 3.6 `pipeline_stages.db` (regenerated per run)
Temporary staging database for pipeline execution. Not preserved between runs.

---

## 4. The Solving Cascade

Clues flow through stages in order. Each stage either produces a HIGH-confidence result
(which stops the cascade) or passes through to the next stage.

```
Stage 0: Signature Solver (mechanical, zero cost)
    ↓ FAIL or LOW
Stage 1: Signature Solver + enrichment (mechanical, zero cost)
    ↓ FAIL or LOW
Stage 2: Haiku (lightweight AI, cheap) — definition extraction only
    ↓ FAIL or LOW
Stage 3: Sonnet Pipeline (full AI reasoning, ~$0.01–0.05/clue)
    ↓ FAIL or very LOW
Stage 4: Leftover processing (manual, human-in-the-loop)
```

Confidence thresholds:
- HIGH (≥70): cascade stops, result stored
- MEDIUM / LOW / FAIL: cascade continues to next stage

---

## 5. Signature Solver (`signature_solver/`)

Zero-cost mechanical solver. No API calls. Uses pattern matching against a catalog of
known wordplay structures.

### 5.1 Entry Point: `solver.py`

`solve(clue_text, answer, db)` → `SolveResult`

Steps:
1. **Definition extraction** — tries 1–4 words from each end of the clue against
   `definition_answers_augmented` + `synonyms_pairs`.
2. **Word analysis** — calls `word_analyzer.py:analyze_phrases()` to tag each
   wordplay word with all possible roles (SYN_F, ABR_F, REV_I, ANA_I, etc.).
3. **Catalog matching** — tries each entry in BASE_CATALOG via `base_matcher.py`.
4. **Slot lookup** — for each F (fodder) slot, `matcher.py:_lookup_slot()` finds
   possible values appearing in the answer.
5. **Combo verification** — `_verify_combo()` / `_verify_reversal_combo()` assembles
   pieces to produce the answer.
6. **Confidence scoring** — `confidence.py:score_result()` assigns 0–100.

### 5.2 `word_analyzer.py`
Builds a per-word role analysis for every token in the wordplay portion of the clue.

Roles assigned per word: `SYN_F` (synonym fodder), `ABR_F` (abbreviation), `REV_I`
(reversal indicator), `ANA_I` (anagram indicator), `CON_I` (container indicator),
`HID_I` (hidden indicator), `HOM_I` (homophone indicator), `DEL_I` (deletion indicator),
`POS_I_*` (positional indicators), `LINK` (filler word).

**Known bottleneck:** synonyms with >4 characters are capped at 20 per word. If the
required synonym is 21st or later in the DB list, it is invisible to the solver.
Fix applied 2026-04-06: also include any synonym whose reverse appears in the answer.

### 5.3 `base_catalog.py`
Defines ~68 base patterns (collapsed from 694 positional variants).

Each `BaseEntry` has:
- `pattern`: tuple of 'I' (indicator) and 'F' (fodder) slots
- `operation`: 'charade', 'anagram', 'container', 'reversal', 'hidden', 'homophone', etc.
- `n_indicator`, `n_fodder`: slot counts
- `tier`: 1–4 (frequency tier for confidence scoring)

`OPERATION_INDICATOR_TYPE` maps each operation to the required indicator type(s).

### 5.4 `base_matcher.py`
`match_base()` iterates BASE_CATALOG and for each entry calls `_place_spans()`.

`_place_spans()`:
- Assigns wordplay words to pattern slots (each slot gets ≥1 word)
- Leaves gaps for indicator words (I slots)
- Validates that indicator word type matches the operation requirement

**Known bug fixed 2026-04-06:** when `n_indicator=0`, `_place_spans` couldn't skip the
indicator word because `assigned_ind` was None. Fix: set `assigned_ind` from
`OPERATION_INDICATOR_TYPE` even when `n_indicator=0`.

### 5.5 `matcher.py`
`_lookup_slot()`: for each F slot, finds possible values (synonyms, abbreviations) that
appear in the answer (or reversed in the answer).

`_verify_combo()`: assembles pieces in order, checks equality with answer.

`_verify_reversal_combo()`: tries permutations of pieces (capped at 4), reverses each
combination, checks equality.

**Known bug fixed 2026-04-06:** reversal_charade didn't try permutations, so
(NIT, LOVER, G) never became (LOVER, NIT, G) → reversed → REVOLTING.

### 5.6 `db.py` (RefDB)
Loads `cryptic_new.db` into memory at startup for fast lookups.

Methods:
- `is_link_word(word)` — True if word is a filler/joiner
- `get_synonyms(word)` → list of UPPERCASE synonyms
- `get_abbreviations(word)` → list of UPPERCASE substitutions
- `get_indicator_types(word)` → list of indicator operation types
- `is_definition_of(phrase, answer)` → True/False
- `get_homophone(word)` → list of sound-alikes

### 5.7 `confidence.py`
Scores a `SolveResult` 0–100 based on:
- Definition match quality
- Word role precision (how many words accounted for)
- Assembly confidence
- Pattern frequency tier

---

## 6. Sonnet Pipeline (`sonnet_pipeline/`)

AI-powered solver. Invokes Claude Sonnet for complex reasoning.

### 6.1 Entry Point: `run.py`

```bash
python -m sonnet_pipeline.run 31240 --source telegraph --write-db
python -m sonnet_pipeline.run 31240 31241 --source telegraph --write-db
```

Flags:
- `--mode 1`: standard mode (used by nightly)
- `--write-db`: persist results to clues_master.db
- `--no-review`: skip human review step
- `--force-api`: bypass signature solver, go straight to Sonnet
- `--source`: 'telegraph', 'dailymail', 'times', 'guardian', 'independent'

### 6.2 `solver.py`
Core logic for each wordplay type.

Assembly methods:
- `try_charade()` — concatenate pieces in order
- `try_anagram()` — shuffle + match
- `try_container()` — one piece inside another
- `try_reversal()` — reverse piece
- `try_deletion()` — remove substring
- `try_homophone()` — sound-alike lookup + concatenation
- `try_hidden()` — answer as substring of clue words
- `try_double_definition()` — two definitions of same answer

### 6.3 `verify_explanation.py`
**THE VERIFIER.** Checks whether a proposed ai_explanation correctly accounts for
every word in the clue and correctly assembles to produce the answer.

Three rules:
1. **Assembly** — pieces concatenate (in stated order) to produce the answer exactly
2. **Mechanism** — every leaf piece and indicator must be authorised by an exact DB row;
   no fallbacks
3. **Residue** — every wordplay word must be accounted as: piece source / indicator /
   DB-listed link word; none left over

The verifier writes `clue_word_roles` rows (source='auto') when it runs.
**NEVER MODIFY THIS FILE without explicit user permission.** It is the integrity gatekeeper.

### 6.4 `word_roles_store.py`
Manages `clue_word_roles` table in `clues_master.db`.

Key functions:
- `write_auto_roles(clue_id, classified_words, conn)` — writes auto-classified roles;
  never overwrites `source='manual'` rows
- `write_manual_role(clue_id, word_index, word_text, role, letters)` — admin override;
  always writes with `source='manual'`
- `get_roles(clue_id)` → list of `(word_index, word_text, role, source, letters, piece_key)`
  ordered by `word_index`

### 6.5 `enricher.py`
`ClueEnricher` — gathers word frequencies, indicators, definitions from RefDB before
the Sonnet call. Provides context to the prompt.

### 6.6 `enrichment_gate.py`
Decides which stage of the cascade to use for each clue based on current confidence.

### 6.7 `sig_adapter.py`
Converts signature solver output format to the Sonnet pipeline format.

### 6.8 `sig_enrichment.py`
Collects synonym/indicator gaps from signature solver failures and queues them to
`pending_enrichments`.

### 6.9 `fifteensquared_pipeline.py` + `tftt_pipeline.py`
Process blog explanations (Fifteensquared, Times for the Times) to extract structured
parses. These are the primary sources for Times, Guardian, and Independent explanations.
Much cheaper than Sonnet — relies on Haiku for definition extraction from blog text.

---

## 7. Flask Web Application (`web/`)

### 7.1 App Factory: `__init__.py`

Creates the Flask app. Registers all blueprints. Sets up:
- Template filters: `wordplay_label`, `source_name`, `def_missing`, `format_answer`,
  `clickable_words`
- Global `RefDB` reference (loaded once at startup)
- Admin session check from HMAC-verified cookie or `?admin=<KEY>` query parameter
- `g.is_admin` boolean available in all routes

**Admin activation:** append `?admin=<ADMIN_KEY>` to any URL. Sets a persistent session
cookie. Cleared by `/admin/logout`.

### 7.2 Routes

#### `browse.py`
- `GET /` — source listing (telegraph, times, guardian, etc.)
- `GET /source/<source>/` — puzzle list for a source with pagination
- `GET /source/<source>/<type>/` — filtered by puzzle type (prize/cryptic/sunday/etc.)

#### `puzzle.py`
- `GET /puzzle/<source>/<puzzle_number>` — full puzzle view with all across/down clues
  and their hint tiers, grid if available

#### `clue.py` — PRIMARY SEO ENTRY POINT
- `GET /clue/<slug>` — individual clue detail page

  Slug format: `{clue_id}-{clue-text-words-truncated-to-12}` e.g.
  `10066249-reportedly-look-that-s-fixed-properly-in-part`

  Legacy slug formats (pre-commit 8efd6532) redirect to new format via 301.

  This route:
  1. Loads clue + structured_explanation + clue_word_roles
  2. Builds `role_groups` for Word-For-Word display
  3. Runs Accept-button scan (primary + secondary) to find enrichment gaps
  4. Computes per-piece `contributes` (letters each piece puts into the answer)
  5. Detects homophone pairs for homophone Accept buttons
  6. Renders `clue.html` with full context

  **Accept button system (primary scan):**
  Walks `clue_word_roles`, groups by `piece_key` or consecutive roles. For each group,
  checks if the phrase→letters pair is in the appropriate reference table. If missing,
  attaches an `accept_target` to the first word's row for template rendering.

  **Accept button system (secondary scan):**
  Parses `ai_explanation` for patterns like `WORD (synonym="phrase")` or
  `[indicator: "phrase"]`. For each multi-word phrase not in the DB, places an
  Accept button at the first matching clue word. Prefers multi-word phrases over
  single-word claims already placed by primary scan.

  **Important regex fix (2026-05-15):** Patterns `[^"\']+` were changed to `[^"]+` in
  all synonym/abbreviation patterns to allow apostrophes within double-quoted phrases
  (e.g. `synonym="look that's fixed"` previously truncated at the apostrophe).

  **W-F-W contributes calculation:**
  For each role group with letters L:
  - If L appears in answer: `contributes = L` (or reversed form if reversed)
  - If wordplay_type is 'homophone' or 'spoonerism': `contributes = effective_letters(L)`
  - Otherwise: `contributes = L`
  Note: compound homophone+charade clues (STARE→STAIR+WELL=STAIRWELL) show the
  intermediate synonym value (STARE) as contribution since the homophone transform
  is not a separate visible step. This is a known display limitation.

#### `admin.py`
- `POST /admin/accept-enrichment/<clue_id>/<word_index>` — accepts an enrichment:
  - type='synonym' → INSERT into `synonyms_pairs`
  - type='abbreviation' → INSERT into `wordplay`
  - type='definition' → INSERT into `definition_answers_augmented`
  - type='indicator' → INSERT into `indicators`
  - type='homophone' → INSERT OR IGNORE into `homophones`
  - Also clears any matching `rejected_enrichments` row.
  - **All inserts go to `cryptic_new.db` with source='admin_clue_page'.**

- `POST /admin/word-role/<clue_id>/<word_index>` — saves a manual role override:
  - Calls `write_manual_role()` in `word_roles_store.py`
  - Returns HTMX fragment + `HX-Refresh: true` to reload page

- `GET/POST /admin/edit/<clue_id>` — inline clue editing (clue_text, answer,
  wordplay_type, explanation, ai_explanation, has_solution)

- `GET /admin/coverage` — coverage dashboard showing % solved/explained per puzzle

- `GET /admin/reverify/<clue_id>` — runs the verifier on a single clue, updates
  `clue_word_roles`, returns result

- `GET /admin/reverify-puzzle/<source>/<puzzle_number>` — re-verifies all clues
  in a puzzle

#### `hints.py`
- `GET /reveal/<clue_id>/<step>` — returns hint fragment for step 1–4:
  1. Definition text
  2. Wordplay type label
  3. Full explanation
  4. Answer

#### `seo.py`
- `GET /robots.txt` — search engine directives
- `GET /sitemap_index.xml` — master sitemap index
- `GET /sitemap/<n>.xml` — individual sitemap pages (1000 URLs each)

#### `clue_seo.py`
JSON-LD schema generators:
- `generate_meta_description()` — 155-char SEO description
- `generate_faq_schema()` — FAQPage structured data
- `generate_breadcrumb_schema()` — BreadcrumbList
- `generate_word_roles_schema()` — DefinedTermSet (per-word explanation)

#### `helper.py`
HTMX widget endpoints for the inline word helper (definition lookup, synonym suggestions).

#### `learn.py`
Static help/tutorial pages explaining how to use the hint system.

#### `tools.py`
Utility endpoints (bulk operations, data exports for admin).

### 7.3 `models.py`
Business logic and data access layer. Key functions:

- `classify_puzzle(source, puzzle_type)` — returns display label and puzzle category
- `get_puzzle_list(source, page, per_page)` → paginated list with coverage %
- `get_puzzle_clues(source, puzzle_number)` → all clues with hint tiers
- `get_clue_by_id(clue_id)` → single clue dict
- `compute_hint_tier(confidence, model_version)` → 'HIGH', 'MEDIUM', 'LOW', 'FAIL', 'PENDING'
- `get_hint_steps(clue)` → ordered list of hint steps available
- `get_hint_content(clue, step)` → content for a specific hint step
- `_build_explanation(clue)` → renders ai_explanation or components to display text
- `compute_solve_source(clue)` → human-readable credit line

**Hint tier logic:**
- HIGH (≥70%): all 4 hints available to all users
- MEDIUM/LOW: wordplay type and explanation hidden from non-admin users
- FAIL/PENDING: only definition + answer available

### 7.4 `coverage.py`
Computes coverage statistics per puzzle and globally.

Key functions used by `clue.py` for the contributes calculation:
- `post_op_letters(letters, ai_expl)` — applies operation (reversal, deletion, etc.)
  to transform piece letters to their final form in the answer
- `effective_letters(L)` — strips brackets, normalises
- `deletion_target_letters(ai_expl)` — identifies letters removed by deletion ops

### 7.5 `db.py`
- `get_db()` — read-only SQLite connection (URI `?mode=ro`), stored in `g`
- `get_admin_db()` — read-write connection, stored in `g`
- `close_db()` — teardown hook (registered with `app.teardown_appcontext`)

### 7.6 `rate_limit.py`
Per-IP request throttling. Reads `RATE_LIMIT_ENABLED` from config.
Cloudflare-aware: uses `X-Forwarded-For` with configurable proxy hop count (default 2).

### 7.7 `session_token.py`
Issues session cookies for rate-limit tracking. Tokens are HMAC-signed.

### 7.8 `config.py`
```python
CLUES_DB = "data/clues_master.db"
PUZZLES_PER_PAGE = 30
ADMIN_KEY = os.environ.get("ADMIN_KEY", "dev-admin-key")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret")
RATE_LIMIT_ENABLED = True
CF_PROXY_HOPS = 2          # Cloudflare sits in front
STRIP_JSONLD = False        # Toggle to strip JSON-LD for debugging
```

### 7.9 Templates (`web/templates/`)

- `base.html` — Layout: navbar, footer, mobile-first responsive grid
- `puzzle.html` — Puzzle view: across/down columns, hint tier badges, grid if available
- `clue.html` — **Main SEO page.** Contains:
  - Progressive hint reveal (HTMX-driven, no JS)
  - Word-For-Word (W-F-W) admin panel (visible to admin only)
  - Per-word role dropdowns + letter fields (HTMX POST to /admin/word-role)
  - Accept buttons for enrichment gaps (HTMX POST to /admin/accept-enrichment)
  - Homophone Accept buttons (separate section, detected from ai_explanation)
  - Re-verify button (HTMX, reloads page after 1.5s delay)
  - JSON-LD structured data in `<head>`
  - Guard: `{% if clue.role_groups and clue.wordplay_type != 'unparsed' %}` —
    W-F-W is suppressed for unparsed clues to avoid showing unaccounted words
- `partials/admin_edit.html` — Inline edit form (HTMX fragment)
- `partials/hint_*.html` — Hint reveal fragments

### 7.10 `run_dev.py`
```python
app.run(debug=True, port=5000, host="0.0.0.0", threaded=True)
```
Debug=True means Flask auto-reloads on file changes. No restart needed after editing
Python files.

---

## 8. Scrapers (`scraper/`)

### 8.1 Orchestrator: `scraper/orchestrator/puzzle_scraper.py`
Master daily scraper. Called by `nightly_run.py` with `--only <source>`.
Delegates to per-source scrapers.

### 8.2 Per-source scrapers
Each source has its own directory with:
- Main scraper script (browser automation or API calls)
- Backfill script (historical puzzles)
- Isolated browser profile (Chrome or Firefox) at `.chrome_profile/` or `.firefox_profile/`

Sources:
- `scraper/telegraph/` — Telegraph Daily Telegraph cryptic
- `scraper/times/` — The Times cryptic
- `scraper/guardian/` — The Guardian cryptic
- `scraper/independent/` — The Independent cryptic
- `scraper/dailymail/` — Daily Mail Quick Cryptic
- `scraper/danword/` — Danword answer lookup (fallback when answer missing)
- `scraper/bigdave/` — Big Dave's Crossword Blog (Telegraph explanations)
- `scraper/fifteensquared/` — Fifteensquared blog (Guardian/Independent explanations)
- `scraper/timesforthetimes/` — Times for the Times blog index

### 8.3 `scraper/danword/danword_lookup.py`
Looks up missing answers from Danword. Called by nightly runner step 2 for DT/DM clues
that have no answer after scraping.

Also contains `build_solution_string()`, `find_puzzle_json()`,
`update_puzzle_grid_solution()` — imported by `admin.py` for grid rebuilding.

---

## 9. ai_explanation Format

The `ai_explanation` field in `clues` is the primary data for display and verification.
The verifier parses this field to assign word roles and check correctness.

### Standard format
```
PIECE1 (synonym="source phrase") [indicator_type: "indicator word"] + PIECE2 (synonym="source") = ANSWER; definition: "phrase"
```

### Examples by wordplay type

**Charade:**
```
SUB (synonym="newspaper employee") + STANCE (synonym="opinion") = SUBSTANCE; definition: "matter"
```

**Homophone:**
```
STAIR sounds like STARE (synonym="look that's fixed") [homophone: "Reportedly"] + WELL (synonym="properly") = STAIRWELL; definition: "part of a residential block"
```

**Anagram:**
```
anagram of "PALE MIST" [anagram: "mixed"] = PALMIEST; definition: "most warm"
```

**Hidden:**
```
hidden in "thiS COURTyard" [hidden: "in part"] = SCOUR; definition: "clean"
```

**Container:**
```
N (abbreviation="nitrogen") inside SEAT (synonym="chair") + E (first letter of "entrance") = SENATE; definition: "upper house"
```

**Double definition:**
```
definition 1: "type of service" | definition 2: "one in a field" = LET
```

**Cryptic definition:**
```
cryptic definition: "part of a staircase"; definition: "full clue"
```

### Synonym/abbreviation annotation
```
WORD (synonym="source phrase")         → looks up synonyms_pairs
WORD (abbreviation="source phrase")    → looks up wordplay table
WORD sounds like WORD                  → looks up homophones table
[indicator_type: "phrase"]             → looks up indicators table
```

---

## 10. Nightly Run: `scripts/nightly_run.py`

Scheduled via Windows Task Scheduler at 2am UTC.

**Steps:**
1. **Scrape** — runs `scraper/orchestrator/puzzle_scraper.py --only telegraph` and
   `--only dailymail` (separate invocations; failure of one doesn't stop the other)
2. **Danword backfill** — for any DT/DM clues with missing answers, runs
   `scraper/danword/danword_lookup.py`
3. **Pipeline** — runs `python -m sonnet_pipeline.run` on each new puzzle
   (weekdays only; weekends skipped)
4. Times TFTT auto-check: **DISABLED** (processed manually)
5. Daily Mash-up: **DISABLED**

```bash
python scripts/nightly_run.py              # full run
python scripts/nightly_run.py --dry-run    # show plan without executing
python scripts/nightly_run.py --skip-scraper
python scripts/nightly_run.py --date 2026-05-14
```

Two separate Python virtualenvs are used:
- `PYTHON_PIPELINE`: AI_Solver venv (Sonnet pipeline)
- `PYTHON_SCRAPER`: cryptic_solver_V2 venv (scrapers + web)

---

## 11. Enrichment (`enrichment/`)

Scripts that populate and maintain `cryptic_new.db`. Run `enrichment/run_all.py` to
execute all in sequence.

**Sacred rule:** never write to `cryptic_new.db` directly from application code without
user permission. All enrichment goes through either:
1. Admin Accept buttons → `accept_enrichment` route
2. `enrichment/apply_candidates.py` (processes `pending_enrichments` queue)
3. Explicit enrichment scripts in `enrichment/`

Key scripts:
- `01_mine_times_notation.py` — parses Times crossword notation for indicators
- `03_mine_indicators_from_tags.py` — extracts indicator words from tagged explanations
- `04_mine_definition_pairs.py` — builds definition→answer pairs
- `05_self_learning_enrichment.py` — auto-enrichment from HIGH-confidence solves
- `07_mine_synonym_pairs.py` — extracts synonym pairs from explanations
- `apply_candidates.py` — applies `pending_enrichments` queue to `cryptic_new.db`
- `audit_candidates.py` — reviews pending enrichments before applying

---

## 12. Leftover Processing Workflow

When a puzzle runs through the automated pipeline and some clues remain FAIL or LOW,
"leftover processing" is the manual human-in-the-loop step.

**The canonical workflow is documented in `memory/feedback_leftover_process.md`.**
Always read that file before starting leftover work.

Headlines:
1. Use a **live SQL query** to get the work list — never `collect_for_review.py` /
   `ingest_claude_review.py` (those silently miss clues).
2. Every clue must end with a definition. Leftover processing is the last line of
   defence — no automated layer is the final word.
3. **Coverage check before any DB write:** every `(synonym=...)`, `(abbreviation=...)`,
   `[indicator: ...]`, and definition phrase must already be in the DB or queued in
   `pending_enrichments`.
4. **Self-check before reporting done:** re-run the work-list query. If it returns rows,
   the work is not finished.
5. Honesty over score. False HIGHs are not acceptable.

**Work-list query (standard):**
```sql
SELECT c.id, c.clue_number, c.clue_text, c.answer,
       se.confidence, se.model_version, se.components
FROM clues c
LEFT JOIN structured_explanations se ON se.clue_id = c.id
WHERE c.source = ? AND c.puzzle_number = ?
  AND (se.confidence IS NULL OR se.confidence < 0.7)
  AND se.model_version NOT IN ('manual_edit', 'manual_approve')
ORDER BY c.clue_number
```

---

## 13. Per-Puzzle Processing Sequence

When a new puzzle arrives (from nightly or manual trigger), the sequence is:

1. **Scrape** — clues added to `clues_master.db` via scraper
2. **Danword** (if answers missing) — fills gaps
3. **Pipeline** — `python -m sonnet_pipeline.run <number> --source <source> --write-db`
4. **Blog check** — for Times/Guardian/Independent, check if blog explanation is available
   (`SELECT COUNT(*) WHERE source=? AND puzzle_number=? AND explanation IS NOT NULL`)
   - If blog available: run TFTT/Fifteensquared pipeline (cheaper, more accurate)
   - If blog missing: STOP leftover work until blog appears
5. **Enrichment** — check `pending_enrichments`, process via Accept buttons or
   `apply_candidates.py`
6. **Re-verify** — run Re-verify on the puzzle to update confidence scores and W-F-W
7. **Leftover processing** — manually write parses for remaining FAIL/LOW clues
8. **Re-verify again** — validate leftover parses
9. **Manual approve** — for clues that are correct but hit verifier limits
10. **Index** — submit puzzle URL to Google Indexing API via dashboard

---

## 14. Key Files Reference

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Standing rules for Claude Code (AI assistant) — READ FIRST |
| `REFACTOR_PLAN.md` | Current project phases and checkboxes |
| `memory/MEMORY.md` | Index of all session handoff notes |
| `memory/feedback_leftover_process.md` | Canonical leftover workflow |
| `memory/feedback_no_parts_dumping.md` | Enrichment rules |
| `memory/verifier_bugs_to_fix.md` | Known verifier gaps (do not fix without permission) |
| `web/routes/clue.py` | Individual clue page — largest most complex route |
| `web/routes/admin.py` | All admin write operations |
| `sonnet_pipeline/verify_explanation.py` | The verifier — DO NOT TOUCH without permission |
| `sonnet_pipeline/word_roles_store.py` | clue_word_roles read/write |
| `scripts/nightly_run.py` | Nightly automation entrypoint |
| `signature_solver/base_catalog.py` | Mechanical solver pattern catalog |
| `signature_solver/matcher.py` | Slot verification + combo assembly |
| `enrichment/apply_candidates.py` | Applies pending_enrichments to cryptic_new.db |

---

## 15. Environment and Secrets

All secrets are in `.env` at the project root:

```
ANTHROPIC_API_KEY=...        # Claude API
TELEGRAPH_EMAIL=...
TELEGRAPH_PASSWORD=...
TIMES_EMAIL=...
TIMES_PASSWORD=...
INDEPENDENT_USERNAME=...
INDEPENDENT_PASSWORD=...
DB_PATH=data/clues_master.db
ADMIN_KEY=...                # Activates admin session in web app
SECRET_KEY=...               # Flask session signing
MW_API_KEY_*=...             # Merriam-Webster thesaurus API keys
SMTP_USER=...                # Gmail for alerts
HF_TOKEN=...                 # Hugging Face
```

---

## 16. Non-Negotiable Rules (from CLAUDE.md)

These rules exist because violations have caused real damage:

1. **Never modify working pipeline stage engines** (anything in `stages/`) to fix edge
   cases. Build helper stages instead.
2. **Never delete files without explicit confirmation.** List what you plan to delete
   and wait for approval.
3. **Never run destructive commands** without showing the exact command first.
4. **Never make bulk changes across multiple files in one go.** One file at a time.
5. **Never touch `sonnet_pipeline/verify_explanation.py`** without explicit permission
   in the same message.
6. **Never write to `cryptic_new.db`** without explicit user permission.
7. **Never overwrite rows with `model_version` = 'manual_edit' or 'manual_approve'.**
   These are human-approved parses and are protected.
8. **A question is not an instruction.** "How does X work?" → explain. Do not touch files.
9. **Verify before claiming.** Test through the actual web route, not a standalone script.
   Show actual output as proof.
10. **Honesty over score.** False HIGHs are not acceptable. The explanation must decode
    the wordplay correctly; a HIGH verifier score is not proof of correctness.

---

## 17. Known Structural Gaps and Limitations

These are documented limitations the new assistant should be aware of and not attempt
to work around with per-clue hacks:

| Issue | Location | Status |
|-------|----------|--------|
| Synonym cap (>4 chars, capped at 20) | `word_analyzer.py` | Partially fixed (reverse-included) |
| Multi-word homophone source display | `clue.py` contributes calculation | Known limitation — compound homophone+charade shows STARE not STAIR |
| Verifier strict-link cascade | `verify_explanation.py` | Listed in `memory/verifier_bugs_to_fix.md` |
| Def-not-in-DB cascade | `verify_explanation.py` | Listed in bugs file |
| Hyphenated answer handling | `verify_explanation.py` | Listed in bugs file |
| 7b false positives | `verify_explanation.py` | Listed in bugs file |
| Multi-word indicator absorbing | `verify_explanation.py` | Listed in bugs file |
| Non-contiguous anagram fodder | `verify_explanation.py` | Listed in bugs file |
| Charade joiners not in LINK_WORDS | `verify_explanation.py` | Listed in bugs file |
| `pending_enrichments` not processable in bulk | Dashboard | **CRITICAL missing feature** — no "Process pending enrichments" button exists; enrichments must be applied one-by-one via Accept buttons |

---

## 18. Diagnostic Commands

Useful queries for understanding current state:

```sql
-- Work list for a puzzle (clues not yet HIGH)
SELECT c.id, c.clue_number, c.clue_text, c.answer,
       se.confidence, se.model_version
FROM clues c
LEFT JOIN structured_explanations se ON se.clue_id = c.id
WHERE c.source = 'telegraph' AND c.puzzle_number = 31240
  AND (se.confidence IS NULL OR se.confidence < 0.7)
  AND (se.model_version IS NULL
       OR se.model_version NOT IN ('manual_edit', 'manual_approve'))
ORDER BY c.clue_number;

-- Check pending enrichments for a puzzle
SELECT * FROM pending_enrichments
WHERE source = 'telegraph' AND puzzle_number = 31240;

-- Check synonym exists
SELECT * FROM synonyms_pairs
WHERE LOWER(word) = 'look that''s fixed';

-- Check word roles for a clue
SELECT word_index, word_text, role, source, letters, piece_key
FROM clue_word_roles
WHERE clue_id = 10066249
ORDER BY word_index;

-- Check confidence distribution for a puzzle
SELECT se.model_version,
       ROUND(se.confidence * 100) as pct,
       c.clue_number, c.answer
FROM clues c
JOIN structured_explanations se ON se.clue_id = c.id
WHERE c.source = 'dailymail' AND c.puzzle_number = 17879
ORDER BY se.confidence;
```

---

## 19. Git History and Commit Style

Recent commits show the convention — present-tense imperative, lower-case:
```
tokenizer, clue page, nightly: structural fixes from Times 29542 session
clue page: structural fixes for coverage, contributes and dropdowns
admin reverify: don't treat manual_edit as score-protected
admin word-role: add letter_position_indicator role + reliable save
```

**Always commit before starting a new phase.** Use git history as a recovery mechanism.

---

*Document generated 2026-05-15. Refer to `memory/MEMORY.md` for most recent session
handoffs, which supersede any stale information here.*
