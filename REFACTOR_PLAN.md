# Cryptic Solver — Refactor Plan
## Save as REFACTOR_PLAN.md in project root

**Created: 2026-02-12**
**Status: NOT STARTED**
**Purpose: Survives context loss. Any fresh Claude instance reads this first.**

---

## 0. CURRENT STATE (as of 2026-02-12)

### Project structure
```
project_root/
├── .venv/
├── data/
│   ├── clues_master.db          # LIVE clue source (new — clues come from here)
│   ├── cryptic_new.db           # Reference tables: synonyms_pairs, wordplay, indicators,
│   │                            #   homophones, definition_answers_augmented, definition_answers, synonyms
│   └── pipeline_stages.db       # Regenerated each run — stage results
├── stages/                      # Pipeline stage engines (well-organized)
│   ├── __init__.py
│   ├── anagram.py               # Was: anagram_stage.py — brute-force anagram + evidence fallback
│   ├── compound.py              # Was: compound_wordplay_analyzer.py — 4,693 lines, DB-backed analysis
│   ├── dd.py                    # Was: dd_stage.py — double definition detection
│   ├── definition.py            # Was: definition_engine.py — base definition engine
│   ├── definition_edges.py      # Was: definition_engine_edges.py — REDUNDANT wrapper (see Phase 4)
│   ├── evidence.py              # Was: anagram_evidence_system_patched.py — 1,715 lines
│   ├── explanation.py           # Was: explanation_builder.py — ORIGINAL (see Phase 3)
│   ├── general.py               # Was: general_wordplay_analyzer.py — stage 5 engine
│   ├── lurker.py                # Was: lurker_stage.py — hidden word detection
│   └── unified_explanation.py   # Was: unified_explanation_builder.py — SUPERSET clone (see Phase 3)
├── anagram_analysis.py          # SHOULD BE in engine/ — compound orchestrator
├── evidence_analysis.py         # SHOULD BE in engine/ — evidence scorer
├── persistence.py               # SHOULD BE in engine/ — DB writer for pipeline_stages.db
├── pipeline_simulator.py        # SHOULD BE in engine/ — orchestrator stages 1-4
├── pipeline_stages.db           # DUPLICATE of data/pipeline_stages.db — remove one
├── presets.py                   # SHOULD BE in engine/ — DB_PATH, criteria (TO BE CONSOLIDATED)
├── resources.py                 # SHOULD BE in engine/ — shared utilities
├── report.py                    # Was: puzzle_report.py — INTENDED single entry point (NOT YET WORKING)
├── run.py                       # CURRENT entry point (temporary — to be replaced by report.py)
├── scratch.py                   # Dev scratch file
└── .gitignore
```

### What works
- Pipeline runs via `run.py` → `pipeline_simulator.py` with criteria from `presets.py`
- Clues sourced from `clues_master.db` (live DB)
- Reference tables from `cryptic_new.db`
- Cascade: DD → Definition → Anagram → Lurker → Compound → General
- Functionally identical to old system (no regressions, no improvements)

### What's broken/incomplete
- `report.py` is supposed to be THE entry point but hasn't been tested
- Criteria selection scattered across `presets.py` AND `pipeline_simulator.py`
- Anthropic API calls in report.py are NO LONGER RELEVANT — must be removed
- Root directory is cluttered with files that belong in a package
- Duplicate `pipeline_stages.db` (root and data/)
- All redundancies from dependency map still present (see Phase 1-5 below)

### Key principle — SACRED RULE
**NEVER modify existing pipeline stage engines to fix edge cases.** Build new helper stages instead. Previous attempts to fix edge cases broke working solves elsewhere.

---

## 1. THE PLAN — SEQUENTIAL PHASES

### ━━━ Phase 0: Consolidate entry point ━━━
**Goal: report.py becomes the single entry point. run.py eliminated. Criteria centralized.**
**Risk: MEDIUM — requires understanding current report.py and run.py**

- [ ] **0.1** Read `run.py` — document exactly what it does
- [ ] **0.2** Read `report.py` — document what exists vs what's missing
- [ ] **0.3** Read criteria handling in `presets.py` and `pipeline_simulator.py`
- [ ] **0.4** Strip ALL Anthropic API code from `report.py` (imports, calls, fallback logic)
- [ ] **0.5** Consolidate criteria selection into `report.py` (or a config it owns)
- [ ] **0.6** Wire `report.py` to call the pipeline cascade (DD → Def → Anagram → Lurker → Compound → General)
- [ ] **0.7** Test: `python report.py` produces same results as `python run.py`
- [ ] **0.8** Delete `run.py`

**Test gate: Full pipeline run via report.py matches previous run.py output**

### ━━━ Phase 1: Safe deletions (ZERO risk) ━━━
**Goal: Remove dead code and unused imports. Nothing breaks.**

- [ ] **1.1** Delete `write_puzzle_report()` from `stages/general.py` (~206 lines of dead code)
- [ ] **1.2** Remove unused imports:
  - `import os` in `evidence_analysis.py`
  - `import sys` in `report.py` (if still present after Phase 0)
  - `import os`, `import itertools`, `field` from dataclasses in `stages/evidence.py`
  - `build_wordlist` import in `stages/definition.py`
- [ ] **1.3** Remove dead `return prompt` in `report.py` (was line 885 in old version — locate and remove)
- [ ] **1.4** Remove duplicate `pipeline_stages.db` from root (keep only `data/pipeline_stages.db`)

**Test gate: Full pipeline run still works identically**

### ━━━ Phase 2: Deduplication (LOW risk, HIGH value) ━━━
**Goal: Single source of truth for shared code**

- [ ] **2.1** Move `format_answer_with_enumeration()` to `resources.py`
  - Remove from: `stages/compound.py`, `stages/explanation.py`, `stages/unified_explanation.py`
  - Add import in those three files
- [ ] **2.2** Create ONE canonical `LINK_WORDS` set in `resources.py`
  - Use tight set (like evidence.py version ~80 words)
  - CRITICAL: Do NOT include cryptic indicators: love, back, cut, hold, lead, left, about, even
  - Remove link_words from: `stages/compound.py`, `stages/explanation.py`, `stages/unified_explanation.py`, `stages/evidence.py`
  - Import from resources.py in all four
- [ ] **2.3** Fix indirect imports: `stages/general.py` should import `norm_letters` from `resources.py` directly, not from `stages/compound.py`

**Test gate: Full pipeline run still works identically**

### ━━━ Phase 3: Explanation builder merge (MEDIUM risk) ━━━
**Goal: One ExplanationBuilder, not two diverging copies**

- [ ] **3.1** Confirm `stages/unified_explanation.py` is a strict superset of `stages/explanation.py`
  - The ONLY difference should be `_build_general_formula` code path for non-anagram clues
- [ ] **3.2** Update `anagram_analysis.py` to import from `stages/unified_explanation.py` instead of `stages/explanation.py`
- [ ] **3.3** Delete `stages/explanation.py`
- [ ] **3.4** Rename `stages/unified_explanation.py` → `stages/explanation.py` (clean name)
- [ ] **3.5** Update all imports referencing the old filenames

**Test gate: Full pipeline run still works identically**

### ━━━ Phase 4: Definition layer cleanup (LOW risk) ━━━
**Goal: Eliminate redundant definition_edges.py wrapper**

- [ ] **4.1** Move `has_separator()` logic into `stages/definition.py` (add to return dict)
- [ ] **4.2** Update `pipeline_simulator.py` to import `definition_candidates` from `stages/definition` (not edges)
- [ ] **4.3** Delete `stages/definition_edges.py`

**Test gate: Full pipeline run still works identically**

### ━━━ Phase 5: File organization (LOW risk) ━━━
**Goal: Clean root directory — only entry points at root level**

- [ ] **5.1** Create `engine/` package with `__init__.py`
- [ ] **5.2** Move to `engine/`:
  - `pipeline_simulator.py`
  - `anagram_analysis.py`
  - `evidence_analysis.py`
  - `persistence.py`
  - `resources.py`
  - `presets.py` (if still needed after Phase 0 consolidation)
- [ ] **5.3** Update ALL imports across entire project
- [ ] **5.4** Add `stage_general` table definition to `engine/persistence.py` (currently bypasses persistence layer)
- [ ] **5.5** Update `stages/general.py` to use persistence layer instead of direct SQL

**Test gate: Full pipeline run still works identically**

### ━━━ Phase 6: Integrate scraper ━━━
**Goal: Daily scraper lives inside the project, runs automatically, populates clues_master.db**
**Currently: Separate project/folder. Uses Playwright + mix of approaches. Currently broken (minor).**

- [ ] **6.1** Copy scraper code into project under `scraper/` directory
- [ ] **6.2** Audit scraper — document what publications it covers, what's broken, and why
- [ ] **6.3** Fix broken scraper (the minor issues preventing it from running)
- [ ] **6.4** Ensure scraper writes to `data/clues_master.db` with correct schema
- [ ] **6.5** Add deduplication — scraper must not create duplicate clues on re-runs
- [ ] **6.6** Automate: set up scheduled task (cron on Linux/Mac, Task Scheduler on Windows)
- [ ] **6.7** Add basic logging/error reporting so failed scrapes are visible
- [ ] **6.8** Test end-to-end: scraper populates DB → report.py solves new clues

**Test gate: Scraper runs unattended, populates clues_master.db, report.py processes new clues**

**Target structure:**
```
project_root/
├── scraper/
│   ├── __init__.py
│   ├── run_scraper.py          # Entry point for daily scrape
│   ├── publishers/             # Per-publication scrapers
│   │   ├── telegraph.py
│   │   ├── times.py
│   │   ├── guardian.py
│   │   ├── ft.py
│   │   └── independent.py
│   └── requirements.txt        # Playwright + any scraper-specific deps
```

### ━━━ Phase 7: Architectural improvements (DEFERRED) ━━━
**Not to be started until Phases 0-6 complete and stable**

- [ ] **7.1** Make pipeline resumable — run stage 5 without re-running 1-4
- [ ] **7.2** Add substitution validation to report.py (verify against DB tables)
- [ ] **7.3** Clean up dead run comparison logic in persistence.py
- [ ] **7.4** Resolve circular dependency: anagram.py ↔ evidence.py (lazy import → clean separation)

---

## 2. DATABASE CONTRACT

### clues_master.db (LIVE — clue source)
| Table | Purpose |
|-------|---------|
| `clues` (or equivalent) | Source of clues for pipeline runs |

### cryptic_new.db (REFERENCE — never wiped)
| Table | Read by |
|-------|---------|
| `synonyms_pairs` | compound.py, report.py, resources.py |
| `wordplay` | compound.py, evidence.py, report.py |
| `indicators` | compound.py, evidence.py, report.py |
| `homophones` | compound.py, report.py |
| `definition_answers_augmented` | resources.py, report.py |
| `definition_answers` | resources.py |
| `synonyms` | resources.py |
| `clues` | resources.py (build_wordlist only — NOT for pipeline input) |

### pipeline_stages.db (REGENERATIVE — wiped each run)
| Table | Written by | Read by |
|-------|-----------|---------|
| `stage_input` | pipeline_simulator → persistence | report.py |
| `stage_dd` | pipeline_simulator → persistence | report.py |
| `stage_definition` | pipeline_simulator → persistence | **(orphaned)** |
| `stage_anagram` | pipeline_simulator → persistence | general.py, report.py |
| `stage_lurker` | pipeline_simulator → persistence | report.py |
| `stage_compound` | anagram_analysis → persistence | general.py, report.py |
| `stage_general` | general.py DIRECTLY (bypasses persistence) | report.py |

---

## 3. FILE DEPENDENCY MAP (post-transition names)

```
presets.py → resources.py → stages/definition.py → stages/dd.py
                          → stages/definition_edges.py (REDUNDANT — Phase 4 removes)
                          → stages/compound.py → stages/explanation.py (DUPLICATE — Phase 3 removes)
                          │                    → stages/unified_explanation.py
                          → pipeline_simulator.py → stages/dd.py
                          │                       → stages/anagram.py ←→ stages/evidence.py (circular, lazy)
                          │                       → stages/lurker.py
                          │                       → persistence.py
                          → evidence_analysis.py → pipeline_simulator.py
                          │                      → stages/evidence.py
                          → anagram_analysis.py → pipeline_simulator.py
                                                → stages/explanation.py
                                                → evidence_analysis.py
                                                → persistence.py
stages/general.py → stages/compound.py
                  → stages/unified_explanation.py

report.py → (standalone — reads pipeline_stages.db via SQL, currently has Anthropic API — TO REMOVE)
```

---

## 4. KNOWN REDUNDANCIES REFERENCE

| ID | Issue | Location | Status |
|----|-------|----------|--------|
| R1 | ExplanationSystemBuilder duplicated | explanation.py + unified_explanation.py | Phase 3 |
| R2 | format_answer_with_enumeration triplicated | compound.py, explanation.py, unified_explanation.py | Phase 2 |
| R3 | FOUR link_words lists (some harm explanations) | compound.py, explanation.py, unified_explanation.py, evidence.py | Phase 2 |
| R4 | Dead write_puzzle_report() ~206 lines | general.py | Phase 1 |
| R5 | stage_general bypasses persistence | general.py | Phase 5 |
| R6 | norm_letters imported indirectly | general.py via compound.py | Phase 2 |
| R7-R9,R14-R15 | Unused imports / dead code | Various | Phase 1 |
| R10-R11 | Orphaned DB tables | pipeline_stages.db | Noted |
| R12 | Full pipeline re-run forced | general.py → anagram_analysis → pipeline_simulator | Phase 7 |
| R13 | Dead multi-run comparison | persistence.py | Phase 7 |
| R16 | Redundant definition_edges.py wrapper | definition_edges.py | Phase 4 |

---

## 5. DANGER ZONES

| ID | Risk | Mitigation |
|----|------|-----------|
| D1 | link_words treat cryptic indicators as filler | Phase 2 — canonical tight set |
| D2 | Validation checks letter math only, not substitution correctness | Phase 7 |
| D3 | ExplanationBuilder divergence risk | Phase 3 — merge to single file |
| D4 | Circular dependency anagram ↔ evidence | Phase 7 — deferred |
| D5 | definition_edges.py redundant wrapper | Phase 4 — delete |

---

## 6. INSTRUCTIONS FOR CLAUDE INSTANCES

When starting a new conversation about this project:

1. **Read this file first** — it is the source of truth
2. **Check the status checkboxes** — completed items will be marked [x]
3. **Follow the phases in order** — do NOT skip ahead
4. **Test after every phase** — the test gate must pass before proceeding
5. **Update this file** — mark items complete as you go
6. **NEVER modify working pipeline stages** (stages/ engines) to fix edge cases
7. **Small, surgical changes** — one item at a time, test, commit
8. **If context is getting large**, stop, update this file with current status, and suggest starting a new thread

---

## 7. FUTURE WORK (after refactor complete)

These are the real goals that the refactor enables:

- **Self-contained daily system**: Scraper → clues_master.db → Solver → Results (fully automated)
- **Container helper stage** — 48 verified patterns ready as test cases (from container_deep_dive.py analysis)
- **Single-word synonym helper** via Merriam-Webster API
- **Deletion/reversal/homophone handlers**
- **New helpers insert between general stage and API fallback in report.py**
- **Each successful result feeds back into database** (compounding knowledge advantage)
- **Mobile-first crossword helper app** (long-term goal)
