# Handover — 2026-05-29 — Pipeline Redesign

## What we did today

### 1. Database cleanup
Dropped 9 WFW/prototype tables from `clues_master.db` (local only — droplet kept
as safety net):

Dropped: `atomic_parse_artifacts`, `atomic_parse_review_items`, `atomic_parse_runs`,
`wfw_proof_attempts`, `manual_evidence_edges`, `manual_evidence_nodes`,
`manual_structured_parses`, `solver_stage_contexts`, `clue_word_roles`

Remaining tables (12): `api_explanations`, `clue_pipeline_state`, `clues`,
`indexing_submissions`, `mashup_clue_selections`, `mashup_puzzles`,
`pending_enrichments`, `puzzle_grids`, `rejected_enrichments`,
`sqlite_sequence`, `structured_explanations`, `telegraph_clues`

### 2. Pipeline walk-through (in progress)
Walked through the pipeline step by step from dashboard click to Phase 0.5 (anagram).
Did NOT yet cover: charade, deletion, reversal, container, acrostic, homophone,
Phase 1 (Signature), Phase 2 (Tier 2/AI), Phase 3 (enrichment + re-run).

### 3. Redesign log started
`documents/REDESIGN_LOG.md` — the single source of truth for the redesign.
**Read this before starting work.**

---

## The redesign in one paragraph

We are not starting from scratch. We are reinstating the sound logic that existed in
`cryptic_solver_archive` but was lost when V2 simplified aggressively. The foundation
is three pre-processing steps that run before any solver: (1) atomise the clue so every
character has a numerical reference, (2) run the definition engine to tag the definition
zone, (3) grammar triage to identify which mechanisms are licensed by the indicator atoms.
Every solver then operates only on the wordplay zone, only if licensed.

---

## Design log entries so far

| Entry | Topic | Status |
|-------|-------|--------|
| 1 | Atomisation | Agreed |
| 2 | Definition engine first | Agreed |
| 3 | Grammar triage (incl. DD special case) | Agreed |
| 4 | Spoonerism — two pieces, indicator atom, Haiku fallback | Agreed |
| 5 | Double definition — Haiku if one/zero halves found | Agreed |
| 6 | Anagram — reinstate archive's 4 rules (indicator, proximity, contiguity, whole words) | Agreed |

---

## Where to resume

**Continue the pipeline walk-through from charade** (Phase 0.5, second solver).
For each solver: explain what it does, what V2 lost vs archive, add a design log entry.

Archive to compare against:
- `C:\Users\shute\PycharmProjects\cryptic_solver_archive\stages\`

Solvers still to cover in Phase 0.5:
- `charade.py` / `try_charade` in `batch_v1_solver.py`
- `deletion.py` / `try_deletion`
- `reversal.py` / `try_reversal`
- `container.py` / `try_container`
- `acrostic.py` / `try_acrostic`
- `homophone.py` / `try_homophone`

Then: Phase 1 (Signature solver), Phase 2 (Tier 2), Phase 3 (enrichment).

---

## Key files

| File | Purpose |
|------|---------|
| `documents/REDESIGN_LOG.md` | Design decisions — update after each session |
| `sonnet_pipeline/run.py` | Pipeline orchestrator — the walk-through source |
| `backfill_ai_exp/batch_v1_solver.py` | V2 mechanical solvers |
| `cryptic_solver_archive/stages/` | Archive solvers to compare against |

---

## Ground rules agreed this session

1. Design log only — no code yet
2. One stage at a time
3. Goal is to reinstate archive's good logic on a proper foundation, not rewrite for its own sake
4. Droplet (`165.232.46.255`, `/opt/cordelia/data/`) is the safety net — do not touch
