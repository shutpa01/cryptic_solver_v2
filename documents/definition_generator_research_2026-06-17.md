# Definition-generator research & corrective plan — 2026-06-17

## The problem (user report)
When a clue's definition is **not** in the reference DB, the definition stage generates an
**over-long definition** that swallows words the wordplay needs, so the clue fails. The
user "hardly ever accepts a suggested def." We want to fix this for many clues at once.

## Method
Harness: `core/_research_defgen.py`. Take clues that currently PASS with a **DB-confirmed**
definition (so we *know* the correct definition = `parse.definition.text`). Re-solve each
**as if that definition were not in the DB** (wrap `defines` to return False for the
answer), forcing the floor/extent to generate one. Compare generated vs known definition
and whether the clue still solves. DB-only wiring (no AI), `auto_signature` off — i.e. the
behaviour the batch/clue-page actually shows.

## Results (sample: `id % 250`, 2380 scanned → 470 clean DB-def passes)
| metric | value |
|---|---|
| generator reproduced the correct def | **301 (64.0%)** |
| generated a **LONGER** def | **133 (28.3%)** ← the weak point |
| generated a shorter def | 22 (4.7%) |
| still solved (pass/pending) without the DB def | 387 (82.3%) |
| **BROKE (was pass → now fail)** | **83 (17.7%)** |
| avg extra words on the longer ones | **+1.87** |

So with the definition removed, **~1 in 6 otherwise-solvable clues breaks**, and **>1 in 4
gets a longer definition** than the truth.

## Failure patterns (from the examples)
Two distinct modes:

**1. Over-absorption** — the generated def grabs adjacent function/wordplay words that
should belong to the wordplay or be links:
- OUTBREAK: `Rash's` → `Rash's gone`  (absorbed wordplay "gone")
- NEEDLING: `annoying` → `is annoying`  (absorbed link "is")
- MATADOR: `Ring performer` → `Ring performer has`
- STABLE: `May claim` → `May claim this`
- ADIPOSE: `fat` → `getting fat`
- SWEETPEA: `Climber` → `Climber we see apt to come`  (+4 words!)
- ITERATES: `Repeats` → `Repeats treatise`

**2. Wrong edge** — the generator put the definition at the *opposite* end of the clue,
because the wordplay can over-reach from either side and nothing pins the correct edge:
- ETHANE (hidden): correct `Gas` (start) → generated `all of it` (end)
- EXCORIATE (anag): correct `run down severely` (end) → `Upset or excite a run` (start)
- ARUNDEL: correct `Sussex town` (end) → `A series was` (start)
- HAPPENSTANCE: `coincidence` → `apes with penchant for coincidence`

## Root cause
The definition is chosen **definition-first**, before/independent of the wordplay:
1. **Floor ordering** — `definition_engine._residue_edge_splits` offers edge windows
   **longest-first**. The first window whose remainder *happens* to reconstruct the answer
   wins, biasing toward long defs / short wordplay.
2. **Grammar EXTENT growth** — `_extend_split` / `grammar.extend_definition_indices` *grow*
   a definition outward by absorbing adjacent function words; without the DB anchor this
   over-grows.
3. **No wordplay-driven boundary & no edge disambiguation** — nothing forces the definition
   to be the *minimal* leftover, and nothing tells the stage which edge is the wordplay.

## Corrective action plan (the user's "wordplay-first, force the residue")
We already know the answer, so let the **wordplay define the boundary**:

**A. Wordplay-maximising boundary (primary fix).**
Enumerate definition boundaries **shortest-definition-first (= longest-wordplay-first)** and
accept the first boundary where the wordplay engine **reconstructs the answer exactly AND
accounts for every non-definition word**. The definition is then the *minimal* leftover edge
— forced, not looked up. This directly removes over-absorption (mode 1) and, by requiring
full wordplay coverage, disambiguates the edge (mode 2).
- Implementation: in the floor path, reverse the window order (shortest def first) and/or
  have the cascade prefer the solve with the **fewest definition words / most accounted
  wordplay**, rather than first-that-reconstructs. Keep it gated to the no-DB-def case so
  confirmed solves are untouched.

**B. Stop EXTENT over-growth on unconfirmed defs.**
Only grow a definition that is DB-confirmed, and only via the post-roles `extend_definition`
(which already refuses to swallow wordplay words). Never grow a floor/AI provisional def.

**C. Haiku as validator, not generator.**
Once A has fixed the boundary, Haiku's job is a **yes/no on the specific short residue**
("is `fat` a definition of ADIPOSE?"), not inventing a phrase. This is far more likely to be
accepted than today's free-form suggestion, and it only runs on the per-clue path.

## How to validate the fix
Re-run `core/_research_defgen.py` after each change. Targets vs the 2026-06-17 baseline
(reproduced 64.0% / longer 28.3% / broke 17.7%):
- reproduced-def ↑ (toward the boundary the wordplay implies)
- longer ↓ and avg-extra-words ↓
- broke ↓ (the headline: fewer otherwise-solvable clues lost when the def isn't in the DB)

## Files
- Harness: `core/_research_defgen.py` (arg = sample mod, default 120)
- Raw output: `_research_defgen_out.txt`
- Relevant code: `core/definition_engine.py` (`find_definitions`, `_residue_edge_splits`,
  `_extend_split`, `_peel_dbe`), `core/definition_fallback.py` (Haiku path).
