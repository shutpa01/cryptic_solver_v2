# Handover — ASTRIDE false-PASS fix (location vs operation indicators)

Date: 2026-06-22
Branch: redesign
Status: IN PROGRESS. Code UNCOMMITTED. **No database has been changed.**

---

## The problem

Clue **10074879**: *"In riding position since start of that bike trip"* = **ASTRIDE**.
It was a **false PASS** — a totally wrong explanation marked pass. The user calls this
"the worst possible type" of error; it must never happen.

The fabricated parse (via the `charade_deletion` engine):
- `since` → PAST → behead → **AST**
- `bike` → **RIDE**
- AST + RIDE = ASTRIDE  ✗ (wrong reasoning, right letters)

The **correct** parse:
- `In riding position` = definition
- `since` → AS
- `start of that` → **T** (first letter of "that" — a SELECTION)
- `bike trip` → **RIDE** (a bike trip = a ride)
- AS + T + RIDE = ASTRIDE  ✓

## The user's model (the principle to build on)

**START is a letter-LOCATION indicator.** It points at a letter; it is NOT a deletion.
A deletion happens only when a location indicator is **combined with a separate
OPERATION indicator**.

- `start of dog` → **D** (location only → select the first letter)
- `start off dog` → behead (location "start" + operation "off")

So letter-location words (start/first/head/end/last/tip…) must never license a deletion
by themselves. Selection handles them when alone; deletion needs an operation word too.

## Two separate root causes

### (A) Location words mis-typed as deletions — being fixed
14 first-letter location words carry a bogus `deletion/head` row in `cryptic_new.db`.
Each ALSO already has the correct selection typing (`acrostic/initial` + `parts/first_use`):

```
start, first, beginning, initial, initially, lead, leader, leading,
opener, opening, capital, introduction, front, tip
```

Deliberately EXCLUDED (genuinely ambiguous — can be real beheads like "topless"):
`head, top, header, heading, topped, topping`.

### (B) The TRIP problem — NOT fixed yet
`trip` is typed **anagram-only**. But `charade_deletion_engine.py:103` `is_glue` accepts
**any** indicator-typed leftover word as "charade glue":

```python
def is_glue(k):
    return bool((is_link and is_link(words[k].text)) or types(k))   # ANY type
```

So "trip" (an anagram word, irrelevant to this charade/deletion) is silently absorbed as
a bogus *"charade indicator"* instead of being flagged as **unaccounted content**. If it
weren't absorbed, the parse would fail honestly. This is role-by-elimination (banned).
Proven: ASTRIDE OLD parse annotates `trip` as `'charade indicator'` while
`indicator_types('trip') == {'anagram'}` and `is_link('trip') == False`.

**Fix direction:** only accept a leftover as glue if its indicator type is RELEVANT to
the operation in play (charade/link/deletion), not anagram or other unrelated types.

## What has been DONE (uncommitted)

The cascade's plain-deletion path is **`core/signature_verifier.solve_deletion`**, NOT the
retired `core/deletion_engine.py`. Changes made there:

1. **core/engine_registry.py**
   - Added `"selection_rules": selection_rules` to the wiring dict (it existed as a local
     function but wasn't exposed).
   - Passed `loc_rules=wiring.get("selection_rules")` into the `deletion_solve(...)` call.

2. **core/signature_verifier.py**
   - Added module map `_LOC_DELOP = {"first":"behead","last":"curtail","outer":"outer","middle":"heartless"}`.
   - `_resolve_deletion_piece(...)`: a letter-location indicator inside the deletion slot
     is now folded into the indicator (accounted, not required to be a link) AND contributes
     its drop-op — but ONLY when a genuine deletion operation word is also present in the
     slot (`has_del_word = bool(typed) or generic`). A location word never licenses a
     deletion alone, so "start of X" stays a selection.
   - Threaded `loc_rules` through `solve_deletion` → `_try_split` → `_resolve_deletion_piece`.

   This change is **INERT on the current DB** (location words are still deletion-typed there,
   so the fold is skipped); it only activates once their deletion typing is removed.

**Verified** (via simulated post-DB wiring that strips the 14 words' deletion typing):
- EWER (1710775, "start off") → PASS (start=location, off=operation, behead) ✓
- LAIR (1711328, "Abandoning front") → PASS ✓
- ALLOT (1711450, "sack leader") → PASS ✓
- ASTRIDE (10074879) → FAIL honestly (no fabrication) ✓

## What is STILL TO DO

1. **Apply the same location/operation split to `core/charade_deletion_engine.py` and
   `core/container_deletion_engine.py`.** A/B confirmed genuine regressions there once the
   14 words lose deletion typing:
   - ALLUSION (1711996, charade_deletion) — "thing leader demolished" → A + LLUSION
     (ILLUSION beheaded). NOTE the operation word here ("demolished") may not be
     deletion-typed in the DB — so this case is harder than EWER and needs care.
   - OYSTER (1711842, container_deletion) — "nurse short of capital" → Y in F[OSTER]
     ("capital"=location, "short"=operation).
   - 1712276, 1713814 (container_deletion) — also flagged.
   - charade_deletion accounts leftover location words via `is_glue` (they keep selection
     typing) BUT its op comes only from `is_del` words — so when the only deletion signal
     was the location word, the gate fails after the DB change. container_deletion's
     `_build` (line ~243) rejects any residue that isn't con/del-indicator or link — no
     glue escape — so it breaks the same way the verifier did.

2. **(B) Tighten `is_glue` / leftover acceptance** so an irrelevant-type word (e.g.
   anagram-typed "trip") is NOT absorbed as glue. Decide scope: charade_deletion only, or
   wherever the "any indicator = glue" pattern appears.

3. **Reclassify the DB** — run `core/_apply_inclusion_dedelete.py --apply` (backs up to
   `cryptic_new.db.bak-incl-dedelete` first; deletes the 14 `deletion/head` rows). DRY-RUN
   confirmed exactly 14 rows. THEN invalidate caches / restart server.

4. **Re-run the full A/B clean** (`core/_ab_inclusion_dedelete.py`) — must show fabrications
   gone AND zero genuine deletion PASS lost.

5. **Verify ASTRIDE + EWER live on :5099**, then commit (one engine per commit, A/B-gated).

## Files created this session (all UNCOMMITTED, all helper scripts)
- `core/_ab_inclusion_dedelete.py` — A/B harness. Strips the 14 words' deletion typing via
  patched wiring (indicator_types + deletion_subtypes), runs the FULL cascade both ways,
  flags PASS changes. Slow (full cascade ×2 per clue, ~tens of minutes for 250).
- `core/_apply_inclusion_dedelete.py` — DB cleanup (dry-run by default; `--apply` to write).
- `logs/ab_dedelete.log` (v1, blanket removal, showed the genuine regressions),
  `logs/ab_dedelete2.log` (v2, with the verifier fix; still showed charade_deletion +
  container_deletion regressions — expected, those engines not yet fixed).

## Separate, harder follow-up: actually SOLVING ASTRIDE correctly
Making it solve as AS+T+RIDE (not just fail honestly) needs:
- the missing synonym `bike trip → RIDE` (DB only has `bike → RIDE`), and
- a selection-piece-inside-a-charade capability (first-letter "start of that → T" as a
  charade piece) — the charade engine doesn't assemble that today.
This is out of scope for the fabrication fix.

## Standing rules reinforced this session
- No fabrication / no false PASS — the cardinal rule.
- Never assign a role by elimination to inflate a pass (the TRIP/is_glue issue is exactly this).
- DB-driven, never hardcoded.
- One engine at a time, A/B-gated (gains + zero regressions), DB backup before any write.
- Verify on the LIVE page (:5099) before claiming anything.
- Communicate in plain English — the user flagged over-condensed jargon in this session.
