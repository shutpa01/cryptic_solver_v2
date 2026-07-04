# Handover — 2026-06-20 — Atom-map signature system (start of build)

Branch `redesign` (NOT pushed). All work below UNCOMMITTED. The user controls commits.

Read these memory files first — they hold the agreed design and the facts:
- `memory/atom_map_signature_architecture.md` (the agreed direction)
- `memory/reference_db_map.md` (which table backs what — do NOT re-derive it)
- `memory/MEMORY.md` (index; the two above are at the top)

---

## 0. READ THIS FIRST — how to behave (hard-won, the user is angry for good reason)

This session repeatedly failed the user. Do not repeat it:

- **Do exactly what was agreed. One small step at a time.** Do not invent side-machinery.
  When unsure of the step, ask for the step — then do that step only.
- **No waffle.** Short replies. The user reads reams of text as bluffing/cover-up and is
  usually right. State facts, show proof, stop.
- **No bluffing.** Never say something is done unless it is, verified. If you haven't run
  it, say so. If a number is untrustworthy, say why.
- **Chase QUALITY, not a target.** The user explicitly said stop chasing the 80%. The bar
  exists (80% cold pass on a new puzzle, excluding DB gaps; the user adjudicates DB-gap
  exclusions) but it is NOT to be gamed or chased with cheap per-clue hacks.
- **Verify before claiming**, against the actual code/DB. You do NOT carry the project's
  history; reconstruct context from files + memory, and don't narrate known facts as
  discoveries.

## 1. The design (what we are building) — an EXPLAINER

We always have the clue AND the answer. The job is to produce the correct wordplay
explanation. The model:

- **Atomise the answer first.** It is N letter-slots; solving = sourcing every slot.
- A **signature** is an atom-map: each answer atom is sourced by one clue piece; the
  OPERATION on a piece is the SHAPE of the map (letters in order = identity/charade;
  reversed = reversal; scrambled = anagram; a source letter dropped = deletion; a piece's
  answer atoms split around another = container). Derived from the ANSWER, not from prose.
- **Indicators VALIDATE, they do not INITIATE.** The structure is read off the atoms; the
  indicator only confirms a DB-backed licence for the operation the atoms imply.
- **Completeness invariant (the precision guard):** every answer atom sourced + every
  CONTENT clue atom roled + every implied operation DB-licensed. (Inert punctuation, e.g.
  a comma, may have no role and is just not displayed.)
- It **REPLACES** the 30-engine cascade (not alongside) and renders through the existing
  WFW page. The cascade is the inferior thing being removed, NOT a parity target.

The user rejected the earlier operation/assembly GENERIC verifier as a false-positive
disaster. The whole point of a detailed atom-map signature is that the verifier is
TRIVIAL (place the signature, check the indicators are DB-licensed) — NOT a search.

## 2. What was built this session — and which parts are SOUND vs FLAWED

New package `core/atomsig/` (self-contained, separate from the cascade):

SOUND — keep:
- `signature.py` — `signature_from_parse(parse, ctx)` reads a SOLVED `Parse` (already a
  one-Link-per-answer-letter atom-map) into an `AtomSignature` (assembly + per-piece
  role/transform/placement + indicators + def edge), length/word-stripped. Enforces the
  completeness invariant (rejects incomplete or unaccounted-word parses). Merges
  multi-word anagram fodder into one piece.
- `harvest.py` — runs the existing cascade DB-only over a corpus of clue+answer pairs and
  converts every PASS via `signature_from_parse`. This is the SOUND seed source: it reads
  real atom-maps (`Parse.links`), no inference. Writes `logs/atomsig/{harvest_report.txt,
  signatures.jsonl, instances.jsonl}`. **It does NOT persist to a catalogue — that is the
  gap (the next step).**
  Run: `.venv/Scripts/python.exe -m core.atomsig.harvest --limit N --seed 7`

FLAWED — set aside (do NOT build on these without rethinking):
- `from_structured.py` + `verifier.py` + `backing.py` — an attempt to convert the
  `structured_explanations` table (183k old-solver decompositions) into atom-maps.
  `structured_explanations` only stores `fodder -> yields` (piece lists), NOT the
  atom-level map. So `verifier.py` INFERS the geometry by SEARCHING piece orderings / DB
  values / transforms. **That is exactly the complex search verifier the user rejected** —
  it can manufacture plausible-but-wrong tilings, and it under-reads sources (synonyms +
  wordplay-abbreviations only; NOT definitions or depth-2 synonyms), so its "36% validated
  on the >=0.85 sample" number is untrustworthy both ways. `backing.py`'s per-mechanism
  idea may be reusable conceptually for the trivial verifier, but the all-orderings search
  in `verifier.py` must NOT be carried forward.

## 3. Evidence (real numbers, not bluffed)

Cascade harvest, 1500 clues, seed 7 (in `logs/atomsig/`):
- base pass rate 773/1500 = 51.5% (WARM/in-sample — the cascade was tuned on this pool, so
  this is an upper bound, NOT a cold number; DB gaps included)
- clean-conversion 681/773 = 88.1% of passes (= the atom-maps we can read out cleanly)
- 145 distinct signatures; the 92 conversion failures are all one cause: "incomplete"
  (a pass whose links don't cover every answer letter = coarse provenance in some engine)
- top shapes: `single | anagram_fodder:anagram:contiguous` (256), charade synonym+synonym,
  container synonym(split)+synonym, hidden, etc. Composites already appear as COMBINATIONS
  of the same vocabulary (anagram+charade, deletion+charade) — the thesis holds.

## 4. Key facts the next thread MUST know (so it doesn't blunder)

- **Atom-maps are NOT stored at scale.** `structured_explanations` (183k) has only
  fodder->yields, no geometry. Only `wfw_solve` (~1k rows) stores the atom-level Parse.
  So to get atom-maps for the whole corpus you must either RE-RUN the cascade (regenerate
  `Parse.links` — but that only covers what the cascade solves now, ~51%, and the user
  rightly sees re-solving already-solved clues as wasteful), or get them another way.
  THIS TENSION IS UNRESOLVED — do not silently re-run; discuss it.
- Reference data lives in `data/cryptic_new.db`: synonyms=`synonyms_pairs`,
  **abbreviations=`wordplay` table (there is NO abbreviations table)**,
  definitions=`definition_answers_augmented`, homophones=`homophones`. The standard wiring
  lookup does NOT read depth-2 synonyms or back wordplay via definitions — a synonym-only
  checker UNDER-counts backing. (Full detail: `memory/reference_db_map.md`.)
- Python: `.venv\Scripts\python.exe`. Clue corpus: `data/clues_master.db` table `clues`
  (eligible = `has_solution=1 AND answer<>''`, ~184k). The puzzle-page "HIGH" tier =
  `structured_explanations.confidence >= 0.7` (code: `web/models.py:compute_hint_tier`);
  the user asked to use **>= 0.85** when selecting from the structured source.

## 5. THE NEXT STEP (agreed: take what we have, build the system, small steps)

Goal: "build the complete system to see if it can work," in small verifiable steps.

**STEP 1 (do this first):** Persist the signatures we ALREADY harvested into a real
catalogue, so the log-file output becomes the foundation the system reads.
- Write to a NEW file `data/atomsig.db` (do NOT touch `clues_master.db` schema).
- Two tables: `signature(key TEXT PRIMARY KEY, count INT, example_clue, example_answer)`
  and `instance(clue_id INT, signature TEXT, assembly, def_pos, operation, info_json)`.
- Source: either read `logs/atomsig/{signatures,instances}.jsonl`, or add a `--persist`
  path to `harvest.py` that commits incrementally (so a stop never loses progress).
- Proof of done: print the row counts and the top 20 signatures from `data/atomsig.db`.
- Keep it small. Do ONLY this. Then stop and report.

**STEP 2 (after Step 1 is confirmed):** Apply a signature to a clue — the TRIVIAL verifier
the design calls for (place the signature's typed slots on the clue words, check every
implied operation has a DB-licensed indicator, build the atom-map Parse). Test on ONE clue
end-to-end (signature -> explanation -> WFW render) before scaling. Do NOT build an
all-orderings search; the signature pins the structure.

**Before scaling anything:** resolve the §4 tension (how to get atom-maps at scale) WITH
THE USER. The honest options are: (a) re-run the cascade to regenerate them (covers ~51%,
user sees as wasteful), (b) the manual click-to-create path (atom-level by construction —
the trustworthy source the design ultimately intends), or (c) salvage `wfw_solve`'s ~1k.

## 6. One-line status

Sound atom-map harvester exists and produces real signatures (681 from a 1500 sample, in
log files); they are NOT yet persisted into a catalogue. That persistence is Step 1. The
structured_explanations converter was a wrong turn (needs the rejected search verifier) —
set it aside.
