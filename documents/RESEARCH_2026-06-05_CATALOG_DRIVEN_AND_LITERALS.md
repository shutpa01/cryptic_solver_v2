# Research — catalog-driven engines + literals: how it works, how we transition

Date: 2026-06-05. Branch: redesign. Status: research only — no code or DB changed.

Purpose: answer two questions the user posed.
1. How would a catalog/signature-driven engine actually work (so labels and literals
   come from the signature, not guessed from which DB table a value fell out of)?
2. How would we transition the current evidence-driven engines to it without
   regressing the solves that already work?

Everything below is grounded in the actual code/data and cited. Where a claim is an
estimate or unknown, it says so.

---

## 1. Why this is the right fix (the two examples, root-caused)

### 1a. The mislabel (GEMINI "House of stone in Italy" = GEM+I+N+I... = GEM·IN·I)
The evidence engine tiles by "any DB value that fits" and labels each piece by which
table the value came from. `Italy → I` is filed in BOTH synonyms_pairs and the
abbreviation table; the lookup returns synonyms first (engine_registry.py:151 before
:157) and the engine takes the first match, so it shows "synonym". It is an
abbreviation (I = country code for Italy). Same for `stone → ST`.

Under a signature, the slot carries its type. `SYN_F+ABR_F charade` says slot 2 is
ABR_F before any lookup, so `I` is looked up as an abbreviation and labelled one. The
mislabel cannot occur. My earlier "option 1" patch (relabel at lookup time) was
treating a symptom the design removes at the root.

### 1b. Literals (e.g. ASHRAM "A horny male maintains silence in retreat",
A=literal + RAM(synonym) + SH(abbreviation))
The evidence charade engine refuses a word's own letters by design
(charade_engine.py:10-11, "NEVER the word's own raw letters — that wildcard is the
free-tiling this engine exists to replace"). So a literal can only leak in by accident
(when its letters happen to be a table entry, e.g. in→IN via the abbreviation table)
and is then mislabelled. The engine has no honest way to *produce* a literal.

The catalog dissolves the free-tiling fear: a literal slot type (LIT_F) in the
signature pins exactly which slot is the literal and where it sits, so the engine takes
that word's own letters THERE and nowhere else. The catalog is what makes literals
both producible and safe.

Root cause of both: the engines drifted off the catalog design (§4) to an
evidence-driven tiler. Confirmed: charade, anagram, anagram_charade all accept a
`templates` argument and ignore it ("accepted for call-site compatibility but unused");
anagram_container is not even passed templates (engine_registry.py:316). The drift is
total.

---

## 2. How a catalog-driven engine works (grounded in code that already existed)

The catalog-driven charade engine is not hypothetical — it existed and was committed,
then rewritten away. It is preserved at commit fb0119d4 (`git show
fb0119d4:core/charade_engine.py`, 298 lines). The data model and loader it used are
still live.

### 2a. The data model (live today, data/clues_master.db)
- `catalog_templates(id, operation, signature, def_pos, count, priority, origin,
  active, version, ...)` — one row per mined signature. 694 rows, all origin='mined'.
  charade has 226 (115 def:start, 111 def:end).
- `catalog_template_slots(template_id, position, role, n_words)` — the typed slots in
  clue order. A slot's `n_words` is how many consecutive clue words form that one piece
  (so SYN_F(2w) is a two-word synonym).
- Loader: `core/catalog_loader.py` reads both tables once into immutable `Template`
  (with `.slots` tuple of `Slot(position, role, n_words)`) and is already injected into
  the wiring (engine_registry.py:238-245). The engines receive the templates today;
  they just ignore them.

Distinct slot roles present (counts): SYN_F 879, ABR_F 790, CON_I 138, ANA_F 117,
ANA_I 98, REV_I 37, HOM_F 1, HID_F 1, HID_I 1. **No LIT_F.** (This is the literal gap —
section 4.)

### 2b. The matching algorithm (the historical engine, fb0119d4)
For each template in priority order, for each DB-confirmed definition split:
1. `split.where` must equal `template.def_pos` (the signature says which edge the
   definition sits on; `DefinitionSplit.where` ∈ {start,end} already carries this —
   definition_engine.py:24). The definition is decided upstream, never by the engine.
2. The wordplay words map onto the typed slots by `n_words`.
3. Each slot is filled STRICTLY by role — `ROLE_MECHANISM = {SYN_F: synonym, ABR_F:
   abbreviation}` — a role-pure, answer-aware lookup (a SYN_F slot accepts only
   synonyms; the word's raw letters are NOT a candidate).
4. `_reconstruct`: choose one value per slot so they concatenate left-to-right to
   EXACTLY the answer (no reordering in a charade).
5. `_verify_charade`: three-state pass/pending/fail, engine-specific (no shared grand
   verifier — the 13-isolated-engines decision is preserved).
6. The mined signature that produced the solve is recorded on the parse
   (`template_id`, `matched_signature`) — design §10's clue↔signature cross-reference,
   which the evidence engine cannot provide (it has no signature).

The label is `ROLE_MECHANISM[slot.role]` (charade_engine.py @fb0119d4 _build) — it comes
from the signature, not from a table guess. That is the whole point.

### 2c. The single defect that got it abandoned
In `_try_template` (fb0119d4):
```
fodder = [t for t in split.wordplay_tokens if not is_link(t.text)]   # PRE-STRIP
links  = [t for t in split.wordplay_tokens if is_link(t.text)]
groups = _slot_groups(template, fodder)                              # contiguous map
```
It pre-strips link words by `is_link` BEFORE mapping the rest onto slots. That violates
the binding rule "never pre-assign link words; links are residue classified LAST"
(memory: feedback-never-preassign-links). The 2026-06-04 session reacted by throwing
out the catalog entirely (commit 7d5518a3) instead of fixing this one step.

This is the crux: **the catalog approach was never in conflict with links-last. The
implementation shortcut was.** The fix (section 3) keeps the catalog and honours the
rule.

---

## 3. The reconciliation — catalog-driven AND links-as-residue

The evidence engine and the catalog engine are not opposites. The evidence engine does
a FREE placement search (DFS over "value piece or skip"); the catalog engine should do
the SAME placement search CONSTRAINED by the signature. Concretely:

Replace the pre-strip with a placement search that allows gaps:
- Place the template's K slots onto K disjoint word-runs in clue order, gaps allowed
  between and around them. Each slot consumes exactly its `n_words` consecutive words.
- Fill each slot by its role (role-pure lookup); reconstruct to the exact answer.
- The leftover (gap) words are classified LAST, exactly as the evidence engines already
  do it (charade_engine.py:38-49 residue logic): is_link or POS-function (ADP/PART/
  AUX/DET/CCONJ/SCONJ/VERB/ADV) → link; an op that needs an indicator takes it from the
  residue; anything else → unaccounted → that branch fails. No role is asserted on a
  fail (feedback-no-role-assignment-on-fail).

So the signature supplies: how many pieces, each piece's role (= its label and its
allowed lookup), their order, and the definition edge. The placement search supplies:
which words form each piece, with gap words left for last-classification. Links are
never pre-assigned; they fall out as residue. Both rules satisfied at once.

This also reframes the precision/measurement finding from the 2026-06-04 sample: the
evidence engine's precision-when-passing was already ~97-100%, so catalog-driven is not
mainly about catching false parses. Its wins are (a) correct labels by construction,
(b) literals become producible and safe, (c) per-template coverage is measurable
(design §10), (d) it is the specified design.

---

## 4. Literals — the missing slot type

### 4a. What exists
- The mechanism is first-class in the data model. The seeding design maps `literal →
  literal(source_word, value=letters)` (CATALOG_SEEDING_DESIGN.md:180) and the
  components JSON mechanism vocabulary includes `literal` (ibid:135). Real charades use
  it (ASHRAM: A=literal). The renderer already maps mechanism `raw` → "Literal"
  (wfw_render.py:58) and the design lists `raw` as a Piece mechanism (§3.3).
- So both ends are ready: the source data labels literals, the renderer displays them.

### 4b. What's missing
- The mined catalog has NO LIT_F slot role (section 2a). The extraction that built
  catalog_templates emitted only SYN_F/ABR_F/ANA_F for charades — literal pieces were
  dropped or collapsed. So no charade signature currently says "this slot is a literal".
- A `ROLE_MECHANISM["LIT_F"]` entry filling from the word's own letters
  (raw(word) == the answer span at that position) — the engine side.

### 4c. How a LIT_F slot fills (engine side)
`ROLE_MECHANISM["LIT_F"] = "raw"`; the candidate for a LIT_F slot of word `w` is the
single value `raw(w)` (uppercase letters of the word), accepted only if it equals the
answer span at that slot's position. Because only a LIT_F slot uses raw letters, and a
LIT_F slot exists only where a signature places it, there is no free raw-tiling. Label
renders as "Literal".

### 4d. How LIT_F signatures get into the catalog (two routes — a user decision)
The mined catalog has none, so we must add them. Options:
- Route A — hand-add the common literal shapes as origin='hand_added' rows (e.g.
  LIT_F+SYN_F, SYN_F+LIT_F, LIT_F+ABR_F, ABR_F+LIT_F, and 3-piece variants), def:start
  and def:end. Fast, targeted, immediately testable, fully under our control. Risk:
  we guess the shapes rather than measuring them.
- Route B — re-mine: the seeding/translation pass (CATALOG_SEEDING_DESIGN.md) already
  emits literal leaves; extending `extract_catalog` to emit LIT_F slots would produce
  literal signatures with real frequencies/priorities. Principled and complete, but
  heavier: it means reviving the prototype seeder, and the actual `extract_catalog.py`
  miner is not in the tracked tree (only build_catalog_report.py is) — locating/
  reviving it is real cost and risk.
- Recommendation: Route A first (a handful of hand-added literal signatures, measured
  on real clues), Route B later for completeness. Note: prevalence of literals in
  charades is asserted-common by the user; a clean automated count is not available
  (the structured_explanations ai_pieces shape is inconsistent — only 28 charade rows
  parsed cleanly with a quick extractor), so an early small measurement is worth doing
  to size Route A's shape list.

---

## 5. Scope — what actually has to change

All four catalog engines bypass their signatures today:
- charade — receives templates, ignores them (evidence DFS).
- anagram — receives templates, ignores them (anagram_engine.py:113).
- anagram_charade — receives templates, ignores them.
- anagram_container — not passed templates at all (engine_registry.py:316).

The hidden and double-definition engines are quick checks and stay as-is (design §5.2).

So "catalog-driven" is a conversion of four engines, charade being the reference. Each
is isolated (13-engines decision), so they convert one at a time with no shared-code
blast radius.

---

## 6. Transition plan (measured, reversible, one engine at a time)

Front-to-back, each step proved on real clues with output shown and regression-checked
before the next (design §7 discipline; CLAUDE.md quality rule).

Step 0 — Literal slot support, catalog side (Route A).
  Hand-add a small set of literal charade signatures (origin='hand_added', active=1).
  Backed up, reversible, soft-disable via `active`. No engine change yet — proves the
  data path and lets the catalog engine (step 1) consume them.

Step 1 — Convert the charade engine to signature-driven WITH links-as-residue.
  Revive the fb0119d4 engine, but replace the pre-strip (§2c) with the gap-allowing
  placement search (§3). Add `ROLE_MECHANISM = {SYN_F: synonym, ABR_F: abbreviation,
  LIT_F: raw}`. Keep its own verifier (isolated).

Step 2 — A/B measurement before cutover (the coverage gate).
  Run BOTH the evidence engine and the new signature engine over the same real sample
  (reuse the 2026-06-04 harness). For every clue the evidence engine currently PASSES:
  - does the signature engine also pass, with correct labels? (expected win: labels)
  - any clue the evidence engine solves that NO signature covers is a catalog-creation
    gap (§8) — record it; fill via a hand-added or re-mined signature before cutover.
  This is the per-mechanism coverage measurement the design demands before retiring an
  engine (§7, §9). Cutover only when the signature engine ≥ evidence engine on coverage
  AND strictly better on labels.

Step 3 — Cut the charade slot of the cascade over to the signature engine; keep the
  evidence engine available (soft, behind a flag) until step 2 holds on a broad sample.

Step 4 — Repeat steps 1-3 for anagram, anagram_charade, anagram_container (each has its
  own roles: ANA_F fodder, indicator slots ANA_I/CON_I, etc.; the role-pure fill and
  residue-last logic generalise).

Step 5 — Per-puzzle render switch (design §13). The live cutover is per puzzle via the
  puzzle_render flag, set only after a puzzle's new-system run is reviewed and covers
  all its clues. Unchanged by this research; it sits on top.

Throughout: the catalog-creation process (§8) handles gaps — classify the gap (missing
DB entry / missing signature / mis-classified indicator / CD), fix at the right layer,
prove through the real path, regression-check, record. A missing signature is added to
catalog_templates (hand_added), not coded around.

---

## 7. Risks and open decisions (for the user)

1. Literal signatures — Route A (hand-add, fast, guess shapes) vs Route B (re-mine,
   complete, heavy, miner not in tree). Recommendation: A then B. DECISION NEEDED.
2. Coverage regression risk at cutover: the mined catalog is the weakest of the three
   historical catalogs (memory: feedback-read-design-fully); some evidence-engine
   solves may have no signature. Mitigated by the step-2 gate (never cut over below
   coverage) but it means catalog-creation work before each engine's cutover. This is
   expected, not a surprise.
3. Multi-word slots and phrase pieces: the evidence engine allows a piece to span up to
   4 words via free phrase lookup; the signature pins `n_words` per slot. Mined
   signatures already encode n_words, but hand-added ones must get it right.
4. The placement search adds back a search the evidence engine has; it must stay
   constrained (slot count, roles, order, n_words, def_pos) so it does not become a
   free tiler with extra steps. The constraint is the signature; keep it strict.
5. Whether to also record template_id for the currently-passing evidence solves during
   transition (so coverage is measurable immediately) — minor, deferred.

---

## 8. One-paragraph answer

It works by making each engine walk its mined signatures in priority order and
instantiate them: the signature says how many pieces, each piece's role (which is both
its label and the only lookup allowed to fill it), their order, and the definition edge;
the engine searches which words form each piece, leaving gap words to be classified as
links LAST. That fixes the labels by construction and, with a new LIT_F role, makes
literals producible and safe (the slot pins where a literal may appear, so no
free-tiling). The catalog-driven engine already existed (fb0119d4); its only sin was
pre-stripping link words, which the gap-allowing placement search removes. We transition
one isolated engine at a time, charade first, gated by an A/B coverage measurement
against the current evidence engine, adding hand-added signatures (incl. literals) for
any gap, and flipping the live view per puzzle via the existing render switch.
