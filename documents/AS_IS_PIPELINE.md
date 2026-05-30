# AS-IS Pipeline Documentation

This document records, as-it-currently-is, what happens when a puzzle is run
through the solver from the dashboard. We build it up bit by bit, one file at a
time: for each file we record the filename and a summary of what it does.

## Starting point

Dashboard → Pipeline tab → select puzzle(s) → click "Run Selected Puzzles".

---

## General design considerations

These are cross-cutting points that apply across the whole pipeline rather than to
one file. Individual sections below refer back here instead of repeating them.

### Atomise the clue and answer once

Today every solver re-derives its own view of the clue and answer: the same work
(stripping the enumeration, reducing to letters, splitting into words) is repeated
inside function after function. This is wasteful and a source of inconsistency,
because each function can make slightly different choices.

Redesign principle — atomise the clue and the answer once, at the very beginning
of a run, and reuse that single atomised representation in every downstream
function. A good version of this atomisation already exists in the committed WFW
(word-by-word) project and could be revived rather than rebuilt.

### Preserve all letter-contributing evidence, even on failure

Every stage that contributes answer letters must keep its evidence, even when the
operation ultimately fails. Today the solvers do the opposite: when a stage fails
it returns nothing and discards the letter-contributing pieces it had already
gathered, so nothing downstream (review, leftover processing) can reuse that work.

Redesign principle — retain the evidence each stage produced about how clue words
map to answer letters, regardless of whether that stage succeeded. This connects
to the WFW evidence-trail / candidate-evidence-preservation work.

FINDING (2026-05-30, DT 100009 "Paused, as that man has and I would, to take in
gallery" = HESITATED): this gap has a direct, costly consequence for enrichment.
The pipeline only harvests enrichment gaps from parses it actually built — the
gap-collector inspects the pieces of a produced explanation and flags those not in
the DB. A clue that fails to assemble produces no explanation, so no pieces, so
nothing is flagged and nothing is queued. In this clue the engine had HE ("man"),
TATE ("gallery") and ID ("I would") and was a single piece short — the missing
reference entry is precisely "has" giving S (the apostrophe-s that contracts "he
has" to HES). We can name the missing entry exactly, yet it is never queued, because
the all-or-nothing failure discards the near-miss that reveals it. The irony: the
clues that would teach the DB the most — the ones failing by one missing piece — are
exactly the ones that contribute nothing to enrichment, while clues that already
solved do. A redesign that retained the near-miss evidence could queue "has -> S"
automatically from this failure.

### Harmonise PARTS subtype classes in the indicator table

The indicator table classifies PARTS-type indicators with subtypes (first_delete,
first_use, last_delete, last_use, tail_delete, outer_delete, outer_use, outer,
center_delete, center_use, inner_use, and so on), but these have not been applied
consistently. The symptom is visible in the solver code: several solvers
(deletion, container, charade) each carry their own hand-written map translating
those PARTS subtypes into the subtype they actually use (head/tail/ends/middle).
That local re-mapping exists only because the underlying table is not consistent.

Redesign principle — harmonise the PARTS subtype classes once, in the indicator
table itself, so every solver can read a single consistent subtype and the
per-solver translation maps disappear.

### An operation may only apply to material near its controlling indicator

An operation (anagram, reversal, deletion, odd/even extraction, containment, and
so on) must not be applied to a word or span unless that material has clear
proximity to the indicator that controls it. Today several solvers gate only on an
indicator being present somewhere in the clue, then apply the operation to any word
regardless of distance — the charade solver is the clearest case, where a single
reversal/deletion/odd-even indicator anywhere licenses that operation on every
word.

This applies to all operators, with no exceptions. Container is the most exposed,
because its indicators are numerous common words (in, around, between, holding,
about, and dozens more); without a proximity rule a stray common word anywhere in
the clue can trigger a spurious container reading.

Redesign principle — bind each operation to a nearby controlling indicator, and
only allow the operation on material in clear proximity to it.

### Single-type engines versus the compound reality of clues

The pipeline is built as a set of single-type engines — one for hidden words, one
for double definitions, one for anagrams, one for containers, and so on — each
looking for exactly one mechanism. For the engines that genuinely detect one
mechanism this is accurate. But real cryptic clues are frequently compound (a
charade of a reversal and a deletion, a container whose inner is itself an anagram,
and so on), and a set of single-type engines cannot represent that cleanly. The
charade solver shows the strain: it quietly absorbs reversal, deletion, odd/even
and first-letter operations but still reports a single "charade" type.

Redesign principle — model wordplay as deeper, compound clue types built from
composable operations, rather than a fixed menu of single-type engines each
emitting one broad label.

### The cascade order is backwards — best engine runs last

The most sophisticated engine (the signature solver — grammar triage, catalogs,
confidence scoring) runs LAST in the cascade, after the cruder earlier mechanical
phases (Phase 0–0.5). Because the cascade is first-to-claim-wins, the cruder
engines take a clue first and the better engine only ever sees the leftovers. A
worse parse can win over a better one purely because it ran earlier.

This also means the raw "who solved what" split (about two-thirds early-mechanical,
under a third signature) is largely an artefact of ordering, not of quality — the
early engines get first dibs, not better answers.

Redesign principle — either run the best engine first, or run engines in
competition and keep the highest-confidence parse, rather than first-to-claim-wins.

### Triage should route to the catalog, not solve in parallel

The grammar-triage patterns and the catalog entries are the same kind of object — a
clue's grammatical shape mapped to a role layout (a template). Today they are wired
as two separate solvers covering the same ground: triage runs its own structural
tests inside the signature engine, while the catalog matcher independently grinds
through its templates. The expensively-mined grammar patterns (166 grammatical
shapes, discovered over several days) are then placed late in the cascade, behind the
cruder phases, and under-wired where they do run — so the hard intellectual work
mostly never pays off. The name "triage" promises a sort-to-the-right-answer step but
the placement delivers a parallel mini-solver that runs last and may never be reached.

Redesign principle — use triage as the front-end router. Take the clue's grammatical
shape, have triage rank which catalog templates are worth trying (most likely first),
then let one matcher try just those. This makes triage what its name implies, removes
the triage/catalog duplication (one classifier feeding one matcher, not two solvers
on the same ground), and puts the learned patterns where they earn their keep — at the
front, narrowing the search — instead of as a late fallback. It also depends on first
measuring what each catalog and template actually covers, since nothing today records
which catalog or template solved a clue.

---

## Big picture: cascade and control flow

Before the file-by-file detail, the overall shape of a run.

It is a cascade. A clue passes through phases in order: Phase 0 (hidden) → Phase 0b
(spoonerism) → Phase 0c (double definition) → Phase 0.5 (the seven V1 mechanical
solvers) → Phase 1 (signature solver) → Phase 1.5 / 1.5b (blog + Haiku) → Phase 2
(Tier 2 / API) → Phase 3 (enrichment re-solve). Each phase skips any clue already
solved by an earlier phase, so a clue drops out of the cascade the moment one phase
claims it.

Within Phase 0.5 the seven solvers are tried in a fixed order (anagram, container,
deletion, charade, reversal, acrostic, homophone) and the first one to return a
result wins. There is no scoring or competition between solvers; the order itself
is the tie-breaker (for example container is tried before charade to limit false
positives).

The solvers are all-or-nothing. Each one either returns a complete parse that
produces the whole answer, or it returns nothing. There is no notion of a partial
parse being handed to the next solver to finish. A clue is either fully claimed by
some phase, or it falls through to Phase 2, and if that also fails it ends as
FAILED (a leftover).

Partial evidence is not retained. Because each solver is all-or-nothing, when one
cannot complete the answer it returns nothing and discards everything it gathered —
the candidate values and the pieces it had already matched against parts of the
answer. Nothing carries forward. A clue that is mostly decodable by one mechanism
but missing a single piece produces no result and no retained evidence; it looks
identical to a clue nothing understood.

These last two points are where the general design considerations bite: the
all-or-nothing, no-retention behaviour is exactly what "Preserve all
letter-contributing evidence, even on failure" addresses, and it compounds with
"Single-type engines versus the compound reality of clues" — a compound clue that
no single engine can finish yields nothing, even when several engines each found
genuine pieces.

---

## Files in the flow

### dashboard/pages/pipeline.py
Streamlit "Pipeline Runner" page. Lets you select puzzles and click Run Selected
Puzzles. Doesn't solve anything itself — for each puzzle it launches the solver
as a subprocess: a blog pipeline if a Times/Guardian/Independent blog exists,
otherwise the default sonnet_pipeline.run.

### sonnet_pipeline/run.py
The orchestrator — the script the dashboard runs for the default path. Given a
source and puzzle number, it solves every clue in that puzzle and writes the
results to the database. The dashboard always runs it in mode 1 (full pipeline).

#### Shared resources (loaded once per run)

Factual summary:
Before any clue is solved, four resources are constructed once and reused across
all clues. At this stage nothing is solved — the resources only read tables into
memory so later phases don't repeat the work.

- ClueEnricher — opens cryptic_new.db and loads five tables: indicators,
  wordplay, homophones, synonyms_pairs, definition_answers_augmented.
- HomophoneEngine — opens cryptic_new.db and loads one table: homophones.
- example_messages — builds a fixed list of worked examples from constants in
  solver.py. No database.
- RefDB — opens cryptic_new.db and loads five tables (indicators, wordplay,
  synonyms_pairs, definition_answers_augmented, homophones), builds a wordlist
  from them, then opens clues_master.db and adds words from the clues table to
  that wordlist.

DESIGN CONSIDERATION:
Problem — the loading is duplicative. ClueEnricher and RefDB load the same five
tables from cryptic_new.db, and HomophoneEngine loads homophones a third time.
The result is two parallel in-memory copies of essentially the same reference
data, held for the whole run. This exists because the resources serve two
separately-developed code paths: ClueEnricher and HomophoneEngine were built for
the paid AI path, RefDB later for the signature/mechanical path, and each path
got its own loader instead of sharing one.
Potential solution — collapse the reference data into a single shared in-memory
store loaded once, and have both the AI path and the mechanical path read from
it. This removes the redundant reads, halves the memory, and gives one source of
truth for synonyms/abbreviations/indicators/definitions/homophones.

#### Clue loading and clue selection

Factual summary:
run_puzzle opens clues_master.db and reads all clues for the given source and
puzzle, ordered by clue number (id, clue number, direction, clue text, answer,
enumeration, existing explanation). It first builds a cross-reference map (clue
number to answer) from the full puzzle, so clues that refer to other clues can be
resolved later. If a single clue was requested it filters to matching rows.

It then decides which clues to work on. When not in force mode, it checks each
clue against the structured_explanations table and removes any clue that already
has a stored explanation scoring 0.7 or higher — these are treated as already
done, and everything else is re-attempted.

Three options control a run, set both in the dashboard and as defaults in
run.py:
- Write to DB — dashboard default on; run.py default also on (WRITE_DB = True).
- Force — dashboard default off; run.py default on (FORCE_API = True).
- Re-run partials — dashboard default off; run.py default off (PARTIALS = False).

The dashboard only adds a flag to the command when its checkbox is ticked. But
each flag in run.py is defined as a store-true argument whose default is the
module-level config value. A store-true flag with a default of True is True
whether or not the flag is passed. So when run.py's default is True, the
dashboard's "off" cannot turn it off.

Net effect today:
- Write to DB is on regardless (both defaults on; harmless).
- Force is on regardless — run.py's True default overrides the dashboard's off,
  so the skip-HIGH-confidence block (guarded by "if not force") never runs and
  every clue is re-attempted every time.
- Re-run partials is genuinely off unless ticked (both defaults off).

DESIGN CONSIDERATION:
Problem — the dashboard selection boxes and the run.py module-level defaults
diverge. The defaults at the top of run.py were convenient for running the file
directly during development, but they also feed the command-line argument
defaults, so the dashboard's intent (Force off) does not survive into the actual
run. This is a product of incremental building: the standalone script and the
dashboard launcher were developed at different times and each carries its own
notion of the defaults.
Potential solution — harmonise the two. Decide a single source of truth for run
options and make the other side defer to it: for example, have the command-line
flags default to off (store_true with default False) so the dashboard checkboxes
fully control behaviour, and keep the development-time convenience values in a
clearly separated block that does not leak into the CLI defaults. The broader
point for the redesign is to remove the redundant, separately-maintained notions
of the same setting.

#### Phase 0 — mechanical hidden-word check

Factual summary:
Phase 0 is the mechanical hidden-word check — zero API cost, the first solving
step. A lookup graph is built once from RefDB. For each clue whose cleaned answer
is 3 or more letters, run.py calls the hidden-word solver (try_hidden in
backfill_ai_exp/backfill_dd_hidden.py), passing the answer's letter count as the
target length.

Acceptance is purely mechanical and does not require a definition. The solver
strips the enumeration, reduces the clue to a letters-only stream, and looks for
the answer's letters appearing contiguously, forwards or reversed, as a proper
substring of the clue. A match is valid if it sits inside a single word without
being the whole word (for example OTIC in "noticed"), or spans several words
without consuming every word in the clue. The first matching span is taken; the
direction is recorded as forward or reverse, and the covered clue words are
recorded.

Separately, and only as a best-effort extra (because the graph was passed in), it
tries to name the definition: it runs the definition engine (definition_candidates)
over the clue, looks for a candidate whose letters equal the answer, and if found
keeps the longest supporting phrase. This definition may be None, and it has no
bearing on acceptance.

On a hit, run.py marks the clue solved, sets the wordplay type to hidden or
hidden_reversed, and builds an explanation string with the hidden letters
highlighted. If writing to the database it writes in two places: a row in
structured_explanations (model_version mechanical_hidden, confidence 1.0,
definition text possibly None) and an update to the clues table itself
(wordplay_type, definition, ai_explanation, has_solution = 1), using COALESCE so
existing non-empty values are preserved.

DESIGN CONSIDERATION (scoped to Phase 0 only):
Problem 1 — label versus behaviour. run.py prints this phase as
"definition-confirmed", but the solver explicitly does not confirm a definition
and will store a hidden solve with definition = None. The printed label
misrepresents what the code does.
Problem 2 — self-certified confidence. A purely mechanical letter-run match is
stored at a fixed confidence of 1.0 with no verification step, so a coincidental
contiguous run is recorded as fully certain.
Problem 3 — it writes the same result in two places (structured_explanations and
denormalised columns on clues). Noting it here; the reasons may become clear
later.
Potential solution — make the printed label match the actual acceptance rule;
derive the stored confidence instead of hardcoding 1.0 (for instance, lower it
when no definition could be found, or when the run sits inside a single word);
and revisit whether the result needs to live in two places.

#### Phase 0b — mechanical spoonerism check

Factual summary:
Phase 0b is the mechanical spoonerism check — zero API cost. It runs only when
RefDB is loaded. For each not-yet-solved clue whose cleaned answer is 4 or more
letters, run.py first applies a gate: the word "spooner" must appear in the clue.
This gate is correct and follows established crossword convention — a spoonerism
clue always signals itself with Spooner. Only then does it call try_spoonerism_v2
in sonnet_pipeline/solver.py.

The solver takes the answer as an uppercase run of letters and tries every split
point into two parts (each at least 2 letters). For each split it swaps the
leading letters — both the single first letter, and initial clusters of 2 to 3
letters — and checks whether both swapped results are valid words using RefDB's
real-word test. Every split that yields two valid words is collected as a
candidate.

If a clue and RefDB are available, it then tries to tie each candidate's two
swapped words back to the clue — matching them directly, or as synonyms, against
single clue words or two-word phrases — and keeps the highest-scoring candidate.
This matching does not gate acceptance: the best candidate is returned even when
neither swapped word could be tied to the clue. With no clue or RefDB, it returns
the first candidate.

So acceptance rests on: the word "spooner" in the clue (correct convention), and
the answer having at least one split whose swapped initials are both real words.
The clue-word mapping is best-effort decoration.

On a hit, run.py builds an explanation naming the two swapped words (clue words
shown only where the mapping was found), records the result at fixed confidence
high, score 100, and writes in two places: a row in structured_explanations
(model_version mechanical_spoonerism, confidence 1.0, definition text None) and an
update to the clues table (wordplay_type spoonerism, ai_explanation,
has_solution = 1). No definition is captured or written.

DESIGN CONSIDERATION (scoped to Phase 0b only):
Problem 1 — beyond the correct Spooner gate, acceptance only needs any valid
initial-swap split to exist; the two swapped words are not required to tie back to
the clue's wording. Combined with RefDB's broad real-word set (enriched from
puzzle answers and clue texts), this can accept a split unrelated to the intended
clue, yet still store it at a fixed confidence of 1.0.
Problem 2 — no definition. Phase 0b never captures or stores a definition, so a
spoonerism solve lacks the definition a clue is expected to contain.
Problem 3 — it writes the same result in two places (noting only; reasons may
become clear later).
Potential solution — require the clue-word mapping to succeed (at least one,
ideally both swapped words tied to the clue) before accepting, and let the stored
confidence reflect how complete that mapping is; capture a definition where
possible.

#### Phase 0c — mechanical double-definition check

Factual summary:
Phase 0c is the mechanical double-definition check — zero API cost. It runs only
when RefDB is loaded. For each not-yet-solved clue whose cleaned answer is 2 or
more letters, run.py calls the DD solver (generate_dd_hypotheses in
backfill_ai_exp/backfill_dd_hidden.py), passing the answer and its letter count.

The check is strict. It works off a definition graph built once from RefDB (the
synonym and definition-answer data, indexed in both directions). The solver tries
every point at which the clue can split into a LEFT phrase and a RIGHT phrase. For
each split it generates definition windows from each side and looks them up in the
graph. A split fires only if LEFT independently produces a candidate, RIGHT
independently produces the same candidate, and — because the answer is supplied
here — that shared candidate equals the actual answer (matched by length and
letters). One further gate: the two matching definition windows together must
cover all but at most one of the clue's words. It returns at most one hit,
carrying the left definition window, the right definition window, and the split
point.

So acceptance requires both halves of the clue to map independently to the real
answer through the reference data, with almost the whole clue accounted for — a
strong gate, in contrast to the weak spoonerism gate.

On a hit, run.py records the result at fixed confidence high, score 100, and
writes in two places: a row in structured_explanations (model_version
mechanical_dd, confidence 1.0) and an update to the clues table (wordplay_type
double_definition, has_solution = 1). But note what is stored as the explanation
and definition: the literal constant string "Double definition". The left and
right definition windows the solver computed are discarded; both the
ai_explanation and the definition column are set to that placeholder.

DESIGN CONSIDERATION (scoped to Phase 0c only):
Problem 1 — the decode is discarded. The solver knows which phrase defines the
answer on the left and which on the right, but Phase 0c throws that away and
stores only the constant "Double definition". The user is shown no actual account
of how the clue works, and the definition column holds a placeholder rather than
real text from the clue.
Problem 2 — fixed confidence 1.0 is self-certified (less of a concern here than
elsewhere because the gate is strong, but still not derived).
Problem 3 — it writes the same result in two places (noting only; reasons may
become clear later).
Potential solution — store the computed left and right definition windows in the
explanation, and put one of the real defining phrases in the definition column
instead of the placeholder.

#### Phase 0.5 — V1 mechanical solvers (overview)

Factual summary:
Phase 0.5 runs the V1 mechanical solvers — zero API cost — only when RefDB is
loaded. The individual solvers live in backfill_ai_exp/batch_v1_solver.py and are
documented one at a time below; this section covers the orchestration only.

For each not-yet-solved clue (answer 3+ letters) it skips cross-reference clues
(those mentioning another clue, such as "5 across"), then:

1. Finds a definition. It calls _v1_find_definition to split the clue into a
   definition and the remaining wordplay words. If that returns no definition, it
   falls back to a paid Haiku call (signature_solver/haiku_definition.py) to
   propose one. If there are still no remaining words, it uses the whole clue
   minus the enumeration.
2. Tries solvers in a fixed order, taking the first that succeeds: anagram,
   container, deletion, charade, reversal, acrostic, homophone. The order is
   deliberate — container and deletion are tried before charade because charade is
   more permissive and would otherwise produce false positives.
3. If none succeeded and no definition was found, it makes one more attempt
   "wordplay-first" via _v1_solve_without_definition, which tries to infer the
   definition from the wordplay.

On success it builds an explanation (_v1_build_explanation) and appends an
in-memory result tagged tier Mechanical, confidence high, score 100.

The database write differs from the earlier phases: it gates through the
ExplanationVerifier. The verifier's verdict is mapped to a confidence (HIGH 1.0,
MEDIUM 0.6, LOW 0.3, FAIL 0.0) and that derived confidence is what goes into
structured_explanations (model_version mechanical_v1). It then writes in two
places as before, updating the clues table (wordplay_type, definition,
ai_explanation, has_solution = 1).

DESIGN CONSIDERATION (scoped to Phase 0.5 orchestration only):
Problem — the in-memory result and the database write disagree. The in-memory
result is always tagged confidence high / score 100, but the stored confidence is
whatever the verifier awarded, which can be as low as 0.0 (FAIL). Regardless of
the verdict, the clues table is still set to has_solution = 1 with the explanation
written. So a clue the verifier judged a FAIL is still recorded as solved, and the
report (which reads the in-memory result) shows it as high/100 while the stored
confidence says 0.0.
Potential solution — make the in-memory result and the stored result share one
confidence (the verifier's), and stop marking a clue solved (has_solution = 1)
when the verifier verdict is FAIL.

##### Phase 0.5 — the definition stage (feeds every solver)

Factual summary:
The definition stage runs first inside Phase 0.5 and feeds every solver. It has
two paths.

Path 1 — DB definition finder (find_definition in
backfill_ai_exp/batch_v1_solver.py). It strips the enumeration, normalises
quotes, and splits the clue into words. It tries definition windows of 1 to 4
words taken from the start of the clue and, separately, from the end — never the
middle — always leaving at least one word for wordplay. Each window is tested
with ref_db.is_definition_of. It keeps the longest matching window; on a length
tie the start window wins. It returns the definition phrase and the remaining
(wordplay) words, or nothing.

The test it relies on, is_definition_of (signature_solver/db.py), is purely
reference-DB backed. It checks both directions — is any synonym of the phrase
equal to the answer, or any synonym of the answer equal to the phrase — using
RefDB's synonyms (which merge synonyms_pairs and definition_answers_augmented).
There is no fuzzy matching: the exact pair must exist in the reference data.

Path 2 — Haiku fallback (find_definition in signature_solver/haiku_definition.py),
called only when Path 1 returns nothing. It makes a paid Haiku API call (model
claude-haiku-4-5, temperature 0, 50 tokens) asking which first-or-last word or
phrase defines the answer, with no real words orphaned at the edge. It takes
Haiku's reply as the definition, then validates only its position — locating the
phrase at the start or end of the clue (skipping only pure-punctuation tokens) and
deriving the remaining wordplay words. On any error it silently returns nothing.
The Haiku-proposed definition is not checked against the reference DB — by
construction it cannot be, because Path 1 (which queries that DB) just failed.

Both paths enforce the edge-anchored rule, but they apply different standards of
proof, and the solvers receive the result identically either way.

Whichever path supplies it, the definition is later written to clues_master.db
(the clues.definition column and the structured_explanations row) by the Phase 0.5
write block; that write is not conditional on any check. The Phase 0.5 result dict
carries no ai_output field, so none of the run's enrichment-gap collectors
(collect_gaps_from_results in sig_enrichment.py, and _actionable_quality in
report.py) pick up this definition. As a result a Phase 0.5 definition is never
queued to pending_enrichments for review.

DESIGN CONSIDERATION (scoped to the definition stage only):
Problem 1 — a Haiku-proposed definition that drives a Phase 0.5 solve is written
to clues_master.db but is never queued for enrichment review, unlike Phase 1,
which does queue its (possibly Haiku-derived) definition. Because Phase 1 skips
clues already solved in Phase 0.5, such a definition has no later opportunity to
be queued either.
Problem 2 — two different standards of proof, indistinguishable downstream. Path 1
requires an exact reference-DB entry; Path 2 is accepted on the model's say-so
with only a positional check. Once returned, nothing records which standard
produced the definition.
Problem 3 — paid API call inside a phase labelled "zero API cost". The Haiku
fallback makes a per-clue paid call whenever Path 1 misses.
Potential solution — have Phase 0.5 queue its Haiku definition for enrichment the
same way Phase 1 does (the queue_enrichment function already exists in
haiku_definition.py), carry a flag marking the definition LLM-asserted versus
DB-verified, and correct the "zero API cost" framing for the phase.

##### Phase 0.5 — anagram solver (try_anagram)

Factual summary:
The anagram solver (try_anagram in backfill_ai_exp/batch_v1_solver.py) looks for a
set of clue words whose combined letters are exactly the answer's letters. It
cleans the answer to letters only and requires at least 3 letters, strips the
enumeration, splits the clue into words, and excludes the definition words (handed
in from the definition stage) so they cannot be used as fodder. It builds a
letter-count for each remaining word, and if there are more than 12 such words it
gives up (a guard against combinatorial explosion).

It then tries every combination of those words. For each combination it adds the
letter-counts; if the total length matches the answer and the letters match
exactly, it is treated as an anagram. One guard: if the answer appears verbatim in
the clue, that combination is skipped. On success it returns the fodder words and
the unused words, typed as anagram. (The function is passed ref_db but never uses
it.)

What the current code does NOT do — these are guards lost in a poor transcription
of an earlier solver, not deliberate design choices:
- No indicator validation. There is no check that the clue contains an anagram
  indicator (confused, broken, mixed, and so on).
- No proximity guard. Nothing requires the fodder to sit near an indicator.
- No contiguity guard. The fodder words can be gathered from anywhere in the clue,
  not as a contiguous run.

DESIGN CONSIDERATION (scoped to the anagram solver):
Problem — by cryptic convention an anagram must be licensed by an indicator, the
fodder must be a contiguous span, and that span sits adjacent to its indicator.
This solver enforces none of these: it accepts any scattered set of non-definition
words whose letters happen to total the answer, with no indicator present. That
admits coincidental letter matches, especially on short answers.
This is also a transcription loss, not an intended design: the section derives
from an earlier solver and was poorly transcribed, dropping guards the original
had.
Potential solution — do not patch this code. Require indicator validation, a
contiguity guard on the fodder, and a proximity guard tying the fodder to its
indicator (see the general design consideration "An operation may only apply to
material near its controlling indicator"). A better implementation already exists
(the WFW project) and should be revived rather than this one repaired.

##### Phase 0.5 — container solver (try_container)

Factual summary:
The container solver (try_container) tries to read the answer as one set of
letters inserted inside another — inner letters placed within outer letters. It
requires a cleaned answer of at least 4 letters and at least 3 remaining wordplay
words.

Unlike the anagram solver, it is indicator-gated: it first scans the remaining
words and requires that at least one is a container indicator (type "container" in
RefDB). If none is present, it returns nothing immediately.

It then gathers the possible letter contributions for each word via the shared
helper _get_word_values (abbreviations, synonyms, reversals, and so on), and only
proceeds if there are between 2 and 8 such word-values. It searches exhaustively:
for every way of choosing an inner substring position inside the answer, it splits
the answer into an inner target (the middle slice) and an outer target (the left
and right remainder joined), requiring the outer to be at least 2 letters. For
every partition of the word-values into an inner group and an outer group, it asks
_try_build_string to assemble the inner target from one group and the outer target
from the other. The first partition where both assemble is returned, as pieces
(outer words first, then inner words) plus the inner string, outer string, and
insertion position.

DESIGN CONSIDERATION (scoped to the container solver):
Problem 1 — the container indicator is gated by presence only. A container
indicator anywhere in the remaining words licenses any inner/outer split the
letters allow. This is especially dangerous for container because its indicators
are common words (in, around, between, holding, and many more), so a stray common
word can trigger a spurious container. See the general design consideration
"An operation may only apply to material near its controlling indicator".
Problem 2 — on failure the solver returns nothing and discards the word-values it
gathered (the inner-source and outer-source letter contributions it had already
matched against parts of the answer). That evidence is lost. See the general
design consideration "Preserve all letter-contributing evidence, even on failure".
Potential solution — require the container indicator to sit in clear proximity to
the inner/outer material it acts on; and retain the letter-contributing pieces this
stage found, whether or not the container assembly succeeded.

##### Phase 0.5 — deletion solver (try_deletion)

Factual summary:
The deletion solver (try_deletion) tries to read the answer as a longer word or
phrase with letters removed. It requires a cleaned answer of at least 2 letters and
at least 2 remaining words.

It is indicator-gated: it scans the remaining words for a deletion indicator —
either an indicator typed "deletion" in RefDB, or a "parts" indicator whose subtype
maps to a deletion (first/last/outer/centre). Each indicator carries a subtype:
head, tail, ends, middle, or general. With no deletion indicator it stops.

It then works in two passes. Pass 1 (synonym/literal deletion): for each deletion
indicator it builds source candidates from the other words — phrases longest-first
down to single words, skipping other deletion indicators and skipping link words
for single words. For each source it looks up synonyms longer than the answer
(deletion must shorten), and the raw word itself if longer, and applies the
deletion for that subtype. If the result equals the answer it returns the source,
the deletion type and the pieces. Pass 2 (specific-letter deletion): it looks for
the "synonym of one word minus the single-letter abbreviation of another" shape
(OLIVER minus R = OLIVE) and returns two pieces.

The helper _apply_deletion is exact about subtypes: head removes the first letter,
tail the last, ends both outer letters, middle keeps the outer letters (and for
odd-length words also tries removing the central letter). The general subtype tries
all of these. It only ever removes one or two letters — there is no
arbitrary-length deletion.

So deletion is indicator-gated and respects the indicator's subtype, but it does
not bind the indicator to a nearby source: the source can be any word or phrase
anywhere in the remaining words, at any distance from the deletion indicator.

DESIGN CONSIDERATION (scoped to the deletion solver):
Problem 1 — skipping link words is crude. In Pass 1 the solver drops a single-word
source purely because RefDB labels it a link word. That assumes the word's role
without proof and removes it from consideration. Link words should be treated like
any other word — left as candidate sources — with a role concluded only from
evidence in the solve, not from a label.
Problem 2 — the PARTS subtype handling here is local patchwork. The solver carries
its own hand-written map from PARTS subtypes onto deletion subtypes because the
indicator table is not consistent. See the general design consideration
"Harmonise PARTS subtype classes in the indicator table".
Problem 3 — the deletion indicator is not bound to a nearby source; any word at any
distance can be the source. See the general design consideration "An operation may
only apply to material near its controlling indicator".
Problem 4 — on failure it returns nothing and discards the source candidates and
synonyms it gathered. See the general design consideration "Preserve all
letter-contributing evidence, even on failure".
Potential solution — stop excluding link words up front; rely on a harmonised
indicator table instead of a local subtype map; require the source to sit in clear
proximity to the deletion indicator; and retain the letter-contributing evidence
whether or not the deletion succeeded.

##### Phase 0.5 — charade solver (try_charade)

The charade solver is large (about 230 lines) and has absorbed most of the other
mechanisms, so it is documented in pieces.

###### Piece 1 — indicator detection

Factual summary:
Before collecting anything, charade scans all the remaining words once to decide
which advanced mechanisms are licensed by an indicator anywhere in the clue. It
sets three things: del_subtypes (deletion subtypes — added for any "deletion"
indicator, or any "parts" indicator whose subtype is in a local map), and the flags
has_reversal_ind and has_odd_even_ind (the latter true for a "parts" indicator with
subtype odd, even, alternate, or pattern). These flags are used later to decide
whether each word should also try reversed forms, odd/even extractions, and
deletion variants.

The local map it uses (PARTS_TO_DEL) translates parts subtypes (first_delete,
first_use, last_delete, last_use, tail_delete, outer_delete, outer_use, outer,
center_delete, center_use) into head/tail/ends/middle.

DESIGN CONSIDERATION (scoped to charade piece 1):
Problem 1 — local PARTS patchwork, and inconsistent with other solvers. Charade's
PARTS_TO_DEL map is not the same as the deletion solver's: the deletion solver also
maps inner_use to middle, but charade's map omits inner_use. So the same parts
subtype is handled differently depending on which solver you are in — concrete
evidence for the general design consideration "Harmonise PARTS subtype classes in
the indicator table".
Problem 2 — the flags are clue-global. A single reversal, odd/even, or deletion
indicator anywhere makes that operation a candidate for every word, with no
requirement that the operated-on word is near the indicator. See the general design
consideration "An operation may only apply to material near its controlling
indicator".
Potential solution — read a single harmonised PARTS subtype from the indicator
table rather than a local map, and bind each licensed operation to material in
clear proximity to its indicator rather than enabling it clue-wide.

###### Piece 2 — per-word value collection

Factual summary:
For each remaining word (including the indicator words from piece 1 — they are not
excluded), charade builds a bag of candidate letter-contributions. Every candidate
is kept only if the value it produces is a substring of the answer. The candidate
types: abbreviation; synonym (up to answer length); reversed abbreviation/synonym
(only if a reversal indicator was flagged); odd/even letters of the word (only if
an odd/even indicator was flagged); deletion variants of longer synonyms (only if a
deletion indicator was flagged); the word's own raw literal letters; and the word's
first letter.

DESIGN CONSIDERATION (scoped to charade piece 2):
Problem 1 — first-letter extraction is ungated. Every word's first letter is added
as a candidate, filtered only by the letter appearing in the answer (true for
almost any letter), so nearly every word contributes its first letter. First-letter
(initialism) is a mechanism that must be licensed by an indicator, and that
indicator must satisfy the proximity rule. See the general design consideration
"An operation may only apply to material near its controlling indicator".
Problem 2 — the deletion variants ignore the subtype. Piece 1 works out the
deletion subtype (head/tail/ends/middle), but here the code only checks that some
deletion indicator exists and then tries every deletion form ("general"). The
subtype must be respected, as the standalone deletion solver does.
Problem 3 — the literal candidate has no length limit. It accepts any word whose
full surface letters fall inside the answer. Literals do legitimately appear in
charades, but in practice only very short words (most commonly "A", and some two-
or three-letter words). Literals should be limited to short words.
Problem 4 (potential issue) — indicator words are not excluded from value
collection, so a single word can be double-counted: licensing an operation as an
indicator in piece 1 and also contributing letters as a piece here.
Potential solution — require an initialism indicator (with proximity) before
offering first letters; respect the detected deletion subtype; cap literals to
short words; and consider excluding a word already consumed as an indicator from
also being a letter source.

###### Piece 3 — phrase lookups

Factual summary:
After the single-word values, charade looks up short phrases. For each word
position it forms the contiguous 2-word and 3-word phrases starting there, makes a
lookup key (lowercased, punctuation stripped), and looks that phrase up in the
abbreviation and synonym tables, keeping an abbreviation if it appears in the
answer and a synonym if it is a substring of the answer within the answer length.
A phrase that yields values is added to the candidate bag tagged with its span (2
or 3) so the assembler knows how many original words it covers. Phrase lookups try
only abbreviation and synonym — not reversal, odd/even, deletion, literal or first
letter. The phrases are contiguous and gated by the same substring-of-answer test.
These are plain charade pieces (a multi-word synonym or abbreviation), not
operations needing an indicator, so the indicator/proximity concerns do not apply.

DESIGN CONSIDERATION (scoped to charade piece 3, minor):
Problem — phrase lookups are capped at span 3; phrases of four or more words are
never tried, a coverage limitation.
Potential solution — allow longer phrase spans where the reference data has
multi-word entries.

###### Piece 4 — the recursive assembler (try_build)

Factual summary:
The assembler walks the candidate entries in clue order and consumes the answer
left-to-right. At each entry it either uses one of the entry's values if that value
is the next prefix of what remains of the answer (recording the entry's covered
word indices as used), or skips the entry. It succeeds when the answer is fully
consumed and every clue word not used as a piece is either a link word or an
indicator; if any leftover word is neither, the assembly is rejected. Because
entries are walked in clue order and values must match as a prefix, the pieces are
required to appear in the same order as the words in the clue (the right-to-left
case is handled separately in piece 5).

Strength to note: the leftover-word rule is a genuine residue check — it stops the
assembler from quietly ignoring clue words; every word must be accounted for as a
piece, a link word, or an indicator.

DESIGN CONSIDERATION (scoped to charade piece 4):
Problem (CONFIRMED BUG) — the covered-word computation is wrong for 3-word phrases.
For a multi-word entry the code does w.split(" ", 1) and then matches only a
consecutive pair. Verified by trace: a 2-word phrase "old boy" gives parts
['old', 'boy'] and covered {0, 1} (correct), but a 3-word phrase "man of war"
gives parts ['man', 'of war'] and covered set() (empty). So a span-3 phrase marks
none of its words used. Consequences: the leftover-word check then sees those three
words as unused and rejects the assembly unless each is a link word or indicator (a
false negative), and because the covered set is empty those words also remain
available for other entries to reuse.
Problem (minor) — duplicate words are tracked imprecisely. For single-word entries
the covered index is found with remaining_words.index(w), which returns the first
occurrence, so if the same word appears twice both uses resolve to that first
index and are not tracked independently.
Potential solution — compute the covered indices from the phrase's full span rather
than a single split-and-pair match, so 3-word (and longer) phrases mark all their
words used; and track word positions by index rather than by value so repeated
words are handled correctly.

###### Piece 5 — the reverse-order retry

Factual summary:
After the forward assembly, if it produced nothing or fewer than two pieces,
charade retries in reverse word order, because many clues read right-to-left (for
example "waste following public" = OVERT + URE). It builds a reversed copy of the
candidate entries, defines a second assembler (try_build_rev) that is an exact
duplicate of the forward assembler except it iterates the reversed list, runs it,
and takes its result. If two or more pieces come back, charade returns them.

DESIGN CONSIDERATION (scoped to charade piece 5, minor):
Problem — a redundant computation. Inside the reverse branch, before the reversed
assembler is defined, the code reassigns the result by calling the forward
assembler again (the same forward search that just ran and gave the too-short
result). That value is immediately overwritten by the reversed assembler's result
on the next line, so the forward re-run is dead work — wasteful, but not a
correctness error.
Potential solution — remove the redundant forward call.

###### Piece 6 — the return shape

Factual summary:
After the forward and (if needed) reverse attempts, charade checks the result: if
there are at least two pieces it returns a dict typed wordplay_type "charade" with
the list of pieces, each carrying its clue_word, the letters it contributed, and
its mechanism. Otherwise it returns nothing. A single-piece result is not accepted
as a charade.

DESIGN CONSIDERATION (scoped to charade piece 6):
Problem — the returned type flattens compound wordplay. The wordplay_type is always
"charade", even when the pieces include reversal, deletion, odd/even or first-letter
operations gathered in piece 2. The per-piece mechanism field records those
sub-operations, but the top-level type collapses a compound wordplay (for example a
charade of a reversal and a deletion) to plain "charade". A key project aim is to
expose deeper, compound clue types rather than broad lossy labels, so this
flattening hides structure the user should see.
Potential solution — emit a structured/compound type that reflects the actual
operations performed, instead of a single broad label.

##### Phase 0.5 — reversal solver (try_reversal)

Factual summary:
The reversal solver checks whether reversing a synonym or the raw form of any
single word spells the answer. For each remaining word it skips the word if RefDB
types it as any kind of indicator, then checks each synonym (up to answer length)
reversed against the answer, and the raw word (same length as the answer) reversed
against the answer. The first exact match returns a reversal result with that
source word as the single piece. It is a whole-answer reversal — one source
reversed to give the entire answer (partial reversals inside a charade are handled
in the charade solver). In its favour, unlike charade it does skip indicator words
as candidate sources.

DESIGN CONSIDERATION (scoped to the reversal solver):
Problem — no reversal indicator is required. A reversal is accepted purely because
some word's synonym (or the word itself) reversed equals the answer, with no
reversal indicator ("reversed", "going back", "returning", and so on) anywhere in
the clue. This is the mechanism-precedence gap: an indicator-requiring mechanism
made available with no indicator. Reversal must require a reversal indicator that
satisfies the proximity rule. See the general design consideration "An operation
may only apply to material near its controlling indicator".
Potential solution — require a reversal indicator, in clear proximity to the word
being reversed, before accepting a reversal.

##### Phase 0.5 — acrostic solver (try_acrostic)

Factual summary:
The acrostic solver checks whether the first letters — or, in a second pass, the
last letters — of a contiguous run of words spell the answer. It requires an answer
of at least 3 letters and at least as many words as answer letters. It slides a
window the exact length of the answer across the words: if the first letters of
that contiguous run spell the answer it returns an acrostic (each word contributing
its first letter); otherwise it tries the same with last letters. The window length
equals the answer length, so it is a pure acrostic — one word per answer letter. In
its favour, the run is contiguous, which is correct for an acrostic.

DESIGN CONSIDERATION (scoped to the acrostic solver):
Problem — no indicator is required. An acrostic is accepted purely because the
initial (or final) letters of some contiguous run spell the answer, with no
acrostic indicator present — nothing like "initially", "firstly", "heads" for first
letters, or "finally", "endings", "tails" for last letters. This is the
mechanism-precedence gap. Acrostic must require the matching indicator (first-letter
versus last-letter) in clear proximity to the run. See the general design
consideration "An operation may only apply to material near its controlling
indicator".
Potential solution — require the appropriate first-letter or last-letter indicator,
in proximity to the contiguous run, before accepting an acrostic.

##### Phase 0.5 — homophone solver (try_homophone)

Factual summary:
The homophone solver checks whether any single word has a homophone, in RefDB's
homophones table, that equals the answer. For each remaining word it looks up that
word's homophones and, if one matches the cleaned answer exactly, returns a
homophone result with that word as the single piece (mechanism sound_of). It is a
whole-answer homophone — one word sounds like the entire answer — and depends
entirely on the homophones table's coverage.

DESIGN CONSIDERATION (scoped to the homophone solver):
Problem 1 — no indicator is required. A homophone is accepted purely because a
word's homophone equals the answer, with no homophone indicator present — nothing
like "we hear", "sounds like", "reportedly", "audibly", "said", "on the radio".
This is the mechanism-precedence gap. Homophone must require a homophone indicator
in clear proximity. See the general design consideration "An operation may only
apply to material near its controlling indicator".
Problem 2 — detection leans on a table that cannot be comprehensive. Homophones are
subtle and open-ended; a homophones table will always miss many, so a lookup-only
approach has inherently weak coverage for this mechanism.
Potential solution — require a homophone indicator with proximity; and once a
homophone reading is licensed, use a cheap Haiku call to judge whether the word
sounds like the answer, rather than depending on the homophones table.

##### Phase 0.5 — wordplay-first fallback (_v1_solve_without_definition)

Factual summary:
This fallback runs only when none of the seven solvers succeeded and no definition
was found. Instead of finding the definition first, it assumes each possible edge
window is the definition and tries to assemble the rest as wordplay. It strips the
enumeration, requires at least 3 words, then builds candidate definitions from
every window of 1 to 5 words taken from the start and from the end of the clue
(always leaving at least 2 words for wordplay). For each candidate, with the
remaining words it runs the mechanical solvers and, at the first one that assembles
to the answer, returns that edge window as the "definition" along with the wordplay
type and pieces.

DESIGN CONSIDERATION (scoped to the wordplay-first fallback):
Problem 1 — the inferred definition has no semantic validation. The window it
returns is simply whichever edge phrase was left over when some wordplay parse
succeeded. It cannot be validated against the reference DB — these are the very
windows the definition stage already tried against the DB and rejected, which is
why this fallback is running. So a coincidental wordplay parse can declare an
arbitrary edge phrase to be the definition with nothing confirming it actually
defines the answer.
Problem 2 — inconsistent and riskier solver order. The main Phase 0.5 path runs
anagram first and deliberately places container before charade, because charade is
the most permissive solver and prone to false positives. This fallback runs charade
first (with anagram fifth), so the most permissive solver gets first claim across
every candidate definition window — the opposite of the deliberate main ordering,
which amplifies false positives.
Potential solution — since the DB is known not to contain it, judge the inferred
definition with a cheap Haiku call and/or queue it for enrichment review, and carry
a flag marking it wordplay-inferred and unvalidated; and align the fallback's solver
order with the main path's deliberate ordering.

#### Phase 1 — the signature solver caller (run.py)

Factual summary:
Phase 1 is a stage in the cascade whose whole job is to call the signature engine
(solve_clue, documented step by step in the walkthrough below) once per clue and file
each result. It runs only when RefDB is loaded. For each clue not already claimed by
Phase 0/0.5, and not a cross-reference clue, it calls solve_clue with the clue text, the
cleaned answer and RefDB; the engine returns a result carrying a confidence score
(0-100), the parse, and the per-word evidence it gathered. In the cascade this calling
stage sits after the cruder Phase 0/0.5 engines, so any clue they already claimed never
reaches it.

It then sorts the returned result three ways:
- HIGH (80 or more): the clue is marked solved, added to the run's results, and stored to
  the database (store_signature_result); if the engine attached a definition, that
  definition is queued for enrichment review.
- MEDIUM (solved but below 80): counted, and kept in an in-memory map for a possible
  Phase 3 upgrade — but not added to the results and not written to the database as a
  solve.
- Anything else: not solved.

Three side-writes happen regardless of band: if there is a definition and the clue's
definition column is still empty it is written in; any DBE-Haiku synonym candidates the
engine collected are queued for enrichment review unconditionally (the rationale in the
code being that the Haiku call was already paid for and the candidates passed an
answer-substring filter); and suggested indicators are queued only when the solve reached
HIGH (the successful solve is the proof the indicator was right).

DESIGN CONSIDERATION (scoped to the Phase 1 caller):
Problem 1 — "zero API cost" is wrong. The print line calls Phase 1 "mechanical, zero API
cost", but the engine it calls makes paid Haiku calls in four of its steps (Steps 3, 6, 7
and 8 of the walkthrough below). The phase is not zero-cost.
Problem 2 — a MEDIUM parse is discarded as durable evidence. It lives only in the
in-memory map for the optional Phase 3 retry; its per-word letter-contributing evidence
reaches no database. See the general consideration "Preserve all letter-contributing
evidence, even on failure".
Problem 3 — a definition that drove a MEDIUM is written but not queued. The definition is
written into the clues table on any band where the column is empty, but it is only queued
for review inside the HIGH branch — so a (possibly Haiku-guessed) definition behind a
medium parse is written to the DB yet never put up for review, and carries no flag marking
it a guess.
Problem 4 — only this stage earns its confidence from a real scorer, yet it sits behind
the self-certifying Phase 0/0.5 engines that claim clues first. Because the cascade is
first-to-claim-wins, a self-certified earlier parse blocks the one engine that actually
scores its own work from ever seeing the clue. See the general consideration "The cascade
order is backwards — best engine runs last".
Potential solution — correct the cost label; persist medium parses (or at least their
evidence) rather than holding them only in memory; queue a medium-backing definition for
review and flag LLM-asserted definitions; and reconsider the ordering so the scored engine
is not gated behind unscored ones.

---

## Signature solver — detailed walkthrough (solve_clue, step by step)

This walkthrough is the internals of the signature engine (solve_clue) that the Phase 1
caller above invokes once per clue. The caller decides what to do with each result (store
a HIGH, hold a MEDIUM, queue enrichments); the steps below are how the engine produces one
result. Read the Phase 1 caller section first for the framing, then these steps for the
detail.

The signature solver's front door is the function solve_clue in
signature_solver/solver.py (line 127). It is the first signature code that runs when a
clue arrives. We walk it one step at a time.

What it receives (the signature, lines 127-129): the full clue text (undivided, still
including the definition); the known answer; the RefDB reference data; and several
optional arguments (min_confidence, extra_catalog, extra_synonyms, extra_indicators,
and the flag _dbe_already_attempted). On a normal first call those optional arguments
are all empty or default — they only carry values when solve_clue calls itself again
later in its fallback chain, so they belong to steps much further down.

The key fact to hold from the outset: signature always knows the answer. It is not a
blind solver discovering the answer; it is given the answer and works out how the clue
produces it.

### Step 1 — normalise the clue and reduce the answer (lines 162-163)

Factual summary:
Before any solving, solve_clue prepares its two inputs.

- It normalises the clue text and splits it into a list of words. Normalising (the
  helper _normalize_clue, lines 114-124) means stripping accents (à becomes a) and
  converting smart quotes and long dashes into plain equivalents, then splitting on
  spaces into a word list (clue_words).
- It reduces the answer to bare letters: uppercased, with spaces and hyphens removed
  (answer_clean). So "TWELFTH NIGHT" becomes "TWELFTHNIGHT".

Nothing is solved here and no definition has been found yet. It has only cleaned up
the two things it was given, so the rest of the code has a tidy word-list and a
bare-letter answer to work with.

DESIGN CONSIDERATION (scoped to step 1):
Problem — this normalising-and-splitting is work other stages also perform for
themselves, independently. This is one of several places the same preparation happens.
See the general design consideration "Atomise the clue and answer once".
Potential solution — atomise the clue and answer a single time at the start of a run
and reuse that representation here rather than re-deriving it.

### Step 2 — extract definition candidates (line 165)

Factual summary:
With the clue split into words and the answer reduced to letters, solve_clue calls
extract_definition_candidates (lines 87-111). This is the step that decides which part
of the clue is the definition and which part is the wordplay.

The helper only ever takes the definition from the two ends of the clue, never the
middle. It walks a window size n from 1 up to 4 words and, for each n, tries two
splits:
- the first n words as the definition, the rest as wordplay;
- the last n words as the definition, everything before as wordplay.

Each candidate split is kept only if the DB confirms the definition phrase defines the
answer (db.is_definition_of). It returns a list of (definition phrase, wordplay words)
pairs — possibly empty, possibly several. This matches the cryptic convention that the
definition sits at one end of the clue.

What is_definition_of actually checks (db.py:273-293): it does not query the database
at solve time. All reference tables are read once into memory when RefDB is built
(_load_all, db.py:30-156); the test reads only the in-memory synonyms dictionary. That
dictionary is a merge of exactly two tables, both from cryptic_new.db:
- synonyms_pairs (columns word, synonym), db.py:66-72;
- definition_answers_augmented (columns definition, answer), appended into the same
  dictionary keyed on definition with answer as the value, db.py:76-88.

No other table is consulted for the definition test. The match is exact after a light
normalisation: keys are lowercased and stripped of punctuation (_normalize_key), the
comparison strips spaces and hyphens, and the only morphological flex is simple
possessive/plural variants (_word_variants, db.py:158-172). It checks both directions
— is any synonym of the phrase equal to the answer, and is any synonym of the answer
equal to the phrase — but both directions read the same merged dictionary, so both
rest on the same two tables. There is no fuzzy or stemmed matching.

DESIGN CONSIDERATION (scoped to step 2):
Problem 1 — entirely DB-gated, exact-match only. A split becomes a candidate only if
the exact (definition, answer) pair sits in synonyms_pairs or
definition_answers_augmented. A correct definition the DB happens not to hold is never
proposed. This is what triggers the Haiku definition fallback in the next step when the
candidate list comes back empty.
Problem 2 — every downstream stage is conditional on the definition guess. The wordplay
is always "whatever is left after removing the definition," so word analysis, matching
and scoring all run on words chosen by this split. A wrong split means the wordplay
stages analyse the wrong words.
Problem 3 — multiple candidates multiply the work. The helper can return several splits,
and solve_clue runs the full analyse-match-score pipeline once per candidate, so a clue
with three plausible definition splits is solved three times over.
Potential solution — connects to the general considerations "Atomise the clue and
answer once" (decide the definition once on a shared representation) and the broader
question of whether definition selection should be exact-DB-gated or allowed a
confidence-ranked set; noted here, not resolved.

### Step 3 — Haiku definition fallback (lines 167-185)

Factual summary:
After Step 2 returns its candidate list, solve_clue sets a flag haiku_tried = False
and then, only if that list is empty — RefDB found no edge window that defines the
answer — it consults Haiku for a definition. This is the first of two Haiku
definition invocation points in solve_clue (the second is the second-chance retry
much further down); the flag exists to stop the two firing redundantly.

When it fires it sets haiku_tried = True, imports find_definition from
haiku_definition, calls it with the full clue text and the answer, and if a result
comes back appends that single (definition phrase, wordplay words) pair to the
candidate list. The whole block is wrapped in a bare try/except that silently
swallows any error, in which case no candidate is added.

The function it calls is the same one documented in the Phase 0.5 definition stage
(Path 2) — a paid claude-haiku-4-5 call, max_tokens 50, temperature 0. It is told
the answer; it is not solving the clue, only being asked to point at the phrase
that defines a known answer. The prompt sent is, verbatim (the answer and clue are
substituted in):

  In this cryptic crossword clue, which word or short phrase is a synonym or
  definition of the answer ANSWER? The definition must be either the FIRST word(s)
  of the clue or the LAST word(s) of the clue — no real words may be left orphaned
  before it (if at the start) or after it (if at the end). Reply with the exact
  phrase from the clue, nothing else.

  Clue: CLUE_TEXT

After the call, the reply is validated by position, not by the DB (the DB just
failed, by construction). The code lowercases and strips punctuation, then requires
Haiku's phrase to equal a contiguous run of words at the start of the clue, or
failing that at the end, allowing only pure-punctuation tokens to be skipped at the
edge (the same orphan rule the prompt states). It also requires at least one
wordplay word to remain, so a reply that consumes the whole clue is rejected. If no
edge window matches exactly, the function returns None and the paid call yields no
candidate — there is no looser fallback.

DESIGN CONSIDERATION (scoped to step 3):
Problem 1 — paid API call inside a phase the orchestrator labels "mechanical, zero
API cost". This is one of four steps in solve_clue that make that label wrong (the
others are the DBE-Haiku, indicator-enrichment, and Haiku second-chance steps below).
Problem 2 — two different standards of proof, indistinguishable downstream. A
candidate produced by Step 2 rests on an exact reference-DB pair; a candidate
produced here rests on the model's say-so with only a positional check. Once the
Haiku pair is appended to the same candidate list, nothing records which standard
produced it, and every later stage (word analysis, matching, scoring) treats the two
identically. See the general design consideration on a single shared definition
stage with a single, flagged LLM fallback.

### Step 4 — grammar triage (orchestration, lines 189-224)

Factual summary:
After definition extraction (and the Haiku fallback) has produced a candidate list,
solve_clue runs grammar-guided triage — a fast, structural engine that tries to
solve the clue mechanically before the catalog matcher is consulted. The whole block
is wrapped in a try/except: if the triage module cannot be imported or throws, it
silently passes and continues to the catalog solver.

The orchestration bands grammar triage's results by confidence:
- For each (definition phrase, wordplay words) candidate it calls grammar_triage with
  that split. If the returned result scores 80 or more it stamps the definition onto
  it and keeps it as the best-so-far when it beats the current best. If the result
  scores 90 or more it returns immediately — the first 90 wins and the remaining
  candidates are never tried.
- If, after the candidate loop, nothing reached 80, it retries grammar_triage once
  more with the whole clue treated as wordplay (no definition removed), and if that
  scores 80 or more it re-attaches the first candidate's definition phrase for display.
- Then, if the best result is 80 or more it is returned; otherwise solve_clue falls
  through to the catalog solver.

The crucial detail is what happens to a grammar-triage result that scores below 80.
grammar_triage only returns a result at all when its structural test has fully rebuilt
the answer (the result is a complete, self-verified parse with word roles and letter
evidence). But the orchestration keeps results only at 80 or more (the three "80 or
more" guards). A sub-80 result — and grammar_triage genuinely returns these, for
example the unindicated anagram at 70 and the positional-feed anagram at 65 — is
computed, then discarded. It is not stored, not passed down, and not used to seed the
catalog solver, which then re-analyses the same words from scratch.

DESIGN CONSIDERATION (scoped to step 4 orchestration):
Problem 1 — first-acceptable, not best. The engine returns at the first result scoring
90, so other definition candidates that might score higher are never tried; the
"keep the best" logic only operates within the 80-89 band.
Problem 2 — a complete, reconstructed parse is discarded on a single confidence number.
A sub-80 grammar result is a fully-assembled parse that already rebuilds the answer,
yet it is thrown away — and the work partly redone by the catalog solver. This is a
sharper instance of the general design consideration "Preserve all letter-contributing
evidence, even on failure": here it is not a failure being discarded but a success,
binned for scoring 70 rather than 80. Because the per-test confidences are hardcoded
constants (see the engine overview below) rather than derived, the cut is blunt — it
bins a coincidental unlicensed match and a genuinely-good-but-just-under-80 match with
exactly the same threshold, retaining neither the evidence nor the distinction between
them.
Problem 3 — the whole-clue retry can double-use the definition words. When no clean
split solves, the entire clue (including the definition words) is fed back as wordplay
fodder, then the first candidate's definition is re-attached for display, so the stated
definition words may also have been consumed as wordplay pieces — the parse and the
displayed definition are not guaranteed consistent (the code comment acknowledges this).
Potential solution — score parses through one real scorer and keep the global best
across candidates; retain a sub-threshold parse and its evidence rather than discarding
it; and make the whole-clue retry honest about which words it consumed.

#### Step 4 — the grammar_triage engine (overview)

Factual summary:
grammar_triage (grammar_triage.py, lines 1276-1413) is given the wordplay words and
the known answer. It starts a one-second timer (TRIAGE_TIMEOUT = 1.0) and checks it
between every test. It computes a ratio — total wordplay letters divided by answer
length — which gates which anagram tests are attempted. It then runs a fixed cascade of
structural tests and returns the first that fires:

1. standalone anagram (try_anagram), when the ratio is 0.8 to 2.5;
2. standalone anagram again, when the ratio exceeds 2.5 (the abbreviation-substitution
   case matters more here);
3. anagram with a positional letter fed into the fodder (try_anagram_with_positional),
   ratio 1.5 to 4.0;
4. standalone pure reversal (try_reversal);
5. the POS-guided path: POS-tag the words with spaCy, look the abstracted tag sequence
   up in data/grammar_catalog.json and verify each candidate role sequence
   (_verify_grammar_roles), then a POS bigram container detector (_try_container);
6. container (try_container) without POS guidance;
7. container + charade compound (try_container_charade);
8. container with a deletion-derived inner (try_container_with_deletion);
9. anagram + charade compound (try_anagram_charade);
10. the general charade (try_charade).

Each test, when it fires, returns a result carrying a confidence baked into its result
builder (for example anagram 90 with an indicator / 70 without, charade 80-95, container
90, reversal 90, the compounds 85). The individual tests are documented one at a time in
the pieces that follow this overview.

DESIGN CONSIDERATION (scoped to the engine overview):
Problem 1 — every result carries a hardcoded confidence, and each constant meets or
beats the thresholds that gate this step (70 for anagram, 80/90 for the rest). So in
practice almost any structural match grammar triage finds is accepted, returns early,
and bypasses confidence.py and the whole catalog/scoring path the rest of the signature
engine relies on. The numbers that gate the cascade are numbers the tests assigned
themselves.
Problem 2 — the licensing is inconsistent. Some tests are properly indicator-gated
(anagram-with-positional, container-with-deletion, anagram-charade, and the positional
pieces inside charade/container). But the three tests that run earliest and most often —
standalone anagram, standalone reversal, container — accept with no indicator at all.
See the general design consideration "An operation may only apply to material near its
controlling indicator". (The anagram case is qualified in its own section below.)
Problem 3 — even the gated tests check presence, not proximity. The positional-indicator
test scans every wordplay word; nothing requires the indicator to be near the material
it operates on — the same proximity gap documented for the V1 solvers.
Problem 4 — the one-second timeout is a silent correctness cliff. A longer clue that
needs more time returns nothing and drops to the catalog solver, so the outcome depends
on how fast the machine happened to be.
Problem 5 — a hard dependency on spaCy for the grammar half. Without the model installed,
the learned grammar-catalog lookup and the POS container detector never run; triage
quietly degrades to the structural tests only.
Problem 6 — dead code: _detect_reversal is defined and never called in the cascade.

##### Step 4 — anagram detection (try_anagram, lines 276-356)

Factual summary:
The anagram detector rests on a single idea: two strings are anagrams if and only if
their letters, sorted, are identical. It sorts the answer's letters once, and reduces
each wordplay word to its uppercase letters-only form. Detection is then the search for
some selection of those letters whose sorted form equals the sorted answer. It never
generates the answer; it only checks letter-multiset equality. It tries four strategies
in order:

1. all wordplay letters together;
2. exclude each single word in turn (this is how an anagram indicator or a link word —
   which contribute no letters — are accommodated: drop one word and the rest anagram);
3. exclude each pair of words (only when there are 4+ words);
4. substitution: a word with a short 1-2 letter DB value (for example western = W,
   area = A) may contribute that value instead of its raw letters; it tries one or two
   such substitutions combined with every combination of excluding the other words.

A word therefore has exactly three possible fates: all of its raw letters are fodder, it
is excluded entirely, or it is swapped wholesale for a short DB value. It works on whole
words only — it cannot take a partial slice of a word into the fodder (no first-three-
letters, no "most of" a word). The single exception lives in a separate test,
try_anagram_with_positional, where one word may feed just its first or last letter into
the fodder, and only when a first/last positional indicator licenses it. There is no
deletion-then-anagram path anywhere (anagram of a word with a letter removed).

The indicator plays no part in detection. Whether a match is declared rests entirely on
the sorted-letters comparison. The indicator only matters afterward, in the result
builder, which sets confidence to 90 when one of the excluded words is an anagram
indicator and 70 when there is none, and labels that excluded word as an anagram
indicator versus a link word. The ratio gate upstream is what keeps the detector from
trying anagrams on clues whose letter counts cannot plausibly work.

DESIGN CONSIDERATION (scoped to anagram detection):
Problem 1 — the whole-word-only constraint is a real coverage gap. The fodder must be
made of whole words (or whole-word substitutions), so the common cryptic pattern of an
anagram over a partial word — "most of", "endlessly", "headless" feeding a truncated
word into the fodder — has no path here. A better implementation (the WFW project)
should be revived rather than this one extended.
Problem 2 — an unindicated anagram match is NOT inherently weak, and downgrading it to
70 is a DB-coverage artefact, not a quality judgement. Because the fodder is whole words
and the fodder letters must sum to exactly the answer length and match its multiset
exactly, an exact whole-word anagram is a strong coincidence to occur by chance: it is
real evidence that the clue is an anagram. When no anagram indicator is found, the more
likely explanation is that the indicator table does not contain that particular signal
word — not that the match is spurious. So scoring such a match at 70 and then discarding
it (because 70 is below the orchestration's 80 floor, see step 4 orchestration above)
throws away a probably-correct parse on the grounds that our own indicator list is
incomplete. The 70-versus-90 split is being driven by indicator presence in an
incomplete DB, not by the quality of the match.
Two caveats keep this honest. First, the argument is length-dependent: for a three- or
four-letter answer surrounded by many small words, an exact whole-word match is much
easier to hit by chance, so for short answers an unindicated match really can be
coincidental. Second, the substitution branch (strategy 4) is the looser one — swapping
in one or two short abbreviations gives the search free letters to play with, which is
where most genuine coincidence risk lives; the clean "all words" or "drop one word"
cases are the convincing ones.
Potential solution — score an anagram on the match itself (exactness, how much of the
clue the fodder consumes, answer length, whether substitution was needed) rather than on
indicator presence; and when no indicator is found for an otherwise-clean exact anagram,
queue the licensing word as a candidate anagram indicator for enrichment instead of
downgrading and discarding the parse.

##### Step 4 — reversal (try_reversal, lines 907-933)

Factual summary:
This test looks for a whole-answer reversal: a single clue word whose value, reversed,
spells the entire answer. For each wordplay word it pulls that word's values and, for
each value, requires two conditions together — the value's length equals the answer
length, and the value reversed equals the answer exactly. The first word/value pair that
satisfies both wins and returns immediately. The matched word is labelled SYN_F or
ABR_F; every other word is labelled a reversal indicator (REV_I) if the DB classifies it
as one, otherwise a link word. It returns confidence 90.

Three facts about its scope:
- It is whole-answer only — the value must be exactly the answer length and reverse to
  the whole answer. The docstring says "full or charade piece," but the code does the
  full case only; partial reversals inside a charade are handled separately in
  try_charade. The docstring overstates the function.
- The values come from DB synonyms and abbreviations only (the shared value helper
  returns synonyms and abbreviations and nothing else). The literal clue word is never
  offered as a value, so a pure literal-letter reversal — a clue word whose own letters
  reverse to the answer — is not a path here. (Note this differs from the hidden engine
  in Phase 0, which works on the literal clue letters: the two are not the same mechanism
  and neither makes the other redundant. A clean whole-word literal reversal in fact
  falls through both — this test because it has no literal value, and the hidden engine
  because it excludes whole-word matches.)
- No reversal indicator is required, for detection or for the score. It returns 90
  whether or not any word is a reversal indicator; with none, every other word simply
  becomes a link word and there is no REV_I role at all, yet the confidence is still 90,
  which short-circuits the whole engine (90 or more returns immediately in solve_clue).
  Separately, the signature token list is hardcoded as [SYN_F, REV_I] even when the word
  roles contain no REV_I word, so the declared signature can claim a reversal indicator
  the parse does not actually have.

DESIGN CONSIDERATION (scoped to reversal):
Problem 1 — no reversal indicator is required, yet it returns at 90 and short-circuits
everything. This is the mechanism-precedence gap (an indicator-requiring mechanism made
available with no indicator), the same one flagged for the Phase 0.5 reversal solver, but
here at a confidence that pre-empts the rest of the signature engine. See the general
design consideration "An operation may only apply to material near its controlling
indicator".
Problem 2 — whether an unindicated reversal is weak depends on the source, and the
source here is the synonym table. The exact-reverse constraint is tighter than the
anagram's letter-multiset constraint, so a reversal built on a solid, direct synonym or
abbreviation is strong evidence in its own right, and a missing indicator is then better
read as a DB-coverage gap than as spuriousness — the same argument made for the anagram.
But unlike the anagram, which leans on the literal clue letters, this test reverses a
looked-up synonym, and the synonym table is known to be polluted (the 31251 audit found
"first" carrying 187 junk synonyms). A junk synonym that happens to reverse to the
answer is a real false-positive route that the clean anagram does not have. So the
strong-evidence reading applies only when the synonym or abbreviation is solid; the
synonym-pollution route is the reason to be more cautious here than with the anagram.
Problem 3 — the hardcoded signature [SYN_F, REV_I] can misdescribe the parse when no
reversal indicator word exists.
Potential solution — require a reversal indicator in clear proximity before accepting,
or, where an otherwise-solid reversal has no indicator, queue the licensing word for
enrichment rather than accepting silently; score the reversal on the directness of the
underlying synonym/abbreviation (penalising weak or pollution-prone synonyms) rather than
on a flat 90; and derive the signature token list from the actual word roles. Cleaning
the polluted synonym table (see the 31251 audit findings) reduces the false-positive
route directly.

##### Step 4 — the POS / grammar-catalog path (lines 1344-1369, _verify_grammar_roles)

Factual summary:
This is the one learned component in an otherwise hand-coded engine — a frequency model
that maps a clue's grammatical shape to the role layouts that have worked for it before.
The file data/grammar_catalog.json holds 166 grammatical patterns; each maps to a ranked
list of "recipes" (which word does which job) with a count of how often each was seen.
For the pattern "adjective then noun", for example, the commonest recipe (78 times) is
"first word an abbreviation, second a synonym", then "both synonyms" (65), and so on.

At solve time it labels each wordplay word with its part of speech using spaCy,
simplifies that to a rough pattern, and if the pattern is one of the 166 it tries the
recipes most-common-first, keeping the first that rebuilds the known answer. A separate
part-of-speech check can also trigger the container test.

A recipe is checked by the same join-the-pieces routine the charade solver uses, so a
verified recipe is filed as a "charade" and scored 80-95.

DESIGN CONSIDERATION (scoped to the POS / grammar-catalog path):
Problem 1 — the anagram recipes cannot fire. The checker only builds the answer by
joining pieces in order; it never rearranges letters. So a recipe calling for anagram
fodder can only pass if the letters are already in order — i.e. it was not an anagram.
Genuine anagrams are caught by the standalone anagram tests earlier, never here, so the
catalog's anagram recipes are effectively dead.
Problem 2 — the recipe is a loose hint, not a rule. A word the recipe expects to
contribute is quietly dropped if it cannot, and the checker also tries tricks the recipe
never named (reversal, first/last letter, two-word phrases). So the grammar layer really
only suggests which words are fodder; the solving is the same join-the-pieces routine
used elsewhere, raising the question of how much the POS step adds.
Problem 3 — every success is filed as "charade" and scored high enough (80+) to
short-circuit the rest of the engine, so richer structure is flattened and the later,
more careful stages never see the clue.
Problem 4 — coverage is bounded twice: by a hard spaCy dependency (no spaCy, no path)
and by the 166 known patterns (an unseen grammatical shape gets no attempt).

DESIGN CONSIDERATION (cross-cutting — placement of triage):
The engine is called "triage" but runs late, inside the signature solver, and routes
nothing — the name promises a front-door classifier, the placement delivers a late-stage
solver. Because the pipeline always has the answer, these answer-aware solve-and-confirm
tests could run at stage 0 to identify the clue type and route, or to compete on score,
rather than being gated behind the cruder, self-certifying Phase 0.5 engines that claim
clues first. This is the concrete form of the general consideration "The cascade order is
backwards — best engine runs last". It compounds with outright duplication: anagram,
reversal, container and charade are implemented both in the Phase 0.5 V1 solvers and
again here, and the two versions have already drifted apart.

##### Step 4 — container (try_container, lines 701-750)

Factual summary:
This reads the answer as outer letters wrapped around inner letters. For each ordered
pair of words it takes an outer value, checks that the answer begins and ends with the
two halves of that outer, and matches the gap in the middle against an inner value. The
first match wins and it scores 90. Outer and inner values come from synonyms and
abbreviations, plus first/last/outer-letter slices — but those slices are only allowed
when a positional indicator licenses them. It is whole-answer only; partial containers
are handled by the container-plus-charade test.

It is called twice in the engine: once when a part-of-speech check (_detect_container,
a "noun then -ing/-ed word then noun" pattern) spots a possible container, and again a
few lines later unconditionally, with no part-of-speech gate and no indicator gate, in a
branch that runs even without spaCy. The two calls are the identical function.

DESIGN CONSIDERATION (scoped to container):
Problem 1 — no container indicator is required. A word is marked a container indicator
if the DB says so, otherwise a link word, but presence is never required, and the test
still returns 90 and short-circuits the engine. This diverges from the Phase 0.5
container solver, which was gated on an indicator being present.
Problem 2 — this is the operator where running ungated is most dangerous, and the
"an unindicated match is strong evidence" reasoning from anagram and reversal does not
carry — it flips. Container is the most permissive operator and the most exposed to
coincidence: its indicators are very common words (in, around, about, holding), and the
operation itself is loose — any split of the answer into prefix, gap and suffix where the
ends are one synonym and the gap is another. A coincidental container is easy to hit, so
this is precisely where requiring an indicator with proximity matters most. See the
general consideration "An operation may only apply to material near its controlling
indicator", which already names container as the most exposed case.
Problem 3 — the grammar-based container detection is dead. The part-of-speech check can
trigger the test, but the unconditional call a few lines later runs it regardless, so the
grammar pre-check can never change the outcome — it only makes the test fire slightly
earlier.
Potential solution — require a container indicator in clear proximity to the outer/inner
material before accepting; do not treat an unindicated container as evidence; and remove
the redundant grammar pre-check (or make it actually gate the test).

##### Step 4 — container plus charade (try_container_charade, lines 753-849)

Factual summary:
This is the compound version of the container test: the container produces only a chunk
of the answer rather than the whole of it, and one or two other pieces fill the letters
before and after that chunk. It inserts an inner value into an outer value, checks that
the result appears somewhere inside the answer, then fills the remaining front and back
from other words' synonyms, abbreviations, or licensed first/last-letter slices. It needs
at least three words, scores 85, and short-circuits the engine. It is not grammar-gated
and runs unconditionally after the standalone container test.

DESIGN CONSIDERATION (scoped to container plus charade):
Problem 1 — as with the standalone container, no container indicator is required, yet it
returns 85 and short-circuits.
Problem 2 — it is looser still than the standalone container and so more exposed to
coincidence: the container now only has to produce a substring of the answer, with one or
two further pieces mopping up the rest, which gives even more freedom to land on a chance
match. The "an unindicated match is strong evidence" reasoning from anagram and reversal
does not apply here either — container is the most permissive operator, and this compound
loosens it further. See the general consideration "An operation may only apply to
material near its controlling indicator".
Potential solution — require a container indicator in clear proximity before accepting,
and do not treat an unindicated container-plus-charade as evidence.

##### Step 4 — container with deletion (try_container_with_deletion, lines 1123-1206)

Factual summary:
This handles a container whose inner piece is a synonym with one end letter removed — for
example DOODLED read as D(OODLE)D, where OODLE is OODLES with its last letter dropped.
Unlike the other container tests it is properly gated: it does nothing unless the clue
contains both a container (or insertion) indicator and a deletion indicator. It then
tries which word is each indicator — allowing a word the DB tags as an indicator to still
serve as fodder — drops one end letter from an inner value, and checks that the outer
wrapped around that shortened inner rebuilds the answer. It needs at least four words and
scores 85. Its own comment notes it is a deliberately separate structural test (new
compound types get their own slot rather than broadening an existing test).

DESIGN CONSIDERATION (scoped to container with deletion):
This is the best-behaved of the container tests, because it is genuinely gated on the two
indicators its mechanism requires; with both present and an exact reconstruction, an
unindicated-style false positive is far less likely here. Three smaller caveats remain.
Problem 1 — it still only checks the indicators are present somewhere, not that they sit
near the material they act on (the proximity gap). See the general consideration "An
operation may only apply to material near its controlling indicator".
Problem 2 — it carries its own local map of deletion subtypes and only ever removes a
single letter from one end (no middle or both-ends deletion, no multi-letter). This is
both a coverage limit and another instance of the general consideration "Harmonise PARTS
subtype classes in the indicator table".
Problem 3 — the inner and outer values come from synonyms and abbreviations, so the
synonym-pollution exposure noted for the other tests remains.
Potential solution — bind each indicator to nearby material; read a single harmonised
deletion subtype rather than a local map, and support the deletion forms it omits; and
clean the synonym source.

##### Step 4 — anagram plus charade (try_anagram_charade, lines 936-1072)

Factual summary:
Here some words are anagram fodder and the rest are charade pieces, and together they
make the answer: the charade pieces sit at the start and/or end and the anagram fills the
gap between them. It is gated — it does nothing unless the clue contains an anagram
indicator. The charade pieces come from synonyms, abbreviations, or licensed
first/last-letter slices; the fodder is whole words' raw letters. It needs at least three
words and scores 85. It keeps a compound type of its own (anagram-plus-charade) rather
than flattening to a single broad label.

DESIGN CONSIDERATION (scoped to anagram plus charade):
Like container-with-deletion, this is reasonably well-behaved because it is gated on the
indicator its mechanism requires, so an unindicated false positive is less likely; and it
preserves the compound nature of the wordplay in its type, which is what the redesign
wants (contrast the V1 charade solver, which flattens everything to "charade"). Three
standing caveats remain.
Problem 1 — the anagram indicator only has to be present, not near the fodder it acts on
(the proximity gap). See the general consideration "An operation may only apply to
material near its controlling indicator".
Problem 2 — the fodder is whole words only, the same limitation as the standalone anagram:
an anagram over a partial word (most-of, endless) cannot be expressed.
Problem 3 — the charade pieces come from synonyms and abbreviations, so the
synonym-pollution exposure remains.
Potential solution — bind the anagram indicator to nearby fodder; allow partial-word
fodder; and clean the synonym source.

##### Step 4 — the general charade (try_charade, lines 455-578)

Factual summary:
This is the catch-all "join the pieces" solver, run last in the engine because it is the
most permissive. For each word it gathers candidate pieces — a synonym or abbreviation
that appears in the answer; the word's own raw letters if they appear (blocked if the
word sits next to an example-marker such as "maybe" or "say"); first/last/outer-letter
slices, but only when a positional indicator licenses them; a reversed synonym if the
reversed form appears in the answer; and two- or three-word phrase lookups. It then lays
these pieces across the answer from left to right, each word used once, preferring the
most direct source (raw letters first, then synonyms and abbreviations, then reversal,
then slices). It refuses a single piece equal to the whole answer (that would be a
definition, not wordplay). It scores at least 80, up to 95 depending on how many pieces
are DB-confirmed, so any successful assembly short-circuits the engine.

One genuine strength: running last is the right ordering — the most permissive solver
gets last claim, not first (the opposite of the Phase 0.5 wordplay-first fallback, which
wrongly runs charade first).

DESIGN CONSIDERATION (scoped to the general charade):
Problem 1 — the reversal piece is not gated. A reversed synonym can be used as a piece
even when no reversal indicator exists anywhere in the clue — the same mechanism-
precedence gap as the standalone reversal test, now inside the charade. See the general
consideration "An operation may only apply to material near its controlling indicator".
Problem 2 — the positional slices are gated, but only on the indicator being present, not
on it being near the word it slices (the proximity gap).
Problem 3 — the top-level type is always "charade" even when pieces include reversal or
slice operations; the per-piece operation is recorded underneath, but the headline label
is lossy — the same flattening flagged for the V1 charade solver. See the general
consideration "Single-type engines versus the compound reality of clues".
Problem 4 — the pieces come from synonyms and abbreviations, so the synonym-pollution
exposure remains.
Potential solution — gate the reversal piece on a reversal indicator in proximity; bind
positional slices to nearby indicators; emit a compound type reflecting the operations
used rather than a flat "charade"; and clean the synonym source.

### Step 5 — the catalog path (solve, solver.py line 535)

Factual summary:
This is what the engine falls to when grammar triage did not produce an 80+ result. It
is slower and more thorough, and it is the one place confidence is earned rather than
hardcoded. It analyses the words once, then tries three catalogs in order:
- the base catalog — about 68 collapsed patterns such as "fodder + fodder charade",
  where each slot can stretch to several words;
- the positional catalog — fixed-span patterns, more entries;
- an older catalog — for patterns the first two do not cover.
For each catalog it walks the entries, and for every match it builds the explanation and
scores it, keeping the highest score. It stops at the first parse that scores 80, and
after the base and positional catalogs it returns straight away if the best so far is
80 or more.

How a match is found and verified (the base catalog is the heart): for each pattern it
generates ways to assign words to slots — a fodder slot takes 1 to 4 words, an indicator
slot 1 to 2, and leftover words become link-word gaps. For each assignment it looks up
what letters each fodder word can supply (synonyms, abbreviations, and so on) and then
rebuilds the answer from those pieces — a match only counts if the pieces, assembled per
the operation, actually make the known answer. Leftover words must all be link words or
indicators of an already-used type, or the match is rejected. So verification is by
reconstruction, with a genuine residue check.

The scorer (confidence.py): by the time scoring runs, the assembly is already proven to
make the answer, so the score is not "did it work" but "how trustworthy and explainable
are the pieces a user would see". It starts at 100 and deducts: a synonym that is not a
real word, -60; the definition reused as its own fodder (circularity), -30; an operation
missing its indicator, -20; a real-word synonym the DB cannot confirm, -20; an unconfirmed
homophone, -15; an unconfirmed abbreviation, -10; an indicator not in the DB or an
unverified link word, -5 each. 80 or more is HIGH, 50-79 MEDIUM, under 50 LOW.

DESIGN CONSIDERATION (scoped to the catalog path):
First, a genuine strength to record, not just problems. This deduction-from-100,
explanation-centred scoring is the right model — it scores the explanation the user will
read rather than the internals, and the indicator discipline is real here (the base
matcher requires the operation's indicator to be present, and for patterns with no
indicator slot, such as reversal-charade, it insists the indicator is among the leftover
words). This is the one part of the whole engine where the confidence number means
something.
Problem 1 — first-acceptable, not best, across the three catalogs. Each matcher breaks at
the first parse scoring 80, and the base and positional matchers return as soon as the
running best is 80 or more, so a later catalog that might score a clue higher is never
consulted. "Keep the best" only operates inside one matcher's loop.
Problem 2 — silent search caps. A pattern tries at most 20 placements and 50 fodder-type
combinations, a fodder slot stretches to at most 4 words and an indicator to 2. A correct
parse that needs more is never found — the same silent coverage cliff as the triage
timeout, with different limits.
Problem 3 — indicator gating is present but porous. It checks the indicator is present,
not that it sits near the material it acts on (the proximity gap, see the general
consideration "An operation may only apply to material near its controlling indicator").
And the scoring penalty for a missing indicator is only -20, so a parse whose only flaw
is a missing indicator still lands at exactly 80 — HIGH. A missing indicator alone does
not keep a clue out of HIGH.
Problem 4 — synonym pollution defeats the scorer. A synonym the DB confirms draws no
penalty, so if the synonym table is polluted (the 31251 "first carries 187 junk synonyms"
problem) a junk synonym counts as confirmed and a wrong parse built on it scores HIGH. The
scorer is only as honest as the table it trusts; cleaning the synonym source (see the
31251 audit) is what protects it.
Problem 5 — a stray debug print fires to stdout whenever the old-catalog branch runs with
extra entries (the Phase 3 path), leftover instrumentation rather than intended output.
Potential solution — score all candidate parses and keep the global best across the three
catalogs rather than the first to clear 80; surface or raise the search caps, and log when
a clue hit them; require indicators in proximity and weight a missing indicator more
heavily than -20; clean the synonym source so "confirmed" means trustworthy; and remove
the debug print.

### Step 6 — the DBE-Haiku fallback (solve_clue lines 243-278)

Factual summary:
This handles "definition by example": when a clue word is marked by "maybe", "perhaps",
"say" or "for example", it stands for an example rather than itself, so the wordplay piece
is something related to it — its category (Garibaldi giving BISCUIT), a famous bearer's
other name (Hill giving DAMON), or a type (tulip giving FLOWER). The synonyms table will
not hold these, so this step asks Haiku.

It runs only if nothing has reached HIGH yet, this is not already the recursive pass (no
injected synonyms), and DBE has not already been tried — so it is a gated last resort. It
finds the example-marked words and, for each, makes a paid Haiku call (find_dbe_candidates,
200 tokens, temperature 0) asking what the word could stand for. Haiku's replies are
filtered to those that appear as a substring of the answer or its reverse (and are not the
whole answer). Any survivors are injected as extra synonyms and solve_clue is called again
recursively; if that re-solve is better it is returned. Separately, every surviving
candidate is queued to pending_enrichments for human review unconditionally — the stated
rationale being that the call was paid for and the candidate passed the filter.

DESIGN CONSIDERATION (scoped to the DBE-Haiku fallback):
A counterweight to record first: the design is fundamentally "propose and queue", the raw
letters of the marked word are correctly blocked (the word stands for an example, not
itself), and a candidate must at least fit the answer — so as an enrichment-discovery
mechanism it is reasonable.
Problem 1 — paid and recursive, another contradiction of the "zero API cost" label for
Phase 1, and it re-enters solve_clue.
Problem 2 — the substring filter is the only gate on Haiku's output, and it is weak for
short candidates. A two- or three-letter candidate appears inside almost any longer answer
by chance, so many spurious suggestions pass and get both injected and queued.
Problem 3 — an injected DBE candidate scores as a confirmed synonym. Because it is added
through the extra-synonyms overlay, the scorer's "is this a known synonym?" check finds it
and applies no penalty, so a re-solve built on an unreviewed Haiku guess can reach HIGH and
be served to the user before the human ever reviews the queued pair. This is the same shape
as the synonym-pollution problem, but self-inflicted by the overlay — the real problem is
that this can serve a HIGH on the guess, not merely queue it.
Problem 4 — unconditional queueing fills the review queue with weak candidates; combined
with the weak filter, the reviewer gets a lot of short-substring noise.
Potential solution — gate Haiku's candidates more tightly than bare substring fit
(minimum length, alignment to a slot the matcher actually needs); mark an injected DBE
candidate as unverified so the scorer treats it as an unconfirmed synonym rather than a
confirmed one, keeping it out of HIGH until reviewed; and queue only the candidate that
actually contributed to a parse rather than every survivor.

### Step 7 — the indicator-enrichment fallback (solve_clue lines 280-447)

Factual summary:
If still nothing has reached HIGH, this step asks: is there a missing indicator that, if
the DB held it, would let the clue solve? If so, it injects it, re-solves, and — only if
it genuinely works — queues that indicator word for enrichment. It runs only if the best
so far is not HIGH and this is not already the indicator-injection retry.

It works in two passes. Pass 1 (rule-based): detect_missing_indicator proposes candidate
indicators — a clue word, an operation type, a subtype, and the gap it would fill —
skipping words already used in the partial parse. For the top three it injects each as an
extra indicator and re-solves; it accepts only if the re-solve is HIGH, beats the best,
and a check confirms the proposed word actually took that indicator role in the winning
parse. If several subtypes tie, it makes a paid Haiku call to pick the right one and
re-solves with that pick if it also verifies, so the queued subtype matches the parse.
Pass 2 (Haiku): only if the rule pass found at least one candidate (so the gap is known),
it makes a paid Haiku call for indicator words, injects the top three, and accepts under
the same "actually used" gate.

DESIGN CONSIDERATION (scoped to the indicator-enrichment fallback):
A genuine strength to record first: the "actually used" gate is disciplined — it will not
accept or queue an indicator unless that word really took the indicator role in a verified
HIGH parse. That is the opposite of the DBE step's unconditional queueing, and worth
crediting.
Problem 1 — paid and deeply recursive. Haiku fires in two places (subtype disambiguation
and indicator suggestion), and the block re-enters solve_clue repeatedly, another
contradiction of the "zero API cost" label and a cost/time concern on hard clues.
Problem 2 — a HIGH here can rest on a proposed indicator. Injecting the indicator is
exactly what removes the scorer's two indicator penalties (-20 for a missing indicator, -5
for one not in the DB), which is how the score climbs to HIGH. The "actually used" gate
ensures it was used and it is queued for review, but, as with the DBE step, the clue can be
served to the user as HIGH on an unconfirmed indicator before the human reviews it.
Problem 3 — this block is hard to reason about. It keeps re-running the whole solver inside
itself: to test a possible fix it does not patch the current attempt but starts the entire
solver again from the top with the proposed indicator added, and looks at the result. If
that works but there is a choice about which kind of indicator it is, it asks Haiku and
runs the whole solver again with Haiku's choice; the Haiku pass does the same for each of
its guesses. So solving one hard clue can set the solver running over and over, several
layers deep inside itself, and each of those re-runs has to carry a small bundle of state
down with it — the synonym guesses from the earlier DBE step, a switch saying "do not do
the DBE step again", and the proposed indicator — and pass them on correctly every time.
When code calls itself repeatedly and hands down a bundle of switches each time, it becomes
very hard to tell what state things are in at any moment and very hard to test every route
through it; a small change in one place can have effects several layers down that are not
obvious. The point is not that it is wrong, but that it is fragile and hard to reason about.
Potential solution — mark an injected indicator as unverified so the scorer keeps the clue
out of HIGH until the indicator is reviewed; and separate the concerns the redesign already
names (one definition stage, one evidence-gathering pass, one matcher) so late discoveries
are fed in once rather than by the solver re-entering itself.

### Step 8 — Haiku definition second-chance and the final return (lines 457-532)

Factual summary:
The second-chance fires when the best result still is not HIGH and Haiku has not yet been
asked for a definition. The first Haiku definition call (Step 3) only ran when the DB
returned no candidates; this covers the other case — the DB did return candidates, they
were weak, and none produced a confident parse, so Haiku never got a turn and gets one now.
It makes the paid definition call, and if Haiku returns something not already tried it
appends it to the candidate list, then retries both grammar triage and the catalog solver
with that new definition, keeping it if it scores.

One detail worth crediting: it appends rather than puts the Haiku guess first, and the code
comment records why — a real bug (1d CARRIAGE), where the DB correctly had "Coach" giving
CARRIAGE but a prepended Haiku guess "bearing" overrode it. Appending means the later
unsolved fallback still uses the DB definition when assembly fails. So this is a deliberate
guard against a Haiku guess displacing a correct DB definition.

The final return then resolves three ways:
- if there is any best result it is returned — even a medium below 80 (run.py does the
  banding afterwards), with the DBE candidates and any suggested indicator attached;
- if there is no result but there were definition candidates, it runs one last solve on the
  first candidate's wordplay at minimum confidence 0 and returns that — an unsolved or low
  result carrying the definition;
- if there were no candidates at all, it solves the whole clue as wordplay with no
  definition.

DESIGN CONSIDERATION (scoped to step 8 and the final return):
Strength to record: the append-not-prepend ordering is a genuine correctness guard against
a Haiku guess displacing a correct DB definition (the CARRIAGE example).
Problem 1 — another paid Haiku call, the second of the two definition-Haiku points, again
under the "zero API cost" label.
Problem 2 — the same two-standards-of-proof issue as Step 3: the Haiku definition is
appended with nothing marking it as LLM-asserted rather than DB-confirmed, so downstream
cannot tell them apart.
Potential solution — carry a flag on each candidate recording whether it came from the DB
or from Haiku, so the standard of proof survives into scoring and review.

---

## Phases 1.5 and 1.5b — blog paths (deferred)

In cascade order, two blog-based phases sit between the signature solver and the paid AI
stage: Phase 1.5 parses Times (TFTT) blog explanations with Haiku, and Phase 1.5b does the
same for fifteensquared (Guardian/Independent) blogs. These are deliberately left
undocumented for now, to be returned to later. The cascade description below picks up at
Phase 2.

## Phase 2 — the paid AI stage (run.py)

Factual summary:
Phase 2 is the paid stage. It takes every clue that no earlier stage solved and, for each,
makes one call to an AI model — Claude Sonnet (claude-sonnet-4-6) — asking it to explain
how the wordplay makes the already-known answer. There is no free-form fallback: if this
one call does not produce a working explanation, the clue is set aside as an unsolved
leftover for manual attention, and nothing further is tried on it.

The call is constrained — a "pick from a menu" rather than "solve it yourself". Before
calling, the pipeline works out from the database the handful of roles each wordplay word
could play: which of its synonyms or abbreviations actually appear in the answer, whether
its first or last letter or its raw letters appear, a possible reversal, and whether it is
a known indicator or link word. It hands the model that menu and asks it to pick, for each
word, which role it plays and how the pieces assemble. If the database offers no usable
role for any word, the pipeline returns without calling the model at all.

The only correctness check is letter assembly: do the chosen pieces' letters join up to
the known answer (optionally dropping link and indicator pieces)? If yes, the solve is
accepted; if not, rejected.

How it is scored: the confidence is fixed, not measured. Every accepted Phase 2 solve is
stored at "medium", score 65, regardless of quality. The explanation verifier does run on
this path, but only afterwards, to harvest enrichment gaps (missing synonyms,
abbreviations, definitions) and queue them for review — it does not score or gate the
solve.

OBSERVATION (user, 2026-05-30): in practice almost nothing is spent on this stage. The
call itself is mid-tier Sonnet capped at 400 output tokens, so each call is modest; but
the low total is driven by volume — the free phases claim most clues first, the stage
skips the API entirely when the database offers no roles, and during the redesign the
nightly is in scrape-only mode so the solving pipeline is barely run.

DESIGN CONSIDERATION (scoped to the paid AI stage):
Strength to record: constraining the model to database-listed roles blocks invented
synonyms — the model can only assemble from values that genuinely exist in the DB and
appear in the answer.
Problem 1 — fixed confidence. Every solve is stored at medium/65 by fiat; the stored
number says nothing about the parse's quality, and because 65 is medium these never reach
HIGH.
Problem 2 — the only gate is letter-assembly against a known answer. Since the answer is
supplied and the pieces are drawn from DB values already in the answer, a plausible-but-
wrong decomposition that happens to join up is accepted; nothing checks the chosen role is
the right one.
Problem 3 — prompt truncation. Only the indicator rule tells the model to capture the full
multi-word phrase; the pieces carry no such instruction, so multi-word synonym pieces get
cut to a single word (the 31251 audit's "Have Spellbound" collapsing to "Spellbound").
Problem 4 — a coverage limit that is the flip side of the strength: if the DB lacks the
needed synonym, the role is never offered and the model cannot use it.
Problem 5 — the caching path is bypassed in practice. The stage can re-run the assembler on
a prior solve's stored AI output instead of re-calling the model, but the force flag
defaults on (see the run.py clue-selection section), so the cache is skipped and every
reached clue re-calls the model on every run.
Potential solution — score the parse through the verifier instead of stamping 65; gate on
more than letter-assembly by checking each chosen role against the DB; instruct the pieces
to capture full multi-word phrases; treat missing roles as enrichment gaps; and let the
dashboard's force setting actually control re-calling.

## Phase 3 — the enrichment re-solve (run.py lines 1125-1233)

Factual summary:
Phase 3 is a feedback loop. It takes what the paid stage (and the blog stages) discovered
while solving — new synonyms, abbreviations, definitions, indicator words, and even new
wordplay patterns — folds them into a copy of the reference data, and then gives the free
signature engine a second go at every clue it did not already solve, now that it knows
more. If the signature engine now scores a clue HIGH, that clue is upgraded and its earlier
result is overwritten.

The steps: it gathers the gaps (missing synonyms, abbreviations, definitions) from every
non-signature result, plus any new patterns and indicator words the paid solves implied; it
makes an enriched copy of the reference data with those injected and builds an extra batch
of catalog patterns; then, only if something was actually injected, it re-runs the
signature engine — with the enriched data and extra patterns — on every clue not already
solved by the signature engine in Phase 1. For any clue that now comes back HIGH it tags it
"Signature+Enriched", replaces the earlier entry, and overwrites the stored result.

The injected knowledge lives in memory for this run only. enrich_refdb clones the reference
data and mutates only the copied dictionaries; nothing here writes those synonyms or
indicators back to the reference database. The only path to permanent learning is the
separate pending_enrichments queue, which a human reviews and approves.

DESIGN CONSIDERATION (scoped to the enrichment re-solve):
A genuine strength to record first: this recycles expensive discoveries back into the free
engine. A clue the paid stage solved at the flat medium/65 — with no real scoring — can be
re-solved here by the signature engine into a properly scored HIGH with a real explanation.
It turns paid, unscored solves into free, scored ones, a real quality and cost win.
Problem 1 — it can launder a wrong paid discovery into a signature-quality HIGH. The
synonyms and indicators being injected come from the paid stage's own output, which is
gated only by letter-assembly and can be plausible-but-wrong, with synonyms possibly
truncated. Once injected, those entries count as confirmed to the scorer (the same overlay
effect as the DBE step), so they draw no penalty. A wrong synonym the paid stage invented
can therefore be fed back in and let the signature engine "verify" a wrong parse to HIGH —
the synonym-pollution pattern, but self-generated within the run.
Problem 2 — the upgrade is destructive and changes attribution. It overwrites the stored
result and replaces the in-memory one, so the prior result is gone and the clue's recorded
solver becomes "Signature+Enriched". This is the last-writer-wins behaviour noted elsewhere
— a later phase overwriting an earlier solve.
Potential solution — mark run-injected synonyms and indicators as unverified so the scorer
treats them as unconfirmed (keeping a re-solve built on them out of HIGH until the queued
enrichment is human-approved); and preserve the prior result and solver attribution rather
than overwriting in place.
