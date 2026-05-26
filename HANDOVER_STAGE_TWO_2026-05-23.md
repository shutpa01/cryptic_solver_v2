# Handover: Stage Two Evidence Package

Date: 2026-05-23

Repo:

```text
C:\Users\shute\PycharmProjects\cryptic_solver_V2
```

This handover is for starting a new chat after the Stage One and Stage Two recovery work.

## User Process Rules

- Use plain English.
- Work one fact at a time when discussing design.
- Do not hide useful evidence just because a clue is not fully solved.
- Do not let old WFW display rows override fresh evidence.
- Do not treat conditional enrichment evidence as proven.
- Do not write database facts until the user has reviewed and accepted them.
- Do not start by coding in the new thread; first confirm the handover and the next step.

## Main Goal

The goal is a unified solver.

That means:

- keep as much of obase / the older production mechanism knowledge as possible;
- use WFW/Stage One grammar and tokenisation to improve evidence quality;
- stop having two competing sources of truth;
- show what was found and what was not found;
- let enrichment improve the DB, then re-run and verify.

The user specifically does not want another parallel solver or another display that contradicts the real evidence.

## Current Stage Definitions

### Stage One

Stage One is now defined as:

```text
surface + definition-boundary stage
```

Its job:

- atomise clue and answer;
- build tokens and spans;
- use POS for phrase boundaries;
- use DB evidence for definition-answer matches;
- output candidate definition/wordplay splits with boundary metadata;
- persist the stage-one context;
- not suggest wordplay mechanisms.

Important Stage One files:

```text
signature_solver/clue_context.py
signature_solver/stage_context_store.py
signature_solver/solver.py
signature_solver/wfw_unified_solver.py
web/routes/admin.py
signature_solver/test_clue_context.py
signature_solver/test_stage_context_store.py
```

Stage One result pinned for Daily Mail 17883 10a:

```text
clue:  Note Spanish male taken with a Mexican city
answer: TIJUANA
definition: Mexican city
wordplay: Note Spanish male taken with a
status: complete_edge_phrase
stage annotations: 0
```

Wordplay annotation is deliberately later:

```text
with_wordplay_annotations(...)
```

For 10a it gives:

```text
Note -> TI
Spanish male -> JUAN
a -> A
```

It must not give:

```text
city -> anything
Mexican city -> anything
```

### Stage Two

Stage Two is now agreed as an internal evidence package between Stage One and Stage Three.

It is not the final explanation.

It is not the public WFW display.

It is a hand-off object saying:

- what was found;
- what was not found;
- what fits the answer;
- what remains unaccounted for;
- what enrichments would help;
- what would work if those enrichments were accepted.

Stage Two must not:

- write to the DB;
- publish a final proof;
- mark conditional evidence as proven;
- hide partial evidence when the clue needs review.

Implemented Stage Two files:

```text
signature_solver/stage_two_casefile.py
signature_solver/test_stage_two_casefile.py
web/routes/admin.py
web/routes/clue.py
web/templates/clue.html
web/templates/admin_stage_two_casefile.html
web/templates/partials/stage_two_casefile.html
```

The main builder is:

```text
build_stage_two_casefile(clue_text, answer, db, stage_one_context=None)
```

It returns a `StageTwoCaseFile`.

The diagnostic page is:

```text
/admin/stage-two/<clue_id>?admin=dev-admin-key
```

Example for Daily Mail 17883 13a:

```text
http://127.0.0.1:5001/admin/stage-two/10068260?admin=dev-admin-key
```

This page exists only to inspect the internal Stage Two package.

## Current Dirty Worktree Facts

The Stage Two work is currently uncommitted.

Relevant modified/untracked files:

```text
M  web/routes/admin.py
M  web/routes/clue.py
M  web/templates/clue.html
?? signature_solver/stage_two_casefile.py
?? signature_solver/test_stage_two_casefile.py
?? web/templates/admin_stage_two_casefile.html
?? web/templates/partials/stage_two_casefile.html
```

There are other dirty files in the repo from earlier sessions. Do not revert anything unless the user explicitly asks.

## Tests Run

These passed after the Stage Two implementation:

```powershell
.\.venv\Scripts\python.exe signature_solver\test_stage_two_casefile.py
.\.venv\Scripts\python.exe signature_solver\test_clue_context.py
.\.venv\Scripts\python.exe signature_solver\test_stage_context_store.py
.\.venv\Scripts\python.exe signature_solver\test_wfw_unified_solver.py
```

These compile checks passed:

```powershell
.\.venv\Scripts\python.exe -m compileall web\routes\admin.py web\routes\clue.py signature_solver\stage_two_casefile.py
```

The Flask render check needed the `AI_Solver` venv because the project `.venv` does not have Flask installed:

```text
C:\Users\shute\PycharmProjects\AI_Solver\.venv\Scripts\python.exe
```

Render checks passed for Stage Two-only admin pages for clue ids:

```text
10068257
10068258
10068259
10068260
```

The check confirmed:

```text
contains "Stage Two evidence": true
contains "WFW needs review": false
```

## Daily Mail 17883 Case Studies

These examples drove the Stage Two contract.

### 10a

```text
clue: Note Spanish male taken with a Mexican city
answer: TIJUANA
```

Stage Two should report:

```text
definition found: Mexican city -> TIJUANA
pieces found: Note -> TI, Spanish male -> JUAN, a -> A
assembly: TI + JUAN + A = TIJUANA
not accounted for: taken
status: answer_fit_needs_review
```

Important user decision:

`taken` does not actually have a role in this clue. The surface and wordplay both work without it. This should be submitted for review, not treated as a total failure.

Also important:

`taken` could be a containment indicator in other clues. Unknown or unused possible indicators must not cause the solver to throw away useful evidence.

### 11a

```text
clue: Large reptile heading off fabulous bird
answer: ROC
```

Old WFW display showed a bad version:

```text
R + ROC without first letter = OC = ROC
```

Fresh evidence shows:

```text
definition found: fabulous bird -> ROC
piece found: reptile -> CROC
operation found: heading off
working pair: reptile + heading off -> ROC
not accounted for: Large
enrichment candidate: Large reptile -> CROC
status: answer_fit_needs_review
```

Key grammar point:

POS says `Large reptile` is a phrase. Since `reptile -> CROC` is useful and `Large` is left over, Stage Two should propose widening the source phrase to `Large reptile -> CROC`.

### 12a

```text
clue: Small child linked to friend in an absolute way
answer: TOTALLY
```

Old WFW display got essentially nothing useful.

Stage Two should report:

```text
definition found: in an absolute way -> TOTALLY
pieces found: Small child -> TOT, friend -> ALLY
assembly: TOT + ALLY = TOTALLY
not accounted for: linked
status: answer_fit_needs_review
```

Important:

This proves that the system is better than the current WFW display conveys.

### 13a

```text
clue: One no longer relevant base sadly in western half of north London suburb
answer: HASBEEN
```

Old WFW display was preposterous:

```text
HA + SBE + E + N = HASBEEN
```

Likely real parse:

```text
definition: One no longer relevant
base sadly: anagram of BASE -> ASBE
north London suburb: HENDON
western half of HENDON: HEN
assembly: H(ASBE)EN = HASBEEN
```

Stage Two currently reports:

```text
working pair: base + sadly -> ASBE
enrichment candidate: One no longer relevant -> HASBEEN
enrichment candidate: north London suburb -> HENDON
conditional assembly: H(ASBE)EN = HASBEEN
status: conditional_needs_enrichment
```

Important user decision:

`base/sadly` should be paired using indicator proximity. The system should leverage obase-style mechanism knowledge here rather than rediscovering basic cryptic operations.

## Why Stage Two Exists

The current clue page can show stale WFW proof attempts that contradict fresh solver evidence.

That is the "two sources of truth" problem.

Stage Two is a step toward one evidence pipeline:

```text
Stage One -> Stage Two evidence package -> Stage Three verifier/proof -> display
```

The Stage Two diagnostic page is just a window into that package.

## Outstanding Stage Three Work

Stage Three has not been implemented in this recovery thread.

The agreed next job is:

```text
Stage Three consumes the Stage Two package and decides what can honestly be verified.
```

Stage Three must:

- accept Stage Two evidence as input;
- use obase/production mechanisms where available;
- verify complete answer construction;
- distinguish proven facts from conditional enrichments;
- return "review needed" when surface words remain unaccounted for;
- never promote a conditional assembly to proven until enrichment is accepted and the clue is rerun;
- preserve useful partial evidence in the output.

Stage Three should not:

- rerun a separate WFW truth path;
- trust stale `wfw_proof_attempts` over current Stage Two evidence;
- publish a polished proof when there are unresolved words or missing DB facts;
- silently fail when it has partial evidence.

## Outstanding Enrichment Actions

This is the most important unfinished area.

The desired workflow is:

```text
run pipeline -> inspect Stage Two/Stage Three evidence -> process enrichments -> reverify -> then inspect final clue page
```

The current Stage Two implementation only creates enrichment candidates in memory.

It does not queue them.

It does not accept them.

It does not write to `pending_enrichments`.

It does not write to `cryptic_new.db`.

### Enrichment Action 1: Define Candidate Shape

Stage Two enrichment candidates need a stable shape before they are passed to Stage Three or review tooling.

Current candidate kinds include:

```text
source_phrase_widening
definition_gap
conditional_source_gap
```

They should be mapped to reviewable DB fact types:

```text
source_phrase_widening -> synonym or abbreviation, depending on source evidence
definition_gap -> definition
conditional_source_gap -> synonym or definition, depending on phrase role
indicator_gap -> indicator
homophone_gap -> homophone
```

Open question:

How should Stage Two express "this is probably a synonym" versus "this is probably a definition" when the same DB table can currently expose both through `RefDB.get_synonyms()`?

### Enrichment Action 2: Preserve Leftover Context

The user explicitly said what is not found is as interesting as what was found.

Example:

If 10a did not have `Spanish male -> JUAN` in the DB, the system should preserve:

```text
leftover clue words: Spanish male / taken / with
leftover answer letters: JUAN
known pieces: Note -> TI, a -> A
definition: Mexican city
```

This leftover evidence should feed enrichment review.

Current Stage Two records unresolved clue words, but it does not yet compute leftover answer-letter blocks in a general way.

Needed:

- compute answer letters not covered by found pieces;
- connect leftover clue spans to leftover answer spans;
- present them as possible enrichment candidates;
- keep link/surface words separate from true unknown source phrases.

### Enrichment Action 3: Grammar-Guided Phrase Widening

Stage Two already proposes:

```text
Large reptile -> CROC
```

because:

```text
reptile -> CROC
Large is unresolved
POS says Large reptile is a phrase
```

This needs to become a general rule:

- if a found source is the root or head of a POS phrase;
- and adjacent words in the phrase remain unresolved;
- and the same value would help explain the answer;
- propose the wider phrase as an enrichment candidate.

Guardrails:

- do not propose one-letter widenings unless there is strong reason;
- do not propose a phrase if the exact phrase/value is already in DB;
- do not widen across definition boundaries;
- do not widen into a known link word unless grammar strongly supports it.

### Enrichment Action 4: Indicator Proximity

The user specifically highlighted obase indicator proximity rules.

Example 13a:

```text
base sadly -> ASBE
```

`sadly` should attach to the nearby source `base`.

Needed:

- use existing obase mechanism knowledge for indicator/source pairing;
- do not require a perfect WFW proof before recording the useful pair;
- include paired evidence in the Stage Two package;
- pass paired evidence to Stage Three.

This should apply beyond anagrams:

```text
heading off + reptile -> ROC
western half + HENDON -> HEN
back/reversed indicators
hidden indicators
container indicators
homophone indicators
deletion indicators
positional indicators
```

### Enrichment Action 5: Conditional Enrichment To Stage Three

Key decision already agreed:

Stage Two can pass conditional enrichments to Stage Three.

Stage Three may say:

```text
This would solve if these enrichments are accepted.
```

Stage Three must not say:

```text
This is solved.
```

until:

```text
enrichments accepted -> DB updated -> solver rerun -> proof verified
```

For 13a this means:

```text
IF One no longer relevant -> HASBEEN
AND north London suburb -> HENDON
THEN H(ASBE)EN = HASBEEN
```

The conditional assembly is useful, but it is not proof.

### Enrichment Action 6: Review UI / Queue Integration

There are currently several enrichment surfaces:

```text
Flask clue-page Accept buttons
Streamlit dashboard review page
pending_enrichments table
sonnet_pipeline/review_gaps.py
enrichment/apply_candidates.py
```

Existing known issue from `CODEBASE_GUIDE.md`:

```text
pending_enrichments not processable in bulk:
critical missing feature; no dashboard "Process pending enrichments" button exists.
```

Needed:

- decide whether Stage Two candidates are only displayed, or also queued;
- if queued, use `pending_enrichments` rather than direct DB writes;
- preserve clue id, source, puzzle number, clue text, answer, stage, candidate kind, and conditional status;
- support accept/reject/edit;
- after accept, re-run the clue or puzzle with the updated DB.

Important:

Do not bypass human review by writing Stage Two candidates directly into the reference DB.

### Enrichment Action 7: Reverify After Enrichment

The user process is explicit:

```text
run pipeline
process enrichments
reverify
then inspect clue
```

Needed:

- after enrichment acceptance, invalidate any stale WFW/proof display for that clue;
- rerun Stage One -> Stage Two -> Stage Three;
- only then update final display/proof state;
- make stale proof attempts visibly stale or stop using them as display truth.

This directly addresses the 11a/13a problem where old WFW rows showed nonsense after fresher evidence existed.

### Enrichment Action 8: Avoid Bad Enrichment

The old enrichment system can be too eager.

Guardrails needed:

- reject candidates already in DB;
- reject candidates in `rejected_enrichments`;
- avoid full-answer-as-source unless it is a definition;
- avoid broad awkward noun phrases unless supported by answer evidence;
- preserve "review" status for surface-only leftovers like `taken`;
- do not infer a mechanism just because a word is near the right letters.

## Stage Two Implementation Notes

The current Stage Two file is intentionally small and read-only.

It currently includes:

```text
StageTwoCaseFile
build_stage_two_casefile(...)
definition candidates
grammar phrases
source candidates
operation candidates
working pairs
assemblies
enrichment candidates
unresolved words
status
```

The current implementation has some example-specific logic:

```text
_conditional_suburb_enrichments(...)
```

That exists to pin the 13a design decision.

It should eventually become a general "conditional assembly" mechanism using normal source/operation rules.

Do not treat the hard-coded HENDON path as the final architecture.

## Important Display Correction

The normal clue page still shows the old WFW display if `clue.atomic_wfw` exists.

That old panel can still show the preposterous 13a parse.

The Stage Two-only page was added because the user was correctly annoyed at being sent to the normal clue page.

Use this diagnostic page for Stage Two inspection:

```text
/admin/stage-two/<clue_id>?admin=dev-admin-key
```

Do not ask the user to inspect Python.

Do not ask the user to inspect the normal clue page when discussing Stage Two evidence.

## Current Local Inspection Server

During the previous thread, a separate Flask inspection server was started on:

```text
http://127.0.0.1:5001
```

It was started with the `AI_Solver` venv because that environment has Flask.

Do not assume it is still running in the new thread.

Check before using:

```powershell
Get-NetTCPConnection -LocalPort 5001 -State Listen -ErrorAction SilentlyContinue
```

If needed, start the app using the environment that has Flask:

```powershell
& 'C:\Users\shute\PycharmProjects\AI_Solver\.venv\Scripts\python.exe' -c "import sys; sys.path.insert(0, r'C:\Users\shute\PycharmProjects\cryptic_solver_V2'); from web import create_app; app=create_app('development'); app.run(debug=False, port=5001, host='127.0.0.1', threaded=True)"
```

Avoid disturbing anything already running on port 5000.

## Suggested Next Chat Opening

The next assistant should start by saying something like:

```text
I have read the Stage Two handover. I understand Stage Two is an internal evidence package, not the final explanation. The next job is to design Stage Three so it consumes that package, preserves conditional enrichment honestly, and does not trust stale WFW display rows.
```

Then proceed one fact at a time.

## Suggested Next Technical Step

Do not start by coding.

First agree the Stage Three contract in plain English.

Proposed Stage Three contract:

```text
Stage Three consumes Stage Two evidence and returns one of:

1. proven explanation;
2. review-needed explanation with unresolved surface words;
3. conditional explanation requiring accepted enrichments;
4. evidence-only failure that still lists what was found.
```

Only after that contract is agreed should implementation begin.

