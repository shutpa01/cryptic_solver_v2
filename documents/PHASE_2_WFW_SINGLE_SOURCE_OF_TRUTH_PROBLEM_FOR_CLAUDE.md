# Phase 2 WFW Single Source Of Truth Problem Statement For Claude

## Request

Please propose the correct architectural fix. Do not just patch the visible messages.

The current app has allowed multiple proof/scoring/display authorities to coexist for a clue. This is producing contradictory states and making fixes appear to work in one layer while another layer still shows stale or incompatible evidence.

We need a single current source of truth for WFW status, WFW display, review messages, and score.

## Current Tables / Stores Involved

The current code uses at least these independent stores:

- `structured_explanations`
  - drives clue score/tier in several views through `confidence` / `model_version`
  - can say `manual_structured_parse`, `manual_structured_parse_needs_db`, `signature_solver_v1`, etc.

- `wfw_proof_attempts`
  - stores proof JSON and `status`
  - clue page currently reads the latest row through `get_latest_wfw_proof_attempt`
  - latest row may be an automatic Stage Three proof, manual-role proof, or older proof attempt

- `manual_structured_parses`
  - stores the human structured parse JSON
  - currently loaded on the clue page and merged into the WFW display
  - does not currently replace the WFW proof source of truth

- `clue_word_roles`
  - legacy/manual word role overrides
  - Stage Three manual-role proof can use these, but precedence is inconsistent

This has created a split-brain system.

## Concrete Failure 1: WRAPPER

Clue:

```text
Telegraph Cryptic #31250, 5 Across
Cover western hip-hopper (7)
Answer: WRAPPER
clue_id: 10069641
```

Stored manual structured parse:

```text
definition: Cover -> WRAPPER
piece1: western -> W, relationship abbreviation, answer box 1
piece2: hip-hopper -> RAPPER, relationship synonym, answer boxes 2-7
```

The manual parse is structurally correct. After the direct-DB-write slice, its DB audit is clean:

```text
missing: []
queued: 0
```

But the page still shows WFW review messages from an older/wrong proof:

```text
WRAPPER = WRAPPER
one or more charade parts lack a stable clue span
one or more assembly parts lack a stable clue span
These clue words need a clear purpose before publication: western, hip-hopper
```

Latest `wfw_proof_attempts` row for this clue still contains:

```text
source_text: Cover
source_value: WRAPPER
source_span: null
```

So the WFW panel is evaluating the old automatic parse `Cover -> WRAPPER`, while the visible manual structured parse says `Cover` is the definition and `western hip-hopper` supplies the wordplay.

That is the central contradiction. The old proof is not just an awkward message; it is the wrong current authority.

## Concrete Failure 2: Manual Word Role Not Authoritative

Clue:

```text
Telegraph Cryptic #31250, 15 Down
Caught game bird flying around university city
Answer: CAMBRIDGE
clue_id: 10069665
```

Manual word roles include:

```text
Caught -> anagram_fodder
game -> anagram_fodder
bird -> anagram_fodder
flying -> anagram_indicator
around -> anagram_indicator
university -> definition
city -> definition
```

But the Stage Three proof reports:

```text
around
purpose: definition_separator_candidate
status: candidate
needed_evidence: grammar evidence showing this word separates definition from wordplay
```

So a manual role set by the user is not fully authoritative in the purpose layer. Automatic/candidate purpose inference can still survive for the same word and cause review status.

This is a related source-of-truth problem: manual evidence exists but is not consistently the authority used by proof/status.

## Concrete Failure 3: Score And WFW Panel Contradict

Observed clue page state:

```text
page score: HIGH
WFW status panel: Review
messages:
  no complete assembly found
  source candidates exist but were not accepted as a complete assembly
  no assembly available for order verification
  words still need WFW role
```

The screenshot example includes answer `DESCARTES`, with WFW review text mentioning:

```text
in -> T [unknown]
Mercedes -> CAR [unknown]
```

and unresolved display blocks for:

```text
Mercedes
car
test
```

A clue should not be simultaneously displayed as HIGH while the current WFW panel says there is no complete accepted assembly. This means score/tier and WFW status are being computed from different authorities.

## Why Display Merging Is Not Enough

The current clue page does this approximate sequence in `web/routes/clue.py`:

1. Read latest `wfw_proof_attempts`.
2. Convert it into `wfw_display`.
3. If admin and manual structured parse exists, merge manual structured parse into the display.

That creates a hybrid display:

- blocks may come from the manual parse
- review messages may come from the old WFW proof
- score may come from `structured_explanations`

This is the wrong model. Merging manual parse blocks into an old proof display does not make the old proof valid, and it can create exactly the contradictory states above.

## Desired Invariant

For each clue, there must be exactly one current WFW authority.

The page score, WFW panel status, WFW display blocks, answer links, review messages, and puzzle-list status should all be derived from that same current authority.

Possible authority order:

1. If a valid manual structured parse exists, it is the current authority.
2. Else if a valid/current manual-role proof exists, it is the current authority.
3. Else use the latest automatic Stage Three proof.
4. If no proof exists, show missing proof state.

But please decide the cleanest design. The important requirement is that stale automatic proof rows must not remain the current authority after a valid manual structured parse has replaced the parse.

## Question For Claude

What is the best fix?

Please decide whether the right approach is:

### Option A: Replace Current Proof Rows

Treat `wfw_proof_attempts` as the current proof table for now, not as history.

When a valid manual structured parse is saved:

- validate parse
- write/audit DB facts
- delete old `wfw_proof_attempts` rows for that clue
- write one new manual-structured proof row as the current proof
- update `structured_explanations` from that same manual proof
- clue page reads that one current proof

When automatic re-run/reverify writes a new proof:

- delete old proof rows for that clue first, unless explicitly preserving history somewhere else
- write one current proof row

### Option B: Keep History But Add Explicit Current Selection

Keep `wfw_proof_attempts` as history but add a current/superseded mechanism, e.g.

- `is_current`
- `superseded_at`
- `superseded_by`
- or a separate `current_wfw_proofs` table

All page/model code must read only the current proof, never just latest historical row.

### Option C: Separate Current Authority Helper

Do not alter table shape yet. Add one helper that always chooses the current authority:

```python
get_current_wfw_authority(clue_id)
```

Everything uses it:

- clue page
- puzzle page
- score/tier display
- reverify/rerun decisions

This helper must prefer valid manual structured parse over stale proof rows.

## Constraints

- Do not solve this by hiding review messages.
- Do not merely merge manual parse display blocks into stale proof displays.
- Do not make clue-specific patches.
- Do not weaken WFW validation to make contradictions pass.
- Do not leave score and WFW status reading different authorities.
- Avoid a broad rewrite if a smaller source-of-truth change can enforce the invariant.

## Acceptance Tests Required

Please include tests for these cases in the proposed solution.

### Test 1: WRAPPER Manual Structured Parse Supersedes Old Proof

Setup:

- clue `Cover western hip-hopper`, answer `WRAPPER`
- old WFW proof says `Cover -> WRAPPER` with `source_span: null`
- manual structured parse says `Cover` definition, `western -> W`, `hip-hopper -> RAPPER`
- DB facts exist or are directly written

Expected:

- current authority is manual structured parse
- WFW status is proven/high
- WFW display does not show old `Cover -> WRAPPER` source block
- no review messages about `western` or `hip-hopper` lacking purpose
- score is HIGH and derives from same authority

### Test 2: Review Cannot Display As HIGH

Setup:

- current WFW authority says review/no complete assembly

Expected:

- page/puzzle score is not HIGH
- WFW status and score agree

### Test 3: Invalid Manual Parse Does Not Hide Automatic Review

Setup:

- manual structured parse exists but fails structural validation or DB audit after direct write

Expected:

- current authority is review
- errors are from the manual parse validation/audit
- no stale automatic proof is mixed into the manual parse display

### Test 4: Manual Word Role Precedence

Setup:

- clue `Caught game bird flying around university city`
- manual role marks `around` as `anagram_indicator`

Expected:

- `around` is not also emitted as `definition_separator_candidate`
- candidate purpose requests do not survive for a word with an authoritative manual role

## Deliverable Requested From Claude

Please produce a Codex implementation instruction with:

- selected design option
- exact files/functions to change
- exact data flow after the change
- migration/table impact if any
- tests to add/update
- explicit anti-patterns to avoid

