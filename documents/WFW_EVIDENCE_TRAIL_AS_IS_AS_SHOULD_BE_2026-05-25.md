# WFW Evidence Trail: As It Is and As It Should Be

Date: 2026-05-25

Scope: this document describes the current WFW evidence trail, the points where evidence is lost or misrepresented, and the target contract that should replace it. It is written for sharing with Claude/Codex before further implementation.

This document is not a code change. It does not claim the WFW system is fixed. It records what the current code and stored evidence show.

## Executive Summary

The WFW system currently has the pieces of a useful evidence trail, but they are not carried through one consistent proof object.

The core problem is this:

1. Stage Two can find source candidates, operation candidates, definition candidates, working pairs, and sometimes assemblies.
2. Stage Three currently promotes only selected evidence into the proof blocks.
3. The WFW display renders Stage Three proof blocks, plus some missing-word fallback and manual-role overlay.
4. Therefore, real candidate evidence can exist but disappear from the WFW page if it was not part of a selected complete assembly.

That is why the page can say, or appear to say, that nothing useful was found when the database and Stage Two actually found useful pieces.

The most important example is ENDEARED. Fresh Stage Two evidence finds candidate pieces such as:

- `Aim -> END`
- `limit -> END`
- `expensive -> DEAR`
- `energy -> E`
- `made more attractive -> ENDEARED`

But the stored Stage Three proof for ENDEARED only displays the accepted definition and an operation block for `Aim`; the candidate source pieces are not carried into proof blocks because no complete assembly was selected. That is an evidence-trail failure, not just a visual bug.

The target state should be: one proof object records all relevant evidence, including accepted evidence, rejected evidence, candidate evidence, failed partial assemblies, manual roles, proof checks, and display blocks derived from that same object. The display should not need to reconstruct evidence from side channels.

## Current Evidence Trail: As It Is

### 1. Inputs

The WFW trail starts from:

- clue text
- answer
- clue id / puzzle id
- stored lexical facts, including synonyms, abbreviations, wordplay facts, definitions, and augmented definitions
- manual word-role rows in the admin UI

Manual roles are currently stored separately from the solver evidence and then reintroduced during reverify/rerun flows.

### 2. Signature Clue Pipeline

Relevant files/functions:

- `sonnet_pipeline/clue_pipeline.py`
  - `run_signature_clue_pipeline`
  - `run_clue_pipeline`
- `sonnet_pipeline/sig_adapter.py`
  - `store_signature_evidence`
- `signature_solver/solver.py`
  - `solve_clue`

The pipeline produces a solve result. During `solve_clue`, Stage Two and Stage Three objects can be built and attached to the solve result.

Important current fact: this pipeline can generate a Stage Two casefile with useful evidence even when Stage Three later fails to produce a complete proof.

### 3. Stage Two Casefile

Relevant file/functions:

- `signature_solver/stage_two_casefile.py`
  - `build_stage_two_casefile`
  - `build_stage_two_from_solve_result`
  - `_source_candidates`
  - `_assemblies`
  - `_unresolved_words`

Stage Two is where the solver gathers candidate evidence.

Observed current behavior:

- `_source_candidates` collects source-like candidates from annotations.
- It filters candidates by whether the candidate value is compatible with the answer.
- `_assemblies` tries to combine source terms and working-pair outputs into a full charade assembly.
- `_first_charade` only returns a successful assembly when the terms concatenate exactly to the answer in order and at least two parts are present.

Current limitation:

Failed partial assemblies are not preserved as first-class evidence. If useful pieces exist but cannot be assembled into the answer, Stage Two can contain candidates while Stage Three has no complete assembly to prove.

### 4. Stage Three Proof

Relevant file/functions:

- `signature_solver/stage_three_proof.py`
  - `build_stage_three_proof`
  - `_blocks`
  - `_definition_check`
  - `_accepted_definition_candidate`
  - `_word_purposes`
  - `_purpose_requests`

Stage Three builds the WFW proof object and proof checks.

Observed current behavior:

- Definition candidates can become `DEF_BLOCK`s.
- Source blocks are normally produced from selected assembly parts.
- Operation blocks can be produced from operation evidence.
- Unresolved words can become review/missing blocks.
- Manual roles can now be read from the casefile and used by `_word_purposes`.
- Manual definition candidates can now be passed into `_definition_check`.

Current limitation:

Unused source candidates are not automatically preserved in the proof blocks. If Stage Two has source candidates but no selected assembly, those candidates can be absent from Stage Three display evidence.

This is the central evidence-loss point.

### 5. Persistence

Relevant file/functions:

- `signature_solver/atomic_parse_store.py`
  - `upsert_solve_result_pipeline_state`
- `web/routes/admin.py`
  - `reverify_clue`
  - `_rerun_clue_inner`
  - `_write_manual_role_stage_three_for_clue`
  - `atomic_reverify_puzzle`

The proof can be written into `wfw_proof_attempts`. The pipeline state can also be written into `clue_pipeline_state`.

Observed current behavior:

- If `solve_result.stage_three_proof` exists, `upsert_solve_result_pipeline_state` can write a Stage Three proof attempt.
- Admin flows now include additional helper logic to rebuild manual-role-aware Stage Three proofs.
- The helper has been observed to write successful manual-role-aware rows for MAUI.

Current limitation:

There have historically been multiple write flows for WFW proof attempts. A correct system must guarantee that every user-visible action that claims to rerun/reverify WFW writes the same proof contract.

Actions that must share one contract:

- clue `Re-run`
- clue `Re-verify`
- puzzle `Re-verify`
- batch/import pipeline reruns
- any future background WFW proof generation

### 6. Display

Relevant files/functions:

- `signature_solver/wfw_display_adapter.py`
  - `display_from_stage_three_proof`
  - `_stage_three_missing_word_blocks`
- `web/routes/clue.py`
  - `clue_page`
  - `_apply_manual_roles_to_wfw_display`

Observed current behavior:

- The display adapter renders Stage Three proof blocks.
- It now has a fallback for missing words from Stage Three word purposes.
- The clue page applies manual roles visually after the proof display is built.

Current limitation:

The display is downstream of the proof object. If the proof object does not include candidate source evidence, the display cannot show it honestly without reconstructing from some separate source. The correct fix is to make Stage Three preserve the candidate evidence, then let the display render it.

Manual-role overlay should not be the main proof mechanism. It can help the UI, but the proof object itself must be complete.

## Concrete Stored Examples

### A. UNDEMOCRATIC: The Good Path

Clue id observed: `10069315`

Stored proof status observed:

- `wfw_proven`
- source: `stage_three_manual_roles`
- schema: `stage_three_proof:v1`

Checks observed as PASS:

- definition evidence
- answer assembly
- word purpose coverage

Observed blocks:

- `contrary to a constitution?` as definition
- `courted` as source, value `COURTED`
- `man` as source, value `MAN`
- `in charge` as source, value `IC`
- `Rogue` as operation indicator

This is the path that currently works. Stage Two had enough evidence for a complete assembly, Stage Three selected that assembly, the source blocks were preserved, and the display could show them.

This is the model the rest of WFW should follow, except that unsuccessful candidate evidence should also be preserved.

### B. MAUI: Manual Roles Can Now Feed Proof

Clue id observed: `10069319`

Observed final state after manual-role-aware rewrites:

- `wfw_proven`
- source: `stage_three_manual_roles`
- answer assembly: `MA + U + I = MAUI`
- word purpose coverage: PASS

Observed blocks:

- `Graduate -> MA`
- `uniform -> U`
- `island, -> I`

Observed word purposes:

- `Graduate`: answer source, verified
- `with`: structural separator, manual
- `uniform`: answer source, verified
- `by`: structural separator, manual
- `island,`: answer source, verified
- `one in Hawaii`: definition phrase members, manual

Earlier observed state before the definition DB fact was available:

- word purpose coverage could pass
- answer assembly could pass
- definition evidence remained REVIEW with a precise gap message:
  - `definition gap: one in Hawaii -> MAUI not in DB`

Meaning:

Manual roles can now be carried into the proof flow for this case. That does not prove the wider WFW system is fixed. It proves that MAUI can be made coherent when manual roles and definition evidence are present and the relevant admin action writes the manual-role-aware proof.

### C. ENDEARED: Candidate Evidence Exists But Is Not Preserved

Clue id observed: `10069320`

Clue:

`Aim to limit expensive energy is made more attractive`

Answer:

`ENDEARED`

Stored proof status observed:

- `wfw_review`
- source: `stage_three_manual_roles`

Observed passing check:

- definition evidence:
  - `made more attractive -> ENDEARED`

Observed failing checks:

- answer assembly
- source evidence
- assembly order
- atomic coverage
- span integrity
- word purpose coverage

Observed stored proof blocks:

- definition block:
  - `made more attractive`
- operation block:
  - `Aim`, token `POS_I_FIRST`

Observed word purposes:

- `Aim`: operation indicator, verified
- `to`: unresolved
- `limit`: unresolved
- `expensive`: answer source, manual
- `energy`: answer source, manual
- `is`: structural separator, manual
- `made more attractive`: definition, verified

Observed database / fresh Stage Two evidence:

- `aim -> END`
- `limit -> END`
- `expensive -> DEAR`
- `energy -> E`
- `made more attractive -> ENDEARED`

Fresh Stage Two found source candidates:

- `Aim`, `SYN_F`, `END`
- `limit`, `SYN_F`, `END`
- `expensive`, `SYN_F`, `DEAR`
- `energy`, `ABR_F`, `E`
- `is`, `SYN_F`, `DE`

Fresh Stage Two found operation candidates:

- `to`, link
- `energy`, anagram indicator candidate
- `is`, link

Fresh Stage Two found no complete assembly.

This means ENDEARED is not simply "nothing found." The system has useful candidate evidence, but the current proof object does not preserve those candidates when no complete assembly is selected.

The honest current message should be closer to:

`No complete verified answer assembly found. Candidate source evidence exists but was not assembled: Aim -> END, limit -> END, expensive -> DEAR, energy -> E.`

The current display should not imply those pieces do not exist.

### D. LEGSPIN: Evidence Gap And Solver Gap

Clue id observed: `10069316`

Stored proof status observed:

- `wfw_review`

Observed failing checks:

- definition evidence
- answer assembly
- source evidence
- assembly order
- atomic coverage
- span integrity
- word purpose coverage

Observed proof/display pattern:

- Several operation candidates appear.
- Several words remain unresolved.
- No complete assembly is available.

Observed no-write pipeline evidence:

- operation candidates included `Retired`, `embracing`, `special`, `of`
- source candidates included at least `special -> IN` / `SP` style evidence
- no complete assembly was found

This is not merely a display failure. It is a real solver/evidence gap as well. The target proof should still preserve all found candidate evidence and explain why no assembly passed.

### E. NASCENT: Definition Found, Assembly Missing

Clue id observed: `10069317`

Stored proof status observed:

- `wfw_review`

Observed:

- definition evidence passed
- answer assembly failed
- proof blocks contained the definition
- non-definition clue words remained unresolved

This is another case where the system should preserve all candidate evidence and unresolved words in one proof object, rather than letting the UI look like a thin or misleading partial view.

## Failure Points

### 1. Candidate Source Evidence Is Not A First-Class Proof Output

As-is:

- Stage Two may find source candidates.
- Stage Three only displays source evidence when it participates in a selected assembly or working pair.
- Candidate source evidence that did not assemble can disappear.

As it should be:

- Every answer-compatible source candidate should be represented in the proof object with a status.
- Possible statuses:
  - `accepted`
  - `candidate`
  - `rejected`
  - `manual`
  - `stale`
  - `conflict`

The display can then show accepted evidence strongly and candidate evidence differently, without pretending it is proven.

### 2. Failed Partial Assemblies Are Not Preserved

As-is:

- `_first_charade` returns a successful charade or nothing useful for the proof.
- Failed combinations are not recorded as explanatory evidence.

As it should be:

- Stage Two or Stage Three should record attempted/partial assemblies.
- Each rejected assembly should have a reason:
  - wrong order
  - duplicate source
  - missing letters
  - extra letters
  - candidate value not answer-compatible
  - operation requirement unmet
  - span/order conflict

For ENDEARED, the proof should be able to say exactly why the available pieces did not produce `ENDEARED`.

### 3. Review Messages Are Technically Accurate But User-Misleading

As-is:

`no complete assembly source list to verify` can be true while source candidates exist.

The user-facing reading is: "the solver found no source evidence." That is false in cases like ENDEARED.

As it should be:

Messages must distinguish:

- no source candidates found
- source candidates found but no complete assembly
- complete assembly found but source evidence unverified
- complete assembly found but order/spans failed

### 4. Manual Roles Are Still Partly A Side Channel

As-is:

- Manual roles can now influence Stage Three in some admin flows.
- The clue display also applies manual roles as a visual overlay.

As it should be:

- Manual roles should be input evidence to Stage Three.
- The proof object should record manual roles, their source, their timestamp if available, and how they affected word purposes/checks.
- The display should render what the proof object says, not repair the proof after the fact.

### 5. Multiple Writers Must Be Unified

As-is:

- There are several routes/actions that can regenerate proof evidence.
- Some admin flows now explicitly write manual-role-aware Stage Three proofs.

As it should be:

Every WFW-writing action must call one shared proof-generation function or must produce byte-for-byte equivalent schema objects.

Required actions:

- `Re-run`
- clue `Re-verify`
- puzzle `Re-verify`
- batch rerun
- import-time proof generation

The user should never have to know which button writes which kind of WFW proof. Buttons with the same name should differ only in scope, not behavior.

### 6. Span Space Must Be Explicit

As-is:

- There have been mismatches between atom spans, clue word spans, and Stage Three word indexes.
- Some fixes already map Stage Three tokens to word-sequential indexes.

As it should be:

Every proof object should declare its span space explicitly:

- raw clue character spans
- clue-word indexes
- atom indexes
- normalized token indexes

Display code should refuse or clearly mark evidence whose span space is missing or incompatible.

### 7. Tests Are Not Yet A Reliable Guardrail

As-is:

- Some behavior is DB-dependent.
- Helper scripts and direct function calls can pass even when a user-visible route remains broken.

As it should be:

Acceptance tests must exercise stored-row and route-level behavior, not only isolated helpers.

For WFW, a test is not enough unless it verifies:

- the correct action was called
- the proof row was written
- the latest row is the one the UI reads
- the proof object contains the expected evidence
- the display adapter renders from that proof object

## Target Evidence Trail: As It Should Be

The WFW system should have one authoritative proof object.

### Required Proof Object Sections

The proof object should include:

1. `clue_context`
   - clue id
   - clue text
   - answer
   - enumeration if available
   - normalized clue words
   - span-space declaration

2. `definition_candidates`
   - phrase
   - span
   - evidence source
   - DB hit/miss
   - accepted/rejected/manual/gap status
   - rejection or gap reason

3. `source_candidates`
   - clue phrase
   - span
   - token type, such as `SYN_F`, `ABR_F`
   - value
   - DB evidence source
   - answer compatibility
   - accepted/candidate/rejected/manual status
   - reason if not accepted

4. `operation_candidates`
   - clue phrase
   - span
   - token type, such as anagram, container, reversal, positional, link
   - status
   - reason

5. `working_pairs`
   - operation
   - input pieces
   - output
   - span
   - verification status

6. `assemblies`
   - selected assembly, if any
   - rejected/partial assemblies
   - part list
   - produced answer string
   - expected answer string
   - missing letters
   - extra letters
   - order result
   - rejection reason

7. `manual_roles`
   - word
   - span/index
   - selected role
   - source of role: manual/auto
   - whether it affected proof

8. `word_purposes`
   - word
   - role
   - status
   - evidence reference
   - whether manual, verified, candidate, or unresolved

9. `checks`
   - definition evidence
   - answer assembly
   - source evidence
   - assembly order
   - atomic coverage
   - span integrity
   - word purpose coverage
   - each with status and precise message

10. `display_blocks`
   - derived from the proof object
   - no independent reconstruction of evidence
   - includes accepted blocks, candidate blocks, manual blocks, unresolved blocks, and gaps

## Target Rendering Contract

The WFW display should render proof evidence by status.

Accepted evidence:

- strong positive styling
- used in passing checks
- has a direct reference to DB/proof evidence

Candidate evidence:

- visible but not proven
- explains what was found
- does not satisfy proof checks unless promoted by a valid assembly or accepted manual proof rule

Manual evidence:

- visible as manual
- can satisfy word-purpose coverage if allowed by the proof rules
- should not silently override verified evidence

Gaps:

- visible and precise
- should say what is missing from DB or what proof step failed

Unresolved words:

- visible only when no accepted, candidate, or manual purpose explains them

Important display rule:

If there is no complete assembly, the page should still show found pieces. It should say "no complete assembly", not make found source candidates vanish.

## Acceptance Criteria

### ENDEARED

The proof object must contain, at minimum:

- definition candidate:
  - `made more attractive -> ENDEARED`, accepted
- source candidates:
  - `Aim -> END`
  - `limit -> END`
  - `expensive -> DEAR`
  - `energy -> E`
- operation/link candidates:
  - any candidates actually found by Stage Two
- assembly result:
  - no complete accepted assembly unless the solver genuinely proves one
- rejected/partial assembly explanation:
  - precise reason no available candidate set produced `ENDEARED`

The display must show the found candidates even if the WFW status remains REVIEW.

### MAUI

When manual roles are:

- `with`: link
- `by`: link
- `one in Hawaii`: definition phrase

and the definition fact exists:

- `one in Hawaii -> MAUI`

the proof should pass:

- definition evidence
- answer assembly
- word purpose coverage

If the definition fact does not exist, the proof should not pass definition evidence. It should emit the precise gap:

`definition gap: one in Hawaii -> MAUI not in DB`

### UNDEMOCRATIC

The existing proven path must remain proven.

The proof must preserve:

- `courted -> COURTED`
- `man -> MAN`
- `in charge -> IC`
- operation indicator from `Rogue`
- definition `contrary to a constitution?`

No change should flatten or lose multi-word source evidence such as `in charge`.

### LEGSPIN

The proof should not pretend the answer is proven unless it is.

It should preserve:

- found operation candidates
- found source candidates
- unresolved words
- precise assembly failure reason

This is a solver/evidence gap case, not only a display case.

### NASCENT

The proof should preserve:

- accepted definition evidence
- all found candidates, if any
- unresolved non-definition words
- reason no answer assembly passed

The display should not collapse to only a definition block if other evidence exists.

## Implementation Plan

This section is a plan, not a claim that the work is already done.

### Step 1: Extend Stage Three Proof Schema

Add first-class arrays for:

- `source_candidates`
- `operation_candidates`
- `rejected_assemblies` or `partial_assemblies`
- `manual_roles`
- `display_blocks`, if display blocks remain stored

Do not rely on display reconstruction to recover evidence that Stage Three omitted.

### Step 2: Preserve Candidate Source Evidence

In Stage Three, carry Stage Two source candidates into the proof object even when they are not part of a selected assembly.

Each source candidate must state:

- phrase
- span
- token type
- value
- whether value is answer-compatible
- whether it was used in the selected assembly
- status and reason

### Step 3: Preserve Failed Assembly Attempts

Modify the assembly-building stage so failed or partial assemblies are recorded with reasons.

This should answer questions like:

- Did the solver have enough letters?
- Did the pieces appear in the wrong order?
- Were there duplicate/conflicting candidates?
- Was an operation required but not proven?
- Were letters missing or extra?

### Step 4: Make Checks Refer To Evidence References

Each proof check should point to the evidence it used or the evidence it found insufficient.

Example:

- answer assembly PASS references selected assembly id
- answer assembly REVIEW references partial assembly ids and failure reasons
- source evidence REVIEW distinguishes "no candidates" from "candidates not assembled"

### Step 5: Move Manual Roles Fully Into Proof Input

Manual roles should be loaded before Stage Three proof generation and recorded in the proof object.

The display should render manual roles from the proof object, not apply them as an independent correction after proof rendering.

### Step 6: Use One WFW Writer

Create or enforce one shared function for WFW proof regeneration.

All these actions must call it:

- clue `Re-run`
- clue `Re-verify`
- puzzle `Re-verify`
- batch/import generation

The difference between buttons should be scope only.

### Step 7: Update Display Adapter

The display adapter should render from the authoritative proof object.

It should show:

- accepted source blocks
- accepted definition blocks
- operation blocks
- candidate source blocks
- manual blocks
- unresolved words
- precise gaps

It should not need to infer missing evidence from DB tables independently.

### Step 8: Add Route-Level Acceptance Tests

Tests must verify the same thing a user sees.

Minimum tests:

- MAUI manual roles plus definition fact produce a proven latest proof row.
- MAUI without definition fact produces precise definition gap.
- ENDEARED latest proof contains candidate source evidence even with no complete assembly.
- ENDEARED display shows the candidate pieces and still says assembly is not proven.
- UNDEMOCRATIC remains proven.
- LEGSPIN remains review but shows found evidence and assembly failure reason.
- Puzzle reverify and clue reverify produce equivalent proof objects for the same clue.
- `Re-run` and `Re-verify` differ only in scope, not proof semantics.

## What This Document Does Not Claim

This document does not claim:

- the current code fully fixes WFW
- ENDEARED is currently solved
- every route currently writes the same proof schema
- display output is currently an authoritative representation of all Stage Two evidence
- manual-role overlay is a sufficient proof mechanism
- helper-script success proves UI success

The current known truth is narrower:

- Some manual-role-aware proof generation now works for MAUI.
- UNDEMOCRATIC is an example of the complete-assembly path working.
- ENDEARED proves that candidate evidence can exist but fail to survive into the visible proof.
- LEGSPIN shows there are still genuine solver/evidence gaps.
- The proof object must become the single source of truth before WFW can be trusted.

## Short Version To Send

The current WFW failure is not just a small bug. The evidence trail is structurally incomplete.

Stage Two can find useful candidate evidence, but Stage Three only promotes selected proof evidence into blocks. If no full assembly is selected, real candidates can vanish from the proof and therefore from the page. ENDEARED demonstrates this: the DB/Stage Two evidence includes `Aim -> END`, `limit -> END`, `expensive -> DEAR`, `energy -> E`, and `made more attractive -> ENDEARED`, but the stored proof displays only the definition and an operation block because no complete assembly was selected.

The fix should be to make Stage Three preserve all candidate evidence, rejected/partial assemblies, manual roles, checks, and display blocks in one authoritative proof object. The display should render that object, not reconstruct evidence from side channels. Every WFW action, including `Re-run`, clue `Re-verify`, puzzle `Re-verify`, and batch generation, must write the same proof contract.

Acceptance should be route-level and stored-row-level: ENDEARED must show found candidate pieces even while REVIEW, MAUI must pass or show a precise definition gap depending on DB evidence, UNDEMOCRATIC must remain proven, and LEGSPIN/NASCENT must preserve found evidence and explain the real missing proof step.
