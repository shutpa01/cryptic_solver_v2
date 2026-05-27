# Handover: Structured Parse Editor Next Thread

Date: 2026-05-27

Project:

```text
C:\Users\shute\PycharmProjects\cryptic_solver_V2
```

This handover is for the next thread so momentum is not lost. The user is
trying to turn the manual evidence work into a genuinely usable structured
crossword parse editor. This is not just a UI tidy-up. The bigger goal is to
replace the old free-form leftover explanation process with a structured parse
format that both a human admin and an LLM can produce.

## User Priority

The user wants a tool that can always correct a failed clue parse.

The current automatic solver is weak. Even after adding missing DB facts, it
often cannot assemble the correct parse. The user therefore needs a durable,
visual, structured correction path:

- specify which clue words define the answer
- specify which clue words produce answer letters
- specify operations such as container/reversal/anagram/deletion
- colour the clue blocks and answer tiles correctly
- preserve that evidence across rerun/reverify
- eventually allow an LLM to output the same structured parse format

The user does not want a manual free-form paragraph editor. They also do not
want raw internal storage concepts exposed in the UI.

## Emotional / Process Context

The previous work caused serious frustration because implementation exposed
backend concepts such as:

- node type
- edge
- payload JSON
- result label
- word numbers
- group id

That was correctly judged as inward-facing and not a human crossword interface.

Next thread must not repeat this pattern. Do not make a small technical patch
and present it as a solution. The design must be reviewed first, then
implemented faithfully.

Important working rules:

1. Do not go off independently beyond the user's request.
2. Do not implement a new direction before the user agrees.
3. Do not expose graph/storage language in the UI.
4. Do not ask the user to think in computer terms when crossword terms exist.
5. Treat browser-visible behaviour as the product, not merely backend tests.
6. Be honest about what is implemented and what is only designed.

## Current New Design Document

The key document prepared for Claude review is:

```text
documents\PHASE_2_STRUCTURED_PARSE_EDITOR_SPEC_FOR_CLAUDE.md
```

Clickable path:

[PHASE_2_STRUCTURED_PARSE_EDITOR_SPEC_FOR_CLAUDE.md](C:/Users/shute/PycharmProjects/cryptic_solver_V2/documents/PHASE_2_STRUCTURED_PARSE_EDITOR_SPEC_FOR_CLAUDE.md)

This document asks Claude to review the design, not write code.

It frames the main question as:

```text
What structured parse format should both the human admin UI and future LLM
parsing use, so that any failed clue can be corrected, validated, rendered with
colours, preserved across reruns, and eventually promoted into proof evidence?
```

The document explicitly asks Claude to be critical:

- identify clue types the structure cannot represent
- identify field names still too technical
- decide graph-only vs JSON-only vs hybrid persistence
- review the SWALLOW container example
- review the rule that pieces own colours and operations validate combinations
- recommend the smallest safe first implementation slice
- do not write implementation code yet

## Core Design Principle

The most important design rule is:

```text
Operations validate how pieces combine.
Pieces own the answer tile colours.
```

This matters especially for container clues.

Example:

```text
Bird, female, going over fence (7)
Answer: SWALLOW
```

Parse:

```text
Bird = definition
female = SOW
fence = WALL
going over = container indicator
SOW contains WALL
S + WALL + OW = SWALLOW
```

Desired structured entry:

```yaml
definition:
  clue_words: Bird
  answer: SWALLOW

pieces:
  - clue_words: female
    relationship: synonym
    letters: SOW
    answer_boxes: [1, 6, 7]
    colour: blue

  - clue_words: fence
    relationship: synonym
    letters: WALL
    answer_boxes: [2, 3, 4, 5]
    colour: pink

operations:
  - clue_words: going over
    type: container
    outer: female
    inner: fence
    result: SWALLOW
```

Expected visual result:

- `Bird` is the definition block.
- `female` is a blue source block.
- `fence` is a pink source block.
- `going over` is an operation block.
- answer box 1, `S`, is blue.
- answer boxes 2-5, `WALL`, are pink.
- answer boxes 6-7, `OW`, are blue.

The result `SWALLOW` should not flatten all answer tiles into one colour.

## Human-Facing Terms

Use crossword terms, not storage terms.

Good labels:

- definition
- clue words
- synonym
- abbreviation
- literal letters
- foreign word
- answer boxes
- letters produced
- colour
- reversal
- container
- anagram
- deletion
- homophone
- filler/link word

Bad labels in the primary UI:

- node
- edge
- payload JSON
- group id
- transform
- result label
- raw letters
- word numbers, unless unavoidable

If clue word positions must be visible, prefer:

```text
Clue word positions
```

But the better UX is clickable clue words that auto-fill the text and
positions.

## Answer Box Numbering

Human-facing answer boxes must be 1-based.

For `SWALLOW`:

```text
S W A L L O W
1 2 3 4 5 6 7
```

Storage/display internals may convert to 0-based, but the user should never
have to enter 0-based answer positions.

Validation must catch:

- box number less than 1
- box number greater than the answer length
- duplicate claimed boxes
- letters that do not match the cleaned answer at those boxes
- mismatch between letter count and box count

Example:

```text
female -> SOW, boxes 1,6,7
```

Must validate as:

- S = answer box 1
- O = answer box 6
- W = answer box 7

## Existing Implemented Backend

The manual evidence graph backend already exists and should be treated as a
usable lower layer, not as the user-facing design.

Important file:

```text
signature_solver\manual_evidence_store.py
```

Clickable path:

[manual_evidence_store.py](C:/Users/shute/PycharmProjects/cryptic_solver_V2/signature_solver/manual_evidence_store.py)

It includes:

- `manual_evidence_nodes`
- `manual_evidence_edges`
- `payload_json`
- node write/read/delete helpers
- edge write/read/delete helpers
- operation evaluators
- `merge_manual_into_display(...)`

Current node types include:

- source
- definition
- structural
- operator
- transform

Current edge types include:

- input_to
- output_of

Important current limitation:

The graph is implementation-shaped. It can support display and validation, but
it is not the correct mental model for the admin UI or future LLM parse output.

## Existing Implemented Routes

Important file:

```text
web\routes\admin.py
```

Clickable path:

[admin.py](C:/Users/shute/PycharmProjects/cryptic_solver_V2/web/routes/admin.py)

Relevant routes:

- `POST /admin/manual-evidence/<clue_id>`
- `POST /admin/manual-evidence/<clue_id>/edge`
- `POST /admin/manual-evidence/edge/<edge_id>/delete`
- `POST /admin/manual-evidence/node/<node_id>/delete`

Important file:

```text
web\routes\clue.py
```

Clickable path:

[clue.py](C:/Users/shute/PycharmProjects/cryptic_solver_V2/web/routes/clue.py)

It loads manual evidence graph rows for admins and calls
`merge_manual_into_display(...)`. Public display is still unchanged.

## Existing UI State

Important files:

```text
web\templates\clue.html
web\templates\partials\manual_evidence_nodes.html
```

Clickable paths:

[clue.html](C:/Users/shute/PycharmProjects/cryptic_solver_V2/web/templates/clue.html)

[manual_evidence_nodes.html](C:/Users/shute/PycharmProjects/cryptic_solver_V2/web/templates/partials/manual_evidence_nodes.html)

Current UI has been improved from the worst state. It now says:

- `Admin: manual parse editor`
- `Clue words give letters`
- `Clue words apply an operation`
- `Operation result goes in the answer`
- `Mark definition or filler`
- saved clue evidence
- saved connections

But it is still not good enough. The user got lost while using it.

Known remaining UI problems:

1. User still has to type clue word positions manually.
2. "same meaning" should be "synonym".
3. "Operation result goes in the answer" is confusing.
4. "Result label" is incomprehensible and should be removed or auto-filled.
5. Container entry is not specialised enough.
6. Container colouring must preserve two colours: outer and inner pieces.
7. The user should not manually wire connections for ordinary operations.

## Current Verification State

Recent checks run successfully after the UI wording patch:

```text
.venv\Scripts\python.exe -m py_compile signature_solver\manual_evidence_store.py web\routes\admin.py web\routes\clue.py
```

Template tests also passed:

- manual evidence partial renders with readable text
- old visible labels `node_type`, `edge_type`, `payload_json` absent from the
  partial
- admin clue page renders with `Admin: manual parse editor`
- old visible labels `Admin: manual evidence`, `Node type`, `Payload JSON`
  absent from the clue page
- populated connection form can render

Browser check:

- fresh Flask instance on port `5004` showed the new UI text
- old Flask on port `5003` was still serving stale template memory

Important lesson:

If the user says they restarted Flask but sees old UI, check which process and
port they are actually using. A stale Flask process can make a correct template
change appear invisible.

## Worktree Warning

The worktree is dirty and has many unrelated files.

Do not clean, reset, or delete anything casually.

Current status includes known modified/untracked items such as:

- `signature_solver/stage_three_proof.py`
- `web/routes/admin.py`
- `web/routes/clue.py`
- `web/templates/clue.html`
- `signature_solver/manual_evidence_store.py`
- `web/templates/partials/manual_evidence_nodes.html`
- many `documents/*` files
- many scraper JSON files
- `.claude/worktrees/*`
- `.codex/`

The next thread must not treat the dirty worktree as permission to revert or
clean. Only touch files needed for the agreed task.

## Recommended Next Sequence

### Step 1: Wait for Claude review

The next useful input should be Claude's review of:

```text
documents\PHASE_2_STRUCTURED_PARSE_EDITOR_SPEC_FOR_CLAUDE.md
```

Do not implement from the spec before reviewing Claude's response with care.

### Step 2: Review Claude response critically

Check:

- did Claude preserve the human/LLM shared structured parse goal?
- did Claude preserve split colouring for container clues?
- did Claude avoid raw graph UI?
- did Claude recommend a storage strategy?
- did Claude identify missing clue types?
- did Claude propose a smallest safe first slice?
- did Claude avoid weakening the design into another micro patch?

### Step 3: Agree the implementation slice with the user

Likely first implementation slice should be one of:

Option A:

- fix current UI labels and 1-based answer boxes
- remove/auto-fill result label
- keep existing graph backend

Option B:

- add a container-specific editor first
- support `SOW` around `WALL` equals `SWALLOW`
- preserve split colours from source pieces

Option C:

- add structured parse JSON storage first
- generate graph rows from JSON

Do not choose this without user agreement.

### Step 4: Implement with browser-visible acceptance tests

For any implementation, tests must include:

- Python compile
- store/helper unit smoke tests
- Flask test-client render
- browser check against a fresh Flask process
- explicit SWALLOW container test

## Must-Have Acceptance Test: SWALLOW

Use this clue as the product-level test:

```text
Bird, female, going over fence (7)
Answer: SWALLOW
```

The editor must allow:

- definition: Bird
- piece: female -> SOW, answer boxes 1,6,7, blue
- piece: fence -> WALL, answer boxes 2,3,4,5, pink
- operation: going over = container, outer=SOW, inner=WALL

Expected display:

- blue `female` source block
- pink `fence` source block
- operation block for `going over`
- definition block for `Bird`
- answer tile 1 blue
- answer tiles 2-5 pink
- answer tiles 6-7 blue
- operation validates `S + WALL + OW = SWALLOW`
- rerun does not delete the evidence

## Anti-Patterns to Avoid

Do not:

- expose node/edge/payload language in the primary UI
- ask the user to type JSON
- ask the user to understand graph connections
- flatten container result colours into one colour
- call a field "result label"
- implement only a cosmetic label patch and imply the workflow is solved
- skip browser verification
- use 0-based answer boxes in the human UI
- delete manual evidence during rerun
- promote manual evidence publicly without a separate agreed slice

## Current Best Architectural Instinct

The likely best long-term design is hybrid:

```text
structured parse JSON is authoritative
manual evidence graph rows are generated display artefacts
```

Reason:

- the structured parse is the right model for human and LLM editing
- the graph is already useful for display merge
- graph rows can be regenerated as display rules improve

But this is not yet agreed. Claude has been asked to challenge it.

## Exact Next Prompt Suggestion

If starting a new thread after Claude replies, use something like:

```text
Please review Claude's response to
documents\PHASE_2_STRUCTURED_PARSE_EDITOR_SPEC_FOR_CLAUDE.md with extreme care.
Do not implement yet. Check whether it preserves the goal: a shared human/LLM
structured parse format, split-colour container support, 1-based answer boxes,
and no graph/storage concepts in the primary UI. Then recommend the smallest
safe implementation slice.
```

If the user asks to implement:

```text
Implement only the agreed slice. Verify with the SWALLOW container clue and a
fresh browser check. Do not touch unrelated dirty worktree files.
```

## Bottom Line

We are trying to build the correction layer that makes the whole project
unstuck.

The next thread should not optimise for speed. It should preserve the design,
review carefully, implement a small but meaningful slice, and verify it in the
browser with the SWALLOW example.
