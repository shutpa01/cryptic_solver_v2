# Phase 2 Stage Three Word Purpose Fixes — Codex Instruction

## Task

Fix two word-purpose classification gaps in Stage Three so that anagram
indicators and separator words receive verified grammatical purposes when
the assembly evidence supports them.

One file changes: signature_solver/stage_three_proof.py.
No other file changes.

Do not change stage_two_casefile.py, wfw_display_adapter.py, admin.py,
test_stage_three_proof.py, or any other file.


---

## Background

Stage Three classifies every clue word with a grammatical or cryptic purpose
(answer_source, operation_indicator, definition_phrase_member, etc.).
word_purpose_coverage is PASS only when no word ends up as "unresolved_purpose".

Two words in the ENDEARMENT clue ("Unusually tender name for sweetheart?",
answer ENDEARMENT, clue_id 10067996) currently fail classification:

"Unusually" is the anagram indicator. Stage Two stores it as an operation
candidate with token "ANA_I". _blocks() already yields an OP_BLOCK for it,
but always with status "candidate". In _word_purposes(), an OP_BLOCK with
status "candidate" maps to "operation_indicator_candidate" (not verified),
so word_purpose_candidates is REVIEW. The fix: when the selected assembly
is an anagram answer_fit, ANA_I operation candidates should be promoted to
status "verified" in the OP_BLOCK they yield, so _word_purposes() classifies
them as "operation_indicator" (verified).

"for" is the link word between the wordplay span (tender name, indices 1-3)
and the definition span (sweetheart, indices 4-5). It carries no Stage Two
evidence and ends up as "unresolved_purpose", making word_purpose_coverage
REVIEW. The fix: after all Stage Two evidence passes, if a word has no
classification at all and sits in the positional gap between the end of all
SOURCE_BLOCKs and the start of all DEF_BLOCKs, classify it as
"structural_separator" (verified). This is a purely positional rule — it
fires only when all other classification passes have produced nothing.


---

## Change A: Promote ANA_I to verified when adjacent to selected anagram fodder

Location: _blocks (lines ~977-991 in the current file).

The current block that iterates operation candidates:

    for operation in _ops:
        span = operation.get("span")
        if span and tuple(span) in operation_spans:
            continue
        yield {
            "kind": "OP_BLOCK",
            "role": operation.get("role") or "operation",
            "text": operation.get("text", ""),
            "span": span,
            "token": operation.get("token"),
            "source": operation.get("source"),
            "status": operation.get("span_status") or "candidate",
        }
        if span:
            operation_spans.add(tuple(span))

Replace with:

    _anagram_assembly = (
        assembly is not None
        and assembly.get("kind") == "anagram"
        and assembly.get("status") == "answer_fit"
    )
    _selected_source_start = None
    _selected_source_end = None
    if _anagram_assembly:
        _part_spans = [
            p.get("span")
            for p in (assembly.get("parts") or [])
            if p.get("span") and len(p.get("span")) == 2
        ]
        if _part_spans:
            _selected_source_start = min(s[0] for s in _part_spans)
            _selected_source_end = max(s[1] for s in _part_spans)
    for operation in _ops:
        span = operation.get("span")
        if span and tuple(span) in operation_spans:
            continue
        _op_is_verified_ana_i = (
            _anagram_assembly
            and operation.get("token") == "ANA_I"
            and span is not None
            and len(span) == 2
            and _selected_source_start is not None
            and (span[1] == _selected_source_start or span[0] == _selected_source_end)
        )
        _op_status = "verified" if _op_is_verified_ana_i else (operation.get("span_status") or "candidate")
        yield {
            "kind": "OP_BLOCK",
            "role": operation.get("role") or "operation",
            "text": operation.get("text", ""),
            "span": span,
            "token": operation.get("token"),
            "source": operation.get("source"),
            "status": _op_status,
        }
        if span:
            operation_spans.add(tuple(span))

Before the loop, compute _anagram_assembly (True when the selected assembly
is an anagram answer_fit), then _selected_source_start and _selected_source_end
from the assembly parts' spans. These mark the extent of the selected fodder.

A ANA_I candidate is promoted to "verified" only when ALL of these hold:

  1. _anagram_assembly is True
  2. The operation token is "ANA_I"
  3. The operation has a valid two-element span
  4. The operation span is adjacent to the selected source span:
       span[1] == _selected_source_start  (indicator immediately before fodder)
     OR
       span[0] == _selected_source_end    (indicator immediately after fodder)

For ENDEARMENT: assembly parts tender [1,2] and name [2,3] give
_selected_source_start=1, _selected_source_end=3. Unusually [0,1] has
span[1]=1 == _selected_source_start=1, so it is adjacent and verified.
An unrelated ANA_I elsewhere in the clue (not touching the fodder span)
would not be promoted and remains candidate.

The adjacency check is intentionally narrow: it requires the indicator
span to touch the fodder span boundary. It does not attempt to verify
indicators that are separated from the fodder by other words.


---

## Change B: Classify positional gap words as structural_separator

Location: _word_purposes (lines ~1119-1236 in the current file).

### Step B1: Add three computed variables before the word loop

The current lines immediately before the word loop:

    manual_by_index = _manual_roles_by_index(manual_roles)
    for index, word in enumerate(words):

Replace with:

    manual_by_index = _manual_roles_by_index(manual_roles)
    _source_block_spans = [
        tuple(block["span"])
        for block in blocks
        if block.get("kind") == "SOURCE_BLOCK"
        and block.get("span")
        and len(block.get("span")) == 2
    ]
    _wordplay_end = max(s[1] for s in _source_block_spans) if _source_block_spans else None
    _definition_start = min(s[0] for s in definition_spans) if definition_spans else None
    for index, word in enumerate(words):

definition_spans is already computed two lines earlier in the same function.
These three new variables do not replace any existing variable; they add new
computed state only.

_wordplay_end is the maximum end index of all SOURCE_BLOCKs, which marks
where verified wordplay coverage ends.
_definition_start is the minimum start index of all DEF_BLOCKs, which marks
where the definition begins.
A word at index i is in the gap when _wordplay_end <= i < _definition_start.


### Step B2: Add separator check inside the word loop

Inside the loop, just before the existing "if role is None:" fallback, insert
the separator check. The current closing lines of the loop body are:

        if role is None:
            role = "unresolved_purpose"
            status = "unresolved"
        yield {
            "index": index,
            "text": word,
            "purpose": role,
            "status": status,
            "evidence": evidence,
        }

Replace with:

        if (role is None
                and _wordplay_end is not None
                and _definition_start is not None
                and _wordplay_end <= index < _definition_start):
            role = "structural_separator"
            status = "verified"
        if role is None:
            role = "unresolved_purpose"
            status = "unresolved"
        yield {
            "index": index,
            "text": word,
            "purpose": role,
            "status": status,
            "evidence": evidence,
        }

The condition is role is None (not "role is None or unresolved_purpose").
Words that already have any classification — including REVIEW_BLOCK words
that were tagged "unresolved_purpose" by Stage Two — are not overridden.
Only words that are entirely uncovered by all earlier passes receive the
structural_separator treatment. This is the intended conservative semantics:
positional evidence is used only as a last resort when no other evidence exists.

The separator check fires before the "if role is None: unresolved_purpose"
fallback so that gap words are classified as structural_separator rather
than falling through to unresolved.


---

## What not to change

Do not change the schema string "stage_three_proof:v1".
Do not change _blocks() for any other operation token type. Only "ANA_I"
is promoted; all other tokens keep their existing status logic.
Do not add defensive getattr to any other function.
Do not change _definition_check, _source_check, _assembly_check, or any
other check function.
Do not change _purpose_for_operation_candidate, _purpose_for_manual_role,
or _definition_purpose.
Do not change test_stage_three_proof.py or any test fixture.


---

## Verification

Run three checks in order. All three must pass before declaring done.

### Check 1 — syntax only

    C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe
        -m py_compile signature_solver\stage_three_proof.py

Expected: no output, exit 0.


### Check 2 — regression suite

    C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe
        signature_solver\test_stage_three_proof.py

Expected output:

    Stage Three proof contract passed

All assertions must pass. This uses the project virtualenv (no Flask needed
for the test — the import chain stage_three_proof -> stage_two_casefile -> db
does not require Flask).


### Check 3 — focused ENDEARMENT verification

Write and run a standalone script (do not save it to the repo) using the
project virtualenv:

    C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe

The script must:

1. Add the project root to sys.path so signature_solver is importable.

2. Import and run solve_clue:

    import sys
    sys.path.insert(0, r"C:\Users\shute\PycharmProjects\cryptic_solver_V2")
    from signature_solver.db import RefDB
    from signature_solver.solver import solve_clue

    db = RefDB()
    sr = solve_clue(
        "Unusually tender name for sweetheart?",
        "ENDEARMENT",
        db,
    )

3. Access sr.stage_three_proof. If it is None, print "stage_three_proof is None"
   and stop — that is a failure.

4. Call proof = sr.stage_three_proof.as_dict()

5. Print the word_purposes list:
    for wp in proof["word_purposes"]:
        print(wp["index"], wp["text"], wp["purpose"], wp["status"])

6. Find and print the status of each Stage Three check:
    for check in proof["checks"]:
        print(check["name"], check["status"])

Expected output includes these lines (index, text, purpose, status):

    0 Unusually operation_indicator verified
    1 tender answer_source verified
    2 name answer_source verified
    3 for structural_separator verified
    4 sweetheart definition_phrase_member verified

Expected check statuses include:

    word_purpose_coverage PASS
    word_purpose_candidates PASS

If word_purpose_coverage or word_purpose_candidates is REVIEW, print the
full detail field of those checks and identify which words are responsible.
Do not declare done unless both are PASS.

Note: sweetheart has a trailing "?" in the raw clue. After punctuation
stripping, it becomes "sweetheart". If the _clue_words function strips it
differently, the printed text may vary — but the purpose and status must
still be "definition_phrase_member" and "verified".

Note on environment: solve_clue, RefDB, and build_stage_three_proof have no
Flask dependency. The project virtualenv is sufficient for this check.


---

## After writing

Paste:
  1. The full replacement block for Change A (from "_anagram_assembly = ("
     through the final "operation_spans.add(tuple(span))" line, including
     the _selected_source_start/_selected_source_end computation and the
     _op_is_verified_ana_i condition).
  2. The three new variables added by Step B1 (from "_source_block_spans = ["
     through "_definition_start = ...").
  3. The separator check added by Step B2 (from "if (role is None" through
     "status = 'verified'").

Then run all three verification checks and paste the full output of each.
