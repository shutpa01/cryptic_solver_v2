# Phase 2 Structured Parse Editor — Slice 3 Codex Instruction

Date: 2026-05-28 (Revision 2)

## What this slice does

Add reversal and anagram validation to `_validate_structured_parse`, fix the
DB-fact audit so it uses fodder (not final answer letters) for reversal/anagram
pieces, add a new admin route, and add a new form in the clue page.

No changes to Stage Three, the WFW display adapter, the solver, proof storage,
the merge function, the container validation path, or the charade validation
path.

## Current state (already working — do not touch unless stated below)

- `signature_solver/manual_evidence_store.py`:
  - `_validate_structured_parse` handles pieces, container operations,
    complete-coverage check for source-only parses, colour uniqueness, and
    relationship validation.
  - `merge_structured_parse_into_display` handles N pieces, operations, and
    filler. Do not modify it.
- `web/routes/admin.py`:
  - `POST /admin/structured-parse/<clue_id>/container` — container route
  - `POST /admin/structured-parse/<clue_id>/source` — charade route
  - `_structured_parse_db_entries` — **must be changed** (see File 2a below).
    It currently uses `piece.letters` for synonym/abbreviation audit. For
    reversal/anagram pieces that is wrong — audit must use `operation.fodder`.
  - `_structured_parse_summary` — do not modify in this slice.
- `web/templates/clue.html`:
  - Charade form posts to `/source` (label: "Charade parse editor")
  - Container form posts to `/container` (label: "Container parse editor")
  - The new form goes between them.

## Critical design decision — piece.letters vs fodder

`piece.letters` is the final answer letters occupying the answer boxes.
It is validated against the answer by the existing piece-loop code:

```python
elif letters[i] != cleaned[idx]:
    errors.append("piece %s: letter %r at box %d does not match answer letter %r" ...)
```

Do not change this. Do not make `piece.letters` store pre-operation fodder.

The pre-operation input is stored in a new field on the operation:
`operation.fodder`.

For reversal:
- `piece.letters = "TRAP"` (final answer letters)
- `operation.fodder = "PART"` (pre-reversal input)
- `operation.result = "TRAP"` (== fodder reversed == piece.letters)

For anagram:
- `piece.letters = "TEAM"` (final answer letters)
- `operation.fodder = "MATE"` (pre-anagram input, same letter multiset)
- `operation.result = "TEAM"` (== piece.letters)

**Why `_structured_parse_db_entries` must use fodder:** the DB fact to audit
is "this clue word gives these letters" — for a reversal, that fact is
`family -> KIN`, not `family -> NIK`. Using `piece.letters` for the audit
would queue/check the wrong synonym or abbreviation fact, and enrichment
results would be broken. See File 2a below.

The merge function already maps `piece.letters` to `piece.answer_boxes`
correctly. No merge change is needed.

## File 1: `signature_solver/manual_evidence_store.py`

### Change: add reversal and anagram validation inside `_validate_structured_parse`

Locate the `for op in operations:` loop at around line 1082. Inside it, after
the `if op_type == "container":` block (which ends with its elif/else), add:

```python
        elif op_type == "reversal":
            input_id = op.get("input_piece_id")
            fodder = (op.get("fodder") or "").upper()
            result = (op.get("result") or "").upper()

            if not input_id or input_id not in piece_ids:
                errors.append(
                    "operation %s: input_piece_id %r not in pieces"
                    % (oid, input_id)
                )
            else:
                input_piece = next(p for p in pieces if p["id"] == input_id)
                piece_letters = (input_piece.get("letters") or "").upper()

                # The input piece must cover all answer boxes exactly once
                input_boxes = set(input_piece.get("answer_boxes") or [])
                expected_boxes = set(range(1, len(cleaned) + 1))
                if input_boxes != expected_boxes:
                    errors.append(
                        "operation %s: piece %s boxes %s must cover all "
                        "answer boxes (1..%d)"
                        % (oid, input_id, sorted(input_boxes), len(cleaned))
                    )

                if not fodder:
                    errors.append(
                        "operation %s: fodder is required for reversal" % oid
                    )
                elif not result:
                    errors.append(
                        "operation %s: result is required for reversal" % oid
                    )
                else:
                    if result != fodder[::-1]:
                        errors.append(
                            "operation %s: result %r is not the reverse of "
                            "fodder %r" % (oid, result, fodder)
                        )
                    if result != piece_letters:
                        errors.append(
                            "operation %s: result %r does not match piece %s "
                            "letters %r" % (oid, result, input_id, piece_letters)
                        )

        elif op_type == "anagram":
            input_id = op.get("input_piece_id")
            fodder = (op.get("fodder") or "").upper()
            result = (op.get("result") or "").upper()

            if not input_id or input_id not in piece_ids:
                errors.append(
                    "operation %s: input_piece_id %r not in pieces"
                    % (oid, input_id)
                )
            else:
                input_piece = next(p for p in pieces if p["id"] == input_id)
                piece_letters = (input_piece.get("letters") or "").upper()

                # The input piece must cover all answer boxes exactly once
                input_boxes = set(input_piece.get("answer_boxes") or [])
                expected_boxes = set(range(1, len(cleaned) + 1))
                if input_boxes != expected_boxes:
                    errors.append(
                        "operation %s: piece %s boxes %s must cover all "
                        "answer boxes (1..%d)"
                        % (oid, input_id, sorted(input_boxes), len(cleaned))
                    )

                if not fodder:
                    errors.append(
                        "operation %s: fodder is required for anagram" % oid
                    )
                elif not result:
                    errors.append(
                        "operation %s: result is required for anagram" % oid
                    )
                else:
                    if sorted(fodder) != sorted(result) or len(fodder) != len(result):
                        errors.append(
                            "operation %s: result %r is not an anagram of "
                            "fodder %r" % (oid, result, fodder)
                        )
                    if result != piece_letters:
                        errors.append(
                            "operation %s: result %r does not match piece %s "
                            "letters %r" % (oid, result, input_id, piece_letters)
                        )
```

Place this code immediately after the closing of the `if op_type == "container":`
block, inside the same `for op in operations:` loop. The loop currently has:

```python
        if op_type == "container":
            ...container validation...
```

After that block, add the `elif op_type == "reversal":` and
`elif op_type == "anagram":` branches shown above.

Do not change any other part of `_validate_structured_parse`.
Do not change the container block.
Do not change the source-only coverage check at the top of the operation loop.

## File 2a: `web/routes/admin.py` — fix `_structured_parse_db_entries`

This function currently reads `piece.letters` as the audit value for
synonym/abbreviation facts. For reversal/anagram pieces, that is wrong.

Replace the body of `_structured_parse_db_entries` with:

```python
def _structured_parse_db_entries(parse_dict):
    entries = []
    pieces = parse_dict.get("pieces") or []
    operations = parse_dict.get("operations") or []

    # For pieces that feed reversal/anagram operations, audit against
    # operation.fodder (the pre-operation input), not piece.letters
    # (the post-operation answer letters).
    piece_fodder = {}
    for op in operations:
        op_type = (op.get("type") or "").lower()
        if op_type in ("reversal", "anagram"):
            input_id = op.get("input_piece_id")
            fodder = (op.get("fodder") or "").strip()
            if input_id and fodder:
                piece_fodder[input_id] = fodder

    for piece in pieces:
        rel = (piece.get("relationship") or "").lower()
        word = (piece.get("clue_text") or "").strip()
        pid = piece.get("id")
        # Use fodder for reversal/anagram input pieces; piece.letters otherwise
        letters = piece_fodder.get(pid) or (piece.get("letters") or "").strip()
        if not word or not letters:
            continue
        if rel == "synonym":
            entries.append({"type": "synonym", "word": word, "value": letters})
        elif rel == "abbreviation":
            entries.append(
                {"type": "abbreviation", "word": word, "value": letters}
            )

    for op in operations:
        op_type = (op.get("type") or "").strip().lower()
        word = (op.get("clue_text") or "").strip()
        if op_type and word:
            entries.append({"type": "indicator", "word": word, "value": op_type})

    defn = parse_dict.get("definition") or {}
    if defn.get("clue_text") and parse_dict.get("answer"):
        entries.append({
            "type": "definition",
            "word": defn.get("clue_text"),
            "value": parse_dict.get("answer"),
        })
    return entries
```

The rest of `web/routes/admin.py` is unchanged.

## File 2b: `web/routes/admin.py` — add route

### Add route: `POST /admin/structured-parse/<clue_id>/single-operation`

Insert this route immediately after the `save_source_structured_parse` function
ends and before `delete_structured_parse_route` begins.

The route follows the exact same pattern as `save_source_structured_parse`:
fetch clue, parse form, build parse_dict, validate, write, score, render.

Form field names use the `op_` prefix throughout:

| Form field              | Description                                    |
|-------------------------|------------------------------------------------|
| `op_def_text`           | Definition clue words                          |
| `op_piece_text`         | Piece clue words                               |
| `op_piece_relationship` | Relationship (synonym/abbreviation/etc.)       |
| `op_piece_letters`      | Final answer letters (post-operation)          |
| `op_piece_boxes`        | Answer boxes, comma-separated 1-based integers |
| `op_piece_colour`       | Piece colour                                   |
| `op_type`               | "reversal" or "anagram"                        |
| `op_indicator_text`     | Indicator/operation clue words                 |
| `op_fodder`             | Pre-operation input letters                    |
| `op_indicator_colour`   | Indicator colour (optional)                    |
| `op_filler_text`        | Optional filler words (semicolon-separated)    |

Partial-row rule: if `op_piece_text`, `op_piece_letters`, or `op_piece_boxes` is
non-empty but any of the three is missing, add a form error naming the missing
field. Do not silently drop.

The `parse_dict` produced by this route:

```python
{
    "version": 1,
    "clue_id": clue_id,
    "answer": answer,
    "source": "human",
    "confidence": "verified",
    "created_by": "admin",
    "definition": {
        "id": "def1",
        "clue_text": def_text,
        "clue_word_positions": _find_word_positions(def_text, clue_text),
        "answer": answer,
    },
    "pieces": [
        {
            "id": "piece1",
            "clue_text": piece_text,
            "clue_word_positions": _find_word_positions(piece_text, clue_text),
            "relationship": piece_relationship,
            "letters": piece_letters,     # final answer letters (post-operation)
            "answer_boxes": piece_boxes,
            "mapping": "positional",
            "colour": piece_colour,
        }
    ],
    "transform_pieces": [],
    "operations": [
        {
            "id": "op1",
            "type": op_type,              # "reversal" or "anagram"
            "clue_text": indicator_text,
            "clue_word_positions": _find_word_positions(indicator_text, clue_text),
            "input_piece_id": "piece1",
            "fodder": fodder.upper(),     # pre-operation input letters
            "result": piece_letters,      # == piece.letters (post-operation)
            "colour": indicator_colour,
        }
    ],
    "filler": filler,
}
```

Notes:
- `fodder` comes from the `op_fodder` field, stripped and uppercased.
- `result` is set to `piece_letters` (same value already validated in the
  piece). This means the operation `result` field is never independently wrong
  if the form input is consistent.
- If `op_type` is not "reversal" or "anagram", add a form error:
  `"op_type must be reversal or anagram"`.
- Filler parsing: same semicolon-split pattern as `save_source_structured_parse`.
- Error rendering, write, score, and re-render: identical to both existing
  routes. Copy that pattern exactly.

Imports inside the route function (same as existing routes):

```python
from signature_solver.manual_evidence_store import (
    _find_word_positions,
    _validate_structured_parse,
    write_structured_parse,
)
```

## File 3: `web/templates/clue.html`

### Add a collapsed `<details>` section between the charade form and the container form

The manual parse editor section (starting at the outer `<details class="mt-3">
<summary>Admin: manual parse editor</summary>`) already contains:

1. `{% set mp = clue.manual_parse_form or {} %}` — must stay here, above all forms
2. The charade `<form>` (border sky-200)
3. The container `<form>` (border teal-200)
4. The `{% with %}{% include "partials/manual_evidence_nodes.html" %}{% endwith %}`
5. The legacy graph editor `<details>`

Insert the new reversal/anagram section between items 2 and 3 above. It must
be a `<details>` element — collapsed by default (no `open` attribute). The
`<form>` lives inside the `<details>`, not as the outer element.

The complete block to insert:

```html
<details class="rounded border border-amber-200 bg-white/80">
    <summary class="text-xs font-semibold uppercase text-amber-700 cursor-pointer
                    select-none px-3 py-2 hover:bg-amber-50">
        Reversal / anagram parse editor
    </summary>
    <div class="p-3 grid grid-cols-1 gap-3">

        <!-- Clue reference: visible once the section is opened, outside the form -->
        <div class="rounded border border-slate-200 bg-slate-50 px-3 py-2">
            <p class="text-xs font-semibold uppercase text-slate-500">Clue</p>
            <p class="text-sm font-semibold text-slate-900">
                {{ clue.clue_text }}
                {% if clue.enumeration %}
                <span class="font-normal text-slate-500">({{ clue.enumeration }})</span>
                {% endif %}
            </p>
            {% if clue.answer %}
            <p class="mt-1 text-xs text-slate-500">
                Answer: <span class="font-mono font-semibold text-slate-800">{{ clue.answer }}</span>
            </p>
            {% endif %}
        </div>

        <form hx-post="/admin/structured-parse/{{ clue.id }}/single-operation"
              hx-target="#manual-evidence-list-{{ clue.id }}"
              hx-swap="outerHTML"
              class="grid grid-cols-1 gap-3 text-sm">
            <p class="text-xs text-slate-500">Use for clues where one source
            piece is reversed or anagrammed to give the answer. Fodder letters
            = what the clue word gives before the operation. Letters produced
            = what ends up in the answer boxes (= fodder reversed or
            anagrammed).</p>

            <!-- Row 1: definition + filler -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                    Definition clue words
                    <input name="op_def_text"
                           class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                </label>
                <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                    Optional filler/link words (semicolon-separated)
                    <input name="op_filler_text"
                           class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                </label>
            </div>

            <!-- Row 2: piece (5 columns) -->
            <div class="grid grid-cols-2 md:grid-cols-5 gap-2">
                <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                    Piece clue words
                    <input name="op_piece_text"
                           class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                </label>
                <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                    Relationship
                    <select name="op_piece_relationship"
                            class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                        <option value="synonym">synonym</option>
                        <option value="abbreviation">abbreviation</option>
                        <option value="literal_letters">literal letters</option>
                        <option value="initial_letters">initial letters</option>
                        <option value="foreign">foreign word</option>
                        <option value="pronoun">pronoun/name</option>
                        <option value="single_letter">single letter</option>
                        <option value="hidden_letters">hidden letters</option>
                    </select>
                </label>
                <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                    Fodder letters (pre-operation)
                    <input name="op_fodder"
                           class="rounded border border-slate-300 px-2 py-1 bg-white uppercase font-mono text-xs">
                </label>
                <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                    Letters produced (post-operation, in answer boxes)
                    <input name="op_piece_letters"
                           class="rounded border border-slate-300 px-2 py-1 bg-white uppercase font-mono text-xs">
                </label>
                <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                    Answer boxes (1-based)
                    <input name="op_piece_boxes" placeholder="1,2,3"
                           class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                </label>
            </div>

            <!-- Row 3: piece colour -->
            <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700 w-fit">
                Piece colour
                <select name="op_piece_colour"
                        class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                    <option value="blue">blue</option>
                    <option value="pink">pink</option>
                    <option value="yellow">yellow</option>
                    <option value="orange">orange</option>
                    <option value="purple">purple</option>
                </select>
            </label>

            <!-- Row 4: operation -->
            <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
                <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                    Operation type
                    <select name="op_type"
                            class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                        <option value="reversal">reversal</option>
                        <option value="anagram">anagram</option>
                    </select>
                </label>
                <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                    Indicator words
                    <input name="op_indicator_text" placeholder="e.g. rejected"
                           class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
                </label>
                <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                    Indicator colour
                    <select name="op_indicator_colour"
                            class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                        <option value="blue">blue</option>
                        <option value="pink">pink</option>
                        <option value="yellow">yellow</option>
                        <option value="orange">orange</option>
                        <option value="purple">purple</option>
                    </select>
                </label>
            </div>

            <button type="submit"
                    class="text-xs px-3 py-1.5 rounded border border-amber-500 bg-amber-50
                           text-amber-800 hover:bg-amber-100 cursor-pointer font-semibold w-fit">
                Save reversal/anagram parse
            </button>
        </form>
    </div>
</details>
```

**Structure rules:**
- The `<details>` has no `open` attribute — it is collapsed by default.
- The clue reference `<div>` is inside `<details>` but outside `<form>`. It is
  display-only text; not part of the form submission.
- The `<form>` has no border or background class — those are on the `<details>`.
- `{% set mp = clue.manual_parse_form or {} %}` is already above this block at
  the top of the violet `<div>`. Do not move it or add a second copy.
- The `{% with %}{% include "partials/manual_evidence_nodes.html" %}{% endwith %}`
  block stays where it is, after all three parser forms, directly inside the
  violet `<div>`. Do not wrap it in any new `<details>`.

**Prefill note:** The new form is blank on load by design. It does not
pre-populate from any existing parse. It must not overwrite an existing parse
for this clue unless the form is explicitly submitted. The table has
`clue_id UNIQUE`, so submission always overwrites — that is intentional and
correct behaviour only on submit.

Do not remove or modify the charade form or container form.

## File 4: `signature_solver/test_structured_parse_validation.py` (new file)

Create this file in `signature_solver/`. It contains only direct calls to
`_validate_structured_parse` and `_structured_parse_db_entries`. No Flask
app context, no live DB connections.

```python
"""Direct validation tests for structured parse — reversal and anagram."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.manual_evidence_store import _validate_structured_parse
from web.routes.admin import _structured_parse_db_entries


def _reversal_parse(answer, piece_letters, fodder, result,
                    piece_colour="blue", boxes=None):
    if boxes is None:
        boxes = list(range(1, len(piece_letters) + 1))
    return {
        "version": 1,
        "clue_id": 1,
        "answer": answer,
        "source": "human",
        "confidence": "verified",
        "definition": {
            "id": "def1",
            "clue_text": "something",
            "clue_word_positions": [0],
            "answer": answer,
        },
        "pieces": [{
            "id": "piece1",
            "clue_text": "source",
            "clue_word_positions": [1],
            "relationship": "synonym",
            "letters": piece_letters,
            "answer_boxes": boxes,
            "mapping": "positional",
            "colour": piece_colour,
        }],
        "transform_pieces": [],
        "operations": [{
            "id": "op1",
            "type": "reversal",
            "clue_text": "rejected",
            "clue_word_positions": [2],
            "input_piece_id": "piece1",
            "fodder": fodder,
            "result": result,
            "colour": piece_colour,
        }],
        "filler": [],
    }


def _anagram_parse(answer, piece_letters, fodder, result,
                   piece_colour="blue", boxes=None):
    if boxes is None:
        boxes = list(range(1, len(piece_letters) + 1))
    return {
        "version": 1,
        "clue_id": 1,
        "answer": answer,
        "source": "human",
        "confidence": "verified",
        "definition": {
            "id": "def1",
            "clue_text": "something",
            "clue_word_positions": [0],
            "answer": answer,
        },
        "pieces": [{
            "id": "piece1",
            "clue_text": "source",
            "clue_word_positions": [1],
            "relationship": "synonym",
            "letters": piece_letters,
            "answer_boxes": boxes,
            "mapping": "positional",
            "colour": piece_colour,
        }],
        "transform_pieces": [],
        "operations": [{
            "id": "op1",
            "type": "anagram",
            "clue_text": "scrambled",
            "clue_word_positions": [2],
            "input_piece_id": "piece1",
            "fodder": fodder,
            "result": result,
            "colour": piece_colour,
        }],
        "filler": [],
    }


def run_tests():
    # --- REVERSAL VALIDATION ---

    # Valid reversal: PART reversed = TRAP
    errors = _validate_structured_parse(
        _reversal_parse("TRAP", "TRAP", "PART", "TRAP"), "TRAP"
    )
    assert errors == [], "valid reversal should pass: %s" % errors

    # Invalid reversal: fodder wrong (TARP reversed = PRAT, not TRAP)
    errors = _validate_structured_parse(
        _reversal_parse("TRAP", "TRAP", "TARP", "TRAP"), "TRAP"
    )
    assert any("not the reverse" in e for e in errors), (
        "wrong fodder should fail: %s" % errors
    )

    # Invalid reversal: result does not match piece letters
    errors = _validate_structured_parse(
        _reversal_parse("TRAP", "TRAP", "PART", "PRAT"), "TRAP"
    )
    assert any("piece" in e and "letters" in e for e in errors), (
        "result != piece letters should fail: %s" % errors
    )

    # Invalid reversal: missing fodder
    parse = _reversal_parse("TRAP", "TRAP", "PART", "TRAP")
    parse["operations"][0]["fodder"] = ""
    errors = _validate_structured_parse(parse, "TRAP")
    assert any("fodder" in e for e in errors), (
        "missing fodder should fail: %s" % errors
    )

    # Invalid reversal: piece does not cover all answer boxes
    # answer = TRAP (4 letters), piece only covers boxes 1,2
    errors = _validate_structured_parse(
        _reversal_parse("TRAP", "TR", "RT", "TR", boxes=[1, 2]), "TRAP"
    )
    assert any("cover" in e or "box" in e for e in errors), (
        "partial coverage reversal should fail: %s" % errors
    )

    # --- ANAGRAM VALIDATION ---

    # Valid anagram: MATE anagrammed = TEAM
    errors = _validate_structured_parse(
        _anagram_parse("TEAM", "TEAM", "MATE", "TEAM"), "TEAM"
    )
    assert errors == [], "valid anagram should pass: %s" % errors

    # Invalid anagram: extra letter in fodder
    errors = _validate_structured_parse(
        _anagram_parse("TEAM", "TEAM", "MATES", "TEAM"), "TEAM"
    )
    assert any("not an anagram" in e for e in errors), (
        "wrong letter count should fail: %s" % errors
    )

    # Invalid anagram: wrong letters in fodder
    errors = _validate_structured_parse(
        _anagram_parse("TEAM", "TEAM", "MAZE", "TEAM"), "TEAM"
    )
    assert any("not an anagram" in e for e in errors), (
        "wrong letters should fail: %s" % errors
    )

    # Invalid anagram: result does not match piece letters
    errors = _validate_structured_parse(
        _anagram_parse("TEAM", "TEAM", "MATE", "MATE"), "TEAM"
    )
    assert any("piece" in e and "letters" in e for e in errors), (
        "result != piece letters should fail: %s" % errors
    )

    # Invalid anagram: piece does not cover all answer boxes
    errors = _validate_structured_parse(
        _anagram_parse("TEAM", "TE", "ET", "TE", boxes=[1, 2]), "TEAM"
    )
    assert any("cover" in e or "box" in e for e in errors), (
        "partial coverage anagram should fail: %s" % errors
    )

    # --- DB AUDIT USES FODDER NOT PIECE.LETTERS ---

    # For a reversal piece with relationship synonym, the audit entry must
    # use operation.fodder (the pre-reversal value), not piece.letters
    reversal_audit_parse = _reversal_parse("TRAP", "TRAP", "PART", "TRAP")
    reversal_audit_parse["pieces"][0]["clue_text"] = "part"
    entries = _structured_parse_db_entries(reversal_audit_parse)
    synonym_entries = [e for e in entries if e["type"] == "synonym"]
    assert len(synonym_entries) == 1, (
        "expected exactly one synonym entry: %s" % synonym_entries
    )
    assert synonym_entries[0]["value"] == "PART", (
        "synonym audit must use fodder PART, not piece.letters TRAP: %s"
        % synonym_entries
    )

    # For a charade (no operation), audit still uses piece.letters
    pa = {
        "version": 1, "clue_id": 1, "answer": "PA", "source": "human",
        "confidence": "verified",
        "definition": {"id": "def1", "clue_text": "answer",
                       "clue_word_positions": [1], "answer": "PA"},
        "pieces": [
            {"id": "piece1", "clue_text": "Quiet", "clue_word_positions": [0],
             "relationship": "abbreviation", "letters": "P",
             "answer_boxes": [1], "mapping": "positional", "colour": "blue"},
            {"id": "piece2", "clue_text": "a", "clue_word_positions": [2],
             "relationship": "literal_letters", "letters": "A",
             "answer_boxes": [2], "mapping": "positional", "colour": "pink"},
        ],
        "transform_pieces": [], "operations": [], "filler": [],
    }
    pa_entries = _structured_parse_db_entries(pa)
    abbrev_entries = [e for e in pa_entries if e["type"] == "abbreviation"]
    assert len(abbrev_entries) == 1, (
        "expected one abbreviation entry for PA: %s" % abbrev_entries
    )
    assert abbrev_entries[0]["value"] == "P", (
        "charade audit must use piece.letters P: %s" % abbrev_entries
    )

    # --- EXISTING REGRESSIONS ---

    # SWALLOW container still passes
    swallow = {
        "version": 1, "clue_id": 1, "answer": "SWALLOW",
        "source": "human", "confidence": "verified",
        "definition": {"id": "def1", "clue_text": "Bird,",
                       "clue_word_positions": [0], "answer": "SWALLOW"},
        "pieces": [
            {"id": "piece1", "clue_text": "female", "clue_word_positions": [2],
             "relationship": "synonym", "letters": "SOW",
             "answer_boxes": [1, 6, 7], "mapping": "positional", "colour": "blue"},
            {"id": "piece2", "clue_text": "fence", "clue_word_positions": [4],
             "relationship": "synonym", "letters": "WALL",
             "answer_boxes": [2, 3, 4, 5], "mapping": "positional", "colour": "pink"},
        ],
        "transform_pieces": [],
        "operations": [{"id": "op1", "type": "container", "clue_text": "going over",
                        "clue_word_positions": [3, 4], "outer_piece_id": "piece1",
                        "inner_piece_id": "piece2", "result": "SWALLOW",
                        "colour": "blue"}],
        "filler": [],
    }
    errors = _validate_structured_parse(swallow, "SWALLOW")
    assert errors == [], "SWALLOW container should still pass: %s" % errors

    # PA charade still passes
    errors = _validate_structured_parse(pa, "PA")
    assert errors == [], "PA charade should still pass: %s" % errors

    # PA with one piece missing box 2 fails coverage
    pa_bad = {**pa, "pieces": [pa["pieces"][0]]}
    errors = _validate_structured_parse(pa_bad, "PA")
    assert any("not cover" in e or "box" in e for e in errors), (
        "missing box coverage should fail: %s" % errors
    )

    # Filler does not cause validation errors
    pa_with_filler = {
        **pa,
        "filler": [{"id": "fill1", "clue_text": "with",
                    "clue_word_positions": [3], "role": "link"}],
    }
    errors = _validate_structured_parse(pa_with_filler, "PA")
    assert errors == [], "PA charade with filler should pass: %s" % errors

    print("Structured parse validation tests passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
```

## Verification steps

Run in order. All must pass before reporting done:

```
python -m py_compile signature_solver/manual_evidence_store.py
python -m py_compile web/routes/admin.py
python signature_solver/test_structured_parse_validation.py
python web/test_clue_wfw_render_contract.py
python signature_solver/test_wfw_display_adapter.py
```

Notes:
- `python -m py_compile web/templates/clue.html` is not valid — Jinja templates
  cannot be compiled with py_compile. Template syntax errors will surface in
  `web/test_clue_wfw_render_contract.py`, which loads the Flask app.
- All five commands must pass. A file that compiles but has wrong logic is still
  broken.

## What not to touch

```
signature_solver/stage_three_proof.py
signature_solver/wfw_display_adapter.py
signature_solver/manual_evidence_store.py  — only _validate_structured_parse changes
web/routes/clue.py
web/routes/admin.py  — only _structured_parse_db_entries replacement + new route
web/templates/partials/atomic_parse.html
web/templates/partials/manual_evidence_nodes.html
Automatic solver files
Proof attempt storage
Reference DB files
The container route and its validation path
The charade route and its validation path
_structured_parse_summary
```

## Anti-patterns

- Do not make `piece.letters` store pre-operation fodder. It stores final answer
  letters and is validated against the answer.
- Do not add `result == cleaned_answer` checks for reversal/anagram operations.
  The piece's own letter-match validation covers correctness at the answer boxes.
  (Container is different because its result IS the full answer.)
- Do not invent a new operation without a separate instruction.
- Do not add deletion in this slice.
- Do not modify `merge_structured_parse_into_display`.
- Do not use live DB clue rows in the validation test file.
- Do not use `python -m py_compile` on `.html` files.
