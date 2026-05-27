# Phase 2 Manual Parse Direct DB Write — Codex Instruction

Date: 2026-05-28

## What this slice does

Change `_apply_structured_parse_score_if_valid` in `web/routes/admin.py` so
that when a manual structured parse has missing DB facts, those facts are
written directly to `cryptic_new.db` rather than queued in
`pending_enrichments`. A re-audit confirms the writes succeeded. Only fall back
to `manual_structured_parse_needs_db` if the write genuinely fails.

This is a save/scoring path change only.

## Files to change

```
web/routes/admin.py   — two changes (add helper, replace failure path)
web/test_manual_parse_direct_write.py   — new test file
```

Do not touch:
```
signature_solver/*
web/templates/*
Any route other than the scoring helper
pending_enrichments behaviour for solver-generated enrichment
Stage Two, Stage Three, WFW display
Manual parse validation rules
```

## Change 1: add `_write_structured_parse_db_facts` helper

Locate the end of `_audit_structured_parse_db_facts`. Its last two lines are:

```python
    return {"missing": missing, "queued": queued}
```

Immediately after that closing line (before the next blank lines and next
function), insert:

```python
def _write_structured_parse_db_facts(parse_dict):
    """Write DB facts implied by a manual structured parse directly to
    cryptic_new.db.  Uses the same normalise/insert path as other WFW fact
    writes.  _normalise_wfw_db_entry calls abort(400) on unknown entry types;
    all types currently produced by _structured_parse_db_entries are known.
    Returns (added, existing) lists of normalised entries.
    """
    entries = _structured_parse_db_entries(parse_dict)
    return _write_wfw_db_entries(entries)
```

No other change to the area around `_audit_structured_parse_db_facts`.

## Change 2: replace the failure path in `_apply_structured_parse_score_if_valid`

Locate this exact block inside `_apply_structured_parse_score_if_valid`:

```python
    audit = _audit_structured_parse_db_facts(
        db, clue, parse_dict, queue_missing=True)
    if audit["missing"]:
        existing = db.execute(
            "SELECT 1 FROM structured_explanations WHERE clue_id = ?",
            (clue_id,),
        ).fetchone()
        if existing:
            db.execute(
                """UPDATE structured_explanations
                   SET confidence = 0.0,
                       model_version = 'manual_structured_parse_needs_db'
                   WHERE clue_id = ?""",
                (clue_id,),
            )
        else:
            db.execute(
                """INSERT INTO structured_explanations
                   (clue_id, confidence, model_version)
                   VALUES (?, 0.0, 'manual_structured_parse_needs_db')""",
                (clue_id,),
            )
        errors = [
            "missing DB fact: %s %s -> %s"
            % (item["type"], item["word"], item["value"])
            for item in audit["missing"]
        ]
        if audit["queued"]:
            errors.append("%d missing DB fact(s) queued for review" % audit["queued"])
        return {
            "ok": False,
            "errors": errors,
            "missing": audit["missing"],
            "queued": audit["queued"],
        }
```

Replace it with:

```python
    audit = _audit_structured_parse_db_facts(
        db, clue, parse_dict, queue_missing=False)
    if audit["missing"]:
        added, existing_written = _write_structured_parse_db_facts(parse_dict)
        audit = _audit_structured_parse_db_facts(
            db, clue, parse_dict, queue_missing=False)
        if audit["missing"]:
            _se_row = db.execute(
                "SELECT 1 FROM structured_explanations WHERE clue_id = ?",
                (clue_id,),
            ).fetchone()
            if _se_row:
                db.execute(
                    """UPDATE structured_explanations
                       SET confidence = 0.0,
                           model_version = 'manual_structured_parse_needs_db'
                       WHERE clue_id = ?""",
                    (clue_id,),
                )
            else:
                db.execute(
                    """INSERT INTO structured_explanations
                       (clue_id, confidence, model_version)
                       VALUES (?, 0.0, 'manual_structured_parse_needs_db')""",
                    (clue_id,),
                )
            errors = [
                "manual parse DB write failed: %s %s -> %s"
                % (item["type"], item["word"], item["value"])
                for item in audit["missing"]
            ]
            return {
                "ok": False,
                "errors": errors,
                "missing": audit["missing"],
                "added": added,
                "existing": existing_written,
            }
```

### What changed and why

| Before | After | Reason |
|--------|-------|--------|
| `queue_missing=True` | `queue_missing=False` | Stop adding to `pending_enrichments` for manual parse facts |
| No write attempt | `_write_structured_parse_db_facts` called | Write missing facts directly |
| Single audit | Audit → write → re-audit | Confirm writes succeeded before scoring |
| `"missing DB fact: ..."` | `"manual parse DB write failed: ..."` | This path is now a genuine write failure, not a queue notification |
| `"queued for review"` appended | Removed | No queueing in this path |
| Returns `"queued"` key | Returns `"added"` / `"existing"` keys | `"queued"` key not used by any caller; new keys expose write outcome |
| Variable named `existing` | Renamed to `existing_written` / `_se_row` | Avoids shadowing the return value from `_write_structured_parse_db_facts` |

### Verify no caller reads `"queued"` from the return value

All three call sites read only `result.get("ok")` and `result.get("errors", [])`.
The `"queued"` key is not consumed by any caller. Confirm this by searching for
`"queued"` near `_apply_structured_parse_score_if_valid` call sites before
making the change.

## File: `web/test_manual_parse_direct_write.py` (new)

```python
"""Tests for the direct DB write path in _apply_structured_parse_score_if_valid.

Uses monkeypatching to avoid touching the live cryptic_new.db or live clue data.
"""

import os
import sys
import sqlite3
import unittest
from unittest.mock import patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers to build minimal in-memory state
# ---------------------------------------------------------------------------

def _make_clue_db(clue_id=1, answer="BEATNIK"):
    """Return an in-memory sqlite3 connection with the minimum schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE clues (
            id INTEGER PRIMARY KEY,
            clue_text TEXT,
            answer TEXT,
            source TEXT,
            puzzle_number TEXT,
            definition TEXT,
            wordplay_type TEXT,
            ai_explanation TEXT,
            enumeration TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE structured_explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clue_id INTEGER NOT NULL UNIQUE,
            confidence REAL,
            model_version TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE pending_enrichments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            word TEXT,
            letters TEXT,
            answer TEXT,
            clue_text TEXT,
            source TEXT,
            puzzle_number TEXT
        )
    """)
    conn.execute(
        "INSERT INTO clues (id, clue_text, answer, source, puzzle_number) "
        "VALUES (?, 'test clue', ?, 'telegraph', '99999')",
        (clue_id, answer),
    )
    conn.commit()
    return conn


def _make_parse_dict(clue_id=1, answer="BEATNIK"):
    """Minimal valid structured parse dict for testing."""
    return {
        "version": 1,
        "clue_id": clue_id,
        "answer": answer,
        "source": "human",
        "confidence": "verified",
        "definition": {
            "id": "def1",
            "clue_text": "Bohemian",
            "clue_word_positions": [0],
            "answer": answer,
        },
        "pieces": [
            {
                "id": "piece1",
                "clue_text": "Pound",
                "clue_word_positions": [1],
                "relationship": "synonym",
                "letters": "BEAT",
                "answer_boxes": [1, 2, 3, 4],
                "mapping": "positional",
                "colour": "blue",
            },
            {
                "id": "piece2",
                "clue_text": "family",
                "clue_word_positions": [2],
                "relationship": "synonym",
                "letters": "NIK",
                "answer_boxes": [5, 6, 7],
                "mapping": "positional",
                "colour": "pink",
            },
        ],
        "transform_pieces": [],
        "operations": [],
        "filler": [],
    }


# ---------------------------------------------------------------------------
# Test 1: _structured_parse_db_entries emits the right facts
# ---------------------------------------------------------------------------

def test_db_entries_extraction():
    """_structured_parse_db_entries returns the right fact types and values."""
    from web.routes.admin import _structured_parse_db_entries

    parse = _make_parse_dict()
    entries = _structured_parse_db_entries(parse)

    types = {e["type"] for e in entries}
    assert "definition" in types, "expected definition entry"
    assert "synonym" in types, "expected synonym entries"

    def_entries = [e for e in entries if e["type"] == "definition"]
    assert len(def_entries) == 1
    assert def_entries[0]["word"] == "Bohemian"
    assert def_entries[0]["value"] == "BEATNIK"

    syn_entries = [e for e in entries if e["type"] == "synonym"]
    words = {e["word"] for e in syn_entries}
    assert "Pound" in words
    assert "family" in words
    beat_entry = next(e for e in syn_entries if e["word"] == "Pound")
    assert beat_entry["value"] == "BEAT"

    print("test_db_entries_extraction passed")


# ---------------------------------------------------------------------------
# Test 1b: reversal fodder used for DB entry, not piece.letters
# ---------------------------------------------------------------------------

def test_db_entries_reversal_uses_fodder():
    """For reversal pieces, synonym fact uses operation.fodder not piece.letters."""
    from web.routes.admin import _structured_parse_db_entries

    parse = {
        "version": 1, "clue_id": 1, "answer": "TRAP",
        "source": "human", "confidence": "verified",
        "definition": {"id": "def1", "clue_text": "something",
                       "clue_word_positions": [0], "answer": "TRAP"},
        "pieces": [{
            "id": "piece1", "clue_text": "part",
            "clue_word_positions": [1], "relationship": "synonym",
            "letters": "TRAP",  # post-reversal
            "answer_boxes": [1, 2, 3, 4], "mapping": "positional", "colour": "blue",
        }],
        "transform_pieces": [],
        "operations": [{
            "id": "op1", "type": "reversal", "clue_text": "rejected",
            "clue_word_positions": [2], "input_piece_id": "piece1",
            "fodder": "PART",  # pre-reversal
            "result": "TRAP", "colour": "blue",
        }],
        "filler": [],
    }

    entries = _structured_parse_db_entries(parse)
    syn_entries = [e for e in entries if e["type"] == "synonym"]
    assert len(syn_entries) == 1, "expected one synonym entry: %s" % syn_entries
    assert syn_entries[0]["value"] == "PART", (
        "synonym must use fodder PART not piece.letters TRAP: %s" % syn_entries
    )
    print("test_db_entries_reversal_uses_fodder passed")


# ---------------------------------------------------------------------------
# Test 2: direct write path — scores HIGH, no pending_enrichments
# ---------------------------------------------------------------------------

def test_direct_write_scores_high():
    """When DB facts are missing, they are written directly and clue scores HIGH."""
    from web.routes.admin import _apply_structured_parse_score_if_valid

    db = _make_clue_db(clue_id=1, answer="BEATNIK")

    # Write the parse to the manual_structured_parses table
    from signature_solver.manual_evidence_store import (
        write_structured_parse, ensure_tables,
    )
    ensure_tables(conn=db)
    write_structured_parse(1, _make_parse_dict(), source="human",
                           status="verified", conn=db)
    db.commit()

    # Patch: first audit returns missing; _write_wfw_db_entries succeeds;
    # second audit returns no missing.
    call_count = {"n": 0}

    def fake_audit(db_, clue_, parse_, queue_missing=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: missing facts
            return {"missing": [{"type": "synonym", "word": "Pound", "value": "BEAT"}],
                    "queued": 0}
        # Second call: facts now present
        return {"missing": [], "queued": 0}

    def fake_write(parse_):
        return ([{"type": "synonym", "word": "Pound", "value": "BEAT"}], [])

    with patch("web.routes.admin._audit_structured_parse_db_facts", fake_audit), \
         patch("web.routes.admin._write_structured_parse_db_facts", fake_write):
        result = _apply_structured_parse_score_if_valid(db, 1)

    assert result is not None, "expected a result"
    assert result.get("ok") is True, "expected ok=True: %s" % result
    row = db.execute(
        "SELECT confidence, model_version FROM structured_explanations "
        "WHERE clue_id = 1"
    ).fetchone()
    assert row is not None
    assert row["confidence"] == 1.0, "expected confidence 1.0: %s" % row["confidence"]
    assert row["model_version"] == "manual_structured_parse", (
        "expected manual_structured_parse: %s" % row["model_version"]
    )
    pending = db.execute("SELECT COUNT(*) FROM pending_enrichments").fetchone()[0]
    assert pending == 0, "expected no pending_enrichments: got %d" % pending

    print("test_direct_write_scores_high passed")


# ---------------------------------------------------------------------------
# Test 3: write fails — scores LOW, no pending_enrichments, correct error text
# ---------------------------------------------------------------------------

def test_write_failure_scores_low():
    """If write cannot satisfy audit, clue scores LOW with correct error text."""
    from web.routes.admin import _apply_structured_parse_score_if_valid

    db = _make_clue_db(clue_id=1, answer="BEATNIK")

    from signature_solver.manual_evidence_store import (
        write_structured_parse, ensure_tables,
    )
    ensure_tables(conn=db)
    write_structured_parse(1, _make_parse_dict(), source="human",
                           status="verified", conn=db)
    db.commit()

    missing_fact = {"type": "synonym", "word": "Pound", "value": "BEAT"}

    def fake_audit_always_missing(db_, clue_, parse_, queue_missing=False):
        return {"missing": [missing_fact], "queued": 0}

    def fake_write_noop(parse_):
        # Simulates a write that doesn't actually insert anything
        return ([], [])

    with patch("web.routes.admin._audit_structured_parse_db_facts",
               fake_audit_always_missing), \
         patch("web.routes.admin._write_structured_parse_db_facts", fake_write_noop):
        result = _apply_structured_parse_score_if_valid(db, 1)

    assert result is not None
    assert result.get("ok") is False, "expected ok=False: %s" % result
    assert any("manual parse DB write failed" in e
               for e in result.get("errors", [])), (
        "expected 'manual parse DB write failed' in errors: %s" % result
    )
    assert "queued for review" not in str(result.get("errors", [])), (
        "must not say 'queued for review': %s" % result
    )

    row = db.execute(
        "SELECT confidence, model_version FROM structured_explanations "
        "WHERE clue_id = 1"
    ).fetchone()
    assert row is not None
    assert row["confidence"] == 0.0
    assert row["model_version"] == "manual_structured_parse_needs_db"

    pending = db.execute("SELECT COUNT(*) FROM pending_enrichments").fetchone()[0]
    assert pending == 0, "expected no pending_enrichments: got %d" % pending

    print("test_write_failure_scores_low passed")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests():
    test_db_entries_extraction()
    test_db_entries_reversal_uses_fodder()
    test_direct_write_scores_high()
    test_write_failure_scores_low()
    print("All direct-write tests passed")
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
python -m py_compile web/routes/admin.py
python web/test_manual_parse_direct_write.py
python web/test_clue_wfw_render_contract.py
python signature_solver/test_wfw_display_adapter.py
```

Manual smoke test after Flask restart:

1. Open a clue page as admin.
2. Submit a valid manual structured parse with a synonym or definition fact that
   is not yet in `cryptic_new.db`.
3. Confirm the clue scores `manual_structured_parse` with confidence 1.0
   immediately — no "needs DB" warning.
4. Confirm the fact appears in the relevant reference table in `cryptic_new.db`.
5. Confirm no new row in `pending_enrichments`.
6. Run puzzle-level Re-verify; confirm the clue stays at 1.0.

## What not to touch

```
_audit_structured_parse_db_facts       — signature and body unchanged
_write_wfw_db_entries                  — unchanged
_normalise_wfw_db_entry                — unchanged
_insert_wfw_db_entry                   — unchanged
_structured_parse_db_entries           — unchanged
_queue_pending_enrichment              — unchanged (still used by solver path)
pending_enrichments solver workflow    — unchanged
All three save routes                  — unchanged (they call the scoring helper)
Manual parse validation                — unchanged
Stage Two, Stage Three, WFW display    — unchanged
```

## Anti-patterns

- Do not set `queue_missing=True` anywhere in the manual parse scoring path.
- Do not remove `_queue_pending_enrichment` from the codebase — it is still
  used by the solver enrichment path.
- Do not try to clean up existing `pending_enrichments` rows from prior saves.
- Do not use `existing` as a local variable name in the replacement block —
  it shadows the return value from `_write_structured_parse_db_facts`. Use
  `existing_written` and `_se_row` as shown in the replacement code above.
- Do not add `INSERT OR IGNORE` to `_insert_wfw_db_entry` — the
  `_wfw_db_entry_exists` pre-check already prevents duplicate writes.
