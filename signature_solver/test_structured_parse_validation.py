"""Direct validation tests for structured parse reversal and anagram."""

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
    errors = _validate_structured_parse(
        _reversal_parse("TRAP", "TRAP", "PART", "TRAP"), "TRAP"
    )
    assert errors == [], "valid reversal should pass: %s" % errors

    errors = _validate_structured_parse(
        _reversal_parse("TRAP", "TRAP", "TARP", "TRAP"), "TRAP"
    )
    assert any("not the reverse" in e for e in errors), (
        "wrong fodder should fail: %s" % errors
    )

    errors = _validate_structured_parse(
        _reversal_parse("TRAP", "TRAP", "PART", "PRAT"), "TRAP"
    )
    assert any("piece" in e and "letters" in e for e in errors), (
        "result != piece letters should fail: %s" % errors
    )

    parse = _reversal_parse("TRAP", "TRAP", "PART", "TRAP")
    parse["operations"][0]["fodder"] = ""
    errors = _validate_structured_parse(parse, "TRAP")
    assert any("fodder" in e for e in errors), (
        "missing fodder should fail: %s" % errors
    )

    errors = _validate_structured_parse(
        _reversal_parse("TRAP", "TR", "RT", "TR", boxes=[1, 2]), "TRAP"
    )
    assert any("cover" in e or "box" in e for e in errors), (
        "partial coverage reversal should fail: %s" % errors
    )

    errors = _validate_structured_parse(
        _anagram_parse("TEAM", "TEAM", "MATE", "TEAM"), "TEAM"
    )
    assert errors == [], "valid anagram should pass: %s" % errors

    errors = _validate_structured_parse(
        _anagram_parse("TEAM", "TEAM", "MATES", "TEAM"), "TEAM"
    )
    assert any("not an anagram" in e for e in errors), (
        "wrong letter count should fail: %s" % errors
    )

    errors = _validate_structured_parse(
        _anagram_parse("TEAM", "TEAM", "MAZE", "TEAM"), "TEAM"
    )
    assert any("not an anagram" in e for e in errors), (
        "wrong letters should fail: %s" % errors
    )

    errors = _validate_structured_parse(
        _anagram_parse("TEAM", "TEAM", "MATE", "MATE"), "TEAM"
    )
    assert any("piece" in e and "letters" in e for e in errors), (
        "result != piece letters should fail: %s" % errors
    )

    errors = _validate_structured_parse(
        _anagram_parse("TEAM", "TE", "ET", "TE", boxes=[1, 2]), "TEAM"
    )
    assert any("cover" in e or "box" in e for e in errors), (
        "partial coverage anagram should fail: %s" % errors
    )

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

    pa = {
        "version": 1,
        "clue_id": 1,
        "answer": "PA",
        "source": "human",
        "confidence": "verified",
        "definition": {
            "id": "def1",
            "clue_text": "answer",
            "clue_word_positions": [1],
            "answer": "PA",
        },
        "pieces": [
            {
                "id": "piece1",
                "clue_text": "Quiet",
                "clue_word_positions": [0],
                "relationship": "abbreviation",
                "letters": "P",
                "answer_boxes": [1],
                "mapping": "positional",
                "colour": "blue",
            },
            {
                "id": "piece2",
                "clue_text": "a",
                "clue_word_positions": [2],
                "relationship": "literal_letters",
                "letters": "A",
                "answer_boxes": [2],
                "mapping": "positional",
                "colour": "pink",
            },
        ],
        "transform_pieces": [],
        "operations": [],
        "filler": [],
    }
    pa_entries = _structured_parse_db_entries(pa)
    abbrev_entries = [e for e in pa_entries if e["type"] == "abbreviation"]
    assert len(abbrev_entries) == 1, (
        "expected one abbreviation entry for PA: %s" % abbrev_entries
    )
    assert abbrev_entries[0]["value"] == "P", (
        "charade audit must use piece.letters P: %s" % abbrev_entries
    )

    swallow = {
        "version": 1,
        "clue_id": 1,
        "answer": "SWALLOW",
        "source": "human",
        "confidence": "verified",
        "definition": {
            "id": "def1",
            "clue_text": "Bird,",
            "clue_word_positions": [0],
            "answer": "SWALLOW",
        },
        "pieces": [
            {
                "id": "piece1",
                "clue_text": "female",
                "clue_word_positions": [2],
                "relationship": "synonym",
                "letters": "SOW",
                "answer_boxes": [1, 6, 7],
                "mapping": "positional",
                "colour": "blue",
            },
            {
                "id": "piece2",
                "clue_text": "fence",
                "clue_word_positions": [4],
                "relationship": "synonym",
                "letters": "WALL",
                "answer_boxes": [2, 3, 4, 5],
                "mapping": "positional",
                "colour": "pink",
            },
        ],
        "transform_pieces": [],
        "operations": [{
            "id": "op1",
            "type": "container",
            "clue_text": "going over",
            "clue_word_positions": [3, 4],
            "outer_piece_id": "piece1",
            "inner_piece_id": "piece2",
            "result": "SWALLOW",
            "colour": "blue",
        }],
        "filler": [],
    }
    errors = _validate_structured_parse(swallow, "SWALLOW")
    assert errors == [], "SWALLOW container should still pass: %s" % errors

    errors = _validate_structured_parse(pa, "PA")
    assert errors == [], "PA charade should still pass: %s" % errors

    pa_bad = {**pa, "pieces": [pa["pieces"][0]]}
    errors = _validate_structured_parse(pa_bad, "PA")
    assert any("not cover" in e or "box" in e for e in errors), (
        "missing box coverage should fail: %s" % errors
    )

    pa_with_filler = {
        **pa,
        "filler": [{
            "id": "fill1",
            "clue_text": "with",
            "clue_word_positions": [3],
            "role": "link",
        }],
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
