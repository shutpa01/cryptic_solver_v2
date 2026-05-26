"""Regression tests for the WFW proof-gated record."""

import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.wfw_proof import build_wfw_proof_from_obase


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "clues_master.db")


def _load_clue(clue_number, direction):
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """SELECT c.clue_text, c.answer, c.definition, c.ai_explanation,
                      se.definition_text, se.components
               FROM clues c
               JOIN structured_explanations se ON se.clue_id = c.id
               WHERE c.source = 'telegraph'
                 AND c.puzzle_number = '31243'
                 AND c.clue_number = ?
                 AND c.direction = ?""",
            (clue_number, direction),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise AssertionError("missing Telegraph 31243 %s %s fixture" % (
            clue_number, direction))
    return row


def run_tests():
    declare = _load_clue("17", "across")
    record = build_wfw_proof_from_obase(
        declare["clue_text"],
        declare["answer"],
        json.loads(declare["components"]),
        ai_explanation=declare["ai_explanation"],
        definition_text=declare["definition_text"] or declare["definition"],
    )
    assert record.status == "wfw_proven"
    assert record.proof_attempt.status == "proven"
    assert "".join(
        placement.answer_letter
        for placement in record.proof_attempt.placements
    ) == "DECLARE"

    realise = _load_clue("15", "across")
    container = build_wfw_proof_from_obase(
        realise["clue_text"],
        realise["answer"],
        json.loads(realise["components"]),
        ai_explanation=realise["ai_explanation"],
        definition_text=realise["definition_text"] or realise["definition"],
    )
    assert container.status == "wfw_proven"
    assert container.proof_attempt.status == "proven"
    assert container.proof_attempt.transformations[0].operation == "container"
    assert [
        placement.source_block_id
        for placement in container.proof_attempt.placements
    ] == [
        "obase_piece_0_base",
        "obase_piece_1_base",
        "obase_piece_1_base",
        "obase_piece_2_base",
        "obase_piece_0_base",
        "obase_piece_0_base",
        "obase_piece_0_base",
    ]

    selected = _load_clue("24", "across")
    anagram = build_wfw_proof_from_obase(
        selected["clue_text"],
        selected["answer"],
        json.loads(selected["components"]),
        ai_explanation=selected["ai_explanation"],
        definition_text=selected["definition_text"] or selected["definition"],
    )
    assert anagram.status == "wfw_proven"
    assert anagram.proof_attempt.status == "proven"
    assert anagram.proof_attempt.transformations[0].operation == "anagram"
    assert "".join(
        placement.answer_letter
        for placement in anagram.proof_attempt.placements
    ) == "SELECTED"
    assert anagram.proof_attempt.placements[4].source_block_id == (
        "obase_piece_1_base")

    ideas = _load_clue("3", "down")
    hidden = build_wfw_proof_from_obase(
        ideas["clue_text"],
        ideas["answer"],
        json.loads(ideas["components"]),
        ai_explanation=ideas["ai_explanation"],
        definition_text=ideas["definition_text"] or ideas["definition"],
    )
    assert hidden.status == "wfw_proven"
    assert hidden.proof_attempt.status == "proven"
    assert hidden.proof_attempt.transformations[0].operation == "hidden"
    assert "".join(
        placement.answer_letter
        for placement in hidden.proof_attempt.placements
    ) == "IDEAS"
    assert hidden.coverage["uncovered"] == []

    data = record.as_dict()
    assert data["atom_context"]["answer_text"] == "DECLARE"
    assert data["proof_attempt"]["status"] == "proven"

    print("WFW proof-gate regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
