"""Regression tests for treating obase as WFW evidence proposer."""

import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.wfw_atoms import build_wfw_atom_context
from signature_solver.wfw_obase_bridge import prove_obase_charade


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "clues_master.db")


def _load_31243_17a():
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """SELECT c.clue_text, c.answer, c.ai_explanation, se.components
               FROM clues c
               JOIN structured_explanations se ON se.clue_id = c.id
               WHERE c.source = 'telegraph'
                 AND c.puzzle_number = '31243'
                 AND c.clue_number = '17'
                 AND c.direction = 'across'"""
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise AssertionError("missing Telegraph 31243 17A fixture")
    return row


def run_tests():
    row = _load_31243_17a()
    ctx = build_wfw_atom_context(row["clue_text"], row["answer"])
    components = json.loads(row["components"])

    attempt = prove_obase_charade(
        ctx, components, ai_explanation=row["ai_explanation"])
    assert attempt.status == "proven"
    assert attempt.operation == "charade"
    assert [proposal.transform for proposal in attempt.proposals] == [
        "reversal", "trim_last"]
    assert [transform.operation for transform in attempt.transformations] == [
        "reversal", "trim_last"]
    assert "".join(p.answer_letter for p in attempt.placements) == "DECLARE"
    assert [p.answer_position for p in attempt.placements] == list(range(1, 8))

    reversal = attempt.transformations[0]
    assert reversal.controller_atom_ids
    trim = attempt.transformations[1]
    assert trim.controller_atom_ids
    assert trim.removed_char_atom_ids == (
        "obase_piece_1_base_char_0005",)

    wrong = dict(components)
    wrong["ai_pieces"] = list(reversed(wrong["ai_pieces"]))
    rejected = prove_obase_charade(
        ctx, wrong, ai_explanation=row["ai_explanation"])
    assert rejected.status == "rejected"
    assert "CLAREDE != DECLARE" in rejected.objections[0]

    print("WFW obase bridge regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
