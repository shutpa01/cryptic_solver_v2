"""Regression tests for stage-one context persistence."""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.clue_context import build_clue_context
from signature_solver.stage_context_store import (
    get_latest_stage_context,
    write_stage_context,
)


def run_tests():
    conn = sqlite3.connect(":memory:")
    ctx = build_clue_context(
        "Note Spanish male taken with a Mexican city",
        "TIJUANA",
        db=None,
        annotate=False,
        use_pos_model=False,
    )

    row_id = write_stage_context(
        123, "dailymail", "17883", ctx, status="built", conn=conn)
    assert row_id == 1

    stored = get_latest_stage_context(123, conn=conn)
    assert stored["clue_id"] == 123
    assert stored["source"] == "dailymail"
    assert stored["puzzle_number"] == "17883"
    assert stored["stage_name"] == "stage_01_context"
    assert stored["status"] == "built"

    context = stored["context"]
    assert context["clue_text"] == "Note Spanish male taken with a Mexican city"
    assert context["answer"] == "TIJUANA"
    assert context["atom_context"]["answer_text"] == "TIJUANA"
    assert len(context["atom_context"]["answer_atoms"]) == 7
    assert context["tokens"][0]["text"] == "Note"
    assert context["tokens"][0]["atom_ids"]
    assert context["tokens"][1]["text"] == "Spanish"

    print("Stage context store regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
