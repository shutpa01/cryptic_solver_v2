"""Regression tests for Stage Three review-queue payloads."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.db import RefDB
from signature_solver.stage_three_proof import build_stage_three_proof
from signature_solver.stage_three_review_queue import (
    pending_enrichments_from_stage_three_proof,
    review_items_from_stage_three_proof,
)
from signature_solver.stage_two_casefile import build_stage_two_casefile


def run_tests():
    db = RefDB()
    tijuana = build_stage_three_proof(build_stage_two_casefile(
        "Note Spanish male taken with a Mexican city",
        "TIJUANA",
        db,
    ))
    items = review_items_from_stage_three_proof(tijuana, clue_id=10068257)
    assert all(item["clue_id"] == 10068257 for item in items)
    assert len(items) == 2
    assert any(
        item["review_type"] == "stage_three:mechanism_indicator_evidence"
        and item["payload"]["request"]["text"] == "taken"
        for item in items
    )
    assert any(
        item["review_type"] == "stage_three:grammar_separator_evidence"
        and item["payload"]["request"]["text"] == "with"
        for item in items
    )
    assert pending_enrichments_from_stage_three_proof(tijuana) == ()

    hasbeen = build_stage_three_proof(build_stage_two_casefile(
        "One no longer relevant base sadly in western half of north London suburb",
        "HASBEEN",
        db,
    ))
    hasbeen_items = review_items_from_stage_three_proof(hasbeen, clue_id=10068260)
    assert any(
        item["review_type"] == "stage_three:mechanical_proof"
        for item in hasbeen_items
    )
    assert not any(
        item["review_type"] == "stage_three:answer_assembly"
        for item in hasbeen_items
    )
    assert not any(
        item["review_type"] == "stage_three:definition_phrase_evidence"
        and item["payload"]["request"]["text"] == "One no longer relevant"
        for item in hasbeen_items
    )
    assert not any(
        item["review_type"] == "stage_three:conditional_source_evidence"
        and item["payload"]["request"]["text"] == "north London suburb"
        for item in hasbeen_items
    )
    assert not any(
        item["review_type"] == "stage_three:mechanism_indicator_evidence"
        for item in hasbeen_items
    )
    assert not any(
        item["review_type"] == "stage_three:conditional_source_gap"
        and item["payload"]["request"]["text"] == "north London suburb"
        for item in hasbeen_items
    )
    pending = pending_enrichments_from_stage_three_proof(
        hasbeen,
        source="dailymail",
        puzzle_number=17883,
    )
    assert any(
        row["type"] == "definition"
        and row["word"] == "One no longer relevant"
        and row["letters"] == "HASBEEN"
        for row in pending
    )
    assert any(
        row["type"] == "synonym"
        and row["word"] == "north London suburb"
        and row["letters"] == "HENDON"
        and row["source"] == "dailymail"
        and row["puzzle_number"] == 17883
        for row in pending
    )
    assert review_items_from_stage_three_proof({"schema": "other"}) == ()
    assert pending_enrichments_from_stage_three_proof({"schema": "other"}) == ()

    print("Stage Three review queue contract passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
