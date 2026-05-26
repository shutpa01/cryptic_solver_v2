"""Regression test for WFW-native human correction records."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.db import RefDB
from signature_solver.wfw_display_adapter import display_from_wfw_proof_attempt
from signature_solver.wfw_manual_correction import build_manual_wfw_correction


def run_tests():
    db = RefDB()
    correction = build_manual_wfw_correction(
        "Sit across top mount",
        "BESTRIDE",
        definition_span=(0, 2),
        pieces=[
            {"clue_span": (2, 3), "value": "BEST", "mechanism": "synonym"},
            {"clue_span": (3, 4), "value": "RIDE", "mechanism": "synonym"},
        ],
        db=db,
    )

    assert correction["status"] == "wfw_manual_proven_with_db_gaps"
    assert correction["definition"]["text"] == "Sit across"
    assert [block["text"] for block in correction["working_blocks"]] == [
        "top", "mount"]
    assert [block["value"] for block in correction["working_blocks"]] == [
        "BEST", "RIDE"]
    assert "".join(
        placement["answer_letter"] for placement in correction["placements"]
    ) == "BESTRIDE"
    assert {
        "type": "definition",
        "definition": "Sit across",
        "answer": "BESTRIDE",
    } in correction["missing_enrichments"]
    assert {
        "type": "synonym",
        "word": "mount",
        "synonym": "RIDE",
    } in correction["missing_enrichments"]
    assert not any(
        item.get("word") == "top" and item.get("synonym") == "BEST"
        for item in correction["missing_enrichments"]
    )
    display = display_from_wfw_proof_attempt({
        "id": 1,
        "status": "wfw_proven",
        "proof_source": "manual_wfw_correction",
        "proof": correction,
    })
    assert display["status"] == "wfw_proven"
    assert display["operations"][0]["detail"] == "BEST + RIDE = BESTRIDE"
    assert [block["text"] for block in display["blocks"]] == [
        "Sit across", "top", "mount"]
    assert [link["letter"] for link in display["answer_links"]] == list(
        "BESTRIDE")
    assert display["answer_links"][0]["source_text"] == "top"
    assert display["answer_links"][4]["source_text"] == "mount"

    print("WFW manual correction regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
