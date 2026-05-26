"""Regression tests for WFW-native gap collection."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.db import RefDB
from signature_solver.wfw_gap_collector import collect_wfw_gaps


def run_tests():
    db = RefDB()

    gaps = collect_wfw_gaps("Principal cable", "LEAD", db=db)
    gap_dicts = [gap.as_dict() for gap in gaps]
    assert any(
        gap["type"] == "synonym"
        and gap["word"].lower() == "cable"
        and gap["value"] == "LEAD"
        for gap in gap_dicts
    ), "reverse-only cable/LEAD evidence must surface as a DB gap"

    no_gaps = collect_wfw_gaps("Welcomes largely old set at work", "GREETS", db=db)
    assert not any(
        gap.word.lower() == "welcomes" and gap.value == "GREETS"
        for gap in no_gaps
    ), "direct definition evidence should not be reported as a gap"

    print("WFW gap collector regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
