"""Regression tests proving solve_clue is WFW-native first."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.db import RefDB
from signature_solver.solver import solve_clue


def run_tests():
    db = RefDB()

    sr = solve_clue(
        "Search advanced into blacksmith's workshop", "FORAGE", db)
    assert sr.high_confidence
    assert sr.solver_authority == "wfw_unified"
    assert sr.result.signature == ["WFW_CONTAINER"]
    assert sr.wfw_unified_result.status == "solved"
    assert sr.wfw_assemblies[0].status == "materialised"
    assembly = sr.wfw_assemblies[0].as_dict()
    assert [block["value"] for block in assembly["working_blocks"]] == [
        "FORGE", "A", "FORAGE"]
    assert "".join(
        placement["answer_letter"] for placement in assembly["placements"]
    ) == "FORAGE"

    unsolved = solve_clue(
        "This deliberately has no obvious cryptic route", "XYZ", db)
    assert hasattr(unsolved, "wfw_unified_result")
    assert unsolved.solver_authority in {
        "legacy_with_wfw_evidence", "wfw_unified"}

    print("WFW live solver path regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
