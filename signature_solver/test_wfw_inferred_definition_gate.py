"""Regression tests for inferred-definition proof gating."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.db import RefDB
from signature_solver.wfw_unified_proof import build_wfw_proof_from_unified_result
from signature_solver.wfw_unified_solver import solve_wfw_unified


def run_tests():
    db = RefDB()
    result = solve_wfw_unified(
        "Useful for traditional dancers same ploy to vary",
        "MAYPOLES",
        db=db,
    )
    proof = build_wfw_proof_from_unified_result(result)
    assert proof["token_parse"], "mechanical WFW assembly should be preserved"
    assert proof["token_parse"]["confidence"] == (
        "mechanically_verified_inferred_definition")
    assert proof["status"] == "wfw_review"
    assert "definition_not_db_verified" in proof["objections"]

    surface_result = solve_wfw_unified(
        "Place attended by a group of police is calm",
        "PLACID",
        db=db,
    )
    surface_proof = build_wfw_proof_from_unified_result(surface_result)
    assert surface_proof["token_parse"], (
        "wordplay assembly should preserve licensed surface words")
    assert surface_proof["token_parse"]["confidence"] == (
        "mechanically_verified")
    assert surface_proof["status"] == "wfw_proven"
    assert not surface_proof["objections"]

    dd_result = solve_wfw_unified("Principal cable", "LEAD", db=db)
    dd_proof = build_wfw_proof_from_unified_result(dd_result)
    assert dd_proof["token_parse"], (
        "definition-candidate double definitions should be preserved")
    assert dd_proof["token_parse"]["operation"] == "double_definition"
    assert dd_proof["token_parse"]["confidence"] == (
        "mechanically_verified")
    assert dd_proof["status"] == "wfw_proven"
    assert not dd_proof["objections"]

    print("WFW inferred-definition gate regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
