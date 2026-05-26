"""Regression tests for the version-4 WFW structural proof gate."""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.db import RefDB
from signature_solver.wfw_display_adapter import display_from_wfw_proof_attempt
from signature_solver.wfw_proof_validator import validate_unified_wfw_proof
from signature_solver.wfw_unified_proof import build_wfw_proof_from_unified_result
from signature_solver.wfw_unified_solver import solve_wfw_unified


def _proven_user_proof():
    db = RefDB()
    result = solve_wfw_unified(
        "American elector gutted seeing exploiter",
        "USER",
        db=db,
    )
    proof = build_wfw_proof_from_unified_result(result)
    assert proof["status"] == "wfw_proven"
    return proof


def run_tests():
    proof = _proven_user_proof()
    assert validate_unified_wfw_proof(proof) == ()

    unused_indicator = copy.deepcopy(proof)
    unused_indicator["token_parse"]["blocks"].append({
        "block_id": "cov_99",
        "kind": "OP_BLOCK",
        "span": [3, 4],
        "span_space": "full_clue_tokens",
        "text": "seeing",
        "token": "ANA_I",
        "value": None,
        "role": "anagram_indicator",
        "input_value": None,
    })
    objections = validate_unified_wfw_proof(unused_indicator)
    assert "v4_operation_block_not_used:seeing" in objections

    stale_attempt = {
        "id": 999,
        "status": "wfw_proven",
        "proof_source": "wfw_unified_solver",
        "proof": unused_indicator,
    }
    display = display_from_wfw_proof_attempt(stale_attempt)
    assert display["status"] == "wfw_review"
    assert "v4_operation_block_not_used:seeing" in display["objections"]

    missing_placement = copy.deepcopy(proof)
    missing_placement["assembly"]["placements"] = (
        missing_placement["assembly"]["placements"][:-1]
    )
    objections = validate_unified_wfw_proof(missing_placement)
    assert "v4_answer_positions_not_fully_covered" in objections

    unused_source = copy.deepcopy(proof)
    unused_source["token_parse"]["blocks"].append({
        "block_id": "src_99",
        "kind": "SOURCE_BLOCK",
        "span": [3, 4],
        "span_space": "full_clue_tokens",
        "text": "seeing",
        "token": "SYN_F",
        "value": "SEEING",
        "role": "piece_99",
        "input_value": None,
    })
    objections = validate_unified_wfw_proof(unused_source)
    assert "v4_source_block_not_used:seeing" in objections

    wrong_output = copy.deepcopy(proof)
    wrong_output["assembly"]["transformations"][0]["output_block_id"] = "wfw_src_0"
    objections = validate_unified_wfw_proof(wrong_output)
    assert "v4_operation_output_mismatch:gutted" in objections

    print("WFW proof validator regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
