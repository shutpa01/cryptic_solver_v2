"""Regression tests for WFW proof persistence."""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.wfw_proof_store import (
    ensure_table,
    get_latest_wfw_proof_attempt,
    write_wfw_proof_attempt,
)


def run_tests():
    conn = sqlite3.connect(":memory:")
    ensure_table(conn)

    first = {
        "status": "wfw_review",
        "source": "obase_structured",
        "clue_text": "State education rejected, Bordeaux unfinished",
        "answer": "DECLARE",
        "objections": ["wrong_order"],
    }
    second = {
        "status": "wfw_proven",
        "source": "obase_structured",
        "clue_text": "State education rejected, Bordeaux unfinished",
        "answer": "DECLARE",
        "proof_attempt": {"status": "proven"},
        "objections": [],
    }

    first_id = write_wfw_proof_attempt(17, "dt", "31243", first, conn)
    second_id = write_wfw_proof_attempt(17, "dt", "31243", second, conn)

    assert first_id != second_id
    latest = get_latest_wfw_proof_attempt(17, conn)
    assert latest["id"] == second_id
    assert latest["status"] == "wfw_proven"
    assert latest["proof_source"] == "obase_structured"
    assert latest["proof"]["answer"] == "DECLARE"
    assert latest["proof"]["objections"] == []
    assert get_latest_wfw_proof_attempt(999, conn) is None

    print("WFW proof store regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
