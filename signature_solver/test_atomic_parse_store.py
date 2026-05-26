"""Regression tests for atomic parser artifact persistence."""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.atomic_parse_store import (
    ensure_table,
    get_latest_atomic_artifact,
    get_pipeline_state,
    upsert_pipeline_state,
    write_atomic_artifact,
)


def run_tests():
    conn = sqlite3.connect(":memory:")
    ensure_table(conn)

    artifact = {
        "status": "solved",
        "confidence": 90,
        "solver_version": "test",
        "clue_context": {"tokens": [{"text": "Set"}]},
        "annotations": [{"token": "SYN_F"}],
        "gt2_bundles": [{"operation": "charade"}],
        "token_parses": [{"operation": "charade"}],
        "wfw": [{"answer": "CODE", "tokens": [], "answer_links": []}],
    }
    first_id = write_atomic_artifact(
        123, "Set of expectations about lyric poem", "CODE", artifact, conn)
    second = dict(artifact)
    second["confidence"] = 95
    second_id = write_atomic_artifact(
        123, "Set of expectations about lyric poem", "CODE", second, conn)

    assert first_id != second_id
    latest = get_latest_atomic_artifact(123, conn)
    assert latest["id"] == second_id
    assert latest["confidence"] == 95
    assert latest["wfw"][0]["answer"] == "CODE"
    assert get_latest_atomic_artifact(999, conn) is None

    upsert_pipeline_state(
        123, "daily_mail", "17884",
        "Set of expectations about lyric poem", "CODE", artifact, conn)
    current = get_pipeline_state(123, conn)
    assert current["confidence"] == 90
    assert current["stage_one"]["tokens"][0]["text"] == "Set"

    updated = dict(artifact)
    updated["confidence"] = 99
    updated["clue_context"] = {"tokens": [{"text": "Updated"}]}
    upsert_pipeline_state(
        123, "daily_mail", "17884",
        "Set of expectations about lyric poem", "CODE", updated, conn)
    current = get_pipeline_state(123, conn)
    assert current["confidence"] == 99
    assert current["stage_one"]["tokens"][0]["text"] == "Updated"
    count = conn.execute(
        "SELECT COUNT(*) FROM clue_pipeline_state WHERE clue_id = 123"
    ).fetchone()[0]
    assert count == 1

    print("Atomic parse store regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
