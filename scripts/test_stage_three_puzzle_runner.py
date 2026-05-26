"""Regression test for the Stage Three puzzle dry-runner."""

import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_stage_three_puzzle import run_stage_three_puzzle


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")


def run_tests():
    before = _queue_counts()
    summary = run_stage_three_puzzle("dailymail", "17883")
    after = _queue_counts()
    assert before == after

    assert summary["source"] == "dailymail"
    assert summary["puzzle_number"] == "17883"
    assert summary["total"] == 32
    assert summary["pass"] + summary["review"] == summary["total"]
    assert (
        summary["mechanical_pass"] + summary["mechanical_review"]
        == summary["total"]
    )
    assert sum(summary["buckets"].values()) == summary["total"]
    assert summary["pending_enrichments"] == sum(
        len(clue["pending_enrichments"]) for clue in summary["clues"])
    assert summary["review_items"] == sum(
        len(clue["review_items"]) for clue in summary["clues"])

    tijuana = next(clue for clue in summary["clues"] if clue["ref"] == "10A")
    assert tijuana["answer"] == "TIJUANA"
    assert tijuana["mechanical_pass"] is True
    assert tijuana["bucket"] == "purpose_review"
    assert len(tijuana["pending_enrichments"]) == 0

    hasbeen = next(clue for clue in summary["clues"] if clue["ref"] == "13A")
    assert hasbeen["answer"] == "HASBEEN"
    assert hasbeen["bucket"] == "mechanical_review"
    assert any(
        row["word"] == "north London suburb"
        and row["letters"] == "HENDON"
        for row in hasbeen["pending_enrichments"]
    )

    sparrowhawk = next(clue for clue in summary["clues"] if clue["ref"] == "29A")
    assert sparrowhawk["answer"] == "SPARROWHAWK"
    assert not any(
        "disturbed bird" in row["word"]
        for row in sparrowhawk["pending_enrichments"]
    )

    json.dumps(summary, sort_keys=True)
    print("Stage Three puzzle runner contract passed")
    return True


def _queue_counts():
    conn = sqlite3.connect(CLUES_DB)
    try:
        return {
            "pending_enrichments": conn.execute(
                "SELECT COUNT(*) FROM pending_enrichments").fetchone()[0],
            "atomic_parse_review_items": conn.execute(
                "SELECT COUNT(*) FROM atomic_parse_review_items").fetchone()[0],
        }
    finally:
        conn.close()


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
