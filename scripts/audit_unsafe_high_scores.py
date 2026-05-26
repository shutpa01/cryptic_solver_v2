"""Audit clues that would be unsafe to present as HIGH.

HIGH is safe only for a proven WFW proof or explicit manual approval.  This
script reports legacy high-confidence rows that lack that proof.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = ROOT / "data" / "clues_master.db"


def audit(source, puzzle, db_path=CLUES_DB):
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT c.id, c.clue_number, c.direction, c.clue_text, c.answer,
                      c.wordplay_type, c.definition, se.confidence,
                      se.model_version,
                      wfw.status AS wfw_status
               FROM clues c
               LEFT JOIN structured_explanations se ON se.clue_id = c.id
               LEFT JOIN (
                   SELECT clue_id, MAX(id) AS latest_id
                   FROM wfw_proof_attempts
                   GROUP BY clue_id
               ) latest_wfw ON latest_wfw.clue_id = c.id
               LEFT JOIN wfw_proof_attempts wfw ON wfw.id = latest_wfw.latest_id
               WHERE c.source = ? AND c.puzzle_number = ?
               ORDER BY
                 CASE WHEN c.direction = 'across' THEN 0 ELSE 1 END,
                 CAST(c.clue_number AS INTEGER),
                 c.clue_number""",
            (source, str(puzzle)),
        ).fetchall()
    finally:
        conn.close()

    unsafe = []
    for row in rows:
        confidence = row["confidence"]
        if confidence is None:
            continue
        score = confidence * 100 if confidence <= 1 else confidence
        if score < 70:
            continue
        if row["wfw_status"] == "wfw_proven":
            continue
        if row["model_version"] == "manual_approve":
            continue
        unsafe.append(row)
    return unsafe


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--puzzle", required=True)
    args = parser.parse_args(argv)
    rows = audit(args.source, args.puzzle)
    print("Unsafe legacy HIGH audit: %s #%s" % (args.source, args.puzzle))
    print("unsafe_high: %d" % len(rows))
    for row in rows:
        print(
            "  %s%s %s: confidence=%s model=%s wfw=%s | %s" % (
                row["clue_number"],
                (row["direction"] or "?")[0].lower(),
                row["answer"],
                row["confidence"],
                row["model_version"],
                row["wfw_status"] or "none",
                row["clue_text"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
