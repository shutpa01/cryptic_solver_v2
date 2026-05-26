"""Audit WFW DB gaps for one puzzle without writing to the database."""
from __future__ import annotations

import argparse
import contextlib
import io
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from signature_solver.db import RefDB
from signature_solver.wfw_gap_collector import collect_wfw_gaps


CLUES_DB = ROOT / "data" / "clues_master.db"


def load_clues(conn, source, puzzle):
    return conn.execute(
        """SELECT id, clue_number, direction, clue_text, answer
           FROM clues
           WHERE source = ? AND puzzle_number = ?
           ORDER BY
             CASE WHEN direction = 'across' THEN 0 ELSE 1 END,
             CAST(clue_number AS INTEGER),
             clue_number""",
        (source, str(puzzle)),
    ).fetchall()


def audit(source, puzzle, db_path=CLUES_DB):
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = load_clues(conn, source, puzzle)
    finally:
        conn.close()

    with contextlib.redirect_stdout(io.StringIO()):
        ref_db = RefDB()

    findings = []
    for row in rows:
        gaps = collect_wfw_gaps(row["clue_text"], row["answer"], db=ref_db)
        for gap in gaps:
            findings.append({
                "clue_id": row["id"],
                "clue_number": row["clue_number"],
                "direction": row["direction"],
                "clue_text": row["clue_text"],
                "answer": row["answer"],
                **gap.as_dict(),
            })
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--puzzle", required=True)
    args = parser.parse_args(argv)

    findings = audit(args.source, args.puzzle)
    print("WFW gap audit: %s #%s" % (args.source, args.puzzle))
    print("gaps: %d" % len(findings))
    for item in findings:
        print(
            "%s%s %s: %s -> %s [%s]" % (
                item["clue_number"],
                item["direction"][0].lower(),
                item["answer"],
                item["word"],
                item["value"],
                item["reason"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
