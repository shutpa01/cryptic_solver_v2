"""Run WFW-vs-obase compatibility across multiple puzzles."""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = ROOT / "data" / "clues_master.db"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_wfw_obase_compatibility import audit as audit_puzzle


def latest_puzzles(limit, min_solved=1, db_path=CLUES_DB):
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT source, puzzle_number, publication_date,
                      COUNT(*) AS clue_count,
                      SUM(CASE WHEN has_solution = 1 THEN 1 ELSE 0 END) AS solved_count,
                      MAX(id) AS latest_id
               FROM clues
               GROUP BY source, puzzle_number, publication_date
               HAVING solved_count >= ?
               ORDER BY publication_date DESC, latest_id DESC
               LIMIT ?""",
            (min_solved, limit),
        ).fetchall()
    finally:
        conn.close()
    return rows


def run_corpus(puzzles):
    totals = Counter()
    puzzle_rows = []
    regressions = []
    unsafe = []
    for puzzle in puzzles:
        source = puzzle["source"]
        number = puzzle["puzzle_number"]
        findings = audit_puzzle(source, number)
        counts = Counter(item["compatibility"] for item in findings)
        totals.update(counts)
        puzzle_rows.append((source, number, puzzle["publication_date"], counts))
        regressions.extend(
            (source, number, item)
            for item in findings
            if item["compatibility"] == "regression"
        )
        unsafe.extend(
            (source, number, item)
            for item in findings
            if item["compatibility"] == "obase_unsafe"
        )
    return totals, puzzle_rows, regressions, unsafe


def print_report(totals, puzzle_rows, regressions, unsafe):
    print("WFW/obase corpus audit")
    for key in ("covered", "mechanical_review", "obase_unsafe",
                "regression", "wfw_extra", "both_unsolved"):
        print("%s: %d" % (key, totals[key]))
    print("")
    print("By puzzle:")
    for source, number, date, counts in puzzle_rows:
        bits = ", ".join(
            "%s=%d" % (key, counts[key])
            for key in ("covered", "mechanical_review", "obase_unsafe",
                        "regression", "both_unsolved")
        )
        print("  %s #%s %s: %s" % (source, number, date, bits))
    print("")
    print("Regressions:")
    for source, number, item in regressions:
        print("  %s #%s %s %s: %s" % (
            source, number, item["label"], item["answer"],
            item["clue_text"],
        ))
    print("")
    print("Unsafe obase rows rejected by WFW:")
    for source, number, item in unsafe:
        print("  %s #%s %s %s: %s" % (
            source, number, item["label"], item["answer"],
            item["clue_text"],
        ))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--min-solved", type=int, default=1)
    args = parser.parse_args(argv)
    puzzles = latest_puzzles(args.limit, min_solved=args.min_solved)
    totals, puzzle_rows, regressions, unsafe = run_corpus(puzzles)
    print_report(totals, puzzle_rows, regressions, unsafe)
    return 0 if not regressions else 1


if __name__ == "__main__":
    raise SystemExit(main())
