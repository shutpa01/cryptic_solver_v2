"""Read-only audit of strict atomic parse coverage for one puzzle."""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signature_solver.clue_context import build_clue_context  # noqa: E402
from signature_solver.db import RefDB  # noqa: E402
from signature_solver.solver import solve_clue  # noqa: E402
from signature_solver.token_parse_assembler import assemble_token_parses  # noqa: E402


CLUES_DB = ROOT / "data" / "clues_master.db"


def main():
    parser = argparse.ArgumentParser(
        description="Audit atomic parser coverage for a puzzle. Read-only.")
    parser.add_argument("--source", default="telegraph")
    parser.add_argument("--puzzle-number", required=True)
    args = parser.parse_args()

    rows = _fetch_puzzle(args.source, args.puzzle_number)
    if not rows:
        raise SystemExit("No clues found for %s %s" % (
            args.source, args.puzzle_number))

    db = RefDB()
    results = []
    for row in rows:
        results.append(_audit_row(row, db))

    _print_report(args.source, args.puzzle_number, results)
    return 0


def _fetch_puzzle(source, puzzle_number):
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute(
            """SELECT id, clue_number, direction, clue_text, answer,
                      definition, wordplay_type
               FROM clues
               WHERE source = ? AND puzzle_number = ?
               ORDER BY direction, CAST(clue_number AS INTEGER), clue_number""",
            (source, puzzle_number),
        ))
    finally:
        conn.close()


def _audit_row(row, db):
    sr = solve_clue(row["clue_text"], row["answer"], db)
    token_parses = list(getattr(sr, "token_parses", []) or [])

    # If solve_clue did not carry a context/parse, run the strict assembler
    # directly on the canonical context. This is still read-only.
    if not token_parses:
        context = build_clue_context(
            row["clue_text"], row["answer"], db, annotate=True)
        token_parses = assemble_token_parses(context)
    else:
        context = getattr(sr, "clue_context", None)

    operation = token_parses[0].operation if token_parses else ""
    reason = "atomic_ok" if token_parses else _classify_gap(row, sr, context)
    return {
        "id": row["id"],
        "ref": "%s%s" % (
            row["clue_number"],
            (row["direction"] or "?")[0].upper()),
        "answer": row["answer"],
        "clue": row["clue_text"],
        "stored_type": row["wordplay_type"] or "",
        "solver_confidence": getattr(sr, "confidence", 0),
        "solver_high": bool(getattr(sr, "high_confidence", False)),
        "atomic": bool(token_parses),
        "operation": operation,
        "reason": reason,
    }


def _classify_gap(row, solve_result, context):
    if context is None:
        return "no_context"
    if not context.definition_candidates:
        return "missing_definition_span"
    if not getattr(solve_result, "result", None):
        return "solver_no_candidate"
    if getattr(solve_result, "high_confidence", False):
        return "old_solved_no_atomic_parse"
    annotations = context.annotations
    if not any(ann.token.endswith("_I") or ann.token.startswith("POS_I_")
               for ann in annotations):
        return "missing_indicator_or_whole_clue_type"
    if not any(ann.token in ("SYN_F", "ABR_F", "HOM_F", "POS_F")
               for ann in annotations):
        return "missing_source_value"
    return "assembly_or_coverage_rejected"


def _print_report(source, puzzle_number, results):
    total = len(results)
    atomic = sum(1 for r in results if r["atomic"])
    high = sum(1 for r in results if r["solver_high"])

    print("Atomic parse audit: %s %s" % (source, puzzle_number))
    print("Total clues: %d" % total)
    print("Solver HIGH: %d/%d" % (high, total))
    print("Atomic parses: %d/%d" % (atomic, total))
    print()
    print("Operation buckets:")
    for op, count in Counter(
            r["operation"] or "(none)" for r in results).most_common():
        print("  %-24s %2d" % (op, count))
    print()
    print("Failure buckets:")
    for reason, count in Counter(
            r["reason"] for r in results if not r["atomic"]).most_common():
        print("  %-32s %2d" % (reason, count))
    print()
    print("Clues:")
    print("%-4s %-16s %-5s %-24s %-32s %s" % (
        "Ref", "Answer", "Atom", "Operation", "Reason", "Clue"))
    for r in results:
        print("%-4s %-16s %-5s %-24s %-32s %s" % (
            r["ref"],
            r["answer"][:16],
            "yes" if r["atomic"] else "no",
            r["operation"] or "-",
            r["reason"],
            r["clue"],
        ))


if __name__ == "__main__":
    raise SystemExit(main())
