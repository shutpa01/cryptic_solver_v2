"""Dry-run Stage Three proof gate for one puzzle.

This script does not write to the database.  It builds Stage Two case files,
runs the Stage Three verifier, and reports the review/enrichment actions that
an outer admin/run layer could later persist.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from signature_solver.db import RefDB  # noqa: E402
from signature_solver.stage_three_proof import PASS  # noqa: E402
from signature_solver.stage_three_proof import build_stage_three_proof  # noqa: E402
from signature_solver.stage_three_review_queue import (  # noqa: E402
    pending_enrichments_from_stage_three_proof,
    review_items_from_stage_three_proof,
)
from signature_solver.stage_three_write_layer import (  # noqa: E402
    write_stage_three_puzzle_results,
)
from signature_solver.stage_two_casefile import build_stage_two_casefile  # noqa: E402


CLUES_DB = ROOT / "data" / "clues_master.db"
MECHANICAL_CHECKS = {
    "definition_evidence",
    "answer_assembly",
    "source_evidence",
    "assembly_order",
    "operation_evidence",
    "operation_attachment",
    "mechanism_rules",
    "atomic_coverage",
    "span_integrity",
}
PURPOSE_CHECKS = {
    "word_purpose_coverage",
    "word_purpose_candidates",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Dry-run Stage Three proof gate for one puzzle.")
    parser.add_argument("--source", default="dailymail")
    parser.add_argument("--puzzle-number", "--puzzle", required=True)
    parser.add_argument(
        "--json", action="store_true",
        help="Emit JSON instead of a compact text report.")
    parser.add_argument(
        "--actions", action="store_true",
        help="Include per-clue review and pending-enrichment actions.")
    parser.add_argument(
        "--write", action="store_true",
        help="Persist Stage Three artifacts, review items, and DB enrichment queue rows.")
    args = parser.parse_args(argv)

    summary = run_stage_three_puzzle(args.source, args.puzzle_number)
    write_counts = None
    if args.write:
        write_counts = write_stage_three_puzzle_results(summary)
        summary["write_counts"] = write_counts
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_report(summary, show_actions=args.actions)
        if write_counts is not None:
            print("")
            print("Write counts:")
            for key, value in sorted(write_counts.items()):
                print("  %-24s %s" % (key, value))
    return 0


def run_stage_three_puzzle(source, puzzle_number, db=None):
    rows = _fetch_puzzle(source, puzzle_number)
    if not rows:
        raise ValueError("No clues found for %s %s" % (source, puzzle_number))

    db = db or RefDB()
    clue_results = []
    for row in rows:
        clue_results.append(_run_clue(row, db, source, puzzle_number))

    return {
        "source": source,
        "puzzle_number": str(puzzle_number),
        "total": len(clue_results),
        "buckets": dict(Counter(item["bucket"] for item in clue_results)),
        "pass": sum(1 for item in clue_results if item["status"] == PASS),
        "review": sum(1 for item in clue_results if item["status"] != PASS),
        "mechanical_pass": sum(
            1 for item in clue_results if item["mechanical_pass"]),
        "mechanical_review": sum(
            1 for item in clue_results if not item["mechanical_pass"]),
        "purpose_issue": sum(
            1 for item in clue_results if item["purpose_review"]),
        "purpose_review": sum(
            1 for item in clue_results if item["bucket"] == "purpose_review"),
        "db_enrichment_review": sum(
            1 for item in clue_results if item["pending_enrichments"]),
        "db_enrichment_bucket": sum(
            1 for item in clue_results
            if item["bucket"] == "db_enrichment_review"),
        "failed_checks": dict(Counter(
            check
            for item in clue_results
            for check in item["failed_checks"]
        )),
        "review_types": dict(Counter(
            review["review_type"]
            for item in clue_results
            for review in item["review_items"]
        )),
        "pending_enrichment_types": dict(Counter(
            pending["type"]
            for item in clue_results
            for pending in item["pending_enrichments"]
        )),
        "pending_enrichments": sum(
            len(item["pending_enrichments"]) for item in clue_results),
        "review_items": sum(
            len(item["review_items"]) for item in clue_results),
        "clues": clue_results,
    }


def _run_clue(row, db, source, puzzle_number):
    casefile = build_stage_two_casefile(row["clue_text"], row["answer"], db)
    proof = build_stage_three_proof(casefile)
    failed_checks = [
        check.name for check in proof.checks
        if check.status != PASS
    ]
    mechanical_failed = [
        check for check in failed_checks
        if check in MECHANICAL_CHECKS
    ]
    purpose_failed = [
        check for check in failed_checks
        if check in PURPOSE_CHECKS
    ]
    pending = pending_enrichments_from_stage_three_proof(
        proof,
        clue_text=row["clue_text"],
        source=source,
        puzzle_number=puzzle_number,
    )
    review = review_items_from_stage_three_proof(proof, clue_id=row["id"])
    bucket = _clue_bucket(
        proof.status,
        not mechanical_failed,
        bool(purpose_failed),
        bool(pending),
    )
    return {
        "clue_id": row["id"],
        "ref": "%s%s" % (
            row["clue_number"], (row["direction"] or "?")[0].upper()),
        "clue": row["clue_text"],
        "answer": row["answer"],
        "stage_one_context": casefile.stage_one_context.as_dict(),
        "stage_two_casefile": casefile.as_dict(),
        "stage_three_proof": proof.as_dict(),
        "status": proof.status,
        "failed_checks": failed_checks,
        "mechanical_pass": not mechanical_failed,
        "mechanical_failed": mechanical_failed,
        "purpose_review": bool(purpose_failed),
        "purpose_failed": purpose_failed,
        "bucket": bucket,
        "review_items": list(review),
        "pending_enrichments": list(pending),
    }


def _clue_bucket(status, mechanical_pass, purpose_review, has_pending):
    if status == PASS:
        return "final_pass"
    if not mechanical_pass:
        return "mechanical_review"
    if has_pending:
        return "db_enrichment_review"
    if purpose_review:
        return "purpose_review"
    return "review"


def _fetch_puzzle(source, puzzle_number):
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute(
            """SELECT id, clue_number, direction, clue_text, answer
               FROM clues
               WHERE source = ? AND puzzle_number = ?
               ORDER BY direction, CAST(clue_number AS INTEGER), clue_number""",
            (source, str(puzzle_number)),
        ))
    finally:
        conn.close()


def print_report(summary, show_actions=False):
    print("Stage Three dry run: %s %s" % (
        summary["source"], summary["puzzle_number"]))
    print("Final PASS: %d/%d" % (summary["pass"], summary["total"]))
    print("Final REVIEW: %d/%d" % (summary["review"], summary["total"]))
    print("Mechanical proof pass: %d/%d" % (
        summary["mechanical_pass"], summary["total"]))
    print("Mechanical proof review: %d/%d" % (
        summary["mechanical_review"], summary["total"]))
    print("Purpose/grammar issue present: %d/%d" % (
        summary["purpose_issue"], summary["total"]))
    print("Purpose-only next action: %d/%d" % (
        summary["purpose_review"], summary["total"]))
    print("DB enrichment issue present: %d/%d" % (
        summary["db_enrichment_review"], summary["total"]))
    print("DB-enrichment next action: %d/%d" % (
        summary["db_enrichment_bucket"], summary["total"]))
    print("Review items: %d" % summary["review_items"])
    print("Pending DB enrichments: %d" % summary["pending_enrichments"])
    print("")
    print("Buckets:")
    for name, count in sorted(summary["buckets"].items()):
        print("  %-24s %2d" % (name, count))
    print("")
    print("Failed checks:")
    for name, count in sorted(summary["failed_checks"].items()):
        print("  %-28s %2d" % (name, count))
    print("")
    print("Review action types:")
    for name, count in sorted(summary["review_types"].items()):
        print("  %-44s %2d" % (name, count))
    print("")
    print("Pending enrichment types:")
    for name, count in sorted(summary["pending_enrichment_types"].items()):
        print("  %-16s %2d" % (name, count))
    print("")
    print("%-4s %-16s %-8s %-20s %-5s %-7s %-7s %s" % (
        "Ref", "Answer", "Status", "Bucket", "Mech", "Review", "DB", "Clue"))
    for clue in summary["clues"]:
        print("%-4s %-16s %-8s %-20s %-5s %-7d %-7d %s" % (
            clue["ref"],
            clue["answer"][:16],
            clue["status"],
            clue["bucket"],
            "PASS" if clue["mechanical_pass"] else "REV",
            len(clue["review_items"]),
            len(clue["pending_enrichments"]),
            clue["clue"],
        ))
        if show_actions:
            _print_clue_actions(clue)


def _print_clue_actions(clue):
    for item in clue["review_items"]:
        payload = item.get("payload") or {}
        request = payload.get("request") or {}
        if request:
            detail = request.get("needed_evidence") or request.get("reason") or ""
            text = request.get("text") or ""
            print("      review: %s | %s | %s" % (
                item["review_type"], text, detail))
            continue
        check_names = [
            check.get("name") for check in payload.get("failed_checks") or []
            if check.get("name")
        ]
        if check_names:
            print("      review: %s | %s" % (
                item["review_type"], ", ".join(check_names)))
        else:
            print("      review: %s | %s" % (
                item["review_type"], item.get("summary") or ""))
    for item in clue["pending_enrichments"]:
        print("      db: %s | %s -> %s" % (
            item["type"], item["word"], item["letters"]))


if __name__ == "__main__":
    raise SystemExit(main())
