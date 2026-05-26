"""Group WFW review gates for one puzzle without writing to the database."""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_wfw_obase_compatibility import audit as compatibility_audit
from signature_solver.db import RefDB
from signature_solver.wfw_gap_collector import collect_wfw_gaps
from signature_solver.wfw_unified_solver import solve_wfw_unified


def audit(source, puzzle):
    findings = compatibility_audit(source, puzzle)
    review_items = [
        item for item in findings
        if item["compatibility"] == "mechanical_review"
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        db = RefDB()

    rows = []
    reason_counts = Counter()
    for item in review_items:
        result = solve_wfw_unified(item["clue_text"], item["answer"], db=db)
        gaps = collect_wfw_gaps(
            item["clue_text"], item["answer"], db=db, wfw_result=result)
        gap_rows = []
        for gap in gaps:
            reasons = tuple(reason for reason in gap.reason.split(";")
                            if reason)
            for reason in reasons:
                reason_counts[reason] += 1
            gap_rows.append({
                "word": gap.word,
                "value": gap.value,
                "reasons": reasons,
            })
        rows.append({
            **item,
            "gaps": gap_rows,
        })
    return rows, reason_counts


def print_report(rows, reason_counts, source, puzzle):
    print("WFW review-gate audit: %s #%s" % (source, puzzle))
    print("review_cases: %d" % len(rows))
    print("")
    print("Gate counts:")
    for reason, count in reason_counts.most_common():
        print("  %d %s" % (count, reason))
    print("")
    print("Review cases:")
    for item in rows:
        print(
            "  {label} {answer}: WFW={wfw_operation} "
            "[{wfw_confidence}]".format(**item)
        )
        for gap in item["gaps"]:
            print("    %s -> %s [%s]" % (
                gap["word"],
                gap["value"],
                ";".join(gap["reasons"]),
            ))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--puzzle", required=True)
    args = parser.parse_args(argv)
    rows, reason_counts = audit(args.source, args.puzzle)
    print_report(rows, reason_counts, args.source, args.puzzle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
