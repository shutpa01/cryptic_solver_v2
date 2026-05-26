"""Export atomic review items into the existing DB-gap review format.

The output is intentionally compatible with ``sonnet_pipeline.review_gaps``.
Atomic failures often need a human to choose the actual enrichment, so these
rows start as type ``review`` and carry the full clue context. In the review
tool choose ``e`` to edit the suggestion into a synonym, abbreviation,
definition, indicator, or homophone before accepting it.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLUES_DB = ROOT / "data" / "clues_master.db"
OUTPUT_DIR = ROOT / "documents"


def main():
    parser = argparse.ArgumentParser(
        description="Export open atomic review items for enrichment review.")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--source")
    parser.add_argument("--puzzle-number")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    try:
        path, count = export_atomic_review_gaps(
            run_id=args.run_id,
            source=args.source,
            puzzle_number=args.puzzle_number,
            output_dir=args.output_dir,
        )
    except NoAtomicReviewItems as exc:
        raise SystemExit(str(exc))

    print("Wrote %s (%d atomic review item%s)" % (
        path, count, "" if count == 1 else "s"))
    print("Review with:")
    print("  .\\.venv\\Scripts\\python.exe -m sonnet_pipeline.review_gaps %s" % path)
    return 0


class NoAtomicReviewItems(Exception):
    pass


def export_atomic_review_gaps(
        run_id=None, source=None, puzzle_number=None, output_dir=OUTPUT_DIR):
    args = argparse.Namespace(
        run_id=run_id,
        source=source,
        puzzle_number=puzzle_number,
        output_dir=str(output_dir),
    )
    rows = _fetch_rows(args)
    if not rows:
        raise NoAtomicReviewItems("No open atomic review items found.")

    source = args.source or rows[0]["source"] or "atomic"
    puzzle = args.puzzle_number or rows[0]["puzzle_number"] or "unknown"
    run_id = args.run_id or rows[0]["run_id"]

    gaps = [_gap_from_row(row) for row in rows]
    data = {
        "source": source,
        "puzzle": str(puzzle),
        "stats": {
            "total": len(rows),
            "assembled": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "failed": len(rows),
            "avg_score": 0,
        },
        "gaps": gaps,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "run%s" % run_id if run_id is not None else "open"
    path = out_dir / (
        "pending_gaps_atomic_%s_%s_%s.json" % (source, puzzle, suffix))
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path, len(gaps)


def _fetch_rows(args):
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        where = ["ri.status = 'open'"]
        params = []
        if args.run_id is not None:
            where.append("ri.run_id = ?")
            params.append(args.run_id)
        if args.source:
            where.append("c.source = ?")
            params.append(args.source)
        if args.puzzle_number:
            where.append("c.puzzle_number = ?")
            params.append(str(args.puzzle_number))
        return list(conn.execute(
            """SELECT ri.id AS review_id, ri.run_id, ri.review_type,
                      ri.summary, ri.payload_json,
                      c.id AS clue_id, c.source, c.puzzle_number,
                      c.clue_number, c.direction, c.clue_text, c.answer
               FROM atomic_parse_review_items ri
               JOIN clues c ON c.id = ri.clue_id
               WHERE %s
               ORDER BY c.direction, CAST(c.clue_number AS INTEGER),
                        c.clue_number, ri.id""" % " AND ".join(where),
            params,
        ))
    finally:
        conn.close()


def _gap_from_row(row):
    payload = {}
    if row["payload_json"]:
        try:
            payload = json.loads(row["payload_json"])
        except json.JSONDecodeError:
            payload = {}
    note_parts = [
        "Atomic review #%s: %s" % (row["review_id"], row["summary"]),
    ]
    reason = payload.get("reason")
    if reason:
        note_parts.append("Reason: %s" % reason)
    annotation_summary = payload.get("annotation_summary") or {}
    counts = annotation_summary.get("counts") or {}
    if counts:
        compact = ", ".join(
            "%s=%s" % (key, counts[key])
            for key in sorted(counts)
            if key
        )
        note_parts.append("Evidence tokens: %s" % compact)

    return {
        "type": "review",
        "word": "",
        "letters": "",
        "answer": row["answer"],
        "clue": row["clue_text"],
        "clue_id": row["clue_id"],
        "clue_number": row["clue_number"],
        "direction": row["direction"],
        "score": payload.get("solver_confidence") or 0,
        "source": row["source"],
        "puzzle_number": row["puzzle_number"],
        "note": " | ".join(note_parts),
        "atomic_review": {
            "review_id": row["review_id"],
            "run_id": row["run_id"],
            "review_type": row["review_type"],
            "payload": payload,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
