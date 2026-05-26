"""Build WFW proof attempts for one puzzle.

obase supplies structured proposals. WFW decides whether each proposal is
actually proved letter by letter.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = ROOT / "data" / "clues_master.db"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from signature_solver.wfw_proof import build_wfw_proof_from_obase
from signature_solver.wfw_proof_store import (
    ensure_table,
    write_wfw_proof_attempt,
)


def load_clues(conn, source, puzzle_number):
    rows = conn.execute(
        """SELECT c.id, c.source, c.puzzle_number, c.clue_number, c.direction,
                  c.clue_text, c.answer, c.definition, c.ai_explanation,
                  se.components, se.definition_text
           FROM clues c
           LEFT JOIN (
               SELECT clue_id, MAX(id) AS latest_id
               FROM structured_explanations
               GROUP BY clue_id
           ) latest ON latest.clue_id = c.id
           LEFT JOIN structured_explanations se ON se.id = latest.latest_id
           WHERE c.source = ?
             AND c.puzzle_number = ?
           ORDER BY
             CASE WHEN c.direction = 'across' THEN 0 ELSE 1 END,
             CAST(c.clue_number AS INTEGER),
             c.clue_number""",
        (source, str(puzzle_number)),
    ).fetchall()
    return rows


def build_proof_for_row(row):
    if not row["components"]:
        return None, "missing_structured_components"
    try:
        components = json.loads(row["components"])
    except json.JSONDecodeError:
        return None, "bad_structured_components_json"
    proof = build_wfw_proof_from_obase(
        row["clue_text"],
        row["answer"],
        components,
        ai_explanation=row["ai_explanation"] or "",
        definition_text=row["definition_text"] or row["definition"],
    )
    return proof, None


def run(source, puzzle_number, db_path=CLUES_DB, write=False):
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    counts = {
        "total": 0,
        "wfw_proven": 0,
        "wfw_review": 0,
        "missing_structured_components": 0,
        "bad_structured_components_json": 0,
        "written": 0,
    }
    try:
        if write:
            ensure_table(conn)
        rows = load_clues(conn, source, puzzle_number)
        for row in rows:
            counts["total"] += 1
            proof, error = build_proof_for_row(row)
            if error:
                counts[error] += 1
                continue
            counts[proof.status] += 1
            if write:
                write_wfw_proof_attempt(
                    row["id"],
                    row["source"],
                    row["puzzle_number"],
                    proof,
                    conn=conn,
                )
                counts["written"] += 1
        if write:
            conn.commit()
    finally:
        conn.close()
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--puzzle", required=True)
    parser.add_argument("--db", default=str(CLUES_DB))
    parser.add_argument(
        "--write",
        action="store_true",
        help="Save proof attempts. Without this, only prints counts.",
    )
    args = parser.parse_args(argv)
    counts = run(args.source, args.puzzle, Path(args.db), write=args.write)
    mode = "write" if args.write else "dry-run"
    print("WFW proof puzzle run (%s): %s #%s" % (
        mode, args.source, args.puzzle))
    for key in (
        "total",
        "wfw_proven",
        "wfw_review",
        "missing_structured_components",
        "bad_structured_components_json",
        "written",
    ):
        print("%s: %s" % (key, counts[key]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
