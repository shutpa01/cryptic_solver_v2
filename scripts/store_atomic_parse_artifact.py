"""Solve one clue and append its atomic parse artifact.

This writes only to signature_solver.atomic_parse_store's additive table.
It does not update clues, structured_explanations, or clue_word_roles.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signature_solver.atomic_parse_store import (  # noqa: E402
    artifact_from_solve_result,
    write_atomic_artifact,
)
from signature_solver.db import RefDB  # noqa: E402
from signature_solver.solver import solve_clue  # noqa: E402


CLUES_DB = ROOT / "data" / "clues_master.db"


def main():
    parser = argparse.ArgumentParser(
        description="Append one atomic parse artifact for a clue id.")
    parser.add_argument("--clue-id", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    row = _fetch_clue(args.clue_id)
    if row is None:
        raise SystemExit("No clue found for id %s" % args.clue_id)

    db = RefDB()
    sr = solve_clue(row["clue_text"], row["answer"], db)
    artifact = artifact_from_solve_result(sr)

    if args.dry_run:
        print(json.dumps({
            "clue_id": args.clue_id,
            "clue_text": row["clue_text"],
            "answer": row["answer"],
            "status": artifact["status"],
            "confidence": artifact["confidence"],
            "annotations": len(artifact["annotations"]),
            "gt2_bundles": len(artifact["gt2_bundles"]),
            "token_parses": len(artifact["token_parses"]),
            "wfw": len(artifact["wfw"]),
        }, indent=2, sort_keys=True))
        return 0

    artifact_id = write_atomic_artifact(
        args.clue_id, row["clue_text"], row["answer"], artifact)
    print("Wrote atomic_parse_artifacts.id=%s for clue_id=%s" % (
        artifact_id, args.clue_id))
    return 0


def _fetch_clue(clue_id):
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT id, clue_text, answer FROM clues WHERE id = ?",
            (clue_id,),
        ).fetchone()
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
