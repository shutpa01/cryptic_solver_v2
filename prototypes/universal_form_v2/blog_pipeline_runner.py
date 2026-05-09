"""End-to-end blog pipeline runner.

Pipeline: blog -> word_role_mapper -> assembly_enumerator -> verifier.

Outputs a JSON per clue with the mapping, the assembled form (or none),
and the verifier verdict.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signature_solver.db import RefDB

from .word_role_mapper import map_clue_words
from .db_anchored_mapper import map_clue_words_db
from .assembly_enumerator import assemble
from .verifier import verify
from .renderer import wordplay_type


def fetch_clues(source: str, limit: int, puzzle_range=None):
    conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    conn.row_factory = sqlite3.Row
    where = "source = ? AND answer != '' AND answer IS NOT NULL " \
            "AND explanation IS NOT NULL AND explanation != ''"
    params = [source]
    if puzzle_range:
        lo, hi = puzzle_range
        where += " AND CAST(puzzle_number AS INTEGER) BETWEEN ? AND ?"
        params.extend([lo, hi])
    rows = conn.execute(
        f"""SELECT id, source, puzzle_number, clue_number, direction,
                   clue_text, answer, explanation, definition
            FROM clues WHERE {where}
            ORDER BY id LIMIT ?""",
        params + [limit]).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run_one(c, db):
    # db is the RefDB; we also need a raw sqlite connection for the mapper.
    # Reuse db.conn if present.
    raw_conn = getattr(db, "conn", None)
    if raw_conn is None:
        import sqlite3
        raw_conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    m = map_clue_words_db(
        c["clue_text"], c["answer"], c["explanation"], raw_conn,
        clue_definition=c.get("definition"),
    )
    a = assemble(m)
    verdict = None
    if a.form is not None:
        try:
            verdict = verify(a.form, c["clue_text"], db)
        except Exception as e:
            verdict = None
    pattern = a.pattern
    if not pattern and any(t.role == "piece" for t in m.tags):
        pattern = "no_assembly_found"
    elif not pattern:
        pattern = "no_pieces"
    return {
        "clue_id": c["id"],
        "source": c["source"],
        "puzzle_number": c["puzzle_number"],
        "clue_number": c["clue_number"],
        "direction": c["direction"],
        "clue_text": c["clue_text"],
        "answer": c["answer"],
        "blog": c["explanation"],
        "mapping": m.to_dict(),
        "assembly_pattern": pattern,
        "form": a.form.to_dict() if a.form else None,
        "wordplay_type": (wordplay_type(a.form) if a.form else None),
        "verdict": verdict.to_dict() if verdict else None,
        "notes": list(a.notes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="times")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--puzzle-range", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pr = None
    if args.puzzle_range:
        lo, hi = args.puzzle_range.split("-")
        pr = (int(lo), int(hi))
    clues = fetch_clues(args.source, args.limit, pr)
    db = RefDB(str(PROJECT_ROOT / "data" / "cryptic_new.db"))

    results = [run_one(c, db) for c in clues]

    pattern_counts = Counter(r["assembly_pattern"] for r in results)
    verdicts = Counter()
    for r in results:
        if r["verdict"]:
            verdicts[r["verdict"]["verdict"]] += 1
        elif r["form"] is None:
            verdicts["NO_FORM"] += 1
        else:
            verdicts["NO_VERDICT"] += 1

    summary = {
        "n": len(clues),
        "verdicts": dict(verdicts),
        "patterns": dict(pattern_counts),
        "per_clue": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2),
                              encoding="utf-8")
    print(f"n={summary['n']}")
    print(f"verdicts: {summary['verdicts']}")
    print(f"patterns:")
    for k, v in pattern_counts.most_common():
        print(f"  {v:>3d}  {k}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
