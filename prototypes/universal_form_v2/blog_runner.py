"""Run the mechanical blog parser across a sample of clues.

Reads (clue, answer, blog) tuples from `clues_master.db`, runs the parser,
records results to a JSON file. No DB writes.

Usage:
    python -m prototypes.universal_form_v2.blog_runner \
        --source times --limit 100 \
        --out prototypes/universal_form_v2/runs/blog_parse_times_100.json
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

from .blog_parser import parse_blog


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
                   clue_text, answer, explanation
            FROM clues WHERE {where}
            ORDER BY RANDOM() LIMIT ?""",
        params + [limit]).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run(clues):
    results = []
    statuses = Counter()
    patterns = Counter()
    wp_counts = Counter()
    for c in clues:
        pr = parse_blog(c["clue_text"], c["answer"], c["explanation"])
        d = pr.to_dict()
        d["clue_id"] = c["id"]
        d["source"] = c["source"]
        d["puzzle_number"] = c["puzzle_number"]
        d["clue_number"] = c["clue_number"]
        d["direction"] = c["direction"]
        d["clue_text"] = c["clue_text"]
        d["answer"] = c["answer"]
        d["blog_text"] = c["explanation"]
        results.append(d)
        statuses[d["status"]] += 1
        if d["pattern"]:
            patterns[d["pattern"]] += 1
        if d.get("wordplay_type"):
            wp_counts[d["wordplay_type"]] += 1
    return {
        "n": len(clues),
        "statuses": dict(statuses),
        "patterns": dict(patterns),
        "wordplay_types": dict(wp_counts),
        "per_clue": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="times")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--puzzle-range", default=None,
                     help="lo-hi range, e.g. 29500-29530")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pr = None
    if args.puzzle_range:
        lo, hi = args.puzzle_range.split("-")
        pr = (int(lo), int(hi))
    clues = fetch_clues(args.source, args.limit, pr)
    summary = run(clues)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2),
                              encoding="utf-8")
    print(f"n={summary['n']}")
    print(f"statuses: {summary['statuses']}")
    print(f"patterns: {summary['patterns']}")
    print(f"top wordplay types:")
    for wp, count in sorted(summary['wordplay_types'].items(),
                            key=lambda x: -x[1])[:15]:
        print(f"  {count:>3d}  {wp}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
