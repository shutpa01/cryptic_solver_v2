"""Backfill fifteensquared explanations for guardian/independent puzzles.

Fetches from fifteensquared.net, writes explanation + definition
to clues_master.db for each matched clue.

Usage:
    python scripts/backfill_fifteensquared.py                         # all missing
    python scripts/backfill_fifteensquared.py --source guardian       # one source
    python scripts/backfill_fifteensquared.py --limit 10              # test run
    python scripts/backfill_fifteensquared.py --dry-run               # fetch + parse, no DB writes
"""

import argparse
import os
import re
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scraper", "fifteensquared"))

from sonnet_pipeline.fifteensquared_pipeline import fetch_fifteensquared
from sonnet_pipeline.solver import clean

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
RATE_LIMIT = 2.5


def get_missing_puzzles(source_filter=None, cutoff='2021-03-29'):
    """Get guardian/independent puzzles with answers but no explanations."""
    conn = sqlite3.connect(CLUES_DB, timeout=30)

    sources = []
    if source_filter:
        sources = [source_filter]
    else:
        sources = ['guardian', 'independent']

    all_puzzles = []
    for source in sources:
        rows = conn.execute('''
            SELECT puzzle_number, publication_date, COUNT(*) as total
            FROM clues
            WHERE source = ?
            AND publication_date >= ?
            AND answer IS NOT NULL AND answer != ''
            GROUP BY puzzle_number
            HAVING SUM(CASE WHEN explanation IS NOT NULL AND explanation != '' THEN 1 ELSE 0 END) = 0
            ORDER BY CAST(puzzle_number AS INTEGER)
        ''', (source, cutoff)).fetchall()
        for r in rows:
            all_puzzles.append((source, r[0], r[1], r[2]))

    conn.close()
    return all_puzzles


def write_explanations(source, puzzle_number, fs_clues, dry_run=False):
    """Match fifteensquared clues to DB clues by answer, write explanation + definition."""
    conn = sqlite3.connect(CLUES_DB, timeout=30)
    conn.row_factory = sqlite3.Row

    db_clues = conn.execute('''
        SELECT id, clue_number, direction, answer
        FROM clues WHERE source = ? AND puzzle_number = ?
    ''', (source, str(puzzle_number))).fetchall()

    db_by_answer = {}
    for c in db_clues:
        ans = clean(c["answer"]) if c["answer"] else ""
        if ans:
            db_by_answer[ans] = c

    matched = 0
    for fc in fs_clues:
        fc_answer = clean(fc.get("answer", ""))
        if not fc_answer:
            continue

        db_clue = db_by_answer.get(fc_answer)
        if not db_clue:
            continue

        explanation = fc.get("explanation", "")
        definition = fc.get("definition", "")

        if not explanation and not definition:
            continue

        if not dry_run:
            updates = []
            params = []
            if explanation:
                updates.append("explanation = ?")
                params.append(explanation)
            if definition:
                updates.append("definition = COALESCE(NULLIF(definition, ''), ?)")
                params.append(definition)

            if updates:
                params.append(db_clue["id"])
                conn.execute(
                    "UPDATE clues SET %s WHERE id = ?" % ", ".join(updates),
                    params
                )

        matched += 1

    if not dry_run:
        conn.commit()
    conn.close()
    return matched


def main():
    parser = argparse.ArgumentParser(description="Backfill fifteensquared explanations")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", type=str, default=None, choices=['guardian', 'independent'])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("BACKFILL FIFTEENSQUARED EXPLANATIONS")
    print("=" * 60)

    puzzles = get_missing_puzzles(args.source)
    if args.limit:
        puzzles = puzzles[:args.limit]

    print(f"  {len(puzzles)} puzzles to process")

    total_matched = 0
    total_fetched = 0
    not_found = 0

    for i, (source, pnum, pub_date, clue_count) in enumerate(puzzles):
        print(f"  [{i+1}/{len(puzzles)}] {source} #{pnum} ({clue_count} clues)...", end="", flush=True)

        fs_clues = fetch_fifteensquared(int(pnum), source, pub_date)

        if fs_clues:
            matched = write_explanations(source, pnum, fs_clues, dry_run=args.dry_run)
            total_fetched += 1
            total_matched += matched
            label = "would match" if args.dry_run else "matched"
            print(f" {len(fs_clues)} fetched, {matched} {label}")
        else:
            not_found += 1
            print(f" not found")

        time.sleep(RATE_LIMIT)

    print(f"\nDone:")
    print(f"  Fetched: {total_fetched}")
    print(f"  Not found: {not_found}")
    print(f"  Total clues {'would be ' if args.dry_run else ''}matched: {total_matched}")


if __name__ == "__main__":
    main()
