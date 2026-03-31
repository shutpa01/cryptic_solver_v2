"""Backfill TFTT explanations for times puzzles missing them.

Fetches from timesforthetimes.co.uk, writes explanation + definition
to clues_master.db for each matched clue.

Usage:
    python scripts/backfill_tftt.py                    # all missing
    python scripts/backfill_tftt.py --limit 10         # test run
    python scripts/backfill_tftt.py --dry-run          # fetch + parse, no DB writes
"""

import argparse
import os
import re
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scraper", "timesforthetimes"))

from sonnet_pipeline.tftt_pipeline import fetch_tftt
from sonnet_pipeline.solver import clean

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
RATE_LIMIT = 2.5  # seconds between requests


def get_missing_puzzles(cutoff='2021-03-29'):
    """Get times puzzles with answers but no explanations."""
    conn = sqlite3.connect(CLUES_DB, timeout=30)
    rows = conn.execute('''
        SELECT puzzle_number, COUNT(*) as total
        FROM clues
        WHERE source = 'times'
        AND publication_date >= ?
        AND CAST(strftime("%w", publication_date) AS INTEGER) BETWEEN 1 AND 5
        AND answer IS NOT NULL AND answer != ''
        GROUP BY puzzle_number
        HAVING SUM(CASE WHEN explanation IS NOT NULL AND explanation != '' THEN 1 ELSE 0 END) = 0
        ORDER BY CAST(puzzle_number AS INTEGER)
    ''', (cutoff,)).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def write_explanations(puzzle_number, tftt_clues, dry_run=False):
    """Match TFTT clues to DB clues by answer, write explanation + definition."""
    conn = sqlite3.connect(CLUES_DB, timeout=30)
    conn.row_factory = sqlite3.Row

    db_clues = conn.execute('''
        SELECT id, clue_number, direction, answer
        FROM clues WHERE source = 'times' AND puzzle_number = ?
    ''', (str(puzzle_number),)).fetchall()

    # Build lookup by clean answer
    db_by_answer = {}
    for c in db_clues:
        ans = clean(c["answer"]) if c["answer"] else ""
        if ans:
            db_by_answer[ans] = c

    matched = 0
    for tc in tftt_clues:
        tc_answer = clean(tc.get("answer", ""))
        if not tc_answer:
            continue

        db_clue = db_by_answer.get(tc_answer)
        if not db_clue:
            continue

        explanation = tc.get("explanation", "")
        definition = tc.get("definition", "")

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
    parser = argparse.ArgumentParser(description="Backfill TFTT explanations")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("BACKFILL TFTT EXPLANATIONS")
    print("=" * 60)

    puzzles = get_missing_puzzles()
    if args.limit:
        puzzles = puzzles[:args.limit]

    print(f"  {len(puzzles)} puzzles to process")

    total_matched = 0
    total_fetched = 0
    not_found = 0

    for i, (pnum, clue_count) in enumerate(puzzles):
        print(f"  [{i+1}/{len(puzzles)}] Times #{pnum} ({clue_count} clues)...", end="", flush=True)

        tftt_clues = fetch_tftt(int(pnum))

        if tftt_clues:
            matched = write_explanations(pnum, tftt_clues, dry_run=args.dry_run)
            total_fetched += 1
            total_matched += matched
            label = "would match" if args.dry_run else "matched"
            print(f" {len(tftt_clues)} fetched, {matched} {label}")
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
