"""
run_all.py — Run all enrichment scripts in sequence.

Usage:
    python -m enrichment.run_all              # Full run (writes to DB)
    python -m enrichment.run_all --dry-run    # Preview only
"""

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

from enrichment.common import CRYPTIC_DB

SCRIPTS = [
    '01_mine_times_notation',
    '03_mine_indicators_from_tags',
    '04_mine_definition_pairs',
]

TABLES = [
    'wordplay',
    'indicators',
    'definition_answers',
    'definition_answers_augmented',
    'synonyms_pairs',
]


def get_row_counts(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    counts = {}
    for table in TABLES:
        try:
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.OperationalError:
            counts[table] = 0
    conn.close()
    return counts


def main():
    parser = argparse.ArgumentParser(description='Run all enrichment scripts')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    extra_args = ['--dry-run'] if args.dry_run else []
    mode = "[DRY RUN] " if args.dry_run else ""

    print(f"\n{'=' * 60}")
    print(f"{mode}Enrichment Pipeline")
    print(f"{'=' * 60}")
    print(f"Database: {CRYPTIC_DB}")

    # Row counts before
    before = get_row_counts(str(CRYPTIC_DB))
    print(f"\nTable row counts BEFORE:")
    for table, count in before.items():
        print(f"  {table:35s}: {count:>10,}")

    # Run each script
    for script in SCRIPTS:
        print(f"\n{'-' * 60}")
        print(f"Running {script}...")
        print(f"{'-' * 60}")

        result = subprocess.run(
            [sys.executable, '-m', f'enrichment.{script}'] + extra_args,
            capture_output=False,
        )
        if result.returncode != 0:
            print(f"  ERROR: {script} failed with exit code {result.returncode}")

    # Row counts after
    after = get_row_counts(str(CRYPTIC_DB))

    print(f"\n{'=' * 60}")
    print(f"{mode}ENRICHMENT SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n{'Table':35s} {'Before':>10s} {'After':>10s} {'Change':>10s}")
    print(f"{'-' * 67}")
    for table in TABLES:
        b = before[table]
        a = after[table]
        change = a - b
        sign = '+' if change > 0 else ''
        print(f"  {table:33s} {b:>10,} {a:>10,} {sign}{change:>9,}")
    print()


if __name__ == '__main__':
    main()
