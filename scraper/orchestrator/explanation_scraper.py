#!/usr/bin/env python3
"""Explanation Scraper Orchestrator

Runs explanation scrapers that enrich existing clue rows with explanations and definitions.
Typically run in the afternoon after puzzles have been solved and blogged about.

Usage:
    python explanation_scraper.py              # Run all explanation scrapers
    python explanation_scraper.py --only tftt  # Run only Times for the Times
"""

import argparse
import subprocess
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notify import send_failure_email, send_summary_email

BASE_PATH = Path(__file__).resolve().parent.parent  # scraper/
PROJECT_ROOT = BASE_PATH.parent
PYTHON = str(PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe')
CLUES_MASTER_DB = PROJECT_ROOT / 'data' / 'clues_master.db'

# Explanation scraper definitions
SCRAPERS = {
    'tftt': {
        'name': 'Times for the Times',
        'script': BASE_PATH / 'timesforthetimes' / 'timesforthetimes_scraper.py',
        'args': ['--daily'],
        'timeout': 120,
    },
    'fifteensquared': {
        'name': 'Fifteen Squared',
        'script': BASE_PATH / 'fifteensquared' / 'fifteensquared_scraper.py',
        'args': ['--daily'],
        'timeout': 180,
    },
    'bigdave': {
        'name': 'Big Dave 44',
        'script': BASE_PATH / 'bigdave' / 'bigdave_scraper.py',
        'args': ['--daily'],
        'timeout': 180,
    },
}


def run_scraper(name: str, display_name: str, script: Path, args: list, timeout: int) -> bool:
    """Run a scraper subprocess. Returns True on success."""
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {display_name} ({script.name})")
    print(f"{'=' * 60}")

    if not script.exists():
        print(f"  Script not found: {script}")
        return False

    cmd = [PYTHON, str(script)] + args

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(script.parent),
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.stdout.close()
        except Exception:
            pass
        try:
            process.stderr.close()
        except Exception:
            pass
        print(f"  TIMEOUT: {display_name} exceeded {timeout}s")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    if stdout:
        print(stdout)
    if stderr:
        print("STDERR:", stderr)

    return process.returncode == 0


def get_explanation_stats() -> str:
    """Gather and print explanation coverage statistics. Returns stats as string."""
    if not CLUES_MASTER_DB.exists():
        print("  Master DB not found")
        return ""

    conn = sqlite3.connect(CLUES_MASTER_DB)
    cursor = conn.cursor()

    lines = []

    cursor.execute("SELECT COUNT(*) FROM clues")
    total = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM clues
        WHERE explanation IS NOT NULL AND explanation != ''
    """)
    explained = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(*) FROM clues
        WHERE definition IS NOT NULL AND definition != ''
    """)
    defined = cursor.fetchone()[0]

    lines.append("Explanation coverage:")
    lines.append(f"  Total clues:      {total:,}")
    if total:
        lines.append(f"  With explanation: {explained:,} ({100*explained/total:.1f}%)")
        lines.append(f"  With definition:  {defined:,} ({100*defined/total:.1f}%)")

    # Per-source stats
    cursor.execute("""
        SELECT source, COUNT(*) as total,
               SUM(CASE WHEN explanation IS NOT NULL AND explanation != '' THEN 1 ELSE 0 END) as explained
        FROM clues GROUP BY source ORDER BY total DESC
    """)
    lines.append("\nBy source:")
    for row in cursor.fetchall():
        pct = (100 * row[2] / row[1]) if row[1] else 0
        lines.append(f"  {row[0]:15} {row[2]:>7,} / {row[1]:>7,} ({pct:.1f}%)")

    conn.close()

    stats = '\n'.join(lines)
    print(f"\n{stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description='Run explanation scrapers')
    parser.add_argument('--only', choices=list(SCRAPERS.keys()),
                        help='Run only this scraper')
    args = parser.parse_args()

    print(f"Explanation Scraper — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.only:
        scrapers_to_run = {args.only: SCRAPERS[args.only]}
    else:
        scrapers_to_run = SCRAPERS

    results = {}
    for key, config in scrapers_to_run.items():
        success = run_scraper(
            key, config['name'], config['script'], config['args'], config['timeout']
        )
        results[key] = success

        if not success:
            send_failure_email(config['name'], f"{config['name']} scraper failed or timed out")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for key, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {SCRAPERS[key]['name']:25} {status}")

    # Use display names for email
    display_results = {SCRAPERS[k]['name']: v for k, v in results.items()}
    stats = get_explanation_stats()

    send_summary_email("Explanation Scraper", display_results, stats)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
