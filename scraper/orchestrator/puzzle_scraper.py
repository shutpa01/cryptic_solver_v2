#!/usr/bin/env python3
"""Puzzle Scraper Orchestrator

Runs all newspaper puzzle scrapers. Each scraper writes directly to the clues table
in clues_master.db — no sync step needed.

After scrapers run, reconciles against the expected puzzle schedule to report
exactly which puzzles were captured and which are missing.

Usage:
    python puzzle_scraper.py                # Run all scrapers
    python puzzle_scraper.py --only guardian    # Run only guardian
    python puzzle_scraper.py --only telegraph   # Run only telegraph
    python puzzle_scraper.py --only times       # Run only times
    python puzzle_scraper.py --only independent # Run only independent
"""

import argparse
import re
import subprocess
import sqlite3
import sys
import time
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notify import send_failure_email, _send_email

BASE_PATH = Path(__file__).resolve().parent.parent  # scraper/
PROJECT_ROOT = BASE_PATH.parent
PYTHON = str(PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe')
CLUES_MASTER_DB = PROJECT_ROOT / 'data' / 'clues_master.db'

# ── Expected puzzle schedule ──────────────────────────────────────────────
# Each entry: label, source, puzzle_number range (lo, hi), days of week
# Days: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
EXPECTED_PUZZLES = [
    # Telegraph — daily cryptic is 31xxx, toughie+prize cryptic share 2400-4000 (different days)
    ('Telegraph Cryptic',       'telegraph', 27000, 35000, [0, 1, 2, 3, 4, 5]),
    ('Telegraph Toughie',       'telegraph', 2400,  4000,  [1, 2, 3, 4]),
    ('Telegraph Prize Cryptic', 'telegraph', 2400,  4000,  [6]),
    ('Telegraph Prize Toughie', 'telegraph', 1,     500,   [6]),
    # Times
    ('Times Cryptic',           'times',     26000, 32000, [0, 1, 2, 3, 4, 5]),
    ('Sunday Times Cryptic',    'times',     4700,  6000,  [6]),
    # Guardian — cryptic Mon-Fri, prize Sat (both use 29xxx numbers). Quiptic/Everyman discontinued.
    ('Guardian Cryptic',        'guardian',  21000, 32000, [0, 1, 2, 3, 4]),
    ('Guardian Prize',          'guardian',  21000, 32000, [5]),
]

# Scraper definitions: (name, script_path, args, timeout_seconds)
SCRAPERS = {
    'telegraph': {
        'script': BASE_PATH / 'telegraph' / 'telegraph_daily.py',
        'args': [],
        'timeout': 300,  # Browser scraper — needs more time
        'retries': 2,    # Browser scrapers get a retry for transient failures
    },
    'times': {
        'script': BASE_PATH / 'times' / 'times_all.py',
        'args': [],
        'timeout': 300,  # Browser scraper
        'retries': 2,    # Browser scrapers get a retry for transient failures
    },
    'guardian': {
        'script': BASE_PATH / 'guardian' / 'guardian_all.py',
        'args': [],
        'timeout': 120,  # API scraper — fast
        'retries': 1,
    },
    'independent': {
        'script': BASE_PATH / 'independent' / 'independent_all.py',
        'args': [],
        'timeout': 120,  # HTTP scraper — fast
        'retries': 1,
    },
}

DANWORD_SCRIPT = BASE_PATH / 'danword' / 'danword_lookup.py'
DANWORD_TIMEOUT = 600  # ~30 clues × 8s each + overhead


def run_scraper(name: str, script: Path, args: list, timeout: int) -> bool:
    """Run a scraper subprocess. Returns True on success."""
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {name} ({script.name})")
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
        print(f"  TIMEOUT: {name} exceeded {timeout}s")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    if stdout:
        print(stdout)
    if stderr:
        print("STDERR:", stderr)

    return process.returncode == 0


def reconcile(today: date | None = None) -> tuple[list[tuple[str, str, str]], int, int]:
    """Check DB for today's expected puzzles. Returns (results, found, expected).

    Each result is (label, status, detail) where status is 'OK' or 'MISSING'.
    """
    if today is None:
        today = date.today()
    dow = today.weekday()
    today_str = today.isoformat()

    if not CLUES_MASTER_DB.exists():
        return [], 0, 0

    conn = sqlite3.connect(CLUES_MASTER_DB)
    cursor = conn.cursor()

    results = []
    found = 0
    expected = 0

    for label, source, num_lo, num_hi, days in EXPECTED_PUZZLES:
        if dow not in days:
            continue

        expected += 1

        # Check for a puzzle in this series with today's date
        cursor.execute("""
            SELECT puzzle_number, COUNT(*) as clue_count
            FROM clues
            WHERE source = ? AND publication_date = ?
              AND CAST(puzzle_number AS INTEGER) BETWEEN ? AND ?
            GROUP BY puzzle_number
            ORDER BY CAST(puzzle_number AS INTEGER) DESC
            LIMIT 1
        """, (source, today_str, num_lo, num_hi))
        row = cursor.fetchone()

        if row:
            found += 1
            results.append((label, 'OK', f'#{row[0]} ({row[1]} clues)'))
        else:
            # Check most recent puzzle in this series (for context)
            cursor.execute("""
                SELECT puzzle_number, publication_date
                FROM clues
                WHERE source = ? AND CAST(puzzle_number AS INTEGER) BETWEEN ? AND ?
                GROUP BY puzzle_number
                ORDER BY publication_date DESC
                LIMIT 1
            """, (source, num_lo, num_hi))
            last = cursor.fetchone()
            if last:
                results.append((label, 'MISSING', f'last was #{last[0]} on {last[1]}'))
            else:
                results.append((label, 'MISSING', 'no puzzles in DB'))

    conn.close()
    return results, found, expected


def get_stats() -> str:
    """Gather and print clues table statistics. Returns stats as string."""
    if not CLUES_MASTER_DB.exists():
        print("  Master DB not found")
        return ""

    conn = sqlite3.connect(CLUES_MASTER_DB)
    cursor = conn.cursor()

    lines = []

    cursor.execute("SELECT COUNT(*) FROM clues")
    total = cursor.fetchone()[0]
    lines.append(f"Total clues: {total:,}")

    cursor.execute("""
        SELECT source, COUNT(*) as cnt
        FROM clues
        GROUP BY source
        ORDER BY cnt DESC
    """)
    lines.append("\nBy source:")
    for row in cursor.fetchall():
        lines.append(f"  {row[0]}: {row[1]:,}")

    cursor.execute("""
        SELECT COUNT(*) FROM clues
        WHERE publication_date >= date('now', '-7 days')
    """)
    recent = cursor.fetchone()[0]
    lines.append(f"\nAdded in last 7 days: {recent:,}")

    conn.close()

    stats = '\n'.join(lines)
    print(f"\n{stats}")
    return stats


def get_puzzle_snapshot() -> dict[str, set[str]]:
    """Snapshot current puzzles in DB, keyed by source -> set of puzzle_numbers."""
    if not CLUES_MASTER_DB.exists():
        return {}
    conn = sqlite3.connect(CLUES_MASTER_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT source, puzzle_number FROM clues GROUP BY source, puzzle_number")
    snapshot = {}
    for source, pnum in cursor.fetchall():
        snapshot.setdefault(source, set()).add(pnum)
    conn.close()
    return snapshot


def diff_snapshots(before: dict[str, set[str]], after: dict[str, set[str]]) -> list[tuple[str, str, int]]:
    """Return list of (source, puzzle_number, clue_count) for newly added puzzles."""
    if not CLUES_MASTER_DB.exists():
        return []
    new_puzzles = []
    conn = sqlite3.connect(CLUES_MASTER_DB)
    cursor = conn.cursor()
    for source in after:
        old = before.get(source, set())
        for pnum in sorted(after[source] - old, key=lambda x: int(x) if x.isdigit() else 0):
            cursor.execute(
                "SELECT COUNT(*) FROM clues WHERE source = ? AND puzzle_number = ?",
                (source, pnum))
            count = cursor.fetchone()[0]
            new_puzzles.append((source, pnum, count))
    conn.close()
    return new_puzzles


def find_answerless_puzzles(today: date | None = None) -> list[tuple[str, str, int]]:
    """Find today's puzzles where ALL clues have empty answers (prize puzzles).

    Returns list of (source, puzzle_number, clue_count) for puzzles needing
    danword backfill.
    """
    if today is None:
        today = date.today()
    today_str = today.isoformat()

    if not CLUES_MASTER_DB.exists():
        return []

    conn = sqlite3.connect(CLUES_MASTER_DB)
    cursor = conn.cursor()

    # Find puzzles published today where every clue is answerless
    cursor.execute("""
        SELECT source, puzzle_number, COUNT(*) as total,
               SUM(CASE WHEN answer IS NULL OR answer = '' THEN 1 ELSE 0 END) as missing
        FROM clues
        WHERE publication_date = ?
          AND source IN ('telegraph', 'times')
        GROUP BY source, puzzle_number
        HAVING missing = total AND total > 0
        ORDER BY source, puzzle_number
    """, (today_str,))

    results = [(row[0], row[1], row[2]) for row in cursor.fetchall()]
    conn.close()
    return results


def run_danword_backfill(puzzles: list[tuple[str, str, int]]) -> list[tuple[str, str, int, int]]:
    """Run danword lookup for each answerless puzzle.

    Returns list of (source, puzzle_number, found, total) results.
    """
    if not DANWORD_SCRIPT.exists():
        print(f"  Danword script not found: {DANWORD_SCRIPT}")
        return []

    results = []
    for source, puzzle_number, clue_count in puzzles:
        print(f"\n  Danword: {source} #{puzzle_number} ({clue_count} clues)")

        cmd = [PYTHON, str(DANWORD_SCRIPT),
               '--source', source, '--puzzle', str(puzzle_number)]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(DANWORD_SCRIPT.parent),
        )

        try:
            stdout, stderr = process.communicate(timeout=DANWORD_TIMEOUT)
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
            print(f"    TIMEOUT after {DANWORD_TIMEOUT}s")
            results.append((source, puzzle_number, 0, clue_count))
            continue

        if stdout:
            print(stdout)
        if stderr:
            print("    STDERR:", stderr[:500])

        # Parse found/total from output (last line: "Done: X/Y answers written")
        found = 0
        for line in (stdout or '').splitlines():
            m = re.search(r'Done:\s*(\d+)/(\d+)', line)
            if m:
                found = int(m.group(1))
                break

        results.append((source, puzzle_number, found, clue_count))

    return results


def main():
    parser = argparse.ArgumentParser(description='Run puzzle scrapers')
    parser.add_argument('--only', choices=list(SCRAPERS.keys()),
                        help='Run only this scraper')
    args = parser.parse_args()

    print(f"Puzzle Scraper — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.only:
        scrapers_to_run = {args.only: SCRAPERS[args.only]}
    else:
        scrapers_to_run = SCRAPERS

    # Snapshot DB before scraping
    before = get_puzzle_snapshot()

    results = {}
    for name, config in scrapers_to_run.items():
        max_attempts = config.get('retries', 1)
        success = False

        for attempt in range(1, max_attempts + 1):
            success = run_scraper(name, config['script'], config['args'], config['timeout'])
            if success:
                break
            if attempt < max_attempts:
                print(f"  {name} failed on attempt {attempt}/{max_attempts}, retrying in 10s...")
                time.sleep(10)
            else:
                print(f"  {name} failed after {max_attempts} attempt(s)")

        results[name] = success

        if not success:
            send_failure_email(name, f"{name} scraper failed after {max_attempts} attempt(s)")

    # Snapshot DB after scraping
    after = get_puzzle_snapshot()
    new_puzzles = diff_snapshots(before, after)

    # Summary
    print(f"\n{'=' * 60}")
    print("SCRAPER RESULTS")
    print(f"{'=' * 60}")
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name:15} {status}")

    # New puzzles added
    print(f"\n{'=' * 60}")
    print("NEW PUZZLES ADDED")
    print(f"{'=' * 60}")
    if new_puzzles:
        for source, pnum, count in new_puzzles:
            print(f"  {source:15} #{pnum:>8}  ({count} clues)")
    else:
        print("  None")

    # Danword backfill for answerless (prize) puzzles
    danword_results = []
    answerless = find_answerless_puzzles()
    if answerless:
        print(f"\n{'=' * 60}")
        print(f"DANWORD BACKFILL ({len(answerless)} answerless puzzles)")
        print(f"{'=' * 60}")
        danword_results = run_danword_backfill(answerless)
        for source, pnum, found, total in danword_results:
            status = f"{found}/{total} answers"
            print(f"  {source:15} #{pnum:>8}  {status}")
    else:
        print(f"\n  No answerless puzzles to backfill today")

    # Reconcile against expected schedule
    print(f"\n{'=' * 60}")
    print("PUZZLE RECONCILIATION")
    print(f"{'=' * 60}")
    recon_results, recon_found, recon_expected = reconcile()
    for label, status, detail in recon_results:
        mark = '+' if status == 'OK' else 'X'
        print(f"  [{mark}] {label:28} {detail}")
    print(f"\n  Result: {recon_found}/{recon_expected} expected puzzles captured")

    stats = get_stats()

    # Build email body with reconciliation front and centre
    missing = [r for r in recon_results if r[1] == 'MISSING']

    if missing:
        subject = f"Puzzle Scraper: {len(missing)} MISSING ({recon_found}/{recon_expected})"
    else:
        subject = f"Puzzle Scraper: all {recon_expected} puzzles OK"

    email_lines = [
        f"Date: {datetime.now().strftime('%A %Y-%m-%d %H:%M')}",
        "",
        "NEW PUZZLES ADDED",
        "-" * 40,
    ]
    if new_puzzles:
        for source, pnum, count in new_puzzles:
            email_lines.append(f"  {source:15} #{pnum:>8}  ({count} clues)")
    else:
        email_lines.append("  None")

    if danword_results:
        email_lines.append("")
        email_lines.append("DANWORD BACKFILL")
        email_lines.append("-" * 40)
        for source, pnum, found, total in danword_results:
            email_lines.append(f"  {source:15} #{pnum:>8}  {found}/{total} answers")

    email_lines.append("")
    email_lines.append("PUZZLE RECONCILIATION")
    email_lines.append("-" * 40)
    for label, status, detail in recon_results:
        mark = '+' if status == 'OK' else 'X'
        email_lines.append(f"  [{mark}] {label:28} {detail}")
    email_lines.append(f"\n  {recon_found}/{recon_expected} expected puzzles captured")

    email_lines.append("")
    email_lines.append("SCRAPER PROCESSES")
    email_lines.append("-" * 40)
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        email_lines.append(f"  {name:20} {status}")

    if stats:
        email_lines.append("")
        email_lines.append(stats)

    _send_email(subject, '\n'.join(email_lines))

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
