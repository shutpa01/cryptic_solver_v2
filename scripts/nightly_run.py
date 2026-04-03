"""Nightly automated run: scrape → solve DT + Daily Mail → sync honeypot.

Designed to run at 2am UTC via Windows Task Scheduler.
Solves Telegraph and Daily Mail puzzles only (not Times/Guardian/Independent
which rely on blog explanations).

Usage:
    python scripts/nightly_run.py              # full run
    python scripts/nightly_run.py --dry-run    # show what would run, don't execute
"""

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = ROOT / "data" / "clues_master.db"
PYTHON_PIPELINE = r"C:\Users\shute\PycharmProjects\AI_Solver\.venv\Scripts\python.exe"
PYTHON_SCRAPER = str(ROOT / ".venv" / "Scripts" / "python.exe")
SCRAPER_SCRIPT = str(ROOT / "scraper" / "orchestrator" / "puzzle_scraper.py")
LOG_DIR = ROOT / "logs"

# Sources to solve (not Times/Guardian/Independent — they use blog explanations)
SOLVE_SOURCES = ["telegraph", "dailymail"]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def run_scraper():
    """Run the puzzle scraper to fetch today's puzzles."""
    log("Running scraper...")
    result = subprocess.run(
        [PYTHON_SCRAPER, SCRAPER_SCRIPT],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    if result.returncode != 0:
        log(f"Scraper failed (exit {result.returncode})")
        if result.stderr:
            log(f"  stderr: {result.stderr[-500:]}")
        return False
    log("Scraper completed")
    return True


def find_todays_puzzles(target_date):
    """Find puzzle numbers for today's DT and Daily Mail."""
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    puzzles = []
    for source in SOLVE_SOURCES:
        rows = conn.execute(
            """SELECT DISTINCT puzzle_number FROM clues
               WHERE source = ? AND publication_date = ?
                 AND answer IS NOT NULL AND answer != ''""",
            (source, target_date),
        ).fetchall()
        for r in rows:
            puzzles.append((source, r[0]))
    conn.close()
    return puzzles


def run_pipeline(source, puzzle_number):
    """Run the Sonnet pipeline on a single puzzle."""
    log(f"Running pipeline: {source} #{puzzle_number}")
    cmd = [
        PYTHON_PIPELINE, "-m", "sonnet_pipeline.run",
        "--mode", "1",
        "--no-review",
        "--write-db",
        "--source", source,
        str(puzzle_number),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,  # 30 min max per puzzle
    )
    if result.returncode != 0:
        log(f"  Pipeline failed (exit {result.returncode})")
        if result.stderr:
            log(f"  stderr: {result.stderr[-300:]}")
        return False

    # Extract summary from output
    output = result.stdout or ""
    for line in output.splitlines()[-10:]:
        if "HIGH" in line or "solved" in line.lower() or "cost" in line.lower():
            log(f"  {line.strip()}")
    log(f"  Pipeline completed for {source} #{puzzle_number}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Nightly scrape + solve")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD), default: today")
    parser.add_argument("--skip-scraper", action="store_true", help="Skip scraper, just run pipeline")
    args = parser.parse_args()

    target_date = args.date or date.today().isoformat()

    # Ensure log directory exists
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"nightly_{target_date}.log"

    log("=" * 60)
    log(f"NIGHTLY RUN — {target_date}")
    log("=" * 60)

    # Step 1: Scrape
    if not args.skip_scraper:
        if args.dry_run:
            log("[DRY RUN] Would run scraper")
        else:
            if not run_scraper():
                log("Scraper failed — continuing to pipeline (puzzles may already exist)")

    # Step 2: Find today's puzzles
    puzzles = find_todays_puzzles(target_date)
    if not puzzles:
        log(f"No puzzles found for {target_date} in {SOLVE_SOURCES}")
        return

    log(f"Found {len(puzzles)} puzzle(s) to solve:")
    for source, pnum in puzzles:
        log(f"  {source} #{pnum}")

    # Step 3: Run pipeline on each
    results = []
    for source, pnum in puzzles:
        if args.dry_run:
            log(f"[DRY RUN] Would run pipeline on {source} #{pnum}")
            results.append((source, pnum, True))
        else:
            ok = run_pipeline(source, pnum)
            results.append((source, pnum, ok))

    # Summary
    log("")
    log("=" * 60)
    successes = sum(1 for _, _, ok in results if ok)
    failures = len(results) - successes
    log(f"DONE — {successes} succeeded, {failures} failed")
    for source, pnum, ok in results:
        log(f"  {'OK' if ok else 'FAIL'}: {source} #{pnum}")
    log("=" * 60)


if __name__ == "__main__":
    main()
