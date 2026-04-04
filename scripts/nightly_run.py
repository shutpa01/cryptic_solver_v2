"""Nightly automated run: scrape all → danword backfill → solve DT + Daily Mail.

Designed to run at 2am UTC via Windows Task Scheduler.

Flow:
  1. Run all scrapers (all sources, all days)
  2. Danword backfill for any puzzles with missing answers
  3. Pipeline for DT + Daily Mail (weekdays only)
  4. Times/Guardian/Independent are skipped — run manually after blog answers available

Usage:
    python scripts/nightly_run.py              # full run
    python scripts/nightly_run.py --dry-run    # show what would run, don't execute
    python scripts/nightly_run.py --skip-scraper  # skip scraper step
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
DANWORD_SCRIPT = str(ROOT / "scraper" / "danword" / "danword_lookup.py")
LOG_DIR = ROOT / "logs"

# Sources to auto-solve (not Times/Guardian/Independent — they use blog explanations)
SOLVE_SOURCES = ["telegraph", "dailymail"]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_scraper():
    """Run the puzzle scraper to fetch today's puzzles from all sources."""
    log("Step 1: Running all scrapers...")
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
        log(f"  Scraper failed (exit {result.returncode})")
        if result.stderr:
            log(f"  stderr: {result.stderr[-500:]}")
        return False
    # Show summary lines from output
    for line in (result.stdout or "").splitlines():
        if "clues" in line.lower() or "saved" in line.lower() or "skip" in line.lower():
            log(f"  {line.strip()}")
    log("  Scraper completed")
    return True


def run_danword_backfill(target_date):
    """Run Danword lookup for any puzzles with missing answers."""
    log("Step 2: Danword backfill for missing answers...")
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    rows = conn.execute("""
        SELECT source, puzzle_number, COUNT(*) as total,
               SUM(CASE WHEN answer IS NULL OR answer = '' THEN 1 ELSE 0 END) as missing
        FROM clues
        WHERE publication_date = ?
          AND source IN ('telegraph', 'times', 'guardian', 'independent', 'dailymail')
        GROUP BY source, puzzle_number
        HAVING missing > 0
    """, (target_date,)).fetchall()
    conn.close()

    if not rows:
        log("  No puzzles with missing answers")
        return

    for source, puzzle_number, total, missing in rows:
        log(f"  Danword: {source} #{puzzle_number} ({missing}/{total} missing)")
        try:
            result = subprocess.run(
                [PYTHON_PIPELINE, DANWORD_SCRIPT,
                 '--source', source, '--puzzle', str(puzzle_number)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            if result.returncode == 0:
                # Count how many were found
                for line in (result.stdout or "").splitlines():
                    if "found" in line.lower() or "written" in line.lower():
                        log(f"    {line.strip()}")
            else:
                log(f"    Danword failed (exit {result.returncode})")
        except subprocess.TimeoutExpired:
            log(f"    Danword timed out")
        except Exception as e:
            log(f"    Danword error: {e}")


def find_todays_puzzles(target_date):
    """Find puzzle numbers for today's DT and Daily Mail (weekdays only)."""
    # Check if today is a weekday
    try:
        d = date.fromisoformat(target_date)
    except ValueError:
        return []

    if d.weekday() >= 5:  # Saturday or Sunday
        log(f"  Weekend ({d.strftime('%A')}) — skipping pipeline for DT/Daily Mail")
        return []

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
    log(f"  Pipeline: {source} #{puzzle_number}")
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
        log(f"    Pipeline failed (exit {result.returncode})")
        if result.stderr:
            log(f"    stderr: {result.stderr[-300:]}")
        return False

    # Extract summary from output
    output = result.stdout or ""
    for line in output.splitlines()[-10:]:
        if "HIGH" in line or "solved" in line.lower() or "cost" in line.lower():
            log(f"    {line.strip()}")
    log(f"    Completed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Nightly scrape + solve")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD), default: today")
    parser.add_argument("--skip-scraper", action="store_true", help="Skip scraper step")
    parser.add_argument("--skip-danword", action="store_true", help="Skip Danword backfill")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip pipeline step")
    args = parser.parse_args()

    target_date = args.date or date.today().isoformat()
    target_day = date.fromisoformat(target_date).strftime('%A')

    # Ensure log directory exists
    LOG_DIR.mkdir(exist_ok=True)

    log("=" * 60)
    log(f"NIGHTLY RUN — {target_date} ({target_day})")
    log("=" * 60)

    # Step 1: Scrape all sources
    if not args.skip_scraper:
        if args.dry_run:
            log("[DRY RUN] Would run all scrapers")
        else:
            run_scraper()

    # Step 2: Danword backfill for missing answers
    if not args.skip_danword:
        if args.dry_run:
            log("[DRY RUN] Would run Danword backfill")
        else:
            run_danword_backfill(target_date)

    # Step 3: Pipeline for DT + Daily Mail (weekdays only)
    if not args.skip_pipeline:
        log("Step 3: Pipeline (DT + Daily Mail, weekdays only)...")
        puzzles = find_todays_puzzles(target_date)
        if not puzzles:
            log("  No puzzles to solve")
        else:
            log(f"  Found {len(puzzles)} puzzle(s):")
            for source, pnum in puzzles:
                log(f"    {source} #{pnum}")

            results = []
            for source, pnum in puzzles:
                if args.dry_run:
                    log(f"  [DRY RUN] Would run pipeline on {source} #{pnum}")
                    results.append((source, pnum, True))
                else:
                    ok = run_pipeline(source, pnum)
                    results.append((source, pnum, ok))

            successes = sum(1 for _, _, ok in results if ok)
            failures = len(results) - successes
            log(f"  Pipeline: {successes} succeeded, {failures} failed")

    log("")
    log("=" * 60)
    log("NIGHTLY RUN COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    main()
