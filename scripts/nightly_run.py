"""Nightly automated run: scrape all → danword backfill → solve DT + DM → TFTT → mashup.

Designed to run at 2am UTC via Windows Task Scheduler.

Flow:
  1. Run all scrapers (all sources) — clues + answers uploaded to honeypot/cordelia
  2. Danword backfill for any puzzles with missing answers
  3. Pipeline for Telegraph + Daily Mail ONLY (no blog coverage, must use Sonnet)
  4. Times TFTT — if blog has posted, run blog+Haiku (10x cheaper than Sonnet)
     If not posted, retry_tftt.py runs at 4am via separate Task Scheduler job.
  5. Cordelia's Daily Mash-up

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

# Pipeline sources — only those without blog coverage
# Times/Guardian/Independent run manually after blogs post (cheaper via TFTT/FS+Haiku)
SOLVE_SOURCES = ["telegraph", "dailymail"]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    # Replace chars that Windows cp1252 can't encode
    safe_msg = str(msg).encode('cp1252', errors='replace').decode('cp1252')
    print(f"[{ts}] {safe_msg}", flush=True)


def run_scraper():
    """Run the puzzle scraper to fetch today's puzzles from all sources."""
    log("Step 1: Running all scrapers...")
    try:
        result = subprocess.run(
            [PYTHON_SCRAPER, SCRAPER_SCRIPT],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        log("  Scraper TIMEOUT (30 min) — continuing without scraper")
        return False
    except Exception as e:
        log(f"  Scraper ERROR: {e} — continuing")
        return False

    if result.returncode != 0:
        log(f"  Scraper failed (exit {result.returncode})")
        if result.stderr:
            log(f"  stderr: {result.stderr[-500:]}")
        return False
    # Show summary lines from output
    for line in (result.stdout or "").splitlines():
        if any(k in line.lower() for k in ("clues", "saved", "skip", "indexing", "submitted", "err ")):
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
                timeout=900,
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
    """Find puzzle numbers for today's Telegraph + Daily Mail (weekdays only)."""
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
    """Run the Sonnet pipeline on a single puzzle. Always returns True/False, never raises."""
    log(f"  Pipeline: {source} #{puzzle_number}")
    cmd = [
        PYTHON_PIPELINE, "-m", "sonnet_pipeline.run",
        "--mode", "1",
        "--no-review",
        "--write-db",
        "--source", source,
        str(puzzle_number),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,  # 60 min max per puzzle
        )
    except subprocess.TimeoutExpired:
        log(f"    Pipeline TIMEOUT (30 min) — skipping {source} #{puzzle_number}")
        return False
    except Exception as e:
        log(f"    Pipeline ERROR: {e} — skipping {source} #{puzzle_number}")
        return False

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


def _get_times_puzzle(target_date):
    """Get today's Times puzzle number from the DB (already scraped in Step 1)."""
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    row = conn.execute(
        "SELECT DISTINCT puzzle_number FROM clues "
        "WHERE source = 'times' AND publication_date = ?",
        (target_date,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def _times_already_solved(target_date):
    """Check if today's Times puzzle already has TFTT explanations."""
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    row = conn.execute("""
        SELECT COUNT(*) FROM clues
        WHERE source = 'times' AND publication_date = ?
          AND explanation IS NOT NULL AND explanation != ''
    """, (target_date,)).fetchone()
    conn.close()
    return row[0] > 0


def _check_tftt_available(puzzle_number):
    """Lightweight HTTP check: does a TFTT blog post exist for this puzzle?"""
    import requests

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
    }

    # Try direct URL (fast)
    url = f"https://timesforthetimes.co.uk/times-cryptic-{puzzle_number}"
    try:
        resp = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        if resp.status_code == 200:
            return True
    except Exception:
        pass

    # Try WordPress search API
    try:
        resp = requests.get(
            "https://timesforthetimes.co.uk/wp-json/wp/v2/posts",
            headers=headers, timeout=10,
            params={"search": str(puzzle_number), "per_page": 3, "categories": "11,21"},
        )
        if resp.status_code == 200:
            for post in resp.json():
                if str(puzzle_number) in post.get("slug", ""):
                    return True
    except Exception:
        pass

    return False


def run_tftt_pipeline(puzzle_number):
    """Run the TFTT+Haiku pipeline on a Times puzzle via subprocess."""
    log(f"  Running TFTT pipeline for Times #{puzzle_number}...")
    cmd = [
        PYTHON_PIPELINE, "-m", "sonnet_pipeline.tftt_pipeline",
        str(puzzle_number), "--write-db",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,  # 10 min max
        )
    except subprocess.TimeoutExpired:
        log(f"    TFTT pipeline TIMEOUT (10 min)")
        return False
    except Exception as e:
        log(f"    TFTT pipeline ERROR: {e}")
        return False

    if result.returncode != 0:
        log(f"    TFTT pipeline failed (exit {result.returncode})")
        if result.stderr:
            log(f"    stderr: {result.stderr[-300:]}")
        return False

    # Show summary lines
    for line in (result.stdout or "").splitlines()[-10:]:
        if any(k in line.lower() for k in ("high", "medium", "low", "cost", "score", "clue")):
            log(f"    {line.strip()}")
    log(f"    TFTT pipeline completed")
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

    # Step 3: Pipeline for DT + Daily Mail only (weekdays only)
    if not args.skip_pipeline:
        log("Step 3: Pipeline (Telegraph + Daily Mail, weekdays only)...")
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

    # Step 4: Times TFTT — run blog+Haiku if TFTT has posted
    log("Step 4: Times TFTT blog check...")
    times_puzzle = _get_times_puzzle(target_date)
    if not times_puzzle:
        log("  No Times puzzle found for today")
    elif _times_already_solved(target_date):
        log(f"  Times #{times_puzzle} already has explanations — skipping")
    elif args.dry_run:
        log(f"  [DRY RUN] Would check TFTT for Times #{times_puzzle}")
    else:
        if _check_tftt_available(times_puzzle):
            log(f"  TFTT blog found for Times #{times_puzzle}")
            run_tftt_pipeline(times_puzzle)
        else:
            log(f"  TFTT not yet posted for Times #{times_puzzle} — retry at 4am")

    # Step 5: Cordelia's Daily Mash-up — DISABLED 2026-05-13.
    # Re-enable by uncommenting the block below; the generate_mashup helper
    # still exists in scripts/generate_mashup.py.
    log("Step 5: Cordelia's Daily Mash-up: DISABLED")
    # tomorrow = (date.fromisoformat(target_date) + timedelta(days=1)).isoformat()
    # log(f"Step 5: Cordelia's Daily Mash-up (preparing {tomorrow})...")
    # if args.dry_run:
    #     log(f"  [DRY RUN] Would generate mash-up for {tomorrow}")
    # else:
    #     try:
    #         sys.path.insert(0, str(ROOT))
    #         from scripts.generate_mashup import generate_mashup
    #         mashup_number = generate_mashup(tomorrow)
    #         if mashup_number:
    #             log(f"  Mash-up #{mashup_number} generated for {tomorrow}")
    #             # Run pipeline on unsolved clues
    #             conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    #             unsolved = conn.execute("""
    #                 SELECT COUNT(*) FROM clues
    #                 WHERE source = 'cordelia' AND puzzle_number = ?
    #                   AND answer IS NOT NULL AND answer != ''
    #                   AND (has_solution IS NULL OR has_solution = 0)
    #                   AND clue_text NOT LIKE 'See %%'
    #             """, (str(mashup_number),)).fetchone()[0]
    #             conn.close()
    #             if unsolved > 0:
    #                 log(f"  Running pipeline on {unsolved} unsolved clues...")
    #                 run_pipeline("cordelia", str(mashup_number))
    #             else:
    #                 log(f"  All clues already explained — no pipeline needed")
    #         else:
    #             log(f"  Mash-up skipped (already exists or no eligible base puzzle)")
    #     except Exception as e:
    #         log(f"  Mash-up generation error: {e}")

    log("")
    log("=" * 60)
    log("NIGHTLY RUN COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        import traceback
        for line in traceback.format_exc().splitlines():
            log(f"  {line}")
        sys.exit(1)
