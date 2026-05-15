"""Nightly automated run: scrape only, all sources.

Designed to run at 2am UTC via Windows Task Scheduler.

Workflow rule (2026-05-13, see memory/workflow_enrichment_before_indexing.md):
the nightly run is LOCAL ONLY — scrape every source we collect from,
but invoke nothing that costs money or touches the public site. No
Danword, no Sonnet API, no Indexing API, no droplet upload. The
enrichment + upload + indexing flow is human-driven the next day
once each puzzle is fully enriched.

Flow:
  1. Scrape all five sources (telegraph, times, guardian, independent,
     dailymail)
  2. Danword backfill — DISABLED
  3. Pipeline (Sonnet) — DISABLED
  4. Times TFTT — DISABLED
  5. Cordelia's Daily Mash-up — DISABLED

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

SCRAPE_SOURCES = ["telegraph", "times", "guardian", "independent", "dailymail"]
# SOLVE_SOURCES kept for reference; Step 3 is disabled so nothing reads
# this. If pipeline runs are ever reinstated, restore from SCRAPE_SOURCES
# or a deliberate subset.
SOLVE_SOURCES: list = []


def log(msg):
    ts = time.strftime("%H:%M:%S")
    # Replace chars that Windows cp1252 can't encode
    safe_msg = str(msg).encode('cp1252', errors='replace').decode('cp1252')
    print(f"[{ts}] {safe_msg}", flush=True)


def run_scraper():
    """Scrape today's Telegraph and Daily Mail puzzles.

    puzzle_scraper.py's --only flag takes a single source, so we invoke
    it once per source. Failure of one source doesn't stop the other.
    """
    overall_ok = True
    for source in SCRAPE_SOURCES:
        log(f"Step 1: Scraping {source}...")
        try:
            result = subprocess.run(
                [PYTHON_SCRAPER, SCRAPER_SCRIPT, "--only", source],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            log(f"  {source} scraper TIMEOUT (30 min) — continuing")
            overall_ok = False
            continue
        except Exception as e:
            log(f"  {source} scraper ERROR: {e} — continuing")
            overall_ok = False
            continue

        if result.returncode != 0:
            log(f"  {source} scraper failed (exit {result.returncode})")
            if result.stderr:
                log(f"  stderr: {result.stderr[-500:]}")
            overall_ok = False
            continue

        # Show summary lines from output
        for line in (result.stdout or "").splitlines():
            if any(k in line.lower() for k in
                   ("clues", "saved", "skip", "indexing", "submitted", "err ")):
                log(f"  {line.strip()}")
        log(f"  {source} scraper completed")
    return overall_ok


def run_danword_backfill(target_date):
    """Run Danword lookup for any DT/DM puzzles with missing answers."""
    log("Step 2: Danword backfill for missing answers (DT + DM only)...")
    placeholders = ",".join("?" * len(SOLVE_SOURCES))
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    rows = conn.execute(f"""
        SELECT source, puzzle_number, COUNT(*) as total,
               SUM(CASE WHEN answer IS NULL OR answer = '' THEN 1 ELSE 0 END) as missing
        FROM clues
        WHERE publication_date = ?
          AND source IN ({placeholders})
        GROUP BY source, puzzle_number
        HAVING missing > 0
    """, (target_date, *SOLVE_SOURCES)).fetchall()
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

    # Step 1: Scrape Telegraph + Daily Mail
    if not args.skip_scraper:
        if args.dry_run:
            log("[DRY RUN] Would scrape: " + ", ".join(SCRAPE_SOURCES))
        else:
            run_scraper()

    # Step 2: Danword backfill — DISABLED 2026-05-15. The
    # workflow rule is local-only nightly with zero API spend;
    # Danword is the most expensive scrape step and must not run
    # unattended. Re-enable by uncommenting the block below; the
    # helper function definitions remain in this file.
    log("Step 2: Danword backfill: DISABLED")

    # Step 3: Pipeline (Sonnet API) — DISABLED 2026-05-15. Same
    # workflow rule: no API spend in the nightly. Puzzles are
    # enriched manually the next day before any upload or
    # Indexing API submission.
    log("Step 3: Pipeline (Sonnet): DISABLED")

    # Step 4: Times TFTT — DISABLED 2026-05-13 along with the
    # restriction of the nightly to DT+DM only. Times / Guardian /
    # Independent are processed manually as blogs appear; the
    # helpers and retry_tftt.py can still be invoked by hand.
    log("Step 4: Times TFTT auto-check: DISABLED")

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
