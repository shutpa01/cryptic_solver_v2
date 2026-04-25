"""Retry TFTT at 4am — picks up Times puzzles that weren't blogged by 2am.

Designed to run at 4am UTC via Windows Task Scheduler.
If TFTT has posted, runs blog+Haiku pipeline (~$0.10).
If not, logs and exits — user runs manually after 5am.

Usage:
    python scripts/retry_tftt.py               # today
    python scripts/retry_tftt.py --date 2026-04-15
    python scripts/retry_tftt.py --dry-run
"""

import argparse
import sqlite3
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = ROOT / "data" / "clues_master.db"
PYTHON_PIPELINE = r"C:\Users\shute\PycharmProjects\AI_Solver\.venv\Scripts\python.exe"
LOG_DIR = ROOT / "logs"


def log(msg):
    ts = time.strftime("%H:%M:%S")
    safe_msg = str(msg).encode('cp1252', errors='replace').decode('cp1252')
    print(f"[{ts}] {safe_msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Retry TFTT for Times puzzle")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target_date = args.date or date.today().isoformat()

    LOG_DIR.mkdir(exist_ok=True)

    log("=" * 60)
    log(f"TFTT RETRY — {target_date}")
    log("=" * 60)

    # Find today's Times puzzle
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    row = conn.execute(
        "SELECT DISTINCT puzzle_number FROM clues "
        "WHERE source = 'times' AND publication_date = ?",
        (target_date,)
    ).fetchone()

    if not row:
        log("No Times puzzle found for today — nothing to do")
        conn.close()
        return

    puzzle_number = row[0]

    # Check if already solved (2am run succeeded)
    solved = conn.execute("""
        SELECT COUNT(*) FROM clues
        WHERE source = 'times' AND publication_date = ?
          AND explanation IS NOT NULL AND explanation != ''
    """, (target_date,)).fetchone()[0]
    conn.close()

    if solved > 0:
        log(f"Times #{puzzle_number} already has explanations (2am run succeeded) — skipping")
        return

    log(f"Times #{puzzle_number} has no explanations yet — checking TFTT...")

    if args.dry_run:
        log(f"[DRY RUN] Would check TFTT for Times #{puzzle_number}")
        return

    # Lightweight HTTP check
    import requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
    }

    found = False
    url = f"https://timesforthetimes.co.uk/times-cryptic-{puzzle_number}"
    try:
        resp = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        if resp.status_code == 200:
            found = True
    except Exception:
        pass

    if not found:
        try:
            resp = requests.get(
                "https://timesforthetimes.co.uk/wp-json/wp/v2/posts",
                headers=headers, timeout=10,
                params={"search": str(puzzle_number), "per_page": 3, "categories": "11,21"},
            )
            if resp.status_code == 200:
                for post in resp.json():
                    if str(puzzle_number) in post.get("slug", ""):
                        found = True
                        break
        except Exception:
            pass

    if not found:
        log(f"TFTT still not posted for Times #{puzzle_number} — manual run needed")
        return

    log(f"TFTT blog found for Times #{puzzle_number} — running pipeline...")
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
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        log("TFTT pipeline TIMEOUT (10 min)")
        return
    except Exception as e:
        log(f"TFTT pipeline ERROR: {e}")
        return

    if result.returncode != 0:
        log(f"TFTT pipeline failed (exit {result.returncode})")
        if result.stderr:
            log(f"stderr: {result.stderr[-300:]}")
        return

    for line in (result.stdout or "").splitlines()[-10:]:
        if any(k in line.lower() for k in ("high", "medium", "low", "cost", "score", "clue")):
            log(f"  {line.strip()}")
    log("TFTT pipeline completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        import traceback
        for line in traceback.format_exc().splitlines():
            log(f"  {line}")
        sys.exit(1)
