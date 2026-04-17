#!/usr/bin/env python3
"""Puzzle Scraper Orchestrator

Runs all newspaper puzzle scrapers. Each scraper writes directly to the clues table
in clues_master.db — no sync step needed.

After scrapers run, reconciles against the expected puzzle schedule to report
exactly which puzzles were captured and which are missing.

Usage:Predictably the times fi
    python puzzle_scraper.py                # Run all scrapers
    python puzzle_scraper.py --only guardian    # Run only guardian
    python puzzle_scraper.py --only telegraph   # Run only telegraph
    python puzzle_scraper.py --only times       # Run only times
    python puzzle_scraper.py --only independent # Run only independent
    python puzzle_scraper.py --only dailymail   # Run only dailymail
"""

import argparse
import json
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
GIT_BASH = r'C:\Program Files\Git\bin\bash.exe'


def _rsync(local_path, remote_path, timeout=300):
    """Run rsync via Git Bash (provides full MSYS2 environment including SSH)."""
    # Convert Windows path to MSYS-style: C:\Users\x → /c/Users/x
    s = str(local_path).replace('\\', '/')
    if len(s) >= 2 and s[1] == ':':
        s = '/' + s[0].lower() + s[2:]
    cmd = f'rsync -cz {s} {remote_path}'
    return subprocess.run(
        [GIT_BASH, '-c', cmd],
        capture_output=True, text=True, timeout=timeout,
    )


# ── Expected puzzle schedule ──────────────────────────────────────────────
# Each entry: label, source, puzzle_number range (lo, hi), days of week
# Days: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
EXPECTED_PUZZLES = [
    # Telegraph — daily cryptic is 31xxx, toughie+prize cryptic share 2400-4000 (different days)
    ('Telegraph Cryptic',       'telegraph', 27000, 35000, [0, 1, 2, 3, 4, 5]),
    ('Telegraph Toughie',       'telegraph-toughie', 2400,  4000,  [1, 2, 3, 4]),
    ('Telegraph Prize Cryptic', 'telegraph', 2400,  4000,  [6]),
    ('Telegraph Prize Toughie', 'telegraph', 1,     500,   [6]),
    # Times
    ('Times Cryptic',           'times',     26000, 32000, [0, 1, 2, 3, 4, 5]),
    ('Sunday Times Cryptic',    'times',     4700,  6000,  [6]),
    # Guardian — cryptic Mon-Fri, Everyman Sunday
    ('Guardian Cryptic',        'guardian',  21000, 32000, [0, 1, 2, 3, 4]),
    ('Observer Everyman',       'guardian',   4000,  5000, [6]),
    # Independent — cryptic every day
    ('Independent Cryptic',     'independent', 10000, 19999, [0, 1, 2, 3, 4, 5]),
    ('Independent Sunday',      'independent',  1000,  1999, [6]),
    # Daily Mail — cryptic weekdays only, paper puzzle numbers ~17k range
    ('Daily Mail Cryptic',      'dailymail', 15000, 20000, [0, 1, 2, 3, 4]),
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
        'script': BASE_PATH / 'independent' / 'independent_edition.py',
        'args': [],
        'timeout': 120,  # HTTP scraper — fast
        'retries': 1,
    },
    'dailymail': {
        'script': BASE_PATH / 'dailymail' / 'dailymail_daily.py',
        'args': [],
        'timeout': 60,   # API scraper — very fast, no auth needed
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
        print(f"  It seems to be tkiScript not found: {script}")
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


def find_missing_answers(today: date | None = None) -> list[tuple[str, str, str]]:
    """Find clues from today's puzzles that have no answer.

    Returns list of (source, puzzle_number, clue_number) for each missing answer.
    """
    if today is None:
        today = date.today()
    today_str = today.isoformat()

    if not CLUES_MASTER_DB.exists():
        return []

    conn = sqlite3.connect(CLUES_MASTER_DB)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT source, puzzle_number, clue_number
        FROM clues
        WHERE publication_date = ?
          AND (answer IS NULL OR answer = '')
        ORDER BY source, CAST(puzzle_number AS INTEGER), clue_number
    """, (today_str,))
    results = [(row[0], row[1], row[2]) for row in cursor.fetchall()]
    conn.close()
    return results


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
    """Find today's puzzles that have any clues with empty answers.

    Returns list of (source, puzzle_number, clue_count) for puzzles needing
    danword backfill. clue_count is the number of missing answers.
    """
    if today is None:
        today = date.today()
    today_str = today.isoformat()

    if not CLUES_MASTER_DB.exists():
        return []

    conn = sqlite3.connect(CLUES_MASTER_DB)
    cursor = conn.cursor()

    # Find puzzles published today with any answerless clues
    cursor.execute("""
        SELECT source, puzzle_number, COUNT(*) as total,
               SUM(CASE WHEN answer IS NULL OR answer = '' THEN 1 ELSE 0 END) as missing
        FROM clues
        WHERE publication_date = ?
          AND source IN ('telegraph', 'times', 'guardian', 'independent', 'dailymail')
        GROUP BY source, puzzle_number
        HAVING missing > 0
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
            results.append((source, puzzle_number, 0, clue_count, 'TIMEOUT', []))
            continue

        if stdout:
            print(stdout)
        if stderr:
            print("    STDERR:", stderr[:500])

        # Parse found/total, grid status, and conflicts from output
        found = 0
        grid_status = ''
        conflicts = []
        for line in (stdout or '').splitlines():
            m = re.search(r'Done:\s*(\d+)/(\d+)', line)
            if m:
                found = int(m.group(1))
            mg = re.search(r'^Grid:\s*(.+)$', line)
            if mg:
                grid_status = mg.group(1).strip()
            mc = re.search(r'^\s+\(\d+,\d+\):\s+(.+)$', line)
            if mc:
                conflicts.append(mc.group(1).strip())

        results.append((source, puzzle_number, found, clue_count, grid_status, conflicts))

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
        for source, pnum, found, total, grid, conflicts in danword_results:
            grid_tag = f"  [{grid}]" if grid else ""
            print(f"  {source:15} #{pnum:>8}  {found}/{total} answers{grid_tag}")
            for c in conflicts:
                print(f"    CONFLICT: {c}")
    else:
        print(f"\n  No answerless puzzles to backfill today")

    # Missing answers report
    missing_answers = find_missing_answers()
    print(f"\n{'=' * 60}")
    print("MISSING ANSWERS")
    print(f"{'=' * 60}")
    if missing_answers:
        print(f"  {len(missing_answers)} clues still missing answers:")
        for source, pnum, cnum in missing_answers:
            print(f"  {source:15} #{pnum:>8}  clue {cnum}")
    else:
        print("  All today's clues have answers")

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
        for source, pnum, found, total, grid, conflicts in danword_results:
            grid_tag = f"  [{grid}]" if grid else ""
            email_lines.append(f"  {source:15} #{pnum:>8}  {found}/{total} answers{grid_tag}")
            if conflicts:
                email_lines.append(f"    *** CROSSING CONFLICTS ***")
                for c in conflicts:
                    email_lines.append(f"    {c}")

    email_lines.append("")
    email_lines.append("MISSING ANSWERS")
    email_lines.append("-" * 40)
    if missing_answers:
        email_lines.append(f"  {len(missing_answers)} clues still missing answers:")
        for source, pnum, cnum in missing_answers:
            email_lines.append(f"  {source:15} #{pnum:>8}  clue {cnum}")
    else:
        email_lines.append("  All today's clues have answers")

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

    # Sync DBs to live sites
    _sync_honeypot()
    _sync_cordelia()

    # Submit URLs to Google Indexing API (always — future puzzles are priority)
    _submit_to_indexing_api(new_puzzles)


def _sync_honeypot():
    """Copy clues_master.db to the honeypot server and restart."""
    import subprocess
    import shutil

    db_path = CLUES_MASTER_DB
    if not db_path.exists():
        print("Honeypot sync: DB not found, skipping")
        return

    # Checkpoint WAL so all data is in the main .db file before upload
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
    except Exception as e:
        print(f"  WAL checkpoint failed: {e}")

    print("\nSyncing DB to honeypot site...")
    try:
        result = _rsync(db_path, "root@134.209.21.34:/opt/honeypot/data/clues.db")
        if result.returncode != 0:
            print(f"  rsync failed: {result.stderr}")
            return

        result = subprocess.run(
            ["ssh", "root@134.209.21.34", "systemctl restart honeypot"],
            timeout=120,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("  Honeypot synced and restarted")
        else:
            print(f"  Restart failed: {result.stderr}")

        # Regenerate sitemaps on the droplet with the freshly uploaded DB
        result = subprocess.run(
            ["ssh", "root@134.209.21.34",
             "cd /opt/honeypot && python3 generate_sitemaps.py --domain https://clairesclues.xyz"],
            timeout=120,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("  Sitemaps regenerated")
        else:
            print(f"  Sitemap generation failed: {result.stderr}")
    except Exception as e:
        print(f"  Honeypot sync error: {e}")


CORDELIA_DROPLET = "root@165.232.46.255"
CRYPTIC_NEW_DB = PROJECT_ROOT / 'data' / 'cryptic_new.db'


def _sync_cordelia():
    """Copy both databases to the Cordelia server and restart."""
    import subprocess

    clues_db = CLUES_MASTER_DB
    ref_db = CRYPTIC_NEW_DB

    if not clues_db.exists():
        print("Cordelia sync: clues_master.db not found, skipping")
        return

    # Checkpoint WAL so all data is in the main .db files before upload
    for db in [clues_db, ref_db]:
        if db.exists():
            try:
                conn = sqlite3.connect(str(db))
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.close()
            except Exception as e:
                print(f"  WAL checkpoint failed for {db.name}: {e}")

    print("\nSyncing DBs to Cordelia (justcordelia.com)...")
    try:
        # Upload clues_master.db
        result = _rsync(clues_db, f"{CORDELIA_DROPLET}:/opt/cordelia/data/clues_master.db", timeout=600)
        if result.returncode != 0:
            print(f"  rsync clues_master.db failed: {result.stderr}")
            return
        print("  clues_master.db uploaded")

        # Upload cryptic_new.db
        if ref_db.exists():
            result = _rsync(ref_db, f"{CORDELIA_DROPLET}:/opt/cordelia/data/cryptic_new.db", timeout=600)
            if result.returncode != 0:
                print(f"  SCP cryptic_new.db failed: {result.stderr}")
                return
            print("  cryptic_new.db uploaded")
        else:
            print("  cryptic_new.db not found, skipping")

        # Restart service
        result = subprocess.run(
            ["ssh", CORDELIA_DROPLET, "systemctl restart cordelia"],
            timeout=120,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("  Cordelia synced and restarted")
        else:
            print(f"  Restart failed: {result.stderr}")

    except Exception as e:
        print(f"  Cordelia sync error: {e}")


INDEXING_SA_PATH = PROJECT_ROOT / "impressions" / "indexing_service_account.json"
INDEXING_SUBMITTED_PATH = PROJECT_ROOT / "impressions" / "submitted_future_urls.json"
CORDELIA_BASE_URL = "https://justcordelia.com"
INDEXING_DAILY_QUOTA = 200

# Source priority order for indexing — DT first, Indy last
INDEXING_SOURCE_ORDER = ["telegraph", "dailymail", "times", "guardian", "independent"]

# Weekday cryptic ranges for future puzzle prediction
FUTURE_PUZZLE_RANGES = [
    ("telegraph", 31000, 31999),
    ("dailymail", 16000, 19999),
    ("times", 26000, 39999),
    ("guardian", 20000, 39999),
    ("independent", 1, 19999),
]


def _make_clue_slug(clue_id, clue_text):
    """Build honeypot clue URL slug: {id}-{slugified-text}."""
    text = re.sub(r"[^a-z0-9]+", "-", clue_text.lower().strip()).strip("-")
    if not text:
        return None
    words = text.split("-")[:12]
    return f"{clue_id}-{'-'.join(words)}"


def _get_future_puzzle_numbers(conn, n=5):
    """Predict next n puzzle numbers for each weekday cryptic source."""
    results = []
    for source, lo, hi in FUTURE_PUZZLE_RANGES:
        row = conn.execute(
            "SELECT MAX(CAST(puzzle_number AS INTEGER)) FROM clues "
            "WHERE source = ? AND CAST(puzzle_number AS INTEGER) BETWEEN ? AND ?",
            (source, lo, hi),
        ).fetchone()
        if not row or row[0] is None:
            continue
        latest = row[0]
        for i in range(1, n + 1):
            results.append((source, str(latest + i)))
    return results


def _load_submitted_futures():
    """Load set of previously submitted future puzzle URLs."""
    if INDEXING_SUBMITTED_PATH.exists():
        try:
            with open(INDEXING_SUBMITTED_PATH, encoding="utf-8") as f:
                return set(json.load(f))
        except (json.JSONDecodeError, ValueError):
            pass
    return set()


def _save_submitted_futures(submitted_set):
    """Persist submitted future puzzle URLs to disk."""
    INDEXING_SUBMITTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEXING_SUBMITTED_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(submitted_set), f, indent=2)


def _submit_to_indexing_api(new_puzzles):
    """Submit URLs to Google's Indexing API with priority ordering.

    Priority: future puzzles (new only) > today's puzzles > today's clue URLs.
    Cordelia URLs only. Capped at daily quota.
    Source order: DT > DM > Times > Guardian > Independent.
    """
    if not INDEXING_SA_PATH.exists():
        print("\nIndexing API: service account not found, skipping")
        return

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_file(
            str(INDEXING_SA_PATH),
            scopes=["https://www.googleapis.com/auth/indexing"],
        )
        service = build("indexing", "v3", credentials=creds)

        conn = sqlite3.connect(str(CLUES_MASTER_DB))
        urls = []

        # --- Priority 1: Future puzzle pages (Cordelia only, new URLs only) ---
        previously_submitted = _load_submitted_futures()
        future = _get_future_puzzle_numbers(conn, n=14)
        source_rank = {s: i for i, s in enumerate(INDEXING_SOURCE_ORDER)}
        future.sort(key=lambda x: source_rank.get(x[0], 99))

        current_future_urls = set()
        for source, pnum in future:
            url = f"{CORDELIA_BASE_URL}/{source}/cryptic/{pnum}"
            current_future_urls.add(url)
            if url not in previously_submitted:
                urls.append(url)

        future_count = len(urls)

        # --- Priority 2: Today's new puzzle pages (Cordelia only) ---
        sorted_new = sorted(new_puzzles, key=lambda x: source_rank.get(x[0], 99))
        for source, puzzle_number, _count in sorted_new:
            urls.append(f"{CORDELIA_BASE_URL}/{source}/cryptic/{puzzle_number}")

        today_count = len(urls) - future_count

        # --- Priority 3: Today's clue URLs (Cordelia, fill remaining quota) ---
        clue_urls = []
        for source, puzzle_number, _count in sorted_new:
            rows = conn.execute(
                "SELECT id, clue_text FROM clues WHERE source = ? AND puzzle_number = ?",
                (source, puzzle_number),
            ).fetchall()
            for clue_id, clue_text in rows:
                slug = _make_clue_slug(clue_id, clue_text)
                if slug:
                    clue_urls.append(f"{CORDELIA_BASE_URL}/clue/{slug}")
        conn.close()

        remaining = INDEXING_DAILY_QUOTA - len(urls)
        if remaining > 0:
            urls.extend(clue_urls[:remaining])

        clue_count = len(urls) - future_count - today_count

        print(f"\n{'=' * 60}")
        print(f"GOOGLE INDEXING API — {len(urls)} URLs (quota {INDEXING_DAILY_QUOTA})")
        print(f"  Future puzzles (new): {future_count}")
        print(f"  Today's puzzles: {today_count}")
        print(f"  Clue URLs: {clue_count}")
        print(f"{'=' * 60}")

        submitted = 0
        errors = 0
        for url in urls:
            try:
                service.urlNotifications().publish(
                    body={"url": url, "type": "URL_UPDATED"}
                ).execute()
                submitted += 1
            except Exception as e:
                print(f"  ERR {url} — {e}")
                errors += 1

        print(f"  Submitted: {submitted}, Errors: {errors}")

        # Update tracking — keep only current future URLs, add newly submitted
        _save_submitted_futures(previously_submitted | current_future_urls)

    except Exception as e:
        print(f"\nIndexing API error: {e}")


if __name__ == "__main__":
    main()
