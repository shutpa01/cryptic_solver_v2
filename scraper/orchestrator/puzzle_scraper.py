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


def _rsync_json_dir(local_dir, remote_dir, timeout=300):
    """Rsync only *.json files from a directory to a remote directory.

    Trailing slashes on both sides matter: copies CONTENTS of local_dir into
    remote_dir, not the directory itself. --mkpath creates the remote dir
    if missing. No --delete: missing local files are left alone on remote.
    """
    s = str(local_dir).replace('\\', '/')
    if len(s) >= 2 and s[1] == ':':
        s = '/' + s[0].lower() + s[2:]
    s = s.rstrip('/') + '/'
    remote = remote_dir.rstrip('/') + '/'
    cmd = (
        f"rsync -cz --mkpath --include='*.json' --exclude='*' "
        f"{s} {remote}"
    )
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
DANWORD_TIMEOUT = 900  # ~30 clues × 15s each + Firefox startup overhead


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

    # Danword backfill for puzzles with missing answers
    answerless = find_answerless_puzzles()
    if answerless:
        danword_results = run_danword_backfill(answerless)
    else:
        danword_results = []

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

    # Sync DB to live site (Cordelia only — honeypot retired)
    _sync_cordelia()

    # Submit today's clue URLs to Google Indexing API
    indexing_summary = _submit_to_indexing_api(new_puzzles)
    if indexing_summary:
        email_lines.append("")
        email_lines.append(indexing_summary)

    _send_email(subject, '\n'.join(email_lines))

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


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

        # Upload puzzle JSON files (grid structure source — local-only data,
        # required by web/grid.py:build_grid_from_json so the droplet can
        # render grids using the publisher's authoritative clue numbering
        # instead of falling back to algorithmic re-derivation).
        json_dirs = [
            (PROJECT_ROOT / 'scraper' / 'telegraph', '/opt/cordelia/scraper/telegraph'),
            (PROJECT_ROOT / 'scraper' / 'times', '/opt/cordelia/scraper/times'),
            (PROJECT_ROOT / 'scraper' / 'guardian', '/opt/cordelia/scraper/guardian'),
        ]
        for local_dir, remote_dir in json_dirs:
            if not local_dir.exists():
                print(f"  {local_dir.name}: skipped (local dir missing)")
                continue
            try:
                result = _rsync_json_dir(
                    local_dir,
                    f"{CORDELIA_DROPLET}:{remote_dir}",
                    timeout=300,
                )
                if result.returncode == 0:
                    print(f"  {local_dir.name}: JSON synced")
                else:
                    # Don't abort — DB sync already succeeded, restart should still happen
                    print(f"  {local_dir.name}: JSON sync failed (rc={result.returncode}): "
                          f"{result.stderr.strip()[:200]}")
            except subprocess.TimeoutExpired:
                print(f"  {local_dir.name}: JSON sync timed out")
            except Exception as e:
                print(f"  {local_dir.name}: JSON sync error: {e}")

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
CORDELIA_BASE_URL = "https://justcordelia.com"
INDEXING_DAILY_QUOTA = 200

# Source priority order for indexing — DT first, Indy last
INDEXING_SOURCE_ORDER = ["telegraph", "dailymail", "times", "guardian", "independent"]


def _make_clue_slug(clue_id, clue_text):
    """Build Cordelia clue URL slug: {id}-{slugified-text}."""
    text = re.sub(r"[^a-z0-9]+", "-", clue_text.lower().strip()).strip("-")
    if not text:
        return None
    words = text.split("-")[:12]
    return f"{clue_id}-{'-'.join(words)}"


def _submit_to_indexing_api(new_puzzles):
    """Submit today's clue URLs to Google's Indexing API.

    Cordelia URLs only. Capped at daily quota.
    Source order: DT > DM > Times > Guardian > Independent.

    Returns: summary string for email, or None.
    """
    if not INDEXING_SA_PATH.exists():
        print("\nIndexing API: service account not found, skipping")
        return None

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        creds = service_account.Credentials.from_service_account_file(
            str(INDEXING_SA_PATH),
            scopes=["https://www.googleapis.com/auth/indexing"],
        )
        service = build("indexing", "v3", credentials=creds)

        source_rank = {s: i for i, s in enumerate(INDEXING_SOURCE_ORDER)}
        sorted_new = sorted(new_puzzles, key=lambda x: source_rank.get(x[0], 99))

        conn = sqlite3.connect(str(CLUES_MASTER_DB))
        urls = []
        for source, puzzle_number, _count in sorted_new:
            rows = conn.execute(
                "SELECT id, clue_text FROM clues WHERE source = ? AND puzzle_number = ?",
                (source, puzzle_number),
            ).fetchall()
            for clue_id, clue_text in rows:
                slug = _make_clue_slug(clue_id, clue_text)
                if slug:
                    urls.append(f"{CORDELIA_BASE_URL}/clue/{slug}")
        conn.close()

        urls = urls[:INDEXING_DAILY_QUOTA]

        print(f"\n{'=' * 60}")
        print(f"GOOGLE INDEXING API — {len(urls)} clue URLs (quota {INDEXING_DAILY_QUOTA})")
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

        summary = (
            f"GOOGLE INDEXING API\n"
            f"  Clue URLs: {len(urls)}\n"
            f"  Total submitted: {submitted}, Errors: {errors}"
        )
        return summary

    except Exception as e:
        print(f"\nIndexing API error: {e}")
        return f"GOOGLE INDEXING API: ERROR — {e}"


if __name__ == "__main__":
    main()
