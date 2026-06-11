"""Take a checkpointed backup of the cryptic DBs and the gilts portfolio DB.

Backs up clues_master.db, cryptic_new.db (cryptic solver) and the gilts_scraper
gilts.db. Backups land in OneDrive (off-repo, cloud-synced) and are retained for
one month (older copies are pruned, but the newest of each is always kept). Uses
SQLite's online .backup API so the copy is consistent even if the source has an
active WAL.

Usage:
    python scripts/backup_dbs.py
    python scripts/backup_dbs.py --skip-if-younger-than 6   # hours
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BACKUP_DIR = Path(r"C:\Users\shute\OneDrive\DB Backup")
DBS = [
    REPO / "data" / "clues_master.db",
    REPO / "data" / "cryptic_new.db",
    Path(r"C:\Users\shute\PycharmProjects\gilts_scraper\data\gilts.db"),
]
RETENTION_DAYS = 31


def latest_backup(dbname):
    """Return the newest backup file for `dbname`, or None."""
    matches = sorted(BACKUP_DIR.glob(f"{dbname}_*.db"))
    return matches[-1] if matches else None


def take_backup(src_path, ts):
    """Online backup via SQLite's backup API (WAL-safe)."""
    dst_path = BACKUP_DIR / f"{src_path.stem}_{ts}.db"
    src = sqlite3.connect(str(src_path))
    dst = sqlite3.connect(str(dst_path))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return dst_path


def rotate(dbname, retention_days=RETENTION_DAYS):
    """Delete backups older than `retention_days`, but always keep the newest one
    (so a paused job never leaves a DB with zero backups)."""
    matches = sorted(BACKUP_DIR.glob(f"{dbname}_*.db"))
    cutoff = datetime.now() - timedelta(days=retention_days)
    removed = 0
    for f in matches[:-1]:        # everything except the most recent
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        if mtime < cutoff:
            try:
                f.unlink()
                removed += 1
            except OSError as e:
                print(f"  warning: could not delete {f.name}: {e}",
                      file=sys.stderr)
    return removed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-if-younger-than", type=float, default=0.0,
                   metavar="HOURS",
                   help="Skip a DB if its newest backup is younger than this "
                        "many hours. 0 means always back up.")
    args = p.parse_args()

    if not BACKUP_DIR.exists():
        print(f"ERROR: backup dir does not exist: {BACKUP_DIR}",
              file=sys.stderr)
        return 1

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    cutoff = (datetime.now() - timedelta(hours=args.skip_if_younger_than)
              if args.skip_if_younger_than > 0 else None)

    summary = []
    for src in DBS:
        if not src.exists():
            print(f"  skip {src.name}: not found")
            summary.append((src.name, "missing"))
            continue

        if cutoff is not None:
            last = latest_backup(src.stem)
            if last is not None:
                last_mtime = datetime.fromtimestamp(last.stat().st_mtime)
                if last_mtime > cutoff:
                    age_hours = (datetime.now() - last_mtime).total_seconds() / 3600
                    print(f"  skip {src.name}: last backup {age_hours:.1f}h "
                          f"old (< {args.skip_if_younger_than}h threshold)")
                    summary.append((src.name, f"skipped ({age_hours:.1f}h)"))
                    continue

        try:
            dst = take_backup(src, ts)
            size_mb = dst.stat().st_size / (1024 * 1024)
            removed = rotate(src.stem)
            note = f"{size_mb:.0f} MB"
            if removed:
                note += f", rotated {removed}"
            print(f"  backed up {src.name} -> {dst.name} ({note})")
            summary.append((src.name, note))
        except Exception as e:
            print(f"  ERROR backing up {src.name}: {e}", file=sys.stderr)
            summary.append((src.name, f"FAILED: {e}"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
