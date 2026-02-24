"""Migrate explanations, definitions, and wordplay_type from cryptic_new.db to clues_master.db.

Uses Python dict for fast matching — loads both sides into memory, matches by key.

Matching key: (master_source, puzzle_number, UPPER(answer))

Usage:
    python migrate_explanations.py              # Dry run (shows counts)
    python migrate_explanations.py --apply      # Apply changes
"""

import argparse
import sqlite3
import time
from pathlib import Path

CRYPTIC_NEW_DB = Path(r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\cryptic_new.db")
CLUES_MASTER_DB = Path(r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db")

# Map cryptic_new source → master source, plus puzzle_number transform
SOURCE_MAP = {
    'telegraph':        'telegraph',
    'toughie':          'telegraph',
    'sunday_telegraph': 'telegraph',
    'times':            'times',
    'guardian':         'guardian',
    'independent':      'independent',
}


def normalise_pn(source, pn):
    """Normalise puzzle_number for matching."""
    if source == 'times' and pn and pn.startswith('Times '):
        return pn[6:]
    return pn


def migrate(apply=False):
    t0 = time.time()

    # ── Load cryptic_new data ──
    print("Loading cryptic_new.db...")
    conn_cn = sqlite3.connect(str(CRYPTIC_NEW_DB))
    cur_cn = conn_cn.cursor()
    cur_cn.execute("""
        SELECT source, puzzle_number, answer, explanation, definition, wordplay_type
        FROM clues
        WHERE source IN ('telegraph', 'toughie', 'sunday_telegraph', 'times', 'guardian', 'independent')
          AND puzzle_number IS NOT NULL AND answer IS NOT NULL
          AND (
              (explanation IS NOT NULL AND explanation != '')
              OR (definition IS NOT NULL AND definition != '')
              OR (wordplay_type IS NOT NULL AND wordplay_type != '')
          )
    """)
    cn_rows = cur_cn.fetchall()
    conn_cn.close()
    print(f"  {len(cn_rows):,} rows with data to migrate ({time.time()-t0:.1f}s)")

    # Build lookup: key → (explanation, definition, wordplay_type)
    # If multiple rows have same key, last one wins (doesn't matter, they should be identical)
    cn_lookup = {}
    by_source = {}
    for source, pn, answer, expl, defn, wtype in cn_rows:
        master_source = SOURCE_MAP[source]
        master_pn = normalise_pn(source, pn)
        key = (master_source, master_pn, answer.strip().upper())
        cn_lookup[key] = (expl, defn, wtype)
        by_source[source] = by_source.get(source, 0) + 1

    print(f"  Unique match keys: {len(cn_lookup):,}")
    print("  By source:")
    for src, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"    {src}: {cnt:,}")

    # ── Load clues_master IDs + match keys ──
    print(f"\nLoading clues_master.db... ({time.time()-t0:.1f}s)")
    conn_cm = sqlite3.connect(str(CLUES_MASTER_DB))
    cur_cm = conn_cm.cursor()

    cur_cm.execute("SELECT COUNT(*) FROM clues WHERE explanation IS NOT NULL AND explanation != ''")
    pre_expl = cur_cm.fetchone()[0]
    cur_cm.execute("SELECT COUNT(*) FROM clues WHERE definition IS NOT NULL AND definition != ''")
    pre_def = cur_cm.fetchone()[0]
    cur_cm.execute("SELECT COUNT(*) FROM clues WHERE wordplay_type IS NOT NULL AND wordplay_type != ''")
    pre_wt = cur_cm.fetchone()[0]
    print(f"  BEFORE: explanations={pre_expl:,}, definitions={pre_def:,}, wordplay_type={pre_wt:,}")

    cur_cm.execute("SELECT id, source, puzzle_number, answer FROM clues")
    master_rows = cur_cm.fetchall()
    print(f"  {len(master_rows):,} total rows ({time.time()-t0:.1f}s)")

    # ── Match and collect updates ──
    print(f"\nMatching...")
    updates_expl = []
    updates_def = []
    updates_wt = []
    matched = 0

    for row_id, source, pn, answer in master_rows:
        if not answer or not pn:
            continue
        key = (source, pn, answer.strip().upper())
        data = cn_lookup.get(key)
        if not data:
            continue

        matched += 1
        expl, defn, wtype = data

        if expl:
            updates_expl.append((expl, row_id))
        if defn:
            updates_def.append((defn, row_id))
        if wtype:
            updates_wt.append((wtype, row_id))

    print(f"  Matched: {matched:,}")
    print(f"  Explanations to write: {len(updates_expl):,}")
    print(f"  Definitions to write: {len(updates_def):,}")
    print(f"  Wordplay types to write: {len(updates_wt):,}")
    print(f"  Unmatched keys: {len(cn_lookup) - matched:,}")

    if not apply:
        print(f"\nDRY RUN — use --apply to write changes ({time.time()-t0:.1f}s)")
        conn_cm.close()
        return

    # ── Apply updates ──
    print(f"\nApplying updates...")

    cur_cm.execute("BEGIN")

    print(f"  Writing {len(updates_expl):,} explanations...")
    cur_cm.executemany("UPDATE clues SET explanation = ? WHERE id = ?", updates_expl)

    print(f"  Writing {len(updates_def):,} definitions...")
    cur_cm.executemany("UPDATE clues SET definition = ? WHERE id = ?", updates_def)

    print(f"  Writing {len(updates_wt):,} wordplay types...")
    cur_cm.executemany("UPDATE clues SET wordplay_type = ? WHERE id = ?", updates_wt)

    conn_cm.commit()

    # Post stats
    cur_cm.execute("SELECT COUNT(*) FROM clues WHERE explanation IS NOT NULL AND explanation != ''")
    post_expl = cur_cm.fetchone()[0]
    cur_cm.execute("SELECT COUNT(*) FROM clues WHERE definition IS NOT NULL AND definition != ''")
    post_def = cur_cm.fetchone()[0]
    cur_cm.execute("SELECT COUNT(*) FROM clues WHERE wordplay_type IS NOT NULL AND wordplay_type != ''")
    post_wt = cur_cm.fetchone()[0]

    print(f"\n  AFTER:  explanations={post_expl:,}, definitions={post_def:,}, wordplay_type={post_wt:,}")
    print(f"  Explanations: {pre_expl:,} -> {post_expl:,} (+{post_expl - pre_expl:,})")
    print(f"  Definitions:  {pre_def:,} -> {post_def:,} (+{post_def - pre_def:,})")
    print(f"  Wordplay type: {pre_wt:,} -> {post_wt:,} (+{post_wt - pre_wt:,})")

    conn_cm.close()
    print(f"\nDone ({time.time()-t0:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Migrate explanations from cryptic_new to clues_master")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default is dry run)")
    args = parser.parse_args()
    migrate(apply=args.apply)


if __name__ == "__main__":
    main()
