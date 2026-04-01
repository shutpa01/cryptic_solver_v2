"""Upload offline batch results to the database.

Reads data/batch_results.jsonl and replays the writes that store_signature_result
would have made. Also inserts validated enrichment pairs into pending_enrichments
for Excel review.

SAFETY: This script will NEVER overwrite existing explanations or solutions.
It only fills gaps (clues with no solution). A timestamped backup of the
database is created automatically before any writes.

Usage:
    python scripts/batch_upload.py                          # Upload results + enrichment
    python scripts/batch_upload.py --results-only           # Just clue results, no enrichment
    python scripts/batch_upload.py --enrichment-only        # Just enrichment, no clue results
    python scripts/batch_upload.py --dry-run                # Show what would be written
    python scripts/batch_upload.py --file data/backfill_dd_hidden_results.jsonl  # Custom input file
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
CRYPTIC_DB = os.path.join(ROOT, "data", "cryptic_new.db")
RESULTS_PATH = os.path.join(ROOT, "data", "batch_results.jsonl")
VALIDATED_PATH = os.path.join(ROOT, "data", "batch_enrichment_validated.jsonl")


def backup_database():
    """Create a timestamped backup of clues_master.db before any writes."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{CLUES_DB}.bak.{timestamp}"
    print(f"Backing up database to {backup_path} ...")
    shutil.copy2(CLUES_DB, backup_path)
    size_mb = os.path.getsize(backup_path) / (1024 * 1024)
    print(f"Backup complete ({size_mb:.1f} MB)")
    return backup_path


def ensure_structured_table(conn):
    """Create structured_explanations table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS structured_explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clue_id INTEGER UNIQUE,
            definition_text TEXT,
            definition_start INTEGER,
            definition_end INTEGER,
            wordplay_types TEXT,
            components TEXT,
            model_version TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source TEXT,
            puzzle_number TEXT,
            clue_number TEXT
        )
    """)


def upload_results(dry_run=False):
    """Read batch_results.jsonl and write to DB.

    SAFETY: Never overwrites clues that already have solutions (has_solution >= 1).
    Only fills gaps. Every field is individually protected.
    """
    if not os.path.exists(RESULTS_PATH):
        print(f"No results file found at {RESULTS_PATH}")
        return

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Read {len(lines)} lines from {RESULTS_PATH}")

    # Count actions
    counts = {}
    for line in lines:
        rec = json.loads(line)
        action = rec["action"]
        counts[action] = counts.get(action, 0) + 1

    print("Actions found:")
    for action, count in sorted(counts.items()):
        print(f"  {action}: {count}")

    # Pre-scan: check how many clues already have solutions
    conn = sqlite3.connect(CLUES_DB, timeout=60)
    solve_ids = set()
    for line in lines:
        rec = json.loads(line)
        if rec["action"] in ("high_solve", "medium_solve", "hidden_solve", "dd_solve"):
            solve_ids.add(rec["clue_id"])

    already_solved_count = 0
    for cid in solve_ids:
        row = conn.execute(
            "SELECT has_solution FROM clues WHERE id = ?", (cid,)
        ).fetchone()
        if row and row[0] and row[0] >= 1:
            already_solved_count += 1
    conn.close()

    print(f"\nPre-scan:")
    print(f"  Solve records in file: {len(solve_ids)}")
    print(f"  Already solved in DB (will skip): {already_solved_count}")
    print(f"  New solves to write: {len(solve_ids) - already_solved_count}")

    if dry_run:
        print("\n--dry-run: no DB writes.")
        return

    conn = sqlite3.connect(CLUES_DB, timeout=60)
    ensure_structured_table(conn)

    high_written = 0
    medium_written = 0
    hidden_written = 0
    dd_written = 0
    attempted_written = 0
    skipped = 0
    skipped_already_solved = 0

    for line in lines:
        rec = json.loads(line)
        cid = rec["clue_id"]
        action = rec["action"]

        if action in ("skipped_crossref", "error"):
            skipped += 1
            continue

        if action == "attempted":
            # Mark as S-attempted — only if no solution exists
            conn.execute(
                "UPDATE clues SET has_solution = 3 WHERE id = ? AND (has_solution IS NULL OR has_solution = 0)",
                (cid,)
            )
            attempted_written += 1

        elif action in ("high_solve", "medium_solve", "hidden_solve", "dd_solve"):
            p = rec["payload"]

            # --- SAFETY: fetch current state in one query ---
            current = conn.execute(
                "SELECT has_solution, reviewed, wordplay_type, ai_explanation, definition FROM clues WHERE id = ?",
                (cid,)
            ).fetchone()

            if not current:
                skipped += 1
                continue

            cur_has_solution, cur_reviewed, cur_wordplay_type, cur_ai_explanation, cur_definition = current

            # --- SAFETY: never overwrite a clue that already has a solution ---
            if cur_has_solution and cur_has_solution >= 1:
                skipped_already_solved += 1
                continue

            # --- SAFETY: never overwrite manual reviews ---
            already_reviewed = cur_reviewed in (1, 2) if cur_reviewed else False

            # Definition: only write if current is NULL or empty
            if p["definition"] and (not cur_definition or cur_definition.strip() == ""):
                conn.execute(
                    "UPDATE clues SET definition = ? WHERE id = ?",
                    (p["definition"], cid)
                )

            # Wordplay type: only write if current is NULL or empty
            if p["wordplay_type"] and (not cur_wordplay_type or cur_wordplay_type.strip() == ""):
                conn.execute(
                    "UPDATE clues SET wordplay_type = ? WHERE id = ?",
                    (p["wordplay_type"], cid)
                )

            # AI explanation: only write if current is NULL or empty
            if p["ai_explanation"] and (not cur_ai_explanation or cur_ai_explanation.strip() == ""):
                conn.execute(
                    "UPDATE clues SET ai_explanation = ? WHERE id = ?",
                    (p["ai_explanation"], cid)
                )

            # Reviewed: preserve manual reviews
            reviewed = p["reviewed"]
            if already_reviewed:
                reviewed = cur_reviewed

            # has_solution + reviewed
            conn.execute("UPDATE clues SET has_solution = ?, reviewed = ? WHERE id = ?",
                         (p["has_solution"], reviewed, cid))

            # Structured explanations: INSERT only, never overwrite existing
            existing_se = conn.execute(
                "SELECT id FROM structured_explanations WHERE clue_id = ?", (cid,)
            ).fetchone()

            if not existing_se:
                wp_types_json = json.dumps(p["wordplay_types"]) if p["wordplay_types"] else json.dumps(["unknown"])
                components_json = json.dumps(p["components"])

                conn.execute("""
                    INSERT INTO structured_explanations
                    (clue_id, definition_text, definition_start, definition_end,
                     wordplay_types, components, model_version, confidence,
                     source, puzzle_number, clue_number, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (
                    cid, p["definition"], p["definition_start"], p["definition_end"],
                    wp_types_json, components_json,
                    p["model_version"], p["confidence"],
                    p["source"], p["puzzle_number"], p["clue_number"]
                ))

            if action == "high_solve":
                high_written += 1
            elif action == "medium_solve":
                medium_written += 1
            elif action == "hidden_solve":
                hidden_written += 1
            elif action == "dd_solve":
                dd_written += 1

        # Commit every 100 records
        total_written = high_written + medium_written + hidden_written + dd_written + attempted_written
        if total_written > 0 and total_written % 100 == 0:
            conn.commit()

    conn.commit()
    conn.close()

    print(f"\nUpload complete:")
    print(f"  HIGH solves written: {high_written}")
    print(f"  MEDIUM solves written: {medium_written}")
    print(f"  Hidden solves written: {hidden_written}")
    print(f"  DD solves written: {dd_written}")
    print(f"  Attempted (has_solution=3): {attempted_written}")
    print(f"  Skipped (already solved): {skipped_already_solved}")
    print(f"  Skipped (other): {skipped}")


def upload_enrichment(dry_run=False):
    """Insert validated enrichment pairs into pending_enrichments."""
    if not os.path.exists(VALIDATED_PATH):
        print(f"No validated enrichment file at {VALIDATED_PATH}")
        return

    with open(VALIDATED_PATH, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    print(f"Read {len(pairs)} validated enrichment pairs")

    if dry_run:
        print("--dry-run: no DB writes.")
        for p in pairs[:10]:
            print(f"  {p['word']} -> {p['letters']}")
        if len(pairs) > 10:
            print(f"  ... and {len(pairs) - 10} more")
        return

    conn = sqlite3.connect(CLUES_DB, timeout=60)

    inserted = 0
    already_exists = 0

    for p in pairs:
        word = p["word"].lower()
        letters = p["letters"].upper()

        # Check not already in pending or rejected
        existing = conn.execute(
            "SELECT 1 FROM pending_enrichments WHERE LOWER(word) = ? AND UPPER(letters) = ?",
            (word, letters)
        ).fetchone()
        rejected = conn.execute(
            "SELECT 1 FROM rejected_enrichments WHERE LOWER(word) = ? AND UPPER(letters) = ?",
            (word, letters)
        ).fetchone()

        if existing or rejected:
            already_exists += 1
            continue

        # Determine type based on length (same heuristic as batch_enrichment)
        etype = "synonym" if len(letters) >= 3 else "abbreviation"

        conn.execute(
            "INSERT INTO pending_enrichments (type, word, letters, created_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            (etype, word, letters)
        )
        inserted += 1

    conn.commit()
    conn.close()

    print(f"Enrichment upload complete:")
    print(f"  Inserted into pending_enrichments: {inserted}")
    print(f"  Already existed (skipped): {already_exists}")
    print(f"\nNext: python scripts/enrichment_excel.py export")


def main():
    parser = argparse.ArgumentParser(description="Upload offline batch results to DB")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be written")
    parser.add_argument("--results-only", action="store_true", help="Upload clue results only")
    parser.add_argument("--enrichment-only", action="store_true", help="Upload enrichment only")
    parser.add_argument("--file", type=str, help="Path to JSONL results file (default: data/batch_results.jsonl)")
    args = parser.parse_args()

    print("=" * 60)
    print("BATCH UPLOAD")
    print("=" * 60)

    # Create backup once before any writes (skip for dry-run)
    if not args.dry_run:
        backup_database()

    # Override results path if --file provided
    if args.file:
        global RESULTS_PATH
        RESULTS_PATH = os.path.join(ROOT, args.file) if not os.path.isabs(args.file) else args.file

    if not args.enrichment_only:
        print("\n--- Uploading clue results ---")
        upload_results(dry_run=args.dry_run)

    if not args.results_only:
        print("\n--- Uploading enrichment ---")
        upload_enrichment(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
