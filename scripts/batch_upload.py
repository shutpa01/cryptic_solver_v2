"""Upload offline batch results to the database.

Reads data/batch_results.jsonl and replays the writes that store_signature_result
would have made. Also inserts validated enrichment pairs into pending_enrichments
for Excel review.

Usage:
    python scripts/batch_upload.py                          # Upload results + enrichment
    python scripts/batch_upload.py --results-only           # Just clue results, no enrichment
    python scripts/batch_upload.py --enrichment-only        # Just enrichment, no clue results
    python scripts/batch_upload.py --dry-run                # Show what would be written
"""

import argparse
import json
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
CRYPTIC_DB = os.path.join(ROOT, "data", "cryptic_new.db")
RESULTS_PATH = os.path.join(ROOT, "data", "batch_results.jsonl")
VALIDATED_PATH = os.path.join(ROOT, "data", "batch_enrichment_validated.jsonl")


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
    """Read batch_results.jsonl and write to DB."""
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

    for line in lines:
        rec = json.loads(line)
        cid = rec["clue_id"]
        action = rec["action"]

        if action in ("skipped_crossref", "error"):
            skipped += 1
            continue

        if action == "attempted":
            # Mark as S-attempted
            conn.execute(
                "UPDATE clues SET has_solution = 3 WHERE id = ? AND (has_solution IS NULL OR has_solution = 0)",
                (cid,)
            )
            attempted_written += 1

        elif action in ("high_solve", "medium_solve", "hidden_solve", "dd_solve"):
            p = rec["payload"]

            # Check current reviewed status — never overwrite manual reviews
            current = conn.execute("SELECT reviewed FROM clues WHERE id = ?", (cid,)).fetchone()
            already_reviewed = current and current[0] in (1, 2)

            # Update clues table
            if p["definition"]:
                conn.execute(
                    "UPDATE clues SET definition = ? WHERE id = ? AND (definition IS NULL OR definition = '')",
                    (p["definition"], cid)
                )

            conn.execute("UPDATE clues SET wordplay_type = ? WHERE id = ?",
                         (p["wordplay_type"], cid))

            if p["ai_explanation"]:
                conn.execute("UPDATE clues SET ai_explanation = ? WHERE id = ?",
                             (p["ai_explanation"], cid))

            reviewed = p["reviewed"]
            if already_reviewed:
                reviewed = current[0]

            conn.execute("UPDATE clues SET has_solution = ?, reviewed = ? WHERE id = ?",
                         (p["has_solution"], reviewed, cid))

            # Upsert structured_explanations
            existing = conn.execute(
                "SELECT id FROM structured_explanations WHERE clue_id = ?", (cid,)
            ).fetchone()

            wp_types_json = json.dumps(p["wordplay_types"]) if p["wordplay_types"] else json.dumps(["unknown"])
            components_json = json.dumps(p["components"])

            if existing:
                conn.execute("""
                    UPDATE structured_explanations
                    SET definition_text = ?, definition_start = ?, definition_end = ?,
                        wordplay_types = ?, components = ?,
                        model_version = ?, confidence = ?,
                        source = ?, puzzle_number = ?, clue_number = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE clue_id = ?
                """, (
                    p["definition"], p["definition_start"], p["definition_end"],
                    wp_types_json, components_json,
                    p["model_version"], p["confidence"],
                    p["source"], p["puzzle_number"], p["clue_number"],
                    cid
                ))
            else:
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
        if (high_written + medium_written + attempted_written) % 100 == 0:
            conn.commit()

    conn.commit()
    conn.close()

    print(f"\nUpload complete:")
    print(f"  HIGH solves written: {high_written}")
    print(f"  MEDIUM solves written: {medium_written}")
    print(f"  Hidden solves written: {hidden_written}")
    print(f"  DD solves written: {dd_written}")
    print(f"  Attempted (has_solution=3): {attempted_written}")
    print(f"  Skipped: {skipped}")


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
    args = parser.parse_args()

    print("=" * 60)
    print("BATCH UPLOAD")
    print("=" * 60)

    if not args.enrichment_only:
        print("\n--- Uploading clue results ---")
        upload_results(dry_run=args.dry_run)

    if not args.results_only:
        print("\n--- Uploading enrichment ---")
        upload_enrichment(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
