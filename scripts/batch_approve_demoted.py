"""Batch-approve the 19 demoted-HIGH clues with corrected explanations.

Reads the response file (claude_review format), updates each clue's
definition/wordplay_type/ai_explanation, and forces confidence=1.0 +
reviewed=1 on the specified clue_ids. Equivalent to manually clicking
the admin /approve button for each, but done from the corrected drafts.

Usage:
    python scripts/batch_approve_demoted.py <response_file> --dry-run
    python scripts/batch_approve_demoted.py <response_file>
"""
import argparse
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = ROOT / "data" / "clues_master.db"


def parse_response(text):
    results = []
    blocks = re.split(r"^---\s*$", text, flags=re.MULTILINE)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        entry = {}
        for line in block.splitlines():
            line = line.strip()
            if line.upper().startswith("ID:"):
                try:
                    entry["id"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.upper().startswith("WORDPLAY:"):
                entry["wordplay_type"] = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("DEFINITION:"):
                entry["definition"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("EXPLANATION:"):
                entry["explanation"] = line.split(":", 1)[1].strip()
        if "id" in entry and "explanation" in entry:
            results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("response_file")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    text = Path(args.response_file).read_text(encoding="utf-8")
    entries = parse_response(text)
    print(f"Parsed {len(entries)} entries from {args.response_file}")

    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row

    now = datetime.utcnow().isoformat()
    updates = 0
    missing = 0

    for e in entries:
        cid = e["id"]
        row = conn.execute(
            "SELECT source, puzzle_number, clue_number, direction, answer FROM clues WHERE id = ?",
            (cid,),
        ).fetchone()
        if row is None:
            print(f"  MISSING id={cid}")
            missing += 1
            continue
        label = f"{row['clue_number']}{row['direction'][0]}"
        print(f"  [{cid}] {row['source']:12s} #{row['puzzle_number']:6s} {label:4s} "
              f"{row['answer']:15s}  {e['wordplay_type']:18s}  {e['explanation'][:70]}")
        if args.dry_run:
            continue

        conn.execute(
            """
            UPDATE clues
            SET definition = ?, wordplay_type = ?, ai_explanation = ?,
                has_solution = 1, reviewed = 1
            WHERE id = ?
            """,
            (e["definition"], e["wordplay_type"], e["explanation"], cid),
        )

        existing = conn.execute(
            "SELECT id FROM structured_explanations WHERE clue_id = ?", (cid,)
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE structured_explanations
                SET definition_text = ?, wordplay_types = ?, components = ?,
                    model_version = ?, confidence = ?, updated_at = ?
                WHERE clue_id = ?
                """,
                (e["definition"], f'["{e["wordplay_type"]}"]',
                 '{"source":"manual_approve","wordplay_type":"' + e["wordplay_type"] + '"}',
                 "manual_approve", 1.0, now, cid),
            )
        else:
            conn.execute(
                """
                INSERT INTO structured_explanations
                (clue_id, definition_text, wordplay_types, components, model_version,
                 confidence, source, puzzle_number, clue_number, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (cid, e["definition"], f'["{e["wordplay_type"]}"]',
                 '{"source":"manual_approve","wordplay_type":"' + e["wordplay_type"] + '"}',
                 "manual_approve", 1.0, row["source"], row["puzzle_number"],
                 row["clue_number"], now, now),
            )
        updates += 1

    if not args.dry_run:
        conn.commit()
    conn.close()

    verb = "Would update" if args.dry_run else "Updated"
    print(f"\n{verb}: {updates}, Missing: {missing}")


if __name__ == "__main__":
    main()
