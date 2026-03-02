"""Interactive DB gap review — approve/reject pending inserts one by one.

Usage:
    python -m sonnet_pipeline.review_gaps
    python -m sonnet_pipeline.review_gaps documents/pending_gaps_telegraph_31176.json
"""

import glob
import json
import os
import sqlite3
import sys

CRYPTIC_DB = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\cryptic_new.db"
OUTPUT_DIR = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\documents"


def find_latest_gaps_file():
    """Find the most recent pending_gaps_*.json in the output directory."""
    pattern = os.path.join(OUTPUT_DIR, "pending_gaps_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def already_exists(conn, gap):
    """Check if a gap entry already exists in the target table."""
    if gap["type"] == "definition":
        row = conn.execute(
            "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=? AND LOWER(answer)=?",
            (gap["definition"].lower(), gap["answer"].lower())
        ).fetchone()
        return row is not None

    elif gap["type"] == "synonym":
        row = conn.execute(
            "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND LOWER(synonym)=?",
            (gap["word"].lower(), gap["letters"].lower())
        ).fetchone()
        return row is not None

    elif gap["type"] == "abbreviation":
        row = conn.execute(
            "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? AND LOWER(substitution)=? AND category='abbreviation'",
            (gap["word"].lower(), gap["letters"].upper())
        ).fetchone()
        return row is not None

    return False


def insert_gap(conn, gap):
    """Execute the INSERT for an approved gap."""
    if gap["type"] == "definition":
        conn.execute(
            "INSERT INTO definition_answers_augmented (definition, answer, source) VALUES (?, ?, ?)",
            (gap["definition"].lower(), gap["answer"].upper(), "pipeline")
        )
    elif gap["type"] == "synonym":
        conn.execute(
            "INSERT INTO synonyms_pairs (word, synonym, source) VALUES (?, ?, ?)",
            (gap["word"].lower(), gap["letters"].upper(), "pipeline")
        )
    elif gap["type"] == "abbreviation":
        conn.execute(
            "INSERT INTO wordplay (indicator, substitution, category, frequency, confidence, notes) "
            "VALUES (?, ?, 'abbreviation', 0, 'medium', 'pipeline')",
            (gap["word"].lower(), gap["letters"].upper())
        )


def format_sql(gap):
    """Return the SQL statement that would be executed, for display."""
    if gap["type"] == "definition":
        return (
            "INSERT INTO definition_answers_augmented (definition, answer, source)\n"
            "    VALUES ('%s', '%s', 'pipeline')"
            % (gap["definition"].lower(), gap["answer"].upper())
        )
    elif gap["type"] == "synonym":
        return (
            "INSERT INTO synonyms_pairs (word, synonym, source)\n"
            "    VALUES ('%s', '%s', 'pipeline')"
            % (gap["word"].lower(), gap["letters"].upper())
        )
    elif gap["type"] == "abbreviation":
        return (
            "INSERT INTO wordplay (indicator, substitution, category, frequency, confidence, notes)\n"
            "    VALUES ('%s', '%s', 'abbreviation', 0, 'medium', 'pipeline')"
            % (gap["word"].lower(), gap["letters"].upper())
        )
    return "-- unknown gap type: %s" % gap["type"]


def format_clue_context(gap):
    """Return the clue context line for display."""
    d = gap.get("direction", "")
    d_char = d[0].upper() if d else ""
    ref = "%s%s" % (gap["clue_number"], d_char)
    return 'Clue %s: "%s" = %s (%d/100)' % (
        ref, gap.get("clue", "?"), gap["answer"], gap.get("score", 0))


def main():
    # Find gaps file
    if len(sys.argv) > 1:
        gaps_path = sys.argv[1]
    else:
        gaps_path = find_latest_gaps_file()

    if not gaps_path or not os.path.exists(gaps_path):
        print("No pending gaps file found.")
        if not gaps_path:
            print("Run the pipeline first to generate one, or pass a path as argument.")
        else:
            print("File not found: %s" % gaps_path)
        sys.exit(1)

    with open(gaps_path, "r", encoding="utf-8") as f:
        gaps = json.load(f)

    if not gaps:
        print("No gaps in %s" % gaps_path)
        os.remove(gaps_path)
        sys.exit(0)

    print("=" * 70)
    print("DB GAP REVIEW: %s" % os.path.basename(gaps_path))
    print("  %d entries to review" % len(gaps))
    print("  Target DB: %s" % CRYPTIC_DB)
    print("=" * 70)
    print()

    conn = sqlite3.connect(CRYPTIC_DB)
    approved = 0
    skipped = 0
    existed = 0
    all_reviewed = False

    try:
        for i, gap in enumerate(gaps):
            # Check dedup first
            if already_exists(conn, gap):
                print("[%d/%d] %s — already exists, skipping" % (
                    i + 1, len(gaps), gap["type"].upper()))
                existed += 1
                continue

            print("[%d/%d] %s" % (i + 1, len(gaps), gap["type"].upper()))
            print("  %s" % format_clue_context(gap))
            print("  %s" % format_sql(gap))

            while True:
                try:
                    choice = input("  [y]es / [n]o / [q]uit > ").strip().lower()
                except EOFError:
                    choice = "q"
                if choice in ("y", "yes"):
                    insert_gap(conn, gap)
                    conn.commit()
                    approved += 1
                    print("  -> inserted")
                    break
                elif choice in ("n", "no"):
                    skipped += 1
                    break
                elif choice in ("q", "quit"):
                    skipped += len(gaps) - i - existed
                    raise KeyboardInterrupt
                else:
                    print("  Please enter y, n, or q")
            print()

        all_reviewed = True
    except KeyboardInterrupt:
        print("\n\nReview interrupted.")
        all_reviewed = False

    conn.close()

    print()
    print("=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print("  Approved:        %d" % approved)
    print("  Skipped:         %d" % skipped)
    print("  Already existed: %d" % existed)
    print("  Total:           %d" % len(gaps))

    # Only delete the file if all gaps were reviewed
    if all_reviewed:
        try:
            os.remove(gaps_path)
            print("\nDeleted %s" % gaps_path)
        except OSError as e:
            print("\nCould not delete %s: %s" % (gaps_path, e))
    else:
        print("\nFile kept (review incomplete): %s" % gaps_path)


if __name__ == "__main__":
    main()
