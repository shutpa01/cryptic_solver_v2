"""Interactive DB gap review — approve/reject pending inserts one by one.

Usage:
    python -m sonnet_pipeline.review_gaps
    python -m sonnet_pipeline.review_gaps documents/pending_gaps_telegraph_31176.json
"""

import glob
import json
import os
import sqlite3
import subprocess
import sys

CRYPTIC_DB = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\cryptic_new.db"
MASTER_DB = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db"
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


def _sync_definition(master_conn, clue_id, definition):
    """Sync a manual definition to definition_answers_augmented in cryptic_new.db."""
    row = master_conn.execute(
        "SELECT answer FROM clues WHERE id = ?", (clue_id,)
    ).fetchone()
    if not row or not row[0]:
        return
    answer = row[0].upper()
    defn = definition.lower()
    try:
        cryptic_conn = sqlite3.connect(CRYPTIC_DB, timeout=30)
        exists = cryptic_conn.execute(
            "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=? AND LOWER(answer)=?",
            (defn, answer.lower())
        ).fetchone()
        if not exists:
            cryptic_conn.execute(
                "INSERT INTO definition_answers_augmented (definition, answer, source) VALUES (?, ?, ?)",
                (defn, answer, "manual")
            )
            cryptic_conn.commit()
            print("  -> synced def '%s' -> %s to reference DB" % (defn, answer))
        cryptic_conn.close()
    except Exception as e:
        print("  -> warning: could not sync definition: %s" % e)


def manual_entry_phase(source, puzzle, stats):
    """Phase 2: manually enter definition/type/explanation for weak clues."""
    conn = sqlite3.connect(MASTER_DB, timeout=30)
    rows = conn.execute("""
        SELECT id, clue_number, direction, clue_text, answer,
               definition, wordplay_type, ai_explanation, has_solution
        FROM clues
        WHERE source = ? AND puzzle_number = ?
          AND (has_solution IS NULL OR has_solution = 0 OR has_solution = 2)
        ORDER BY CAST(clue_number AS INTEGER)
    """, (source, puzzle)).fetchall()

    if not rows:
        print("\nAll clues have full solutions — nothing to annotate.")
        conn.close()
        return 0

    print()
    print("=" * 70)
    print("MANUAL EXPLANATION ENTRY")
    print("=" * 70)
    print("  %d clues without full solutions" % len(rows))
    print("  Target DB: %s" % MASTER_DB)
    print()

    try:
        choice = input("Enter manual explanations? [y/n] > ").strip().lower()
    except EOFError:
        choice = "n"
    if choice not in ("y", "yes"):
        conn.close()
        return 0

    updated = 0
    try:
        for i, row in enumerate(rows):
            clue_id, clue_num, direction, clue_text, answer = row[:5]
            cur_def, cur_type, cur_expl, cur_solved = row[5:]

            d_char = direction[0].upper() if direction else ""
            ref = "%s%s" % (clue_num, d_char)
            status = "PARTIAL" if cur_solved == 2 else "FAILED"

            print()
            print("[%d/%d] %s — %s: \"%s\" = %s" % (
                i + 1, len(rows), status, ref, clue_text, answer))
            print("  Current: def=%s | type=%s | expl=%s" % (
                cur_def or "None", cur_type or "None",
                (cur_expl[:60] + "...") if cur_expl and len(cur_expl) > 60 else cur_expl or "None"))

            while True:
                try:
                    action = input("  [a]ll / [d]ef / [t]ype / [e]xpl / [s]kip / [q]uit > ").strip().lower()
                except EOFError:
                    action = "q"

                if action in ("q", "quit"):
                    raise KeyboardInterrupt
                if action in ("s", "skip"):
                    break
                if action not in ("a", "all", "d", "def", "t", "type", "e", "expl"):
                    print("  Invalid choice — enter a/d/t/e/s/q")
                    continue

                new_def = cur_def
                new_type = cur_type
                new_expl = cur_expl

                if action in ("a", "all", "d", "def"):
                    try:
                        val = input("    Definition: ").strip()
                    except EOFError:
                        val = ""
                    if val:
                        new_def = val

                if action in ("a", "all", "t", "type"):
                    try:
                        val = input("    Wordplay type: ").strip()
                    except EOFError:
                        val = ""
                    if val:
                        new_type = val

                if action in ("a", "all", "e", "expl"):
                    try:
                        val = input("    Explanation: ").strip()
                    except EOFError:
                        val = ""
                    if val:
                        new_expl = val

                # Check if anything changed
                if (new_def, new_type, new_expl) == (cur_def, cur_type, cur_expl):
                    print("  (no changes)")
                    break

                # Update only changed fields
                sets = []
                params = []
                if new_def != cur_def:
                    sets.append("definition = ?")
                    params.append(new_def)
                if new_type != cur_type:
                    sets.append("wordplay_type = ?")
                    params.append(new_type)
                if new_expl != cur_expl:
                    sets.append("ai_explanation = ?")
                    params.append(new_expl)

                # Mark as solved if all three fields are now populated
                if new_def and new_type and new_expl:
                    sets.append("has_solution = 1")
                elif new_def or new_type or new_expl:
                    sets.append("has_solution = 2")

                params.append(clue_id)
                conn.execute(
                    "UPDATE clues SET %s WHERE id = ?" % ", ".join(sets),
                    params
                )
                conn.commit()

                # Sync new definition to definition_answers_augmented
                if new_def and new_def != cur_def:
                    _sync_definition(conn, clue_id, new_def)

                updated += 1
                print("  -> updated (%s)" % ", ".join(
                    f for f in ["def" if new_def != cur_def else "",
                                "type" if new_type != cur_type else "",
                                "expl" if new_expl != cur_expl else ""] if f))
                break

    except KeyboardInterrupt:
        print("\n\nManual entry interrupted.")

    conn.close()

    print()
    print("Manual entry: %d clue(s) updated" % updated)
    return updated


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
        raw = json.load(f)

    # Support both old format (plain list) and new format (dict with metadata)
    if isinstance(raw, dict):
        gaps = raw.get("gaps", [])
        source = raw.get("source", "")
        puzzle = raw.get("puzzle", "")
        stats = raw.get("stats", {})
    else:
        gaps = raw
        source = ""
        puzzle = ""
        stats = {}

    if not gaps:
        print("No gaps in %s" % gaps_path)
        os.remove(gaps_path)
        sys.exit(0)

    # Header with puzzle summary
    print("=" * 70)
    print("DB GAP REVIEW: %s" % os.path.basename(gaps_path))
    if source and puzzle:
        print("  Puzzle: %s #%s" % (source, puzzle))
    if stats:
        total = stats.get("total", 0)
        print("  Results: %d/%d assembled (avg %.0f/100)" % (
            stats.get("assembled", 0), total, stats.get("avg_score", 0)))
        print("  High: %d | Medium: %d | Low: %d | Failed: %d" % (
            stats.get("high", 0), stats.get("medium", 0),
            stats.get("low", 0), stats.get("failed", 0)))
    print("  Gaps to review: %d" % len(gaps))
    print("  Target DB: %s" % CRYPTIC_DB)
    print("=" * 70)
    print()

    conn = sqlite3.connect(CRYPTIC_DB, timeout=30)
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

    # Phase 2: Re-run pipeline (BEFORE manual entry, so manual adds aren't overwritten)
    reran = False
    failed_count = stats.get("failed", 0)
    medium_count = stats.get("medium", 0)
    low_count = stats.get("low", 0)
    rerun_worthy = failed_count + medium_count + low_count
    if source and puzzle and (approved > 0 or rerun_worthy > 0):
        print()
        if approved > 0:
            print("You approved %d DB entries — re-running may improve results." % approved)
        if rerun_worthy > 0:
            print("Puzzle has %d failed + %d medium + %d low clues." % (
                failed_count, medium_count, low_count))
        try:
            choice = input("\nRe-run pipeline on partials & failed? [y/n] > ").strip().lower()
        except EOFError:
            choice = "n"
        if choice in ("y", "yes"):
            cmd = [
                sys.executable, "-m", "sonnet_pipeline.run",
                puzzle, "--source", source,
                "--write-db", "--partials",
            ]
            print("\nRunning: %s" % " ".join(cmd))
            print("=" * 70)
            subprocess.run(cmd)
            reran = True

    # Phase 3: Manual explanation entry for clues that STILL need help
    # Done AFTER re-run so manual adds aren't overwritten by the pipeline
    if source and puzzle:
        manual_entry_phase(source, puzzle, stats)


if __name__ == "__main__":
    main()
