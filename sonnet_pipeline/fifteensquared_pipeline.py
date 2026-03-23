"""fifteensquared Pipeline — Guardian/Independent solver using human explanations + Haiku parsing.

Mirrors the TFTT pipeline but for fifteensquared.net (Guardian + Independent).

Workflow:
  1. Discover post URL via WordPress API (search or date-range)
  2. Fetch + parse HTML (4 format variants)
  3. Match blog clues to DB clues by answer
  4. Parse explanation with Haiku into structured pieces
  5. Score each parse with confidence system
  6. Store results in clues_master.db

Usage:
    python -m sonnet_pipeline.fifteensquared_pipeline guardian 29958
    python -m sonnet_pipeline.fifteensquared_pipeline independent 12304 --date 2026-03-16
    python -m sonnet_pipeline.fifteensquared_pipeline guardian 29958 --dry-run
"""

import argparse
import os
import re
import sqlite3
import sys

# -- Paths --
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUES_DB = os.path.join(BASE_DIR, "data", "clues_master.db")

# Import the tested scraper
sys.path.insert(0, os.path.join(BASE_DIR, "scraper", "fifteensquared"))
from test_fifteensquared import discover_post_url, parse_post, HEADERS

# Reuse TFTT pipeline components (Haiku parsing, scoring, storage)
from .tftt_pipeline import parse_with_haiku, score_parse, store_tftt_result, clean


def fetch_fifteensquared(puzzle_number, source, pub_date=None):
    """Fetch and parse a fifteensquared page for a Guardian or Independent puzzle.

    Returns list of clue dicts with: clue_number, direction, answer,
    clue_text, enumeration, definition, explanation.
    Returns None if page not found or parsing fails.
    """
    import requests

    url, title, method = discover_post_url(puzzle_number, source, pub_date)
    if not url:
        return None

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print("    fifteensquared HTTP %d for %s" % (resp.status_code, url))
            return None
    except Exception as e:
        print("    fifteensquared fetch error: %s" % e)
        return None

    clues, fmt = parse_post(resp.text)
    if not clues:
        print("    fifteensquared: parsed 0 clues (format: %s)" % fmt)
        return None

    print("    fifteensquared: %d clues from %s (format: %s)" % (len(clues), title or url, fmt))
    return clues


def store_fifteensquared_result(conn, clue_id, parsed, score, definition,
                                raw_explanation="", source_name="guardian"):
    """Store a fifteensquared-parsed result. Wraps store_tftt_result with correct source label."""
    # store_tftt_result writes source="times" in structured_explanations.
    # We need to override that. Call it, then fix the source field.
    store_tftt_result(conn, clue_id, parsed, score, definition, raw_explanation)

    # Fix the source and model_version in structured_explanations
    conn.execute(
        """UPDATE structured_explanations
           SET source = ?, model_version = ?
           WHERE clue_id = ?""",
        (source_name, "fifteensquared+haiku", clue_id),
    )
    conn.commit()


# ============================================================
# Standalone pipeline runner
# ============================================================

def run_fifteensquared_pipeline(source, puzzle_number, pub_date=None,
                                write_db=False, no_fallback=False):
    """Run the fifteensquared pipeline for a Guardian or Independent puzzle."""
    from dotenv import load_dotenv
    load_dotenv()
    import anthropic

    print("=" * 70)
    print("fifteensquared Pipeline — %s puzzle %s" % (source, puzzle_number))
    print("=" * 70)

    # Step 1: Fetch fifteensquared
    print("\n--- Step 1: Fetching fifteensquared blog ---")
    blog_clues = fetch_fifteensquared(puzzle_number, source, pub_date)

    if not blog_clues:
        print("  No fifteensquared page found")
        return

    print("  Fetched %d clues" % len(blog_clues))

    # Load RefDB for scoring
    print("\n--- Loading RefDB ---")
    from signature_solver.db import RefDB
    ref_db = RefDB()

    # Connect to clues_master to match blog clues to DB rows
    conn = sqlite3.connect(CLUES_DB, timeout=30)

    # Build lookup from DB: answer -> (clue_id, clue_text)
    db_rows = conn.execute("""
        SELECT id, clue_number, clue_text, answer
        FROM clues WHERE source = ? AND puzzle_number = ?
    """, (source, str(puzzle_number))).fetchall()

    if not db_rows:
        print("  Puzzle not found in clues_master.db by puzzle_number")
        # For Independent, try by date
        if pub_date:
            db_rows = conn.execute("""
                SELECT id, clue_number, clue_text, answer
                FROM clues WHERE source = ? AND publication_date = ?
            """, (source, pub_date)).fetchall()
            if db_rows:
                print("  Found %d clues by date %s" % (len(db_rows), pub_date))
        if not db_rows:
            print("  No clues found — run the daily scraper first")
            conn.close()
            return

    db_by_answer = {}
    for cid, cnum, ctext, answer in db_rows:
        key = clean(answer)
        db_by_answer[key] = (cid, cnum, ctext, answer)

    # Build compound answer lookup for linked clues (e.g. "1, 4a" = OXFORD + STREET).
    # Guardian API splits these across separate grid entries; the blog combines them.
    see_pattern = re.compile(r'^See (\d+)', re.IGNORECASE)
    by_clue_num = {}
    for cid, cnum, ctext, answer in db_rows:
        by_clue_num.setdefault(cnum, []).append((cid, cnum, ctext, answer))

    db_compound = {}  # cleaned compound answer -> (primary_clue_id, primary_cnum, primary_ctext, full_answer)
    for cid, cnum, ctext, answer in db_rows:
        m = see_pattern.match((ctext or "").strip())
        if not m:
            continue
        parent_num = m.group(1)
        for pid, pnum, ptxt, pans in by_clue_num.get(parent_num, []):
            if see_pattern.match((ptxt or "").strip()):
                continue  # skip if parent is also a "See X"
            compound_key = clean(pans + answer)
            full_answer = pans + " " + answer
            db_compound[compound_key] = (pid, pnum, ptxt, full_answer)

    # Step 2+3: Parse with Haiku and score
    print("\n--- Step 2+3: Haiku parsing + scoring ---")
    client = anthropic.Anthropic()
    haiku_cost = 0
    high_count = 0
    medium_count = 0
    low_count = 0
    failed_count = 0
    compound_updates = []  # track primary clue answer updates for compound clues

    for bc in blog_clues:
        answer_key = clean(bc["answer"])
        db_match = db_by_answer.get(answer_key)
        is_compound = False

        if not db_match:
            # Try compound lookup
            db_match = db_compound.get(answer_key)
            if db_match:
                is_compound = True
                print("  %s %s — compound match" % (bc["clue_number"], bc["answer"]))

        if not db_match:
            print("  %s %s — no DB match" % (bc["clue_number"], bc["answer"]))
            failed_count += 1
            continue

        clue_id, clue_num, clue_text, db_answer = db_match

        # For compound clues, use the blog's answer (properly spaced) for parsing and DB update
        if is_compound:
            db_answer = bc["answer"].upper()  # blog has correct spacing e.g. "OXFORD STREET"
            compound_updates.append((clue_id, db_answer))
        explanation = bc.get("explanation", "")
        blog_definition = bc.get("definition", "")

        if not explanation:
            print("  %s %s — no explanation" % (clue_num, db_answer))
            failed_count += 1
            continue

        # Parse with Haiku
        parsed, usage = parse_with_haiku(client, clue_text, db_answer, explanation)

        if usage:
            haiku_cost += (usage.input_tokens * 0.80 + usage.output_tokens * 4.0) / 1_000_000

        if not parsed:
            print("  %s %s — Haiku parse failed" % (clue_num, db_answer))
            low_count += 1
            continue

        # Score
        score, reasons = score_parse(parsed, db_answer, ref_db)

        if score >= 80:
            band = "HIGH"
            high_count += 1
        elif score >= 70:
            band = "MEDIUM"
            medium_count += 1
        else:
            band = "LOW"
            low_count += 1

        reason_str = ", ".join("%s(%d)" % (r, d) for r, d in reasons) if reasons else "clean"
        print("  %s %-20s %3d %s  %s" % (clue_num, db_answer, score, band, reason_str))

        # Store any successful parse — blog explanations are always worth keeping
        if write_db and parsed:
            store_fifteensquared_result(
                conn, clue_id, parsed, score, blog_definition,
                raw_explanation=explanation, source_name=source,
            )

            # Also backfill puzzle_number if missing (Independent)
            if puzzle_number:
                conn.execute(
                    "UPDATE clues SET puzzle_number = ? WHERE id = ? AND (puzzle_number IS NULL OR puzzle_number = '')",
                    (str(puzzle_number), clue_id),
                )
                conn.commit()

    # Update primary clue answers for compound clues (e.g. "OXFORD" -> "OXFORD STREET")
    if write_db and compound_updates:
        for clue_id, full_answer in compound_updates:
            conn.execute(
                "UPDATE clues SET answer = ? WHERE id = ?",
                (full_answer, clue_id),
            )
        conn.commit()
        print("\n  Updated %d compound answers" % len(compound_updates))

    conn.close()

    # Summary
    print("\n" + "=" * 70)
    print("Results: %d HIGH, %d MEDIUM, %d LOW, %d failed" % (
        high_count, medium_count, low_count, failed_count))
    print("Cost: Haiku $%.4f" % haiku_cost)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="fifteensquared Pipeline for Guardian/Independent")
    parser.add_argument("source", choices=["guardian", "independent"], help="Source")
    parser.add_argument("puzzle", type=int, help="Puzzle number")
    parser.add_argument("--date", help="Publication date YYYY-MM-DD (helps URL discovery)")
    parser.add_argument("--write-db", action="store_true", help="Write results to clues_master.db")
    parser.add_argument("--dry-run", action="store_true", help="Parse and score only")
    parser.add_argument("--no-fallback", action="store_true", help="Skip Sonnet fallback")
    args = parser.parse_args()

    run_fifteensquared_pipeline(
        args.source, args.puzzle,
        pub_date=args.date,
        write_db=args.write_db,
        no_fallback=args.dry_run or args.no_fallback,
    )


if __name__ == "__main__":
    main()
