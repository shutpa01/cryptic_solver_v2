"""FifteenSquared catch-up — parse blog explanations for Guardian/Independent.

Runs as a second pass after the main pipeline. Picks up blog posts that
weren't available at 2am. Zero Sonnet cost — uses only Haiku for parsing.

Usage:
    python scripts/fifteensquared_catchup.py                    # today, both sources
    python scripts/fifteensquared_catchup.py --source guardian   # today, guardian only
    python scripts/fifteensquared_catchup.py --source independent
    python scripts/fifteensquared_catchup.py --date 2026-04-08   # specific date
    python scripts/fifteensquared_catchup.py --dry-run            # show what would run
"""

import argparse
import os
import re
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLUES_DB = ROOT / "data" / "clues_master.db"

SOURCES = ["guardian", "independent"]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    safe = str(msg).encode('cp1252', errors='replace').decode('cp1252')
    print(f"[{ts}] {safe}", flush=True)


def find_unsolved_puzzles(target_date, source_filter=None):
    """Find puzzles with unsolved clues for the given date."""
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row

    sources = [source_filter] if source_filter else SOURCES
    puzzles = []

    for source in sources:
        rows = conn.execute("""
            SELECT c.puzzle_number,
                   COUNT(*) as total,
                   SUM(CASE WHEN se.confidence >= 0.7 THEN 1 ELSE 0 END) as high
            FROM clues c
            LEFT JOIN structured_explanations se ON se.clue_id = c.id
            WHERE c.source = ? AND c.publication_date = ?
              AND c.answer IS NOT NULL AND c.answer != ''
            GROUP BY c.puzzle_number
        """, (source, target_date)).fetchall()

        for r in rows:
            unsolved = r["total"] - (r["high"] or 0)
            if unsolved > 0:
                puzzles.append({
                    "source": source,
                    "puzzle_number": r["puzzle_number"],
                    "total": r["total"],
                    "high": r["high"] or 0,
                    "unsolved": unsolved,
                })

    conn.close()
    return puzzles


def run_fifteensquared_pass(source, puzzle_number, pub_date, dry_run=False):
    """Run FifteenSquared parsing on unsolved clues for one puzzle.

    Returns (solved, skipped, failed) counts.
    """
    from sonnet_pipeline.fifteensquared_pipeline import (
        fetch_fifteensquared, store_fifteensquared_result,
    )
    from sonnet_pipeline.tftt_pipeline import parse_with_haiku, score_parse
    from signature_solver.db import RefDB
    import anthropic

    # Fetch blog explanations
    fs_clues = fetch_fifteensquared(int(puzzle_number), source, pub_date)
    if not fs_clues:
        log(f"  No FifteenSquared page found for {source} #{puzzle_number}")
        return 0, 0, 0

    log(f"  Fetched {len(fs_clues)} clues from FifteenSquared")

    if dry_run:
        return len(fs_clues), 0, 0

    # Load resources
    ref_db = RefDB()
    haiku_client = anthropic.Anthropic()

    # Get unsolved clues from DB
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row

    clues = conn.execute("""
        SELECT c.id, c.clue_text, c.answer, c.clue_number, c.direction
        FROM clues c
        LEFT JOIN structured_explanations se ON se.clue_id = c.id
        WHERE c.source = ? AND c.puzzle_number = ?
          AND c.answer IS NOT NULL AND c.answer != ''
          AND (se.confidence IS NULL OR se.confidence < 0.7)
          AND (c.reviewed IS NULL OR c.reviewed != 1)
    """, (source, str(puzzle_number))).fetchall()

    if not clues:
        log(f"  All clues already solved or reviewed")
        conn.close()
        return 0, 0, 0

    # Build lookup: clean answer -> blog clue
    fs_by_answer = {}
    for fc in fs_clues:
        key = re.sub(r'[^A-Z]', '', fc["answer"].upper())
        fs_by_answer[key] = fc

    solved = 0
    skipped = 0
    failed = 0

    for clue in clues:
        answer_clean = re.sub(r'[^A-Z]', '', clue["answer"].upper())
        fc = fs_by_answer.get(answer_clean)

        if not fc or not fc.get("explanation"):
            skipped += 1
            continue

        # Parse with Haiku
        try:
            parsed, usage = parse_with_haiku(
                haiku_client, clue["clue_text"], clue["answer"], fc["explanation"]
            )
        except Exception as e:
            log(f"    Haiku error on {clue['clue_number']}{clue['direction'][0].upper()}: {e}")
            failed += 1
            continue

        if not parsed:
            failed += 1
            continue

        # Score
        score, reasons = score_parse(parsed, clue["answer"], ref_db)

        if score >= 70:
            store_fifteensquared_result(
                conn, clue["id"], parsed, score,
                fc.get("definition", ""),
                raw_explanation=fc.get("explanation", ""),
                source_name=source,
            )
            solved += 1
            cnum = clue["clue_number"]
            direction = clue["direction"][0].upper()
            log(f"    [{score}] {cnum}{direction}. {clue['answer']}")
        else:
            reason_str = ", ".join("%s(%d)" % (r, d) for r, d in reasons)
            failed += 1

    conn.commit()
    conn.close()

    return solved, skipped, failed


def main():
    parser = argparse.ArgumentParser(description="FifteenSquared catch-up pass")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date (YYYY-MM-DD). Default: today")
    parser.add_argument("--source", choices=["guardian", "independent"],
                        help="Run for one source only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    args = parser.parse_args()

    target_date = args.date or date.today().isoformat()

    log("=" * 60)
    log(f"FIFTEENSQUARED CATCH-UP — {target_date}")
    log("=" * 60)

    puzzles = find_unsolved_puzzles(target_date, args.source)

    if not puzzles:
        log("No unsolved Guardian/Independent puzzles for this date.")
        return

    for p in puzzles:
        log(f"\n{p['source']} #{p['puzzle_number']}: "
            f"{p['unsolved']}/{p['total']} unsolved")

        if args.dry_run:
            log(f"  [DRY RUN] Would attempt FifteenSquared parsing")
            continue

        solved, skipped, failed = run_fifteensquared_pass(
            p["source"], p["puzzle_number"], target_date, dry_run=args.dry_run
        )
        log(f"  Result: {solved} solved, {skipped} no blog match, {failed} failed")

    log("\n" + "=" * 60)
    log("CATCH-UP COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    main()
