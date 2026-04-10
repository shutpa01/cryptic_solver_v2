"""Collect non-HIGH clues from completed pipeline runs for Claude review.

Generates a formatted report that can be pasted into a Claude conversation.
Claude's responses are then parsed by ingest_claude_review.py and run through
the verifier before being written to the database.

Usage:
    python scripts/collect_for_review.py                          # today's DT + DM
    python scripts/collect_for_review.py --date 2026-04-10        # specific date
    python scripts/collect_for_review.py --source dailymail --puzzle 17854  # specific puzzle
    python scripts/collect_for_review.py --threshold 80           # collect below this score
"""

import argparse
import sqlite3
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = ROOT / "data" / "clues_master.db"
OUTPUT_DIR = ROOT / "data"


def collect_clues(source=None, puzzle_number=None, target_date=None, threshold=80):
    """Collect clues below the confidence threshold.

    Returns list of dicts with clue data.
    """
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row

    if source and puzzle_number:
        # Specific puzzle
        where = "c.source = ? AND c.puzzle_number = ?"
        params = (source, str(puzzle_number))
    elif target_date:
        # All puzzles from a date (DT + DM only by default)
        where = "c.publication_date = ? AND c.source IN ('telegraph', 'dailymail')"
        params = (target_date,)
    else:
        target_date = date.today().isoformat()
        where = "c.publication_date = ? AND c.source IN ('telegraph', 'dailymail')"
        params = (target_date,)

    rows = conn.execute(f"""
        SELECT c.id, c.source, c.puzzle_number, c.clue_number, c.direction,
               c.clue_text, c.answer, c.enumeration, c.definition,
               c.wordplay_type, c.ai_explanation,
               se.confidence, se.model_version
        FROM clues c
        LEFT JOIN structured_explanations se ON se.clue_id = c.id
        WHERE {where}
          AND c.answer IS NOT NULL AND c.answer != ''
        ORDER BY c.source, c.puzzle_number, c.direction, CAST(c.clue_number AS INTEGER)
    """, params).fetchall()
    conn.close()

    clues = []
    for r in rows:
        conf = r["confidence"]
        if conf is not None:
            score = int(conf * 100) if conf <= 1 else int(conf)
        elif r["model_version"]:
            score = 0  # FAIL
        else:
            score = 0  # PENDING

        if score < threshold:
            clues.append({
                "id": r["id"],
                "source": r["source"],
                "puzzle_number": r["puzzle_number"],
                "clue_number": r["clue_number"],
                "direction": r["direction"],
                "clue_text": r["clue_text"],
                "answer": r["answer"],
                "enumeration": r["enumeration"],
                "definition": r["definition"],
                "wordplay_type": r["wordplay_type"],
                "ai_explanation": r["ai_explanation"],
                "score": score,
            })

    return clues


def format_report(clues):
    """Format clues into a prompt for Claude."""
    if not clues:
        return "No clues to review.\n"

    lines = []
    lines.append("Please solve each cryptic crossword clue below. For each one, provide:")
    lines.append("- wordplay_type (e.g. charade, anagram, container, hidden, double_definition, reversal, deletion, homophone, spoonerism, acrostic, cryptic_definition)")
    lines.append("- definition (the straight definition part of the clue)")
    lines.append("- explanation (how the wordplay works, showing each piece)")
    lines.append("")
    lines.append("Format each answer EXACTLY like this (must match verifier format):")
    lines.append("")
    lines.append("For charade/container/deletion:")
    lines.append("---")
    lines.append("ID: 12345")
    lines.append("WORDPLAY: charade")
    lines.append("DEFINITION: having three parties")
    lines.append('EXPLANATION: TRI (abbreviation="three") + LATE (synonym="delayed") + R (first letter of "year") + AL (synonym="Al") = TRILATERAL; definition: "having three parties"')
    lines.append("---")
    lines.append("")
    lines.append("For anagram:")
    lines.append("---")
    lines.append("ID: 12346")
    lines.append("WORDPLAY: anagram")
    lines.append("DEFINITION: fruit")
    lines.append("EXPLANATION: anagram of LOGO + ON + BREAD = BLOODORANGE; definition: \"fruit\"")
    lines.append("---")
    lines.append("")
    lines.append("For hidden:")
    lines.append("---")
    lines.append("ID: 12347")
    lines.append("WORDPLAY: hidden")
    lines.append("DEFINITION: skill")
    lines.append('EXPLANATION: hidden in "cARTography"; definition: "skill"')
    lines.append("---")
    lines.append("")
    lines.append("For double definition:")
    lines.append("---")
    lines.append("ID: 12348")
    lines.append("WORDPLAY: double_definition")
    lines.append("DEFINITION: pack")
    lines.append("EXPLANATION: Double definition: pack = STUFF, things = STUFF")
    lines.append("---")
    lines.append("")
    lines.append("For homophone:")
    lines.append("---")
    lines.append("ID: 12349")
    lines.append("WORDPLAY: homophone")
    lines.append("DEFINITION: coast")
    lines.append('EXPLANATION: SHORE sounds like "sure" (mentioned); definition: "coast"')
    lines.append("---")
    lines.append("")
    lines.append("CRITICAL: Use UPPERCASE for letter pieces in explanations. Always end with ; definition: \"...\"")
    lines.append("Use (synonym=\"word\"), (abbreviation=\"word\"), (first letter of \"word\") for piece sources.")
    lines.append("")

    # Group by source/puzzle
    current_group = None
    for c in clues:
        group = f"{c['source']} #{c['puzzle_number']}"
        if group != current_group:
            current_group = group
            lines.append(f"=== {group} ===")
            lines.append("")

        label = f"{c['clue_number']}{c['direction'][0]}"
        score_note = f" [score={c['score']}]" if c['score'] > 0 else " [unsolved]"
        existing = ""
        if c['ai_explanation']:
            existing = f"\n   Current explanation: {c['ai_explanation']}"

        lines.append(f"ID: {c['id']}")
        lines.append(f"{label}. {c['clue_text']} ({c['enumeration']})")
        lines.append(f"   Answer: {c['answer']}{score_note}{existing}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect clues for Claude review")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--puzzle", type=str, default=None)
    parser.add_argument("--threshold", type=int, default=80,
                        help="Collect clues below this confidence score (default: 80)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: data/claude_review_YYYY-MM-DD.txt)")
    args = parser.parse_args()

    target_date = args.date or date.today().isoformat()

    clues = collect_clues(
        source=args.source,
        puzzle_number=args.puzzle,
        target_date=target_date if not args.source else None,
        threshold=args.threshold,
    )

    report = format_report(clues)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / f"claude_review_{target_date}.txt"

    output_path.write_text(report, encoding="utf-8")
    print(f"Collected {len(clues)} clues for review -> {output_path}")
    print(f"Paste the contents into a Claude conversation, then run:")
    print(f"  python scripts/ingest_claude_review.py {output_path.stem}_response.txt")


if __name__ == "__main__":
    main()
