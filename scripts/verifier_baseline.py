"""Snapshot verifier scores for a fixed set of clues, for before/after regression checks.

Reads from clues_master.db and re-runs the verifier on each clue, capturing
score, verdict, and full check list. Saves to JSON for later diffing.

Usage:
    python scripts/verifier_baseline.py --out data/verifier_baseline_2026_04_27.json

Default: snapshots clues from the puzzles processed in this session.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLUES_DB = ROOT / "data" / "clues_master.db"

DEFAULT_PUZZLES = [
    ("telegraph", "31224"),
    ("telegraph", "3366"),
    ("dailymail", "17865"),
    ("independent", "1887"),
    ("guardian", "4149"),
    ("times", "5213"),
]


def snapshot(puzzles, out_path):
    from sonnet_pipeline.verify_explanation import ExplanationVerifier

    verifier = ExplanationVerifier()
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row

    entries = []
    for source, puzzle in puzzles:
        rows = conn.execute("""
            SELECT c.id, c.source, c.puzzle_number, c.clue_number, c.direction,
                   c.clue_text, c.answer, c.definition, c.wordplay_type, c.ai_explanation
            FROM clues c
            WHERE c.source = ? AND c.puzzle_number = ?
              AND c.ai_explanation IS NOT NULL AND c.ai_explanation != ''
            ORDER BY c.direction, CAST(c.clue_number AS INTEGER)
        """, (source, puzzle)).fetchall()

        for r in rows:
            try:
                v = verifier.verify(
                    clue_text=r["clue_text"],
                    answer=r["answer"],
                    wordplay_type=r["wordplay_type"] or "",
                    definition=r["definition"] or "",
                    ai_explanation=r["ai_explanation"],
                )
                entries.append({
                    "id": r["id"],
                    "source": r["source"],
                    "puzzle": r["puzzle_number"],
                    "label": f"{r['clue_number']}{r['direction'][0]}",
                    "answer": r["answer"],
                    "wordplay_type": r["wordplay_type"],
                    "definition": r["definition"],
                    "ai_explanation": r["ai_explanation"],
                    "score": v.get("score"),
                    "verdict": v.get("verdict"),
                    "checks": v.get("checks", []),
                })
            except Exception as e:
                entries.append({
                    "id": r["id"],
                    "source": r["source"],
                    "puzzle": r["puzzle_number"],
                    "label": f"{r['clue_number']}{r['direction'][0]}",
                    "answer": r["answer"],
                    "error": str(e),
                })

    conn.close()

    out_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"Snapshotted {len(entries)} clues to {out_path}")

    by_verdict = {}
    for e in entries:
        v = e.get("verdict", "ERROR")
        by_verdict[v] = by_verdict.get(v, 0) + 1
    print("Verdicts:", by_verdict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    snapshot(DEFAULT_PUZZLES, Path(args.out))


if __name__ == "__main__":
    main()
