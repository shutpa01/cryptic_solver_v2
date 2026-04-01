"""Batch mechanical solver — run hidden word and double definition solvers
on all unsolved clues. Writes results to JSONL, no DB locking.

These are zero-cost mechanical solvers:
- Hidden words: answer appears contiguously in the clue text
- Double definitions: both halves of the clue independently define the answer

Usage:
    python scripts/batch_mechanical.py                     # all unsolved
    python scripts/batch_mechanical.py --limit 1000        # test run
    python scripts/batch_mechanical.py --source guardian    # one source
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sonnet_pipeline.solver import try_hidden, try_double_definition, clean
from sonnet_pipeline.report import _highlight_hidden
from signature_solver.db import RefDB

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
RESULTS_PATH = os.path.join(ROOT, "data", "batch_mechanical_results.jsonl")


def load_clues(limit, source_filter):
    """Load unsolved clues from DB, then close connection."""
    conn = sqlite3.connect(CLUES_DB, timeout=30)
    conn.row_factory = sqlite3.Row

    where = "answer IS NOT NULL AND answer != '' AND clue_text IS NOT NULL AND clue_text != ''"
    where += " AND (has_solution IS NULL OR has_solution = 0)"
    params = []
    if source_filter:
        where += " AND source = ?"
        params.append(source_filter)

    if limit:
        rows = conn.execute("""
            SELECT id, source, puzzle_number, clue_number, clue_text, answer
            FROM clues WHERE %s
            ORDER BY publication_date DESC
            LIMIT ?
        """ % where, params + [limit]).fetchall()
    else:
        rows = conn.execute("""
            SELECT id, source, puzzle_number, clue_number, clue_text, answer
            FROM clues WHERE %s
            ORDER BY publication_date DESC
        """ % where, params).fetchall()

    clues = []
    for r in rows:
        clues.append({
            "id": r["id"],
            "source": r["source"],
            "puzzle_number": r["puzzle_number"],
            "clue_number": r["clue_number"],
            "clue_text": r["clue_text"],
            "answer": r["answer"],
        })

    conn.close()
    return clues


def run_batch(clues, ref_db, results_path):
    """Process clues with hidden and DD solvers, write to JSONL."""
    hidden_count = 0
    dd_count = 0
    skipped = 0
    t0 = time.time()

    with open(results_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(clues):
            clue = row["clue_text"]
            answer = row["answer"]
            answer_clean = clean(answer)
            if not answer_clean or len(answer_clean) < 2:
                skipped += 1
                continue

            # Hidden word check
            hidden_result = try_hidden(clue, answer_clean)
            if hidden_result:
                is_reversed = "reversed" in hidden_result.get("op", "")
                hiding_words = hidden_result.get("words", hidden_result.get("word", ""))

                if is_reversed:
                    highlighted = _highlight_hidden(hiding_words, answer_clean[::-1])
                    expl_text = 'hidden reversed in "%s"' % highlighted
                else:
                    highlighted = _highlight_hidden(hiding_words, answer_clean)
                    expl_text = 'hidden in "%s"' % highlighted

                payload = {
                    "definition": None,
                    "wordplay_type": hidden_result["op"],
                    "ai_explanation": expl_text,
                    "has_solution": 1,
                    "reviewed": 1,
                    "confidence": 1.0,
                    "components": {
                        "ai_pieces": [{
                            "clue_word": hiding_words,
                            "letters": answer_clean,
                            "mechanism": "hidden",
                        }],
                        "assembly": hidden_result,
                        "wordplay_type": hidden_result["op"],
                    },
                    "wordplay_types": [hidden_result["op"]],
                    "definition_start": None,
                    "definition_end": None,
                    "model_version": "mechanical_hidden",
                    "source": row["source"],
                    "puzzle_number": row["puzzle_number"],
                    "clue_number": row["clue_number"],
                }
                f.write(json.dumps({"clue_id": row["id"], "action": "hidden_solve", "payload": payload}) + "\n")
                hidden_count += 1
                continue

            # Double definition check
            dd_result = try_double_definition(clue, answer_clean, ref_db)
            if dd_result:
                left_def = dd_result["left_def"]
                right_def = dd_result["right_def"]
                expl_text = 'Double definition: "%s" and "%s" both mean %s' % (
                    left_def, right_def, answer)

                payload = {
                    "definition": left_def,
                    "wordplay_type": "double_definition",
                    "ai_explanation": expl_text,
                    "has_solution": 1,
                    "reviewed": 1,
                    "confidence": 1.0,
                    "components": {
                        "ai_pieces": [],
                        "assembly": dd_result,
                        "wordplay_type": "double_definition",
                    },
                    "wordplay_types": ["double_definition"],
                    "definition_start": None,
                    "definition_end": None,
                    "model_version": "mechanical_dd",
                    "source": row["source"],
                    "puzzle_number": row["puzzle_number"],
                    "clue_number": row["clue_number"],
                }
                f.write(json.dumps({"clue_id": row["id"], "action": "dd_solve", "payload": payload}) + "\n")
                dd_count += 1
                continue

            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(clues)} ({elapsed:.0f}s) - {hidden_count} hidden, {dd_count} DD")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Hidden: {hidden_count:,}")
    print(f"  DD: {dd_count:,}")
    print(f"  Skipped: {skipped:,}")
    print(f"  Total solved: {hidden_count + dd_count:,}")
    print(f"  Results written to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch mechanical solver (hidden + DD)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clues")
    parser.add_argument("--source", type=str, default=None, help="Filter by source")
    args = parser.parse_args()

    print("=" * 60)
    print("BATCH MECHANICAL SOLVER (hidden + DD)")
    if args.limit:
        print(f"  Limit: {args.limit}")
    if args.source:
        print(f"  Source: {args.source}")
    print("=" * 60)

    print("\nLoading clues from DB...")
    clues = load_clues(args.limit, args.source)
    print(f"Loaded {len(clues):,} clues. DB connection closed.")

    print("Loading RefDB into memory (for DD solver)...")
    ref_db = RefDB()
    print("RefDB loaded. All DB connections closed.\n")

    run_batch(clues, ref_db, RESULTS_PATH)


if __name__ == "__main__":
    main()
