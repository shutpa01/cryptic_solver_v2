"""Wrapper to run the existing explanation parser with exclusion of already-solved clues.

Uses all parsing logic from parse_human_explanations.py unchanged.
Adds --exclude flag to skip clue IDs already solved by WS1/WS2.

Usage:
    python scripts/run_explanation_parser.py --exclude "data/backfill_dd_hidden_results.jsonl,data/batch_v1_results_verified.jsonl"
    python scripts/run_explanation_parser.py --exclude "..." --limit 1000
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.parse_human_explanations import (
    load_clues, parse_explanation, extract_definition_from_expl,
    build_explanation_text, build_payload, norm_letters,
)
from signature_solver.db import RefDB

RESULTS_PATH = os.path.join(ROOT, "data", "parsed_explanations_v2.jsonl")


def load_exclude_ids(exclude_arg):
    """Load clue IDs to skip from comma-separated JSONL file paths."""
    skip_ids = set()
    if not exclude_arg:
        return skip_ids
    for path in exclude_arg.split(","):
        path = path.strip()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        skip_ids.add(json.loads(line)["clue_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
            print(f"  Loaded {len(skip_ids):,} IDs to skip from {path}")
    return skip_ids


def main():
    parser = argparse.ArgumentParser(description="Run explanation parser with exclusions")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--exclude", type=str, default=None,
                        help="Comma-separated JSONL files of already-solved clue IDs to skip")
    parser.add_argument("--output", type=str, default=RESULTS_PATH)
    args = parser.parse_args()

    print("=" * 60)
    print("EXPLANATION PARSER (with exclusions)")
    print("=" * 60)

    skip_ids = load_exclude_ids(args.exclude)

    print("\nLoading clues from DB...")
    clues = load_clues(args.limit, args.source)
    print(f"Loaded {len(clues):,} clues with human explanations.")

    print("Loading RefDB...")
    ref_db = RefDB()
    print("RefDB loaded.\n")

    stats = {}
    skipped = 0
    t0 = time.time()

    with open(args.output, "w", encoding="utf-8") as f:
        for i, row in enumerate(clues):
            if row["id"] in skip_ids:
                skipped += 1
                continue

            answer = row["answer"]
            ans_clean = norm_letters(answer).upper()
            if not ans_clean:
                continue

            result = parse_explanation(row["clue_text"], ans_clean, row["explanation"], ref_db)

            if result:
                wtype = result["wordplay_type"]
                pieces = result.get("pieces", [])
                stats[wtype] = stats.get(wtype, 0) + 1

                definition = extract_definition_from_expl(
                    row["explanation"], row["clue_text"], ans_clean, ref_db)

                expl_text = build_explanation_text(
                    wtype, pieces, definition, answer, row["explanation"])

                payload = build_payload(definition, wtype, pieces, expl_text, row)
                f.write(json.dumps({"clue_id": row["id"], "action": "high_solve",
                                    "payload": payload}, ensure_ascii=False) + "\n")

            if (i + 1) % 10000 == 0:
                elapsed = time.time() - t0
                solved = sum(stats.values())
                print(f"  {i+1}/{len(clues)} ({elapsed:.0f}s) — {solved:,} parsed, {skipped:,} skipped")

    elapsed = time.time() - t0
    solved = sum(stats.values())

    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Skipped (already solved): {skipped:,}")
    print(f"  Parsed: {solved:,}")
    for wt, n in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"    {wt}: {n:,}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
