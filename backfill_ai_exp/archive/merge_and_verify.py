"""Merge results from multiple batch solvers, verify with scorer, keep best per clue.

Reads JSONL files from batch_mechanical, batch_v1_solver, and parse_human_explanations.
Runs the verifier on each, and for each clue_id keeps only the highest-scoring result.

Usage:
    python scripts/merge_and_verify.py                    # merge all three
    python scripts/merge_and_verify.py --dry-run          # show stats without writing
"""

import argparse
import json
import os
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sonnet_pipeline.verify_explanation import ExplanationVerifier

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
OUTPUT_PATH = os.path.join(ROOT, "data", "batch_merged_results.jsonl")

INPUT_FILES = [
    os.path.join(ROOT, "data", "batch_mechanical_results.jsonl"),
    os.path.join(ROOT, "data", "batch_v1_results.jsonl"),
    os.path.join(ROOT, "data", "parsed_explanations_results.jsonl"),
]


def load_clue_data():
    """Load clue_text, answer, and current has_solution for all clues."""
    conn = sqlite3.connect(CLUES_DB, timeout=30)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, clue_text, answer, has_solution FROM clues
        WHERE answer IS NOT NULL AND answer != ''
    """).fetchall()
    conn.close()
    return {r["id"]: {"clue_text": r["clue_text"], "answer": r["answer"],
                       "has_solution": r["has_solution"]} for r in rows}


def score_result(verifier, clue_text, answer, payload):
    """Run the verifier on a result and return the score."""
    try:
        result = verifier.verify(
            clue_text=clue_text,
            answer=answer,
            definition=payload.get("definition") or "",
            wordplay_type=payload.get("wordplay_type") or "",
            ai_explanation=payload.get("ai_explanation") or "",
        )
        return result.get("score", 0), result.get("verdict", "FAIL")
    except Exception:
        return 0, "FAIL"


def main():
    parser = argparse.ArgumentParser(description="Merge and verify batch results")
    parser.add_argument("--dry-run", action="store_true", help="Show stats only")
    parser.add_argument("--min-score", type=int, default=0,
                        help="Minimum score to include (default: 0 = include all)")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("MERGE AND VERIFY BATCH RESULTS", flush=True)
    print("=" * 60, flush=True)

    # Load all results from all input files
    all_results = {}
    for filepath in INPUT_FILES:
        if not os.path.exists(filepath):
            print(f"  Skipping {os.path.basename(filepath)} (not found)", flush=True)
            continue

        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                action = rec.get("action", "")
                if action in ("attempted", "skipped_crossref", "error"):
                    continue
                cid = rec["clue_id"]
                if cid not in all_results:
                    all_results[cid] = []
                all_results[cid].append((os.path.basename(filepath), rec))
                count += 1

        print(f"  Loaded {count:,} results from {os.path.basename(filepath)}", flush=True)

    print(f"\n  Total clue IDs with results: {len(all_results):,}", flush=True)
    duplicates = sum(1 for v in all_results.values() if len(v) > 1)
    print(f"  Clue IDs with multiple results: {duplicates:,}", flush=True)

    # Load clue data for verification
    print("\n  Loading clue data from DB...", flush=True)
    clue_data = load_clue_data()
    print(f"  Loaded {len(clue_data):,} clues", flush=True)

    # Score each result and keep the best per clue_id
    print("\n  Scoring results with verifier...", flush=True)
    verifier = ExplanationVerifier()

    best_results = {}
    score_dist = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "FAIL": 0}
    t0 = time.time()
    processed = 0

    for cid, entries in all_results.items():
        cd = clue_data.get(cid)
        if not cd:
            continue
        if cd["has_solution"] == 1:
            continue

        best_score = -1
        best_entry = None

        for source_file, rec in entries:
            payload = rec.get("payload", {})
            score, verdict = score_result(
                verifier, cd["clue_text"], cd["answer"], payload)

            if score > best_score:
                best_score = score
                best_entry = (score, verdict, source_file, rec)

        if best_entry:
            score, verdict, source_file, rec = best_entry
            if score >= args.min_score:
                best_results[cid] = best_entry
                score_dist[verdict] = score_dist.get(verdict, 0) + 1

        processed += 1
        if processed % 5000 == 0:
            elapsed = time.time() - t0
            print(f"    {processed:,}/{len(all_results):,} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Scored {processed:,} clues in {elapsed:.0f}s", flush=True)
    print(f"\n  Score distribution:", flush=True)
    for verdict in ["HIGH", "MEDIUM", "LOW", "FAIL"]:
        count = score_dist.get(verdict, 0)
        if count:
            print(f"    {verdict}: {count:,}", flush=True)
    print(f"    Total kept: {len(best_results):,}", flush=True)

    # Show source distribution
    source_counts = {}
    for score, verdict, source_file, rec in best_results.values():
        source_counts[source_file] = source_counts.get(source_file, 0) + 1
    print(f"\n  Best results by source:", flush=True)
    for sf, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {sf}: {cnt:,}", flush=True)

    if args.dry_run:
        print("\n  --dry-run: not writing output", flush=True)
        return

    # Write merged results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for cid, (score, verdict, source_file, rec) in best_results.items():
            rec["payload"]["confidence"] = score / 100.0
            f.write(json.dumps(rec) + "\n")

    print(f"\n  Wrote {len(best_results):,} results to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
