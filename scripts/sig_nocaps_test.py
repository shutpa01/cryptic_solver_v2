"""Test signature solver with caps removed on a specific cohort.

DRY RUN ONLY. No DB writes. Temporarily patches the cap constants
in base_matcher to very high values, runs the solver, reports results.

Usage:
    python scripts/sig_nocaps_test.py --load-cohort charade_fails
"""

import argparse
import json
import sqlite3
import sys
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
COHORT_DIR = os.path.join(ROOT, "data", "cohorts")


def main():
    parser = argparse.ArgumentParser(description="Test solver with caps removed (dry run)")
    parser.add_argument("--load-cohort", type=str, required=True)
    args = parser.parse_args()

    # Patch caps BEFORE importing solver
    import signature_solver.base_matcher as bm
    import signature_solver.matcher as m

    old_placements = bm.MAX_PLACEMENTS_PER_ENTRY
    old_fodder = bm.MAX_FODDER_COMBOS
    bm.MAX_PLACEMENTS_PER_ENTRY = 10000
    bm.MAX_FODDER_COMBOS = 10000

    # Also patch _product_capped cap in _verify_combo
    old_product_cap = 200  # hardcoded in _verify_combo call
    # We need to patch _product_capped itself to ignore the cap
    original_product_capped = m._product_capped
    def _product_uncapped(lists, cap=999999):
        return original_product_capped(lists, cap=999999)
    m._product_capped = _product_uncapped

    print(f"Caps patched: placements {old_placements}->{bm.MAX_PLACEMENTS_PER_ENTRY}, "
          f"fodder {old_fodder}->{bm.MAX_FODDER_COMBOS}, product uncapped")

    print("Loading RefDB...")
    from signature_solver.db import RefDB
    ref_db = RefDB()

    from signature_solver.solver import (
        extract_definition_candidates, solve, _normalize_clue
    )

    # Load cohort
    with open(os.path.join(COHORT_DIR, f"{args.load_cohort}.json")) as f:
        clue_ids = json.load(f)

    conn = sqlite3.connect(CLUES_DB, timeout=30)
    placeholders = ",".join("?" for _ in clue_ids)
    rows = conn.execute(f"""
        SELECT id, source, puzzle_number, clue_number, clue_text, answer
        FROM clues WHERE id IN ({placeholders})
        ORDER BY source, publication_date DESC, CAST(clue_number AS INTEGER)
    """, clue_ids).fetchall()
    conn.close()

    total = len(rows)
    print(f"Processing {total} clues (all previously failed)...\n")

    solved_high = 0
    solved_medium = 0
    solved_low = 0
    still_failed = 0

    solved_examples = []
    t0 = time.time()

    for i, (cid, source, pnum, cnum, clue_text, answer) in enumerate(rows):
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{total} ({time.time()-t0:.1f}s)")

        answer_clean = answer.upper().replace(" ", "").replace("-", "")
        clue_words = _normalize_clue(clue_text).strip().split()

        candidates = extract_definition_candidates(clue_words, answer_clean, ref_db)
        if not candidates:
            still_failed += 1
            continue

        best_confidence = -1
        solved = False
        for def_phrase, wp_words in candidates:
            dp = 'start' if clue_words[:len(def_phrase.split())] == clue_words[:len(clue_words) - len(wp_words)] else 'end'
            sr = solve(wp_words, answer_clean, ref_db, min_confidence=0, def_pos=dp)
            if sr.solved and sr.confidence > best_confidence:
                best_confidence = sr.confidence
                solved = True
            if sr.solved and sr.confidence >= 80:
                break

        if solved:
            if best_confidence >= 80:
                solved_high += 1
                if len(solved_examples) < 10:
                    solved_examples.append(
                        f"  HIGH {best_confidence}: {source} #{pnum} {cnum}: {clue_text} = {answer}")
            elif best_confidence >= 50:
                solved_medium += 1
            else:
                solved_low += 1
        else:
            still_failed += 1

    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"CAPS REMOVED TEST -- {total} previously-failed clues in {elapsed:.1f}s")
    print(f"{'=' * 70}")
    print(f"\n  Solved HIGH:    {solved_high}")
    print(f"  Solved MEDIUM:  {solved_medium}")
    print(f"  Solved LOW:     {solved_low}")
    print(f"  Still failed:   {still_failed}")
    print(f"  Total:          {total}")
    print(f"\n  Recovery rate:  {solved_high}/{total} = {100*solved_high//total if total else 0}%")

    if solved_examples:
        print(f"\n--- Newly solved (HIGH) ---")
        for ex in solved_examples:
            print(ex)

    print()


if __name__ == "__main__":
    main()
