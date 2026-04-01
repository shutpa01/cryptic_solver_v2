"""Dry-run test of the new pipeline phases on a specific puzzle.

Runs the enhanced Phase 0 (V1 DD + hidden) and Phase 0.5 (V1 mechanical)
on a puzzle and reports what would be solved mechanically vs sent to Sonnet.

Does NOT write to DB. Compares against existing results.

Usage:
    python scripts/test_new_pipeline.py telegraph 31201
    python scripts/test_new_pipeline.py times 29504
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from signature_solver.db import RefDB, _normalize_key
from backfill_ai_exp.backfill_dd_hidden import (
    build_graph, generate_dd_hypotheses, try_hidden,
    generate_definition_windows, norm_letters, strip_enumeration,
)
from backfill_ai_exp.batch_v1_solver import (
    find_definition, try_anagram, try_charade, try_reversal,
    try_acrostic, try_homophone, build_explanation_text,
)
from sonnet_pipeline.verify_pieces import PieceVerifier

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")


def clean(s):
    return re.sub(r"[^A-Z]", "", s.upper())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("puzzle")
    args = parser.parse_args()

    print(f"Loading RefDB...")
    ref_db = RefDB()
    graph = build_graph(ref_db)
    pv = PieceVerifier(ref_db=ref_db)
    print(f"  Graph: {len(graph):,} keys")

    conn = sqlite3.connect(CLUES_DB, timeout=30)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT c.id, c.clue_number, c.direction, c.clue_text, c.answer,
               c.enumeration, c.definition, c.wordplay_type, c.ai_explanation,
               se.confidence, se.model_version, se.components
        FROM clues c
        LEFT JOIN structured_explanations se ON se.clue_id = c.id
        WHERE c.source = ? AND c.puzzle_number = ?
        ORDER BY c.direction DESC, CAST(c.clue_number AS INTEGER)
    """, (args.source, args.puzzle)).fetchall()

    print(f"\n{'=' * 80}")
    print(f"NEW PIPELINE DRY RUN: {args.source} {args.puzzle} ({len(rows)} clues)")
    print(f"{'=' * 80}\n")

    mechanical_solved = {}  # cid -> {phase, type, explanation, ...}
    would_need_sonnet = []

    for r in rows:
        cid = r["id"]
        clue = r["clue_text"]
        answer = r["answer"]
        if not answer:
            continue
        answer_clean = clean(answer)
        if not answer_clean or len(answer_clean) < 2:
            continue

        total_len = len(norm_letters(answer))
        solved = False

        # --- Phase 0a: Hidden (definition-confirmed) ---
        hidden_result = try_hidden(clue, answer_clean, graph, total_len)
        if hidden_result:
            mechanical_solved[cid] = {
                "phase": "0a-Hidden",
                "answer": answer,
                "clue_number": r["clue_number"],
                "direction": r["direction"],
                "definition": hidden_result.get("definition"),
                "wordplay_type": "hidden" if hidden_result["direction"] == "forward" else "hidden_reversed",
                "explanation": f'Hidden in "{hidden_result["words"]}"',
            }
            solved = True

        # --- Phase 0c: DD (V1 coverage-checked) ---
        if not solved:
            dd_result = generate_dd_hypotheses(clue, graph, total_len=total_len, answer=answer_clean)
            if dd_result:
                mechanical_solved[cid] = {
                    "phase": "0c-DD",
                    "answer": answer,
                    "clue_number": r["clue_number"],
                    "direction": r["direction"],
                    "definition": "Double definition",
                    "wordplay_type": "double_definition",
                    "explanation": "Double definition",
                }
                solved = True

        # --- Phase 0.5: V1 Mechanical solvers (with definition required) ---
        if not solved:
            definition, remaining = find_definition(clue, answer_clean, ref_db)
            if definition:
                # Only run mechanical solvers if we found a definition
                if remaining is None:
                    text = strip_enumeration(clue)
                    remaining = text.split()

                # Try anagram
                ana_result = try_anagram(clue, answer_clean, ref_db,
                                         definition_words=definition.split() if definition else None)
                if ana_result:
                    pieces = [{"clue_word": w, "letters": norm_letters(w).upper(),
                               "mechanism": "anagram_fodder"} for w in ana_result["fodder_words"]]
                    mechanical_solved[cid] = {
                        "phase": "0.5-Anagram",
                        "answer": answer,
                        "clue_number": r["clue_number"],
                        "direction": r["direction"],
                        "definition": definition,
                        "wordplay_type": "anagram",
                        "explanation": build_explanation_text("anagram", pieces, definition, answer),
                    }
                    solved = True

                # Try charade
                if not solved:
                    cha_result = try_charade(remaining, answer_clean, ref_db)
                    if cha_result:
                        mechanical_solved[cid] = {
                            "phase": "0.5-Charade",
                            "answer": answer,
                            "clue_number": r["clue_number"],
                            "direction": r["direction"],
                            "definition": definition,
                            "wordplay_type": "charade",
                            "explanation": build_explanation_text("charade", cha_result["pieces"], definition, answer),
                        }
                        solved = True

                # Try reversal
                if not solved:
                    rev_result = try_reversal(remaining, answer_clean, ref_db)
                    if rev_result:
                        mechanical_solved[cid] = {
                            "phase": "0.5-Reversal",
                            "answer": answer,
                            "clue_number": r["clue_number"],
                            "direction": r["direction"],
                            "definition": definition,
                            "wordplay_type": "reversal",
                            "explanation": build_explanation_text("reversal", rev_result["pieces"], definition, answer),
                        }
                        solved = True

                # Try acrostic
                if not solved:
                    acr_result = try_acrostic(remaining, answer_clean, ref_db)
                    if acr_result:
                        mechanical_solved[cid] = {
                            "phase": "0.5-Acrostic",
                            "answer": answer,
                            "clue_number": r["clue_number"],
                            "direction": r["direction"],
                            "definition": definition,
                            "wordplay_type": "acrostic",
                            "explanation": build_explanation_text("acrostic", acr_result["pieces"], definition, answer),
                        }
                        solved = True

                # Try homophone
                if not solved:
                    hom_result = try_homophone(remaining, answer_clean, ref_db)
                    if hom_result:
                        mechanical_solved[cid] = {
                            "phase": "0.5-Homophone",
                            "answer": answer,
                            "clue_number": r["clue_number"],
                            "direction": r["direction"],
                            "definition": definition,
                            "wordplay_type": "homophone",
                            "explanation": build_explanation_text("homophone", hom_result["pieces"], definition, answer),
                        }
                        solved = True

        if not solved:
            would_need_sonnet.append(r)

    # --- Report ---
    print(f"--- MECHANICAL SOLVES (would NOT need Sonnet) ---\n")
    phase_counts = Counter()
    for cid, info in mechanical_solved.items():
        phase_counts[info["phase"]] += 1
        print(f'  [{info["phase"]:15s}] {info["clue_number"]:>3s}{(info["direction"] or "")[0:1]} '
              f'{info["answer"]:15s} def={info["definition"] or "-":<20s} {info["explanation"][:60]}')

    print(f"\n--- WOULD STILL NEED SONNET ({len(would_need_sonnet)}) ---\n")
    for r in would_need_sonnet:
        # Show what the old pipeline did
        confidence = r["confidence"]
        old_score = f"{confidence * 100:.0f}" if confidence else "-"
        mv = r["model_version"] or "-"
        print(f'  {r["clue_number"]:>3s}{(r["direction"] or "")[0:1]} {r["answer"]:15s} '
              f'old=[{old_score:>3s} {mv[:5]:5s}] {r["clue_text"][:50]}')

    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"  Total clues:      {len(rows)}")
    print(f"  Mechanical:       {len(mechanical_solved)} ({100*len(mechanical_solved)/len(rows):.0f}%)")
    for phase, count in sorted(phase_counts.items()):
        print(f"    {phase}: {count}")
    print(f"  Need Sonnet:      {len(would_need_sonnet)} ({100*len(would_need_sonnet)/len(rows):.0f}%)")
    print(f"  Sonnet savings:   {len(mechanical_solved)} fewer API calls")
    print(f"{'=' * 80}")

    conn.close()


if __name__ == "__main__":
    main()
