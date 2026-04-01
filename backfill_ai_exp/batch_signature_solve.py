"""Batch signature solver — run S across all clues for free enrichment.

Iterative: run multiple times. Each run solves more clues because the
DB gets richer from the previous run's discoveries.

Usage:
    python scripts/batch_signature_solve.py              # Run once
    python scripts/batch_signature_solve.py --rounds 3   # Run 3 iterations
    python scripts/batch_signature_solve.py --source telegraph  # One source only
    python scripts/batch_signature_solve.py --dry-run    # Don't write to DB
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from signature_solver.solver import solve_clue as sig_solve_clue
from signature_solver.db import RefDB
from sonnet_pipeline.solver import try_hidden, try_spoonerism, try_double_definition, clean

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
CRYPTIC_DB = os.path.join(ROOT, "data", "cryptic_new.db")


def extract_unconfirmed(reasons):
    """Parse confidence reasons to extract unconfirmed synonym/abbreviation pairs."""
    pairs = []
    for reason_text, penalty in reasons:
        # SYN 'prize'=CUP unconfirmed (real word)
        m = re.match(r"SYN '(.+?)'=(\w+) unconfirmed", reason_text)
        if m:
            pairs.append(("synonym", m.group(1).lower(), m.group(2).upper()))
            continue
        # ABR 'large'=L unconfirmed
        m = re.match(r"ABR '(.+?)'=(\w+) unconfirmed", reason_text)
        if m:
            pairs.append(("abbreviation", m.group(1).lower(), m.group(2).upper()))
            continue
    return pairs


def add_enrichment(cryptic_conn, enrichment_pairs, dry_run=False):
    """Add verified pairs to the reference DB. Returns count added."""
    added = 0
    c = cryptic_conn.cursor()

    for etype, word, letters in enrichment_pairs:
        if etype == "synonym":
            existing = c.execute(
                "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(synonym)=?",
                (word, letters)
            ).fetchone()
            if not existing:
                if not dry_run:
                    c.execute(
                        "INSERT INTO synonyms_pairs (word, synonym, source) VALUES (?, ?, ?)",
                        (word, letters, "sig_batch")
                    )
                added += 1

        elif etype == "abbreviation":
            existing = c.execute(
                "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? AND UPPER(substitution)=?",
                (word, letters.upper())
            ).fetchone()
            if not existing:
                if not dry_run:
                    c.execute(
                        "INSERT INTO wordplay (indicator, substitution) VALUES (?, ?)",
                        (word, letters)
                    )
                added += 1

    if not dry_run:
        cryptic_conn.commit()
    return added


def run_batch(ref_db, source_filter=None, dry_run=False, write_explanations=True):
    """Run signature solver on all unsolved clues. Returns (solved, enrichment_pairs)."""

    conn = sqlite3.connect(CLUES_DB, timeout=30)
    conn.row_factory = sqlite3.Row

    # Get all clues with answers that don't have explanations yet
    where = "answer IS NOT NULL AND answer != '' AND (has_solution IS NULL OR has_solution = 0)"
    params = []
    if source_filter:
        where += " AND source = ?"
        params.append(source_filter)

    rows = conn.execute("""
        SELECT id, source, puzzle_number, clue_number, direction,
               clue_text, answer, enumeration
        FROM clues
        WHERE %s
        ORDER BY source, CAST(puzzle_number AS INTEGER),
                 CASE direction WHEN 'across' THEN 0 ELSE 1 END,
                 CAST(clue_number AS INTEGER)
    """ % where, params).fetchall()

    print(f"Clues to process: {len(rows)}")

    solved = 0
    hidden_solved = 0
    all_enrichment = []
    t0 = time.time()

    for i, row in enumerate(rows):
        cid = row["id"]
        clue = row["clue_text"] or ""
        answer = row["answer"] or ""
        answer_clean = clean(answer)

        if not answer_clean or len(answer_clean) < 2:
            continue

        # Phase 0: Mechanical hidden word check
        hidden_result = try_hidden(clue, answer_clean)
        if hidden_result:
            hidden_solved += 1
            solved += 1

            if write_explanations and not dry_run:
                hiding_words = hidden_result.get("words", "")
                is_reversed = "reversed" in hidden_result.get("op", "")
                expl = 'hidden reversed in "%s"' % hiding_words if is_reversed else 'hidden in "%s"' % hiding_words

                pieces_data = [{"clue_word": hiding_words, "letters": answer_clean, "mechanism": "hidden"}]
                components = json.dumps({
                    "ai_pieces": pieces_data,
                    "assembly": hidden_result,
                    "wordplay_type": hidden_result["op"],
                })
                conn.execute("""
                    INSERT OR REPLACE INTO structured_explanations
                    (clue_id, components, wordplay_types, definition_text, confidence, model_version, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (cid, components, json.dumps([hidden_result["op"]]),
                      None, 1.0, "mechanical_hidden", row["source"]))
                conn.execute("""
                    UPDATE clues SET
                        wordplay_type = COALESCE(NULLIF(wordplay_type, ''), ?),
                        ai_explanation = COALESCE(NULLIF(ai_explanation, ''), ?),
                        has_solution = 1
                    WHERE id = ?
                """, (hidden_result["op"], expl, cid))
                conn.commit()

            continue  # Don't also run S on hidden words

        # Phase 0b: Spoonerism check
        if "spooner" in clue.lower() and len(answer_clean) >= 4:
            spoon_result = try_spoonerism(answer_clean, ref_db.is_real_word)
            if spoon_result:
                solved += 1
                w1 = spoon_result["word1"]
                w2 = spoon_result["word2"]
                sw1 = spoon_result["swapped1"]
                sw2 = spoon_result["swapped2"]
                expl = 'spoonerism of %s %s (swap initials: %s %s)' % (sw1, sw2, w1, w2)

                if write_explanations and not dry_run:
                    components = json.dumps({
                        "ai_pieces": [{"clue_word": "%s %s" % (sw1, sw2), "letters": answer_clean, "mechanism": "spoonerism"}],
                        "assembly": spoon_result,
                        "wordplay_type": "spoonerism",
                    })
                    conn.execute("""
                        INSERT OR REPLACE INTO structured_explanations
                        (clue_id, components, wordplay_types, definition_text, confidence, model_version, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (cid, components, json.dumps(["spoonerism"]),
                          None, 1.0, "mechanical_spoonerism", row["source"]))
                    conn.execute("""
                        UPDATE clues SET
                            wordplay_type = COALESCE(NULLIF(wordplay_type, ''), 'spoonerism'),
                            ai_explanation = COALESCE(NULLIF(ai_explanation, ''), ?),
                            has_solution = 1
                        WHERE id = ?
                    """, (expl, cid))
                    conn.commit()

                continue

        # Phase 0c: Double definition check
        dd_result = try_double_definition(clue, answer_clean, ref_db)
        if dd_result:
            solved += 1

            if write_explanations and not dry_run:
                left_def = dd_result["left_def"]
                right_def = dd_result["right_def"]
                expl = 'Double definition: "%s" and "%s" both mean %s' % (left_def, right_def, answer)

                components = json.dumps({
                    "ai_pieces": [],
                    "assembly": dd_result,
                    "wordplay_type": "double_definition",
                })
                conn.execute("""
                    INSERT OR REPLACE INTO structured_explanations
                    (clue_id, components, wordplay_types, definition_text, confidence, model_version, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (cid, components, json.dumps(["double_definition"]),
                      left_def, 1.0, "mechanical_dd", row["source"]))
                conn.execute("""
                    UPDATE clues SET
                        wordplay_type = COALESCE(NULLIF(wordplay_type, ''), 'double_definition'),
                        definition = COALESCE(NULLIF(definition, ''), ?),
                        ai_explanation = COALESCE(NULLIF(ai_explanation, ''), ?),
                        has_solution = 1
                    WHERE id = ?
                """, (left_def, expl, cid))
                conn.commit()

            continue

        # Phase 1: Signature solver
        # Skip cross-reference clues
        if re.search(r'\b\d+\s*(?:across|down|ac|dn)\b', clue, re.IGNORECASE):
            continue

        try:
            sr = sig_solve_clue(clue, answer_clean, ref_db)
        except Exception:
            continue

        if sr.solved:
            # Extract unconfirmed pairs from ALL solved clues (any confidence)
            pairs = extract_unconfirmed(sr.confidence_reasons)
            all_enrichment.extend(pairs)

            if sr.confidence >= 60:
                solved += 1

                # Store explanation if high confidence
                if write_explanations and not dry_run and sr.high_confidence:
                    from sonnet_pipeline.sig_adapter import store_signature_result
                    store_signature_result(conn, cid, sr, clue, answer)
                    conn.commit()

        # Progress
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  ... {i+1}/{len(rows)} ({rate:.0f}/sec) — {solved} solved, {len(all_enrichment)} enrichment pairs")

    conn.close()
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s — {solved} solved ({hidden_solved} hidden), {len(all_enrichment)} enrichment pairs")

    return solved, all_enrichment


def main():
    parser = argparse.ArgumentParser(description="Batch signature solver for free enrichment")
    parser.add_argument("--rounds", type=int, default=1, help="Number of iterative rounds")
    parser.add_argument("--source", type=str, default=None, help="Filter by source (e.g. telegraph)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--no-explanations", action="store_true", help="Skip writing explanations, only collect enrichment")
    args = parser.parse_args()

    print("=" * 60)
    print("BATCH SIGNATURE SOLVER")
    print("=" * 60)
    print(f"Rounds: {args.rounds}")
    print(f"Source filter: {args.source or 'all'}")
    print(f"Dry run: {args.dry_run}")
    print()

    total_solved = 0
    total_enrichment = 0

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}")
        print(f"{'='*60}")

        # Load fresh RefDB each round (picks up enrichment from previous round)
        print("Loading reference DB...")
        ref_db = RefDB()

        solved, enrichment_pairs = run_batch(
            ref_db,
            source_filter=args.source,
            dry_run=args.dry_run,
            write_explanations=not args.no_explanations,
        )

        total_solved += solved

        # Deduplicate enrichment
        unique_pairs = list(set(enrichment_pairs))
        print(f"\nUnique enrichment pairs this round: {len(unique_pairs)}")

        if unique_pairs and not args.dry_run:
            cryptic_conn = sqlite3.connect(CRYPTIC_DB, timeout=30)
            added = add_enrichment(cryptic_conn, unique_pairs, dry_run=args.dry_run)
            cryptic_conn.close()
            total_enrichment += added
            print(f"Added to DB: {added} new pairs")

            # Show some examples
            for etype, word, letters in unique_pairs[:10]:
                print(f"  {etype:12} {word:20} -> {letters}")
            if len(unique_pairs) > 10:
                print(f"  ... and {len(unique_pairs) - 10} more")
        elif unique_pairs:
            print("(dry run — not adding to DB)")
            for etype, word, letters in unique_pairs[:10]:
                print(f"  {etype:12} {word:20} -> {letters}")

        if solved == 0 and len(unique_pairs) == 0:
            print("\nNo progress this round — stopping early")
            break

    print(f"\n{'='*60}")
    print(f"DONE: {total_solved} clues solved, {total_enrichment} DB entries added across {round_num} rounds")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
