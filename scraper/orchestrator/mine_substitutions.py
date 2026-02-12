#!/usr/bin/env python3
"""
Mine substitution frequency from clues database.

Two-pass approach:
1. Count ALL instances first
2. Look at distribution (percentiles)
3. Assign confidence based on relative ranking

This avoids the problem where 50 and 3000 both = "very_high"
"""

import sqlite3
from pathlib import Path
from collections import defaultdict

# Configuration
CLUES_DB = Path(r"C:\Users\shute\PycharmProjects\AI_Solver\data\clues_master.db")
PATTERNS_DB = Path(r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db")


def mine_substitutions():
    """Count instances, then rank and assign confidence."""

    # Connect to both databases
    clues_conn = sqlite3.connect(CLUES_DB)
    patterns_conn = sqlite3.connect(PATTERNS_DB)

    clues_cur = clues_conn.cursor()
    patterns_cur = patterns_conn.cursor()

    # Get all substitutions
    patterns_cur.execute("""
        SELECT id, indicator, substitution, confidence, frequency
        FROM wordplay
    """)
    substitutions = patterns_cur.fetchall()

    print(f"PASS 1: Counting {len(substitutions)} substitutions against 1.2M clues...")
    print()

    # PASS 1: Count all instances
    results = []
    measurable_counts = []

    for sub_id, word, letters, old_confidence, old_frequency in substitutions:
        # Single letters can't be measured reliably
        if len(letters) == 1:
            results.append({
                'id': sub_id,
                'word': word,
                'substitution': letters,
                'frequency': old_frequency or 0,
                'old_confidence': old_confidence,
                'new_confidence': old_confidence or 'high',
                'changed': False,
                'unmeasurable': True
            })
            continue

        # Count clues where word appears AND answer contains letters
        clues_cur.execute("""
            SELECT COUNT(*) FROM clues 
            WHERE (LOWER(clue_text) LIKE ? 
                OR LOWER(clue_text) LIKE ?
                OR LOWER(clue_text) LIKE ?
                OR LOWER(clue_text) = ?)
            AND UPPER(answer) LIKE ?
        """, (
            f'% {word.lower()} %',
            f'{word.lower()} %',
            f'% {word.lower()}',
            word.lower(),
            f'%{letters.upper()}%'
        ))

        count = clues_cur.fetchone()[0]

        results.append({
            'id': sub_id,
            'word': word,
            'substitution': letters,
            'frequency': count,
            'old_confidence': old_confidence,
            'new_confidence': None,  # Set in pass 2
            'changed': False,
            'unmeasurable': False
        })

        if count > 0:
            measurable_counts.append(count)

        # Progress
        if len(results) % 100 == 0:
            print(f"  Counted {len(results)}/{len(substitutions)}...")

    # PASS 2: Calculate percentiles and assign confidence
    print()
    print("PASS 2: Analyzing distribution and assigning confidence...")

    if measurable_counts:
        measurable_counts.sort()
        n = len(measurable_counts)

        # Percentile thresholds
        p90 = measurable_counts[int(n * 0.90)]
        p75 = measurable_counts[int(n * 0.75)]
        p50 = measurable_counts[int(n * 0.50)]
        p25 = measurable_counts[int(n * 0.25)]

        print()
        print("=" * 60)
        print("DISTRIBUTION OF COUNTS")
        print("=" * 60)
        print(f"Min:                    {measurable_counts[0]}")
        print(f"25th percentile:        {p25}")
        print(f"50th percentile (median): {p50}")
        print(f"75th percentile:        {p75}")
        print(f"90th percentile:        {p90}")
        print(f"Max:                    {measurable_counts[-1]}")
        print()
        print("Confidence thresholds:")
        print(f"  very_high: >= {p90} (top 10%)")
        print(f"  high:      >= {p75} (top 25%)")
        print(f"  medium:    >= {p50} (top 50%)")
        print(f"  low:       >= {p25} (top 75%)")
        print(f"  very_low:  < {p25}")

        # Assign confidence based on percentile
        for r in results:
            if r['unmeasurable']:
                continue

            count = r['frequency']
            if count >= p90:
                r['new_confidence'] = 'very_high'
            elif count >= p75:
                r['new_confidence'] = 'high'
            elif count >= p50:
                r['new_confidence'] = 'medium'
            elif count >= p25:
                r['new_confidence'] = 'low'
            else:
                r['new_confidence'] = 'very_low'

            r['changed'] = r['old_confidence'] != r['new_confidence']

    # Update database
    print()
    print("Updating wordplay table...")

    updated = 0
    unmeasurable_count = 0
    for r in results:
        if r['unmeasurable']:
            unmeasurable_count += 1
            continue
        patterns_cur.execute("""
            UPDATE wordplay 
            SET frequency = ?, confidence = ?
            WHERE id = ?
        """, (r['frequency'], r['new_confidence'], r['id']))
        if r['changed']:
            updated += 1

    patterns_conn.commit()

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    by_confidence = defaultdict(int)
    for r in results:
        if not r['unmeasurable']:
            by_confidence[r['new_confidence']] += 1

    print(f"Total substitutions: {len(results)}")
    print(f"Unmeasurable (single letters): {unmeasurable_count}")
    print(f"Measured: {len(results) - unmeasurable_count}")
    print()
    for conf in ['very_high', 'high', 'medium', 'low', 'very_low']:
        print(f"  {conf}: {by_confidence[conf]}")

    print(f"\nConfidence changed: {updated}")

    # Top 20
    print()
    print("TOP 20 SUBSTITUTIONS:")
    measurable = [r for r in results if not r['unmeasurable']]
    top = sorted(measurable, key=lambda x: -x['frequency'])[:20]
    for r in top:
        print(
            f"  {r['word']} → {r['substitution']}: {r['frequency']} ({r['new_confidence']})")

    # Bottom 20 (with any matches)
    print()
    print("BOTTOM 20 (with matches):")
    with_matches = [r for r in measurable if r['frequency'] > 0]
    bottom = sorted(with_matches, key=lambda x: x['frequency'])[:20]
    for r in bottom:
        print(
            f"  {r['word']} → {r['substitution']}: {r['frequency']} ({r['new_confidence']})")

    clues_conn.close()
    patterns_conn.close()

    print()
    print("Done!")


if __name__ == "__main__":
    mine_substitutions()