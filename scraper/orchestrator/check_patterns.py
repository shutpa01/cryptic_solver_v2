#!/usr/bin/env python3
"""Check if scraped puzzles match known grid patterns.

Run this on your machine where the database is located:
    python check_patterns.py
"""

import json
import sqlite3
import os
from collections import defaultdict

DB_PATH = os.getenv('DB_PATH',
                    r"C:\Users\shute\PycharmProjects\AI_Solver\data\clues_master.db")
PATTERNS_FILE = "common_grid_patterns.json"


def load_patterns():
    """Load the grid patterns from JSON."""
    with open(PATTERNS_FILE, 'r') as f:
        patterns = json.load(f)
    return patterns


def get_puzzle_clue_pattern(cursor, table, puzzle_number):
    """Extract clue pattern from a puzzle."""
    cursor.execute(f"""
        SELECT clue_number, direction 
        FROM {table} 
        WHERE puzzle_number = ?
        ORDER BY direction, CAST(clue_number AS INTEGER)
    """, (str(puzzle_number),))

    rows = cursor.fetchall()

    across = []
    down = []

    for clue_num, direction in rows:
        clue_id = f"{clue_num}{direction[0].lower()}"
        if direction.lower() == 'across':
            across.append(clue_id)
        else:
            down.append(clue_id)

    return sorted(set(across), key=lambda x: int(x[:-1])), sorted(set(down),
                                                                  key=lambda x: int(
                                                                      x[:-1]))


def pattern_matches(puzzle_across, puzzle_down, pattern):
    """Check if puzzle matches a pattern."""
    pattern_across = sorted(pattern['across_clues'], key=lambda x: int(x[:-1]))
    pattern_down = sorted(pattern['down_clues'], key=lambda x: int(x[:-1]))

    return puzzle_across == pattern_across and puzzle_down == pattern_down


def analyze_source(cursor, table, patterns, limit=50):
    """Analyze puzzles from a source."""
    # Get distinct puzzles
    cursor.execute(f"""
        SELECT DISTINCT puzzle_number 
        FROM {table} 
        ORDER BY puzzle_number DESC 
        LIMIT {limit}
    """)

    puzzles = [row[0] for row in cursor.fetchall()]

    results = {
        'matched': 0,
        'unmatched': 0,
        'pattern_counts': defaultdict(int),
        'unmatched_puzzles': []
    }

    for puzzle_num in puzzles:
        across, down = get_puzzle_clue_pattern(cursor, table, puzzle_num)

        if not across or not down:
            continue

        matched = False
        for pattern in patterns:
            if pattern_matches(across, down, pattern):
                results['matched'] += 1
                results['pattern_counts'][pattern['pattern_id']] += 1
                matched = True
                break

        if not matched:
            results['unmatched'] += 1
            results['unmatched_puzzles'].append({
                'puzzle': puzzle_num,
                'across': across,
                'down': down,
                'total': len(across) + len(down)
            })

    return results


def main():
    print("=" * 60)
    print("GRID PATTERN ANALYSIS")
    print("=" * 60)

    # Load patterns
    try:
        patterns = load_patterns()
        print(f"Loaded {len(patterns)} grid patterns\n")
    except FileNotFoundError:
        print(f"Pattern file not found: {PATTERNS_FILE}")
        print("Please ensure common_grid_patterns.json is in the current directory")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Tables to check
    tables = {
        'telegraph_clues': 'Telegraph',
        'guardian_clues': 'Guardian',
        'times_clues': 'Times',
        'independent_clues': 'Independent',
        'ft_clues': 'FT'
    }

    for table, name in tables.items():
        print(f"\n{'=' * 40}")
        print(f"{name}")
        print('=' * 40)

        # Check table exists
        cursor.execute(f"""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='{table}'
        """)

        if not cursor.fetchone():
            print("Table not found - skipping")
            continue

        # Count puzzles
        cursor.execute(f"SELECT COUNT(DISTINCT puzzle_number) FROM {table}")
        total = cursor.fetchone()[0]
        print(f"Total puzzles: {total}")

        if total == 0:
            continue

        # Analyze
        results = analyze_source(cursor, table, patterns, limit=50)

        print(f"Matched patterns: {results['matched']}")
        print(f"Unmatched: {results['unmatched']}")

        if results['pattern_counts']:
            print("\nPattern distribution:")
            for pattern_id, count in sorted(results['pattern_counts'].items()):
                print(f"  {pattern_id}: {count}")

        if results['unmatched_puzzles']:
            print("\nUnmatched puzzles (first 3):")
            for p in results['unmatched_puzzles'][:3]:
                print(f"  Puzzle {p['puzzle']}: {p['total']} clues")
                print(f"    Across: {p['across']}")
                print(f"    Down: {p['down']}")

    conn.close()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()