#!/usr/bin/env python3
"""Backfill Daily Mail grid solutions from the mailplus API.

Re-fetches the daily bundle for each DM puzzle date and builds the grid
solution from clue positions and answers. Only updates puzzles that have
a NULL solution in puzzle_grids.
"""

import sqlite3
import time
from datetime import datetime
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR.parent.parent / "data" / "clues_master.db"
API_URL = "https://api.mailplus.co.uk/puzzles/mail-plus/data/{}/bundle.json"
GAME_ID = 6  # Cryptic crossword


def build_grid_solution(game_data):
    """Build a flat solution string from DM clue positions and answers."""
    data = game_data.get('data', {})
    rows = int(data.get('rows', 15))
    cols = int(data.get('cols', 15))

    grid = [[' '] * cols for _ in range(rows)]
    has_any = False

    for clue in data.get('hor', []):
        r_idx = clue['r'] - 1
        c_idx = clue['c'] - 1
        ans = clue.get('answer', '').replace(' ', '')
        for i, ch in enumerate(ans):
            if 0 <= c_idx + i < cols:
                grid[r_idx][c_idx + i] = ch.upper()
                has_any = True

    for clue in data.get('ver', []):
        r_idx = clue['r'] - 1
        c_idx = clue['c'] - 1
        ans = clue.get('answer', '').replace(' ', '')
        for i, ch in enumerate(ans):
            if 0 <= r_idx + i < rows:
                grid[r_idx + i][c_idx] = ch.upper()
                has_any = True

    if not has_any:
        return None, rows, cols

    return ''.join(''.join(row) for row in grid), rows, cols


def main():
    conn = sqlite3.connect(str(DB_PATH))

    # Get DM puzzles needing grid solutions
    rows = conn.execute("""
        SELECT DISTINCT c.puzzle_number, c.publication_date
        FROM clues c
        JOIN puzzle_grids g ON c.source = g.source AND c.puzzle_number = g.puzzle_number
        WHERE c.source = 'dailymail'
          AND (g.solution IS NULL OR g.solution = '')
        ORDER BY c.publication_date
    """).fetchall()

    print(f"DM puzzles needing grid solutions: {len(rows)}")

    succeeded = 0
    failed = 0
    failed_dates = []

    for i, (puzzle_number, pub_date) in enumerate(rows, 1):
        if not pub_date:
            failed += 1
            continue

        url = API_URL.format(pub_date)
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                failed += 1
                failed_dates.append((pub_date, f'http_{r.status_code}'))
                time.sleep(0.3)
                continue

            bundle = r.json()
            game_data = None
            for game in bundle.get('games', []):
                if game.get('gameId') == GAME_ID:
                    game_data = game
                    break

            if not game_data:
                failed += 1
                failed_dates.append((pub_date, 'no_game_6'))
                time.sleep(0.3)
                continue

            solution, grid_rows, grid_cols = build_grid_solution(game_data)
            if not solution:
                failed += 1
                failed_dates.append((pub_date, 'no_solution'))
                time.sleep(0.3)
                continue

            conn.execute("""
                UPDATE puzzle_grids SET solution = ?, grid_rows = ?, grid_cols = ?
                WHERE source = 'dailymail' AND puzzle_number = ?
            """, (solution, grid_rows, grid_cols, puzzle_number))
            conn.commit()
            succeeded += 1

            if succeeded % 100 == 0:
                print(f"  --- {i}/{len(rows)}: {succeeded} OK, {failed} failed ---")

        except Exception as e:
            failed += 1
            failed_dates.append((pub_date, str(e)[:50]))

        time.sleep(0.3)

    conn.close()

    print(f"\n{'=' * 60}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")

    if failed_dates:
        from collections import Counter
        by_reason = Counter(r for _, r in failed_dates)
        print(f"\nFailures by reason:")
        for reason, cnt in by_reason.most_common():
            print(f"  {reason}: {cnt}")


if __name__ == "__main__":
    main()
