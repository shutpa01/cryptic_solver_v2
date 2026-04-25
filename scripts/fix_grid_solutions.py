"""One-time fix: patch puzzle_grids solution strings using JSON grid_arrays.

The API solution string uses spaces for both black cells AND unchecked cells.
This script reads the JSON grid_array (authoritative for black vs white) and
replaces spaces with dots where the cell is actually white (unchecked).

parse_grid_solution() already handles dots as empty white cells.
"""

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = ROOT / "data" / "clues_master.db"
sys.path.insert(0, str(ROOT))

from scraper.danword.danword_lookup import find_puzzle_json


def fix_solution(solution, grid_array, rows, cols):
    """Replace spaces with dots where grid_array says the cell is white."""
    if not grid_array or len(solution) != rows * cols:
        return solution
    chars = list(solution)
    fixed = 0
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if chars[idx] == ' ' and grid_array[r][c].get('Blank') != 'blank':
                chars[idx] = '.'
                fixed += 1
    return ''.join(chars), fixed


def main():
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT source, puzzle_number, solution, grid_rows, grid_cols FROM puzzle_grids WHERE solution IS NOT NULL"
    ).fetchall()

    total = 0
    patched = 0

    for row in rows:
        source = row["source"]
        pnum = row["puzzle_number"]
        solution = row["solution"]
        grid_rows = row["grid_rows"]
        grid_cols = row["grid_cols"]

        json_path = find_puzzle_json(source, pnum)
        if json_path is None:
            continue

        total += 1

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        if "json" in data:
            grid_array = data["json"].get("grid")
        elif "data" in data:
            grid_array = data["data"].get("grid")
        else:
            grid_array = data.get("grid")

        if not grid_array:
            continue

        new_solution, fixed_count = fix_solution(solution, grid_array, grid_rows, grid_cols)
        if fixed_count > 0:
            conn.execute(
                "UPDATE puzzle_grids SET solution = ? WHERE source = ? AND puzzle_number = ?",
                (new_solution, source, pnum),
            )
            conn.commit()
            patched += 1
            print(f"  {source} #{pnum}: {fixed_count} unchecked cells fixed")

    conn.close()
    print(f"\nDone: {total} puzzles checked, {patched} patched")


if __name__ == "__main__":
    main()
