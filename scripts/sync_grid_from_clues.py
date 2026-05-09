"""Propagate clue answers into puzzle_grids.solution for a puzzle.

The puzzle page renders from puzzle_grids.solution, not from clues.answer.
After manually editing clue answers, run this to copy the letters into
the grid string at the correct slot positions.

Usage:
    python scripts/sync_grid_from_clues.py guardian 4150
"""

import argparse
import re
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "clues_master.db"


def compute_slots(sol: list[str], rows: int, cols: int):
    """Walk the grid (' '=black, anything else=white) and return slot starts.

    Returns (across, down) dicts keyed by clue number, value (start_row, start_col, length).
    """
    def is_white(r, c):
        return 0 <= r < rows and 0 <= c < cols and sol[r*cols + c] != " "

    across, down = {}, {}
    n = 0
    for r in range(rows):
        for c in range(cols):
            if not is_white(r, c):
                continue
            sa = (c == 0 or not is_white(r, c - 1)) and is_white(r, c + 1)
            sd = (r == 0 or not is_white(r - 1, c)) and is_white(r + 1, c)
            if not (sa or sd):
                continue
            n += 1
            if sa:
                length = 0
                cc = c
                while is_white(r, cc):
                    length += 1
                    cc += 1
                across[n] = (r, c, length)
            if sd:
                length = 0
                rr = r
                while is_white(rr, c):
                    length += 1
                    rr += 1
                down[n] = (r, c, length)
    return across, down


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("puzzle_number")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    grow = conn.execute(
        "SELECT solution, grid_rows, grid_cols FROM puzzle_grids WHERE source=? AND puzzle_number=?",
        (args.source, args.puzzle_number),
    ).fetchone()
    if not grow or not grow["solution"]:
        print(f"No grid found for {args.source} #{args.puzzle_number}")
        sys.exit(1)
    sol = list(grow["solution"])
    rows, cols = grow["grid_rows"], grow["grid_cols"]
    print(f"Loaded grid {rows}x{cols} ({len(sol)} cells)")

    across, down = compute_slots(sol, rows, cols)

    clues = conn.execute(
        "SELECT clue_number, direction, answer FROM clues "
        "WHERE source=? AND puzzle_number=? AND answer IS NOT NULL AND answer != ''",
        (args.source, args.puzzle_number),
    ).fetchall()

    filled = 0
    skipped = 0
    collisions = 0
    for cl in clues:
        n = int(cl["clue_number"])
        direction = cl["direction"]
        letters = re.sub(r"[^A-Za-z]", "", cl["answer"]).upper()
        slot = (across if direction == "across" else down).get(n)
        if not slot:
            print(f"  !! {n}{direction[0]}: no slot in grid; skipping")
            skipped += 1
            continue
        sr, sc, slen = slot
        if len(letters) != slen:
            print(f"  !! {n}{direction[0]} '{cl['answer']}': length {len(letters)} != slot length {slen}; skipping")
            skipped += 1
            continue
        for i, ch in enumerate(letters):
            if direction == "across":
                r, c = sr, sc + i
            else:
                r, c = sr + i, sc
            existing = sol[r*cols + c]
            if existing not in (".", ch):
                print(f"  !! collision at ({r},{c}) for {n}{direction[0]}: {existing!r} -> {ch!r}")
                collisions += 1
            sol[r*cols + c] = ch
        filled += 1

    new_sol = "".join(sol)
    print(f"\nFilled {filled} clue(s); skipped {skipped}; {collisions} collision(s).")
    print("Grid now:")
    for i in range(rows):
        print(f"  {new_sol[i*cols:(i+1)*cols]!r}")

    conn.execute(
        "UPDATE puzzle_grids SET solution=? WHERE source=? AND puzzle_number=?",
        (new_sol, args.source, args.puzzle_number),
    )
    conn.commit()
    print("\nWritten back to puzzle_grids.solution")


if __name__ == "__main__":
    main()
