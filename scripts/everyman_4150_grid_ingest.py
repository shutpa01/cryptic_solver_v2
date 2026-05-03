"""Build the 4150 grid structure from the PDF and save to puzzle_grids.

Black cells come from the rectangle objects in the larger of the two
grids on the PDF (the empty puzzle for the new week). White cells
become dots, black cells become spaces — matching the convention
used by build_everyman_grid() when no solution is yet known.

Verifies the grid against the already-ingested clue list before writing.
"""

import sqlite3
import sys
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "everyman_4150.pdf"
DB_PATH = PROJECT_ROOT / "data" / "clues_master.db"

PUZZLE_NUMBER = "4150"
SOURCE = "guardian"
ROWS, COLS = 15, 15

# Bounds of the empty 4150 grid on the PDF (verified by inspection)
GRID_X0, GRID_X1 = 36.0, 399.0
GRID_Y0, GRID_Y1 = 410.0, 773.0


def extract_black_cells() -> set[tuple[int, int]]:
    """Return set of (row, col) for black cells, 0-indexed from top-left."""
    with pdfplumber.open(str(PDF_PATH)) as pdf:
        page = pdf.pages[0]
        rects = [
            r for r in page.rects
            if r.get("non_stroking_color") == 0.0
            and r.get("fill")
            and r["y0"] >= GRID_Y0 and r["y1"] <= GRID_Y1
            and r["x0"] >= GRID_X0 and r["x1"] <= GRID_X1
        ]
    cell = (GRID_X1 - GRID_X0) / COLS
    out: set[tuple[int, int]] = set()
    for r in rects:
        col = round((r["x0"] - GRID_X0) / cell)
        # PDF y-up: row 0 (top) is at the highest y value
        row = round((GRID_Y1 - r["y1"]) / cell)
        if 0 <= col < COLS and 0 <= row < ROWS:
            out.add((row, col))
    return out


def build_grid_string(black: set[tuple[int, int]]) -> str:
    """White cells are '.', black cells are ' '. Flat 225-char string."""
    chars = []
    for r in range(ROWS):
        for c in range(COLS):
            chars.append(" " if (r, c) in black else ".")
    return "".join(chars)


def compute_slots(black: set[tuple[int, int]]) -> tuple[dict, dict]:
    """Return (across_slots, down_slots) keyed by clue number.

    A cell is a slot start if:
      - across: cell is white, and (cell to left is black or c == 0),
                and cell to right is white
      - down:   cell is white, and (cell above is black or r == 0),
                and cell below is white
    """
    def is_white(r, c):
        return 0 <= r < ROWS and 0 <= c < COLS and (r, c) not in black

    across, down = {}, {}
    n = 0
    for r in range(ROWS):
        for c in range(COLS):
            if not is_white(r, c):
                continue
            starts_across = (c == 0 or not is_white(r, c - 1)) and is_white(r, c + 1)
            starts_down = (r == 0 or not is_white(r - 1, c)) and is_white(r + 1, c)
            if not (starts_across or starts_down):
                continue
            n += 1
            if starts_across:
                length = 0
                cc = c
                while is_white(r, cc):
                    length += 1
                    cc += 1
                across[n] = (r, c, length)
            if starts_down:
                length = 0
                rr = r
                while is_white(rr, c):
                    length += 1
                    rr += 1
                down[n] = (r, c, length)
    return across, down


def main():
    black = extract_black_cells()
    print(f"Black cells in PDF: {len(black)}")
    grid = build_grid_string(black)
    print(f"Grid string: {len(grid)} chars (rows*cols = {ROWS*COLS})")
    assert len(grid) == ROWS * COLS

    print()
    print("Grid layout (`.` white, ` ` black):")
    for r in range(ROWS):
        print("  " + grid[r*COLS:(r+1)*COLS])

    across, down = compute_slots(black)
    print()
    print(f"Slots derived from grid: {len(across)} across, {len(down)} down")

    # Compare against ingested clues
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT clue_number, direction, enumeration FROM clues "
        "WHERE source=? AND puzzle_number=? ORDER BY direction, CAST(clue_number AS INTEGER)",
        (SOURCE, PUZZLE_NUMBER),
    ).fetchall()
    by_dir = {"across": {}, "down": {}}
    for r in rows:
        n = int(r["clue_number"])
        # Sum the enumeration ignoring separators
        digits = "".join(ch if ch.isdigit() else " " for ch in r["enumeration"]).split()
        total = sum(int(d) for d in digits) if digits else 0
        by_dir[r["direction"]][n] = total

    print()
    print("Cross-check (clue # -> grid-slot length vs clue enum length):")
    mismatches = []
    for direction, slots in (("across", across), ("down", down)):
        for n, (r, c, length) in sorted(slots.items()):
            clue_total = by_dir[direction].get(n)
            ok = clue_total is not None and clue_total == length
            mark = "OK" if ok else "MISMATCH"
            print(f"  {direction:>6} {n:>2}: grid-slot len={length}, clue enum sums to {clue_total} [{mark}]")
            if not ok:
                mismatches.append((direction, n, length, clue_total))

    if mismatches:
        print(f"\n!! {len(mismatches)} mismatch(es) — NOT writing grid")
        sys.exit(1)

    # Write
    conn.execute(
        """INSERT INTO puzzle_grids (source, puzzle_number, solution, grid_rows, grid_cols)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(source, puzzle_number) DO UPDATE SET
             solution = excluded.solution,
             grid_rows = excluded.grid_rows,
             grid_cols = excluded.grid_cols""",
        (SOURCE, PUZZLE_NUMBER, grid, ROWS, COLS),
    )
    conn.commit()
    conn.close()
    print(f"\nSaved structure grid for {SOURCE} #{PUZZLE_NUMBER} ({ROWS}x{COLS}, "
          f"{sum(c == ' ' for c in grid)} black cells)")


if __name__ == "__main__":
    main()
