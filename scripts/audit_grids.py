"""Audit every puzzle in the database for grid availability.

Reports:
1. Puzzles with stored grid solution in puzzle_grids
2. Puzzles with JSON file that can build a grid
3. Puzzles where reconstruct_grid succeeds
4. Puzzles with NO grid at all (the failures we need to fix)

Usage:
    python scripts/audit_grids.py
    python scripts/audit_grids.py --source telegraph
    python scripts/audit_grids.py --fix   # Store grids for puzzles that can be reconstructed
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLUES_DB = ROOT / "data" / "clues_master.db"


def audit(source_filter=None, fix=False):
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row

    # Get all puzzles with at least one answer
    where = ""
    params = []
    if source_filter:
        where = "AND c.source = ?"
        params.append(source_filter)

    puzzles = conn.execute(f"""
        SELECT c.source, c.puzzle_number, c.publication_date,
               COUNT(*) as total_clues,
               SUM(CASE WHEN c.answer IS NOT NULL AND c.answer != '' THEN 1 ELSE 0 END) as with_answers
        FROM clues c
        WHERE c.answer IS NOT NULL AND c.answer != '' {where}
        GROUP BY c.source, c.puzzle_number
        HAVING with_answers > 0
        ORDER BY c.source, CAST(c.puzzle_number AS INTEGER)
    """, params).fetchall()

    print(f"Total puzzles to audit: {len(puzzles)}")

    # Check which have stored grids
    stored = set()
    rows = conn.execute("SELECT source, puzzle_number FROM puzzle_grids WHERE solution IS NOT NULL AND solution != ''").fetchall()
    for r in rows:
        stored.add((r["source"], r["puzzle_number"]))

    # Check which have grid JSON files
    has_json = set()
    rows2 = conn.execute("SELECT source, puzzle_number FROM puzzle_grids WHERE api_id IS NOT NULL").fetchall()
    for r in rows2:
        has_json.add((r["source"], r["puzzle_number"]))

    # Categories
    cat_stored = []       # Has stored solution string
    cat_json_only = []    # Has JSON but no stored solution
    cat_reconstruct = []  # No JSON, no stored, but reconstruct works
    cat_no_grid = []      # Nothing works
    cat_no_answers = []   # Has clues but no answers (can't build grid)

    # Import grid functions
    from web.grid import reconstruct_grid, build_grid_from_json, parse_grid_solution
    from scraper.danword.danword_lookup import find_puzzle_json
    import json

    t0 = time.time()
    for i, p in enumerate(puzzles):
        src = p["source"]
        pnum = p["puzzle_number"]
        key = (src, pnum)
        total = p["total_clues"]
        answers = p["with_answers"]

        if answers == 0:
            cat_no_answers.append((src, pnum, p["publication_date"], total, 0))
            continue

        # 1. Stored solution?
        if key in stored:
            cat_stored.append((src, pnum, p["publication_date"], total, answers))
            continue

        # 2. JSON file available?
        if key in has_json:
            # Try building from JSON
            clue_data = conn.execute("""
                SELECT clue_number, direction, answer FROM clues
                WHERE source = ? AND puzzle_number = ? AND answer IS NOT NULL AND answer != ''
            """, (src, str(pnum))).fetchall()
            clue_list = [dict(c) for c in clue_data]

            json_path = find_puzzle_json(src, int(pnum))
            if json_path:
                try:
                    with open(json_path, encoding="utf-8") as f:
                        data = json.load(f)
                    # Extract solution from JSON settings
                    j = data.get("json", data)
                    settings = j.get("settings", {})
                    solution = settings.get("solution", "")
                    copy = j.get("copy", {})
                    gridsize = copy.get("gridsize", {})
                    grid_rows = int(gridsize.get("rows", 15))
                    grid_cols = int(gridsize.get("cols", 15))

                    if solution and len(solution) == grid_rows * grid_cols:
                        if fix:
                            conn.execute("""
                                UPDATE puzzle_grids SET solution = ?, grid_rows = ?, grid_cols = ?
                                WHERE source = ? AND puzzle_number = ?
                            """, (solution, grid_rows, grid_cols, src, str(pnum)))
                            cat_stored.append((src, pnum, p["publication_date"], total, answers))
                        else:
                            cat_json_only.append((src, pnum, p["publication_date"], total, answers))
                        continue
                except Exception:
                    pass

            cat_json_only.append((src, pnum, p["publication_date"], total, answers))
            continue

        # 3. Try reconstruction
        clue_data = conn.execute("""
            SELECT clue_number, direction, answer FROM clues
            WHERE source = ? AND puzzle_number = ? AND answer IS NOT NULL AND answer != ''
        """, (src, str(pnum))).fetchall()
        clue_list = [dict(c) for c in clue_data]

        grid = reconstruct_grid(clue_list)
        if grid:
            if fix:
                # Build solution string from grid cells
                cells = grid["cells"]
                solution = ""
                for row in cells:
                    for cell in row:
                        if cell is None:
                            solution += " "
                        else:
                            solution += cell.get("letter", " ")
                conn.execute("""
                    INSERT OR REPLACE INTO puzzle_grids (source, puzzle_number, solution, grid_rows, grid_cols)
                    VALUES (?, ?, ?, ?, ?)
                """, (src, str(pnum), solution, grid["rows"], grid["cols"]))
                cat_stored.append((src, pnum, p["publication_date"], total, answers))
            else:
                cat_reconstruct.append((src, pnum, p["publication_date"], total, answers))
        else:
            cat_no_grid.append((src, pnum, p["publication_date"], total, answers))

        # Progress
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  ... {i+1}/{len(puzzles)} ({elapsed:.0f}s)")

    if fix:
        conn.commit()

    conn.close()
    elapsed = time.time() - t0

    # Report
    print(f"\nAudit complete in {elapsed:.0f}s")
    print(f"=" * 60)
    print(f"Stored grid solution:     {len(cat_stored):6}")
    print(f"JSON only (no solution):  {len(cat_json_only):6}")
    print(f"Reconstructable:          {len(cat_reconstruct):6}")
    print(f"NO GRID (failures):       {len(cat_no_grid):6}")
    print(f"No answers (can't build): {len(cat_no_answers):6}")
    print(f"{'=' * 60}")
    print(f"Total:                    {len(puzzles):6}")

    if cat_no_grid:
        print(f"\n--- Puzzles with NO GRID ({len(cat_no_grid)}) ---")
        by_source = {}
        for src, pnum, pub, total, ans in cat_no_grid:
            by_source.setdefault(src, []).append((pnum, pub, total, ans))
        for src in sorted(by_source):
            items = by_source[src]
            print(f"\n  {src} ({len(items)} puzzles):")
            for pnum, pub, total, ans in items[:10]:
                print(f"    #{pnum} ({pub}): {total} clues, {ans} answers")
            if len(items) > 10:
                print(f"    ... and {len(items) - 10} more")

    if cat_reconstruct and not fix:
        print(f"\n--- Reconstructable ({len(cat_reconstruct)}) — run with --fix to store ---")
        by_source = {}
        for src, pnum, pub, total, ans in cat_reconstruct:
            by_source.setdefault(src, []).append((pnum, pub, total, ans))
        for src in sorted(by_source):
            items = by_source[src]
            print(f"  {src}: {len(items)} puzzles can be stored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit puzzle grids")
    parser.add_argument("--source", type=str, default=None, help="Filter by source")
    parser.add_argument("--fix", action="store_true", help="Store grids for reconstructable puzzles")
    args = parser.parse_args()

    audit(source_filter=args.source, fix=args.fix)
