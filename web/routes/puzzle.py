"""Puzzle routes — individual puzzle page with clues."""

import json
import re

from flask import Blueprint, render_template, abort, request, Response, g

from web.models import (
    classify_puzzle, TYPE_LABELS, _is_valid_type, get_puzzle_clues,
    get_puzzle_date, compute_hint_tier, get_hint_steps, compute_solve_source,
    get_puzzle_grid_data, get_puzzle_grid_solution,
)
from web.routes.hints import generate_token
from web.routes.clue import generate_clue_slug
from web.grid import reconstruct_grid, parse_grid_solution, build_grid_from_json

bp = Blueprint("puzzle", __name__)


@bp.route("/<source>/<puzzle_type>/<int:puzzle_number>")
def puzzle(source, puzzle_type, puzzle_number):
    """Puzzle page showing all clues with hint tier badges and reveal buttons."""
    # Validate source/type
    if not _is_valid_type(source, puzzle_type):
        abort(404)

    # Verify puzzle_number falls within the expected range for this type
    actual_slug, _ = classify_puzzle(source, puzzle_number)
    if actual_slug != puzzle_type:
        abort(404)

    clues = get_puzzle_clues(source, puzzle_number)
    if not clues:
        abort(404)

    pub_date = get_puzzle_date(source, puzzle_number)
    type_label = TYPE_LABELS[(source, puzzle_type)]

    # Split into across/down and attach tier info + tokens
    across = []
    down = []
    for clue in clues:
        tier, max_steps = compute_hint_tier(clue)
        steps = get_hint_steps(clue, tier=tier, is_admin=g.get("is_admin", False))
        clue_dict = dict(clue)
        clue_dict["tier"] = tier
        clue_dict["solve_source"] = compute_solve_source(clue)
        clue_dict["max_steps"] = max_steps
        clue_dict["total_steps"] = len(steps)
        clue_dict["steps"] = steps
        # Generate token for clues that have hints to reveal
        if steps:
            clue_dict["token"] = generate_token(clue["id"])
        else:
            clue_dict["token"] = None
        # Slug for individual clue page link
        clue_dict["slug"] = generate_clue_slug(clue["clue_text"] or "", clue["answer"] or "")
        if clue["direction"] == "across":
            across.append(clue_dict)
        else:
            down.append(clue_dict)

    # Detect linked clues: "See N Across/Down" or "See N"
    all_clues_list = across + down
    clue_by_key = {}  # (clue_number, direction) -> clue_dict
    for c in all_clues_list:
        clue_by_key[(str(c["clue_number"]), c["direction"])] = c

    for c in all_clues_list:
        c["is_linked_ref"] = False  # "See X" pointer clue
        c["linked_to"] = None       # ID of the "See X" clue that points to us
        c["linked_label"] = None    # e.g. "8d"
        c["linked_id"] = None       # clue ID of the linked clue

    for c in all_clues_list:
        text = (c.get("clue_text") or "").strip()
        # Match "See N Across/Down" or "See N"
        m = re.match(r"^See (\d+)\s*(Across|Down|across|down)?$", text)
        if m:
            ref_num = m.group(1)
            ref_dir = (m.group(2) or "").lower()
            if not ref_dir:
                # Bare "See N" — guess direction from context
                ref_dir = "across" if (ref_num, "across") in clue_by_key else "down"
            target = clue_by_key.get((ref_num, ref_dir))
            if target:
                c["is_linked_ref"] = True
                target["linked_to"] = c["id"]
                target["linked_label"] = str(c["clue_number"]) + c["direction"][0]
                target["linked_id"] = c["id"]

    return render_template(
        "puzzle.html",
        source=source,
        puzzle_type=puzzle_type,
        type_label=type_label,
        puzzle_number=puzzle_number,
        publication_date=pub_date,
        across=across,
        down=down,
    )


@bp.route("/<source>/<puzzle_type>/<int:puzzle_number>/grid")
def puzzle_grid(source, puzzle_type, puzzle_number):
    """Return the completed crossword grid as an HTMX fragment."""
    if not _is_valid_type(source, puzzle_type):
        abort(404)

    actual_slug, _ = classify_puzzle(source, puzzle_number)
    if actual_slug != puzzle_type:
        abort(404)

    # Path 1: rebuild live from JSON structure + current DB answers
    clue_data = get_puzzle_grid_data(source, puzzle_number)
    grid = build_grid_from_json(source, puzzle_number, clue_data)
    if grid is not None:
        return render_template("partials/grid.html", grid=grid)

    # Path 2: use stored solution string (no JSON available)
    stored = get_puzzle_grid_solution(source, puzzle_number)
    if stored:
        solution, grid_rows, grid_cols = stored
        grid = parse_grid_solution(solution, grid_rows, grid_cols)
        if grid is not None:
            return render_template("partials/grid.html", grid=grid)

    # Path 3: algorithmic reconstruction (last resort)
    if clue_data:
        grid = reconstruct_grid(clue_data)
        if grid is not None:
            return render_template("partials/grid.html", grid=grid)

    return render_template("partials/grid_error.html")


@bp.route("/<source>/<puzzle_type>/<int:puzzle_number>/grid-progress")
def puzzle_grid_progress(source, puzzle_type, puzzle_number):
    """Return the crossword grid showing only solved clues.

    Query param: solved — comma-separated clue IDs that the user has correctly solved.
    Unsolved cells are shown as empty white squares.
    """
    if not _is_valid_type(source, puzzle_type):
        abort(404)

    actual_slug, _ = classify_puzzle(source, puzzle_number)
    if actual_slug != puzzle_type:
        abort(404)

    # Parse solved clue IDs
    solved_param = request.args.get("solved", "")
    solved_ids = set()
    if solved_param:
        for s in solved_param.split(","):
            try:
                solved_ids.add(int(s.strip()))
            except ValueError:
                pass

    # Parse user-provided answers (for prize puzzles where DB has no answer)
    answers_param = request.args.get("answers", "")
    user_answers = {}
    if answers_param:
        try:
            user_answers = json.loads(answers_param)  # {clue_id: "ANSWER"}
        except (json.JSONDecodeError, TypeError):
            pass

    # Get all clue data, then filter to only solved clues
    all_clue_data = get_puzzle_grid_data(source, puzzle_number)

    # Detect linked clue pairs (spanning answers)
    from web.db import get_db
    db = get_db()
    all_puzzle_clues = db.execute(
        "SELECT id, clue_number, direction, answer, clue_text FROM clues WHERE source = ? AND puzzle_number = ?",
        (source, str(puzzle_number)),
    ).fetchall()

    # Build link map: "See X" clue -> main clue
    linked_pairs = {}  # {see_clue_id: main_clue_id}
    clue_by_key = {}
    for c in all_puzzle_clues:
        clue_by_key[(str(c["clue_number"]), c["direction"])] = c
    for c in all_puzzle_clues:
        text = (c["clue_text"] or "").strip()
        m = re.match(r"^See (\d+)\s*(Across|Down|across|down)?$", text)
        if m:
            ref_num = m.group(1)
            ref_dir = (m.group(2) or "").lower()
            if not ref_dir:
                ref_dir = "across" if (ref_num, "across") in clue_by_key else "down"
            target = clue_by_key.get((ref_num, ref_dir))
            if target:
                linked_pairs[c["id"]] = target["id"]

    # Get solved clue_number+direction pairs, and merge user answers for missing DB answers
    solved_clues = set()
    if solved_ids:
        placeholders = ",".join("?" * len(solved_ids))
        rows = db.execute(
            f"SELECT id, clue_number, direction, answer FROM clues WHERE id IN ({placeholders})",
            list(solved_ids),
        ).fetchall()
        for r in rows:
            solved_clues.add((str(r["clue_number"]), r["direction"]))
            if (not r["answer"] or r["answer"].strip() == "") and str(r["id"]) in user_answers:
                user_ans = user_answers[str(r["id"])]
                # Check if this is a linked clue — need to split the answer
                # Find the paired clue
                see_id = None
                main_id = None
                for see, main in linked_pairs.items():
                    if main == r["id"]:
                        see_id = see
                        main_id = r["id"]
                        break
                    if see == r["id"]:
                        see_id = r["id"]
                        main_id = main
                        break

                if see_id and main_id:
                    # This is a spanning clue — we'll split after building the grid
                    # For now, store the full answer; splitting happens below
                    all_clue_data.append({
                        "clue_number": r["clue_number"],
                        "direction": r["direction"],
                        "answer": user_ans,
                        "_spanning": True,
                        "_see_id": see_id,
                        "_main_id": main_id,
                    })
                else:
                    all_clue_data.append({
                        "clue_number": r["clue_number"],
                        "direction": r["direction"],
                        "answer": user_ans,
                    })

    # For spanning clues, split the full answer between grid positions
    # First build the full grid to count cells per clue
    _split_spanning_answers(all_clue_data, source, puzzle_number, linked_pairs, clue_by_key, solved_clues)

    # Build grid with only solved answers
    solved_clue_data = [
        c for c in all_clue_data
        if (str(c["clue_number"]), c["direction"]) in solved_clues
    ]

    # Path 1: JSON — works with partial data
    grid = build_grid_from_json(source, puzzle_number, solved_clue_data)
    if grid is not None:
        return render_template("partials/grid.html", grid=grid)

    # Path 2: stored solution — blank unsolved cells
    stored = get_puzzle_grid_solution(source, puzzle_number)
    if stored:
        solution, grid_rows, grid_cols = stored
        full_grid = parse_grid_solution(solution, grid_rows, grid_cols)
        if full_grid is not None:
            _blank_unsolved(full_grid, solved_clue_data, all_clue_data)
            return render_template("partials/grid.html", grid=full_grid)

    # Path 3: reconstruct from FULL data, then blank unsolved
    if all_clue_data:
        full_grid = reconstruct_grid(all_clue_data)
        if full_grid is not None:
            _blank_unsolved(full_grid, solved_clue_data, all_clue_data)
            return render_template("partials/grid.html", grid=full_grid)

    return render_template("partials/grid_error.html")


def _split_spanning_answers(all_clue_data, source, puzzle_number, linked_pairs, clue_by_key, solved_clues):
    """Split spanning answers (e.g. BATTLE OF HASTINGS) between two grid positions.

    Builds a temporary full grid to count cells per clue, then replaces
    the full answer with the correct portion for each grid position.
    """
    spanning = [c for c in all_clue_data if c.get("_spanning")]
    if not spanning:
        return

    # Build full grid from ALL clue data (to get cell counts)
    full_data = get_puzzle_grid_data(source, puzzle_number)
    temp_grid = build_grid_from_json(source, puzzle_number, full_data)
    if temp_grid is None:
        temp_grid = reconstruct_grid(full_data) if full_data else None
    if temp_grid is None:
        return

    # Count cells per clue position
    cells = temp_grid["cells"]
    rows = len(cells)
    cols = len(cells[0]) if rows > 0 else 0
    clue_cell_counts = {}  # (clue_number, direction) -> cell count

    for r in range(rows):
        for c_idx in range(cols):
            cell = cells[r][c_idx]
            if cell is None or "number" not in cell:
                continue
            num = str(cell["number"])
            is_across = (c_idx + 1 < cols and cells[r][c_idx + 1] is not None and
                         (c_idx == 0 or cells[r][c_idx - 1] is None))
            is_down = (r + 1 < rows and cells[r + 1][c_idx] is not None and
                       (r == 0 or cells[r - 1][c_idx] is None))
            if is_across:
                count = 0
                ci = c_idx
                while ci < cols and cells[r][ci] is not None:
                    count += 1
                    ci += 1
                clue_cell_counts[(num, "across")] = count
            if is_down:
                count = 0
                ri = r
                while ri < rows and cells[ri][c_idx] is not None:
                    count += 1
                    ri += 1
                clue_cell_counts[(num, "down")] = count

    # Now split each spanning answer
    for c in spanning:
        full_answer = c["answer"].replace(" ", "").upper()
        see_id = c["_see_id"]
        main_id = c["_main_id"]

        # Find the see clue's position
        see_clue = None
        for pc in clue_by_key.values():
            if pc["id"] == see_id:
                see_clue = pc
                break

        main_key = (str(c["clue_number"]), c["direction"])
        see_key = (str(see_clue["clue_number"]), see_clue["direction"]) if see_clue else None

        main_cells = clue_cell_counts.get(main_key, 0)
        see_cells = clue_cell_counts.get(see_key, 0) if see_key else 0

        if main_cells + see_cells == len(full_answer):
            # Split correctly
            c["answer"] = full_answer[:main_cells]
            # Add the see clue's portion
            if see_clue and see_key:
                solved_clues.add(see_key)
                all_clue_data.append({
                    "clue_number": see_clue["clue_number"],
                    "direction": see_clue["direction"],
                    "answer": full_answer[main_cells:],
                })
        # Clean up internal keys
        c.pop("_spanning", None)
        c.pop("_see_id", None)
        c.pop("_main_id", None)


def _blank_unsolved(grid, solved_clue_data, all_clue_data):
    """Keep only solved clue letters in the grid, blank the rest.

    Walks the grid to find clue start positions, traces each word across/down,
    and keeps letters only for solved clues.
    """
    cells = grid["cells"]
    rows = len(cells)
    cols = len(cells[0]) if rows > 0 else 0

    # Build solved answer lookup: {(clue_number, direction): "ANSWER"}
    solved_lookup = {}
    for c in solved_clue_data:
        ans = (c.get("answer") or "").replace(" ", "").upper()
        if ans:
            solved_lookup[(str(c["clue_number"]), c["direction"])] = ans

    # Find all clue starts and trace their cells
    solved_cells = set()  # (row, col) positions to keep

    for r in range(rows):
        for c in range(cols):
            cell = cells[r][c]
            if cell is None or "number" not in cell:
                continue
            num = str(cell["number"])

            # Check across: is there a cell to the right and no cell to the left?
            is_across = (c + 1 < cols and cells[r][c + 1] is not None and
                         (c == 0 or cells[r][c - 1] is None))
            # Check down: is there a cell below and no cell above?
            is_down = (r + 1 < rows and cells[r + 1][c] is not None and
                       (r == 0 or cells[r - 1][c] is None))

            if is_across and (num, "across") in solved_lookup:
                ans = solved_lookup[(num, "across")]
                for i, ch in enumerate(ans):
                    if c + i < cols and cells[r][c + i] is not None:
                        solved_cells.add((r, c + i))

            if is_down and (num, "down") in solved_lookup:
                ans = solved_lookup[(num, "down")]
                for i, ch in enumerate(ans):
                    if r + i < rows and cells[r + i][c] is not None:
                        solved_cells.add((r + i, c))

    # Blank all cells NOT in solved_cells
    for r in range(rows):
        for c in range(cols):
            if cells[r][c] is not None and (r, c) not in solved_cells:
                cells[r][c]["letter"] = ""


@bp.route("/<source>/<puzzle_type>/<int:puzzle_number>/crossings")
def puzzle_crossings(source, puzzle_type, puzzle_number):
    """Return crossing letter patterns for unsolved clues as JSON.

    Query param: solved — comma-separated clue IDs the user has solved.
    Returns: {clue_id: "B_L_P__K", ...} for each unsolved clue with crossings.
    """
    if not _is_valid_type(source, puzzle_type):
        abort(404)

    actual_slug, _ = classify_puzzle(source, puzzle_number)
    if actual_slug != puzzle_type:
        abort(404)

    # Parse solved clue IDs
    solved_param = request.args.get("solved", "")
    solved_ids = set()
    if solved_param:
        for s in solved_param.split(","):
            try:
                solved_ids.add(int(s.strip()))
            except ValueError:
                pass

    if not solved_ids:
        return Response("{}", mimetype="application/json")

    # Parse user-provided answers
    answers_param = request.args.get("answers", "")
    user_answers = {}
    if answers_param:
        try:
            user_answers = json.loads(answers_param)
        except (json.JSONDecodeError, TypeError):
            pass

    from web.db import get_db
    db = get_db()

    # Get ALL clues for this puzzle (need id, clue_number, direction, answer, clue_text)
    all_clues = db.execute(
        """SELECT id, clue_number, direction, answer, enumeration, clue_text
           FROM clues WHERE source = ? AND puzzle_number = ?""",
        (source, str(puzzle_number)),
    ).fetchall()

    # Build solved answer lookup (DB answer, or user answer as fallback)
    solved_lookup = {}
    for c in all_clues:
        if c["id"] in solved_ids:
            ans = c["answer"]
            if not ans or ans.strip() == "":
                ans = user_answers.get(str(c["id"]), "")
            if ans:
                solved_lookup[(str(c["clue_number"]), c["direction"])] = ans.replace(" ", "").upper()

    # Build the full grid — merge user answers for clues missing DB answers
    all_clue_data = get_puzzle_grid_data(source, puzzle_number)
    existing_keys = {(str(c["clue_number"]), c["direction"]) for c in all_clue_data}
    for c in all_clues:
        key = (str(c["clue_number"]), c["direction"])
        if key not in existing_keys and str(c["id"]) in user_answers:
            all_clue_data.append({
                "clue_number": c["clue_number"],
                "direction": c["direction"],
                "answer": user_answers[str(c["id"])],
            })

    full_grid = build_grid_from_json(source, puzzle_number, all_clue_data)
    if full_grid is None:
        stored = get_puzzle_grid_solution(source, puzzle_number)
        if stored:
            solution, grid_rows, grid_cols = stored
            full_grid = parse_grid_solution(solution, grid_rows, grid_cols)
        elif all_clue_data:
            full_grid = reconstruct_grid(all_clue_data)

    if full_grid is None:
        return Response("{}", mimetype="application/json")

    cells = full_grid["cells"]
    rows = len(cells)
    cols = len(cells[0]) if rows > 0 else 0

    # Map each cell to its letter from the full grid
    full_letters = {}
    for r in range(rows):
        for c in range(cols):
            if cells[r][c] is not None and cells[r][c].get("letter"):
                full_letters[(r, c)] = cells[r][c]["letter"]

    # Find which cells are filled by solved clues
    solved_cells = set()
    for r in range(rows):
        for c in range(cols):
            cell = cells[r][c]
            if cell is None or "number" not in cell:
                continue
            num = str(cell["number"])
            is_across = (c + 1 < cols and cells[r][c + 1] is not None and
                         (c == 0 or cells[r][c - 1] is None))
            is_down = (r + 1 < rows and cells[r + 1][c] is not None and
                       (r == 0 or cells[r - 1][c] is None))
            if is_across and (num, "across") in solved_lookup:
                ans = solved_lookup[(num, "across")]
                for i in range(len(ans)):
                    if c + i < cols and cells[r][c + i] is not None:
                        solved_cells.add((r, c + i))
            if is_down and (num, "down") in solved_lookup:
                ans = solved_lookup[(num, "down")]
                for i in range(len(ans)):
                    if r + i < rows and cells[r + i][c] is not None:
                        solved_cells.add((r + i, c))

    # Build enumeration lookup and detect linked clues
    enum_lookup = {}
    linked_clues = set()  # clue keys that are part of a spanning pair
    for c in all_clues:
        enum_lookup[(str(c["clue_number"]), c["direction"])] = c["enumeration"] or ""
        text = (c["clue_text"] or "").strip()
        m = re.match(r"^See (\d+)\s*(Across|Down|across|down)?$", text)
        if m:
            ref_num = m.group(1)
            ref_dir = (m.group(2) or "").lower() or ("across" if ref_num in [str(x["clue_number"]) for x in all_clues if x["direction"] == "across"] else "down")
            linked_clues.add((str(c["clue_number"]), c["direction"]))
            linked_clues.add((ref_num, ref_dir))

    def _insert_breaks(pat, enum_str):
        """Insert dashes at word break positions based on enumeration.

        e.g. pattern "R______E" + enum "(3,5)" -> "R__-____E"
             pattern "F______________" + enum "(7,8)" -> "F______-________"
        """
        if not enum_str:
            return pat
        import re as _re
        nums = _re.findall(r"\d+", enum_str)
        if not nums or sum(int(n) for n in nums) != len(pat):
            return pat
        if len(nums) <= 1:
            return pat
        # Determine separator: hyphen in enum = hyphen, comma = space-like dash
        sep = "-"
        result = []
        pos = 0
        for i, n in enumerate(nums):
            result.append(pat[pos:pos + int(n)])
            pos += int(n)
            if i < len(nums) - 1:
                result.append(sep)
        return "".join(result)

    # For each UNSOLVED clue, trace its cells and build crossing pattern
    crossings = {}
    for r in range(rows):
        for c in range(cols):
            cell = cells[r][c]
            if cell is None or "number" not in cell:
                continue
            num = str(cell["number"])
            is_across = (c + 1 < cols and cells[r][c + 1] is not None and
                         (c == 0 or cells[r][c - 1] is None))
            is_down = (r + 1 < rows and cells[r + 1][c] is not None and
                       (r == 0 or cells[r - 1][c] is None))

            # Across
            if is_across and (num, "across") not in solved_lookup:
                pattern = []
                ci = c
                while ci < cols and cells[r][ci] is not None:
                    if (r, ci) in solved_cells:
                        pattern.append(full_letters.get((r, ci), "_"))
                    else:
                        pattern.append("_")
                    ci += 1
                pat = "".join(pattern)
                if any(ch != "_" for ch in pat):
                    # Don't insert word breaks for linked/spanning clues — JS combines them
                    if (num, "across") not in linked_clues:
                        pat = _insert_breaks(pat, enum_lookup.get((num, "across"), ""))
                    for clue in all_clues:
                        if str(clue["clue_number"]) == num and clue["direction"] == "across":
                            crossings[str(clue["id"])] = pat
                            break

            # Down
            if is_down and (num, "down") not in solved_lookup:
                pattern = []
                ri = r
                while ri < rows and cells[ri][c] is not None:
                    if (ri, c) in solved_cells:
                        pattern.append(full_letters.get((ri, c), "_"))
                    else:
                        pattern.append("_")
                    ri += 1
                pat = "".join(pattern)
                if any(ch != "_" for ch in pat):
                    if (num, "down") not in linked_clues:
                        pat = _insert_breaks(pat, enum_lookup.get((num, "down"), ""))
                    for clue in all_clues:
                        if str(clue["clue_number"]) == num and clue["direction"] == "down":
                            crossings[str(clue["id"])] = pat
                            break

    return Response(json.dumps(crossings), mimetype="application/json")
