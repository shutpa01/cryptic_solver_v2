"""Puzzle routes — individual puzzle page with clues."""

from flask import Blueprint, render_template, abort

from web.models import (
    classify_puzzle, TYPE_LABELS, _is_valid_type, get_puzzle_clues,
    get_puzzle_date, compute_hint_tier, get_hint_steps, compute_solve_source,
    get_puzzle_grid_data, get_puzzle_grid_solution,
)
from web.routes.hints import generate_token
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
        steps = get_hint_steps(clue)
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
        if clue["direction"] == "across":
            across.append(clue_dict)
        else:
            down.append(clue_dict)

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
