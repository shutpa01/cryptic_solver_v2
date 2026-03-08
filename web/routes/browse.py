"""Browse routes — home page and puzzle lists."""

from flask import Blueprint, render_template, request, abort

from web.models import BROWSE_SOURCES, TYPE_LABELS, _is_valid_type, get_puzzle_list

bp = Blueprint("browse", __name__)


@bp.route("/")
def home():
    """Home page — browse by source and type."""
    return render_template("home.html", sources=BROWSE_SOURCES)


@bp.route("/<source>/<puzzle_type>/")
def puzzle_list(source, puzzle_type):
    """Paginated puzzle list for a source/type combination."""
    if not _is_valid_type(source, puzzle_type):
        abort(404)

    page = request.args.get("page", 1, type=int)
    if page < 1:
        page = 1

    type_label = TYPE_LABELS[(source, puzzle_type)]
    puzzles, total_pages = get_puzzle_list(source, puzzle_type, page)

    return render_template(
        "puzzle_list.html",
        source=source,
        puzzle_type=puzzle_type,
        type_label=type_label,
        puzzles=puzzles,
        page=page,
        total_pages=total_pages,
    )
