"""Browse routes — home page, puzzle lists, and clue search."""

import re

from flask import Blueprint, render_template, request, abort, jsonify, g

from web.db import get_db
from web.models import BROWSE_SOURCES, TYPE_LABELS, _is_valid_type, get_puzzle_list, classify_puzzle

bp = Blueprint("browse", __name__)

SEARCH_SOURCES = ('telegraph', 'times', 'dailymail', 'guardian', 'independent')

_SOURCE_NAMES = {
    "telegraph": "Telegraph", "times": "Times", "dailymail": "Daily Mail",
    "guardian": "Guardian", "independent": "Independent", "cordelia": "Cordelia",
}


def _source_name(source):
    return _SOURCE_NAMES.get(source, source.title())


def _public_sources():
    """BROWSE_SOURCES filtered for non-admin visitors (hides cordelia)."""
    if getattr(g, "is_admin", False):
        return BROWSE_SOURCES
    return [s for s in BROWSE_SOURCES if s[0] != "cordelia"]


@bp.route("/")
def home():
    """Home page — browse by source and type."""
    return render_template("home.html", sources=_public_sources())


@bp.route("/puzzles")
def puzzles():
    """Puzzles page — browse all publications."""
    return render_template("puzzles.html", sources=_public_sources())


@bp.route("/about")
def about():
    """About Cordelia."""
    return render_template("about.html")


@bp.route("/search")
def search():
    """Search clues by text or puzzle number."""
    q = request.args.get("q", "").strip()
    if not q or len(q) < 3:
        return render_template("search.html", q=q, results=[], too_short=len(q) > 0)

    db = get_db()
    placeholders = ",".join("?" for _ in SEARCH_SOURCES)

    # Check if it looks like a puzzle number (e.g. "DT 31180", "29504", "times 5212")
    puzzle_match = re.match(
        r'^(?:dt|telegraph|times|guardian|independent|daily\s*mail)?\s*#?(\d{4,6})$',
        q, re.IGNORECASE,
    )
    if puzzle_match:
        num = puzzle_match.group(1)
        rows = db.execute(
            f"""SELECT DISTINCT source, puzzle_number, publication_date
               FROM clues
               WHERE puzzle_number = ? AND source IN ({placeholders})
               LIMIT 10""",
            (num, *SEARCH_SOURCES),
        ).fetchall()
        puzzles = []
        for r in rows:
            type_slug, type_label = classify_puzzle(r["source"], r["puzzle_number"], r["publication_date"])
            if type_slug:
                puzzles.append({
                    "source": r["source"],
                    "puzzle_number": r["puzzle_number"],
                    "type_slug": type_slug,
                    "type_label": type_label,
                    "publication_date": r["publication_date"],
                })
        if puzzles:
            return render_template("search.html", q=q, results=[], puzzles=puzzles)

    # Search by clue text
    words = q.split()[:6]
    conditions = []
    params = []
    for word in words:
        conditions.append("lower(c.clue_text) LIKE ?")
        params.append(f"%{word.lower()}%")

    where_clause = " AND ".join(conditions)
    rows = db.execute(
        f"""SELECT c.id, c.clue_text, c.answer, c.enumeration,
                   c.source, c.puzzle_number, c.publication_date
           FROM clues c
           WHERE {where_clause}
             AND c.answer IS NOT NULL AND length(c.answer) > 0
             AND c.source IN ({placeholders})
           ORDER BY c.publication_date DESC
           LIMIT 50""",
        (*params, *SEARCH_SOURCES),
    ).fetchall()

    from web.routes.clue import generate_clue_slug
    results = []
    for r in rows:
        slug = generate_clue_slug(r["clue_text"], clue_id=r["id"])
        results.append({**dict(r), "slug": slug})

    return render_template("search.html", q=q, results=results, puzzles=[])


@bp.route("/search/suggest")
def search_suggest():
    """Typeahead suggestions for the search box."""
    q = request.args.get("q", "").strip()
    if not q or len(q) < 3:
        return jsonify([])

    db = get_db()
    placeholders = ",".join("?" for _ in SEARCH_SOURCES)

    # Puzzle number search
    puzzle_match = re.match(
        r'^(?:dt|telegraph|times|guardian|independent|daily\s*mail)?\s*#?(\d{4,6})$',
        q, re.IGNORECASE,
    )
    if puzzle_match:
        num = puzzle_match.group(1)
        rows = db.execute(
            f"""SELECT DISTINCT source, puzzle_number, publication_date
               FROM clues
               WHERE puzzle_number = ? AND source IN ({placeholders})
               LIMIT 10""",
            (num, *SEARCH_SOURCES),
        ).fetchall()
        results = []
        for r in rows:
            type_slug, type_label = classify_puzzle(r["source"], r["puzzle_number"], r["publication_date"])
            if type_slug:

                sname = _source_name(r["source"])
                results.append({
                    "type": "puzzle",
                    "text": f"{sname} {type_label} #{r['puzzle_number']}",
                    "date": r["publication_date"] or "",
                    "url": f"/{r['source']}/{type_slug}/{r['puzzle_number']}",
                })
        if results:
            return jsonify(results)

    # Clue text search
    words = q.split()[:6]
    conditions = []
    params = []
    for word in words:
        conditions.append("lower(c.clue_text) LIKE ?")
        params.append(f"%{word.lower()}%")

    where_clause = " AND ".join(conditions)
    rows = db.execute(
        f"""SELECT c.id, c.clue_text, c.answer, c.enumeration, c.source
           FROM clues c
           WHERE {where_clause}
             AND c.answer IS NOT NULL AND length(c.answer) > 0
             AND c.source IN ({placeholders})
           ORDER BY c.publication_date DESC
           LIMIT 8""",
        (*params, *SEARCH_SOURCES),
    ).fetchall()

    from web.routes.clue import generate_clue_slug
    results = []
    for r in rows:
        slug = generate_clue_slug(r["clue_text"], clue_id=r["id"])
        if slug:
            enum = f" ({r['enumeration']})" if r["enumeration"] else ""
            results.append({
                "type": "clue",
                "text": r["clue_text"] + enum,
                "answer": r["answer"],
                "source": _source_name(r["source"]),
                "url": f"/clue/{slug}",
            })
    return jsonify(results)


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
