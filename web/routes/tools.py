"""Standalone crossword tool pages — SEO entry points for anagram solver
and pattern matcher.

Each tool page reuses the existing /helper/ endpoints via HTMX. When a
result matches a recent puzzle that's ≥80% solved, Cordelia offers the
full puzzle experience.
"""

import re

from flask import Blueprint, request, render_template, abort, current_app, make_response
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from web.db import get_db
from web.models import classify_puzzle
from web.session_token import issue_session_cookie, has_valid_session

bp = Blueprint("tools", __name__)

ALLOWED_SOURCES = ('telegraph', 'times', 'dailymail', 'guardian', 'independent')

SOURCE_NAMES = {
    "telegraph": "Telegraph", "times": "Times", "dailymail": "Daily Mail",
    "guardian": "Guardian", "independent": "Independent",
}


@bp.route("/tools")
def tools_index():
    return render_template("tools.html")


@bp.route("/tools/anagram")
def tools_anagram():
    prefill = request.args.get("letters", "").strip()
    response = make_response(render_template("tools_anagram.html", prefill_letters=prefill))
    return issue_session_cookie(response)


@bp.route("/tools/synonym")
def tools_synonym():
    prefill = request.args.get("word", "").strip()
    response = make_response(render_template("tools_synonym.html", prefill=prefill))
    return issue_session_cookie(response)


@bp.route("/tools/pattern")
def tools_pattern():
    prefill = request.args.get("pattern", "").strip()
    response = make_response(render_template("tools_pattern.html", prefill_pattern=prefill))
    return issue_session_cookie(response)


@bp.route("/tools/puzzle-match")
def puzzle_match():
    """Given an answer, find recent puzzles containing it with ≥80% solve rate."""
    token = request.args.get("ht", "")
    if not token:
        abort(403)
    s = URLSafeTimedSerializer(current_app.config["SECRET_KEY"])
    try:
        s.loads(token, max_age=7200, salt="helper-access")
    except (BadSignature, SignatureExpired):
        abort(403)
    # Page-load session cookie required (matches helper endpoints).
    if not has_valid_session():
        abort(403)

    answer = request.args.get("answer", "").strip().upper()
    answer_clean = re.sub(r'[^A-Z]', '', answer)
    if not answer_clean or len(answer_clean) < 3:
        return ""

    db = get_db()
    placeholders = ",".join("?" for _ in ALLOWED_SOURCES)

    # Find puzzles containing this answer
    rows = db.execute(
        f"""SELECT DISTINCT source, puzzle_number, publication_date
            FROM clues
            WHERE UPPER(REPLACE(answer, ' ', '')) = ?
              AND source IN ({placeholders})
            ORDER BY publication_date DESC
            LIMIT 10""",
        (answer_clean, *ALLOWED_SOURCES),
    ).fetchall()

    if not rows:
        return ""

    # Check solve rate for each candidate puzzle
    matches = []
    seen = set()
    for r in rows:
        key = (r["source"], r["puzzle_number"])
        if key in seen:
            continue
        seen.add(key)

        stats = db.execute(
            """SELECT COUNT(*) as total,
                      SUM(CASE WHEN has_solution = 1 THEN 1 ELSE 0 END) as solved
               FROM clues WHERE source = ? AND puzzle_number = ?""",
            (r["source"], r["puzzle_number"]),
        ).fetchone()

        if not stats or stats["total"] == 0:
            continue
        if stats["solved"] / stats["total"] < 0.8:
            continue

        type_slug, type_label = classify_puzzle(
            r["source"], r["puzzle_number"], r["publication_date"]
        )
        if type_slug:
            sname = SOURCE_NAMES.get(r["source"], r["source"].title())
            matches.append({
                "source_name": sname,
                "puzzle_number": r["puzzle_number"],
                "total": stats["total"],
                "url": f"/{r['source']}/{type_slug}/{r['puzzle_number']}",
            })

        if len(matches) >= 3:
            break

    if not matches:
        return ""

    return render_template("partials/puzzle_match.html", matches=matches)
