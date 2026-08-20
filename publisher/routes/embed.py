"""The embed page — the entire widget, served for framing."""

import json
import sqlite3

from flask import Blueprint, abort, current_app, render_template, request

from publisher.auth import (
    csp_frame_ancestors, framing_origin, get_key_config, mint_token,
    origin_allowed,
)
from publisher import explanations
from publisher.puzzles import PuzzleNotFound, build_model, summarise

bp = Blueprint("embed", __name__)

SHELLS = {
    "telegraph": "shells/telegraph.html",
}


@bp.route("/embed/<source>/<number>")
def embed(source, number):
    """Render the widget for one puzzle, framed by one publisher.

    The key travels in the URL and is public. What actually restricts framing
    is the `frame-ancestors` header set below; the origin check is here so a
    copied key fails at the door rather than rendering a widget whose API calls
    then all fail.
    """
    key = request.args.get("k", "")
    config = get_key_config(key)
    if config is None:
        abort(403, "unknown publisher key")

    if source not in (config.get("sources") or []):
        abort(403, "this key is not licensed for that source")

    if not origin_allowed(config, framing_origin()):
        abort(403, "this widget is not licensed for that origin")

    shell = request.args.get("shell") or config.get("shell") or "telegraph"
    template = SHELLS.get(shell)
    if template is None:
        abort(404, "unknown shell")

    try:
        model, _ = build_model(source, number)
    except PuzzleNotFound:
        abort(404, "puzzle not available")

    response = current_app.make_response(render_template(
        template,
        model_json=json.dumps(model, separators=(",", ":")),
        model=model,
        token=mint_token(key, source, number),
        publisher=config.get("name", ""),
        source=source,
        number=number,
        renew_after=int(current_app.config["TOKEN_MAX_AGE"] * 0.6),
        # The card's own stylesheet, inlined with the shell. The full
        # explanation is now the site's rendered card verbatim
        # (publisher/explanations.card_html), so it needs the styles the site
        # gives it. Its contract is that it carries no page-shell rules, so it
        # cannot reach anything outside the card.
        card_css=explanations.card_stylesheet(),
    ))
    response.headers["Content-Security-Policy"] = csp_frame_ancestors(config)
    # No caching: the HTML carries a short-lived token, and a cached copy would
    # serve a dead one.
    response.headers["Cache-Control"] = "no-store"
    return response


_index_cache = {}


def _dev_index(source):
    """Every local puzzle, labelled with what it can actually demonstrate.

    Three things vary and all three matter when testing: whether the feed
    carries answers (Check and Reveal), whether the clue database has verified
    explanations (the hint ladder), and whether it is an embargoed prize puzzle
    (the honesty paths). Without the labels you open a puzzle at random and
    find half the product greyed out.
    """
    if source in _index_cache:
        return _index_cache[source]

    puzzles = summarise(source)

    coverage = {}
    db = sqlite3.connect(f"file:{current_app.config['CLUES_DB']}?mode=ro", uri=True)
    try:
        for number, total, passed in db.execute(
            """SELECT cl.puzzle_number, COUNT(*),
                      SUM(CASE WHEN LOWER(s.status)='pass' THEN 1 ELSE 0 END)
               FROM clues cl LEFT JOIN wfw_solve s ON s.clue_id = cl.id
               WHERE cl.source = ? GROUP BY cl.puzzle_number""",
            (source,),
        ):
            coverage[str(number)] = (total, passed or 0)
    finally:
        db.close()

    for puzzle in puzzles:
        total, passed = coverage.get(str(puzzle["display_number"]), (0, 0))
        puzzle["explained"] = passed
        puzzle["clue_total"] = total
        puzzle["fully_explained"] = bool(total) and passed >= total

    # Best first: answers AND a full set of explanations exercises everything.
    puzzles.sort(key=lambda p: (
        not (p["has_answers"] and p["fully_explained"]),
        not p["fully_explained"],
        not p["has_answers"],
    ))
    _index_cache[source] = puzzles
    return puzzles


@bp.route("/")
def index():
    """A local index of what can be demoed. Development only."""
    if not current_app.config.get("DEBUG"):
        abort(404)
    return render_template("dev_index.html", puzzles=_dev_index("telegraph"))
