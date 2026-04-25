"""Individual clue page — the primary SEO entry point.

Each of the 500k clues gets its own URL with the clue text and answer in
the slug, e.g. /clue/companions-shredded-corset-ESCORT

Blueprint registration (already in web/__init__.py):
    from web.routes.clue import bp as clue_bp
    app.register_blueprint(clue_bp)
"""

import re

from flask import Blueprint, render_template, request, abort, g

from web.db import get_db
from web.models import (
    classify_puzzle, compute_hint_tier, get_hint_steps, compute_solve_source,
    get_hint_content, clue_slug,
)
from flask import make_response

from web.routes.hints import generate_token
from web.routes.clue_seo import (
    generate_meta_description, generate_faq_schema, generate_breadcrumb_schema,
)
from web.rate_limit import rate_limit
from web.session_token import issue_session_cookie

bp = Blueprint("clue", __name__)


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

def generate_clue_slug(clue_text, answer=None, clue_id=None):
    """Create a URL-safe slug from clue ID and clue text.

    Format: {id}-{clue-text-words}
    Example: 2046138-parisian-is-running-home-to-host-a-european

    Answer is NOT included in the slug — it would defeat the purpose
    of progressive hints.
    """
    if not clue_id:
        return None
    # Clean clue text: lowercase, replace non-alphanumeric with hyphens
    text = clue_text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if not text:
        return None
    # Truncate to keep URLs reasonable
    words = text.split("-")[:12]
    text = "-".join(words)
    return f"{clue_id}-{text}"


def parse_clue_slug(slug):
    """Extract clue ID from slug.

    Returns (clue_id, slug_text) or (None, None).
    """
    if not slug:
        return None, None
    parts = slug.split("-", 1)
    if not parts or not parts[0].isdigit():
        return None, None
    clue_id = int(parts[0])
    slug_text = parts[1] if len(parts) > 1 else ""
    return clue_id, slug_text


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@bp.route("/clue/<slug>")
@rate_limit(scope="clue_page", limit=60, window=60)
def clue_page(slug):
    """Individual clue page — looked up by URL slug.

    Slug format: clue-text-words-ANSWER
    Example: /clue/companions-shredded-corset-ESCORT

    Shows the clue with progressive hint buttons (Definition, Type,
    Explanation, Answer), source info, link to full puzzle, and
    "also seen in" if the clue appears in multiple puzzles.

    Includes FAQPage and BreadcrumbList JSON-LD for SEO.

    Optional ?id= param to select a specific clue when slug has duplicates.
    """
    # Look up by clue ID embedded in slug
    clue_id, slug_text = parse_clue_slug(slug)

    # Also support ?id= parameter for backwards compatibility
    if not clue_id:
        clue_id = request.args.get("id", type=int)

    if not clue_id:
        abort(404)

    from web.models import get_clue_by_id
    clue = get_clue_by_id(clue_id)
    if clue is None:
        abort(404)

    # Find other appearances of the same clue text + answer
    db = get_db()
    other_rows = db.execute(
        """SELECT id, source, puzzle_number, publication_date, clue_number, direction,
                  clue_text, enumeration, answer, definition, wordplay_type,
                  explanation, ai_explanation
           FROM clues
           WHERE answer = ? AND clue_text = ? AND id != ?
             AND source IN ('telegraph', 'times', 'guardian', 'independent', 'dailymail')
           ORDER BY publication_date DESC
           LIMIT 5""",
        (clue["answer"], clue["clue_text"], clue_id),
    ).fetchall()
    matches = list(other_rows)

    clue_dict = dict(clue)

    # Tier and hints
    tier, max_steps = compute_hint_tier(clue)
    steps = get_hint_steps(clue, tier=tier, is_admin=g.get("is_admin", False))
    clue_dict["tier"] = tier
    clue_dict["solve_source"] = compute_solve_source(clue)
    clue_dict["max_steps"] = max_steps
    clue_dict["total_steps"] = len(steps)
    clue_dict["steps"] = steps
    clue_dict["token"] = generate_token(clue["id"]) if steps else None

    # Puzzle context
    source = clue["source"]
    puzzle_number = clue["puzzle_number"]
    pub_date = clue["publication_date"] if "publication_date" in clue.keys() else None
    type_slug, type_label = classify_puzzle(source, puzzle_number, pub_date)
    clue_dict["type_slug"] = type_slug
    clue_dict["type_label"] = type_label
    clue_dict["puzzle_url"] = (
        f"/{source}/{type_slug}/{puzzle_number}" if type_slug else None
    )

    # Other appearances of the same clue
    other_appearances = []
    for other in matches:
        o_source = other["source"]
        o_pnum = other["puzzle_number"]
        o_pub = other["publication_date"] if "publication_date" in other.keys() else None
        o_type_slug, o_type_label = classify_puzzle(o_source, o_pnum, o_pub)
        o_slug = generate_clue_slug(other["clue_text"], clue_id=other["id"])
        other_appearances.append({
            "source": o_source,
            "puzzle_number": o_pnum,
            "publication_date": o_pub,
            "type_label": o_type_label,
            "answer": other["answer"],
            "clue_url": f"/clue/{o_slug}",
        })

    # SEO data
    meta_description = generate_meta_description(clue_dict)
    faq_schema = generate_faq_schema(clue_dict, steps)
    breadcrumb_schema = generate_breadcrumb_schema(clue_dict)

    from web.models import get_source_puzzle_url
    source_puzzle_url = get_source_puzzle_url(source, puzzle_number)

    response = make_response(render_template(
        "clue.html",
        clue=clue_dict,
        other_appearances=other_appearances,
        source_puzzle_url=source_puzzle_url,
        meta_description=meta_description,
        faq_schema=faq_schema,
        breadcrumb_schema=breadcrumb_schema,
    ))
    return issue_session_cookie(response)
