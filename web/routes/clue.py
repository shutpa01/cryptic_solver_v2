"""Individual clue page — the primary SEO entry point.

Each of the 500k clues gets its own URL with the clue text and answer in
the slug, e.g. /clue/companions-shredded-corset-ESCORT

Blueprint registration (already in web/__init__.py):
    from web.routes.clue import bp as clue_bp
    app.register_blueprint(clue_bp)
"""

import re

from flask import Blueprint, render_template, request, abort

from web.db import get_db
from web.models import (
    classify_puzzle, compute_hint_tier, get_hint_steps, compute_solve_source,
    get_hint_content, clue_slug,
)
from web.routes.hints import generate_token
from web.routes.clue_seo import (
    generate_meta_description, generate_faq_schema, generate_breadcrumb_schema,
)

bp = Blueprint("clue", __name__)


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

def generate_clue_slug(clue_text, answer):
    """Create a URL-safe slug from clue text and answer.

    Format: clue-text-words-here-ANSWER
    Example: "Companions shredded corset" + "ESCORT" -> "companions-shredded-corset-ESCORT"

    The answer is kept uppercase to visually separate it from the clue words
    and to make parsing unambiguous (last uppercase segment = answer).
    """
    # Clean clue text: lowercase, replace non-alphanumeric with hyphens
    text = clue_text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")

    if not text:
        return None

    # Clean answer: uppercase, alphanumeric only
    ans = re.sub(r"[^A-Za-z0-9]", "", (answer or "")).upper()
    if not ans:
        return None

    return f"{text}-{ans}"


def get_clue_by_slug(slug):
    """Reverse a slug back to a DB lookup.

    Parses the slug to extract the answer (last uppercase segment) and
    clue text words, then searches the database.

    Returns a list of matching clue rows (with structured_explanations
    joined), ordered by best explanation first.
    """
    if not slug:
        return []

    # Split slug into parts
    parts = slug.split("-")
    if len(parts) < 2:
        return []

    # Find the answer: scan from the end for consecutive uppercase parts.
    # The answer is always the last segment(s), all uppercase.
    answer_parts = []
    clue_parts = []
    found_answer = False

    for i in range(len(parts) - 1, -1, -1):
        p = parts[i]
        if not found_answer:
            # Check if this part is all uppercase (the answer)
            if p and p == p.upper() and p.isalpha():
                answer_parts.insert(0, p)
            else:
                # Once we hit a non-uppercase part, everything before is clue text
                found_answer = True
                clue_parts = parts[:i + 1]
        # If we already found the transition, we're done
        if found_answer:
            break

    if not found_answer:
        # All parts were uppercase — unlikely but handle it
        # Treat last part as answer, rest as clue
        answer_parts = [parts[-1]]
        clue_parts = parts[:-1]

    if not answer_parts or not clue_parts:
        return []

    answer = "".join(answer_parts)

    db = get_db()

    # Use LIKE on first few clue words to narrow candidates, plus exact answer match
    where_clauses = ["UPPER(c.answer) = ?"]
    params = [answer.upper()]

    for w in clue_parts[:3]:
        if w:
            where_clauses.append("LOWER(c.clue_text) LIKE ?")
            params.append(f"%{w}%")

    sql = """SELECT c.id, c.source, c.puzzle_number, c.publication_date,
                    c.clue_number, c.direction, c.clue_text, c.enumeration,
                    c.answer, c.definition, c.wordplay_type, c.explanation,
                    c.ai_explanation, se.components, se.confidence, se.model_version
             FROM clues c
             LEFT JOIN structured_explanations se ON se.clue_id = c.id
             WHERE c.source IN ('telegraph', 'times', 'guardian', 'independent')
               AND c.clue_text IS NOT NULL
               AND %s
             ORDER BY
                 se.confidence DESC,
                 c.publication_date DESC
             LIMIT 50""" % " AND ".join(where_clauses)

    rows = db.execute(sql, params).fetchall()

    # Filter to exact slug match in Python
    matches = []
    for row in rows:
        row_slug = generate_clue_slug(row["clue_text"], row["answer"])
        if row_slug == slug:
            matches.append(row)

    return matches


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@bp.route("/clue/<slug>")
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
    # If specific id requested, use that
    specific_id = request.args.get("id", type=int)
    if specific_id:
        from web.models import get_clue_by_id
        clue = get_clue_by_id(specific_id)
        if clue is None:
            abort(404)
        # Verify slug matches
        actual_slug = generate_clue_slug(clue["clue_text"], clue["answer"])
        if actual_slug != slug:
            abort(404)
        matches = get_clue_by_slug(slug)
        # Remove the selected clue from "others"
        matches = [m for m in matches if m["id"] != specific_id]
    else:
        matches = get_clue_by_slug(slug)
        if not matches:
            abort(404)
        clue = matches[0]
        matches = matches[1:]

    clue_dict = dict(clue)

    # Tier and hints
    tier, max_steps = compute_hint_tier(clue)
    steps = get_hint_steps(clue)
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
        o_slug = generate_clue_slug(other["clue_text"], other["answer"])
        other_appearances.append({
            "source": o_source,
            "puzzle_number": o_pnum,
            "publication_date": o_pub,
            "type_label": o_type_label,
            "answer": other["answer"],
            "clue_url": f"/clue/{o_slug}?id={other['id']}",
        })

    # SEO data
    meta_description = generate_meta_description(clue_dict)
    faq_schema = generate_faq_schema(clue_dict, steps)
    breadcrumb_schema = generate_breadcrumb_schema(clue_dict)

    return render_template(
        "clue.html",
        clue=clue_dict,
        other_appearances=other_appearances,
        meta_description=meta_description,
        faq_schema=faq_schema,
        breadcrumb_schema=breadcrumb_schema,
    )
