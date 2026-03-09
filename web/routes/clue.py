"""Individual clue page — the SEO entry point."""

from flask import Blueprint, render_template, request, abort

from web.models import (
    get_clues_by_slug, get_clue_by_id, classify_puzzle, TYPE_LABELS,
    compute_hint_tier, get_hint_steps, clue_slug,
)
from web.routes.hints import generate_token

bp = Blueprint("clue", __name__)


@bp.route("/clue/<slug>")
def clue_page(slug):
    """Individual clue page — looked up by URL slug.

    Shows the clue with hint buttons, source info, and link to full puzzle.
    If multiple clues share the same slug, shows the most recent with
    links to others.

    Optional ?id= param to select a specific clue when slug has duplicates.
    """
    # If specific id requested, use that
    specific_id = request.args.get("id", type=int)
    if specific_id:
        clue = get_clue_by_id(specific_id)
        if clue is None:
            abort(404)
        # Verify slug matches
        actual_slug = clue_slug(clue["clue_text"], clue["enumeration"])
        if actual_slug != slug:
            abort(404)
        matches = get_clues_by_slug(slug)
        # Remove the selected clue from "others"
        matches = [m for m in matches if m["id"] != specific_id]
    else:
        matches = get_clues_by_slug(slug)
        if not matches:
            abort(404)
        clue = matches[0]
        matches = matches[1:]
    clue_dict = dict(clue)

    # Tier and hints
    tier, max_steps = compute_hint_tier(clue)
    steps = get_hint_steps(clue)
    clue_dict["tier"] = tier
    clue_dict["max_steps"] = max_steps
    clue_dict["steps"] = steps
    clue_dict["token"] = generate_token(clue["id"]) if steps else None

    # Puzzle context
    source = clue["source"]
    puzzle_number = clue["puzzle_number"]
    type_slug, type_label = classify_puzzle(source, puzzle_number, clue["publication_date"])
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
        o_type_slug, o_type_label = classify_puzzle(o_source, o_pnum, other["publication_date"])
        o_slug = clue_slug(other["clue_text"], other["enumeration"])
        other_appearances.append({
            "source": o_source,
            "puzzle_number": o_pnum,
            "publication_date": other["publication_date"],
            "type_label": o_type_label,
            "answer": other["answer"],
            "clue_url": f"/clue/{o_slug}?id={other['id']}",
        })

    return render_template(
        "clue.html",
        clue=clue_dict,
        other_appearances=other_appearances,
    )
