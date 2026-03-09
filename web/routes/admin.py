"""Admin routes — inline clue editing with DB write access."""

import re
import sys
from pathlib import Path

from flask import Blueprint, abort, g, redirect, render_template, request, session, url_for

from web.db import get_admin_db, get_db

bp = Blueprint("admin", __name__, url_prefix="/admin")

# Import grid rebuild functions from danword_lookup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scraper.danword.danword_lookup import (
    build_solution_string,
    find_puzzle_json,
    update_puzzle_grid_solution,
)


def _require_admin():
    """Abort 403 if not in admin session."""
    if not g.get("is_admin"):
        abort(403)


@bp.route("/logout")
def logout():
    """Clear admin session and redirect back."""
    session.pop("admin", None)
    referrer = request.referrer or "/"
    return redirect(referrer)


@bp.route("/edit/<int:clue_id>", methods=["GET"])
def edit_form(clue_id):
    """Return the inline edit form for a clue (HTMX fragment)."""
    _require_admin()

    db = get_db()
    clue = db.execute(
        """SELECT c.*, se.components
           FROM clues c
           LEFT JOIN structured_explanations se ON se.clue_id = c.id
           WHERE c.id = ?""",
        (clue_id,),
    ).fetchone()
    if clue is None:
        abort(404)

    # Build display explanation: ai_explanation first, then from components
    from web.models import _build_explanation
    display_explanation = _build_explanation(clue)

    # Wordplay type options for the dropdown
    from web import WORDPLAY_LABELS
    wordplay_options = [("", "— None —")] + sorted(WORDPLAY_LABELS.items(), key=lambda x: x[1])

    return render_template(
        "partials/admin_edit.html",
        clue=clue,
        display_explanation=display_explanation,
        wordplay_options=wordplay_options,
    )


@bp.route("/edit/<int:clue_id>", methods=["POST"])
def edit_save(clue_id):
    """Save clue edits and return success fragment."""
    _require_admin()

    db = get_admin_db()
    clue = db.execute("SELECT * FROM clues WHERE id = ?", (clue_id,)).fetchone()
    if clue is None:
        abort(404)

    # Read form fields
    clue_text = request.form.get("clue_text", "").strip()
    answer = request.form.get("answer", "").strip().upper()
    answer = re.sub(r"[^A-Z]", "", answer)  # letters only
    definition = request.form.get("definition", "").strip()
    wordplay_type = request.form.get("wordplay_type", "").strip()
    if wordplay_type == "__custom__":
        wordplay_type = ""
    ai_explanation = request.form.get("ai_explanation", "").strip()

    # Track what changed
    changes = []
    old_answer = clue["answer"] or ""

    if clue_text and clue_text != (clue["clue_text"] or ""):
        changes.append("clue text")
    if answer != re.sub(r"[^A-Z]", "", old_answer.upper()) if old_answer else answer:
        changes.append("answer")
    if definition != (clue["definition"] or ""):
        changes.append("definition")
    if wordplay_type != (clue["wordplay_type"] or ""):
        changes.append("wordplay type")
    if ai_explanation != (clue["ai_explanation"] or ""):
        changes.append("explanation")

    # Update the clues table
    db.execute(
        """UPDATE clues
           SET clue_text = ?, answer = ?, definition = ?, wordplay_type = ?, ai_explanation = ?
           WHERE id = ?""",
        (
            clue_text or clue["clue_text"],
            answer or None,
            definition or None,
            wordplay_type or None,
            ai_explanation or None,
            clue_id,
        ),
    )
    db.commit()

    # Rebuild grid if answer changed
    grid_rebuilt = False
    if answer != re.sub(r"[^A-Z]", "", old_answer.upper()) if old_answer else answer:
        source = clue["source"]
        puzzle_number = clue["puzzle_number"]
        json_path = find_puzzle_json(source, puzzle_number)
        if json_path:
            # Fetch all current answers for this puzzle
            rows = db.execute(
                """SELECT clue_number, direction, answer FROM clues
                   WHERE source = ? AND puzzle_number = ?
                   AND answer IS NOT NULL AND answer != ''""",
                (source, puzzle_number),
            ).fetchall()
            clue_answers = {(r["clue_number"], r["direction"]): r["answer"] for r in rows}
            result = build_solution_string(str(json_path), clue_answers)
            if result:
                sol, grid_rows, grid_cols = result
                update_puzzle_grid_solution(source, puzzle_number, sol, grid_rows, grid_cols)
                grid_rebuilt = True

    return render_template(
        "partials/admin_saved.html",
        changes=changes,
        grid_rebuilt=grid_rebuilt,
        clue_id=clue_id,
    )
