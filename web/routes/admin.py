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

    # Update the clues table — auto-approve on save
    db.execute(
        """UPDATE clues
           SET clue_text = ?, answer = ?, definition = ?, wordplay_type = ?, ai_explanation = ?,
               reviewed = 1, has_solution = 1
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


@bp.route("/rerun/<int:clue_id>", methods=["POST"])
def rerun_clue(clue_id):
    """Re-run a clue through the full pipeline (TFTT + Sonnet) and return result as HTMX fragment."""
    _require_admin()

    db = get_admin_db()
    clue = db.execute("SELECT * FROM clues WHERE id = ?", (clue_id,)).fetchone()
    if clue is None:
        abort(404)

    source = clue["source"]
    puzzle_number = clue["puzzle_number"]
    answer = clue["answer"]
    clue_text = clue["clue_text"]

    # Clear previous results
    db.execute(
        "UPDATE clues SET definition = NULL, wordplay_type = NULL, "
        "ai_explanation = NULL, reviewed = NULL WHERE id = ?",
        (clue_id,),
    )
    db.execute(
        "DELETE FROM structured_explanations WHERE clue_id = ?",
        (clue_id,),
    )
    db.commit()

    success = False
    message = ""

    # For Times clues, try TFTT first
    if source == "times" and answer:
        try:
            from sonnet_pipeline.tftt_pipeline import (
                fetch_tftt, parse_with_haiku, score_parse, store_tftt_result
            )
            from signature_solver.db import RefDB
            import anthropic as _anthropic

            tftt_clues = fetch_tftt(int(puzzle_number))
            if tftt_clues:
                # Find matching clue by answer
                import re
                answer_clean = re.sub(r'[^A-Za-z]', '', answer).upper()
                tc = None
                for t in tftt_clues:
                    if re.sub(r'[^A-Za-z]', '', t["answer"]).upper() == answer_clean:
                        tc = t
                        break

                if tc and tc.get("explanation"):
                    haiku_client = _anthropic.Anthropic()
                    ref_db = RefDB()
                    parsed, usage = parse_with_haiku(
                        haiku_client, clue_text, answer, tc["explanation"]
                    )
                    if parsed:
                        score, reasons = score_parse(parsed, answer, ref_db)
                        if score >= 70:
                            import sqlite3
                            conn = sqlite3.connect(
                                str(PROJECT_ROOT / "data" / "clues_master.db"), timeout=30
                            )
                            store_tftt_result(
                                conn, clue_id, parsed, score,
                                tc.get("definition", ""),
                                raw_explanation=tc.get("explanation", "")
                            )
                            conn.close()
                            success = True
        except Exception as e:
            message = "TFTT error: %s" % e

    # Fall back to Sonnet explainer if TFTT didn't work
    if not success:
        from web.explainer import generate_explanation
        try:
            success, message, result = generate_explanation(clue_id)
        except Exception as e:
            return '<div class="mt-2 text-xs text-red-600 bg-red-50 rounded px-2 py-1">Error: %s</div>' % str(e)

        if not success:
            return '<div class="mt-2 text-xs text-red-600 bg-red-50 rounded px-2 py-1">Failed: %s</div>' % message

    # Return full button row matching the puzzle page layout
    from web.models import get_clue_by_id, compute_hint_tier, get_hint_steps, compute_solve_source
    from web.routes.hints import generate_token
    clue = get_clue_by_id(clue_id)
    new_tier, _ = compute_hint_tier(clue)
    steps = get_hint_steps(clue)
    new_token = generate_token(clue_id)
    solve_source = compute_solve_source(clue)
    return render_template(
        "partials/admin_rerun_result.html",
        clue=clue, tier=new_tier, steps=steps,
        token=new_token, solve_source=solve_source,
    )


@bp.route("/approve/<int:clue_id>", methods=["POST"])
def approve_clue(clue_id):
    """Mark a clue as approved (reviewed=1, has_solution=1)."""
    _require_admin()

    db = get_admin_db()
    clue = db.execute("SELECT * FROM clues WHERE id = ?", (clue_id,)).fetchone()
    if clue is None:
        abort(404)

    db.execute(
        "UPDATE clues SET reviewed = 1, has_solution = 1 WHERE id = ?",
        (clue_id,),
    )
    # Set confidence to HIGH (1.0) in structured_explanations
    db.execute(
        "UPDATE structured_explanations SET confidence = 1.0 WHERE clue_id = ?",
        (clue_id,),
    )
    db.commit()

    # Return refreshed button row
    from web.models import get_clue_by_id, compute_hint_tier, get_hint_steps, compute_solve_source
    from web.routes.hints import generate_token
    clue = get_clue_by_id(clue_id)
    new_tier, _ = compute_hint_tier(clue)
    steps = get_hint_steps(clue)
    new_token = generate_token(clue_id)
    solve_source = compute_solve_source(clue)
    return render_template(
        "partials/admin_rerun_result.html",
        clue=clue, tier=new_tier, steps=steps,
        token=new_token, solve_source=solve_source,
    )


@bp.route("/enrich", methods=["POST"])
def enrich_db():
    """Add an entry to the reference DB (cryptic_new.db)."""
    _require_admin()

    import sqlite3

    etype = request.form.get("type", "")
    word = request.form.get("word", "").strip()
    value = request.form.get("value", "").strip()

    if not word or not value:
        return '<span class="text-red-500">Both fields required.</span>'

    cryptic_db = PROJECT_ROOT / "data" / "cryptic_new.db"
    conn = sqlite3.connect(str(cryptic_db), timeout=30)

    msg = ""
    if etype == "synonym":
        existing = conn.execute(
            "SELECT 1 FROM synonyms_pairs WHERE word = ? AND synonym = ?",
            (word.lower(), value.upper()),
        ).fetchone()
        if existing:
            msg = '<span class="text-gray-500">Already exists: %s = %s</span>' % (word, value)
        else:
            conn.execute(
                "INSERT INTO synonyms_pairs (word, synonym, source) VALUES (?, ?, 'flask_admin')",
                (word.lower(), value.upper()),
            )
            conn.commit()
            msg = '<span class="text-green-600">Added synonym: %s = %s</span>' % (word, value)

    elif etype == "abbreviation":
        existing = conn.execute(
            "SELECT 1 FROM wordplay WHERE indicator = ? AND substitution = ?",
            (word.lower(), value.upper()),
        ).fetchone()
        if existing:
            msg = '<span class="text-gray-500">Already exists: %s = %s</span>' % (word, value)
        else:
            conn.execute(
                "INSERT INTO wordplay (indicator, substitution, category, confidence, notes) "
                "VALUES (?, ?, 'flask_admin', 'high', '')",
                (word.lower(), value.upper()),
            )
            conn.commit()
            msg = '<span class="text-green-600">Added abbreviation: %s = %s</span>' % (word, value)

    elif etype == "definition":
        existing = conn.execute(
            "SELECT 1 FROM definition_answers_augmented WHERE definition = ? AND answer = ?",
            (word.lower(), value.upper()),
        ).fetchone()
        if existing:
            msg = '<span class="text-gray-500">Already exists: %s = %s</span>' % (word, value)
        else:
            conn.execute(
                "INSERT INTO definition_answers_augmented (definition, answer, source) "
                "VALUES (?, ?, 'flask_admin')",
                (word.lower(), value.upper()),
            )
            conn.commit()
            msg = '<span class="text-green-600">Added definition: %s = %s</span>' % (word, value)

    elif etype == "indicator":
        existing = conn.execute(
            "SELECT 1 FROM indicators WHERE word = ? AND wordplay_type = ?",
            (word.lower(), value),
        ).fetchone()
        if existing:
            msg = '<span class="text-gray-500">Already exists: %s = %s</span>' % (word, value)
        else:
            conn.execute(
                "INSERT INTO indicators (word, wordplay_type, confidence, source) "
                "VALUES (?, ?, 'high', 'flask_admin')",
                (word.lower(), value),
            )
            conn.commit()
            msg = '<span class="text-green-600">Added indicator: %s = %s</span>' % (word, value)

    else:
        msg = '<span class="text-red-500">Unknown type: %s</span>' % etype

    conn.close()
    return msg
