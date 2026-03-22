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

    # Update the clues table — save only, no auto-approve
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

    # Return refreshed button row with updated tier badge
    from web.models import get_clue_by_id, compute_hint_tier, get_hint_steps, compute_solve_source
    from web.routes.hints import generate_token
    clue = get_clue_by_id(clue_id)
    new_tier, _ = compute_hint_tier(clue)
    steps = get_hint_steps(clue, is_admin=True)
    new_token = generate_token(clue_id)
    solve_source = compute_solve_source(clue)
    return render_template(
        "partials/admin_rerun_result.html",
        clue=clue, tier=new_tier, steps=steps,
        token=new_token, solve_source=solve_source,
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

    # For Guardian/Independent, try fifteensquared first
    if source in ("guardian", "independent") and answer:
        try:
            from sonnet_pipeline.fifteensquared_pipeline import (
                fetch_fifteensquared, store_fifteensquared_result
            )
            from sonnet_pipeline.tftt_pipeline import parse_with_haiku, score_parse
            from signature_solver.db import RefDB
            import anthropic as _anthropic

            # Get publication date for URL discovery
            pub_date = clue["publication_date"] if "publication_date" in clue.keys() else None

            fs_clues = fetch_fifteensquared(int(puzzle_number), source, pub_date)
            if fs_clues:
                import re as _re
                answer_clean = _re.sub(r'[^A-Za-z]', '', answer).upper()
                fc = None
                for f in fs_clues:
                    if _re.sub(r'[^A-Za-z]', '', f["answer"]).upper() == answer_clean:
                        fc = f
                        break

                if fc and fc.get("explanation"):
                    haiku_client = _anthropic.Anthropic()
                    ref_db = RefDB()
                    parsed, usage = parse_with_haiku(
                        haiku_client, clue_text, answer, fc["explanation"]
                    )
                    if parsed:
                        score, reasons = score_parse(parsed, answer, ref_db)
                        if score >= 70:
                            store_fifteensquared_result(
                                db, clue_id, parsed, score,
                                fc.get("definition", ""),
                                raw_explanation=fc.get("explanation", ""),
                                source_name=source,
                            )
                            success = True
        except Exception as e:
            import traceback
            traceback.print_exc()
            message = "fifteensquared error: %s" % e

    # For Times clues, try TFTT first
    if not success and source == "times" and answer:
        try:
            from sonnet_pipeline.tftt_pipeline import (
                fetch_tftt, parse_with_haiku, score_parse, store_tftt_result
            )
            from signature_solver.db import RefDB
            import anthropic as _anthropic

            tftt_clues = fetch_tftt(int(puzzle_number))
            if tftt_clues:
                import re as _re
                answer_clean = _re.sub(r'[^A-Za-z]', '', answer).upper()
                tc = None
                for t in tftt_clues:
                    if _re.sub(r'[^A-Za-z]', '', t["answer"]).upper() == answer_clean:
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
                            # Use Flask's admin DB connection (already open)
                            store_tftt_result(
                                db, clue_id, parsed, score,
                                tc.get("definition", ""),
                                raw_explanation=tc.get("explanation", "")
                            )
                            success = True
        except Exception as e:
            import traceback
            traceback.print_exc()
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
    steps = get_hint_steps(clue, is_admin=True)
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
    # Upsert confidence to HIGH (1.0) in structured_explanations
    existing_se = db.execute(
        "SELECT id FROM structured_explanations WHERE clue_id = ?", (clue_id,)
    ).fetchone()
    if existing_se:
        db.execute(
            "UPDATE structured_explanations SET confidence = 1.0 WHERE clue_id = ?",
            (clue_id,),
        )
    else:
        db.execute(
            """INSERT INTO structured_explanations
               (clue_id, definition_text, model_version, confidence,
                source, puzzle_number, clue_number)
               VALUES (?, ?, 'manual_approve', 1.0, ?, ?, ?)""",
            (
                clue_id,
                clue["definition"],
                clue["source"],
                clue["puzzle_number"],
                clue["clue_number"],
            ),
        )
    db.commit()

    # Return refreshed button row
    from web.models import get_clue_by_id, compute_hint_tier, get_hint_steps, compute_solve_source
    from web.routes.hints import generate_token
    clue = get_clue_by_id(clue_id)
    new_tier, _ = compute_hint_tier(clue)
    steps = get_hint_steps(clue, is_admin=True)
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


@bp.route("/set-answer/<int:clue_id>", methods=["POST"])
def set_answer(clue_id):
    """Set the answer for a clue (admin only). Used for prize puzzles."""
    _require_admin()

    answer = request.form.get("answer", "").strip().upper()
    answer = re.sub(r"[^A-Z ]", "", answer)  # letters and spaces only

    if not answer:
        return '<span class="text-xs text-red-500">No answer provided.</span>'

    db = get_admin_db()
    clue = db.execute("SELECT * FROM clues WHERE id = ?", (clue_id,)).fetchone()
    if clue is None:
        return '<span class="text-xs text-red-500">Clue not found.</span>'

    db.execute("UPDATE clues SET answer = ? WHERE id = ?", (answer, clue_id))
    db.commit()

    return f'<span class="text-xs text-green-600 font-bold">Answer set: {answer}</span>'


@bp.route("/queue-enrichment/<int:clue_id>", methods=["POST"])
def queue_enrichment(clue_id):
    """Extract pieces from a clue's explanation and queue for dashboard enrichment."""
    _require_admin()

    import json

    db = get_admin_db()
    clue = db.execute(
        "SELECT clue_text, answer, source, puzzle_number FROM clues WHERE id = ?",
        (clue_id,),
    ).fetchone()
    if clue is None:
        return '{"queued": 0}', 200, {"Content-Type": "application/json"}

    # Get pieces from structured_explanations
    se = db.execute(
        "SELECT components FROM structured_explanations WHERE clue_id = ?",
        (clue_id,),
    ).fetchone()
    if not se or not se["components"]:
        return '{"queued": 0}', 200, {"Content-Type": "application/json"}

    comps = json.loads(se["components"])
    pieces = comps.get("ai_pieces", [])

    queued = 0
    for p in pieces:
        mechanism = p.get("mechanism", "")
        clue_word = p.get("clue_word", "").strip()
        letters = p.get("letters", "").strip().upper()

        if not clue_word or not letters:
            continue

        # Only queue synonym and abbreviation mappings
        if mechanism == "synonym":
            etype = "synonym"
        elif mechanism == "abbreviation":
            etype = "abbreviation"
        else:
            continue

        # Skip if already in pending
        existing = db.execute(
            "SELECT 1 FROM pending_enrichments WHERE type = ? AND word = ? AND letters = ?",
            (etype, clue_word.lower(), letters),
        ).fetchone()
        if existing:
            continue

        db.execute("""
            INSERT INTO pending_enrichments
            (type, word, letters, answer, clue_text, source, puzzle_number, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            etype, clue_word.lower(), letters,
            clue["answer"] or "", clue["clue_text"] or "",
            clue["source"], clue["puzzle_number"],
        ))
        queued += 1

    db.commit()
    return json.dumps({"queued": queued}), 200, {"Content-Type": "application/json"}


@bp.route("/save-all-answers", methods=["POST"])
def save_all_answers():
    """Save multiple answers to the DB at once (admin only).

    Expects JSON body: {clue_id: "ANSWER", ...}
    Only updates clues that currently have no answer.
    Detects spanning clues and splits the answer between grid positions.
    """
    _require_admin()

    import json as _json
    data = request.get_json(silent=True) or {}

    db = get_admin_db()

    # Detect linked clue pairs in the submitted answers
    clue_info = {}
    clue_ids = []
    for k in data:
        try:
            clue_ids.append(int(k))
        except ValueError:
            pass

    if clue_ids:
        placeholders = ",".join("?" * len(clue_ids))
        rows = db.execute(
            f"SELECT id, clue_number, direction, clue_text, answer, source, puzzle_number FROM clues WHERE id IN ({placeholders})",
            clue_ids,
        ).fetchall()
        for r in rows:
            clue_info[r["id"]] = dict(r)

    # Find linked pairs: "See X" clue -> main clue
    linked_pairs = {}  # see_id -> main_id
    all_puzzle_clues = {}
    for ci in clue_info.values():
        text = (ci["clue_text"] or "").strip()
        m = re.match(r"^See (\d+)\s*(Across|Down|across|down)?$", text)
        if m:
            ref_num = m.group(1)
            ref_dir = (m.group(2) or "").lower()
            # Find the target clue in our submitted answers
            for other in clue_info.values():
                if str(other["clue_number"]) == ref_num and (not ref_dir or other["direction"] == ref_dir):
                    linked_pairs[ci["id"]] = other["id"]
                    break

    # Get grid cell counts for spanning clues
    cell_counts = {}
    if linked_pairs:
        # Use any clue to get source/puzzle_number
        sample = next(iter(clue_info.values()))
        source = sample["source"]
        puzzle_number = sample["puzzle_number"]

        from web.models import get_puzzle_grid_data
        from web.grid import build_grid_from_json, reconstruct_grid
        grid_data = get_puzzle_grid_data(source, puzzle_number)
        temp_grid = build_grid_from_json(source, puzzle_number, grid_data)
        if temp_grid is None and grid_data:
            temp_grid = reconstruct_grid(grid_data)
        if temp_grid:
            cells = temp_grid["cells"]
            rows_count = len(cells)
            cols_count = len(cells[0]) if rows_count > 0 else 0
            for r in range(rows_count):
                for c in range(cols_count):
                    cell = cells[r][c]
                    if cell is None or "number" not in cell:
                        continue
                    num = str(cell["number"])
                    is_across = (c + 1 < cols_count and cells[r][c + 1] is not None and
                                 (c == 0 or cells[r][c - 1] is None))
                    is_down = (r + 1 < rows_count and cells[r + 1][c] is not None and
                               (r == 0 or cells[r - 1][c] is None))
                    if is_across:
                        cnt = 0
                        ci = c
                        while ci < cols_count and cells[r][ci] is not None:
                            cnt += 1
                            ci += 1
                        cell_counts[(num, "across")] = cnt
                    if is_down:
                        cnt = 0
                        ri = r
                        while ri < rows_count and cells[ri][c] is not None:
                            cnt += 1
                            ri += 1
                        cell_counts[(num, "down")] = cnt

    # Track which IDs are the "see" side of a pair — skip saving for these
    see_ids = set(linked_pairs.keys())

    saved = 0
    for clue_id_str, answer in data.items():
        try:
            clue_id = int(clue_id_str)
        except ValueError:
            continue

        # Skip "See X" clues — the main clue handles both
        if clue_id in see_ids:
            continue

        answer = re.sub(r"[^A-Z ]", "", answer.upper().strip())
        if not answer:
            continue

        ci = clue_info.get(clue_id)
        if ci is None:
            continue

        # Check if this is the main side of a spanning pair
        see_id = None
        for sid, mid in linked_pairs.items():
            if mid == clue_id:
                see_id = sid
                break

        if see_id and see_id in clue_info:
            # Split the answer between main and see positions
            see_ci = clue_info[see_id]
            main_key = (str(ci["clue_number"]), ci["direction"])
            see_key = (str(see_ci["clue_number"]), see_ci["direction"])
            main_cells = cell_counts.get(main_key, 0)
            see_cells = cell_counts.get(see_key, 0)
            raw = answer.replace(" ", "")

            if main_cells + see_cells == len(raw):
                main_answer = raw[:main_cells]
                see_answer = raw[main_cells:]
                # Save main
                row = db.execute("SELECT answer FROM clues WHERE id = ?", (clue_id,)).fetchone()
                if row and (not row["answer"] or row["answer"].strip() == ""):
                    db.execute("UPDATE clues SET answer = ? WHERE id = ?", (main_answer, clue_id))
                    saved += 1
                # Save see
                row2 = db.execute("SELECT answer FROM clues WHERE id = ?", (see_id,)).fetchone()
                if row2 and (not row2["answer"] or row2["answer"].strip() == ""):
                    db.execute("UPDATE clues SET answer = ? WHERE id = ?", (see_answer, see_id))
                    saved += 1
                continue

        # Normal (non-spanning) clue
        row = db.execute("SELECT answer FROM clues WHERE id = ?", (clue_id,)).fetchone()
        if row is None:
            continue
        if not row["answer"] or row["answer"].strip() == "":
            db.execute("UPDATE clues SET answer = ? WHERE id = ?", (answer, clue_id))
            saved += 1

    db.commit()
    return _json.dumps({"saved": saved}), 200, {"Content-Type": "application/json"}
