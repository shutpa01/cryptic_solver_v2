"""Admin routes — inline clue editing with DB write access."""

import re
import sys
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

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


@bp.route("/stage-two/<int:clue_id>")
def stage_two_casefile(clue_id):
    """Plain inspection page for the read-only Stage Two case file."""
    _require_admin()
    db = get_db()
    clue = db.execute(
        """SELECT id, source, puzzle_number, publication_date, clue_number,
                  direction, clue_text, enumeration, answer
           FROM clues
           WHERE id = ?""",
        (clue_id,),
    ).fetchone()
    if clue is None:
        abort(404)

    clue_dict = dict(clue)
    clue_dict["stage_two_casefile"] = None
    clue_dict["stage_two_casefile_error"] = None
    try:
        from signature_solver.stage_two_casefile import build_stage_two_casefile
        ref_db = current_app.get_shared_ref_db()
        casefile = build_stage_two_casefile(
            clue["clue_text"],
            clue["answer"],
            ref_db,
        )
        clue_dict["stage_two_casefile"] = casefile.as_dict()
    except Exception as exc:
        clue_dict["stage_two_casefile_error"] = str(exc)

    return render_template("admin_stage_two_casefile.html", clue=clue_dict)


def _store_wfw_rerun_result(conn, clue_id, clue, sr, answer_clean):
    """Persist a proven WFW result as the authoritative clue-page state."""
    from signature_solver.wfw_proof_store import write_wfw_proof_attempt
    from signature_solver.wfw_unified_proof import (
        build_wfw_proof_from_unified_result,
    )

    proof = build_wfw_proof_from_unified_result(
        getattr(sr, "wfw_unified_result", None))
    if not proof or proof.get("status") != "wfw_proven":
        return False

    token_parse = proof.get("token_parse") or {}
    blocks = token_parse.get("blocks") or []
    definition = next(
        (block.get("text") for block in blocks
         if block.get("kind") == "DEF_BLOCK"),
        None,
    )
    wordplay_type = token_parse.get("operation") or "wfw"
    explanation = _wfw_explanation_summary(proof)

    conn.execute(
        """UPDATE clues
           SET definition = ?, wordplay_type = ?, ai_explanation = ?,
               has_solution = 1, reviewed = 1
           WHERE id = ?""",
        (definition, wordplay_type, explanation, clue_id),
    )
    conn.execute(
        """INSERT INTO structured_explanations
           (clue_id, definition_text, wordplay_types, components,
            model_version, confidence, source, puzzle_number, clue_number,
            created_at, updated_at)
           VALUES (?, ?, ?, ?, 'wfw_unified', 1.0, ?, ?, ?,
                   CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
        (
            clue_id,
            definition,
            _json_dumps([wordplay_type]),
            _json_dumps({
                "source": "wfw_unified",
                "proof_status": proof.get("status"),
                "operation": wordplay_type,
                "answer": answer_clean,
            }),
            clue["source"],
            clue["puzzle_number"],
            clue["clue_number"],
        ),
    )
    row_id = write_wfw_proof_attempt(
        clue_id, clue["source"], clue["puzzle_number"], proof, conn=conn)
    if not row_id:
        raise RuntimeError("WFW proof was not written")
    return True


def _json_dumps(value):
    import json
    return json.dumps(value, sort_keys=True)


def _wfw_explanation_summary(proof):
    token_parse = proof.get("token_parse") or {}
    blocks = token_parse.get("blocks") or []
    pieces = [
        "%s -> %s" % (block.get("text") or "", block.get("value") or "")
        for block in blocks
        if block.get("kind") == "SOURCE_BLOCK"
    ]
    operation = token_parse.get("operation") or "wfw"
    if pieces:
        return "%s: %s" % (operation, "; ".join(pieces))
    return operation


@bp.route("/logout")
def logout():
    """Clear admin session and redirect back."""
    session.pop("admin", None)
    referrer = request.referrer or "/"
    return redirect(referrer)


# Roles accepted in the per-clue word-role dropdown on the clue page.
# Includes the verifier's structural labels (definition, link,
# indicator, anagram_fodder, ...) plus charade_joiner plus the
# DB-derived categories from the wordplay table that the verifier
# now records directly (abbreviation, single_letter, roman_numeral,
# nato_phonetic, cricket, chemistry, musical, foreign_*).
WORD_ROLE_CHOICES = (
    # Structural
    "definition",
    "link",
    "surface",
    "indicator",  # legacy generic; new code uses a specific *_indicator
    "anagram_fodder",
    "spoonerism_fodder",
    "hidden_source",
    "positional_source",
    "reversal_source",
    "deletion_source",
    "dbe_marker",
    "charade_joiner",
    "literal_source",
    "letter_source",
    "possessive_source",
    "letter_position_indicator",
    "unaccounted",
    # Letters-producing (verifier-derived from wordplay.category)
    "synonym",
    "synonym_source",  # legacy alias retained
    "abbreviation",
    "abbreviation_source",  # legacy alias retained
    "single_letter",
    "double_letter",
    "roman_numeral",
    "nato_phonetic",
    "cricket",
    "chemistry",
    "musical",
    "name",
    "shape",
    "example",
    "british_slang",
    "slang",
    "pronoun",
    "reference",
    "suffix",
    "first_letter",
    "substitution",
    "foreign",
    "foreign_french",
    "foreign_german",
    "foreign_spanish",
    "foreign_italian",
    "foreign_latin",
    "cryptic_synonym",
    "misc",
    # Indicator-type roles (derived from indicators.wordplay_type)
    "anagram_indicator",
    "container_indicator",
    "reversal_indicator",
    "deletion_indicator",
    "homophone_indicator",
    "hidden_indicator",
    "insertion_indicator",
    "acrostic_indicator",
    "parts_indicator",
    "positional_indicator",
    "selection_indicator",
    "alternating_indicator",
    "spoonerism_indicator",
    "charade_indicator",
    "first_letter_indicator",
    "last_letter_indicator",
    "middle_letter_indicator",
    "outer_letter_indicator",
    "substitution_indicator",
)


@bp.route("/accept-enrichment/<int:clue_id>/<int:word_index>", methods=["POST"])
def accept_enrichment(clue_id, word_index):
    """Accept an inline enrichment proposed on the clue page.

    Inserts the (word, letters) pair into the appropriate reference
    table (synonyms_pairs / definition_answers_augmented / wordplay /
    indicators) with source='admin_clue_page' for traceability. Also
    clears any matching rejected_enrichments row so the entry isn't
    blocked from later automatic queueing.
    """
    _require_admin()
    etype = (request.form.get("type") or "").strip()
    word = (request.form.get("word") or "").strip()
    letters = (request.form.get("letters") or "").strip()
    if not (etype and word and letters):
        abort(400)
    import sqlite3 as _sqlite3
    ref = _sqlite3.connect(
        str(PROJECT_ROOT / "data" / "cryptic_new.db"), timeout=10)
    try:
        if etype == "synonym":
            ref.execute(
                "INSERT INTO synonyms_pairs (word, synonym, source) "
                "VALUES (?, ?, 'admin_clue_page')",
                (word, letters.upper()))
        elif etype == "abbreviation":
            ref.execute(
                "INSERT INTO wordplay "
                "(indicator, substitution, category, confidence) "
                "VALUES (?, ?, 'abbreviation', 'high')",
                (word, letters.upper()))
        elif etype == "definition":
            ref.execute(
                "INSERT INTO definition_answers_augmented "
                "(definition, answer, source) "
                "VALUES (?, ?, 'admin_clue_page')",
                (word, letters.upper()))
        elif etype == "indicator":
            ref.execute(
                "INSERT INTO indicators "
                "(word, wordplay_type, source) "
                "VALUES (?, ?, 'admin_clue_page')",
                (word, letters.lower()))
        elif etype == "homophone":
            ref.execute(
                "INSERT OR IGNORE INTO homophones (word, homophone) "
                "VALUES (?, ?)",
                (word.lower(), letters.lower()))
        else:
            abort(400)
        ref.commit()
    finally:
        ref.close()
    # Clear any rejection so the entry stays accepted
    db = get_admin_db()
    db.execute(
        "DELETE FROM rejected_enrichments WHERE type=? "
        "AND LOWER(word)=? AND UPPER(letters)=?",
        (etype, word.lower(),
         letters.upper() if etype != "indicator" else letters.lower()))
    db.commit()
    return ('<span class="text-xs text-emerald-700 font-medium">'
            f'Accepted: {etype}</span>')


@bp.route("/word-role/<int:clue_id>/<int:word_index>", methods=["POST"])
def set_word_role(clue_id, word_index):
    """Save a manual role override for a single clue word.

    Returns a tiny "Saved" indicator that HTMX swaps in next to the
    dropdown. The underlying row in clue_word_roles is written with
    source='manual' and survives future verifier auto-classifications.
    """
    _require_admin()
    role = (request.form.get("role") or "").strip()
    word_text = (request.form.get("word_text") or "").strip()
    letters_raw = (request.form.get("letters") or "").strip().upper()
    letters = letters_raw if letters_raw else None
    if role not in WORD_ROLE_CHOICES:
        abort(400)
    if not word_text:
        abort(400)
    from sonnet_pipeline.word_roles_store import write_manual_role
    write_manual_role(clue_id, word_index, word_text, role, letters=letters)
    return '<span class="text-xs text-emerald-600">Saved</span>'


@bp.route("/wfw-correction/<int:clue_id>", methods=["POST"])
def save_wfw_correction(clue_id):
    """Add human WFW facts to the reference DB, deduped.

    This is deliberately not a proof writer.  The human action is to add the
    missing crossword facts; the next solve/re-run must prove the clue using
    the normal WFW path.
    """
    _require_admin()
    db = get_admin_db()
    clue = db.execute(
        """SELECT id, source, puzzle_number, clue_text, answer
           FROM clues
           WHERE id = ?""",
        (clue_id,),
    ).fetchone()
    if clue is None:
        abort(404)

    entries = []
    definition_text = (request.form.get("definition_text") or "").strip()
    definition_answer = (
        request.form.get("definition_answer") or clue["answer"] or ""
    ).strip()
    if definition_text and definition_answer:
        entries.append({
            "type": "definition",
            "word": definition_text,
            "value": definition_answer,
        })

    fact_type = (request.form.get("fact_type") or "").strip()
    fact_word = (request.form.get("fact_word") or "").strip()
    fact_value = (request.form.get("fact_value") or "").strip()
    if fact_type and fact_word and fact_value:
        entries.append({
            "type": fact_type,
            "word": fact_word,
            "value": fact_value,
        })

    entries.extend(_parse_db_fact_lines(request.form.get("db_facts") or ""))

    # Backwards compatibility for the earlier span-based form while we replace
    # it: convert spans into DB facts instead of writing a sidecar proof.
    definition_span = _parse_span(request.form.get("definition_span") or "")
    pieces = _parse_manual_pieces(request.form.get("pieces") or "")
    if definition_span or pieces:
        entries.extend(_entries_from_span_form(
            clue["clue_text"], clue["answer"], definition_span, pieces))

    if not entries:
        abort(400)

    added, existing = _write_wfw_db_entries(entries)

    # The page may have a stale WFW proof from before these facts were added.
    # The user-facing truth after this action is the DB; the clue should be
    # re-run to create a fresh proof.
    try:
        db.execute("DELETE FROM wfw_proof_attempts WHERE clue_id = ?", (clue_id,))
    except Exception:
        pass
    db.commit()

    from flask import current_app, make_response
    for entry in added:
        if hasattr(current_app, "patch_word_coverage_db"):
            current_app.patch_word_coverage_db(
                entry["type"], entry["word"], entry["value"])

    parts = []
    if added:
        parts.append("%d added" % len(added))
    if existing:
        parts.append("%d already in DB" % len(existing))
    msg = "WFW DB updated: " + ", ".join(parts)
    response = make_response(
        '<span class="text-xs text-emerald-700 font-semibold">%s. Now re-run the clue.</span>'
        % _html_escape(msg)
    )
    return response


def _parse_span(value):
    match = re.match(r"^\s*(\d+)\s*[-:,]\s*(\d+)\s*$", value or "")
    if not match:
        return None
    start, end = int(match.group(1)), int(match.group(2))
    if start < 0 or end <= start:
        return None
    return (start, end)


def _parse_manual_pieces(value):
    pieces = []
    for raw_line in (value or "").replace(";", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(
            r"^\s*(\d+)\s*[-:,]\s*(\d+)\s*=\s*([A-Za-z -]+)"
            r"(?:\s*/\s*([a-z_]+))?\s*$",
            line,
        )
        if not match:
            abort(400)
        pieces.append({
            "clue_span": (int(match.group(1)), int(match.group(2))),
            "value": match.group(3).strip().upper(),
            "mechanism": (match.group(4) or "synonym").strip(),
        })
    return pieces


def _parse_db_fact_lines(value):
    """Parse admin-entered WFW facts.

    Accepted forms:
      mount=RIDE
      synonym: mount=RIDE
      definition: sit across=BESTRIDE
      abbreviation: west=W
      indicator: returned=reversal
      homophone: flour=flower
    """
    entries = []
    for raw_line in (value or "").replace(";", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        etype = "synonym"
        if ":" in line:
            prefix, rest = line.split(":", 1)
            prefix = prefix.strip().lower().replace(" ", "_")
            if prefix in ("synonym", "definition", "abbreviation",
                          "indicator", "homophone"):
                etype = prefix
                line = rest.strip()
        match = re.match(r"^(.+?)\s*(?:=|->|→)\s*(.+)$", line)
        if not match:
            abort(400)
        entries.append({
            "type": etype,
            "word": match.group(1).strip(),
            "value": match.group(2).strip(),
        })
    return entries


def _entries_from_span_form(clue_text, answer, definition_span, pieces):
    from signature_solver.wfw_atoms import build_wfw_atom_context
    atom_context = build_wfw_atom_context(clue_text, answer)
    entries = []
    if definition_span is not None:
        definition_text = _token_span_text(atom_context, definition_span)
        if definition_text:
            entries.append({
                "type": "definition",
                "word": definition_text,
                "value": answer,
            })
    for piece in pieces:
        source_text = _token_span_text(atom_context, piece["clue_span"])
        if source_text:
            entries.append({
                "type": piece.get("mechanism") or "synonym",
                "word": source_text,
                "value": piece["value"],
            })
    return entries


def _token_span_text(atom_context, span):
    start, end = span
    tokens = atom_context.clue_tokens[start:end]
    return " ".join(token.text for token in tokens).strip()


def _write_wfw_db_entries(entries):
    import sqlite3
    ref = sqlite3.connect(
        str(PROJECT_ROOT / "data" / "cryptic_new.db"), timeout=30)
    added = []
    existing = []
    try:
        for entry in entries:
            normalised = _normalise_wfw_db_entry(entry)
            if normalised is None:
                continue
            if _wfw_db_entry_exists(ref, normalised):
                existing.append(normalised)
                continue
            _insert_wfw_db_entry(ref, normalised)
            added.append(normalised)
        ref.commit()
    finally:
        ref.close()
    return added, existing


def _normalise_wfw_db_entry(entry):
    etype = (entry.get("type") or "synonym").strip().lower()
    word = (entry.get("word") or "").strip()
    value = (entry.get("value") or "").strip()
    if etype not in ("synonym", "definition", "abbreviation",
                     "indicator", "homophone"):
        abort(400)
    if not word or not value:
        return None
    if etype == "indicator":
        value = value.lower().replace(" ", "_")
    elif etype == "homophone":
        word = word.lower()
        value = value.lower()
    else:
        value = value.upper()
    return {"type": etype, "word": word, "value": value}


def _wfw_db_entry_exists(conn, entry):
    etype, word, value = entry["type"], entry["word"], entry["value"]
    if etype == "synonym":
        return conn.execute(
            "SELECT 1 FROM synonyms_pairs "
            "WHERE LOWER(word)=LOWER(?) AND UPPER(synonym)=UPPER(?) "
            "LIMIT 1",
            (word, value),
        ).fetchone() is not None
    if etype == "definition":
        return conn.execute(
            "SELECT 1 FROM definition_answers_augmented "
            "WHERE LOWER(definition)=LOWER(?) AND UPPER(answer)=UPPER(?) "
            "LIMIT 1",
            (word, value),
        ).fetchone() is not None
    if etype == "abbreviation":
        return conn.execute(
            "SELECT 1 FROM wordplay "
            "WHERE LOWER(indicator)=LOWER(?) AND UPPER(substitution)=UPPER(?) "
            "LIMIT 1",
            (word, value),
        ).fetchone() is not None
    if etype == "indicator":
        return conn.execute(
            "SELECT 1 FROM indicators "
            "WHERE LOWER(word)=LOWER(?) AND LOWER(wordplay_type)=LOWER(?) "
            "LIMIT 1",
            (word, value),
        ).fetchone() is not None
    if etype == "homophone":
        return conn.execute(
            "SELECT 1 FROM homophones "
            "WHERE LOWER(word)=LOWER(?) AND LOWER(homophone)=LOWER(?) "
            "LIMIT 1",
            (word, value),
        ).fetchone() is not None
    return False


def _insert_wfw_db_entry(conn, entry):
    etype, word, value = entry["type"], entry["word"], entry["value"]
    if etype == "synonym":
        conn.execute(
            "INSERT INTO synonyms_pairs (word, synonym, source) "
            "VALUES (?, ?, 'admin_wfw_fact')",
            (word.lower(), value.upper()),
        )
    elif etype == "definition":
        conn.execute(
            "INSERT INTO definition_answers_augmented "
            "(definition, answer, source) VALUES (?, ?, 'admin_wfw_fact')",
            (word.lower(), value.upper()),
        )
    elif etype == "abbreviation":
        conn.execute(
            "INSERT INTO wordplay "
            "(indicator, substitution, category, confidence, notes) "
            "VALUES (?, ?, 'abbreviation', 'high', 'admin_wfw_fact')",
            (word.lower(), value.upper()),
        )
    elif etype == "indicator":
        conn.execute(
            "INSERT INTO indicators "
            "(word, wordplay_type, confidence, source) "
            "VALUES (?, ?, 'high', 'admin_wfw_fact')",
            (word.lower(), value.lower()),
        )
    elif etype == "homophone":
        conn.execute(
            "INSERT INTO homophones (word, homophone) VALUES (?, ?)",
            (word.lower(), value.lower()),
        )


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
            answer or "",
            definition or "",
            wordplay_type or "",
            ai_explanation or "",
            clue_id,
        ),
    )

    # Mark as manually edited so reverify won't overwrite
    db.execute(
        """UPDATE structured_explanations SET model_version = 'manual_edit'
           WHERE clue_id = ? AND model_version NOT IN ('manual_approve', 'manual_edit')""",
        (clue_id,),
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
                sol, grid_rows, grid_cols, _ = result
                update_puzzle_grid_solution(source, puzzle_number, sol, grid_rows, grid_cols)

    # Return refreshed button row with updated tier badge
    from web.models import get_clue_by_id, compute_hint_tier, get_hint_steps, compute_solve_source
    from web.routes.hints import generate_token
    clue = get_clue_by_id(clue_id)
    new_tier, _ = compute_hint_tier(clue)
    steps = get_hint_steps(clue, is_admin=True)
    new_token = generate_token(clue_id)
    solve_source = compute_solve_source(clue)
    response_html = render_template(
        "partials/admin_rerun_result.html",
        clue=clue, tier=new_tier, steps=steps,
        token=new_token, solve_source=solve_source,
    )

    # If the answer was just cleared, also remove it from solve-mode localStorage
    # so it can't be redisplayed in solve mode or pushed back to the DB by
    # the "Save all to DB" button.
    if old_answer and not answer:
        response_html += (
            '<script>(function(){try{'
            f'var k="solve_{clue["source"]}_{clue["puzzle_number"]}";'
            'var s=JSON.parse(localStorage.getItem(k)||"{}");'
            f'delete s["{clue_id}"];'
            'localStorage.setItem(k,JSON.stringify(s));'
            '}catch(e){}})();</script>'
        )

    return response_html


@bp.route("/reverify-clue/<int:clue_id>", methods=["POST"])
def reverify_clue(clue_id):
    """Re-verify a single clue's existing parse with current DB state and
    manual word-role assignments. Does NOT touch the stored parse text —
    only updates structured_explanations.confidence based on the
    verifier's verdict.
    """
    _require_admin()
    from sonnet_pipeline.verify_explanation import ExplanationVerifier
    db = get_admin_db()
    clue = db.execute(
        """SELECT c.id, c.clue_text, c.answer, c.definition,
                  c.wordplay_type, c.ai_explanation,
                  se.model_version
           FROM clues c
           LEFT JOIN structured_explanations se ON se.clue_id = c.id
           WHERE c.id = ?""",
        (clue_id,),
    ).fetchone()
    if clue is None:
        abort(404)
    wfw_reverify_error = None
    try:
        _write_manual_role_stage_three_for_clue(db, clue_id)
        db.commit()
    except Exception as exc:
        wfw_reverify_error = str(exc)
    if not clue["ai_explanation"]:
        extra = (
            ' <span class="text-xs text-rose-700">WFW reverify failed: %s</span>'
            % wfw_reverify_error
            if wfw_reverify_error else ""
        )
        return ('<span class="text-xs text-amber-700">No parse to verify</span>'
                + extra)
    # manual_edit / manual_approve still re-verifies: the explanation
    # itself is never modified, only confidence is updated based on
    # current DB state and manual word-role overrides. Protection
    # applies to destructive paths (Re-run) — not to scoring.
    verifier = ExplanationVerifier()
    result = verifier.verify(
        clue["clue_text"], clue["answer"], clue["definition"],
        clue["wordplay_type"], clue["ai_explanation"],
        clue_id=clue_id, db_conn=db,
    )
    score = result.get("score", 0)
    verdict = result.get("verdict", "FAIL")
    confidence = min(score / 100.0, 0.6)
    existing = db.execute(
        "SELECT 1 FROM structured_explanations WHERE clue_id = ?",
        (clue_id,)).fetchone()
    if existing:
        db.execute(
            "UPDATE structured_explanations SET confidence = ? "
            "WHERE clue_id = ?", (confidence, clue_id))
    else:
        db.execute(
            "INSERT INTO structured_explanations "
            "(clue_id, confidence, model_version) "
            "VALUES (?, ?, 'reverified')", (clue_id, confidence))
    db.commit()
    tier_colour = {"HIGH": "emerald", "MEDIUM": "amber",
                   "LOW": "orange", "FAIL": "rose"}.get(verdict, "slate")
    # Build a compact list of failing checks so the admin can see why
    # without having to ask. Passing checks are omitted to keep it brief.
    failed_lines = []
    if wfw_reverify_error:
        failed_lines.append(
            '<li class="text-rose-700">WFW reverify failed: %s</li>'
            % wfw_reverify_error
        )
    for ch in result.get("checks", []):
        if ch.get("status") not in ("verified", "skipped"):
            detail = ch.get("detail", "")
            failed_lines.append(
                f'<li class="text-rose-700">✗ {detail}</li>'
            )
    failed_html = (
        f'<ul class="mt-1 text-xs list-none space-y-0.5">{"".join(failed_lines)}</ul>'
        if failed_lines else ""
    )
    return (f'<span class="text-xs font-semibold text-{tier_colour}-700">'
            f'{verdict} {score}</span>'
            f'{failed_html}')


@bp.route("/rerun/<int:clue_id>", methods=["POST"])
def rerun_clue(clue_id):
    """Re-run a clue through the pipeline and return result as HTMX fragment.

    ?mechanical=1 — mechanical solvers only (signature + V1), zero API cost.
    Without parameter — full pipeline including Sonnet fallback.
    """
    from flask import request as _req
    mechanical_only = _req.args.get("mechanical") == "1"
    force = _req.args.get("force") == "1"
    try:
        return _rerun_clue_inner(clue_id, mechanical_only=mechanical_only, force=force)
    except Exception as e:
        import traceback
        print(f"[RERUN OUTER] {e}")
        traceback.print_exc()
        return _with_hx_refresh(
            '<div class="mt-2 text-xs text-red-600 bg-red-50 rounded px-2 py-1">Error: %s</div>' % str(e))

def _rerun_clue_inner(clue_id, mechanical_only=False, force=False):
    _require_admin()

    db = get_admin_db()
    clue = db.execute("SELECT * FROM clues WHERE id = ?", (clue_id,)).fetchone()
    if clue is None:
        abort(404)

    source = clue["source"]
    puzzle_number = clue["puzzle_number"]
    answer = clue["answer"]
    clue_text = clue["clue_text"]

    # Protect manually reviewed clues unless force is set
    if clue["reviewed"] == 1 and not force:
        return _with_hx_refresh(
            '<div class="mt-2 text-xs text-blue-700 bg-blue-50 border border-blue-200 rounded px-2 py-1">Manually reviewed — use Force Re-run to override.</div>')

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
    db.execute(
        "DELETE FROM wfw_proof_attempts WHERE clue_id = ?",
        (clue_id,),
    )
    db.commit()
    print(f"[RERUN] Cleared clue {clue_id} ({clue_text[:40]}), mechanical_only={mechanical_only}")

    success = False
    message = ""
    evidence_artifact_id = None
    signature_pipeline_result = None
    sr = None
    unified_pipeline_ran = False
    import re as _re
    import json as _json
    from flask import current_app
    from sonnet_pipeline.verify_explanation import ExplanationVerifier

    if answer and clue_text:
        answer_clean = _re.sub(r'[^A-Za-z]', '', answer).upper()
        from sonnet_pipeline.word_roles_store import get_roles as _get_roles
        manual_roles = [
            row for row in _get_roles(clue_id, conn=db)
            if row[3] == "manual"
        ]

    # Phase 0: WFW unified solver. DB facts should feed WFW first, and a
    # successful WFW solve must be persisted as a WFW proof for the clue page.
    if not success and not unified_pipeline_ran and answer and clue_text:
        try:
            from signature_solver.db import RefDB
            from sonnet_pipeline.clue_pipeline import run_clue_pipeline

            # Use a fresh DB view here. Admin rerun is explicitly a refresh
            # action, so it should not depend on a long-lived Flask cache.
            unified_pipeline_ran = True
            signature_pipeline_result = run_clue_pipeline(
                db, clue_id, source, puzzle_number, clue["clue_number"],
                clue["direction"], clue_text, answer_clean,
                clue["enumeration"], clue["ai_explanation"], RefDB(),
                dd_graph=current_app.get_shared_dd_graph(),
                manual_roles=manual_roles, write_db=True,
                store_solution=True,
                solver_version="admin_rerun_signature:v3")
            sr = signature_pipeline_result.solve_result
            evidence_artifact_id = signature_pipeline_result.evidence_ids.get(
                "atomic_artifact_id")
            db.commit()
            if signature_pipeline_result.solved:
                success = True
            elif (sr and sr.high_confidence
                    and getattr(sr, "solver_authority", None) == "wfw_unified"):
                if not _store_wfw_rerun_result(
                        db, clue_id, clue, sr, answer_clean):
                    raise RuntimeError("WFW solve did not produce a proven proof")
                db.commit()
                success = True
        except Exception as e:
            import traceback
            print(f"[RERUN WFW] Error: {e}")
            traceback.print_exc()
            return (
                '<div class="mt-2 text-xs text-red-600 bg-red-50 rounded '
                'px-2 py-1">Unified pipeline error: %s</div>' % str(e)
            )

    # ── MECHANICAL SOLVERS ──────────────────────────────────────────
    # Priority order: V1 solvers (best explanations) > Hidden/DD > Signature solver
    # V1 produces clear, user-facing explanations with source words shown.
    # Signature solver is the fallback — its explanations are opaque.

    if signature_pipeline_result is not None:
        try:
            _write_manual_role_stage_three_for_clue(db, clue_id)
            db.commit()
        except Exception as e:
            import traceback
            print(f"[RERUN WFW MANUAL ROLES] Error: {e}")
            traceback.print_exc()

    # Phase 1: V1 mechanical solvers (best explanations, zero API cost)
    if not success and not unified_pipeline_ran and answer and clue_text:
        try:
            from signature_solver.db import RefDB
            from backfill_ai_exp.batch_v1_solver import (
                find_definition as v1_find_def,
                solve_without_definition as v1_solve_no_def,
                try_anagram as v1_anagram, try_charade as v1_charade,
                try_container as v1_container, try_deletion as v1_deletion,
                try_reversal as v1_reversal,
                try_acrostic as v1_acrostic, try_homophone as v1_homophone,
                build_explanation_text as v1_build_expl,
            )

            ref_db = RefDB()
            definition, remaining = v1_find_def(clue_text, answer_clean, ref_db)

            mech_result = None
            mech_wtype = None
            mech_pieces = None

            if definition and remaining:
                for try_fn, wtype, piece_key in [
                    (lambda: v1_anagram(clue_text, answer_clean, ref_db,
                                        definition_words=definition.split()), "anagram", "fodder_words"),
                    (lambda: v1_container(remaining, answer_clean, ref_db), "container", "pieces"),
                    (lambda: v1_deletion(remaining, answer_clean, ref_db), "deletion", "pieces"),
                    (lambda: v1_charade(remaining, answer_clean, ref_db), "charade", "pieces"),
                    (lambda: v1_reversal(remaining, answer_clean, ref_db), "reversal", "pieces"),
                    (lambda: v1_acrostic(remaining, answer_clean, ref_db), "acrostic", "pieces"),
                    (lambda: v1_homophone(remaining, answer_clean, ref_db), "homophone", "pieces"),
                ]:
                    r = try_fn()
                    if r:
                        mech_wtype = wtype
                        if wtype == "anagram":
                            mech_pieces = [{"clue_word": w, "letters": _re.sub(r'[^A-Za-z]', '', w).upper(),
                                            "mechanism": "anagram_fodder"} for w in r["fodder_words"]]
                        else:
                            mech_pieces = r["pieces"]
                        mech_result = r
                        break

            # If definition-first failed, try wordplay-first (infer definition)
            if not mech_result and not definition:
                inferred_def, inferred_wtype, inferred_pieces = v1_solve_no_def(
                    clue_text, answer_clean, ref_db)
                if inferred_def and inferred_pieces:
                    definition = inferred_def
                    mech_wtype = inferred_wtype
                    mech_pieces = inferred_pieces
                    mech_result = True

            if mech_result and mech_pieces:
                    print(f"[RERUN V1] Solved as {mech_wtype}: {mech_pieces}")
                    expl_text = v1_build_expl(mech_wtype, mech_pieces, definition, answer)
                    print(f"[RERUN V1] Explanation: {expl_text[:80]}")
                    components = _json.dumps({
                        "ai_pieces": mech_pieces,
                        "assembly": {"op": mech_wtype},
                        "wordplay_type": mech_wtype,
                    })
                    # Gate through verifier — V1 doesn't get automatic 1.0
                    _verifier = ExplanationVerifier()
                    _vresult = _verifier.verify(clue_text, answer, definition, mech_wtype, expl_text, clue_id=clue_id, db_conn=db)
                    _conf_map = {"HIGH": 0.6, "MEDIUM": 0.6, "LOW": 0.3, "FAIL": 0.0}
                    _final_conf = _conf_map.get(_vresult["verdict"], 0.0)
                    db.execute("""
                        INSERT OR REPLACE INTO structured_explanations
                        (clue_id, components, wordplay_types, definition_text, confidence, model_version, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (clue_id, components, _json.dumps([mech_wtype]),
                          definition, _final_conf, "mechanical_v1", source))
                    db.execute("""
                        UPDATE clues SET
                            wordplay_type = ?, definition = ?, ai_explanation = ?, has_solution = 1
                        WHERE id = ?
                    """, (mech_wtype, definition, expl_text, clue_id))
                    db.commit()
                    success = True
        except Exception as e:
            import traceback
            print(f"[RERUN V1] Error: {e}")
            traceback.print_exc()

    # Phase 2: Hidden word + DD check (zero API cost)
    if not success and not unified_pipeline_ran and answer and clue_text:
        try:
            from backfill_ai_exp.backfill_dd_hidden import (
                generate_dd_hypotheses,
                try_hidden,
                norm_letters as dd_norm,
            )

            dd_graph = current_app.get_shared_dd_graph()
            total_len = len(dd_norm(answer))

            # Try hidden word
            hidden_result = try_hidden(clue_text, answer_clean, dd_graph, total_len)
            if hidden_result:
                op = "hidden_reversed" if hidden_result["direction"] == "reverse" else "hidden"
                hiding_words = hidden_result.get("words", "")
                hidden_def = hidden_result.get("definition")
                pieces = [{"clue_word": hiding_words, "letters": answer_clean, "mechanism": "hidden"}]
                components = _json.dumps({
                    "ai_pieces": pieces,
                    "assembly": {"op": op},
                    "wordplay_type": op,
                })
                db.execute("""
                    INSERT OR REPLACE INTO structured_explanations
                    (clue_id, components, wordplay_types, definition_text, confidence, model_version, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (clue_id, components, _json.dumps([op]), hidden_def, 0.6, "mechanical_hidden", source))
                from sonnet_pipeline.report import _highlight_hidden
                highlighted = _highlight_hidden(hiding_words, answer_clean[::-1] if op == "hidden_reversed" else answer_clean)
                expl = 'hidden in "%s"' % highlighted
                db.execute("""
                    UPDATE clues SET wordplay_type = ?, definition = ?, ai_explanation = ?, has_solution = 1
                    WHERE id = ?
                """, (op, hidden_def, expl, clue_id))
                db.commit()
                success = True

            # Try DD
            if not success:
                dd_result = generate_dd_hypotheses(clue_text, dd_graph, total_len=total_len, answer=dd_norm(answer))
                if dd_result:
                    components = _json.dumps({
                        "ai_pieces": [],
                        "assembly": {"op": "double_definition", "left_def": dd_result["left_def"], "right_def": dd_result["right_def"]},
                        "wordplay_type": "double_definition",
                    })
                    db.execute("""
                        INSERT OR REPLACE INTO structured_explanations
                        (clue_id, components, wordplay_types, definition_text, confidence, model_version, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (clue_id, components, _json.dumps(["double_definition"]),
                          "Double definition", 0.6, "mechanical_dd", source))
                    db.execute("""
                        UPDATE clues SET wordplay_type = 'double_definition', definition = 'Double definition',
                        ai_explanation = 'Double definition', has_solution = 1
                        WHERE id = ?
                    """, (clue_id,))
                    db.commit()
                    success = True
        except Exception as e:
            import traceback
            print(f"[RERUN DD/HIDDEN] Error: {e}")
            traceback.print_exc()

    # Phase 3: Signature solver (fallback — opaque explanations)
    if not success and not unified_pipeline_ran and answer and clue_text:
        try:
            from sonnet_pipeline.sig_adapter import store_signature_result

            if sr is None:
                from signature_solver.db import RefDB
                from sonnet_pipeline.clue_pipeline import (
                    run_clue_pipeline,
                )
                signature_pipeline_result = run_clue_pipeline(
                    db, clue_id, source, puzzle_number, clue["clue_number"],
                    clue["direction"], clue_text, answer_clean,
                    clue["enumeration"], clue["ai_explanation"], RefDB(),
                    dd_graph=current_app.get_shared_dd_graph(),
                    manual_roles=manual_roles,
                    write_db=True, store_solution=False,
                    solver_version="admin_rerun_signature:v3")
                sr = signature_pipeline_result.solve_result
                evidence_artifact_id = (
                    signature_pipeline_result.evidence_ids.get(
                        "atomic_artifact_id")
                )
            if sr and sr.high_confidence:
                store_signature_result(db, clue_id, sr, clue_text, answer_clean)
                success = True
        except Exception as e:
            import traceback
            print(f"[RERUN S] Error: {e}")
            traceback.print_exc()

    if mechanical_only and not success:
        from flask import make_response
        evidence_html = (
            '<span class="text-xs font-semibold text-emerald-700 '
            'bg-emerald-50 border border-emerald-200 rounded px-2 py-1">'
            'Evidence #%s</span> ' % evidence_artifact_id
            if evidence_artifact_id else ""
        )
        return make_response(
            evidence_html
            + '<div class="mt-2 text-xs text-amber-700 bg-amber-50 '
            + 'border border-amber-200 rounded px-2 py-1">'
            + 'Mechanical solvers found no solution. Check DB pieces or '
            + 'try Re-run + Sonnet.</div>')

    # For Guardian/Independent, try fifteensquared (only if S didn't solve)
    if not success and source in ("guardian", "independent") and answer:
        try:
            from sonnet_pipeline.fifteensquared_pipeline import (
                fetch_fifteensquared, store_fifteensquared_result
            )
            from sonnet_pipeline.tftt_pipeline import parse_with_haiku, score_parse
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
                    ref_db = current_app.get_shared_ref_db()
                    parsed, usage = parse_with_haiku(
                        haiku_client, clue_text, answer, fc["explanation"]
                    )
                    if parsed:
                        score, reasons = score_parse(parsed, answer, ref_db)
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
                    ref_db = current_app.get_shared_ref_db()
                    parsed, usage = parse_with_haiku(
                        haiku_client, clue_text, answer, tc["explanation"]
                    )
                    if parsed:
                        score, reasons = score_parse(parsed, answer, ref_db)
                        # Always store — store_tftt_result gates has_solution/reviewed via verifier
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

    # Sonnet explainer deactivated — used for enrichment only, not direct solves.
    # Sonnet explanations have too many false positives (fabricated pieces,
    # wrong indicators). To reactivate, uncomment below.
    # if not success:
    #     from web.explainer import generate_explanation
    #     try:
    #         success, message, result = generate_explanation(clue_id)
    #     except Exception as e:
    #         import traceback, sys
    #         traceback.print_exc()
    #         sys.stderr.flush()
    #         sys.stdout.flush()
    #         return '<div class="mt-2 text-xs text-red-600 bg-red-50 rounded px-2 py-1">Error: %s<br><pre>%s</pre></div>' % (str(e), traceback.format_exc())
    #
    #     if not success:
    #         return '<div class="mt-2 text-xs text-red-600 bg-red-50 rounded px-2 py-1">Failed: %s</div>' % message

    # Queue unverified pieces for DB+ review
    try:
        from sonnet_pipeline.verify_explanation import ExplanationVerifier
        import sqlite3 as _sqlite3

        fresh_clue = db.execute("SELECT clue_text, answer, definition, wordplay_type, ai_explanation FROM clues WHERE id = ?", (clue_id,)).fetchone()
        if fresh_clue and fresh_clue["ai_explanation"]:
            _verifier = ExplanationVerifier()
            vresult = _verifier.verify(
                fresh_clue["clue_text"], fresh_clue["answer"],
                fresh_clue["definition"], fresh_clue["wordplay_type"],
                fresh_clue["ai_explanation"],
                clue_id=clue_id,
                db_conn=db,
            )
            _ref = _sqlite3.connect(str(PROJECT_ROOT / "data" / "cryptic_new.db"), timeout=10)
            for check in vresult.get("checks", []):
                if check["status"] == "unverifiable" and check["check"] in ("synonym", "abbreviation"):
                    m = re.match(r"'(.+?)'\s*(?:=|->)\s*(\w+)", check["detail"])
                    if m:
                        word = m.group(1).strip().lower()
                        letters = m.group(2).strip().upper()
                        gtype = check["check"]
                        already = False
                        if gtype == "synonym":
                            already = _ref.execute("SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(synonym)=?", (word, letters)).fetchone() is not None
                        elif gtype == "abbreviation":
                            already = _ref.execute("SELECT 1 FROM wordplay WHERE LOWER(indicator)=? AND UPPER(substitution)=?", (word, letters)).fetchone() is not None
                        elif gtype == "definition":
                            already = _ref.execute("SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=? AND UPPER(answer)=?", (word, letters)).fetchone() is not None
                        if not already:
                            rejected = db.execute("SELECT 1 FROM rejected_enrichments WHERE type=? AND LOWER(word)=? AND UPPER(letters)=?", (gtype, word, letters)).fetchone()
                            if rejected:
                                continue
                            existing_pending = db.execute("SELECT 1 FROM pending_enrichments WHERE LOWER(word)=? AND UPPER(letters)=?", (word, letters)).fetchone()
                            if not existing_pending:
                                db.execute(
                                    "INSERT INTO pending_enrichments (type, word, letters, answer, clue_text, source, puzzle_number) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (gtype, word, letters, fresh_clue["answer"], fresh_clue["clue_text"], source, puzzle_number))
            _ref.close()
            db.commit()
    except Exception:
        pass  # Don't let gap collection failure block the re-run result

    # If nothing solved but we found a definition, store it anyway
    if not success and not unified_pipeline_ran and answer and clue_text:
        try:
            from signature_solver.solver import extract_definition_candidates, _normalize_clue
            ref_db = current_app.get_shared_ref_db()
            clue_words = _normalize_clue(clue_text).strip().split()
            candidates = extract_definition_candidates(clue_words, answer_clean, ref_db)
            if not candidates:
                from signature_solver.haiku_definition import find_definition as haiku_find_def
                haiku_result = haiku_find_def(clue_text, answer)
                if haiku_result:
                    candidates = [haiku_result]
            if candidates:
                found_def = candidates[0][0]
                db.execute("UPDATE clues SET definition = ? WHERE id = ? AND definition IS NULL",
                           (found_def, clue_id))
                db.commit()
        except Exception:
            pass

    # Return full button row matching the puzzle page layout
    try:
        from web.models import get_clue_by_id, compute_hint_tier, get_hint_steps, compute_solve_source
        from web.routes.hints import generate_token
        clue = get_clue_by_id(clue_id)
        new_tier, _ = compute_hint_tier(clue)
        steps = get_hint_steps(clue, tier=new_tier, is_admin=True)
        new_token = generate_token(clue_id)
        solve_source = compute_solve_source(clue)
        from flask import make_response
        response = make_response(render_template(
            "partials/admin_rerun_result.html",
            clue=clue, tier=new_tier, steps=steps,
            token=new_token, solve_source=solve_source,
            evidence_artifact_id=evidence_artifact_id,
        ))
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        return _with_hx_refresh(
            '<div class="mt-2 text-xs text-red-600 bg-red-50 rounded px-2 py-1">Render error: %s</div>' % str(e))


def _with_hx_refresh(html):
    from flask import make_response
    response = make_response(html)
    response.headers["HX-Refresh"] = "true"
    return response


@bp.route("/approve/<int:clue_id>", methods=["POST"])
def approve_clue(clue_id):
    """Mark a clue as approved at a given tier (reviewed=1, has_solution=1)."""
    _require_admin()

    db = get_admin_db()
    clue = db.execute("SELECT * FROM clues WHERE id = ?", (clue_id,)).fetchone()
    if clue is None:
        abort(404)

    # Tier from request: HIGH (default), MEDIUM, or LOW
    tier = request.form.get("tier", "HIGH").upper()
    confidence_map = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.2}
    confidence = confidence_map.get(tier, 1.0)

    db.execute(
        "UPDATE clues SET reviewed = 1, has_solution = 1 WHERE id = ?",
        (clue_id,),
    )
    # Upsert confidence in structured_explanations
    existing_se = db.execute(
        "SELECT id FROM structured_explanations WHERE clue_id = ?", (clue_id,)
    ).fetchone()
    if existing_se:
        db.execute(
            "UPDATE structured_explanations SET confidence = ?, model_version = 'manual_approve' WHERE clue_id = ?",
            (confidence, clue_id),
        )
    else:
        db.execute(
            """INSERT INTO structured_explanations
               (clue_id, definition_text, model_version, confidence,
                source, puzzle_number, clue_number)
               VALUES (?, ?, 'manual_approve', ?, ?, ?, ?)""",
            (
                clue_id,
                clue["definition"],
                confidence,
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

    # Patch the in-memory RefDB with the new entry (avoids 12s full reload)
    from flask import current_app
    if hasattr(current_app, 'patch_word_coverage_db'):
        current_app.patch_word_coverage_db(etype, word, value)

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

    # Validate answer length against enumeration
    enum_str = clue["enumeration"] or ""
    nums = re.findall(r"\d+", enum_str)
    if nums:
        expected_len = sum(int(n) for n in nums)
        answer_letters = re.sub(r"[^A-Z]", "", answer)
        if len(answer_letters) != expected_len:
            return f'<span class="text-xs text-red-500">Answer is {len(answer_letters)} letters but enumeration ({enum_str}) needs {expected_len}.</span>'

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
        # Reclassify: "abbreviation" with 3+ letter result is really a synonym
        import re as _re
        letters_clean = _re.sub(r"[^A-Z]", "", letters)
        if mechanism == "synonym":
            etype = "synonym"
        elif mechanism == "abbreviation":
            etype = "abbreviation" if len(letters_clean) < 3 else "synonym"
        else:
            continue

        # Skip if previously rejected
        rejected = db.execute(
            "SELECT 1 FROM rejected_enrichments WHERE type=? AND LOWER(word)=? AND UPPER(letters)=?",
            (etype, clue_word.lower(), letters.upper()),
        ).fetchone()
        if rejected:
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
    skipped = 0
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
            # Validate length against enumeration
            enum_str = ci.get("enumeration") or ""
            enum_nums = re.findall(r"\d+", enum_str)
            if enum_nums:
                expected = sum(int(n) for n in enum_nums)
                if len(answer.replace(" ", "")) != expected:
                    skipped += 1
                    continue
            db.execute("UPDATE clues SET answer = ? WHERE id = ?", (answer, clue_id))
            saved += 1

    db.commit()
    return _json.dumps({"saved": saved, "skipped": skipped}), 200, {"Content-Type": "application/json"}


@bp.route("/silly/<int:clue_id>", methods=["POST"])
def toggle_silly(clue_id):
    """Toggle Cordelia's Silly Award on a clue."""
    _require_admin()
    db = get_admin_db()
    current = db.execute("SELECT silly_award FROM clues WHERE id = ?", (clue_id,)).fetchone()
    if not current:
        return "Clue not found", 404
    new_val = 0 if current["silly_award"] else 1
    db.execute("UPDATE clues SET silly_award = ? WHERE id = ?", (new_val, clue_id))
    db.commit()
    if new_val:
        return (
            '<span id="silly-%d" hx-post="/admin/silly/%d" hx-target="#silly-%d" hx-swap="outerHTML">'
            '<span class="text-lg cursor-pointer" title="Cordelia\'s Silly Award — what were they thinking?!">&#127942;</span>'
            '</span>' % (clue_id, clue_id, clue_id)
        )
    return (
        '<span id="silly-%d" hx-post="/admin/silly/%d" hx-target="#silly-%d" hx-swap="outerHTML">'
        '<span class="text-lg cursor-pointer opacity-20 hover:opacity-100" title="Award Cordelia\'s Silly Award">&#127942;</span>'
        '</span>' % (clue_id, clue_id, clue_id)
    )


def _build_manual_definition_candidates(answer, manual_roles, ref_db):
    """Build definition candidates from consecutive definition manual roles.

    For each consecutive run of words manually marked 'definition', check
    whether RefDB contains that phrase -> answer. Return candidates with
    boundary_status 'manual_definition' for DB hits or 'manual_definition_gap'
    for missing DB facts.
    """
    def_entries = [
        r for r in manual_roles if r.get("role") == "definition"
    ]
    if not def_entries:
        return []

    runs = []
    current = [def_entries[0]]
    for entry in def_entries[1:]:
        if entry["index"] == current[-1]["index"] + 1:
            current.append(entry)
        else:
            runs.append(current)
            current = [entry]
    runs.append(current)

    candidates = []
    for run in runs:
        span = [run[0]["index"], run[-1]["index"] + 1]
        text = " ".join(e["text"] for e in run)
        db_hit = ref_db.is_definition_of(text, answer)
        candidates.append({
            "boundary_status": (
                "manual_definition" if db_hit else "manual_definition_gap"
            ),
            "objections": [],
            "span": span,
            "text": text,
            "missing_answer": None if db_hit else answer,
        })
    return candidates


def _manual_roles_for_clue(db, clue_id):
    rows = db.execute(
        "SELECT word_index, word_text, role, letters "
        "FROM clue_word_roles WHERE clue_id = ? ORDER BY word_index",
        (clue_id,),
    ).fetchall()
    return [
        {
            "index": row["word_index"],
            "text": row["word_text"],
            "role": row["role"],
            "letters": row["letters"],
        }
        for row in rows
    ]


def _casefile_from_stage_two_json(stage_two, clue):
    from types import SimpleNamespace

    return SimpleNamespace(
        clue_text=stage_two.get("clue_text") or clue["clue_text"],
        answer=stage_two.get("answer") or clue["answer"],
        definition_candidates=tuple(
            stage_two.get("definition_candidates") or ()),
        grammar_phrases=tuple(stage_two.get("grammar_phrases") or ()),
        source_candidates=tuple(stage_two.get("source_candidates") or ()),
        operation_candidates=tuple(
            stage_two.get("operation_candidates") or ()),
        working_pairs=tuple(stage_two.get("working_pairs") or ()),
        assemblies=tuple(stage_two.get("assemblies") or ()),
        enrichment_candidates=tuple(
            stage_two.get("enrichment_candidates") or ()),
        unresolved_words=tuple(stage_two.get("unresolved_words") or ()),
        status=stage_two.get("status") or "unknown",
    )


def _write_manual_role_stage_three_for_clue(db, clue_id):
    """Write one manual-role-aware Stage Three proof from retained Stage Two."""
    import json

    from signature_solver.stage_three_proof import build_stage_three_proof
    from signature_solver.wfw_proof_store import write_wfw_proof_attempt

    row = db.execute(
        """SELECT c.id, c.source, c.puzzle_number, c.clue_text, c.answer,
                  cps.stage_two_json
           FROM clues c
           LEFT JOIN clue_pipeline_state cps ON cps.clue_id = c.id
           WHERE c.id = ?""",
        (clue_id,),
    ).fetchone()
    if row is None:
        return {"status": "missing_clue"}
    if not row["stage_two_json"]:
        return {"status": "missing_stage_two"}

    stage_two = json.loads(row["stage_two_json"])
    casefile = _casefile_from_stage_two_json(stage_two, row)
    casefile.manual_roles = _manual_roles_for_clue(db, clue_id)
    ref_db = current_app.get_shared_ref_db()
    casefile.manual_definition_candidates = (
        _build_manual_definition_candidates(
            casefile.answer, casefile.manual_roles, ref_db)
    )

    stage_three_proof = build_stage_three_proof(casefile).as_dict()
    proof = dict(stage_three_proof)
    proof["status"] = (
        "wfw_proven" if stage_three_proof.get("status") == "PASS"
        else "wfw_review"
    )
    proof["source"] = "stage_three_manual_roles"
    proof_id = write_wfw_proof_attempt(
        clue_id, row["source"], row["puzzle_number"], proof, conn=db)
    db.execute(
        """UPDATE clue_pipeline_state
           SET stage_three_json = ?, updated_at = CURRENT_TIMESTAMP
           WHERE clue_id = ?""",
        (json.dumps(stage_three_proof, sort_keys=True), clue_id),
    )
    return {
        "status": proof["status"],
        "proof_id": proof_id,
        "stage_three_status": stage_three_proof.get("status"),
    }


@bp.route("/reverify/<source>/<int:puzzle_number>", methods=["POST"])
@bp.route("/atomic-reverify/<source>/<int:puzzle_number>", methods=["POST"])
def atomic_reverify_puzzle(source, puzzle_number):
    """Re-run Stage Three over retained puzzle enrichment.

    This deliberately does not solve clues again.  It reads the current
    Stage Two evidence package from clue_pipeline_state, applies the current
    Stage Three proof gate, and records a fresh wfw_proof_attempt for display.
    """
    _require_admin()
    db = get_admin_db()
    rows = db.execute(
        """SELECT c.id, c.clue_text, c.answer, cps.stage_two_json
           FROM clues c
           LEFT JOIN clue_pipeline_state cps ON cps.clue_id = c.id
           WHERE c.source = ? AND c.puzzle_number = ?
           ORDER BY c.direction, CAST(c.clue_number AS INTEGER), c.clue_number""",
        (source, str(puzzle_number)),
    ).fetchall()
    if not rows:
        return '<div class="bg-teal-50 border border-teal-200 rounded p-3 text-teal-800 text-sm">No clues found for this puzzle.</div>'

    proven = 0
    review = 0
    missing = 0
    errors = []
    for row in rows:
        if not row["stage_two_json"]:
            missing += 1
            continue
        try:
            result = _write_manual_role_stage_three_for_clue(db, row["id"])
            if result.get("status") == "wfw_proven":
                proven += 1
            else:
                review += 1
        except Exception as exc:
            errors.append("%s: %s" % (row["answer"], exc))
    db.commit()

    body = (
        '<strong>WFW reverified from retained enrichment</strong>: '
        f'{proven} proven, {review} review'
    )
    if missing:
        body += f', {missing} missing retained Stage Two evidence'
    if errors:
        body += '<br><strong>Errors:</strong> ' + _html_escape(
            '; '.join(errors[:5]))
    return (
        f'<div class="bg-teal-50 border border-teal-200 rounded p-3 text-teal-800 text-sm">'
        f'{body}</div>'
        f'<script>setTimeout(function(){{ window.location.reload(); }}, 2000);</script>'
    )


@bp.route("/legacy-reverify/<source>/<int:puzzle_number>", methods=["POST"])
def reverify_puzzle(source, puzzle_number):
    """Re-run the mechanical verifier on all clues in a puzzle. Zero API cost.

    Re-scores existing explanations using current RefDB data.
    Does NOT solve, clear, or replace any existing explanations.
    """
    _require_admin()

    from sonnet_pipeline.verify_explanation import ExplanationVerifier

    db = get_admin_db()
    clues = db.execute(
        """SELECT c.id, c.clue_text, c.answer, c.definition, c.wordplay_type,
                  c.ai_explanation, se.confidence AS old_confidence,
                  se.model_version AS se_model
           FROM clues c
           LEFT JOIN structured_explanations se ON se.clue_id = c.id
           WHERE c.source = ? AND c.puzzle_number = ?
           AND c.ai_explanation IS NOT NULL AND c.ai_explanation != ''""",
        (source, str(puzzle_number)),
    ).fetchall()

    if not clues:
        return '<div class="bg-teal-50 border border-teal-200 rounded p-3 text-teal-800 text-sm">No clues with explanations to verify.</div>'

    verifier = ExplanationVerifier()
    upgraded = 0
    downgraded = 0
    unchanged = 0
    gaps_queued = 0
    upgraded_list = []
    downgraded_list = []

    import sqlite3 as _sqlite3
    ref_conn = _sqlite3.connect(str(PROJECT_ROOT / "data" / "cryptic_new.db"), timeout=10)

    # manual_approve is a hard "trust me, the score is fixed" tag and
    # stays protected from auto re-score. manual_edit only protects the
    # explanation TEXT from auto-solver overwrite — re-verifying still
    # makes sense since the score reflects current DB state and any
    # manual word-role overrides.
    MANUAL_MODELS = ("manual_approve",)

    for clue in clues:
        # Never re-score manually approved clues
        if clue["se_model"] in MANUAL_MODELS:
            unchanged += 1
            continue

        # Pass clue_id so word roles get persisted to clue_word_roles
        # and any manual overrides are honoured in word_coverage.
        result = verifier.verify(
            clue["clue_text"], clue["answer"],
            clue["definition"], clue["wordplay_type"],
            clue["ai_explanation"],
            clue_id=clue["id"],
            db_conn=db,
        )

        # Queue unverified pieces for DB+ review
        for check in result.get("checks", []):
            if check["status"] == "unverifiable" and check["check"] in ("synonym", "abbreviation"):
                import re as _re
                m = _re.match(r"'(.+?)'\s*(?:=|->)\s*(\w+)", check["detail"])
                if m:
                    word = m.group(1).strip().lower()
                    letters = m.group(2).strip().upper()
                    gtype = check["check"]
                    already = False
                    if gtype == "synonym":
                        already = ref_conn.execute(
                            "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(synonym)=?",
                            (word, letters)).fetchone() is not None
                    elif gtype == "abbreviation":
                        already = ref_conn.execute(
                            "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? AND UPPER(substitution)=?",
                            (word, letters)).fetchone() is not None
                    elif gtype == "definition":
                        already = ref_conn.execute(
                            "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=? AND UPPER(answer)=?",
                            (word, letters)).fetchone() is not None
                    if not already:
                        rejected = db.execute(
                            "SELECT 1 FROM rejected_enrichments WHERE type=? AND LOWER(word)=? AND UPPER(letters)=?",
                            (gtype, word.lower(), letters.upper())).fetchone()
                        if rejected:
                            continue
                        existing_pending = db.execute(
                            "SELECT 1 FROM pending_enrichments WHERE LOWER(word)=? AND UPPER(letters)=?",
                            (word, letters)).fetchone()
                        if not existing_pending:
                            db.execute(
                                "INSERT INTO pending_enrichments (type, word, letters, answer, clue_text, source, puzzle_number) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (gtype, word, letters, clue["answer"], clue["clue_text"], source, str(puzzle_number)))
                            gaps_queued += 1

        new_confidence = min(result["score"] / 100.0, 0.6)
        old_confidence = clue["old_confidence"]

        # Update or insert structured_explanations
        existing = db.execute(
            "SELECT 1 FROM structured_explanations WHERE clue_id = ?",
            (clue["id"],),
        ).fetchone()

        if existing:
            db.execute(
                "UPDATE structured_explanations SET confidence = ? WHERE clue_id = ?",
                (new_confidence, clue["id"]),
            )
        else:
            db.execute(
                """INSERT INTO structured_explanations (clue_id, confidence, model_version)
                   VALUES (?, ?, 'reverified')""",
                (clue["id"], new_confidence),
            )

        # Also mark as solved
        db.execute(
            "UPDATE clues SET has_solution = 1 WHERE id = ? AND (has_solution IS NULL OR has_solution = 0)",
            (clue["id"],),
        )

        if old_confidence is not None:
            old_score = round(old_confidence * 100) if old_confidence <= 1 else old_confidence
            if result["score"] > old_score:
                upgraded += 1
                upgraded_list.append(f'{clue["answer"]} {int(old_score)}->{result["score"]}')
            elif result["score"] < old_score:
                downgraded += 1
                downgraded_list.append(f'{clue["answer"]} {int(old_score)}->{result["score"]}')
            else:
                unchanged += 1
        else:
            upgraded += 1
            upgraded_list.append(f'{clue["answer"]} NEW->{result["score"]}')

    ref_conn.close()
    db.commit()

    total = len(clues)
    lines = []
    lines.append(f'<strong>{upgraded} upgraded</strong>, {unchanged} unchanged, {downgraded} downgraded out of {total} clues')
    if gaps_queued:
        lines.append(f'{gaps_queued} DB entries queued for review')
    if upgraded_list:
        lines.append('<strong>Upgraded:</strong> ' + ', '.join(upgraded_list))
    if downgraded_list:
        lines.append('<strong>Downgraded:</strong> ' + ', '.join(downgraded_list))
    body = '<br>'.join(lines)
    return (
        f'<div class="bg-teal-50 border border-teal-200 rounded p-3 text-teal-800 text-sm">'
        f'{body}</div>'
        f'<script>setTimeout(function(){{ window.location.reload(); }}, 2000);</script>'
    )


def _html_escape(text):
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
