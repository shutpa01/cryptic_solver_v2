"""Data access layer — queries against clues_master.db."""

import json
import re

from flask import current_app

from web.db import get_db


# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------

def clue_slug(clue_text, enumeration=None):
    """Generate a URL slug from clue text + enumeration.

    'Staggers back with beers' (7) -> 'staggers-back-with-beers-7'
    """
    text = clue_text.lower().strip()
    # Replace non-alphanumeric with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    # Append enumeration if present
    if enumeration:
        enum_clean = re.sub(r"[^0-9,/-]+", "", enumeration)
        if enum_clean:
            text += "-" + re.sub(r"[^0-9]", "-", enum_clean).strip("-")
    return text

# ---------------------------------------------------------------------------
# Puzzle-type classification
# ---------------------------------------------------------------------------
# Telegraph prize cryptic = Sunday 3xxx series + Saturday puzzles in 31xxx series.
# Telegraph cryptic = Mon-Fri puzzles in 31xxx series.
# Classification uses publication_date day-of-week for the 31xxx split.

# All browsable source/type combinations
BROWSE_SOURCES = [
    ("telegraph", "cryptic", "Telegraph Cryptic"),
    ("telegraph", "prize", "Telegraph Prize Cryptic"),
    ("times", "cryptic", "Times Cryptic"),
    ("times", "sunday", "Times Sunday"),
    ("guardian", "cryptic", "Guardian Cryptic"),
    ("independent", "cryptic", "Independent Cryptic"),
]

# Label lookup for display
TYPE_LABELS = {
    ("telegraph", "cryptic"): "Cryptic",
    ("telegraph", "prize"): "Prize Cryptic",
    ("times", "cryptic"): "Cryptic",
    ("times", "sunday"): "Sunday",
    ("guardian", "cryptic"): "Cryptic",
    ("independent", "cryptic"): "Cryptic",
}


def _is_valid_type(source, type_slug):
    """Check if a source/type combination is browsable."""
    return (source, type_slug) in TYPE_LABELS


def classify_puzzle(source, puzzle_number, publication_date=None):
    """Return (type_slug, type_label) for a puzzle, or (None, None).

    For Telegraph 31xxx, needs publication_date to distinguish Sat prize from
    Mon-Fri cryptic.  If date not provided, looks it up from the DB.
    """
    try:
        num = int(puzzle_number)
    except (ValueError, TypeError):
        return None, None

    if source == "telegraph":
        if 3000 <= num <= 3999:
            return "prize", "Prize Cryptic"
        if 31000 <= num <= 31999:
            if publication_date is None:
                db = get_db()
                row = db.execute(
                    "SELECT publication_date FROM clues WHERE source = ? AND puzzle_number = ? LIMIT 1",
                    (source, str(puzzle_number)),
                ).fetchone()
                publication_date = row["publication_date"] if row else None
            if publication_date and _is_saturday(publication_date):
                return "prize", "Prize Cryptic"
            return "cryptic", "Cryptic"
        return None, None

    elif source == "times":
        if 5000 <= num <= 9999:
            return "sunday", "Sunday"
        if 26000 <= num <= 39999:
            return "cryptic", "Cryptic"
        return None, None

    elif source == "guardian":
        if 1 <= num <= 39999:
            return "cryptic", "Cryptic"
        return None, None

    elif source == "independent":
        if 1 <= num <= 19999:
            return "cryptic", "Cryptic"
        return None, None

    return None, None


def _is_saturday(date_str):
    """Check if a YYYY-MM-DD date string is a Saturday."""
    from datetime import date
    try:
        d = date.fromisoformat(date_str)
        return d.weekday() == 5  # Monday=0, Saturday=5
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Puzzle list
# ---------------------------------------------------------------------------

def _puzzle_filter_sql(source, type_slug):
    """Return (WHERE clause fragment, params) for filtering puzzles by type.

    Telegraph prize = 3xxx Sunday series + 31xxx Saturdays.
    Telegraph cryptic = 31xxx Mon-Fri only.
    """
    if source == "telegraph" and type_slug == "prize":
        return (
            """(source = ? AND (
                (CAST(puzzle_number AS INTEGER) BETWEEN 3000 AND 3999
                    AND CAST(strftime('%w', publication_date) AS INTEGER) = 0)
                OR (CAST(puzzle_number AS INTEGER) BETWEEN 31000 AND 31999
                    AND CAST(strftime('%w', publication_date) AS INTEGER) = 6)
            ))""",
            [source],
        )
    elif source == "telegraph" and type_slug == "cryptic":
        return (
            """(source = ? AND CAST(puzzle_number AS INTEGER) BETWEEN 31000 AND 31999
                AND CAST(strftime('%w', publication_date) AS INTEGER) != 6)""",
            [source],
        )
    elif source == "times" and type_slug == "sunday":
        return (
            "source = ? AND CAST(puzzle_number AS INTEGER) BETWEEN 5000 AND 9999",
            [source],
        )
    elif source == "times" and type_slug == "cryptic":
        return (
            "source = ? AND CAST(puzzle_number AS INTEGER) BETWEEN 26000 AND 39999",
            [source],
        )
    elif source == "guardian" and type_slug == "cryptic":
        return (
            "source = ? AND puzzle_number IS NOT NULL AND puzzle_number != ''",
            [source],
        )
    elif source == "independent" and type_slug == "cryptic":
        return (
            "source = ? AND puzzle_number IS NOT NULL AND puzzle_number != ''",
            [source],
        )
    return None, None


def get_puzzle_list(source, type_slug, page=1):
    """Return paginated puzzle list with clue counts and coverage stats.

    Returns (puzzles, total_pages) where each puzzle is a dict with:
    puzzle_number, publication_date, clue_count, hints_high, coverage_pct
    """
    db = get_db()
    per_page = current_app.config["PUZZLES_PER_PAGE"]
    where, params = _puzzle_filter_sql(source, type_slug)
    if where is None:
        return [], 0

    # Count total puzzles for pagination
    total = db.execute(
        "SELECT COUNT(DISTINCT puzzle_number) FROM clues WHERE %s" % where,
        params,
    ).fetchone()[0]

    total_pages = max(1, (total + per_page - 1) // per_page)
    offset = (page - 1) * per_page

    rows = db.execute(
        """SELECT puzzle_number,
                  MAX(publication_date) AS publication_date,
                  COUNT(*) AS clue_count,
                  SUM(CASE WHEN definition IS NOT NULL
                            AND wordplay_type IS NOT NULL THEN 1 ELSE 0 END) AS hints_high,
                  SUM(CASE WHEN definition IS NOT NULL THEN 1 ELSE 0 END) AS hints_with_def
           FROM clues
           WHERE %s
           GROUP BY puzzle_number
           ORDER BY MAX(publication_date) DESC
           LIMIT ? OFFSET ?""" % where,
        params + [per_page, offset],
    ).fetchall()

    puzzles = []
    for r in rows:
        clue_count = r["clue_count"]
        high = r["hints_high"] or 0
        coverage_pct = round(100 * high / clue_count) if clue_count else 0
        puzzles.append({
            "puzzle_number": r["puzzle_number"],
            "publication_date": r["publication_date"],
            "clue_count": clue_count,
            "hints_high": high,
            "coverage_pct": coverage_pct,
        })

    return puzzles, total_pages


# ---------------------------------------------------------------------------
# Puzzle clues
# ---------------------------------------------------------------------------

def get_puzzle_clues(source, puzzle_number):
    """Return all clues for a puzzle, ordered by direction then clue number.

    Joins structured_explanations to get component data where available.
    """
    db = get_db()
    rows = db.execute(
        """SELECT c.id, c.clue_number, c.direction, c.clue_text, c.enumeration,
                  c.answer, c.definition, c.wordplay_type, c.explanation, c.ai_explanation,
                  c.silly_award,
                  se.components, se.confidence, se.model_version
           FROM clues c
           LEFT JOIN structured_explanations se ON se.clue_id = c.id
           WHERE c.source = ? AND c.puzzle_number = ?
           ORDER BY
               CASE c.direction WHEN 'across' THEN 0 WHEN 'down' THEN 1 ELSE 2 END,
               CAST(c.clue_number AS INTEGER)""",
        (source, str(puzzle_number)),
    ).fetchall()
    return rows


def get_puzzle_date(source, puzzle_number):
    """Return the publication date for a puzzle, or None."""
    db = get_db()
    row = db.execute(
        "SELECT publication_date FROM clues WHERE source = ? AND puzzle_number = ? LIMIT 1",
        (source, str(puzzle_number)),
    ).fetchone()
    return row["publication_date"] if row else None


def get_puzzle_grid_data(source, puzzle_number):
    """Return lightweight clue data for grid reconstruction.

    Returns list of dicts with clue_number, direction, answer — or empty list.
    """
    db = get_db()
    rows = db.execute(
        """SELECT clue_number, direction, answer FROM clues
           WHERE source = ? AND puzzle_number = ? AND answer IS NOT NULL AND answer != ''""",
        (source, str(puzzle_number)),
    ).fetchall()
    return [dict(r) for r in rows]


def get_puzzle_grid_solution(source, puzzle_number):
    """Return stored grid solution string from puzzle_grids table.

    Returns (solution, grid_rows, grid_cols) or None if not available.
    """
    db = get_db()
    try:
        row = db.execute(
            """SELECT solution, grid_rows, grid_cols FROM puzzle_grids
               WHERE source = ? AND puzzle_number = ?""",
            (source, str(puzzle_number)),
        ).fetchone()
    except Exception:
        # Table may not exist yet
        return None
    if row is None:
        return None
    return row["solution"], row["grid_rows"], row["grid_cols"]


# ---------------------------------------------------------------------------
# Hint tier
# ---------------------------------------------------------------------------

def _has_explanation(clue):
    """Check if a clue has OUR OWN explanation content (components or ai).

    Does NOT include scraped 'explanation' — that's third-party content
    (Big Dave, Fifteen Squared, TftT) which we must not serve.
    """
    for field in ("components", "ai_explanation"):
        val = clue[field] if field in clue.keys() else None
        if val:
            return True
    return False


def compute_hint_tier(clue):
    """Return (tier_name, max_steps) for a clue row.

    Tier is based on the pipeline confidence score (matching puzzle reports):
      HIGH:    confidence >= 80
      MEDIUM:  confidence 40-79
      LOW:     confidence < 40
      FAIL:    pipeline ran but produced no usable result
      PENDING: never processed by any pipeline

    max_steps is based on available hint data (definition, type, explanation).
    """
    # Step count based on what hint data exists
    has_def = bool(clue["definition"])
    has_type = bool(clue["wordplay_type"])
    has_expl = _has_explanation(clue)

    if has_def and has_type and has_expl:
        max_steps = 4
    elif has_def and has_type:
        max_steps = 3
    elif has_def:
        max_steps = 2
    else:
        max_steps = 0

    # Tier based on pipeline confidence score (stored as 0-1 decimal in DB)
    confidence = clue["confidence"] if "confidence" in clue.keys() else None
    if confidence is not None:
        # Normalise to 0-100 scale if stored as decimal
        score = confidence * 100 if confidence <= 1 else confidence
        if score >= 80:
            return "HIGH", max_steps
        elif score >= 40:
            return "MEDIUM", max_steps
        else:
            return "LOW", max_steps
    else:
        # Distinguish "never run" from "ran but no confidence"
        mv = clue["model_version"] if "model_version" in clue.keys() else None
        if mv is not None:
            return "FAIL", max_steps
        return "PENDING", max_steps


def compute_solve_source(clue):
    """Return engine source label: S, SE, P, or fail."""
    mv = clue["model_version"] if "model_version" in clue.keys() else None
    if mv is None:
        return "fail"
    if mv == "signature_solver_v1":
        return "S"
    if mv == "signature_solver_enriched_v1":
        return "SE"
    # Any other model version = P (Sonnet pipeline)
    return "P"


def get_hint_steps(clue, tier=None, is_admin=False):
    """Return ordered list of available hint steps for a clue.

    Each step is a dict: {step: int, label: str, type: str}
    Types: 'definition', 'wordplay_type', 'explanation', 'answer'

    For non-admin users, LOW and FAIL tiers only show definition + answer
    (wordplay type and explanation are hidden to avoid showing bad content).
    """
    steps = []
    n = 1

    # Determine if we should show the full explanation
    show_full = is_admin or tier in ("HIGH", "MEDIUM", "PENDING", None)

    if clue["definition"]:
        steps.append({"step": n, "label": "Definition", "type": "definition"})
        n += 1
    if show_full and clue["wordplay_type"]:
        steps.append({"step": n, "label": "Wordplay type", "type": "wordplay_type"})
        n += 1
    if show_full and _has_explanation(clue):
        steps.append({"step": n, "label": "Explanation", "type": "explanation"})
        n += 1
    # Answer is always available as the final step
    steps.append({"step": n, "label": "Answer", "type": "answer"})
    return steps


def get_clues_by_slug(slug):
    """Find clues matching a slug, most recent first.

    Returns a list of clue rows (with structured_explanations joined).
    Generates slugs on the fly and matches against the requested slug.
    Since SQLite can't do regex slug generation, we search by the words
    in the slug and filter in Python.
    """
    db = get_db()
    # Extract search words from the slug (drop the trailing enumeration digits)
    parts = slug.split("-")
    # Find the content words (non-numeric, or numeric but not at the end)
    search_words = []
    for i, p in enumerate(parts):
        if p and not (p.isdigit() and i >= len(parts) - 2):
            search_words.append(p)

    if not search_words:
        return []

    # Use LIKE for the first few words to narrow candidates
    where_clauses = []
    params = []
    for w in search_words[:3]:
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
             ORDER BY c.publication_date DESC
             LIMIT 50""" % " AND ".join(where_clauses)

    rows = db.execute(sql, params).fetchall()

    # Filter to exact slug match in Python
    matches = []
    for row in rows:
        row_slug = clue_slug(row["clue_text"], row["enumeration"])
        if row_slug == slug:
            matches.append(row)

    return matches


def get_clue_by_id(clue_id):
    """Fetch a full clue row with structured_explanations data, or None."""
    db = get_db()
    row = db.execute(
        """SELECT c.*, se.components, se.confidence, se.model_version
           FROM clues c
           LEFT JOIN structured_explanations se ON se.clue_id = c.id
           WHERE c.id = ?""",
        (clue_id,),
    ).fetchone()
    return row


def get_hint_content(clue, step_type):
    """Return the display content for a given hint step type."""
    if step_type == "definition":
        return clue["definition"]
    elif step_type == "wordplay_type":
        return clue["wordplay_type"]
    elif step_type == "explanation":
        return _build_explanation(clue)
    elif step_type == "answer":
        return clue["answer"]
    return None


def _correct_mechanism(mech, word, letters):
    """Fix mislabelled piece mechanisms based on actual letter positions."""
    import re
    w = re.sub(r"[^A-Za-z]", "", word).upper()
    lt = letters.upper()
    if not w or len(w) < 2:
        return mech
    # "first_letter" producing 2+ chars that are first+last = outer_letters
    if mech == "first_letter" and len(lt) >= 2 and lt == w[0] + w[-1]:
        return "outer_letters"
    # "first_letter" but letters don't start with first char = last_letter check
    if mech == "first_letter" and len(lt) == 1 and lt != w[0] and lt == w[-1]:
        return "last_letter"
    return mech


def _describe_p_piece(p):
    """Describe a single P pipeline piece in the same style as S explanations.

    Each piece has: mechanism, clue_word, letters, and optionally
    indicator, source, deleted, deleted_word.
    """
    import re
    mech = p.get("mechanism", "")
    word = p.get("clue_word", "")
    letters = p.get("letters", "")
    indicator = p.get("indicator", "")
    source = p.get("source", "")
    deleted = p.get("deleted", "")
    deleted_word = p.get("deleted_word", "")

    if not mech or not word or not letters:
        return None

    # Fix mislabelled mechanisms
    mech = _correct_mechanism(mech, word, letters)

    # Mechanism-specific descriptions
    if mech == "synonym":
        return f'{letters} (synonym of "{word}")'
    elif mech == "abbreviation":
        return f'{letters} (abbreviation of "{word}")'
    elif mech == "anagram_fodder":
        return f'"{word}"'
    elif mech == "first_letter":
        if indicator:
            return f'{letters} ("{indicator}" of "{word}" = first letter(s))'
        return f'{letters} (first letter(s) of "{word}")'
    elif mech == "last_letter":
        if indicator:
            return f'{letters} ("{indicator}" of "{word}" = last letter(s))'
        return f'{letters} (last letter(s) of "{word}")'
    elif mech == "outer_letters":
        if indicator:
            return f'{letters} ("{indicator}" of "{word}" = outer letters)'
        return f'{letters} (outer letters of "{word}")'
    elif mech == "inner_letters":
        if indicator:
            return f'{letters} ("{indicator}" of "{word}" = inner letters)'
        return f'{letters} (inner letters of "{word}")'
    elif mech == "alternating":
        if indicator:
            return f'{letters} ("{indicator}" of "{word}" = alternating letters)'
        return f'{letters} (alternating letters of "{word}")'
    elif mech == "reversal":
        if indicator:
            return f'{letters} (reverse ["{indicator}"] of "{word}")'
        return f'{letters} (reverse of "{word}")'
    elif mech == "deletion":
        if source and deleted:
            if deleted_word:
                return f'{letters} ({source} minus {deleted} ["{deleted_word}"])'
            elif indicator:
                return f'{letters} ({source} minus {deleted} ["{indicator}"])'
            return f'{letters} ({source} minus {deleted})'
        if indicator:
            return f'{letters} ("{indicator}" of "{word}")'
        return f'{letters} (from "{word}")'
    elif mech == "homophone":
        if indicator:
            return f'{letters} (sounds like ["{indicator}"] "{word}")'
        return f'{letters} (sounds like "{word}")'
    elif mech == "hidden":
        return f'"{word}"'
    elif mech == "raw":
        return f'{letters} ("{word}")'
    else:
        return f'{letters} ("{word}", {mech})'


def _build_explanation(clue):
    """Build a human-readable explanation from components or raw text."""
    # Manual ai_explanation takes priority (human-curated)
    ai_expl = clue["ai_explanation"] if "ai_explanation" in clue.keys() else None
    if ai_expl:
        return ai_expl

    # Fall back to structured components from pipeline
    comps_json = clue["components"] if "components" in clue.keys() else None
    if comps_json:
        try:
            comps = json.loads(comps_json)
            pieces = comps.get("ai_pieces", [])
            wtype = comps.get("wordplay_type", "")
            if pieces:
                part_strs = []
                for p in pieces:
                    desc = _describe_p_piece(p)
                    if desc:
                        part_strs.append(desc)
                if not part_strs:
                    return None

                # Format based on wordplay type
                if wtype == "anagram":
                    ana_words = [p.get("clue_word", "") for p in pieces
                                 if p.get("mechanism") == "anagram_fodder"]
                    extras = [s for s, p in zip(part_strs, pieces)
                              if p.get("mechanism") != "anagram_fodder"]
                    fodder = " ".join(ana_words) if ana_words else ""
                    # Find indicator from non-piece clue words (not yet available)
                    if fodder:
                        result = f'Anagram of "{fodder}"'
                        if extras:
                            result = f'{" + ".join(extras)} + {result}'
                        answer = clue["answer"] if "answer" in clue.keys() else ""
                        return f'{result} = {answer}'
                elif wtype == "hidden":
                    hid_words = [p.get("clue_word", "") for p in pieces
                                 if p.get("mechanism") == "hidden"]
                    answer = clue["answer"] if "answer" in clue.keys() else ""
                    if hid_words:
                        return f'Hidden in "{" ".join(hid_words)}" = {answer}'

                answer = clue["answer"] if "answer" in clue.keys() else ""
                return f'{" + ".join(part_strs)} = {answer}'
        except (json.JSONDecodeError, TypeError):
            pass
    return None
