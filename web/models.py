"""Data access layer — queries against clues_master.db."""

import json

from flask import current_app

from web.db import get_db

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
]

# Label lookup for display
TYPE_LABELS = {
    ("telegraph", "cryptic"): "Cryptic",
    ("telegraph", "prize"): "Prize Cryptic",
    ("times", "cryptic"): "Cryptic",
    ("times", "sunday"): "Sunday",
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
                  c.definition, c.wordplay_type, c.explanation, c.ai_explanation,
                  se.components, se.confidence
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

    HIGH (4 steps): definition + wordplay_type + explanation content
    MEDIUM (3): definition + wordplay_type
    LOW (2): definition only
    NONE (0): nothing useful
    """
    has_def = bool(clue["definition"])
    has_type = bool(clue["wordplay_type"])
    has_expl = _has_explanation(clue)

    if has_def and has_type and has_expl:
        return "HIGH", 4
    elif has_def and has_type:
        return "MEDIUM", 3
    elif has_def:
        return "LOW", 2
    else:
        return "NONE", 0


def get_hint_steps(clue):
    """Return ordered list of available hint steps for a clue.

    Each step is a dict: {step: int, label: str, type: str}
    Types: 'definition', 'wordplay_type', 'explanation', 'answer'
    """
    steps = []
    n = 1
    if clue["definition"]:
        steps.append({"step": n, "label": "Definition", "type": "definition"})
        n += 1
    if clue["wordplay_type"]:
        steps.append({"step": n, "label": "Wordplay type", "type": "wordplay_type"})
        n += 1
    if _has_explanation(clue):
        steps.append({"step": n, "label": "Explanation", "type": "explanation"})
        n += 1
    # Answer is always available as the final step
    steps.append({"step": n, "label": "Answer", "type": "answer"})
    return steps


def get_clue_by_id(clue_id):
    """Fetch a full clue row with structured_explanations data, or None."""
    db = get_db()
    row = db.execute(
        """SELECT c.*, se.components, se.confidence
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
            if pieces:
                parts = []
                for p in pieces:
                    mech = p.get("mechanism", "")
                    word = p.get("clue_word", "")
                    letters = p.get("letters", "")
                    if mech and word and letters:
                        # Fix mislabelled mechanisms
                        mech = _correct_mechanism(mech, word, letters)
                        parts.append(f"{word} → {letters} ({mech})")
                if parts:
                    return " + ".join(parts)
        except (json.JSONDecodeError, TypeError):
            pass
    return None
