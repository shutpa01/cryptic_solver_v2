"""Helper widget routes — solver's toolkit for word lookups, pattern search,
anagram solving, and similar clue search."""

import re
import sqlite3
from pathlib import Path

from flask import Blueprint, request, render_template, abort, g

# Stop words to ignore when searching for similar clues
_STOP_WORDS = frozenset(
    "a an the in on of to for and or but is it its by at with from as "
    "be not no has had have was were are been this that up so if do does".split()
)

bp = Blueprint("helper", __name__)

_BASE = Path(__file__).resolve().parent.parent.parent
REF_DB = str(_BASE / "data" / "cryptic_new.db")
CLUES_DB = str(_BASE / "data" / "clues_master.db")


def _get_ref_db():
    """Return a read-only connection to the reference database (cryptic_new.db)."""
    if "ref_db" not in g:
        uri = f"file:{REF_DB}?mode=ro"
        g.ref_db = sqlite3.connect(uri, uri=True)
        g.ref_db.row_factory = sqlite3.Row
    return g.ref_db


# Flask-Limiter decorator would go here, e.g.:
# @limiter.limit("30/minute")
@bp.route("/helper/lookup")
def lookup():
    """Look up a word and return synonyms, indicators, abbreviations,
    definitions, and homophones from the reference DB.

    Query params:
        word (str, required) — the word to look up
        letters (int, optional) — desired answer length, filters results if provided

    Returns an HTML fragment for HTMX.
    """
    word = request.args.get("word", "").strip()
    letters_raw = request.args.get("letters", "").strip()

    if not word:
        abort(400)

    letters = None
    if letters_raw:
        try:
            letters = int(letters_raw)
        except ValueError:
            pass
        if letters is not None and (letters < 1 or letters > 30):
            letters = None

    word_lower = word.lower()
    db = _get_ref_db()

    # 1. "Could mean" — combined synonyms + definition answers, deduplicated
    if letters:
        synonyms = db.execute(
            """SELECT DISTINCT val FROM (
                   SELECT UPPER(synonym) AS val FROM synonyms_pairs
                   WHERE LOWER(word) = ? AND LENGTH(synonym) = ?
                   UNION
                   SELECT UPPER(answer) AS val FROM definition_answers_augmented
                   WHERE LOWER(definition) = ? AND LENGTH(answer) = ?
               ) ORDER BY val LIMIT 15""",
            (word_lower, letters, word_lower, letters),
        ).fetchall()
        meanings_list = [r["val"] for r in synonyms]
    else:
        synonyms = db.execute(
            """SELECT DISTINCT val, LENGTH(val) as len FROM (
                   SELECT UPPER(synonym) AS val FROM synonyms_pairs
                   WHERE LOWER(word) = ?
                   UNION
                   SELECT UPPER(answer) AS val FROM definition_answers_augmented
                   WHERE LOWER(definition) = ?
               ) ORDER BY LENGTH(val), val LIMIT 200""",
            (word_lower, word_lower),
        ).fetchall()
        # Group by length, top 5 per group, alphabetical
        by_len = {}
        for r in synonyms:
            l = r["len"]
            if l not in by_len:
                by_len[l] = []
            by_len[l].append(r["val"])
        meanings_list = []
        for l in sorted(by_len):
            all_words = sorted(by_len[l])
            shown = all_words[:5]
            more = len(all_words) - 5 if len(all_words) > 5 else 0
            meanings_list.append({"length": l, "words": shown, "more": more})

    # 2. Indicators — alphabetical by type
    indicators = db.execute(
        """SELECT wordplay_type, confidence FROM indicators
           WHERE LOWER(word) = ?
           ORDER BY wordplay_type""",
        (word_lower,),
    ).fetchall()

    # 3. Abbreviations — by length then alphabetical
    if letters:
        abbreviations = db.execute(
            """SELECT DISTINCT substitution FROM wordplay
               WHERE LOWER(indicator) = ? AND LENGTH(substitution) = ?
               ORDER BY substitution
               LIMIT 10""",
            (word_lower, letters),
        ).fetchall()
    else:
        abbreviations = db.execute(
            """SELECT DISTINCT substitution FROM wordplay
               WHERE LOWER(indicator) = ?
               ORDER BY LENGTH(substitution), substitution
               LIMIT 15""",
            (word_lower,),
        ).fetchall()

    # 4. Homophones — alphabetical
    if letters:
        homophones = db.execute(
            """SELECT DISTINCT homophone FROM homophones
               WHERE LOWER(word) = ? AND LENGTH(homophone) = ?
               ORDER BY homophone
               LIMIT 10""",
            (word_lower, letters),
        ).fetchall()
    else:
        homophones = db.execute(
            """SELECT DISTINCT homophone FROM homophones
               WHERE LOWER(word) = ?
               ORDER BY homophone
               LIMIT 10""",
            (word_lower,),
        ).fetchall()

    return render_template(
        "partials/helper_results.html",
        word=word,
        letters=letters,
        meanings=meanings_list,
        meanings_grouped=letters is None,
        indicators=[
            {"type": r["wordplay_type"].replace("_", " ").title()}
            for r in indicators
        ],
        abbreviations=[r["substitution"] for r in abbreviations],
        homophones=[r["homophone"] for r in homophones],
    )


def _get_clues_db():
    """Return a read-only connection to clues_master.db."""
    if "clues_pattern_db" not in g:
        uri = f"file:{CLUES_DB}?mode=ro"
        g.clues_pattern_db = sqlite3.connect(uri, uri=True)
        g.clues_pattern_db.row_factory = sqlite3.Row
    return g.clues_pattern_db


@bp.route("/helper/meanings")
def meanings_expand():
    """Return all meanings for a word at a specific letter count.

    Used when the user clicks a length group to expand it.
    """
    word = request.args.get("word", "").strip()
    letters = request.args.get("letters", type=int)
    if not word or not letters:
        abort(400)

    word_lower = word.lower()
    db = _get_ref_db()
    rows = db.execute(
        """SELECT DISTINCT val FROM (
               SELECT UPPER(synonym) AS val FROM synonyms_pairs
               WHERE LOWER(word) = ? AND LENGTH(synonym) = ?
               UNION
               SELECT UPPER(answer) AS val FROM definition_answers_augmented
               WHERE LOWER(definition) = ? AND LENGTH(answer) = ?
           ) ORDER BY val""",
        (word_lower, letters, word_lower, letters),
    ).fetchall()

    words = [r["val"] for r in rows]
    return (
        '<span class="text-blue-400 text-xs font-bold">(%d)</span> '
        '<span class="text-gray-800">%s</span>'
    ) % (letters, ", ".join(words))


# Flask-Limiter decorator would go here, e.g.:
# @limiter.limit("20/minute")
@bp.route("/helper/pattern")
def pattern_search():
    """Search for words matching a pattern.

    The user types known letters and ? or space for unknowns.
    Dashes separate words in multi-word answers.

    Query params:
        pattern (str, required) — e.g. "S?O?E" or "OX?ORD-S??EET"

    Returns an HTML fragment for HTMX.
    """
    raw = request.args.get("pattern", "").strip().upper()
    if not raw:
        abort(400)

    # Normalise: spaces and ? both mean unknown letter, - means word break
    # Convert to SQL LIKE pattern: ? -> _, - -> space (multi-word answers stored with spaces)
    clean = re.sub(r'[^A-Z0-9?\- ]', '', raw)
    if not clean or len(clean) > 25:
        abort(400)

    # Build SQL LIKE patterns — one with spaces (for multi-word), one without
    pattern_spaced = ""
    pattern_joined = ""
    display = ""
    for ch in clean:
        if ch in ('?', ' '):
            pattern_spaced += '_'
            pattern_joined += '_'
            display += '?'
        elif ch == '-':
            pattern_spaced += ' '
            # No character in joined version — word break is just visual
            display += '-'
        else:
            pattern_spaced += ch
            pattern_joined += ch
            display += ch

    total_letters = len(pattern_joined)
    if total_letters < 2:
        abort(400)

    # Search distinct answers from clues_master — try both spaced and joined forms
    db = _get_clues_db()
    results = set()

    for pat, pat_len in [(pattern_spaced, len(pattern_spaced)), (pattern_joined, len(pattern_joined))]:
        rows = db.execute(
            """SELECT DISTINCT UPPER(answer) AS ans
               FROM clues
               WHERE UPPER(answer) LIKE ?
               AND LENGTH(answer) = ?
               LIMIT 50""",
            (pat, pat_len),
        ).fetchall()
        for r in rows:
            results.add(r["ans"])

    # Filter by "must include" letters if specified
    include = request.args.get("include", "").strip().upper()
    include_letters = re.sub(r'[^A-Z]', '', include)

    if include_letters:
        filtered = []
        for word in results:
            word_letters = list(word.replace(" ", ""))
            match = True
            for ch in include_letters:
                if ch in word_letters:
                    word_letters.remove(ch)
                else:
                    match = False
                    break
            if match:
                filtered.append(word)
        matches = sorted(filtered)[:50]
    else:
        matches = sorted(results)[:50]

    return render_template(
        "partials/pattern_results.html",
        pattern=display,
        total_letters=total_letters,
        matches=matches,
    )


def _letter_signature(word):
    """Sorted uppercase letters only — the anagram fingerprint."""
    return "".join(sorted(re.sub(r'[^A-Z]', '', word.upper())))


# Flask-Limiter decorator would go here, e.g.:
# @limiter.limit("20/minute")
@bp.route("/helper/anagram")
def anagram_search():
    """Find anagrams of the given letters.

    User enters jumbled letters with dashes for word breaks.
    e.g. "ulbs-eky" → finds "BLUE SKY" (4,3)
    e.g. "sohre" → finds "HORSE", "SHORE", etc.

    Only returns real answers from our clue database — never nonsense.

    Query params:
        letters (str, required) — jumbled letters, dashes for word breaks
    """
    raw = request.args.get("letters", "").strip().upper()
    if not raw:
        abort(400)

    # Parse: letters + dash structure
    clean = re.sub(r'[^A-Z\-]', '', raw)
    if not clean or len(clean) > 25:
        abort(400)

    parts = clean.split('-')
    word_lengths = [len(p) for p in parts if p]
    all_letters = ''.join(parts)
    total = len(all_letters)

    if total < 2:
        abort(400)

    signature = "".join(sorted(all_letters))
    display = "-".join(parts)
    db = _get_clues_db()
    results = set()

    if len(word_lengths) == 1:
        # Single word — search for answers of this exact length
        rows = db.execute(
            """SELECT DISTINCT UPPER(answer) AS ans
               FROM clues
               WHERE LENGTH(answer) = ?
               AND answer IS NOT NULL""",
            (total,),
        ).fetchall()
        for r in rows:
            if _letter_signature(r["ans"]) == signature:
                results.add(r["ans"])

    else:
        # Multi-word with dashes — search both spaced and joined forms
        enum_pattern = ",".join(str(l) for l in word_lengths)

        # Search by enumeration match (most reliable)
        rows = db.execute(
            """SELECT DISTINCT UPPER(answer) AS ans, answer AS raw_ans
               FROM clues
               WHERE enumeration = ?
               AND answer IS NOT NULL""",
            (enum_pattern,),
        ).fetchall()
        for r in rows:
            if _letter_signature(r["ans"]) == signature:
                results.add(r["raw_ans"].upper())

        # Also try joined form (no spaces)
        rows2 = db.execute(
            """SELECT DISTINCT UPPER(answer) AS ans
               FROM clues
               WHERE LENGTH(answer) = ?
               AND answer IS NOT NULL""",
            (total,),
        ).fetchall()
        for r in rows2:
            if _letter_signature(r["ans"]) == signature:
                results.add(r["ans"])

    matches = sorted(results)[:50]

    return render_template(
        "partials/anagram_results.html",
        letters=display,
        total_letters=total,
        word_lengths=word_lengths,
        matches=matches,
    )


@bp.route("/helper/similar")
def similar():
    """Search for clues with similar wording in the 500k clue database.

    Query params:
        q (str, required) — search text (clue or partial clue)
        enum (str, optional) — enumeration to filter by (e.g. "5" or "3-6")
        exclude_id (int, optional) — clue ID to exclude from results

    Results are filtered to match enumeration and sorted by word overlap
    (most matching words first).

    Returns an HTML fragment for HTMX.
    """
    query = request.args.get("q", "").strip()
    enum = request.args.get("enum", "").strip()
    label = request.args.get("label", "").strip()
    clue_id = request.args.get("clue_id", type=int)
    exclude_id = request.args.get("exclude_id", type=int)

    if not query or len(query) < 3:
        return render_template("partials/similar_results.html", query=query, results=[])

    # Extract significant words (drop stop words and short words)
    words = [w for w in re.findall(r"[a-zA-Z]+", query.lower()) if w not in _STOP_WORDS and len(w) > 2]

    if len(words) < 1:
        return render_template("partials/similar_results.html", query=query, results=[])

    # Limit to 6 most significant words (longest first — more distinctive)
    words.sort(key=len, reverse=True)
    search_words = words[:6]

    # Use OR query to cast a wide net, then rank by overlap
    conditions = ["clue_text LIKE ?"] * len(search_words)
    params = [f"%{w}%" for w in search_words]

    or_where = " OR ".join(conditions)
    full_where = f"({or_where}) AND answer IS NOT NULL AND length(answer) > 0"

    # Strictly filter by enumeration
    if enum:
        full_where += " AND enumeration = ?"
        params.append(enum)

    # Exclude the source clue by exact text match
    full_where += " AND clue_text != ?"
    params.append(query)

    if exclude_id:
        full_where += " AND id != ?"
        params.append(exclude_id)

    db = _get_clues_db()
    rows = db.execute(f"""
        SELECT clue_text, answer, enumeration, source, puzzle_number
        FROM clues
        WHERE {full_where}
        LIMIT 500
    """, params).fetchall()

    # Score each result by weighted overlap — longer words worth more
    def _overlap_score(clue_text):
        ct = clue_text.lower()
        return sum(len(w) for w in search_words if w in ct)

    # Minimum score threshold: require the 2 longest search words to match
    sorted_lens = sorted((len(w) for w in search_words), reverse=True)
    min_score = sum(sorted_lens[:2]) if len(sorted_lens) >= 2 else sum(sorted_lens)

    # Deduplicate by clue_text+answer, score and rank
    seen = set()
    scored = []
    for r in rows:
        key = (r["clue_text"].lower(), r["answer"])
        if key not in seen:
            seen.add(key)
            score = _overlap_score(r["clue_text"])
            if score >= min_score:
                scored.append((score, dict(r)))

    # Sort by overlap score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [item for _, item in scored[:20]]

    # Look up the source clue's answer for accuracy checking
    source_answer = None
    if clue_id:
        row = db.execute(
            "SELECT answer FROM clues WHERE id = ?", (clue_id,)
        ).fetchone()
        if row and row["answer"]:
            source_answer = row["answer"].upper().strip()

    return render_template(
        "partials/similar_results.html",
        query=query,
        words=search_words,
        results=results,
        label=label,
        enum=enum,
        clue_id=clue_id,
        has_answer=source_answer is not None,
        source_answer=source_answer,
        is_admin=g.get("is_admin", False),
    )
