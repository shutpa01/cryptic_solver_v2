"""Helper widget routes — solver's toolkit for word lookups, pattern search,
anagram solving, and similar clue search.

All endpoints require a valid helper token (ht parameter) to prevent
bulk scraping of the reference database. Tokens are generated per page
load and expire after 2 hours.
"""

import re
import sqlite3
from pathlib import Path

import json

from flask import Blueprint, request, render_template, abort, g, jsonify, current_app
from web.word_cache import match_pattern, match_pattern_user
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

# Stop words to ignore when searching for similar clues
_STOP_WORDS = frozenset(
    "a an the in on of to for and or but is it its by at with from as "
    "be not no has had have was were are been this that up so if do does".split()
)

bp = Blueprint("helper", __name__)

HELPER_TOKEN_MAX_AGE = 7200  # 2 hours


def generate_helper_token():
    """Generate a signed token for helper endpoint access."""
    s = URLSafeTimedSerializer(current_app.config["SECRET_KEY"])
    return s.dumps({"helper": True}, salt="helper-access")


_rate_limit_store = {}  # IP -> (count, window_start)
RATE_LIMIT_MAX = 60     # requests per window
RATE_LIMIT_WINDOW = 60  # seconds


@bp.before_request
def _require_helper_token():
    """Validate helper token and enforce rate limiting.
    Admin users bypass both checks."""
    if getattr(g, 'is_admin', False):
        return

    # Token validation
    token = request.args.get("ht", "")
    if not token:
        abort(403)
    s = URLSafeTimedSerializer(current_app.config["SECRET_KEY"])
    try:
        s.loads(token, max_age=HELPER_TOKEN_MAX_AGE, salt="helper-access")
    except (BadSignature, SignatureExpired):
        abort(403)

    # Rate limiting
    import time
    ip = request.remote_addr or "unknown"
    now = time.time()
    if ip in _rate_limit_store:
        count, window_start = _rate_limit_store[ip]
        if now - window_start > RATE_LIMIT_WINDOW:
            _rate_limit_store[ip] = (1, now)
        elif count >= RATE_LIMIT_MAX:
            abort(429)
        else:
            _rate_limit_store[ip] = (count + 1, window_start)
    else:
        _rate_limit_store[ip] = (1, now)

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
    enum_exact = request.args.get("enum_exact", "").strip()

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

    # Parse exact enumeration pattern for filtering (e.g. "3,2,4")
    enum_pattern = None
    if enum_exact:
        import re as _re
        nums = _re.findall(r'\d+', enum_exact)
        if nums:
            enum_pattern = [int(n) for n in nums]

    word_lower = word.lower()
    db = _get_ref_db()

    # 1. "Could mean" — combined synonyms + definition answers, deduplicated
    if enum_pattern:
        # Exact enumeration filter: only synonyms whose word lengths match the pattern
        total_letters = sum(enum_pattern)
        synonyms = db.execute(
            """SELECT DISTINCT val FROM (
                   SELECT UPPER(synonym) AS val FROM synonyms_pairs
                   WHERE LOWER(word) = ? AND LENGTH(REPLACE(synonym, ' ', '')) = ?
                   UNION
                   SELECT UPPER(answer) AS val FROM definition_answers_augmented
                   WHERE LOWER(definition) = ? AND LENGTH(REPLACE(answer, ' ', '')) = ?
               ) ORDER BY val""",
            (word_lower, total_letters, word_lower, total_letters),
        ).fetchall()

        def _matches_enum(val, pattern):
            """Check if a synonym's word lengths match the enumeration pattern."""
            parts = val.split()
            if len(parts) != len(pattern):
                return False
            return all(len(p) == expected for p, expected in zip(parts, pattern))

        meanings_list = []
        for r in synonyms:
            val = r["val"]
            if _matches_enum(val, enum_pattern):
                meanings_list.append(val)
        meanings_list.sort()
    elif letters:
        synonyms = db.execute(
            """SELECT DISTINCT val FROM (
                   SELECT UPPER(synonym) AS val FROM synonyms_pairs
                   WHERE LOWER(word) = ? AND LENGTH(REPLACE(synonym, ' ', '')) = ?
                   UNION
                   SELECT UPPER(answer) AS val FROM definition_answers_augmented
                   WHERE LOWER(definition) = ? AND LENGTH(REPLACE(answer, ' ', '')) = ?
               ) ORDER BY val LIMIT 15""",
            (word_lower, letters, word_lower, letters),
        ).fetchall()
        meanings_list = [r["val"] for r in synonyms]
    else:
        synonyms = db.execute(
            """SELECT DISTINCT val, LENGTH(REPLACE(val, ' ', '')) as len FROM (
                   SELECT UPPER(synonym) AS val FROM synonyms_pairs
                   WHERE LOWER(word) = ?
                   UNION
                   SELECT UPPER(answer) AS val FROM definition_answers_augmented
                   WHERE LOWER(definition) = ?
               ) ORDER BY LENGTH(REPLACE(val, ' ', '')), val""",
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

    # 2. Indicators — alphabetical by type, include subtype
    indicators = db.execute(
        """SELECT wordplay_type, subtype, confidence FROM indicators
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

    # Parse enumeration for filter button (e.g. "3,2,4" -> total 9, display "3,2,4")
    enum_raw = request.args.get("enum", "").strip()
    enum_total = 0
    enum_display = enum_exact or enum_raw
    if enum_display:
        import re as _re
        nums = _re.findall(r'\d+', enum_display)
        enum_total = sum(int(n) for n in nums) if nums else 0

    return render_template(
        "partials/helper_results.html",
        word=word,
        letters=letters,
        meanings=meanings_list,
        meanings_grouped=letters is None and enum_pattern is None,
        enum_filtered=enum_pattern is not None,
        indicators=[
            {"type": r["wordplay_type"].replace("_", " ").title(),
             "subtype": (r["subtype"] or "").replace("_", " ") if r["subtype"] and r["subtype"] not in ("general", "") else ""}
            for r in indicators
        ],
        abbreviations=[r["substitution"] for r in abbreviations],
        homophones=[r["homophone"] for r in homophones],
        enum_total=enum_total,
        enum_display=enum_display,
        helper_token=request.args.get("ht", ""),
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
               WHERE LOWER(word) = ? AND LENGTH(REPLACE(synonym, ' ', '')) = ?
               UNION
               SELECT UPPER(answer) AS val FROM definition_answers_augmented
               WHERE LOWER(definition) = ? AND LENGTH(REPLACE(answer, ' ', '')) = ?
           ) ORDER BY val""",
        (word_lower, letters, word_lower, letters),
    ).fetchall()

    words = [r["val"] for r in rows]
    word_spans = []
    for w in words:
        word_spans.append(
            '<span class="synonym-pick text-gray-800 cursor-pointer hover:bg-blue-200 '
            'hover:rounded px-0.5" onclick="synonymToSolve(\'%s\')" title="Use as answer">%s</span>' % (w, w)
        )
    return (
        '<span class="text-blue-400 text-xs font-bold">(%d)</span> '
        '%s'
    ) % (letters, ", ".join(word_spans))


# Flask-Limiter decorator would go here, e.g.:
# @limiter.limit("20/minute")
@bp.route("/helper/pattern-counts")
def pattern_counts_batch():
    """Batch pattern count — accepts multiple patterns in one request.

    Query param: patterns — JSON object {"clueId": {"pattern": "S?O?E", "enum": "5"}, ...}
    Returns: JSON object {"clueId": count, ...}
    """
    raw = request.args.get("patterns", "")
    if not raw:
        return jsonify({})

    try:
        queries = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return jsonify({})

    results = {}

    for clue_id, info in queries.items():
        pat_raw = info.get("pattern", "").strip().upper()
        enum_val = info.get("enum", "").strip()
        if not pat_raw or len(pat_raw) < 2:
            results[clue_id] = 0
            continue

        clean = re.sub(r'[^A-Z0-9?\- ]', '', pat_raw)
        pattern_joined = ""
        pattern_spaced = ""
        for ch in clean:
            if ch in ('?', ' '):
                pattern_joined += '_'
                pattern_spaced += '_'
            elif ch == '-':
                pattern_spaced += ' '
            else:
                pattern_joined += ch
                pattern_spaced += ch

        if len(pattern_joined) < 2:
            results[clue_id] = 0
            continue

        # Skip all-unknown patterns (no known letters) — count is meaningless
        if all(c == '_' for c in pattern_joined):
            results[clue_id] = 0
            continue

        enum_parts = re.findall(r'\d+', enum_val) if enum_val else []

        # Clues-only cache with enum-formatted answers (handles hyphenated enums)
        seen = match_pattern_user(pattern_joined)
        if getattr(g, 'is_admin', False):
            # Admin also gets broad word cache for prize puzzles without answers
            seen |= match_pattern(pattern_joined, pattern_spaced)

        if enum_val and enum_parts:
            if len(enum_parts) == 1:
                seen = {w for w in seen if ' ' not in w}
            elif len(enum_parts) > 1:
                def _matches_enum(answer, parts):
                    words = answer.split()
                    if len(words) == len(parts):
                        return all(len(w) == int(p) for w, p in zip(words, parts))
                    return False
                seen = {w for w in seen if _matches_enum(w, enum_parts)}

        results[clue_id] = len(seen)

    return jsonify(results)


@bp.route("/helper/word-info")
def word_info():
    """Quick reverse lookup — what does this word mean in crossword context?

    Returns a short text like "bird, number, colour, cleric" showing
    what definitions map to this answer word.
    """
    word = request.args.get("word", "").strip().upper()
    if not word or len(word) < 2:
        return ""

    ref = _get_ref_db()
    meanings = set()

    # Try both with-space and without-space variants
    variants = [word]
    if " " in word:
        variants.append(word.replace(" ", ""))
    elif len(word) > 5:
        # Try adding spaces at common break points (from clues DB)
        db = _get_clues_db()
        spaced = db.execute(
            "SELECT DISTINCT answer FROM clues WHERE UPPER(REPLACE(answer,' ','')) = ? AND answer LIKE '% %' LIMIT 1",
            (word,),
        ).fetchone()
        if spaced:
            variants.append(spaced["answer"].upper())

    for variant in variants:
        # Reverse synonym lookup: what words have this as a synonym?
        rows = ref.execute(
            "SELECT DISTINCT LOWER(word) AS w FROM synonyms_pairs WHERE UPPER(synonym) = ? LIMIT 20",
            (variant,),
        ).fetchall()
        for r in rows:
            meanings.add(r["w"])

        # Also check definition_answers_augmented
        rows2 = ref.execute(
            "SELECT DISTINCT LOWER(definition) AS d FROM definition_answers_augmented WHERE UPPER(answer) = ? LIMIT 20",
            (variant,),
        ).fetchall()
        for r in rows2:
            meanings.add(r["d"])

    if not meanings:
        return "no definitions found"

    # Show up to 8, sorted by length (shorter = more useful)
    sorted_meanings = sorted(meanings, key=len)[:8]
    return ", ".join(sorted_meanings)


@bp.route("/helper/pattern-count")
def pattern_count():
    """Fast count-only pattern match — just returns the number of matches.

    Only searches clues table (not reference DB) for speed.
    Used by crossing letter match counts where we need many counts fast.
    """
    raw = request.args.get("pattern", "").strip().upper()
    enum_val = request.args.get("enum", "").strip()
    if not raw or len(raw) < 2:
        return "0"

    clean = re.sub(r'[^A-Z0-9?\- ]', '', raw)
    pattern_joined = ""
    pattern_spaced = ""
    for ch in clean:
        if ch in ('?', ' '):
            pattern_joined += '_'
            pattern_spaced += '_'
        elif ch == '-':
            pattern_spaced += ' '
        else:
            pattern_joined += ch
            pattern_spaced += ch

    if len(pattern_joined) < 2:
        return "0"

    # Skip all-unknown patterns (no known letters)
    if all(c == '_' for c in pattern_joined):
        return "0"

    enum_parts = re.findall(r'\d+', enum_val) if enum_val else []

    # Clues-only cache with enum-formatted answers (handles hyphenated enums)
    seen = match_pattern_user(pattern_joined)
    if getattr(g, 'is_admin', False):
        # Admin also gets broad word cache for prize puzzles without answers
        seen |= match_pattern(pattern_joined, pattern_spaced)

    # Filter by enumeration word breaks
    if enum_val and enum_parts:
        if len(enum_parts) == 1:
            seen = {w for w in seen if ' ' not in w}
        elif len(enum_parts) > 1:
            def _matches_enum(answer, parts):
                words = answer.split()
                if len(words) == len(parts):
                    return all(len(w) == int(p) for w, p in zip(words, parts))
                return False
            seen = {w for w in seen if _matches_enum(w, enum_parts)}

    return str(len(seen))


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

    # Normalise: ? means unknown letter, - or , means word break
    # Convert to SQL LIKE pattern: ? -> _, - -> space (multi-word answers stored with spaces)
    clean = re.sub(r'[^A-Z0-9?\-,]', '', raw)
    clean = clean.replace(',', '-')  # treat comma same as dash
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

    # Derive enumeration from dash pattern if user typed word breaks
    # e.g. "I?-S?I?????" -> word_lengths = [2, 7] -> enum_parts = ['2', '7']
    dash_parts = [p for p in clean.split('-') if p]
    has_word_breaks = len(dash_parts) > 1
    word_lengths = [len(re.sub(r'[^A-Z?]', '', p)) for p in dash_parts]

    enum_val = request.args.get("enum", "").strip()
    if not enum_val and has_word_breaks:
        enum_val = ",".join(str(wl) for wl in word_lengths)
    enum_parts = re.findall(r'\d+', enum_val) if enum_val else []

    # Clues-only cache with enum-formatted answers (handles hyphenated enums)
    results = match_pattern_user(pattern_joined)
    if getattr(g, 'is_admin', False):
        # Admin also gets broad word cache for prize puzzles without answers
        results |= match_pattern(pattern_joined, pattern_spaced)

    # Filter by "must include" letters if specified
    include = request.args.get("include", "").strip().upper()
    include_letters = re.sub(r'[^A-Z]', '', include)

    # Always filter by enumeration when word breaks are specified
    if has_word_breaks and not enum_parts:
        enum_parts = [str(wl) for wl in word_lengths]

    if enum_parts:
        if len(enum_parts) == 1:
            # Single word — exclude multi-word answers
            results = {w for w in results if ' ' not in w}
        elif len(enum_parts) > 1:
            # Multi-word — only keep answers that are genuinely multi-word
            # with the right word lengths. Don't force-split single words.
            expected_lengths = [int(p) for p in enum_parts]
            filtered = set()
            for w in results:
                if ' ' in w:
                    words = w.split()
                    if len(words) == len(expected_lengths) and all(
                            len(wd) == el for wd, el in zip(words, expected_lengths)):
                        filtered.add(w)
                # Single words are NOT reformatted — they don't match
                # the multi-word enumeration the user requested
            results = filtered

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
        matches = sorted(filtered)
    else:
        matches = sorted(results)

    # Look up enumerations for display
    db = _get_clues_db()
    match_enums = {}
    for word in matches:
        clean_word = word.replace(' ', '')
        row = db.execute(
            "SELECT enumeration FROM clues WHERE UPPER(REPLACE(answer,' ','')) = ? AND enumeration IS NOT NULL AND enumeration != '' LIMIT 1",
            (clean_word,),
        ).fetchone()
        if row:
            match_enums[word] = row[0]

    return render_template(
        "partials/pattern_results.html",
        pattern=display,
        total_letters=total_letters,
        matches=matches,
        match_enums=match_enums,
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

    # Parse: letters + separator structure (treat , and - interchangeably)
    clean = re.sub(r'[^A-Z,\-]', '', raw)
    clean = clean.replace(',', '-')  # normalise separators
    if not clean or len(clean) > 25:
        abort(400)

    parts = [p for p in clean.split('-') if p]
    word_lengths = [len(p) for p in parts]
    all_letters = ''.join(parts)
    total = len(all_letters)

    if total < 2:
        abort(400)

    signature = "".join(sorted(all_letters))
    display = "-".join(parts)
    db = _get_clues_db()
    results = set()

    if len(word_lengths) == 1:
        # Single word — search clue answers of this exact length
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

        # Also try joined form (no spaces) — but only add if we can
        # format it to match the requested word lengths
        rows2 = db.execute(
            """SELECT DISTINCT UPPER(answer) AS ans
               FROM clues
               WHERE LENGTH(answer) = ?
               AND answer IS NOT NULL""",
            (total,),
        ).fetchall()
        for r in rows2:
            if _letter_signature(r["ans"]) == signature:
                # Format as multi-word to match the requested enumeration
                ans = r["ans"].replace(" ", "")
                formatted = []
                pos = 0
                for wl in word_lengths:
                    formatted.append(ans[pos:pos + wl])
                    pos += wl
                results.add(" ".join(formatted))

    # Also search reference DB (synonyms + definitions)
    ref = _get_ref_db()
    for table, col in [("synonyms_pairs", "synonym"), ("definition_answers_augmented", "answer")]:
        ref_rows = ref.execute(
            f"SELECT DISTINCT UPPER({col}) AS ans FROM {table} WHERE LENGTH({col}) = ?",
            (total,),
        ).fetchall()
        for r in ref_rows:
            if _letter_signature(r["ans"]) == signature:
                results.add(r["ans"])

    # Filter out the input itself (don't show fodder as a result)
    input_upper = all_letters.upper()
    matches = sorted(r for r in results if r.replace(" ", "") != input_upper)

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

    db = _get_clues_db()

    # Look up the source clue's answer
    source_answer = None
    if clue_id:
        row = db.execute(
            "SELECT answer FROM clues WHERE id = ?", (clue_id,)
        ).fetchone()
        if row and row["answer"]:
            source_answer = row["answer"].upper().strip()

    if not source_answer and (not query or len(query) < 3):
        return render_template("partials/similar_results.html", query=query, results=[])

    # Search by answer — find other clues with the same answer
    if source_answer:
        params = [source_answer]
        where = "UPPER(answer) = ? AND clue_text != ?"
        params.append(query)
        if exclude_id:
            where += " AND id != ?"
            params.append(exclude_id)

        rows = db.execute(f"""
            SELECT clue_text, answer, enumeration, source, puzzle_number
            FROM clues
            WHERE {where}
              AND answer IS NOT NULL AND length(answer) > 0
            ORDER BY publication_date DESC
            LIMIT 200
        """, params).fetchall()

        # Deduplicate by clue text
        seen = set()
        results = []
        for r in rows:
            key = r["clue_text"].lower()
            if key not in seen:
                seen.add(key)
                results.append(dict(r))
            if len(results) >= 20:
                break
    else:
        results = []

    return render_template(
        "partials/similar_results.html",
        query=query,
        results=results,
        label=label,
        enum=enum,
        clue_id=clue_id,
        has_answer=source_answer is not None,
        source_answer=source_answer,
        is_admin=g.get("is_admin", False),
    )
