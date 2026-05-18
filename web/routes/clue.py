"""Individual clue page — the primary SEO entry point.

Each of the 500k clues gets its own URL: id prefix + clue text, e.g.
/clue/2046138-parisian-is-running-home-to-host-a-european . The answer
is deliberately NOT in the slug — that would leak it before the user
chooses to reveal it.

Legacy URL formats from before commit 8efd6532 (text-ANSWER and
text-ENUMERATION suffixes) are handled by the redirect helpers
old_slug_to_new_id and enum_slug_to_new_id, which 301 to the current
answer-free format.

Blueprint registration (already in web/__init__.py):
    from web.routes.clue import bp as clue_bp
    app.register_blueprint(clue_bp)
"""

import re

from flask import Blueprint, render_template, request, abort, g, redirect, url_for

from web.db import get_db
from web.models import (
    classify_puzzle, compute_hint_tier, get_hint_steps, compute_solve_source,
    get_hint_content, clue_slug,
)
from flask import make_response

from web.routes.hints import generate_token
from web.routes.clue_seo import (
    generate_meta_description, generate_faq_schema, generate_breadcrumb_schema,
    generate_word_roles_schema,
)
from web.rate_limit import rate_limit
from web.session_token import issue_session_cookie

bp = Blueprint("clue", __name__)


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

def generate_clue_slug(clue_text, answer=None, clue_id=None):
    """Create a URL-safe slug from clue ID and clue text.

    Format: {id}-{clue-text-words}
    Example: 2046138-parisian-is-running-home-to-host-a-european

    Answer is NOT included in the slug — it would defeat the purpose
    of progressive hints.
    """
    if not clue_id:
        return None
    # Clean clue text: lowercase, replace non-alphanumeric with hyphens
    text = clue_text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    if not text:
        return None
    # Truncate to keep URLs reasonable
    words = text.split("-")[:12]
    text = "-".join(words)
    return f"{clue_id}-{text}"


def parse_clue_slug(slug):
    """Extract clue ID from slug.

    Returns (clue_id, slug_text) or (None, None).
    """
    if not slug:
        return None, None
    parts = slug.split("-", 1)
    if not parts or not parts[0].isdigit():
        return None, None
    clue_id = int(parts[0])
    slug_text = parts[1] if len(parts) > 1 else ""
    return clue_id, slug_text


def _build_old_slug(clue_text, answer):
    """Reconstruct the pre-`8efd6532` slug for matching: text-words-ANSWER."""
    if not clue_text or not answer:
        return None
    text = re.sub(r"[^a-z0-9]+", "-", clue_text.lower().strip()).strip("-")
    ans = re.sub(r"[^A-Za-z0-9]", "", answer).upper()
    if not text or not ans:
        return None
    return f"{text}-{ans}"


def old_slug_to_new_id(slug):
    """If `slug` matches the pre-`8efd6532` format, return the clue id, else None.

    Old format put the answer as a trailing all-uppercase block, e.g.
    `companions-shredded-corset-ESCORT`. Some clues share text+answer across
    sources/puzzles; we accept any match (caller redirects to one canonical id).
    """
    if not slug:
        return None
    parts = slug.split("-")
    if len(parts) < 2:
        return None

    # Collect trailing uppercase alphanumeric parts that contain at least
    # one letter (the answer block, possibly split if it had digits).
    answer_parts = []
    for i in range(len(parts) - 1, -1, -1):
        p = parts[i]
        if p and p.isalnum() and p == p.upper() and any(c.isalpha() for c in p):
            answer_parts.insert(0, p)
        else:
            break
    if not answer_parts:
        return None
    clue_parts = parts[:len(parts) - len(answer_parts)]
    if not clue_parts:
        return None

    answer = "".join(answer_parts)

    db = get_db()
    where_clauses = ["UPPER(answer) = ?"]
    params = [answer]
    for w in clue_parts[:3]:
        if w:
            where_clauses.append("LOWER(clue_text) LIKE ?")
            params.append(f"%{w}%")
    sql = (
        "SELECT id, clue_text, answer FROM clues "
        "WHERE source IN ('telegraph','times','guardian','independent','dailymail') "
        "  AND clue_text IS NOT NULL "
        f"  AND {' AND '.join(where_clauses)} "
        "ORDER BY publication_date DESC LIMIT 50"
    )
    rows = db.execute(sql, params).fetchall()
    for row in rows:
        if _build_old_slug(row["clue_text"], row["answer"]) == slug:
            return row["id"]
    return None


def _build_enum_slug(clue_text, enumeration):
    """Reconstruct an enumeration-suffixed slug for matching: text-words-N(-M)."""
    if not clue_text or not enumeration:
        return None
    text = re.sub(r"[^a-z0-9]+", "-", clue_text.lower().strip()).strip("-")
    enum = re.sub(r"[^0-9]+", "-", enumeration).strip("-")
    if not text or not enum:
        return None
    return f"{text}-{enum}"


def _enum_slug_db_lookup(text_parts, enum_parts, slug):
    """DB-side lookup for one (text_parts, enum_parts) split. Returns id or None.

    Filter words: prefer distinctive (>=3 char) words. Short tokens like
    'a', 'of', 'i', 'm' match too broadly and let LIMIT 50 truncate before
    the right row appears.
    """
    enum_dash = "-".join(enum_parts)
    enum_comma = ",".join(enum_parts)
    enum_candidates = list({enum_dash, enum_comma})

    long_words = [w for w in text_parts if len(w) >= 3]
    filter_words = (sorted(long_words, key=len, reverse=True)[:3]
                    if len(long_words) >= 1 else text_parts[:3])

    db = get_db()
    placeholders = ",".join("?" for _ in enum_candidates)
    where_clauses = [f"enumeration IN ({placeholders})"]
    params = list(enum_candidates)
    for w in filter_words:
        if w:
            where_clauses.append("LOWER(clue_text) LIKE ?")
            params.append(f"%{w}%")
    sql = (
        "SELECT id, clue_text, enumeration FROM clues "
        "WHERE source IN ('telegraph','times','guardian','independent','dailymail') "
        "  AND clue_text IS NOT NULL "
        f"  AND {' AND '.join(where_clauses)} "
        "ORDER BY publication_date DESC LIMIT 200"
    )
    rows = db.execute(sql, params).fetchall()
    for row in rows:
        if _build_enum_slug(row["clue_text"], row["enumeration"]) == slug:
            return row["id"]
    return None


def enum_slug_to_new_id(slug):
    """If `slug` matches the oldest enumeration-suffix format, return clue id, else None.

    The pre-pre-`8efd6532` URL format used the puzzle enumeration as the
    trailing suffix, e.g. `ring-back-about-origin-of-incredibly-small-tree-5`
    where `-5` is the enumeration "(5)", or `-5-3` for "(5,3)".

    DB stores enumeration as `5`, `5,3`, or `5-3` depending on the original
    setter convention; the slug uses `-` between numbers regardless.
    """
    if not slug:
        return None
    parts = slug.split("-")
    if len(parts) < 2:
        return None

    # Collect trailing all-digit parts (potential enumeration blocks).
    trailing_digits = []
    for i in range(len(parts) - 1, -1, -1):
        p = parts[i]
        if p and p.isdigit():
            trailing_digits.insert(0, p)
        else:
            break
    if not trailing_digits:
        return None

    # Try maximal-greedy enum first, then progressively shorter splits.
    # Some clue texts contain trailing numbers (e.g. "after 11") that get
    # absorbed into the enum guess; peel them off and retry.
    for n_enum in range(len(trailing_digits), 0, -1):
        enum_parts = trailing_digits[-n_enum:]
        text_parts = parts[:len(parts) - n_enum]
        if not text_parts:
            continue
        match = _enum_slug_db_lookup(text_parts, enum_parts, slug)
        if match is not None:
            return match
    return None


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@bp.route("/clue/<slug>")
@rate_limit(scope="clue_page", limit=60, window=60)
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
    # Look up by clue ID embedded in slug
    clue_id, slug_text = parse_clue_slug(slug)
    from web.models import get_clue_by_id
    clue = get_clue_by_id(clue_id) if clue_id else None

    if clue is None:
        # parse_clue_slug may have misread a leading digit as a clue ID
        # (e.g. "/clue/6-ceding-power-..." where "6" is part of the clue text).
        # Try old-format redirect helpers before giving up.
        old_id = old_slug_to_new_id(slug)
        if old_id is None:
            # Even older format (enumeration-suffixed). 14k+ Googlebot 404s
            # observed Apr 28-30 2026 came from this format, throttling the
            # crawl rate by 99%. Recovers PageRank from those URLs.
            old_id = enum_slug_to_new_id(slug)
        if old_id is not None:
            row = get_clue_by_id(old_id)
            new_slug = generate_clue_slug(row["clue_text"], clue_id=old_id) if row else str(old_id)
            return redirect(url_for("clue.clue_page", slug=new_slug or str(old_id)), code=301)
        # Also support ?id= parameter for backwards compatibility
        clue_id = request.args.get("id", type=int)
        if clue_id:
            clue = get_clue_by_id(clue_id)

    if clue is None:
        abort(404)
    clue_id = clue["id"]

    # Find other appearances of the same clue text + answer
    db = get_db()
    other_rows = db.execute(
        """SELECT id, source, puzzle_number, publication_date, clue_number, direction,
                  clue_text, enumeration, answer, definition, wordplay_type,
                  explanation, ai_explanation
           FROM clues
           WHERE answer = ? AND clue_text = ? AND id != ?
             AND source IN ('telegraph', 'times', 'guardian', 'independent', 'dailymail')
           ORDER BY publication_date DESC
           LIMIT 5""",
        (clue["answer"], clue["clue_text"], clue_id),
    ).fetchall()
    matches = list(other_rows)

    clue_dict = dict(clue)

    # Tier and hints
    tier, max_steps = compute_hint_tier(clue)
    steps = get_hint_steps(clue, tier=tier, is_admin=g.get("is_admin", False))
    clue_dict["tier"] = tier
    clue_dict["solve_source"] = compute_solve_source(clue)
    clue_dict["max_steps"] = max_steps
    clue_dict["total_steps"] = len(steps)
    clue_dict["steps"] = steps
    clue_dict["token"] = generate_token(clue["id"]) if steps else None

    # Pre-render definition / wordplay-type / explanation for inline
    # display. The answer is rendered separately in the template (it's
    # already inline). Reveal buttons are gone — these now appear on
    # initial page load for both SEO and one-glance user experience.
    # NOTE: the actual wordplay_type content is replaced below with
    # mechanism_label (computed from role_groups) when available, so
    # the page reads e.g. "CHARADE CONTAINER" instead of just "container".
    inline_hints = []
    for step_type in ("definition", "wordplay_type", "explanation"):
        content = get_hint_content(clue, step_type)
        if content:
            inline_hints.append({"type": step_type, "content": content})
    clue_dict["inline_hints"] = inline_hints

    # Per-clue word roles — populated by the verifier (auto) plus any
    # admin manual overrides. Empty list when no rows exist for the
    # clue: the template suppresses the section entirely so pages stay
    # visually consistent.
    #
    # get_roles returns 6-tuples ordered by word_index. We group
    # consecutive words sharing the same piece_key into one display row
    # (e.g. "US city" -> one row labelled "synonym → LA US city").
    # Words with piece_key=None stay solo.
    from sonnet_pipeline.word_roles_store import get_roles as _get_word_roles
    word_roles = _get_word_roles(clue["id"])
    role_groups = []
    word_role_rows = []
    for wi, wt, r, s, letters, piece_key in word_roles:
        word_role_rows.append({
            "word_index": wi, "word_text": wt, "role": r,
            "source": s, "letters": letters, "piece_key": piece_key,
        })
        if (piece_key is not None and role_groups
                and role_groups[-1]["piece_key"] == piece_key):
            role_groups[-1]["words"].append(wt)
            # For anagram fodder each clue word contributes its own
            # letters (A PRIME TV → A + PRIME + TV), so concatenate.
            # For synonyms / abbreviations the piece value (e.g. QUE
            # for "manuel's gag" → QUE) is shared by every word in the
            # group, so leave the existing letters alone — concatenating
            # would produce "QUE QUE".
            if letters and r == "anagram_fodder":
                if role_groups[-1].get("letters"):
                    role_groups[-1]["letters"] = (
                        role_groups[-1]["letters"] + " " + letters)
                else:
                    role_groups[-1]["letters"] = letters
            elif letters and not role_groups[-1].get("letters"):
                role_groups[-1]["letters"] = letters
        else:
            role_groups.append({
                "piece_key": piece_key,
                "role": r,
                "letters": letters,
                "words": [wt],
                "source": s,
            })
    # Display normalisation: clue words that contribute a single letter
    # to the answer all sit under the same display label, regardless of
    # whether the underlying mechanism is positional (first letter of X),
    # an abbreviation lookup (charge -> C), or a wordplay single_letter
    # entry. Without this, "Relative in charge of sound" -> SONIC shows
    # "in" as positional_source and "charge" as abbreviation -- visually
    # inconsistent for two single-letter contributions. Apply only to
    # display; auto/manual rows on disk keep their precise mechanical
    # role.
    _single_letter_eq = {"positional_source", "abbreviation",
                         "abbreviation_source", "single_letter",
                         "literal_source"}
    for grp in role_groups:
        if (grp.get("role") in _single_letter_eq
                and grp.get("letters")
                and len(grp["letters"]) == 1):
            grp["role"] = "single_letter"
    # For cryptic_definition clues, always show a single clean definition
    # row in the word-by-word section (the verifier writes messy unaccounted
    # rows for the cryptic-hint words which looks wrong). Also ensure the
    # admin panel has at least one row so the Accept button can appear.
    if (clue["wordplay_type"] or "") == "cryptic_definition" and clue["clue_text"]:
        _cd_words = clue["clue_text"].split()
        # Replace role_groups with a single clean "definition" row
        role_groups = [{
            "piece_key": None,
            "role": "definition",
            "letters": None,
            "contributes": None,
            "words": _cd_words,
            "source": "synthetic",
        }]
        # Reclassify all word_role_rows as "definition" so the admin
        # panel shows no "unaccounted" noise for CD clues.
        for _r in word_role_rows:
            _r["role"] = "definition"
        if not word_role_rows:
            word_role_rows = [{
                "word_index": 0,
                "word_text": _cd_words[0],
                "role": "definition",
                "source": "synthetic",
                "letters": None,
                "piece_key": None,
            }]
    clue_dict["word_roles"] = word_role_rows
    clue_dict["role_groups"] = role_groups

    # Per-piece enrichment-needed flag for the admin override panel.
    # We walk word_role_rows, group multi-word pieces (by piece_key or
    # consecutive 'definition' rows), do one DB lookup per group, and
    # attach an accept_target to the FIRST row of each group when the
    # lookup misses. The template renders an Accept button only on rows
    # that carry an accept_target.
    import sqlite3 as _sqlite3
    import re as _re_acc
    from pathlib import Path as _Path
    _PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
    answer_clean_for_def = _re_acc.sub(
        r"[^A-Z]", "", (clue["answer"] or "").upper())

    # Pre-scan the explanation for multi-word indicator brackets like
    # [reversal: "looking back"]. For each, find the consecutive clue
    # words that match the phrase and remember which row indices form
    # the phrase. The piece-grouping loop below uses this to merge
    # consecutive *_indicator rows that together carry a multi-word
    # bracket, so the Accept button surfaces the multi-word phrase
    # (e.g. "looking back" -> reversal) rather than the individual
    # single-word constituents that may already each be in DB.
    _ai_expl_for_groups = clue["ai_explanation"] or ""
    _multiword_by_index = {}  # word_index -> first_index in phrase span
    _ann_pat = _re_acc.compile(
        r'[\[;]\s*(\w+(?:\s+\w+)?)\s*:\s*["\']([^"\']+)["\']',
        _re_acc.IGNORECASE,
    )
    _lower_words = [
        (row["word_text"] or "").lower() for row in word_role_rows
    ]
    for _m in _ann_pat.finditer(_ai_expl_for_groups):
        _phrase_words = _re_acc.findall(
            r"[a-zA-Z]+(?:'[a-zA-Z]+)?",
            _m.group(2).lower(),
        )
        if len(_phrase_words) < 2:
            continue
        # Locate the contiguous run of clue word indices matching the
        # phrase. Use the first hit only.
        for _start in range(len(_lower_words) - len(_phrase_words) + 1):
            if all(_lower_words[_start + _j] == _phrase_words[_j]
                   for _j in range(len(_phrase_words))):
                for _j in range(len(_phrase_words)):
                    _multiword_by_index[_start + _j] = _start
                break

    piece_groups_for_accept = []
    cur = None
    for i, row in enumerate(word_role_rows):
        pk = row["piece_key"]
        rl = row["role"]
        # Multi-word indicator merge: two consecutive *_indicator rows
        # belong to the same group when they share a multi-word phrase
        # span discovered above.
        same_indicator_phrase = (
            cur is not None
            and rl and rl.endswith("_indicator")
            and (cur.get("role") or "").endswith("_indicator")
            and i in _multiword_by_index
            and cur.get("_phrase_start") == _multiword_by_index[i]
        )
        same_group = (
            cur is not None and (
                (pk is not None and pk == cur["piece_key"])
                or (pk is None and cur["piece_key"] is None
                    and rl == "definition" and cur["role"] == "definition")
                or same_indicator_phrase
            )
        )
        if same_group:
            cur["words"].append(row["word_text"])
        else:
            if cur:
                piece_groups_for_accept.append(cur)
            cur = {"first_index": i, "piece_key": pk, "role": rl,
                   "words": [row["word_text"]], "letters": row["letters"],
                   "_phrase_start": _multiword_by_index.get(i)}
    if cur:
        piece_groups_for_accept.append(cur)

    ref_acc = _sqlite3.connect(
        str(_PROJECT_ROOT / "data" / "cryptic_new.db"))
    accept_by_index = {}
    # Try both the phrase as joined from clue words AND the form with
    # the trailing possessive 's stripped (and vice versa). Without
    # this, the DB might have "oscar winner's" while the parse uses
    # "Oscar winner" (or vice versa) and the lookup misses on one
    # side -- mirrors the verifier's _phrase_variants logic so the
    # Accept button surfaces only when the pair is genuinely missing
    # in both forms.
    def _phrase_variants(p):
        p = (p or "").strip()
        yield p
        if p.lower().endswith("'s") or p.lower().endswith("’s"):
            yield p[:-2]
        else:
            yield p + "'s"

    for pg in piece_groups_for_accept:
        phrase = " ".join(pg["words"]).strip()
        role = pg["role"] or ""
        letters = pg["letters"]
        target = in_db = None
        if role in ("synonym", "synonym_source") and letters:
            target = ("synonym", phrase, letters.upper())
            for ph in _phrase_variants(phrase):
                if in_db:
                    break
                in_db = ref_acc.execute(
                    "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? "
                    "AND UPPER(synonym)=? LIMIT 1",
                    (ph.lower(), letters.upper())).fetchone()
        elif role in ("abbreviation", "abbreviation_source") and letters:
            target = ("abbreviation", phrase, letters.upper())
            for ph in _phrase_variants(phrase):
                if in_db:
                    break
                in_db = ref_acc.execute(
                    "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? "
                    "AND UPPER(substitution)=? LIMIT 1",
                    (ph.lower(), letters.upper())).fetchone()
        elif role == "definition":
            target = ("definition", phrase, answer_clean_for_def)
            for ph in _phrase_variants(phrase):
                if in_db:
                    break
                in_db = ref_acc.execute(
                    "SELECT 1 FROM definition_answers_augmented "
                    "WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
                    (ph.lower(), answer_clean_for_def)).fetchone()
        elif role.endswith("_indicator"):
            op_type = role[:-len("_indicator")]
            target = ("indicator", phrase, op_type)
            in_db = ref_acc.execute(
                "SELECT 1 FROM indicators WHERE LOWER(word)=? "
                "AND LOWER(wordplay_type)=? LIMIT 1",
                (phrase.lower(), op_type.lower())).fetchone()
        if target and not in_db:
            accept_by_index[pg["first_index"]] = target
    # Secondary scan: surface Accept buttons for parse claims that
    # the classifier's strict gate refused to claim (because the full
    # multi-word phrase is not in DB). Without this, FRAIL's parse
    # "RAIL (synonym='water bird')" leaves water/bird as unaccounted
    # word_role_rows and the Accept button never appears -- the user
    # has no way to add the enrichment from the clue page.
    #
    # Re-open the ref DB; we just closed it.
    ref_acc = _sqlite3.connect(
        str(_PROJECT_ROOT / "data" / "cryptic_new.db"))
    try:
        # Build a quick lookup so we can find the first clue-word
        # index for any source-phrase word.
        _lower_words_for_acc = [
            (row["word_text"] or "").lower() for row in word_role_rows
        ]
        # Bracket-form indicator claims like [parts: "occasionally"]
        # or [reversal: "looking back"]. Surface an Accept button on
        # the first matching clue word if the phrase isn't in DB.
        for _bm in _re_acc.finditer(
                r'[\[;]\s*(\w+(?:\s+\w+)?)\s*:\s*["\']([^"\']+)["\']',
                clue["ai_explanation"] or "",
                _re_acc.IGNORECASE):
            _op_type = _bm.group(1).strip().lower()
            _phrase = _bm.group(2).strip()
            if not _phrase:
                continue
            _r = ref_acc.execute(
                "SELECT 1 FROM indicators WHERE LOWER(word)=? "
                "AND LOWER(wordplay_type)=? LIMIT 1",
                (_phrase.lower(), _op_type)).fetchone()
            if _r:
                continue
            _phrase_tokens = _re_acc.findall(
                r"[a-zA-Z]+(?:'[a-zA-Z]+)?", _phrase.lower())
            if not _phrase_tokens:
                continue
            _first_tok = _phrase_tokens[0]
            for _i, _lw in enumerate(_lower_words_for_acc):
                if _lw == _first_tok:
                    _existing = accept_by_index.get(_i)
                    if (_existing is None
                            or (len(_phrase_tokens) > 1
                                and len(_existing[1].split()) == 1)):
                        accept_by_index[_i] = (
                            "indicator", _phrase, _op_type)
                        # Remove any single-word entries the primary scan
                        # set for the remaining tokens of this phrase.
                        for _j, _pt in enumerate(_phrase_tokens[1:], 1):
                            if _i + _j < len(_lower_words_for_acc):
                                _e = accept_by_index.get(_i + _j)
                                if _e and len(_e[1].split()) == 1:
                                    del accept_by_index[_i + _j]
                    break
        for _pat, _kind in [
            (r'(\w+)\s*\(\s*synonym\s*=\s*["\']([^"]+)["\']\s*\)',
             'synonym'),
            (r'(\w+)\s*\(\s*synonym\s+of\s+["\']([^"]+)["\']\s*\)',
             'synonym'),
            (r'(\w+)\s*\(\s*abbreviation\s*=\s*["\']([^"]+)["\']\s*\)',
             'abbreviation'),
            (r'(\w+)\s*\(\s*abbreviation\s+of\s+["\']([^"]+)["\']\s*\)',
             'abbreviation'),
            # Paren-less forms used inside deletion/container clauses, e.g.
            # RAIT (deletion="RABBIT", BB dropped, RABBIT synonym="inferior cricketer")
            (r'([A-Z]+)\s+synonym\s*=\s*["\']([^"]+)["\']',
             'synonym'),
            (r'([A-Z]+)\s+abbreviation\s*=\s*["\']([^"]+)["\']',
             'abbreviation'),
        ]:
            for _cm in _re_acc.finditer(
                    _pat, clue["ai_explanation"] or "",
                    _re_acc.IGNORECASE):
                _ltrs = _cm.group(1)
                _src = _cm.group(2)
                if not (_ltrs and _src):
                    continue
                # Letters must be uppercase (it's a piece value) and
                # at least one letter -- skips matches inside narrative.
                if not _re_acc.match(r"^[A-Z']+$", _ltrs):
                    continue
                # Skip if the source phrase has only one word -- the
                # primary scan above already handles single-word
                # claims via the synonym_source role.
                _phrase_tokens = _re_acc.findall(
                    r"[a-zA-Z]+(?:'[a-zA-Z]+)?", _src.lower())
                if len(_phrase_tokens) < 2:
                    continue
                # DB lookup with the trailing-'s variants.
                _hit = False
                for _ph in _phrase_variants(_src):
                    if _kind == 'synonym':
                        r = ref_acc.execute(
                            "SELECT 1 FROM synonyms_pairs WHERE "
                            "LOWER(word)=? AND UPPER(synonym)=? LIMIT 1",
                            (_ph.lower(), _ltrs.upper())).fetchone()
                    else:
                        r = ref_acc.execute(
                            "SELECT 1 FROM wordplay WHERE "
                            "LOWER(indicator)=? AND UPPER(substitution)=? "
                            "LIMIT 1",
                            (_ph.lower(), _ltrs.upper())).fetchone()
                    if r:
                        _hit = True
                        break
                if _hit:
                    continue
                # Locate the first clue word matching the start of the
                # source phrase. Attach an accept_target there so the
                # button appears on that word's admin row.
                _first_tok = _phrase_tokens[0]
                for _i, _lw in enumerate(_lower_words_for_acc):
                    if _lw == _first_tok:
                        _existing = accept_by_index.get(_i)
                        if (_existing is None
                                or (len(_phrase_tokens) > 1
                                    and len(_existing[1].split()) == 1)):
                            accept_by_index[_i] = (
                                _kind, _src, _ltrs.upper())
                        break
    finally:
        ref_acc.close()

    # CD: surface Accept button for the full clue text → answer definition.
    # The verifier's CD check requires this exact mapping in the DB.
    if ((clue["wordplay_type"] or "") == "cryptic_definition"
            and word_role_rows
            and 0 not in accept_by_index):
        _ref_cd = _sqlite3.connect(str(_PROJECT_ROOT / "data" / "cryptic_new.db"))
        try:
            _ct = (clue["clue_text"] or "").strip()
            _cd_hit = _ref_cd.execute(
                "SELECT 1 FROM definition_answers_augmented "
                "WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
                (_ct.lower(), answer_clean_for_def)).fetchone()
            if not _cd_hit:
                accept_by_index[0] = ("definition", _ct, answer_clean_for_def)
        finally:
            _ref_cd.close()

    for i, row in enumerate(word_role_rows):
        row["accept_target"] = accept_by_index.get(i)

    # Compute the per-piece "contributes" string — the actual letters
    # each piece puts into the answer at its final position. For most
    # pieces this equals letters (synonym MAN -> MAN, reversal MUR ->
    # MUR). For anagram fodder the fodder is what goes IN; the
    # contribution is the permutation that comes OUT in the answer
    # (ASIA fodder for SAMURAI contributes SAAI — the answer letters
    # left over after MUR is positioned). Heuristic: locate each
    # non-anagram piece's letters as a contiguous run in the answer;
    # remaining positions form the anagram contribution. If the
    # leftover multiset doesn't match the fodder, leave contributes
    # blank rather than guess.
    answer_for_contrib = _re_acc.sub(
        r"[^A-Z]", "", (clue["answer"] or "").upper())
    from web.coverage import (
        post_op_letters as _post_op_letters,
        effective_letters as _effective_letters,
        deletion_target_letters as _deletion_target_letters,
    )
    _ai_expl_for_contrib = clue["ai_explanation"] or ""
    # Identify letter groups consumed by a deletion clause in the
    # explanation. Pieces whose letters match a deletion target are
    # not contributions to the answer; they are the letter that was
    # removed. Surface them in the contributes column with the
    # brackets convention so the reader sees the deleted letter
    # without it being counted as an addition.
    _deletion_targets_remaining = list(
        _deletion_target_letters(_ai_expl_for_contrib))
    non_anagram_letters = []
    anagram_groups = []
    for grp in role_groups:
        rl = grp.get("role") or ""
        L = grp.get("letters")
        if rl == "anagram_fodder":
            anagram_groups.append(grp)
        elif L:
            eff = _effective_letters(L).upper()
            if eff and eff in _deletion_targets_remaining:
                grp["is_deletion_target"] = True
                _deletion_targets_remaining.remove(eff)
                continue
            non_anagram_letters.append(eff)
    used_positions = [False] * len(answer_for_contrib)
    # piece_contributions maps source-letters → post-operation letters
    # actually placed in the answer. Synonym RUM under a reversal claim
    # contributes MUR to SAMURAI; synonym ADMIRE under a deletion claim
    # contributes ADMIR to ADMIRAL. post_op_letters does the substitution.
    piece_contributions = {}
    for piece_letters in non_anagram_letters:
        effective = _post_op_letters(piece_letters, _ai_expl_for_contrib)
        placed = False
        for candidate in (effective, effective[::-1]):
            for i in range(len(answer_for_contrib) - len(candidate) + 1):
                if all(not used_positions[i + j]
                       for j in range(len(candidate))) \
                        and answer_for_contrib[i:i + len(candidate)] == candidate:
                    for j in range(len(candidate)):
                        used_positions[i + j] = True
                    piece_contributions[piece_letters] = candidate
                    placed = True
                    break
            if placed:
                break
    leftover = "".join(
        answer_for_contrib[i]
        for i in range(len(answer_for_contrib))
        if not used_positions[i]
    )
    # For hidden parses, derive per-word contribution by intersecting
    # each hidden_source word's letter positions with the hidden span.
    # E.g. INERTIA hidden in "Maître Nicolas" — "maitre" contributes
    # AITRE, "nicolas" contributes NI; together AITRENI = answer
    # reversed.
    import unicodedata as _unicode_acc
    def _ascii_fold(s):
        return _unicode_acc.normalize("NFKD", s or "").encode(
            "ascii", "ignore").decode("ascii")
    clue_text_for_hidden = _ascii_fold(clue["clue_text"] or "")
    expl_for_hidden = clue["ai_explanation"] or ""
    hidden_match = _re_acc.search(
        r'hidden(?:\s+reversed)?\s+in\s+["\']([^"\']+)["\']',
        expl_for_hidden, _re_acc.IGNORECASE)
    hidden_is_reversed = bool(
        _re_acc.search(r"hidden\s+reversed\s+in",
                       expl_for_hidden, _re_acc.IGNORECASE)
        or (clue_dict.get("wordplay_type") or "") == "hidden_reversed"
    )
    hidden_span_in_clue = None  # (start, end) over the stripped clue
    if hidden_match:
        span_raw = hidden_match.group(1)
        span_text = _ascii_fold(span_raw)
        span_letters = _re_acc.sub(r"[^a-z]", "", span_text.lower())
        clue_stripped = _re_acc.sub(
            r"[^a-z]", "", clue_text_for_hidden.lower())
        pos = clue_stripped.find(span_letters)
        if pos >= 0:
            # The full source region is at [pos, pos + len(span_letters)).
            # Inside the parse's span text, the answer letters are written
            # UPPERCASE. Find the upper-letter slice within the span and
            # narrow hidden_span_in_clue to that slice.
            upper_indices = [
                k for k, ch in enumerate(
                    _re_acc.sub(r"[^A-Za-z]", "", span_text))
                if ch.isupper()
            ]
            if upper_indices:
                hidden_span_in_clue = (
                    pos + upper_indices[0],
                    pos + upper_indices[-1] + 1,
                )
            else:
                hidden_span_in_clue = (pos, pos + len(span_letters))
    # Build per-word letter-offset ranges in the stripped clue.
    word_offsets = []  # list of (word_lower, start, end)
    cursor = 0
    clue_stripped_for_words = _re_acc.sub(
        r"[^a-z]", "", clue_text_for_hidden.lower())
    for tok in _re_acc.findall(
            r"[a-zA-Z]+(?:&[a-zA-Z]+)*(?:’[a-zA-Z]+)?",
            clue_text_for_hidden.lower().replace("’", "’")):
        tok_letters = _re_acc.sub(r"[^a-z]", "", tok)
        word_offsets.append((tok, cursor, cursor + len(tok_letters)))
        cursor += len(tok_letters)

    for grp in role_groups:
        rl = grp.get("role") or ""
        L = grp.get("letters")
        if rl == "anagram_fodder":
            fodder_letters = _re_acc.sub(r"[^A-Z]", "", (L or "").upper())
            if len(grp["words"]) > 1 and fodder_letters and sorted(leftover) == sorted(fodder_letters):
                grp["contributes"] = leftover
            elif fodder_letters:
                grp["contributes"] = fodder_letters
            else:
                grp["contributes"] = None
        elif rl == "hidden_source":
            if not hidden_span_in_clue:
                grp["contributes"] = None
                continue
            # Find the offset of this group's first word in the clue.
            grp_word = grp["words"][0].lower()
            word_range = next(
                ((s, e) for (w, s, e) in word_offsets
                 if w == grp_word), None)
            if not word_range:
                grp["contributes"] = None
                continue
            ws, we = word_range
            hs, he = hidden_span_in_clue
            ovl_start = max(ws, hs)
            ovl_end = min(we, he)
            if ovl_end > ovl_start:
                contrib_letters = clue_stripped_for_words[ovl_start:ovl_end].upper()
                # For hidden_reversed, the whole span is reversed before
                # placement in the answer — apply that reversal to each
                # word's contribution so the letters shown are the ones
                # landing in the answer (mirrors the RUM->MUR treatment).
                if hidden_is_reversed:
                    contrib_letters = contrib_letters[::-1]
                grp["contributes"] = contrib_letters
            else:
                grp["contributes"] = None
        elif grp.get("is_deletion_target"):
            # Deletion target piece — show its letters wrapped in
            # brackets so the reader sees the deleted letter without it
            # being counted as an addition to the answer.
            L_eff = _effective_letters(L or "").upper()
            grp["contributes"] = "(" + L_eff + ")" if L_eff else None
        else:
            # If the piece was actually placed reversed, show the reversed
            # form (the letters that landed in the answer).
            placed_contrib = piece_contributions.get(
                _effective_letters(L or "").upper())
            if placed_contrib is not None:
                grp["contributes"] = placed_contrib
            else:
                wptype = (clue_dict.get("wordplay_type") or "").lower()
                if wptype in ("homophone", "spoonerism") and L:
                    grp["contributes"] = _effective_letters(L or "").upper() or leftover or answer_for_contrib
                else:
                    grp["contributes"] = L

    # Render-time post-pass: surface a literal S piece sourced from a
    # possessive 's in the clue. The parser writes "+ S (from clue)" for
    # the S contributed by a "...'s" token, but the classifier has already
    # claimed that token under its primary role (typically synonym), so
    # the S never gets its own row.
    #
    # IMPORTANT (2026-05-14 fix): only surface the synthetic row if the
    # S is genuinely part of the answer. Yesterday's unguarded version
    # surfaced it whenever "S (from clue)" appeared in the parse,
    # papering over wrong parses (e.g. RIBCAGE = RIB + CAGE has no
    # trailing S, but the parse claimed "+ S (from clue)" and the
    # renderer dutifully showed an S row, making the wrong parse look
    # complete). The new guard sums the existing piece contribs and
    # only adds the synthetic S when (sum + 'S') matches the answer
    # multiset — i.e. the S really does belong.
    import re as _re
    expl_text = clue_dict.get("ai_explanation") or ""
    if _re.search(r"\bS\s*\(\s*from\s+clue\s*\)", expl_text):
        possessive_indices = [
            i for i, grp in enumerate(role_groups)
            if any(w.endswith("'s") or w.endswith("’s")
                   for w in grp["words"])
        ]
        if len(possessive_indices) == 1:
            # Sum existing piece contributions. Mirrors the coverage
            # helper's logic: dedupe by piece_key, skip non-source roles,
            # but include the contributes column when present so deletion
            # post-op letters are correctly counted.
            _seen_pk = set()
            _existing = ""
            for grp in role_groups:
                rl = grp.get("role") or ""
                if rl not in (
                    "synonym", "synonym_source",
                    "abbreviation", "abbreviation_source",
                    "single_letter", "positional_source",
                    "reversal_source", "deletion_source",
                    "anagram_fodder", "literal_source",
                ):
                    continue
                pk = grp.get("piece_key")
                if pk is not None:
                    if pk in _seen_pk:
                        continue
                    _seen_pk.add(pk)
                # contributes is the post-op letters that land in
                # the answer; fall back to letters when no contributes
                # was computed (e.g. some hidden_source rows).
                contrib = grp.get("contributes") or grp.get("letters") or ""
                _existing += _re.sub(r"[^A-Z]", "", str(contrib).upper())
            _ans_letters = _re.sub(
                r"[^A-Z]", "", (clue["answer"] or "").upper())
            # Only surface the synthetic row when adding S makes the
            # piece-letter multiset equal the answer multiset.
            if (sorted(_existing + "S") == sorted(_ans_letters)
                    and sorted(_existing) != sorted(_ans_letters)):
                host_i = possessive_indices[0]
                host_words = role_groups[host_i]["words"]
                role_groups.insert(host_i + 1, {
                    "piece_key": None,
                    "role": "possessive_source",
                    "letters": "S",
                    "contributes": "S",
                    "words": host_words,
                    "source": "synthetic",
                })

    # Auto-generate the mechanism label from the role groups.
    # "Charade" is added when 2+ distinct pieces concatenate; each
    # distinct indicator type (anagram, container, reversal, deletion,
    # homophone, hidden, ...) contributes its name. Examples:
    #   THREATS  : 2 pieces + container_indicator -> "CHARADE CONTAINER"
    #   THEATRES : 1 piece + anagram_indicator     -> "ANAGRAM"
    #   DESIREE  : 2 pieces, no indicator          -> "CHARADE"
    #   PIECE    : 1 piece + homophone_indicator   -> "HOMOPHONE"
    # Falls back to the stored wordplay_type (uppercased, underscores
    # to spaces) when no role data is available.
    def _mechanism_label(groups, fallback):
        pieces = set()
        indicator_types = []
        for g in groups:
            if g.get("piece_key") is not None:
                pieces.add(g["piece_key"])
            role = g.get("role") or ""
            if role.endswith("_indicator"):
                base = role[:-len("_indicator")]
                if base and base not in indicator_types:
                    indicator_types.append(base)
        parts = []
        if len(pieces) >= 2:
            parts.append("charade")
        parts.extend(indicator_types)
        if not parts:
            return (fallback or "").replace("_", " ").upper() or None
        return " ".join(parts).replace("_", " ").upper()

    clue_dict["mechanism_label"] = _mechanism_label(
        role_groups, clue_dict.get("wordplay_type"),
    )

    from web.coverage import coverage_warning as _coverage_warning
    clue_dict["coverage_warning"] = _coverage_warning(
        clue["id"], clue["answer"], clue_dict.get("wordplay_type"),
        clue_dict.get("tier"),
        ai_explanation=clue_dict.get("ai_explanation"),
    )

    # Replace the wordplay_type inline-hint content with the richer
    # mechanism_label when we computed one. Falls back silently when
    # no role data exists for the clue (mechanism_label is None or
    # empty in that case).
    if clue_dict["mechanism_label"]:
        for h in inline_hints:
            if h["type"] == "wordplay_type":
                h["content"] = clue_dict["mechanism_label"]
                h["is_mechanism_label"] = True
                break

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

    # Prev/next within the same puzzle — across by clue_number first,
    # then down by clue_number. Admin-only navigation lives on the
    # action row so reviewers can step through every clue without
    # bouncing back to the puzzle page.
    sibling_clues = db.execute(
        """SELECT id, clue_text, clue_number, direction
           FROM clues
           WHERE source = ? AND puzzle_number = ?
           ORDER BY CASE WHEN direction='across' THEN 0 ELSE 1 END,
                    CAST(clue_number AS INTEGER)""",
        (source, puzzle_number),
    ).fetchall()
    prev_id = next_id = None
    for i, sib in enumerate(sibling_clues):
        if sib["id"] == clue_id:
            if i > 0:
                prev_id = sibling_clues[i - 1]["id"]
            if i + 1 < len(sibling_clues):
                next_id = sibling_clues[i + 1]["id"]
            break
    def _sibling_url(sib_id):
        if not sib_id:
            return None
        sib = next((s for s in sibling_clues if s["id"] == sib_id), None)
        if not sib:
            return None
        return f"/clue/{generate_clue_slug(sib['clue_text'], clue_id=sib_id)}"
    clue_dict["prev_url"] = _sibling_url(prev_id)
    clue_dict["next_url"] = _sibling_url(next_id)

    # Other appearances of the same clue
    other_appearances = []
    for other in matches:
        o_source = other["source"]
        o_pnum = other["puzzle_number"]
        o_pub = other["publication_date"] if "publication_date" in other.keys() else None
        o_type_slug, o_type_label = classify_puzzle(o_source, o_pnum, o_pub)
        o_slug = generate_clue_slug(other["clue_text"], clue_id=other["id"])
        other_appearances.append({
            "source": o_source,
            "puzzle_number": o_pnum,
            "publication_date": o_pub,
            "type_label": o_type_label,
            "answer": other["answer"],
            "clue_url": f"/clue/{o_slug}",
        })

    # SEO data
    meta_description = generate_meta_description(clue_dict)
    faq_schema = generate_faq_schema(clue_dict, steps)
    breadcrumb_schema = generate_breadcrumb_schema(clue_dict)
    word_roles_schema = generate_word_roles_schema(
        clue_dict, role_groups, clue_dict.get("mechanism_label"),
    )

    from web.models import get_source_puzzle_url
    source_puzzle_url = get_source_puzzle_url(source, puzzle_number)

    role_choices = sorted([
        # Structural
        "definition",
        "link",
        "charade_joiner",
        "dbe_marker",
        "unaccounted",
        # Source roles (clue word produces letters)
        "synonym_source",
        "abbreviation_source",
        "positional_source",
        "reversal_source",
        "deletion_source",
        "literal_source",
        "letter_source",
        "possessive_source",
        "anagram_fodder",
        "spoonerism_fodder",
        "hidden_source",
        # Mechanism indicators
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
        "substitution_indicator",
        "letter_position_indicator",
        # Legacy generic indicator
        "indicator",
    ])

    # Detect homophone pairs from the explanation that are missing from
    # the homophones table and surface them as Accept buttons.
    # Pattern: "ANSWER sounds like WORD" anywhere in the explanation.
    import re as _re_hp
    import sqlite3 as _sqlite3_hp
    _homophone_accepts = []
    _expl_for_hp = clue_dict.get("ai_explanation") or ""
    _ref_hp = _sqlite3_hp.connect(
        str(_Path(__file__).resolve().parent.parent.parent / "data" / "cryptic_new.db"),
        timeout=5)
    try:
        for _m in _re_hp.finditer(
                r"\b([A-Z]+)\s+sounds\s+like\s+([A-Z]+)\b", _expl_for_hp):
            _answer_word = _m.group(1).lower()
            _source_word = _m.group(2).lower()
            _in_db = _ref_hp.execute(
                "SELECT 1 FROM homophones WHERE "
                "(LOWER(word)=? AND LOWER(homophone)=?) OR "
                "(LOWER(word)=? AND LOWER(homophone)=?) LIMIT 1",
                (_answer_word, _source_word, _source_word, _answer_word),
            ).fetchone()
            if not _in_db:
                _homophone_accepts.append(
                    (_m.group(1), _m.group(2)))  # (ANSWER, SOURCE) uppercase
    finally:
        _ref_hp.close()

    response = make_response(render_template(
        "clue.html",
        clue=clue_dict,
        other_appearances=other_appearances,
        source_puzzle_url=source_puzzle_url,
        meta_description=meta_description,
        faq_schema=faq_schema,
        breadcrumb_schema=breadcrumb_schema,
        word_roles_schema=word_roles_schema,
        role_choices=role_choices,
        homophone_accepts=_homophone_accepts,
    ))
    return issue_session_cookie(response)
