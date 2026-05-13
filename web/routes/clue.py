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
    piece_groups_for_accept = []
    cur = None
    for i, row in enumerate(word_role_rows):
        pk = row["piece_key"]
        rl = row["role"]
        same_group = (
            cur is not None and (
                (pk is not None and pk == cur["piece_key"])
                or (pk is None and cur["piece_key"] is None
                    and rl == "definition" and cur["role"] == "definition")
            )
        )
        if same_group:
            cur["words"].append(row["word_text"])
        else:
            if cur:
                piece_groups_for_accept.append(cur)
            cur = {"first_index": i, "piece_key": pk, "role": rl,
                   "words": [row["word_text"]], "letters": row["letters"]}
    if cur:
        piece_groups_for_accept.append(cur)

    ref_acc = _sqlite3.connect(
        str(_PROJECT_ROOT / "data" / "cryptic_new.db"))
    accept_by_index = {}
    for pg in piece_groups_for_accept:
        phrase = " ".join(pg["words"]).strip()
        role = pg["role"] or ""
        letters = pg["letters"]
        target = in_db = None
        if role in ("synonym", "synonym_source") and letters:
            target = ("synonym", phrase, letters.upper())
            in_db = ref_acc.execute(
                "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? "
                "AND UPPER(synonym)=? LIMIT 1",
                (phrase.lower(), letters.upper())).fetchone()
        elif role in ("abbreviation", "abbreviation_source") and letters:
            target = ("abbreviation", phrase, letters.upper())
            in_db = ref_acc.execute(
                "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? "
                "AND UPPER(substitution)=? LIMIT 1",
                (phrase.lower(), letters.upper())).fetchone()
        elif role == "definition":
            target = ("definition", phrase, answer_clean_for_def)
            in_db = ref_acc.execute(
                "SELECT 1 FROM definition_answers_augmented "
                "WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
                (phrase.lower(), answer_clean_for_def)).fetchone()
        elif role.endswith("_indicator"):
            op_type = role[:-len("_indicator")]
            target = ("indicator", phrase, op_type)
            in_db = ref_acc.execute(
                "SELECT 1 FROM indicators WHERE LOWER(word)=? "
                "AND LOWER(wordplay_type)=? LIMIT 1",
                (phrase.lower(), op_type.lower())).fetchone()
        if target and not in_db:
            accept_by_index[pg["first_index"]] = target
    ref_acc.close()

    for i, row in enumerate(word_role_rows):
        row["accept_target"] = accept_by_index.get(i)

    # Render-time post-pass: surface a literal S piece sourced from a
    # possessive 's in the clue. The parser writes "+ S (from clue)" for
    # the S contributed by a "...'s" token, but the classifier has already
    # claimed that token under its primary role (typically synonym), so
    # the S never gets its own row. We add a synthetic possessive row
    # next to the host token so the word-by-word panel shows every piece
    # the wordplay uses. No DB write — purely a display layer.
    import re as _re
    expl_text = clue_dict.get("ai_explanation") or ""
    if _re.search(r"\bS\s*\(\s*from\s+clue\s*\)", expl_text):
        possessive_indices = [
            i for i, grp in enumerate(role_groups)
            if any(w.endswith("'s") or w.endswith("’s")
                   for w in grp["words"])
        ]
        if len(possessive_indices) == 1:
            host_i = possessive_indices[0]
            host_words = role_groups[host_i]["words"]
            role_groups.insert(host_i + 1, {
                "piece_key": None,
                "role": "possessive_source",
                "letters": "S",
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

    response = make_response(render_template(
        "clue.html",
        clue=clue_dict,
        other_appearances=other_appearances,
        source_puzzle_url=source_puzzle_url,
        meta_description=meta_description,
        faq_schema=faq_schema,
        breadcrumb_schema=breadcrumb_schema,
        word_roles_schema=word_roles_schema,
    ))
    return issue_session_cookie(response)
