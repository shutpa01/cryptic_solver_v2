"""Reference-DB queries behind the four tools.

Mirrors the proven behaviour of the site's `/helper/*` endpoints without
importing them — see `publisher_build_decisions`; this package stays liftable,
so the duplication is the decision.

Two rules hold everywhere in this module:

* **Results are capped**, and the cap is returned alongside so the caller can
  say "showing 100 of more" rather than implying it found exactly 100. The
  licensing design accepts that an endpoint answering arbitrary queries can be
  walked; the job is to make walking it slow, bounded and visible.
* **Nothing is invented.** Every word returned came out of the curated tables.
  A tool that pads its results with a generic dictionary would destroy the
  thing the whole product is sold on. The one addition is the puzzle's own
  answer, put into the pattern shortlist on purpose (see `choose_options`) —
  it comes from the feed, not from a dictionary, and never appears for a
  puzzle whose solution we do not hold.
"""

import hashlib
import re
import sqlite3
import threading

RESULT_CAP = 100
SYNONYM_CAP = 60
ABBREVIATION_CAP = 20

# The shortlist an entry is offered. Mirrors Config.MATCH_OPTIONS, which is
# what the routes actually pass; this is only the default for direct callers.
OPTION_LIMIT = 9

_anagram_lock = threading.Lock()
_anagram_index = {}     # length -> {signature: set(display)}


def _connect(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _clean(word):
    return re.sub(r"[^A-Z]", "", (word or "").upper())


def normalise_key(text):
    """Collapse a lookup key the way the reference DB's stored keys are built.

    A VERBATIM copy of `signature_solver.db._normalize_key`, duplicated rather
    than imported because this package stays liftable (see
    `publisher_build_decisions`). It must not drift: the DB's `norm_word` and
    `norm_def` columns were written with that function, and a key built any
    other way silently matches nothing.

    Word-joining punctuation becomes a space, so 'pen-pushers' meets 'pen
    pushers'; other punctuation is dropped, so "Jill's companion" meets
    'jills companion'.
    """
    text = re.sub(r"[-‐-―/]+", " ", (text or "").lower())
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def signature(word):
    """Sorted letters — the anagram fingerprint."""
    return "".join(sorted(_clean(word)))


def _variants(word_lower):
    """The word plus the plural and possessive forms worth also trying.

    Used by the SYNONYM tool only. Weaker than `match_variants` below on
    purpose for now: widening what a typed word can mean is a separate question
    from what a clue word can DO, and it has not been measured. See the note on
    `indicator_roles`.
    """
    out = [word_lower]
    if len(word_lower) >= 4 and word_lower.endswith("s") and not word_lower.endswith("ss"):
        out.append(word_lower[:-1])
    if word_lower.endswith("'s"):
        out.append(word_lower[:-2])
    return out


def match_variants(text):
    """Every form to try against the reference tables, original first.

    The SOLVER matches a clue word against its regular inflections and its
    contraction/possessive forms — `core/engine_registry.py:185` unions the
    indicator types over `_match_variants`. Asking a narrower question here
    makes the panel silent about a role the card is showing: 2026-08-25,
    "maintain" is the container indicator in Telegraph 31323 1 across while the
    table holds only "maintains" and "maintaining".

    `core.inflect` and `core.contractions` are IMPORTED, not copied like
    `normalise_key` above. Both are leaf modules with no imports of their own,
    and the rule this package keeps is that it imports nothing from `web/`.
    Copying inflection rules that must agree with the engine's, exactly, is how
    this hole would reopen quietly.
    """
    from core import contractions, inflect

    text = (text or "").replace("’", "'").replace("‘", "'")
    out = []

    def add(candidate):
        for variant in inflect.phrase_variants(candidate):
            if variant not in out:
                out.append(variant)

    add(text)
    for form in contractions.forms(text):
        add(form)
    return out


def indicator_roles(db, word, word_lower):
    """What this clue word can DO, across the forms the engine would try.

    Variant order, the word as written first, each role tagged with the form
    that matched. `form` is empty when it is the word itself; when it is not,
    the panel says so rather than implying a row that does not exist.
    """
    out, seen = [], set()
    for variant in match_variants(word):
        key = normalise_key(variant)
        if not key:
            continue
        for r in db.execute(
            "SELECT word, wordplay_type, subtype FROM indicators "
            "WHERE norm_word = ? ORDER BY wordplay_type", (key,),
        ).fetchall():
            ident = (r["wordplay_type"], r["subtype"])
            if ident in seen:
                continue
            seen.add(ident)
            out.append({
                "type": (r["wordplay_type"] or "").replace("_", " ").title(),
                "subtype": (r["subtype"] or "").replace("_", " ")
                           if r["subtype"] and r["subtype"] != "general" else "",
                # The ROW'S OWN word, not the variant we generated to find it:
                # stripping the "s" off "Parisian's" gives the stem "parisian'",
                # which is how the engine gets there but is not a word to show.
                "form": "" if key == word_lower else (r["word"] or variant),
            })
    return out


# --- word lookup (clicking a word in the clue) ---------------------------

def lookup(ref_db, word, letters=None, entry_length=None):
    """Everything the reference DB knows about one clue word.

    Returns could-mean values grouped by length (the shape a solver scans),
    plus indicator roles, abbreviations and homophones.

    `entry_length` expands the one group that fits the entry the solver is
    actually on. Every other length stays trimmed to five. Showing five of
    thirty-eight six-letter candidates and no way to see the rest is a dead end
    precisely where the answer must be — and we know the length, so there is no
    reason to make them guess.

    Keyed on `norm_word` / `norm_def`, which is how our own solver reads these
    tables (`core/live_db.py:79`). Matching on LOWER(word) instead misses every
    row whose key differs from its display form — 39,265 synonym rows, among
    them "Jill's companion" -> JACK. If the solver could find it, so must this:
    the clue was solved with these very tables.
    """
    word_lower = normalise_key(word)
    if not word_lower:
        return {"word": word, "meanings": [], "indicators": [],
                "abbreviations": [], "homophones": []}

    db = _connect(ref_db)
    try:
        if letters:
            rows = db.execute(
                """SELECT DISTINCT val FROM (
                       SELECT UPPER(synonym) AS val FROM synonyms_pairs
                       WHERE norm_word = ? AND LENGTH(REPLACE(synonym,' ','')) = ?
                       UNION
                       SELECT UPPER(answer) AS val FROM definition_answers_augmented
                       WHERE norm_def = ? AND LENGTH(REPLACE(answer,' ','')) = ?
                   ) ORDER BY val LIMIT ?""",
                (word_lower, letters, word_lower, letters, RESULT_CAP),
            ).fetchall()
            meanings = [{"length": letters, "words": [r["val"] for r in rows], "more": 0}]
        else:
            rows = db.execute(
                """SELECT DISTINCT val, LENGTH(REPLACE(val,' ','')) AS len FROM (
                       SELECT UPPER(synonym) AS val FROM synonyms_pairs
                       WHERE norm_word = ?
                       UNION
                       SELECT UPPER(answer) AS val FROM definition_answers_augmented
                       WHERE norm_def = ?
                   ) ORDER BY LENGTH(REPLACE(val,' ','')), val""",
                (word_lower, word_lower),
            ).fetchall()
            by_length = {}
            for r in rows:
                by_length.setdefault(r["len"], []).append(r["val"])
            meanings = []
            for length in sorted(by_length):
                words = sorted(by_length[length])
                shown = RESULT_CAP if length == entry_length else 5
                meanings.append({
                    "length": length,
                    "words": words[:shown],
                    "more": max(0, len(words) - shown),
                    "fits": length == entry_length,
                })

        indicators = indicator_roles(db, word, word_lower)

        abbreviation_sql = ("SELECT DISTINCT substitution FROM wordplay "
                            "WHERE norm_ind = ?")
        if letters:
            abbreviations = [r["substitution"] for r in db.execute(
                abbreviation_sql + " AND LENGTH(substitution) = ? "
                "ORDER BY substitution LIMIT 15", (word_lower, letters)).fetchall()]
        else:
            abbreviations = [r["substitution"] for r in db.execute(
                abbreviation_sql + " ORDER BY LENGTH(substitution), substitution "
                "LIMIT 15", (word_lower,)).fetchall()]

        homophone_sql = ("SELECT DISTINCT homophone FROM homophones "
                         "WHERE norm_word = ?")
        if letters:
            homophones = [r["homophone"] for r in db.execute(
                homophone_sql + " AND LENGTH(homophone) = ? ORDER BY homophone "
                "LIMIT 10", (word_lower, letters)).fetchall()]
        else:
            homophones = [r["homophone"] for r in db.execute(
                homophone_sql + " ORDER BY homophone LIMIT 10",
                (word_lower,)).fetchall()]
    finally:
        db.close()

    return {
        "word": word,
        "meanings": meanings,
        "indicators": indicators,
        "abbreviations": abbreviations,
        "homophones": homophones,
    }


def word_info(ref_db, clues_db, word, limit=8):
    """What a RESULT word means — the ⓘ beside a match.

    The reverse of the word lookup: not "what can this clue word become" but
    "if I put this word in the grid, what would it be doing there". A solver
    scanning nine words that all fit needs that to choose between them, and
    without it the list is nine strings.

    Mirrors the site's `/helper/word-info` (`web/routes/helper.py:528`): the
    reverse synonym lookup, plus the definitions that answer to this word,
    shortest first.
    """
    clean = (word or "").strip().upper()
    if not clean or len(clean) < 2:
        return {"word": word, "meanings": []}

    variants = [clean]
    if " " in clean:
        variants.append(clean.replace(" ", ""))
    elif len(clean) > 5:
        # A solid entry may be a spaced answer in the corpus. The site looks the
        # spacing up rather than guessing where the break goes.
        db = _connect(clues_db)
        try:
            row = db.execute(
                "SELECT answer FROM clues WHERE UPPER(REPLACE(answer,' ','')) = ? "
                "AND answer LIKE '% %' LIMIT 1", (clean,)).fetchone()
        finally:
            db.close()
        if row:
            variants.append(row["answer"].upper())

    meanings = set()
    db = _connect(ref_db)
    try:
        for variant in variants:
            for row in db.execute(
                    "SELECT DISTINCT LOWER(word) AS v FROM synonyms_pairs "
                    "WHERE UPPER(synonym) = ? LIMIT 20", (variant,)).fetchall():
                meanings.add(row["v"])
            for row in db.execute(
                    "SELECT DISTINCT LOWER(definition) AS v FROM "
                    "definition_answers_augmented WHERE UPPER(answer) = ? LIMIT 20",
                    (variant,)).fetchall():
                meanings.add(row["v"])
    finally:
        db.close()

    # Shortest first: a one-word gloss tells a solver more than a phrase.
    return {"word": clean, "meanings": sorted(meanings, key=len)[:limit]}


# --- synonym tool --------------------------------------------------------

def synonyms(ref_db, word, length=None, include=""):
    """Synonyms in both directions, plus abbreviations.

    Both directions because a solver does not care which way round the pair
    happened to be stored.
    """
    raw = (word or "").strip()
    if not raw or len(raw) > 50:
        return {"word": raw, "synonyms": [], "abbreviations": [], "capped": False}

    # Normalised keys, as the solver uses (see `lookup`): a typed apostrophe or
    # hyphen must not be the reason a known phrase comes back empty.
    word_lower = normalise_key(raw)
    word_upper = raw.upper().strip(".,;:!?\"'()-")

    db = _connect(ref_db)
    found = set()
    abbreviations = set()
    try:
        for variant in _variants(word_lower):
            for r in db.execute(
                "SELECT DISTINCT UPPER(synonym) AS s FROM synonyms_pairs "
                "WHERE norm_word = ?", (variant,)):
                found.add(r["s"])
            for r in db.execute(
                "SELECT DISTINCT UPPER(answer) AS a FROM definition_answers_augmented "
                "WHERE norm_def = ?", (variant,)):
                found.add(r["a"])
            for r in db.execute(
                "SELECT DISTINCT UPPER(substitution) AS s FROM wordplay "
                "WHERE norm_ind = ?", (variant,)):
                if r["s"]:
                    abbreviations.add(r["s"])

        for target in {word_upper, word_upper.replace(" ", "")}:
            for r in db.execute(
                "SELECT DISTINCT UPPER(word) AS w FROM synonyms_pairs "
                "WHERE UPPER(synonym) = ?", (target,)):
                found.add(r["w"])
            for r in db.execute(
                "SELECT DISTINCT UPPER(definition) AS d FROM definition_answers_augmented "
                "WHERE UPPER(answer) = ?", (target,)):
                found.add(r["d"])
    finally:
        db.close()

    found.discard(word_upper)
    found.discard(word_upper.replace(" ", ""))

    if length:
        found = {s for s in found if len(re.sub(r"[^A-Z]", "", s)) == length}
        abbreviations = {a for a in abbreviations if len(a) == length}

    wanted = _clean(include)
    if wanted:
        found = {s for s in found if _contains_all(s, wanted)}
        abbreviations = {a for a in abbreviations if _contains_all(a, wanted)}

    ordered = sorted(found, key=lambda s: (len(s), s))
    return {
        "word": raw,
        "synonyms": ordered[:SYNONYM_CAP],
        "abbreviations": sorted(abbreviations, key=len)[:ABBREVIATION_CAP],
        "capped": len(ordered) > SYNONYM_CAP,
    }


def _contains_all(candidate, wanted):
    """Order-independent multiset containment — for known letters, unknown spots."""
    letters = list(_clean(candidate))
    for ch in wanted:
        if ch in letters:
            letters.remove(ch)
        else:
            return False
    return True


# --- pattern tool --------------------------------------------------------

def fits_pattern(word, normalised, enumeration=None):
    """True when `word` could sit in a grid pattern of letters and dots.

    `normalised` is corpus.normalise_pattern output — upper case, dots for the
    squares still empty.
    """
    if not word or not normalised:
        return False
    clean = _clean(word)
    if len(clean) != len(normalised):
        return False
    if not re.match("^" + normalised.replace(".", "[A-Z]") + "$", clean):
        return False
    parts = re.findall(r"\d+", enumeration or "")
    if parts and sum(int(p) for p in parts) != len(clean):
        parts = []          # the enumeration describes something else; ignore it
    return _enum_ok(word, parts) if parts else True


def _sample_order(words, seed):
    """A fixed, arbitrary order for `words` — the same one every time.

    The fillers that sit beside the answer must not be the alphabetically
    first nine, or the answer stands out the moment it is not one of them.
    Neither may they reshuffle: open the same entry twice and get two
    different shortlists and the widget looks like it is guessing. Hashing
    each word with the pattern gives both — spread, and stability.
    """
    return sorted(words, key=lambda w: hashlib.md5(
        (seed + "|" + w).encode("utf-8")).hexdigest())


def choose_options(matches, answer=None, limit=OPTION_LIMIT, seed=""):
    """The words an entry is offered: at most `limit`, alphabetical, and with
    `answer` among them whenever there is one.

    The answer is put in deliberately. Above the limit the list is a shortlist,
    not a tally, and a shortlist that could exclude the answer would send the
    solver away from it — the opposite of the help the chip promises. `answer`
    is None for a puzzle whose solution we do not hold (an embargoed prize),
    and then nothing is added: the widget never serves an answer it has been
    refused elsewhere.

    Nothing else is invented — every filler came out of the curated corpus.
    """
    pool = [w for w in matches if w != answer]
    room = limit - 1 if answer else limit
    chosen = _sample_order(pool, seed)[:max(room, 0)]
    if answer:
        chosen.append(answer)
    return sorted(chosen)


def pattern_matches(corpus_module, clues_db, ref_db, pattern, enumeration=None,
                    include="", answer=None, limit=OPTION_LIMIT, ceiling=None):
    """Words fitting a grid pattern, listed.

    Runs against the same corpus as the match count so the two can never
    disagree — a count of 34 above a list of 12 would read as broken.

    `total` stays honest about how many words really fit; `matches` is the
    shortlist the solver is shown, which is capped at `limit` and always holds
    the answer. The scan is deliberately NOT bounded by default: the panel
    prints "showing 9 of 41", and a bounded scan would print the bound —
    "showing 9 of 200" is a number the widget cannot stand behind. Pass
    `ceiling` only where a floor is acceptable.
    """
    normalised = corpus_module.normalise_pattern(pattern)
    if normalised is None:
        return {"matches": [], "capped": False, "total": 0}

    by_length = corpus_module.load(clues_db, ref_db)
    bucket = by_length.get(len(normalised)) or []

    enum_parts = re.findall(r"\d+", enumeration or "")
    if enum_parts and sum(int(p) for p in enum_parts) != len(normalised):
        enum_parts = []

    regex = re.compile("^" + normalised.replace(".", "[A-Z]") + "$")
    wanted = _clean(include)

    seen = {}
    hit_ceiling = False
    for clean, display in bucket:
        if clean in seen or not regex.match(clean):
            continue
        if enum_parts and not _enum_ok(display, enum_parts):
            continue
        if wanted and not _contains_all(clean, wanted):
            continue
        seen[clean] = display
        if ceiling and len(seen) >= ceiling:
            hit_ceiling = True
            break

    # An answer that does not fit the pattern is not offered. The solver has a
    # wrong letter in the grid, and quietly listing the answer anyway would
    # hand them a free Check — Check is what the paper sells.
    if answer and not fits_pattern(answer, normalised, enumeration):
        answer = None
    if answer and wanted and not _contains_all(answer, wanted):
        answer = None       # the solver's own filter wins over the shortcut

    total = len(seen)
    if answer and answer not in seen.values():
        total += 1          # never report a total below the list length
    chosen = choose_options(list(seen.values()), answer, limit, normalised)
    return {
        "matches": chosen,
        "capped": total > len(chosen) or hit_ceiling,
        "total": total,
    }


def _enum_ok(display, enum_parts):
    words = display.split()
    if len(enum_parts) == 1:
        return len(words) == 1
    if len(words) != len(enum_parts):
        return False
    return all(len(w) == int(p) for w, p in zip(words, enum_parts))


# --- anagram tool --------------------------------------------------------

def _build_anagram_index(corpus_module, clues_db, ref_db, length):
    by_length = corpus_module.load(clues_db, ref_db)
    index = {}
    for clean, display in by_length.get(length) or []:
        index.setdefault("".join(sorted(clean)), set()).add(display)
    return index


def anagrams(corpus_module, clues_db, ref_db, letters, grid_pattern=None):
    """Anagrams of `letters`, optionally filtered by the grid pattern.

    That composition is the point: the fodder comes from the words the solver
    picked out of the clue, and the letters already in the grid narrow it. No
    paper offers it — the Guardian's own Anagram Helper makes you type the
    fodder by hand and ignores the grid entirely.
    """
    parts = [p for p in re.split(r"[-,\s]+", (letters or "").upper()) if _clean(p)]
    parts = [_clean(p) for p in parts]
    all_letters = "".join(parts)
    if len(all_letters) < 2 or len(all_letters) > 25:
        return {"matches": [], "capped": False, "total": 0, "letters": all_letters}

    length = len(all_letters)
    with _anagram_lock:
        if length not in _anagram_index:
            _anagram_index[length] = _build_anagram_index(
                corpus_module, clues_db, ref_db, length)
        index = _anagram_index[length]

    found = set(index.get("".join(sorted(all_letters)), ()))
    found = {f for f in found if _clean(f) != all_letters}

    if len(parts) > 1:
        enum_parts = [str(len(p)) for p in parts]
        found = {f for f in found if _enum_ok(f, enum_parts)}

    if grid_pattern:
        normalised = corpus_module.normalise_pattern(grid_pattern)
        if normalised and len(normalised) == length:
            regex = re.compile("^" + normalised.replace(".", "[A-Z]") + "$")
            found = {f for f in found if regex.match(_clean(f))}

    ordered = sorted(found)
    return {
        "matches": ordered[:RESULT_CAP],
        "capped": len(ordered) > RESULT_CAP,
        "total": len(ordered),
        "letters": all_letters,
    }


def invalidate():
    global _anagram_index
    with _anagram_lock:
        _anagram_index = {}
