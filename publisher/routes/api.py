"""The served side of the widget: match counts, check, reveal, the four tools,
the hint ladder, and token renewal.

Everything here is same-origin (the widget is an iframe on our own domain), so
there is no CORS layer. Every route is scoped to the one puzzle its token was
minted for.
"""

import threading

from flask import Blueprint, current_app, g, jsonify, request

from publisher import corpus, explanations, reference
from publisher.auth import (
    AuthError, mint_token, require_token, scoped_to, token_age_ok_for_renew,
)
from publisher.puzzles import PuzzleNotFound, build_model

bp = Blueprint("api", __name__)

# Parsed puzzles, keyed by (source, number). Small, read-only and identical for
# every solver, so one copy per process serves everyone.
_cache = {}
_cache_lock = threading.Lock()


def _load(source, number):
    """Return (model, solutions, letters) for a puzzle, parsed once per process.

    The flattened letter map is cached HERE, against the same key as the model,
    rather than being looked up by a field inside the model. Deriving the cache
    key from model contents is what let a bad `number` field collapse every
    puzzle onto one cache entry and serve the wrong answers.
    """
    key = (source, str(number))
    with _cache_lock:
        if key not in _cache:
            model, solutions = build_model(source, number)
            _cache[key] = (model, solutions, _flatten_solutions(model, solutions))
        return _cache[key]


def _flatten_solutions(model, solutions):
    """Per-entry solutions -> {"r,c": "A"}."""
    out = {}
    for entry in model["entries"]:
        answer = solutions.get(entry["id"])
        if not answer or len(answer) != len(entry["cells"]):
            continue
        for (r, c), letter in zip(entry["cells"], answer):
            out[f"{r},{c}"] = letter
    return out


def _entry_map(model):
    return {e["id"]: e for e in model["entries"]}


def _scoped_puzzle():
    """The puzzle this request's token allows, or an error response."""
    source = g.pub["s"]
    number = g.pub["n"]
    if not scoped_to(source, number):
        return None, (jsonify({"error": "token out of scope"}), 403)
    try:
        return _load(source, number), None
    except PuzzleNotFound:
        return None, (jsonify({"error": "puzzle not available"}), 404)


@bp.route("/api/token/renew", methods=["POST"])
def renew():
    """Exchange a live (or just-expired) token for a fresh one.

    Keeps the token life short without ever expiring under a solver who is
    simply taking their time over a hard puzzle.
    """
    body = request.get_json(silent=True) or {}
    try:
        payload = token_age_ok_for_renew(body.get("token", ""))
    except AuthError as e:
        return jsonify({"error": e.message}), e.status
    return jsonify({
        "token": mint_token(payload["k"], payload["s"], payload["n"]),
        "renew_after": int(current_app.config["TOKEN_MAX_AGE"] * 0.6),
    })


@bp.route("/api/match-counts", methods=["POST"])
@require_token
def match_counts():
    """How many words each entry is offered — never more than MATCH_OPTIONS.

    Body: {"patterns": {"a1": "S?O?E", ...}, "crossed": ["a1", ...]}
    Returns {"counts": {"a1": {"n": 4, "capped": false}, ...}}

    The number is the LENGTH OF THE SHORTLIST the solver gets on clicking it,
    not the true tally, and the two are computed the same way here and in
    /api/tools/pattern so they can never disagree.

    Three refusals, all deliberate and all enforced here rather than in the
    browser:

    * A pattern with no unknown squares — a count on a full entry is a free
      Check, and Check is what the paper sells.
    * A pattern with no letters at all — an empty entry has no meaningful
      count, and a red zero on every empty entry would destroy the one signal
      the product is sold on.
    * More words fit than the limit — a chip reading "67" is one the solver
      cannot act on, and a board covered in them cannot be scanned. The
      exception is `crossed`: entries the caller says have every crossing
      letter in place. Those can never narrow further from the grid, so they
      are offered the shortlist anyway rather than staying silent forever.
    """
    loaded, error = _scoped_puzzle()
    if error:
        return error
    model, solutions, _letters = loaded
    entries = _entry_map(model)

    body = request.get_json(silent=True) or {}
    patterns = body.get("patterns") or {}
    if not isinstance(patterns, dict):
        return jsonify({"error": "patterns must be an object"}), 400
    if len(patterns) > 100:
        return jsonify({"error": "too many patterns"}), 400
    crossed = body.get("crossed")
    crossed = set(crossed) if isinstance(crossed, list) else set()

    limit = current_app.config["MATCH_OPTIONS"]
    offer_when_crossed = current_app.config["MATCH_OPTIONS_WHEN_CROSSED"]
    counts = {}
    for entry_id, pattern in patterns.items():
        entry = entries.get(entry_id)
        if entry is None or not isinstance(pattern, str):
            counts[entry_id] = None
            continue
        if len(pattern) != entry["len"]:
            counts[entry_id] = None
            continue
        if not any(ch in "?._ " for ch in pattern):
            counts[entry_id] = None      # full entry: not a free Check
            continue

        # One past the limit is all we need to know: it separates "this entry
        # has closed down to a handful" from "hundreds fit", and the early exit
        # keeps a whole-grid request cheap on every keystroke.
        words, over = corpus.match_words(
            current_app.config["CLUES_DB"], current_app.config["REF_DB"],
            pattern, entry.get("enum"), limit + 1,
        )
        if words is None:
            counts[entry_id] = None      # empty entry: no meaningful count
            continue
        if over:
            counts[entry_id] = (
                {"n": limit, "capped": True}
                if offer_when_crossed and entry_id in crossed else None
            )
            continue
        counts[entry_id] = {
            "n": len(_shortlist(entry, solutions, words, pattern, limit)),
            "capped": False,
        }

    return jsonify({"counts": counts})


def _answer_for(entry, solutions):
    """The entry's answer as the corpus would hold it, or None.

    None means "we do not hold it" — an embargoed prize puzzle, where Check and
    Reveal both say so. The shortlist must be silent in exactly the same
    circumstances, or the widget contradicts itself and leaks what the feed
    withheld.
    """
    answer = solutions.get(entry["id"])
    return corpus.display_form(answer, entry.get("enum")) if answer else None


def _shortlist(entry, solutions, words, pattern, limit):
    """The words this entry is offered, from an already-scanned match set.

    Built by the same call the pattern tool makes, so the number on the grid is
    the length of the list the click produces — never a promise the list then
    contradicts.
    """
    normalised = corpus.normalise_pattern(pattern) or ""
    answer = _answer_for(entry, solutions)
    # An answer that does not fit what is in the grid is not offered: the
    # solver has a wrong letter, and listing the answer anyway would tell them
    # so for nothing.
    if answer and not reference.fits_pattern(answer, normalised, entry.get("enum")):
        answer = None
    return reference.choose_options(words, answer, limit, normalised)


@bp.route("/api/check", methods=["POST"])
@require_token
def check():
    """Mark which of the supplied letters are wrong.

    Body: {"letters": {"r,c": "A", ...}, "scope": "entry"|"grid",
           "entry": "a1"}
    Returns {"wrong": ["r,c", ...], "unknown": ["r,c", ...]}

    `unknown` is the honest part: a prize puzzle held under embargo has no
    solution here, and those squares are reported as uncheckable rather than
    silently passed as correct.
    """
    loaded, error = _scoped_puzzle()
    if error:
        return error
    model, _solutions, solution_letters = loaded
    entries = _entry_map(model)

    body = request.get_json(silent=True) or {}
    letters = body.get("letters") or {}
    scope = body.get("scope", "grid")
    entry_id = body.get("entry")

    if scope == "entry":
        entry = entries.get(entry_id)
        if entry is None:
            return jsonify({"error": "unknown entry"}), 400
        wanted = {f"{r},{c}" for r, c in entry["cells"]}
    else:
        wanted = None

    wrong, unknown = [], []
    for key, value in letters.items():
        if wanted is not None and key not in wanted:
            continue
        expected = solution_letters.get(key)
        if expected is None:
            unknown.append(key)
        elif (value or "").upper() != expected:
            wrong.append(key)

    return jsonify({"wrong": sorted(wrong), "unknown": sorted(unknown)})


@bp.route("/api/reveal", methods=["POST"])
@require_token
def reveal():
    """Return solution letters for one cell, one entry, or the whole grid.

    Body: {"scope": "cell"|"entry"|"grid", "cell": "r,c", "entry": "a1"}
    Returns {"letters": {"r,c": "A", ...}, "unavailable": bool}
    """
    loaded, error = _scoped_puzzle()
    if error:
        return error
    model, _solutions, solution_letters = loaded
    entries = _entry_map(model)

    body = request.get_json(silent=True) or {}
    scope = body.get("scope", "cell")

    if scope == "cell":
        keys = [body.get("cell", "")]
    elif scope == "entry":
        entry = entries.get(body.get("entry"))
        if entry is None:
            return jsonify({"error": "unknown entry"}), 400
        keys = [f"{r},{c}" for r, c in entry["cells"]]
    else:
        keys = list(solution_letters.keys())

    out = {k: solution_letters[k] for k in keys if k in solution_letters}
    return jsonify({"letters": out, "unavailable": len(out) < len(keys)})




# --- the four tools -------------------------------------------------------
#
# Each is scoped to the puzzle in front of the solver by its token, capped, and
# rate limited per publisher key. The licensing design is honest that an
# endpoint answering arbitrary queries can be walked given enough time; these
# make it slow, bounded and visible rather than pretending it is impossible.

def _dbs():
    return current_app.config["CLUES_DB"], current_app.config["REF_DB"]


@bp.route("/api/tools/lookup", methods=["POST"])
@require_token
def tools_lookup():
    """What the reference DB knows about one word of the clue."""
    loaded, error = _scoped_puzzle()
    if error:
        return error
    model = loaded[0]
    body = request.get_json(silent=True) or {}
    word = (body.get("word") or "").strip()
    if not word or len(word) > 60:
        return jsonify({"error": "word required"}), 400
    letters = body.get("letters")
    letters = letters if isinstance(letters, int) and 1 <= letters <= 30 else None

    entry = _entry_map(model).get(body.get("entry"))
    entry_length = entry["len"] if entry and entry["len"] else None

    _clues, ref = _dbs()
    return jsonify(reference.lookup(ref, word, letters, entry_length))


@bp.route("/api/tools/word-info", methods=["POST"])
@require_token
def tools_word_info():
    """The ⓘ beside a result: what this word means in a crossword.

    Same reverse lookup the site runs from its match lists
    (`web/routes/helper.py:528`). It describes a corpus word, not this puzzle,
    so it gives nothing away that the word itself did not.
    """
    _loaded, error = _scoped_puzzle()
    if error:
        return error
    body = request.get_json(silent=True) or {}
    word = (body.get("word") or "").strip()
    if not word or len(word) > 60:
        return jsonify({"error": "word required"}), 400
    clues, ref = _dbs()
    return jsonify(reference.word_info(ref, clues, word))


@bp.route("/api/tools/synonym", methods=["POST"])
@require_token
def tools_synonym():
    """The synonym tool — deliberately separate from word lookup.

    It is for a word that is NOT in the clue, or a suspected definition.
    """
    _loaded, error = _scoped_puzzle()
    if error:
        return error
    body = request.get_json(silent=True) or {}
    word = (body.get("word") or "").strip()
    if not word:
        return jsonify({"error": "word required"}), 400
    length = body.get("length")
    length = length if isinstance(length, int) and 1 <= length <= 30 else None
    _clues, ref = _dbs()
    return jsonify(reference.synonyms(ref, word, length, body.get("include") or ""))


@bp.route("/api/tools/pattern", methods=["POST"])
@require_token
def tools_pattern():
    """Pattern search. Prefills from the entry, so it usually arrives filled in.

    This is where the grid's number is cashed in: click a chip and the words it
    counted are listed, at most MATCH_OPTIONS of them, alphabetical, with the
    answer among them. `total` still reports how many really fit, so a widened
    pattern reads "showing 9 of 41" rather than pretending nine is all there is.
    """
    loaded, error = _scoped_puzzle()
    if error:
        return error
    model, solutions, _letters = loaded
    body = request.get_json(silent=True) or {}
    pattern = (body.get("pattern") or "").strip()
    if not pattern or len(pattern) > 30:
        return jsonify({"error": "pattern required"}), 400

    enumeration = body.get("enum")
    entry = _entry_map(model).get(body.get("entry"))
    if entry and entry.get("stub_of"):
        entry = _entry_map(model).get(entry["stub_of"])
    if enumeration is None and entry:
        enumeration = entry.get("enum")

    # The answer of the entry the search was launched from — and only if it
    # fits the pattern actually being searched, which `pattern_matches` checks.
    # A solver who edits the box into some other pattern is not handed this
    # entry's answer for a pattern it does not match.
    answer = _answer_for(entry, solutions) if entry else None

    clues, ref = _dbs()
    return jsonify(reference.pattern_matches(
        corpus, clues, ref, pattern, enumeration, body.get("include") or "",
        answer=answer,
        limit=current_app.config["MATCH_OPTIONS"]))


@bp.route("/api/tools/anagram", methods=["POST"])
@require_token
def tools_anagram():
    """Anagram search, optionally filtered by the letters already in the grid.

    Fodder from the clue words the solver picked, narrowed by the grid: that
    composition is the thing no paper offers and no setter can hand over.
    """
    loaded, error = _scoped_puzzle()
    if error:
        return error
    model = loaded[0]
    body = request.get_json(silent=True) or {}
    letters = (body.get("letters") or "").strip()
    if not letters:
        return jsonify({"error": "letters required"}), 400

    grid_pattern = None
    if body.get("pattern"):
        entry = _entry_map(model).get(body.get("entry"))
        candidate = str(body["pattern"])
        # Only apply the grid filter when it describes the same number of
        # squares as the fodder has letters, or it would silently return none.
        if entry is None or len(candidate) == entry["len"]:
            grid_pattern = candidate

    clues, ref = _dbs()
    return jsonify(reference.anagrams(corpus, clues, ref, letters, grid_pattern))


# --- the hint ladder ------------------------------------------------------

_clue_index_cache = {}

# Rungs that give the answer away, directly or by spelling out the wordplay.
_ANSWER_BEARING_STEPS = ("answer", "explanation")


def _withheld(solutions, entry):
    """True when this entry's answer must not be served at all.

    The puzzle feed is the authority on whether an answer is published. A prize
    puzzle arrives under embargo with no solution letters, and Check and Reveal
    both say so — but clues_master.db may well hold the answer, scraped once the
    embargo lifted. Serving it through the hint ladder would contradict the same
    widget's own Reveal, and is exactly the leak the licensing design calls a
    hard requirement to prevent. When in doubt the feed wins.
    """
    return entry["id"] not in solutions


def _clue_index(source, display_number):
    key = (source, display_number)
    with _cache_lock:
        if key not in _clue_index_cache:
            _clue_index_cache[key] = explanations.load_clue_index(
                current_app.config["CLUES_DB"], source, display_number)
        return _clue_index_cache[key]


@bp.route("/api/hints", methods=["POST"])
@require_token
def hints():
    """One step of the ladder: definition, clue type, answer, full explanation.

    ONE step per request, never all four. Shipping the set and hiding three in
    the DOM would put the answer in the page for anyone who opened dev tools,
    which is the same as not having a ladder at all.
    """
    loaded, error = _scoped_puzzle()
    if error:
        return error
    model, solutions, _letters = loaded

    body = request.get_json(silent=True) or {}
    step = body.get("step")
    if step not in ("definition", "clue_type", "answer", "explanation"):
        return jsonify({"error": "unknown step"}), 400

    entry = _entry_map(model).get(body.get("entry"))
    if entry is None:
        return jsonify({"error": "unknown entry"}), 400
    if entry.get("stub_of"):
        entry = _entry_map(model).get(entry["stub_of"])

    if step in _ANSWER_BEARING_STEPS and _withheld(solutions, entry):
        return jsonify({"step": step, "value": None,
                        "unavailable": "The answer to this puzzle has not been "
                                       "published yet."})

    index = _clue_index(model["source"], model.get("display_number"))
    clue_id = index.get(entry["id"])
    if clue_id is None:
        return jsonify({"step": step, "value": None,
                        "unavailable": "This puzzle has no explanations yet."})

    steps = explanations.steps_for(current_app.config["CLUES_DB"], clue_id)
    if steps is None:
        return jsonify({"step": step, "value": None,
                        "unavailable": "This puzzle has no explanations yet."})

    value = steps.get(step)
    if value is None:
        return jsonify({"step": step, "value": None,
                        "unavailable": "Not recorded for this clue."})
    return jsonify({"step": step, "value": value})


@bp.route("/api/hints/available", methods=["POST"])
@require_token
def hints_available():
    """Which rungs exist for this entry, without giving any of them away.

    The widget needs to know whether to offer the ladder at all; it must learn
    that without receiving the content.
    """
    loaded, error = _scoped_puzzle()
    if error:
        return error
    model, solutions, _letters = loaded
    body = request.get_json(silent=True) or {}

    entry = _entry_map(model).get(body.get("entry"))
    if entry is None:
        return jsonify({"error": "unknown entry"}), 400
    if entry.get("stub_of"):
        entry = _entry_map(model).get(entry["stub_of"])

    index = _clue_index(model["source"], model.get("display_number"))
    clue_id = index.get(entry["id"])
    if clue_id is None:
        return jsonify({"steps": []})

    steps = explanations.steps_for(current_app.config["CLUES_DB"], clue_id) or {}
    withheld = _withheld(solutions, entry)
    return jsonify({"steps": [
        name for name in ("definition", "clue_type", "answer", "explanation")
        if steps.get(name) and not (withheld and name in _ANSWER_BEARING_STEPS)
    ]})
