"""The corpus behind the match count.

This is the killer feature of the licensing pitch: the number that recounts the
instant a crossing letter lands, answering "what actually fits _R?E?T?D" against
the whole curated corpus. It only exists served — no flat table delivers it.

Loaded once per process, grouped by length so a query is a regex over one
bucket rather than a scan. Sources: solved clue answers, `synonyms_pairs`, and
`definition_answers_augmented`.

Only ever exposed as a COUNT, never as a list. That is the extraction defence
the licensing design asks for: a count leaks a bit per query, a list leaks the
corpus. Counting also stops at a ceiling (100 by default), so a harvester
cannot use a wide pattern to measure bucket sizes either.
"""

import re
import sqlite3
import threading

_lock = threading.Lock()
_by_length = None       # {length: [(clean, display), ...]}
_stats = None


def _format_with_enum(clean, enumeration):
    """Space a clean answer according to its enumeration.

    ('ADLIBBING', '2-7') -> 'AD LIBBING'. Without this the enumeration filter
    below cannot tell a genuine (2-7) match from a nine-letter solid word.
    """
    if not enumeration:
        return clean
    parts = re.findall(r"\d+", enumeration)
    if len(parts) <= 1:
        return clean
    if sum(int(p) for p in parts) != len(clean):
        return clean            # length disagreement — do not guess a split
    out = []
    pos = 0
    for p in parts:
        n = int(p)
        out.append(clean[pos:pos + n])
        pos += n
    return " ".join(out)


def _add(entries, raw, enumeration=None):
    if not raw:
        return
    display = raw.strip().upper()
    if enumeration:
        display = _format_with_enum(re.sub(r"[^A-Z]", "", display), enumeration)
    clean = re.sub(r"[^A-Z]", "", display)
    if len(clean) < 2:
        return
    entries.setdefault(clean, set()).add(display)


def _build(clues_db, ref_db):
    entries = {}
    counts = {}

    conn = sqlite3.connect(f"file:{clues_db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT DISTINCT UPPER(answer), enumeration FROM clues "
            "WHERE answer IS NOT NULL AND answer != ''"
        ).fetchall()
        for answer, enumeration in rows:
            _add(entries, answer, enumeration)
        counts["clue answers"] = len(rows)
    finally:
        conn.close()

    conn = sqlite3.connect(f"file:{ref_db}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT DISTINCT UPPER(synonym) FROM synonyms_pairs "
            "WHERE synonym IS NOT NULL AND synonym != ''"
        ).fetchall()
        for (synonym,) in rows:
            _add(entries, synonym)
        counts["synonyms"] = len(rows)

        rows = conn.execute(
            "SELECT DISTINCT UPPER(answer) FROM definition_answers_augmented "
            "WHERE answer IS NOT NULL AND answer != ''"
        ).fetchall()
        for (answer,) in rows:
            _add(entries, answer)
        counts["definitions"] = len(rows)
    finally:
        conn.close()

    # SORTED, not set order. The bucket order decides which display form wins
    # for a word and which nine of sixty-two get shown, and Python randomises
    # set iteration per process — so an unsorted build gave a different
    # shortlist after every server restart. The sampling is meant to be stable.
    grouped = {}
    for clean in sorted(entries):
        grouped.setdefault(len(clean), []).extend(
            (clean, d) for d in sorted(entries[clean]))

    counts["distinct words"] = len(entries)
    return grouped, counts


def load(clues_db, ref_db):
    """Build the cache if needed and return it. Safe to call concurrently."""
    global _by_length, _stats
    if _by_length is None:
        with _lock:
            if _by_length is None:
                _by_length, _stats = _build(clues_db, ref_db)
    return _by_length


def stats(clues_db, ref_db):
    load(clues_db, ref_db)
    return dict(_stats or {})


def invalidate():
    global _by_length, _stats
    with _lock:
        _by_length = None
        _stats = None


def normalise_pattern(pattern):
    """Grid pattern -> a regex-ready string of letters and dots.

    Accepts '?', '.', '_' and space as unknowns. Returns None when the pattern
    is unusable: too short, or with no known letter at all (which would just
    count the whole bucket).
    """
    if not pattern:
        return None
    cleaned = re.sub(r"[^A-Za-z?._ ]", "", pattern).upper()
    normalised = re.sub(r"[?._ ]", ".", cleaned)
    if len(normalised) < 2 or all(ch == "." for ch in normalised):
        return None
    return normalised


def _enum_ok(display, enum_parts):
    if not enum_parts:
        return True
    words = display.split()
    if len(enum_parts) == 1:
        return len(words) == 1
    if len(words) != len(enum_parts):
        return False
    return all(len(w) == int(p) for w, p in zip(words, enum_parts))


def match_words(clues_db, ref_db, pattern, enumeration=None, ceiling=100):
    """Corpus words fitting `pattern`, stopping at `ceiling`.

    Returns (words, capped). `words` is None — never [] — when the pattern
    carries no information (empty, or all unknowns); see the note in
    `count_matches` on why that distinction is load-bearing. `capped` is True
    when the scan stopped at the ceiling, so more words fit than are listed.

    The early exit is what keeps this cheap enough to run over every entry in
    the grid on each keystroke: with a ceiling of ten, an entry that hundreds
    of words fit costs ten matches, not a bucket scan.
    """
    normalised = normalise_pattern(pattern)
    if normalised is None:
        return None, False

    by_length = load(clues_db, ref_db)
    bucket = by_length.get(len(normalised))
    if not bucket:
        return [], False

    enum_parts = re.findall(r"\d+", enumeration or "")
    if enum_parts and sum(int(p) for p in enum_parts) != len(normalised):
        # The enumeration does not describe this many squares — trust the
        # squares and drop the filter rather than return a confident zero.
        enum_parts = []

    regex = re.compile("^" + normalised.replace(".", "[A-Z]") + "$")
    seen = {}
    for clean, display in bucket:
        if clean in seen:
            continue
        if not regex.match(clean):
            continue
        if not _enum_ok(display, enum_parts):
            continue
        seen[clean] = display
        if len(seen) >= ceiling:
            return list(seen.values()), True
    return list(seen.values()), False


def count_matches(clues_db, ref_db, pattern, enumeration=None, ceiling=100):
    """How many corpus words fit `pattern`, stopping at `ceiling`.

    Returns (count, capped). `capped` is True when counting stopped at the
    ceiling, so the caller can render "over" rather than a misleading number.

    A count of None means "no meaningful count", NOT "nothing fits". The
    distinction matters more here than anywhere else in the widget: a zero is
    rendered red and means the letters in the grid are wrong. Returning 0 for
    an entry with no letters in it yet would put a red zero on every empty
    entry in the puzzle and destroy the one signal the product is sold on.
    """
    words, capped = match_words(clues_db, ref_db, pattern, enumeration, ceiling)
    return (None if words is None else len(words)), capped


def display_form(answer, enumeration=None):
    """An answer as the corpus would hold it: letters only, spaced by its
    enumeration. ('ADLIBBING', '2-7') -> 'AD LIBBING'.

    Used when the answer is placed beside corpus words, so a multi-word answer
    is not the one entry in the list that looks foreign.
    """
    if not answer:
        return None
    clean = re.sub(r"[^A-Z]", "", answer.upper())
    if not clean:
        return None
    return _format_with_enum(clean, enumeration)
