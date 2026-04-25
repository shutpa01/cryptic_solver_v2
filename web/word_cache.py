"""In-memory word cache for fast pattern matching.

Loads all distinct uppercase answers/synonyms from:
  - clues_master.db (clues.answer)
  - cryptic_new.db (synonyms_pairs.synonym)
  - cryptic_new.db (definition_answers_augmented.answer)

Words are grouped by length for O(n) regex matching instead of
full-table-scan SQL LIKE queries.

The cache is built once on first access and held in module-level state.
Call invalidate() to force a rebuild (e.g. after enrichment).
"""

import re
import sqlite3
import threading
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
_CLUES_DB = str(_BASE / "data" / "clues_master.db")
_REF_DB = str(_BASE / "data" / "cryptic_new.db")

_lock = threading.Lock()
_by_length = None  # dict[int, list[str]]  — all distinct uppercase words grouped by length
_by_length_user = None  # dict[int, list[tuple[str, str]]]  — (clean, display) from clues only


def _build():
    """Build the word cache from all three source tables."""
    words = set()

    conn = sqlite3.connect(f"file:{_CLUES_DB}?mode=ro", uri=True)
    for row in conn.execute(
        "SELECT DISTINCT UPPER(answer) AS a FROM clues "
        "WHERE answer IS NOT NULL AND answer != ''"
    ):
        words.add(row[0])
    conn.close()

    ref = sqlite3.connect(f"file:{_REF_DB}?mode=ro", uri=True)
    for row in ref.execute(
        "SELECT DISTINCT UPPER(synonym) AS a FROM synonyms_pairs "
        "WHERE synonym IS NOT NULL AND synonym != ''"
    ):
        words.add(row[0])
    for row in ref.execute(
        "SELECT DISTINCT UPPER(answer) AS a FROM definition_answers_augmented "
        "WHERE answer IS NOT NULL AND answer != ''"
    ):
        words.add(row[0])
    ref.close()

    words.discard(None)

    grouped = {}
    for w in words:
        n = len(w)
        if n not in grouped:
            grouped[n] = []
        grouped[n].append(w)

    return grouped


def get_by_length():
    """Return the length-grouped word dict, building it on first call."""
    global _by_length
    if _by_length is None:
        with _lock:
            if _by_length is None:
                _by_length = _build()
    return _by_length


def _format_with_enum(answer_clean, enumeration):
    """Insert spaces into a clean answer based on its enumeration.

    E.g. ('ADLIBBING', '2-7') -> 'AD LIBBING'
         ('EARHOLE', '3-4')   -> 'EAR HOLE'
         ('HELLO', '5')       -> 'HELLO'
         ('HELLO', None)      -> 'HELLO'
    """
    if not enumeration:
        return answer_clean
    nums = re.findall(r'\d+', enumeration)
    if len(nums) <= 1:
        return answer_clean
    total = sum(int(n) for n in nums)
    if total != len(answer_clean):
        return answer_clean  # length mismatch — don't guess
    pos = 0
    parts = []
    for n_str in nums:
        n = int(n_str)
        parts.append(answer_clean[pos:pos + n])
        pos += n
    return ' '.join(parts)


def _build_user():
    """Build user word cache from clues table only, with enum-formatted answers."""
    entries = {}  # clean_answer -> set of display_answers

    conn = sqlite3.connect(f"file:{_CLUES_DB}?mode=ro", uri=True)
    for row in conn.execute(
        "SELECT DISTINCT UPPER(REPLACE(answer, ' ', '')) AS clean, enumeration "
        "FROM clues WHERE answer IS NOT NULL AND answer != ''"
    ):
        clean = row[0]
        enum_val = row[1]
        if not clean:
            continue
        if clean not in entries:
            entries[clean] = set()
        entries[clean].add(_format_with_enum(clean, enum_val))
    conn.close()

    # Group by length of clean answer (no spaces)
    grouped = {}
    for clean, displays in entries.items():
        n = len(clean)
        if n not in grouped:
            grouped[n] = []
        for display in displays:
            grouped[n].append((clean, display))
    return grouped


def invalidate():
    """Clear the cache so it rebuilds on next access."""
    global _by_length, _by_length_user
    with _lock:
        _by_length = None
        _by_length_user = None


def _get_user_cache():
    """Return the user length-grouped cache, building on first call."""
    global _by_length_user
    if _by_length_user is None:
        with _lock:
            if _by_length_user is None:
                _by_length_user = _build_user()
    return _by_length_user


def _like_to_regex(pat):
    """Convert a SQL LIKE pattern (with _ wildcards) to a compiled regex."""
    escaped = re.escape(pat)
    return re.compile("^" + escaped.replace("_", ".") + "$")


def match_pattern_user(pattern_joined):
    """Match pattern against clues answers only, returning enum-formatted results.

    For regular users: answers come only from the clues table, and are
    formatted with spaces based on their enumeration. This means the
    existing enum filter works correctly for hyphenated answers like
    ADLIBBING (2-7) -> 'AD LIBBING'.
    """
    by_len = _get_user_cache()
    results = set()

    joined_len = len(pattern_joined)
    if joined_len >= 2 and not all(c == '_' for c in pattern_joined):
        regex = _like_to_regex(pattern_joined)
        for clean, display in by_len.get(joined_len, []):
            if regex.match(clean):
                results.add(display)

    return results


def match_pattern(pattern_joined, pattern_spaced=None):
    """Return a set of matching words for the given pattern(s).

    pattern_joined: SQL LIKE pattern with _ for unknowns, no spaces
                    (e.g. "S_O_E")
    pattern_spaced: optional SQL LIKE pattern with spaces for multi-word
                    answers (e.g. "S_O_E _ALL")

    Returns a set of uppercase strings.
    """
    by_len = get_by_length()
    results = set()

    # Search joined pattern
    joined_len = len(pattern_joined)
    if joined_len >= 2 and not all(c == '_' for c in pattern_joined):
        regex = _like_to_regex(pattern_joined)
        for w in by_len.get(joined_len, []):
            if regex.match(w):
                results.add(w)

    # Search spaced pattern (for multi-word answers stored with spaces)
    if pattern_spaced and pattern_spaced != pattern_joined:
        spaced_len = len(pattern_spaced)
        if spaced_len >= 2:
            regex = _like_to_regex(pattern_spaced)
            for w in by_len.get(spaced_len, []):
                if regex.match(w):
                    results.add(w)

    return results
