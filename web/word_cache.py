"""In-memory word cache for the pattern finder and the anagram solver.

ONE word list, shared by both tools and by every user (2026-10-10, user: "why would we
not use all the words in the entire database?" — the pattern finder used to give users
past answers only, and the anagram solver three columns). It holds every word the two
databases know, each with a TIER that ranks it in the results (results are capped, so
the order decides what is seen):

  0  a past crossword answer       clues.answer (clues_master.db + cryptic_new.db)
  1  a synonym / definition answer synonyms_pairs.synonym, definition_answers_augmented.answer
  2  any other word we hold        pronunciations, homophones, and the single words among
                                   synonym headwords, clue definitions and indicators

Multi-word entries are taken only from the columns that hold ANSWERS (tiers 0-1) and from
the pronouncing dictionary (hyphenated compounds). The other columns hold clue phrases
("source of legitimacy in post"), so only their single words are kept.

Built once per process (warm() starts it in the background on the first request) and
held in module state. Call invalidate() to force a rebuild.
"""

import re
import sqlite3
import threading
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
_CLUES_DB = str(_BASE / "data" / "clues_master.db")
_REF_DB = str(_BASE / "data" / "cryptic_new.db")

# (db, sql, tier, multi-word allowed). A query returning a second column gives the
# enumeration used to space a past answer (ADLIBBING (2-7) -> AD LIBBING).
_SOURCES = [
    (_CLUES_DB, "SELECT DISTINCT answer, enumeration FROM clues", 0, True),
    (_REF_DB, "SELECT DISTINCT answer, enumeration FROM clues", 0, True),
    (_REF_DB, "SELECT DISTINCT synonym FROM synonyms_pairs", 1, True),
    (_REF_DB, "SELECT DISTINCT answer FROM definition_answers_augmented", 1, True),
    (_REF_DB, "SELECT DISTINCT word FROM pronunciations", 2, True),
    (_REF_DB, "SELECT DISTINCT word FROM homophones", 2, True),
    (_REF_DB, "SELECT DISTINCT homophone FROM homophones", 2, True),
    (_REF_DB, "SELECT DISTINCT word FROM synonyms_pairs", 2, False),
    (_REF_DB, "SELECT DISTINCT definition FROM definition_answers_augmented", 2, False),
    (_REF_DB, "SELECT DISTINCT word FROM indicators", 2, False),
]

_WORD_OK = re.compile(r"^[A-Z][A-Z '\-]*[A-Z]$")

_lock = threading.Lock()
_by_length = None   # dict[int, list[tuple[clean, display]]]
_by_sig = None      # dict[str, list[display]] — sorted-letter fingerprint -> words
_tier = None        # dict[clean, int]
_warming = False    # a warm() thread has been started for the current build


def _normalise(raw):
    """Upper-case display form (word breaks as single spaces), or None if not a word."""
    s = (raw or "").upper().strip()
    if not _WORD_OK.match(s):
        return None
    s = re.sub(r"[\s\-]+", " ", s.replace("'", ""))
    return s if 2 <= len(s.replace(" ", "")) <= 25 else None


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


def signature(letters):
    """Sorted uppercase letters only — the anagram fingerprint."""
    return "".join(sorted(re.sub(r'[^A-Z]', '', (letters or "").upper())))


def _build():
    """Read every source once; return (by_length, by_sig, tier)."""
    displays = {}   # clean -> set of display forms
    tier = {}       # clean -> best (lowest) tier
    for db, sql, t, multi_ok in _SOURCES:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            for row in conn.execute(sql):
                disp = _normalise(row[0])
                if not disp:
                    continue
                clean = disp.replace(" ", "")
                if len(row) > 1:
                    disp = _format_with_enum(clean, row[1]) if row[1] else disp
                if " " in disp and not multi_ok:
                    continue
                displays.setdefault(clean, set()).add(disp)
                if t < tier.get(clean, 3):
                    tier[clean] = t
        finally:
            conn.close()

    by_length, by_sig = {}, {}
    for clean, ds in displays.items():
        lst = by_length.setdefault(len(clean), [])
        sig = by_sig.setdefault(signature(clean), [])
        for d in ds:
            lst.append((clean, d))
            sig.append(d)
    return by_length, by_sig, tier


def _get():
    global _by_length, _by_sig, _tier
    if _by_length is None:
        with _lock:
            if _by_length is None:
                _by_length, _by_sig, _tier = _build()
    return _by_length, _by_sig, _tier


def warm():
    """Build the cache in a background thread so the first search does not wait.
    Starts at most one thread per build (it is called on every request)."""
    global _warming
    if _by_length is None and not _warming:
        _warming = True
        threading.Thread(target=_get, name="word-cache-warm", daemon=True).start()


def invalidate():
    """Clear the cache so it rebuilds on next access."""
    global _by_length, _by_sig, _tier, _warming
    with _lock:
        _by_length = _by_sig = _tier = None
        _warming = False


def rank_key(display):
    """Sort key for results: past answers first, then synonyms, then other words."""
    _, _, tier = _get()
    clean = display.replace(" ", "")
    return (tier.get(clean, 3), display)


def one_per_word(displays):
    """Collapse forms of the same letters to one, keeping the spaced form (SKY BLUE over
    SKYBLUE) — for result lists where no enumeration has chosen between them."""
    best = {}
    for d in displays:
        clean = d.replace(" ", "")
        if clean not in best or (" " in d and " " not in best[clean]):
            best[clean] = d
    return set(best.values())


def _like_to_regex(pat):
    """Convert a SQL LIKE pattern (with _ wildcards) to a compiled regex."""
    escaped = re.escape(pat)
    return re.compile("^" + escaped.replace("_", ".") + "$")


def match_pattern_user(pattern_joined):
    """Words matching a pattern (SQL LIKE, _ for unknowns, no spaces), as display forms
    with their word breaks, so the callers' enumeration filters work (AD LIBBING)."""
    by_len, _, _ = _get()
    results = set()
    if len(pattern_joined) >= 2 and not all(c == '_' for c in pattern_joined):
        regex = _like_to_regex(pattern_joined)
        for clean, display in by_len.get(len(pattern_joined), []):
            if regex.match(clean):
                results.add(display)
    return results


def match_pattern(pattern_joined, pattern_spaced=None):
    """Same word list as match_pattern_user — users and admin now search alike."""
    return match_pattern_user(pattern_joined)


def match_anagram(letters):
    """Every word (display form) whose letters are an anagram of `letters`."""
    _, by_sig, _ = _get()
    return set(by_sig.get(signature(letters), []))
