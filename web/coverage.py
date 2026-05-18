"""Coverage warning helper.

Flags clues whose stored explanation scores HIGH but whose source
letters in clue_word_roles do not sum to the answer's letters.
Used by both the puzzle page (per-clue badge) and the clue page
(badge next to the tier).
"""

import re
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_SOURCE_ROLES = {
    "synonym_source", "synonym", "abbreviation_source", "abbreviation",
    "positional_source", "reversal_source", "deletion_source",
    "literal_source", "letter_source", "possessive_source",
    "anagram_fodder", "spoonerism_fodder",
    "single_letter", "double_letter", "roman_numeral", "nato_phonetic",
    "cricket", "chemistry", "musical", "name", "shape", "example",
    "british_slang", "slang", "pronoun", "reference", "suffix",
    "first_letter", "substitution", "foreign", "foreign_french",
    "foreign_german", "foreign_spanish", "foreign_italian",
    "foreign_latin", "cryptic_synonym", "misc",
}

# Wordplay types that decompose into letter-producing pieces. The
# stored letters on source-role rows sum to the answer (charade,
# container, reversal, deletion, anagram). Other types either have
# no pieces (DD, CD) or use a mechanism that doesn't store letters
# directly (homophone, hidden, alternating, acrostic, palindrome,
# spoonerism, &lit, etc.).
_PIECE_KEYWORDS = ("charade", "anagram", "container",
                   "reversal", "deletion")
_NON_PIECE_KEYWORDS = (
    "homophone", "homonym", "spoonerism", "sponerism",
    "hidden", "alternat", "acrostic", "palindrome",
    "double", "cryptic", "straight", "lit", "shift",
    "substitut", "selection", "initial", "outer",
    "and_lit", "alternate",
)


def _is_piece_based(wordplay_type):
    if not wordplay_type:
        return False
    wt = wordplay_type.lower()
    if any(kw in wt for kw in _NON_PIECE_KEYWORDS):
        return False
    return any(kw in wt for kw in _PIECE_KEYWORDS)


def effective_letters(letters):
    """Strip parenthesized or bracketed content from a letters string.

    Convention: a synonym followed by a deletion can be written
    ADMIR(e), PHA(b)LET, etc. — the bracketed letters are deleted,
    so the answer-side contribution is everything outside the brackets.
    Used by coverage sum and placement.
    """
    if not letters:
        return ""
    return re.sub(r"\([^)]*\)|\[[^\]]*\]", "", letters)


_ORDINAL_MAP = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}


def _apply_prose_deletion(source_letters, explanation):
    """Find a prose deletion clause applied to source_letters in the
    explanation and return the post-deletion letters.

    Handles forms produced by the existing pipeline:
      - ANNUAL (synonym="yearbook") with second A removed
      - IMPERIAL (synonym="sovereign") with A (abbreviation="area") removed
      - ADMIRE synonym="esteem", last letter dropped
      - X (...) minus A

    Returns source_letters unchanged if no recognised deletion is
    found nearby OR if source_letters is itself the post-op result of
    a wrapper like X (deletion="Y", ...) — in that case the deletion
    clause inside those parens describes how X was produced, not what
    should still happen to X.
    """
    L = source_letters
    idx = explanation.upper().find(L)
    if idx < 0:
        return L
    tail_start = explanation[idx + len(L):]
    # If the immediate parens after L name a deletion/reversal/anagram/
    # spoonerism operation, L is already the post-operation result; do
    # not re-apply any clause from within those parens.
    if re.match(
            r"\s*\(\s*(?:deletion|reversal|anagram|spoonerism)\s*=",
            tail_start, re.IGNORECASE):
        return L
    # Restrict the search to before the next piece boundary so a
    # deletion clause inside a later piece does not bleed onto this one.
    boundary = re.search(r"[+;]", tail_start)
    tail = tail_start[:boundary.start()] if boundary else tail_start[:300]
    # Positional drops (first/last/middle/outer letter[s] dropped/removed)
    m = re.search(
        r"\b(first|last|middle|outer)\s+letters?\s+(?:dropped|removed|deleted)",
        tail, re.IGNORECASE)
    if m:
        kind = m.group(1).lower()
        if kind == "first" and len(L) > 1:
            return L[1:]
        if kind == "last" and len(L) > 1:
            return L[:-1]
        if kind == "outer" and len(L) > 2:
            return L[1:-1]
        if kind == "middle" and len(L) >= 3:
            mid = len(L) // 2
            return L[:mid] + L[mid + 1:]
    # "with [Nth] X removed/dropped" / "minus X"
    m = re.search(
        r"(?:with|minus)\s+(?:the\s+)?(first|last|second|third|fourth|"
        r"fifth|sixth|seventh|eighth|ninth|tenth)?\s*"
        r"([A-Z]+)(?:\s*\([^)]*\))?\s+(?:removed|dropped|deleted|out)",
        tail, re.IGNORECASE)
    if m:
        position = (m.group(1) or "").lower()
        rm = m.group(2).upper()
        if position == "last":
            i = L.rfind(rm)
            if i >= 0:
                return L[:i] + L[i + len(rm):]
        elif position in _ORDINAL_MAP:
            n = _ORDINAL_MAP[position]
            count = 0
            for i in range(len(L) - len(rm) + 1):
                if L[i:i + len(rm)] == rm:
                    count += 1
                    if count == n:
                        return L[:i] + L[i + len(rm):]
        else:
            i = L.find(rm)
            if i >= 0:
                return L[:i] + L[i + len(rm):]
    return L


def deletion_target_letters(explanation):
    """Return a list of letter-tokens named as deletion targets in the
    explanation. Covers two forms produced by the pipeline:
      - "with [Nth] X removed" / "minus X"
      - "X (...) dropped" inside a piece-source claim
    Each entry is the literal letter group the clause names (e.g. "A",
    "P"). Used by the coverage check to subtract consumed letters that
    must not appear in the answer sum.
    """
    if not explanation:
        return []
    targets = []
    for m in re.finditer(
            r"(?:with|minus)\s+(?:the\s+)?(?:first|last|second|third|fourth|"
            r"fifth|sixth|seventh|eighth|ninth|tenth)?\s*"
            r"([A-Z]+)(?:\s*\([^)]*\))?\s+(?:removed|dropped|deleted|out)\b",
            explanation, re.IGNORECASE):
        targets.append(m.group(1).upper())
    # Bare piece-source form "X (abbreviation=...) dropped". Restrict to
    # abbreviation / single_letter / cricket / nato_phonetic claims so
    # the regex doesn't run greedy across nested parens and wrongly
    # capture an outer piece like INT (deletion="PINT", P dropped) as
    # the target.
    for m in re.finditer(
            r"\b([A-Z]+)\s*\(\s*(?:abbreviation|single_letter|cricket|"
            r"nato_phonetic|roman_numeral|chemistry|musical|name|"
            r"foreign_\w+|cryptic_synonym|substitution|first_letter)\s*=",
            explanation, re.IGNORECASE):
        # Find the closing paren of this piece-source claim, then check
        # whether 'dropped/removed/deleted' follows within a few words.
        start = m.start()
        # Walk parens to balance.
        depth = 0
        end = None
        for i in range(start, min(len(explanation), start + 200)):
            if explanation[i] == "(":
                depth += 1
            elif explanation[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            continue
        post = explanation[end + 1:end + 30]
        if re.match(r"\s+(?:dropped|removed|deleted)\b", post,
                    re.IGNORECASE):
            targets.append(m.group(1).upper())
    return targets


def post_op_letters(source_letters, explanation):
    """If a piece's source letters (e.g. ADMIRE for synonym esteem)
    appears as the operand of a deletion / reversal / anagram in the
    explanation, return the post-operation letters that actually land
    in the answer (e.g. ADMIR). Otherwise return source_letters
    unchanged. Convention: dropdown letters hold the source value;
    contributes column and coverage sum use the post-op value.
    """
    if not source_letters or not explanation:
        return (source_letters or "").upper()
    L = source_letters.upper()
    for op in ("deletion", "reversal", "anagram", "spoonerism"):
        pat = (r"([A-Z']+)\s*\(\s*" + op + r"\s*=\s*\"" +
               re.escape(L) + r"\"")
        m = re.search(pat, explanation, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # Fallback: prose deletion forms produced by the pipeline.
    return _apply_prose_deletion(L, explanation)


def coverage_warning(clue_id, answer, wordplay_type, tier,
                     ai_explanation=None, conn=None):
    # Run on every tier that has a parse: HIGH/MEDIUM/LOW. PENDING and
    # FAIL clues have either no parse or a known-broken one, so the
    # coverage badge would be redundant noise there. Earlier this gate
    # only fired on HIGH, which let wrong-piece-sum parses (e.g. a
    # MEDIUM-scoring RIB+CAGE+S claiming to make RIBCAGE) slip past
    # unflagged.
    if tier in ("PENDING", "FAIL"):
        return False
    if not _is_piece_based(wordplay_type):
        return False
    answer_letters = re.sub(r"[^A-Z]", "", (answer or "").upper())
    if not answer_letters:
        return False

    own_conn = False
    if conn is None:
        conn = sqlite3.connect(
            str(PROJECT_ROOT / "data" / "clues_master.db"))
        own_conn = True
    try:
        rows = conn.execute(
            "SELECT role, letters, piece_key FROM clue_word_roles "
            "WHERE clue_id = ?",
            (clue_id,),
        ).fetchall()
    finally:
        if own_conn:
            conn.close()

    # Dedupe by piece_key for non-fodder roles: a multi-word piece like
    # "Manuel's gag" → QUE has the same letters stored on every row, so
    # summing them all triples QUE. Anagram fodder is the opposite: each
    # word contributes its own letters (a=A, prime=PRIME, tv=TV), so we
    # must concatenate every row.
    seen_piece_keys = set()
    piece_letters = ""
    for role, letters, piece_key in rows:
        if role not in _SOURCE_ROLES or role == "literal_source" or not letters:
            continue
        if role == "anagram_fodder":
            piece_letters += re.sub(
                r"[^A-Z]", "", effective_letters(letters).upper())
            continue
        if piece_key is not None:
            if piece_key in seen_piece_keys:
                continue
            seen_piece_keys.add(piece_key)
        effective = post_op_letters(effective_letters(letters), ai_explanation)
        piece_letters += re.sub(r"[^A-Z]", "", effective)

    # "X (from clue)" pieces — covers literal letters that have no
    # standalone clue word, e.g. the S from a possessive 's. Mirror
    # the clue-page render guard added 2026-05-14: only count an
    # "S (from clue)" piece when including it makes the multiset
    # match the answer. Without this guard a parse over-claiming an
    # extra S (e.g. RIB + CAGE + S = RIBCAGE) would have the
    # coverage_warning badge fire even after the render suppresses
    # the synthetic possessive row, because coverage was summing the
    # spurious S regardless. Other "X (from clue)" letters (non-S)
    # are still always counted.
    if ai_explanation:
        for m in re.finditer(
                r"\b(\w+)\s*\(\s*from\s+clue\s*\)",
                ai_explanation, re.IGNORECASE):
            piece = re.sub(r"[^A-Z]", "", m.group(1).upper())
            if piece == "S":
                # Only include the S if including it helps coverage.
                if (sorted(piece_letters + "S") == sorted(answer_letters)
                        and sorted(piece_letters) != sorted(answer_letters)):
                    piece_letters += "S"
                # else: drop the spurious S; render does the same.
            else:
                # Only add if no letter in piece would exceed its count in
                # the answer. Prevents double-counting when clue_word_roles
                # already carries the same letters as this (from clue) piece.
                combined = piece_letters + piece
                if all(combined.count(ch) <= answer_letters.count(ch)
                       for ch in set(piece)):
                    piece_letters += piece

    # Subtract any letters named as deletion targets. The deletion
    # target is consumed by the deletion, so it must not count as an
    # addition to the answer.
    if ai_explanation:
        remaining = list(piece_letters)
        for target in deletion_target_letters(ai_explanation):
            for ch in target:
                if ch in remaining:
                    remaining.remove(ch)
        piece_letters = "".join(remaining)

    return sorted(piece_letters) != sorted(answer_letters)
