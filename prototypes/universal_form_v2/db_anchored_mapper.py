"""DB-anchored blog mapper (rebuild).

The previous mapper anchored pieces to clue words via blog `(gloss)`
text. This one uses the cryptic_new.db synonyms / abbreviations tables
to anchor pieces — handling TFTT's terse format where blogs omit
glosses but the answer-pieces are still recognisable from the clue.

See MAPPER_REBUILD.md for the design.

Public entry point:
    map_clue_words_db(clue_text, answer, blog_text, conn,
                      clue_definition=None) -> Mapping

`conn` is a read-only sqlite3 connection to cryptic_new.db.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from .word_role_mapper import Mapping, Tag, _norm_letters, _norm


# --- Op-word vocabulary ---------------------------------------------------

OP_WORDS = {
    # container family
    "in": ("container", None),
    "inside": ("container", None),
    "within": ("container", None),
    "containing": ("container", None),
    "contained": ("container", None),
    "around": ("container", None),
    "surrounding": ("container", None),
    "enveloping": ("container", None),
    "outside": ("container", None),
    # reversal family
    "reversed": ("reversal", None),
    "reversal": ("reversal", None),
    "rev": ("reversal", None),
    "back": ("reversal", None),
    "backwards": ("reversal", None),
    "returning": ("reversal", None),
    # anagram family
    "anagram": ("anagram", None),
    "scrambled": ("anagram", None),
    "mixed": ("anagram", None),
    # hidden family
    "hidden": ("hidden", None),
    "concealed": ("hidden", None),
    # homophone
    "homophone": ("homophone", None),
    "sounds": ("homophone", None),
}

CHARADE_JOIN_WORDS = {
    "then", "and", "next", "with", "before", "after",
}


# --- Tokenisation events --------------------------------------------------

@dataclass
class Event:
    kind: str       # PIECE | INSERT | DROP | GLOSS | INDICATOR
                    # | OP_WORD | CHARADE_JOIN | ANAGRAM_FODDER
    value: str      # text or letters
    pos: int = 0    # offset in blog (for debugging)


_PIECE_RE = re.compile(
    r"[A-Z]+\.[A-Z]+(?:\.[A-Z]+)*\.?"        # period abbrev R.E.
    r"|"
    # Leading {x} or [x], optionally repeated for alternate-letter
    # notation {e}N{g}A{g}E (every-other-letter extraction).
    r"(?:\{[a-z]+\}|\[[a-z]+\])"
    r"(?:[A-Z]+(?:\{[a-z]+\}|\[[a-z]+\])?)+[A-Z]*"
    r"|"
    r"[A-Z]+"
    r"(?:(?:\{[a-z]+\}|\[[a-z]+\])[A-Z]+)*"
    r"(?:\{[a-z]+\}|\[[a-z]+\])?[A-Z]*"
)
_PAREN_RE = re.compile(r"\(([^()]+)\)")
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_BAREWORD_RE = re.compile(r"[a-z][a-z\-]*")


_MIXED_CASE_RE = re.compile(r"[A-Za-z]{2,}")


def _classify_mixed_case(token: str):
    """Detect TFTT mixed-case positional notation.

    Examples:
      'Bottle'  -> ('first', 'B', 'bottle')   first letter
      'bottlE'  -> ('last', 'E', 'bottle')    last letter
      'BottlE'  -> ('outer', 'BE', 'bottle')  outer letters
      'bASIn'   -> ('middle', 'ASI', 'basin') middle letters
      'pUnTeR'  -> ('alternate', 'UTR', 'punter') (rare, fallback)

    Returns (kind, kept_letters, source_word) or None if not a
    positional pattern.
    """
    if len(token) < 3:
        return None
    upper_indices = [i for i, c in enumerate(token) if c.isupper()]
    lower_indices = [i for i, c in enumerate(token) if c.islower()]
    if not upper_indices or not lower_indices:
        return None  # all-upper or all-lower: not mixed
    if len(upper_indices) > len(token) - 2:
        return None  # mostly upper: probably a regular piece + bareword
    source = token.lower()
    kept = "".join(token[i] for i in upper_indices)
    n = len(token)
    # First letter pattern: only first char is upper
    if upper_indices == [0] and len(kept) == 1:
        return ("first", kept, source)
    # Last letter: only last char is upper
    if upper_indices == [n - 1] and len(kept) == 1:
        return ("last", kept, source)
    # Outer: first and last upper, middle lower
    if upper_indices[0] == 0 and upper_indices[-1] == n - 1 \
            and len(upper_indices) == 2:
        return ("outer", kept, source)
    # Middle/inner: lowercase at start and end, uppercase contiguous in middle
    if upper_indices[0] > 0 and upper_indices[-1] < n - 1 \
            and upper_indices == list(range(upper_indices[0],
                                              upper_indices[-1] + 1)):
        return ("middle", kept, source)
    # Fallback: alternating
    return ("alternate", kept, source)


_PROSE_DELETION_RE = re.compile(
    r"\b([A-Z]{2,})\b"                                  # PIECE (uppercase)
    r"(?:\s*\([^)]*\))?"                                # optional gloss
    r"\s+(?:minus|without|losing|less|lacking)\s+"      # deletion verb
    r"(?:its|the)?\s*"
    r"(last|first|final|initial|central|middle|"
    r"external|outer|interior|inner|inside)"            # kind
    r"\s+letters?\b",
    re.IGNORECASE)

_PROSE_KIND_TO_BRACES = {
    "last": "tail",
    "final": "tail",
    "first": "head",
    "initial": "head",
    "central": "heart",
    "middle": "heart",
    "external": "outer",
    "outer": "outer",
    "interior": "heart",
    "inner": "heart",
    "inside": "heart",
}


_SUBSTRING_DELETION_RE = re.compile(
    r"\b([A-Z]{2,})\b"                                  # WORD1
    r"(?:\s*\([^)]*\))?"                                # optional gloss
    r"\s+[\-–—]?\s*"                          # optional dash
    r"(?:\(\s*(?:gives?|giving|drops?|dropping|"
    r"missing|without|losing|loses?|sans|"
    r"absent|removing|removed|sheds?|"
    r"shedding|less|lacking)"
    r"(?:\s+\w+)?\s*\)|"
    r"(?:minus|loses?|losing|missing|without|"
    r"lacking|expelled\s+from|sans|absent|"
    r"sheds?|shedding|drops?|dropping|"
    r"gives?|giving|less))\s+"
    r"(?:its\s+|\(\s*\w+\s*\)\s+)?"
    r"(?P<word2>[A-Z]{1,3})\b"                          # WORD2 (1-3 chars)
    r"(?:\s*\([^)]*\))?",
    re.IGNORECASE)


def _convert_substring_deletion(blog: str) -> str:
    """Convert 'WORD1 minus WORD2' to curly-brace notation when WORD2
    is a prefix/suffix of WORD1.

    Examples:
      'FETCHING loses F'  -> '{f}ETCHING'
      'TUNIS minus IS'    -> 'TUN{is}'
      'VOMIT minus V'     -> '{v}OMIT'
    Also handles 'X expelled from WORD1' shape (TUN case).
    """
    def replace(m):
        w1 = m.group(1)
        w2 = m.group("word2")
        if not w1 or not w2 or len(w2) >= len(w1):
            return m.group(0)
        if w1.startswith(w2):
            new = "{" + w2.lower() + "}" + w1[len(w2):]
        elif w1.endswith(w2):
            new = w1[:-len(w2)] + "{" + w2.lower() + "}"
        else:
            # Mid-word: w2 must occur exactly once inside w1 (not at
            # the start or end). Schema-supported via heart-style
            # deletion when w2 is a single letter.
            occurrences = [i for i in range(1, len(w1) - len(w2))
                            if w1[i:i + len(w2)] == w2]
            if len(occurrences) != 1:
                return m.group(0)
            p = occurrences[0]
            new = w1[:p] + "{" + w2.lower() + "}" + w1[p + len(w2):]
        # Preserve the leading gloss if any
        orig = m.group(0)
        gloss_match = re.match(r"\b[A-Z]{2,}\b\s*\(([^)]*)\)", orig)
        if gloss_match:
            return f"{new} ({gloss_match.group(1)})"
        return new

    blog = _SUBSTRING_DELETION_RE.sub(replace, blog)
    # Also handle the inverted shapes:
    #   "remove WORD2 from WORD1"
    #   "expel WORD2 from WORD1"
    #   "removal of WORD2 from WORD1"
    #   "removing WORD2 from WORD1"
    #   "deleting WORD2 from WORD1"
    #   "WORD2 dropped from WORD1"
    inv_re = re.compile(
        r"\b(?:remove|expel|removal\s+of|removing|"
        r"deleting|deletion\s+of)\s+"
        r"([A-Z]{1,3})\b"
        r"(?:\s*\(([^)]*)\))?"                # optional w2 gloss
        r"\s+(?:from|out\s+of)\s+([A-Z]{2,})\b"
        r"(?:\s*\(([^)]*)\))?",                # optional w1 gloss
        re.IGNORECASE)

    def repl_inv(m):
        w2 = m.group(1)
        w2_gloss = m.group(2)
        w1 = m.group(3)
        w1_gloss = m.group(4)
        if not w1 or not w2 or len(w2) >= len(w1):
            return m.group(0)
        if w1.startswith(w2):
            new = "{" + w2.lower() + "}" + w1[len(w2):]
        elif w1.endswith(w2):
            new = w1[:-len(w2)] + "{" + w2.lower() + "}"
        else:
            occurrences = [i for i in range(1, len(w1) - len(w2))
                            if w1[i:i + len(w2)] == w2]
            if len(occurrences) != 1:
                return m.group(0)
            p = occurrences[0]
            new = w1[:p] + "{" + w2.lower() + "}" + w1[p + len(w2):]
        if w1_gloss:
            new = f"{new} ({w1_gloss})"
        # Preserve the dropped-letter gloss as a bracket-indicator so
        # the clue word it refers to gets tagged (residue-friendly).
        if w2_gloss:
            new = f"{new} [{w2_gloss}]"
        return new

    blog = inv_re.sub(repl_inv, blog)
    return blog


def _convert_prose_deletion(blog: str) -> str:
    """Pre-process blog: convert prose-described deletions on uppercase
    pieces into curly-brace notation that the existing parser handles.

    Examples:
      'STIFF (strong) minus the last letter'   -> 'STIF{F} (strong)'
      'JEDI without first letter (leaderless)' -> '{J}EDI (leaderless)'
      'DRIVES (roads) minus the central letters' -> 'DR{IV}ES (roads)'
      'SINNERS minus the external letters'     -> '{S}INNER{S}'
    """
    def replace(m):
        word = m.group(1)
        kind_str = m.group(2).lower()
        kind = _PROSE_KIND_TO_BRACES.get(kind_str)
        if not kind or len(word) < 3:
            return m.group(0)  # leave unchanged
        # Dropped letters must be LOWERCASE for the existing _PIECE_RE
        # to recognise the curly-brace deletion notation.
        if kind == "tail":
            new = word[:-1] + "{" + word[-1].lower() + "}"
        elif kind == "head":
            new = "{" + word[0].lower() + "}" + word[1:]
        elif kind == "heart":
            mid = len(word) // 2
            if len(word) % 2 == 1:
                new = word[:mid] + "{" + word[mid].lower() + "}" + word[mid + 1:]
            else:
                new = (word[:mid - 1]
                        + "{" + word[mid - 1:mid + 1].lower() + "}"
                        + word[mid + 1:])
        elif kind == "outer":
            if len(word) >= 3:
                new = ("{" + word[0].lower() + "}"
                        + word[1:-1]
                        + "{" + word[-1].lower() + "}")
            else:
                return m.group(0)
        else:
            return m.group(0)
        # Preserve any gloss that was matched
        # Find the gloss (if any) in the original match
        orig = m.group(0)
        gloss_match = re.search(r"\(([^)]*)\)", orig)
        if gloss_match:
            return f"{new} ({gloss_match.group(1)})"
        return new

    return _PROSE_DELETION_RE.sub(replace, blog)


_LOOSE_CURLY_RE = re.compile(r"(\b[A-Z]+)\s+(\{[a-z]+\})")
_LOOSE_CURLY_TRAIL_RE = re.compile(r"(\{[a-z]+\})\s+([A-Z]+\b)")

# Prose positional: "last letter of XXX" / "back of XXX" / "first of XXX" /
# "front of XXX" / "head of XXX" / "tail of XXX". Convert the WORD to
# mixed-case so the existing mixed-case parser produces a positional
# event. The optional parenthetical (e.g. "last letter (back) of FIESTA")
# is preserved.
_PROSE_POSITIONAL_LAST_RE = re.compile(
    r"\b(?:last\s+letters?|final\s+letters?|end|tail|back|bottom)"
    r"(?:\s*\([^)]*\))?\s+of\s+([A-Z][A-Za-z]+)\b",
    re.IGNORECASE)
_PROSE_POSITIONAL_FIRST_RE = re.compile(
    r"\b(?:first\s+letters?|initial\s+letters?|head|top|front|start|opening)"
    r"(?:\s*\([^)]*\))?\s+of\s+([A-Z][A-Za-z]+)\b",
    re.IGNORECASE)


def _convert_prose_positional(blog: str) -> str:
    def repl_last(m):
        word = m.group(1)
        if len(word) < 2:
            return m.group(0)
        new = word[:-1].lower() + word[-1].upper()
        return new

    def repl_first(m):
        word = m.group(1)
        if len(word) < 2:
            return m.group(0)
        new = word[0].upper() + word[1:].lower()
        return new

    blog = _PROSE_POSITIONAL_LAST_RE.sub(repl_last, blog)
    blog = _PROSE_POSITIONAL_FIRST_RE.sub(repl_first, blog)
    # Multi-word "first letters of WORD WORD WORD" — convert each word
    # to mixed-case (capital first, rest lowercase) to emit POSITIONAL
    # events. Common in TFTT for things like "first letters of Good
    # Employer" -> "Good Employer".
    multi_first = re.compile(
        r"\b(?:first\s+letters?|tips|initial\s+letters?|"
        r"starts|beginnings)\s*(?:\([^)]*\))?\s+of\s+"
        r"((?:[A-Z][a-z]+(?:\s+(?:and\s+)?)?)+)\b",
        re.IGNORECASE)

    def repl_multi_first(m):
        words = re.findall(r"[A-Z][a-z]+", m.group(1))
        # Each word with capital first only — already mixed-case
        return " ".join(words)

    blog = multi_first.sub(repl_multi_first, blog)
    multi_last = re.compile(
        r"\b(?:last\s+letters?|tails|ends|"
        r"final\s+letters?)\s*(?:\([^)]*\))?\s+of\s+"
        r"((?:[A-Z][a-z]+(?:\s+(?:and\s+)?)?)+)\b",
        re.IGNORECASE)

    def repl_multi_last(m):
        words = re.findall(r"[A-Z][a-z]+", m.group(1))
        return " ".join(w[:-1].lower() + w[-1].upper() for w in words)

    blog = multi_last.sub(repl_multi_last, blog)
    return blog


_ANAGRAMMED_GLOSS_RE = re.compile(
    r"\b([A-Z]+(?:\s+[A-Z]+)*)\b\s+"  # FODDER (one or more uppercase tokens)
    r"(?:anagrammed|scrambled|jumbled|rearranged)\s+"
    r"\(([^)]+)\)",
    re.IGNORECASE)

# "(An )?Anagram [indicator] of FODDER" — common blog phrasing.
_ANAGRAM_OF_RE = re.compile(
    r"(?:^|[\s\-,;.])(?:[Aa]n\s+)?[Aa]nagram"
    r"(?:\s*\[([^\]]+)\])?"
    r"(?:\s*\(([^)]+)\))?"
    r"\s+of\s+"
    r"([A-Z]+(?:\s+[A-Z]+)*)\b")


def _convert_anagram_of(blog: str) -> str:
    """Convert 'Anagram [indicator] of X Y' to '*(X Y) [indicator]'."""
    def repl(m):
        bracket_ind = m.group(1)
        paren_ind = m.group(2)
        fodder = m.group(3)
        ind = bracket_ind or paren_ind or ""
        prefix = m.group(0)[0] if m.group(0) and m.group(0)[0] in " \t\n,.;-" else ""
        if ind:
            return f"{prefix}*({fodder}) [{ind}]"
        return f"{prefix}*({fodder})"
    return _ANAGRAM_OF_RE.sub(repl, blog)


def _convert_anagrammed_gloss(blog: str) -> str:
    """Convert 'ARE anagrammed (blunders)' to '*(ARE) [blunders]'.

    The bracketed gloss IS the anagram indicator anchor in the clue.
    Rewrite to the canonical TFTT shape so the existing parser
    handles it.
    """
    def replace(m):
        fodder = m.group(1)
        indicator_anchor = m.group(2)
        return f"*({fodder}) [{indicator_anchor}]"
    return _ANAGRAMMED_GLOSS_RE.sub(replace, blog)


def tokenise(blog: str) -> list:
    blog = _convert_prose_deletion(blog)
    blog = _convert_substring_deletion(blog)
    blog = _convert_prose_positional(blog)
    blog = _convert_anagrammed_gloss(blog)
    blog = _convert_anagram_of(blog)
    # Strip tilde-separators within uppercase-letter sequences. Some
    # TFTT writers use '~' to mark a charade joint inside a piece
    # (e.g. UN~TESTED, CHA~I). The tilde adds no letters and isn't
    # an op marker — drop it so the piece tokenises as one chunk.
    blog = re.sub(r"(?<=[A-Z])~(?=[A-Z])", "", blog)
    # Glue space-separated curly-brace deletion fragments to neighbouring
    # uppercase pieces. Some blogs write 'INK {ling}' with whitespace
    # rather than the canonical 'INK{ling}'. Same on the leading side
    # for '{p} INTER' → '{p}INTER'.
    blog = _LOOSE_CURLY_RE.sub(r"\1\2", blog)
    blog = _LOOSE_CURLY_TRAIL_RE.sub(r"\1\2", blog)
    # Convert LETTER(lowercase)LETTER pattern (heart-style deletion
    # written with parens, e.g. L(ad)Y) into curly-brace form so it
    # parses as ONE piece with heart deletion. Only fires when there's
    # no whitespace around the parens — pure attached form.
    blog = re.sub(
        r"([A-Z]+)\(([a-z]+)\)(?=[A-Z])",
        lambda m: f"{m.group(1)}{{{m.group(2)}}}",
        blog)
    # Truncate blog at the first ". " that ends the analysis sentence —
    # after that point bloggers add explanatory prose ("Japan's
    # legislature, for example...") whose stray uppercase tokens get
    # mis-parsed as PIECEs. We keep period-prefixed abbreviations like
    # R.E. by requiring a SPACE after the period, AND preceded by a
    # closing paren / lowercase letter / digit (signals end of clue
    # analysis, not mid-abbreviation).
    m = re.search(r"(?<=[)a-z\]0-9 ])\.\s+[A-Z]", blog)
    if m:
        blog = blog[:m.start() + 1]
    # Also cut at common explanation markers — phrases bloggers use
    # to start prose commentary after the wordplay analysis.
    explanation_markers = re.compile(
        r"(?:^|[,.;]\s+)(?:in\s+the\s+case\s+of|"
        r"as\s+in|i\.e\.|this\s+is|meaning|where|"
        r"who\s+lived|literally|"
        r"a\s+chestnut|here|"
        r"and\s+a\s+chestnut)\b",
        re.IGNORECASE)
    m2 = explanation_markers.search(blog)
    if m2:
        # Cut before the marker phrase
        blog = blog[:m2.start()]
    events = []
    pos = 0
    n = len(blog)
    while pos < n:
        # Skip whitespace
        if blog[pos] in " \t\n":
            pos += 1
            continue
        # `->` derivation arrow (TODO #20): consume and skip the next
        # PIECE token (it duplicates the preceding piece+indicator's
        # transformed value).
        if blog[pos] == "-" and pos + 1 < n and blog[pos + 1] == ">":
            pos += 2
            # Skip whitespace
            while pos < n and blog[pos] in " \t":
                pos += 1
            # Skip the following PIECE token if any
            m = _PIECE_RE.match(blog, pos)
            if m and m.group(0) and any(c.isupper() for c in m.group(0)):
                pos = m.end()
            # Also skip following CHARADE_JOIN/op-word context until
            # we hit something structural — actually just continue
            # processing from here; the next CHARADE_JOIN or PIECE
            # naturally resumes.
            continue
        # ASTERISK_ANAGRAM prefix: *(FODDER)
        if blog[pos] == "*" and pos + 1 < n and blog[pos + 1] == "(":
            close = blog.find(")", pos + 2)
            if close > 0:
                content = blog[pos + 2:close].strip()
                events.append(Event("ANAGRAM_FODDER", content, pos))
                pos = close + 1
                continue
            pos += 1
            continue
        # INDICATOR: [phrase]
        if blog[pos] == "[":
            close = blog.find("]", pos)
            if close > 0:
                inner = blog[pos + 1:close].strip()
                events.append(Event("INDICATOR", inner, pos))
                pos = close + 1
                continue
            pos += 1
            continue
        # PAREN content: ANAGRAM postfix? GLOSS | INSERT | DROP
        if blog[pos] == "(":
            close = blog.find(")", pos)
            if close > 0:
                content = blog[pos + 1:close].strip()
                # Look ahead for trailing `*` (postfix anagram)
                after = close + 1
                if after < n and blog[after] == "*":
                    events.append(Event("ANAGRAM_FODDER", content, pos))
                    pos = after + 1
                    continue
                # Disambiguation: DROP only when paren is DIRECTLY
                # attached to an uppercase letter (no whitespace).
                # Otherwise it's a GLOSS (or INSERT for all-uppercase).
                attached = (pos > 0 and blog[pos - 1].isalpha())
                if re.fullmatch(r"[A-Z]+", content):
                    events.append(Event("INSERT", content, pos))
                elif attached and re.fullmatch(r"[a-z]{1,5}", content):
                    events.append(Event("DROP", content.upper(), pos))
                else:
                    events.append(Event("GLOSS", content, pos))
                pos = close + 1
                continue
            pos += 1
            continue
        # MIXED-CASE positional notation: e.g. Bottle, bASIn
        m = _MIXED_CASE_RE.match(blog, pos)
        if m:
            tok = m.group(0)
            mc = _classify_mixed_case(tok)
            if mc is not None:
                kind, kept, source = mc
                events.append(Event("POSITIONAL",
                                     f"{kind}|{kept}|{source}", pos))
                pos = m.end()
                continue
        # PIECE: uppercase letter run with optional curly/bracket
        m = _PIECE_RE.match(blog, pos)
        if m and m.group(0) and any(c.isupper() for c in m.group(0)):
            tok = m.group(0)
            # Postfix-asterisk anagram: WORD* with no parens
            # e.g. "LOAVES B*" or "FACELIFT* D" — the asterisk
            # immediately after a piece marks the piece as anagram
            # fodder. Emit ANAGRAM_FODDER event for the matched
            # uppercase letters.
            after_end = m.end()
            if (after_end < n and blog[after_end] == "*"
                    and re.fullmatch(r"[A-Z]+", tok)):
                events.append(Event("ANAGRAM_FODDER", tok, pos))
                pos = after_end + 1
                continue
            # Special case: 1-char uppercase PIECE followed by whitespace
            # then a bareword that's NOT an op-word/charade-join — TFTT
            # often writes "B ottle" (with space) for first-letter-of-bottle.
            # Combine into a POSITIONAL event.
            if len(tok) == 1 and tok.isalpha():
                end = m.end()
                ws_end = end
                while ws_end < n and blog[ws_end] in " \t":
                    ws_end += 1
                bw = _BAREWORD_RE.match(blog, ws_end)
                if bw:
                    bw_word = bw.group(0).rstrip(".").lower()
                    if (bw_word not in OP_WORDS
                            and bw_word not in CHARADE_JOIN_WORDS
                            and len(bw_word) >= 2):
                        combined = tok + bw.group(0)
                        mc = _classify_mixed_case(combined)
                        if mc:
                            kind, kept, source = mc
                            events.append(Event(
                                "POSITIONAL",
                                f"{kind}|{kept}|{source}", pos))
                            pos = bw.end()
                            continue
            # Alternate-letter detection: {e}N{g}A{g}E or N{g}A{g}E etc.
            # Pattern of single drops alternating with single keeps -> positional.
            alt_parts = re.findall(r"\{[a-z]\}|\[[a-z]\]|[A-Z]", tok)
            if len(alt_parts) >= 4:
                kept_chars = [p for p in alt_parts if len(p) == 1]
                drop_chars = [p[1] for p in alt_parts if len(p) > 1]
                # Reconstruct full source by concatenating parts in order
                full_letters = []
                for p in alt_parts:
                    if len(p) == 1:
                        full_letters.append(p.lower())
                    else:
                        full_letters.append(p[1].lower())
                full = "".join(full_letters)
                kept = "".join(kept_chars)
                if kept_chars and drop_chars and len(kept) >= 2:
                    # Determine pattern: drop-keep-drop-keep or keep-drop-keep-drop
                    role_pattern = "".join(
                        "K" if len(p) == 1 else "D" for p in alt_parts)
                    is_alternating = (role_pattern.startswith("DKDK") or
                                       role_pattern.startswith("KDKD"))
                    if is_alternating:
                        events.append(Event("POSITIONAL",
                                              f"alternate|{kept}|{full}",
                                              pos))
                        pos = m.end()
                        continue
            events.append(Event("PIECE", tok, pos))
            pos = m.end()
            continue
        # BAREWORD: op or charade-join
        m = _BAREWORD_RE.match(blog, pos)
        if m:
            word = m.group(0).rstrip(".").lower()
            if word in OP_WORDS:
                # Suppress container "in"/"of" when preceded by a
                # positional-letters phrase like "alternating letters",
                # "first letters of", "interior letters", "every other".
                # In those contexts the preposition is a locator, not an
                # op-word — the positional extraction is already encoded
                # via mixed-case piece tokens.
                op_kind = OP_WORDS[word][0]
                if op_kind == "container" and word in ("in", "of"):
                    # Look back over recent events for "letters" or
                    # "alternating" / "alternate" / "first" / "last" /
                    # "interior" / "every other" within ~5 words.
                    look_back = blog[max(0, pos - 60):pos].lower()
                    if re.search(
                        r"\b(?:alternating|alternate|every\s+other|"
                        r"first|last|interior|outer|odd|even)\s+"
                        r"letters?\b", look_back):
                        # locator, skip
                        pos = m.end()
                        continue
                events.append(Event("OP_WORD", word, pos))
            elif word in CHARADE_JOIN_WORDS:
                events.append(Event("CHARADE_JOIN", word, pos))
            # else noise
            pos = m.end()
            continue
        # CHARADE_JOIN punctuation
        if blog[pos] in "+,;":
            events.append(Event("CHARADE_JOIN", blog[pos], pos))
            pos += 1
            continue
        # Other — skip
        pos += 1
    return events


# --- Stage 2 — Build piece records from events ---------------------------

@dataclass
class PieceRecord:
    value: str            # the contributed letters (post-deletion)
    full_value: str       # pre-deletion (== value if no deletion)
    sub_kind: Optional[str] = None  # tail/head/heart/outer (deletion)
                                      # OR first/last/outer/middle (positional)
    role_hint: str = "synonym"      # synonym / abbreviation / literal /
                                      # positional
    is_anagram_fodder: bool = False
    is_outer: bool = False          # for container insertion patterns
    inner_piece: Optional["PieceRecord"] = None
    gloss: Optional[str] = None     # if a gloss followed the piece
    positional_kind: Optional[str] = None  # for positional pieces:
                                             # first/last/outer/middle/alternate


def _expand_piece_token(token: str) -> tuple:
    """Resolve TFTT deletion in a piece token. Mirrors the helper in
    word_role_mapper but kept local for clarity.
    """
    if "{" not in token and "[" not in token:
        kept = token.replace(".", "")
        return kept, kept, None
    parts_kept, parts_dropped, brace_pos = [], [], []
    pos = 0
    while pos < len(token):
        if token[pos] in "{[":
            close_char = "}" if token[pos] == "{" else "]"
            close = token.index(close_char, pos)
            parts_dropped.append(token[pos + 1:close].upper())
            brace_pos.append((pos, close))
            pos = close + 1
        else:
            j = pos
            while j < len(token) and token[j] not in "{[":
                j += 1
            parts_kept.append(token[pos:j])
            pos = j
    kept = "".join(parts_kept).replace(".", "")
    dropped = "".join(parts_dropped)
    if not brace_pos:
        return kept, kept, None
    n = len(brace_pos)
    if n >= 2 and brace_pos[0][0] == 0 \
            and brace_pos[-1][1] == len(token) - 1:
        # outer — letters at both ends dropped
        full = "".join(parts_dropped[:1]) + kept + "".join(
            parts_dropped[-1:])
        return kept, full, "outer"
    p0, c0 = brace_pos[0]
    if p0 == 0:
        # head: dropped letters on the left
        full = dropped + kept
        return kept, full, "head"
    if c0 == len(token) - 1:
        # tail: dropped on the right
        full = kept + dropped
        return kept, full, "tail"
    # heart: drop in middle. Reconstruct full by walking the original
    # token in order — kept and dropped fragments interleave at their
    # actual positions, so a positional concat is correct.
    full_parts = []
    pos = 0
    while pos < len(token):
        if token[pos] in "{[":
            close_char = "}" if token[pos] == "{" else "]"
            close = token.index(close_char, pos)
            full_parts.append(token[pos + 1:close].upper())
            pos = close + 1
        else:
            j = pos
            while j < len(token) and token[j] not in "{[":
                j += 1
            full_parts.append(token[pos:j])
            pos = j
    full = "".join(full_parts).replace(".", "")
    return kept, full, "heart"


def build_pieces(events: list) -> list:
    """Walk the event stream, group into PieceRecord list with charade
    boundaries detected via CHARADE_JOIN events.
    """
    pieces = []
    pending_outer = None   # accumulating an outer piece (e.g. A(POST)LE)
    last_piece = None
    last_op = None
    # When an "anagram" OP_WORD appears, subsequent PIECEs (until a
    # CHARADE_JOIN) are anagram fodder — their letters come literally
    # from the clue, not via a synonym lookup.
    anagram_active = False
    for ev in events:
        if ev.kind == "PIECE":
            kept, full, sub = _expand_piece_token(ev.value)
            if pending_outer is not None:
                # Continuation of an outer piece: A + (inner) + LE
                # Append `kept` to the outer's value
                pending_outer.value += kept
                pending_outer.full_value += full
                last_piece = pending_outer
                continue
            role_hint = "literal" if anagram_active else "synonym"
            rec = PieceRecord(value=kept, full_value=full,
                               sub_kind=sub,
                               role_hint=role_hint,
                               is_anagram_fodder=anagram_active)
            pieces.append(rec)
            last_piece = rec
        elif ev.kind == "INSERT":
            # An INSERT (uppercase letters in parens) creates an inner
            # piece and turns the immediately-preceding piece into the
            # outer of a container.
            if last_piece is not None:
                inner = PieceRecord(value=ev.value, full_value=ev.value,
                                     sub_kind=None)
                last_piece.is_outer = True
                last_piece.inner_piece = inner
                pieces.append(inner)
                pending_outer = last_piece
            else:
                # Standalone INSERT — treat as a piece
                rec = PieceRecord(value=ev.value, full_value=ev.value)
                pieces.append(rec)
        elif ev.kind == "DROP":
            # Apply to the most recent piece as a deletion subkind
            if last_piece is not None and last_piece.sub_kind is None:
                last_piece.full_value = last_piece.value + ev.value
                last_piece.sub_kind = "tail"
        elif ev.kind == "GLOSS":
            if last_piece is not None and last_piece.gloss is None:
                last_piece.gloss = ev.value
        elif ev.kind == "ANAGRAM_FODDER":
            # *(FODDER) or (fodder)* — anagram of the contained letters
            letters = "".join(c for c in ev.value if c.isalpha()).upper()
            rec = PieceRecord(value=letters, full_value=letters,
                               is_anagram_fodder=True,
                               role_hint="literal",
                               gloss=ev.value)  # for source anchoring
            pieces.append(rec)
            last_piece = rec
            anagram_active = True
        elif ev.kind == "POSITIONAL":
            # Mixed-case positional notation produced by the tokeniser.
            # ev.value is "kind|kept|source"
            try:
                kind, kept, source = ev.value.split("|", 2)
            except ValueError:
                continue
            rec = PieceRecord(
                value=kept, full_value=kept,
                role_hint="positional",
                positional_kind=kind,
                gloss=source,  # source word = lowercase form of token
            )
            pieces.append(rec)
            last_piece = rec
        elif ev.kind in ("CHARADE_JOIN",):
            pending_outer = None  # close any pending outer
            last_piece = None
            anagram_active = False
        elif ev.kind == "OP_WORD":
            last_op = OP_WORDS.get(ev.value)
            pending_outer = None
            last_piece = None
            if last_op and last_op[0] == "anagram":
                anagram_active = True
            else:
                anagram_active = False
        # INDICATOR events handled at indicator-anchoring stage
    return pieces


# --- Stage 3 — DB-anchor pieces to clue words ---------------------------

_STOP_WORDS = {"of", "for", "to", "the", "a", "an", "in", "is", "by"}


def _stem(w: str) -> str:
    """Trivial singular/plural stemmer: strip trailing 's', 'es' if longer."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _norm_match(w: str) -> str:
    """Normalised form for matching: lowercase, strip punctuation,
    strip apostrophe-s, basic stem."""
    w = (w or "").lower()
    # Strip surrounding punctuation
    w = w.strip(".,;:!?\"'()-‘’“”")
    # Strip trailing 's (possessive/contraction)
    if w.endswith("'s") or w.endswith("’s"):
        w = w[:-2]
    elif w.endswith("s'"):
        w = w[:-2]
    return _stem(w)


def _phrase_in_clue_strict(phrase: str, clue_words: list, used: set
                             ) -> Optional[tuple]:
    """Strict phrase match — finds an exact contiguous span containing
    every word of `phrase` in the SAME order. Used for bracket
    indicators where the phrase is bound (e.g. "to begin with")."""
    target = [_norm_match(w) for w in re.findall(r"[A-Za-z'‘’]+", phrase or "")]
    target = [w for w in target if w]
    if not target:
        return None
    n_t = len(target)
    clue_norm = [_norm_match(w) for w in clue_words]
    for i in range(len(clue_words) - n_t + 1):
        if any(j in used for j in range(i, i + n_t)):
            continue
        if clue_norm[i:i + n_t] == target:
            return (i, i + n_t)
    return None


def _phrase_in_clue(phrase: str, clue_words: list, used: set
                     ) -> Optional[tuple]:
    """Find the smallest contiguous span of clue_words containing the
    content words of `phrase`.

    Rules:
      - Strip punctuation, apostrophe-s; basic stem normalisation.
      - Ignore stop words (of/for/to/the/a/an/in/is/by) in the phrase.
      - Drop phrase content words that aren't in the clue (TODO #13).
      - Find the smallest contiguous span containing every remaining
        phrase word, in any order.

    Returns (start, end_exclusive) or None.
    """
    raw_words = re.findall(r"[A-Za-z'‘’]+", phrase or "")
    target_norm = [_norm_match(w) for w in raw_words]
    target = [w for w in target_norm if w and w not in _STOP_WORDS]
    if not target:
        return None
    clue_norm = [_norm_match(w) for w in clue_words]
    # Drop target words that don't appear in the clue (relaxed match)
    target_in_clue = [w for w in target if w in clue_norm]
    if not target_in_clue:
        return None
    target_set = set(target_in_clue)
    # Find the smallest contiguous span containing every word in target_set,
    # avoiding 'used' indices.
    best = None
    n = len(clue_words)
    for i in range(n):
        if i in used:
            continue
        seen = set()
        for j in range(i, n):
            if j in used:
                break
            if clue_norm[j] in target_set:
                seen.add(clue_norm[j])
            if seen >= target_set:
                length = j - i + 1
                if best is None or length < best[1] - best[0]:
                    best = (i, j + 1)
                break
    return best


def _db_lookup_for_value(value: str, conn) -> set:
    """Return set of words (lowercase) that have `value` as a synonym
    or abbreviation in the live DB.
    """
    out = set()
    val = (value or "").upper()
    if not val:
        return out
    for row in conn.execute(
            "SELECT LOWER(word) FROM synonyms_pairs "
            "WHERE UPPER(synonym) = ? COLLATE NOCASE", (val,)):
        out.add(row[0])
    # Abbreviations live in the `wordplay` table (category='abbreviation')
    for row in conn.execute(
            "SELECT LOWER(indicator) FROM wordplay "
            "WHERE UPPER(substitution) = ? COLLATE NOCASE "
            "AND category = 'abbreviation'", (val,)):
        out.add(row[0])
    return out


_OP_PRIORITY = {
    # Specific structural ops first
    "anagram": 1, "reversal": 1, "container": 1, "insertion": 1,
    "hidden": 1, "homophone": 1,
    # Positional/extraction (common for indicators next to positional pieces)
    "parts": 2, "acrostic": 2, "alternating": 2, "selection": 2,
    # Deletion last — deletion is correct when explicitly indicated,
    # but for ambiguous words like "initially" (which can be parts/first
    # OR deletion/head) the parts interpretation is usually right.
    "deletion": 3,
}


def _indicator_types_for_word(phrase: str, conn) -> list:
    """Query indicators DB for op types. Returns (wordplay_type, subtype)
    tuples ORDERED by op priority (specific ops before positional/parts)
    and confidence."""
    out = []
    p = (phrase or "").lower().strip(".,;:!?\"'()-‘’")
    if not p:
        return out
    if p.endswith("'s") or p.endswith("’s"):
        p = p[:-2]
    rows = conn.execute(
        "SELECT wordplay_type, subtype, confidence FROM indicators "
        "WHERE LOWER(word)=? COLLATE NOCASE", (p,)).fetchall()
    # Fallback to individual content words
    if not rows:
        words = [w for w in re.findall(r"[a-z']+", p)
                 if w not in _STOP_WORDS]
        for w in words:
            for r in conn.execute(
                    "SELECT wordplay_type, subtype, confidence FROM "
                    "indicators WHERE LOWER(word)=? COLLATE NOCASE", (w,)):
                rows.append(tuple(r))
    # Order: specific-op priority, then high>medium>low confidence
    _conf_score = {"high": 0, "medium": 1, "low": 2}
    rows.sort(key=lambda r: (_OP_PRIORITY.get(r[0], 99),
                              _conf_score.get(r[2], 9)))
    return [(r[0], r[1]) for r in rows]


_DB_OP_TO_FORM_OP = {
    "container": "container",
    "insertion": "container",
    "anagram": "anagram",
    "reversal": "reversal",
    "deletion": "deletion",
    "hidden": "hidden",
    "homophone": "homophone",
    "acrostic": "acrostic",
    "parts": "positional",
    "alternating": "positional",
    "selection": "positional",
}


def _anchor_piece(piece: PieceRecord, clue_words: list,
                   used: set, conn) -> Optional[tuple]:
    """Try to anchor a piece's value to a clue word.

    Returns (span, mechanism, db_confirmed) or None.
    db_confirmed is True iff the DB has the (clue_phrase, value) pair.
    When False, the anchor was made via gloss text only — the caller
    should treat this as a shadow-DB candidate.
    """
    val = piece.value or piece.full_value
    if not val:
        return None
    mech_for_value = ("abbreviation" if len(val) <= 2 else "synonym")
    db_words = _db_lookup_for_value(val, conn)

    # 1. Gloss-first anchor: if blog supplied a gloss, find that exact
    #    phrase in the clue. Try STRICT match first (preserves stop
    #    words and order — needed for "(a bit)", "(Nutter in)" where
    #    every word is meaningful). Fall back to relaxed matching if
    #    strict fails.
    if piece.gloss:
        span = _phrase_in_clue_strict(piece.gloss, clue_words, used)
        if not span:
            span = _phrase_in_clue(piece.gloss, clue_words, used)
        if span:
            phrase_lower = " ".join(_norm(w) for w in
                                     clue_words[span[0]:span[1]])
            confirmed = phrase_lower in db_words
            # If literal letters of the clue phrase equal the value,
            # it's a literal piece regardless of DB.
            phrase_letters = "".join(_norm_letters(w) for w in
                                      clue_words[span[0]:span[1]])
            if phrase_letters == val:
                return span, "literal", True
            return span, mech_for_value, confirmed

    # 2a. For curly-brace deletion pieces, try matching FULL letters
    # (pre-deletion) to a clue word. {g}OT has full="GOT" → match "got".
    if piece.sub_kind and piece.full_value and piece.full_value != val:
        for i, w in enumerate(clue_words):
            if i in used:
                continue
            if _norm_letters(w) == piece.full_value:
                return (i, i + 1), "literal", True

    # 2. Exact letters match (literal): clue word's letters == value
    for i, w in enumerate(clue_words):
        if i in used:
            continue
        if _norm_letters(w) == val:
            return (i, i + 1), "literal", True

    # 3. DB-driven anchor when no gloss
    if db_words:
        clue_norm = [_norm(w) for w in clue_words]
        # Multi-word phrases first (longest), then single words
        for span_size in (4, 3, 2):
            for i in range(len(clue_words) - span_size + 1):
                if any(j in used for j in range(i, i + span_size)):
                    continue
                phrase = " ".join(clue_norm[i:i + span_size])
                if phrase in db_words:
                    return (i, i + span_size), mech_for_value, True
        for i, w in enumerate(clue_words):
            if i in used:
                continue
            if clue_norm[i] in db_words:
                return (i, i + 1), mech_for_value, True
    return None


# --- Stage 4 — Definition assignment ------------------------------------

def _assign_definition(clue_words: list, used: set,
                        clue_definition: Optional[str],
                        skip_phrase_match: bool = False) -> Optional[tuple]:
    """Pick the definition span. Prefer the value of `clues.definition`
    if it can be found in the clue text; else longest end-or-start run.

    When `skip_phrase_match` is True, the `clue_definition` phrase
    isn't used — only the longest-untagged-run heuristic. Used when
    `clue_definition` was deemed conflicting with a piece's gloss
    or an indicator's anchor word.
    """
    if clue_definition and not skip_phrase_match:
        span = _phrase_in_clue(clue_definition, clue_words, used)
        if span:
            return span
    # Fallback: contiguous unaccounted runs at start/end
    untagged = [i for i in range(len(clue_words)) if i not in used]
    if not untagged:
        return None
    runs = []
    cur = [untagged[0]]
    for i in untagged[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    n = len(clue_words)
    edge = [r for r in runs if r[0] == 0 or r[-1] == n - 1]
    chosen = max(edge, key=len) if edge else max(runs, key=len)
    return (chosen[0], chosen[-1] + 1)


# --- Stage 5 — Indicator by-elimination ----------------------------------

def _attach_floating_ops(floating_ops: list, used: set,
                          clue_words: list, conn) -> list:
    """Attach floating ops to unaccounted clue words.

    Priority (TODOs #16, #18):
      1. Skip leftovers that are in LINK_WORDS, UNLESS the word is in
         the indicators DB as the matching op type (then the indicator
         role overrides — common for "in"/"of" which are dual-use).
      2. Among remaining, prefer leftovers that are in indicators DB
         as the matching op type.
      3. Fall back to first non-link leftover.

    Returns list of (span, op, sub_kind) tags.
    """
    from .verifier import LINK_WORDS as _LW
    out = []
    leftover = sorted(i for i in range(len(clue_words)) if i not in used)
    for op, sub in floating_ops:
        if not leftover:
            break
        chosen = None
        # Pass 1: prefer non-link words tagged in DB for THIS op
        for idx in leftover:
            w_norm = clue_words[idx].lower().strip(".,;:!?\"'()-")
            if w_norm in _LW:
                continue
            types = _indicator_types_for_word(clue_words[idx], conn)
            for wp_type, _sub in types:
                form_op = _DB_OP_TO_FORM_OP.get(wp_type)
                if form_op == op:
                    chosen = idx
                    break
            if chosen is not None:
                break
        # Pass 2: any non-link word
        if chosen is None:
            for idx in leftover:
                w_norm = clue_words[idx].lower().strip(".,;:!?\"'()-")
                if w_norm not in _LW:
                    chosen = idx
                    break
        # Pass 3: link word that's DB-tagged for THIS op (last-resort
        # — covers cases like 'in' as container indicator).
        if chosen is None:
            for idx in leftover:
                types = _indicator_types_for_word(clue_words[idx], conn)
                for wp_type, _sub in types:
                    form_op = _DB_OP_TO_FORM_OP.get(wp_type)
                    if form_op == op:
                        chosen = idx
                        break
                if chosen is not None:
                    break
        # Pass 4: any leftover at all
        if chosen is None:
            chosen = leftover[0]
        leftover.remove(chosen)
        used.add(chosen)
        out.append(((chosen, chosen + 1), op, sub))
    return out


# --- Top-level entry point ------------------------------------------------

_DD_RE = re.compile(r"\bdouble\s+def(?:initions?)?\b", re.IGNORECASE)
_CD_RE = re.compile(
    r"\bcryptic\s+def(?:inition)?s?\b|"
    r"\b(?:and\s+)?lit(?:eral)?\b\.?", re.IGNORECASE)
_HIDDEN_RE = re.compile(
    r"^\s*(?:hidden|concealed|lurking)\b\s*"
    r"(?:reversed\s*)?"
    r"(?:in|across|inside|within|of)?\s*", re.IGNORECASE)


def _is_dd_blog(blog: str) -> bool:
    return bool(blog and _DD_RE.search(blog))


def _is_cd_blog(blog: str) -> bool:
    if not blog:
        return False
    if _CD_RE.search(blog):
        return True
    return False


def _is_hidden_blog(blog: str) -> bool:
    """Detect 'Hidden in croquetteS A LA MInute' style blogs."""
    if not blog:
        return False
    return bool(_HIDDEN_RE.match(blog.strip()))


def _build_hidden_mapping(clue_text: str, answer: str,
                            clue_words: list, blog: str, conn,
                            shadow_candidates: list,
                            clue_definition: Optional[str] = None
                            ) -> Mapping:
    """Build a Mapping for 'Hidden in WORDS' blogs.

    The blog (after stripping the 'Hidden in' prefix) lists the source
    words that contain the answer as a hidden span. Each word becomes
    a literal leaf; the form is `hidden [indicator] of [literal*]`.
    """
    n = len(clue_words)
    is_reversed = bool(re.search(r"hidden\s+reversed", blog,
                                   re.IGNORECASE))
    rest = _HIDDEN_RE.sub("", blog.strip()).strip().rstrip(".,;:")
    source_words = re.findall(r"[A-Za-z]+", rest)
    if not source_words:
        return None
    src_norm = [w.lower() for w in source_words]
    clue_norm = [_norm(w) for w in clue_words]
    span_start = None
    for i in range(n - len(src_norm) + 1):
        if [clue_norm[i + j] for j in range(len(src_norm))] == src_norm:
            span_start = i
            break
    if span_start is None:
        return None
    span_end = span_start + len(source_words)

    used = set(range(span_start, span_end))
    tags = []
    # Each source word becomes a literal piece
    for i, w in enumerate(source_words):
        idx = span_start + i
        tags.append(Tag(
            span=(idx, idx + 1),
            words=[clue_words[idx]],
            role="piece",
            value=clue_words[idx].upper().replace(" ", ""),
            mechanism="literal",
            notes=["hidden_source"]))

    # Definition (use clue_definition if available)
    if clue_definition:
        def_span = _phrase_in_clue(clue_definition, clue_words, used)
        if def_span:
            tags.append(Tag(span=def_span,
                              words=clue_words[def_span[0]:def_span[1]],
                              role="definition"))
            used.update(range(def_span[0], def_span[1]))

    # Anchor a hidden indicator to a leftover clue word (DB-priority)
    floating_ops = [("hidden", None)]
    if is_reversed:
        floating_ops.append(("reversal", None))
    op_tags = _attach_floating_ops(floating_ops, used, clue_words, conn)
    for spn, op, sub in op_tags:
        words = clue_words[spn[0]:spn[1]]
        tags.append(Tag(span=spn, words=words, role="indicator",
                          operation=op, sub_kind=sub,
                          notes=["hidden-indicator anchored by elimination"]))
        # Also write a shadow candidate (this often lifts indicator coverage)
        shadow_candidates.append({
            "kind": "indicator",
            "source_word": " ".join(words),
            "value": " ".join(words),
            "operation": op,
            "subtype": sub,
            "evidence": "hidden indicator anchored from clue residue",
        })

    # Any remaining unaccounted go as 'unaccounted'
    leftover = [i for i in range(n) if i not in used]
    if leftover:
        runs = []
        cur = [leftover[0]]
        for i in leftover[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
        runs.append(cur)
        for run in runs:
            sp = (run[0], run[-1] + 1)
            tags.append(Tag(span=sp,
                              words=clue_words[sp[0]:sp[1]],
                              role="unaccounted"))

    m = Mapping(
        clue_text=clue_text, answer=answer,
        clue_words=clue_words, tags=tags,
        unmapped_pieces=[],
        floating_ops=[],  # consumed via op_tags
        notes=["hidden form"],
    )
    m.shadow_candidates = shadow_candidates  # type: ignore
    return m


def _build_cd_mapping(clue_text: str, answer: str, clue_words: list,
                       conn, shadow_candidates: list) -> Mapping:
    """Build a Mapping for a cryptic-definition clue.

    The whole clue is the definition; there's no wordplay to assemble.
    Form: a single synonym leaf where source_word IS the clue and
    value IS the answer. Verifier passes assembly trivially. Bridge
    on the leaf checks whether the clue (or a sub-phrase) appears in
    definition_answers_augmented or synonyms_pairs for the answer —
    if not, it becomes a shadow candidate.
    """
    n = len(clue_words)
    tags = []
    if n > 0:
        tag = Tag(span=(0, n), words=list(clue_words),
                   role="piece", value=answer,
                   mechanism="synonym",
                   notes=["cryptic_definition"])
        tags.append(tag)
        if not _has_definition_in_db(" ".join(clue_words), answer, conn):
            shadow_candidates.append({
                "kind": "definition",
                "source_word": " ".join(clue_words),
                "value": answer,
                "evidence": "CD blog; whole clue not in DB as definition",
            })
    m = Mapping(
        clue_text=clue_text, answer=answer,
        clue_words=clue_words, tags=tags,
        unmapped_pieces=[],
        floating_ops=[("cryptic_definition", None)],
        notes=["CD form"],
    )
    m.shadow_candidates = shadow_candidates  # type: ignore
    return m


def _has_definition_in_db(phrase: str, answer: str, conn) -> bool:
    p = (phrase or "").lower().strip(".,;:!?\"'()-")
    a = (answer or "").upper()
    # Build candidate forms: original, apostrophe-s stripped, trailing-s
    # stripped. Try each.
    candidates = [p]
    if p.endswith("'s") or p.endswith("’s"):
        candidates.append(p[:-2])
    elif p.endswith(" s"):
        candidates.append(p[:-2])
    elif p.endswith("s") and not p.endswith("ss"):
        candidates.append(p[:-1])
    for cand in candidates:
        if not cand:
            continue
        rows = conn.execute(
            "SELECT 1 FROM definition_answers_augmented "
            "WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
            (cand, a)).fetchone()
        if rows:
            return True
        rows = conn.execute(
            "SELECT 1 FROM synonyms_pairs "
            "WHERE LOWER(word)=? AND UPPER(synonym)=? LIMIT 1",
            (cand, a)).fetchone()
        if rows:
            return True
    return False


def _build_dd_mapping(clue_text: str, answer: str, clue_words: list,
                       conn, shadow_candidates: list) -> Mapping:
    """Build a Mapping for a double-definition clue.

    Try every split point; pick the one where both halves match the
    answer in the DB (definition_answers_augmented / synonyms_pairs).
    Falls back to midpoint.
    """
    n = len(clue_words)
    chosen = None
    if n >= 2:
        for i in range(1, n):
            left = " ".join(clue_words[:i])
            right = " ".join(clue_words[i:])
            l_ok = _has_definition_in_db(left, answer, conn)
            r_ok = _has_definition_in_db(right, answer, conn)
            if l_ok and r_ok:
                chosen = i
                break
        if chosen is None:
            # Try with apostrophe-s stripping
            for i in range(1, n):
                left = " ".join(clue_words[:i]).rstrip("’'s ")
                right = " ".join(clue_words[i:]).rstrip("’'s ")
                if _has_definition_in_db(left, answer, conn) \
                        and _has_definition_in_db(right, answer, conn):
                    chosen = i
                    break
        if chosen is None:
            chosen = n // 2
    else:
        chosen = 1

    tags = []
    if chosen and 0 < chosen < n:
        left_words = clue_words[:chosen]
        right_words = clue_words[chosen:]
        l_ok = _has_definition_in_db(" ".join(left_words), answer, conn)
        r_ok = _has_definition_in_db(" ".join(right_words), answer, conn)
        tags.append(Tag(span=(0, chosen), words=left_words,
                          role="piece", value=answer,
                          mechanism="synonym",
                          notes=["dd_left"
                                 + ("" if l_ok else " (db_miss)")]))
        tags.append(Tag(span=(chosen, n), words=right_words,
                          role="piece", value=answer,
                          mechanism="synonym",
                          notes=["dd_right"
                                 + ("" if r_ok else " (db_miss)")]))
        if not l_ok:
            shadow_candidates.append({
                "kind": "definition",
                "source_word": " ".join(left_words),
                "value": answer,
                "evidence": "DD blog; left half not in DB",
            })
        if not r_ok:
            shadow_candidates.append({
                "kind": "definition",
                "source_word": " ".join(right_words),
                "value": answer,
                "evidence": "DD blog; right half not in DB",
            })
    m = Mapping(
        clue_text=clue_text, answer=answer,
        clue_words=clue_words, tags=tags,
        unmapped_pieces=[], floating_ops=[("double_definition", None)],
        notes=["DD form"],
    )
    m.shadow_candidates = shadow_candidates  # type: ignore
    return m


def map_clue_words_db(clue_text: str, answer: str, blog_text: str,
                       conn: sqlite3.Connection,
                       clue_definition: Optional[str] = None
                       ) -> Mapping:
    clue_words = re.findall(r"[A-Za-z][A-Za-z']*", clue_text or "")
    answer_clean = _norm_letters(answer)
    blog = blog_text or ""
    notes: list = []

    # DD short-circuit (TODO #4)
    if _is_dd_blog(blog):
        return _build_dd_mapping(clue_text, answer_clean, clue_words,
                                   conn, [])
    # CD short-circuit (cryptic definition)
    if _is_cd_blog(blog):
        return _build_cd_mapping(clue_text, answer_clean, clue_words,
                                   conn, [])
    # Hidden short-circuit
    if _is_hidden_blog(blog):
        h = _build_hidden_mapping(clue_text, answer_clean, clue_words,
                                    blog, conn, [], clue_definition)
        if h is not None:
            return h

    events = tokenise(blog)
    pieces = build_pieces(events)
    # Collect floating ops (OP_WORD events that didn't get a gloss/bracket
    # attached — track them in event order)
    floating_ops_set = []
    for ev in events:
        if ev.kind == "OP_WORD":
            op_info = OP_WORDS.get(ev.value)
            if op_info and op_info not in floating_ops_set:
                floating_ops_set.append(op_info)
    # If we saw an INSERT (container shorthand), add container op
    if any(ev.kind == "INSERT" for ev in events):
        if ("container", None) not in floating_ops_set:
            floating_ops_set.insert(0, ("container", None))
    # If we saw an ANAGRAM_FODDER, add anagram op
    if any(ev.kind == "ANAGRAM_FODDER" for ev in events):
        if ("anagram", None) not in floating_ops_set:
            floating_ops_set.insert(0, ("anagram", None))

    used = set()
    tags_by_span = {}
    shadow_candidates = []

    # Definition first (when supplied) — but defer when clue.definition
    # is a subset of a piece's gloss (likely the def column captured
    # only part of what's actually a piece source phrase, e.g. "team"
    # in "cricket team" for AXIS).
    def_span_pre = None
    conflicts_with_piece = False
    if clue_definition:
        def_norm = set(_norm_match(w) for w in
                        re.findall(r"[A-Za-z']+", clue_definition))
        for p in pieces:
            if not p.gloss:
                continue
            gloss_norm = set(_norm_match(w) for w in
                              re.findall(r"[A-Za-z']+", p.gloss))
            if not gloss_norm or not def_norm:
                continue
            # Conflict in either direction: def ⊆ gloss (AXIS pattern,
            # piece's gloss is broader than the def_db's slice) OR
            # gloss ⊆ def (COMMERCE pattern, def_db has captured part
            # of a piece's gloss alongside the actual definition).
            if def_norm.issubset(gloss_norm) or gloss_norm.issubset(def_norm):
                # But only block if there's strict OVERLAP — if def
                # contains words the piece doesn't (and vice versa),
                # they're simply different phrases.
                if def_norm & gloss_norm:
                    conflicts_with_piece = True
                    break
        # Also conflict: any indicator-anchor word from blog is in def
        # span. Bloggers often parenthetical the indicator anchor like
        # "(out)" or "[blunders]"; if def_db captured that anchor as
        # part of the definition phrase, the indicator can't anchor.
        if not conflicts_with_piece:
            indicator_words = set()
            for ev in events:
                if ev.kind == "INDICATOR":
                    for w in re.findall(r"[A-Za-z']+", ev.value):
                        indicator_words.add(_norm_match(w))
            if indicator_words & def_norm:
                conflicts_with_piece = True
        if not conflicts_with_piece:
            def_span_pre = _phrase_in_clue(clue_definition, clue_words,
                                              used)
            if def_span_pre:
                words = clue_words[def_span_pre[0]:def_span_pre[1]]
                tags_by_span[def_span_pre] = Tag(
                    span=def_span_pre, words=words, role="definition")
                used.update(range(def_span_pre[0], def_span_pre[1]))

    # Anchor pieces
    piece_records_by_span = {}
    for piece in pieces:
        anchor = _anchor_piece(piece, clue_words, used, conn)
        if anchor:
            span, mech, db_confirmed = anchor
            words = clue_words[span[0]:span[1]]
            # Mechanism precedence:
            # 1. Anagram-fodder pieces are always 'literal'
            # 2. Positional pieces are 'positional' (with kind in sub_kind)
            # 3. Anchor matched by exact letters → 'literal'
            # 4. Otherwise use whichever of role_hint / mech is more
            #    specific.
            if piece.is_anagram_fodder:
                final_mech = "literal"
                effective_sub = piece.sub_kind
            elif piece.role_hint == "positional":
                final_mech = "positional"
                effective_sub = piece.positional_kind
            elif mech == "literal":
                final_mech = "literal"
                effective_sub = piece.sub_kind
            else:
                final_mech = mech or piece.role_hint or "synonym"
                effective_sub = piece.sub_kind
            tag = Tag(span=span, words=words, role="piece",
                       value=piece.value,
                       mechanism=final_mech,
                       sub_kind=effective_sub)
            if not db_confirmed and final_mech in ("synonym", "abbreviation"):
                # Anchor came via gloss text but DB doesn't have the
                # exact phrase — write a shadow candidate so the entry
                # gets queued for the shadow store.
                shadow_candidates.append({
                    "kind": final_mech,
                    "source_word": " ".join(words),
                    "value": piece.value,
                    "evidence": "anchored via blog gloss; DB miss",
                })
                tag.notes.append("db_miss; shadow_candidate")
            tags_by_span[span] = tag
            piece_records_by_span[span] = piece
            used.update(range(span[0], span[1]))
        else:
            # Floating piece — by-elimination later, write as shadow candidate
            shadow_candidates.append({
                "kind": ("abbreviation" if len(piece.value or "") <= 2
                          else "synonym"),
                "source_word": None,  # filled in by-elimination below
                "value": piece.value,
                "evidence": "blog piece, no DB match",
            })

    # Definition (fallback if not pre-anchored above)
    if def_span_pre is None:
        # If pre-anchor was deferred due to conflict, also skip the
        # phrase-match in the fallback to prevent re-finding the same
        # conflicting span. Use only the longest-untagged-run heuristic.
        # AND temporarily mark INDICATOR-event spans as used so the
        # heuristic doesn't grab them either.
        skip_phrase = clue_definition is not None and conflicts_with_piece
        ind_blocked = set()
        if conflicts_with_piece:
            for ev in events:
                if ev.kind != "INDICATOR":
                    continue
                ind_span = (_phrase_in_clue_strict(ev.value, clue_words, used)
                             or _phrase_in_clue(ev.value, clue_words, used))
                if ind_span:
                    for i in range(ind_span[0], ind_span[1]):
                        if i not in used:
                            ind_blocked.add(i)
        for i in ind_blocked:
            used.add(i)
        try:
            def_span = _assign_definition(clue_words, used,
                                            clue_definition,
                                            skip_phrase_match=skip_phrase)
        finally:
            for i in ind_blocked:
                used.discard(i)
        if def_span:
            words = clue_words[def_span[0]:def_span[1]]
            tags_by_span[def_span] = Tag(
                span=def_span, words=words, role="definition")
            used.update(range(def_span[0], def_span[1]))

    # INDICATOR events: bracketed phrases ARE clue-word indicators.
    # Anchor them via phrase match and look up DB for op type.
    bracketed_indicators = []
    for ev in events:
        if ev.kind != "INDICATOR":
            continue
        # Try strict match first (preserves stop words and order),
        # fall back to relaxed match.
        span = _phrase_in_clue_strict(ev.value, clue_words, used)
        if not span:
            span = _phrase_in_clue(ev.value, clue_words, used)
        if not span:
            continue
        types = _indicator_types_for_word(ev.value, conn)
        # Detect adjacent piece kind — if present, prefer the matching
        # indicator type (parts/acrostic for positional pieces;
        # deletion/parts for curly-brace deletion pieces).
        adj_positional = False
        adj_deletion = False
        for psp, ptag in tags_by_span.items():
            if ptag.role != "piece":
                continue
            if not (psp[1] == span[0] or span[1] == psp[0]):
                continue
            if ptag.mechanism == "positional":
                adj_positional = True
            if ptag.sub_kind in ("head", "tail", "outer", "heart"):
                adj_deletion = True
        chosen_op, chosen_sub = None, None
        floating_op_types = {op for op, _ in floating_ops_set}
        # Pass 1: floating-op match
        for wp_type, subtype in types:
            form_op = _DB_OP_TO_FORM_OP.get(wp_type)
            if form_op and form_op in floating_op_types:
                chosen_op, chosen_sub = form_op, subtype
                break
        # Pass 2: context-aware priority for adjacent piece kind
        if chosen_op is None and adj_deletion:
            for wp_type, subtype in types:
                if wp_type in ("deletion", "parts"):
                    chosen_op = _DB_OP_TO_FORM_OP.get(wp_type)
                    chosen_sub = subtype
                    break
        if chosen_op is None and adj_positional:
            for wp_type, subtype in types:
                if wp_type in ("parts", "acrostic", "alternating"):
                    chosen_op = _DB_OP_TO_FORM_OP.get(wp_type)
                    chosen_sub = subtype
                    break
        # Pass 3: take highest-priority DB type
        if chosen_op is None and types:
            wp_type, subtype = types[0]
            chosen_op = _DB_OP_TO_FORM_OP.get(wp_type)
            chosen_sub = subtype
        # Pass 4: fallbacks when DB has no match
        if chosen_op is None and adj_deletion:
            chosen_op = "deletion"
            chosen_sub = None
        if chosen_op is None and adj_positional:
            chosen_op = "positional"
            chosen_sub = None
        words = clue_words[span[0]:span[1]]
        tags_by_span[span] = Tag(
            span=span, words=words, role="indicator",
            operation=chosen_op, sub_kind=chosen_sub,
            notes=[f"bracket-indicator '{ev.value}'"
                   + (f" db_op={chosen_op}" if chosen_op else "")])
        used.update(range(span[0], span[1]))
        if chosen_op:
            bracketed_indicators.append(chosen_op)
            # Also queue as shadow-candidate (so DB grows even when DB had it)
            shadow_candidates.append({
                "kind": "indicator",
                "source_word": " ".join(words),
                "value": " ".join(words),
                "operation": chosen_op,
                "subtype": chosen_sub,
                "evidence": "bracket indicator with DB-confirmed op",
            })

    # Remove ops already satisfied by bracketed indicators
    floating_ops_set = [
        (op, sub) for (op, sub) in floating_ops_set
        if op not in bracketed_indicators
    ]

    # If a single floating op remains AND there's a tagged indicator
    # with operation=None (no DB classification), assign the floating
    # op to it. Common when blog has [bracketed-indicator] but the
    # word isn't in the indicators DB.
    if len(floating_ops_set) == 1:
        unclassified = [s for s, t in tags_by_span.items()
                        if t.role == "indicator" and t.operation is None]
        if len(unclassified) == 1:
            floating_op, floating_sub = floating_ops_set[0]
            unclass_span = unclassified[0]
            old_tag = tags_by_span[unclass_span]
            tags_by_span[unclass_span] = Tag(
                span=unclass_span, words=old_tag.words,
                role="indicator", operation=floating_op,
                sub_kind=floating_sub,
                notes=old_tag.notes + [
                    f"floating_op auto-assigned: {floating_op}"])
            floating_ops_set = []
            # Queue as shadow candidate so pass-2 verifier can find it.
            shadow_candidates.append({
                "kind": "indicator",
                "source_word": " ".join(old_tag.words),
                "value": " ".join(old_tag.words),
                "operation": floating_op,
                "subtype": floating_sub,
                "evidence": "auto-assigned floating-op to unclassified bracket indicator",
            })

    # Merge adjacent {positional, deletion-via-curly-brace} pieces with
    # their bracket indicators. The bracket's words get absorbed into
    # the piece's source span so residue accounts for them.
    _piece_spans = sorted([s for s, t in tags_by_span.items()
                            if t.role == "piece"])
    _matchable_ind_ops = ("positional", "acrostic", "deletion")
    _ind_spans = sorted([s for s, t in tags_by_span.items()
                          if t.role == "indicator"
                          and t.operation in _matchable_ind_ops])
    for pspan in _piece_spans:
        ptag = tags_by_span.get(pspan)
        if ptag is None:
            continue
        # Eligible: positional pieces OR pieces with deletion sub_kind
        is_positional = ptag.mechanism == "positional"
        is_deletion = ptag.sub_kind in ("head", "tail", "outer", "heart")
        if not (is_positional or is_deletion):
            continue
        for ispan in list(_ind_spans):
            if ispan not in tags_by_span:
                continue
            itag = tags_by_span[ispan]
            # Op type compatible with piece kind?
            if is_positional and itag.operation not in (
                    "positional", "acrostic"):
                continue
            if is_deletion and itag.operation not in (
                    "deletion", "positional"):
                continue
            # Adjacent if indicator starts where piece ends (or vice versa)
            if ispan[0] == pspan[1] or ispan[1] == pspan[0]:
                new_start = min(pspan[0], ispan[0])
                new_end = max(pspan[1], ispan[1])
                new_span = (new_start, new_end)
                merged = Tag(
                    span=new_span,
                    words=clue_words[new_start:new_end],
                    role="piece", value=ptag.value,
                    mechanism=ptag.mechanism,
                    sub_kind=ptag.sub_kind,
                    notes=ptag.notes + [
                        f"merged_indicator='{' '.join(itag.words)}'"])
                del tags_by_span[pspan]
                del tags_by_span[ispan]
                tags_by_span[new_span] = merged
                break

    # Floating-op anchoring: residue clue words become indicators
    floating_remaining = list(floating_ops_set)
    op_tags = _attach_floating_ops(floating_remaining, used, clue_words,
                                     conn)
    for span, op, sub in op_tags:
        words = clue_words[span[0]:span[1]]
        tags_by_span[span] = Tag(
            span=span, words=words, role="indicator",
            operation=op, sub_kind=sub,
            notes=["floating-op anchored by elimination"])
        # Emit a shadow candidate for this indicator. The DB may already
        # have it (the writer skips dupes), so this is just a queue.
        shadow_candidates.append({
            "kind": "indicator",
            "source_word": " ".join(words),
            "value": " ".join(words),  # the indicator word itself
            "operation": op,
            "subtype": sub,
            "evidence": "blog implies this op; clue word taken by-elimination",
        })

    # By-elimination: assign unanchored shadow_candidates a source_word
    # from any remaining unaccounted clue words.
    from .verifier import LINK_WORDS as _LW
    leftover = [i for i in range(len(clue_words)) if i not in used]
    # Prefer non-link content words for by-elimination — link words
    # almost never abbreviate to single letters in real cryptic clues,
    # and assigning them produces noisy shadow candidates ("of → F",
    # "and → IN"). Sort leftovers so non-link words come first.
    def _is_link(i):
        w = clue_words[i].lower().strip(".,;:!?\"'()-")
        return w in _LW
    leftover.sort(key=lambda i: (1 if _is_link(i) else 0, i))
    sc_idx = 0
    for cand in shadow_candidates:
        if cand["source_word"] is not None:
            continue
        if sc_idx >= len(leftover):
            break
        idx = leftover[sc_idx]
        cand["source_word"] = clue_words[idx]
        # Mark this candidate as low-confidence — by-elimination
        # assignment has no semantic evidence. Useful for downstream
        # filters when promoting candidates to the live DB.
        word_norm = clue_words[idx].lower().strip(".,;:!?\"'()-")
        is_link = word_norm in _LW
        cand["evidence"] = (
            f"by-elimination on leftover word"
            f"{' (link word, very low confidence)' if is_link else ''}")
        cand["confidence"] = "low" if is_link else "medium"
        # If this is a single-letter abbreviation candidate AND that
        # letter matches the first or last of the source word, it's
        # almost certainly a positional extraction, not an abbreviation.
        # Re-type to keep the shadow DB clean.
        val = cand.get("value") or ""
        if (cand["kind"] == "abbreviation" and len(val) == 1
                and val.isalpha() and len(word_norm) >= 2):
            wupper = word_norm.upper()
            if val.upper() == wupper[0]:
                cand["kind"] = "positional"
                cand["positional_kind"] = "first"
                cand["evidence"] = (cand.get("evidence") or "") + \
                    "; first letter of source"
            elif val.upper() == wupper[-1]:
                cand["kind"] = "positional"
                cand["positional_kind"] = "last"
                cand["evidence"] = (cand.get("evidence") or "") + \
                    "; last letter of source"
        # And add a piece tag pointing at this word
        span = (idx, idx + 1)
        # Use the candidate's (possibly re-typed) kind for the tag mech.
        # Schema-wise, "positional" leaves carry sub_kind=first/last.
        tag_mech = cand["kind"]
        tag_sub = cand.get("positional_kind") if tag_mech == "positional" else None
        tag = Tag(span=span, words=[clue_words[idx]],
                   role="piece", value=cand["value"],
                   mechanism=tag_mech, sub_kind=tag_sub,
                   notes=[f"by-elimination, value={cand['value']}, "
                          f"shadow_candidate"])
        tags_by_span[span] = tag
        used.add(idx)
        sc_idx += 1

    # POS-aware widening: fold adjacent unaccounted content words into
    # neighbouring pieces. Strict rules:
    #   - Only link-listed words may bridge the gap (NOT used indicators
    #     or used pieces — those are real roles, not connectives).
    #   - Prefer the RIGHTWARD piece (the noun that the modifier
    #     typically modifies) over the leftward one.
    from .verifier import LINK_WORDS as _LW
    _piece_spans_now = sorted(
        [s for s, t in tags_by_span.items() if t.role == "piece"])
    for idx in range(len(clue_words)):
        if idx in used:
            continue
        word_norm = clue_words[idx].lower().strip(",.;:!?\"'()-‘’")
        if word_norm in _LW:
            continue  # legitimate link word
        right_span = None
        left_span = None
        for pspan in _piece_spans_now:
            if pspan not in tags_by_span:
                continue
            if idx < pspan[0]:
                gap = list(range(idx + 1, pspan[0]))
                if all(j not in used and clue_words[j].lower().strip(
                        ",.;:!?\"'()-‘’") in _LW for j in gap):
                    if right_span is None or pspan[0] < right_span[0]:
                        right_span = pspan
            elif idx > pspan[1] - 1:
                gap = list(range(pspan[1], idx))
                if all(j not in used and clue_words[j].lower().strip(
                        ",.;:!?\"'()-‘’") in _LW for j in gap):
                    if left_span is None or pspan[1] > left_span[1]:
                        left_span = pspan
        # Prefer rightward fold (modifier-attaches-to-following-noun)
        best_span = right_span if right_span is not None else left_span
        best_dir = "right" if right_span is not None else "left"
        if best_span is None:
            continue
        ptag = tags_by_span.get(best_span)
        if ptag is None:
            continue
        # Widen the piece span to include this word and any link-word
        # gap between.
        if best_dir == "right":
            new_start = idx
            new_end = best_span[1]
            extra = list(range(idx, best_span[0]))
        else:
            new_start = best_span[0]
            new_end = idx + 1
            extra = list(range(best_span[1], idx + 1))
        new_span = (new_start, new_end)
        if new_span in tags_by_span and new_span != best_span:
            continue
        new_tag = Tag(
            span=new_span,
            words=clue_words[new_start:new_end],
            role="piece", value=ptag.value,
            mechanism=ptag.mechanism, sub_kind=ptag.sub_kind,
            notes=ptag.notes + [
                f"pos_widened to absorb '{clue_words[idx]}'"])
        del tags_by_span[best_span]
        tags_by_span[new_span] = new_tag
        used.update(extra)
        # Refresh _piece_spans_now since tags changed
        _piece_spans_now = sorted(
            [s for s, t in tags_by_span.items() if t.role == "piece"])

    # Indicator extension: a single-word indicator immediately
    # followed by an unaccounted content word (no piece/def between)
    # often forms a phrasal indicator together — "providing cover",
    # "tucking into", "fooling around". Absorb the adjacent unaccounted
    # word(s) into the indicator span up to the next tagged role.
    from .verifier import LINK_WORDS as _LW_IND
    ind_spans_now = sorted(
        [s for s, t in tags_by_span.items() if t.role == "indicator"])
    for ispan in ind_spans_now:
        itag = tags_by_span.get(ispan)
        if itag is None:
            continue
        # Don't extend a multi-word indicator further (already a phrase).
        if ispan[1] - ispan[0] > 1:
            continue
        # Absorb the next unaccounted CONTENT word (not link words).
        # Stop at the first link word, used index, or after 1 word
        # (be conservative — only adjective+noun pairs).
        next_idx = ispan[1]
        if next_idx >= len(clue_words) or next_idx in used:
            continue
        next_word = clue_words[next_idx]
        next_word_norm = (next_word.lower()
                          .strip(",.;:!?\"'()-‘’"))
        if next_word_norm in _LW_IND:
            continue
        # Skip if next word is uppercase letters (a piece, not a
        # phrasal continuation) — e.g. "turned MI" where MI is a piece.
        if next_word.isupper() and len(next_word) <= 4:
            continue
        new_span = (ispan[0], next_idx + 1)
        if new_span in tags_by_span and new_span != ispan:
            continue
        del tags_by_span[ispan]
        tags_by_span[new_span] = Tag(
            span=new_span,
            words=clue_words[new_span[0]:new_span[1]],
            role="indicator",
            operation=itag.operation,
            sub_kind=itag.sub_kind,
            notes=itag.notes + [
                f"extended to absorb '{next_word}'"])
        used.add(next_idx)
        # Queue extended phrase as shadow indicator candidate so
        # bridge.indicators in pass-2 can verify it.
        if itag.operation:
            shadow_candidates.append({
                "kind": "indicator",
                "source_word": " ".join(clue_words[new_span[0]:new_span[1]]),
                "value": " ".join(clue_words[new_span[0]:new_span[1]]),
                "operation": itag.operation,
                "subtype": itag.sub_kind,
                "evidence": "indicator extended via adjacent content word",
            })

    # Definition extension: if a definition tag exists, extend its
    # span to absorb adjacent unaccounted content words (through
    # link-word bridges). Handles cases where clue.definition has
    # only the keyword but the actual definition phrase is longer
    # (e.g. clue.definition='fail' for TURING TEST when the real
    # phrase is 'ChatGPT may fail this').
    def_spans_now = sorted(
        [s for s, t in tags_by_span.items() if t.role == "definition"])
    if def_spans_now:
        ds = def_spans_now[0]
        new_start, new_end = ds
        # Extend leftward through unaccounted words and link bridges
        i = new_start - 1
        absorbed_left = []
        while i >= 0 and i not in used:
            absorbed_left.append(i)
            i -= 1
        if absorbed_left:
            new_start = absorbed_left[-1]
        # Extend rightward
        i = new_end
        absorbed_right = []
        while i < len(clue_words) and i not in used:
            absorbed_right.append(i)
            i += 1
        if absorbed_right:
            new_end = absorbed_right[-1] + 1
        if (new_start, new_end) != ds:
            old_tag = tags_by_span[ds]
            del tags_by_span[ds]
            tags_by_span[(new_start, new_end)] = Tag(
                span=(new_start, new_end),
                words=clue_words[new_start:new_end],
                role="definition",
                notes=old_tag.notes + ["extended"])
            used.update(absorbed_left + absorbed_right)

    # Anything still unaccounted → tag as 'unaccounted'
    leftover = [i for i in range(len(clue_words)) if i not in used]
    if leftover:
        # Group consecutive
        runs = []
        cur = [leftover[0]]
        for i in leftover[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
        runs.append(cur)
        for run in runs:
            span = (run[0], run[-1] + 1)
            tags_by_span[span] = Tag(
                span=span, words=clue_words[span[0]:span[1]],
                role="unaccounted")

    sorted_tags = [tags_by_span[s] for s in sorted(tags_by_span.keys())]
    floating_unanchored = floating_remaining

    m = Mapping(
        clue_text=clue_text, answer=answer_clean,
        clue_words=clue_words, tags=sorted_tags,
        unmapped_pieces=[],
        floating_ops=floating_unanchored,
        notes=notes,
    )
    # Attach shadow_candidates as an extra attribute (not in dataclass)
    m.shadow_candidates = shadow_candidates  # type: ignore
    return m
