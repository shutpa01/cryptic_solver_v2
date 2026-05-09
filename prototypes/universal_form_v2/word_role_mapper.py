"""Map each clue word to a role using the TFTT blog as evidence.

Input  : clue_text, answer, blog_text
Output : a Mapping with a Tag per contiguous clue-word span.
         Each tag has role in {definition, piece, indicator, link,
         unaccounted}, plus a value/mechanism (for pieces) or
         operation (for indicators).

This module does NOT build a form. It only assigns roles. The
downstream assembly enumerator takes the tagged pieces and tries
compositions until one produces the answer.

Blog conventions handled:
  - `TOKEN (gloss)`           -> piece (synonym / abbreviation /
                                  literal — chosen by token shape)
  - `TOKEN{x} (gloss)`        -> piece with deletion sub-info
  - `bareword (gloss)`        -> indicator: bareword maps to an op,
                                  gloss identifies the clue word
  - `[indicator]` square-brackets -> indicator: the bracketed phrase
                                  IS the indicator clue word
  - `{x}TOKEN`, `TOKEN{x}`   -> deletion notation
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# --- Output types ---------------------------------------------------------

@dataclass
class Tag:
    span: tuple              # (start_idx, end_idx_exclusive) in clue words
    words: list              # the actual words the span covers
    role: str                # definition / piece / indicator / link / unaccounted
    value: Optional[str] = None      # piece value (UPPERCASE letters)
    mechanism: Optional[str] = None  # synonym / abbreviation / literal / etc.
    operation: Optional[str] = None  # container / anagram / reversal / etc.
    sub_kind: Optional[str] = None   # e.g. "tail" for deletion, "first" for parts
    notes: list = field(default_factory=list)

    def to_dict(self):
        d = {"span": list(self.span), "words": list(self.words),
             "role": self.role}
        if self.value is not None:
            d["value"] = self.value
        if self.mechanism:
            d["mechanism"] = self.mechanism
        if self.operation:
            d["operation"] = self.operation
        if self.sub_kind:
            d["sub_kind"] = self.sub_kind
        if self.notes:
            d["notes"] = list(self.notes)
        return d


@dataclass
class Mapping:
    clue_text: str
    answer: str
    clue_words: list         # tokenisation we used
    tags: list               # one Tag per span; spans cover the whole clue
    unmapped_pieces: list    # blog tokens we couldn't anchor to a clue word
    floating_ops: list = field(default_factory=list)
    """Operations the blog mentions but couldn't anchor to a clue word.
    Each entry is an (op, sub_kind) tuple. The enumerator uses these
    as a hint for what operations to attempt."""
    notes: list = field(default_factory=list)

    def to_dict(self):
        return {
            "clue_text": self.clue_text,
            "answer": self.answer,
            "clue_words": list(self.clue_words),
            "tags": [t.to_dict() for t in self.tags],
            "unmapped_pieces": list(self.unmapped_pieces),
            "floating_ops": [list(op) for op in self.floating_ops],
            "notes": list(self.notes),
        }


# --- Token patterns -------------------------------------------------------

# TFTT piece-token: one or more uppercase runs, optionally interleaved with
# {lowercase} braces. Captures e.g. BAN, RIGHT, DEARLY, REN{t}, {cha}RT{ed}.
_PIECE_TOKEN = (
    # period-separated abbreviations like R.E., E.G., V.E.D.
    r"[A-Z]+\.[A-Z]+(?:\.[A-Z]+)*\.?"
    r"|"
    # standard with optional curly-brace deletion (and bracket deletion)
    r"[A-Z]+(?:\{[a-z]+\}|\[[a-z]+\])?(?:[A-Z]+)?(?:\{[a-z]+\}|\[[a-z]+\])?(?:[A-Z]+)?"
)

# A bareword in the blog (lowercase, possibly with apostrophes/hyphens).
_BAREWORD = r"[a-z][a-z\-]*"

# Combined token pattern: TOKEN or bareword, with optional gloss (...)
# and optional [indicator] in brackets. Brackets and parens may also appear
# as standalone segments (no preceding token), in which case they're glosses
# for the immediately preceding token.
_TOKEN_WITH_GLOSS = re.compile(
    r"(?P<token>" + _PIECE_TOKEN + r"|" + _BAREWORD + r")"
    r"(?:\s*\((?P<gloss>[^()]+)\))?"
)


# Map of TFTT operator words to (operation, sub_kind)
_OP_WORDS = {
    # container family
    "in": ("container", None),
    "inside": ("container", None),
    "within": ("container", None),
    "containing": ("container", None),
    "contained": ("container", None),
    "contains": ("container", None),
    "around": ("container", None),
    "surrounding": ("container", None),
    "wrapped": ("container", None),
    "enveloping": ("container", None),
    "outside": ("container", None),
    # reversal family
    "reversed": ("reversal", None),
    "reversal": ("reversal", None),
    "back": ("reversal", None),
    "backwards": ("reversal", None),
    "up": ("reversal", None),
    "upwards": ("reversal", None),
    "returning": ("reversal", None),
    # anagram family
    "anagram": ("anagram", None),
    "scrambled": ("anagram", None),
    "mixed": ("anagram", None),
    "mixed-up": ("anagram", None),
    # hidden family
    "hidden": ("hidden", None),
    "concealed": ("hidden", None),
    "lurking": ("hidden", None),
    # homophone
    "homophone": ("homophone", None),
    "sounds": ("homophone", None),
    # charade connectives (not real ops, but useful)
    # (we DON'T tag '+', ',', 'next', 'then' as indicators — they
    # signal the implicit charade rather than an indicator word)
}


# --- Helpers --------------------------------------------------------------

def _tokenise_clue(clue_text: str) -> list:
    """Words from clue_text, lowercase-normalised but preserving original."""
    return re.findall(r"[A-Za-z][A-Za-z']*", clue_text)


def _norm(s: str) -> str:
    return (s or "").lower().strip(" ,.;:!?\"'()-")


def _norm_letters(s: str) -> str:
    return "".join(c for c in (s or "").upper() if c.isalpha())


def _expand_curly(token: str) -> tuple:
    """Resolve TFTT deletion notation in a piece token.

    Supports both {curly} and [bracket] forms (TFTT uses both).

    'REN{t}'        -> kept='REN', dropped='T', kind='tail'
    '{n}EUTER'      -> kept='EUTER', dropped='N', kind='head'
    '{cha}RT{ed}'   -> kept='RT', dropped='CHA+ED', kind='outer'
    'DIC{t}ATED'    -> kept='DICATED', dropped='T', kind='heart'
    'A[rchbisho]P'  -> kept='AP', dropped='RCHBISHO', kind='heart'
    """
    # Period-only abbreviations (R.E.) — strip dots, no deletion
    if '{' not in token and '[' not in token:
        kept = token.replace('.', '')
        return kept, "", None
    parts_kept = []
    parts_dropped = []
    pos = 0
    brace_positions = []
    while pos < len(token):
        if token[pos] in '{[':
            close_char = '}' if token[pos] == '{' else ']'
            close = token.index(close_char, pos)
            parts_dropped.append(token[pos + 1:close].upper())
            brace_positions.append((pos, close))
            pos = close + 1
        else:
            j = pos
            while j < len(token) and token[j] not in '{[':
                j += 1
            parts_kept.append(token[pos:j])
            pos = j
    kept = "".join(parts_kept).replace('.', '')
    dropped = "".join(parts_dropped)
    if not brace_positions:
        return kept, "", None
    n = len(brace_positions)
    if n >= 2:
        first_pos = brace_positions[0][0]
        if first_pos == 0 and \
                brace_positions[-1][1] == len(token) - 1:
            return kept, dropped, "outer"
    p0, c0 = brace_positions[0]
    if p0 == 0:
        return kept, dropped, "head"
    if c0 == len(token) - 1:
        return kept, dropped, "tail"
    return kept, dropped, "heart"


def _find_phrase_in_clue(phrase: str, clue_words: list,
                         already_used: set) -> Optional[tuple]:
    """Find a contiguous span in clue_words matching phrase, that doesn't
    overlap with already_used indices.

    Match is case-insensitive on letter content only.
    """
    target = [_norm(w) for w in re.findall(r"[A-Za-z']+", phrase)]
    if not target:
        return None
    n = len(target)
    norm_clue = [_norm(w) for w in clue_words]
    for i in range(len(clue_words) - n + 1):
        if any(j in already_used for j in range(i, i + n)):
            continue
        if norm_clue[i:i + n] == target:
            return (i, i + n)
    # Looser match: substring of each clue word
    for i in range(len(clue_words) - n + 1):
        if any(j in already_used for j in range(i, i + n)):
            continue
        if all(target[k] == norm_clue[i + k] or
               target[k] in norm_clue[i + k] or
               norm_clue[i + k] in target[k]
               for k in range(n)):
            return (i, i + n)
    return None


def _classify_piece_mechanism(token: str, gloss: str) -> str:
    """Pick a mechanism label for a piece based on token + gloss shape.

    Most cryptic abbreviations are 1-2 letters (R, S, AC, BC, etc.).
    3+ letters tend to be real synonyms, even short ones (BAN, USE, LO).
    Downstream can confirm with a DB lookup.
    """
    t_letters = _norm_letters(token)
    g_letters = _norm_letters(gloss or "")
    if g_letters and g_letters == t_letters:
        return "literal"  # gloss IS the token (rare)
    if len(t_letters) <= 2:
        return "abbreviation"
    return "synonym"


# --- Main mapper ----------------------------------------------------------

def map_clue_words(clue_text: str, answer: str,
                    blog_text: str) -> Mapping:
    """Map each clue word to a role using the blog as evidence."""
    clue_words = _tokenise_clue(clue_text)
    answer_clean = _norm_letters(answer)

    tags_by_span = {}     # (start, end) -> Tag
    used_indices = set()  # clue-word indices already accounted for
    unmapped = []
    notes = []

    blog = blog_text or ""

    # Walk the blog left to right, recognising pieces and operator-words.
    pos = 0
    last_token = None     # last bare/uppercase token (for trailing-gloss)
    last_was_uppercase = False
    last_op_word = None   # most recent op-word (for bracket inheritance)
    last_op_info = None   # (op, sub) for last_op_word
    pending_anagram = False  # True if previous char was '*' for *(FODDER)
    seen_ops = set()      # set of (op, sub) tuples seen as bare op-words
                            # — used as a fallback signal for the enumerator
    while pos < len(blog):
        # Asterisk-anagram prefix: *(FODDER) means anagram of FODDER
        if blog[pos] == '*':
            pending_anagram = True
            pos += 1
            continue
        # Skip whitespace and punctuation that aren't structural
        if blog[pos] in " \t\n,.;:!?":
            pos += 1
            continue

        # Standalone gloss in parens (e.g. "(admits)" after "around")
        # OR parenthetical drop notation (e.g. "(t)IGHTENED", "CHA(p)")
        if blog[pos] == '(' and last_token is not None:
            close = blog.find(')', pos)
            if close > 0:
                gloss = blog[pos + 1:close]
                # Disambiguation: short all-lowercase content immediately
                # after an UPPERCASE token = a deletion drop, not a gloss.
                if (last_was_uppercase and
                        re.fullmatch(r"[a-z]{1,5}", gloss.strip())):
                    # CHA(p) form: "p" was dropped from the end of CHAP.
                    # Extend the existing last-piece tag if any, else
                    # leave as a flag — the piece tag is already created.
                    # Mark the most recent piece tag with deletion sub.
                    # (For simplicity we reflect this by editing the
                    # most recent tag's sub_kind to 'tail' if it isn't
                    # already, and adjusting its value to include the
                    # dropped letters in pre-deletion form.)
                    if tags_by_span:
                        last_span = max(tags_by_span.keys(),
                                         key=lambda s: s[0])
                        t = tags_by_span[last_span]
                        if t.role == "piece" and t.sub_kind is None:
                            t.sub_kind = "tail"
                            t.notes.append(
                                f"parenthetical_drop:{gloss.strip().upper()}")
                    last_token = None
                    last_was_uppercase = False
                    pos = close + 1
                    continue
                if not last_was_uppercase:
                    # Operator-word + (gloss): map gloss to clue word
                    op_info = _OP_WORDS.get(last_token.lower())
                    if op_info:
                        op, sub = op_info
                        span = _find_phrase_in_clue(gloss, clue_words,
                                                      used_indices)
                        if span:
                            words = clue_words[span[0]:span[1]]
                            tags_by_span[span] = Tag(
                                span=span, words=words,
                                role="indicator", operation=op,
                                sub_kind=sub)
                            used_indices.update(range(span[0], span[1]))
                last_token = None
                last_was_uppercase = False
                pos = close + 1
                continue
            else:
                pos += 1
                continue

        # Bracketed indicator: [word] or [phrase]
        if blog[pos] == '[':
            close = blog.find(']', pos)
            if close > 0:
                inner = blog[pos + 1:close]
                # If the previous bareword was an op-word, inherit its
                # operation type for this bracket.
                inferred_op = None
                inferred_sub = None
                if last_op_info is not None:
                    inferred_op, inferred_sub = last_op_info
                span = _find_phrase_in_clue(inner, clue_words,
                                              used_indices)
                if span:
                    words = clue_words[span[0]:span[1]]
                    note = f"bracket-indicator '{inner}'"
                    if inferred_op:
                        note += f" inferred_op={inferred_op}"
                    tags_by_span[span] = Tag(
                        span=span, words=words, role="indicator",
                        operation=inferred_op, sub_kind=inferred_sub,
                        notes=[note])
                    used_indices.update(range(span[0], span[1]))
                pos = close + 1
                continue
            pos += 1
            continue

        # Try to match a TOKEN (piece or operator-word) at this position
        m = _TOKEN_WITH_GLOSS.match(blog, pos)
        if not m:
            pos += 1
            continue
        token = m.group("token")
        gloss = m.group("gloss")
        is_uppercase = token[:1].isupper() and any(c.isupper() for c in token)
        if is_uppercase:
            # Piece. If we have a pending '*' anagram-prefix, this fodder
            # IS an anagram. Use the token's letters as the anagram fodder
            # value and synthesise an indicator tag (since '*' has no
            # explicit clue word — caller will need to pick the right
            # clue-word indicator from context).
            if pending_anagram:
                pending_anagram = False
                # Tag this as anagram fodder. We don't yet have a single
                # clue word to anchor it to — use the token value.
                # Mark with mechanism='literal' (raw fodder).
                kept_only = re.sub(r"[^A-Z]", "", token.upper())
                # Add a synthetic anagram indicator — use empty op tag
                # so the enumerator's anagram trial can fire.
                # We don't have a clue word to anchor; use the gloss as
                # source if present.
                if gloss:
                    span = _find_phrase_in_clue(gloss, clue_words,
                                                  used_indices)
                    if span:
                        words = clue_words[span[0]:span[1]]
                        tags_by_span[span] = Tag(
                            span=span, words=words, role="piece",
                            value=kept_only, mechanism="literal",
                            notes=["asterisk_anagram_fodder"])
                        used_indices.update(range(span[0], span[1]))
                last_token = token
                last_was_uppercase = True
                # Force an anagram indicator tag if not already present.
                # Use a placeholder span at end of clue (not anchored).
                pos = m.end()
                continue
            # Piece. Tag the clue word(s) matching gloss (if any).
            kept, dropped, del_kind = _expand_curly(token)
            value = kept
            if gloss:
                span = _find_phrase_in_clue(gloss, clue_words,
                                              used_indices)
                if span:
                    words = clue_words[span[0]:span[1]]
                    mech = _classify_piece_mechanism(token, gloss)
                    sub = del_kind  # if curly braces present
                    tags_by_span[span] = Tag(
                        span=span, words=words, role="piece",
                        value=value, mechanism=mech, sub_kind=sub)
                    used_indices.update(range(span[0], span[1]))
                else:
                    unmapped.append({"token": token, "gloss": gloss,
                                      "reason": "gloss not in clue"})
            else:
                unmapped.append({"token": token, "gloss": None,
                                  "reason": "no gloss"})
            last_token = token
            last_was_uppercase = True
        else:
            # Bareword. Track op-words for bracket inheritance.
            op_info = _OP_WORDS.get(token.lower())
            if op_info:
                last_op_word = token
                last_op_info = op_info
                if gloss:
                    op, sub = op_info
                    span = _find_phrase_in_clue(gloss, clue_words,
                                                  used_indices)
                    if span:
                        words = clue_words[span[0]:span[1]]
                        tags_by_span[span] = Tag(
                            span=span, words=words, role="indicator",
                            operation=op, sub_kind=sub)
                        used_indices.update(range(span[0], span[1]))
                        # Once attached, clear so a later bracket
                        # doesn't double-attach
                        last_op_info = None
                else:
                    # Bare op-word with no anchor — record as a
                    # "floating" op so the enumerator can still try
                    # the operation. We add a Tag with an empty span
                    # (using a sentinel) and role='indicator'.
                    seen_ops.add(op_info)
            last_token = token
            last_was_uppercase = False
        pos = m.end()

    # Fill in the rest of the clue: contiguous unaccounted regions become
    # candidate definitions (the longest such region at start or end of the
    # clue is the most likely definition).
    untagged = [i for i in range(len(clue_words))
                if i not in used_indices]
    # Group consecutive untagged indices into spans
    runs = []
    if untagged:
        cur = [untagged[0]]
        for i in untagged[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
        runs.append(cur)

    # Heuristic: definition is at the start or end of the clue (cryptic
    # convention). Among the unaccounted runs, prefer one anchored at
    # word-index 0 or word-index n-1; among those, take the longer.
    # Other runs become 'unaccounted'.
    if runs:
        n = len(clue_words)
        edge_runs = [r for r in runs if r[0] == 0 or r[-1] == n - 1]
        if edge_runs:
            chosen = max(edge_runs, key=len)
        else:
            chosen = max(runs, key=len)
        for run in runs:
            span = (run[0], run[-1] + 1)
            words = clue_words[span[0]:span[1]]
            role = "definition" if run is chosen else "unaccounted"
            tags_by_span[span] = Tag(span=span, words=words, role=role)

    # Sort tags by span start
    sorted_tags = [tags_by_span[s] for s in sorted(tags_by_span.keys())]

    return Mapping(
        clue_text=clue_text, answer=answer_clean,
        clue_words=clue_words, tags=sorted_tags,
        unmapped_pieces=unmapped,
        floating_ops=sorted(seen_ops),
        notes=notes,
    )
