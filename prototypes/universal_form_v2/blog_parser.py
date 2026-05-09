"""Mechanical TFTT blog parser.

Reads a clue + answer + blog text and emits a candidate `Form`. Pure
regex/string-matching — no LLM, no production solver dependency.

Returns a ParseResult with one of three statuses:

  - "form": parser built a form and it mechanically produces the answer.
  - "form_unverified": parser built a form but assembly doesn't match
    the answer (extraction misread the blog or the form is incomplete).
  - "no_match": parser couldn't recognise the blog's structure at all.

Per the agreed plan, the parser does NOT touch live DBs. Each piece it
identifies (synonym candidate, indicator candidate, definition candidate)
is returned in `extraction` so a downstream step can write to the shadow
store.

Patterns supported in v0:

  - Anagram:        "anagram [IND] of FODDER" / "Anagram of FODDER"
  - Hidden:         "hidden" + (optionally) the spanning words
  - Reversal:       "reversal of PIECE" / "PIECE reversed [IND]"
  - Container:      "X in Y" with parenthesised glosses
  - Charade:        "X (gloss) + Y (gloss) [+ ...]" or "X + Y."
  - Double defn:    explicit "double definition"
  - TFTT deletion:  "PIE{c}E" curly-brace notation

Compound shapes (charade-of-deletion, container-of-reversal, etc.) work
when each constituent uses one of the recognised patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .schema import (
    Form, Node, Definition, lit, syn, abbr, raw,
    charade, anagram, reversal, container, deletion, hidden,
    double_definition, unknown,
)
from .renderer import wordplay_type


# --- Result types ---------------------------------------------------------

@dataclass
class ExtractionItem:
    """One piece of evidence to consider for the shadow DB."""
    kind: str          # 'definition' | 'synonym' | 'abbreviation' | 'indicator'
    source_word: str   # the clue word(s)
    value: str         # the contributed letters
    sub: str = ""      # subtype: e.g. 'anagram' / 'reversal' / 'first_letter'

    def to_dict(self):
        return {"kind": self.kind, "source_word": self.source_word,
                "value": self.value, "sub": self.sub}


@dataclass
class ParseResult:
    status: str            # form | form_unverified | no_match
    form: Optional[Form]
    pattern: str           # which top-level parser branch fired
    notes: list = field(default_factory=list)
    extraction: list = field(default_factory=list)  # ExtractionItems

    def to_dict(self):
        return {
            "status": self.status,
            "pattern": self.pattern,
            "wordplay_type": (wordplay_type(self.form)
                              if self.form else None),
            "form": self.form.to_dict() if self.form else None,
            "notes": list(self.notes),
            "extraction": [e.to_dict() for e in self.extraction],
        }


# --- Tokens / regexes ----------------------------------------------------

# A 'piece-letter-token' = an UPPERCASE letter run, possibly with TFTT
# curly-brace deletion markers (e.g. REN{t}, DIC{t}ATED, {cha}RT{ed}).
# Stays a single conceptual token even though it has lowercase chars.
_PIECE_TOKEN_RE = r"(?:[A-Z]+(?:\{[a-z]+\})?(?:[A-Z]+)?(?:\{[a-z]+\})?(?:[A-Z]+)?)"

# A 'gloss' = (...) directly after a piece token, optionally containing
# nested parentheses or commas. We match the most common shape — a single
# parenthesis level containing the gloss text.
_GLOSS_RE = r"\(([^()]*)\)"

# Indicator in square brackets
_IND_RE = r"\[([^\]]+)\]"

# Token + optional gloss + optional indicator
PIECE_RE = re.compile(
    r"(?P<token>" + _PIECE_TOKEN_RE + r")"
    r"(?:\s*\(" + r"(?P<gloss>[^()]*)" + r"\))?"
    r"(?:\s*\[" + r"(?P<ind>[^\]]+)" + r"\])?"
)


def _norm_letters(s: str) -> str:
    """Letters-only uppercase."""
    return "".join(c for c in (s or "").upper() if c.isalpha())


def _expand_curly(token: str) -> tuple[str, str, list]:
    """Resolve TFTT curly-brace notation in a piece token.

    Returns (kept_letters, dropped_letters, deletion_kind_hints).
    'REN{t}'    -> ('REN', 'T', ['tail'])
    '{cha}RT{ed}' -> ('RT', 'CHA+ED', ['head','tail'])
    'DIC{t}ATED' -> ('DICATED', 'T', ['heart'])  (best-effort)
    """
    kept_parts = []
    dropped_parts = []
    kinds = []
    pos = 0
    while pos < len(token):
        if token[pos] == '{':
            close = token.index('}', pos)
            dropped = token[pos + 1:close].upper()
            dropped_parts.append(dropped)
            # Heuristic for deletion kind:
            if pos == 0:
                kinds.append('head')
            elif close == len(token) - 1:
                kinds.append('tail')
            else:
                kinds.append('heart')
            pos = close + 1
        else:
            # Run of uppercase
            j = pos
            while j < len(token) and token[j] != '{':
                j += 1
            kept_parts.append(token[pos:j])
            pos = j
    kept = "".join(kept_parts)
    return kept, "".join(dropped_parts), kinds


def _build_piece_node(token: str, gloss: str, source_word: str,
                      extraction: list) -> Node:
    """Build a leaf (or small subtree) for one piece token + gloss.

    If the token has curly-brace deletion, wrap a literal in a deletion
    op. Otherwise the gloss decides the leaf op (synonym/abbreviation/raw).
    """
    has_curly = '{' in token
    if has_curly:
        kept, dropped, kinds = _expand_curly(token)
        kind = kinds[0] if kinds else 'tail'
        # Inner leaf: the un-trimmed source. Use the gloss source if
        # present; otherwise use the source_word.
        full_letters = kept + (dropped if kinds == ['head']
                               else dropped if kinds == ['tail']
                               else dropped)  # approximation only
        # Mark as synonym when there's a gloss; else literal.
        if gloss:
            inner = syn(source_word=source_word, value=full_letters)
            extraction.append(ExtractionItem(
                "synonym", source_word, full_letters))
        else:
            inner = lit(source_word=source_word, value=full_letters)
        return deletion(inner, kind=kind, indicator="")
    # No curly braces — pick leaf op from gloss content
    val = _norm_letters(token)
    if gloss is None or gloss.strip() == "":
        # Bare token — literal/raw
        return lit(source_word=source_word or token, value=val)
    g = gloss.strip()
    # If gloss is ALL CAPS itself, it might be 'redundant' (X (X)).
    # Otherwise treat as synonym.
    if _norm_letters(g) == val:
        # Tautology — same as raw
        return raw(source_word=source_word or g, value=val)
    # Heuristic: short upper letters (1-3) likely abbreviation
    if len(val) <= 3 and val.isalpha():
        extraction.append(ExtractionItem("abbreviation", g, val))
        return abbr(source_word=g, value=val)
    extraction.append(ExtractionItem("synonym", g, val))
    return syn(source_word=g, value=val)


# --- Top-level parser branches -------------------------------------------

# 1. Double definition --------------------------------------------------------

_DD_PATTERNS = [
    re.compile(r"\bdouble\s+def(?:initions?)?\b", re.IGNORECASE),
    re.compile(r"\bDD\b"),
]


def _try_double_definition(clue_text: str, answer: str, blog: str,
                            extraction: list) -> Optional[Form]:
    if not any(p.search(blog) for p in _DD_PATTERNS):
        return None
    # Split the clue text in two roughly equal halves and emit each as
    # a synonym leaf. We can't tell from "double definition" alone where
    # the split is. The verifier will check both halves are answers in
    # the DB.
    words = re.findall(r"[A-Za-z']+", clue_text)
    if len(words) < 2:
        return None
    mid = len(words) // 2
    left = " ".join(words[:mid])
    right = " ".join(words[mid:])
    left_node = syn(source_word=left, value=answer)
    right_node = syn(source_word=right, value=answer)
    extraction.append(ExtractionItem("definition", left, answer))
    extraction.append(ExtractionItem("definition", right, answer))
    return Form(
        tree=double_definition(left_node, right_node),
        definition=Definition(phrase=left, answer=answer),
        link_words=[],
    )


# 2. Anagram --------------------------------------------------------------

# Examples:
#   "Anagram [unusually] of TRIBAL YET"
#   "anagram of ENUMERATION"
#   "anagram of TENNIS GAMES minus E"
#   "Anagram [cast] of MADE PORN"

_ANAGRAM_RE = re.compile(
    r"\b[Aa]nagram\s+"
    r"(?:\[(?P<ind>[^\]]+)\]\s+)?"
    r"of\s+"
    r"(?P<fodder>[A-Z]+(?:\s+[A-Z]+)*)"
    r"(?:\s+minus\s+(?P<minus>[A-Z]+))?"
)


def _try_anagram(clue_text: str, answer: str, blog: str,
                  extraction: list) -> Optional[Form]:
    m = _ANAGRAM_RE.search(blog)
    if not m:
        return None
    ind = (m.group("ind") or "").strip().strip('"\'')
    fodder = m.group("fodder").strip()
    minus = (m.group("minus") or "").strip()
    fodder_words = fodder.split()
    # Each fodder word becomes a literal leaf
    fodder_leaves = [lit(source_word=w, value=_norm_letters(w))
                     for w in fodder_words]
    if minus:
        # Wrap fodder in a deletion of the minus letter(s)
        # Convention: the minus letters are removed from the combined fodder
        # before anagramming. Modelled as anagram(charade(fodder...)) with
        # a deletion of the minus letters somewhere — for v0 simplicity, we
        # subtract by removing them from one fodder leaf if found.
        for i, lf in enumerate(fodder_leaves):
            if minus in lf.value:
                # Remove first occurrence
                idx = lf.value.find(minus)
                lf.value = lf.value[:idx] + lf.value[idx + len(minus):]
                break
    # Definition is the clue minus fodder words minus indicator words
    used = set(_words(fodder)) | set(_words(ind))
    def_phrase = " ".join(w for w in re.findall(r"[A-Za-z']+", clue_text)
                          if w.lower() not in used).strip()
    if ind:
        extraction.append(ExtractionItem("indicator", ind, "", "anagram"))
    if def_phrase:
        extraction.append(ExtractionItem("definition", def_phrase, answer))
    return Form(
        tree=anagram(*fodder_leaves, indicator=ind or None),
        definition=Definition(phrase=def_phrase, answer=answer),
        link_words=[],
    )


# 3. Hidden ---------------------------------------------------------------

# TFTT often just says "hidden" terse. Sometimes "hidden in TEXT".
_HIDDEN_RE = re.compile(
    r"\bhidden\s*(?:reversed)?\s*"
    r"(?:in\s+(?P<text>[^.]+))?",
    re.IGNORECASE,
)


def _try_hidden(clue_text: str, answer: str, blog: str,
                 extraction: list) -> Optional[Form]:
    m = _HIDDEN_RE.search(blog)
    if not m:
        return None
    is_reversed = bool(re.search(r"hidden\s+reversed", blog, re.IGNORECASE))
    text = (m.group("text") or "").strip(' ".,;:')
    if not text:
        # Find the spanning words from the clue itself
        clue_letters = "".join(c for c in clue_text.upper() if c.isalpha())
        target = answer if not is_reversed else answer[::-1]
        if target not in clue_letters:
            return None
        # Find the words spanning the answer
        text = clue_text  # fallback: whole clue
    span_words = re.findall(r"[A-Za-z']+", text)
    if not span_words:
        return None
    leaves = [lit(source_word=w, value=_norm_letters(w))
              for w in span_words]
    inner = hidden(*leaves, indicator=None)
    if is_reversed:
        tree = reversal(inner, indicator=None)
    else:
        tree = inner
    # Definition is the rest of the clue
    used = {w.lower() for w in span_words}
    def_phrase = " ".join(w for w in re.findall(r"[A-Za-z']+", clue_text)
                          if w.lower() not in used).strip()
    if def_phrase:
        extraction.append(ExtractionItem("definition", def_phrase, answer))
    return Form(
        tree=tree,
        definition=Definition(phrase=def_phrase, answer=answer),
        link_words=[],
    )


# 4. Container ------------------------------------------------------------

# Patterns:
#   X (gloss) in Y (gloss)
#   X (gloss) in (containing) Y (gloss)
#   "X in Y" without glosses

_CONTAINER_RE = re.compile(
    r"(?P<inner>" + _PIECE_TOKEN_RE + r")(?:\s*\((?P<inner_gloss>[^)]+)\))?"
    r"\s+(?:in|inside|within|contained\s+by|containing|surrounded\s+by)\s+"
    r"(?:\((?:[^)]*)\)\s*)?"  # optional indicator gloss like "(crossed by)"
    r"(?P<outer>" + _PIECE_TOKEN_RE + r")(?:\s*\((?P<outer_gloss>[^)]+)\))?"
)


def _try_container(clue_text: str, answer: str, blog: str,
                    extraction: list) -> Optional[Form]:
    m = _CONTAINER_RE.search(blog)
    if not m:
        return None
    inner_token = m.group("inner")
    outer_token = m.group("outer")
    inner_gloss = m.group("inner_gloss")
    outer_gloss = m.group("outer_gloss")
    inner_node = _build_piece_node(inner_token, inner_gloss,
                                    inner_gloss or "", extraction)
    outer_node = _build_piece_node(outer_token, outer_gloss,
                                    outer_gloss or "", extraction)
    # Definition: rough heuristic — clue words not glossed
    used = set()
    for g in (inner_gloss, outer_gloss):
        if g:
            used.update(_words(g))
    def_phrase = " ".join(w for w in re.findall(r"[A-Za-z']+", clue_text)
                          if w.lower() not in used).strip()
    if def_phrase:
        extraction.append(ExtractionItem("definition", def_phrase, answer))
    return Form(
        tree=container(outer=outer_node, inner=inner_node, indicator=None),
        definition=Definition(phrase=def_phrase, answer=answer),
        link_words=[],
    )


# 5. Reversal -------------------------------------------------------------

_REVERSAL_PREFIX_RE = re.compile(
    r"\breversal\s+(?:\([^)]*\)\s+)?of\s+(?P<piece>[A-Z]+(?:\s+[A-Z]+)*)",
    re.IGNORECASE,
)
# "X reversed" or "X reversed [indicator]" — postfix form
_REVERSAL_POSTFIX_RE = re.compile(
    r"(?<![A-Za-z])(?P<piece>[A-Z]+(?:\s+[A-Z]+)*)\s+"
    r"reversed(?:\s+\[(?P<ind>[^\]]+)\])?"
)


def _try_reversal(clue_text: str, answer: str, blog: str,
                   extraction: list) -> Optional[Form]:
    m = _REVERSAL_PREFIX_RE.search(blog)
    if m:
        pieces = m.group("piece").strip().split()
    else:
        m = _REVERSAL_POSTFIX_RE.search(blog)
        if not m:
            return None
        pieces = m.group("piece").strip().split()
    leaves = [lit(source_word=p, value=_norm_letters(p)) for p in pieces]
    if len(leaves) == 1:
        inner = leaves[0]
    else:
        inner = charade(*leaves)
    return Form(
        tree=reversal(inner, indicator=None),
        definition=Definition(phrase="", answer=answer),
        link_words=[],
    )


# 5b. Acrostic ------------------------------------------------------------

# Examples:
#   "The first letters of Croquet and Often"
#   "first letters of A and B and C"
#   "initial letters of X and Y"

_ACROSTIC_RE = re.compile(
    r"(?:[Tt]he\s+)?(?:first|initial)\s+letters?\s+of\s+"
    r"(?P<words>[A-Z][a-zA-Z]+(?:\s+(?:and\s+)?[A-Z][a-zA-Z]+)+)",
)


def _try_acrostic(clue_text: str, answer: str, blog: str,
                   extraction: list) -> Optional[Form]:
    m = _ACROSTIC_RE.search(blog)
    if not m:
        return None
    raw_words = re.findall(r"[A-Z][a-zA-Z]+", m.group("words"))
    # Filter out 'and'
    src_words = [w for w in raw_words if w.lower() != "and"]
    if not src_words:
        return None
    leaves = [lit(source_word=w, value=_norm_letters(w))
              for w in src_words]
    from .schema import acrostic
    tree = acrostic(*leaves, indicator=None, kind="first")
    return Form(
        tree=tree,
        definition=Definition(phrase="", answer=answer),
        link_words=[],
    )


# 5c. Homophone -----------------------------------------------------------

# Examples:
#   "sounds like our answer"
#   "homophone of X"
#   "X sounds like Y"

_HOMOPHONE_RE = re.compile(
    r"\b(?:sounds\s+like|homophone\s+of|homophone)\b",
    re.IGNORECASE,
)


def _try_homophone(clue_text: str, answer: str, blog: str,
                    extraction: list) -> Optional[Form]:
    if not _HOMOPHONE_RE.search(blog):
        return None
    # Without specific source-word identification we can't build a tight
    # form. Emit a homophone op with the answer as the value.
    from .schema import homophone_leaf, Node
    leaf = homophone_leaf(source_word=clue_text, value=answer)
    tree = Node(operation="homophone", indicator=None,
                sources=[leaf])
    return Form(
        tree=tree,
        definition=Definition(phrase="", answer=answer),
        link_words=[],
    )


# 6. Charade --------------------------------------------------------------

# Sequence of piece tokens with optional glosses, joined by + or commas.
# Examples:
#   "LEG IT (run away)."  -> single piece (terse charade)
#   "LO (look), USE (take advantage of)"
#   "DISC (record), ONCE (previously),  {cha}RT{ed} [at fourth and fifth positions only]"
#   "AB (sailor), DIC{t}ATED (read aloud) [not enjoying the first time]"
#   "UPPER + CASE."
#   "PAST + A."

_CHARADE_PIECES_RE = re.compile(
    r"(?P<token>" + _PIECE_TOKEN_RE + r")"
    r"(?:\s*\((?P<gloss>[^)]*)\))?"
    r"(?:\s*\[(?P<ind>[^\]]+)\])?"
)


def _try_charade(clue_text: str, answer: str, blog: str,
                  extraction: list) -> Optional[Form]:
    # Find all piece+gloss tokens in order
    pieces = []
    for m in _CHARADE_PIECES_RE.finditer(blog):
        pieces.append((m.group("token"), m.group("gloss"), m.group("ind")))
    if not pieces:
        return None
    # Reject if zero glosses and the tokens don't concat to the answer —
    # likely random caps from the blog text (proper nouns etc.).
    answer_clean = _norm_letters(answer)
    nodes = []
    for tok, gl, ind in pieces:
        node = _build_piece_node(tok, gl, gl or "", extraction)
        if ind:
            # Indicator on a single piece — typically the deletion
            # explanation (e.g. "[at fourth and fifth positions only]").
            # Attach to the deletion sub-op if curly braces are present.
            if isinstance(node, Node) and node.operation == "deletion":
                node.indicator = ind
                extraction.append(ExtractionItem(
                    "indicator", ind, "", "deletion"))
        nodes.append(node)
    # Heuristic check: the concatenated values produce the answer
    concat = "".join(_letters_yielded(n) for n in nodes)
    if concat != answer_clean:
        # Try permutations — sometimes the blog lists pieces out of order
        from itertools import permutations
        if 1 < len(nodes) <= 5:
            for perm in permutations(nodes):
                if "".join(_letters_yielded(n) for n in perm) == answer_clean:
                    nodes = list(perm)
                    break
            else:
                return None
        else:
            return None
    # Definition heuristic: clue words not in any gloss
    used = set()
    for tok, gl, ind in pieces:
        if gl:
            used.update(_words(gl))
        if ind:
            used.update(_words(ind))
    def_phrase = " ".join(w for w in re.findall(r"[A-Za-z']+", clue_text)
                          if w.lower() not in used).strip()
    if def_phrase:
        extraction.append(ExtractionItem("definition", def_phrase, answer))
    if len(nodes) == 1:
        tree = nodes[0]
    else:
        tree = charade(*nodes)
    return Form(
        tree=tree,
        definition=Definition(phrase=def_phrase, answer=answer),
        link_words=[],
    )


def _letters_yielded(node: Node) -> str:
    """Conservative letter-yield walker for charade verification.

    Handles literal/syn/abbr/raw leaves and deletion sub-ops only.
    Anything else returns the leaf's value as-is.
    """
    if node.operation == "deletion":
        if not node.sources:
            return ""
        kid = node.sources[0]
        kid_letters = (kid.value or "")
        kind = node.deletion_kind or "tail"
        if kind == "tail":
            return kid_letters[:-1]
        if kind == "head":
            return kid_letters[1:]
        if kind == "outer" and len(kid_letters) >= 3:
            return kid_letters[1:-1]
        if kind == "heart" and len(kid_letters) >= 3:
            mid = len(kid_letters) // 2
            if len(kid_letters) % 2 == 1:
                return kid_letters[:mid] + kid_letters[mid + 1:]
            return kid_letters[:mid - 1] + kid_letters[mid + 1:]
        return kid_letters
    return node.value or ""


# --- Verification --------------------------------------------------------

def _form_produces(form: Form) -> bool:
    """Walk the tree, return True iff it mechanically produces the answer.

    Mirrors the verifier's _produces, but kept local to avoid an import
    cycle and reduce dependency surface for the parser.
    """
    answer = _norm_letters(form.definition.answer)
    return _walk(form.tree, answer)


def _walk(node: Node, target: str) -> bool:
    op = node.operation
    if op in ("literal", "synonym", "abbreviation", "raw",
              "positional", "homophone"):
        return (node.value or "").upper() == target
    if op == "charade":
        # split target into N pieces matching children
        return _split_charade(node.sources, target)
    if op == "anagram":
        letters = "".join(_yield(c) for c in node.sources)
        return sorted(letters) == sorted(target)
    if op == "reversal":
        return _walk(node.sources[0], target[::-1])
    if op == "container":
        if len(node.sources) != 2:
            return False
        outer, inner = node.sources
        outer_v = _yield(outer)
        inner_v = _yield(inner)
        if not outer_v or not inner_v:
            return False
        if len(outer_v) + len(inner_v) != len(target):
            return False
        for p in range(1, len(outer_v)):
            if outer_v[:p] + inner_v + outer_v[p:] == target:
                return True
        return False
    if op == "deletion":
        if not node.sources:
            return False
        src = _yield(node.sources[0])
        kind = node.deletion_kind or "tail"
        if kind == "tail":
            return src[:-1] == target
        if kind == "head":
            return src[1:] == target
        if kind == "outer" and len(src) >= 3:
            return src[1:-1] == target
        if kind == "heart" and len(src) >= 3:
            mid = len(src) // 2
            if len(src) % 2 == 1:
                return src[:mid] + src[mid + 1:] == target
            return src[:mid - 1] + src[mid + 1:] == target
        return False
    if op == "hidden":
        text = "".join(_yield(c) for c in node.sources)
        return target in text
    if op == "double_definition":
        return any(_walk(c, target) for c in node.sources)
    return False


def _yield(node: Node) -> str:
    """Best-effort single letter-yield."""
    if node.value:
        return node.value.upper()
    if node.operation == "charade":
        return "".join(_yield(c) for c in node.sources)
    if node.operation == "reversal" and node.sources:
        return _yield(node.sources[0])[::-1]
    return ""


def _split_charade(children, target):
    """Recursive: every child gets a contiguous slice of target."""
    if not children:
        return target == ""
    if len(children) == 1:
        return _walk(children[0], target)
    first, *rest = children
    fv = _yield(first)
    if not fv:
        return False
    if not target.startswith(fv):
        return False
    return _split_charade(rest, target[len(fv):])


def _words(s: str) -> list:
    return [w.lower() for w in re.findall(r"[A-Za-z']+", s or "")]


# --- Top-level parse ------------------------------------------------------

# Order matters: more specific first.
_PARSER_BRANCHES = [
    ("double_definition", _try_double_definition),
    ("anagram", _try_anagram),
    ("hidden", _try_hidden),
    ("acrostic", _try_acrostic),
    ("reversal", _try_reversal),
    ("container", _try_container),
    ("homophone", _try_homophone),
    ("charade", _try_charade),
]


def parse_blog(clue_text: str, answer: str,
                blog_text: str) -> ParseResult:
    """Try each parser branch in order. Return the first form that
    mechanically produces the answer; otherwise the first form that
    parsed but didn't verify; otherwise no_match.
    """
    answer_clean = _norm_letters(answer)
    if not (blog_text and blog_text.strip()):
        return ParseResult(status="no_match", form=None,
                            pattern="", notes=["blog text empty"])
    best_unverified = None
    for name, branch in _PARSER_BRANCHES:
        extraction = []
        try:
            form = branch(clue_text, answer_clean, blog_text, extraction)
        except Exception as e:
            continue
        if form is None:
            continue
        if _form_produces(form):
            return ParseResult(
                status="form", form=form, pattern=name,
                notes=[], extraction=extraction)
        if best_unverified is None:
            best_unverified = ParseResult(
                status="form_unverified", form=form, pattern=name,
                notes=[f"{name} parsed but assembly didn't match answer"],
                extraction=extraction)
    if best_unverified is not None:
        return best_unverified
    return ParseResult(status="no_match", form=None, pattern="",
                        notes=["no parser branch matched"])
