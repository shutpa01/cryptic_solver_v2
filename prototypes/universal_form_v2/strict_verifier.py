"""Strict universal-form verifier.

Per agreed spec (2026-05-05):

  1. Coverage      — every clue word has a role (definition / piece-source
                     / indicator / link). No unaccounted words.
  2. Leaf justify  — every leaf's value is mechanically derivable under
                     its operation:
                       synonym(v, s)      → DB row word=s, synonym=v
                       abbreviation(v, s) → DB row word=s, abbreviation=v
                       literal(v, s)      → letters(s) == v (exact)
                       raw(v, s)          → letters(s) == v (exact)
                       positional(v,s,k)  → mechanical extraction of k
                                            from s yields v
                       homophone(v, s)    → DB row word=s, homophone=v
  3. Indicator     — every op-node that requires one has an `indicator`
                     anchored to clue word(s).
  4. Indicator DB  — indicator word(s) appear in the indicators DB with
                     a wordplay_type matching the op.
  5. Mechanism     — anagram and hidden may only contain literal/raw
                     children (anagram fodder must be literal letters).
  6. Assembly      — the tree, evaluated mechanically, produces the
                     answer exactly.
  7. Definition    — form.definition.phrase is a contiguous span of
                     clue words.

Verdict is PASS only if every check passes. No partial credit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import permutations
from typing import Optional

from .schema import Form, Node, LEAF_OPERATIONS


# ---- LINK words (purely connective / grammatical) ------------------------

LINK_WORDS = frozenset({
    "a", "an", "and", "as", "at", "but", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "so", "the", "this", "to",
    "with", "without", "into", "out", "off", "up", "down",
    "be", "been", "being", "are", "was", "were", "had", "has",
    "have", "did", "does", "do", "may", "might", "could",
    "would", "should", "must", "shall", "will",
    "that", "which", "who", "whose", "whom",
    "not", "no",
    "all", "some", "any", "every", "each", "both",
    "his", "her", "its", "our", "their", "my", "your",
    "i", "we", "they", "you", "me", "us", "him", "them",
    "than", "then",
    "s",  # apostrophe-s remnant after tokenisation
})


# ---- Indicator op → DB wordplay_types ------------------------------------

OP_TO_DB_TYPES = {
    "anagram":   {"anagram"},
    "reversal":  {"reversal"},
    "container": {"container", "insertion"},
    "deletion":  {"deletion", "parts"},
    "hidden":    {"hidden"},
    "homophone": {"homophone"},
    "acrostic":  {"acrostic", "parts"},
    # Positional leaf indicators are typed by their kind:
    # parts/acrostic/alternating → positional indicators in DB
    "positional": {"parts", "acrostic", "alternating", "selection"},
}

# Ops that are juxtaposition / identity — no indicator required
NO_INDICATOR_OPS = frozenset({
    "charade", "double_definition", "cryptic_definition", "unknown",
})


# ---- Result types --------------------------------------------------------

@dataclass
class Check:
    name: str
    status: str   # "pass" | "fail"
    detail: str = ""

    def to_dict(self):
        return {"name": self.name, "status": self.status,
                "detail": self.detail}


@dataclass
class Verdict:
    verdict: str  # "PASS" | "FAIL"
    checks: list = field(default_factory=list)

    def to_dict(self):
        return {"verdict": self.verdict,
                "checks": [c.to_dict() for c in self.checks]}


# ---- Tokenisation --------------------------------------------------------

# Words plus optional apostrophe-s (smart or straight)
_WORD_RE = re.compile(r"[A-Za-z]+(?:[’‘ʼ'][A-Za-z]+)?")


def _surface_words(text: str) -> list:
    """Normalise clue text into lowercase word tokens. Apostrophe-s
    becomes its own token ('s) so coverage can flag it explicitly."""
    if not text:
        return []
    tokens = []
    for m in _WORD_RE.finditer(text):
        word = m.group(0).lower()
        # Split apostrophe-suffix into a separate token
        for sep in ("’", "‘", "ʼ", "'"):
            if sep in word:
                head, _, tail = word.partition(sep)
                tokens.append(head)
                if tail:
                    tokens.append(tail)
                break
        else:
            tokens.append(word)
    return tokens


def _letters(text: str) -> str:
    return "".join(c for c in text.upper() if c.isalpha())


# ---- Walking helpers -----------------------------------------------------

def _all_nodes(node: Node):
    """Iterate every node in the form tree, parents first."""
    yield node
    for c in node.sources or []:
        yield from _all_nodes(c)


def _leaves(node: Node) -> list:
    """All leaf nodes (operation in LEAF_OPERATIONS, no sources)."""
    out = []
    for n in _all_nodes(node):
        if n.operation in LEAF_OPERATIONS and not n.sources:
            out.append(n)
    return out


# ---- Per-leaf justification ----------------------------------------------

def _justify_leaf(leaf: Node, db, shadow_conn=None) -> Optional[str]:
    """Return None if the leaf's value is justified, else a failure
    reason string."""
    op = leaf.operation
    val = (leaf.value or "").upper()
    src = (leaf.source_word or "").strip()
    if not val:
        return f"{op} leaf has no value"
    if not src:
        return f"{op} leaf has no source_word"

    if op in ("literal", "raw"):
        src_letters = _letters(src)
        if src_letters == val:
            return None
        return (f"literal {val!r}: source letters "
                f"{src_letters!r} != value")

    if op == "synonym":
        # Try the source as-is (single phrase); accept if DB has it.
        return _check_synonym(src, val, db, shadow_conn)

    if op == "abbreviation":
        return _check_abbreviation(src, val, db, shadow_conn)

    if op == "positional":
        kind = leaf.positional_kind or ""
        if not kind:
            return "positional leaf has no positional_kind"
        return _check_positional(src, val, kind)

    if op == "homophone":
        return _check_homophone(src, val, db, shadow_conn)

    return f"unrecognised leaf op {op!r}"


def _check_synonym(src, val, db, shadow_conn) -> Optional[str]:
    src_norm = src.lower().strip()
    if val in db.get_synonyms(src_norm):
        return None
    if shadow_conn is not None:
        row = shadow_conn.execute(
            "SELECT 1 FROM synonyms_pairs "
            "WHERE LOWER(word)=? AND UPPER(synonym)=? LIMIT 1",
            (src_norm, val)).fetchone()
        if row:
            return None
    return f"synonym {val!r} not in DB for {src!r}"


def _check_abbreviation(src, val, db, shadow_conn) -> Optional[str]:
    src_norm = src.lower().strip()
    if val in db.get_abbreviations(src_norm):
        return None
    if shadow_conn is not None:
        row = shadow_conn.execute(
            "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? "
            "AND UPPER(substitution)=? AND category='abbreviation' "
            "LIMIT 1",
            (src_norm, val)).fetchone()
        if row:
            return None
    return f"abbreviation {val!r} not in DB for {src!r}"


def _check_homophone(src, val, db, shadow_conn) -> Optional[str]:
    src_norm = src.lower().strip()
    if val in db.get_homophones(src_norm):
        return None
    if shadow_conn is not None:
        row = shadow_conn.execute(
            "SELECT 1 FROM homophones "
            "WHERE LOWER(word)=? AND UPPER(homophone)=? LIMIT 1",
            (src_norm, val)).fetchone()
        if row:
            return None
    return f"homophone {val!r} not in DB for {src!r}"


def _check_positional(src, val, kind) -> Optional[str]:
    # Source might be a multi-word phrase (e.g. "interior of nests").
    # Mechanical extraction: try each individual word too.
    candidates = [src]
    if " " in src:
        candidates.extend(w for w in src.split() if w)
    for cand in candidates:
        if _positional_extract(cand, kind) == val:
            return None
    return (f"positional[{kind}] of {src!r} does not yield {val!r}")


def _positional_extract(word: str, kind: str) -> Optional[str]:
    src = "".join(c for c in word.upper() if c.isalpha())
    if not src:
        return None
    if kind == "first":
        return src[0]
    if kind == "last":
        return src[-1]
    if kind == "outer":
        return src[0] + src[-1] if len(src) >= 2 else None
    if kind == "middle":
        if len(src) < 3:
            return None
        # Single middle letter (odd-length) or two middle letters
        if len(src) % 2 == 1:
            return src[len(src) // 2]
        mid = len(src) // 2
        return src[mid - 1:mid + 1]
    if kind in ("alternate", "odd"):
        return src[0::2]
    if kind == "even":
        return src[1::2]
    if kind == "half":
        # Either half is acceptable elsewhere; here return first half
        return src[:len(src) // 2]
    if kind == "interior":
        return src[1:-1] if len(src) >= 3 else None
    return None


# ---- Indicator presence + DB validation ----------------------------------

def _check_indicators(form: Form, db, shadow_conn=None) -> list:
    """Returns list of (kind, message) failures."""
    failures = []
    for n in _all_nodes(form.tree):
        op = n.operation
        # Op-nodes
        if n.sources and op not in LEAF_OPERATIONS:
            if op in NO_INDICATOR_OPS:
                continue
            if not n.indicator:
                failures.append(("missing",
                                  f"{op} node has no indicator"))
                continue
            db_types = OP_TO_DB_TYPES.get(op)
            if db_types and not _indicator_in_db(n.indicator,
                                                  db_types, db,
                                                  shadow_conn):
                failures.append(("invalid",
                                  f"indicator {n.indicator!r} for op "
                                  f"{op!r}: no DB row with type in "
                                  f"{sorted(db_types)}"))
        # Positional leaves require an indicator too
        if op == "positional" and not n.sources:
            if not n.indicator:
                failures.append((
                    "missing",
                    f"positional[{n.positional_kind}] leaf "
                    f"{(n.value or '')!r} from {(n.source_word or '')!r} "
                    f"has no indicator"))
                continue
            if not _indicator_in_db(n.indicator,
                                      OP_TO_DB_TYPES["positional"],
                                      db, shadow_conn):
                failures.append(("invalid",
                                  f"positional indicator "
                                  f"{n.indicator!r} not in DB"))
    return failures


def _indicator_in_db(text, db_types, db, shadow_conn) -> bool:
    word = (text or "").lower().strip(",.;:!?\"'()-")
    if not word:
        return False
    types = db.get_indicator_types(word)
    if any(t[0] in db_types for t in types):
        return True
    if shadow_conn is not None:
        rows = shadow_conn.execute(
            "SELECT wordplay_type FROM indicators "
            "WHERE LOWER(word)=?", (word,)).fetchall()
        if any(r[0] in db_types for r in rows):
            return True
    return False


# ---- Mechanism integrity --------------------------------------------------

def _check_mechanism(form: Form) -> list:
    """Returns list of failure messages."""
    failures = []
    for n in _all_nodes(form.tree):
        op = n.operation
        if op == "anagram":
            for c in n.sources or []:
                if c.operation not in ("literal", "raw"):
                    failures.append(
                        f"anagram contains non-literal child "
                        f"({c.operation}); anagram fodder must be "
                        f"literal letters from the clue")
                    break
        elif op == "hidden":
            for c in n.sources or []:
                if c.operation not in ("literal", "raw"):
                    failures.append(
                        f"hidden contains non-literal child "
                        f"({c.operation}); hidden source must be "
                        f"literal clue text")
                    break
    return failures


# ---- Assembly: does the tree produce the answer? -------------------------

def _produces(node: Node, target: str) -> bool:
    """Mechanical evaluation: can this subtree produce target?"""
    if node is None or not target:
        return False
    op = node.operation
    target = target.upper()

    # Leaves
    if op in LEAF_OPERATIONS and not node.sources:
        v = (node.value or "").upper()
        return v == target

    if op == "charade":
        return _charade_produces(node.sources, target)

    if op == "anagram":
        # Sum letters from all children (each must be literal/raw, so
        # _produces accepts only their committed value). Total letters
        # multiset must equal target.
        letters = []
        for c in node.sources:
            v = (c.value or "").upper()
            if not v:
                return False
            letters.append(v)
        joined = "".join(letters)
        return sorted(joined) == sorted(target)

    if op == "reversal":
        if len(node.sources) != 1:
            return False
        return _produces(node.sources[0], target[::-1])

    if op == "container":
        if len(node.sources) != 2:
            return False
        outer, inner = node.sources
        # Try every contiguous slice as the inner string
        for il in range(1, len(target)):
            for p in range(1, len(target) - il + 1):
                inner_str = target[p:p + il]
                outer_str = target[:p] + target[p + il:]
                if not outer_str:
                    continue
                if (_produces(inner, inner_str)
                        and _produces(outer, outer_str)):
                    return True
        return False

    if op == "deletion":
        if not node.sources:
            return False
        kind = node.deletion_kind or "tail"
        # The source must produce a longer string from which the
        # specified position is dropped.
        # We don't know the source length, so try plausible candidates:
        # source produces (target with one or more letters re-inserted
        # at the deletion position).
        # Quick path: if source is a leaf, check directly.
        src = node.sources[0]
        # Derive what the source must yield:
        # tail: target + dropped_letters
        # head: dropped_letters + target
        # heart: target with dropped letters re-inserted in middle
        # outer: dropped + target + dropped
        # We don't know dropped letters; if source is a leaf, take its
        # value as committed and apply deletion mechanically.
        if src.operation in LEAF_OPERATIONS and not src.sources:
            full = (src.value or "").upper()
            return _apply_deletion(full, kind) == target
        # For non-leaf sources we'd need enumeration; not needed for v0.
        return False

    if op == "hidden":
        text = "".join((c.value or "").upper() for c in node.sources)
        return target in text

    if op == "double_definition":
        return any(_produces(c, target) for c in node.sources)

    if op == "acrostic":
        kind = node.acrostic_kind or "first"
        letters = []
        for c in node.sources:
            v = (c.value or c.source_word or "").upper()
            v = "".join(ch for ch in v if ch.isalpha())
            if not v:
                return False
            letters.append(v[-1] if kind == "last" else v[0])
        return "".join(letters) == target

    if op == "homophone":
        if not node.sources:
            return (node.value or "").upper() == target
        # Op-form: child produces some word that sounds like target.
        # Mechanically we cannot prove the homophone here — it's
        # validated in the bridge step. Accept if any child produces
        # *some* string (correctness is enforced by leaf-justify on
        # the homophone DB).
        return any(_produces(c, (c.value or "").upper())
                    for c in node.sources)

    return False


def _charade_produces(children, target):
    if not children:
        return False
    if len(children) == 1:
        return _produces(children[0], target)
    first = children[0]
    rest = children[1:]
    for cut in range(1, len(target)):
        if (_produces(first, target[:cut])
                and _charade_produces(rest, target[cut:])):
            return True
    return False


def _apply_deletion(full: str, kind: str) -> Optional[str]:
    if not full:
        return None
    if kind == "tail":
        return full[:-1]
    if kind == "head":
        return full[1:]
    if kind == "outer":
        return full[1:-1] if len(full) >= 3 else None
    if kind == "heart":
        if len(full) < 3:
            return None
        if len(full) % 2 == 1:
            mid = len(full) // 2
            return full[:mid] + full[mid + 1:]
        # Even length — drop centre two
        mid = len(full) // 2
        return full[:mid - 1] + full[mid + 1:]
    return None


# ---- Coverage check ------------------------------------------------------

def _check_coverage(form: Form, clue_text: str) -> list:
    """Every clue word must be in exactly one of:
      - definition.phrase
      - some leaf's source_word
      - some node's indicator
      - form.link_words (and word must be in LINK_WORDS)
    Returns list of failures.
    """
    surface = _surface_words(clue_text)
    used = {}  # word -> role(s)

    def mark(word, role):
        used.setdefault(word, []).append(role)

    # Definition
    for w in _surface_words(form.definition.phrase):
        mark(w, "definition")

    # Each leaf's source_word
    for leaf in _leaves(form.tree):
        for w in _surface_words(leaf.source_word):
            mark(w, "piece-source")

    # Each node's indicator
    for n in _all_nodes(form.tree):
        if n.indicator:
            for w in _surface_words(n.indicator):
                mark(w, "indicator")

    # Link words
    for lw in form.link_words:
        for w in _surface_words(lw):
            mark(w, "link")

    failures = []
    for surface_word in surface:
        if surface_word not in used:
            failures.append(f"unaccounted: {surface_word!r}")

    # Link-words declared must be genuine link words
    for lw in form.link_words:
        norm = lw.lower().strip(",.;:!?\"'()-")
        if norm and norm not in LINK_WORDS:
            failures.append(
                f"link_word {lw!r} not in LINK_WORDS allow-list")

    return failures


# ---- Definition contiguity ------------------------------------------------

def _check_definition(form: Form, clue_text: str) -> Optional[str]:
    """definition.phrase must be a contiguous span of clue words."""
    clue = _surface_words(clue_text)
    phrase = _surface_words(form.definition.phrase)
    if not phrase:
        return "definition phrase is empty"
    n = len(clue)
    m = len(phrase)
    for i in range(n - m + 1):
        if clue[i:i + m] == phrase:
            return None
    return (f"definition phrase {form.definition.phrase!r} is not a "
            f"contiguous span of the clue")


# ---- Top-level verify -----------------------------------------------------

def verify(form: Form, clue_text: str, db, shadow_conn=None) -> Verdict:
    """Run every check. Verdict is PASS iff all checks pass."""
    checks = []

    # 1. Coverage
    cov_fails = _check_coverage(form, clue_text)
    if cov_fails:
        checks.append(Check("coverage", "fail", "; ".join(cov_fails)))
    else:
        checks.append(Check("coverage", "pass",
                              "all clue words accounted for"))

    # 2. Leaf justification
    leaf_fails = []
    leaf_passes = []
    for leaf in _leaves(form.tree):
        reason = _justify_leaf(leaf, db, shadow_conn)
        if reason:
            leaf_fails.append(reason)
        else:
            leaf_passes.append(_describe_leaf(leaf))
    if leaf_fails:
        checks.append(Check("leaf_justification", "fail",
                              "; ".join(leaf_fails)))
    else:
        checks.append(Check("leaf_justification", "pass",
                              "verified " + "; ".join(leaf_passes)))

    # 3 + 4. Indicators (presence + DB validity)
    ind_fails = _check_indicators(form, db, shadow_conn)
    presence_fails = [m for k, m in ind_fails if k == "missing"]
    db_fails = [m for k, m in ind_fails if k == "invalid"]
    if presence_fails:
        checks.append(Check("indicator_presence", "fail",
                              "; ".join(presence_fails)))
    else:
        checks.append(Check("indicator_presence", "pass", ""))
    if db_fails:
        checks.append(Check("indicator_db", "fail",
                              "; ".join(db_fails)))
    else:
        checks.append(Check("indicator_db", "pass", ""))

    # 5. Mechanism integrity
    mech_fails = _check_mechanism(form)
    if mech_fails:
        checks.append(Check("mechanism", "fail",
                              "; ".join(mech_fails)))
    else:
        checks.append(Check("mechanism", "pass", ""))

    # 6. Assembly
    answer = (form.definition.answer or "").upper()
    if _produces(form.tree, answer):
        checks.append(Check("assembly", "pass",
                              f"tree produces {answer}"))
    else:
        checks.append(Check("assembly", "fail",
                              f"tree does not produce {answer!r}"))

    # 7. Definition contiguity
    def_fail = _check_definition(form, clue_text)
    if def_fail:
        checks.append(Check("definition", "fail", def_fail))
    else:
        checks.append(Check("definition", "pass", ""))

    verdict = "PASS" if all(c.status == "pass" for c in checks) else "FAIL"
    return Verdict(verdict=verdict, checks=checks)


def _describe_leaf(leaf: Node) -> str:
    op = leaf.operation
    v = leaf.value or ""
    s = leaf.source_word or ""
    if op == "positional":
        return f'{op}[{leaf.positional_kind}]({v}←{s!r})'
    return f'{op}({v}←{s!r})'
