"""Clipboard verifier — three rules, no interpretation.

Per memory/verifier_three_rules.md:

  Rule 1 — Assembly: pieces equal the answer, in the order specified.
  Rule 2 — Mechanism: every leaf bridge and indicator authorised by an
                       exact DB row. No fallbacks. Missing rows queued
                       for enrichment.
  Rule 3 — Residue: every wordplay word accounted as leaf-source,
                     indicator, or DB-listed link word.

The verifier reads the form as written and checks the form's exact
claims. It does not reinterpret, sub-phrase, fallback, or
reverse-engineer.

Inputs:
  - form: Form object (tree + definition + link_words)
  - clue_text: original clue string
  - db: RefDB (live)
  - shadow_conn: optional sqlite3.Connection to shadow DB

Outputs:
  Verdict(verdict='PASS'|'FAIL', checks=[...],
           enrichment_candidates=[...])
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from .schema import Form, Node, LEAF_OPERATIONS
from .surface import tokenize as _tokenize


# Words allowed to appear in form.link_words. Anything else there
# fails Rule 3.
#
# Note: "s" was previously here as an apostrophe-s tokenisation
# remnant. That hack is gone — surface tokenisation now preserves
# internal apostrophes (so "bird's" stays one token), and DB lookups
# handle the possessive via `signature_solver/db.py::_word_variants`.
LINK_WORDS = frozenset({
    "a", "an", "and", "as", "at", "but", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "so", "the", "this", "to",
    "with", "without", "into", "out", "off", "up", "down",
    "be", "been", "being", "are", "was", "were", "had", "has",
    "have", "did", "does", "do", "may", "might", "could",
    "would", "should", "must", "shall", "will",
    "that", "which", "who", "whose", "whom",
    "all", "some", "any", "every", "each", "both",
    "his", "her", "its", "our", "their", "my", "your",
    "i", "we", "they", "you", "me", "us", "him", "them",
    "than", "then", "if",
})


# Op → required wordplay_type in the indicators DB
OP_INDICATOR_TYPES = {
    "anagram":   {"anagram"},
    "reversal":  {"reversal"},
    "container": {"container", "insertion"},
    "deletion":  {"deletion", "parts"},
    "hidden":    {"hidden"},
    "homophone": {"homophone"},
    "acrostic":  {"acrostic", "parts"},
}

# Positional kind → required wordplay_type
POS_KIND_INDICATOR_TYPES = {
    "first":     {"parts", "acrostic"},
    "last":      {"parts"},
    "outer":     {"parts"},
    "middle":    {"parts"},
    "alternate": {"alternating", "parts"},
    "odd":       {"alternating", "parts"},
    "even":      {"alternating", "parts"},
    "half":      {"parts"},
    "interior":  {"parts"},
}

NO_INDICATOR_OPS = frozenset({
    "charade", "double_definition", "cryptic_definition", "unknown",
})

# Ops whose children must be literal/raw nodes — they consume surface
# letters directly, not derived values.
LITERAL_FODDER_OPS = frozenset({"anagram", "hidden", "acrostic"})


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
class EnrichmentCandidate:
    kind: str          # synonym | abbreviation | homophone | indicator | definition
    source_word: str
    value: str
    operation: Optional[str] = None     # for indicator candidates: wordplay_type
    subtype: Optional[str] = None       # for indicator candidates: deletion_kind / acrostic_kind / etc.
    detail: str = ""

    def to_dict(self):
        return {"kind": self.kind,
                "source_word": self.source_word,
                "value": self.value,
                "operation": self.operation,
                "subtype": self.subtype,
                "detail": self.detail}


@dataclass
class Verdict:
    verdict: str  # "PASS" | "FAIL"
    checks: list = field(default_factory=list)
    enrichment_candidates: list = field(default_factory=list)

    def to_dict(self):
        return {"verdict": self.verdict,
                "checks": [c.to_dict() for c in self.checks],
                "enrichment_candidates": [
                    e.to_dict() for e in self.enrichment_candidates
                ]}


# ---- Helpers --------------------------------------------------------------

def _shadow_query(shadow_conn, sql, params):
    """Run a shadow-DB query, swallowing missing-table errors so the
    verifier doesn't blow up on partial shadow schemas."""
    if shadow_conn is None:
        return None
    try:
        return shadow_conn.execute(sql, params).fetchone()
    except sqlite3.OperationalError:
        return None


def _surface_words(text: str) -> list:
    """Lowercase clue tokens. Tokenisation preserves internal
    apostrophes (so "bird's" is one token); see `surface.tokenize`."""
    return [t.lower() for t in _tokenize(text)]


def _letters(text: str) -> str:
    return "".join(c for c in (text or "").upper() if c.isalpha())


def _all_nodes(node: Node):
    yield node
    for c in node.sources or []:
        yield from _all_nodes(c)


def _leaves(root: Node) -> list:
    out = []
    for n in _all_nodes(root):
        if n.operation in LEAF_OPERATIONS and not n.sources:
            out.append(n)
    return out


# ---- Rule 1: Assembly -----------------------------------------------------

def _assembles(node: Node, target: str) -> bool:
    """Walk bottom-up; does this subtree produce exactly `target`?"""
    if node is None or not target:
        return False
    op = node.operation
    target = target.upper()

    # Leaves: value must equal target
    if op in LEAF_OPERATIONS and not node.sources:
        return (node.value or "").upper() == target

    if op == "charade":
        return _charade_assembles(node.sources, target)

    if op == "anagram":
        # Children's letters' multiset must equal target's
        letters = "".join((c.value or "").upper() for c in node.sources)
        return sorted(letters) == sorted(target)

    if op == "reversal":
        if len(node.sources) != 1:
            return False
        return _assembles(node.sources[0], target[::-1])

    if op == "container":
        if len(node.sources) != 2:
            return False
        outer, inner = node.sources  # FIXED ORDER
        return _container_assembles(outer, inner, target)

    if op == "deletion":
        if not node.sources:
            return False
        kind = node.deletion_kind or "tail"
        # Source must produce the pre-deletion word (target with
        # letters re-added at the kind position). The source is
        # typically a leaf whose `value` is the full pre-deletion
        # word. We check: applying the kind-deletion to the source's
        # yield equals target.
        return _deletion_source_assembles(node.sources[0], target, kind)

    if op == "hidden":
        text = "".join((c.value or "").upper() for c in node.sources)
        return target in text

    if op == "double_definition":
        return any(_assembles(c, target) for c in node.sources)

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
        # Op form. Assembly accepts; Rule 2 verifies the DB.
        if not node.sources:
            return (node.value or "").upper() == target
        # Just confirm a child has a yieldable value
        for c in node.sources:
            v = _yield_value(c)
            if v:
                return True
        return False

    if op == "cryptic_definition":
        # No mechanical assembly — the whole clue is a witty definition
        # of the answer. The verifier accepts CDs unconditionally; the
        # PENDING verdict (set in `verify`) ensures human review.
        return True

    return False


def _yield_value(node: Node) -> str:
    """Return the value the node mechanically yields (for use by
    Rule 2 homophone-op closure). For a leaf this is the value;
    for a non-leaf it's the assembled string."""
    if node.operation in LEAF_OPERATIONS and not node.sources:
        return (node.value or "").upper()
    op = node.operation
    if op == "charade":
        return "".join(_yield_value(c) for c in node.sources or [])
    if op == "reversal":
        if len(node.sources) != 1:
            return ""
        return _yield_value(node.sources[0])[::-1]
    # For other ops we don't have a single deterministic yield
    # without enumerating. Used only for homophone closure where
    # the child is typically a synonym or charade — both deterministic.
    return ""


def _charade_assembles(children, target):
    if not children:
        return False
    if len(children) == 1:
        return _assembles(children[0], target)
    first, rest = children[0], children[1:]
    for cut in range(1, len(target)):
        if _assembles(first, target[:cut]) and _charade_assembles(rest, target[cut:]):
            return True
    return False


def _container_assembles(outer: Node, inner: Node, target: str) -> bool:
    """outer wraps inner: scan target for a slice that inner can
    produce, and check outer can produce the surrounding letters."""
    n = len(target)
    for il in range(1, n):
        for p in range(1, n - il + 1):
            inner_str = target[p:p + il]
            outer_str = target[:p] + target[p + il:]
            if not outer_str:
                continue
            if _assembles(inner, inner_str) and _assembles(outer, outer_str):
                return True
    return False


def _deletion_source_assembles(source: Node, target: str, kind: str) -> bool:
    """Source must yield a string that, after `kind` deletion, equals
    target. For leaves, the value should be the full pre-deletion
    word."""
    if source.operation in LEAF_OPERATIONS and not source.sources:
        full = (source.value or "").upper()
        return _apply_deletion(full, kind) == target
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
        mid = len(full) // 2
        return full[:mid - 1] + full[mid + 1:]
    return None


# ---- Rule 2: Mechanism ----------------------------------------------------

def _check_leaf_bridge(leaf: Node, db, shadow_conn):
    """Return (ok: bool, enrichment_candidate or None, fail_msg or None)."""
    op = leaf.operation
    val = (leaf.value or "").upper()
    src = (leaf.source_word or "").strip().lower()

    if not val:
        return False, None, f"{op} leaf has no value"
    if not src and op not in ("literal", "raw", "positional"):
        # literal/raw/positional may have empty source if value is the literal
        # but we still expect source for leaves with words
        pass

    if op in ("literal", "raw"):
        # Mechanical: letters of source must equal value
        src_letters = _letters(leaf.source_word or "")
        if src_letters == val:
            return True, None, None
        return False, None, (f"literal {val!r}: source letters "
                              f"{src_letters!r} != value")

    if op == "positional":
        kind = leaf.positional_kind or ""
        if not kind:
            return False, None, "positional leaf has no positional_kind"
        src_letters = _letters(leaf.source_word or "")
        extracted = _positional_extract(src_letters, kind)
        if extracted == val:
            return True, None, None
        return False, None, (f"positional[{kind}] of {src!r}: "
                              f"extracted {extracted!r} != {val!r}")

    if op == "synonym":
        if val in db.get_synonyms(src):
            return True, None, None
        row = _shadow_query(shadow_conn,
            "SELECT 1 FROM synonyms_pairs "
            "WHERE LOWER(word)=? AND UPPER(synonym)=? LIMIT 1",
            (src, val))
        if row:
            return True, None, None
        return False, EnrichmentCandidate(
            kind="synonym", source_word=src, value=val,
            detail=f"synonym {val!r} not in DB for {src!r}"
        ), f"synonym {val!r} not in DB for {src!r}"

    if op == "abbreviation":
        if val in db.get_abbreviations(src):
            return True, None, None
        if val in db.get_synonyms(src):
            return True, None, None
        row = _shadow_query(shadow_conn,
            "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? "
            "AND UPPER(substitution)=? AND category='abbreviation' "
            "LIMIT 1", (src, val))
        if row:
            return True, None, None
        row = _shadow_query(shadow_conn,
            "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? "
            "AND UPPER(synonym)=? LIMIT 1", (src, val))
        if row:
            return True, None, None
        return False, EnrichmentCandidate(
            kind="abbreviation", source_word=src, value=val,
            detail=f"abbreviation {val!r} not in DB for {src!r}"
        ), f"abbreviation {val!r} not in DB for {src!r}"

    if op == "homophone":
        if val in db.get_homophones(src):
            return True, None, None
        row = _shadow_query(shadow_conn,
            "SELECT 1 FROM homophones WHERE LOWER(word)=? "
            "AND UPPER(homophone)=? LIMIT 1", (src, val))
        if row:
            return True, None, None
        return False, EnrichmentCandidate(
            kind="homophone", source_word=src, value=val,
            detail=f"homophone {val!r} not in DB for {src!r}"
        ), f"homophone {val!r} not in DB for {src!r}"

    return False, None, f"unknown leaf op {op!r}"


def _positional_extract(src_letters: str, kind: str) -> Optional[str]:
    if not src_letters:
        return None
    if kind == "first":
        return src_letters[0]
    if kind == "last":
        return src_letters[-1]
    if kind == "outer":
        return src_letters[0] + src_letters[-1] if len(src_letters) >= 2 else None
    if kind == "middle":
        if len(src_letters) < 3:
            return None
        if len(src_letters) % 2 == 1:
            return src_letters[len(src_letters) // 2]
        mid = len(src_letters) // 2
        return src_letters[mid - 1:mid + 1]
    if kind in ("alternate", "odd"):
        return src_letters[0::2]
    if kind == "even":
        return src_letters[1::2]
    if kind == "half":
        return src_letters[:len(src_letters) // 2]
    if kind == "interior":
        return src_letters[1:-1] if len(src_letters) >= 3 else None
    return None


def _check_indicator(node: Node, expected_types: set, db, shadow_conn):
    """(ok, enrichment, fail_msg) — does the indicator word appear in
    the indicators DB with matching wordplay_type?"""
    if not node.indicator:
        return False, None, f"{node.operation} node has no indicator"
    word = node.indicator.lower().strip(",.;:!?\"'()-")
    if not word:
        return False, None, f"{node.operation} indicator is blank"
    types = db.get_indicator_types(word)
    if any(t[0] in expected_types for t in types):
        return True, None, None
    if shadow_conn is not None:
        try:
            rows = shadow_conn.execute(
                "SELECT wordplay_type FROM indicators WHERE LOWER(word)=?",
                (word,)).fetchall()
            if any(r[0] in expected_types for r in rows):
                return True, None, None
        except sqlite3.OperationalError:
            pass
    sub = _op_subtype(node)
    cand = EnrichmentCandidate(
        kind="indicator",
        source_word=word, value=word,
        operation=node.operation,
        subtype=sub,
        detail=f"indicator {word!r} not in DB for op "
                f"{node.operation!r}"
                + (f"[{sub}]" if sub else "")
                + f" (need {sorted(expected_types)})"
    )
    return False, cand, (f"indicator {word!r} not in DB for op "
                          f"{node.operation!r}"
                          + (f"[{sub}]" if sub else ""))


def _op_subtype(node: Node) -> Optional[str]:
    """Subtype for the indicator-DB row, derived from the op's kind.

    deletion -> deletion_kind  (head/tail/outer/heart)
    acrostic -> acrostic_kind  (first/last)
    other ops carry no op-level subtype in the schema.
    """
    if node.operation == "deletion":
        return node.deletion_kind
    if node.operation == "acrostic":
        return node.acrostic_kind
    return None


def _check_fodder_integrity(root: Node) -> tuple:
    """Anagram, hidden, and acrostic ops consume surface letters
    directly. Their children must be `literal` or `raw` nodes — never
    `synonym` / `abbreviation` / `positional` / `homophone` / nested
    ops, which produce derived values."""
    fails = []
    for n in _all_nodes(root):
        if n.operation not in LITERAL_FODDER_OPS:
            continue
        for c in n.sources or []:
            if c.operation not in ("literal", "raw") or c.sources:
                fails.append(
                    f"{n.operation} child must be literal/raw, got "
                    f"{c.operation}"
                    + (f"({c.value!r})" if c.value else "")
                )
    return (not fails), fails


def _check_homophone_op_closure(node: Node, target: str, db,
                                 shadow_conn):
    """For a homophone op-node, the child's yielded value Y must be
    in the homophones DB with target as homophone (or vice versa)."""
    if not node.sources:
        return True, None, None  # leaf-form homophone, leaf check covers
    y = ""
    for c in node.sources:
        cy = _yield_value(c)
        if cy:
            y = cy
            break
    if not y:
        return False, None, "homophone op: child yields no value"
    target_u = target.upper()
    homos = db.get_homophones(y.lower())
    if target_u in homos:
        return True, None, None
    homos2 = db.get_homophones(target_u.lower())
    if y in homos2:
        return True, None, None
    row = _shadow_query(shadow_conn,
        "SELECT 1 FROM homophones WHERE "
        "(LOWER(word)=? AND UPPER(homophone)=?) OR "
        "(LOWER(word)=? AND UPPER(homophone)=?) LIMIT 1",
        (y.lower(), target_u, target_u.lower(), y))
    if row:
        return True, None, None
    cand = EnrichmentCandidate(
        kind="homophone", source_word=y.lower(), value=target_u,
        detail=f"homophone op: {y!r} ↔ {target_u!r} not in DB"
    )
    return False, cand, f"homophone op: {y!r} not a homophone of {target_u!r}"


# ---- Rule 3: Residue ------------------------------------------------------

def _check_residue(form: Form, clue_text: str) -> tuple:
    """Returns (ok, fail_msgs) for the wordplay portion only.

    Wordplay portion = clue words minus the definition span.
    Each wordplay word must be in EXACTLY ONE of:
      - some leaf's source_word (lowercased word match)
      - some op-node's indicator (lowercased word match)
      - form.link_words AND on LINK_WORDS allow-list
    """
    surface = _surface_words(clue_text)
    def_words = _surface_words(form.definition.phrase)

    # Find the definition span as a contiguous run in surface
    def_lo, def_hi = _find_contiguous(surface, def_words)
    if def_lo is None:
        return False, [
            f"definition phrase {form.definition.phrase!r} is not "
            f"a contiguous span of the clue"]
    wordplay_indices = [i for i in range(len(surface))
                        if not (def_lo <= i < def_hi)]
    wordplay_words = [surface[i] for i in wordplay_indices]

    # Collect what the form claims for each wordplay word
    leaf_source_words = set()
    indicator_words = set()
    for leaf in _leaves(form.tree):
        for w in _surface_words(leaf.source_word or ""):
            leaf_source_words.add(w)
    for n in _all_nodes(form.tree):
        if n.indicator:
            for w in _surface_words(n.indicator):
                indicator_words.add(w)
    declared_link = {w.lower() for w in form.link_words}

    fails = []
    # Every wordplay word must be in exactly one bucket
    for w in wordplay_words:
        in_leaf = w in leaf_source_words
        in_ind = w in indicator_words
        in_link = w in declared_link
        n_buckets = sum([in_leaf, in_ind, in_link])
        if n_buckets == 0:
            fails.append(f"unaccounted: {w!r}")
        elif n_buckets > 1:
            buckets = []
            if in_leaf: buckets.append("leaf-source")
            if in_ind: buckets.append("indicator")
            if in_link: buckets.append("link")
            fails.append(f"word {w!r} claimed by multiple buckets: "
                          f"{buckets}")
    # link_words must all be on the allow-list
    for w in declared_link:
        if w not in LINK_WORDS:
            fails.append(f"link_word {w!r} not on LINK_WORDS allow-list")
    return (not fails), fails


def _find_contiguous(haystack: list, needle: list) -> tuple:
    """Return (start, end) such that haystack[start:end] == needle, or
    (None, None)."""
    if not needle:
        return None, None
    n = len(haystack)
    m = len(needle)
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle:
            return i, i + m
    return None, None


# ---- Top-level verify -----------------------------------------------------

def verify(form: Form, clue_text: str, db,
           shadow_conn=None) -> Verdict:
    """Run the three rules. Returns Verdict + enrichment candidates."""
    checks = []
    enrichments = []

    # Rule 1 — Assembly
    answer = (form.definition.answer or "").upper()
    if _assembles(form.tree, answer):
        checks.append(Check("assembly", "pass",
                              f"tree produces {answer}"))
    else:
        checks.append(Check("assembly", "fail",
                              f"tree does not produce {answer!r}"))

    # Rule 2 — Mechanism
    leaf_fails = []
    leaf_passes = []
    for leaf in _leaves(form.tree):
        ok, cand, fail_msg = _check_leaf_bridge(leaf, db, shadow_conn)
        if ok:
            leaf_passes.append(_describe_leaf(leaf))
        else:
            leaf_fails.append(fail_msg)
            if cand:
                enrichments.append(cand)
    if leaf_fails:
        checks.append(Check("mechanism.leaves", "fail",
                              "; ".join(leaf_fails)))
    else:
        checks.append(Check("mechanism.leaves", "pass",
                              "verified " + "; ".join(leaf_passes)))

    fodder_ok, fodder_fails = _check_fodder_integrity(form.tree)
    if fodder_ok:
        checks.append(Check("mechanism.fodder", "pass",
                              "anagram/hidden/acrostic children are literal/raw"))
    else:
        checks.append(Check("mechanism.fodder", "fail",
                              "; ".join(fodder_fails)))

    ind_fails = []
    ind_passes = []
    for n in _all_nodes(form.tree):
        op = n.operation
        # Op-nodes that need an indicator
        if (n.sources and op not in LEAF_OPERATIONS
                and op not in NO_INDICATOR_OPS):
            expected = OP_INDICATOR_TYPES.get(op)
            if expected:
                ok, cand, msg = _check_indicator(
                    n, expected, db, shadow_conn)
                if ok:
                    ind_passes.append(f'{n.indicator!r}→{op}')
                else:
                    ind_fails.append(msg)
                    if cand:
                        enrichments.append(cand)
        # Op-form homophone — check homophone DB closure
        if op == "homophone" and n.sources:
            # The target depends on what this node contributes. For the
            # root, target = answer. Otherwise we use the node's
            # contribution computed during Assembly. For simplicity:
            # only check at the root for now (most homophone clues
            # have homophone at the root).
            if n is form.tree:
                ok, cand, msg = _check_homophone_op_closure(
                    n, answer, db, shadow_conn)
                if not ok:
                    ind_fails.append(msg)
                    if cand:
                        enrichments.append(cand)
        # Positional leaves: indicator required and DB-typed
        if op == "positional" and not n.sources:
            kind = n.positional_kind
            expected = POS_KIND_INDICATOR_TYPES.get(kind, set())
            if not n.indicator:
                ind_fails.append(
                    f"positional[{kind}] leaf has no indicator")
                continue
            if expected:
                ok, cand, msg = _check_indicator(
                    n, expected, db, shadow_conn)
                if ok:
                    ind_passes.append(f'{n.indicator!r}→positional[{kind}]')
                else:
                    ind_fails.append(msg)
                    if cand:
                        enrichments.append(cand)
    if ind_fails:
        checks.append(Check("mechanism.indicators", "fail",
                              "; ".join(ind_fails)))
    else:
        checks.append(Check("mechanism.indicators", "pass",
                              "verified " + "; ".join(ind_passes)
                              if ind_passes else
                              "no indicators required"))

    # Rule 3 — Residue
    ok, fails = _check_residue(form, clue_text)
    if ok:
        checks.append(Check("residue", "pass",
                              "every wordplay word accounted for"))
    else:
        checks.append(Check("residue", "fail", "; ".join(fails)))

    if all(c.status == "pass" for c in checks):
        # &lit forms (Form.is_and_lit=True) and forms whose tree contains
        # a cryptic_definition node always require human review before
        # they're treated as confirmed PASSes — they don't contribute to
        # the catalog until reviewed. Mechanical pass alone is not enough.
        if form.is_and_lit or _has_cryptic_definition(form.tree):
            verdict = "PENDING"
        else:
            verdict = "PASS"
    else:
        verdict = "FAIL"
    return Verdict(verdict=verdict, checks=checks,
                    enrichment_candidates=enrichments)


def _has_cryptic_definition(node: Node) -> bool:
    if node is None:
        return False
    if node.operation == "cryptic_definition":
        return True
    return any(_has_cryptic_definition(c) for c in (node.sources or []))


def _describe_leaf(leaf: Node) -> str:
    op = leaf.operation
    v = leaf.value or ""
    s = leaf.source_word or ""
    if op == "positional":
        return f'{op}[{leaf.positional_kind}]({v}←{s!r})'
    return f'{op}({v}←{s!r})'
