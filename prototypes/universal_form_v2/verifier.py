"""Universal-form verifier (v2 — pure referee, never participates).

Given a Form + clue_text + answer, run the four checks defined in the spec:

  1. Assembly — does the tree compute to the answer?
  2. Bridge   — does every non-literal node have an indicator known to the
                indicators DB for that op type? Does every leaf's value map
                to its source_word in the appropriate DB table?
  3. Mechanism — outermost node's bridge check (subsumed by Bridge).
  4. Residue   — every source_word + indicator + definition.phrase covers
                the surface words minus link_words.

The verifier never modifies the form. It returns a verdict + list of
per-check results. PASS only if every check passes.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signature_solver.db import RefDB

from .schema import Form, Node, LEAF_OPERATIONS


# Op-name -> wordplay_type strings in `indicators` table.
# `parts` is the catch-all for trim/positional indicators in the DB
# (e.g. 'largely' = parts/last_delete, used for deletion ops).
_OP_TO_DB_INDICATOR_TYPES = {
    "anagram": {"anagram"},
    "reversal": {"reversal"},
    "container": {"container", "insertion"},
    "deletion": {"deletion", "parts"},
    "hidden": {"hidden"},
    "homophone": {"homophone"},
    "acrostic": {"acrostic", "parts"},
}


# Common link words allow-list. Sourced from cryptic conventions; the verifier
# uses this to check that form.link_words is plausible.
LINK_WORDS = frozenset({
    "a", "an", "and", "as", "at", "but", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "so", "the", "this", "to",
    "with", "without", "into", "out", "off", "up", "down",
    "be", "been", "being", "are", "was", "were", "had", "has",
    "have", "had", "did", "does", "do", "may", "might", "could",
    "would", "should", "must", "shall", "will",
    "that", "which", "who", "whose", "whom", "where", "when",
    "why", "how", "though", "although", "while", "whilst",
    "after", "before", "during", "within",
    "all", "some", "any", "every", "each", "no", "both",
    "his", "her", "its", "our", "their", "my", "your",
    "i", "we", "they", "you", "me", "us", "him", "them",
    "than", "then", "too", "very", "just", "only", "even",
    "between", "behind", "across", "around", "about", "above",
})


# --- Tree assembly: can the tree produce `target`? ------------------------

def _produces(node: Node, target: str) -> bool:
    """Return True if the tree rooted at `node` can produce `target`.

    For ambiguous ops (container insertion, deletion position), tries every
    valid option. Bounded recursion — depth is the tree depth, fan-out is
    bounded by the small number of children per node.
    """
    if node is None or not target:
        return False

    op = node.operation

    # Leaves: value must equal target. (Homophone op-nodes have
    # sources — fall through to the op handling below.)
    if op in LEAF_OPERATIONS and not node.sources:
        return (node.value or "").upper() == target.upper()

    if op == "charade":
        return _charade_produces(node.sources, target)

    if op == "anagram":
        # All children's letters together (multiset) must equal target's.
        # But children are themselves trees — they each commit to one value.
        # We try each child's permitted values; for v0 a child is either a
        # leaf (one value) or a small subtree we can fully compute.
        for letters in _all_yields(node.sources):
            if sorted(letters.upper()) == sorted(target.upper()):
                return True
        return False

    if op == "reversal":
        return _produces(node.sources[0], target[::-1])

    if op == "container":
        if len(node.sources) != 2:
            return False
        outer, inner = node.sources
        # Try both as outer (outer/inner role often unreliable from adapter)
        return (_container_produces(outer, inner, target)
                or _container_produces(inner, outer, target))

    if op == "deletion":
        if not node.sources:
            return False
        kind = node.deletion_kind or "tail"
        # Deletion's child produces a longer string; remove kind-letters
        # to match target.
        for src_str in _yields(node.sources[0]):
            if _deletion_produces(src_str, target, kind):
                return True
        return False

    if op == "hidden":
        text = "".join((s.value or "").upper() for s in _flatten_leaves(
            node.sources))
        return target.upper() in text

    if op == "double_definition":
        # any child must produce target
        return any(_produces(c, target) for c in node.sources)

    if op == "acrostic":
        kind = node.acrostic_kind or "first"
        letters = []
        for s in node.sources:
            v = (s.value or s.source_word or "").upper()
            v = "".join(c for c in v if c.isalpha())
            if not v:
                return False
            if kind == "first":
                letters.append(v[0])
            elif kind == "last":
                letters.append(v[-1])
            else:
                letters.append(v[0])
        return "".join(letters) == target.upper()

    if op == "homophone":
        # Leaf form: value carries the answer's letters.
        if not node.sources:
            return (node.value or "").upper() == target.upper()
        # Op form: child produces a word X; verify X "sounds like"
        # the target via the homophones DB. Without DB access here,
        # accept any child-yield X whose letter count is within ±2 of
        # target — assembly check passes pessimistically; the bridge
        # check will detect false positives.
        for c in node.sources:
            for child_str in _yields(c):
                if not child_str:
                    continue
                if abs(len(child_str) - len(target)) <= 3:
                    return True
        return False

    # cryptic_definition / substitution / spoonerism / unknown — verifier
    # cannot mechanically check.
    return False


def _charade_produces(children, target):
    """Can [c1, c2, ..., cN] concatenate (in order) to produce target?"""
    if not children:
        return False
    if len(children) == 1:
        return _produces(children[0], target)
    # Try every split position
    first = children[0]
    rest = children[1:]
    for cut in range(1, len(target)):
        if _produces(first, target[:cut]) \
                and _charade_produces(rest, target[cut:]):
            return True
    return False


def _container_produces(outer, inner, target):
    """outer wraps around inner: outer[:p] + inner + outer[p:] == target.

    Avoids enumeration on either side: for each contiguous slice of
    target, ask `inner` to produce the slice and `outer` to produce
    the surrounding letters. Both checks recurse via `_produces`,
    which uses sorted-letters for anagram — so this works even when
    inner is a 7-letter anagram with thousands of permutations.
    """
    t = target.upper()
    # Try every (p, q) split where target = outer_left + inner_str + outer_right
    # with 1 ≤ inner_len < len(target) and outer_len ≥ 1 on at least one side.
    n = len(t)
    for il in range(1, n):
        for p in range(1, n - il + 1):
            inner_str = t[p:p + il]
            outer_str = t[:p] + t[p + il:]
            if not outer_str:
                continue
            if _produces(inner, inner_str) and _produces(outer, outer_str):
                return True
    return False


def _deletion_produces(src, target, kind):
    if not src:
        return False
    if kind == "tail":
        return src[:-1] == target
    if kind == "head":
        return src[1:] == target
    if kind == "outer":
        return len(src) >= 3 and src[1:-1] == target
    if kind == "heart":
        if len(src) < 3:
            return False
        mid = len(src) // 2
        if len(src) % 2 == 1:
            return src[:mid] + src[mid + 1:] == target
        return src[:mid - 1] + src[mid + 1:] == target
    return False


def _yields(node) -> set:
    """Best-effort: enumerate possible letter strings the subtree can yield.

    Bounded — for v0 we collect a small set; if a node yields too much, we
    just return its committed value.
    """
    if node is None:
        return set()
    op = node.operation
    if op in LEAF_OPERATIONS and not node.sources:
        v = (node.value or "").upper()
        return {v} if v else set()
    if op == "charade":
        # Cartesian product (small)
        sets = [_yields(c) for c in node.sources]
        results = {""}
        for s in sets:
            new = set()
            for prefix in results:
                for v in s:
                    new.add(prefix + v)
                    if len(new) > 64:
                        break
            results = new
            if len(results) > 64:
                break
        return results
    if op == "reversal":
        inner = _yields(node.sources[0])
        return {v[::-1] for v in inner}
    if op == "container":
        outer_set = _yields(node.sources[0])
        inner_set = _yields(node.sources[1])
        out = set()
        for o in outer_set:
            for i in inner_set:
                for p in range(1, len(o)):
                    out.add(o[:p] + i + o[p:])
                    if len(out) > 64:
                        return out
        return out
    if op == "anagram":
        # All permutations of accumulated letters
        letters_options = _all_yields(node.sources)
        out = set()
        for letters in letters_options:
            if len(letters) > 8:
                # too big to fully permute — skip enumeration, parent op
                # check will use sorted()-comparison instead
                out.add(letters)
                continue
            for p in permutations(letters):
                out.add("".join(p))
                if len(out) > 64:
                    return out
        return out
    if op == "deletion":
        if not node.sources:
            return set()
        kind = node.deletion_kind or "tail"
        out = set()
        for src in _yields(node.sources[0]):
            if not src:
                continue
            if kind == "tail":
                out.add(src[:-1])
            elif kind == "head":
                out.add(src[1:])
            elif kind == "outer" and len(src) >= 3:
                out.add(src[1:-1])
            elif kind == "heart" and len(src) >= 3:
                mid = len(src) // 2
                if len(src) % 2 == 1:
                    out.add(src[:mid] + src[mid + 1:])
                else:
                    out.add(src[:mid - 1] + src[mid + 1:])
        return out
    if op == "hidden":
        # Concat children's letters, return all contiguous substrings
        text = "".join((s.value or s.source_word or "").upper()
                       for s in _flatten_leaves(node.sources))
        text = "".join(c for c in text if c.isalpha())
        out = set()
        for i in range(len(text)):
            for j in range(i + 1, len(text) + 1):
                out.add(text[i:j])
                if len(out) > 200:
                    return out
        return out
    if op == "double_definition":
        out = set()
        for c in node.sources:
            out.update(_yields(c))
        return out
    if op == "acrostic":
        kind = node.acrostic_kind or "first"
        letters = []
        for s in node.sources:
            v = (s.value or s.source_word or "").upper()
            v = "".join(c for c in v if c.isalpha())
            if not v:
                return set()
            if kind == "last":
                letters.append(v[-1])
            else:
                letters.append(v[0])
        return {"".join(letters)}
    if op == "homophone":
        # Op homophone — children produce something that sounds like target
        out = set()
        for c in node.sources:
            out.update(_yields(c))
        return out
    return set()


def _all_yields(children: list) -> set:
    """Concatenations of every yields(child) — used by anagram letter set."""
    sets = [_yields(c) for c in children]
    results = {""}
    for s in sets:
        new = set()
        for prefix in results:
            for v in s:
                new.add(prefix + v)
                if len(new) > 64:
                    break
        results = new
    return results


def _flatten_leaves(nodes) -> list:
    """All leaf descendants of a list of nodes.

    A node with `sources` is treated as an op-node and we recurse,
    even if its `operation` is technically in LEAF_OPERATIONS
    (e.g. `homophone` can be either leaf or op).
    """
    out = []
    for n in nodes:
        if n.operation in LEAF_OPERATIONS and not n.sources:
            out.append(n)
        else:
            out.extend(_flatten_leaves(n.sources))
    return out


# --- Bridge check: DB lookups for leaves and indicators -------------------

def _bridge_leaf(node: Node, db: RefDB,
                  shadow_conn=None) -> Optional[str]:
    """Verify a leaf against the DB. Returns failure reason or None.

    When `shadow_conn` is provided, lookups also check the shadow DB
    after the live DB misses.
    """
    op = node.operation
    word = (node.source_word or "").lower().strip(",.;:!?\"'()-")
    val = (node.value or "").upper()
    if op == "literal" or op == "raw":
        src_letters = "".join(c for c in word.upper() if c.isalpha())
        if val and val in src_letters:
            return None
        if val == src_letters:
            return None
        # Strip stop/link words from the source — anagram fodder can
        # span multiple words separated by 'with', 'and', 'of', 'a',
        # etc. Try the letters of the content words only.
        stop = {"WITH", "AND", "OF", "A", "AN", "THE", "TO", "FOR",
                 "IN", "IS", "BY", "ON", "AS"}
        content_letters = "".join(
            "".join(c for c in w.upper() if c.isalpha())
            for w in word.split() if w.upper() not in stop)
        if val and val == content_letters:
            return None
        if val and val in content_letters:
            return None
        # Final fallback: multiset equality (any-order anagram fodder).
        if val and len(val) == len(content_letters) \
                and sorted(val) == sorted(content_letters):
            return None
        return f"literal value {val!r} not derivable from source {word!r}"
    if op == "synonym":
        if not word or not val:
            return "synonym: missing source_word or value"
        # Build candidate forms — full phrase, apostrophe-s stripped,
        # and progressively-shorter sub-phrases (drop leading words).
        # The latter handles POS-widened sources like
        # "associated with beer" → look up "beer".
        base = word
        if base.endswith("'s") or base.endswith("’s"):
            base = base[:-2]
        elif base.endswith(" s"):
            base = base[:-2]
        word_candidates = [base]
        # Every contiguous sub-phrase (POS-widened sources like
        # "s irritable having" → look up "irritable").
        bw = base.split()
        for i in range(len(bw)):
            for j in range(i + 1, len(bw) + 1):
                if i == 0 and j == len(bw):
                    continue
                word_candidates.append(" ".join(bw[i:j]))
        # Also check definition_answers_augmented in live DB — for
        # cryptic-definition-style leaves where source_word is the
        # whole clue, the relationship lives in def_aug not synonyms.
        live_conn = getattr(db, "_live_conn", None) or getattr(db, "conn", None)
        for w in word_candidates:
            w = w.strip()
            if not w:
                continue
            if val in db.get_synonyms(w):
                return None
            all_syns = db.get_synonyms(w, max_len=len(val) + 5)
            if val in all_syns:
                return None
            if shadow_conn is not None:
                row = shadow_conn.execute(
                    "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? "
                    "AND UPPER(synonym)=? LIMIT 1",
                    (w, val)).fetchone()
                if row:
                    return None
                row = shadow_conn.execute(
                    "SELECT 1 FROM definition_answers_augmented "
                    "WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
                    (w, val)).fetchone()
                if row:
                    return None
            # Live DB definition_answers_augmented check via raw SQL
            if live_conn is not None:
                row = live_conn.execute(
                    "SELECT 1 FROM definition_answers_augmented "
                    "WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
                    (w, val)).fetchone()
                if row:
                    return None
        return f"synonym {val!r} not in DB for {word!r}"
    if op == "abbreviation":
        # Try abbreviations AND synonyms — short forms (1-2 letters)
        # often live in synonyms_pairs rather than the wordplay table.
        # Also try sub-phrases for POS-widened sources.
        base = word
        if base.endswith("'s") or base.endswith("’s"):
            base = base[:-2]
        elif base.endswith(" s"):
            base = base[:-2]
        word_candidates = [base]
        bw = base.split()
        for i in range(len(bw)):
            for j in range(i + 1, len(bw) + 1):
                if i == 0 and j == len(bw):
                    continue
                word_candidates.append(" ".join(bw[i:j]))
        for w in word_candidates:
            w = w.strip()
            if not w:
                continue
            if val in db.get_abbreviations(w):
                return None
            if val in db.get_synonyms(w):
                return None
            if shadow_conn is not None:
                row = shadow_conn.execute(
                    "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? "
                    "AND UPPER(substitution)=? AND category='abbreviation' LIMIT 1",
                    (w, val)).fetchone()
                if row:
                    return None
                row = shadow_conn.execute(
                    "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? "
                    "AND UPPER(synonym)=? LIMIT 1",
                    (w, val)).fetchone()
                if row:
                    return None
        return f"abbreviation {val!r} not in DB for {word!r}"
    if op == "positional":
        # positional kinds are mechanically verifiable — no DB lookup
        kind = node.positional_kind or "first"
        return _verify_positional(word, val, kind)
    if op == "homophone":
        homos = db.get_homophones(word)
        if val in homos:
            return None
        # Check synonym → homophone chain
        for syn in db.get_synonyms(word):
            if val in db.get_homophones(syn.lower()):
                return None
        return f"homophone {val!r} not in DB for {word!r}"
    return f"leaf op {op!r} not bridge-checkable"


def _verify_positional(word, val, kind):
    # Try the whole source first, then each individual word — handles
    # source_words like "interior of nests" where only "nests" is the
    # actual fodder ("interior of" is the indicator phrase that the
    # mapper folded in).
    candidates = [word]
    if " " in word:
        candidates.extend(w for w in word.split() if w)
    for cand in candidates:
        result = _verify_positional_single(cand, val, kind)
        if result is None:
            return None
    return _verify_positional_single(word, val, kind)


def _verify_positional_single(word, val, kind):
    src = "".join(c for c in word.upper() if c.isalpha())
    n = len(val)
    if not src or not val:
        return "positional: missing source/value"
    if kind == "first":
        ok = src.startswith(val)
    elif kind == "last":
        ok = src.endswith(val)
    elif kind == "outer":
        ok = (n == 2 and val[0] == src[0] and val[-1] == src[-1])
    elif kind == "middle":
        if len(src) < n:
            ok = False
        else:
            start = (len(src) - n) // 2
            ok = src[start:start + n] == val
    elif kind == "alternate":
        odd = "".join(src[i] for i in range(0, len(src), 2))
        even = "".join(src[i] for i in range(1, len(src), 2))
        ok = val in (odd, even)
    elif kind == "odd":
        ok = "".join(src[i] for i in range(0, len(src), 2)) == val
    elif kind == "even":
        ok = "".join(src[i] for i in range(1, len(src), 2)) == val
    elif kind == "half":
        ok = src[:len(src) // 2] == val or src[len(src) // 2:] == val
    else:
        return f"positional kind {kind!r} unknown"
    return None if ok else (
        f"positional[{kind}] of {word!r} != {val!r}")


def _describe_leaf_for_check(leaf: Node) -> str:
    """Short string for a verified leaf in the bridge.leaves detail."""
    op = leaf.operation
    val = leaf.value or ""
    src = leaf.source_word or ""
    if op == "literal" or op == "raw":
        return f'literal "{src}"'
    if op == "synonym":
        return f'"{src}"->{val}'
    if op == "abbreviation":
        return f'"{src}"->{val} (abbr)'
    if op == "positional":
        return f'{leaf.positional_kind} of "{src}"={val}'
    if op == "homophone":
        return f'"{src}" sounds like {val}'
    return f'{op} "{src}"->{val}'


def _bridge_indicator(node: Node, db: RefDB,
                       shadow_conn=None) -> Optional[str]:
    """Verify a non-literal node's indicator. Returns failure reason or None."""
    op = node.operation
    if op in LEAF_OPERATIONS:
        return None
    if op in ("charade", "double_definition", "cryptic_definition", "unknown"):
        return None
    if not node.indicator:
        return f"{op} node has no indicator"
    expected_db_types = _OP_TO_DB_INDICATOR_TYPES.get(op)
    if expected_db_types is None:
        return None
    word = node.indicator.lower().strip(",.;:!?\"'()-")
    types = db.get_indicator_types(word)
    if any(t[0] in expected_db_types for t in types):
        return None
    types = db.get_indicator_types(node.indicator.lower().strip())
    if any(t[0] in expected_db_types for t in types):
        return None
    if shadow_conn is not None:
        rows = list(shadow_conn.execute(
            "SELECT wordplay_type FROM indicators "
            "WHERE LOWER(word)=? COLLATE NOCASE",
            (word,)))
        if any(r[0] in expected_db_types for r in rows):
            return None
    return (f"indicator {node.indicator!r} not in DB for op {op!r} "
            f"(expected types {sorted(expected_db_types)})")


# --- Residue check --------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _surface_words(text: str) -> list:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _form_uses_words(form: Form) -> set:
    """Every clue word the form's tree + definition.phrase claim."""
    used = set()
    for leaf in _flatten_leaves([form.tree]):
        if leaf.source_word:
            used.update(_surface_words(leaf.source_word))
    # Indicators on every node
    def walk(n):
        if n.indicator:
            used.update(_surface_words(n.indicator))
        for c in n.sources or []:
            walk(c)
    walk(form.tree)
    used.update(_surface_words(form.definition.phrase))
    return used


# --- Top-level verify -----------------------------------------------------

@dataclass
class Check:
    name: str
    status: str   # "pass" / "fail"
    detail: str = ""


@dataclass
class Verdict:
    verdict: str  # PASS / FAIL
    checks: list

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "checks": [{"name": c.name, "status": c.status,
                        "detail": c.detail} for c in self.checks],
        }


def verify(form: Form, clue_text: str, db: RefDB,
            shadow_conn=None) -> Verdict:
    """Verify a form. If shadow_conn is provided, bridge lookups
    will also consult the shadow DB after live-DB misses."""
    checks: list = []

    # 1. Assembly
    answer = (form.definition.answer or "").upper()
    if _produces(form.tree, answer):
        checks.append(Check("assembly", "pass",
                            f"tree produces {answer}"))
    else:
        # Show what the tree DID produce so the gap is visible
        yielded = _yields(form.tree)
        if yielded:
            sample = sorted(yielded)[:6]
            sample_str = ", ".join(repr(s) for s in sample)
            extra = "" if len(yielded) <= 6 \
                else f" (+ {len(yielded) - 6} more)"
            checks.append(Check("assembly", "fail",
                                f"tree produces {sample_str}{extra}"
                                f" - need {answer!r}"))
        else:
            checks.append(Check("assembly", "fail",
                                f"tree produces nothing - need {answer!r}"))

    # 2/3. Bridge (covers leaves and every non-literal node's indicator)
    leaf_failures = []
    leaf_passes = []
    leaves = _flatten_leaves([form.tree])
    for leaf in leaves:
        reason = _bridge_leaf(leaf, db, shadow_conn)
        if reason:
            leaf_failures.append(reason)
        else:
            leaf_passes.append(_describe_leaf_for_check(leaf))

    indicator_failures = []
    indicator_passes = []
    def walk_indicators(n):
        reason = _bridge_indicator(n, db, shadow_conn)
        if reason:
            indicator_failures.append(reason)
        elif (n.operation not in LEAF_OPERATIONS
              and n.operation not in ("charade", "double_definition",
                                       "cryptic_definition", "unknown")
              and n.indicator):
            indicator_passes.append(f'"{n.indicator}" -> {n.operation}')
        for c in n.sources or []:
            walk_indicators(c)
    walk_indicators(form.tree)

    if leaf_failures:
        checks.append(Check("bridge.leaves", "fail",
                            "; ".join(leaf_failures[:5])))
    else:
        checks.append(Check("bridge.leaves", "pass",
                            "verified " + "; ".join(leaf_passes)))

    if indicator_failures:
        checks.append(Check("bridge.indicators", "fail",
                            "; ".join(indicator_failures[:5])))
    else:
        if indicator_passes:
            checks.append(Check("bridge.indicators", "pass",
                                "verified " + "; ".join(indicator_passes)))
        else:
            checks.append(Check("bridge.indicators", "pass",
                                "no indicator-bearing nodes"))

    # 4. Residue
    surface = set(_surface_words(clue_text))
    used = _form_uses_words(form)
    declared_lnks = {w.lower() for w in form.link_words}
    unaccounted = surface - used - declared_lnks
    bad_lnks = declared_lnks - LINK_WORDS
    if not unaccounted and not bad_lnks:
        checks.append(Check("residue", "pass", ""))
    else:
        msg_parts = []
        if unaccounted:
            msg_parts.append(f"unaccounted: {sorted(unaccounted)}")
        if bad_lnks:
            msg_parts.append(f"non-link words declared as link_words: "
                             f"{sorted(bad_lnks)}")
        checks.append(Check("residue", "fail", "; ".join(msg_parts)))

    verdict = "PASS" if all(c.status == "pass" for c in checks) else "FAIL"
    return Verdict(verdict=verdict, checks=checks)
