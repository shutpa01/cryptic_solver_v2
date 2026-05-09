"""Tree-aware matcher — production Phase-1 matcher in waiting.

Walks tree-shape catalog signatures (per `catalog_v1.json`) against
clue text and yields candidate `Form` objects, each structurally
plausible. The clipboard verifier is the next step that checks
per-rule details.

Design lives in `TREE_MATCHER_DESIGN.md` (next to this file).

Currently built (this file):
  * leaves:  literal, synonym, abbreviation, homophone, raw, positional
             (positional requires a 2-word span: indicator + source)
  * ops:     charade, deletion, double_definition, anagram, hidden,
             acrostic, reversal, container, homophone (op)
  * link-word slot support: charade, double_definition, anagram,
    hidden, and acrostic allow LINK_WORDS-listed words as gaps between
    children's spans. Gap words are recorded on form.link_words so the
    verifier's residue check accounts for them.

Spoonerism deferred — verifier's _assembles has no spoonerism case.

All `_bind_*` helpers yield (Node, link_words_list) tuples. The
recursive composition merges link_words lists from siblings; the
top-level `match_signature` lifts them onto the Form.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Iterator, List, Optional, Tuple

from signature_solver.solver import extract_definition_candidates
from signature_solver.db import RefDB

from .schema import (
    Form, Definition, Node, lit, syn, charade, deletion,
    abbr, raw, positional, homophone_leaf, double_definition,
    anagram, reversal, hidden, acrostic, container,
)
from .surface import tokenize as _surface_tokenize
from .clipboard_verifier import (
    _positional_extract, POS_KIND_INDICATOR_TYPES, LINK_WORDS,
    _assembles as _verifier_assembles,
)


MAX_BINDINGS = 100

# TODO: target-driven binding. The matcher currently enumerates every
# (synonym × child) combination and then filters by mechanical assembly.
# For deeply-nested charades with multiple synonym leaves this can be
# slow (50^4 = 6.25M combinations per template). The proper fix is to
# pass the answer-target down through the binding so each leaf only
# yields values that could possibly fit the parent's slot — for charade,
# values that are prefixes of the remaining target. Deferred. For now,
# slow per-clue cases are acceptable in the per-puzzle batch model.


def _letters_only(s: str) -> str:
    """Letters-only uppercase. Used to derive literal values from
    clue words that may carry possessive 's or other punctuation."""
    return re.sub(r"[^A-Za-z]", "", s or "").upper()


# --- Public API ----------------------------------------------------------

def match_signature(
    catalog_entry: dict,
    clue_text: str,
    answer: str,
    db: RefDB,
    shadow_conn: Optional[sqlite3.Connection] = None,
    and_lit: bool = False,
) -> List[Form]:
    """Walk the catalog entry's structure tree against the clue.

    Returns a list of structurally-plausible Form objects, capped at
    MAX_BINDINGS. The caller is expected to pass each Form to the
    clipboard verifier; the matcher does not check per-rule details
    or judge correctness.

    When `and_lit=True`, the WHOLE clue is treated as both the
    definition phrase and the wordplay window — every emitted form
    is marked `is_and_lit=True` so the verifier returns PENDING.
    Used as the fallback pass when no standard split-form
    interpretation produces a PASS.
    """
    structure = catalog_entry["structure"]
    answer_u = _letters_only(answer)
    if not answer_u:
        return []

    clue_words = _surface_tokenize(clue_text)
    if not clue_words:
        return []

    syn_cache: dict = {}
    ind_cache: dict = {}

    out: List[Form] = []

    # Special case: top-level double_definition uses the WHOLE clue as
    # the wordplay (and as the definition phrase). Both halves are
    # definitions of the answer, so the standard def+wp split doesn't
    # apply.
    if structure.get("op") == "double_definition":
        for tree_node, link_words in _bind(
                structure, list(clue_words),
                db, shadow_conn, syn_cache, ind_cache):
            if not _assembles_to_answer(tree_node, answer_u):
                continue
            form = Form(
                tree=tree_node,
                definition=Definition(phrase=clue_text, answer=answer_u),
                link_words=list(link_words),
            )
            out.append(form)
            if len(out) >= MAX_BINDINGS:
                return out
        return out

    # &lit mode: skip the standard def+wp split. Treat the whole clue
    # as both wp window and definition phrase. Every form is marked
    # is_and_lit so the verifier returns PENDING.
    if and_lit:
        for tree_node, link_words in _bind(
                structure, list(clue_words),
                db, shadow_conn, syn_cache, ind_cache):
            if not _assembles_to_answer(tree_node, answer_u):
                continue
            form = Form(
                tree=tree_node,
                definition=Definition(phrase=clue_text, answer=answer_u),
                link_words=list(link_words),
                is_and_lit=True,
            )
            out.append(form)
            if len(out) >= MAX_BINDINGS:
                return out
        return out

    for def_phrase, wp_words in extract_definition_candidates(
            clue_words, answer_u, db):
        if not wp_words:
            continue
        for tree_node, link_words in _bind(
                structure, list(wp_words),
                db, shadow_conn, syn_cache, ind_cache):
            if not _assembles_to_answer(tree_node, answer_u):
                continue
            form = Form(
                tree=tree_node,
                definition=Definition(phrase=def_phrase, answer=answer_u),
                link_words=list(link_words),
            )
            out.append(form)
            if len(out) >= MAX_BINDINGS:
                return out
    return out


# --- Recursive binding ---------------------------------------------------

def _bind(node: dict, span: list, db: RefDB,
          shadow_conn, syn_cache: dict, ind_cache: dict
          ) -> Iterator[Tuple[Node, List[str]]]:
    """Bind `node` to the contiguous wordplay `span`. Yields
    (Node, link_words_list) tuples, where link_words is the list of
    LINK_WORDS-listed surface words that this binding (recursively)
    treated as gaps between children's spans."""
    if node.get("leaf"):
        yield from _bind_leaf(node, span, db, syn_cache)
        return
    op = node["op"]
    if op == "charade":
        yield from _bind_charade(node, span, db, shadow_conn,
                                  syn_cache, ind_cache)
    elif op == "deletion":
        yield from _bind_deletion(node, span, db, shadow_conn,
                                   syn_cache, ind_cache)
    elif op == "double_definition":
        yield from _bind_double_definition(node, span, db, shadow_conn,
                                            syn_cache, ind_cache)
    elif op == "anagram":
        yield from _bind_anagram(node, span, db, shadow_conn,
                                  syn_cache, ind_cache)
    elif op == "hidden":
        yield from _bind_hidden(node, span, db, shadow_conn,
                                 syn_cache, ind_cache)
    elif op == "acrostic":
        yield from _bind_acrostic(node, span, db, shadow_conn,
                                   syn_cache, ind_cache)
    elif op == "reversal":
        yield from _bind_reversal(node, span, db, shadow_conn,
                                   syn_cache, ind_cache)
    elif op == "container":
        yield from _bind_container(node, span, db, shadow_conn,
                                    syn_cache, ind_cache)
    elif op == "homophone":
        yield from _bind_homophone_op(node, span, db, shadow_conn,
                                       syn_cache, ind_cache)
    # spoonerism deferred — verifier's _assembles has no spoonerism case yet.


def _bind_leaf(node: dict, span: list, db: RefDB,
                syn_cache: dict
                ) -> Iterator[Tuple[Node, List[str]]]:
    op = node["op"]
    if op == "literal":
        if len(span) != 1:
            return
        word = span[0]
        value = _letters_only(word)
        if not value:
            return
        yield (lit(source_word=word, value=value), [])
        return
    if op == "synonym":
        phrase = " ".join(w.lower() for w in span)
        values = syn_cache.get(phrase)
        if values is None:
            values = list(db.get_synonyms(phrase))
            syn_cache[phrase] = values
        for v in values:
            yield (syn(source_word=phrase, value=v.upper()), [])
        return
    if op == "raw":
        if len(span) != 1:
            return
        word = span[0]
        value = _letters_only(word)
        if not value:
            return
        yield (raw(source_word=word, value=value), [])
        return
    if op == "abbreviation":
        # Per the design: DB requirement is wordplay(category=abbreviation)
        # OR synonyms_pairs — the verifier accepts either, so the matcher
        # proposes values from both.
        phrase = " ".join(w.lower() for w in span)
        cache_key = ("abbr", phrase)
        values = syn_cache.get(cache_key)
        if values is None:
            uniq = set()
            for v in db.get_abbreviations(phrase):
                uniq.add(v.upper())
            for v in db.get_synonyms(phrase):
                uniq.add(v.upper())
            values = sorted(uniq)
            syn_cache[cache_key] = values
        for v in values:
            yield (abbr(source_word=phrase, value=v), [])
        return
    if op == "homophone":
        # Leaf-form homophone (vs op-form, which is a non-leaf).
        phrase = " ".join(w.lower() for w in span)
        cache_key = ("homo", phrase)
        values = syn_cache.get(cache_key)
        if values is None:
            values = sorted({v.upper() for v in db.get_homophones(phrase)})
            syn_cache[cache_key] = values
        for v in values:
            yield (homophone_leaf(source_word=phrase, value=v), [])
        return
    if op == "positional":
        # Positional leaves require a licensing indicator. The leaf
        # takes a 2-word span (indicator + source, like deletion).
        # Non-adjacent positional indicators are a v2 enhancement.
        if len(span) != 2:
            return
        kind = node.get("positional_kind")
        if not kind:
            return
        expected = POS_KIND_INDICATOR_TYPES.get(kind, set())
        for ind_pos in (0, 1):
            ind_word = span[ind_pos]
            src_word = span[1 - ind_pos]
            ind_key = ("ind_types", ind_word.lower())
            types = syn_cache.get(ind_key)
            if types is None:
                types = db.get_indicator_types(ind_word.lower())
                syn_cache[ind_key] = types
            if not any(t[0] in expected for t in types):
                continue
            src_letters = _letters_only(src_word)
            value = _positional_extract(src_letters, kind)
            if not value:
                continue
            leaf = positional(source_word=src_word, value=value, kind=kind)
            leaf.indicator = ind_word
            yield (leaf, [])
        return


def _bind_charade(node: dict, span: list, db: RefDB,
                   shadow_conn, syn_cache: dict, ind_cache: dict
                   ) -> Iterator[Tuple[Node, List[str]]]:
    """Charade: ≥1 children, no indicator. Span is partitioned into
    n contiguous non-empty child sub-spans, with optional LINK_WORDS
    gaps between/around children."""
    children = node["children"]
    n = len(children)
    L = len(span)
    if L < n:
        return
    for sub_spans, own_links in _partition_with_link_gaps(span, n):
        for combo, child_links in _child_product(
                children, sub_spans, db, shadow_conn,
                syn_cache, ind_cache):
            yield (charade(*combo), own_links + child_links)


def _bind_double_definition(node: dict, span: list, db: RefDB,
                              shadow_conn, syn_cache: dict, ind_cache: dict
                              ) -> Iterator[Tuple[Node, List[str]]]:
    """DD: exactly two children, no indicator. Both children must
    independently yield the answer (verifier check). LINK_WORDS gaps
    between halves are allowed."""
    children = node.get("children", [])
    if len(children) != 2 or len(span) < 2:
        return
    for sub_spans, own_links in _partition_with_link_gaps(span, 2):
        for combo, child_links in _child_product(
                children, sub_spans, db, shadow_conn,
                syn_cache, ind_cache):
            yield (double_definition(*combo), own_links + child_links)


def _bind_anagram(node: dict, span: list, db: RefDB,
                    shadow_conn, syn_cache: dict, ind_cache: dict
                    ) -> Iterator[Tuple[Node, List[str]]]:
    """Anagram: ≥1 literal/raw children + 1 indicator at a span
    endpoint. The remaining fodder span is partitioned among children
    with optional LINK_WORDS gaps."""
    children = node.get("children", [])
    n = len(children)
    if n < 1 or len(span) < n + 1:
        return
    expected = {"anagram"}
    for ind_pos in (0, len(span) - 1):
        ind_word = span[ind_pos]
        if not _has_indicator_authority(ind_word, expected, db, ind_cache):
            continue
        fodder_span = span[1:] if ind_pos == 0 else span[:-1]
        for sub_spans, own_links in _partition_with_link_gaps(fodder_span, n):
            for combo, child_links in _child_product(
                    children, sub_spans, db, shadow_conn,
                    syn_cache, ind_cache):
                yield (anagram(*combo, indicator=ind_word),
                        own_links + child_links)


def _bind_hidden(node: dict, span: list, db: RefDB,
                   shadow_conn, syn_cache: dict, ind_cache: dict
                   ) -> Iterator[Tuple[Node, List[str]]]:
    """Hidden: ≥1 literal/raw children whose concatenated letters
    contain the answer; 1 indicator. Span shape mirrors anagram; gaps
    allowed in the fodder area."""
    children = node.get("children", [])
    n = len(children)
    if n < 1 or len(span) < n + 1:
        return
    expected = {"hidden"}
    for ind_pos in (0, len(span) - 1):
        ind_word = span[ind_pos]
        if not _has_indicator_authority(ind_word, expected, db, ind_cache):
            continue
        fodder_span = span[1:] if ind_pos == 0 else span[:-1]
        for sub_spans, own_links in _partition_with_link_gaps(fodder_span, n):
            for combo, child_links in _child_product(
                    children, sub_spans, db, shadow_conn,
                    syn_cache, ind_cache):
                yield (hidden(*combo, indicator=ind_word),
                        own_links + child_links)


def _bind_acrostic(node: dict, span: list, db: RefDB,
                     shadow_conn, syn_cache: dict, ind_cache: dict
                     ) -> Iterator[Tuple[Node, List[str]]]:
    """Acrostic: ≥2 literal/raw children, each contributing one letter
    (first or last per acrostic_kind); 1 indicator at a span endpoint.
    Gaps allowed in the fodder area."""
    children = node.get("children", [])
    n = len(children)
    if n < 2 or len(span) < n + 1:
        return
    kind = node.get("acrostic_kind", "first")
    expected = {"acrostic", "parts"}
    for ind_pos in (0, len(span) - 1):
        ind_word = span[ind_pos]
        if not _has_indicator_authority(ind_word, expected, db, ind_cache):
            continue
        fodder_span = span[1:] if ind_pos == 0 else span[:-1]
        for sub_spans, own_links in _partition_with_link_gaps(fodder_span, n):
            for combo, child_links in _child_product(
                    children, sub_spans, db, shadow_conn,
                    syn_cache, ind_cache):
                yield (acrostic(*combo, indicator=ind_word, kind=kind),
                        own_links + child_links)


def _bind_reversal(node: dict, span: list, db: RefDB,
                     shadow_conn, syn_cache: dict, ind_cache: dict
                     ) -> Iterator[Tuple[Node, List[str]]]:
    """Reversal: exactly 1 child + 1 indicator at a span endpoint.
    No own gaps — child takes the entire remainder of the span."""
    children = node.get("children", [])
    if len(children) != 1 or len(span) < 2:
        return
    expected = {"reversal"}
    for ind_pos in (0, len(span) - 1):
        ind_word = span[ind_pos]
        if not _has_indicator_authority(ind_word, expected, db, ind_cache):
            continue
        child_span = span[1:] if ind_pos == 0 else span[:-1]
        for child_node, child_links in _bind(
                children[0], child_span, db, shadow_conn,
                syn_cache, ind_cache):
            yield (reversal(child_node, indicator=ind_word), child_links)


def _bind_container(node: dict, span: list, db: RefDB,
                      shadow_conn, syn_cache: dict, ind_cache: dict
                      ) -> Iterator[Tuple[Node, List[str]]]:
    """Container: 2 children (outer, inner) with indicator typically
    BETWEEN them (so ind_pos can be any interior position). The clue
    surface order may have outer first or inner first; try both. No
    own gaps in this slice — children take all words on each side of
    the indicator."""
    children = node.get("children", [])
    if len(children) != 2 or len(span) < 3:
        return
    expected = {"container", "insertion"}
    n = len(span)
    for ind_pos in range(1, n - 1):
        ind_word = span[ind_pos]
        if not _has_indicator_authority(ind_word, expected, db, ind_cache):
            continue
        before = span[:ind_pos]
        after = span[ind_pos + 1:]
        for outer_span, inner_span in ((before, after), (after, before)):
            for outer_node, outer_links in _bind(
                    children[0], outer_span, db, shadow_conn,
                    syn_cache, ind_cache):
                for inner_node, inner_links in _bind(
                        children[1], inner_span, db, shadow_conn,
                        syn_cache, ind_cache):
                    yield (container(outer_node, inner_node,
                                      indicator=ind_word),
                            outer_links + inner_links)


def _bind_homophone_op(node: dict, span: list, db: RefDB,
                         shadow_conn, syn_cache: dict, ind_cache: dict
                         ) -> Iterator[Tuple[Node, List[str]]]:
    """Homophone (op form): exactly 1 child whose yielded value sounds
    like the answer; 1 indicator at a span endpoint."""
    children = node.get("children", [])
    if len(children) != 1 or len(span) < 2:
        return
    expected = {"homophone"}
    for ind_pos in (0, len(span) - 1):
        ind_word = span[ind_pos]
        if not _has_indicator_authority(ind_word, expected, db, ind_cache):
            continue
        child_span = span[1:] if ind_pos == 0 else span[:-1]
        for child_node, child_links in _bind(
                children[0], child_span, db, shadow_conn,
                syn_cache, ind_cache):
            yield (Node(operation="homophone", indicator=ind_word,
                          sources=[child_node]),
                    child_links)


def _bind_deletion(node: dict, span: list, db: RefDB,
                    shadow_conn, syn_cache: dict, ind_cache: dict
                    ) -> Iterator[Tuple[Node, List[str]]]:
    """Deletion: exactly 1 child + 1 indicator at a span endpoint."""
    children = node.get("children", [])
    if len(children) != 1 or len(span) < 2:
        return
    kind = node.get("deletion_kind", "tail")
    expected = {"deletion", "parts"}
    for ind_pos in (0, len(span) - 1):
        ind_word = span[ind_pos]
        if not _has_indicator_authority(ind_word, expected, db, ind_cache):
            continue
        child_span = span[1:] if ind_pos == 0 else span[:-1]
        for child_node, child_links in _bind(
                children[0], child_span, db, shadow_conn,
                syn_cache, ind_cache):
            yield (deletion(child_node, indicator=ind_word, kind=kind),
                    child_links)


# --- Helpers -------------------------------------------------------------

def _child_product(children, sub_spans, db, shadow_conn,
                    syn_cache: dict, ind_cache: dict, idx: int = 0,
                    acc_nodes: Optional[list] = None,
                    acc_links: Optional[list] = None
                    ) -> Iterator[Tuple[List[Node], List[str]]]:
    """Cartesian product across children, threading link_words from
    each child's binding into a single accumulated list."""
    if acc_nodes is None:
        acc_nodes = []
    if acc_links is None:
        acc_links = []
    if idx == len(children):
        yield (list(acc_nodes), list(acc_links))
        return
    for child_node, child_links in _bind(
            children[idx], sub_spans[idx], db, shadow_conn,
            syn_cache, ind_cache):
        yield from _child_product(
            children, sub_spans, db, shadow_conn,
            syn_cache, ind_cache,
            idx + 1,
            acc_nodes + [child_node],
            acc_links + list(child_links))


def _has_indicator_authority(word: str, expected: set, db: RefDB,
                              cache: dict) -> bool:
    key = word.lower()
    types = cache.get(key)
    if types is None:
        types = db.get_indicator_types(key)
        cache[key] = types
    return any(t[0] in expected for t in types)


def _partition_with_link_gaps(span: list, n: int):
    """Yield (sub_spans, gap_link_words) where:
      * sub_spans is a list of n non-empty contiguous slices of span,
        in order
      * gap_link_words is the list of span positions NOT in any
        sub_span; every such position must be in LINK_WORDS, otherwise
        the partition is rejected

    The span layout is:  gap_0, child_1, gap_1, child_2, ..., child_n, gap_n
    where each child has size ≥ 1 and each gap has size ≥ 0. This
    enumerates every legal arrangement, then filters out partitions
    whose gaps contain non-link words."""
    L = len(span)
    if L < n:
        return
    slot_count = 2 * n + 1

    def gen_sizes(remaining: int, slot_idx: int):
        if slot_idx == slot_count:
            if remaining == 0:
                yield ()
            return
        is_child = (slot_idx % 2 == 1)
        min_s = 1 if is_child else 0
        for s in range(min_s, remaining + 1):
            for rest in gen_sizes(remaining - s, slot_idx + 1):
                yield (s,) + rest

    for sizes in gen_sizes(L, 0):
        sub_spans = []
        gap_words = []
        pos = 0
        valid = True
        for i, sz in enumerate(sizes):
            if i % 2 == 0:  # gap slot
                for p in range(pos, pos + sz):
                    if span[p].lower() not in LINK_WORDS:
                        valid = False
                        break
                if not valid:
                    break
                gap_words.extend(span[pos:pos + sz])
            else:  # child slot
                sub_spans.append(span[pos:pos + sz])
            pos += sz
        if valid:
            yield (sub_spans, gap_words)


def _ordered_partitions(L: int, n: int):
    """Legacy partitioner with no gap support. Retained for code paths
    that explicitly don't allow link-word gaps."""
    if n == 1:
        yield (L,)
        return
    for first_end in range(1, L - n + 2):
        for rest in _ordered_partitions(L - first_end, n - 1):
            yield (first_end,) + tuple(first_end + r for r in rest)


# --- Pre-filter (delegates to verifier) ----------------------------------

def _assembles_to_answer(tree: Node, answer: str) -> bool:
    """Pre-filter: does the tree mechanically produce the answer?
    Delegates to the verifier's `_assembles` so all op semantics are
    checked exactly the same way the verifier will check them later."""
    return _verifier_assembles(tree, answer)


# --- Canary self-test ----------------------------------------------------

def _canary_test():
    """Two canaries on the charade(deletion[outer](literal),synonym)
    signature:

      * ELITE — the canonical example we built by hand
      * HOWL  — Telegraph 31063 14a, "The wingless bird's screech".
                Tests the apostrophe-preserving tokenisation: 'bird's'
                must be ONE token whose synonym lookup finds OWL via
                the production possessive variant."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from .clipboard_verifier import verify

    db = RefDB()
    catalog_entry = {
        "id": "charade(deletion[outer](literal),synonym)",
        "structure": {
            "op": "charade",
            "children": [
                {"op": "deletion", "deletion_kind": "outer",
                 "children": [{"op": "literal", "leaf": True}]},
                {"op": "synonym", "leaf": True},
            ],
        },
    }
    canaries = [
        ("ELITE", "Cream tea uncovered, with fewer calories?"),
        ("HOWL",  "The wingless bird's screech"),
    ]

    for answer, clue in canaries:
        print(f"\n=== canary: {answer} ===")
        print(f"clue: {clue!r}")
        forms = match_signature(catalog_entry, clue, answer, db)
        print(f"matched {len(forms)} candidate form(s)")
        passes = []
        for f in forms:
            v = verify(f, clue, db)
            if v.verdict == "PASS":
                passes.append((f, v))
        print(f"verifier-PASS: {len(passes)}")
        for i, (f, _) in enumerate(passes[:3], 1):
            print(f"-- PASS {i} --")
            print(f"  def:  {f.definition.phrase!r} -> {f.definition.answer}")
            print(f"  tree: {_compact(f.tree)}")
            if f.link_words:
                print(f"  link: {f.link_words}")


def _compact(n: Node) -> str:
    if not n.sources:
        v = n.value or ""
        s = n.source_word or ""
        if n.operation == "deletion":
            return f"deletion[{n.deletion_kind}]({v})"
        return f"{n.operation}({v} <- {s!r})"
    ind = f" [{n.indicator}]" if n.indicator else ""
    chs = ", ".join(_compact(c) for c in n.sources)
    if n.operation == "deletion":
        return f"deletion[{n.deletion_kind}]{ind}({chs})"
    return f"{n.operation}{ind}({chs})"


if __name__ == "__main__":
    _canary_test()
