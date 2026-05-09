"""Build a form from role-tagged pieces by enumerating compositions.

Input  : Mapping (from word_role_mapper)
Output : a Form whose tree mechanically produces the answer, or None
         with a reason if no composition fits.

The strategy is bounded generate-and-test:

  1. Build leaves from the pieces (synonym/abbreviation/literal,
     wrapped in a deletion if the piece has curly-brace sub-info).
  2. Determine which operations the indicators imply (container,
     anagram, reversal, deletion, hidden).
  3. Try compositions in increasing complexity. The first whose
     mechanical assembly equals the answer wins.

This is independent of the production solver and the catalog — it
relies only on the role-tagged pieces produced by the mapper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Optional

from .schema import (
    Form, Node, Definition, lit, syn, abbr, raw,
    charade, anagram, reversal, container, deletion, hidden,
    double_definition,
)
from .word_role_mapper import Mapping, Tag
from .verifier import LINK_WORDS


@dataclass
class AssemblyResult:
    form: Optional[Form]
    pattern: str           # which composition fired
    notes: list = field(default_factory=list)

    def to_dict(self):
        return {
            "form": self.form.to_dict() if self.form else None,
            "pattern": self.pattern,
            "notes": list(self.notes),
        }


# --- Leaf construction ----------------------------------------------------

def _build_leaf(tag: Tag) -> Node:
    """Build a leaf (or deletion-wrapped leaf) from a piece tag.

    For pieces with curly-brace deletion sub_kind set on a non-positional
    mechanism, special-case `heart` shapes whose kept letters are
    exactly first+last of the source — these are positional[outer]
    extractions, mechanically verifiable without any DB lookup.
    """
    src = " ".join(tag.words)
    val = (tag.value or "").upper()
    mech = tag.mechanism or "synonym"
    if mech == "positional":
        from .schema import positional
        kind = tag.sub_kind or "first"
        return positional(source_word=src, value=val, kind=kind)

    # Detect "outer letters" pattern from heart-style deletion notation:
    # T[erritor]Y, S{tor}E, etc. The piece's value is exactly 2 chars,
    # equal to source's first+last; treat as positional[outer].
    notes_text = " ".join(tag.notes or [])
    if (tag.sub_kind == "heart" and len(val) == 2
            and len(src) >= 3):
        src_letters = "".join(c for c in src.upper() if c.isalpha())
        # If the source contains a multi-letter word whose first & last
        # match val, treat as positional[outer].
        for w in src.split():
            wn = "".join(c for c in w.upper() if c.isalpha())
            if (len(wn) >= 3 and wn[0] == val[0] and wn[-1] == val[-1]):
                from .schema import positional
                return positional(source_word=src, value=val, kind="outer")

    if mech == "abbreviation":
        base = abbr(source_word=src, value=val)
    elif mech == "literal":
        base = lit(source_word=src, value=val)
    elif mech == "raw":
        base = raw(source_word=src, value=val)
    else:
        base = syn(source_word=src, value=val)

    return base


# --- Operations from indicators -------------------------------------------

@dataclass
class Operations:
    container_ind: list  # list of indicator Tags
    anagram_ind: list
    reversal_ind: list
    deletion_ind: list
    hidden_ind: list
    homophone_ind: list
    untagged_ind: list   # indicators with no operation set


def _gather_ops(mapping: Mapping) -> Operations:
    cont, ana, rev, dele, hid, hom, untag = [], [], [], [], [], [], []
    for t in mapping.tags:
        if t.role != "indicator":
            continue
        op = t.operation
        if op == "container":
            cont.append(t)
        elif op == "anagram":
            ana.append(t)
        elif op == "reversal":
            rev.append(t)
        elif op == "deletion":
            dele.append(t)
        elif op == "hidden":
            hid.append(t)
        elif op == "homophone":
            hom.append(t)
        else:
            untag.append(t)
    # Floating ops (bare op-words in blog with no clue-word anchor)
    # become synthetic indicator tags so the enumerator's trial
    # functions still fire. The form will then have indicator=None on
    # the corresponding op node — verifier may flag it but assembly
    # can succeed.
    for op_tuple in getattr(mapping, "floating_ops", []) or []:
        op, sub = op_tuple
        synthetic = Tag(span=(-1, -1), words=[], role="indicator",
                         operation=op, sub_kind=sub,
                         notes=["floating_op (no clue-word anchor)"])
        if op == "container" and not cont:
            cont.append(synthetic)
        elif op == "anagram" and not ana:
            ana.append(synthetic)
        elif op == "reversal" and not rev:
            rev.append(synthetic)
        elif op == "deletion" and not dele:
            dele.append(synthetic)
        elif op == "hidden" and not hid:
            hid.append(synthetic)
        elif op == "homophone" and not hom:
            hom.append(synthetic)
    return Operations(cont, ana, rev, dele, hid, hom, untag)


def _ind_word(tags: list) -> Optional[str]:
    if not tags:
        return None
    if not tags[0].words:
        return None  # floating op — no clue-word anchor
    return " ".join(tags[0].words)


# --- Letter yield helpers (deterministic, no permutation) -----------------

def _yield(node: Node) -> str:
    """Single-value letter yield. For non-leaves, computes deterministically."""
    if node.value is not None:
        return node.value.upper()
    op = node.operation
    if op == "charade":
        return "".join(_yield(c) for c in node.sources)
    if op == "reversal" and node.sources:
        return _yield(node.sources[0])[::-1]
    if op == "deletion" and node.sources:
        src = _yield(node.sources[0])
        kind = node.deletion_kind or "tail"
        if kind == "tail":
            return src[:-1]
        if kind == "head":
            return src[1:]
        if kind == "outer" and len(src) >= 3:
            return src[1:-1]
        if kind == "heart" and len(src) >= 3:
            mid = len(src) // 2
            if len(src) % 2 == 1:
                return src[:mid] + src[mid + 1:]
            return src[:mid - 1] + src[mid + 1:]
    return ""


# --- Composition trials ---------------------------------------------------

def _try_dd(leaves, answer, ops, mapping=None):
    """Double-definition: two synonym leaves, both with value=answer."""
    if mapping is None:
        return None
    if not any(op == "double_definition"
                for op, _ in (mapping.floating_ops or [])):
        return None
    if len(leaves) != 2:
        return None
    if _yield(leaves[0]) != answer or _yield(leaves[1]) != answer:
        return None
    tree = double_definition(leaves[0], leaves[1])
    return AssemblyResult(form=Form(tree=tree, definition=None,
                                       link_words=[]),
                            pattern="double_definition")


def _try_cd(leaves, answer, ops, mapping=None):
    """Cryptic definition: one synonym leaf, value=answer, source=whole clue."""
    if mapping is None:
        return None
    if not any(op == "cryptic_definition"
                for op, _ in (mapping.floating_ops or [])):
        return None
    if len(leaves) != 1:
        return None
    if _yield(leaves[0]) != answer:
        return None
    tree = leaves[0]
    return AssemblyResult(form=Form(tree=tree, definition=None,
                                       link_words=[]),
                            pattern="cryptic_definition")


def _try_single_piece(leaves, answer, ops):
    if len(leaves) != 1:
        return None
    if _yield(leaves[0]) == answer:
        return AssemblyResult(form=None,  # caller fills definition
                                pattern="single_piece")
    return None


def _try_container_anagram(leaves, answer, ops):
    """Compound: container(anagram(X), Y) or container(Y, anagram(X)).

    Fires only when BOTH a container indicator AND an anagram indicator
    are present. Tries every (fodder, inner) split of the leaves.
    Anagram fodder may span multiple leaves (charade of fodders).
    """
    if not ops.container_ind or not ops.anagram_ind:
        return None
    if len(leaves) < 2:
        return None
    cont_ind = _ind_word(ops.container_ind)
    ana_ind = _ind_word(ops.anagram_ind)
    # Compound shape only makes sense if both indicators are
    # anchored to clue words. A floating container indicator (no
    # anchor) means the simple-anagram interpretation is more
    # honest — let _try_anagram handle it.
    if cont_ind is None or ana_ind is None:
        return None
    n = len(leaves)
    # Cap leaves to keep subsets enumeration tractable
    if n > 6:
        return None
    from itertools import combinations as _comb
    answer_sorted = sorted(answer)
    for fsize in range(1, n):
        for fodder_idx in _comb(range(n), fsize):
            fodder_set = set(fodder_idx)
            fodder_leaves = [leaves[i] for i in fodder_idx]
            inner_leaves = [leaves[i] for i in range(n) if i not in fodder_set]
            fodder_letters = "".join(_yield(l) for l in fodder_leaves)
            if not fodder_letters:
                continue
            for inner_perm in permutations(inner_leaves):
                inner_letters = "".join(_yield(l) for l in inner_perm)
                if not inner_letters:
                    continue
                if len(fodder_letters) + len(inner_letters) != len(answer):
                    continue
                # Multiset check: combined letters should match answer
                if sorted(fodder_letters + inner_letters) != answer_sorted:
                    continue
                inner_node = (inner_perm[0] if len(inner_perm) == 1
                               else charade(*inner_perm))
                fodder_sorted = sorted(fodder_letters)
                il = len(inner_letters)
                # Find positions where the answer contains the inner letters
                # contiguously; the surrounding letters must be a permutation
                # of fodder.
                for p in range(1, len(answer) - il):
                    if answer[p:p + il] == inner_letters:
                        outer = answer[:p] + answer[p + il:]
                        if sorted(outer) == fodder_sorted:
                            ana_node = (anagram(*fodder_leaves,
                                                 indicator=ana_ind))
                            tree = container(outer=ana_node,
                                              inner=inner_node,
                                              indicator=cont_ind)
                            return AssemblyResult(
                                form=Form(tree=tree, definition=None,
                                           link_words=[]),
                                pattern="container_anagram")
                # Inverted: anagram(fodder) sits inside inner_letters
                # (e.g. WATERSIDE: anagram(TRADE IS) inside WE)
                fl = len(fodder_letters)
                if il > 1:
                    for q in range(1, len(answer) - fl):
                        # answer[q:q+fl] must be a permutation of fodder
                        if (sorted(answer[q:q + fl]) == fodder_sorted
                            and answer[:q] + answer[q + fl:] == inner_letters):
                            ana_node = (anagram(*fodder_leaves,
                                                 indicator=ana_ind))
                            tree = container(outer=inner_node,
                                              inner=ana_node,
                                              indicator=cont_ind)
                            return AssemblyResult(
                                form=Form(tree=tree, definition=None,
                                           link_words=[]),
                                pattern="container_anagram_inverted")
    return None


def _try_anagram(leaves, answer, ops):
    if not ops.anagram_ind:
        return None
    letters = "".join(_yield(l) for l in leaves)
    if sorted(letters) != sorted(answer):
        return None
    ind = _ind_word(ops.anagram_ind)
    tree = anagram(*leaves, indicator=ind)
    return AssemblyResult(form=Form(tree=tree, definition=None,
                                       link_words=[]),
                            pattern="anagram")


def _try_charade(leaves, answer, ops):
    if len(leaves) < 2:
        return None
    n = len(leaves)
    if n > 6:
        return None  # too many to enumerate
    for perm in permutations(leaves):
        if "".join(_yield(l) for l in perm) == answer:
            tree = charade(*perm)
            return AssemblyResult(form=Form(tree=tree, definition=None,
                                              link_words=[]),
                                   pattern="charade")
    return None


def _try_container(leaves, answer, ops):
    if not ops.container_ind:
        return None
    if len(leaves) < 2:
        return None
    ind = _ind_word(ops.container_ind)
    n = len(leaves)
    # Pick (outer, inner) from leaves. With 2 leaves, just two
    # orderings. With 3+, container wraps two of them; the rest
    # charade alongside.
    if n == 2:
        for outer, inner in [(leaves[0], leaves[1]),
                              (leaves[1], leaves[0])]:
            ov, iv = _yield(outer), _yield(inner)
            if not ov or not iv or len(ov) + len(iv) != len(answer):
                continue
            for p in range(1, len(ov)):
                if ov[:p] + iv + ov[p:] == answer:
                    tree = container(outer=outer, inner=inner,
                                      indicator=ind)
                    return AssemblyResult(form=Form(tree=tree,
                                                      definition=None,
                                                      link_words=[]),
                                            pattern="container")
        return None
    # n >= 3: pick 2 leaves for container, charade the rest
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            outer, inner = leaves[i], leaves[j]
            ov, iv = _yield(outer), _yield(inner)
            if not ov or not iv:
                continue
            # All possible insertion positions
            con_results = []
            for p in range(1, len(ov)):
                con_results.append(ov[:p] + iv + ov[p:])
            rest = [leaves[k] for k in range(n) if k != i and k != j]
            for con_letters in con_results:
                # Charade with rest
                if not rest:
                    if con_letters == answer:
                        tree = container(outer=outer, inner=inner,
                                          indicator=ind)
                        return AssemblyResult(
                            form=Form(tree=tree, definition=None,
                                       link_words=[]),
                            pattern="container")
                    continue
                # Build a container node and charade with rest in some order
                con_node = container(outer=outer, inner=inner,
                                      indicator=ind)
                for perm in permutations([con_node] + rest):
                    parts = []
                    for n2 in perm:
                        if n2 is con_node:
                            parts.append(con_letters)
                        else:
                            parts.append(_yield(n2))
                    if "".join(parts) == answer:
                        tree = charade(*perm)
                        return AssemblyResult(
                            form=Form(tree=tree, definition=None,
                                       link_words=[]),
                            pattern="container_charade")
    # n == 3: also try (1 outer, container(inner1, inner2) as inner) —
    # covers PALIMPSEST-shaped double-nested containers (inner pair
    # is itself a container, then wrapped by outer).
    if n == 3 and len(ops.container_ind) >= 1:
        for i in range(n):
            outer = leaves[i]
            ov = _yield(outer)
            if not ov:
                continue
            inner_pieces = [leaves[k] for k in range(n) if k != i]
            for inner_perm in permutations(inner_pieces):
                a_node, b_node = inner_perm
                a, b = _yield(a_node), _yield(b_node)
                if not a or not b:
                    continue
                # Container of (a, b) at every position
                for ip in range(1, len(a)):
                    inner_letters = a[:ip] + b + a[ip:]
                    if len(ov) + len(inner_letters) != len(answer):
                        continue
                    for p in range(1, len(ov)):
                        if ov[:p] + inner_letters + ov[p:] == answer:
                            inner_node = container(outer=a_node,
                                                     inner=b_node,
                                                     indicator=ind)
                            tree = container(outer=outer,
                                              inner=inner_node,
                                              indicator=ind)
                            return AssemblyResult(
                                form=Form(tree=tree, definition=None,
                                            link_words=[]),
                                pattern="container_double_nested")
    # n == 3: also try (1 outer, 2-piece-charade as inner) — covers
    # SQUADRON-shaped clues where two pieces (QUAD + R) both go inside
    # the third (SON).
    if n == 3:
        for i in range(n):
            outer = leaves[i]
            ov = _yield(outer)
            if not ov:
                continue
            inner_pieces = [leaves[k] for k in range(n) if k != i]
            for inner_perm in permutations(inner_pieces):
                iv = "".join(_yield(p) for p in inner_perm)
                if not iv:
                    continue
                if len(ov) + len(iv) != len(answer):
                    continue
                for p in range(1, len(ov)):
                    if ov[:p] + iv + ov[p:] == answer:
                        inner_node = charade(*inner_perm)
                        tree = container(outer=outer,
                                          inner=inner_node,
                                          indicator=ind)
                        return AssemblyResult(
                            form=Form(tree=tree, definition=None,
                                       link_words=[]),
                            pattern="container_2inner")
    # n == 3: also try (2-piece-charade as outer, 1 leaf as inner) —
    # covers SNIPE-shaped clues where two pieces (S + E) wrap the
    # third (NIP).
    if n == 3:
        for i in range(n):
            inner_leaf = leaves[i]
            iv = _yield(inner_leaf)
            if not iv:
                continue
            outer_pieces = [leaves[k] for k in range(n) if k != i]
            for outer_perm in permutations(outer_pieces):
                ov = "".join(_yield(p) for p in outer_perm)
                if not ov:
                    continue
                if len(ov) + len(iv) != len(answer):
                    continue
                for p in range(1, len(ov)):
                    if ov[:p] + iv + ov[p:] == answer:
                        outer_node = charade(*outer_perm)
                        tree = container(outer=outer_node,
                                          inner=inner_leaf,
                                          indicator=ind)
                        return AssemblyResult(
                            form=Form(tree=tree, definition=None,
                                       link_words=[]),
                            pattern="container_2outer")
    # General n >= 3: pick disjoint subsets for outer + inner, charade
    # the rest around the container. Covers HINDU-style clues where 2
    # outer pieces wrap 1 inner piece + 1 trailing piece.
    if n >= 3 and n <= 6:
        from itertools import combinations as _comb
        idxs = list(range(n))
        for outer_size in range(1, n - 1):
            for outer_idx in _comb(idxs, outer_size):
                outer_set = set(outer_idx)
                rest_idxs = [k for k in idxs if k not in outer_set]
                for inner_size in range(1, len(rest_idxs)):
                    for inner_idx in _comb(rest_idxs, inner_size):
                        inner_set = set(inner_idx)
                        leftover_idx = [k for k in rest_idxs
                                          if k not in inner_set]
                        outer_pieces = [leaves[k] for k in outer_idx]
                        inner_pieces = [leaves[k] for k in inner_idx]
                        leftover_pieces = [leaves[k] for k in leftover_idx]
                        for outer_perm in permutations(outer_pieces):
                            ov = "".join(_yield(p) for p in outer_perm)
                            if not ov:
                                continue
                            for inner_perm in permutations(inner_pieces):
                                iv = "".join(_yield(p) for p in inner_perm)
                                if not iv:
                                    continue
                                container_letter_options = []
                                for p in range(1, len(ov)):
                                    container_letter_options.append(
                                        ov[:p] + iv + ov[p:])
                                if not leftover_pieces:
                                    if any(c == answer for c
                                            in container_letter_options):
                                        outer_node = (outer_perm[0]
                                            if len(outer_perm) == 1
                                            else charade(*outer_perm))
                                        inner_node = (inner_perm[0]
                                            if len(inner_perm) == 1
                                            else charade(*inner_perm))
                                        tree = container(outer=outer_node,
                                                          inner=inner_node,
                                                          indicator=ind)
                                        return AssemblyResult(
                                            form=Form(tree=tree,
                                                        definition=None,
                                                        link_words=[]),
                                            pattern="container_general")
                                    continue
                                # Charade container with leftovers in any
                                # order
                                outer_node = (outer_perm[0]
                                    if len(outer_perm) == 1
                                    else charade(*outer_perm))
                                inner_node = (inner_perm[0]
                                    if len(inner_perm) == 1
                                    else charade(*inner_perm))
                                con_node = container(outer=outer_node,
                                                      inner=inner_node,
                                                      indicator=ind)
                                for con_letters in container_letter_options:
                                    for perm in permutations(
                                            [con_node] + leftover_pieces):
                                        parts = []
                                        for n2 in perm:
                                            if n2 is con_node:
                                                parts.append(con_letters)
                                            else:
                                                parts.append(_yield(n2))
                                        if "".join(parts) == answer:
                                            tree = charade(*perm)
                                            return AssemblyResult(
                                                form=Form(tree=tree,
                                                            definition=None,
                                                            link_words=[]),
                                                pattern="container_general")
    return None


def _try_reversal(leaves, answer, ops):
    if not ops.reversal_ind:
        return None
    ind = _ind_word(ops.reversal_ind)
    n = len(leaves)
    if n == 1:
        if _yield(leaves[0])[::-1] == answer:
            tree = reversal(leaves[0], indicator=ind)
            return AssemblyResult(form=Form(tree=tree, definition=None,
                                              link_words=[]),
                                   pattern="reversal")
        return None
    # n >= 2: try reversing each leaf in turn within a charade
    if n > 5:
        return None
    for rev_idx in range(n):
        rev_node = reversal(leaves[rev_idx], indicator=ind)
        rest = [leaves[k] for k in range(n) if k != rev_idx]
        for perm in permutations([rev_node] + rest):
            parts = []
            for nd in perm:
                parts.append(_yield(nd))
            if "".join(parts) == answer:
                tree = charade(*perm) if len(perm) > 1 else perm[0]
                return AssemblyResult(
                    form=Form(tree=tree, definition=None,
                               link_words=[]),
                    pattern="reversal_charade")
    # Or reverse the whole charade
    for perm in permutations(leaves):
        full = "".join(_yield(l) for l in perm)
        if full[::-1] == answer:
            tree = reversal(charade(*perm), indicator=ind)
            return AssemblyResult(form=Form(tree=tree, definition=None,
                                              link_words=[]),
                                   pattern="full_reversal")
    return None


def _try_container_reversal(leaves, answer, ops):
    """Compound: container with reversal applied to outer or inner.

    Shape A: container(reversal(outer), inner)  — outer reversed,
             then wraps inner.
    Shape B: container(outer, reversal(inner))  — outer wraps inner
             which has been reversed.

    Fires only when BOTH a reversal indicator AND a container indicator
    are present and anchored. Distinct from `_try_reversal_container`
    (which puts the reversal AROUND the container). Different shape.
    """
    if not ops.reversal_ind or not ops.container_ind:
        return None
    if len(leaves) < 2:
        return None
    rev_ind = _ind_word(ops.reversal_ind)
    cont_ind = _ind_word(ops.container_ind)
    if rev_ind is None or cont_ind is None:
        return None
    n = len(leaves)
    if n > 5:
        return None
    from itertools import combinations as _comb
    for inner_size in range(1, n):
        for inner_idx in _comb(range(n), inner_size):
            inner_set = set(inner_idx)
            inner_leaves = [leaves[i] for i in inner_idx]
            outer_leaves = [leaves[i] for i in range(n) if i not in inner_set]
            if not outer_leaves or not inner_leaves:
                continue
            for inner_perm in permutations(inner_leaves):
                inner_letters = "".join(_yield(l) for l in inner_perm)
                if not inner_letters:
                    continue
                inner_node_plain = (inner_perm[0] if len(inner_perm) == 1
                                     else charade(*inner_perm))
                inner_letters_rev = inner_letters[::-1]
                il = len(inner_letters)
                for outer_perm in permutations(outer_leaves):
                    outer_letters = "".join(_yield(l) for l in outer_perm)
                    if not outer_letters:
                        continue
                    if len(outer_letters) + il != len(answer):
                        continue
                    outer_node_plain = (outer_perm[0]
                                          if len(outer_perm) == 1
                                          else charade(*outer_perm))
                    outer_letters_rev = outer_letters[::-1]
                    # Shape A: container(reversal(outer), inner)
                    for p in range(1, len(outer_letters_rev)):
                        if (outer_letters_rev[:p] + inner_letters
                            + outer_letters_rev[p:]) == answer:
                            rev_outer = reversal(outer_node_plain,
                                                   indicator=rev_ind)
                            tree = container(outer=rev_outer,
                                              inner=inner_node_plain,
                                              indicator=cont_ind)
                            return AssemblyResult(
                                form=Form(tree=tree, definition=None,
                                            link_words=[]),
                                pattern="container_reversal_outer")
                    # Shape B: container(outer, reversal(inner))
                    for p in range(1, len(outer_letters)):
                        if (outer_letters[:p] + inner_letters_rev
                            + outer_letters[p:]) == answer:
                            rev_inner = reversal(inner_node_plain,
                                                   indicator=rev_ind)
                            tree = container(outer=outer_node_plain,
                                              inner=rev_inner,
                                              indicator=cont_ind)
                            return AssemblyResult(
                                form=Form(tree=tree, definition=None,
                                            link_words=[]),
                                pattern="container_reversal_inner")
    return None


def _try_reversal_container(leaves, answer, ops):
    """Compound: reversal(container(...)) — outer reversal wrapping a
    container. Fires only when BOTH a reversal indicator AND a
    container indicator are present in the clue, both anchored.

    Pattern: outer of the container can be a single leaf or a charade
    of multiple leaves; inner of the container is the remaining leaf(es).
    """
    if not ops.reversal_ind or not ops.container_ind:
        return None
    if len(leaves) < 2:
        return None
    rev_ind = _ind_word(ops.reversal_ind)
    cont_ind = _ind_word(ops.container_ind)
    if rev_ind is None or cont_ind is None:
        return None
    n = len(leaves)
    if n > 5:
        return None
    rev_answer = answer[::-1]
    from itertools import combinations as _comb
    for inner_size in range(1, n):
        for inner_idx in _comb(range(n), inner_size):
            inner_set = set(inner_idx)
            inner_leaves = [leaves[i] for i in inner_idx]
            outer_leaves = [leaves[i] for i in range(n) if i not in inner_set]
            if not outer_leaves or not inner_leaves:
                continue
            for inner_perm in permutations(inner_leaves):
                inner_letters = "".join(_yield(l) for l in inner_perm)
                if not inner_letters:
                    continue
                inner_node = (inner_perm[0] if len(inner_perm) == 1
                                else charade(*inner_perm))
                il = len(inner_letters)
                for outer_perm in permutations(outer_leaves):
                    outer_letters = "".join(_yield(l) for l in outer_perm)
                    if not outer_letters:
                        continue
                    if len(outer_letters) + il != len(answer):
                        continue
                    # container(outer_perm, inner) at every internal position
                    # of outer_letters; check if reversed result equals answer
                    for p in range(1, len(outer_letters)):
                        if (outer_letters[:p] + inner_letters
                            + outer_letters[p:]) == rev_answer:
                            outer_node = (outer_perm[0]
                                            if len(outer_perm) == 1
                                            else charade(*outer_perm))
                            cont_node = container(outer=outer_node,
                                                    inner=inner_node,
                                                    indicator=cont_ind)
                            tree = reversal(cont_node, indicator=rev_ind)
                            return AssemblyResult(
                                form=Form(tree=tree, definition=None,
                                            link_words=[]),
                                pattern="reversal_container")
    return None


def _try_homophone(leaves, answer, ops):
    """homophone of single piece (or charade of pieces)."""
    if not ops.homophone_ind:
        return None
    if not leaves:
        return None
    ind = _ind_word(ops.homophone_ind)
    from .schema import Node
    if len(leaves) == 1:
        # Single piece: homophone(piece). Letters likely don't match
        # target literally; that's the whole point.
        tree = Node(operation="homophone", indicator=ind,
                     sources=[leaves[0]])
        return AssemblyResult(form=Form(tree=tree, definition=None,
                                          link_words=[]),
                                pattern="homophone")
    # Multi-piece: homophone(charade(pieces))
    if len(leaves) > 4:
        return None
    inner = charade(*leaves)
    tree = Node(operation="homophone", indicator=ind, sources=[inner])
    return AssemblyResult(form=Form(tree=tree, definition=None,
                                      link_words=[]),
                            pattern="homophone_charade")


def _try_hidden(leaves, answer, ops):
    if not ops.hidden_ind:
        return None
    if not leaves:
        return None
    text = "".join(_yield(l) for l in leaves)
    if answer in text:
        ind = _ind_word(ops.hidden_ind)
        tree = hidden(*leaves, indicator=ind)
        return AssemblyResult(form=Form(tree=tree, definition=None,
                                          link_words=[]),
                                pattern="hidden")
    return None


# --- Top-level entry ------------------------------------------------------

_TRIALS = [
    _try_single_piece,
    _try_container_anagram,
    _try_reversal_container,
    _try_container_reversal,
    _try_anagram,
    _try_charade,
    _try_container,
    _try_reversal,
    _try_homophone,
    _try_hidden,
]


def assemble(mapping: Mapping) -> AssemblyResult:
    """Try to build a form from the mapping's pieces. Returns the first
    composition whose assembly equals the answer.
    """
    pieces = [t for t in mapping.tags if t.role == "piece"]
    if not pieces:
        return AssemblyResult(form=None, pattern="",
                                notes=["no pieces tagged"])
    leaves = [_build_leaf(p) for p in pieces]
    ops = _gather_ops(mapping)
    answer = mapping.answer

    # CD trial first (cryptic-definition — single piece, no wordplay)
    cd_result = _try_cd(leaves, answer, ops, mapping=mapping)
    if cd_result is not None and cd_result.form is not None:
        def_tags = [t for t in mapping.tags if t.role == "definition"]
        def_phrase = " ".join(
            w for t in def_tags for w in t.words) if def_tags else \
            " ".join(mapping.clue_words)
        cd_result.form.definition = Definition(phrase=def_phrase,
                                                  answer=answer)
        cd_result.form.link_words = []
        return cd_result

    # DD trial first (special — needs mapping context)
    dd_result = _try_dd(leaves, answer, ops, mapping=mapping)
    if dd_result is not None and dd_result.form is not None:
        # Fill in definition + link words
        def_tags = [t for t in mapping.tags if t.role == "definition"]
        def_phrase = " ".join(
            w for t in def_tags for w in t.words) if def_tags else ""
        dd_result.form.definition = Definition(phrase=def_phrase,
                                                  answer=answer)
        dd_link_words = []
        for t in mapping.tags:
            if t.role in ("link", "unaccounted"):
                for w in t.words:
                    if w.lower() in LINK_WORDS:
                        dd_link_words.append(w)
        dd_result.form.link_words = dd_link_words
        return dd_result

    # Run every trial; pick the result whose form covers the most
    # tagged indicators. This avoids the "first match wins" trap
    # where a simple charade preempts a compound shape that uses
    # an additional tagged indicator (SNIPE, FRESCO, etc.).
    indicator_words = set()
    for t in mapping.tags:
        if t.role == "indicator":
            for w in t.words:
                indicator_words.add(w.lower())

    candidates = []
    for trial in _TRIALS:
        result = trial(leaves, answer, ops)
        if result is None:
            continue
        # _try_single_piece returns form=None by design (the caller
        # builds the form from the single leaf). Reconstruct here so
        # the candidate participates in coverage-based selection.
        if result.form is None and result.pattern == "single_piece":
            result = AssemblyResult(
                form=Form(tree=leaves[0], definition=None,
                            link_words=[]),
                pattern="single_piece")
        if result.form is None:
            continue
        # Count how many indicator words appear in the form's tree
        used_inds = set()
        def _walk(n):
            if n.indicator:
                for w in n.indicator.split():
                    used_inds.add(w.lower())
            for c in n.sources or []:
                _walk(c)
        _walk(result.form.tree)
        coverage = len(used_inds & indicator_words)
        candidates.append((coverage, len(_TRIALS) - _TRIALS.index(trial),
                            result))

    if candidates:
        # Highest coverage wins; tie-break by trial earlier in the list
        # (more specific shapes registered first).
        candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
        result = candidates[0][2]
        def_tags = [t for t in mapping.tags if t.role == "definition"]
        def_phrase = " ".join(
            w for t in def_tags for w in t.words) if def_tags else ""
        result.form.definition = Definition(phrase=def_phrase,
                                              answer=answer)
        link_words = []
        for t in mapping.tags:
            if t.role in ("link", "unaccounted"):
                for w in t.words:
                    if w.lower() in LINK_WORDS:
                        link_words.append(w)
        result.form.link_words = link_words
        # Attach unused tagged indicators to matching leaves/nodes.
        # Positional/acrostic leaves can carry an indicator; if the
        # mapping has a tagged positional/acrostic indicator that
        # didn't make it into the form, attach it to a matching leaf.
        # This stops the residue check flagging them as unaccounted.
        used_ind_text = set()
        def _collect(n):
            if n.indicator:
                used_ind_text.add(n.indicator)
            for c in n.sources or []:
                _collect(c)
        _collect(result.form.tree)
        unused_indicators = []
        for t in mapping.tags:
            if t.role != "indicator":
                continue
            ind_text = " ".join(t.words)
            if ind_text in used_ind_text:
                continue
            unused_indicators.append((t.operation, ind_text))
        if unused_indicators:
            def _walk_attach(n):
                if n.operation == "positional" and n.indicator is None:
                    for op, txt in unused_indicators:
                        if op == "positional":
                            n.indicator = txt
                            unused_indicators.remove((op, txt))
                            return True
                if n.operation == "acrostic" and n.indicator is None:
                    for op, txt in unused_indicators:
                        if op == "acrostic":
                            n.indicator = txt
                            unused_indicators.remove((op, txt))
                            return True
                for c in n.sources or []:
                    if _walk_attach(c):
                        return True
                return False
            while unused_indicators:
                if not _walk_attach(result.form.tree):
                    break
        return result

    # Nothing fit — surface the shape so we can see what's missing
    return AssemblyResult(
        form=None, pattern="",
        notes=[f"{len(pieces)} pieces, ops=("
               f"con={len(ops.container_ind)},"
               f"ana={len(ops.anagram_ind)},"
               f"rev={len(ops.reversal_ind)},"
               f"del={len(ops.deletion_ind)},"
               f"hid={len(ops.hidden_ind)},"
               f"untag={len(ops.untagged_ind)})"
               f" — no composition produced answer"])
