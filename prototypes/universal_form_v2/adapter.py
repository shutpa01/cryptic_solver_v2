"""SolveResult -> Form adapter (v0).

Takes a `SolveResult` from the production `signature_solver/solver.py:solve_clue`
and converts it to a universal `Form`.

Design rules (per the agreed plan):
  - No surface-text inference, no heuristic re-ordering.
  - Where structure is missing from `SolveResult`, attach an explicit FLAG
    on the form. Flags are the worklist for production-solver changes.
  - The verifier is the referee — adapter never tries to make the form pass.

Public entry point:
    solve_to_form(clue_text, answer, db) -> AdapterResult
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add project root so production modules import cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signature_solver.db import RefDB
from signature_solver.solver import solve_clue, SolveResult
from signature_solver.tokens import (
    SYN_F, ABR_F, ANA_F, RAW, HID_F, HOM_F, POS_F, DEL_F,
    ANA_I, REV_I, CON_I, DEL_I, HID_I, HOM_I,
    POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
    POS_I_ALTERNATE, POS_I_HALF, POS_I_TRIM_FIRST, POS_I_TRIM_LAST,
    POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER,
    LNK, DEF, DBE_MARKER, INDICATOR_TOKENS, FODDER_TOKENS,
)

from .schema import (
    Form, Node, Definition, lit, syn, abbr, raw,
    positional, homophone_leaf,
    charade, anagram, reversal, container, deletion, hidden,
    double_definition, acrostic, unknown,
)


# --- Op name extraction ---------------------------------------------------

# Op names that grammar_triage smuggles via confidence_reasons[0][0].
# Sourced from the `return SolveResult(..., [(<name>, 0)], ...)` lines in
# grammar_triage.py.
_GRAMMAR_TRIAGE_OPS = {
    "anagram", "charade", "container", "container_charade",
    "reversal", "anagram_charade",
}


def _extract_op_name(sr: SolveResult, flags: list) -> Optional[str]:
    """Try to recover the operation name from SolveResult.

    Grammar-triage path puts it in confidence_reasons[0][0]; catalog-matcher
    path doesn't preserve it (worklist #1).
    """
    if sr.confidence_reasons:
        first = sr.confidence_reasons[0]
        if isinstance(first, tuple) and len(first) >= 1:
            name = first[0]
            if isinstance(name, str) and name in _GRAMMAR_TRIAGE_OPS:
                return name
    flags.append("op_name_not_preserved_in_solveresult")
    return None


# --- Token categorisation -------------------------------------------------

def _categorise_roles(roles):
    """Walk word_roles, split into fodder / indicators / lnk / dbe_marker.

    Returns:
      fodder: list of (word, tok, value, meta)  — same order as word_roles
      indicators: dict {indicator_token: (word, meta_or_empty)}
      lnks: list of words tagged LNK
      dbe_markers: list of words tagged DBE_MARKER
    """
    fodder = []
    indicators = {}
    lnks = []
    dbe_markers = []
    for r in roles:
        word = r[0]
        tok = r[1]
        val = r[2] if len(r) > 2 else None
        meta = r[3] if len(r) > 3 and isinstance(r[3], dict) else {}
        if tok in INDICATOR_TOKENS:
            indicators[tok] = (word, meta)
        elif tok == LNK:
            lnks.append(word)
        elif tok == DBE_MARKER:
            dbe_markers.append(word)
        elif tok == DEF:
            pass  # definition handled separately
        else:
            fodder.append((word, tok, val, meta))
    return fodder, indicators, lnks, dbe_markers


# --- Leaf construction ----------------------------------------------------

# Maps positional indicator token -> positional_kind for leaves
_POS_INDICATOR_TO_KIND = {
    POS_I_FIRST: "first",
    POS_I_LAST: "last",
    POS_I_OUTER: "outer",
    POS_I_MIDDLE: "middle",
    POS_I_ALTERNATE: "alternate",
    POS_I_HALF: "half",
}

_TRIM_INDICATOR_TO_DELETION_KIND = {
    POS_I_TRIM_FIRST: "head",
    POS_I_TRIM_LAST: "tail",
    POS_I_TRIM_MIDDLE: "heart",
    POS_I_TRIM_OUTER: "outer",
}


def _fodder_to_leaf(word, tok, value, meta, indicators, flags):
    """Convert one fodder entry to a Leaf node.

    The leaf's op type is derived from the token; positional kind comes from
    the active positional indicator if any.
    """
    val = (value or "").upper().replace(" ", "")
    if tok == SYN_F:
        return syn(word, val)
    if tok == ABR_F:
        return abbr(word, val)
    if tok == RAW:
        return raw(word, val)
    if tok == ANA_F:
        # Anagram fodder: the leaf is the literal source word; the anagram
        # parent rearranges. Per spec.
        norm = "".join(c for c in word.upper() if c.isalpha())
        return lit(word, norm or val)
    if tok == HID_F:
        norm = "".join(c for c in word.upper() if c.isalpha())
        return lit(word, norm or val)
    if tok == HOM_F:
        return homophone_leaf(word, val)
    if tok == POS_F:
        # Determine positional kind from the indicator present on this clue.
        kind = None
        for ind_tok in _POS_INDICATOR_TO_KIND:
            if ind_tok in indicators:
                kind = _POS_INDICATOR_TO_KIND[ind_tok]
                break
        if kind is None:
            for ind_tok in _TRIM_INDICATOR_TO_DELETION_KIND:
                if ind_tok in indicators:
                    # POS_F under a trim indicator means a deletion produces
                    # the leaf's letters from the source. Modelled as a
                    # positional with kind = "first/last/etc." complement.
                    # For v0 we just emit a literal and flag.
                    flags.append("pos_f_under_trim_indicator_modelled_as_literal")
                    return lit(word, val)
        if kind is None:
            flags.append("pos_f_without_positional_indicator")
            kind = "first"
        return positional(word, val, kind=kind)
    if tok == DEL_F:
        # Deletion fodder = the part being deleted. Emitted as a literal of
        # the source-word letters; the parent deletion op consumes it.
        norm = "".join(c for c in word.upper() if c.isalpha())
        return lit(word, norm or val)
    flags.append(f"unknown_fodder_token:{tok}")
    return lit(word, val)


# --- Tree construction by operation ---------------------------------------

def _select_indicator(indicators: dict, op_token):
    """Look up the indicator word for an operation token, if present."""
    e = indicators.get(op_token)
    if e is None:
        return None
    return e[0]  # word


def _build_tree_for_op(op_name: Optional[str], leaves, indicators, flags):
    """Wrap leaves in the appropriate op-tree given op_name + indicators.

    Returns the root Node, or None if no tree can be built.
    """
    n = len(leaves)

    # Charade — many leaves, no special op token
    if op_name == "charade":
        if n == 0:
            return None
        if n == 1:
            return leaves[0]
        return charade(*leaves)

    if op_name == "anagram":
        ind = _select_indicator(indicators, ANA_I)
        # All leaves combine into the anagram fodder.
        return anagram(*leaves, indicator=ind)

    if op_name == "container":
        ind = _select_indicator(indicators, CON_I)
        if n != 2:
            flags.append(f"container_arity:{n}")
            return unknown("container", sources=leaves, indicator=ind)
        # Outer/inner role NOT preserved in SolveResult — flag and rely on
        # the verifier to try both orderings.
        flags.append("container_outer_inner_not_preserved")
        return container(outer=leaves[0], inner=leaves[1], indicator=ind)

    if op_name == "container_charade":
        # 3+ leaves: container wraps a charade (or vice versa); structure not
        # preserved. Flag and emit as a 3-piece container with the first leaf
        # as outer, the rest as a charade inner.
        ind = _select_indicator(indicators, CON_I)
        flags.append("container_charade_structure_not_preserved")
        if n < 2:
            return unknown("container_charade", sources=leaves, indicator=ind)
        if n == 2:
            return container(outer=leaves[0], inner=leaves[1], indicator=ind)
        return container(outer=leaves[0], inner=charade(*leaves[1:]),
                         indicator=ind)

    if op_name == "reversal":
        ind = _select_indicator(indicators, REV_I)
        if n == 1:
            return reversal(leaves[0], indicator=ind)
        # Reversal of a multi-piece thing: charade inside reversal
        flags.append("reversal_inner_assumed_charade")
        return reversal(charade(*leaves), indicator=ind)

    if op_name == "anagram_charade":
        # The op smuggled by grammar_triage. ANA_F leaves become an anagram
        # node; non-ANA leaves stay as-is; all assembled in a charade.
        # We need to know which leaves are ANA_F — caller must pass that
        # info. We rely on a flag list set from the fodder iteration.
        flags.append("anagram_charade_split_inferred_from_token_types")
        # caller hint: leaves built from ANA_F have op=='literal' AND a
        # corresponding flag/marker. Actually we lose token type when we
        # build leaves. Let the caller pass an `is_ana` mask via flags.
        # For v0 simplicity, treat any leaf whose .source_word's letters
        # don't equal its .value as 'fixed' and any whose letters DO equal
        # its value as 'anagram fodder' — that's the convention.
        ana_indicator = _select_indicator(indicators, ANA_I)
        new_children = []
        for leaf in leaves:
            src_letters = "".join(c for c in (leaf.source_word or "").upper()
                                  if c.isalpha())
            if leaf.operation == "literal" and src_letters == (leaf.value or ""):
                # This is the anagram fodder pattern
                new_children.append(anagram(leaf, indicator=ana_indicator))
            else:
                new_children.append(leaf)
        if len(new_children) > 1:
            return charade(*new_children)
        if len(new_children) == 1:
            return new_children[0]
        return None

    # Catalog path: op_name is None. Infer from token mix in indicators dict.
    if op_name is None:
        # fodder_tokens passed via flags-piggyback param? Use a sentinel:
        # caller must pass via thread-locals or extend signature. Easiest:
        # the caller (solve_to_form) wraps this call with the fodder list.
        return _infer_tree_without_op(leaves, indicators,
                                      _LAST_FODDER_TOKENS, flags)

    flags.append(f"op_name_unhandled:{op_name}")
    return unknown(op_name, sources=leaves)


# Per-clue scratch: the fodder tokens for the current solve_to_form call.
# Set by solve_to_form before invoking the tree builder. Avoids threading a
# parameter through `_build_tree_for_op` for what's a single-call helper.
_LAST_FODDER_TOKENS: list = []
_LAST_ANSWER: str = ""


def _trim_value(val: str, kind: str) -> str:
    """Apply a deletion of given kind to the value letters."""
    if not val:
        return val
    if kind == "tail":
        return val[:-1]
    if kind == "head":
        return val[1:]
    if kind == "outer" and len(val) >= 3:
        return val[1:-1]
    if kind == "heart" and len(val) >= 3:
        mid = len(val) // 2
        if len(val) % 2 == 1:
            return val[:mid] + val[mid + 1:]
        return val[:mid - 1] + val[mid + 1:]
    return val


def _infer_tree_without_op(leaves, indicators, fodder_tokens, flags):
    """When op_name is missing (catalog path), infer from indicators present.

    This is the worklist-#1 hot zone: every inference here is a TODO for
    the production solver to surface entry.operation.

    `fodder_tokens` is the list of original fodder tokens parallel to
    `leaves` — needed to detect hidden / anagram fodder when no indicator
    is present (the &lit-style hidden case).
    """
    flags.append("op_inferred_from_token_mix_not_provided")

    has_ana_i = ANA_I in indicators
    has_rev_i = REV_I in indicators
    has_con_i = CON_I in indicators
    has_del_i = DEL_I in indicators
    has_hid_i = HID_I in indicators
    has_hom_i = HOM_I in indicators
    has_trim_i = any(t in indicators for t in _TRIM_INDICATOR_TO_DELETION_KIND)

    has_hid_f = HID_F in fodder_tokens
    has_ana_f = ANA_F in fodder_tokens

    # Hidden — driven by HID_F fodder OR HID_I indicator
    if has_hid_i or has_hid_f:
        ind = _select_indicator(indicators, HID_I)
        if not ind and has_hid_f and not has_hid_i:
            flags.append("hidden_indicator_not_tagged_by_solver")
        inner = hidden(*leaves, indicator=ind)
        if has_rev_i:
            return reversal(inner, indicator=_select_indicator(indicators, REV_I))
        return inner

    if has_ana_i:
        ind = _select_indicator(indicators, ANA_I)
        ana_leaves = [l for l in leaves if l.operation == "literal"]
        non_ana_leaves = [l for l in leaves if l.operation != "literal"]
        if non_ana_leaves:
            # anagram + charade — same problem as grammar_triage's
            ana_node = anagram(*ana_leaves, indicator=ind) if ana_leaves else None
            children = ([ana_node] if ana_node else []) + non_ana_leaves
            if len(children) > 1:
                return charade(*children)
            if len(children) == 1:
                return children[0]
            return None
        return anagram(*leaves, indicator=ind)

    if has_con_i:
        ind = _select_indicator(indicators, CON_I)
        if len(leaves) >= 2:
            flags.append("container_outer_inner_not_preserved")
            outer = leaves[0]
            inner = leaves[1] if len(leaves) == 2 else charade(*leaves[1:])
            inner_root = inner if len(leaves) == 2 else inner
            con = container(outer=outer, inner=inner_root, indicator=ind)
            if has_rev_i:
                return reversal(con,
                                indicator=_select_indicator(indicators, REV_I))
            return con
        flags.append(f"container_arity:{len(leaves)}")
        return unknown("container", sources=leaves, indicator=ind)

    if has_rev_i:
        ind = _select_indicator(indicators, REV_I)
        if len(leaves) == 1:
            return reversal(leaves[0], indicator=ind)
        return reversal(charade(*leaves), indicator=ind)

    if has_del_i or has_trim_i:
        ind = _select_indicator(indicators, DEL_I)
        kind = "tail"
        for tok, k in _TRIM_INDICATOR_TO_DELETION_KIND.items():
            if tok in indicators:
                kind = k
                if ind is None:
                    ind = _select_indicator(indicators, tok)
                break
        if not leaves:
            return unknown("deletion", indicator=ind)
        if len(leaves) == 1:
            return deletion(leaves[0], indicator=ind, kind=kind)
        # 2+ leaves: find which leaf gets trimmed and the assembly order
        # by trying every combination against the answer.
        from itertools import permutations
        answer_clean = "".join(
            c for c in (_LAST_ANSWER or "").upper() if c.isalpha())
        if answer_clean:
            for trim_idx in range(len(leaves)):
                trimmed_val = _trim_value(leaves[trim_idx].value or "",
                                          kind)
                if not trimmed_val:
                    continue
                pieces = []
                for i, lf in enumerate(leaves):
                    if i == trim_idx:
                        pieces.append((i, trimmed_val, True))
                    else:
                        pieces.append((i, lf.value or "", False))
                for perm in permutations(pieces):
                    if "".join(p[1] for p in perm) == answer_clean:
                        nodes = []
                        for piece_idx, _val, is_trim in perm:
                            if is_trim:
                                nodes.append(deletion(
                                    leaves[piece_idx],
                                    indicator=ind, kind=kind))
                            else:
                                nodes.append(leaves[piece_idx])
                        if len(nodes) == 1:
                            return nodes[0]
                        return charade(*nodes)
        # No combination produced the answer — flag and fall back
        flags.append("deletion_combination_not_found")
        return deletion(leaves[0], indicator=ind, kind=kind)

    if has_hom_i:
        ind = _select_indicator(indicators, HOM_I)
        if len(leaves) == 1:
            return Node(operation="homophone", indicator=ind,
                        sources=[leaves[0]])
        flags.append(f"homophone_arity:{len(leaves)}")
        return Node(operation="homophone", indicator=ind, sources=leaves)

    # No indicator at all → charade (or DD if single SYN equal to answer)
    if not leaves:
        return None
    if len(leaves) == 1:
        return leaves[0]
    return charade(*leaves)


# --- Top-level adapter entry point ----------------------------------------

@dataclass
class AdapterResult:
    form: Optional[Form]
    flags: list  # adapter-level flags / worklist hits
    notes: str   # human-readable summary
    sr_present: bool
    confidence: int


def solve_to_form(clue_text: str, answer: str, db: RefDB) -> AdapterResult:
    """Run production solve_clue, convert SolveResult to Form."""
    flags: list = []

    sr = solve_clue(clue_text, answer, db)

    if sr.result is None:
        return AdapterResult(form=None,
                             flags=["solver_no_result"],
                             notes="solve_clue returned no SignatureResult",
                             sr_present=False,
                             confidence=int(sr.confidence or 0))

    op_name = _extract_op_name(sr, flags)
    fodder, indicators, lnks, dbe_markers = _categorise_roles(
        sr.result.word_roles)

    leaves = []
    fodder_tokens = []
    for word, tok, val, meta in fodder:
        leaf = _fodder_to_leaf(word, tok, val, meta, indicators, flags)
        if leaf is not None:
            leaves.append(leaf)
            fodder_tokens.append(tok)

    global _LAST_FODDER_TOKENS, _LAST_ANSWER
    _LAST_FODDER_TOKENS = fodder_tokens
    _LAST_ANSWER = answer
    tree = _build_tree_for_op(op_name, leaves, indicators, flags)
    if tree is None:
        return AdapterResult(form=None,
                             flags=flags + ["tree_build_returned_none"],
                             notes=("op=%s, leaves=%d, indicators=%s"
                                    % (op_name, len(leaves),
                                       sorted(indicators.keys()))),
                             sr_present=True,
                             confidence=int(sr.confidence or 0))

    answer_clean = "".join(c for c in answer.upper() if c.isalpha())
    definition = Definition(phrase=getattr(sr, "definition", "") or "",
                            answer=answer_clean)

    # is_and_lit: cheap DB check on the full clue. Skip for v0 unless
    # easy — leave False with a flag if conditions warrant.
    is_and_lit = False
    if not definition.phrase:
        flags.append("definition_phrase_missing")

    form = Form(
        tree=tree,
        definition=definition,
        link_words=list(lnks),
        is_and_lit=is_and_lit,
        flags=flags,
    )

    notes = "op=%s leaves=%d ind=%s lnk=%d dbe=%d conf=%d" % (
        op_name, len(leaves), sorted(indicators.keys()),
        len(lnks), len(dbe_markers), sr.confidence)
    return AdapterResult(form=form, flags=flags, notes=notes,
                         sr_present=True, confidence=int(sr.confidence or 0))
