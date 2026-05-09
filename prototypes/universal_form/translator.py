"""Flat structured_explanations.components -> universal Form.

The flat form (what every generator currently writes) has:

    {
      "ai_pieces":      [{"clue_word", "letters", "mechanism"}, ...],
      "assembly":       {"op": "...", ...extra fields per op...},
      "wordplay_type"   or  "wordplay_types"
    }

The universal form needs a recursive tree with the outermost operation at
the top. Translation policy (decided 2026-05-02):

    1. Trust assembly.op as the inner / primary operation.
    2. When wordplay_types lists an op not in assembly.op, that op becomes
       the outer wrapper. Order matters: the wordplay_types entry that's
       NOT in assembly.op is the outer.
    3. Anything we can't yet name -> Op(operation="unknown", flat_op=...)
       so the simulation runs and the report surfaces what to add next.

Indicators aren't reliably stored in components; we recover them from
`[type: "word"]` annotations in the ai_explanation prose when present.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .schema import (
    Form, Definition, Leaf, Op, Node,
    LEAF_OPERATIONS,
    lit, syn, abbr, pos, charade, container, anagram, reversal,
    deletion, hidden, double_definition, unknown,
)


# --- Mechanism -> leaf type ------------------------------------------------

# Map flat-form per-piece mechanism -> a function that builds a Leaf.
# A return value of None means "translator can't yet build a leaf for this
# mechanism" — handled by emitting an Op(unknown, flat_op=mechanism) wrapper.

_POSITIONAL_KINDS_BY_MECH = {
    "first_letter":     "first",
    "last_letter":      "last",
    "middle_letters":   "middle",
    "outer_letters":    "outer",
    "second_letter":    "first",   # not a true match; flagged as imprecise
    "core_letters":     "middle",
    "even_letters":     "even",
    "odd_letters":      "odd",
    "alternate_letters": "alternate",
}


def _kind_matches(src: str, val: str, kind: str) -> bool:
    """Does positional kind actually produce val from src?"""
    if not src or not val:
        return False
    n = len(val)
    if kind == "first":
        return src.startswith(val)
    if kind == "last":
        return src.endswith(val)
    if kind == "middle":
        if len(src) < n:
            return False
        start = (len(src) - n) // 2
        return src[start:start + n] == val
    if kind == "outer":
        if n == 2:
            return val[0] == src[0] and val[-1] == src[-1]
        half = n // 2
        return val == src[:half] + src[-(n - half):]
    if kind == "odd":
        return "".join(src[i] for i in range(0, len(src), 2)) == val
    if kind == "even":
        return "".join(src[i] for i in range(1, len(src), 2)) == val
    if kind == "alternate":
        odd = "".join(src[i] for i in range(0, len(src), 2))
        even = "".join(src[i] for i in range(1, len(src), 2))
        return val in (odd, even)
    return False


def _detect_positional(src: str, val: str) -> Optional[str]:
    """If `val` is a positional extraction of `src`, return the kind.

    Both inputs UPPERCASE and letter-only. Returns one of first/last/
    middle/outer/odd/even, or None if no positional fit.
    """
    if not src or not val or len(val) >= len(src):
        return None
    if src.startswith(val):
        return "first"
    if src.endswith(val):
        return "last"
    if len(val) == 1:
        # Middle letter (single)
        mid = len(src) // 2
        if src[mid] == val:
            return "middle"
    if len(val) == 2 and val[0] == src[0] and val[-1] == src[-1]:
        return "outer"
    if len(val) >= 2:
        # Outer extraction for n>=2: half from front + half from back
        half = len(val) // 2
        if val == src[:half] + src[-(len(val) - half):]:
            return "outer"
    # Middle (multi-letter, contiguous middle slice)
    if 2 <= len(val) < len(src):
        start = (len(src) - len(val)) // 2
        if src[start:start + len(val)] == val:
            return "middle"
    # Odd / even letters
    odd = "".join(src[i] for i in range(0, len(src), 2))
    if odd == val:
        return "odd"
    even = "".join(src[i] for i in range(1, len(src), 2))
    if even == val:
        return "even"
    return None


def _piece_to_leaf(piece: dict) -> tuple[Optional[Node], Optional[str]]:
    """Translate one ai_pieces entry to a Node (Leaf or Op) or None.

    Returns (node, reason_or_None). For anagram_fodder pieces whose
    letters are a permutation of their clue_word, returns an anagram Op
    wrapping a literal of the clue_word — that's what the piece actually
    represents.
    """
    mech = (piece.get("mechanism") or "").lower()
    word = piece.get("clue_word") or ""
    letters = (piece.get("letters") or "").upper()

    if not letters:
        return None, "empty letters"

    if mech == "synonym":
        # If source-word letters equal value letters, the piece is really
        # a literal (the clue word IS the answer letters, no synonym
        # needed).
        norm_src = re.sub(r"[^A-Z]", "", (word or "").upper())
        norm_val = re.sub(r"[^A-Z]", "", letters)
        if norm_src and norm_val and norm_src == norm_val:
            return lit(word, letters), \
                f"synonym_relabelled_as_literal:{word!r}={letters!r}"
        return syn(word, letters), None
    if mech == "abbreviation":
        return abbr(word, letters), None
    if mech in ("literal", "anagram_fodder"):
        # Source-word letters and value letters must agree somehow.
        norm_src = re.sub(r"[^A-Z]", "", (word or "").upper())
        norm_val = re.sub(r"[^A-Z]", "", letters)

        # Anagram fodder: value is a permutation of source. The piece is
        # really anagram(literal source). Wrap accordingly. Indicator
        # gets attached later from the surface.
        if mech == "anagram_fodder" and norm_src and norm_val \
                and norm_src != norm_val \
                and sorted(norm_src) == sorted(norm_val):
            inner_lit = lit(word, norm_src)
            return anagram(inner_lit, indicator=""), \
                f"anagram_fodder_promoted_to_anagram:" \
                f"{word!r}->{letters!r}"

        if norm_src and norm_val and norm_src != norm_val:
            # Mis-tagged literal — recover the actual mechanism.
            kind = _detect_positional(norm_src, norm_val)
            if kind:
                return pos(word, letters, kind=kind), \
                    f"literal_relabelled_as_positional[{kind}]:" \
                    f"{word!r}->{letters!r}"
            return abbr(word, letters), \
                f"literal_relabelled_as_abbreviation:" \
                f"{word!r}->{letters!r}"
        return lit(word or letters, letters), None
    if mech == "hidden" or mech == "hidden_reversed":
        # Hidden pieces are literal spans of the clue
        return lit(word or letters, letters), None
    if mech in _POSITIONAL_KINDS_BY_MECH:
        kind = _POSITIONAL_KINDS_BY_MECH[mech]
        # Validate: does this kind actually produce the value? If not,
        # try to detect the real kind from the source/value pair.
        norm_src = re.sub(r"[^A-Z]", "", (word or "").upper())
        norm_val = re.sub(r"[^A-Z]", "", (letters or "").upper())
        if norm_src and norm_val and \
                not _kind_matches(norm_src, norm_val, kind):
            real_kind = _detect_positional(norm_src, norm_val)
            if real_kind and real_kind != kind:
                return pos(word, letters, kind=real_kind), \
                    f"positional_kind_corrected:{mech}->{real_kind} " \
                    f"({word!r}->{letters!r})"
        return pos(word, letters, kind=kind), None
    if mech == "indicator":
        # Indicator-typed pieces are surface words for the operation; the
        # universal form puts them on the parent Op.indicator instead. The
        # piece itself contributes no letters — drop it (caller handles).
        return None, "drop_indicator_piece"

    # Mechanism the translator can't yet name — keep its letters as a
    # literal leaf so assembly can still be attempted, but flag.
    return lit(word or letters, letters), f"unknown_mechanism:{mech}"


# --- Assembly handlers — one per recognised assembly.op ---------------------

@dataclass
class TranslationReport:
    flag: str          # one of: clean, partial_translation,
                       #         requires_re_derivation, malformed, unknown_op
    notes: list        # per-piece or per-op gaps surfaced

    def to_dict(self) -> dict:
        return {"flag": self.flag, "notes": list(self.notes)}


def _make_leaves(pieces: list, notes: list) -> list[Node]:
    """Build a list of leaves from pieces, accumulating notes for any gaps."""
    leaves: list[Node] = []
    for p in pieces:
        leaf, reason = _piece_to_leaf(p)
        if leaf is None:
            if reason and reason != "drop_indicator_piece":
                notes.append(f"piece dropped: {reason}")
            continue
        if reason:
            notes.append(reason)
        leaves.append(leaf)
    return leaves


def _translate_charade(pieces: list, asm: dict, indicator: Optional[str],
                       notes: list) -> Node:
    leaves = _make_leaves(pieces, notes)
    if not leaves:
        return unknown("charade", indicator=indicator)
    if len(leaves) == 1:
        return leaves[0]
    return charade(*leaves, indicator=indicator)


def _translate_anagram(pieces: list, asm: dict, indicator: Optional[str],
                       notes: list) -> Node:
    leaves = _make_leaves(pieces, notes)
    return anagram(*leaves, indicator=indicator) if leaves else \
        unknown("anagram", indicator=indicator)


def _translate_container(pieces: list, asm: dict, indicator: Optional[str],
                         notes: list) -> Node:
    leaves = _make_leaves(pieces, notes)
    if len(leaves) < 2:
        notes.append(f"container needed >=2 children, got {len(leaves)}")
        return unknown("container", sources=leaves, indicator=indicator)
    # assembly often carries inner/outer letter strings — use them when
    # present to assign roles correctly.
    inner_letters = (asm.get("inner") or "").upper()
    outer_letters = (asm.get("outer") or "").upper()
    outer = inner = None
    if inner_letters or outer_letters:
        for lf in leaves:
            if isinstance(lf, Leaf):
                if lf.value.upper() == inner_letters and inner is None:
                    inner = lf
                elif lf.value.upper() == outer_letters and outer is None:
                    outer = lf
        # If only one was matched, fill the other from the unmatched leaf.
        unmatched = [lf for lf in leaves if lf is not inner and lf is not outer]
        if outer is None and unmatched:
            outer = unmatched[0]
        if inner is None and unmatched and unmatched[0] is not outer:
            inner = unmatched[0]
    if outer is None or inner is None:
        # Default: shorter piece is inner (cryptic convention)
        sorted_leaves = sorted(leaves,
                               key=lambda lf: len(getattr(lf, "value", "")))
        inner = sorted_leaves[0]
        outer = sorted_leaves[-1] if sorted_leaves[-1] is not inner else \
            sorted_leaves[1]
    return container(outer=outer, inner=inner, indicator=indicator)


def _translate_reversal(pieces: list, asm: dict, indicator: Optional[str],
                        notes: list) -> Node:
    leaves = _make_leaves(pieces, notes)
    if len(leaves) == 1:
        return reversal(leaves[0], indicator=indicator or "")
    if len(leaves) > 1:
        # Reversal of a charade
        return reversal(charade(*leaves), indicator=indicator or "")
    return unknown("reversal", indicator=indicator)


def _translate_hidden(pieces: list, asm: dict, indicator: Optional[str],
                      notes: list) -> Node:
    span_words = asm.get("words") or asm.get("word") or ""
    span_words_str = span_words if isinstance(span_words, str) \
        else " ".join(span_words)
    if span_words_str:
        # Split the span into clue tokens; each token becomes a literal leaf.
        toks = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", span_words_str)
        leaves: list[Node] = [lit(t) for t in toks]
        return hidden(*leaves, indicator=indicator) if leaves \
            else unknown("hidden", indicator=indicator)
    leaves = _make_leaves(pieces, notes)
    if leaves:
        return hidden(*leaves, indicator=indicator)
    notes.append("hidden has no span words and no pieces")
    return unknown("hidden", indicator=indicator)


def _translate_hidden_reversed(pieces: list, asm: dict,
                               indicator: Optional[str],
                               notes: list) -> Node:
    inner = _translate_hidden(pieces, asm, indicator=None, notes=notes)
    return reversal(inner, indicator=indicator or "")


def _translate_hidden_in_word(pieces: list, asm: dict,
                              indicator: Optional[str], notes: list) -> Node:
    word = asm.get("word")
    if word:
        return hidden(lit(word), indicator=indicator)
    return _translate_hidden(pieces, asm, indicator, notes)


def _translate_dd(pieces: list, asm: dict, indicator: Optional[str],
                  notes: list) -> Node:
    left_def = asm.get("left_def") or asm.get("left") or ""
    right_def = asm.get("right_def") or asm.get("right") or ""
    if left_def and right_def:
        ans_placeholder = "?"  # replaced by Form.definition.answer
        return double_definition(
            syn(left_def, ans_placeholder),
            syn(right_def, ans_placeholder),
        )
    notes.append("DD missing left_def/right_def — emitting unknown")
    return unknown("double_definition", indicator=indicator)


def _translate_deletion(pieces: list, asm: dict, indicator: Optional[str],
                        notes: list) -> Node:
    leaves = _make_leaves(pieces, notes)
    if leaves:
        # Single-piece deletion: the source's letters reduced to the leaf's
        # value. Without a kind hint from the flat form, emit kind=None and
        # let the verifier flag.
        return deletion(leaves[0], indicator=indicator or "", kind="tail")
    return unknown("deletion", indicator=indicator)


def _translate_homophone(pieces: list, asm: dict,
                         indicator: Optional[str], notes: list) -> Node:
    notes.append("homophone op not yet supported in basic vocab")
    return unknown("homophone", sources=_make_leaves(pieces, notes),
                   indicator=indicator)


def _translate_cd(pieces: list, asm: dict,
                  indicator: Optional[str], notes: list) -> Node:
    notes.append("cryptic_definition op not yet supported in basic vocab")
    return unknown("cryptic_definition",
                   sources=_make_leaves(pieces, notes),
                   indicator=indicator)


def _translate_acrostic(pieces: list, asm: dict,
                        indicator: Optional[str], notes: list) -> Node:
    # Acrostic is positional applied to the first letters of multiple
    # source words. Until we promote it into a basic op, emit a charade
    # of positional leaves where the mechanism allows.
    leaves = _make_leaves(pieces, notes)
    if leaves:
        return charade(*leaves, indicator=indicator)
    return unknown("acrostic", indicator=indicator)


def _translate_spoonerism(pieces: list, asm: dict,
                          indicator: Optional[str], notes: list) -> Node:
    notes.append("spoonerism op not yet supported in basic vocab")
    return unknown("spoonerism", sources=_make_leaves(pieces, notes),
                   indicator=indicator)


_ASSEMBLY_HANDLERS = {
    "charade":           _translate_charade,
    "anagram":           _translate_anagram,
    "container":         _translate_container,
    "reversal":          _translate_reversal,
    "hidden":            _translate_hidden,
    "hidden_reversed":   _translate_hidden_reversed,
    "hidden_in_word":    _translate_hidden_in_word,
    "hidden_word":       _translate_hidden,        # alias seen 2 times
    "double_definition": _translate_dd,
    "deletion":          _translate_deletion,
    "homophone":         _translate_homophone,
    "cryptic_definition": _translate_cd,
    "acrostic":          _translate_acrostic,
    "spoonerism":        _translate_spoonerism,
}


# --- Indicator recovery from prose -----------------------------------------

# `[type: "word"]` annotations the live mechanical/sonnet generators emit.
_INDICATOR_BRACKET_RE = re.compile(
    r'\[\s*([a-zA-Z_]+)\s*:\s*["\']([^"\']+)["\']\s*\]', re.IGNORECASE)


def recover_indicators(ai_explanation: str) -> dict[str, str]:
    """Pull `[op: "word"]` annotations from the prose. Returns a dict
    mapping op_type -> indicator_word. First occurrence wins."""
    out: dict[str, str] = {}
    if not ai_explanation:
        return out
    for op_type, word in _INDICATOR_BRACKET_RE.findall(ai_explanation):
        op_type = op_type.lower().strip().replace(" ", "_")
        if op_type not in out:
            out[op_type] = word.strip()
    return out


# --- Top-level translation entry point -------------------------------------

@dataclass
class TranslationResult:
    form: Optional[Form]
    report: TranslationReport
    components_raw: dict


def translate(components_json: Optional[str], wordplay_types: Optional[str],
              definition_text: Optional[str], answer: str,
              ai_explanation: Optional[str] = None,
              clue_text: Optional[str] = None) -> TranslationResult:
    """Translate a structured_explanations row into a universal Form.

    Inputs are the columns as they appear in the DB (JSON strings, etc.).
    Returns a TranslationResult with the Form (or None if untranslatable)
    and a TranslationReport explaining the flag.

    `clue_text`, when supplied, is used to populate `form.link_words`:
    any surface word that is not claimed by the form's tree or definition
    AND is in the global LINK_WORDS allow-list is added to form.link_words.
    This makes the form structurally complete in the spec's sense
    (residue check needs form.link_words to be present and authoritative).
    """
    notes: list[str] = []

    if not components_json:
        return TranslationResult(
            form=None,
            report=TranslationReport(flag="requires_re_derivation",
                                     notes=["components is NULL"]),
            components_raw={},
        )

    try:
        components = json.loads(components_json)
    except Exception as e:
        return TranslationResult(
            form=None,
            report=TranslationReport(flag="malformed",
                                     notes=[f"json parse: {e}"]),
            components_raw={},
        )

    if isinstance(components, list):
        return TranslationResult(
            form=None,
            report=TranslationReport(flag="malformed",
                                     notes=["components is a list, not dict"]),
            components_raw={"_list": components},
        )

    pieces = components.get("ai_pieces") or []
    assembly = components.get("assembly") or {}
    asm_op = (assembly.get("op") or "").lower().strip()

    # Empty ai_pieces with no DD/hidden span data means the structured form
    # is too thin to translate — the explanation lives in prose only.
    has_dd_data = bool(assembly.get("left_def") or assembly.get("right_def"))
    has_hidden_span = bool(assembly.get("words") or assembly.get("word"))
    if not pieces and not has_dd_data and not has_hidden_span:
        return TranslationResult(
            form=None,
            report=TranslationReport(
                flag="requires_re_derivation",
                notes=[f"ai_pieces empty and no span/DD data; "
                       f"assembly.op={asm_op!r}"]),
            components_raw=components,
        )

    # Recover indicators from prose annotations
    ind_map = recover_indicators(ai_explanation or "")

    # Pick the inner-op handler
    if not asm_op:
        return TranslationResult(
            form=None,
            report=TranslationReport(
                flag="requires_re_derivation",
                notes=[f"no assembly.op; pieces={len(pieces)}"]),
            components_raw=components,
        )

    # Compound assembly.ops like "deletion+anagram", "container_reversal".
    # These are real but rare. Emit unknown for now and surface for naming.
    if asm_op not in _ASSEMBLY_HANDLERS:
        return TranslationResult(
            form=Form(
                tree=unknown(asm_op,
                             sources=_make_leaves(pieces, notes),
                             indicator=ind_map.get(asm_op)),
                definition=Definition(
                    phrase=definition_text or "", answer=answer),
                link_words=[],
            ),
            report=TranslationReport(flag="unknown_op",
                                     notes=notes + [f"assembly.op={asm_op!r}"]),
            components_raw=components,
        )

    handler = _ASSEMBLY_HANDLERS[asm_op]
    inner_indicator = ind_map.get(asm_op)
    inner_tree = handler(pieces, assembly, inner_indicator, notes)

    # Wordplay-types-driven outer wrapper
    wt_list = _parse_wordplay_types(wordplay_types)
    outer_op = _outer_op_from_wordplay_types(wt_list, asm_op)
    if outer_op:
        wrapper = _wrap_outer(inner_tree, outer_op,
                              indicator=ind_map.get(outer_op), notes=notes)
        if wrapper is not None:
            inner_tree = wrapper

    # Patch DD synonym leaves with the actual answer
    _patch_dd_answer(inner_tree, answer)

    # Surface-driven inference (order matters):
    #   (1) Acrostic detection — a charade of all positional[first]
    #       (or [last], etc.) leaves IS an acrostic, not a charade.
    #       Promote to acrostic; the surface acrostic indicator (e.g.
    #       'at first') sits on the acrostic node.
    #   (2) Reversal inference — pieces in reverse clue order with a
    #       reversal indicator in the surface => wrap in reversal.
    #   (3) Positional indicator inference — for any remaining
    #       positional leaves, attach indicators from the surface.
    if clue_text:
        inner_tree, ac_note = _maybe_promote_acrostic(
            inner_tree, clue_text)
        if ac_note:
            notes.append(ac_note)
    # Reorder charade pieces if stored order doesn't assemble
    inner_tree, ro_note = _maybe_reorder_charade(inner_tree, answer)
    if ro_note:
        notes.append(ro_note)
    # Swap container outer/inner if assignment doesn't produce the answer
    inner_tree, sw_note = _maybe_swap_container_roles(inner_tree, answer)
    if sw_note:
        notes.append(sw_note)
    if clue_text:
        inner_tree, cn_note = _maybe_infer_container(
            inner_tree, clue_text, answer)
        if cn_note:
            notes.append(cn_note)
    if clue_text:
        inner_tree, inferred_note = _maybe_infer_reversal(
            inner_tree, clue_text, answer)
        if inferred_note:
            notes.append(inferred_note)
    if clue_text:
        inferred_pos = _attach_positional_indicators(inner_tree, clue_text)
        for note in inferred_pos:
            notes.append(note)
        # Attach anagram indicators (from surface) to any anagram nodes
        # without one — typically the anagram_fodder-promoted nodes.
        inferred_ana = _attach_anagram_indicators(inner_tree, clue_text)
        for note in inferred_ana:
            notes.append(note)
        # General op-indicator attachment: container, deletion, reversal,
        # homophone.
        inferred_ops = _attach_op_indicators(inner_tree, clue_text)
        for note in inferred_ops:
            notes.append(note)

    flag = "clean" if not notes else "partial_translation"
    form = Form(
        tree=inner_tree,
        definition=Definition(phrase=definition_text or "", answer=answer),
        link_words=_derive_link_words(inner_tree, definition_text or "",
                                      clue_text or ""),
    )
    return TranslationResult(
        form=form,
        report=TranslationReport(flag=flag, notes=notes),
        components_raw=components,
    )


# --- Acrostic promotion ---------------------------------------------------

# Surface phrases / single words used as acrostic indicators (kind = first).
# Sourced from cryptic_new.db indicators where wordplay_type='acrostic' or
# 'parts'. Multi-word phrases listed first for greedy matching.
_ACROSTIC_INDICATORS_FIRST = [
    "at first", "to start", "to begin", "head of", "heads of",
    "leader of", "leaders of", "starting", "starts", "start",
    "first", "initially", "initial", "beginning", "leading",
    "lead", "front of", "top of", "primarily", "originally",
    "front",
]
_ACROSTIC_INDICATORS_LAST = [
    "in the end", "at the end", "at last", "tail of", "tails of",
    "back of", "ends of", "end of", "ending in", "ends", "end",
    "last", "finally", "final", "ultimately", "ultimate",
    "bottom of",
]


def _maybe_promote_acrostic(tree, clue_text: str
                            ) -> tuple[object, Optional[str]]:
    """If the tree is a charade where every child is a positional leaf
    of the same kind (first/last), promote it to an acrostic operation.

    The acrostic's children become literals of the source words. The
    indicator (e.g. 'at first') is found in the surface and attached to
    the acrostic node.

    Returns (new_tree, note_or_None).
    """
    if not isinstance(tree, Op) or tree.operation != "charade":
        return tree, None
    leaves = [c for c in tree.sources if isinstance(c, Leaf)]
    if len(leaves) < 2 or len(leaves) != len(tree.sources):
        return tree, None
    # All leaves must be positional with the SAME kind
    kinds = {lf.positional_kind for lf in leaves
             if lf.operation == "positional"}
    if len(kinds) != 1:
        return tree, None
    if not all(lf.operation == "positional" for lf in leaves):
        return tree, None
    kind = kinds.pop()
    if kind not in ("first", "last"):
        return tree, None  # extend later for middle/odd/etc.

    surface_lower = (clue_text or "").lower()
    surface_tokens = re.findall(
        r"[a-zA-Z]+(?:'[a-zA-Z]+)?", surface_lower)
    candidates = (_ACROSTIC_INDICATORS_FIRST if kind == "first"
                  else _ACROSTIC_INDICATORS_LAST)
    indicator: Optional[str] = None
    for phrase in candidates:
        if " " in phrase:
            if phrase in surface_lower:
                indicator = phrase
                break
        else:
            if phrase in surface_tokens:
                indicator = phrase
                break

    # The acrostic's children become literals (the source words supplied
    # the letter — we no longer need a positional leaf wrapping them).
    new_children = [Leaf("literal", lf.source_word,
                         lf.source_word.upper().replace(" ", ""))
                    for lf in leaves]
    new_tree = Op("acrostic", indicator, new_children,
                  acrostic_kind=kind)
    note = (f"promoted_to_acrostic[{kind}]: charade of {len(leaves)} "
            f"positional[{kind}] leaves"
            + (f"; indicator={indicator!r}" if indicator
               else "; no surface indicator found"))
    return new_tree, note


# --- Container role-swap correction --------------------------------------

def _maybe_swap_container_roles(tree, answer: str
                                ) -> tuple[object, Optional[str]]:
    """If the form is a container whose stored outer/inner assignment
    doesn't produce the answer, but swapping them does, swap.

    The translator's container handler picks roles by length heuristic
    when the assembly hint is ambiguous; sometimes that's wrong.
    """
    if not isinstance(tree, Op) or tree.operation != "container":
        return tree, None
    if len(tree.sources) != 2:
        return tree, None
    answer_clean = re.sub(r"[^A-Z]", "", (answer or "").upper())
    outer, inner = tree.sources
    outer_letters = re.sub(r"[^A-Z]", "",
                           _node_letters(outer).upper())
    inner_letters = re.sub(r"[^A-Z]", "",
                           _node_letters(inner).upper())
    # Try inserting inner into outer (current orientation)
    cur_works = any(
        outer_letters[:p] + inner_letters + outer_letters[p:] == answer_clean
        for p in range(1, len(outer_letters)))
    if cur_works:
        return tree, None
    # Try the swap: inner becomes outer, outer becomes inner
    swap_works = any(
        inner_letters[:p] + outer_letters + inner_letters[p:] == answer_clean
        for p in range(1, len(inner_letters)))
    if swap_works:
        new_tree = Op("container", indicator=tree.indicator,
                      sources=[inner, outer])
        return new_tree, (
            f"swapped_container_roles: stored "
            f"outer/inner didn't produce answer; swapped")
    return tree, None


def _node_letters(node) -> str:
    """Best-effort letter computation for a node — used by translator
    inferences that need to compare against the answer."""
    if isinstance(node, Leaf):
        return node.value
    if not isinstance(node, Op):
        return ""
    op = node.operation
    children = [_node_letters(c) for c in node.sources]
    if op == "charade":
        return "".join(children)
    if op == "anagram":
        return "".join(children)  # multiset, not in order
    if op == "reversal":
        return children[0][::-1] if children else ""
    if op == "deletion":
        if not children:
            return ""
        kind = node.deletion_kind
        src = children[0]
        if kind == "head":
            return src[1:]
        if kind == "tail":
            return src[:-1]
        if kind == "outer":
            return src[1:-1]
        return src[:-1]
    if op == "container" and len(children) == 2:
        return children[0] + children[1]  # placeholder
    return "".join(children)


# --- Charade order permutation --------------------------------------------

def _maybe_reorder_charade(tree, answer: str) -> tuple[object, Optional[str]]:
    """If a charade's children in stored order don't concatenate to the
    answer but some permutation does, reorder the children. The setter's
    intent was the order that produces the answer; the lazy stored data
    sometimes lists pieces in the order they appeared in the surface
    rather than in assembly order.

    Returns (new_tree, note_or_None).
    """
    import itertools
    if not isinstance(tree, Op) or tree.operation != "charade":
        return tree, None
    leaves = [c for c in tree.sources if isinstance(c, Leaf)]
    if len(leaves) < 2 or len(leaves) != len(tree.sources):
        return tree, None
    answer_clean = re.sub(r"[^A-Z]", "", (answer or "").upper())
    letters_list = [re.sub(r"[^A-Z]", "", lf.value.upper())
                    for lf in leaves]
    if "".join(letters_list) == answer_clean:
        return tree, None  # already correct order
    # Try permutations (limit to reasonable size)
    if len(leaves) > 5:
        return tree, None
    for perm in itertools.permutations(range(len(leaves))):
        candidate = "".join(letters_list[i] for i in perm)
        if candidate == answer_clean:
            new_sources = [leaves[i] for i in perm]
            new_tree = Op("charade", indicator=tree.indicator,
                          sources=new_sources)
            return new_tree, (
                f"reordered_charade: stored order {[lf.source_word for lf in leaves]} "
                f"didn't assemble; reordered to "
                f"{[leaves[i].source_word for i in perm]}")
    return tree, None


# --- Container inference --------------------------------------------------

# Surface words/phrases used as container indicators. Sourced from
# cryptic_new.db indicators where wordplay_type IN ('container','insertion').
_CONTAINER_INDICATOR_PHRASES = [
    "wrapped around", "filled with", "swallowing up", "swallowing",
    "containing", "around", "encircling", "enveloping", "enclosing",
    "surrounding", "embracing", "engulfing", "smothering", "covering",
    "hugging", "harbouring", "harboring", "concealing", "to cover",
    "to contain", "to hold", "holding", "carrying", "bearing",
    "housing", "trapping", "catching", "gripping", "absorbing",
    "into", "in", "inside", "within", "during", "amongst", "among",
    "amid", "amidst", "inserted in", "going into", "put into",
    "placed in", "set in", "buried in", "hidden in", "lost in",
    "encased in", "imprisoned in", "trapped in", "stuck in",
    "stowed in", "stored in", "tucked in", "bound in", "wedged in",
    "kept in", "penning",
]


def _maybe_infer_container(tree, clue_text: str,
                           answer: str) -> tuple[object, Optional[str]]:
    """If a charade's children don't concatenate to the answer but
    inserting one into another does, AND the surface has a container
    indicator, convert to container.

    Handles 2-piece and 3-piece charades. For 3+ pieces, tries grouping
    adjacent pairs as the inner charade and the remaining as the outer.

    Returns (new_tree, note_or_None).
    """
    if not isinstance(tree, Op) or tree.operation != "charade":
        return tree, None
    leaves = [c for c in tree.sources if isinstance(c, Leaf)]
    if len(leaves) < 2 or len(leaves) != len(tree.sources):
        return tree, None

    answer_clean = re.sub(r"[^A-Z]", "", (answer or "").upper())
    letters_list = [re.sub(r"[^A-Z]", "", lf.value.upper())
                    for lf in leaves]
    concat = "".join(letters_list)
    if concat == answer_clean:
        return tree, None  # plain charade works; nothing to infer

    indicator = _find_container_indicator(clue_text or "")
    if not indicator:
        return tree, None

    # Try each contiguous subset of pieces as the inner group; the rest
    # become the outer. Tests every insertion position in the outer.
    n = len(leaves)
    for inner_start in range(n):
        for inner_end in range(inner_start, n):
            inner_idxs = list(range(inner_start, inner_end + 1))
            outer_idxs = [i for i in range(n) if i not in inner_idxs]
            if not outer_idxs:
                continue
            inner_letters = "".join(letters_list[i] for i in inner_idxs)
            outer_letters = "".join(letters_list[i] for i in outer_idxs)
            for pos in range(1, len(outer_letters)):
                cand = (outer_letters[:pos] + inner_letters
                        + outer_letters[pos:])
                if cand == answer_clean:
                    inner_node = (leaves[inner_idxs[0]]
                                  if len(inner_idxs) == 1
                                  else Op("charade", indicator=None,
                                          sources=[leaves[i] for i in inner_idxs]))
                    outer_node = (leaves[outer_idxs[0]]
                                  if len(outer_idxs) == 1
                                  else Op("charade", indicator=None,
                                          sources=[leaves[i] for i in outer_idxs]))
                    new_tree = Op("container", indicator=indicator,
                                  sources=[outer_node, inner_node])
                    return new_tree, (
                        f"inferred_container: pieces don't concatenate "
                        f"but inner({len(inner_idxs)})-into-outer({len(outer_idxs)})"
                        f" does; wrapped with indicator {indicator!r}")
    return tree, None


def _find_container_indicator(clue_text: str) -> Optional[str]:
    surface_lower = clue_text.lower()
    surface_tokens = re.findall(
        r"[a-zA-Z]+(?:'[a-zA-Z]+)?", surface_lower)
    # Multi-word phrases (those containing a space) first
    multi = [p for p in _CONTAINER_INDICATOR_PHRASES if " " in p]
    single = [p for p in _CONTAINER_INDICATOR_PHRASES if " " not in p]
    for phrase in multi:
        if phrase in surface_lower:
            return phrase
    for tok in surface_tokens:
        if tok in single:
            return tok
    return None


# --- Reversal inference ---------------------------------------------------

# Reversal indicator words from the cryptic_new.db indicators table where
# wordplay_type='reversal'. Hardcoded here so the translator runs standalone.
# Source: SELECT word FROM indicators WHERE wordplay_type='reversal'.
# Multi-word phrases ('sent over', 'going up') are checked first.
_REVERSAL_INDICATOR_PHRASES = (
    "sent over", "sent back", "going up", "thrown back", "turned over",
    "turned up", "turned back", "comes back", "coming back", "comes up",
    "rolled up", "wound up", "going back", "turned round",
)
_REVERSAL_INDICATOR_WORDS = {
    "back", "backward", "backwards", "reversed", "reverse", "returned",
    "returning", "returns", "returned", "rising", "risen", "rises",
    "rose", "up", "upset", "reflected", "reflects", "round", "about",
    "over", "sent", "ascending", "ascended", "westward", "northward",
    "raised", "lifted", "rebounded", "recoiled", "thrown",
}


def _maybe_infer_reversal(tree, clue_text: str,
                          answer: str) -> tuple[object, Optional[str]]:
    """Detect a missing reversal layer.

    Conditions to wrap the form in a reversal:
      1. Tree is a charade or acrostic with multiple ordered leaf
         children whose source_words appear in the clue surface.
      2. Those source_words appear in REVERSE order in the clue surface
         compared to the form's piece order.
      3. The clue surface contains a known reversal indicator (phrase
         or single word).

    On a match: reorder the children to clue order and wrap with a
    reversal{indicator} node.

    Returns (new_tree, note_or_None).
    """
    if not isinstance(tree, Op) or tree.operation not in (
            "charade", "acrostic"):
        return tree, None
    leaves = [c for c in tree.sources if isinstance(c, Leaf)]
    if len(leaves) < 2 or len(leaves) != len(tree.sources):
        return tree, None

    # Surface positions of each leaf's source_word
    surface_lower = (clue_text or "").lower()
    positions: list[Optional[int]] = []
    surface_tokens = re.findall(
        r"[a-zA-Z]+(?:'[a-zA-Z]+)?", surface_lower)

    def _surface_index(src_word: str) -> Optional[int]:
        target = (src_word or "").lower().strip()
        if not target:
            return None
        for i, tok in enumerate(surface_tokens):
            if tok == target or tok == target + "'s" or target == tok:
                return i
        # Multi-word source: look for first word
        first = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", target)
        if first:
            for i, tok in enumerate(surface_tokens):
                if tok == first[0]:
                    return i
        return None

    for lf in leaves:
        positions.append(_surface_index(lf.source_word))
    if any(p is None for p in positions):
        return tree, None

    # Are positions strictly DECREASING? (pieces in reverse clue order)
    is_reversed_order = all(positions[i] > positions[i + 1]
                            for i in range(len(positions) - 1))
    if not is_reversed_order:
        return tree, None

    # Look for a reversal indicator in the clue surface
    indicator = _find_reversal_indicator(surface_lower, surface_tokens)
    if not indicator:
        return tree, None

    # Reorder leaves to clue order and wrap with reversal. Preserve
    # the original op type (charade or acrostic) and any kind/indicator
    # already attached to it.
    reordered = sorted(leaves, key=lambda lf: _surface_index(lf.source_word))
    inner_kwargs: dict = {"indicator": tree.indicator,
                          "sources": list(reordered)}
    if tree.operation == "acrostic":
        inner_kwargs["acrostic_kind"] = tree.acrostic_kind
    new_inner = Op(tree.operation, **inner_kwargs)
    new_tree = Op("reversal", indicator=indicator, sources=[new_inner])
    return new_tree, (
        f"inferred_reversal: pieces were in reverse clue order; "
        f"wrapped in reversal indicator={indicator!r}")


def _find_reversal_indicator(surface_lower: str,
                             surface_tokens: list) -> Optional[str]:
    # Multi-word phrases first
    for phrase in _REVERSAL_INDICATOR_PHRASES:
        if phrase in surface_lower:
            return phrase
    # Single words
    for tok in surface_tokens:
        if tok in _REVERSAL_INDICATOR_WORDS:
            return tok
    return None


# --- Positional indicator inference ----------------------------------------

# Per positional kind, ordered list of indicator phrases (longest first
# so multi-word matches take precedence). Sourced from the indicators DB
# (wordplay_type IN 'parts','acrostic') and pruned to high-signal
# phrases for each kind.
_POSITIONAL_INDICATORS_BY_KIND = {
    "first": [
        "at first", "to start", "to begin", "head of", "heads of",
        "leader of", "leaders of", "starting", "starts", "start",
        "first", "initially", "initial", "beginning", "leading",
        "lead", "front of", "top of", "primarily", "originally",
    ],
    "last": [
        "in the end", "at the end", "at last", "tail of", "tails of",
        "back of", "ends of", "end of", "ending in", "ends", "end",
        "last", "finally", "final", "ultimately", "ultimate",
        "bottom of",
    ],
    "middle": [
        "middle of", "centre of", "center of", "heart of",
        "middle", "centre", "center", "heart", "core",
    ],
    "outer": [
        "extremes of", "limits of", "edges of", "fringes of",
        "borders of", "outer", "outside", "extremely", "extreme",
        "extremes", "limits", "edges", "fringes", "borders",
        "outermost", "ends",
    ],
    "odd": [
        "odd", "odds", "odd letters of", "odd letter of",
    ],
    "even": [
        "even", "evens", "even letters of", "even letter of",
    ],
    "alternate": [
        "alternately", "alternate", "regularly", "regular intervals",
        "every other",
    ],
}


def _attach_positional_indicators(tree, clue_text: str) -> list:
    """Walk the form, attach positional indicators where missing.

    For each positional leaf with no positional_indicator: scan the
    surface for an indicator phrase matching the leaf's kind. Attach
    the first match. Each indicator phrase can only be claimed once
    across all leaves (so 'at first' covers ALL first-letter leaves
    but is only claimed once in residue).

    Returns a list of notes describing what was attached.
    """
    notes: list = []
    surface_lower = (clue_text or "").lower()

    # Group positional leaves by kind, in tree order
    leaves_by_kind: dict = {}
    for n in _walk_for_positional(tree):
        leaves_by_kind.setdefault(n.positional_kind, []).append(n)

    for kind, leaves in leaves_by_kind.items():
        if not any(lf.positional_indicator is None for lf in leaves):
            continue
        candidates = _POSITIONAL_INDICATORS_BY_KIND.get(kind, [])
        chosen: Optional[str] = None
        for phrase in candidates:
            if phrase in surface_lower:
                chosen = phrase
                break
            # Single-word fallback — match as a whole token
            if " " not in phrase:
                tokens = re.findall(
                    r"[a-zA-Z]+(?:'[a-zA-Z]+)?", surface_lower)
                if phrase in tokens:
                    chosen = phrase
                    break
        if chosen:
            # Attach to the FIRST leaf without an indicator
            for lf in leaves:
                if lf.positional_indicator is None:
                    lf.positional_indicator = chosen
                    notes.append(
                        f"inferred_positional_indicator: "
                        f"{chosen!r} attached to "
                        f"positional[{kind}] leaf "
                        f"{lf.source_word!r}")
                    break
    return notes


def _walk_for_positional(node):
    out: list = []
    if isinstance(node, Leaf):
        if node.operation == "positional":
            out.append(node)
        return out
    if isinstance(node, Op):
        for c in node.sources:
            out.extend(_walk_for_positional(c))
    return out


# --- Anagram indicator inference ------------------------------------------

# Surface phrases / words used as anagram indicators. Sourced from
# cryptic_new.db indicators where wordplay_type='anagram'. Multi-word
# phrases first for greedy matching. Common ones picked by frequency.
_ANAGRAM_INDICATOR_PHRASES = [
    "all over the place", "out of order", "in disarray", "going wrong",
    "in a stew", "in pieces", "fell apart", "knocked about",
]
_ANAGRAM_INDICATOR_WORDS = {
    "anagram", "anagrammed", "scrambled", "muddled", "mixed",
    "confused", "broken", "shaken", "stirred", "cooked", "twisted",
    "twist", "wild", "crazy", "mad", "mangled", "warped", "wrong",
    "odd", "strange", "strangely", "oddly", "out", "off", "around",
    "about", "different", "differently", "fixed", "fix", "altered",
    "adjusted", "changed", "moving", "moved", "tossed", "rambling",
    "wriggling", "wriggles", "wriggle", "drunk", "drunken",
    "distributed", "distribution", "rebuilt", "rebuilding",
    "redesigned", "rearranged", "rearranging", "set", "shaping",
    "shapeless", "scattered", "scatters", "tortured", "spilling",
    "shifting", "shifted", "kneaded", "blended", "spinning", "spun",
    "novel", "new", "fresh", "weird", "weirdly", "freaky",
    "spilled", "splashed", "ruptured", "ruined", "shuffled",
    "messed", "mess", "wretched", "fluctuating", "fluttering",
    "flapping", "wandering", "wandered", "agitated", "stewed",
    "tossing", "thrown", "loose", "loosely", "free", "freely",
    "running", "wriggly", "uneasy", "twirling", "rolling", "boiled",
    "thrashed", "thrashing", "shaken", "trembling", "performing",
    "performed", "remodelled", "improvised", "modified",
    "modified", "manipulated", "doctored", "edited", "renovated",
    "rough", "rocky", "lively", "amended",
}


def _attach_anagram_indicators(tree, clue_text: str) -> list:
    """Walk the form, attach surface anagram indicators to anagram
    nodes that have none."""
    notes: list = []
    surface_lower = (clue_text or "").lower()
    surface_tokens = re.findall(
        r"[a-zA-Z]+(?:'[a-zA-Z]+)?", surface_lower)

    anagram_nodes: list = []
    for n in _walk_for_op(tree, "anagram"):
        if not n.indicator:
            anagram_nodes.append(n)
    if not anagram_nodes:
        return notes

    # Collect surface words already claimed by the form so we don't
    # try to use them as indicators.
    claimed_words: set = set()
    for tok in _collect_tree_words(tree):
        for w in re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?",
                            (tok or "").lower()):
            claimed_words.add(w[:-2] if w.endswith("'s") else w)

    # Find an anagram indicator in the surface (not already claimed)
    for phrase in _ANAGRAM_INDICATOR_PHRASES:
        if phrase in surface_lower:
            phrase_words = re.findall(r"[a-zA-Z]+", phrase)
            if any(w in claimed_words for w in phrase_words):
                continue
            for n in anagram_nodes:
                if not n.indicator:
                    n.indicator = phrase
                    notes.append(
                        f"inferred_anagram_indicator: {phrase!r} attached")
                    break
            return notes
    for tok in surface_tokens:
        if tok in claimed_words:
            continue
        if tok in _ANAGRAM_INDICATOR_WORDS:
            for n in anagram_nodes:
                if not n.indicator:
                    n.indicator = tok
                    notes.append(
                        f"inferred_anagram_indicator: {tok!r} attached")
                    break
            return notes
    return notes


def _walk_for_op(node, op_name: str):
    out: list = []
    if isinstance(node, Op):
        if node.operation == op_name:
            out.append(node)
        for c in node.sources:
            out.extend(_walk_for_op(c, op_name))
    return out


# --- General indicator attachment for ops missing them --------------------

# Per op, ordered list of surface words/phrases used as indicators
# (longest-first for greedy matching). Sourced from the indicators DB.
_OP_INDICATOR_PHRASES = {
    "container": [
        "wrapped around", "filled with", "swallowing", "containing",
        "encircling", "enveloping", "enclosing", "surrounding",
        "embracing", "engulfing", "smothering", "covering", "to cover",
        "harbouring", "harboring", "concealing", "to contain",
        "to hold", "holding", "carrying", "bearing", "housing",
        "trapping", "catching", "gripping", "absorbing", "penning",
        "imprisoning", "confining", "encasing", "around", "outside",
        "outside of", "encircle", "across", "round", "spans",
    ],
    "deletion": [
        "lacking", "without", "missing", "less", "minus", "removing",
        "removed", "remove", "drops", "dropped", "drop", "discarding",
        "discarded", "leaving", "losing", "lost", "ditches", "ditched",
        "abandons", "abandoned", "shedding", "shed", "rejecting",
        "rejected", "reject", "deletes", "deleted", "delete",
        "omitting", "omitted", "omit", "excluded", "exclude",
        "excluding", "absent", "gone", "no", "not", "non", "shorn",
        "stripped", "stripping", "out of", "free of", "free from",
        "minus the", "headless", "tailless", "endless", "topless",
        "bottomless", "without one", "taken from", "taken out",
        "removed from",
    ],
    "reversal": [
        "sent over", "sent back", "going up", "thrown back",
        "turned over", "turned up", "turned back", "comes back",
        "coming back", "comes up", "rolled up", "wound up",
        "going back", "turned round", "back", "backwards", "reversed",
        "reverse", "returned", "returning", "rising", "up", "upset",
        "reflected", "round", "over", "ascending", "raised",
    ],
    "homophone": [
        "we hear", "they say", "it's said", "spoken", "voiced",
        "audibly", "audible", "broadcast", "reported", "they tell us",
        "by the sound", "sounds like", "to the ear", "for the audience",
        "heard", "in conversation", "verbally",
    ],
}


def _attach_op_indicators(tree, clue_text: str) -> list:
    """Walk the form, attach surface op-indicators to ops with none.

    Skips leaves and exempt ops (charade, double_definition). Anagram
    is handled by _attach_anagram_indicators (more specific). Container
    / deletion / reversal / homophone are handled here.
    """
    notes: list = []
    surface_lower = (clue_text or "").lower()
    surface_tokens = re.findall(
        r"[a-zA-Z]+(?:'[a-zA-Z]+)?", surface_lower)

    # Words already claimed by the form so we don't re-use them as
    # indicators for additional ops.
    claimed: set = set()
    for tok in _collect_tree_words(tree):
        for w in re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?",
                            (tok or "").lower()):
            claimed.add(w[:-2] if w.endswith("'s") else w)

    # Walk every op node missing an indicator
    needs: list = []
    _walk_ops_missing_indicator(tree, needs)
    if not needs:
        return notes

    for node in needs:
        op = node.operation
        if op not in _OP_INDICATOR_PHRASES:
            continue
        candidates = _OP_INDICATOR_PHRASES[op]
        chosen: Optional[str] = None
        for phrase in candidates:
            phrase_words = re.findall(r"[a-zA-Z]+", phrase)
            if any(w in claimed for w in phrase_words):
                continue
            if " " in phrase:
                if phrase in surface_lower:
                    chosen = phrase
                    break
            else:
                if phrase in surface_tokens:
                    chosen = phrase
                    break
        if chosen:
            node.indicator = chosen
            for w in re.findall(r"[a-zA-Z]+", chosen):
                claimed.add(w)
            notes.append(
                f"inferred_{op}_indicator: {chosen!r} attached")
    return notes


def _walk_ops_missing_indicator(node, out: list) -> None:
    if isinstance(node, Op):
        if node.operation not in (
                "charade", "double_definition", "literal",
                "anagram",  # anagram has its own attacher
                "unknown",
        ) and not node.indicator:
            out.append(node)
        for c in node.sources:
            _walk_ops_missing_indicator(c, out)


# --- Link-word derivation (from surface; used to populate form.link_words) -

# Mirrors the global LINK_WORDS allow-list in verifier.py. Kept in sync
# manually so the translator can run standalone (no verifier import).
_LINK_WORDS = {
    "of", "in", "the", "a", "an", "to", "for", "with", "and", "or",
    "by", "from", "as", "on", "at", "but", "so", "yet", "if", "not",
    "nor", "up", "it", "its", "into", "onto", "within", "without",
    "that", "which", "when", "where", "while", "how", "why", "who",
    "this", "these", "those", "such", "one", "ones", "some", "any",
    "all", "here", "there",
    "is", "are", "be", "been", "being", "was", "were",
    "has", "have", "had", "having",
    "will", "would", "could", "should", "must", "may", "might",
    "get", "gets", "got", "getting",
    "give", "gives", "gave", "given", "giving",
    "make", "makes", "made", "making",
    "need", "needs",
    "thus", "hence", "therefore", "maybe",
    "dont", "doesnt", "didnt", "wont", "wouldnt", "cant", "isnt", "arent",
    "once",
}


def _derive_link_words(tree, definition_phrase: str,
                       clue_text: str) -> list:
    """Identify surface connectors as form.link_words ONLY when the form
    is otherwise structurally complete.

    Rule (agreed 2026-05-03): a surface word is a "link word" only when
    there is no other unaccounted word it could be part of. If the form
    leaves any non-link word unaccounted, none of the connectors get the
    link-word free pass either — they might actually be part of a missing
    indicator phrase (e.g. "taken from" is a deletion indicator; calling
    "from" a link word silently absorbs the wider gap).

    Implementation:
      1. Tokenise surface; classify each word as claimed / link-candidate
         (in LINK_WORDS) / unaccounted.
      2. If there are zero unaccounted words, the link-candidates become
         form.link_words.
      3. If there is even one unaccounted word, link-candidates are
         deferred too — they show up as 'unaccounted (link candidate)'
         in residue.
    """
    if not clue_text:
        return []
    claimed: set = set()
    for tok in _collect_tree_words(tree) + [definition_phrase]:
        for w in re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", (tok or "").lower()):
            claimed.add(w[:-2] if w.endswith("'s") else w)
    clue = re.sub(r"\s*\([\d,\-\s/]+\)\s*$", "", clue_text)
    surface = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", clue.lower())

    link_candidates: list = []
    has_unaccounted_non_link = False
    for w in surface:
        nw = w[:-2] if w.endswith("'s") else w
        if nw in claimed:
            continue
        if nw in _LINK_WORDS:
            link_candidates.append(nw)
        else:
            has_unaccounted_non_link = True

    if has_unaccounted_non_link:
        # Form is incomplete; don't claim any link words. Residue will
        # surface every gap, including connectors.
        return []
    return link_candidates


def _collect_tree_words(node) -> list:
    """Pre-order walk collecting source_words and indicators from leaves
    and ops. Local helper — does not import from schema beyond what's
    already in module scope."""
    out: list = []
    if isinstance(node, Leaf):
        out.append(node.source_word)
        if node.positional_indicator:
            out.append(node.positional_indicator)
        return out
    if isinstance(node, Op):
        if node.indicator:
            out.append(node.indicator)
        for c in node.sources:
            out.extend(_collect_tree_words(c))
    return out


def _parse_wordplay_types(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).lower().strip() for x in parsed if x]
        if isinstance(parsed, str):
            return [parsed.lower().strip()]
    except Exception:
        pass
    return [s.strip().lower() for s in raw.split(",") if s.strip()]


def _outer_op_from_wordplay_types(wt_list: list[str],
                                  asm_op: str) -> Optional[str]:
    """If wordplay_types lists an op that's NOT in assembly.op, that's the
    outer wrapper. Returns the outer op name or None.
    """
    asm_tokens = set(re.findall(r"[a-z_]+", asm_op))
    for wt in wt_list:
        if wt and wt not in asm_tokens and wt != asm_op:
            return wt
    return None


def _wrap_outer(inner: Node, outer_op: str, indicator: Optional[str],
                notes: list) -> Optional[Node]:
    if outer_op == "reversal":
        return reversal(inner, indicator=indicator or "")
    if outer_op == "container":
        # Outer container with one child is degenerate; skip.
        notes.append(
            f"outer container wrapper requested but no second child; "
            f"left as inner-only")
        return None
    if outer_op == "anagram":
        return anagram(inner, indicator=indicator or "")
    if outer_op == "deletion":
        return deletion(inner, indicator=indicator or "", kind="tail")
    if outer_op == "charade":
        # outer 'charade' usually means flat noise (multiple pieces). Don't
        # wrap; keep inner as-is.
        return None
    notes.append(f"unhandled outer op: {outer_op!r}")
    return None


def _patch_dd_answer(node: Node, answer: str) -> None:
    """The DD handler emits placeholder '?' for the synonym values. Once the
    Form's answer is known, propagate it into the DD leaves."""
    if isinstance(node, Op):
        if node.operation == "double_definition":
            for child in node.sources:
                if isinstance(child, Leaf) and child.value == "?":
                    child.value = answer.upper().replace(" ", "")
        for c in node.sources:
            _patch_dd_answer(c, answer)


# --- CLI / smoke test ------------------------------------------------------

if __name__ == "__main__":
    import sqlite3
    conn = sqlite3.connect("data/clues_master.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT se.id, se.components, se.wordplay_types, se.definition_text,
               se.confidence, se.model_version,
               c.clue_text, c.answer, c.ai_explanation
        FROM structured_explanations se JOIN clues c ON c.id=se.clue_id
        WHERE se.components IS NOT NULL
        LIMIT 20
    """).fetchall()
    flag_counts = {}
    for r in rows:
        result = translate(
            r["components"], r["wordplay_types"], r["definition_text"],
            r["answer"] or "", r["ai_explanation"])
        flag = result.report.flag
        flag_counts[flag] = flag_counts.get(flag, 0) + 1
        print(f'{r["model_version"]:<25} {r["answer"]:<15} '
              f'[{flag:<22}] {result.report.notes}')
    print()
    print("Flags:", flag_counts)
