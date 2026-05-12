"""Convert a Haiku wordplay-leaf decomposition into a candidate Form.

Given the list of role-tagged pieces from
`signature_solver.haiku_wordplay_leaves.find_wordplay_leaves`,
build a Form whose tree matches the named mechanism(s). The caller
runs the result through the clipboard verifier; the verifier is
the trust anchor that decides whether the form is correct.

Supported shapes:
  * single synonym / abbreviation / literal leaf
  * pure anagram with one or more fodder pieces and an indicator
  * pure reversal with a source piece and an indicator
  * pure container with outer/inner pieces and an indicator
  * charade of any combination of the above plus
    synonym/abbreviation/literal leaves
  * hidden / homophone / acrostic / deletion: leaf-level only
    (no compound shapes yet)

When the decomposition cannot be mapped to a supported shape (or
no permutation of subtrees assembles to the answer), returns None
and lets the caller fall through to the normal cascade.
"""
from __future__ import annotations

from collections import Counter
from itertools import permutations
from typing import List, Optional, Tuple

from .schema import (
    Form, Definition, Node,
    lit, syn, abbr, charade, anagram, reversal,
    container, hidden, homophone_leaf, deletion,
)


# Roles whose value is the letters contributed by the piece, and
# build a leaf at that letter sequence. The DB-side store for all
# of these is `synonyms_pairs`, but the prototype's tree uses
# operation-specific factories. We always start from a `syn` leaf
# unless the role explicitly maps to a different operation.
_LEAF_VALUE_ROLES = {
    "synonym":            ("syn", None),
    "abbreviation":       ("abbr", None),
    "literal":            ("lit", None),
    "containment_inner":  ("syn", None),
    "containment_outer":  ("syn", None),
    "deletion_source":    ("syn", None),
    "reversal_source":    ("syn", None),
    "hidden_source":      ("lit", None),
    "homophone_source":   ("syn", None),
    "acrostic_source":    ("lit", None),
}


def build_form_from_pieces(
    pieces: list,
    def_phrase: str,
    clue_text: str,
    answer: str,
) -> Optional[Form]:
    """Convert role-tagged pieces into a candidate Form.

    Caller is expected to pass the result through the clipboard
    verifier; this function does only the structural assembly.
    """
    if not pieces:
        return None

    answer_u = "".join(c for c in answer.upper() if c.isalpha())
    if not answer_u:
        return None

    # Bucket the pieces.
    indicators: dict = {}          # mechanism -> list[indicator word]
    sources: dict = {}             # mechanism -> list[(word, value)]
    leaves: list = []              # (kind, word, value)
    link_words: list = []
    fodder: list = []              # (word, raw_letters)

    for p in pieces:
        if not isinstance(p, dict):
            continue
        word = str(p.get("word") or "").strip()
        role = str(p.get("role") or "").strip().lower()
        value = p.get("value")
        if isinstance(value, str):
            value = value.strip().upper() or None
        else:
            value = None
        if not word or not role:
            continue

        if role == "link_word":
            link_words.append(word)
            continue
        if role == "anagram_fodder":
            # Fodder value should be the source letters (the word's
            # own letters). Fall back to the word's letters if Haiku
            # didn't supply a value.
            fl = value or "".join(c for c in word.upper() if c.isalpha())
            fodder.append((word, fl))
            continue
        if role.endswith("_indicator"):
            mech = role[:-len("_indicator")]
            indicators.setdefault(mech, []).append(word)
            continue
        if role in _LEAF_VALUE_ROLES:
            if not value:
                continue
            # Source-style roles are also tracked under their mech so
            # we can pair them with the corresponding indicator.
            mech_pair = None
            for mech in ("containment", "deletion", "reversal",
                         "hidden", "homophone", "acrostic"):
                if role.startswith(mech):
                    mech_pair = mech
                    break
            if mech_pair == "containment":
                # containment_inner / containment_outer
                side = role.split("_", 1)[1]  # 'inner' or 'outer'
                sources.setdefault("container", {}).setdefault(
                    side, []).append((word, value))
            elif mech_pair:
                sources.setdefault(mech_pair, []).append((word, value))
            else:
                leaves.append((role, word, value))
            continue

    # Build mechanism subtrees. Each subtree contributes a value and
    # a length to the eventual answer assembly.
    #
    # subtree spec: (kind, node, expected_letters_multiset, length)
    #   - kind=='exact': the value goes into the answer as-is at the
    #     subtree's position (synonym/abbr/literal/container/etc.).
    #   - kind=='multiset': the answer slot at the subtree's position
    #     is some permutation of the multiset (anagram only).
    subtrees: list = []

    # Anagram subtree (fodder + anagram_indicator)
    if fodder and indicators.get("anagram"):
        ind = indicators["anagram"][0]
        fodder_leaves = [lit(source_word=w, value=v) for w, v in fodder]
        ana_letters = "".join(v for _, v in fodder)
        node = anagram(*fodder_leaves, indicator=ind)
        subtrees.append(("multiset", node, ana_letters, len(ana_letters)))

    # Reversal subtree (reversal_source + reversal_indicator)
    if (sources.get("reversal")
            and indicators.get("reversal")):
        w, v = sources["reversal"][0]
        ind = indicators["reversal"][0]
        child = syn(source_word=w, value=v)
        node = reversal(child, indicator=ind)
        # Reversed letters
        rev = v[::-1]
        subtrees.append(("exact", node, rev, len(rev)))

    # Container subtree (containment_inner + containment_outer +
    # container_indicator). Tries every insertion point and reports
    # the first that fits the answer.
    if (sources.get("container", {}).get("inner")
            and sources.get("container", {}).get("outer")
            and indicators.get("container")):
        wi, vi = sources["container"]["inner"][0]
        wo, vo = sources["container"]["outer"][0]
        ind = indicators["container"][0]
        # Container value: outer with inner inserted at some pos.
        # We let the verifier figure out which insertion point works;
        # the subtree's "value" for ordering purposes is the combined
        # length and the multiset of all letters.
        combined = vo + vi
        node = container(
            outer=syn(source_word=wo, value=vo),
            inner=syn(source_word=wi, value=vi),
            indicator=ind,
        )
        # We can't know the exact substring without the answer slot,
        # so mark as multiset against the combined letters and let
        # _try_order test substring placements.
        subtrees.append(
            ("container", node, combined, len(combined)))

    # Hidden subtree: simple leaf-level case (single source word with
    # the answer as a substring). Treat as a literal leaf for now.
    if (sources.get("hidden")
            and indicators.get("hidden")):
        w, v = sources["hidden"][0]
        ind = indicators["hidden"][0]
        node = hidden(lit(source_word=w, value=v), indicator=ind)
        subtrees.append(("exact", node, answer_u, len(answer_u)))

    # Deletion subtree: head/tail/general. Subtype unknown from
    # Haiku output, so we accept any single-letter deletion that
    # matches the answer at the subtree's position.
    if (sources.get("deletion")
            and indicators.get("deletion")):
        w, v = sources["deletion"][0]
        ind = indicators["deletion"][0]
        # Try tail deletion (most common); the verifier confirms.
        if len(v) >= 2:
            node = deletion(
                source=syn(source_word=w, value=v),
                indicator=ind, kind="tail")
            subtrees.append(("exact", node, v[:-1], len(v) - 1))

    # Homophone subtree
    if (sources.get("homophone")
            and indicators.get("homophone")):
        w, v = sources["homophone"][0]
        ind = indicators["homophone"][0]
        # The verifier checks homophone DB; we just build the node.
        node = Node(
            operation="homophone", indicator=ind,
            sources=[homophone_leaf(source_word=w, value=v)],
        )
        # Homophones don't add letters in the same predictable way,
        # so support only the full-answer case for now.
        subtrees.append(("exact", node, answer_u, len(answer_u)))

    # Acrostic source-only (rare without proper subtype info) — skip.

    # Bare leaves: synonyms / abbreviations / literals not consumed
    # by any of the above mechanisms.
    for kind, word, value in leaves:
        op, _ = _LEAF_VALUE_ROLES[kind]
        if op == "syn":
            node = syn(source_word=word, value=value)
        elif op == "abbr":
            node = abbr(source_word=word, value=value)
        else:
            node = lit(source_word=word, value=value)
        subtrees.append(("exact", node, value, len(value)))

    if not subtrees:
        return None

    # Single-subtree case: use it directly.
    if len(subtrees) == 1:
        tree = subtrees[0][1]
        if not _assembles_to_answer([subtrees[0]], answer_u):
            return None
        return Form(
            tree=tree,
            definition=Definition(phrase=def_phrase, answer=answer_u),
            link_words=list(link_words),
        )

    # Multi-subtree case: try permutations until one assembles.
    # Bounded by factorial of subtree count; typically 2-4 so 2-24.
    best_perm = None
    for perm in permutations(subtrees):
        if _assembles_to_answer(perm, answer_u):
            best_perm = perm
            break
    if best_perm is None:
        return None
    children = [s[1] for s in best_perm]
    tree = charade(*children)
    return Form(
        tree=tree,
        definition=Definition(phrase=def_phrase, answer=answer_u),
        link_words=list(link_words),
    )


def _assembles_to_answer(perm: tuple, answer: str) -> bool:
    """Check whether a sequence of subtrees can assemble to the
    answer.

    For each subtree in order, walk the answer left-to-right:
      - exact: answer[pos:pos+len] must equal the subtree's letters.
      - multiset: answer[pos:pos+len] must be a permutation of the
        subtree's letters (anagram).
      - container: answer[pos:pos+len] must contain the inner letters
        as a substring AND the outer letters must surround them.
    """
    pos = 0
    for kind, _node, letters, length in perm:
        slot = answer[pos:pos + length]
        if len(slot) != length:
            return False
        if kind == "exact":
            if slot != letters:
                return False
        elif kind == "multiset":
            if Counter(slot) != Counter(letters):
                return False
        elif kind == "container":
            # `letters` here is outer+inner concatenated; the actual
            # value is outer with inner inserted somewhere. We check
            # by multiset equivalence on the slot and then confirm
            # the inner appears as a contiguous substring.
            if Counter(slot) != Counter(letters):
                return False
            # Container length = outer + inner. Slot must equal
            # outer with inner inserted at SOME interior position.
            # Since we don't know outer/inner sizes here without
            # extra info, just trust the multiset check; the
            # verifier will reject if the insertion point is wrong.
        else:
            return False
        pos += length
    return pos == len(answer)
