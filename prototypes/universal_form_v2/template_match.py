"""Template matcher — apply a catalog template to a clue.

Given:
  - a catalog template (structural, abstract — see CATALOG_DESIGN.md)
  - a clue's words + per-word role analyses (from word_analyzer)
  - the answer (for known-correct verification)

The matcher enumerates valid layouts (which spans go to which leaves
and indicators), instantiates the template with concrete values from
the role analyses, and reports each combination whose assembly
produces the answer AND passes the strict verifier.

Used by catalog_runner.py to measure catalog utility on held-out clues.

No production touched. Read-only on RefDB and shadow.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from typing import Optional

from .schema import Form, Definition, Node, LEAF_OPERATIONS
from .strict_verifier import verify
from signature_solver.tokens import (
    SYN_F, ABR_F, RAW,
    ANA_I, REV_I, CON_I, DEL_I, HID_I, HOM_I,
    POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
    POS_I_ALTERNATE, POS_I_HALF,
)


# Map our template ops to the production indicator-token type
OP_TO_IND_TOKEN = {
    "anagram":   {ANA_I},
    "reversal":  {REV_I},
    "container": {CON_I},
    "deletion":  {DEL_I},
    "hidden":    {HID_I},
    "homophone": {HOM_I},
    "acrostic":  {POS_I_FIRST},
}

POS_KIND_TO_TOKEN = {
    "first":     {POS_I_FIRST},
    "last":      {POS_I_LAST},
    "outer":     {POS_I_OUTER},
    "middle":    {POS_I_MIDDLE},
    "alternate": {POS_I_ALTERNATE},
    "half":      {POS_I_HALF},
}


# Maximum span length per leaf/indicator slot (in clue words)
MAX_SPAN = 3


@dataclass
class SlotSpec:
    """One slot in a flattened template — either a leaf or an indicator."""
    kind: str            # "leaf" or "indicator"
    op: str              # operation (synonym, anagram, container, ...)
    positional_kind: Optional[str] = None
    deletion_kind: Optional[str] = None
    acrostic_kind: Optional[str] = None
    path: tuple = ()     # path in template (for re-assembly)


def flatten_template(template: dict) -> list:
    """Walk a catalog `structure` and return SlotSpecs in left-to-right
    order. Indicators come before their leaves' subtrees so that the
    layout assignment is easier (indicator anchors a position; leaves
    flank it)."""
    slots = []

    def walk(node, path):
        op = node["op"]
        is_leaf = node.get("leaf", False)
        if is_leaf:
            slots.append(SlotSpec(
                kind="leaf", op=op,
                positional_kind=node.get("positional_kind"),
                path=tuple(path),
            ))
            return
        # Op-node: emit indicator slot if op needs one
        if op in {"anagram", "reversal", "container", "deletion",
                  "hidden", "homophone", "acrostic"}:
            slots.append(SlotSpec(
                kind="indicator", op=op,
                positional_kind=None,
                deletion_kind=node.get("deletion_kind"),
                acrostic_kind=node.get("acrostic_kind"),
                path=tuple(path),
            ))
        for i, c in enumerate(node.get("children", [])):
            walk(c, path + ["children", i])

    walk(template, [])
    return slots


def _enumerate_disjoint_spans(n_words, span_lengths):
    """Yield every way to place `len(span_lengths)` disjoint contiguous
    spans on a timeline of `n_words`, where the i-th span has length
    span_lengths[i]. Spans may appear in any order — the assignment is
    `position_index -> (lo, hi)`.

    The remaining unused indices (not covered by any span) are returned
    as `free`.

    Yields: (spans, free_indices) where `spans` is a list of (lo, hi).
    """
    n = len(span_lengths)
    total = sum(span_lengths)
    if total > n_words:
        return
    # Treat each span as a "block" of fixed length. Place the n blocks
    # on the timeline in some order, with non-negative gaps between.
    # Equivalent to choosing n + 1 non-negative gaps that sum to
    # n_words - total, multiplied by all orderings of blocks.
    from itertools import permutations as _perm
    free_total = n_words - total
    for ordering in _perm(range(n)):
        # Distribute free_total words into n+1 gaps
        for gaps in _distribute(free_total, n + 1):
            spans = [None] * n
            pos = 0
            for g_idx, gap in enumerate(gaps):
                pos += gap
                if g_idx == n:
                    break
                slot_idx = ordering[g_idx]
                spans[slot_idx] = (pos, pos + span_lengths[slot_idx])
                pos += span_lengths[slot_idx]
            covered = set()
            for s in spans:
                for k in range(s[0], s[1]):
                    covered.add(k)
            free = [i for i in range(n_words) if i not in covered]
            yield spans, free


def _distribute(n_items, n_gaps):
    """Yield every way to distribute n_items into n_gaps (each 0+)."""
    if n_gaps == 1:
        yield [n_items]
        return
    for first in range(n_items + 1):
        for rest in _distribute(n_items - first, n_gaps - 1):
            yield [first] + rest


def _enumerate_assignments(n_words, slot_max_lengths):
    """Yield every (slot_spans, def_span, link_indices) layout.

    `slot_max_lengths` is a list of max lengths per slot — we enumerate
    each slot's length 1..max for that slot.
    """
    n_slots = len(slot_max_lengths)
    if n_slots == 0:
        for def_lo in range(n_words):
            for def_hi in range(def_lo + 1, n_words + 1):
                yield {
                    "slot_spans": [],
                    "def_span": (def_lo, def_hi),
                    "link_indices": [
                        i for i in range(n_words)
                        if not (def_lo <= i < def_hi)
                    ],
                }
        return
    # Enumerate slot lengths
    for slot_lengths in product(*[range(1, m + 1)
                                    for m in slot_max_lengths]):
        if sum(slot_lengths) >= n_words:
            continue  # need ≥ 1 word for definition
        # Place the slots
        for spans, free in _enumerate_disjoint_spans(
                n_words, list(slot_lengths)):
            # Definition: pick a contiguous run within `free`
            if not free:
                continue
            # Find runs of consecutive free indices
            runs = []
            cur = [free[0]]
            for k in free[1:]:
                if k == cur[-1] + 1:
                    cur.append(k)
                else:
                    runs.append(cur)
                    cur = [k]
            runs.append(cur)
            for run in runs:
                # Definition span lengths 1..len(run)
                for d_len in range(1, len(run) + 1):
                    for d_start in range(len(run) - d_len + 1):
                        d_lo = run[d_start]
                        d_hi = d_lo + d_len
                        link_idx = [
                            i for i in free
                            if not (d_lo <= i < d_hi)
                        ]
                        yield {
                            "slot_spans": list(spans),
                            "def_span": (d_lo, d_hi),
                            "link_indices": link_idx,
                        }


def _slot_value_candidates(slot: SlotSpec, span, words, single_analyses,
                            phrase_analyses) -> list:
    """For a given slot at `span = (lo, hi)`, return list of (token_str,
    candidate_values) tuples."""
    lo, hi = span
    span_words = words[lo:hi]
    span_text = " ".join(span_words).lower()

    if slot.kind == "leaf":
        if slot.op == "literal" or slot.op == "raw":
            # Value = letters of the span itself
            letters = "".join(c for c in span_text.upper() if c.isalpha())
            return [letters]
        if slot.op == "synonym":
            wa = (phrase_analyses.get((lo, hi)) if hi - lo > 1
                  else single_analyses[lo])
            return list(wa.roles.get(SYN_F, [])) if wa else []
        if slot.op == "abbreviation":
            wa = (phrase_analyses.get((lo, hi)) if hi - lo > 1
                  else single_analyses[lo])
            vals = []
            if wa:
                vals.extend(wa.roles.get(ABR_F, []))
                # Also try synonyms (some short synonyms live there)
                for v in wa.roles.get(SYN_F, []):
                    if 1 <= len(v) <= 3 and v not in vals:
                        vals.append(v)
            return vals
        if slot.op == "positional":
            kind = slot.positional_kind
            letters = "".join(c for c in span_text.upper() if c.isalpha())
            if kind == "first":
                return [letters[0]] if letters else []
            if kind == "last":
                return [letters[-1]] if letters else []
            if kind == "outer":
                if len(letters) >= 2:
                    return [letters[0] + letters[-1]]
                return []
            if kind == "middle":
                if len(letters) % 2 == 1 and len(letters) >= 3:
                    return [letters[len(letters) // 2]]
                if len(letters) % 2 == 0 and len(letters) >= 4:
                    mid = len(letters) // 2
                    return [letters[mid - 1:mid + 1]]
                return []
            if kind in ("alternate", "odd"):
                return [letters[0::2]] if letters else []
            if kind == "even":
                return [letters[1::2]] if letters else []
            return []
        if slot.op == "homophone":
            wa = (phrase_analyses.get((lo, hi)) if hi - lo > 1
                  else single_analyses[lo])
            return list(wa.roles.get("HOM_F", [])) if wa else []
        return []

    # Indicator slot — value is the word(s); we just need to check the
    # span has a matching indicator role
    expected_tokens = OP_TO_IND_TOKEN.get(slot.op, set())
    wa = (phrase_analyses.get((lo, hi)) if hi - lo > 1
          else single_analyses[lo])
    if not wa:
        return []
    if any(tok in wa.roles for tok in expected_tokens):
        return [span_text]
    return []


def _build_form(template: dict, slots: list, layout: dict,
                values: list, words: list, definition_phrase: str,
                answer: str) -> Optional[Form]:
    """Build a concrete Form by walking the template and substituting
    values + indicators per the layout."""

    def walk(node, path):
        op = node["op"]
        is_leaf = node.get("leaf", False)
        if is_leaf:
            # Find this leaf's slot
            for i, s in enumerate(slots):
                if s.kind == "leaf" and s.path == tuple(path):
                    span = layout["slot_spans"][i]
                    val = values[i]
                    src = " ".join(words[span[0]:span[1]])
                    n = Node(operation=op, value=val, source_word=src,
                              positional_kind=s.positional_kind)
                    return n
            return None
        # Non-leaf op
        children = [walk(c, path + ["children", i])
                    for i, c in enumerate(node.get("children", []))]
        if any(c is None for c in children):
            return None
        indicator = None
        # Find this node's indicator slot, if any
        for i, s in enumerate(slots):
            if s.kind == "indicator" and s.path == tuple(path):
                span = layout["slot_spans"][i]
                indicator = " ".join(words[span[0]:span[1]])
                break
        n = Node(operation=op, indicator=indicator, sources=children,
                  deletion_kind=node.get("deletion_kind"),
                  acrostic_kind=node.get("acrostic_kind"))
        return n

    tree = walk(template, [])
    if tree is None:
        return None
    link_words = [words[i] for i in layout["link_indices"]]
    return Form(tree=tree,
                 definition=Definition(phrase=definition_phrase,
                                       answer=answer),
                 link_words=link_words)


def match_template(template: dict, words: list, single_analyses: list,
                    phrase_analyses: dict, answer: str,
                    db, shadow_conn=None) -> list:
    """Return list of (Form, verdict) for every layout/value combo that
    produces the answer AND strict-verifies."""
    slots = flatten_template(template)
    n_words = len(words)
    n_slots = len(slots)
    if n_words < n_slots + 1:  # +1 for definition
        return []

    # Pre-filter 1: every required indicator-token type must appear
    # somewhere in the clue's analyses, else the template can never fit.
    needed_tokens = set()
    for s in slots:
        if s.kind == "indicator":
            needed_tokens.update(OP_TO_IND_TOKEN.get(s.op, set()))
        elif s.kind == "leaf" and s.op == "positional":
            needed_tokens.update(POS_KIND_TO_TOKEN.get(
                s.positional_kind, set()))
    if needed_tokens:
        present = set()
        for wa in single_analyses:
            present.update(wa.roles.keys())
        for wa in phrase_analyses.values():
            present.update(wa.roles.keys())
        if not (needed_tokens & present):
            return []

    # Slot length policy:
    #   - leaves get 1-2 words (most are single, some 2-word phrases)
    #   - indicators get 1-2 words ("knocked back", "to the right")
    slot_max_lengths = []
    for s in slots:
        if s.kind == "leaf" and s.op in ("literal", "raw"):
            slot_max_lengths.append(2)
        elif s.kind == "leaf":
            slot_max_lengths.append(2)
        else:  # indicator
            slot_max_lengths.append(2)

    matches = []
    layouts_tried = 0
    for layout in _enumerate_assignments(n_words, slot_max_lengths):
        layouts_tried += 1
        if layouts_tried > 5000:
            break
        # For each slot, gather candidate values
        candidates_per_slot = []
        ok = True
        for i, slot in enumerate(slots):
            span = layout["slot_spans"][i]
            cands = _slot_value_candidates(
                slot, span, words, single_analyses, phrase_analyses)
            if not cands:
                ok = False
                break
            candidates_per_slot.append(cands)
        if not ok:
            continue
        # Cap product size
        prod = 1
        for c in candidates_per_slot:
            prod *= len(c)
        if prod > 200:
            continue
        # Build definition phrase
        d_lo, d_hi = layout["def_span"]
        def_phrase = " ".join(words[d_lo:d_hi])
        # Try every value combination
        for values in product(*candidates_per_slot):
            form = _build_form(template, slots, layout, list(values),
                                 words, def_phrase, answer)
            if form is None:
                continue
            # Quick assembly check (saves time on full verify)
            from .strict_verifier import _produces
            if not _produces(form.tree, answer.upper()):
                continue
            v = verify(form, " ".join(words), db, shadow_conn)
            if v.verdict == "PASS":
                matches.append((form, v))
                if len(matches) >= 3:
                    return matches
    return matches
