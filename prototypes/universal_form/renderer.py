"""Form -> human-readable explanation in three sections.

Format (BACH-style, agreed 2026-05-03):

    ANSWER

    Definition: "<phrase>"

    Wordplay: <one-line summary of the mechanism>
      <step-by-step derivation, each step naming the indicator that
       licenses it>

    Word roles:
      <every clue word listed with its role>

The role table is the honesty surface - every surface word must have a
role. Anything not claimed by the form (and not in form.link_words) shows
as "(unaccounted)" so the reader sees the gap immediately.

Two output modes:

    render(form, clue_text)   the user-facing three-section explanation
    render_tree(form)         debug-friendly indented tree printout
"""
from __future__ import annotations

import re
from typing import Optional, Any  # noqa: F401

from .schema import Form, Leaf, Op, Node


# Mirrors the LINK_WORDS allow-list in verifier.py - used only to label
# leftover surface words as "link" vs "unaccounted" in the role table.
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render(form: Form, clue_text: str = "") -> str:
    """Three-section human-readable explanation."""
    answer = form.definition.answer
    lines: list[str] = []
    lines.append(answer)
    lines.append("")
    lines.append(f'Definition: "{form.definition.phrase}"')
    lines.append("")
    wordplay_type = compound_wordplay_type(form.tree)
    lines.append(f'Wordplay type: {wordplay_type}')
    lines.append("")
    lines.append("Derivation:")
    _, steps = _summarise_wordplay(form.tree, answer)
    for step in steps:
        lines.append(f'  {step}')
    lines.append("")
    lines.append("Word roles:")
    for w, role in _word_roles(form, clue_text):
        lines.append(f'  {w:<16}{role}')
    if form.is_and_lit:
        lines.append("")
        lines.append("Note: this is an &lit (the whole clue both defines "
                     "and gives the wordplay).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compound wordplay type naming
# ---------------------------------------------------------------------------
#
# Naming convention agreed 2026-05-03: chronologically EARLIEST operation
# first (so "deletion anagram" means deletion then anagram, not anagram
# then deletion). The tree is read inside-out; the deepest op is the
# earliest applied, the root is the latest.
#
# Examples:
#   charade(syn, abbr)                  -> "charade"
#   charade(positional[first] x N)      -> "acrostic"
#   anagram(deletion(literal), literal) -> "deletion anagram"
#   reversal(charade(positional[first])) -> "acrostic reversal"
#   container(syn, syn)                 -> "container"
#   double_definition(syn, syn)         -> "double definition"

def compound_wordplay_type(node: Node) -> str:
    """Return the compound wordplay type name for the form's tree."""
    if isinstance(node, Leaf):
        return _leaf_type_name(node)
    if not isinstance(node, Op):
        return "(unrecognised)"

    op = node.operation
    base = _op_base_name(op)

    # If this op has any sub-Op children, prefix their compound names
    # (chronologically earliest first; tree-wise, deepest first).
    op_children = [c for c in node.sources if isinstance(c, Op)]
    if not op_children:
        return base

    inner_names = [compound_wordplay_type(c) for c in op_children]
    # Multiple op-children: join them with ' + ' (e.g. anagram of
    # (deletion, reversal) → "deletion + reversal anagram"). Single op
    # child is the common case ("deletion anagram").
    inner = inner_names[0] if len(inner_names) == 1 \
        else " + ".join(inner_names)
    return f'{inner} {base}'


def _op_base_name(op: str) -> str:
    return {
        "charade": "charade",
        "container": "container",
        "anagram": "anagram",
        "reversal": "reversal",
        "deletion": "deletion",
        "hidden": "hidden",
        "double_definition": "double definition",
        "acrostic": "acrostic",
        "unknown": "(unknown)",
    }.get(op, op)


def _leaf_type_name(leaf: Leaf) -> str:
    op = leaf.operation
    if op == "positional":
        return f"positional[{leaf.positional_kind or '?'}]"
    return op


def render_tree(form: Form) -> str:
    """Debug printout - every node on a line, ASCII tree edges."""
    lines: list[str] = []
    _render_tree_node(form.tree, prefix="", is_last=True, lines=lines)
    lines.append("")
    lines.append(f'definition: "{form.definition.phrase}" -> '
                 f'{form.definition.answer}')
    if form.link_words:
        lines.append(f'link_words: {form.link_words}')
    if form.is_and_lit:
        lines.append("is_and_lit: True")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Wordplay summary + step-by-step derivation
# ---------------------------------------------------------------------------

def _summarise_wordplay(node: Node, answer: str) -> tuple[str, list[str]]:
    """Return (one-line headline, list of step lines).

    The headline names the type of wordplay. The steps walk the tree from
    the deepest leaves up to the answer, naming the indicator that
    licenses each operation.
    """
    if isinstance(node, Leaf):
        return _leaf_summary(node, answer), []

    if not isinstance(node, Op):
        return "(unrecognised tree)", []

    # Headline = "type of wordplay"
    summary = _node_type_summary(node)
    # Steps = derivation, deepest first
    steps = _derivation_steps(node, answer)
    return summary, steps


def _node_type_summary(node: Op) -> str:
    """One-line description of what kind of wordplay this is."""
    op = node.operation
    if op == "charade":
        n = len(node.sources)
        return f"charade - concatenation of {n} pieces"
    if op == "container":
        return "container - one piece wrapping another"
    if op == "anagram":
        return "anagram of one or more pieces"
    if op == "reversal":
        if node.sources and isinstance(node.sources[0], Op):
            inner = _node_type_summary(node.sources[0])
            return f"reversal of: {inner}"
        return "reversal of one piece"
    if op == "deletion":
        return "deletion (letter dropped from a piece)"
    if op == "hidden":
        return "hidden - the answer appears as a substring of the clue"
    if op == "double_definition":
        return "double definition - two phrases each define the answer"
    if op == "acrostic":
        kind = node.acrostic_kind or "first"
        n = len(node.sources)
        return (f"acrostic - {kind} letter of each of {n} consecutive "
                f"surface words")
    if op == "unknown":
        return f"(unrecognised operation: {node.flat_op or '?'})"
    return op


def _derivation_steps(node: Op, target_answer: str) -> list[str]:
    """Walk the tree bottom-up. Each step names the result and the
    indicator that licenses the operation."""
    steps: list[str] = []
    _emit_steps(node, steps, is_root=True, target_answer=target_answer)
    return steps


def _emit_steps(node: Node, steps: list[str], is_root: bool,
                target_answer: str) -> str:
    """Recurse: emit steps for children, then the step for this node.
    Returns the letters this node produces."""
    if isinstance(node, Leaf):
        return _leaf_value_for_step(node)

    if not isinstance(node, Op):
        return "?"

    child_values = [_emit_steps(c, steps, is_root=False,
                                target_answer=target_answer)
                    for c in node.sources]
    op = node.operation
    ind = node.indicator
    label = target_answer if is_root else _result_letters(node, child_values)

    if op == "charade":
        eq = " + ".join(child_values)
        steps.append(f'{eq} = {label}')
    elif op == "container":
        if len(child_values) == 2:
            o, i = child_values
            ind_phrase = f' (indicated by "{ind}")' if ind else ""
            steps.append(f'{o} wraps around {i} = {label}{ind_phrase}')
    elif op == "anagram":
        eq = " + ".join(child_values)
        ind_phrase = f' (indicated by "{ind}")' if ind else ""
        steps.append(f'anagram of {eq} = {label}{ind_phrase}')
    elif op == "reversal":
        if child_values:
            ind_phrase = f' (indicated by "{ind}")' if ind else ""
            steps.append(f'{child_values[0]} reversed = {label}{ind_phrase}')
    elif op == "deletion":
        if child_values:
            kind = node.deletion_kind or "letter(s)"
            kind_phrase = {
                "head":  "first letter dropped",
                "tail":  "last letter dropped",
                "outer": "first and last letters dropped",
                "heart": "middle letter(s) dropped",
            }.get(kind, f"{kind} dropped")
            ind_phrase = f' (indicated by "{ind}")' if ind else ""
            steps.append(
                f'{child_values[0]} with {kind_phrase} = {label}{ind_phrase}')
    elif op == "hidden":
        span = " ".join(c.source_word for c in node.sources
                        if isinstance(c, Leaf))
        ind_phrase = f' (indicated by "{ind}")' if ind else ""
        steps.append(f'{label} appears inside "{span}"{ind_phrase}')
    elif op == "double_definition":
        if len(node.sources) == 2:
            a, b = node.sources
            a_word = a.source_word if isinstance(a, Leaf) \
                else _result_letters(a, [])
            b_word = b.source_word if isinstance(b, Leaf) \
                else _result_letters(b, [])
            steps.append(f'"{a_word}" defines {label}')
            steps.append(f'"{b_word}" also defines {label}')
    elif op == "acrostic":
        kind = node.acrostic_kind or "first"
        words = [c.source_word for c in node.sources
                 if isinstance(c, Leaf)]
        kind_word = {"first": "first", "last": "last",
                     "middle": "middle"}.get(kind, kind)
        ind_phrase = f' (indicated by "{ind}")' if ind else ""
        steps.append(
            f'{kind_word} letter of each of '
            f'"{" / ".join(words)}" = {label}{ind_phrase}')

    return label


def _result_letters(node: Op, child_values: list) -> str:
    """Quick best-effort result letters for non-root nodes (used in
    intermediate step labels)."""
    op = node.operation
    if op == "charade":
        return "".join(child_values)
    if op == "anagram":
        return "(anagrammed)"
    if op == "reversal":
        return child_values[0][::-1] if child_values else "?"
    if op == "deletion":
        if not child_values:
            return "?"
        src = child_values[0]
        kind = node.deletion_kind
        if kind == "head":
            return src[1:]
        if kind == "tail":
            return src[:-1]
        if kind == "outer":
            return src[1:-1]
        return src[:-1]
    if op == "container":
        if len(child_values) != 2:
            return "?"
        outer, inner = child_values
        return outer + inner  # placeholder; verifier resolves position
    if op == "hidden":
        return "(hidden)"
    if op == "double_definition":
        return child_values[0] if child_values else "?"
    if op == "acrostic":
        # Compute the acrostic letters from the source_words on-the-fly
        kind = node.acrostic_kind or "first"
        out = []
        for c in node.sources:
            if isinstance(c, Leaf):
                src = re.sub(r"[^A-Z]", "",
                             (c.source_word or "").upper())
                if src:
                    if kind == "first":
                        out.append(src[0])
                    elif kind == "last":
                        out.append(src[-1])
                    elif kind == "middle":
                        out.append(src[len(src) // 2])
        return "".join(out)
    return "?"


def _leaf_value_for_step(leaf: Leaf) -> str:
    return leaf.value.upper().replace(" ", "")


def _leaf_summary(leaf: Leaf, answer: str) -> str:
    op = leaf.operation
    if op == "literal":
        return f'{answer}: the word "{leaf.source_word}" from the clue'
    if op == "synonym":
        return f'{answer}: synonym of "{leaf.source_word}"'
    if op == "abbreviation":
        return f'{answer}: abbreviation of "{leaf.source_word}"'
    if op == "positional":
        kind_phrase = _positional_phrase(leaf.positional_kind or "letter",
                                         len(leaf.value))
        return f'{answer}: {kind_phrase} of "{leaf.source_word}"'
    return f'{answer}: {op} of "{leaf.source_word}"'


def _positional_phrase(kind: str, n: int) -> str:
    if kind == "first":
        return "first letter" if n == 1 else f"first {n} letters"
    if kind == "last":
        return "last letter" if n == 1 else f"last {n} letters"
    if kind == "middle":
        return "middle letter" if n == 1 else f"middle {n} letters"
    if kind == "outer":
        return "outer letters"
    if kind == "initial":
        return "initial letter"
    if kind == "final":
        return "final letter"
    if kind == "odd":
        return "odd-numbered letters"
    if kind == "even":
        return "even-numbered letters"
    if kind == "alternate":
        return "alternate letters"
    return f"{kind} letter(s)"


# ---------------------------------------------------------------------------
# Word roles - every clue word with its role
# ---------------------------------------------------------------------------

def _word_roles(form: Form, clue_text: str) -> list[tuple[str, str]]:
    """Walk the surface words and assign each one a role from the form.

    Returns a list of (clue_word, role_description) in clue order.

    Role priority:
        1. Definition phrase
        2. Op indicator (named with the op type)
        3. Positional indicator
        4. Leaf source_word (named with the leaf op + value)
        5. form.link_words (declared link)
        6. (unaccounted) - visible signal that the form doesn't claim it
    """
    if not clue_text:
        return []
    # Strip enumeration
    clue = re.sub(r"\s*\([\d,\-\s/]+\)\s*$", "", clue_text)
    surface_tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", clue)
    if not surface_tokens:
        return []

    # Build a list of role-claims: (claim_words_lower, role_string)
    claims: list[tuple[list[str], str]] = []

    def _tok(s: str) -> list[str]:
        return [_normalise(w.lower()) for w in
                re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", s or "")]

    # 1. Definition
    if form.definition.phrase:
        claims.append((_tok(form.definition.phrase), "definition"))

    # 2 & 3. Indicators (op + positional). Also leaf source_words.
    def _walk(node):
        if isinstance(node, Leaf):
            # Leaf source_word claim
            label = _leaf_role_label(node)
            claims.append((_tok(node.source_word), label))
            if node.positional_indicator:
                claims.append((_tok(node.positional_indicator),
                               f"positional indicator ({node.positional_kind})"))
            return
        if isinstance(node, Op):
            if node.indicator:
                claims.append((_tok(node.indicator),
                               f"{node.operation} indicator"))
            for c in node.sources:
                _walk(c)
    _walk(form.tree)

    # 4. form.link_words declared
    for lw in form.link_words:
        claims.append((_tok(lw), "link word"))

    # Now assign each surface word to the FIRST matching claim. Each
    # claim's word slots are consumed once.
    claim_remaining = [list(words) for words, _ in claims]
    roles: list[tuple[str, str]] = []
    for w in surface_tokens:
        wn = _normalise(w.lower())
        assigned = None
        for i, (words, role) in enumerate(claims):
            if wn in claim_remaining[i]:
                claim_remaining[i].remove(wn)
                assigned = role
                break
        if assigned is None:
            # Not claimed by form. Show as unaccounted unless globally a
            # link word AND form.link_words didn't list it (which still
            # counts as a form gap, just a less serious one).
            if wn in _LINK_WORDS:
                assigned = "(link word - not declared in form.link_words)"
            else:
                assigned = "(unaccounted)"
        roles.append((w, assigned))
    return roles


def _leaf_role_label(leaf: Leaf) -> str:
    op = leaf.operation
    val = leaf.value
    if op == "literal":
        return f'source -> {val} (literal)'
    if op == "synonym":
        return f'source -> {val} (synonym)'
    if op == "abbreviation":
        return f'source -> {val} (abbreviation)'
    if op == "positional":
        kind = leaf.positional_kind or "?"
        return f'source -> {val} ({kind} letter)'
    return f'source -> {val} ({op})'


def _normalise(w: str) -> str:
    return w[:-2] if w.endswith("'s") else w


# ---------------------------------------------------------------------------
# Debug tree printer
# ---------------------------------------------------------------------------

def _render_tree_node(node: Node, prefix: str, is_last: bool,
                      lines: list[str]) -> None:
    branch = "+--"
    extension = "    " if is_last else "|   "

    if isinstance(node, Leaf):
        meta = ""
        if node.operation == "positional":
            meta = f' [{node.positional_kind}'
            if node.positional_indicator:
                meta += f', ind="{node.positional_indicator}"'
            meta += "]"
        lines.append(
            f'{prefix}{branch} {node.operation}: '
            f'"{node.source_word}" -> {node.value}{meta}')
        return

    meta = ""
    if node.indicator:
        meta = f' [ind="{node.indicator}"]'
    if node.operation == "deletion" and node.deletion_kind:
        meta += f' [{node.deletion_kind}]'
    if node.operation == "unknown" and node.flat_op:
        meta += f' [flat_op="{node.flat_op}"]'

    lines.append(f'{prefix}{branch} {node.operation}{meta}')

    children = list(node.sources)
    for i, child in enumerate(children):
        last_child = (i == len(children) - 1)
        _render_tree_node(child, prefix + extension, last_child, lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .examples import ALL_EXAMPLES
    EXAMPLE_CLUES = {
        "RIDE": "Republican fish",
        "REPAST": "Old man, among others, gets a meal",
        "REVOLTING": "Lover and nit, good back",
        "NEAPOLITAN": "Cook Antonio, mostly pale native of southern Italy",
        "FEAST": "Twist of fate's causing blow-out",
        "TAR": "Pitch that a rainstorm covers",
        "SULKY (DD)": "Light carriage being put out",
        "VINDICATION": "Start of victory sign for defence",
    }
    for name, form in ALL_EXAMPLES.items():
        print("=" * 78)
        print(f"  {name}  ({form.definition.answer})")
        print("=" * 78)
        print()
        print(render(form, EXAMPLE_CLUES.get(name, "")))
        print()
