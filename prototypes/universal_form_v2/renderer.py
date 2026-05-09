"""Render a Form as human-readable prose + extract wordplay-type path.

Two helpers:
  - wordplay_type(form): a string showing the operation tree path,
    e.g. 'container > reversal' or 'anagram > deletion + literal'.
    This is the universal-form's extension of the flat wordplay_type label.
  - render(form): a plain-English explanation a user could read.
"""
from __future__ import annotations

from .schema import Form, Node, LEAF_OPERATIONS


# --- Wordplay type (the path through the tree) ----------------------------

def wordplay_type(form: Form) -> str:
    """Return the operation path through the tree as a single string.

    Outer-op-first per the schema. Examples:
      anagram of literal -> "anagram"
      charade of [synonym, abbreviation, synonym] -> "charade"
      reversal of charade -> "reversal > charade"
      container of [synonym, reversal-of-synonym] -> "container > reversal"
      anagram of [literal, deletion-of-synonym] -> "anagram > deletion"
    """
    if form is None or form.tree is None:
        return "(no form)"
    return _node_type_path(form.tree)


def _node_type_path(node: Node) -> str:
    op = node.operation
    if op in LEAF_OPERATIONS:
        return op
    sources = node.sources or []
    inner_ops = []
    for s in sources:
        if s.operation in LEAF_OPERATIONS:
            continue
        inner_ops.append(_node_type_path(s))
    if not inner_ops:
        return op
    if len(inner_ops) == 1:
        return f"{op} > {inner_ops[0]}"
    # Multiple non-leaf children — list them
    return f"{op} > [{', '.join(inner_ops)}]"


# --- Human-readable rendering ---------------------------------------------

def render(form: Form) -> str:
    """Render a form as a multi-line prose explanation."""
    if form is None or form.tree is None:
        return "(no form)"
    answer = (form.definition.answer or "").upper()
    pieces = [_render_node(form.tree, answer)]
    pieces.append(f"= {answer}")
    if form.definition.phrase:
        pieces.append(f'definition: "{form.definition.phrase}"')
    return "\n".join(pieces)


def _render_node(node: Node, target: str) -> str:
    """Render one tree node as prose text. Recurses on children."""
    op = node.operation
    ind = node.indicator
    val = (node.value or "").upper()
    src = node.source_word

    if op == "literal":
        return f'"{src}"' if src else val
    if op == "raw":
        return f'{val} ("{src}")'
    if op == "synonym":
        return f'{val} (synonym of "{src}")'
    if op == "abbreviation":
        return f'{val} (abbreviation of "{src}")'
    if op == "homophone":
        return f'{val} (sounds like "{src}")'
    if op == "positional":
        kind = node.positional_kind or "first"
        labels = {
            "first": "first letter(s) of",
            "last":  "last letter(s) of",
            "outer": "outer letters of",
            "middle": "middle letter(s) of",
            "alternate": "alternate letters of",
            "odd": "odd letters of",
            "even": "even letters of",
            "half": "half of",
        }
        label = labels.get(kind, f"{kind} letters of")
        return f'{val} ({label} "{src}")'

    children_strs = [_render_node(c, target) for c in (node.sources or [])]

    if op == "charade":
        return " + ".join(children_strs)

    if op == "anagram":
        ind_str = f' ["{ind}"]' if ind else ' (anagram indicator missing)'
        if len(children_strs) == 1:
            return f'anagram{ind_str} of {children_strs[0]}'
        return f'anagram{ind_str} of [{", ".join(children_strs)}]'

    if op == "reversal":
        ind_str = f' ["{ind}"]' if ind else ' (reversal indicator missing)'
        return f'reverse{ind_str} of {children_strs[0]}'

    if op == "container":
        ind_str = f' ["{ind}"]' if ind else ' (container indicator missing)'
        if len(children_strs) >= 2:
            return f'{children_strs[1]} inside{ind_str} {children_strs[0]}'
        return f'container{ind_str} of [{", ".join(children_strs)}]'

    if op == "deletion":
        ind_str = f' ["{ind}"]' if ind else ' (deletion indicator missing)'
        kind = node.deletion_kind or "tail"
        kind_label = {"head": "first letter dropped",
                      "tail": "last letter dropped",
                      "outer": "outer letters dropped",
                      "heart": "middle dropped"}.get(kind, kind)
        return f'{children_strs[0]} with {kind_label}{ind_str}'

    if op == "hidden":
        ind_str = f' ["{ind}"]' if ind else ' (hidden indicator missing)'
        joined = " ".join(children_strs)
        return f'hidden{ind_str} in {joined}'

    if op == "double_definition":
        return " / ".join(children_strs)

    if op == "acrostic":
        kind = node.acrostic_kind or "first"
        ind_str = f' ["{ind}"]' if ind else ''
        return f'{kind}-letters{ind_str} of [{", ".join(children_strs)}]'

    return f'{op}({", ".join(children_strs)})'
