"""Palindrome clue type — the ONLY thing it changes on the base screen.

Per the design there is ONE base screen for every clue type (core/wfw_render); a type
changes only what is specific to it. For PALINDROME the specific thing is: highlight
the palindrome indicator in the clue line (the word(s) telling you to read both ways).
There is no clue-letter source to light — the answer's symmetry is the wordplay — so
the breakdown is definition + indicator, on the shared base.
"""

from html import escape
from core import wfw_render


def render(ctx, parse):
    """Render a palindrome Parse on the base screen with its indicator highlighted."""
    return wfw_render.render_parse(parse, ctx=ctx,
                                   clue_line_html=_clue_line(ctx, parse),
                                   coloured=False)


def _clue_line(ctx, parse):
    """The clue as written, with the palindrome indicator word(s) highlighted."""
    lit = set()
    for a in parse.annotations:
        if a.role == "indicator":
            lit.update(a.clue_atom_ids)
    out = []
    for atom in ctx.clue_atoms:
        ch = escape(atom.char)
        out.append('<span class="wfw-lit">%s</span>' % ch
                   if atom.atom_id in lit else ch)
    return "".join(out)
