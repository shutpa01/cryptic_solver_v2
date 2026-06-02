"""Double-definition clue type — its bespoke change on the shared base screen.

Per the per-clue-type UI rule: one base screen (core/wfw_render), each type changes
only what is specific to it. For DD, the specific thing is the clue line: the two
definition halves are highlighted in the two source colours (matching their rows
in the word-by-word breakdown), so the user sees at a glance that the clue is two
definitions of the same answer. Everything else — tiles, breakdown, verdict — is
the shared base. Rendered coloured so each half + its breakdown row share a colour.
"""

from html import escape
from core import wfw_render


def render(ctx, parse):
    return wfw_render.render_parse(parse, ctx=ctx,
                                   clue_line_html=_clue_line(ctx, parse),
                                   coloured=True)


def _clue_line(ctx, parse):
    """The clue text, with each definition half tinted its source colour."""
    colour = {}
    for si, s in enumerate(parse.sources[:len(wfw_render.PALETTE)]):
        fg, fill = wfw_render._colour(si)
        for aid in s.clue_atom_ids:
            colour[aid] = (fg, fill)
    out = []
    for atom in ctx.clue_atoms:
        ch = escape(atom.char)
        if atom.atom_id in colour:
            fg, fill = colour[atom.atom_id]
            out.append('<span style="background:%s;color:%s;border-radius:3px;'
                       'padding:0 .05em">%s</span>' % (fill, fg, ch))
        else:
            out.append(ch)
    return "".join(out)
