"""Acrostic clue type — the ONLY thing it changes on the base screen.

Per the design there is ONE base screen for every clue type (core/wfw_render); a
type changes only what is specific to it. For ACROSTIC the single specific thing is:
light, in the clue line, the exact letters that were taken — the first (or last)
letter of each fodder word — where they actually sit. Everything else (tiles,
word-by-word breakdown stating each word's role, verdict) is the shared base.

The taken letters are precisely parse.links' clue_atom_ids (the §5.5 per-letter
provenance), so this mirrors the hidden screen's clue line.
"""

from html import escape
from core import wfw_render


def render(ctx, parse):
    """Render an acrostic Parse on the base screen: the selected letters lit in the
    clue line, uncoloured (single-letter pieces don't benefit from per-piece colour);
    the breakdown still states every word's role."""
    return wfw_render.render_parse(parse, ctx=ctx,
                                   clue_line_html=_clue_line(ctx, parse),
                                   coloured=False)


def _clue_line(ctx, parse):
    """The clue exactly as written, with the taken acrostic letters highlighted in
    place (the first/last letter of each fodder word)."""
    lit = {l.clue_atom_id for l in parse.links}
    out = []
    for atom in ctx.clue_atoms:
        ch = escape(atom.char)
        out.append('<span class="wfw-lit">%s</span>' % ch
                   if atom.atom_id in lit else ch)
    return "".join(out)
