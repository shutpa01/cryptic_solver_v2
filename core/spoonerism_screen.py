"""Spoonerism clue type — the ONLY thing it changes on the base screen.

For SPOONERISM the specific thing is: highlight the Spooner indicator and the source
word(s) in the clue line (the words whose initial sounds, swapped, sound like the
answer). The match is by sound, so there is no per-letter answer provenance — the
breakdown is definition + indicator + source, on the shared base.
"""

from html import escape
from core import wfw_render


def render(ctx, parse):
    """Render a spoonerism Parse on the base screen with its indicator + source lit."""
    return wfw_render.render_parse(parse, ctx=ctx,
                                   clue_line_html=_clue_line(ctx, parse),
                                   coloured=False)


def _clue_line(ctx, parse):
    """The clue as written, with the indicator and source word(s) highlighted."""
    lit = set()
    for a in parse.annotations:
        if a.role == "indicator":
            lit.update(a.clue_atom_ids)
    for s in parse.sources:
        if s.mechanism == "spoonerism":
            lit.update(s.clue_atom_ids)
    out = []
    for atom in ctx.clue_atoms:
        ch = escape(atom.char)
        out.append('<span class="wfw-lit">%s</span>' % ch
                   if atom.atom_id in lit else ch)
    return "".join(out)
