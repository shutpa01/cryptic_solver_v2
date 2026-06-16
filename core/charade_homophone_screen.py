"""Charade + homophone clue type — its bespoke screen (per the per-clue-type UI rule).

Structurally a charade: the answer is assembled left-to-right from pieces, so it wants
the same thing charade does from the base screen — COLOUR: each piece gets a palette
colour and the answer tiles it produced wear the same colour, so the build reads
visually. The one piece that is a HOMOPHONE additionally carries a 'sounds like X' note
(from its link transform), which the base screen shows on that piece's row. The clue
line is shown as written — a piece's letters come from its DB value or its homophone
source, not from specific clue characters.

The provisional nature (a residue definition not DB-confirmed, or a Haiku-suggested
homophone source) is badged by the base screen: the PENDING verdict and per-row tag.
"""

from core import wfw_render


def render(ctx, parse):
    return wfw_render.render_parse(parse, ctx=ctx, coloured=True)
