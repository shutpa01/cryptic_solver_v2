"""Anagram+container clue type — its bespoke screen (per the per-clue-type UI rule).

Same per-piece COLOUR as the other catalog screens: the outer piece (its two
segments, around the inner) wears one colour, the inner piece another, and the
answer tiles wear the colour of the piece that produced them. The engine emits one
source per piece, so the shared base screen's per-source colouring does the rest.
"""

from core import wfw_render


def render(ctx, parse):
    return wfw_render.render_parse(parse, ctx=ctx, coloured=True)
