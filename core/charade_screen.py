"""Charade clue type — its bespoke screen (per the per-clue-type UI rule).

Charade is meaning-based and assembled left-to-right, so the one thing it wants
from the base screen is COLOUR: each fodder word gets its own palette colour and
the answer tiles it produced wear the same colour, so the build reads visually
(word <-> tiles). There is no in-clue letter highlighting — a charade piece's
letters come from its DB value (synonym/abbreviation/literal), not from specific
clue characters — so the clue line is shown exactly as written.

The pending nature (a provisional Haiku definition, or a provisional piece) is
recognised and badged by the base screen: the PENDING verdict, the provisional
banner, and the per-row "provisional" tag.
"""

from core import wfw_render


def render(ctx, parse):
    return wfw_render.render_parse(parse, ctx=ctx, coloured=True)
