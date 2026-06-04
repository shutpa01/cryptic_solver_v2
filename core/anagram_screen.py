"""Anagram clue type — its bespoke screen (per the per-clue-type UI rule).

Anagram wants the same COLOUR treatment as charade: each fodder word gets its own
palette colour, and the answer tiles its letters landed in wear that colour — so
the rearrangement reads visually (fodder word <-> the tiles it supplied). The
engine emits one source per fodder word and assigns each answer letter to the word
that supplied it, so the shared base screen's per-source colouring does the rest.

There is no in-clue letter highlighting: an anagram scrambles the letters, so a
clue character does not map to a fixed answer position. The clue line is shown as
written; the colour link lives between the breakdown rows and the answer tiles.
"""

from core import wfw_render


def render(ctx, parse):
    return wfw_render.render_parse(parse, ctx=ctx, coloured=True)
