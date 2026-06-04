"""Anagram+charade clue type — its bespoke screen (per the per-clue-type UI rule).

A charade with one anagram piece, so it wants the same per-piece COLOUR as charade:
each piece (the anagram span included) gets its own colour and the answer tiles it
produced wear it. The anagram piece reads "Anagram of <fodder>"; the others read as
synonym/abbreviation pieces. The shared base screen does this from the per-source
colouring; the engine emits one source per piece.
"""

from core import wfw_render


def render(ctx, parse):
    return wfw_render.render_parse(parse, ctx=ctx, coloured=True)
