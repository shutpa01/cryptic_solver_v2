"""Homophone clue type — the ONLY thing it changes on the base screen.

Per the design there is ONE base screen for every clue type (core/wfw_render); a
type changes only what is specific to it. For HOMOPHONE the answer is the SOUND of
one source word, so there are no per-letter clue characters (the links carry
clue_atom_id=None). The single type-specific touch is therefore to light the SOURCE
word in the clue line — the word whose sound is the answer — rather than individual
letters. Everything else (tiles, word-by-word breakdown stating each word's role,
the "sounds like X" note from the link transform, verdict) is the shared base.

Mirrors the acrostic/hidden screens' clue-line treatment, but the lit span is the
whole source word, not selected letters.
"""
from html import escape
from core import wfw_render


def render(ctx, parse):
    """Render a homophone Parse on the base screen: the source word lit in the clue
    line, uncoloured (one source, the amber accent suffices); the breakdown still
    states every word's role and what the source sounds like."""
    return wfw_render.render_parse(parse, ctx=ctx,
                                   clue_line_html=_clue_line(ctx, parse),
                                   coloured=False)


def _clue_line(ctx, parse):
    """The clue exactly as written, with the SOURCE word (whose sound is the answer)
    highlighted in place."""
    lit = set(parse.sources[0].clue_atom_ids) if parse.sources else set()
    out = []
    for atom in ctx.clue_atoms:
        ch = escape(atom.char)
        out.append('<span class="wfw-lit">%s</span>' % ch
                   if atom.atom_id in lit else ch)
    return "".join(out)
