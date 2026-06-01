"""Hidden clue type — the ONLY thing it changes on the base screen.

Per the design: there is ONE base screen for every clue type (core/wfw_render),
so the user always sees the same layout. A type changes only what is specific to
it. For HIDDEN, the single specific thing is: light the answer's letters where
they actually sit inside the host word in the clue line (de[BRIE]fs). Everything
else — tiles, word breakdown, verdict — is the shared base, untouched.

So this module builds just the hidden clue line and hands it to the base.
"""

from html import escape
from core import wfw_render


def render(ctx, parse):
    """Render a hidden Parse on the base screen.

    Hidden's two specifics: the host letters are lit in the clue line, and the
    screen is UNCOLOURED (per-piece colour is awkward for a hidden run). The
    word-by-word breakdown still states every word's role.
    """
    return wfw_render.render_parse(parse, ctx=ctx,
                                   clue_line_html=_clue_line(ctx, parse),
                                   coloured=False)


def _clue_line(ctx, parse):
    """The clue text exactly as written, but with the hidden answer letters
    highlighted in place inside the host word(s)."""
    lit = {l.clue_atom_id for l in parse.links}      # the exact clue chars used
    out = []
    for atom in ctx.clue_atoms:
        ch = escape(atom.char)
        if atom.atom_id in lit:
            out.append('<span class="wfw-lit">%s</span>' % ch)
        else:
            out.append(ch)
    return "".join(out)
