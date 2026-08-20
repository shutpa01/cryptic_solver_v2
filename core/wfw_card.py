"""The rendered WFW card for a STORED parse — shared by the solver's pages and
the public site (week-only relaunch, 2026-07-13).

The public clue page must show THE SAME card the solver shows (user: "base it
on what we have") — same renderers, same screens, same colours — so the card
builder lives here in core and both callers use it. The import chain is
deliberately light: store -> wfw_model/wfw_atoms, screens -> wfw_render, and
wfw_render imports nothing from core at module level. NO engine wiring is
touched — rendering a stored parse never solves anything.

SCREENS and _manual_hidden_line moved here from core/wfw_web.py (which now
imports them back) so the site does not have to import the heavy solver app.
"""

from html import escape

from core import wfw_render
from core import (acrostic_screen, anagram_charade_screen, anagram_container_screen,
                  anagram_screen, charade_homophone_screen, charade_screen, dd_screen,
                  hidden_screen, homophone_screen, palindrome_screen, spoonerism_screen)
from core.wfw_atoms import build_wfw_atom_context


def _manual_hidden_line(ctx, parse):
    """Lit clue line for a MANUAL/PREFILL parse of a hidden clue, or None.

    The hidden ENGINE's parses carry per-letter links, and its screen lights the
    host letters (de[BRIE]fs). Manual and prefill parses store the whole host
    phrase as one raw piece with a 'hidden indicator' annotation — no per-letter
    links — so the generic renderer showed a plain clue line (regression seen
    2026-07-13 on SALSA/KARACHI/NOTIFIED prefills). Here the run is DERIVED, not
    guessed: only when the parse carries a hidden indicator, and only for a piece
    whose value sits as a contiguous run (forward, or reversed for the
    hidden-reversed case) strictly inside that piece's own clue letters. Anything
    else returns None and the caller renders the plain line."""
    notes = " ".join((a.note or "") for a in parse.annotations
                     if getattr(a, "role", "") == "indicator").lower()
    if "hidden indicator" not in notes:
        return None
    atom_by_id = {a.atom_id: a for a in ctx.clue_atoms}
    lit = set()
    for s in parse.sources:
        atoms = sorted((atom_by_id[i] for i in s.clue_atom_ids
                        if i in atom_by_id and atom_by_id[i].kind == "letter"),
                       key=lambda a: a.index)
        letters = "".join(a.normalized for a in atoms)
        val = "".join(c for c in (s.value or "").upper() if c.isalpha())
        if not val or len(val) >= len(letters):      # not a host with a run INSIDE it
            continue
        pos = letters.find(val)
        if pos < 0:
            pos = letters.find(val[::-1])            # hidden reversed
        if pos < 0:
            continue
        lit.update(a.atom_id for a in atoms[pos:pos + len(val)])
    if not lit:
        return None
    return "".join('<span class="wfw-lit">%s</span>' % escape(a.char)
                   if a.atom_id in lit else escape(a.char)
                   for a in ctx.clue_atoms)


SCREENS = {"hidden": hidden_screen.render, "acrostic": acrostic_screen.render,
           "homophone": homophone_screen.render,
           "dd": dd_screen.render,
           # a manual/prefill DD promoted from def+whole-answer-synonym carries
           # solved_by='manual'/'prefill', so dispatch on the OPERATION too, not just
           # solved_by='dd' — both land on the same DD screen.
           "double_definition": dd_screen.render,
           "charade": charade_screen.render, "anagram": anagram_screen.render,
           "anagram_charade": anagram_charade_screen.render,
           "anagram_container": anagram_container_screen.render,
           "container": anagram_container_screen.render,
           "container_charade": anagram_container_screen.render,
           "charade_homophone": charade_homophone_screen.render,
           "palindrome": palindrome_screen.render,
           "spoonerism": spoonerism_screen.render}


def render_stored_parse(parse, ctx=None, comment=""):
    """The card HTML for an already-loaded Parse — the ONE dispatch used by the
    solver's clue page, /hs, and the public site.

    `comment` is the clue's reviewer note. Only a REVERSE ANAGRAM renders it (the
    comment IS that clue type's explanation); every other type ignores it."""
    if ctx is None:
        ctx = build_wfw_atom_context(parse.clue_text, parse.answer_text)
    screen = SCREENS.get(parse.operation) or SCREENS.get(parse.solved_by)
    if screen:
        return screen(ctx, parse)
    return wfw_render.render_parse(parse, ctx=ctx, comment=comment,
                                   clue_line_html=_manual_hidden_line(ctx, parse))


def stored_card(clue_id, db_path=None):
    """The rendered card for this clue's stored PASS parse, or None.

    None when: no stored parse, not a pass, or a junk pass row with no pieces
    at all (a handful of pre-schema rows) — the public serving rule treats
    None as 'this page does not exist' (410)."""
    from core import store
    conn = store.connect(db_path)
    try:
        parse = store.load_parse(conn, clue_id)
        # The reviewer's comment. A reverse anagram carries its explanation there and
        # nowhere else, so the card must have it; read it here, where the connection is
        # already open, rather than mirroring the lookup in each caller.
        comment = store.get_note(conn, clue_id) or ""
    finally:
        conn.close()
    if parse is None or (getattr(parse, "status", "") or "") != "pass":
        return None
    if not (parse.sources or parse.definition):
        return None
    return render_stored_parse(parse, comment=comment)
