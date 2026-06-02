"""Working replica clue page — enter a clue ID, run it through every engine.

Its own Flask app on its own port (5099), separate from the live site. You type
a clue id; it loads that clue and answer from clues_master.db, runs it through
all engines that currently exist (via core.engine_registry), and renders the
word-for-word result. This is the true test: real clues, real DB, real engines.

Run:  python -m core.wfw_web      then open  http://127.0.0.1:5099/
"""

import sqlite3
import os
from html import escape

from flask import Flask, request

from core import engine_registry
from core import wfw_render
from core import hidden_screen
from core import dd_screen
from core import admin_db
from core import store
from core.wfw_atoms import build_wfw_atom_context

# One base screen for every type; a type supplies only its specific changes.
SCREENS = {"hidden": hidden_screen.render, "dd": dd_screen.render}
_ENGINE_LABELS = {"hidden": "hidden", "dd": "double definition"}

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                  "data", "clues_master.db")

app = Flask(__name__)
_WIRING = None


def wiring():
    global _WIRING
    if _WIRING is None:
        _WIRING = engine_registry.make_db_wiring()
    return _WIRING


def _load_clue(clue_id):
    conn = sqlite3.connect(DB)
    try:
        row = conn.execute(
            "SELECT clue_text, answer, source, puzzle_number "
            "FROM clues WHERE id = ?",
            (clue_id,)).fetchone()
    finally:
        conn.close()
    return row


FORM = """
<form method="get" action="/" style="margin:1rem 0;font-family:system-ui">
  <label>Clue ID(s):
    <input name="id" value="{cid}" placeholder="e.g. 1710251, 1710240"
           style="font-size:1.1rem;padding:.3rem;width:22rem">
  </label>
  <button style="font-size:1.1rem;padding:.3rem .9rem">Run</button>
</form>
"""


@app.route("/")
def index():
    raw = (request.args.get("id") or "").strip()
    return _page(_body(raw))


@app.route("/admin", methods=["POST"])
def admin():
    raw = (request.form.get("id") or "").strip()
    kind = request.form.get("kind")
    if kind == "definition":
        msg = admin_db.add_definition(request.form.get("definition"),
                                      request.form.get("answer"))
    elif kind == "synonym":
        msg = admin_db.add_synonym(request.form.get("word"),
                                   request.form.get("synonym"))
    elif kind == "indicator":
        msg = admin_db.add_indicator(request.form.get("word"),
                                     request.form.get("type"))
    else:
        msg = "Unknown add."
    notice = '<div class="wfw-notice">%s</div>' % escape(msg)
    # The live DB checks mean a new definition/indicator is seen on this very
    # re-render — the carried clue ids below will reflect it immediately.
    return _page(notice + _body(raw))


def _body(raw):
    body = FORM.format(cid=escape(raw, quote=True)) + _admin_panel(raw)
    tokens = [t for t in raw.replace(",", " ").split() if t]
    for token in tokens:
        body += _render_one(token)
    return body


def _admin_panel(raw):
    """Collapsible admin panel: add a definition / synonym / indicator straight to
    the reference DB. Each form carries the current clue id(s) so the page re-runs
    them after the add, showing the effect at once."""
    cid = escape(raw, quote=True)
    types = ["hidden", "anagram", "container", "reversal", "deletion",
             "acrostic", "homophone", "charade"]
    opts = "".join('<option value="%s">%s</option>' % (t, t) for t in types)
    return f"""
<details class="wfw-admin">
  <summary>Admin &mdash; add to reference DB</summary>
  <form method="post" action="/admin" class="wfw-af">
    <input type="hidden" name="kind" value="definition">
    <input type="hidden" name="id" value="{cid}">
    <span class="wfw-af-l">Definition</span>
    <input name="definition" placeholder="nearest the bottom">
    <input name="answer" placeholder="NETHERMOST">
    <button>Add</button>
  </form>
  <form method="post" action="/admin" class="wfw-af">
    <input type="hidden" name="kind" value="indicator">
    <input type="hidden" name="id" value="{cid}">
    <span class="wfw-af-l">Indicator</span>
    <input name="word" placeholder="a little">
    <select name="type">{opts}</select>
    <button>Add</button>
  </form>
  <form method="post" action="/admin" class="wfw-af">
    <input type="hidden" name="kind" value="synonym">
    <input type="hidden" name="id" value="{cid}">
    <span class="wfw-af-l">Synonym</span>
    <input name="word" placeholder="word in clue">
    <input name="synonym" placeholder="value/answer fragment">
    <button>Add</button>
  </form>
</details>
"""


def _render_one(token):
    """Solve one clue id (which PERSISTS the Parse), then render straight from the
    DB — so the page can only show what was actually preserved."""
    try:
        clue_id = int(token)
    except ValueError:
        return f'<p class="warn">Not a numeric clue id: {escape(token)}</p>'
    row = _load_clue(clue_id)
    if row is None:
        return f'<p class="warn">No clue with id {clue_id}.</p>'
    clue_text, answer, src, pnum = row
    # Solve — this writes the Parse to the substrate of record (core.store).
    engine_registry.solve_clue_text(clue_text, answer, wiring(),
                                    source=src, puzzle_number=pnum, clue_id=clue_id)
    # Render from the persisted Parse, not the in-memory one.
    conn = store.connect()
    try:
        parse = store.load_parse(conn, clue_id)
    finally:
        conn.close()
    if parse is None:
        avail = ", ".join(_ENGINE_LABELS.get(k, k) for k in SCREENS)
        return (f'<div class="wfw-card"><div class="wfw-clue">{escape(clue_text)}'
                f'</div><p>Answer: <strong>{escape(answer)}</strong></p>'
                f'<p class="warn">No engine claimed this clue '
                f'(engines available: {escape(avail)}).</p></div>')
    ctx = build_wfw_atom_context(parse.clue_text, parse.answer_text)
    screen = SCREENS.get(parse.solved_by)
    return screen(ctx, parse) if screen else wfw_render.render_parse(parse, ctx=ctx)


def _page(body):
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WFW true test</title>
<style>{wfw_render.PAGE_CSS}
  .warn {{ color:#b00; font-weight:700; }}
  .tag {{ color:#888; font-size:.85rem; }}
  .wfw-notice {{ background:#ecfdf5; border:1px solid #6ee7b7; color:#065f46;
                 border-radius:10px; padding:.6rem .85rem; margin:.5rem 0;
                 font-size:.95rem; }}
  .wfw-admin {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px;
                padding:.5rem 1rem; margin:1rem 0; }}
  .wfw-admin summary {{ cursor:pointer; font-weight:700; color:#334155;
                        padding:.35rem 0; }}
  .wfw-af {{ display:flex; gap:.5rem; align-items:center; margin:.5rem 0;
             flex-wrap:wrap; }}
  .wfw-af-l {{ min-width:6rem; font-weight:700; color:#475569; font-size:.85rem; }}
  .wfw-af input, .wfw-af select {{ font-size:1rem; padding:.3rem .4rem;
             border:1px solid #cbd5e1; border-radius:6px; }}
  .wfw-af input {{ flex:1; min-width:8rem; }}
  .wfw-af button {{ font-size:.95rem; padding:.3rem .8rem; border-radius:6px;
             border:1px solid #2563eb; background:#2563eb; color:#fff;
             cursor:pointer; }}
</style></head><body>
  <p class="wfw-tag">word-for-word true test &middot; real clues, all engines</p>
  {body}
</body></html>"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5099, debug=False)
