"""Working replica clue page — enter a clue ID, run it through every engine.

Its own Flask app on its own port (5099), separate from the live site. You type
a clue id; it loads that clue and answer from clues_master.db, runs it through
all engines that currently exist (via core.engine_registry), and renders the
word-for-word result. This is the true test: real clues, real DB, real engines.

Per clue you also get, inline:
  - the ENRICHMENT NEEDED rows (the clue's own queued pending_enrichments), each
    editable, with Approve (-> reference DB) / Reject (-> rejected_enrichments);
  - an ADMIN add panel (definition / synonym / indicator) on EVERY clue, pass or
    fail, for adding straight to the reference DB.
A single-clue re-run re-solves only that clue and keeps the rest of the batch on
screen (the others render from their already-stored result).

Run:  python -m core.wfw_web      then open  http://127.0.0.1:5099/
"""

import sqlite3
import os
from html import escape

from flask import Flask, request

from core import engine_registry
from core import wfw_render
from core import hidden_screen
from core import acrostic_screen
from core import homophone_screen
from core import dd_screen
from core import charade_screen
from core import anagram_screen
from core import anagram_charade_screen
from core import anagram_container_screen
from core import palindrome_screen
from core import spoonerism_screen
from core import admin_db
from core import store
from core.wfw_atoms import build_wfw_atom_context

SCREENS = {"hidden": hidden_screen.render, "acrostic": acrostic_screen.render,
           "homophone": homophone_screen.render,
           "dd": dd_screen.render,
           "charade": charade_screen.render, "anagram": anagram_screen.render,
           "anagram_charade": anagram_charade_screen.render,
           "anagram_container": anagram_container_screen.render,
           "container": anagram_container_screen.render,
           "container_charade": anagram_container_screen.render,
           "palindrome": palindrome_screen.render,
           "spoonerism": spoonerism_screen.render}
_ENGINE_LABELS = {"hidden": "hidden", "dd": "double definition",
                  "charade": "charade", "anagram": "anagram",
                  "anagram_charade": "anagram + charade",
                  "anagram_container": "anagram + container",
                  "palindrome": "palindrome",
                  "spoonerism": "spoonerism",
                  "substitution": "substitution"}
# indicator types offered in the per-clue admin panel + enrichment edit.
_IND_TYPES = ["hidden", "anagram", "container", "reversal", "deletion",
              "acrostic", "homophone", "charade"]

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                  "data", "clues_master.db")

app = Flask(__name__)
_WIRING = None


def wiring():
    """The full wiring, AI ON. Used only for an explicit per-clue re-run, so AI is
    on-demand, never part of a batch solve."""
    global _WIRING
    if _WIRING is None:
        _WIRING = engine_registry.make_db_wiring()
    return _WIRING


def batch_wiring():
    """DB-only view of the wiring (every AI touch-point nulled) for batch solving, so
    running a whole puzzle makes zero AI calls and is instant. Shares the full
    wiring's DB connection and caches, so it is free to build each call."""
    return engine_registry.db_only(wiring())


def reload_wiring():
    """Drop the cached wiring and rebuild it, so reference-DB edits made since
    startup are picked up WITHOUT restarting the server."""
    global _WIRING
    _WIRING = None
    return wiring()


def _load_clue(clue_id):
    conn = sqlite3.connect(DB)
    try:
        return conn.execute(
            "SELECT clue_text, answer, source, puzzle_number "
            "FROM clues WHERE id = ?", (clue_id,)).fetchone()
    finally:
        conn.close()


FORM = """
<form method="get" action="/" style="margin:1rem 0;font-family:system-ui">
  <label>Clue ID(s):
    <input name="id" value="{cid}" placeholder="e.g. 1710251, 1710240"
           style="font-size:1.1rem;padding:.3rem;width:22rem">
  </label>
  <button style="font-size:1.1rem;padding:.3rem .9rem">Run</button>
</form>
"""

RELOAD_FORM = """
<form method="post" action="/reload" style="display:inline-block;margin:0 0 1rem">
  <input type="hidden" name="id" value="{cid}">
  <button class="wfw-reload" title="Rebuild the in-memory DB snapshot from disk, then re-run every clue above">&#8635; Reload DB &amp; re-run all</button>
</form>
"""


@app.route("/")
def index():
    raw = (request.args.get("id") or "").strip()
    return _page(_body(raw))                       # fresh run: solve all


@app.route("/reload", methods=["POST"])
def reload_route():
    """Rebuild the DB snapshot, then re-run. `only` (a per-clue button) re-solves
    JUST that clue and keeps the rest of the batch on screen; otherwise re-solve all."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    reload_wiring()
    notice = '<div class="wfw-notice">Reference DB reloaded from disk.</div>'
    # A per-clue re-run (only set) is an explicit on-demand action -> AI on. A
    # re-run-all (no only) is a batch -> DB-only, no AI.
    return _page(notice + _body(raw, resolve_only={only} if only else None,
                                ai=bool(only)),
                 scroll_to=only)


@app.route("/admin", methods=["POST"])
def admin():
    """Add an entry to the reference DB from a clue's admin panel, then re-solve just
    that clue (batch preserved)."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    msg = _do_add(request.form)
    # No reload_wiring here: definitions/indicators are seen on the re-solve via the
    # wiring's LIVE DB check, so the effect shows at once without the costly ~1GB RefDB
    # rebuild. A new SYNONYM (RefDB-snapshotted) needs the per-clue "Reload DB & re-run".
    notice = '<div class="wfw-notice">%s</div>' % escape(msg)
    return _page(notice + _body(raw, resolve_only={only} if only else None),
                 scroll_to=only)


@app.route("/enrich", methods=["POST"])
def enrich():
    """Approve one queued enrichment: write the (possibly edited) values to the
    reference DB, drop it from the queue, then re-solve just that clue. Same as /admin
    re the live-vs-reload trade-off (no RefDB rebuild here)."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    pid = (request.form.get("pending_id") or "").strip()
    msg = _do_add(request.form)
    if pid and not msg.startswith(("Definition and", "Word and", "Indicator word")):
        admin_db.delete_pending(pid)               # accepted -> leave the queue
    notice = '<div class="wfw-notice">%s</div>' % escape(msg)
    # Approving an enrichment then re-solving that one clue is an on-demand action.
    return _page(notice + _body(raw, resolve_only={only} if only else None,
                                ai=bool(only)),
                 scroll_to=only)


@app.route("/reject", methods=["POST"])
def reject():
    """Reject one queued enrichment (-> rejected_enrichments). No DB add, no re-solve;
    the batch re-renders from stored results and the enrichment list refreshes."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    pid = (request.form.get("pending_id") or "").strip()
    msg = admin_db.reject_pending(pid) if pid else "Nothing to reject."
    notice = '<div class="wfw-notice">%s</div>' % escape(msg)
    return _page(notice + _body(raw, resolve_only=set()),   # render all from store
                 scroll_to=only)


@app.route("/setstatus", methods=["POST"])
def setstatus():
    """Manually override one clue's verdict. Re-renders from store (no re-solve), so
    the manual status sticks until the clue is explicitly re-run."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    status = (request.form.get("status") or "").strip()
    msg = "No clue/status."
    if only and status in ("pass", "pending", "fail"):
        conn = store.connect()
        try:
            store.set_status(conn, int(only), status)
        finally:
            conn.close()
        msg = "Status of clue %s set to %s." % (only, status)
    notice = '<div class="wfw-notice">%s</div>' % escape(msg)
    return _page(notice + _body(raw, resolve_only=set()), scroll_to=only)


@app.route("/setdef", methods=["POST"])
def setdef():
    """Set a DISPLAY-ONLY definition for one clue (no reference-DB write, no checks) —
    for &lit clues. Re-renders from store (no re-solve)."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    text = (request.form.get("definition") or "").strip()
    msg = "No clue/definition."
    if only and text:
        row = _load_clue(int(only))
        answer = row[1] if row else ""
        conn = store.connect()
        try:
            store.set_manual_definition(conn, int(only), text, answer)
        finally:
            conn.close()
        msg = "Display definition set (not added to DB): %r" % text
    notice = '<div class="wfw-notice">%s</div>' % escape(msg)
    return _page(notice + _body(raw, resolve_only=set()), scroll_to=only)


def _do_add(form):
    kind = form.get("kind")
    if kind == "definition":
        return admin_db.add_definition(form.get("definition"), form.get("answer"))
    if kind == "synonym":
        return admin_db.add_synonym(form.get("word"), form.get("synonym"))
    if kind == "indicator":
        return admin_db.add_indicator(form.get("word"), form.get("type"))
    return "Unknown add."


def _body(raw, resolve_only=None, ai=False):
    """Render the page body. `resolve_only` None -> re-solve every clue; a set ->
    re-solve only those ids, render the rest from their stored result. `ai` True
    uses the full AI wiring for the re-solve (per-clue, on demand); False uses the
    DB-only batch wiring so a whole-puzzle run makes no AI calls."""
    cid = escape(raw, quote=True)
    body = FORM.format(cid=cid) + RELOAD_FORM.format(cid=cid)
    tokens = [t for t in raw.replace(",", " ").split() if t]
    for token in tokens:
        resolve = resolve_only is None or token in resolve_only
        body += _render_one(token, raw, resolve, ai=ai)
    return body


def _render_one(token, raw_list, resolve=True, ai=False):
    """Render one clue card: the breakdown, its enrichment rows, its admin panel,
    and a per-clue reload button. Re-solves the clue only when `resolve` is True;
    otherwise it renders from the stored parse so a batch survives a single re-run.
    `ai` True uses the full AI wiring (per-clue, on demand); False uses the DB-only
    batch wiring."""
    try:
        clue_id = int(token)
    except ValueError:
        return f'<p class="warn">Not a numeric clue id: {escape(token)}</p>'
    row = _load_clue(clue_id)
    if row is None:
        return f'<p class="warn">No clue with id {clue_id}.</p>'
    clue_text, answer, src, pnum = row
    if resolve:
        w = wiring() if ai else batch_wiring()
        engine_registry.solve_clue_text(clue_text, answer, w,
                                        source=src, puzzle_number=pnum, clue_id=clue_id)
    conn = store.connect()
    try:
        parse = store.load_parse(conn, clue_id)
        ctx = store.load_atoms(conn, clue_id) if parse is not None else None
    finally:
        conn.close()

    if parse is None:
        note = ("No engine claimed this clue." if resolve
                else "Not yet run — use re-run below.")
        card = (f'<div class="wfw-card"><div class="wfw-clue">{escape(clue_text)}'
                f'</div><p>Answer: <strong>{escape(answer)}</strong></p>'
                f'<p class="warn">{note}</p></div>')
    else:
        if ctx is None:
            ctx = build_wfw_atom_context(parse.clue_text, parse.answer_text)
        screen = SCREENS.get(parse.operation) or SCREENS.get(parse.solved_by)
        card = screen(ctx, parse) if screen else wfw_render.render_parse(parse, ctx=ctx)

    status = parse.status if parse is not None else "fail"
    return (_cid_label(clue_id) + card
            + _enrichment_block(clue_text, answer, clue_id, raw_list)
            + _clue_controls(clue_id, raw_list, status)
            + _clue_admin_panel(clue_id, raw_list)
            + _reload_clue_button(clue_id, raw_list))


def _clue_controls(clue_id, raw_list, status):
    """Per-clue manual controls: override the verdict, and set a display-only
    definition (no DB write) for &lit clues. Both re-render from store, so they are
    not overwritten unless the clue is explicitly re-run."""
    h = _hidden(raw_list, clue_id)
    opts = "".join('<option value="%s"%s>%s</option>'
                   % (s, " selected" if s == status else "", s)
                   for s in ("pass", "pending", "fail"))
    return (
        '<div class="wfw-ctl">'
        f'<form method="post" action="/setstatus" class="wfw-cform">{h}'
        f'<span class="wfw-ctl-l">Status</span><select name="status">{opts}</select>'
        '<button>Set</button></form>'
        f'<form method="post" action="/setdef" class="wfw-cform">{h}'
        '<span class="wfw-ctl-l">Definition (display only)</span>'
        '<input name="definition" placeholder="type to display, not added to DB">'
        '<button>Set</button></form>'
        '</div>')


def _hidden(raw_list, clue_id):
    return ('<input type="hidden" name="id" value="%s">'
            '<input type="hidden" name="only" value="%d">'
            % (escape(raw_list, quote=True), clue_id))


def _enrichment_block(clue_text, answer, clue_id, raw_list):
    """The clue's own queued enrichments, each editable, with Approve / Reject. Shows
    nothing when the clue has no queued gap (so a clean pass stays clutter-free)."""
    rows = admin_db.pending_for_clue(clue_text, answer)
    if not rows:
        return ""
    out = ['<div class="wfw-enrich"><div class="wfw-enrich-h">Enrichment needed</div>']
    for pid, typ, word, letters, ans in rows:
        out.append(_enrich_row(pid, typ, word, letters, ans, clue_id, raw_list))
    out.append("</div>")
    return "".join(out)


def _enrich_row(pid, typ, word, letters, ans, clue_id, raw_list):
    """One editable enrichment row: pre-filled fields for its type + Approve + Reject."""
    h = _hidden(raw_list, clue_id) + ('<input type="hidden" name="pending_id" value="%d">'
                                      % pid)
    w = escape(word or "", quote=True)
    v = escape(letters or "", quote=True)
    a = escape(ans or "", quote=True)
    if typ == "definition":
        fields = (f'<input name="definition" value="{w}">'
                  f'<span class="wfw-arr">&rarr;</span>'
                  f'<input name="answer" value="{a or v}">')
        kind = "definition"
    elif typ == "synonym":
        fields = (f'<input name="word" value="{w}">'
                  f'<span class="wfw-arr">&rarr;</span>'
                  f'<input name="synonym" value="{v}">')
        kind = "synonym"
    elif typ == "indicator":
        opts = "".join('<option value="%s"%s>%s</option>'
                       % (t, " selected" if t == (letters or "").lower() else "", t)
                       for t in _IND_TYPES)
        fields = (f'<input name="word" value="{w}">'
                  f'<select name="type">{opts}</select>')
        kind = "indicator"
    else:
        return (f'<div class="wfw-erow"><span class="wfw-etype">{escape(typ or "?")}'
                f'</span><span class="wfw-emuted">{w} &rarr; {v}</span></div>')
    return (
        f'<div class="wfw-erow">'
        f'<span class="wfw-etype wfw-etype-{kind}">{kind}</span>'
        f'<form method="post" action="/enrich" class="wfw-eform">{h}'
        f'<input type="hidden" name="kind" value="{kind}">{fields}'
        f'<button class="wfw-ok">Approve</button></form>'
        f'<form method="post" action="/reject" class="wfw-eform-r">'
        f'<input type="hidden" name="id" value="{escape(raw_list, quote=True)}">'
        f'<input type="hidden" name="only" value="{clue_id}">'
        f'<input type="hidden" name="pending_id" value="{pid}">'
        f'<button class="wfw-no">Reject</button></form>'
        f'</div>')


def _clue_admin_panel(clue_id, raw_list):
    """A collapsible add-to-reference-DB panel on EVERY clue (pass or fail). Each form
    carries the batch id list + this clue id, so the add re-solves just this clue."""
    h = _hidden(raw_list, clue_id)
    opts = "".join('<option value="%s">%s</option>' % (t, t) for t in _IND_TYPES)
    return f"""
<details class="wfw-admin">
  <summary>Add to reference DB</summary>
  <form method="post" action="/admin" class="wfw-af">
    <input type="hidden" name="kind" value="definition">{h}
    <span class="wfw-af-l">Definition</span>
    <input name="definition" placeholder="nearest the bottom">
    <input name="answer" placeholder="NETHERMOST">
    <button>Add</button>
  </form>
  <form method="post" action="/admin" class="wfw-af">
    <input type="hidden" name="kind" value="indicator">{h}
    <span class="wfw-af-l">Indicator</span>
    <input name="word" placeholder="a little">
    <select name="type">{opts}</select>
    <button>Add</button>
  </form>
  <form method="post" action="/admin" class="wfw-af">
    <input type="hidden" name="kind" value="synonym">{h}
    <span class="wfw-af-l">Synonym</span>
    <input name="word" placeholder="word in clue">
    <input name="synonym" placeholder="value/answer fragment">
    <button>Add</button>
  </form>
</details>
"""


def _reload_clue_button(clue_id, raw_list):
    """Per-clue button: reload the DB snapshot and re-run JUST this clue, keeping the
    rest of the batch on screen."""
    return (
        '<form method="post" action="/reload" style="margin:.4rem 0 0">'
        '<input type="hidden" name="id" value="%s">'
        '<input type="hidden" name="only" value="%d">'
        '<button class="wfw-reload wfw-reload-clue" title="Rebuild the DB snapshot, '
        'then re-run only this clue">&#8635; Reload DB &amp; re-run this clue</button>'
        '</form>' % (escape(raw_list, quote=True), clue_id))


def _cid_label(clue_id):
    return ('<div id="clue-%d" style="font-size:.8rem;font-weight:700;color:#64748b;'
            'letter-spacing:.06em;margin:1.2rem 0 -.5rem;scroll-margin-top:.6rem">'
            'CLUE ID %d</div>' % (clue_id, clue_id))


def _page(body, scroll_to=None):
    # After a per-clue action, jump back to that clue instead of the top of the page.
    scroll = ""
    if scroll_to:
        scroll = ("<script>(function(){var e=document.getElementById('clue-%s');"
                  "if(e)e.scrollIntoView();})();</script>" % escape(str(scroll_to)))
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
  .wfw-enrich {{ background:#fff7ed; border:1px solid #fed7aa; border-radius:12px;
                 padding:.6rem .85rem; margin:.5rem 0 0; }}
  .wfw-enrich-h {{ font-weight:800; color:#9a3412; font-size:.8rem;
                   text-transform:uppercase; letter-spacing:.05em; margin-bottom:.4rem; }}
  .wfw-erow {{ display:flex; gap:.5rem; align-items:center; margin:.35rem 0;
               flex-wrap:wrap; }}
  .wfw-etype {{ min-width:5.5rem; font-size:.7rem; font-weight:800; color:#fff;
                background:#9a3412; border-radius:5px; padding:.15rem .4rem;
                text-align:center; text-transform:uppercase; }}
  .wfw-etype-synonym {{ background:#1d4ed8; }}
  .wfw-etype-indicator {{ background:#7c3aed; }}
  .wfw-etype-definition {{ background:#0f766e; }}
  .wfw-eform, .wfw-eform-r {{ display:inline-flex; gap:.4rem; align-items:center; }}
  .wfw-erow input, .wfw-erow select {{ font-size:.95rem; padding:.25rem .4rem;
               border:1px solid #cbd5e1; border-radius:6px; }}
  .wfw-arr {{ color:#94a3b8; }}
  .wfw-ok {{ font-size:.85rem; padding:.25rem .7rem; border-radius:6px; border:none;
             background:#16a34a; color:#fff; cursor:pointer; }}
  .wfw-no {{ font-size:.85rem; padding:.25rem .7rem; border-radius:6px; border:none;
             background:#dc2626; color:#fff; cursor:pointer; }}
  .wfw-emuted {{ color:#64748b; }}
  .wfw-ctl {{ display:flex; gap:1rem; align-items:center; flex-wrap:wrap;
              margin:.5rem 0 0; padding:.45rem .85rem; background:#f8fafc;
              border:1px solid #e2e8f0; border-radius:12px; }}
  .wfw-cform {{ display:inline-flex; gap:.4rem; align-items:center; }}
  .wfw-ctl-l {{ font-weight:700; color:#475569; font-size:.8rem; }}
  .wfw-ctl select, .wfw-ctl input {{ font-size:.95rem; padding:.25rem .4rem;
              border:1px solid #cbd5e1; border-radius:6px; }}
  .wfw-ctl button {{ font-size:.85rem; padding:.25rem .7rem; border-radius:6px;
              border:1px solid #475569; background:#475569; color:#fff; cursor:pointer; }}
  .wfw-admin {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px;
                padding:.4rem .9rem; margin:.5rem 0 0; }}
  .wfw-admin summary {{ cursor:pointer; font-weight:700; color:#334155;
                        padding:.3rem 0; font-size:.9rem; }}
  .wfw-af {{ display:flex; gap:.5rem; align-items:center; margin:.5rem 0;
             flex-wrap:wrap; }}
  .wfw-af-l {{ min-width:6rem; font-weight:700; color:#475569; font-size:.85rem; }}
  .wfw-af input, .wfw-af select {{ font-size:1rem; padding:.3rem .4rem;
             border:1px solid #cbd5e1; border-radius:6px; }}
  .wfw-af input {{ flex:1; min-width:8rem; }}
  .wfw-af button {{ font-size:.95rem; padding:.3rem .8rem; border-radius:6px;
             border:1px solid #2563eb; background:#2563eb; color:#fff; cursor:pointer; }}
  .wfw-reload {{ font-size:1rem; padding:.3rem .9rem; border-radius:6px;
             border:1px solid #0d9488; background:#0d9488; color:#fff;
             cursor:pointer; margin-left:.5rem; }}
  .wfw-reload-clue {{ font-size:.85rem; padding:.25rem .7rem; margin-left:0;
             background:#0f766e; border-color:#0f766e; }}
</style></head><body>
  <p class="wfw-tag">word-for-word true test &middot; real clues, all engines</p>
  {body}
  {scroll}
</body></html>"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5099, debug=False)
