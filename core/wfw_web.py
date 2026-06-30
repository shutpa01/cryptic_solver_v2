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

import re
import sqlite3
import os
from html import escape

from flask import Flask, request, redirect

from core import engine_registry
from core import wfw_render
from core import hidden_screen
from core import acrostic_screen
from core import homophone_screen
from core import dd_screen
from core import charade_screen
from core import charade_homophone_screen
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
           "charade_homophone": charade_homophone_screen.render,
           "palindrome": palindrome_screen.render,
           "spoonerism": spoonerism_screen.render}
_ENGINE_LABELS = {"hidden": "hidden", "dd": "double definition",
                  "charade": "charade", "anagram": "anagram",
                  "anagram_charade": "anagram + charade",
                  "anagram_container": "anagram + container",
                  "charade_homophone": "charade + homophone",
                  "palindrome": "palindrome",
                  "spoonerism": "spoonerism",
                  "substitution": "substitution"}
# indicator types offered in the per-clue admin panel + enrichment edit.
_IND_TYPES = ["hidden", "anagram", "container", "reversal", "deletion",
              "acrostic", "homophone", "charade", "alternation"]
# Sub-types the SOLVING CODE actually recognises, per indicator type. Only `deletion`
# has any (core.deletion.SUBTYPE_OP). Each is (stored-value, intuitive-label): the value
# is what the code reads, the label is the clear descriptor shown to the user (the DB
# names like "head"/"middle"/"empty" are counter-intuitive on their own).
_IND_SUBTYPES = {
    "deletion": [("", "— no sub-type —"),
                 ("head", "remove first letter (behead)"),
                 ("tail", "remove last letter (curtail)"),
                 ("ends", "remove outer letters"),
                 ("middle", "remove middle letter"),
                 ("empty", "hollow — remove inner letters"),
                 ("general", "letters named by another word")],
}

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                  "data", "clues_master.db")

app = Flask(__name__)
_WIRING = None


import time as _time
BOOT_ID = _time.strftime("%H:%M:%S", _time.localtime())   # changes on every server restart


def _grid_redirect(only, msg=""):
    """Post/Redirect/Get: after a grid action, redirect to the canonical grid GET so the
    URL is clean (/rolegrid, not /gridrole), a refresh won't resubmit, and the page is
    re-rendered fresh from one code path. Preserves the CLUTCH context (`from`, the
    clue-page id-string this clue was opened within) submitted by the grid form, so the
    "back to clue page" link still returns to the whole clutch after an action."""
    from urllib.parse import quote
    extra = ""
    try:
        frm = (request.form.get("from") or request.args.get("from") or "").strip()
    except Exception:
        frm = ""
    if frm:
        extra = "&from=%s" % quote(frm, safe="")
    return redirect("/rolegrid?id=%s&notice=%s%s" % (only, quote(msg), extra))


@app.after_request
def _no_cache(resp):
    """Never let the browser serve a stale page — this is an admin tool that changes on
    every edit, and a cached page made it look like code changes 'did nothing'."""
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


def wiring():
    """The full wiring, AI ON. Used only for an explicit per-clue re-run, so AI is
    on-demand, never part of a batch solve."""
    global _WIRING
    if _WIRING is None:
        _WIRING = engine_registry.make_db_wiring()
        # QUEUE-FOR-APPROVAL (not auto-file): a discovered signature is queued for human
        # approval rather than written straight to the catalog. auto_signature (auto-file)
        # is deliberately OFF; the per-clue re-run turns on auto_signature_queue (see
        # _render_one), so discovery only pays its cost on an explicit single-clue re-run.
        _WIRING["auto_signature"] = False
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


def apply_add_to_wiring(form):
    """Fold a just-added reference entry into the CACHED wiring incrementally, instead
    of reload_wiring() (which rebuilds everything and rescans a 643k-row table). It is
    the same cached wiring the page run uses, so a clue-level add stays as fast as
    solving one clue. Falls back to a full reload if the wiring predates invalidate()."""
    inv = wiring().get("invalidate")
    if not inv:
        reload_wiring()
        return
    kind = form.get("kind")
    if kind == "definition":
        inv("definition", definition=form.get("definition"), answer=form.get("answer"))
    elif kind == "synonym":
        inv("synonym", word=form.get("word"), synonym=form.get("synonym"))
    elif kind == "indicator":
        inv("indicator", word=form.get("word"), wordplay_type=form.get("type"))
    elif kind == "link":
        inv("link", word=form.get("word"))
    elif kind == "literal":
        inv("literal", word=form.get("word"))
    elif kind == "homophone":
        inv("homophone", word=form.get("word"))
    else:
        reload_wiring()


def _norm_phrase(s):
    """Loose normalisation for matching a typed definition to a clue edge phrase:
    lowercase, punctuation -> space, collapse whitespace. So "Grain Kilns",
    "grain kilns" and a token like "kilns," all compare equal."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", (s or "").lower()).split())


def _forced_def_for(clue_id):
    conn = store.connect()
    try:
        return store.get_forced_definition(conn, clue_id)
    finally:
        conn.close()


def _filler_for(clue_id):
    conn = store.connect()
    try:
        return store.get_clue_filler(conn, clue_id)
    finally:
        conn.close()


def _load_clue(clue_id):
    conn = sqlite3.connect(DB)
    try:
        return conn.execute(
            "SELECT clue_text, answer, source, puzzle_number, direction, enumeration, "
            "clue_number FROM clues WHERE id = ?", (clue_id,)).fetchone()
    finally:
        conn.close()


def enum_space(answer, enumeration):
    """Re-insert word breaks into a (possibly spaceless) answer using the clue's
    enumeration, so the page shows 'ALL IN GOOD TIME (3,2,4,4)' rather than the
    concatenated 'ALLINGOODTIME' with a derived '(13)'. A comma in the enumeration is a
    space, a hyphen a hyphen. Returns the answer UNCHANGED when the letter count does not
    match the enumeration total (so it can never corrupt an answer), or when there is only
    one part. Idempotent: re-spacing an already-spaced answer yields the same string."""
    if not answer or not enumeration:
        return answer
    nums = re.findall(r"\d+", enumeration)
    if len(nums) < 2:
        return answer
    seps = re.findall(r"[^\d()\s]+", enumeration)      # separators between the numbers
    letters = [c for c in answer if c.isalpha()]
    if sum(int(n) for n in nums) != len(letters):
        return answer
    parts, i = [], 0
    for n in nums:
        n = int(n)
        parts.append("".join(letters[i:i + n]))
        i += n
    out = parts[0]
    for k in range(1, len(parts)):
        sep = seps[k - 1] if k - 1 < len(seps) else ","
        out += ("-" if "-" in sep else " ") + parts[k]
    return out


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
    # Render from the STORE and re-solve only clues never solved before — a plain page
    # load / back-from-hand-solver is instant and does not re-run the whole clutch. Use
    # the Reload buttons to force a fresh solve (per-clue or whole-puzzle).
    return _page(_body(raw, fill_missing=True))


@app.route("/reload", methods=["POST"])
def reload_route():
    """Rebuild the DB snapshot, then re-run. `only` (a per-clue button) re-solves
    JUST that clue and keeps the rest of the batch on screen; otherwise re-solve all."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    reload_wiring()
    notice = '<div class="wfw-notice">Reference DB reloaded from disk.</div>'
    # A per-clue re-run is DB-only by default (fast); the AI fallback runs only when the
    # "with AI fallback" box is ticked. A re-run-all (no only) is always DB-only.
    use_ai = bool(only) and bool(request.form.get("ai"))
    return _page(notice + _body(raw, resolve_only={only} if only else None,
                                ai=use_ai, discover=bool(only)),
                 scroll_to=only)


@app.route("/approvesig", methods=["POST"])
def approvesig():
    """Approve a queued auto-discovered signature: file it into the catalog and reload
    the wiring so it is live, then re-render."""
    from core import signature_queue
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    sid = (request.form.get("sig_id") or "").strip()
    msg = "No signature id."
    if sid:
        tid, info = signature_queue.approve(int(sid))
        if tid is not None:
            reload_wiring()                       # make the new signature live
            msg = "Approved signature %s (template %s); catalog reloaded." % (info, tid)
        else:
            msg = "Could not approve: %s" % info
    notice = '<div class="wfw-notice">%s</div>' % escape(msg)
    # Keep the clutch on screen and re-solve just this clue so the new signature's solve
    # appears in place (was: blank screen — the form carried no batch id).
    return _page(notice + _body(raw, resolve_only={only} if only else set()),
                 scroll_to=only or None)


@app.route("/rejectsig", methods=["POST"])
def rejectsig():
    """Reject a queued auto-discovered signature (remembered so it is not re-queued)."""
    from core import signature_queue
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    sid = (request.form.get("sig_id") or "").strip()
    if sid:
        signature_queue.reject(int(sid))
    notice = '<div class="wfw-notice">Signature rejected.</div>'
    return _page(notice + _body(raw, resolve_only=set()), scroll_to=only or None)


@app.route("/admin", methods=["POST"])
def admin():
    """Add an entry to the reference DB from a clue's admin panel, then re-solve just
    that clue (batch preserved)."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    msg = _do_add(request.form)
    # Fold the just-added piece into the cached wiring INCREMENTALLY (drop the affected
    # caches + patch the prebuilt index with the one new entry) instead of rebuilding
    # everything — the rebuild rescanned a 643k-row table, which is why a clue-level add
    # was slow. The adders write the normalized-key columns, so the live query sees the
    # new row at once; this just clears the stale cached (failing) result.
    apply_add_to_wiring(request.form)
    notice = '<div class="wfw-notice">%s</div>' % escape(msg)
    return _page(notice + _body(raw, resolve_only={only} if only else None),
                 scroll_to=only)


@app.route("/enrich", methods=["POST"])
def enrich():
    """Approve one queued enrichment: write the (possibly edited) values to the
    reference DB, drop it from the queue, then re-solve just that clue. Rebuilds the
    wiring (see /admin) so the approved piece is seen, then re-solves DB-only."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    pid = (request.form.get("pending_id") or "").strip()
    msg = _do_add(request.form)
    if pid and not msg.startswith(("Definition and", "Word and", "Indicator word")):
        admin_db.delete_pending(pid)               # accepted -> leave the queue
    # Fold the approved piece into the cached wiring incrementally (see /admin) — no
    # full rebuild / 643k-row rescan.
    apply_add_to_wiring(request.form)
    notice = '<div class="wfw-notice">%s</div>' % escape(msg)
    # DB-only re-solve: the piece is now in the reference DB, so it should solve
    # mechanically. AI fallback is opt-in via the per-clue "Reload DB & re-run" box.
    return _page(notice + _body(raw, resolve_only={only} if only else None),
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
    if only and status in ("pass", "pending", "fail", "invalid"):
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


@app.route("/forcedef", methods=["POST"])
def forcedef():
    """Pin one clue's definition to the typed phrase and RE-SOLVE it (DB-only unless AI
    ticked). Unlike /setdef (display only), this re-runs the cascade with `defines`
    confirming only that edge phrase, so the wordplay must account for every other word.
    The pin persists (survives later re-runs) until cleared."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    text = (request.form.get("definition") or "").strip()
    use_ai = bool(request.form.get("ai"))
    msg = "No clue/definition."
    if only and text:
        row = _load_clue(int(only))
        answer = row[1] if row else ""
        conn = store.connect()
        try:
            store.set_forced_definition(conn, int(only), text)
        finally:
            conn.close()
        # Pinning a definition is a human assertion that it IS the definition — so PERSIST
        # it to the reference DB (definition_answers_augmented), not just this one clue, so
        # defines() knows it everywhere and a future re-run needs no pin. Fold it into the
        # cached wiring so the re-solve below already sees it.
        addmsg = admin_db.add_definition(text, answer)
        apply_add_to_wiring({"kind": "definition", "definition": text, "answer": answer})
        msg = "Definition pinned to %r, added to DB (%s); clue re-solved." % (text, addmsg)
    notice = '<div class="wfw-notice">%s</div>' % escape(msg)
    return _page(notice + _body(raw, resolve_only={only} if only else None, ai=use_ai),
                 scroll_to=only)


@app.route("/setfiller", methods=["POST"])
def setfiller():
    """Tag a word as SURFACE FILLER for THIS clue (setter padding, no cryptic role) and
    re-solve it. The tag lives only on this clue (wfw_filler) and is NEVER written to the
    shared link_words table, so common words like 'get'/'will' can't pollute links. The
    re-solve accounts the word like a link for this clue, so a near-solve becomes a pass."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    word = (request.form.get("word") or "").strip()
    msg = "No clue/word."
    if only and word:
        conn = store.connect()
        try:
            store.add_clue_filler(conn, int(only), word)
        finally:
            conn.close()
        msg = "Tagged %r as surface filler (this clue only, not added to link words)." % word
    notice = '<div class="wfw-notice">%s</div>' % escape(msg)
    return _page(notice + _body(raw, resolve_only={only} if only else None), scroll_to=only)


@app.route("/clearfiller", methods=["POST"])
def clearfiller():
    """Untag a surface-filler word on this clue and re-solve."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    word = (request.form.get("word") or "").strip()
    if only:
        conn = store.connect()
        try:
            store.clear_clue_filler(conn, int(only), word or None)
        finally:
            conn.close()
    if request.form.get("surface") == "grid":
        _resolve_one(int(only)) if only else None
        return _grid_redirect(only, "Surface-filler tag removed; re-solved.")
    notice = '<div class="wfw-notice">Surface-filler tag removed; clue re-solved.</div>'
    return _page(notice + _body(raw, resolve_only={only} if only else None), scroll_to=only)


@app.route("/clearforcedef", methods=["POST"])
def clearforcedef():
    """Remove a clue's pinned definition and re-solve it with the normal definition stage."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    msg = "No clue."
    if only:
        conn = store.connect()
        try:
            store.clear_forced_definition(conn, int(only))
        finally:
            conn.close()
        msg = "Pinned definition cleared; clue re-solved normally."
    if request.form.get("surface") == "grid":
        _resolve_one(int(only)) if only else None
        return _grid_redirect(only, msg)
    notice = '<div class="wfw-notice">%s</div>' % escape(msg)
    return _page(notice + _body(raw, resolve_only={only} if only else None),
                 scroll_to=only)


@app.route("/unforce", methods=["POST"])
def unforce_route():
    """Admin UNFORCE: drop ALL of this clue's manual overrides (filler / pinned definition /
    forced indicator) and lift the freeze, then re-solve from scratch. The ONLY way a frozen
    forced pass is allowed to change."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    msg = "No clue."
    if only:
        conn = store.connect()
        try:
            store.unforce(conn, int(only))
        finally:
            conn.close()
        msg = ("Unforced clue %s — all manual overrides cleared and freeze lifted; "
               "re-solved from scratch." % only)
    if request.form.get("surface") == "grid":
        _resolve_one(int(only)) if only else None
        return _grid_redirect(only, msg)
    notice = '<div class="wfw-notice">%s</div>' % escape(msg)
    return _page(notice + _body(raw, resolve_only={only} if only else None),
                 scroll_to=only)


def _do_add(form):
    kind = form.get("kind")
    if kind == "definition":
        return admin_db.add_definition(form.get("definition"), form.get("answer"))
    if kind == "synonym":
        return admin_db.add_synonym(form.get("word"), form.get("synonym"))
    if kind == "indicator":
        return admin_db.add_indicator(form.get("word"), form.get("type"),
                                      form.get("subtype"))
    if kind == "link":
        return admin_db.add_link_word(form.get("word"))
    if kind == "literal":
        return admin_db.add_literal(form.get("word"))
    if kind == "homophone":
        return admin_db.add_homophone(form.get("word"), form.get("homophone"))
    return "Unknown add."


def _signature_queue_html(displayed_ids=()):
    """Top banner for ORPHAN auto-discovered signatures only — those whose clue is NOT in the
    current clutch (so they would otherwise be invisible). In-view clues render their own
    suggestion inline via _clue_signature_suggestions, so they are excluded here."""
    from core import signature_queue
    try:
        rows = signature_queue.list_pending()
    except Exception:
        return ""
    shown = {str(i) for i in (displayed_ids or ())}
    rows = [r for r in rows if str(r.get("clue_id")) not in shown]
    if not rows:
        return ""
    items = []
    for r in rows:
        parse = escape(r["parse_text"] or "").replace("\n", "<br>")
        items.append(
            '<div class="wfw-erow" style="align-items:flex-start">'
            '<span class="wfw-etype" style="background:#6d28d9">SIGNATURE</span>'
            '<div style="flex:1">'
            '<code>%s</code><br>'
            '<span class="tag">from %s [%s] %s</span>'
            '<div style="margin:.25rem 0;font-size:.85rem;color:#444">%s</div>'
            '</div>'
            '<form method="post" action="/approvesig" style="display:inline">'
            '<input type="hidden" name="sig_id" value="%d">'
            '<input type="hidden" name="id" value="%s">'
            '<button>Approve</button></form>'
            '<form method="post" action="/rejectsig" style="display:inline">'
            '<input type="hidden" name="sig_id" value="%d">'
            '<input type="hidden" name="id" value="%s">'
            '<button>Reject</button></form>'
            '</div>'
            % (escape(r["signature"]), escape(str(r["clue_id"])),
               escape(r["answer"] or ""), escape(r["clue_text"] or ""), parse,
               r["id"], "", r["id"], ""))
    return ('<div class="wfw-enrich" style="background:#f5f3ff;border-color:#ddd6fe">'
            '<div class="wfw-enrich-h" style="color:#5b21b6">'
            'Signatures discovered &mdash; awaiting approval (%d)</div>%s</div>'
            % (len(rows), "".join(items)))


def _clue_signature_suggestions(clue_id, raw_list):
    """Per-clue: auto-discovered signature(s) awaiting approval FOR THIS CLUE, rendered INSIDE
    the clue card (not a global top banner). Approve/Reject carry the current clutch (`id`) and
    this clue (`only`) so approving keeps the clutch on screen and re-solves the clue in place."""
    from core import signature_queue
    try:
        rows = [r for r in signature_queue.list_pending()
                if str(r.get("clue_id")) == str(clue_id)]
    except Exception:
        return ""
    if not rows:
        return ""
    h = ('<input type="hidden" name="id" value="%s">'
         '<input type="hidden" name="only" value="%d">'
         % (escape(raw_list, quote=True), clue_id))
    items = []
    for r in rows:
        parse = escape(r["parse_text"] or "").replace("\n", "<br>")
        items.append(
            '<div class="wfw-erow" style="align-items:flex-start">'
            '<span class="wfw-etype" style="background:#6d28d9">SUGGESTED SIGNATURE</span>'
            '<div style="flex:1"><code>%s</code>'
            '<div style="margin:.25rem 0;font-size:.85rem;color:#444">%s</div></div>'
            '<form method="post" action="/approvesig" style="display:inline">%s'
            '<input type="hidden" name="sig_id" value="%d">'
            '<button class="wfw-ok">Approve &amp; re-solve</button></form>'
            '<form method="post" action="/rejectsig" style="display:inline">%s'
            '<input type="hidden" name="sig_id" value="%d">'
            '<button class="wfw-no">Reject</button></form>'
            '</div>'
            % (escape(r["signature"]), parse, h, r["id"], h, r["id"]))
    return ('<div class="wfw-enrich" style="background:#f5f3ff;border-color:#ddd6fe">'
            '<div class="wfw-enrich-h" style="color:#5b21b6">'
            'Suggested signature &mdash; approve to add &amp; re-solve this clue</div>%s</div>'
            % "".join(items))


def _stored_parse_ids(tokens):
    """The subset of these clue ids that ALREADY have a stored parse (a wfw_solve row),
    in ONE query — so the clue page can render them from the store instead of re-running
    the cascade. Non-numeric tokens are ignored."""
    ids = [int(t) for t in tokens if t.isdigit()]
    if not ids:
        return set()
    conn = store.connect()
    try:
        q = ("SELECT clue_id FROM wfw_solve WHERE clue_id IN (%s)"
             % ",".join("?" * len(ids)))
        return {r[0] for r in conn.execute(q, ids)}
    finally:
        conn.close()


def _body(raw, resolve_only=None, ai=False, discover=False, fill_missing=False):
    """Render the page body. `resolve_only` None -> re-solve every clue; a set ->
    re-solve only those ids, render the rest from their stored result. `fill_missing`
    True -> render every clue from the STORE and re-solve ONLY clues that have no stored
    parse yet (the default for a plain page load / back-navigation, so an already-solved
    clutch is instant and nothing is needlessly re-run); takes precedence over
    `resolve_only`. `ai` True uses the full AI wiring for the re-solve (per-clue, on
    demand); False uses the DB-only batch wiring so a whole-puzzle run makes no AI calls.
    `discover` True turns on the auto-signature DISCOVERY+QUEUE for the re-solved clue(s)."""
    cid = escape(raw, quote=True)
    # Render the clue cards FIRST (a per-clue re-run may discover + queue a signature
    # during the solve), THEN build the queue banner so it reflects anything just queued.
    cards = ""
    tokens = [t for t in raw.replace(",", " ").split() if t]
    stored = _stored_parse_ids(tokens) if fill_missing else set()
    for token in tokens:
        if fill_missing:
            resolve = not (token.isdigit() and int(token) in stored)  # only never-solved
        else:
            resolve = resolve_only is None or token in resolve_only
        cards += _render_one(token, raw, resolve, ai=ai, discover=discover)
    return (FORM.format(cid=cid) + RELOAD_FORM.format(cid=cid)
            + _signature_queue_html([t for t in tokens if t.isdigit()]) + cards)


def _render_one(token, raw_list, resolve=True, ai=False, discover=False):
    """Render one clue card: the breakdown, its enrichment rows, its admin panel,
    and a per-clue reload button. Re-solves the clue only when `resolve` is True;
    otherwise it renders from the stored parse so a batch survives a single re-run.
    `ai` True uses the full AI wiring (per-clue, on demand); False uses the DB-only
    batch wiring. `discover` True enables auto-signature DISCOVERY+QUEUE for this clue
    (used on the per-clue re-run, where the ~2s discovery cost is acceptable)."""
    try:
        clue_id = int(token)
    except ValueError:
        return f'<p class="warn">Not a numeric clue id: {escape(token)}</p>'
    row = _load_clue(clue_id)
    if row is None:
        return f'<p class="warn">No clue with id {clue_id}.</p>'
    clue_text, answer, src, pnum, direction, enumeration, cnum = row
    # Space the answer per the clue's enumeration (ALLINGOODTIME -> ALL IN GOOD TIME), so a
    # fresh solve stores the spaced form and the displayed enumeration is (3,2,4,4) not (13).
    answer = enum_space(answer, enumeration)
    forced = _forced_def_for(clue_id)
    filler = _filler_for(clue_id)
    if resolve:
        w = wiring() if ai else batch_wiring()
        # Apply this clue's manual role overrides (filler + forced definition + forced
        # indicator) via the single shared helper — the same forces, now also applied on
        # the batch/A-B paths. No-op when the clue has none, so behaviour is unchanged.
        from core import clue_overrides
        w = clue_overrides.apply_forced_overrides(w, clue_id)
        if discover:
            w = dict(w)
            w["auto_signature_queue"] = True   # discover + queue a signature if it fails
        engine_registry.solve_clue_text(clue_text, answer, w,
                                        source=src, puzzle_number=pnum, clue_id=clue_id,
                                        direction=direction)
    conn = store.connect()
    try:
        parse = store.load_parse(conn, clue_id)
        ctx = store.load_atoms(conn, clue_id) if parse is not None else None
    finally:
        conn.close()
    # FIX THE PAST TOO: stored parses solved before this fix carry the spaceless answer, so
    # re-space at render time (idempotent for freshly-solved clues). Only parse.answer_text
    # matters for the displayed enumeration/answer; ctx is a frozen dataclass and its
    # answer_text is not used by the answer/enumeration rendering.
    if parse is not None:
        parse.answer_text = enum_space(parse.answer_text, enumeration)

    # SURFACE FILLER: tagging a word as filler IS the human approval. So a clue that
    # assembles cleanly once its tagged filler word is accounted is a genuine PASS — not
    # forced to pending. The approval persists in the DB: the tag lives in wfw_filler and
    # the pass is re-derived from it on every solve (and persisted to wfw_solve), so it
    # survives reloads. A clue still pending for ANOTHER reason (e.g. a provisional
    # definition) keeps that pending verdict — only the filler word itself is approved here.
    if filler and parse is not None and parse.status == "pass":
        fil = {(x or "").strip().lower() for x in filler}
        used = sorted({(a.text or "").strip().lower() for a in (parse.annotations or [])
                       if getattr(a, "role", "") == "link"} & fil)
        if used:
            conn2 = store.connect()
            try:
                store.set_status(conn2, clue_id, "pass")   # persist the approved pass
            finally:
                conn2.close()

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

    forced_banner = ""
    if forced:
        cw = _norm_phrase(clue_text).split()
        tw = _norm_phrase(forced).split()
        is_edge = bool(tw) and (cw[:len(tw)] == tw or cw[-len(tw):] == tw)
        if is_edge:
            forced_banner = ('<div class="wfw-notice wfw-forced">Definition pinned to '
                             '<strong>%s</strong> &mdash; wordplay must cover the rest.'
                             '</div>' % escape(forced))
        else:
            forced_banner = ('<div class="wfw-notice wfw-forced-bad">Pinned definition '
                             '<strong>%s</strong> is not a start/end phrase of this clue, '
                             'so it cannot take. Clear it or retype the exact edge words.'
                             '</div>' % escape(forced))

    status = parse.status if parse is not None else "fail"
    unaccounted = []
    if parse is not None and ctx is not None:
        try:
            unaccounted = list(parse.unexplained_words(ctx))
        except Exception:
            unaccounted = []
    return (_cid_label(clue_id, src, pnum, cnum, direction) + forced_banner + card
            + _note_block(clue_id)
            + _filler_block(clue_id, raw_list, unaccounted, filler)
            + _enrichment_block(clue_text, answer, clue_id, raw_list)
            + _clue_signature_suggestions(clue_id, raw_list)
            + _clue_controls(clue_id, raw_list, status)
            + _clue_admin_panel(clue_id, raw_list)
            + _reload_clue_button(clue_id, raw_list))


def _clue_controls(clue_id, raw_list, status):
    """Per-clue manual controls: override the verdict, set a display-only definition
    (no DB write) for &lit clues, and PIN the definition + re-solve (the real override
    when the definition stage grabbed too many words). All re-render from store, so they
    are not overwritten unless the clue is explicitly re-run."""
    h = _hidden(raw_list, clue_id)
    opts = "".join('<option value="%s"%s>%s</option>'
                   % (s, " selected" if s == status else "", s)
                   for s in ("pass", "pending", "fail", "invalid"))
    forced = _forced_def_for(clue_id)
    clear_btn = ""
    if forced:
        clear_btn = (f'<form method="post" action="/clearforcedef" class="wfw-cform">{h}'
                     f'<span class="wfw-ctl-l">Pinned: <em>{escape(forced)}</em></span>'
                     '<button>Clear pin &amp; re-solve</button></form>')
    conn = store.connect()
    try:
        frozen = store.is_frozen(conn, clue_id)
    finally:
        conn.close()
    unforce_btn = ""
    if frozen:
        unforce_btn = (
            f'<form method="post" action="/unforce" class="wfw-cform">{h}'
            '<span class="wfw-ctl-l">&#128274; FROZEN (forced pass &mdash; will not revert)'
            '</span><button>Unforce &amp; re-solve</button></form>')
    return (
        '<div class="wfw-ctl">'
        f'{unforce_btn}'
        f'<form method="post" action="/setstatus" class="wfw-cform">{h}'
        f'<span class="wfw-ctl-l">Status</span><select name="status">{opts}</select>'
        '<button>Set</button></form>'
        f'<form method="post" action="/setdef" class="wfw-cform">{h}'
        '<span class="wfw-ctl-l">Definition (display only)</span>'
        '<input name="definition" placeholder="type to display, not added to DB">'
        '<button>Set</button></form>'
        f'<form method="post" action="/forcedef" class="wfw-cform">{h}'
        '<span class="wfw-ctl-l">Pin definition &amp; re-solve</span>'
        '<input name="definition" placeholder="exact edge words, e.g. kilns">'
        '<label class="wfw-ai"><input type="checkbox" name="ai" value="1">AI</label>'
        '<button>Pin &amp; re-solve</button></form>'
        f'{clear_btn}'
        '</div>')


def _hidden(raw_list, clue_id):
    return ('<input type="hidden" name="id" value="%s">'
            '<input type="hidden" name="only" value="%d">'
            % (escape(raw_list, quote=True), clue_id))


def _filler_block(clue_id, raw_list, unaccounted, tagged):
    """Per-clue SURFACE-FILLER controls. Each currently-unaccounted clue word gets a
    button to tag it as filler (accounted for THIS clue only, never written to the shared
    link_words table); each already-tagged word gets an untag button. Renders nothing when
    there's neither — so a clean pass stays clutter-free."""
    tagged = {(t or "").strip().lower() for t in (tagged or ())}
    show = [w for w in (unaccounted or []) if (w or "").strip().lower() not in tagged]
    if not show and not tagged:
        return ""
    h = _hidden(raw_list, clue_id)

    def btn(action, word, label, bg):
        return ('<form method="post" action="%s" style="display:inline-block;margin:.15rem 0">'
                '%s<input type="hidden" name="word" value="%s">'
                '<button class="wfw-reload wfw-reload-clue" style="background:%s;'
                'border-color:%s;margin-left:.4rem">%s</button></form>'
                % (action, h, escape(word, quote=True), bg, bg, label))

    out = ['<div class="wfw-notice" style="background:#faf5ff;border:1px solid #d8b4fe">'
           '<b>Surface filler</b> &mdash; setter padding with no cryptic role; tag for THIS '
           'clue only (never added to link words):']
    for w in show:
        out.append(btn('/setfiller', w, '&#43; mark &ldquo;%s&rdquo; filler' % escape(w),
                       '#9333ea'))
    for w in sorted(tagged):
        out.append(btn('/clearfiller', w, '&#215; untag &ldquo;%s&rdquo;' % escape(w),
                       '#94a3b8'))
    out.append('</div>')
    return "".join(out)


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
    import json
    h = _hidden(raw_list, clue_id)
    opts = "".join('<option value="%s">%s</option>' % (t, t) for t in _IND_TYPES)
    # Data-driven sub-types: a per-type map fills the subtype dropdown on type change, so
    # ANY type with sub-types (deletion now, selection later) shows them — no hardcoding.
    sub_js = (
        "<script>window.WFW_SUBTYPES=%s;"
        "window.wfwSub=window.wfwSub||function(sel){"
        "var s=sel.form.querySelector('select[name=subtype]');"
        "var subs=(window.WFW_SUBTYPES||{})[sel.value]||[];s.innerHTML='';"
        "if(!subs.length){s.style.display='none';return;}"
        "for(var i=0;i<subs.length;i++){var o=document.createElement('option');"
        "o.value=subs[i][0];o.textContent=subs[i][1];s.appendChild(o);}"
        "s.style.display='';};</script>"
        % json.dumps({t: s for t, s in _IND_SUBTYPES.items()}))
    return f"""
{sub_js}
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
    <select name="type" onchange="wfwSub(this)">{opts}</select>
    <select name="subtype" style="display:none" title="indicator sub-type"></select>
    <button>Add</button>
  </form>
  <form method="post" action="/admin" class="wfw-af">
    <input type="hidden" name="kind" value="link">{h}
    <span class="wfw-af-l">Link word</span>
    <input name="word" placeholder="joining word, e.g. has">
    <button>Add</button>
  </form>
  <form method="post" action="/admin" class="wfw-af">
    <input type="hidden" name="kind" value="synonym">{h}
    <span class="wfw-af-l">Synonym</span>
    <input name="word" placeholder="word in clue">
    <input name="synonym" placeholder="value/answer fragment">
    <button>Add</button>
  </form>
  <form method="post" action="/admin" class="wfw-af">
    <input type="hidden" name="kind" value="literal">{h}
    <span class="wfw-af-l">Literal</span>
    <input name="word" placeholder="function word, e.g. pe (used as its own letters)">
    <button>Add</button>
  </form>
  <form method="post" action="/admin" class="wfw-af">
    <input type="hidden" name="kind" value="homophone">{h}
    <span class="wfw-af-l">Homophone</span>
    <input name="word" placeholder="word in clue">
    <input name="homophone" placeholder="sounds like, e.g. air = heir">
    <button>Add</button>
  </form>
</details>
"""


# ---- atom-level hand-solver (atomised) ----------------------------------------
_HS_CSS = ("<style>"
           ".hs-catom{display:inline-flex;align-items:center;justify-content:center;"
           "min-width:1.3rem;height:2rem;border:1px solid #cbd5e1;border-radius:6px;"
           "font-family:'SF Mono','Courier New',monospace;font-weight:700;cursor:pointer;"
           "background:#fff}"
           ".hs-gap{display:inline-block;width:.7rem}"
           ".hs-atom{display:inline-flex;align-items:center;justify-content:center;"
           "min-width:2.2rem;height:2.4rem;margin:.15rem;border:2px solid #cbd5e1;"
           "border-radius:9px;font-weight:800;font-family:'SF Mono','Courier New',monospace;"
           "background:#fff}"
           ".hs-row{margin:.8rem 0}</style>")

HANDSOLVE_JS = """
function initHS(root, clueId, seed){
 const PAL=['#fca5a5','#fcd34d','#86efac','#93c5fd','#c4b5fd','#f9a8d4','#a5f3fc','#fdba74','#d9f99d','#f5d0fe'];
 const catoms=Array.from(root.querySelectorAll('.hs-catom'));
 const aatoms=Array.from(root.querySelectorAll('.hs-atom'));
 const piecesDiv=root.querySelector('.hs-pieces');
 const result=root.querySelector('.hs-result');
 let pieces=(seed.pieces||[]).map(function(p){return {text:p.text,value:p.value,mech:p.mech||'',atoms:new Set(p.atoms||[]),pos:new Set(),def:false};});
 if(seed.def){pieces.push({text:seed.def.text,value:'',mech:'definition',atoms:new Set(seed.def.atoms||[]),pos:new Set(),def:true});}
 let active=-1; let selClue=new Set(); let selAns=new Set();
 function esc(s){return (s||'').replace(/[&<>]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];});}
 function col(i){return PAL[i%PAL.length];}
 function atomEl(aid){return catoms.find(function(x){return x.dataset.aid===aid;});}
 function pcAtom(aid){for(let i=0;i<pieces.length;i++){if(pieces[i].atoms.has(aid))return i;}return -1;}
 function pcPos(p){for(let i=0;i<pieces.length;i++){if(pieces[i].pos.has(p))return i;}return -1;}
 function letters(aids){return aids.map(function(aid){var c=atomEl(aid);return c?c.textContent:'';}).join('').toUpperCase().replace(/[^A-Z]/g,'');}
 function pieceVal(pc){return pc.value||letters(Array.from(pc.atoms));}
 function draw(){
  catoms.forEach(function(a){var aid=a.dataset.aid;var pi=pcAtom(aid);
   a.style.background=pi>=0?(pieces[pi].def?'#cbd5e1':col(pi)):(selClue.has(aid)?'#1d4ed8':'#fff');
   a.style.color=(selClue.has(aid)&&pi<0)?'#fff':'#0f172a';
   a.style.outline=(pi>=0&&pi===active)?'2px solid #1d4ed8':'none';});
  aatoms.forEach(function(a){var p=+a.dataset.pos;var pi=pcPos(p);
   a.style.background=pi>=0?col(pi):(selAns.has(p)?'#1d4ed8':'#fff');
   a.style.color=(selAns.has(p)&&pi<0)?'#fff':'#0f172a';
   a.style.outline=(pi>=0&&pi===active)?'2px solid #1d4ed8':'none';});
  piecesDiv.innerHTML=pieces.map(function(pc,i){
   var aps=Array.from(pc.pos).sort(function(a,b){return a-b;}).join(',');
   var lbl; if(pc.def){lbl='DEF: '+esc(pc.text||letters(Array.from(pc.atoms)));}
   else if(pc.text){lbl=esc(pc.text)+' &rarr; '+esc(pieceVal(pc));}
   else {lbl=esc(pieceVal(pc))+' (literal)';}
   return '<span class="hs-chip" data-i="'+i+'" style="display:inline-block;cursor:pointer;background:'+(pc.def?'#e5e7eb':col(i))+';border:'+(i===active?'2px solid #1d4ed8':'1px solid #94a3b8')+';border-radius:6px;padding:.2rem .5rem;margin:.15rem;font-weight:700">'+lbl+(aps?(' @'+aps):'')+'</span>';
  }).join('')+'<div style="margin-top:.5rem"><button class="hs-lit" style="font-size:.8rem">link selected clue + answer atoms (literal)</button> <button class="hs-rm" style="font-size:.8rem;margin-left:.3rem">remove active piece</button></div>';
  Array.from(piecesDiv.querySelectorAll('.hs-chip')).forEach(function(ch){ch.onclick=function(){var i=+ch.dataset.i;active=(active===i)?-1:i;selClue.clear();selAns.clear();draw();};});
  piecesDiv.querySelector('.hs-lit').onclick=function(){
   if(!selClue.size||!selAns.size){alert('Select the clue atoms AND the answer atoms they make, then press this.');return;}
   pieces.push({text:'',value:letters(Array.from(selClue)),mech:'literal',atoms:new Set(selClue),pos:new Set(selAns),def:false});selClue.clear();selAns.clear();active=-1;draw();};
  piecesDiv.querySelector('.hs-rm').onclick=function(){if(active>=0){pieces.splice(active,1);active=-1;selClue.clear();selAns.clear();draw();}};
 }
 catoms.forEach(function(a){a.addEventListener('click',function(){
  var aid=a.dataset.aid;var owner=pcAtom(aid);
  if(active>=0){ if(pieces[active].atoms.has(aid))pieces[active].atoms.delete(aid); else if(owner<0)pieces[active].atoms.add(aid); }
  else { if(owner>=0)return; if(selClue.has(aid))selClue.delete(aid); else selClue.add(aid); }
  draw();});});
 aatoms.forEach(function(a){a.addEventListener('click',function(){
  var p=+a.dataset.pos;var owner=pcPos(p);
  if(active>=0){ if(pieces[active].pos.has(p))pieces[active].pos.delete(p); else if(owner<0)pieces[active].pos.add(p); }
  else { if(owner>=0)return; if(selAns.has(p))selAns.delete(p); else selAns.add(p); }
  draw();});});
 root.querySelector('.hs-verify').onclick=function(){result.innerHTML='<i>Verify (the gate) is wired next.</i>';};
 draw();
}
"""

HANDSOLVE_FORM = """
<form method="get" action="/handsolve" style="margin:1rem 0;font-family:system-ui">
  <label>Clue ID(s) / range:
    <input name="id" value="{cid}" placeholder="e.g. 10074740  or  10074740-10074745"
           style="font-size:1.1rem;padding:.3rem;width:24rem">
  </label>
  <button style="font-size:1.1rem;padding:.3rem .9rem">Hand-solve</button>
</form>
"""


def _parse_hs_ids(raw, cap=40):
    """Clue ids from the box: comma/space separated, A-B = inclusive range, capped."""
    ids = []
    for tok in raw.replace(",", " ").split():
        if "-" in tok:
            a, _, b = tok.partition("-")
            if a.isdigit() and b.isdigit():
                ids.extend(range(int(a), int(b) + 1)); continue
        if tok.isdigit():
            ids.append(int(tok))
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i); out.append(i)
        if len(out) >= cap:
            break
    return out


def _handsolve_block(clue_id):
    """One clue's hand-solve canvas. CARRIES ACROSS the candidate pieces the cascade
    already found (the same ones the clue page shows, read from the stored parse), plus
    the clue character-atoms (each selectable, so the possessive 's can be peeled) and the
    answer letter-atoms."""
    row = _load_clue(clue_id)
    if row is None:
        return f'<p class="warn">No clue with id {clue_id}.</p>'
    clue_text, answer, src, pnum, direction, enumeration, cnum = row
    answer = enum_space(answer, enumeration)
    ctx = build_wfw_atom_context(clue_text, answer, direction=direction)
    # CARRY ACROSS: the candidate pieces + definition the cascade already found.
    conn = store.connect()
    try:
        parse = store.load_parse(conn, clue_id)
    finally:
        conn.close()
    seed_pieces, seed_def = [], None
    if parse is not None:
        for s in (parse.sources or []):
            seed_pieces.append({"text": s.text, "value": s.value,
                                "mech": getattr(s, "mechanism", "") or "",
                                "atoms": list(getattr(s, "clue_atom_ids", []) or [])})
        if parse.definition is not None:
            seed_def = {"text": parse.definition.text,
                        "atoms": list(getattr(parse.definition, "clue_atom_ids", []) or [])}
    if seed_pieces or seed_def:
        carried = ''                         # shown as the interactive chips below
    else:
        carried = ('<div class="hs-row"><i>No stored solve for this clue yet &mdash; run it '
                   'on the main page first to carry its pieces across.</i></div>')
    import json as _json
    seed_json = _json.dumps({"pieces": seed_pieces, "def": seed_def})
    wordof, wi = {}, 0                       # atom_id -> index of the word it belongs to
    for t in ctx.clue_tokens:
        if t.kind == "word":
            for aid in t.atom_ids:
                wordof[aid] = wi
            wi += 1
    cbits = []
    for a in ctx.clue_atoms:
        if a.kind == "space":
            cbits.append('<span class="hs-gap"></span>')
        else:
            cbits.append('<span class="hs-catom" data-aid="%s" data-word="%d">%s</span>'
                         % (a.atom_id, wordof.get(a.atom_id, -1), escape(a.char)))
    atiles = "".join('<span class="hs-atom" data-pos="%d">%s</span>'
                     % (a.letter_position, escape(a.char))
                     for a in ctx.answer_atoms if a.kind == "letter")
    rid = "hs-%d" % clue_id
    return (_cid_label(clue_id, src, pnum, cnum, direction)
            + f'<div id="{rid}" class="hs-block" style="margin-bottom:2rem">'
            + f'<div class="wfw-clue" style="font-size:1.1rem;margin:.3rem 0">'
              f'{escape(clue_text)}</div>'
            + carried
            + '<div class="hs-row"><b>Clue atoms</b>:<br>' + "".join(cbits) + '</div>'
            + '<div class="hs-row"><b>Pieces from the solver</b> &mdash; click one to select '
              'it, then click the answer atoms it makes:</div>'
            + '<div class="hs-pieces"></div>'
            + '<div class="hs-row"><b>Answer</b>:<br>' + atiles + '</div>'
            + '<div class="hs-row" style="font-size:.85rem;color:#475569">'
              '<b>Derivative piece</b> (e.g. Tea&rarr;CHA): click its chip, then the answer '
              'atoms it makes. <b>Literal letters</b> (e.g. RADE from <i>grader</i>): with no '
              'chip selected, click the clue atoms and the answer atoms they make, then press '
              '&ldquo;link selected clue + answer atoms&rdquo;. With a chip active, click its '
              'clue atoms to add/remove them (to peel a letter like the &rsquo;s).</div>'
            + '<div class="hs-row"><button class="hs-verify wfw-reload" '
              'style="background:#16a34a;border-color:#16a34a">Verify</button></div>'
            + '<div class="hs-result" style="margin-top:1rem"></div>'
            + '</div>'
            + f'<script>initHS(document.getElementById("{rid}"), {clue_id}, {seed_json});</script>')


# ---------------------------------------------------------------------------------
# ROLE-GRID hand-solver (the redesign's hand-solver; supersedes the atom one above).
# v1 (incremental): DISPLAY each clue word with the role the current solver assigned,
# and let the admin FORCE AN INDICATOR (a contiguous span -> a chosen indicator type),
# which is the one role-primitive the page did not already have (force-definition and
# filler exist). Fodder assignment + inline enrichment are follow-ups.
# ---------------------------------------------------------------------------------

# Indicator types offered when forcing an indicator: (value, label). Positional ones carry
# their subtype encoded 'charade_positional:after' (clue_overrides splits it).
_FORCE_IND_OPTIONS = (
    ("anagram", "anagram"), ("container", "container"), ("insertion", "insertion"),
    ("reversal", "reversal"), ("deletion", "deletion"), ("hidden", "hidden"),
    ("homophone", "homophone"), ("acrostic", "acrostic"),
    ("alternation", "alternation"), ("selection", "selection"),
    ("charade_positional:after", "positional — after"),
    ("charade_positional:before", "positional — before"),
)
_FORCE_IND_TYPES = frozenset(v for v, _ in _FORCE_IND_OPTIONS)


# mechanism -> friendly role label shown in the grid
_MECH_LABEL = {"synonym": "synonym", "abbreviation": "abbreviation",
               "first_letter": "first letter", "last_letter": "last letter",
               "selection": "selection", "anagram_fodder": "anagram fodder",
               "raw": "literal", "homophone": "homophone"}


def _word_roles(ctx, parse, filler_set):
    """Map each clue WORD to the role the stored parse gives it. Returns a list of dicts
    {idx, text, role, label, value} in clue order, where `role` is the colour category
    (definition/piece/indicator/link/filler/none), `label` is what to show, and `value`
    is the letters the piece produced (e.g. D, TORS) — so the synonym IS shown."""
    # atom_id -> (role, label, value)
    amap = {}
    if parse is not None:
        if parse.definition is not None:
            # A guessed (source='pending') definition is shown as "unidentified definition"
            # here too, so the grid never presents the floor's edge guess as confirmed
            # (memory: definition-floor-redesign). Role category stays 'definition'.
            _dlabel = ("unidentified definition"
                       if getattr(parse.definition, "source", "db") == "pending"
                       else "definition")
            for aid in (parse.definition.clue_atom_ids or ()):
                amap[aid] = ("definition", _dlabel, "")
        for s in (parse.sources or []):
            mech = getattr(s, "mechanism", "") or ""
            label = _MECH_LABEL.get(mech, mech or "piece")
            val = (getattr(s, "value", "") or "")
            for aid in (s.clue_atom_ids or ()):
                amap[aid] = ("piece", label, val)
        for a in (parse.annotations or []):
            note = getattr(a, "note", "") or a.role
            for aid in (a.clue_atom_ids or ()):
                amap[aid] = (a.role, note, "")
    out, wi = [], 0
    fil = {(x or "").strip().lower() for x in (filler_set or ())}
    for t in ctx.clue_tokens:
        if t.kind != "word":
            continue
        role, label, value = "none", "—", ""
        for aid in t.atom_ids:
            if aid in amap:
                role, label, value = amap[aid]
                break
        if role == "none" and (t.text or "").strip().lower() in fil:
            role, label = "filler", "filler"
        out.append({"idx": wi, "text": t.text, "role": role,
                    "label": label, "value": value})
        wi += 1
    return out


def _rolegrid_block(clue_id, back_raw=None):
    """One clue's role grid: the words listed vertically with their current roles, the
    forced-indicator control, current forces, and (if frozen) the unforce control.

    `back_raw` is the id-string of the CLUTCH this clue was opened within (space-joined
    ids). The "back to clue page" link returns to that whole clutch, anchored to THIS
    clue (#clue-<id>), so solving one clue in a batch does not collapse the view to a
    single clue. Defaults to this clue alone when opened on its own."""
    row = _load_clue(clue_id)
    if row is None:
        return f'<p class="warn">No clue with id {clue_id}.</p>'
    clue_text, answer, src, pnum, direction, enumeration, cnum = row
    answer = enum_space(answer, enumeration)
    ctx = build_wfw_atom_context(clue_text, answer, direction=direction)
    conn = store.connect()
    try:
        parse = store.load_parse(conn, clue_id)
        filler = store.get_clue_filler(conn, clue_id)
        forced = store.get_forced_definition(conn, clue_id)
        forced_ind = store.get_forced_indicators(conn, clue_id)
        frozen = store.is_frozen(conn, clue_id)
    finally:
        conn.close()
    status = parse.status if parse is not None else "fail"
    rows = _word_roles(ctx, parse, filler)
    h = ('<input type="hidden" name="id" value="%s">'
         '<input type="hidden" name="only" value="%d">'
         '<input type="hidden" name="surface" value="grid">'
         '<input type="hidden" name="from" value="%s">'
         % (escape(str(clue_id), quote=True), clue_id,
            escape(back_raw or str(clue_id), quote=True)))

    # Per-word DB values (synonyms/abbreviations the reference DB holds for that word), so a
    # freshly-added synonym is VISIBLE here even when the clue does not fully solve yet.
    w_db = batch_wiring()
    lookup_all = w_db["lookup_all"]

    def db_values(word):
        seen, out = set(), []
        try:
            for v, m in lookup_all(word):
                v = (v or "").upper()
                if m in ("synonym", "abbreviation") and v and v not in seen:
                    seen.add(v)
                    out.append(v)
                if len(out) >= 10:
                    break
        except Exception:
            pass
        return out

    # ONE editable form: every word has a role dropdown defaulted to "keep" (showing the
    # role the solver gave it). You change only the wrong ones, then Apply once. Adjacent
    # words set to the same role form one phrase; a synonym value is reused from what is
    # already known (the "makes"/"DB has" columns) so you rarely type one.
    ind_opts = "".join('<option value="indicator:%s">indicator — %s</option>' % (v, lbl)
                       for v, lbl in _FORCE_IND_OPTIONS)

    def role_select(idx, current_label):
        return (
            '<select name="role_%d" form="gridapply-%d" class="rg-rsel">'
            '<option value="keep" selected>keep (%s)</option>'
            '<option value="definition">definition</option>'
            '%s'
            '<option value="synonym">synonym</option>'
            '<option value="link">link word</option>'
            '<option value="filler">filler</option>'
            '</select>' % (idx, clue_id, escape(current_label or "—"), ind_opts))

    grid = ['<table class="rg-tbl">'
            '<tr><th>word</th><th>role</th><th>value<br><span class="rg-th2">'
            '(synonym only, if new)</span></th><th>makes</th><th>DB has</th></tr>']
    for r in rows:
        val = ('<span class="rg-val">%s</span>' % escape(r["value"])) if r["value"] else ""
        dbv = db_values(r["text"])
        dbcell = ('<span class="rg-dbv">%s</span>'
                  % escape(", ".join(dbv))) if dbv else '<span class="rg-dbnone">—</span>'
        grid.append(
            '<tr><td class="rg-word">%s</td>'
            '<td>%s</td>'
            '<td><input name="val_%d" form="gridapply-%d" size="10"></td>'
            '<td>%s</td><td>%s</td></tr>'
            % (escape(r["text"]), role_select(r["idx"], r["label"]),
               r["idx"], clue_id, val, dbcell))
    grid.append("</table>")

    set_form = (
        '<form method="post" action="/gridapply" id="gridapply-%d" class="rg-set">%s'
        '<div class="rg-setrow"><button class="rg-apply">Apply &amp; re-solve</button>'
        '<span class="rg-note">Change only the dropdowns that are wrong, then click once. '
        'Adjacent words with the same role become one phrase (e.g. set &ldquo;some&rdquo; to '
        'synonym next to a discovered &ldquo;girls&rdquo; to make &ldquo;some girls&rdquo;). '
        'A synonym value is reused from &ldquo;makes/DB has&rdquo; &mdash; only type one if '
        'those are empty.</span></div>'
        '</form>' % (clue_id, h))

    # Current overrides on this clue, each with a one-click clear.
    cur_items = []
    if forced:
        cur_items.append(
            'definition pinned to <b>%s</b> '
            '<form method="post" action="/clearforcedef" class="rg-inline">%s'
            '<button class="rg-x">clear</button></form>' % (escape(forced), h))
    for p, t in forced_ind:
        cur_items.append(
            'indicator <b>%s</b> = %s '
            '<form method="post" action="/clearforceind" class="rg-inline">%s'
            '<input type="hidden" name="phrase" value="%s">'
            '<button class="rg-x">clear</button></form>'
            % (escape(p), escape(t), h, escape(p, quote=True)))
    for w in sorted(filler):
        cur_items.append(
            'filler <b>%s</b> '
            '<form method="post" action="/clearfiller" class="rg-inline">%s'
            '<input type="hidden" name="word" value="%s">'
            '<button class="rg-x">clear</button></form>' % (escape(w), h,
                                                            escape(w, quote=True)))
    cur = ('<div class="rg-cur"><b>Overrides on this clue:</b> '
           + " &nbsp;·&nbsp; ".join(cur_items) + '</div>') if cur_items else ""

    unforce = ""
    if frozen:
        unforce = ('<div class="rg-frozen">&#128274; FROZEN — forced pass, will not revert. '
                   '<form method="post" action="/unforce" class="rg-inline">%s'
                   '<button class="rg-x">unforce &amp; re-solve</button></form></div>' % h)

    from urllib.parse import quote
    back_id = back_raw if back_raw else str(clue_id)
    back_q = quote(back_id, safe="")
    back = ('<div class="rg-back"><a href="/?id=%s#clue-%d">'
            '&larr; back to clue page</a></div>' % (back_q, clue_id))
    # PREV/NEXT within the hand solver: step through the CLUTCH (the `from` context)
    # without leaving the role grid. Shown only when this clue sits in a clutch of >1.
    nav = ""
    clutch_ids = _parse_hs_ids(back_id)
    if clue_id in clutch_ids and len(clutch_ids) > 1:
        pos = clutch_ids.index(clue_id)

        def _arrow(nid, label):
            if nid is None:
                return '<span class="rg-nav rg-nav-off">%s</span>' % label
            return ('<a class="rg-nav" href="/rolegrid?id=%d&amp;from=%s">%s</a>'
                    % (nid, back_q, label))
        prev_id = clutch_ids[pos - 1] if pos > 0 else None
        next_id = clutch_ids[pos + 1] if pos < len(clutch_ids) - 1 else None
        nav = ('<div class="rg-navrow">%s<span class="rg-navpos">%d of %d</span>%s</div>'
               % (_arrow(prev_id, "&larr; prev clue"), pos + 1, len(clutch_ids),
                  _arrow(next_id, "next clue &rarr;")))
    return (
        _cid_label(clue_id, src, pnum, cnum, direction)
        + '<div class="rg-block">'
        + back
        + nav
        + '<div class="rg-clue">%s</div>' % escape(clue_text)
        + '<div class="rg-status rg-st-%s">%s — %s</div>'
          % (status, escape(answer), status.upper())
        + unforce
        + "".join(grid)
        + set_form + cur
        + '</div>')


_RG_CSS = """<style>
.rg-block{border:1px solid #94a3b8;border-radius:8px;padding:1.1rem 1.2rem;
  margin:.6rem 0 2rem;font-family:system-ui;color:#0f172a;font-size:1rem}
.rg-clue{font-size:1.3rem;font-weight:600;margin:.2rem 0 .5rem;color:#0f172a}
.rg-status{font-size:1.1rem;font-weight:700;margin:.2rem 0 .9rem;letter-spacing:.04em}
.rg-status.rg-st-pass{color:#15803d}.rg-status.rg-st-fail{color:#b91c1c}
.rg-status.rg-st-pending{color:#b45309}
.rg-tbl{border-collapse:collapse;margin:.4rem 0 1rem;font-size:1rem}
.rg-tbl th{text-align:left;font-size:.8rem;color:#334155;font-weight:700;
  padding:.25rem .8rem;border-bottom:2px solid #cbd5e1}
.rg-tbl td{padding:.35rem .8rem;border-top:1px solid #e2e8f0;color:#0f172a}
.rg-pick{text-align:center}.rg-tbl input[type=checkbox]{width:1.1rem;height:1.1rem}
.rg-word{font-weight:700;font-size:1.05rem}
.rg-role{font-weight:600}
.rg-role.rg-definition{color:#0369a1}
.rg-role.rg-indicator{color:#7c3aed}
.rg-role.rg-piece{color:#15803d}
.rg-role.rg-link{color:#64748b}
.rg-role.rg-filler{color:#64748b}
.rg-role.rg-none{color:#b91c1c}
.rg-val{font-weight:700;font-family:ui-monospace,Menlo,Consolas,monospace;
  background:#f1f5f9;padding:.05rem .4rem;border-radius:4px;color:#0f172a}
.rg-dbv{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.85rem;color:#334155}
.rg-dbnone{color:#cbd5e1}
.rg-back{margin:0 0 .5rem}.rg-back a{color:#0d9488;text-decoration:none;font-weight:600}
.rg-navrow{display:flex;align-items:center;gap:1rem;margin:0 0 .6rem;font-size:1rem}
.rg-nav{color:#0d9488;text-decoration:none;font-weight:700;padding:.25rem .7rem;
  border:1px solid #0d9488;border-radius:6px}
.rg-nav:hover{background:#0d9488;color:#fff}
.rg-nav-off{color:#cbd5e1;border-color:#e2e8f0;cursor:default}
.rg-navpos{color:#64748b;font-size:.85rem}
.rg-set{margin:.6rem 0;background:#f8fafc;border:1px solid #cbd5e1;border-radius:6px;
  padding:.7rem .9rem}
.rg-setrow{display:flex;gap:.6rem;align-items:center;flex-wrap:wrap;font-size:1rem}
.rg-rolesel{font-size:1rem;padding:.2rem}
.rg-set select,.rg-set input{font-size:1rem;padding:.2rem .3rem}
.rg-set button{font-size:1rem;padding:.3rem .9rem;background:#0d9488;color:#fff;
  border:1px solid #0d9488;border-radius:6px;cursor:pointer}
.rg-rsel{font-size:.95rem;padding:.2rem}
.rg-apply{font-size:1.05rem!important;font-weight:600;padding:.45rem 1.2rem!important}
.rg-th2{font-weight:400;color:#94a3b8;font-size:.75rem}
.rg-cond{display:none;align-items:center;gap:.3rem}
.rg-note{font-size:.85rem;color:#475569;margin-top:.5rem}
.rg-cur{margin:.7rem 0;font-size:.95rem;color:#0f172a}
.rg-frozen{margin:.4rem 0 .8rem;font-size:1rem;font-weight:600;color:#b45309}
.rg-inline{display:inline}
.rg-x{font-size:.75rem;padding:.1rem .45rem;cursor:pointer}
</style>
<script>
// show the conditional field (type / value) only for the chosen role
document.addEventListener('change', function(e){
  if(!e.target.classList || !e.target.classList.contains('rg-rolesel')) return;
  var form=e.target.closest('form'), role=e.target.value;
  form.querySelectorAll('.rg-cond').forEach(function(c){ c.style.display='none'; });
  var show=form.querySelector('.rg-cond-'+role);
  if(show) show.style.display='inline-flex';
});
// set the initial state on load
document.addEventListener('DOMContentLoaded', function(){
  document.querySelectorAll('.rg-rolesel').forEach(function(s){
    s.dispatchEvent(new Event('change',{bubbles:true})); });
});
</script>"""

ROLEGRID_FORM = ('<form method="get" action="/rolegrid" style="margin:1rem 0;'
                 'font-family:system-ui">'
                 '<input name="id" value="{cid}" placeholder="clue id" size="12">'
                 '<button>Open role grid</button></form>')


@app.route("/rolegrid")
def rolegrid_route():
    """Role-grid hand-solver for one clue (or several, comma/space/range separated)."""
    raw = (request.args.get("id") or "").strip()
    notice = (request.args.get("notice") or "").strip()
    back_from = (request.args.get("from") or "").strip()   # the clue-page CLUTCH, if any
    ids = _parse_hs_ids(raw)
    marker = ('<div style="font-size:.75rem;color:#94a3b8;margin:.3rem 0">'
              'hand-solver build %s</div>' % escape(BOOT_ID))
    head = marker
    if notice:
        head += '<div class="wfw-notice">%s</div>' % escape(notice)
    body = _RG_CSS + head + ROLEGRID_FORM.format(cid=escape(raw, quote=True))
    if not ids:
        return _page(body + '<p class="warn">Enter a clue id, e.g. 10075290.</p>')
    # Back link returns to the CLUTCH this clue was opened within: the `from` clue-page
    # id-string when present (so opening ONE clue from a clutch still returns to the
    # whole clutch), else the ids rendered here. Carried through grid POSTs via `from`.
    back_ids = _parse_hs_ids(back_from) if back_from else ids
    back_raw = " ".join(str(i) for i in (back_ids or ids))
    for cid in ids:
        body += _rolegrid_block(cid, back_raw=back_raw)
    return _page(body)


@app.route("/gridrole", methods=["POST"])
def gridrole_route():
    """Set the ticked clue words to a role from the role grid, then re-solve. One entry
    point dispatching by `role`:
      definition -> pin this clue's definition to the span (per-clue, store.forced_def)
      indicator  -> force the span as an indicator of `wptype` (per-clue, forced_indicator)
      filler     -> tag each ticked word as surface filler (per-clue)
      synonym    -> add `span = value` to the reference DB (global) — the FOX->TOD case
    """
    only = (request.form.get("only") or "").strip()
    role = (request.form.get("role") or "").strip().lower()
    wptype = (request.form.get("wptype") or "").strip().lower()
    value = (request.form.get("value") or "").strip()
    widxs = sorted(int(x) for x in request.form.getlist("w") if x.isdigit())
    msg = "Tick one or more words first."
    if only and widxs:
        row = _load_clue(int(only))
        contiguous = _contiguous(widxs)
        phrase = _span_phrase(row[0], row[4], widxs) if row is not None else ""
        if not phrase:
            msg = "Could not read the selected words."
        elif role in ("definition", "indicator", "synonym") and not contiguous:
            msg = "For a %s the ticked words must be contiguous (one phrase)." % role
        elif role == "definition":
            conn = store.connect()
            try:
                store.set_forced_definition(conn, int(only), phrase)
            finally:
                conn.close()
            _resolve_one(int(only))
            msg = "Definition set to %r (this clue); re-solved." % phrase
        elif role == "indicator" and wptype in _FORCE_IND_TYPES:
            conn = store.connect()
            try:
                store.add_forced_indicator(conn, int(only), phrase, wptype)
            finally:
                conn.close()
            _resolve_one(int(only))
            label = dict(_FORCE_IND_OPTIONS).get(wptype, wptype)
            msg = "Forced %r as a %s indicator (this clue); re-solved." % (phrase, label)
        elif role == "filler":
            conn = store.connect()
            try:
                for i in widxs:
                    store.add_clue_filler(conn, int(only), _span_phrase(row[0], row[4], [i]))
            finally:
                conn.close()
            _resolve_one(int(only))
            msg = "Tagged %r as filler (this clue); re-solved." % phrase
        elif role == "synonym" and value:
            add_form = {"kind": "synonym", "word": phrase, "synonym": value}
            addmsg = _do_add(add_form)
            apply_add_to_wiring(add_form)
            _resolve_one(int(only))
            msg = "Added synonym %r = %r to the reference DB; re-solved. (%s)" % (
                phrase, value, addmsg)
        elif role == "synonym":
            msg = "Type the synonym value (e.g. TOD) before applying."
        else:
            msg = "Choose a role (and its type/value) before applying."
    return _grid_redirect(only, msg)


@app.route("/gridapply", methods=["POST"])
def gridapply_route():
    """Apply ALL role choices from the grid at once, then re-solve ONCE. Each word carries a
    role_<idx> (default 'keep'); adjacent words with the SAME role form one phrase. A synonym
    value is taken from val_<idx> if typed, else REUSED from a known DB value of the word(s)
    that is a substring of the answer (so an existing synonym is extended without retyping)."""
    only = (request.form.get("only") or "").strip()
    if not only:
        return _grid_redirect("", "No clue.")
    cid = int(only)
    row = _load_clue(cid)
    if row is None:
        return _grid_redirect(only, "No clue.")
    clue_text, answer, src, pnum, direction, enumeration, cnum = row
    ctx = build_wfw_atom_context(clue_text, enum_space(answer, enumeration),
                                 direction=direction)
    words = [t.text for t in ctx.clue_tokens if t.kind == "word"]
    n = len(words)
    roles = [(request.form.get("role_%d" % i) or "keep").strip() for i in range(n)]
    vals = [(request.form.get("val_%d" % i) or "").strip() for i in range(n)]

    # group consecutive words sharing the same non-keep role into one phrase
    groups, i = [], 0
    while i < n:
        if roles[i] == "keep":
            i += 1
            continue
        j = i
        while j + 1 < n and roles[j + 1] == roles[i]:
            j += 1
        groups.append((roles[i], list(range(i, j + 1))))
        i = j + 1

    ans_letters = "".join(c for c in (answer or "").upper() if c.isalpha())
    la = batch_wiring()["lookup_all"]

    def reuse_value(idxs):
        """The longest already-known DB value (synonym/abbr) of any word in the group that
        is a substring of the answer — i.e. the synonym the grid already shows."""
        best = ""
        for k in idxs:
            try:
                for v, m in la(words[k]):
                    v = (v or "").upper()
                    if (m in ("synonym", "abbreviation") and v and v in ans_letters
                            and len(v) > len(best)):
                        best = v
            except Exception:
                pass
        return best

    msgs, need = [], []
    conn = store.connect()
    try:
        for role, idxs in groups:
            phrase = " ".join(words[k] for k in idxs)
            if role == "definition":
                store.set_forced_definition(conn, cid, phrase)
                msgs.append("definition = %r" % phrase)
            elif role.startswith("indicator:"):
                wptype = role[len("indicator:"):]
                if wptype in _FORCE_IND_TYPES:
                    store.add_forced_indicator(conn, cid, phrase, wptype)
                    label = dict(_FORCE_IND_OPTIONS).get(wptype, wptype)
                    msgs.append("%r = %s indicator" % (phrase, label))
            elif role == "filler":
                for k in idxs:
                    store.add_clue_filler(conn, cid, words[k])
                msgs.append("%r = filler" % phrase)
            elif role == "link":
                _do_add({"kind": "link", "word": phrase})
                apply_add_to_wiring({"kind": "link", "word": phrase})
                msgs.append("%r = link word" % phrase)
            elif role == "synonym":
                v = next((vals[k] for k in idxs if vals[k]), "") or reuse_value(idxs)
                if v:
                    v = v.upper()
                    _do_add({"kind": "synonym", "word": phrase, "synonym": v})
                    apply_add_to_wiring({"kind": "synonym", "word": phrase, "synonym": v})
                    msgs.append("%r = %s (synonym)" % (phrase, v))
                else:
                    need.append(phrase)
    finally:
        conn.close()
    _resolve_one(cid)
    parts = []
    if msgs:
        parts.append("Applied: " + "; ".join(msgs))
    if need:
        parts.append("type a value for synonym(s): " + ", ".join("%r" % p for p in need))
    if not msgs and not need:
        parts.append("No changes")
    return _grid_redirect(only, " — ".join(parts) + "; re-solved.")


@app.route("/forceind", methods=["POST"])
def forceind_route():
    """Force a contiguous span of clue words to be an indicator of the chosen type, for
    THIS clue only, then re-solve. Stored in wfw_forced_indicator (never the shared
    indicators table); applied via clue_overrides on every solve path."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    wptype = (request.form.get("wptype") or "").strip().lower()
    widxs = sorted(int(x) for x in request.form.getlist("w") if x.isdigit())
    msg = "Select one or more contiguous words and a type."
    if only and wptype and widxs and wptype in _FORCE_IND_TYPES:
        row = _load_clue(int(only))
        if row is not None and _contiguous(widxs):
            phrase = _span_phrase(row[0], row[4], widxs)
            if phrase:
                conn = store.connect()
                try:
                    store.add_forced_indicator(conn, int(only), phrase, wptype)
                finally:
                    conn.close()
                msg = ("Forced %r as a %s indicator (this clue only); re-solved."
                       % (phrase, wptype))
        elif not _contiguous(widxs):
            msg = "Selected words must be contiguous (one phrase)."
        else:
            _resolve_one(int(only))
    return _grid_redirect(only, msg)


@app.route("/gridadd", methods=["POST"])
def gridadd_route():
    """Inline enrichment from the role grid: ADD a reference-DB entry for the selected span
    (synonym/definition/indicator/link), then re-solve. Reuses the proven _do_add +
    apply_add_to_wiring path (same as /admin), so the add is global and visible at once.
    This is the FOX->TOD flow: select the word, choose 'synonym fodder', type TOD, add."""
    only = (request.form.get("only") or "").strip()
    kind = (request.form.get("kind") or "").strip().lower()
    value = (request.form.get("value") or "").strip()
    wptype = (request.form.get("type") or "").strip().lower()
    answer = (request.form.get("answer") or "").strip()
    widxs = sorted(int(x) for x in request.form.getlist("w") if x.isdigit())
    msg = "Select one or more contiguous words, a kind, and (where needed) a value."
    if only and kind and widxs and _contiguous(widxs):
        row = _load_clue(int(only))
        phrase = _span_phrase(row[0], row[4], widxs) if row is not None else ""
        if phrase:
            add_form = None
            if kind == "synonym" and value:
                add_form = {"kind": "synonym", "word": phrase, "synonym": value}
            elif kind == "definition":
                add_form = {"kind": "definition", "definition": phrase, "answer": answer}
            elif kind == "indicator" and wptype:
                add_form = {"kind": "indicator", "word": phrase, "type": wptype,
                            "subtype": ""}
            elif kind == "link":
                add_form = {"kind": "link", "word": phrase}
            if add_form is None:
                msg = "Missing a required value for that kind (e.g. the synonym value)."
            else:
                msg = _do_add(add_form)
                apply_add_to_wiring(add_form)
                _resolve_one(int(only))
                msg = "Added (%s) for %r: %s; clue re-solved." % (kind, phrase, msg)
    elif widxs and not _contiguous(widxs):
        msg = "Selected words must be contiguous (one phrase)."
    return _grid_redirect(only, msg)


@app.route("/clearforceind", methods=["POST"])
def clearforceind_route():
    """Remove one forced indicator (by phrase) and re-solve."""
    only = (request.form.get("only") or "").strip()
    phrase = (request.form.get("phrase") or "").strip()
    if only and phrase:
        conn = store.connect()
        try:
            store.clear_forced_indicator(conn, int(only), phrase)
        finally:
            conn.close()
        _resolve_one(int(only))
    return _grid_redirect(only, "Forced indicator cleared; re-solved.")


def _contiguous(idxs):
    return bool(idxs) and idxs == list(range(idxs[0], idxs[0] + len(idxs)))


def _span_phrase(clue_text, direction, widxs):
    """The surface phrase for the given word indices (clue order)."""
    ctx = build_wfw_atom_context(clue_text, "X", direction=direction)
    words = [t.text for t in ctx.clue_tokens if t.kind == "word"]
    if not widxs or widxs[-1] >= len(words):
        return ""
    return " ".join(words[i] for i in widxs)


def _resolve_one(clue_id):
    """Re-solve a single clue through the cascade (DB-only) WITH its overrides applied, and
    persist — so a freshly-forced role takes effect immediately. Mirrors the page's solve
    path (batch wiring + clue_overrides). Best-effort; never raises to the route."""
    try:
        row = _load_clue(clue_id)
        if row is None:
            return
        clue_text, answer, src, pnum, direction, enumeration, cnum = row
        answer = enum_space(answer, enumeration)
        from core import clue_overrides
        w = clue_overrides.apply_forced_overrides(batch_wiring(), clue_id)
        engine_registry.solve_clue_text(clue_text, answer, w, source=src,
                                        puzzle_number=pnum, clue_id=clue_id,
                                        direction=direction)
    except Exception:
        pass


# ============================================================================
# SPAN-ASSIGNMENT HAND-SOLVER (redesign 2026-06-24) — additive, alongside the old grid.
# VERTICAL grid: one ROW per clue word, preloaded with its CURRENT role. Columns:
# check / word / role / BRINGS. You tick a group, pick a role (synonym values come from a
# DB lookup, never a solve), Assign (held in MEMORY only — no DB write, no solve), repeat,
# then Resolve ONCE (the single commit: DB writes + one solve). Non-contiguous ticks are
# allowed (anagram fodder). Memory: handsolver-redesign-direction.
# Routes: /hs (grid) + /hslookup (AJAX candidates) + /hsresolve (commit all + solve).
# ============================================================================

_HS_ROLE_COLOUR = {"definition": "#0f766e", "piece": "#1d4ed8", "indicator": "#7c3aed",
                   "link": "#64748b", "filler": "#9333ea", "none": "#ffffff"}

_SPAN_CSS = """<style>
.g-root{font-family:system-ui;margin:1rem 0}
.g-ans{font-size:1.05rem;margin:.3rem 0 .6rem}
.g-ans b{font-family:'SF Mono',monospace;letter-spacing:.15em}
.g-tbl{border-collapse:collapse;width:100%;max-width:46rem}
.g-tbl th{font-size:.7rem;text-transform:uppercase;letter-spacing:.04em;color:#64748b;text-align:left;padding:.2rem .5rem}
.g-tbl td{padding:.3rem .5rem;border-top:1px solid #eef2f7;vertical-align:middle}
.g-tbl td.g-word{font-weight:700;font-size:1.02rem}
.g-chk{width:1.1rem;height:1.1rem;cursor:pointer}
.r-brings{font-family:'SF Mono',monospace;color:#0f172a}
.g-bar{display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;padding:.55rem .8rem;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;margin:.6rem 0}
.g-bar select,.g-bar input{font-size:.95rem;padding:.25rem .45rem;border:1px solid #cbd5e1;border-radius:6px}
.g-assign{background:#0d9488;color:#fff;border:none;border-radius:8px;padding:.3rem .8rem;font-weight:700;cursor:pointer}
.g-list{margin:.5rem 0;display:flex;flex-wrap:wrap;gap:.4rem}
.g-tag{border:1px solid #cbd5e1;border-radius:8px;padding:.2rem .55rem;font-size:.9rem;background:#fff}
.g-tag .g-rm{margin-left:.35rem;color:#dc2626;text-decoration:none;font-weight:800}
.g-resolve{background:#1d4ed8;color:#fff;border:none;border-radius:10px;padding:.45rem 1rem;font-weight:800;cursor:pointer;font-size:1rem;margin-top:.4rem}
.g-card{margin-top:1rem}
</style>"""

_SPAN_JS = r"""
function initGrid(rootId, DATA){
 var root=document.getElementById(rootId);
 var assignments=(DATA.assignments||[]);
 var tbody=root.querySelector('#g-tbody');
 var bar=root.querySelector('#g-bar'), selLbl=root.querySelector('#g-sel');
 var roleSel=root.querySelector('#g-role'), itype=root.querySelector('#g-itype'), isub=root.querySelector('#g-isub');
 var candWrap=root.querySelector('#g-cand'), candSel=root.querySelector('#g-candsel'), addInp=root.querySelector('#g-add'), delEl=root.querySelector('#g-del');
 var listDiv=root.querySelector('#g-list'), payload=root.querySelector('#g-payload');
 var ROLECOL={definition:'#0f766e',synonym:'#1d4ed8',letters:'#0891b2',indicator:'#7c3aed',link:'#64748b',filler:'#9333ea',none:'#94a3b8'};
 function saveAssignments(){try{var fd=new FormData();fd.append('only',DATA.cid);fd.append('payload',JSON.stringify(assignments));fetch('/hssave',{method:'POST',body:fd});}catch(e){}}
 function checkedIdx(){return Array.prototype.slice.call(tbody.querySelectorAll('input.g-chk:checked')).map(function(c){return +c.value;}).sort(function(a,b){return a-b;});}
 function phraseOf(idx){return idx.map(function(i){return DATA.words[i];}).join(' ');}
 function assignOf(i){for(var k=0;k<assignments.length;k++){if(assignments[k].idx.indexOf(i)>=0)return assignments[k];}return null;}
 function drawRows(){
  Array.prototype.slice.call(tbody.querySelectorAll('tr')).forEach(function(tr){
   var i=+tr.dataset.i, a=assignOf(i);
   var rc=tr.querySelector('.r-role'), bc=tr.querySelector('.r-brings');
   if(a){var col=ROLECOL[a.role]||'#334155';
    rc.innerHTML='<b style="color:'+col+'">'+a.role+(a.isub?('/'+a.isub):'')+'</b>';
    bc.textContent=(a.role==='synonym'||a.role==='letters')?(a.value||''):'';
    tr.style.background='#f8fafc';
   }else{var c=DATA.current[i]||{};
    rc.innerHTML='<span style="color:#94a3b8">'+(c.label||'—')+'</span>';
    bc.textContent=c.value||'';
    tr.style.background='';
   }
  });
 }
 function drawList(){
  listDiv.innerHTML=assignments.map(function(a,k){
   var col=ROLECOL[a.role]||'#334155';
   var v=(a.role==='synonym'||a.role==='letters')?(' = '+a.value):((a.role==='indicator')?(' ('+a.itype+(a.isub?('/'+a.isub):'')+')'):'');
   return '<span class="g-tag" style="border-color:'+col+'"><b style="color:'+col+'">'+a.role+'</b> '+phraseOf(a.idx)+v+' <a href="#" data-k="'+k+'" class="g-rm">×</a></span>';
  }).join('');
  Array.prototype.slice.call(listDiv.querySelectorAll('.g-rm')).forEach(function(x){x.onclick=function(e){e.preventDefault();assignments.splice(+x.dataset.k,1);drawRows();drawList();saveAssignments();};});
 }
 function updateBar(){var idx=checkedIdx();if(idx.length){bar.style.display='';selLbl.textContent=phraseOf(idx);}else{bar.style.display='none';}}
 function clearChecks(){Array.prototype.slice.call(tbody.querySelectorAll('input.g-chk')).forEach(function(c){c.checked=false;});updateBar();}
 function inferSub(){
  if(roleSel.value!=='indicator'||itype.value!=='deletion')return;
  var syn='';assignments.forEach(function(a){if(a.role==='synonym'&&a.value)syn=a.value;});
  if(!syn)return;
  fetch('/hsinfer?base='+encodeURIComponent(syn)+'&answer='+encodeURIComponent(DATA.answer)).then(function(r){return r.json();}).then(function(o){if(o&&o.subtype)isub.value=o.subtype;});
 }
 function roleFields(){var r=roleSel.value;
  itype.style.display=(r==='indicator')?'':'none';
  isub.style.display=(r==='indicator'&&itype.value==='deletion')?'':'none';
  candWrap.style.display=(r==='synonym'||r==='letters')?'':'none';
  if(candSel)candSel.style.display=(r==='synonym')?'':'none';   // 'letters' = type only, no
  if(delEl)delEl.style.display=(r==='synonym')?'':'none';        // DB candidates / no prune
  if(addInp)addInp.placeholder=(r==='letters')?'exact letters, e.g. G':'new value';
  if(r==='synonym')fetchCands();
  if(r==='indicator'&&itype.value==='deletion')inferSub();
 }
 function delRow(word,value){var f=document.createElement('form');f.method='post';f.action='/hsdelete';
  function h(n,v){var i=document.createElement('input');i.type='hidden';i.name=n;i.value=v;f.appendChild(i);}
  h('only',DATA.cid);h('from',DATA.back||DATA.cid);h('kind','synonym');h('word',word);h('value',value);
  document.body.appendChild(f);f.submit();}
 function fetchCands(){var idx=checkedIdx();if(!idx.length){candSel.innerHTML='';if(delEl)delEl.innerHTML='';return;}
  candSel.innerHTML='<option>…</option>';var phr=phraseOf(idx);
  fetch('/hslookup?id='+DATA.cid+'&phrase='+encodeURIComponent(phr)).then(function(r){return r.json();}).then(function(list){
   if(!list.length){candSel.innerHTML='<option value="">(none in DB — add below)</option>';}
   else{candSel.innerHTML='<option value="">— pick —</option>'+list.map(function(o){return '<option value="'+o.v+'">'+o.v+' ('+o.m+')</option>';}).join('');}
   if(delEl){var del=list.filter(function(o){return o.del;});
    delEl.innerHTML=del.length?('rogue? prune: '+del.map(function(o){return '<a href="#" class="g-delx" data-v="'+o.v+'">'+o.v+' ×</a>';}).join(' &nbsp; ')):'';
    Array.prototype.slice.call(delEl.querySelectorAll('.g-delx')).forEach(function(x){x.onclick=function(e){e.preventDefault();delRow(phr,x.dataset.v);};});}
  }).catch(function(){candSel.innerHTML='<option value="">(lookup failed — add below)</option>';});
 }
 tbody.addEventListener('change',function(e){if(e.target.classList&&e.target.classList.contains('g-chk')){updateBar();if(roleSel.value==='synonym')fetchCands();}});
 roleSel.addEventListener('change',roleFields);
 itype.addEventListener('change',roleFields);
 var msgEl=root.querySelector('#g-msg');
 function note(t){if(msgEl)msgEl.textContent=t||'';}
 function assignNow(){
  var idx=checkedIdx();if(!idx.length){note('tick a word first');return;}
  var r=roleSel.value, a={idx:idx,role:r};
  if(r==='synonym'){var v=((addInp.value||'').trim()||candSel.value||'').toUpperCase();if(!v){note('pick or type a value');return;}a.value=v;}
  if(r==='letters'){var lv=(addInp.value||'').trim().toUpperCase();if(!lv){note('type the exact letters');return;}a.value=lv;}
  if(r==='indicator'){a.itype=itype.value;a.isub=(itype.value==='deletion')?isub.value:'';}
  assignments=assignments.filter(function(x){return !x.idx.some(function(i){return idx.indexOf(i)>=0;});});
  assignments.push(a);addInp.value='';note('');drawRows();drawList();clearChecks();saveAssignments();
 }
 root.querySelector('#g-assign').addEventListener('click',assignNow);
 // picking an existing synonym from the dropdown ADDS it immediately (no separate Assign click)
 candSel.addEventListener('change',function(){if(roleSel.value==='synonym'&&candSel.value)assignNow();});
 root.querySelector('#g-resolve').addEventListener('click',function(){payload.value=JSON.stringify(assignments);root.querySelector('#g-form').submit();});
 drawRows();drawList();updateBar();roleFields();
}
"""


def _infer_synonym_value(candidates, ans_letters, lookup_all):
    """Best DB synonym/abbreviation value for the span (slice-1 inference): prefer a value
    that is a substring of the answer (a charade piece); else, if the span has exactly one
    DB value, use it; else '' (the caller asks the user to type — the easy-add path)."""
    sub, allv = "", []
    for phrase in candidates:
        try:
            for v, m in lookup_all(phrase):
                v = (v or "").upper()
                if m in ("synonym", "abbreviation") and v:
                    if v in ans_letters and len(v) > len(sub):
                        sub = v
                    if v not in allv:
                        allv.append(v)
        except Exception:
            pass
    if sub:
        return sub
    return allv[0] if len(allv) == 1 else ""


def _span_surface(clue_id, back_raw=None):
    """Render the VERTICAL assignment grid for one clue: a row per clue word (preloaded with
    its current role) with a checkbox + role + 'brings' column, plus the role picker, the
    in-memory assignment list, a single Resolve, and the solved breakdown card."""
    import json
    row = _load_clue(clue_id)
    if row is None:
        return '<p class="warn">No clue with id %d.</p>' % clue_id
    clue_text, answer, src, pnum, direction, enumeration, cnum = row
    answer = enum_space(answer, enumeration)
    ctx = build_wfw_atom_context(clue_text, answer, direction=direction)
    conn = store.connect()
    try:
        parse = store.load_parse(conn, clue_id)
        filler = store.get_clue_filler(conn, clue_id)
        saved = store.get_hs_assignments(conn, clue_id)   # restore prior assignments
        note = store.get_note(conn, clue_id)              # restore the user note
    finally:
        conn.close()
    rows = _word_roles(ctx, parse, filler)        # [{idx,text,role,label,value}], clue order
    back = back_raw or str(clue_id)
    try:
        saved_list = json.loads(saved) if saved else []
    except Exception:
        saved_list = []

    trs = "".join(
        '<tr data-i="%d"><td><input type="checkbox" class="g-chk" value="%d"></td>'
        '<td class="g-word">%s</td><td class="r-role"></td><td class="r-brings"></td></tr>'
        % (r["idx"], r["idx"], escape(r["text"])) for r in rows)
    itype_opts = "".join('<option value="%s">%s</option>' % (v, escape(lab))
                         for v, lab in sorted(_FORCE_IND_OPTIONS,
                                              key=lambda o: o[1].lower()))
    isub_opts = "".join('<option value="%s">%s</option>' % (v, escape(lab))
                        for v, lab in _IND_SUBTYPES["deletion"])
    data = {"cid": clue_id,
            "back": back,                         # the CLUTCH id-string, so JS-built forms
                                                  # (e.g. synonym prune) keep it, not collapse
                                                  # to this single clue
            "words": [r["text"] for r in rows],
            "answer": "".join(c for c in answer.upper() if c.isalpha()),
            "current": [{"label": r["label"], "value": r["value"]} for r in rows],
            "assignments": saved_list}

    if parse is not None:
        screen = SCREENS.get(parse.operation) or SCREENS.get(parse.solved_by)
        card = screen(ctx, parse) if screen else wfw_render.render_parse(parse, ctx=ctx)
    else:
        card = '<p>Not solved yet — assign roles and Resolve.</p>'

    rootid = "g-%d" % clue_id
    from urllib.parse import quote
    # PREV/NEXT within the hand-solver: step through the CLUTCH (the `from` context) WITHOUT
    # leaving /hs. Shown only when this clue sits in a clutch of >1 (mirrors the role grid).
    nav_html = ""
    clutch_ids = _parse_hs_ids(back)
    if clue_id in clutch_ids and len(clutch_ids) > 1:
        bq = quote(back, safe="")
        pos = clutch_ids.index(clue_id)

        def _hsarrow(nid, label):
            if nid is None:
                return '<span style="color:#cbd5e1">%s</span>' % label
            return ('<a href="/hs?id=%d&amp;from=%s" style="text-decoration:none;'
                    'font-weight:700;color:#0d9488">%s</a>' % (nid, bq, label))
        prev_id = clutch_ids[pos - 1] if pos > 0 else None
        next_id = clutch_ids[pos + 1] if pos < len(clutch_ids) - 1 else None
        nav_html = ('<div style="display:flex;gap:1.2rem;align-items:center;margin:.3rem 0;'
                    'font-size:.95rem">%s<span style="color:#64748b">%d of %d</span>%s</div>'
                    % (_hsarrow(prev_id, "&larr; prev clue"), pos + 1, len(clutch_ids),
                       _hsarrow(next_id, "next clue &rarr;")))
    cur_status = parse.status if parse is not None else ""
    status_opts = "".join(
        '<option value="%s"%s>%s</option>'
        % (v, " selected" if v == cur_status else "", lab)
        for v, lab in (("pass", "PASS"), ("pending", "PENDING"), ("fail", "FAIL"),
                       ("invalid", "INVALID (missing indicator/operation)")))
    p = [_SPAN_CSS, "<script>%s</script>" % _SPAN_JS,
         '<div id="%s" class="g-root">' % rootid,
         _cid_label(clue_id, src, pnum, cnum, direction),
         '<a href="/?id=%s#clue-%d" style="display:inline-block;margin:.25rem 0;'
         'text-decoration:none;font-weight:700;color:#0d9488">&larr; back to clue page</a>'
         % (quote(back, safe=""), clue_id),
         nav_html,
         '<div style="margin:.3rem 0;font-size:1.1rem;font-weight:600">%s</div>'
         % escape(clue_text),
         '<div class="g-ans">Answer: <b>%s</b></div>' % escape(data["answer"]),
         '<p style="font-size:.85rem;color:#64748b;margin:.2rem 0">Tick the word(s) of a group, '
         'pick a role, choose/add a value, Assign. Repeat for every word, then Resolve once.</p>',
         '<table class="g-tbl"><thead><tr><th></th><th>word</th><th>role</th><th>brings</th>'
         '</tr></thead><tbody id="g-tbody">%s</tbody></table>' % trs,
         '<div id="g-bar" class="g-bar" style="display:none">',
         '<span>Selected: <b id="g-sel"></b></span>',
         '<select id="g-role">'
         '<option value="definition">definition</option>'
         '<option value="synonym">synonym</option>'
         '<option value="letters">letters (exact)</option>'
         '<option value="indicator">indicator</option>'
         '<option value="link">link word</option>'
         '<option value="filler">filler</option>'
         '<option value="none">none (clear)</option></select>',
         '<select id="g-itype" style="display:none">%s</select>' % itype_opts,
         '<select id="g-isub" style="display:none">%s</select>' % isub_opts,
         '<span id="g-cand" style="display:none">value: <select id="g-candsel"></select> '
         'or add <input id="g-add" placeholder="new value" size="12"> '
         '<span id="g-del" style="margin-left:.4rem;font-size:.85rem;color:#b45309"></span></span>',
         '<button type="button" id="g-assign" class="g-assign">Assign</button>',
         '<span id="g-msg" style="color:#dc2626;font-size:.85rem"></span>',
         '</div>',
         '<div id="g-list" class="g-list"></div>',
         '<form method="post" action="/hsresolve" id="g-form">',
         '<input type="hidden" name="only" value="%d">' % clue_id,
         '<input type="hidden" name="from" value="%s">' % escape(back, quote=True),
         '<input type="hidden" name="payload" id="g-payload">',
         '<button type="button" id="g-resolve" class="g-resolve">Resolve &amp; solve</button>',
         '</form>',
         '<form method="post" action="/hsstatus" style="margin:.5rem 0;display:flex;'
         'gap:.4rem;align-items:center;flex-wrap:wrap">',
         '<input type="hidden" name="only" value="%d">' % clue_id,
         '<input type="hidden" name="from" value="%s">' % escape(back, quote=True),
         '<span style="font-size:.85rem;color:#64748b">Mark verdict:</span>',
         '<select name="status">%s</select>' % status_opts,
         '<button type="submit" style="background:#475569;color:#fff;border:none;'
         'border-radius:8px;padding:.3rem .75rem;font-weight:700;cursor:pointer">'
         'Set status</button>',
         '<span style="font-size:.78rem;color:#94a3b8">INVALID = unsolvable as written; '
         'the mark is frozen so it sticks.</span>',
         '</form>',
         '<form method="post" action="/hsnote" style="margin:.5rem 0">',
         '<input type="hidden" name="only" value="%d">' % clue_id,
         '<input type="hidden" name="from" value="%s">' % escape(back, quote=True),
         '<div style="font-size:.85rem;color:#64748b;margin-bottom:.2rem">'
         'Note (shows on the clue page for the user):</div>',
         '<textarea name="note" rows="3" style="width:100%%;max-width:46rem;box-sizing:'
         'border-box;border:1px solid #cbd5e1;border-radius:8px;padding:.4rem;'
         'font-family:inherit;font-size:.95rem">%s</textarea>' % escape(note),
         '<div><button type="submit" style="background:#0d9488;color:#fff;border:none;'
         'border-radius:8px;padding:.3rem .75rem;font-weight:700;cursor:pointer;'
         'margin-top:.3rem">Save note</button></div>',
         '</form>',
         '<div class="g-card">%s</div>' % card,
         '</div>',
         '<script>initGrid("%s", %s);</script>' % (rootid, json.dumps(data))]
    return "".join(p)


def _hs_redirect(only, msg="", back_raw=None):
    """Post/Redirect/Get back to the span surface (clean URL, no resubmit on refresh)."""
    from urllib.parse import quote
    extra = ("&from=%s" % quote(back_raw, safe="")) if back_raw else ""
    return redirect("/hs?id=%s&notice=%s%s" % (only, quote(msg), extra))


@app.route("/hs")
def hs_route():
    """The span-assignment hand-solver for ONE clue."""
    cid = (request.args.get("id") or "").strip()
    if not cid.isdigit():
        return _page('<p class="warn">Enter a clue id, e.g. '
                     '<a href="/hs?id=10075533">/hs?id=10075533</a></p>')
    back = (request.args.get("from") or cid).strip()
    notice = (request.args.get("notice") or "").strip()
    body = ('<div class="wfw-notice">%s</div>' % escape(notice)) if notice else ""
    body += _span_surface(int(cid), back)
    return _page(body)


@app.route("/hslookup")
def hslookup_route():
    """AJAX: the DB synonym/abbreviation candidate values for a ticked word-group, so the grid
    can offer them at Assign time. A LOOKUP, never a solve. Returns JSON [{v,m,del}] where
    `del` marks a value that is a DIRECT synonyms_pairs row (so it can be pruned if rogue);
    values that appear only via the bidirectional lookup are not directly deletable."""
    import json, sqlite3
    phrase = (request.args.get("phrase") or "").strip()
    out = []
    if phrase:
        direct = set()
        try:
            con = sqlite3.connect(admin_db.CRYPTIC_DB)
            for (v,) in con.execute("SELECT synonym FROM synonyms_pairs "
                                    "WHERE lower(word)=lower(?)", (phrase,)):
                direct.add((v or "").upper())
            con.close()
        except Exception:
            pass
        try:
            la = batch_wiring()["lookup_all"]
            seen = set()
            for v, m in la(phrase):
                v = (v or "").upper()
                if m in ("synonym", "abbreviation") and v and v not in seen:
                    seen.add(v)
                    out.append({"v": v, "m": m, "del": v in direct})
                if len(out) >= 30:
                    break
        except Exception:
            out = []
    return app.response_class(json.dumps(out), mimetype="application/json")


@app.route("/hsdelete", methods=["POST"])
def hsdelete_route():
    """Delete a POLLUTING reference-DB row for a ticked span (recoverable in deleted_entries),
    reconcile the wiring, and re-solve — so pruning a rogue entry can unlock the solve on its
    own. Synonyms are the common case; definition/link supported too."""
    only = (request.form.get("only") or "").strip()
    back = (request.form.get("from") or only).strip()
    kind = (request.form.get("kind") or "synonym").strip()
    word = (request.form.get("word") or "").strip()
    value = (request.form.get("value") or "").strip()
    if not only.isdigit() or not word:
        return _hs_redirect(only, "Nothing to delete.", back)
    cid = int(only)
    if kind == "synonym":
        msg = admin_db.delete_synonym(word, value)
        apply_add_to_wiring({"kind": "synonym", "word": word, "synonym": value})
    elif kind == "definition":
        msg = admin_db.delete_definition(word, value)
        apply_add_to_wiring({"kind": "definition", "definition": word, "answer": value})
    elif kind == "link":
        msg = admin_db.delete_link(word)
        apply_add_to_wiring({"kind": "link", "word": word})
    else:
        msg = "Unknown delete kind."
    # The invalidate above clears the cached lookup for the row, so the next live query no
    # longer sees the just-deleted row — no ~9s full reload needed.
    _resolve_one(cid)
    return _hs_redirect(only, msg + " Re-solved.", back)


@app.route("/hsinfer")
def hsinfer_route():
    """AJAX: infer a DELETION sub-type from the answer and an assigned synonym value — the op
    whose result equals the answer (FLORIST + outer-delete -> LORIS gives sub-type 'ends').
    A pure computation, never a solve. Returns {"subtype": "..."} or {}."""
    import json
    from core import deletion
    base = (request.args.get("base") or "").strip().upper()
    answer = (request.args.get("answer") or "").strip().upper()
    out = {}
    if base and answer and len(base) > len(answer):
        for sub, _lab in _IND_SUBTYPES["deletion"]:
            op = deletion.SUBTYPE_OP.get(sub)
            try:
                if op and deletion.apply_op(op, base) == answer:
                    out = {"subtype": sub}
                    break
            except Exception:
                pass
    return app.response_class(json.dumps(out), mimetype="application/json")


# hand role -> the signature fodder slot it implies (for signature creation)
_HSROLE_FODDER = {"synonym": "SYN_F"}
# indicator type -> (operation, indicator slot role)
_HSIND_OP = {"deletion": ("deletion", "DEL_I"), "anagram": ("anagram", "ANA_I"),
             "reversal": ("reversal", "REV_I"), "container": ("container", "CON_I"),
             "insertion": ("container", "CON_I")}


def _cand_from_assignments(assigns, n_total, answer=""):
    """Map the hand-solver's assignments to a catalog signature candidate
    {operation, def_pos, roles, n_words}, or None when the shape isn't one we can file.
    Handles POSITIONAL deletion (one synonym base + indicator) and NAMED deletion: with a
    deletion indicator and a second synonym whose value is removed from the base to make the
    answer (DERAILMENT - DER = AILMENT), that synonym becomes REM_F (the named-removal
    source). The definition must sit at a clue edge; link/filler are residue."""
    def_idx = None
    syns = []          # [(idx_list, value)]
    indicator = None   # (idx_list, slot_role)
    op = None
    for a in assigns:
        try:
            idx = sorted(int(i) for i in a.get("idx", []))
        except Exception:
            return None
        role = (a.get("role") or "").strip()
        if not idx:
            continue
        if role == "definition":
            if def_idx is not None:
                return None                       # one definition only
            def_idx = idx
        elif role in _HSROLE_FODDER:              # synonym -> a fodder value
            syns.append((idx, (a.get("value") or "").strip().upper()))
        elif role == "indicator":
            spec = _HSIND_OP.get((a.get("itype") or "").split(":")[0])
            if spec is None:
                return None                       # an operation we don't file yet
            op = spec[0]
            indicator = (idx, spec[1])
        elif role in ("link", "filler"):
            continue                              # residue, not a slot
        else:
            return None
    if def_idx is None or (not syns and indicator is None):
        return None

    # NAMED deletion: the synonym whose value is removed from another (the base) to spell the
    # answer is the REM_F source; the other is the SYN_F base.
    rem_key = None
    if op == "deletion" and len(syns) >= 2 and answer:
        from core import deletion
        ans = "".join(c for c in answer.upper() if c.isalpha())
        for bidx, bval in syns:
            for ridx, rval in syns:
                if ridx is bidx or not bval or not rval:
                    continue
                try:
                    if rval in deletion.removed_runs(bval, ans):
                        rem_key = tuple(ridx)
                        break
                except Exception:
                    pass
            if rem_key:
                break

    wp = []
    for idx, _val in syns:
        tok = "REM_F" if (rem_key is not None and tuple(idx) == rem_key) else "SYN_F"
        wp.append((idx[0], tok, len(idx)))
    if indicator is not None:
        wp.append((indicator[0][0], indicator[1], len(indicator[0])))
    if not wp:
        return None
    if def_idx[0] == 0:
        def_pos = "start"
    elif def_idx[-1] == n_total - 1:
        def_pos = "end"
    else:
        return None                               # definition must be at a clue edge
    wp.sort()                                     # slots in clue order
    return {"operation": op or "charade", "def_pos": def_pos,
            "roles": [r for _, r, _ in wp], "n_words": [n for _, _, n in wp]}


def _try_create_signature(cid, cand):
    """A Resolve that leaves the clue unsolved may imply a signature the catalog lacks.
    Create it (the catalog auto-backs-up), re-solve, and KEEP it ONLY if the clue now PASSES
    — otherwise roll it back, so an unverified shape never pollutes the catalog. Returns a
    one-line note for the user, or '' when nothing was attempted."""
    from core import catalog_creator as CC
    import sqlite3
    try:
        sig = CC._signature_str(cand)
    except Exception:
        return ""
    con = sqlite3.connect(CC._CLUES_DB)
    try:
        if con.execute("SELECT 1 FROM catalog_templates WHERE signature=?",
                       (sig,)).fetchone():
            return ""                             # already present; the miss is elsewhere
    finally:
        con.close()
    try:
        tid = CC.add_signature(cand, note="hand-solver: %s (clue %s)" % (sig, cid))
    except Exception:
        return ""
    if tid is None:
        return ""
    # Load the new catalog signature with ONE full reload (the catalog isn't a live query),
    # then verify on the cached wiring (not a second make_db_wiring). Keep the signature ONLY
    # if it is the one that now solves the clue (matched==sig), so we never keep a shape that
    # merely coincides with another passing signature. A raw solve works because the
    # hand-solve committed every piece to the reference DB.
    reload_wiring()
    row = _load_clue(cid)
    keep = False
    if row is not None:
        ct, ans, _s, _pn, _d, enum, _cn = row
        st, _name, msig, _p = CC.verify(ct, enum_space(ans, enum), batch_wiring())
        keep = (st == "pass" and msig == sig)
    if keep:
        _resolve_one(cid)
        return "created the missing signature %s — the clue now solves." % sig
    con = sqlite3.connect(CC._CLUES_DB)                # rollback: not solved by the new sig
    try:
        con.execute("DELETE FROM catalog_templates WHERE id=?", (tid,))
        con.execute("DELETE FROM catalog_template_slots WHERE template_id=?", (tid,))
        con.commit()
    finally:
        con.close()
    reload_wiring()
    _resolve_one(cid)
    return ("signature %s did not solve the clue, so it was rolled back "
            "(check the assignments or a missing DB value)." % sig)


@app.route("/hssave", methods=["POST"])
def hssave_route():
    """Save the hand-solver's current assignment list (JSON) for a clue, with NO solve — so
    each Assign persists immediately and a failed Resolve (or leaving the page) never loses
    the work. Restored into the grid on the next /hs load. Returns a tiny ack."""
    only = (request.form.get("only") or "").strip()
    payload = request.form.get("payload") or ""
    if only.isdigit():
        conn = store.connect()
        try:
            store.set_hs_assignments(conn, int(only), payload)
        finally:
            conn.close()
    return app.response_class("ok", mimetype="text/plain")


@app.route("/hsnote", methods=["POST"])
def hsnote_route():
    """Save a free-text note for a clue from the hand-solver — shown on the clue page for the
    user. No solve, no reference-DB write. Redirects back to /hs."""
    only = (request.form.get("only") or "").strip()
    back = (request.form.get("from") or only).strip()
    note = (request.form.get("note") or "").strip()
    if not only.isdigit():
        return _hs_redirect(only, "No clue.", back)
    conn = store.connect()
    try:
        store.set_note(conn, int(only), note)
    finally:
        conn.close()
    return _hs_redirect(only, "Note saved." if note else "Note cleared.", back)


def _note_block(clue_id):
    """The user-facing note for a clue (authored in the hand-solver), or '' if none."""
    conn = store.connect()
    try:
        note = store.get_note(conn, clue_id)
    finally:
        conn.close()
    if not note:
        return ""
    return ('<div style="margin:.5rem 0;padding:.6rem .85rem;background:#fffbeb;'
            'border:1px solid #fcd34d;border-radius:10px;color:#92600a;font-size:.95rem">'
            '<strong>Note:</strong> %s</div>' % escape(note).replace("\n", "<br>"))


@app.route("/hsstatus", methods=["POST"])
def hsstatus_route():
    """Manually set a clue's verdict from the hand-solver (pass/pending/fail/INVALID) and
    FREEZE it so the mark sticks through later batch re-runs. INVALID = the clue cannot be
    solved as written (a missing indicator or operation) — a common, legitimate outcome,
    not a solver failure. Redirects back to /hs."""
    only = (request.form.get("only") or "").strip()
    back = (request.form.get("from") or only).strip()
    status = (request.form.get("status") or "").strip()
    if not only.isdigit() or status not in ("pass", "pending", "fail", "invalid"):
        return _hs_redirect(only, "No clue/status.", back)
    cid = int(only)
    conn = store.connect()
    try:
        if store.load_parse(conn, cid) is None:
            # No stored parse yet — persist a minimal one so the manual verdict has a row.
            row = _load_clue(cid)
            if row is not None:
                from core.wfw_model import Parse
                stub = Parse(clue_text=row[0], answer_text=enum_space(row[1], row[5]),
                             status=status, operation="", solved_by="manual")
                store.save_parse(conn, cid, stub)
        store.set_status(conn, cid, status)
        store.set_frozen(conn, cid)
        conn.commit()
    finally:
        conn.close()
    return _hs_redirect(only, "Status set to %s (frozen so it sticks)." % status.upper(),
                        back)


@app.route("/hsresolve", methods=["POST"])
def hsresolve_route():
    """The SINGLE commit: apply ALL in-memory assignments (payload JSON) to the reference DB
    DIRECTLY (your Resolve click is the approval) + per-clue overrides, then solve ONCE.
    No per-assignment re-solve. Whole-clue definition -> CD and signature creation are next."""
    import json
    only = (request.form.get("only") or "").strip()
    back = (request.form.get("from") or only).strip()
    payload = (request.form.get("payload") or "").strip()
    if not only.isdigit():
        return _hs_redirect(only, "No clue.", back)
    cid = int(only)
    row = _load_clue(cid)
    if row is None:
        return _hs_redirect(only, "No clue.", back)
    clue_text, answer, src, pnum, direction, enumeration, cnum = row
    answer = enum_space(answer, enumeration)
    ctx = build_wfw_atom_context(clue_text, answer, direction=direction)
    allwords = [t.text for t in ctx.clue_tokens if t.kind == "word"]
    ans_letters = "".join(c for c in answer.upper() if c.isalpha())
    try:
        assigns = json.loads(payload) if payload else []
    except Exception:
        assigns = []
    if not assigns:
        return _hs_redirect(only, "No assignments to resolve.", back)

    applied = []
    conn = store.connect()
    try:
        for a in assigns:
            try:
                idx = sorted(int(i) for i in a.get("idx", []) if 0 <= int(i) < len(allwords))
            except Exception:
                idx = []
            if not idx:
                continue
            phrase = " ".join(allwords[i] for i in idx)
            role = (a.get("role") or "").strip()
            if role == "definition":
                store.set_forced_definition(conn, cid, phrase)
                admin_db.add_definition(phrase, ans_letters)
                apply_add_to_wiring({"kind": "definition", "definition": phrase,
                                     "answer": ans_letters})
                applied.append("def=%r" % phrase)
            elif role == "synonym":
                val = (a.get("value") or "").strip().upper()
                if val:
                    admin_db.add_synonym(phrase, val)
                    apply_add_to_wiring({"kind": "synonym", "word": phrase, "synonym": val})
                    applied.append("%r=%s" % (phrase, val))
            elif role == "letters":               # LITERAL piece — per-clue only, NO DB write
                val = (a.get("value") or "").strip().upper()
                if val:
                    applied.append("%r=%s (letters)" % (phrase, val))
            elif role == "indicator":
                itype = (a.get("itype") or "").strip()
                isub = (a.get("isub") or "").strip() or None
                base = itype.split(":")[0]
                if base:
                    admin_db.add_indicator(phrase, base, isub)
                    apply_add_to_wiring({"kind": "indicator", "word": phrase, "type": base})
                    store.add_forced_indicator(conn, cid, phrase, itype)
                    applied.append("%r=%s%s" % (phrase, base, ("/" + isub) if isub else ""))
            elif role == "link":
                admin_db.add_link_word(phrase)
                apply_add_to_wiring({"kind": "link", "word": phrase})
                applied.append("%r=link" % phrase)
            elif role == "filler":
                for i in idx:
                    store.add_clue_filler(conn, cid, allwords[i])
                applied.append("%r=filler" % phrase)
            elif role == "none":
                # Clear any persisted override for these words so the wrong role does NOT
                # return on reload; the word becomes free (a leftover/fodder candidate).
                fd = store.get_forced_definition(conn, cid)
                if fd and fd.strip().lower() == phrase.strip().lower():
                    store.clear_forced_definition(conn, cid)
                store.clear_forced_indicator(conn, cid, phrase)
                for i in idx:
                    store.clear_clue_filler(conn, cid, allwords[i])
                applied.append("%r=cleared" % phrase)
        store.set_hs_assignments(conn, cid, payload)   # persist the assignment set
    finally:
        conn.close()
    # Each add was folded into the cached wiring INCREMENTALLY (apply_add_to_wiring) — fast.
    # No full reload_wiring() here (that ~9s rebuild was the cause of the slow Resolve).
    #
    # ROLE-DRIVEN SOLVE — ONE solver. The assignments are now applied as forced per-clue
    # overrides (definition/indicator/filler) plus DB enrichment (synonym/abbreviation/link).
    # The NORMAL cascade then solves the clue RESPECTING them. The hand-solver only POINTS the
    # engines at the right roles; it never assembles the answer itself (no parallel builder).
    def _applied_msg(tail):
        return ("Applied %d: %s; %s" % (len(applied), "; ".join(applied), tail)
                if applied else (tail[0].upper() + tail[1:] if tail else "Done."))

    _resolve_one(cid)                                  # apply_forced_overrides + the cascade
    conn3 = store.connect()
    try:
        cp = store.load_parse(conn3, cid)
    finally:
        conn3.close()
    if cp is not None and cp.status == "pass":
        return _hs_redirect(only, _applied_msg("solved."), back)
    if cp is not None and cp.operation == "cd":
        return _hs_redirect(only, _applied_msg("the whole clue is taken as a cryptic "
                            "definition — pending your confirmation."), back)

    # The cascade did not solve it. The assignments may imply a catalog signature the catalog
    # lacks — create it, but KEEP it ONLY if the cascade then PASSES (cascade-verified, never a
    # hand-built parse). Otherwise report the cascade's own fail evidence.
    sig_msg = ""
    cand = _cand_from_assignments(assigns, len(allwords), answer)
    if cand is not None:
        sig_msg = _try_create_signature(cid, cand)
    if sig_msg:
        return _hs_redirect(only, _applied_msg(sig_msg), back)
    tail = ("did NOT solve — " + (cp.warnings[0] if (cp is not None and cp.warnings)
            else "the cascade could not assemble the answer from these roles"))
    return _hs_redirect(only, _applied_msg(tail), back)


@app.route("/handsolve")
def handsolve_route():
    """Atom-level hand-solver for one clue or several (id box / A-B range)."""
    raw = (request.args.get("id") or "").strip()
    ids = _parse_hs_ids(raw)
    body = _HS_CSS + HANDSOLVE_FORM.format(cid=escape(raw, quote=True))
    if not ids:
        return _page(body + '<p class="warn">Enter one or more clue ids, '
                            'e.g. 10074740 or 10074740-10074745.</p>')
    body += "<script>" + HANDSOLVE_JS + "</script>"
    for cid in ids:
        body += _handsolve_block(cid)
    return _page(body)


def _reload_clue_button(clue_id, raw_list):
    """Per-clue button: reload the DB snapshot and re-run JUST this clue, keeping the
    rest of the batch on screen."""
    from urllib.parse import quote
    _frm = quote(raw_list or "", safe="")     # carry the CLUTCH into the role grid
    return (
        '<form method="post" action="/reload" style="margin:.4rem 0 0">'
        '<input type="hidden" name="id" value="%s">'
        '<input type="hidden" name="only" value="%d">'
        '<button class="wfw-reload wfw-reload-clue" title="Rebuild the DB snapshot, '
        'then re-run only this clue (DB-only unless AI fallback is ticked)">'
        '&#8635; Reload DB &amp; re-run this clue</button>'
        '<label style="font-size:.8rem;margin-left:.6rem" '
        'title="Run the AI piece fallback too (slower). Off = fast DB-only re-run.">'
        '<input type="checkbox" name="ai" value="on"> with AI fallback</label>'
        '</form>'
        '<a href="/hs?id=%d&amp;from=%s" class="wfw-reload wfw-reload-clue" '
        'style="display:inline-block;text-decoration:none;background:#0d9488;'
        'border-color:#0d9488;margin:.4rem 0" '
        'title="Open the span hand-solver (redesign) for this clue, carrying the clutch">'
        '&#9776; Hand-solver</a>'
        '<a href="/rolegrid?id=%d&amp;from=%s" class="wfw-reload wfw-reload-clue" '
        'style="display:inline-block;text-decoration:none;background:#94a3b8;'
        'border-color:#94a3b8;margin:.4rem 0 .4rem .4rem" '
        'title="Open the OLD role-grid hand-solver for this clue">'
        '&#9776; Old grid</a>'
        '<a href="/handsolve?id=%d" class="wfw-reload wfw-reload-clue" '
        'style="display:inline-block;text-decoration:none;background:#7c3aed;'
        'border-color:#7c3aed;margin:.4rem 0 .4rem .4rem" '
        'title="Open the (legacy) atom-level hand-solver for this clue">'
        '&#9998; Atoms</a>'
        % (escape(raw_list, quote=True), clue_id, clue_id, _frm, clue_id, _frm, clue_id))


def _cid_label(clue_id, source=None, puzzle_number=None, clue_number=None, direction=None):
    bits = []
    if source or puzzle_number:
        bits.append((((str(source).upper() + " ") if source else "")
                     + (str(puzzle_number) if puzzle_number else "")).strip())
    if clue_number:
        dirtxt = (" " + str(direction).upper()) if direction else ""
        bits.append(str(clue_number) + dirtxt)
    pub = "".join(" &middot; " + escape(b) for b in bits if b)
    return ('<div id="clue-%d" style="font-size:.8rem;font-weight:700;color:#64748b;'
            'letter-spacing:.06em;margin:1.2rem 0 -.5rem;scroll-margin-top:.6rem">'
            'CLUE ID %d%s</div>' % (clue_id, clue_id, pub))


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
