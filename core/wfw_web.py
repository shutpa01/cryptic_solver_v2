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

from flask import Flask, request

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
              "acrostic", "homophone", "charade"]
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
    return _page(_body(raw))                       # fresh run: solve all


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
    return _page(notice + _body(raw, resolve_only=set()))


@app.route("/rejectsig", methods=["POST"])
def rejectsig():
    """Reject a queued auto-discovered signature (remembered so it is not re-queued)."""
    from core import signature_queue
    raw = (request.form.get("id") or "").strip()
    sid = (request.form.get("sig_id") or "").strip()
    if sid:
        signature_queue.reject(int(sid))
    notice = '<div class="wfw-notice">Signature rejected.</div>'
    return _page(notice + _body(raw, resolve_only=set()))


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
    return "Unknown add."


def _signature_queue_html():
    """Global banner: auto-DISCOVERED catalog signatures awaiting approval (queue for
    approval). Each row shows the proposed signature, the clue that triggered it, and the
    verified would-be parse, with Approve (-> file into the catalog) / Reject."""
    from core import signature_queue
    try:
        rows = signature_queue.list_pending()
    except Exception:
        return ""
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


def _body(raw, resolve_only=None, ai=False, discover=False):
    """Render the page body. `resolve_only` None -> re-solve every clue; a set ->
    re-solve only those ids, render the rest from their stored result. `ai` True
    uses the full AI wiring for the re-solve (per-clue, on demand); False uses the
    DB-only batch wiring so a whole-puzzle run makes no AI calls. `discover` True turns
    on the auto-signature DISCOVERY+QUEUE for the re-solved clue(s) (the per-clue re-run)."""
    cid = escape(raw, quote=True)
    # Render the clue cards FIRST (a per-clue re-run may discover + queue a signature
    # during the solve), THEN build the queue banner so it reflects anything just queued.
    cards = ""
    tokens = [t for t in raw.replace(",", " ").split() if t]
    for token in tokens:
        resolve = resolve_only is None or token in resolve_only
        cards += _render_one(token, raw, resolve, ai=ai, discover=discover)
    return (FORM.format(cid=cid) + RELOAD_FORM.format(cid=cid)
            + _signature_queue_html() + cards)


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
            + _filler_block(clue_id, raw_list, unaccounted, filler)
            + _enrichment_block(clue_text, answer, clue_id, raw_list)
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
                   for s in ("pass", "pending", "fail"))
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
    h = _hidden(raw_list, clue_id)
    opts = "".join('<option value="%s">%s</option>' % (t, t) for t in _IND_TYPES)
    sub_opts = "".join('<option value="%s">%s</option>' % (v, escape(lab))
                       for v, lab in _IND_SUBTYPES["deletion"])
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
    <select name="type" onchange="var s=this.form.querySelector('select[name=subtype]'); var d=this.value=='deletion'; s.style.display=d?'':'none'; if(!d)s.selectedIndex=0;">{opts}</select>
    <select name="subtype" style="display:none" title="deletion sub-type — what gets removed">{sub_opts}</select>
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

# DB indicator types offered when forcing an indicator (alphabetical).
_FORCE_IND_TYPES = ("acrostic", "alternation", "anagram", "container", "deletion",
                    "hidden", "homophone", "insertion", "reversal", "selection")


def _word_roles(ctx, parse, filler_set):
    """Map each clue WORD to the role the stored parse gives it. Returns a list of
    dicts {idx, text, role, detail} in clue order. Role is derived by atom-id
    membership: definition / piece (mechanism) / indicator / link / filler / '-'."""
    # atom_id -> (role, detail)
    amap = {}
    if parse is not None:
        if parse.definition is not None:
            for aid in (parse.definition.clue_atom_ids or ()):
                amap[aid] = ("definition", "")
        for s in (parse.sources or []):
            mech = getattr(s, "mechanism", "") or ""
            for aid in (s.clue_atom_ids or ()):
                amap[aid] = ("piece", mech)
        for a in (parse.annotations or []):
            for aid in (a.clue_atom_ids or ()):
                amap[aid] = (a.role, getattr(a, "note", "") or "")
    out, wi = [], 0
    fil = {(x or "").strip().lower() for x in (filler_set or ())}
    for t in ctx.clue_tokens:
        if t.kind != "word":
            continue
        role, detail = "—", ""
        for aid in t.atom_ids:
            if aid in amap:
                role, detail = amap[aid]
                break
        if role == "—" and (t.text or "").strip().lower() in fil:
            role, detail = "filler", "surface filler"
        out.append({"idx": wi, "text": t.text, "role": role, "detail": detail})
        wi += 1
    return out


def _rolegrid_block(clue_id):
    """One clue's role grid: the words listed vertically with their current roles, the
    forced-indicator control, current forces, and (if frozen) the unforce control."""
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
        forced_ind = store.get_forced_indicators(conn, clue_id)
        frozen = store.is_frozen(conn, clue_id)
    finally:
        conn.close()
    status = parse.status if parse is not None else "fail"
    rows = _word_roles(ctx, parse, filler)
    h = ('<input type="hidden" name="id" value="%s">'
         '<input type="hidden" name="only" value="%d">'
         % (escape(str(clue_id), quote=True), clue_id))

    # the grid: a checkbox per word (to select a contiguous span) + its current role
    grid = ['<table class="rg-tbl"><tr><th></th><th>word</th><th>current role</th></tr>']
    for r in rows:
        det = (" <span class='rg-det'>%s</span>" % escape(r["detail"])) if r["detail"] else ""
        grid.append(
            '<tr><td><input type="checkbox" name="w" value="%d" form="forceind-%d"></td>'
            '<td class="rg-word">%s</td><td class="rg-role rg-%s">%s%s</td></tr>'
            % (r["idx"], clue_id, escape(r["text"]),
               escape(r["role"].split()[0] if r["role"] else "x"),
               escape(r["role"]), det))
    grid.append("</table>")

    type_opts = "".join('<option value="%s">%s</option>' % (t, t)
                        for t in _FORCE_IND_TYPES)
    force_form = (
        '<form method="post" action="/forceind" id="forceind-%d" class="rg-form">%s'
        '<span class="rg-l">Force selected words as indicator of type</span>'
        '<select name="wptype">%s</select>'
        '<button>Force indicator &amp; re-solve</button></form>'
        % (clue_id, h, type_opts))

    cur = ""
    if forced_ind:
        items = "".join(
            '<li>%s &rarr; <b>%s</b> '
            '<form method="post" action="/clearforceind" class="rg-inline">%s'
            '<input type="hidden" name="phrase" value="%s">'
            '<button class="rg-x">clear</button></form></li>'
            % (escape(p), escape(t), h, escape(p, quote=True))
            for p, t in forced_ind)
        cur = '<div class="rg-cur"><b>Forced indicators:</b><ul>%s</ul></div>' % items

    unforce = ""
    if frozen:
        unforce = ('<form method="post" action="/unforce" class="rg-form">%s'
                   '<span class="rg-l">&#128274; FROZEN — forced pass, will not revert</span>'
                   '<button>Unforce &amp; re-solve</button></form>' % h)

    return (
        _cid_label(clue_id, src, pnum, cnum, direction)
        + '<div class="rg-block">'
        + '<div class="wfw-clue rg-clue">%s</div>' % escape(clue_text)
        + '<div class="rg-status rg-%s">%s &mdash; %s</div>'
          % (status, escape(answer), status.upper())
        + "".join(grid)
        + force_form + cur + unforce
        + '</div>')


_RG_CSS = """<style>
.rg-block{border:1px solid #cbd5e1;border-radius:8px;padding:1rem;margin:.6rem 0 2rem;
  font-family:system-ui}
.rg-clue{font-size:1.15rem;margin:.2rem 0 .6rem}
.rg-status{font-weight:600;margin:.2rem 0 .8rem;letter-spacing:.05em}
.rg-status.rg-pass{color:#16a34a}.rg-status.rg-fail{color:#dc2626}
.rg-status.rg-pending{color:#d97706}
.rg-tbl{border-collapse:collapse;margin:.4rem 0}
.rg-tbl th{text-align:left;font-size:.75rem;color:#64748b;font-weight:600;padding:.2rem .6rem}
.rg-tbl td{padding:.2rem .6rem;border-top:1px solid #f1f5f9}
.rg-word{font-weight:600}
.rg-role{font-size:.85rem;color:#475569}
.rg-role.rg-definition{color:#0369a1}.rg-role.rg-indicator{color:#7c3aed}
.rg-role.rg-piece{color:#16a34a}.rg-role.rg-link{color:#94a3b8}
.rg-role.rg-filler{color:#94a3b8}
.rg-det{color:#94a3b8;font-size:.8rem}
.rg-form{margin:.6rem 0;display:flex;gap:.5rem;align-items:center;flex-wrap:wrap}
.rg-l{font-size:.85rem;color:#475569}
.rg-cur{margin:.6rem 0;font-size:.9rem}.rg-cur ul{margin:.3rem 0;padding-left:1.2rem}
.rg-inline{display:inline}
.rg-x{font-size:.7rem;padding:.05rem .4rem}
</style>"""

ROLEGRID_FORM = ('<form method="get" action="/rolegrid" style="margin:1rem 0;'
                 'font-family:system-ui">'
                 '<input name="id" value="{cid}" placeholder="clue id" size="12">'
                 '<button>Open role grid</button></form>')


@app.route("/rolegrid")
def rolegrid_route():
    """Role-grid hand-solver for one clue (or several, comma/space/range separated)."""
    raw = (request.args.get("id") or "").strip()
    ids = _parse_hs_ids(raw)
    body = _RG_CSS + ROLEGRID_FORM.format(cid=escape(raw, quote=True))
    if not ids:
        return _page(body + '<p class="warn">Enter a clue id, e.g. 10075290.</p>')
    for cid in ids:
        body += _rolegrid_block(cid)
    return _page(body)


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
    notice = '<div class="wfw-notice">%s</div>' % escape(msg)
    # re-render the role grid for this clue (re-solve happens on the main solve path; here
    # we just re-render the grid, which reads the freshly-applied override on next /reload).
    body = _RG_CSS + ROLEGRID_FORM.format(cid=escape(only, quote=True))
    # trigger a re-solve so the new force takes effect immediately, then show the grid
    _resolve_one(int(only)) if only else None
    body += _rolegrid_block(int(only)) if only else ""
    return _page(notice + body)


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
    notice = '<div class="wfw-notice">Forced indicator cleared; re-solved.</div>'
    body = _RG_CSS + ROLEGRID_FORM.format(cid=escape(only, quote=True))
    body += _rolegrid_block(int(only)) if only else ""
    return _page(notice + body)


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
        '<a href="/rolegrid?id=%d" class="wfw-reload wfw-reload-clue" '
        'style="display:inline-block;text-decoration:none;background:#0d9488;'
        'border-color:#0d9488;margin:.4rem 0" '
        'title="Open the role-grid hand-solver for this clue">'
        '&#9776; Hand-solver</a>'
        '<a href="/handsolve?id=%d" class="wfw-reload wfw-reload-clue" '
        'style="display:inline-block;text-decoration:none;background:#7c3aed;'
        'border-color:#7c3aed;margin:.4rem 0 .4rem .4rem" '
        'title="Open the (legacy) atom-level hand-solver for this clue">'
        '&#9998; Atoms</a>'
        % (escape(raw_list, quote=True), clue_id, clue_id, clue_id))


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
