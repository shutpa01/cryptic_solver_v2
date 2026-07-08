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
_IND_TYPES = ["hidden", "anagram", "container", "insertion", "reversal", "deletion",
              "selection", "acrostic", "homophone", "charade", "alternation", "letter_shift",
              "charade_positional", "palindrome", "spoonerism", "definition by example"]
# Sub-types offered per indicator type for the Add-indicator dropdown. The VALUES are the
# sub-codes the SOLVING ENGINES actually recognise, sourced from the engine maps
# (selection_indicators.CLUE_PAGE_SUBTYPES / SUBTYPE_RULE, deletion.SUBTYPE_OP) so this
# list can never drift out of sync with what the engines accept — a hardcoded copy could,
# and did (it was missing `alternation` entirely). Labels are UI copy (legitimately here).
# Shape is unchanged: {type: [(stored-value, intuitive-label), ...]}.
#   - selection: the five canonical selection rules, straight from the engine constant.
#   - deletion: canonical op codes, each asserted to be a real deletion.SUBTYPE_OP key so a
#     divergence is caught at import; plus "" / "general" (a plain removal, letters named by
#     another word — SUBTYPE_OP.get returns None for these, handled answer-driven).
#   - letter_shift / charade_positional: the directions add_indicator validates.
#   - alternation: its engines key on the wordplay_type ALONE (no sub-type), so it offers a
#     single "no sub-type needed" option (value "") so it can be added from the clue page.
_SUBTYPE_LABELS = {
    ("selection", "first"):     "first letter(s) (initially, primarily)",
    ("selection", "last"):      "last letter(s) (finally, ultimately)",
    ("selection", "outer"):     "outer letters (extremes, ends)",
    ("selection", "middle"):    "middle letter(s) (centrally, heart of)",
    ("selection", "alternate"): "alternate letters (oddly, evenly)",
    ("deletion", "head"):    "remove first letter (behead)",
    ("deletion", "tail"):    "remove last letter (curtail)",
    ("deletion", "ends"):    "remove outer letters",
    ("deletion", "middle"):  "remove middle letter",
    ("deletion", "empty"):   "hollow — remove inner letters",
    ("deletion", "general"): "letters named by another word",
    ("letter_shift", "last_front"): "move last letter to front",
    ("letter_shift", "first_end"):  "move first letter to end",
    ("charade_positional", "after"):  "piece goes AFTER (behind) its neighbour",
    ("charade_positional", "before"): "piece goes BEFORE (ahead of) its neighbour",
    ("alternation", ""): "— no sub-type needed —",
    ("palindrome", ""): "single word or phrase (e.g. reversible, either way)",
    ("palindrome", "opp_pair"): "opposite-direction pair (e.g. east west)",
    ("spoonerism", ""): "whole phrase including Spooner (e.g. 'old Spooner', 'according to Spooner')",
}


def _build_ind_subtypes():
    """Build the per-type sub-type dropdown options from the engines' recognised sub-codes."""
    from core import selection_indicators, deletion
    out = {}
    # selection: exactly the engine's canonical selection rules (source of truth).
    out["selection"] = [(c, _SUBTYPE_LABELS[("selection", c)])
                        for c in selection_indicators.CLUE_PAGE_SUBTYPES]
    # deletion: canonical op codes + the "" / general plain-removal choices. Assert each real
    # op code is one the engine recognises, so this can never offer a dead sub-type.
    _del_codes = ["", "head", "tail", "ends", "middle", "empty", "general"]
    d = []
    for c in _del_codes:
        if c and c != "general":
            assert c in deletion.SUBTYPE_OP, "deletion subtype %r not recognised by engine" % c
        d.append((c, "— no sub-type —" if c == "" else _SUBTYPE_LABELS[("deletion", c)]))
    out["deletion"] = d
    out["letter_shift"] = [(c, _SUBTYPE_LABELS[("letter_shift", c)])
                           for c in ("last_front", "first_end")]
    out["charade_positional"] = [(c, _SUBTYPE_LABELS[("charade_positional", c)])
                                 for c in ("after", "before")]
    out["alternation"] = [("", _SUBTYPE_LABELS[("alternation", "")])]
    # palindrome: a single word / phrase (loader infers single vs phrase from word count),
    # or an opposite-direction pair (both words must appear). spoonerism: a lead word.
    out["palindrome"] = [("", _SUBTYPE_LABELS[("palindrome", "")]),
                         ("opp_pair", _SUBTYPE_LABELS[("palindrome", "opp_pair")])]
    # spoonerism: a WHOLE phrase including the Spooner name (never bare lead words).
    out["spoonerism"] = [("", _SUBTYPE_LABELS[("spoonerism", "")])]
    return out


_IND_SUBTYPES = _build_ind_subtypes()

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


def _capture_signature_review(clue_id, status):
    """If this clue was solved by a PENDING-only signature, log the human verdict as a
    signature review (signature-tiers build §5 Step 8). A pass override = 'confirm' (the
    pending-only signature's reconstruction was faithful); a fail/invalid override =
    'reject'. A 'pending' override records nothing (the human left it unreviewed). Only
    pending-tier templates are logged — a pass-tier solve needs no review track record.
    Never lets a logging error break the verdict override."""
    verdict = {"pass": "confirm", "fail": "reject", "invalid": "reject"}.get(status)
    if verdict is None:
        return
    try:
        conn = store.connect()
        try:
            row = conn.execute("SELECT template_id FROM wfw_solve WHERE clue_id = ?",
                               (clue_id,)).fetchone()
        finally:
            conn.close()
        tid = row[0] if row else None
        if tid is None:
            return
        from core.catalog_loader import load_template_tiers
        if load_template_tiers().get(tid) != "pending":
            return
        from core import signature_reviews
        signature_reviews.record(tid, clue_id, verdict)
    except Exception:
        pass


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
        _capture_signature_review(int(only), status)   # log pending-only sig reviews
        msg = "Status of clue %s set to %s." % (only, status)
    notice = '<div class="wfw-notice">%s</div>' % escape(msg)
    return _page(notice + _body(raw, resolve_only=set()), scroll_to=only)


@app.route("/cluecomment", methods=["POST"])
def cluecomment():
    """Save (or clear) the user's free-text comment for one clue, from the inline box below
    the summary. Shares the wfw_notes store with the hand-solver note. Re-renders from store
    (no re-solve, no reference-DB write), so the comment sticks until changed."""
    raw = (request.form.get("id") or "").strip()
    only = (request.form.get("only") or "").strip()
    note = "" if request.form.get("clear") else (request.form.get("note") or "").strip()
    msg = "No clue."
    if only.isdigit():
        conn = store.connect()
        try:
            store.set_note(conn, int(only), note)
        finally:
            conn.close()
        msg = "Comment saved." if note else "Comment cleared."
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
    if kind == "substitution":
        return admin_db.add_substitution(form.get("word"), form.get("value"))
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
            + _note_block(clue_id, raw_list)
            + _filler_block(clue_id, raw_list, unaccounted, filler)
            + _enrichment_block(clue_text, answer, clue_id, raw_list)
            + _clue_signature_suggestions(clue_id, raw_list)
            + _clue_controls(clue_id, raw_list, status)
            + _clue_admin_panel(clue_id, raw_list)
            + _reload_clue_button(clue_id, raw_list))


def _clue_controls(clue_id, raw_list, status):
    """Per-clue manual controls: override the STATUS (pass/fail/pending/invalid) and PIN the
    definition + re-solve (the real override when the definition stage grabbed too many words,
    and the way to solve a CD — pin the definition edge words). Re-renders from store, not
    overwritten unless the clue is explicitly re-run. (The display-only Definition and Unforce
    controls stay removed — the hand-solver /hs owns those now.)"""
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
    return (
        '<div class="wfw-ctl">'
        f'<form method="post" action="/setstatus" class="wfw-cform">{h}'
        f'<span class="wfw-ctl-l">Status</span><select name="status">{opts}</select>'
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
    <input type="hidden" name="kind" value="substitution">{h}
    <span class="wfw-af-l">Abbreviation</span>
    <input name="word" placeholder="word in clue, e.g. point">
    <input name="value" placeholder="letters, e.g. E (&rarr; wordplay table)">
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
 var addpc=root.querySelector('.hs-addpc');
 if(addpc){addpc.onclick=function(){
  var t=(root.querySelector('.hs-pctext').value||'').trim();
  var v=(root.querySelector('.hs-pcval').value||'').trim().toUpperCase();
  if(!v){alert('Type the value (the letters this piece contributes to the answer).');return;}
  pieces.push({text:t,value:v,mech:'derivative',atoms:new Set(),pos:new Set(),def:false});
  root.querySelector('.hs-pctext').value='';root.querySelector('.hs-pcval').value='';
  active=pieces.length-1;selClue.clear();selAns.clear();draw();};}
 var commit=root.querySelector('.hs-commit');
 if(commit){commit.onclick=function(){
  var out=pieces.map(function(pc){return {text:pc.text,value:pieceVal(pc),mech:pc.mech,atoms:Array.from(pc.atoms),pos:Array.from(pc.pos),def:!!pc.def};});
  var fd=new FormData();fd.append('only',clueId);fd.append('payload',JSON.stringify(out));
  result.innerHTML='<i>committing…</i>';
  fetch('/handsolvecommit',{method:'POST',body:fd}).then(function(r){return r.json();}).then(function(o){
   result.innerHTML='<b style="color:'+(o.ok?'#16a34a':'#dc2626')+'">'+esc(o.msg)+'</b>'+(o.ok?(' <a href="/?id='+clueId+'">view on the clue page</a>'):'');
  }).catch(function(e){result.innerHTML='<b style="color:#dc2626">commit failed</b>';});};}
 var uncommit=root.querySelector('.hs-uncommit');
 if(uncommit){uncommit.onclick=function(){
  var fd=new FormData();fd.append('only',clueId);
  fetch('/handsolveuncommit',{method:'POST',body:fd}).then(function(r){return r.json();}).then(function(o){
   result.innerHTML='<b style="color:'+(o.ok?'#16a34a':'#dc2626')+'">'+esc(o.msg)+'</b>';
  }).catch(function(e){result.innerHTML='<b style="color:#dc2626">uncommit failed</b>';});};}
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
            + '<div class="hs-row" style="font-size:.9rem">Add a derivative piece: '
              '<input class="hs-pctext" placeholder="word(s), e.g. rock band" size="18"> '
              '&rarr; <input class="hs-pcval" placeholder="value, e.g. REM" size="10"> '
              '<button class="hs-addpc">Add piece</button> '
              '<span style="color:#64748b;font-size:.82rem">then click its chip and the clue '
              'atoms + answer tiles it makes.</span></div>'
            + '<div class="hs-row"><button class="hs-commit wfw-reload" '
              'style="background:#7c3aed;border-color:#7c3aed">Commit (manual)</button>'
              '<button class="hs-uncommit" style="margin-left:.5rem;background:#fff;'
              'color:#7c3aed;border:1px solid #7c3aed;border-radius:8px;padding:.35rem .8rem;'
              'font-weight:700;cursor:pointer">Uncommit</button>'
              '<span style="margin-left:.6rem;color:#64748b;font-size:.82rem">records exactly '
              'what you tagged, frozen — no DB write, no solver.</span></div>'
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
    ("letter_shift", "letter shift"),
    ("charade_positional:after", "positional — after"),
    ("charade_positional:before", "positional — before"),
    ("palindrome", "palindrome"), ("spoonerism", "spoonerism"),
)
_FORCE_IND_TYPES = frozenset(v for v, _ in _FORCE_IND_OPTIONS)


# mechanism -> friendly role label shown in the grid
_MECH_LABEL = {"synonym": "synonym", "abbreviation": "abbreviation",
               "first_letter": "first letter", "last_letter": "last letter",
               "selection": "selection", "anagram_fodder": "anagram fodder",
               "raw": "literal", "homophone": "homophone"}


import collections as _collections
_HSUnit = _collections.namedtuple("_HSUnit", ["text", "atom_ids"])


def _hs_word_units(ctx):
    """Clue WORDS for the hand solver, with HYPHENATED words SPLIT into their parts:
    'line-up' -> 'line' + 'up', so each part can take its own role (line=synonym, up=indicator).
    A word token's atom_ids align 1:1 with its text characters, so each part keeps its own atoms
    (the hyphen atom itself is dropped — it carries no letter)."""
    units = []
    for t in ctx.clue_tokens:
        if t.kind != "word":
            continue
        text, aids = t.text, list(t.atom_ids)
        if "-" in text and len(aids) == len(text):
            start = 0
            for i in range(len(text) + 1):
                if i == len(text) or text[i] == "-":
                    if i > start:
                        units.append(_HSUnit(text[start:i], tuple(aids[start:i])))
                    start = i + 1
        else:
            units.append(_HSUnit(text, tuple(aids)))
    return units


def _word_roles(ctx, parse, filler_set, split_hyphens=False):
    """Map each clue WORD to the role the stored parse gives it. Returns a list of dicts
    {idx, text, role, label, value} in clue order, where `role` is the colour category
    (definition/piece/indicator/link/filler/none), `label` is what to show, and `value`
    is the letters the piece produced (e.g. D, TORS) — so the synonym IS shown.
    split_hyphens: hand-solver mode — 'line-up' becomes two rows so each part gets its own role."""
    # atom_id -> (role, label, value)
    amap = {}
    if parse is not None:
        if parse.definition is not None:
            # A guessed (source='pending') definition is shown as "unidentified definition"
            # here too, so the grid never presents the floor's edge guess as confirmed
            # (memory: definition-floor-redesign). Role category stays 'definition'.
            _dlabel = ("unidentified definition"
                       if getattr(parse.definition, "source", "db") == "pending"
                       else ("definition by example"
                             if getattr(parse.definition, "mechanism", "") == "definition_by_example"
                             else "definition"))
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
    units = (_hs_word_units(ctx) if split_hyphens else
             [_HSUnit(t.text, tuple(t.atom_ids)) for t in ctx.clue_tokens if t.kind == "word"])
    for u in units:
        role, label, value = "none", "—", ""
        for aid in u.atom_ids:
            if aid in amap:
                role, label, value = amap[aid]
                break
        if role == "none" and (u.text or "").strip().lower() in fil:
            role, label = "filler", "filler"
        out.append({"idx": wi, "text": u.text, "role": role,
                    "label": label, "value": value})
        wi += 1
    return out


# mechanism (stored on a parse Source) -> hand-solver PIECE role. Mirror of the commit map
# (letters->raw, substitution->abbreviation, anagram->anagram_fodder). An unknown mechanism
# (alternate, first_letter, homophone, ...) falls back to 'synonym' — an editable starting point.
_REV_MECH = {"raw": "letters", "abbreviation": "substitution",
             "anagram_fodder": "anagram", "synonym": "synonym",
             "selection": "selection", "replacement_letter": "replacement"}


def _itype_from_note(note):
    """Best-effort recover the hand-solver indicator (type, subtype) from a stored annotation
    note (manual "reversal/general indicator" or engine "container indicator" / "deletion: cut
    ..." forms). Longest type name first so 'charade_positional' beats 'charade'. Falls back to
    a bare type (or none); the user adjusts."""
    n = (note or "").strip().lower()
    itype = ""
    for t in sorted(_IND_TYPES, key=len, reverse=True):
        if t in n:
            itype = t
            break
    isub = ""
    for sv, _lbl in _IND_SUBTYPES.get(itype, []):
        if sv and sv in n:
            isub = sv
            break
    return itype, isub


def _assignments_from_diagnosis(clue_id, pnum, rows):
    """Seed the hand-solver grid from a triage diagnosis's structured reading (the `hs_seed`
    list in documents/triage/diagnoses_<pnum>.json) — the user REVIEWS and adjusts Claude's
    proposed pieces instead of re-deriving them. Purely a UI prefill: nothing is applied
    until the user's own Resolve / Commit. Seed words are matched to grid rows left-to-right
    (gaps allowed, e.g. anagram fodder around a link word); any entry that doesn't match
    cleanly aborts the whole seed (falls back to the parse seed)."""
    from core import triage
    try:
        d = triage.load_diagnoses(pnum).get(str(clue_id)) or {}
    except Exception:
        return []
    seed = d.get("hs_seed") or []
    if not seed:
        return []

    def norm(t):
        return "".join(c for c in (t or "").lower() if c.isalnum())
    row_norms = [norm(r["text"]) for r in rows]
    used, out = set(), []
    for e in seed:
        words = [norm(w) for w in str(e.get("words") or "").split() if norm(w)]
        if not words:
            return []
        idx, start = [], 0
        for w in words:
            found = next((i for i in range(start, len(row_norms))
                          if i not in used and row_norms[i] == w), None)
            if found is None:
                return []                     # stale seed (text/units changed) — abort whole seed
            idx.append(found)
            used.add(found)
            start = found + 1
        a = {"idx": idx, "role": (e.get("role") or "").strip()}
        for k in ("value", "rule", "itype", "isub", "dkind", "cut"):
            if e.get(k):
                a[k] = e[k]
        if e.get("tiles"):
            try:
                a["pos"] = sorted(int(p) for p in e["tiles"])
            except Exception:
                return []
        out.append(a)
    return out


def _assignments_from_parse(ctx, parse):
    """Seed the hand-solver grid with the roles the STORED parse already gives, as EDITABLE
    assignments — so re-tagging ONE word doesn't mean reassigning every word. Pieces carry their
    answer TILES (from the parse links); definition / indicators / links map to their roles.
    Word indices are `_hs_word_units` positions (the same grid the commit reads)."""
    if parse is None:
        return []
    units = _hs_word_units(ctx)
    aid2idx = {}
    for i, u in enumerate(units):
        for aid in (u.atom_ids or ()):
            aid2idx[aid] = i

    def idxs(atom_ids):
        return sorted({aid2idx[a] for a in (atom_ids or ()) if a in aid2idx})

    out = []
    for si, s in enumerate(parse.sources or []):
        idx = idxs(getattr(s, "clue_atom_ids", ()))
        if not idx:
            continue
        pos = sorted(l.answer_pos for l in (parse.links or [])
                     if getattr(l, "source_index", None) == si)
        role = _REV_MECH.get(getattr(s, "mechanism", "") or "", "synonym")
        out.append({"idx": idx, "role": role, "pos": pos,
                    "value": (getattr(s, "value", "") or "")})
    if parse.definition is not None:
        d_idx = idxs(getattr(parse.definition, "clue_atom_ids", ()))
        if d_idx:
            _dk = ("dbe" if getattr(parse.definition, "mechanism", "") == "definition_by_example"
                   else "def")
            out.append({"idx": d_idx, "role": "definition", "dkind": _dk})
    for an in (parse.annotations or []):
        idx = idxs(getattr(an, "clue_atom_ids", ()))
        if not idx:
            continue
        r = getattr(an, "role", "")
        if r == "indicator":
            it, isb = _itype_from_note(getattr(an, "note", ""))
            out.append({"idx": idx, "role": "indicator", "itype": it, "isub": isb})
        elif r == "link":
            out.append({"idx": idx, "role": "link"})
        elif r == "deletion":
            out.append({"idx": idx, "role": "deletion", "value": ""})
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
        # MANUAL-SOLVE GUARD: a committed manual solution (human-authored, frozen) must never
        # be overwritten by the cascade. Skip re-solving it entirely; /handsolveuncommit lifts
        # the freeze first, so an uncommit re-solve is not blocked here.
        conn = store.connect()
        try:
            sp = store.load_parse(conn, clue_id)
            if (sp is not None and getattr(sp, "solved_by", "") == "manual"
                    and store.is_frozen(conn, clue_id)):
                return
        finally:
            conn.close()
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
 var dkind=root.querySelector('#g-dkind'), selrule=root.querySelector('#g-selrule');
 var candWrap=root.querySelector('#g-cand'), candSel=root.querySelector('#g-candsel'), addInp=root.querySelector('#g-add'), delEl=root.querySelector('#g-del');
 var cutWrap=root.querySelector('#g-cutwrap'), cutEl=root.querySelector('#g-cut'), cutPrev=root.querySelector('#g-cutprev');
 var listDiv=root.querySelector('#g-list'), payload=root.querySelector('#g-payload');
 var ROLECOL={definition:'#0f766e',synonym:'#1d4ed8',substitution:'#0e7490',letters:'#0891b2',selection:'#b45309',anagram:'#0369a1',deletion:'#b45309',indicator:'#7c3aed',link:'#64748b',filler:'#9333ea',none:'#94a3b8'};
 function isValued(r){return r==='synonym'||r==='substitution';}          // types/picks a value
 function isPiece(r){return r==='synonym'||r==='substitution'||r==='letters'||r==='replacement'||r==='selection'||r==='anagram';} // lands on tiles
 // The engine's selection rules (core.selection.SPAN_RULES) mirrored on plain letters, so the
 // value is DERIVED from the ticked word(s) — never free-typed — and cannot fabricate.
 function selCands(letters,rule){var la=(letters||'').split(''),n=la.length;
  switch(rule){
   case 'first':return n?[la[0]]:[];
   case 'last':return n?[la[n-1]]:[];
   case 'outer':return n>=2?[la[0]+la[n-1]]:[];
   case 'middle':return n<3?[]:(n%2?[la[(n-1)/2]]:[la[n/2-1]+la[n/2]]);
   case 'alternate':if(n<2)return[];var a=[],b=[];for(var i=0;i<n;i++){(i%2?b:a).push(la[i]);}return[a.join(''),b.join('')];
   case 'remove_first':return n>=2?[la.slice(1).join('')]:[];
   case 'remove_last':return n>=2?[la.slice(0,n-1).join('')]:[];
   case 'remove_outer':return n>=3?[la.slice(1,n-1).join('')]:[];
   case 'remove_middle':return n<3?[]:(n%2?[la.slice(0,(n-1)/2).join('')+la.slice((n+1)/2).join('')]:[la.slice(0,n/2-1).join('')+la.slice(n/2+1).join('')]);
  }return [];}
 function fillSelCands(){if(roleSel.value!=='selection')return;var fl=fodderLetters(checkedIdx());
  var cands=fl?selCands(fl,selrule.value):[];
  candSel.innerHTML=cands.length?cands.map(function(c){return '<option value="'+c+'">'+c+'</option>';}).join('')
   :'<option value="">(tick word(s) first / word too short)</option>';
  if(cands.length)addInp.value=cands[0];}
 function fodderLetters(idx){return idx.map(function(i){return (DATA.words[i]||'').toUpperCase().replace(/[^A-Z]/g,'');}).join('');}
 function msort(s){return (s||'').split('').sort().join('');}
 function msub(a,b){var arr=(a||'').split(''),ok=true;(b||'').split('').forEach(function(c){var i=arr.indexOf(c);if(i>=0)arr.splice(i,1);else ok=false;});return {ok:ok,rem:arr.sort().join('')};}
 var PAL=['#fca5a5','#fcd34d','#86efac','#93c5fd','#c4b5fd','#f9a8d4','#a5f3fc','#fdba74','#d9f99d','#f5d0fe','#fda4af','#bef264'];
 var atiles=Array.prototype.slice.call(root.querySelectorAll('.g-atile'));
 var cmsg=root.querySelector('#g-cmsg');
 var selPos=[];
 function pcCol(k){return PAL[k%PAL.length];}
 function posOwner(p){for(var k=0;k<assignments.length;k++){var a=assignments[k];if(a.pos&&a.pos.indexOf(p)>=0)return k;}return -1;}
 function locateValue(v){v=(v||'').toUpperCase();if(!v)return null;var ans=DATA.answer,hits=[];for(var s=0;s+v.length<=ans.length;s++){if(ans.substr(s,v.length)===v){var ps=[],ok=true;for(var j=0;j<v.length;j++){var p=s+j+1;if(posOwner(p)>=0){ok=false;break;}ps.push(p);}if(ok)hits.push(ps);}}return hits.length===1?hits[0]:null;}
 // Remove the FIRST contiguous run `cut` from a derivative `v` -> the SURVIVING letters that land on
 // the answer (e.g. ORATION - O = RATION). Returns null when `cut` is not a contiguous run of `v`.
 function applyCut(v,cut){v=(v||'').toUpperCase();cut=(cut||'').toUpperCase();if(!cut)return v;var i=v.indexOf(cut);return i<0?null:(v.slice(0,i)+v.slice(i+cut.length));}
 function drawCutPrev(){if(!cutPrev)return;var r=roleSel.value;
  if(r==='anagram'){                                            // ANAGRAM fodder: show the fodder
   var fl=fodderLetters(checkedIdx());                          //   letters + the deletion preview
   if(!fl){cutPrev.innerHTML='';return;}
   var acut=(cutEl&&cutEl.value||'').trim().toUpperCase().replace(/[^A-Z]/g,'');
   if(!acut){cutPrev.innerHTML='<span style="color:#64748b">fodder: <b>'+fl+'</b> ('+fl.length+') &mdash; click the tiles it rearranges into</span>';return;}
   var res=msub(fl,acut);
   cutPrev.innerHTML=res.ok?('<span style="color:#b45309">'+fl+' &minus;'+acut+' &rarr; <b>'+(res.rem||'(empty)')+'</b> ('+res.rem.length+' letters to place)</span>')
    :('<span style="color:#dc2626">'+acut+' has a letter not in '+fl+'</span>');
   return;}
  if(!isValued(r)){cutPrev.innerHTML='';return;}
  var v=((addInp.value||'').trim()||candSel.value||'').toUpperCase();var cut=(cutEl&&cutEl.value||'').trim().toUpperCase();
  if(!v||!cut){cutPrev.innerHTML='';return;}
  var surv=applyCut(v,cut);
  cutPrev.innerHTML=(surv===null)?('<span style="color:#dc2626">'+cut+' not a run of '+v+'</span>')
   :('<span style="color:#b45309">'+v+' &minus;'+cut+' &rarr; <b>'+(surv||'(empty)')+'</b></span>');}
 function drawTiles(){atiles.forEach(function(t){var p=+t.dataset.pos;var o=posOwner(p);
  if(o>=0){t.style.background=pcCol(o);t.style.borderColor=pcCol(o);t.style.color='#0f172a';}
  else if(selPos.indexOf(p)>=0){t.style.background='#1d4ed8';t.style.borderColor='#1d4ed8';t.style.color='#fff';}
  else{t.style.background='#fff';t.style.borderColor='#cbd5e1';t.style.color='#0f172a';}});}
 atiles.forEach(function(t){t.addEventListener('click',function(){var p=+t.dataset.pos;var o=posOwner(p);if(o>=0){releasePiece(o);return;}var i=selPos.indexOf(p);if(i>=0)selPos.splice(i,1);else selPos.push(p);drawTiles();});});
 function saveAssignments(){try{var fd=new FormData();fd.append('only',DATA.cid);fd.append('payload',JSON.stringify(assignments));fetch('/hssave',{method:'POST',body:fd});}catch(e){}}
 function checkedIdx(){return Array.prototype.slice.call(tbody.querySelectorAll('input.g-chk:checked')).map(function(c){return +c.value;}).sort(function(a,b){return a-b;});}
 function phraseOf(idx){return idx.map(function(i){return DATA.words[i];}).join(' ');}
 function assignOf(i){for(var k=0;k<assignments.length;k++){if(assignments[k].idx.indexOf(i)>=0)return assignments[k];}return null;}
 function drawRows(){
  Array.prototype.slice.call(tbody.querySelectorAll('tr')).forEach(function(tr){
   var i=+tr.dataset.i, a=assignOf(i);
   var rc=tr.querySelector('.r-role'), bc=tr.querySelector('.r-brings');
   if(a){var k=assignments.indexOf(a);var col=isPiece(a.role)?pcCol(k):(ROLECOL[a.role]||'#334155');
    rc.innerHTML='<b style="color:'+col+'">'+(a.role==='definition'&&a.dkind==='dbe'?'definition by example':a.role)+(a.isub?('/'+a.isub):'')+(a.rule?('/'+a.rule):'')+'</b>';
    bc.innerHTML=isPiece(a.role)?((a.value||'')+(a.cut?(' <span style="color:#b45309">&minus;'+a.cut+'</span>'):'')+(a.pos&&a.pos.length?(' <span style="color:#64748b">@'+a.pos.slice().sort(function(x,y){return x-y;}).join(',')+'</span>'):'')):(a.role==='deletion'?('<span style="color:#b45309">&minus;'+(a.value||'')+'</span>'):'');
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
   var col=isPiece(a.role)?pcCol(k):(ROLECOL[a.role]||'#334155');
   var v=isPiece(a.role)?(' = '+a.value+(a.cut?(' &minus;'+a.cut):'')+(a.pos&&a.pos.length?(' @'+a.pos.slice().sort(function(x,y){return x-y;}).join(',')):'')):((a.role==='indicator')?(' ('+a.itype+(a.isub?('/'+a.isub):'')+')'):(a.role==='deletion'?(' &minus;'+(a.value||'')):''));
   return '<span class="g-tag" style="border-color:'+col+'"><b style="color:'+col+'">'+(a.role==='definition'&&a.dkind==='dbe'?'definition by example':a.role)+(a.rule?('/'+a.rule):'')+'</b> '+phraseOf(a.idx)+v+' <a href="#" data-k="'+k+'" class="g-rm">×</a></span>';
  }).join('');
  Array.prototype.slice.call(listDiv.querySelectorAll('.g-rm')).forEach(function(x){x.onclick=function(e){e.preventDefault();assignments.splice(+x.dataset.k,1);drawRows();drawList();drawTiles();saveAssignments();};});
 }
 // Click a COLOURED (owned) tile to RELEASE its whole piece back into edit: re-tick exactly its
 // words, restore its role/value/cut, and drop its tiles back into the blue selection to re-pick.
 function releasePiece(k){
  var a=assignments[k];if(!a)return;
  assignments.splice(k,1);
  Array.prototype.slice.call(tbody.querySelectorAll('input.g-chk')).forEach(function(c){c.checked=(a.idx.indexOf(+c.value)>=0);});
  roleSel.value=a.role;
  if(a.role==='indicator'&&a.itype)itype.value=a.itype;
  if(a.role==='selection'&&selrule&&a.rule)selrule.value=a.rule;
  roleFields();
  if(a.role==='indicator'&&a.isub&&isub)isub.value=a.isub;
  if(a.role==='definition'&&dkind)dkind.value=a.dkind||'def';
  if(isValued(a.role)||a.role==='letters'||a.role==='replacement'||a.role==='deletion'||a.role==='selection')addInp.value=a.value||'';
  if(cutEl)cutEl.value=a.cut||'';
  selPos=(a.pos||[]).slice();
  updateBar();
  drawCutPrev();drawRows();drawList();drawTiles();saveAssignments();
  note('released — adjust tiles / value, then Assign');
 }
 function updateBar(){var idx=checkedIdx();if(idx.length){bar.style.display='';selLbl.textContent=phraseOf(idx);}else{bar.style.display='none';}}
 function clearChecks(){Array.prototype.slice.call(tbody.querySelectorAll('input.g-chk')).forEach(function(c){c.checked=false;});updateBar();}
 function inferSub(){
  if(roleSel.value!=='indicator'||itype.value!=='deletion')return;
  var syn='';assignments.forEach(function(a){if(a.role==='synonym'&&a.value)syn=a.value;});
  if(!syn)return;
  fetch('/hsinfer?base='+encodeURIComponent(syn)+'&answer='+encodeURIComponent(DATA.answer)).then(function(r){return r.json();}).then(function(o){if(o&&o.subtype)isub.value=o.subtype;});
 }
 // Repopulate the sub-type dropdown from the selected indicator type (data-driven, from
 // DATA.subtypes) so deletion / selection / letter_shift each show THEIR sub-types. Types with
 // no sub-types (reversal, anagram, ...) hide it. Keeps a matching value selected if possible.
 function fillSub(){var subs=(DATA.subtypes||{})[itype.value]||null;
  if(!subs){isub.style.display='none';isub.innerHTML='';return;}
  var cur=isub.value;
  isub.innerHTML=subs.map(function(s){return '<option value="'+s[0]+'">'+s[1]+'</option>';}).join('');
  if(cur){for(var i=0;i<isub.options.length;i++){if(isub.options[i].value===cur){isub.value=cur;break;}}}
  isub.style.display=(roleSel.value==='indicator')?'':'none';}
 function roleFields(){var r=roleSel.value;
  itype.style.display=(r==='indicator')?'':'none';
  if(dkind)dkind.style.display=(r==='definition')?'':'none';    // plain def vs def-by-example
  if(selrule)selrule.style.display=(r==='selection')?'':'none'; // the selection rule picker
  fillSub();                                                    // data-driven sub-type dropdown
  candWrap.style.display=((isPiece(r)&&r!=='anagram')||r==='deletion')?'':'none'; // deletion = type
  if(candSel)candSel.style.display=(isValued(r)||r==='selection')?'':'none';
  if(delEl)delEl.style.display=(r==='synonym'||r==='substitution'||r==='indicator')?'':'none';  // prune UI
  if(cutWrap)cutWrap.style.display=(isValued(r)||r==='anagram')?'':'none'; // delete letters from a
  if(!isValued(r)&&r!=='anagram'&&cutEl)cutEl.value='';          // derivative, or from anagram fodder
  if(addInp)addInp.placeholder=(r==='letters')?'exact letters, e.g. G':((r==='replacement')?'the new letter, e.g. T (blank = the tile letter)':((r==='deletion')?'removed letters, e.g. A (blank = its own letters)':((r==='selection')?'derived from the word by the rule':'new value')));
  drawCutPrev();
  if(isValued(r))fetchCands();
  if(r==='selection')fillSelCands();
  if(r==='indicator')fetchTypes();               // show current DB typings + prune links
  if(r==='indicator'&&itype.value==='deletion')inferSub();
 }
 function delRow(word,value,kind){var f=document.createElement('form');f.method='post';f.action='/hsdelete';
  function h(n,v){var i=document.createElement('input');i.type='hidden';i.name=n;i.value=v;f.appendChild(i);}
  h('only',DATA.cid);h('from',DATA.back||DATA.cid);h('kind',kind||'synonym');h('word',word);h('value',value);
  if(DATA.src&&DATA.pnum){h('src',DATA.src);h('pnum',DATA.pnum);}
  document.body.appendChild(f);f.submit();}
 function fetchTypes(){var idx=checkedIdx();if(!idx.length||!delEl){if(delEl)delEl.innerHTML='';return;}
  var phr=phraseOf(idx);
  fetch('/hstypes?phrase='+encodeURIComponent(phr)).then(function(r){return r.json();}).then(function(list){
   if(!list.length){delEl.innerHTML='<span style="color:#94a3b8">no DB typings for this word</span>';return;}
   delEl.innerHTML='typed in DB — prune a rogue one: '+list.map(function(o){
    var v=o.t+(o.s?('/'+o.s):'');
    return '<a href="#" class="g-delx" data-v="'+v+'" data-k="indicator">'+v+' ×</a>';}).join(' &nbsp; ');
   Array.prototype.slice.call(delEl.querySelectorAll('.g-delx')).forEach(function(x){
    x.onclick=function(e){e.preventDefault();delRow(phr,x.dataset.v,'indicator');};});
  }).catch(function(){delEl.innerHTML='';});}
 function fetchCands(){var idx=checkedIdx();if(!idx.length){candSel.innerHTML='';if(delEl)delEl.innerHTML='';return;}
  candSel.innerHTML='<option>…</option>';var phr=phraseOf(idx);
  fetch('/hslookup?id='+DATA.cid+'&phrase='+encodeURIComponent(phr)).then(function(r){return r.json();}).then(function(list){
   if(!list.length){candSel.innerHTML='<option value="">(none in DB — add below)</option>';}
   else{candSel.innerHTML='<option value="">— pick —</option>'+list.map(function(o){return '<option value="'+o.v+'">'+o.v+' ('+o.m+')</option>';}).join('');}
   if(delEl){var del=list.filter(function(o){return o.del;});
    delEl.innerHTML=del.length?('rogue? prune: '+del.map(function(o){return '<a href="#" class="g-delx" data-v="'+o.v+'" data-k="'+(o.dk||'synonym')+'">'+o.v+' ×</a>';}).join(' &nbsp; ')):'';
    Array.prototype.slice.call(delEl.querySelectorAll('.g-delx')).forEach(function(x){x.onclick=function(e){e.preventDefault();delRow(phr,x.dataset.v,x.dataset.k);};});}
  }).catch(function(){candSel.innerHTML='<option value="">(lookup failed — add below)</option>';});
 }
 tbody.addEventListener('change',function(e){if(e.target.classList&&e.target.classList.contains('g-chk')){updateBar();if(isValued(roleSel.value))fetchCands();if(roleSel.value==='selection')fillSelCands();if(roleSel.value==='indicator')fetchTypes();drawCutPrev();}});
 roleSel.addEventListener('change',roleFields);
 itype.addEventListener('change',roleFields);
 if(selrule)selrule.addEventListener('change',fillSelCands);
 var msgEl=root.querySelector('#g-msg');
 function note(t){if(msgEl)msgEl.textContent=t||'';}
 function assignNow(){
  var idx=checkedIdx();if(!idx.length){note('tick a word first');return;}
  var r=roleSel.value, a={idx:idx,role:r}, survivor=null;
  if(isValued(r)){var v=((addInp.value||'').trim()||candSel.value||'').toUpperCase();if(!v){note('type the value first');return;}a.value=v;
   var cut=(cutEl&&cutEl.value||'').trim().toUpperCase();       // delete a run from the derivative
   if(cut){survivor=applyCut(v,cut);
    if(survivor===null){note('“'+cut+'” is not a run of '+v);return;}
    if(!survivor.length){note('cannot delete the whole value ('+v+')');return;}
    a.cut=cut;}}
  if(r==='letters'||r==='replacement'){var lv=(addInp.value||'').trim().toUpperCase();if(lv)a.value=lv;}
  if(r==='selection'){var sfl=fodderLetters(idx),srl=selrule?selrule.value:'';
   var scands=selCands(sfl,srl);
   if(!scands.length){note('the ticked word(s) ('+sfl+') are too short for the "'+srl+'" rule');return;}
   var sv=((addInp.value||'').trim()||candSel.value||scands[0]||'').toUpperCase();
   if(scands.indexOf(sv)<0){note(sv+' is not the '+srl+' selection of '+sfl+' (must be '+scands.join(' or ')+')');return;}
   a.value=sv;a.rule=srl;}
  if(r==='anagram'){var fl=fodderLetters(idx);if(!fl){note('tick the fodder word(s) first');return;}a.value=fl;}
  if(r==='deletion'){var dv=(addInp.value||'').trim().toUpperCase().replace(/[^A-Z]/g,'')||fodderLetters(idx);
   if(!dv){note('type the removed letters');return;}a.value=dv;}   // named deletion, no tiles
  if(r==='indicator'){a.itype=itype.value;a.isub=((DATA.subtypes||{})[itype.value])?isub.value:'';}
  if(r==='definition'&&dkind)a.dkind=dkind.value;               // 'def' | 'dbe' (label only)
  if(isPiece(r)){
   var placeVal=(survivor!==null)?survivor:a.value;             // what actually lands on the tiles
   var pos=selPos.slice();
   if(!pos.length){                                             // anagram fodder is SCRAMBLED, so it
    var loc=(r!=='anagram'&&placeVal)?locateValue(placeVal):null; //  can't auto-place: click tiles
    if(!loc&&r!=='anagram'&&placeVal&&placeVal.length>=3){      // a LETTER-SHIFT lands the value's
     var rots=[placeVal.slice(-1)+placeVal.slice(0,-1),         //   letters rotated by one: TERNS
               placeVal.slice(1)+placeVal.slice(0,1)];          //   -> STERN (last->front) is in
     for(var ri=0;ri<rots.length;ri++){var rl=locateValue(rots[ri]);if(rl){loc=rl;break;}}}
    if(loc)pos=loc;                                             //   the answer, so auto-place there
    else if(r==='anagram'){note('now click the answer tiles this fodder ('+placeVal+') fills, in any order, then Assign');return;}
    else{note('now click the answer tiles this piece makes ('+(placeVal||'')+'), then Assign');return;}}
   a.pos=pos.sort(function(x,y){return x-y;});
   if(survivor!==null&&a.pos.length!==survivor.length){
    note('you placed '+a.pos.length+' tile(s) but '+v+' −'+a.cut+' = '+survivor+' ('+survivor.length+')');return;}
   if(r==='anagram'){                                           // fodder must CONTAIN the tiles;
    var ts=msort(a.pos.map(function(p){return DATA.answer[p-1];}).join(''));  // surplus = a deletion
    var acut=(cutEl&&cutEl.value||'').trim().toUpperCase().replace(/[^A-Z]/g,'');
    if(acut){                                                   // an explicitly named deletion
     var r1=msub(a.value, acut);
     if(!r1.ok){note('“'+acut+'” has a letter not in the fodder '+a.value);return;}
     if(r1.rem!==ts){note('fodder '+a.value+' − '+acut+' = '+(r1.rem||'(empty)')+' ≠ tiles ('+ts+')');return;}
     a.cut=msort(acut);
    }else{                                                      // surplus fodder is the deletion
     var r2=msub(a.value, ts);
     if(!r2.ok){note('fodder '+a.value+' does not contain all those tiles ('+ts+')');return;}
     if(r2.rem)a.cut=r2.rem;}}
   if((r==='letters'||r==='replacement')&&!a.value){a.value=a.pos.map(function(p){return DATA.answer[p-1];}).join('');}
  }
  assignments=assignments.filter(function(x){return !x.idx.some(function(i){return idx.indexOf(i)>=0;});});
  assignments.push(a);addInp.value='';if(cutEl)cutEl.value='';selPos=[];note('');drawCutPrev();drawRows();drawList();drawTiles();clearChecks();saveAssignments();
 }
 root.querySelector('#g-assign').addEventListener('click',assignNow);
 if(cutEl)cutEl.addEventListener('input',drawCutPrev);
 if(addInp)addInp.addEventListener('input',drawCutPrev);
 // picking a synonym from the dropdown fills its value; then click the answer tiles + Assign.
 // With a deletion typed, DON'T auto-assign (let the user place the survivor first).
 candSel.addEventListener('change',function(){if(isValued(roleSel.value)&&candSel.value){addInp.value=candSel.value;drawCutPrev();if(!(cutEl&&cutEl.value.trim()))assignNow();}
  else if(roleSel.value==='selection'&&candSel.value){addInp.value=candSel.value;}});
 var grs=root.querySelector('#g-resolve');
 if(grs)grs.addEventListener('click',function(){payload.value=JSON.stringify(assignments);var f=root.querySelector('#g-form');f.action='/hsresolve';f.submit();});
 root.querySelector('#g-commit').addEventListener('click',function(){
  var fd=new FormData();fd.append('only',DATA.cid);fd.append('payload',JSON.stringify(assignments));
  var al=root.querySelector('#g-andlit');if(al&&al.checked)fd.append('andlit','1');
  cmsg.textContent='committing…';cmsg.style.color='#64748b';
  fetch('/hsmanualcommit',{method:'POST',body:fd}).then(function(r){return r.json();}).then(function(o){
   if(o.ok){cmsg.textContent='✓ '+o.msg+' — opening the clue page…';cmsg.style.color='#16a34a';
    setTimeout(function(){window.location.href='/?id='+encodeURIComponent(DATA.back||DATA.cid)+'#clue-'+DATA.cid;},800);}
   else{cmsg.textContent='✗ '+o.msg;cmsg.style.color='#dc2626';cmsg.style.fontSize='1rem';try{cmsg.scrollIntoView({block:'center'});}catch(e){}}
  }).catch(function(){cmsg.textContent='commit failed (network)';cmsg.style.color='#dc2626';});});
 root.querySelector('#g-uncommit').addEventListener('click',function(){
  var fd=new FormData();fd.append('only',DATA.cid);
  fetch('/hsmanualuncommit',{method:'POST',body:fd}).then(function(r){return r.json();}).then(function(o){cmsg.textContent=o.msg;cmsg.style.color=o.ok?'#16a34a':'#dc2626';}).catch(function(){cmsg.textContent='uncommit failed';cmsg.style.color='#dc2626';});});
 drawRows();drawList();drawTiles();updateBar();roleFields();
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


def _hs_puzzle_context(clue_id, src, pnum, back):
    """PUZZLE MODE work-list strip for /hs: every clue of the puzzle still FAIL/PENDING now
    (live statuses, so the list shrinks as commits land) plus the current clue, each a link
    that keeps the puzzle context; plus the sig-regression button (the post-publish A/B
    trigger — memory: publish-first-process). Returns strip_html; best-effort — any load
    failure just renders nothing extra."""
    from core import triage
    from urllib.parse import quote
    try:
        meta, clues = triage.collect_puzzle(src, pnum)
    except Exception:
        return ""
    work = [c for c in clues if c["status"] in ("fail", "pending") or c["id"] == clue_id]
    ctx_qs = "&amp;src=%s&amp;pnum=%s" % (quote(src), quote(str(pnum)))
    bq = quote(back, safe="")
    chips = []
    for c in work:
        cur = (c["id"] == clue_id)
        col = {"fail": "#dc2626", "pending": "#b45309"}.get(c["status"], "#16a34a")
        inner = escape("%s%s" % (c["number"], (c["direction"] or "")[:1]))
        style = ("display:inline-block;margin:.12rem;padding:.15rem .5rem;border-radius:7px;"
                 "border:2px solid %s;%stext-decoration:none;font-weight:700;color:#0f172a;"
                 "font-size:.9rem" % (col, "background:#fef9c3;" if cur else "background:#fff;"))
        if cur:
            chips.append('<span style="%s">%s</span>' % (style, inner))
        else:
            chips.append('<a href="/hs?id=%d&amp;from=%s%s" style="%s">%s</a>'
                         % (c["id"], bq, ctx_qs, style, inner))

    sig_st = _sigreg_read()
    if sig_st.get("running"):
        sig_line = ('<div class="wfw-notice" style="background:#fffbeb">Signature regression '
                    'RUNNING (started %s): %s — refresh for progress.</div>'
                    % (escape(sig_st.get("started", "")), escape(sig_st.get("phase", ""))))
    elif sig_st.get("results"):
        sig_line = ('<div class="wfw-notice">Last signature regression: %s</div>'
                    % escape("; ".join("template %s → %s (%s)"
                                       % (r["template"], r["action"], r["summary"][:80])
                                       for r in sig_st["results"])))
    else:
        sig_line = ""
    strip = ('<div style="border:1px solid #e2e8f0;border-radius:10px;padding:.5rem .7rem;'
             'margin:.4rem 0;background:#f8fafc">'
             '<div style="display:flex;gap:.8rem;align-items:center;flex-wrap:wrap">'
             '<b>%s %s</b><span style="color:#64748b;font-size:.9rem">%d pass &middot; '
             '%d pending &middot; %d fail</span>'
             '<form method="post" action="/sigregress" style="display:inline;margin-left:auto">'
             '<input type="hidden" name="src" value="%s">'
             '<input type="hidden" name="pnum" value="%s">'
             '<input type="hidden" name="back" value="hs">'
             '<button style="background:#7c3aed;color:#fff;border:none;border-radius:8px;'
             'padding:.25rem .7rem;font-weight:700;cursor:pointer;font-size:.8rem" '
             'title="Background batch: A/B-check every pending signature (two full solves '
             'each) and promote the clean ones — run it when a day\'s puzzles are done.">'
             'Regression-check pending signatures</button></form></div>'
             '<div style="margin-top:.25rem">%s</div>%s</div>'
             % (escape(src.title()), escape(str(pnum)), meta["pass"], meta["pending"],
                meta["fail"], escape(src, quote=True), escape(str(pnum), quote=True),
                "".join(chips), sig_line))

    return strip


def _hs_diag_banner(clue_id, pnum):
    """The diagnosis banner for ONE clue on /hs: category badge(s) + the LOUD engine-gap
    line + Reading/Problem/Fix + any signature shape, from the nightly diagnosis files.
    Rendered on EVERY /hs view (the clue row knows its own puzzle number), so the warning
    can never vanish just because the user arrived without the puzzle context in the URL.
    Best-effort: no files / no entry → empty string."""
    from core import triage
    try:
        d = triage.load_diagnoses(pnum).get(str(clue_id)) or {}
    except Exception:
        d = {}
    try:
        cls = triage.load_classified(pnum).get(str(clue_id)) or {}
    except Exception:
        cls = {}
    reasons = cls.get("reasons") or []
    bits = []
    badges = "".join('<span style="background:%s;color:#fff;border-radius:6px;'
                     'padding:.1rem .45rem;font-size:.78rem;font-weight:700;'
                     'margin-right:.3rem">%s</span>'
                     % (_CAT_META.get(r, (r, "#64748b"))[1],
                        escape(_CAT_META.get(r, (r.replace("_", " "), ""))[0].upper()))
                     for r in reasons)
    if badges:
        bits.append('<div>%s</div>' % badges)
    if "missing_engine" in reasons:
        bits.append('<div style="color:#b91c1c;font-weight:700;margin-top:.2rem">'
                    'ENGINE GAP &mdash; don\'t chase data. Hand-solve it (Commit manual); '
                    'the manual solve goes on the engine-improvement worklist.</div>')
    for key, lab in (("reading", "Reading"), ("problem", "Problem"), ("action", "Fix")):
        if d.get(key):
            bits.append('<div style="margin-top:.2rem"><b>%s:</b> %s</div>'
                        % (lab, escape(d[key])))
    sig = d.get("signature") or {}
    if sig:
        bits.append('<div style="margin-top:.2rem">signature needed: '
                    '<code>%s</code> (tier: <b>%s</b>)%s</div>'
                    % (escape(sig.get("shape", "")), escape(sig.get("tier", "")),
                       (" &mdash; " + escape(sig["note"])) if sig.get("note") else ""))
    return ('<div style="border:1px solid #fde68a;background:#fffbeb;border-radius:10px;'
            'padding:.45rem .7rem;margin:.3rem 0;font-size:.9rem">%s</div>'
            % "".join(bits)) if bits else ""


def _span_surface(clue_id, back_raw=None, psrc=None, ppnum=None):
    """Render the VERTICAL assignment grid for one clue: a row per clue word (preloaded with
    its current role) with a checkbox + role + 'brings' column, plus the role picker, the
    in-memory assignment list, a single Resolve, and the solved breakdown card. With
    src+pnum (PUZZLE MODE) it adds the work-list strip + diagnosis banner + Re-run — the
    one-surface review flow that replaces the /triage page."""
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
    rows = _word_roles(ctx, parse, filler, split_hyphens=True)  # [{idx,text,role,label,value}];
                                                  # hyphenated words split so each part gets a role
    back = back_raw or str(clue_id)
    try:
        saved_list = json.loads(saved) if saved else []
    except Exception:
        saved_list = []
    if not saved_list:                        # no prior hand-solve -> seed EDITABLE assignments:
        saved_list = _assignments_from_diagnosis(clue_id, pnum, rows)  # the triage reading first
    if not saved_list:                        # else from the stored parse, so re-tagging one
        saved_list = _assignments_from_parse(ctx, parse)   # word doesn't mean reassigning all

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
            "subtypes": _IND_SUBTYPES,            # per-type sub-type options (data-driven dropdown:
                                                  #   deletion / selection / letter_shift)
            "assignments": saved_list,
            "src": psrc or "",                    # puzzle context (PUZZLE MODE) — JS-built
            "pnum": str(ppnum) if ppnum else ""}  # forms carry it so the strip survives

    if parse is not None:
        screen = SCREENS.get(parse.operation) or SCREENS.get(parse.solved_by)
        card = screen(ctx, parse) if screen else wfw_render.render_parse(parse, ctx=ctx)
    else:
        card = '<p>Not solved yet — assign roles and Resolve.</p>'

    rootid = "g-%d" % clue_id
    from urllib.parse import quote
    # PUZZLE MODE extras: the work-list strip + the current clue's diagnosis banner, and a
    # query-string tail that keeps the puzzle context on every /hs link.
    ctx_qs = (("&amp;src=%s&amp;pnum=%s" % (quote(psrc), ppnum)) if (psrc and ppnum) else "")
    ctx_hidden = ('<input type="hidden" name="src" value="%s">'
                  '<input type="hidden" name="pnum" value="%s">'
                  % (escape(psrc or "", quote=True), ppnum or ""))
    strip_html = ""
    if psrc and ppnum:
        strip_html = _hs_puzzle_context(clue_id, psrc, ppnum, back)
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
            return ('<a href="/hs?id=%d&amp;from=%s%s" style="text-decoration:none;'
                    'font-weight:700;color:#0d9488">%s</a>' % (nid, bq, ctx_qs, label))
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
    ans_tiles = "".join(
        '<span class="g-atile" data-pos="%d" style="display:inline-flex;align-items:center;'
        'justify-content:center;min-width:1.7rem;height:2.1rem;margin:.12rem;border:2px solid '
        '#cbd5e1;border-radius:7px;font-weight:800;font-family:monospace;cursor:pointer;'
        'background:#fff">%s</span>' % (i + 1, escape(ch))
        for i, ch in enumerate(data["answer"]))
    p = [_SPAN_CSS, "<script>%s</script>" % _SPAN_JS,
         '<div id="%s" class="g-root">' % rootid,
         _cid_label(clue_id, src, pnum, cnum, direction),
         '<a href="/?id=%s#clue-%d" style="display:inline-block;margin:.25rem 0;'
         'text-decoration:none;font-weight:700;color:#0d9488">&larr; back to clue page</a>'
         % (quote(back, safe=""), clue_id),
         strip_html,
         nav_html,
         '<div style="margin:.3rem 0;font-size:1.1rem;font-weight:600">%s</div>'
         % escape(clue_text),
         '<div class="g-ans">Answer &mdash; for a synonym/letters piece, after its value, '
         'click the tiles it makes '
         '<span style="color:#94a3b8;font-weight:400">(click a coloured tile to release its piece '
         'and re-edit it)</span>:<br><span id="g-atiles" style="margin-top:.2rem;'
         'display:inline-block">%s</span></div>' % ans_tiles,
         '<p style="font-size:.85rem;color:#64748b;margin:.2rem 0">Tick the word(s), pick a role. '
         '<b>synonym</b>: type the value then click the answer tiles it makes. If a letter is '
         'deleted from the derivative (e.g. speech=ORATION, &ldquo;scrapping introduction&rdquo; '
         'removes O), type the deleted run in <b>&minus; delete</b> and the survivor places itself. '
         '<b>letters</b> (literal): just click the answer tiles it makes. '
         '<b>anagram fodder</b>: tick the fodder word(s) &mdash; its value is their letters (shown '
         'in the preview) &mdash; then click the answer tiles they rearrange into (any order); tag '
         'the anagram word separately as an <b>indicator</b> (type anagram). If the fodder is '
         'LONGER than the answer, place it on the (fewer) tiles and account the removed letter with '
         'a <b>deletion</b> role: tick the word that supplies it (e.g. &ldquo;a&rdquo; &rarr; A) and '
         'pick <b>deletion</b> &mdash; no tiles. (Or type the letter directly in <b>&minus; delete</b> '
         'on the fodder.) '
         '<b>indicator/definition/link/filler</b>: no tiles. Then Assign. When every tile is '
         'coloured, <b>Commit (manual)</b>.</p>',
         '<table class="g-tbl"><thead><tr><th></th><th>word</th><th>role</th><th>brings</th>'
         '</tr></thead><tbody id="g-tbody">%s</tbody></table>' % trs,
         '<div id="g-bar" class="g-bar" style="display:none">',
         '<span>Selected: <b id="g-sel"></b></span>',
         '<select id="g-role">'
         '<option value="definition">definition</option>'
         '<option value="synonym">synonym</option>'
         '<option value="substitution">substitution (abbr / symbol)</option>'
         '<option value="letters">letters (exact)</option>'
         '<option value="replacement">replacement letter (unclued)</option>'
         '<option value="selection">selection (letters from word)</option>'
         '<option value="anagram">anagram fodder</option>'
         '<option value="deletion">deletion (letters removed)</option>'
         '<option value="indicator">indicator</option>'
         '<option value="link">link word</option>'
         '<option value="filler">filler</option>'
         '<option value="none">none (clear)</option></select>',
         '<select id="g-itype" style="display:none">%s</select>' % itype_opts,
         '<select id="g-isub" style="display:none">%s</select>' % isub_opts,
         '<select id="g-selrule" style="display:none" title="Which letters the selection '
         'takes from the ticked word(s) — mirrors the engine rules (core.selection).">'
         '<option value="first">first letter</option>'
         '<option value="last">last letter</option>'
         '<option value="outer">outer letters</option>'
         '<option value="middle">middle letter(s)</option>'
         '<option value="alternate">alternate letters</option>'
         '<option value="remove_first">all but first (behead)</option>'
         '<option value="remove_last">all but last (curtail)</option>'
         '<option value="remove_outer">inner letters (ends off)</option>'
         '<option value="remove_middle">all but middle (heartless)</option></select>',
         '<select id="g-dkind" style="display:none" title="A plain definition, or a '
         'definition by example (DBE) — where the clue defines the answer via an example '
         '(e.g. “flower” for a river). Same in every respect but the label.">'
         '<option value="def">definition</option>'
         '<option value="dbe">definition by example (DBE)</option></select>',
         '<span id="g-cand" style="display:none">value: <select id="g-candsel"></select> '
         'or add <input id="g-add" placeholder="new value" size="12"> '
         '<span id="g-cutwrap" style="display:none;margin-left:.35rem">&minus; delete '
         '<input id="g-cut" placeholder="e.g. O" size="5" title="Delete a run of letters from '
         'the derivative (e.g. speech=ORATION, scrapping introduction removes O -> RATION)">'
         '<span id="g-cutprev" style="margin-left:.35rem;font-size:.85rem"></span></span> '
         '<span id="g-del" style="margin-left:.4rem;font-size:.85rem;color:#b45309"></span></span>',
         '<button type="button" id="g-assign" class="g-assign">Assign</button>',
         '<span id="g-msg" style="color:#dc2626;font-size:.85rem"></span>',
         '</div>',
         '<div id="g-list" class="g-list"></div>',
         '<form method="post" action="/hsresolve" id="g-form">',
         '<input type="hidden" name="only" value="%d">' % clue_id,
         '<input type="hidden" name="from" value="%s">' % escape(back, quote=True),
         ctx_hidden,
         '<input type="hidden" name="payload" id="g-payload">',
         # PUBLISH-FIRST PROCESS (memory: publish-first-process): the user's one action per
         # clue is Commit (manual). Resolve & solve / Save pieces / Re-run are gone from the
         # UI (routes kept); signature + engine work happens POST-publish from the frozen
         # manual solves.
         '<label style="font-weight:600;font-size:.9rem;cursor:pointer" '
         'title="All-in-one (&amp;lit): the whole clue is BOTH the wordplay AND the definition '
         '(the same words used twice). Tag the wordplay as usual and tick this; the whole clue is '
         'taken as the definition, so you need no separate definition word. Verdict stays PENDING '
         'for your confirmation, like a cryptic definition.">'
         '<input type="checkbox" id="g-andlit"> &amp;lit (all-in-one)</label>',
         '<button type="button" id="g-commit" class="g-resolve" style="background:#7c3aed;'
         'margin-left:.5rem" title="Record exactly what you tagged + placed on the tiles as '
         'the solution (frozen; your reusable pieces are saved to the reference DB). This is '
         'THE button: check the reading, correct it, Commit.">Commit (manual)</button>',
         '<button type="button" id="g-uncommit" style="margin-left:.4rem;background:#fff;'
         'color:#7c3aed;border:1px solid #7c3aed;border-radius:8px;padding:.35rem .8rem;'
         'font-weight:700;cursor:pointer">Uncommit</button>',
         '<span id="g-cmsg" style="margin-left:.5rem;font-weight:700"></span>',
         '</form>',
         '<form method="post" action="/hsstatus" style="margin:.5rem 0;display:flex;'
         'gap:.4rem;align-items:center;flex-wrap:wrap">',
         '<input type="hidden" name="only" value="%d">' % clue_id,
         '<input type="hidden" name="from" value="%s">' % escape(back, quote=True),
         ctx_hidden,
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
         ctx_hidden,
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
    """Post/Redirect/Get back to the span surface (clean URL, no resubmit on refresh).
    Carries the puzzle context (src/pnum) through when the posting form/link supplied it,
    so puzzle-mode /hs (the work-list strip) survives every action."""
    from urllib.parse import quote
    extra = ("&from=%s" % quote(back_raw, safe="")) if back_raw else ""
    try:
        psrc = (request.form.get("src") or request.args.get("src") or "").strip()
        ppnum = (request.form.get("pnum") or request.args.get("pnum") or "").strip()
    except Exception:
        psrc = ppnum = ""
    if psrc and ppnum.isdigit():
        extra += "&src=%s&pnum=%s" % (quote(psrc), quote(ppnum))
    return redirect("/hs?id=%s&notice=%s%s" % (only, quote(msg), extra))


@app.route("/hs")
def hs_route():
    """The span-assignment hand-solver. Single clue: /hs?id=NNN. PUZZLE MODE:
    /hs?src=telegraph&pnum=31285 (no id) — builds the FAIL/PENDING work list for the
    puzzle and opens its first clue; the work-list strip + prev/next then walk the whole
    puzzle without leaving /hs. This is the review surface (replaces the /triage page)."""
    cid = (request.args.get("id") or "").strip()
    src = (request.args.get("src") or "").strip()
    pnum = (request.args.get("pnum") or "").strip()
    if not cid.isdigit() and src and pnum.isdigit():
        from core import triage
        from urllib.parse import quote
        try:
            meta, clues = triage.collect_puzzle(src, int(pnum))
        except Exception as e:
            return _page('<p class="warn">Could not load %s %s: %s</p>'
                         % (escape(src), escape(pnum), escape(str(e))))
        work = [c for c in clues if c["status"] in ("fail", "pending")]
        if not work:
            return _page('<p>No FAIL or PENDING clues in %s %s — nothing to review.</p>'
                         % (escape(src.title()), escape(pnum)))
        # `from` = the WHOLE puzzle (every clue, page order), NOT just the fails — it is the
        # clue-page clutch, and the clue page is the record of ALL clues that ran. Filtering
        # it made "back to clue page" rebuild a fail-only page (user report 2026-07-09).
        # Fail-hopping belongs to the work-list strip alone.
        ids = ",".join(str(c["id"]) for c in clues)
        notice = (request.args.get("notice") or "").strip()
        return redirect("/hs?id=%d&from=%s&src=%s&pnum=%s%s"
                        % (work[0]["id"], quote(ids, safe=""), quote(src), quote(pnum),
                           ("&notice=%s" % quote(notice)) if notice else ""))
    if not cid.isdigit():
        return _page('<p class="warn">Enter a clue id, e.g. '
                     '<a href="/hs?id=10075533">/hs?id=10075533</a> — or a puzzle, e.g. '
                     '<a href="/hs?src=telegraph&amp;pnum=31285">'
                     '/hs?src=telegraph&amp;pnum=31285</a></p>')
    back = (request.args.get("from") or cid).strip()
    notice = (request.args.get("notice") or "").strip()
    body = ('<div class="wfw-notice">%s</div>' % escape(notice)) if notice else ""
    body += _span_surface(int(cid), back, psrc=src or None,
                          ppnum=int(pnum) if pnum.isdigit() else None)
    return _page(body)


@app.route("/hslookup")
def hslookup_route():
    """AJAX candidate values for a ticked word-group: synonyms + abbreviations (the wordplay
    table) + substitutions. A LOOKUP, never a solve. Abbreviations/substitutions are listed
    FIRST (they are the short wordplay values the user hunts for and must never be crowded out
    of the list by the synonyms), then the synonyms; BOTH groups sorted ALPHABETICALLY so a
    value is easy to find among many. `del` marks a DIRECT synonyms_pairs row (prunable if rogue)."""
    import json, sqlite3
    phrase = (request.args.get("phrase") or "").strip()
    abbr, syn = [], []
    if phrase:
        direct, subs = set(), set()
        try:
            con = sqlite3.connect(admin_db.CRYPTIC_DB)
            for (v,) in con.execute("SELECT synonym FROM synonyms_pairs "
                                    "WHERE lower(word)=lower(?)", (phrase,)):
                direct.add((v or "").upper())
            for (v,) in con.execute("SELECT substitution FROM substitutions "
                                    "WHERE lower(original_word)=lower(?)", (phrase,)):
                if v:
                    subs.add(v.strip().upper())
            con.close()
        except Exception:
            pass
        try:
            la = batch_wiring()["lookup_all"]
            mechs = {}                                      # value -> set of mechanisms
            for v, m in la(phrase):                        # NO early cap — collect every value
                v = (v or "").upper()
                if v and m in ("synonym", "abbreviation"):
                    mechs.setdefault(v, set()).add(m)
            wp_direct = set()
            try:
                con = sqlite3.connect(admin_db.CRYPTIC_DB)
                for (v,) in con.execute("SELECT substitution FROM wordplay "
                                        "WHERE lower(indicator)=lower(?)", (phrase,)):
                    wp_direct.add((v or "").strip().upper())
                con.close()
            except Exception:
                pass
            for v, ms in mechs.items():                    # a value that is EVER an abbreviation
                # `dk` = which table a prune should hit (synonym row wins: it is the
                # common pollution case); None when the value is a lookup artifact.
                dk = ("synonym" if v in direct
                      else ("substitution" if v in wp_direct else None))
                if "abbreviation" in ms:                   # is grouped as an abbreviation
                    abbr.append({"v": v, "m": "abbreviation", "del": bool(dk), "dk": dk})
                else:
                    syn.append({"v": v, "m": "synonym", "del": bool(dk), "dk": dk})
            for v in subs:                                 # the substitutions table
                if v and v not in mechs:
                    abbr.append({"v": v, "m": "substitution", "del": False})
        except Exception:
            abbr, syn = [], []
    abbr.sort(key=lambda d: d["v"])
    syn.sort(key=lambda d: d["v"])
    out = abbr + syn                                       # no cap — alphabetical, so a long
    return app.response_class(json.dumps(out),             # list is still easy to scan/jump
                              mimetype="application/json")


@app.route("/hstypes")
def hstypes_route():
    """AJAX: the DB indicator typings for a phrase — [{"t": type, "s": subtype}] — so the
    hand-solver can SHOW (and prune) a rogue typing. A lookup, never a solve."""
    import json as _j
    import sqlite3 as _s
    phrase = (request.args.get("phrase") or "").strip()
    out = []
    if phrase:
        try:
            con = _s.connect(admin_db.CRYPTIC_DB)
            for t, s in con.execute("SELECT wordplay_type, COALESCE(subtype,'') FROM "
                                    "indicators WHERE lower(word)=lower(?)", (phrase,)):
                out.append({"t": t, "s": s})
            con.close()
        except Exception:
            out = []
    return app.response_class(_j.dumps(out), mimetype="application/json")


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
    elif kind == "substitution":
        msg = admin_db.delete_substitution(word, value)
        apply_add_to_wiring({"kind": "substitution"})       # unknown kind -> full reload
    elif kind == "indicator":
        wp, _, sub = (value or "").partition("/")
        msg = admin_db.delete_indicator(word, wp, sub or None)
        apply_add_to_wiring({"kind": "indicator", "word": word, "type": wp})
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


@app.route("/hsrerun", methods=["POST"])
def hsrerun_route():
    """Re-solve ONE clue from /hs (the user's click; nothing else applied). Resident
    wiring — a couple of seconds, no snapshot rebuild. Mirrors /triagererun."""
    only = (request.form.get("only") or "").strip()
    back = (request.form.get("from") or only).strip()
    if not only.isdigit():
        return _hs_redirect(only, "No clue.", back)
    cid = int(only)
    _resolve_one(cid)
    conn = store.connect()
    try:
        sp = store.load_parse(conn, cid)
    finally:
        conn.close()
    status = ((sp.status if sp is not None else "") or "fail").upper()
    return _hs_redirect(only, "Re-solved: %s." % status, back)


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
_HSROLE_FODDER = {"synonym": "SYN_F", "substitution": "ABR_F",
                  "letters": "LIT_F", "selection": "SEL_F"}
# indicator type -> (operation, indicator slot role)
_HSIND_OP = {"deletion": ("deletion", "DEL_I"), "anagram": ("anagram", "ANA_I"),
             "reversal": ("reversal", "REV_I"), "container": ("container", "CON_I"),
             "insertion": ("container", "CON_I")}
# indicator types that LICENSE a piece (the engines find them via find_indicators /
# selection_rules) rather than defining the operation — residue in the candidate, not a slot.
_HSIND_RESIDUE = frozenset({"selection", "alternation", "alternating", "alternate"})


def _selection_candidates(phrase, rule):
    """The engine's selection rule applied to the phrase's letters — the mirror of the /hs
    grid's selCands (both mirror core.selection.SPAN_RULES, whose lambdas slice generic
    lists, so plain characters work). The commit validates a selection piece against this,
    so the derived value can never be free-typed."""
    from core import selection
    letters = [c for c in (phrase or "").upper() if c.isalpha()]
    fn = selection.SPAN_RULES.get(rule)
    if fn is None:
        return []
    try:
        return ["".join(cand) for cand in fn(letters) if cand]
    except Exception:
        return []


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
        elif role in _HSROLE_FODDER:              # a fodder value slot (SYN/ABR/LIT/SEL)
            syns.append((idx, (a.get("value") or "").strip().upper(), _HSROLE_FODDER[role]))
        elif role == "indicator":
            base = (a.get("itype") or "").split(":")[0]
            if base in _HSIND_RESIDUE:
                continue                          # licenses a piece; residue, not a slot
            spec = _HSIND_OP.get(base)
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
    if op == "deletion" and answer and len([s for s in syns if s[2] == "SYN_F"]) >= 2:
        from core import deletion
        ans = "".join(c for c in answer.upper() if c.isalpha())
        for bidx, bval, btok in syns:
            if btok != "SYN_F":
                continue
            for ridx, rval, rtok in syns:
                if rtok != "SYN_F" or ridx is bidx or not bval or not rval:
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
    for idx, _val, tok in syns:
        if rem_key is not None and tuple(idx) == rem_key:
            tok = "REM_F"
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


def _rollback_signature(tid):
    """Remove a just-filed signature (+ its slots) — the rollback when it doesn't earn its place."""
    from core import catalog_creator as CC
    con = sqlite3.connect(CC._CLUES_DB)
    try:
        con.execute("DELETE FROM catalog_templates WHERE id=?", (tid,))
        con.execute("DELETE FROM catalog_template_slots WHERE template_id=?", (tid,))
        con.commit()
    finally:
        con.close()


def _try_create_signature(cid, cand):
    """A Resolve that leaves the clue unsolved may imply a signature the catalog lacks. Decide
    the signature's RISK CLASS from the rubric (catalog_creator.rubric_tier) and file it
    accordingly:
      * PENDING-only (risky) — filed straight away; safe BY CONSTRUCTION (it runs in the final
        cascade stage, so it can only ever turn this FAIL into an amber 'needs checking', never
        a green pass). Kept iff it fires; else rolled back.
      * PASS-tier (safe shape) — may go green, so it must first clear the automatic before/after
        A/B (0 regressions, no stray new passes on other clues). Handled in _create_pass_signature.
    Returns a one-line note, or '' when nothing was attempted."""
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
    row = _load_clue(cid)
    if row is None:
        return ""
    ct, ans, _s, _pn, _d, enum, _cn = row
    answer = enum_space(ans, enum)
    if CC.rubric_tier(cand, answer) == "pending":
        return _create_pending_signature(cid, cand, sig)
    return _create_pass_signature(cid, cand, sig, ct, answer)


def _create_pending_signature(cid, cand, sig):
    """File a risky signature as PENDING-only and keep it iff the clue now fires as an amber
    'needs checking'. Safe by construction — it can never mint a green pass."""
    from core import catalog_creator as CC
    try:
        tid = CC.add_signature(cand, note="triage pending-only: %s (clue %s)" % (sig, cid),
                               tier="pending")
    except Exception:
        return ""
    if tid is None:
        return ""
    reload_wiring()
    _resolve_one(cid)
    conn = store.connect()
    try:
        cp = store.load_parse(conn, cid)
    finally:
        conn.close()
    if cp is not None and cp.status == "pending":
        return ("filed a PENDING-ONLY signature %s — the clue now shows amber (needs checking); "
                "confirm it to trust the solve." % sig)
    _rollback_signature(tid)
    reload_wiring()
    _resolve_one(cid)
    return ("signature %s did not fire (a needed value may be missing, or the roles don't "
            "assemble), so it was rolled back." % sig)


def _create_pass_signature(cid, cand, sig, clue_text, answer):
    """PASS-tier-eligible (safe shape). A pass-tier signature can affect OTHER clues, so it may
    NOT go green off nothing. File it as PENDING first (amber, harmless in the final stage), so
    the clue is solved-as-needs-checking immediately; it is promoted to green only after the
    automatic before/after A/B confirms it doesn't regress or fabricate on other clues. The
    promotion is a separate gated step: `python -m core.ab_signature promote <template_id> <clue_id>`."""
    from core import catalog_creator as CC
    try:
        tid = CC.add_signature(cand, note="triage pass-eligible (pending until A/B): %s (clue %s)"
                               % (sig, cid), tier="pending")
    except Exception:
        return ""
    if tid is None:
        return ""
    reload_wiring()
    _resolve_one(cid)
    conn = store.connect()
    try:
        cp = store.load_parse(conn, cid)
    finally:
        conn.close()
    if cp is not None and cp.status == "pending":
        return ("filed PASS-tier-eligible signature %s as pending (amber, needs checking) — it "
                "goes green only after the automatic regression check. To run the check and "
                "promote it if clean:  python -m core.ab_signature promote %d %d"
                % (sig, tid, cid))
    _rollback_signature(tid)
    reload_wiring()
    _resolve_one(cid)
    return ("signature %s did not fire (a needed value may be missing, or the roles don't "
            "assemble), so it was rolled back." % sig)


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


def _note_block(clue_id, raw_list):
    """The user-facing comment for a clue, editable inline right below the summary. Shows the
    saved comment (also editable in the hand-solver — same wfw_notes store) and a compact
    box to type/change it. Posts to /cluecomment, which re-renders from store (no re-solve)."""
    conn = store.connect()
    try:
        note = store.get_note(conn, clue_id)
    finally:
        conn.close()
    saved = ""
    if note:
        saved = ('<div style="margin:.5rem 0 .35rem;padding:.6rem .85rem;background:#fffbeb;'
                 'border:1px solid #fcd34d;border-radius:10px;color:#92600a;font-size:.95rem">'
                 '<strong>Comment:</strong> %s</div>' % escape(note).replace("\n", "<br>"))
    h = _hidden(raw_list, clue_id)
    clear_btn = ""
    if note:
        clear_btn = ('<button name="clear" value="1" style="margin-left:.4rem;background:#fff;'
                     'color:#64748b;border:1px solid #cbd5e1;border-radius:8px;padding:.3rem .7rem;'
                     'font-weight:600;cursor:pointer">Clear</button>')
    editor = ('<form method="post" action="/cluecomment" style="margin:.15rem 0 .6rem">%s'
              '<textarea name="note" rows="2" placeholder="Add a brief comment&hellip;" '
              'style="width:100%%;max-width:46rem;box-sizing:border-box;border:1px solid #cbd5e1;'
              'border-radius:8px;padding:.4rem .55rem;font-family:inherit;font-size:.92rem">%s'
              '</textarea>'
              '<div style="margin-top:.25rem"><button style="background:#0d9488;color:#fff;'
              'border:none;border-radius:8px;padding:.3rem .8rem;font-weight:700;cursor:pointer">'
              'Save comment</button>%s</div></form>'
              % (h, escape(note), clear_btn))
    return saved + editor


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
            elif role == "substitution":          # abbr/symbol -> the WORDPLAY table, not synonyms
                val = (a.get("value") or "").strip().upper()
                if val:
                    admin_db.add_substitution(phrase, val)
                    reload_wiring()               # substitutions load as abbreviations at build
                    applied.append("%r=%s (substitution)" % (phrase, val))
            elif role == "letters":               # LITERAL piece — per-clue only, NO DB write
                val = (a.get("value") or "").strip().upper()
                if val:
                    applied.append("%r=%s (letters)" % (phrase, val))
            elif role == "selection":             # SELECTION piece — per-clue only, NO DB write
                val = (a.get("value") or "").strip().upper()
                rule = (a.get("rule") or "").strip()
                if val and val in _selection_candidates(phrase, rule):
                    applied.append("%r=%s (selection/%s)" % (phrase, val, rule))
                elif val:
                    applied.append("%r=%s (selection/%s — NOT what the rule derives, ignored)"
                                   % (phrase, val, rule))
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


def _json(obj):
    import json as _j
    return app.response_class(_j.dumps(obj), mimetype="application/json")


# --- TRIAGE review page ------------------------------------------------------------------
# Self-contained review surface for a nightly triage report. Renders each FAIL/PENDING clue's
# diagnosis with EDITABLE data suggestions you mark Accept/Reject; then ONE "Apply queued"
# button writes every marked add to the reference DB (via the dashboard's adders), reloads the
# wiring ONCE, and re-solves each affected clue ONCE — the human's Apply IS the commit. After a
# re-solve, a clue that passes with clue words still unaccounted is flagged (read from the
# parse; no engine is touched). Missing-signature/engine/unparsed clues are shown read-only.
# Touches none of the existing dashboard pages.

_TRIAGE_CSS = """<style>
.tr-wrap{font-family:system-ui;max-width:54rem}
.tr-clue{border:1px solid #e2e8f0;border-radius:10px;padding:.7rem .9rem;margin:.7rem 0}
.tr-h{font-size:1.05rem}
.tr-st{font-size:.72rem;font-weight:800;padding:.1rem .45rem;border-radius:6px;margin-left:.4rem;color:#fff}
.tr-fail{background:#dc2626}.tr-pending{background:#d97706}.tr-pass{background:#16a34a}
.tr-clue-text{color:#334155;margin:.25rem 0 .35rem}
.tr-gap{font-size:.9rem;color:#475569;background:#f8fafc;border-left:3px solid #cbd5e1;padding:.35rem .55rem;margin:.3rem 0}
.tr-enr{display:flex;align-items:center;gap:.45rem;margin:.35rem 0;font-size:.92rem;flex-wrap:wrap}
.tr-tag{font-size:.66rem;font-weight:800;color:#fff;background:#64748b;border-radius:5px;padding:.12rem .4rem;text-transform:uppercase}
.tr-in{font-family:inherit;font-size:.9rem;padding:.15rem .35rem;border:1px solid #cbd5e1;border-radius:5px}
.tr-act{display:inline-flex;gap:.7rem;align-items:center;font-size:.85rem;margin-left:.3rem}
.tr-code{font-family:monospace;background:#f1f5f9;padding:.1rem .4rem;border-radius:5px}
.tr-done{color:#166534}.tr-rej{color:#94a3b8}
.tr-esc{font-size:.88rem;color:#6d28d9;background:#f5f3ff;border-left:3px solid #a78bfa;padding:.35rem .55rem;margin:.3rem 0}
.tr-applybar{position:sticky;bottom:0;background:#fff;border-top:2px solid #e2e8f0;padding:.7rem 0;margin-top:1.2rem}
.tr-apply-btn{background:#0d9488;color:#fff;border:none;border-radius:9px;padding:.5rem 1.2rem;font-weight:800;cursor:pointer;font-size:1rem}
.tr-cat{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:.4rem .6rem;margin:.35rem 0}
.tr-catlabel{font-size:.72rem;font-weight:800;color:#64748b;text-transform:uppercase;letter-spacing:.04em}
.tr-badge{font-size:.68rem;font-weight:800;color:#fff;border-radius:5px;padding:.12rem .45rem;text-transform:uppercase}
.tr-ev{margin:.35rem 0 .1rem;padding-left:1.1rem;font-size:.9rem;color:#334155}
.tr-ev li{margin:.1rem 0}
.tr-hswrap{margin-top:.35rem;font-size:.9rem}
.tr-hs{display:inline-block;background:#7c3aed;color:#fff;text-decoration:none;font-weight:700;border-radius:7px;padding:.25rem .7rem}
.tr-hsmuted .tr-hs{background:#fff;color:#7c3aed;border:1px solid #c4b5fd;font-weight:600;padding:.15rem .55rem;font-size:.82rem}
.tr-vlabel{font-size:.75rem;font-weight:800;color:#9a3412;text-transform:uppercase;letter-spacing:.03em;margin:.55rem 0 .1rem}
.tr-vlabel span{font-weight:500;text-transform:none;color:#a16207;letter-spacing:0}
.tr-reclass{background:#475569;color:#fff;border:none;border-radius:8px;padding:.35rem .8rem;font-weight:700;cursor:pointer}
.tr-read{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:.5rem .7rem;margin:.4rem 0;
         font-size:.98rem;color:#111827;line-height:1.5}
.tr-read .tr-catlabel{color:#92400e}
.tr-read-note{margin-top:.35rem;color:#334155;font-size:.92rem}
</style>"""

# deterministic-classifier category -> (label, colour) for the badge
_CAT_META = {
    "missing_data": ("missing data", "#b45309"),
    "missing_signature": ("missing signature", "#6d28d9"),
    "missing_engine": ("missing engine", "#b91c1c"),
    "solves_now": ("should solve now", "#166534"),
}


def _classification_block(cid, cls):
    """Render the DETERMINISTIC diagnosis for one clue: category badge(s) + mechanical evidence
    (from core.triage_classify) + the bridge to the hand-solver. No value guesses here — those
    are a separate, labelled block."""
    from urllib.parse import quote
    reasons = cls.get("reasons") or []
    detail = cls.get("detail") or {}
    out = ['<div class="tr-cat"><span class="tr-catlabel">Diagnosis (mechanical):</span> ']
    for r in reasons:
        lab, col = _CAT_META.get(r, (r.replace("_", " "), "#64748b"))
        out.append('<span class="tr-badge" style="background:%s">%s</span> '
                   % (col, escape(lab)))
    ev = []
    if detail.get("no_definition"):
        ev.append("no confirmed definition")
    if detail.get("unresolved_words"):
        ev.append("no known value for: <b>%s</b>"
                  % escape(", ".join(detail["unresolved_words"])))
    for item in (detail.get("novel_shapes") or []):
        sig, tier = item[0], item[1]
        ev.append("new shape needed: <span class=\"tr-code\">%s</span> (tier: <b>%s</b>)"
                  % (escape(sig), escape(tier)))
    if detail.get("double_definition"):
        ev.append("reads as a double definition")
    if detail.get("possible_double_definition"):
        ev.append("may be a double definition missing its other definition (%s)"
                  % escape(str(detail["possible_double_definition"])))
    if detail.get("acrostic"):
        ev.append("initial letters spell the answer (acrostic)")
    if detail.get("hidden"):
        ev.append("answer sits inside the clue letters (hidden)")
    if detail.get("homophone") or detail.get("homophone_entry_maybe_missing"):
        ev.append("homophone involved — a sounds-like entry may be missing")
    if detail.get("material_present_no_assembly"):
        ev.append("pieces present but nothing known assembles them")
    if ev:
        out.append('<ul class="tr-ev">' + "".join("<li>%s</li>" % e for e in ev) + "</ul>")
    hs = ('<a class="tr-hs" href="/hs?id=%d&amp;from=%s">Open in hand-solver &rarr;</a>'
          % (cid, quote(str(cid))))
    if "missing_signature" in reasons:
        out.append('<div class="tr-hswrap">Fix the shape by hand: %s</div>' % hs)
    else:
        out.append('<div class="tr-hswrap tr-hsmuted">%s</div>' % hs)
    out.append("</div>")
    return "".join(out)

_TRIAGE_ITYPES = ["anagram", "deletion", "selection", "reversal", "container", "hidden",
                  "homophone", "charade_positional", "letter_shift", "acrostic"]


def _enr_label(enr):
    t = enr.get("type", "")
    if t == "synonym":
        return "synonym &nbsp; <b>%s</b> = %s" % (escape(enr.get("word", "")),
                                                  escape(enr.get("value", "")))
    if t == "substitution":
        return "abbreviation &nbsp; <b>%s</b> &rarr; %s" % (escape(enr.get("word", "")),
                                                            escape(enr.get("value", "")))
    if t == "definition":
        return "definition &nbsp; <b>%s</b> &rarr; %s" % (escape(enr.get("definition", "")),
                                                          escape(enr.get("answer", "")))
    if t == "indicator":
        sub = ("/" + enr["subtype"]) if enr.get("subtype") else ""
        return "indicator &nbsp; <b>%s</b> = %s%s" % (escape(enr.get("word", "")),
                                                      escape(enr.get("indicator_type", "")),
                                                      escape(sub))
    return escape(str(enr))


def _enr_type_name(t):
    return {"synonym": "synonym", "substitution": "abbreviation",
            "definition": "definition", "indicator": "indicator"}.get(t, t)


def _enr_fields(i, enr):
    """EDITABLE inputs for enrichment i, so a suggestion can be corrected before Accept
    (e.g. an indicator 'cover off' -> 'tear cover off'). Field names carry the row index."""
    t = enr.get("type", "")
    h = '<input type="hidden" name="type_%d" value="%s">' % (i, escape(t, quote=True))

    def v(x):
        return escape(x or "", quote=True)

    if t in ("synonym", "substitution"):
        return h + ('<input class="tr-in" name="word_%d" value="%s" size="16"> = '
                    '<input class="tr-in" name="value_%d" value="%s" size="10">'
                    % (i, v(enr.get("word")), i, v(enr.get("value"))))
    if t == "definition":
        return h + ('<input class="tr-in" name="def_%d" value="%s" size="20"> &rarr; '
                    '<input class="tr-in" name="ans_%d" value="%s" size="12">'
                    % (i, v(enr.get("definition")), i, v(enr.get("answer"))))
    if t == "indicator":
        opts = "".join('<option%s>%s</option>'
                       % (" selected" if o == enr.get("indicator_type") else "", o)
                       for o in _TRIAGE_ITYPES)
        return h + ('<input class="tr-in" name="word_%d" value="%s" size="16"> = '
                    '<select class="tr-in" name="itype_%d">%s</select> '
                    'sub <input class="tr-in" name="sub_%d" value="%s" size="7">'
                    % (i, v(enr.get("word")), i, opts, i, v(enr.get("subtype"))))
    return h


def _triage_surface(src, pnum, notice=""):
    from core import triage
    meta, clues = triage.collect_puzzle(src, pnum)
    diagnoses = triage.load_diagnoses(pnum)          # the human/AI value suggestions (labelled)
    classified = triage.load_classified(pnum)         # the DETERMINISTIC categories (cached)
    rejected = triage.rejected_keys(pnum)
    review = [c for c in clues if c["status"] in ("fail", "pending")]

    reclass = ('<form method="post" action="/triageclassify" style="display:inline">'
               '<input type="hidden" name="src" value="%s">'
               '<input type="hidden" name="pnum" value="%s">'
               '<button class="tr-reclass">Run / refresh diagnosis</button></form>'
               % (escape(src, quote=True), escape(str(pnum), quote=True)))

    # Signature-regression status + button: publish-now / regression-check-later. The
    # button starts a BACKGROUND batch A/B over every pending-tier signature; each is
    # promoted only on a clean diff. Status comes from the job's file.
    sig_st = _sigreg_read()
    if sig_st.get("running"):
        sig_line = ('<div class="wfw-notice" style="background:#fffbeb">Signature '
                    'regression RUNNING (started %s): %s — refresh for progress.</div>'
                    % (escape(sig_st.get("started", "")),
                       escape(sig_st.get("phase", ""))))
    elif sig_st.get("results"):
        sig_line = ('<div class="wfw-notice">Last signature regression: %s</div>'
                    % escape("; ".join("template %s → %s (%s)"
                                       % (r["template"], r["action"], r["summary"][:80])
                                       for r in sig_st["results"])))
    else:
        sig_line = ""

    p = [_TRIAGE_CSS,
         '<form method="post" action="/triageapply"><div class="tr-wrap">',
         '<input type="hidden" name="src" value="%s">' % escape(src, quote=True),
         '<input type="hidden" name="pnum" value="%s">' % escape(str(pnum), quote=True),
         '<h2>Triage &mdash; %s %s</h2>' % (escape(src.title()), escape(str(pnum))),
         '<p><b>%d</b> pass &middot; <b>%d</b> pending &middot; <b>%d</b> fail &nbsp;'
         '(%d unsolved to review). %s '
         '<button class="tr-reclass" style="background:#7c3aed" formaction="/sigregress" '
         'title="Background batch: A/B-check every pending signature (two full solves '
         'each) and promote the clean ones — run it after publishing, whenever suits.">'
         'Regression-check pending signatures</button></p>'
         % (meta["pass"], meta["pending"], meta["fail"], len(review), reclass),
         sig_line]
    if not classified:
        p.append('<div class="wfw-notice">No diagnosis yet — click <b>Run / refresh diagnosis</b>. '
                 'It runs the fixed mechanical checks (the same category every time) and re-solves '
                 'any clue whose pieces are already all present.</div>')
    if notice:
        p.append('<div class="wfw-notice">%s</div>' % escape(notice))

    i = 0
    for c in review:
        d = diagnoses.get(str(c["id"]), {})
        cls = classified.get(str(c["id"]))
        p.append('<div class="tr-clue" id="clue-%d">' % c["id"])
        p.append('<div class="tr-h"><b>%s %s</b> &mdash; %s'
                 '<span class="tr-st tr-%s">%s</span> '
                 '<button class="tr-reclass" style="padding:.15rem .55rem;font-size:.78rem" '
                 'formaction="/triagererun" name="rerun_cid" value="%d" '
                 'title="Re-solve just this clue now (a couple of seconds)">Re-run</button>'
                 '</div>'
                 % (escape(str(c["number"])), escape(c["direction"]), escape(c["answer"]),
                    c["status"], c["status"].upper(), c["id"]))
        p.append('<div class="tr-clue-text">%s</div>' % escape(c["clue_text"]))

        # 1) the DETERMINISTIC diagnosis (category + mechanical evidence + hand-solver bridge)
        if cls:
            p.append(_classification_block(c["id"], cls))

        # 1b) the PROPOSED READING from the diagnoses file (Claude, labelled unverified).
        # COMPACT: three short lines (Reading / Problem / Fix); the technical trace
        # (gaps/signature/note) collapses behind a details toggle. Entries without the
        # compact fields fall back to rendering the long text directly.
        if any(d.get(k) for k in ("reading", "problem", "action", "gaps", "note", "signature")):
            bits, deep = [], []
            compact = bool(d.get("reading") or d.get("problem") or d.get("action"))
            for key, lab in (("reading", "Reading"), ("problem", "Problem"), ("action", "Fix")):
                if d.get(key):
                    bits.append('<div style="margin-top:.25rem"><b>%s:</b> %s</div>'
                                % (lab, escape(d[key])))
            sig = d.get("signature") or {}
            sig_html = ('signature needed: <span class="tr-code">%s</span> (tier: <b>%s</b>)%s'
                        % (escape(sig.get("shape", "")), escape(sig.get("tier", "")),
                           (" &mdash; " + escape(sig["note"])) if sig.get("note") else "")
                        ) if sig else ""
            for extra in (escape(d["gaps"]) if d.get("gaps") else "", sig_html,
                          escape(d["note"]) if d.get("note") else ""):
                if extra:
                    (deep if compact else bits).append(
                        extra if compact else '<div style="margin-top:.3rem">%s</div>' % extra)
            if deep:
                bits.append('<details style="margin-top:.3rem"><summary style="cursor:pointer;'
                            'color:#92400e;font-size:.85rem">technical detail</summary>'
                            '<div class="tr-read-note">%s</div></details>'
                            % "<br><br>".join(deep))
            p.append('<div class="tr-read"><span class="tr-catlabel">Proposed reading '
                     '(Claude &mdash; unverified):</span>%s</div>' % "".join(bits))

        # 2) the LABELLED value suggestions (human/AI, not mechanical) — editable + Accept/Reject
        enrs = d.get("enrichments") or []
        if enrs:
            p.append('<div class="tr-vlabel">Suggested values '
                     '<span>(human/AI &mdash; not mechanical; edit before accepting)</span></div>')
        for enr in enrs:
            key = triage.enrichment_key(c["id"], enr)
            if triage.enrichment_present(enr):
                p.append('<div class="tr-enr tr-done">&#10003; in DB: %s</div>' % _enr_label(enr))
                continue
            row_h = ('<input type="hidden" name="cid_%d" value="%d">'
                     '<input type="hidden" name="key_%d" value="%s">'
                     % (i, c["id"], i, escape(key, quote=True)))
            if key in rejected:
                p.append('<div class="tr-enr tr-rej"><span class="tr-tag">%s</span> '
                         '&#10007; rejected: %s %s'
                         '<label><input type="radio" name="act_%d" value="unreject"> '
                         'un-reject</label></div>'
                         % (escape(_enr_type_name(enr.get("type", ""))), _enr_label(enr),
                            row_h, i))
            else:
                p.append('<div class="tr-enr"><span class="tr-tag">%s</span> %s %s'
                         '<span class="tr-act">'
                         '<label><input type="radio" name="act_%d" value="accept"> Accept</label>'
                         '<label><input type="radio" name="act_%d" value="reject"> Reject</label>'
                         '</span></div>'
                         % (escape(_enr_type_name(enr.get("type", ""))), _enr_fields(i, enr),
                            row_h, i, i))
            i += 1
        # DELETE facility: prune a SPURIOUS reference row that blocks this clue's solve
        # (synonym/abbr/indicator/definition/link). Queued like an Accept; Apply deletes
        # (recoverable in deleted_entries) and re-solves the clue.
        p.append('<div class="tr-enr" style="border-top:1px dashed #e2e8f0;padding-top:.35rem">'
                 '<span class="tr-tag" style="background:#b91c1c">delete</span> '
                 '<input type="hidden" name="cid_%d" value="%d">'
                 '<select class="tr-in" name="delkind_%d">'
                 '<option value="synonym">synonym</option>'
                 '<option value="abbreviation">abbreviation</option>'
                 '<option value="indicator">indicator</option>'
                 '<option value="definition">definition</option>'
                 '<option value="link">link word</option></select> '
                 '<input class="tr-in" name="delword_%d" placeholder="word / phrase" size="14"> '
                 '<input class="tr-in" name="delval_%d" placeholder="value / type[/sub] / answer" '
                 'size="16"> '
                 '<span class="tr-act"><label><input type="radio" name="act_%d" value="delete"> '
                 'Delete on Apply</label></span></div>'
                 % (i, c["id"], i, i, i, i))
        i += 1
        p.append('</div>')

    p.append('<input type="hidden" name="n" value="%d">' % i)
    p.append('<div class="tr-applybar"><button class="tr-apply-btn" type="submit">'
             'Apply queued &amp; re-solve</button></div>')
    p.append('</div></form>')
    return "".join(p)


def _clue_unexplained(cid):
    """(status, [unaccounted words]) for a clue as it now stands — read from the stored parse.
    A PASS that still leaves clue words unaccounted is an unsound pass worth flagging."""
    row = _load_clue(cid)
    if row is None:
        return (None, [])
    ct, ans, _s, _pn, direction, enum, _cn = row
    ans = enum_space(ans, enum)
    ctx = build_wfw_atom_context(ct, ans, direction=direction)
    conn = store.connect()
    try:
        parse = store.load_parse(conn, cid)
    finally:
        conn.close()
    if parse is None:
        return (None, [])
    try:
        uw = list(parse.unexplained_words(ctx))
    except Exception:
        uw = []
    return (parse.status, uw)


def _sigreg_path():
    from core import triage
    return os.path.join(triage.TRIAGE_DIR, "sigregress_status.json")


def _sigreg_read():
    import json
    try:
        with open(_sigreg_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _sigreg_write(d):
    import json
    os.makedirs(os.path.dirname(_sigreg_path()), exist_ok=True)
    with open(_sigreg_path(), "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)


_sigreg_lock = None


def _sigregress_job(rows):
    """BACKGROUND job: A/B-check every active pending-tier signature (two fresh solves
    each, ~20-30 min per signature) and PROMOTE each one whose diff is clean — the
    'publish now, regression-check the day's signatures later' flow. Progress + results
    land in the status file the triage page displays."""
    import re as _re
    from core import ab_signature
    results = []
    for n, (tid, notes) in enumerate(rows, 1):
        trig = [int(x) for x in _re.findall(r"clue (\d+)", notes or "")]

        def prog(phase, d, t, _n=n, _tid=tid):
            st = _sigreg_read()
            st["phase"] = ("template %d (%d of %d): %s %d/%d"
                           % (_tid, _n, len(rows), phase, d, t))
            _sigreg_write(st)

        try:
            action, summary, _ab = ab_signature.try_promote(tid, trigger_clue_ids=trig,
                                                            progress=prog)
        except Exception as e:
            action, summary = "error", "%s: %s" % (type(e).__name__, e)
        if action == "promote":
            try:
                reload_wiring()                   # the template now loads PASS-tier
                for c in trig:
                    _resolve_one(c)               # its clue(s) go green
            except Exception:
                pass
        results.append({"template": tid, "action": action, "summary": summary})
        st = _sigreg_read()
        st.update({"done": n, "results": results})
        _sigreg_write(st)
    st = _sigreg_read()
    st.update({"running": False, "phase": "finished"})
    _sigreg_write(st)


@app.route("/sigregress", methods=["POST"])
def sigregress_route():
    """START the batch signature regression (background thread). The user's click is the
    trigger; each pending signature is promoted ONLY on a clean A/B verdict."""
    import threading
    from urllib.parse import quote
    from datetime import datetime
    global _sigreg_lock
    if _sigreg_lock is None:
        _sigreg_lock = threading.Lock()
    src = (request.form.get("src") or "").strip()
    pnum = (request.form.get("pnum") or "").strip()
    with _sigreg_lock:
        st = _sigreg_read()
        if st.get("running"):
            notice = "Signature regression already running — refresh for progress."
        else:
            import sqlite3
            from core.ab_signature import _CLUES_DB
            con = sqlite3.connect(_CLUES_DB)
            try:
                rows = con.execute("SELECT id, COALESCE(notes,'') FROM catalog_templates "
                                   "WHERE tier='pending' AND active=1").fetchall()
            finally:
                con.close()
            if not rows:
                notice = "No pending signatures to check."
            else:
                _sigreg_write({"running": True,
                               "started": datetime.now().isoformat(timespec="seconds"),
                               "total": len(rows), "done": 0, "phase": "starting",
                               "results": []})
                threading.Thread(target=_sigregress_job, args=(rows,),
                                 daemon=True).start()
                notice = ("Signature regression started on %d pending signature(s) — "
                          "two full solves each; refresh this page for progress."
                          % len(rows))
    if (request.form.get("back") or "").strip() == "hs":
        return redirect("/hs?src=%s&pnum=%s&notice=%s"
                        % (quote(src), quote(str(pnum)), quote(notice)))
    return redirect("/triage?src=%s&pnum=%s&notice=%s"
                    % (quote(src), quote(str(pnum)), quote(notice)))


@app.route("/triagererun", methods=["POST"])
def triagererun_route():
    """Re-solve ONE clue from the triage page (the user's click; nothing else applied).
    Uses the resident wiring — a couple of seconds, no snapshot rebuild."""
    from urllib.parse import quote
    src = (request.form.get("src") or "").strip()
    pnum = (request.form.get("pnum") or "").strip()
    cid = (request.form.get("rerun_cid") or "").strip()
    if not cid.isdigit():
        return redirect("/triage?src=%s&pnum=%s" % (quote(src), quote(str(pnum))))
    _resolve_one(int(cid))
    conn = store.connect()
    try:
        cp = store.load_parse(conn, int(cid))
    finally:
        conn.close()
    st = cp.status if cp is not None else "unknown"
    notice = "Re-ran clue %s: %s." % (cid, st.upper())
    return redirect("/triage?src=%s&pnum=%s&notice=%s&scroll=%s"
                    % (quote(src), quote(str(pnum)), quote(notice), cid))


@app.route("/triage")
def triage_route():
    src = (request.args.get("src") or "telegraph").strip()
    pnum = (request.args.get("pnum") or "").strip()
    notice = (request.args.get("notice") or "").strip()
    scroll = (request.args.get("scroll") or "").strip()
    if not pnum:
        return _page('<p style="font-family:system-ui">Usage: '
                     '<a href="/triage?src=telegraph&amp;pnum=31284">'
                     '/triage?src=telegraph&amp;pnum=31284</a></p>')
    return _page(_triage_surface(src, pnum, notice), scroll_to=scroll or None)


@app.route("/triageclassify", methods=["POST"])
def triageclassify_route():
    """Run the DETERMINISTIC classifier over every fail/pending clue and cache the result. Also
    re-solve any clue the classifier marks 'solves_now' (its pieces, shape and definition are all
    present) — a single re-solve to establish current truth (green -> drops off; still failing ->
    the rare genuine oddity). Slow-ish (discover per clue), so it's an explicit action."""
    from core import triage, triage_classify
    src = (request.form.get("src") or "").strip()
    pnum = (request.form.get("pnum") or "").strip()
    if not pnum:
        return redirect("/triage")
    _meta, clues = triage.collect_puzzle(src, pnum)
    review = [c for c in clues if c["status"] in ("fail", "pending")]
    wiring = batch_wiring()
    out, resolved = {}, 0
    for c in review:
        try:
            cls = triage_classify.classify(c["clue_text"], c["answer"], wiring)
        except Exception as e:
            cls = {"reasons": ["missing_engine"], "detail": {"error": str(e)}}
        out[str(c["id"])] = cls
        if "solves_now" in cls.get("reasons", []):
            _resolve_one(c["id"])           # single re-solve to establish current truth
            resolved += 1
    triage.save_classified(pnum, out)
    from urllib.parse import quote
    notice = ("Diagnosis refreshed: %d clue(s) classified; re-solved %d that should already solve."
              % (len(out), resolved))
    return redirect("/triage?src=%s&pnum=%s&notice=%s"
                    % (quote(src), quote(str(pnum)), quote(notice)))


@app.route("/triageapply", methods=["POST"])
def triageapply_route():
    """Apply the QUEUED enrichments in ONE pass: write each marked add (with your edits) to the
    reference DB, reload the wiring ONCE, then re-solve each affected clue ONCE. The human's
    Apply is the commit. Flags any clue that now passes with words still unaccounted."""
    from core import triage
    src = (request.form.get("src") or "").strip()
    pnum = (request.form.get("pnum") or "").strip()
    try:
        n = int(request.form.get("n") or "0")
    except ValueError:
        n = 0
    added, present, errors, deleted = [], [], [], []
    n_rej = n_unrej = 0
    affected, need_reload = set(), False
    for i in range(n):
        act = request.form.get("act_%d" % i)
        if not act:
            continue
        cid = (request.form.get("cid_%d" % i) or "").strip()
        key = (request.form.get("key_%d" % i) or "").strip()
        t = (request.form.get("type_%d" % i) or "").strip()
        if act == "delete":
            # prune a spurious reference row (recoverable in deleted_entries), then the
            # clue re-solves in the same pass as the accepts.
            dkind = (request.form.get("delkind_%d" % i) or "").strip()
            dword = (request.form.get("delword_%d" % i) or "").strip()
            dval = (request.form.get("delval_%d" % i) or "").strip()
            if not dword:
                continue
            if dkind == "synonym":
                dm = admin_db.delete_synonym(dword, dval)
                apply_add_to_wiring({"kind": "synonym", "word": dword, "synonym": dval})
            elif dkind == "abbreviation":
                dm = admin_db.delete_substitution(dword, dval)
                need_reload = True
            elif dkind == "indicator":
                wp, _, sub = dval.partition("/")
                dm = admin_db.delete_indicator(dword, wp, sub or None)
                apply_add_to_wiring({"kind": "indicator", "word": dword, "type": wp})
            elif dkind == "definition":
                dm = admin_db.delete_definition(dword, dval)
                apply_add_to_wiring({"kind": "definition", "definition": dword,
                                     "answer": dval})
            elif dkind == "link":
                dm = admin_db.delete_link(dword)
                apply_add_to_wiring({"kind": "link", "word": dword})
            else:
                continue
            deleted.append(str(dm))
            if cid.isdigit():
                affected.add(int(cid))
            continue
        if act == "reject":
            if key:
                triage.mark_rejected(pnum, key)
                n_rej += 1
            continue
        if act == "unreject":
            if key:
                triage.mark_rejected(pnum, key, undo=True)
                n_unrej += 1
            continue
        if act != "accept":
            continue
        if t in ("synonym", "substitution"):
            enr = {"type": t, "word": (request.form.get("word_%d" % i) or "").strip(),
                   "value": (request.form.get("value_%d" % i) or "").strip()}
        elif t == "definition":
            enr = {"type": t, "definition": (request.form.get("def_%d" % i) or "").strip(),
                   "answer": (request.form.get("ans_%d" % i) or "").strip()}
        elif t == "indicator":
            enr = {"type": t, "word": (request.form.get("word_%d" % i) or "").strip(),
                   "indicator_type": (request.form.get("itype_%d" % i) or "").strip(),
                   "subtype": (request.form.get("sub_%d" % i) or "").strip() or None}
        else:
            continue
        m = str(triage.apply_enrichment(enr))
        if m.startswith("Added"):
            added.append(m)
            if t == "substitution":
                need_reload = True
            else:
                apply_add_to_wiring(triage.wiring_form(enr))
        elif m.startswith("Already"):
            present.append(m)
        else:
            errors.append(m)
        if cid.isdigit():
            affected.add(int(cid))

    if need_reload:
        reload_wiring()

    results = {}
    for cid in sorted(affected):
        _resolve_one(cid)
        results[cid] = _clue_unexplained(cid)     # (status, [unaccounted words])
    now_pass = [cid for cid in sorted(affected) if results[cid][0] == "pass"]
    warns = ["clue %d now PASSES but these words are unaccounted: %s"
             % (cid, ", ".join(results[cid][1]))
             for cid in sorted(affected) if results[cid][0] == "pass" and results[cid][1]]
    scroll = next((str(cid) for cid in sorted(affected)
                   if results[cid][0] in ("fail", "pending")), "")

    bits = []
    if added:
        bits.append("added %d" % len(added))
    if deleted:
        bits.append("deleted: " + " | ".join(deleted))
    if present:
        bits.append("%d already present" % len(present))
    if n_rej:
        bits.append("rejected %d" % n_rej)
    if n_unrej:
        bits.append("un-rejected %d" % n_unrej)
    if now_pass:
        bits.append("now passing: %s" % ", ".join(str(x) for x in now_pass))
    if errors:
        bits.append("not added: %s" % "; ".join(errors))
    notice = ("; ".join(bits) if bits else "Nothing queued.")
    if warns:
        notice += " — WARNING: " + " | ".join(warns)

    from urllib.parse import quote
    return redirect("/triage?src=%s&pnum=%s&notice=%s%s"
                    % (quote(src), quote(str(pnum)), quote(notice),
                       ("&scroll=%s" % scroll) if scroll else ""))


def _apply_db_adds(db_adds):
    """Write reusable pieces to the reference DB and fold each new row into the resident
    wiring. Items: ("synonym", word, value) / ("definition", phrase, answer) /
    ("indicator", phrase, type, subtype) / ("substitution", word, value). Dedup is built
    into each adder, so repeats are harmless. Shared by the manual-solve commit and the
    save-pieces-only path, so both write identically. Returns (added, present, rejected)
    message lists."""
    added, present, rejected, need_reload = [], [], [], False
    for item in db_adds:
        kind = item[0]
        try:
            if kind == "synonym":
                m = admin_db.add_synonym(item[1], item[2])
            elif kind == "definition":
                m = admin_db.add_definition(item[1], item[2])
            elif kind == "indicator":
                m = admin_db.add_indicator(item[1], item[2], item[3])
            elif kind == "substitution":
                m = admin_db.add_substitution(item[1], item[2])
            else:
                continue
        except Exception as e:
            m = "error adding %r: %s" % (item[1], e)
        m = str(m)
        if m.startswith("Added"):
            added.append(m)
            if kind == "synonym":
                apply_add_to_wiring({"kind": "synonym", "word": item[1], "synonym": item[2]})
            elif kind == "definition":
                apply_add_to_wiring({"kind": "definition", "definition": item[1],
                                     "answer": item[2]})
            elif kind == "indicator":
                apply_add_to_wiring({"kind": "indicator", "word": item[1], "type": item[2]})
            elif kind == "substitution":
                need_reload = True     # substitutions load as abbreviations at build
        elif m.startswith("Already"):
            present.append(m)
        else:
            rejected.append(m)
    if need_reload:
        reload_wiring()               # substitutions need a rebuild to go live
    return added, present, rejected


@app.route("/hssavepieces", methods=["POST"])
def hssavepieces_route():
    """SAVE PIECES ONLY (no manual solve): write the grid's reusable pieces to the reference
    DB, then re-solve the clue through the cascade — so the ENGINE stays the solver of
    record. For data-gap clues: one click adds the missing synonym/definition/abbreviation/
    indicator and shows whether the solver now passes on its own. Nothing is frozen, no
    status is set; a clue that still fails simply still fails. JSON."""
    import json
    only = (request.form.get("only") or "").strip()
    payload = (request.form.get("payload") or "").strip()
    if not only.isdigit():
        return _json({"ok": False, "msg": "No clue."})
    cid = int(only)
    row = _load_clue(cid)
    if row is None:
        return _json({"ok": False, "msg": "No clue."})
    clue_text, answer, src, pnum, direction, enumeration, cnum = row
    answer = enum_space(answer, enumeration)
    ctx = build_wfw_atom_context(clue_text, answer, direction=direction)
    wt = _hs_word_units(ctx)
    ans_letters = "".join(c for c in answer.upper() if c.isalpha())
    try:
        assigns = json.loads(payload) if payload else []
    except Exception:
        assigns = []

    # The SAME reusable-piece rules as the manual commit (synonym / substitution with a
    # value; definition; indicator with a REAL chosen type) — link/letters/anagram fodder/
    # deletion/selection/filler are per-clue and never saved.
    db_adds = []
    for a in assigns:
        try:
            idx = sorted(int(i) for i in a.get("idx", []) if 0 <= int(i) < len(wt))
        except Exception:
            idx = []
        if not idx:
            continue
        role = (a.get("role") or "").strip()
        phrase = " ".join(wt[i].text for i in idx)
        value = (a.get("value") or "").strip().upper()
        if role == "synonym" and value:
            db_adds.append(("synonym", phrase, value))
        elif role == "substitution" and value:
            db_adds.append(("substitution", phrase, value))
        elif role == "definition":
            db_adds.append(("definition", phrase, ans_letters))
        elif role == "indicator":
            it = (a.get("itype") or "").split(":")[0]
            if it:
                db_adds.append(("indicator", phrase, it, (a.get("isub") or "").strip() or None))
    if not db_adds:
        return _json({"ok": False, "msg": "Nothing to save — assign a synonym/abbreviation/"
                      "definition/indicator first (link/letters/filler are per-clue, "
                      "never saved)."})

    added, present, rejected = _apply_db_adds(db_adds)
    _resolve_one(cid)
    conn = store.connect()
    try:
        sp = store.load_parse(conn, cid)
    finally:
        conn.close()
    status = (sp.status if sp is not None else "fail") or "fail"
    msg = "Saved: %d new, %d already in DB." % (len(added), len(present))
    if rejected:
        msg += " Not saved: %s." % "; ".join(rejected)
    msg += " Re-solved: %s%s." % (status.upper(),
                                  " — the engine solves it" if status == "pass" else "")
    return _json({"ok": True, "msg": msg, "status": status})


@app.route("/hsmanualcommit", methods=["POST"])
def hsmanualcommit_route():
    """MANUAL SOLVE commit from the /hs word grid. Builds a FROZEN manual Parse from the
    assignments: each synonym/letters PIECE is placed on the exact answer TILES the human
    clicked (so reversal / container work — REM on tiles 1-3, EG on tiles 4-5 for MERGE),
    and indicator / definition / link / filler are roles with no tiles. NO reference-DB write,
    NO cascade, NO auto-verification — a recorder, the opposite of the banned builder. JSON."""
    import json
    only = (request.form.get("only") or "").strip()
    payload = (request.form.get("payload") or "").strip()
    if not only.isdigit():
        return _json({"ok": False, "msg": "No clue."})
    cid = int(only)
    row = _load_clue(cid)
    if row is None:
        return _json({"ok": False, "msg": "No clue."})
    clue_text, answer, src, pnum, direction, enumeration, cnum = row
    answer = enum_space(answer, enumeration)
    ctx = build_wfw_atom_context(clue_text, answer, direction=direction)
    wt = _hs_word_units(ctx)                       # hyphenated words split (line-up -> line + up)
                                                   # so payload word-indices align with the /hs grid
    ans_letters = "".join(c for c in answer.upper() if c.isalpha())
    N = len(ans_letters)
    try:
        assigns = json.loads(payload) if payload else []
    except Exception:
        assigns = []

    from core.wfw_model import Source, Link, Annotation, Parse

    def atoms_for(idx):
        out = []
        for i in idx:
            if 0 <= i < len(wt):
                out.extend(wt[i].atom_ids)
        return tuple(out)

    def phrase_for(idx):
        return " ".join(wt[i].text for i in idx if 0 <= i < len(wt))

    sources, links, definition, annotations, covered = [], [], None, [], {}
    db_adds = []   # reusable pieces to save to the reference DB AFTER a successful commit
    for a in assigns:
        try:
            idx = sorted(int(i) for i in a.get("idx", []) if 0 <= int(i) < len(wt))
        except Exception:
            idx = []
        if not idx:
            continue
        role = (a.get("role") or "").strip()
        phrase, atoms = phrase_for(idx), atoms_for(idx)
        if role in ("synonym", "letters", "replacement", "substitution", "anagram",
                    "selection"):
            pos = sorted(int(p) for p in (a.get("pos") or [])
                         if str(p).lstrip("-").isdigit())
            value = (a.get("value") or "").strip().upper()
            if role in ("letters", "replacement") and not value:
                value = "".join(ans_letters[p - 1] for p in pos if 1 <= p <= N)
            if role == "anagram" and not value:            # fodder = the ticked clue words' letters
                value = "".join(c for c in phrase.upper() if c.isalpha())
            if role == "selection":                        # derived letters — validate vs the rule
                rule = (a.get("rule") or "").strip()       # so a selection can never be free-typed
                cands = _selection_candidates(phrase, rule)
                if not value or value not in cands:
                    return _json({"ok": False, "msg": "Selection %r (%s) = %r is not what the "
                                  "rule derives (%s)." % (phrase, rule or "no rule", value,
                                  " / ".join(cands) if cands else "nothing — word too short")})
            if not pos:
                return _json({"ok": False, "msg": "The piece %r has no answer tiles — click "
                              "the answer letters it makes, then Assign." % phrase})
            if role == "anagram":                          # fodder must CONTAIN the tiles it fills;
                got = "".join(ans_letters[p - 1] for p in pos if 1 <= p <= N)  # any surplus fodder
                from collections import Counter            # letters are a deletion before the anagram
                short = Counter(got) - Counter(value)      # (10 fodder letters -> a 9-letter anagram)
                if short:
                    return _json({"ok": False, "msg": "Anagram fodder %r does not contain the tiles "
                                  "you clicked (%s) — missing %s." % (value, got,
                                  "".join(sorted(short.elements())))})
            si = len(sources)
            # record the piece's REAL mechanism so the render shows the right label (letters ->
            # "Literal", substitution -> "Substitution", anagram -> "anagram", synonym -> "synonym")
            # and NOT "MANUAL" on every piece; the whole parse is already flagged manual at the top.
            mech = {"letters": "raw", "replacement": "replacement_letter",
                    "substitution": "abbreviation",
                    "anagram": "anagram_fodder", "selection": "selection"}.get(role, "synonym")
            sources.append(Source(clue_atom_ids=atoms, text=phrase, value=value,
                                  mechanism=mech, source="db"))
            if role == "synonym" and value:            # reusable -> save to the DB after commit
                db_adds.append(("synonym", phrase, value))
            elif role == "substitution" and value:     # abbr/symbol -> wordplay table, after commit
                db_adds.append(("substitution", phrase, value))
            tr = "anagram_of" if role == "anagram" else None
            for p in pos:
                if p in covered:
                    return _json({"ok": False, "msg": "Answer tile %d is claimed by two "
                                  "pieces — each tile belongs to exactly one piece." % p})
                covered[p] = si
                links.append(Link(answer_pos=p, source_index=si, operation="manual",
                                  transform=tr))
        elif role == "definition":
            # 'dbe' = definition by example — identical to a plain definition in every code
            # path (still the parse.definition Source), only the rendered label differs; the
            # marker rides on the mechanism so it round-trips through storage.
            _dmech = ("definition_by_example"
                      if (a.get("dkind") or "").strip() == "dbe" else "definition")
            definition = Source(clue_atom_ids=atoms, text=phrase, value=ans_letters,
                                mechanism=_dmech, source="manual")
            db_adds.append(("definition", phrase, ans_letters))   # reusable -> save after commit
        elif role == "indicator":
            it = (a.get("itype") or "").split(":")[0] or "wordplay"
            isb = (a.get("isub") or "").strip()
            annotations.append(Annotation(clue_atom_ids=atoms, text=phrase, role="indicator",
                                           note="%s%s indicator" % (it, ("/" + isb) if isb else ""),
                                           source="manual"))
            _raw_it = (a.get("itype") or "").split(":")[0]        # save only a REAL chosen type
            if _raw_it:
                db_adds.append(("indicator", phrase, _raw_it, isb or None))
        elif role == "deletion":               # a word whose letters are REMOVED (named deletion) —
            value = (a.get("value") or "").strip().upper()   # e.g. "a" -> A dropped before an anagram
            if not value:
                value = "".join(c for c in phrase.upper() if c.isalpha())
            annotations.append(Annotation(clue_atom_ids=atoms, text=phrase, role="deletion",
                                           note="deleted letters: %s" % value, source="manual"))
        elif role in ("link", "filler"):
            annotations.append(Annotation(clue_atom_ids=atoms, text=phrase, role="link",
                                           note=("surface filler" if role == "filler"
                                                 else "link word"), source="manual"))

    if not sources:
        return _json({"ok": False, "msg": "Place at least one piece on the answer tiles "
                      "(assign a synonym/letters role and click the tiles it makes)."})
    missing = [p for p in range(1, N + 1) if p not in covered]
    if missing:
        return _json({"ok": False, "msg": "Not committed — answer tile(s) %s have no piece. "
                      "Every answer letter must be coloured by a piece." % ", ".join(map(str, missing))})

    andlit = bool((request.form.get("andlit") or "").strip())
    if andlit:
        # ALL-IN-ONE (&lit): the whole clue is BOTH the wordplay (the pieces above) AND the
        # definition — the same words used twice. Build the definition from the WHOLE clue so the
        # words carrying wordplay roles are also covered by it (no separate definition tick needed,
        # and unexplained_words is satisfied). Never auto-confirmed, like a cryptic definition.
        all_atoms = tuple(aid for u in wt for aid in u.atom_ids)
        definition = Source(clue_atom_ids=all_atoms, text=clue_text, value=ans_letters,
                            mechanism="definition", source="manual")
    if definition is None:
        return _json({"ok": False, "msg": "Not committed — no definition. Every clue must end with "
                      "a definition: tick the definition word(s) and pick the definition role."})
    # status="pass" for the WRITE (save_parse refuses to persist a non-pass parse once the clue is
    # frozen, store.py:132); for &lit the verdict is downgraded to 'pending' via set_status below
    # (a direct UPDATE that bypasses that guard) so an &lit is never auto-confirmed — like a CD.
    parse = Parse(clue_text=clue_text, answer_text=answer, sources=sources, links=links,
                  annotations=annotations, definition=definition,
                  operation=("andlit" if andlit else "manual"),
                  solved_by="manual", status="pass")
    if andlit:
        parse.warnings = ["all-in-one (&lit) — the whole clue also reads as the definition; "
                          "needs human confirmation to pass"]
    # EVERY clue word must have a role. A manual PASS with clue words left unaccounted is a false
    # pass (the human is asserting a complete solve) — refuse it, listing what is still unaccounted.
    unaccounted = parse.unexplained_words(ctx)
    if unaccounted:
        return _json({"ok": False, "msg": "Not committed — these clue words have NO role: %s. Every "
                      "clue word must be a piece, the definition, an indicator, a link, filler, or a "
                      "deletion." % ", ".join("“%s”" % w for w in unaccounted)})
    conn = store.connect()
    try:
        store.set_hs_assignments(conn, cid, payload)
        store.save_parse(conn, cid, parse, ctx)
        store.set_status(conn, cid, "pending" if andlit else "pass")
        store.set_frozen(conn, cid)
        conn.commit()
    finally:
        conn.close()

    # The commit SUCCEEDED — now save the reusable pieces to the reference DB, so a hand-solve
    # teaches the system: a synonym/definition/abbreviation/indicator supplied here helps future
    # clues instead of being trapped in this one frozen parse. Dedup is built into each adder,
    # so the ADD-NEW box can be used freely without creating duplicates. Link words, letters,
    # anagram fodder, deletions and filler are deliberately NOT saved (per-clue or not reusable;
    # link words in particular are kept out of the DB so they can't overlap with indicators).
    # Done ONLY after a successful commit, so a rejected commit never writes.
    added, present, rejected = _apply_db_adds(db_adds)

    msg = ("Committed a MANUAL solution (%d piece%s) — frozen."
           % (len(sources), "" if len(sources) == 1 else "s"))
    if added:
        msg += " Saved to reference DB: %d new (%s)." % (
            len(added), "; ".join(a.split(": ", 1)[-1] for a in added))
    if present:
        msg += " %d already in DB." % len(present)
    if rejected:
        msg += " Not saved: %s." % "; ".join(rejected)
    return _json({"ok": True, "msg": msg})


@app.route("/hsmanualuncommit", methods=["POST"])
def hsmanualuncommit_route():
    """Clear a committed manual solution and hand the clue back to the cascade."""
    only = (request.form.get("only") or "").strip()
    if not only.isdigit():
        return _json({"ok": False, "msg": "No clue."})
    cid = int(only)
    conn = store.connect()
    try:
        store.clear_frozen(conn, cid)
        conn.commit()
    finally:
        conn.close()
    _resolve_one(cid)
    return _json({"ok": True, "msg": "Uncommitted — handed back to the cascade."})


@app.route("/handsolvecommit", methods=["POST"])
def handsolvecommit_route():
    """MANUAL SOLVE commit for the ATOM-LEVEL colour-tagging tool. Persists the human's placed
    pieces (each piece = its clue atoms -> the answer TILES it colours) + the definition as a
    FROZEN manual Parse. Because the human assigns each piece to its exact answer tiles, this
    represents reversals / containers / anything — NOT a left-to-right concatenation. No
    reference-DB write, no cascade, no auto-verification (the opposite of the banned builder).
    Returns JSON for the tool's result panel."""
    import json
    only = (request.form.get("only") or "").strip()
    payload = (request.form.get("payload") or "").strip()
    if not only.isdigit():
        return _json({"ok": False, "msg": "No clue."})
    cid = int(only)
    row = _load_clue(cid)
    if row is None:
        return _json({"ok": False, "msg": "No clue."})
    clue_text, answer, src, pnum, direction, enumeration, cnum = row
    answer = enum_space(answer, enumeration)
    ctx = build_wfw_atom_context(clue_text, answer, direction=direction)
    ans_letters = "".join(c for c in answer.upper() if c.isalpha())
    N = len(ans_letters)
    try:
        pieces_in = json.loads(payload) if payload else []
    except Exception:
        pieces_in = []

    from core.wfw_model import Source, Link, Parse
    sources, links, definition, covered = [], [], None, {}
    for pc in pieces_in:
        atoms = tuple(str(a) for a in (pc.get("atoms") or []))
        text = (pc.get("text") or "").strip()
        value = (pc.get("value") or "").strip().upper()
        mech = (pc.get("mech") or "manual").strip() or "manual"
        pos = sorted(int(p) for p in (pc.get("pos") or [])
                     if str(p).lstrip("-").isdigit())
        if pc.get("def"):
            definition = Source(clue_atom_ids=atoms, text=text or answer,
                                value=ans_letters, mechanism="definition", source="manual")
            continue
        if not pos:
            continue                       # a piece placed on no tiles contributes nothing
        si = len(sources)
        sources.append(Source(clue_atom_ids=atoms, text=(text or value), value=value,
                              mechanism=mech, source="manual"))
        for p in pos:
            if p in covered:
                return _json({"ok": False, "msg": "Answer tile %d is claimed by two pieces "
                              "— each tile belongs to exactly one piece." % p})
            covered[p] = si
            links.append(Link(answer_pos=p, source_index=si, operation="manual"))

    if not sources:
        return _json({"ok": False, "msg": "Place at least one piece on the answer tiles first."})
    missing = [p for p in range(1, N + 1) if p not in covered]
    if missing:
        return _json({"ok": False, "msg": "Not committed — answer tile(s) %s have no piece. "
                      "Every tile must be coloured by a piece." % ", ".join(map(str, missing))})

    parse = Parse(clue_text=clue_text, answer_text=answer, sources=sources, links=links,
                  annotations=[], definition=definition, operation="manual",
                  solved_by="manual", status="pass")
    conn = store.connect()
    try:
        store.save_parse(conn, cid, parse, ctx)
        store.set_status(conn, cid, "pass")
        store.set_frozen(conn, cid)
        conn.commit()
    finally:
        conn.close()
    return _json({"ok": True, "msg": "Committed a MANUAL solution (%d piece%s) — frozen, and "
                  "NOT written to the reference DB." % (len(sources),
                  "" if len(sources) == 1 else "s")})


@app.route("/handsolveuncommit", methods=["POST"])
def handsolveuncommit_route():
    """Clear a committed manual solution and hand the clue back to the cascade."""
    only = (request.form.get("only") or "").strip()
    if not only.isdigit():
        return _json({"ok": False, "msg": "No clue."})
    cid = int(only)
    conn = store.connect()
    try:
        store.clear_frozen(conn, cid)
        conn.commit()
    finally:
        conn.close()
    _resolve_one(cid)
    return _json({"ok": True, "msg": "Uncommitted — handed back to the cascade."})


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
        'title="Open the hand-solver for this clue">'
        '&#9776; Hand-solver</a>'
        % (escape(raw_list, quote=True), clue_id, clue_id, _frm))


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
