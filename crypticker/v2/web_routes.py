"""Crypticker on the main site: /crypticker (players) + /crypticker/admin (testing).

Players get ONE response: the game page with today's (UK date) approved game
inlined — no API, no fetch, nothing written. Admin (the site's ?admin=key
session, g.is_admin) lists the inventory, builds candidates, plays any item with
the clock off, approves (fixes its release date) or rejects.

Writes (build / approve / reject) are refused unless the request comes from this
machine: the droplet only serves what a deploy copied to it.
"""
import json, os

from flask import Blueprint, abort, current_app, g, redirect, request, url_for
from markupsafe import escape

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import inventory as inv                                    # noqa: E402

crypticker_bp = Blueprint("crypticker", __name__, url_prefix="/crypticker")
TEMPLATE = os.path.join(_HERE, "quad_template.html")


def _page(game, admin=None):
    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()
    # </ inside JSON must not close the <script>
    dump = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    return html.replace("__GAME_JSON__", dump(game)).replace("__ADMIN_JSON__", dump(admin))


def _admin():
    if not g.get("is_admin"):
        abort(404)


def _local_write():
    _admin()
    if request.remote_addr not in ("127.0.0.1", "::1"):
        abort(403, "Crypticker inventory is changed on the local machine only, then deployed.")


@crypticker_bp.route("/")
def today():
    date = inv.uk_today()
    game = inv.game_for(date)
    if game is None:
        resp = current_app.response_class(
            "<!DOCTYPE html><meta name=viewport content='width=device-width'>"
            "<p style='font:18px system-ui;padding:24px'>No Crypticker today — back tomorrow.</p>", 200)
    else:
        admin = {"note": 'Today\'s game · <a href="%s">inventory</a>' % url_for("crypticker.admin"),
                 "clock_off": False} if g.get("is_admin") else None
        resp = current_app.response_class(_page(game, admin), 200)
    resp.headers["Cache-Control"] = "private, no-store" if g.get("is_admin") else "public, max-age=300"
    return resp


@crypticker_bp.route("/play/<pid>")
def play(pid):
    _admin()
    try:
        it = inv.load(pid)
    except (OSError, ValueError):
        abort(404)
    note = '%s · <a href="%s">inventory</a>' % (escape(it["status"].replace("_", " ")), url_for("crypticker.admin"))
    resp = current_app.response_class(_page(it["game"], {"note": note, "clock_off": True}), 200)
    resp.headers["Cache-Control"] = "private, no-store"
    return resp


_ADMIN_CSS = """
body{font:15px system-ui,sans-serif;background:#f1f5f9;color:#1e293b;margin:0;padding:16px}
.wrap{max-width:760px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:22px 0 8px}
.big{font-size:20px;font-weight:800;background:#fff;border-radius:12px;padding:14px 16px;margin:12px 0}
.big.low{background:#fee2e2;color:#991b1b}
table{width:100%;border-collapse:collapse;background:#fff;border-radius:12px;overflow:hidden}
td,th{padding:10px;border-bottom:1px solid #f1f5f9;text-align:left;vertical-align:middle}
th{font-size:12px;text-transform:uppercase;color:#475569}
form{display:inline} button,.btn{font:inherit;font-weight:700;border:none;border-radius:8px;padding:8px 12px;cursor:pointer;text-decoration:none;display:inline-block}
.play{background:#1e293b;color:#fff}.ok{background:#15803d;color:#fff}.no{background:#e2e8f0;color:#1e293b}
.build{background:#d64545;color:#fff;font-size:16px;padding:12px 18px}
.msg{background:#fef3c7;border-radius:10px;padding:10px 14px;margin:10px 0;font-weight:600}
"""


@crypticker_bp.route("/admin")
def admin():
    _admin()
    today = inv.uk_today()
    allit = inv.items()
    sched = inv.schedule()
    ahead = [it for it in sched if it["release_date"] >= today]
    released = [it for it in sched if it["release_date"] < today][::-1]
    to_test = sorted((it for it in allit if it["status"] == "to_test"), key=lambda it: it["built_at"])
    rejected = sum(it["status"] == "rejected" for it in allit)

    def name(it):
        pz = it["game"]["puzzle"]
        last = next(e for e in it["game"]["entries"] if e["is_last"])
        return "%s %s · %s · %d clues · last: %s" % (
            escape(pz["source"]), escape(pz["number"]), escape(pz["quadrant"]),
            len(it["game"]["entries"]), escape(last["answer"]))

    def row(it, buttons):
        return "<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (
            escape(it.get("release_date") or ""), name(it), buttons)

    def btns(it, approve=True, reject=True):
        b = '<a class="btn play" href="%s">Play (clock off)</a> ' % url_for("crypticker.play", pid=it["id"])
        if approve:
            b += '<form method="post" action="%s"><button class="ok">Approve</button></form> ' % url_for("crypticker.approve", pid=it["id"])
        if reject:
            b += '<form method="post" action="%s"><button class="no">Reject</button></form>' % url_for("crypticker.reject", pid=it["id"])
        return b

    msg = request.args.get("msg")
    h = ["<!DOCTYPE html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
         "<title>Crypticker inventory</title><style>%s</style><div class=wrap>" % _ADMIN_CSS,
         "<h1>Crypticker inventory</h1><p>Today (UK): <b>%s</b> · <a href='%s'>today's game</a></p>"
         % (today, url_for("crypticker.today")),
         '<div class="big%s">%d approved puzzle%s ready (today onward)%s</div>'
         % (" low" if len(ahead) < 3 else "", len(ahead), "" if len(ahead) == 1 else "s",
            "" if any(it["release_date"] == today for it in ahead) else " — NOTHING for today"),
         ('<div class="msg">%s</div>' % escape(msg)) if msg else "",
         '<form method="post" action="%s"><button class="build">Build a new puzzle to test</button></form>'
         % url_for("crypticker.build"),
         "<h2>To test (%d)</h2>" % len(to_test),
         "<table><tr><th></th><th>Puzzle</th><th></th></tr>%s</table>"
         % ("".join(row(it, btns(it)) for it in to_test) or "<tr><td colspan=3>None — build one.</td></tr>"),
         "<h2>Scheduled (%d)</h2>" % len(ahead),
         "<table><tr><th>Date</th><th>Puzzle</th><th></th></tr>%s</table>"
         % ("".join(row(it, btns(it, approve=False, reject=it["release_date"] > today)) for it in ahead)
            or "<tr><td colspan=3>None.</td></tr>"),
         "<h2>Released (%d)</h2>" % len(released),
         "<table><tr><th>Date</th><th>Puzzle</th><th></th></tr>%s</table>"
         % ("".join(row(it, btns(it, approve=False, reject=False)) for it in released)
            or "<tr><td colspan=3>None yet.</td></tr>"),
         "<p>Rejected: %d (never rebuilt).</p></div>" % rejected]
    resp = current_app.response_class("".join(h), 200)
    resp.headers["Cache-Control"] = "private, no-store"
    return resp


def _back(msg):
    return redirect(url_for("crypticker.admin", msg=msg))


@crypticker_bp.route("/admin/build", methods=["POST"])
def build():
    _local_write()
    item, tried = inv.build_candidate()
    if item is None:
        return _back("No playable corner found (%d tried)." % len(tried))
    return _back("Built %s — play it, then approve or reject." % item["id"])


@crypticker_bp.route("/admin/<pid>/approve", methods=["POST"])
def approve(pid):
    _local_write()
    it = inv.approve(pid)
    return _back("Approved %s for %s." % (it["id"], it["release_date"]))


@crypticker_bp.route("/admin/<pid>/reject", methods=["POST"])
def reject(pid):
    _local_write()
    try:
        inv.reject(pid)
    except ValueError as e:
        return _back(str(e))
    return _back("Rejected %s." % pid)
