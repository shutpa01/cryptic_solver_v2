"""Share routes — the WFW card as text, for forums that take no screenshots.

Admin only, and deliberately NOT under the /admin prefix. The WFW clue page is
served by core/wfw_web.py through web/solver_mount.py, which rewrites every
quoted root-absolute URL whose first path segment matches one of the WFW app's
own url_map segments — and `admin` IS one of them (core/wfw_web has its own
POST /admin). A button on that page pointing at /admin/... would be rewritten
to /solver/admin/... and 404. `redditmd` is not a WFW segment, so the URL
survives the mount untouched.
"""

from flask import Blueprint, abort, g

bp = Blueprint("share", __name__)


@bp.route("/redditmd/<int:clue_id>")
def reddit_markdown(clue_id):
    """This clue's WFW card as a Reddit comment (text/plain Markdown).

    404 rather than 403 for a non-admin: this is not a public capability and
    there is nothing to be gained by confirming the URL exists.
    """
    if not g.get("is_admin"):
        abort(404)

    from web.reddit_md import for_clue

    md = for_clue(clue_id)
    if md is None:
        # stored_card returns None unless there is a stored PASS parse. Say so
        # plainly — a blank response would read as a broken button when the
        # real answer is "confirm the reading first".
        return ("This clue has no confirmed reading yet, so there is nothing to "
                "post. Confirm it on the clue page (or solve it in /hs) first.",
                409, {"Content-Type": "text/plain; charset=utf-8"})
    return md, 200, {"Content-Type": "text/plain; charset=utf-8"}
