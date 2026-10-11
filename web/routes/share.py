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


# The Facebook post (user, 2026-10-11): a picture of the card, because a post is
# plain text and the colour coding must survive, plus a caption carrying the
# searchable words. Both built by web/facebook_post.py. Same admin gate as above,
# and `fbpost` is not a WFW segment either, so the URL survives the mount.

_NOT_SERVED = ("This clue has no public page, so there is nothing to post. Confirm "
               "it on the clue page (or solve it in /hs) first.")


@bp.route("/fbpost/<int:clue_id>")
def facebook_caption(clue_id):
    """The post's caption, as plain text."""
    if not g.get("is_admin"):
        abort(404)

    from web.facebook_post import caption_for

    text = caption_for(clue_id)
    if text is None:
        return _NOT_SERVED, 409, {"Content-Type": "text/plain; charset=utf-8"}
    return text, 200, {"Content-Type": "text/plain; charset=utf-8"}


@bp.route("/fbpost/<int:clue_id>.png")
def facebook_picture(clue_id):
    """The post's picture: the clue page's panel, photographed by headless Chrome.

    Chrome loads the page from THIS server over loopback — whatever address the
    request came in on (the phone uses the LAN address), the PC reaches itself
    at 127.0.0.1.
    """
    if not g.get("is_admin"):
        abort(404)

    from flask import request
    from web.facebook_post import card_png

    base = "http://127.0.0.1:%s" % request.environ.get("SERVER_PORT", "5001")
    try:
        png = card_png(clue_id, base)
    except Exception as e:      # Chrome missing, page broken: say so, in words
        return ("Could not build the picture: %s" % e, 500,
                {"Content-Type": "text/plain; charset=utf-8"})
    if png is None:
        return _NOT_SERVED, 409, {"Content-Type": "text/plain; charset=utf-8"}
    return png, 200, {"Content-Type": "image/png",
                      "Content-Disposition":
                          'attachment; filename="cordelia-clue-%d.png"' % clue_id}


@bp.route("/fbpost/<int:clue_id>.mp4")
def facebook_video(clue_id):
    """The post's video: the picture held on screen while Cordelia's recording
    plays (a post cannot carry a sound file; Facebook plays video)."""
    if not g.get("is_admin"):
        abort(404)

    from flask import request
    from web.facebook_post import NoRecording, card_mp4

    base = "http://127.0.0.1:%s" % request.environ.get("SERVER_PORT", "5001")
    try:
        mp4 = card_mp4(clue_id, base)
    except NoRecording:
        return ("No video: this clue has no current recording of Cordelia reading "
                "it. Post the picture instead.", 409,
                {"Content-Type": "text/plain; charset=utf-8"})
    except Exception as e:
        return ("Could not build the video: %s" % e, 500,
                {"Content-Type": "text/plain; charset=utf-8"})
    if mp4 is None:
        return _NOT_SERVED, 409, {"Content-Type": "text/plain; charset=utf-8"}
    return mp4, 200, {"Content-Type": "video/mp4",
                      "Content-Disposition":
                          'attachment; filename="cordelia-clue-%d.mp4"' % clue_id}


@bp.route("/fbpost/<int:clue_id>/publish", methods=["POST"])
def facebook_publish(clue_id):
    """Post the built video or picture, with the caption, to the Facebook Page.
    The button asks for a second click before calling this."""
    if not g.get("is_admin"):
        abort(404)

    from flask import jsonify, request
    from web.facebook_post import publish

    kind = (request.get_json(silent=True) or {}).get("kind", "")
    try:
        rec = publish(clue_id, kind)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 409
    return jsonify(ok=True, **rec)


@bp.route("/fbpost/<int:clue_id>/posted")
def facebook_posted(clue_id):
    """Whether this clue is already on the Page — so the button can say so
    instead of offering a second post."""
    if not g.get("is_admin"):
        abort(404)
    from flask import jsonify
    from web.facebook_post import already_posted
    return jsonify(already_posted(clue_id) or {})
