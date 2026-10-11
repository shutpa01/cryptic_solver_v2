"""One clue as a Facebook post: a picture of the card, plus a caption.

The user posts confirmed clues to the site's Facebook Page (2026-10-11). Facebook
posts are plain text, and the card's COLOUR CODING is the point of it ("We need
to keep the color coding though, that's very important"), so the card travels
as a PICTURE and the words travel as the caption:

    picture  the clue page's own panel — clue, source line, approved prose and
             the WFW card — screenshotted exactly as the video frames are
             (scripts/youtube_capture.py: PANEL, TRIM_JS, make_driver). This is
             not a second renderer: Chrome loads /clue/<slug> from the running
             server and photographs it.
    caption  the searchable text no picture can carry: the clue and enumeration
             first (that line becomes the snippet), the paper, number and clue,
             the answer, the approved prose, and the link to the clue page.

    video    the picture held on screen while Cordelia's recording plays (user,
             2026-10-11) — a post cannot carry a sound file, but Facebook plays
             video. Only for a clue whose recording is current, the same rule as
             the clue page's Listen button (web.routes.clue._has_current_audio).

All three are refused for a clue without a public page (web.serving.is_served):
the picture would be an error page and the link would 410.
"""

import subprocess
from pathlib import Path

# The last picture built per clue, so the video does not photograph the page a
# second time (~15 s). logs/ is gitignored (.gitignore:132).
OUT_DIR = Path(__file__).resolve().parent.parent / "logs" / "fbpost"

SITE = "https://justcordelia.com"

# Narrow viewport, as the vertical Short uses (scripts/youtube_short.py:77): laid
# out narrow, the panel wraps into a portrait block, which is what a feed shows
# well. Scale 2 gives ~1240px wide — Facebook's own upload size is 1080-2048.
WINDOW = "620,1400"
SCALE = "2"

# The listen button is a control, not content — it means nothing in a picture.
HIDE_LISTEN_JS = """
document.querySelectorAll('.listen-row').forEach(e => { e.style.display = 'none'; });
"""


def _clue(clue_id):
    from web.db import get_db
    return get_db().execute(
        "SELECT id, source, puzzle_number, publication_date, clue_number, "
        "direction, clue_text, enumeration, answer FROM clues WHERE id = ?",
        (clue_id,)).fetchone()


def _served_clue(clue_id):
    """The clue row when it has a public page, else None."""
    from web.serving import is_served
    row = _clue(clue_id)
    if row is None or not is_served(row["source"], clue_id):
        return None
    return row


def _slug(row):
    from web.routes.clue import generate_clue_slug
    return generate_clue_slug(row["clue_text"] or "", clue_id=row["id"])


def _puzzle_line(row):
    """'Telegraph Prize Toughie #246, 10 Across' — the clue page's own source line
    (web/templates/clue.html:83-95: source_name, type_label, #number, clue number)."""
    from flask import current_app
    from web.models import classify_puzzle
    _, type_label = classify_puzzle(row["source"], row["puzzle_number"],
                                    row["publication_date"])
    name = current_app.jinja_env.filters["source_name"](row["source"])
    paper = " ".join(p for p in (name, type_label) if p)
    clue = ("%s %s" % (row["clue_number"] or "", (row["direction"] or "").title())).strip()
    return "%s #%s, %s" % (paper, row["puzzle_number"], clue)


def caption_for(clue_id):
    """The post text, or None when the clue has no public page."""
    row = _served_clue(clue_id)
    if row is None:
        return None
    from web.routes.clue import _approved_prose
    enum = row["enumeration"] or ""
    lines = ["“%s”%s" % (row["clue_text"] or "",
                                   " (%s)" % enum if enum else ""),
             _puzzle_line(row),
             "",
             "Answer: %s" % (row["answer"] or "")]
    prose = _approved_prose(clue_id)
    if prose:
        sentence, gloss = prose
        lines += ["", sentence]
        if gloss:
            lines.append("%s — %s" % (row["answer"] or "", gloss))
    lines += ["", "Full colour-coded explanation: %s/clue/%s" % (SITE, _slug(row))]
    return "\n".join(lines)


def card_png(clue_id, base_url):
    """PNG bytes of the clue page's panel, or None when the clue has no public page.

    `base_url` is the running server Chrome loads the page from. Raises
    RuntimeError when the page loads but the panel or card is missing — a wrong
    picture is worse than none.
    """
    row = _served_clue(clue_id)
    if row is None:
        return None
    from selenium.webdriver.common.by import By
    from scripts.youtube_capture import PANEL, TRIM_JS, make_driver

    d = make_driver(SCALE, window=WINDOW)
    try:
        d.get("%s/clue/%s" % (base_url.rstrip("/"), _slug(row)))
        state = d.execute_script(TRIM_JS, PANEL)
        if state != "ok":
            raise RuntimeError("the clue page rendered without its %s (page title %r)"
                               % ("panel" if state == "no-panel" else "card", d.title))
        d.execute_script(HIDE_LISTEN_JS)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / ("%d.png" % int(clue_id))
        d.find_element(By.CSS_SELECTOR, PANEL).screenshot(str(out))
        return out.read_bytes()
    finally:
        d.quit()


class NoRecording(Exception):
    """The clue is served but has no current recording of Cordelia reading it."""


def card_mp4(clue_id, base_url):
    """MP4 bytes: the card picture for as long as the recording plays. None when
    the clue has no public page; NoRecording when it has no current recording.

    Uses the picture card_png last built for this clue (the button builds it just
    before asking for the video), else builds one.
    """
    if _served_clue(clue_id) is None:
        return None
    from web.routes.clue import _has_current_audio
    if not _has_current_audio(clue_id):
        raise NoRecording()
    from core import clue_audio
    from scripts.youtube_assemble import ffmpeg_bin
    try:
        ffmpeg = ffmpeg_bin("ffmpeg")
    except SystemExit as e:                 # its CLI exit, not ours to obey
        raise RuntimeError(str(e))

    png = OUT_DIR / ("%d.png" % int(clue_id))
    if not png.exists():
        card_png(clue_id, base_url)
    out = OUT_DIR / ("%d.mp4" % int(clue_id))
    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-loop", "1", "-framerate", "2", "-i", str(png),
           "-i", str(clue_audio.mp3_path(clue_id)),
           # H.264 needs even dimensions; the screenshot can be odd.
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
           "-c:v", "libx264", "-tune", "stillimage", "-r", "30",
           "-c:a", "aac", "-b:a", "160k",
           "-shortest", "-movflags", "+faststart", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if res.returncode != 0:
        raise RuntimeError("ffmpeg failed: %s" % (res.stderr or "").strip()[-300:])
    return out.read_bytes()


# ---------------------------------------------------------------------------
# POSTING — straight to the Page (user, 2026-10-11: downloading and re-uploading
# was "a bit clunky"; "yes build the posting button"). Page token from the
# "Cordelia's reels" Meta app, filed in .env as FB_PAGE_ID / FB_PAGE_TOKEN; a
# Page token made from a long-lived user token does not expire. Read at call
# time, so a new token needs no restart.
# ---------------------------------------------------------------------------

GRAPH = "https://graph.facebook.com/v23.0"
GRAPH_VIDEO = "https://graph-video.facebook.com/v23.0"


def _page_creds():
    from dotenv import dotenv_values
    env = dotenv_values(Path(__file__).resolve().parent.parent / ".env")
    pid, tok = env.get("FB_PAGE_ID"), env.get("FB_PAGE_TOKEN")
    if not (pid and tok):
        raise RuntimeError("FB_PAGE_ID / FB_PAGE_TOKEN are not in .env")
    return pid, tok


def _posted_path(clue_id):
    return OUT_DIR / ("%d.posted.json" % int(clue_id))


def already_posted(clue_id):
    """The record of an earlier post of this clue, or None. One post per clue:
    a double click must not put the same clue on the Page twice."""
    import json
    p = _posted_path(clue_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def publish(clue_id, kind):
    """Post this clue to the Page: kind 'video' (the mp4) or 'picture' (the png),
    with the caption. Uses the files the button has just built and shown — it
    never posts something the user has not seen. Returns the record written.
    """
    import json
    import time
    import requests

    if kind not in ("video", "picture"):
        raise ValueError("kind must be video or picture")
    prev = already_posted(clue_id)
    if prev:
        raise RuntimeError("already posted on %s: %s" % (prev["when"], prev.get("url", "")))
    caption = caption_for(clue_id)
    if caption is None:
        raise RuntimeError("this clue has no public page")
    media = OUT_DIR / ("%d.%s" % (int(clue_id), "mp4" if kind == "video" else "png"))
    if not media.exists():
        raise RuntimeError("build the %s first (Facebook post button)" % kind)

    pid, tok = _page_creds()
    with open(media, "rb") as fh:
        if kind == "video":
            r = requests.post("%s/%s/videos" % (GRAPH_VIDEO, pid),
                              data={"description": caption, "access_token": tok},
                              files={"source": (media.name, fh, "video/mp4")},
                              timeout=600)
        else:
            r = requests.post("%s/%s/photos" % (GRAPH, pid),
                              data={"message": caption, "published": "true",
                                    "access_token": tok},
                              files={"source": (media.name, fh, "image/png")},
                              timeout=300)
    try:
        j = r.json()
    except ValueError:
        j = {}
    if r.status_code != 200 or "error" in j:
        raise RuntimeError("Facebook refused: %s"
                           % ((j.get("error") or {}).get("message") or r.text[:300]))

    obj_id = j.get("post_id") or j.get("id")
    url = "https://www.facebook.com/%s" % obj_id
    rec = {"clue_id": int(clue_id), "kind": kind, "id": obj_id, "url": url,
           "when": time.strftime("%Y-%m-%d %H:%M")}
    _posted_path(clue_id).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec
