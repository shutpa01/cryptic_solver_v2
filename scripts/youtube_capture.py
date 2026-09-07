"""Capture one frame per clue of a puzzle, straight off the clue page.

    python scripts/youtube_capture.py                        # latest served telegraph puzzle
    python scripts/youtube_capture.py --puzzle 31324
    python scripts/youtube_capture.py --source times --limit 3

Writes to logs/youtube/<source>-<puzzle>/ (logs/ is gitignored, .gitignore:130):
    clue_01.png ... clue_NN.png   one per served clue, in the puzzle's own order
    manifest.json                 clue order, ids, labels, text, enumeration —
                                  the source for the video's per-clue chapters

WHY IT SERVES THE SITE ITSELF
-----------------------------
It starts the Flask app in-process on a loopback port and points Chrome at that,
rather than at justcordelia.com. Two reasons, both verified 2026-08-21:

  1. Cloudflare challenges headless Chrome on the public site — the probe got
     "Just a moment..." and no card at all.
  2. It removes the dependency on a dev server happening to be running. Nothing
     external has to be up for a nightly capture to work.

The rendering is the SAME code either way — web/templates/clue.html printing
core.wfw_card via web.serving.get_card. This is not a second renderer.

WHAT IS IN THE FRAME
--------------------
The white clue panel only: clue text with enumeration, the source/number/date
line, and the WFW card (type badge, answer tiles, assembly line, per-piece rows,
definition). Everything after the card inside that panel — the prev/next clue
nav and the "See the rest of this puzzle" box — is hidden at capture time, as is
the "Back to the puzzle" button. Site navigation, breadcrumb, promo block and
footer are outside the panel and never enter an element screenshot.

Nothing on the site is modified: the hiding is done in the captured browser only.
"""

import argparse
import json
import shutil
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on sys.path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from werkzeug.serving import make_server

from web import create_app

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "logs" / "youtube"

# The clue panel. No id on it, so it is addressed by the classes it is built
# with (web/templates/clue.html:57). Asserted per clue — a miss is fatal, never
# a silently empty frame.
PANEL = "div.bg-white.rounded-lg.shadow-sm.border"

# Hide everything in the panel that sits AFTER the card, plus the way-back
# button. Structural, so it does not break when a class name changes.
TRIM_JS = """
const panel = document.querySelector(arguments[0]);
if (!panel) { return 'no-panel'; }
const card = panel.querySelector('.wfw-card');
if (!card) { return 'no-card'; }
let n = card.nextElementSibling;
while (n) { n.style.display = 'none'; n = n.nextElementSibling; }
panel.querySelectorAll('a').forEach(a => {
    if (a.textContent.trim().startsWith('\\u2190')) { a.style.display = 'none'; }
});
return 'ok';
"""


class Server:
    """The app on a loopback port, for the life of the capture."""

    def __init__(self):
        self.app = create_app("development")
        self.srv = make_server("127.0.0.1", 0, self.app, threaded=True)
        self.port = self.srv.socket.getsockname()[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.srv.shutdown()

    @property
    def base(self):
        return "http://127.0.0.1:%d" % self.port


def latest_served_puzzle(app, source):
    """The most recently published puzzle of `source` whose every clue is served."""
    with app.app_context():
        from web.db import get_db
        from web.serving import served_puzzle_numbers
        served = served_puzzle_numbers([source])
        rows = get_db().execute(
            "SELECT puzzle_number, MAX(publication_date) AS pub FROM clues "
            "WHERE source = ? AND puzzle_number IS NOT NULL "
            "GROUP BY puzzle_number ORDER BY pub DESC", (source,)).fetchall()
    for r in rows:
        if (source, str(r["puzzle_number"])) in served:
            return str(r["puzzle_number"]), r["pub"]
    return None, None


def clue_list(app, source, puzzle):
    """The puzzle's clues in its own order (across then down — the order
    web/routes/puzzle.py:84-87 builds), each with its page slug.

    Clues whose OWN page does not exist are dropped. A "See N" continuation stub
    counts as solved for the puzzle-level gate but has no card of its own, so its
    clue page 410s (web/serving.py:135-155). Filming one would capture an error
    page, so it is skipped and reported.
    """
    with app.app_context():
        from web.models import get_puzzle_clues, get_puzzle_date, classify_puzzle
        from web.routes.clue import generate_clue_slug
        from web.serving import is_served
        rows = get_puzzle_clues(source, puzzle)
        pub = get_puzzle_date(source, puzzle)
        # Recorded here so the assembler can title the intro card without
        # needing an app context of its own.
        type_slug, type_label = classify_puzzle(source, puzzle, pub)
        across = [c for c in rows if c["direction"] == "across"]
        down = [c for c in rows if c["direction"] != "across"]
        out, skipped = [], []
        for c in across + down:
            label = "%s %s" % (c["clue_number"], (c["direction"] or "").title())
            if not is_served(source, c["id"]):
                skipped.append(label)
                continue
            out.append({
                "clue_id": c["id"],
                "label": label,
                "number": c["clue_number"],
                "direction": c["direction"],
                "clue_text": c["clue_text"],
                "enumeration": c["enumeration"],
                "slug": generate_clue_slug(c["clue_text"] or "", clue_id=c["id"]),
            })
    return out, skipped, pub, type_slug, type_label


def make_driver(scale, window="1400,1200"):
    """`window` is the CSS viewport. The default is the landscape video's and must not
    change. The vertical Short passes a NARROW one: a panel laid out at 1400 is a thin
    wide band, and squeezing that into a 1080x1920 frame leaves 85% of a phone screen
    empty with the clue in tiny type. Laid out narrow, the same panel wraps and fills
    the frame (user, 2026-09-07: "just the clue displaying in tiny text")."""
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=%s" % window)
    opts.add_argument("--hide-scrollbars")
    # Renders at `scale`x so the panel has real pixels to spare when it is scaled
    # into a 1080p frame. At 1x the captured panel is ~740px wide and upscaling
    # it to fill the frame is visibly soft.
    opts.add_argument("--force-device-scale-factor=%s" % scale)
    d = webdriver.Chrome(options=opts)
    d.set_page_load_timeout(60)
    return d


def main():
    ap = argparse.ArgumentParser(description="Capture a frame per clue of a puzzle")
    ap.add_argument("--source", default="telegraph")
    ap.add_argument("--puzzle", default=None,
                    help="puzzle number; default is the latest fully-served one")
    ap.add_argument("--scale", default="3", help="device scale factor (default 3)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clues (testing)")
    ap.add_argument("--out", default=None, help="output dir (default logs/youtube/...)")
    args = ap.parse_args()

    app = create_app("development")
    puzzle, pub = (args.puzzle, None)
    if puzzle is None:
        puzzle, pub = latest_served_puzzle(app, args.source)
        if puzzle is None:
            sys.exit("No fully-served %s puzzle found." % args.source)

    clues, skipped, pub2, type_slug, type_label = clue_list(
        app, args.source, puzzle)
    pub = pub or pub2
    if not clues:
        sys.exit("%s %s has no served clue pages." % (args.source, puzzle))

    out_dir = Path(args.out) if args.out else OUT_ROOT / ("%s-%s" % (args.source, puzzle))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print("%s #%s (%s) — %d clue pages%s"
          % (args.source, puzzle, pub, len(clues),
             (", %d stub(s) skipped: %s" % (len(skipped), ", ".join(skipped)))
             if skipped else ""))

    todo = clues[:args.limit] if args.limit else clues
    d = make_driver(args.scale)
    frames = []
    try:
        with Server() as srv:
            for i, c in enumerate(todo, 1):
                # One retry. A miss here has been transient (2026-08-21: clue 21
                # returned no-panel on a page that serves 200 with the panel and
                # card both present, and succeeded on reload). A miss that
                # survives a reload is real and must stop the run — a wrong frame
                # is worse than no video.
                state = None
                for attempt in (1, 2):
                    d.get("%s/clue/%s" % (srv.base, c["slug"]))
                    state = d.execute_script(TRIM_JS, PANEL)
                    if state == "ok":
                        break
                    print("  retry %s (%s): %s at %s"
                          % (c["label"], c["clue_id"], state, d.current_url))
                if state != "ok":
                    raise SystemExit(
                        "clue %s (%s): %s after 2 attempts — the panel or card did "
                        "not render, so the frame would be wrong. Stopping rather "
                        "than writing it. Page title was %r."
                        % (c["label"], c["clue_id"], state, d.title))
                panel = d.find_element(By.CSS_SELECTOR, PANEL)
                name = "clue_%02d.png" % i
                panel.screenshot(str(out_dir / name))
                w, h = panel.rect["width"], panel.rect["height"]
                frames.append(dict(c, frame=name, css_width=w, css_height=h))
                print("  %-8s %-40s %s (%.0fx%.0f css)"
                      % (c["label"], (c["clue_text"] or "")[:40], name, w, h))
    finally:
        d.quit()

    manifest = {
        "source": args.source,
        "puzzle_number": puzzle,
        "publication_date": pub,
        "type_slug": type_slug,
        "type_label": type_label,
        "scale": args.scale,
        "skipped_stubs": skipped,
        "frames": frames,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\n%d frame(s) + manifest.json -> %s" % (len(frames), out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
