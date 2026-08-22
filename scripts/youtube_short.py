"""Build one clue into a vertical Short — the tease, then the reveal.

    python scripts/youtube_short.py                      # latest captured puzzle
    python scripts/youtube_short.py --puzzle 31324 --clue "9 Down"
    python scripts/youtube_short.py --teaser 6 --reveal 14

Writes short.mp4 (1080x1920) into the puzzle's logs/youtube/<source>-<puzzle>/ dir,
plus short_description.txt.

WHY A SHORT AT ALL
------------------
The long video only reaches people already searching for that day's puzzle. The
Short is the only part of this aimed at people who are not searching — it is where
subscribers come from. Same pipeline, same pages, no new content.

WHY TWO FRAMES
--------------
A Short that opens on the answer has nothing to hold anyone. So the clue page is
captured TWICE: once with the card hidden (clue and enumeration only — a question),
then once whole (answer, wordplay, definition — the payoff). Nothing is invented for
this; the second frame is exactly the frame the long video uses.

WHICH CLUE
----------
Default is the clue with the richest breakdown — the most wordplay rows — because it
has the most to show and the most to teach. Deterministic, so re-running picks the
same one. --clue overrides with a label like "9 Down".

This does NOT upload. Building it is safe to automate; publishing to a channel is a
decision. Upload it by hand, or extend scripts/youtube_upload.py once the daily
long-form upload has proved itself over a few days.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selenium.webdriver.common.by import By

from scripts.youtube_capture import PANEL, TRIM_JS, Server, clue_list, make_driver
from scripts.youtube_assemble import ffmpeg_bin, render_banner, run

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "logs" / "youtube"

SW, SH = 1080, 1920          # YouTube Shorts: vertical, 9:16
SBANNER_H = 110              # the Short's own banner, rendered at 1080 wide
MARGIN = 40
BG = "0xF1F5F9"

# Hide the card as well as the trailing furniture, leaving the clue and the
# source line — the question, with no answer anywhere on screen.
TEASER_JS = TRIM_JS.replace(
    "let n = card.nextElementSibling;",
    "card.style.display = 'none';\nlet n = card.nextElementSibling;")


def pick_clue(frames, label):
    if label:
        for f in frames:
            if f["label"].lower() == label.lower():
                return f
        sys.exit("No clue labelled %r. Available: %s"
                 % (label, ", ".join(f["label"] for f in frames)))
    # Tallest captured panel == most wordplay rows == most to explain.
    return max(frames, key=lambda f: f.get("css_height", 0))


def capture_pair(source, clue, out_dir):
    """Teaser and reveal frames for one clue, from the same page load."""
    d = make_driver("3")
    try:
        with Server() as srv:
            d.get("%s/clue/%s" % (srv.base, clue["slug"]))
            if d.execute_script(TEASER_JS, PANEL) != "ok":
                sys.exit("teaser: panel/card missing for %s" % clue["label"])
            d.find_element(By.CSS_SELECTOR, PANEL).screenshot(
                str(out_dir / "short_teaser.png"))
            # Same load, card restored — the reveal is the identical panel.
            d.execute_script(
                "document.querySelector('%s').querySelector('.wfw-card')"
                ".style.display = '';" % PANEL)
            d.find_element(By.CSS_SELECTOR, PANEL).screenshot(
                str(out_dir / "short_reveal.png"))
    finally:
        d.quit()
    return out_dir / "short_teaser.png", out_dir / "short_reveal.png"


TOP_Y = 430      # where the clue line sits, in BOTH frames


def vertical_frame(ff, src, banner, dst):
    """One vertical frame: panel scaled to the full width, anchored at a FIXED
    top edge.

    Not centred. The teaser panel is two lines tall and the reveal is five times
    that, so centring each would make the clue jump up the screen at the very
    moment the viewer is reading it. Anchored, the clue stays exactly where it
    was and the breakdown unfolds beneath it.

    The y is clamped so a tall reveal can never run under the banner.
    """
    # Banner at the TOP, matching the long-form video — at the bottom of a phone
    # screen it competes with the player's own controls and is never looked at.
    vf = ("[0:v]scale=w=%d:h=-2[card];"
          "color=c=%s:s=%dx%d[bg];"
          "[bg][card]overlay=x=(W-w)/2:y='min(%d\\,main_h-%d-overlay_h)'[mid];"
          "[mid][2:v]overlay=x=0:y=0"
          % (SW - MARGIN * 2, BG, SW, SH, TOP_Y, MARGIN))
    run([ff, "-y", "-loglevel", "error", "-i", str(src),
         "-f", "lavfi", "-i", "color=c=%s:s=%dx%d" % (BG, SW, SH),
         "-i", str(banner), "-filter_complex", vf, "-frames:v", "1", str(dst)])


def main():
    ap = argparse.ArgumentParser(description="Build a vertical Short for one clue")
    ap.add_argument("--source", default="telegraph")
    ap.add_argument("--puzzle", default=None)
    ap.add_argument("--clue", default=None, help='e.g. "9 Down"')
    ap.add_argument("--teaser", type=float, default=6.0, help="seconds on the clue")
    ap.add_argument("--reveal", type=float, default=14.0, help="seconds on the answer")
    args = ap.parse_args()

    if args.puzzle:
        cap = OUT_ROOT / ("%s-%s" % (args.source, args.puzzle))
    else:
        dirs = [p for p in OUT_ROOT.glob("*-*") if (p / "manifest.json").exists()]
        if not dirs:
            sys.exit("Nothing captured yet — run scripts/youtube_capture.py first.")
        cap = max(dirs, key=lambda p: (p / "manifest.json").stat().st_mtime)
    man = json.loads((cap / "manifest.json").read_text())

    clue = pick_clue(man["frames"], args.clue)
    print("%s #%s — %s: %s (%s)"
          % (man["source"], man["puzzle_number"], clue["label"],
             clue["clue_text"], clue["enumeration"]))

    ff = ffmpeg_bin("ffmpeg")
    # Its own banner at the Short's width — reusing the 2560-wide landscape one
    # and scaling it down softens the type for no reason.
    banner = cap / "banner_short.png"
    render_banner(banner, SW, SBANNER_H)

    teaser_png, reveal_png = capture_pair(args.source, clue, cap)
    vertical_frame(ff, teaser_png, banner, cap / "short_f1.png")
    vertical_frame(ff, reveal_png, banner, cap / "short_f2.png")

    listing = cap / "short_concat.txt"
    listing.write_text(
        "file 'short_f1.png'\nduration %s\n"
        "file 'short_f2.png'\nduration %s\n"
        "file 'short_f2.png'\n" % (args.teaser, args.reveal))

    out = cap / "short.mp4"
    total = args.teaser + args.reveal
    run([ff, "-y", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", str(listing),
         "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
         # -t alone bounds the output. NOT -shortest: with a concat of stills it
         # ends the output at the FIRST image's duration (proved 2026-08-21 — a
         # 20s Short came out 6.03s, exactly the teaser). The silent audio is
         # infinite, so -t is what has to stop it.
         "-t", "%.3f" % total,
         "-fps_mode", "cfr", "-r", "30",
         "-c:v", "libx264", "-preset", "slow", "-crf", "16",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-c:a", "aac", "-b:a", "128k", str(out)])

    paper = man["source"].title()
    desc = (
        "%s Cryptic #%s, %s: %s (%s)\n\n"
        "Every clue of this puzzle is explained in full — search justcordelia.com.\n\n"
        "#Shorts #crossword #crypticcrossword #puzzles\n"
        % (paper, man["puzzle_number"], clue["label"], clue["clue_text"],
           clue["enumeration"]))
    (cap / "short_description.txt").write_text(desc, encoding="utf-8")

    probe = run([ffmpeg_bin("ffprobe"), "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height:format=duration",
                 "-of", "default=nw=1", str(out)])
    print("\n%s" % out)
    print(probe.stdout.strip())
    print("short_description.txt written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
