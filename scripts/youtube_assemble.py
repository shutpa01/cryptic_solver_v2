"""Assemble a captured puzzle into a 1080p video, with chapters.

    python scripts/youtube_assemble.py                     # latest captured puzzle
    python scripts/youtube_assemble.py --puzzle 31324
    python scripts/youtube_assemble.py --seconds 15

Reads logs/youtube/<source>-<puzzle>/ (what youtube_capture.py wrote) and adds:
    banner.png        the permanent on-screen banner, rendered once
    frame_NN.png      each clue padded onto a fixed 1920x1080 canvas
    video.mp4         the finished upload
    chapters.txt      per-clue timestamps, ready for the description
    description.txt   the full description, chapters included

WHY EACH FRAME IS PADDED, NOT SCALED TO FIT
-------------------------------------------
Captured panels vary in height — 1230px to 1740px on telegraph 31324 — because a
clue with six wordplay rows is taller than one with two. Scaling each to fill the
frame would make the card visibly grow and shrink between clues. Instead every
panel is scaled to the SAME height and centred, so the card holds still and only
its width changes.

THE TEN-SECOND FLOOR
--------------------
YouTube only renders chapters when every chapter is at least 10 seconds long (and
there are at least three, the first starting at 00:00). Chapters are the whole
per-clue SEO argument, so --seconds below 10 is refused rather than silently
producing a video with no chapters.

THE BANNER
----------
Burned into every frame, never overlapping the card: the video's own instruction
to SEARCH for justcordelia.com. It is on screen rather than in the description
because a description link is nofollow and buys nothing — a branded search does.
It is rendered by the same headless Chrome that captures the clues, so its type
matches the site rather than being an ffmpeg font approximation.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "logs" / "youtube"

# 1440p by default, not 1080p. Nothing here moves, and it is almost all thin
# coloured text on white — the case chroma subsampling and a low bitrate treat
# worst. YouTube allocates a markedly higher bitrate to a 1440p stream than a
# 1080p one, so uploading above 1080 is the cheapest available sharpness.
RESOLUTIONS = {"1080p": (1920, 1080), "1440p": (2560, 1440), "2160p": (3840, 2160)}
W, H = RESOLUTIONS["1440p"]

BANNER_FRAC = 0.085      # banner height as a share of frame height
MARGIN_FRAC = 0.022      # tighter than before: the card should own the frame
BG = "0xF1F5F9"          # slate-100, the site's own page ground

MIN_SECONDS = 10                           # YouTube's chapter floor


def geometry(w, h):
    """Banner, margin and content box for a frame size. The banner sits at the
    TOP: at the bottom it reads as a footer and the eye never goes there."""
    banner_h = int(round(h * BANNER_FRAC))
    margin = int(round(h * MARGIN_FRAC))
    return banner_h, margin, (w - margin * 2, h - banner_h - margin * 2)


BANNER_H, MARGIN, (CONTENT_W, CONTENT_H) = geometry(W, H)
GAP_FRAC = 0.022

# Rendered at the frame's own width so the type is never upscaled.
BANNER_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
  html,body{margin:0;padding:0;width:%(w)dpx;height:%(h)dpx;overflow:hidden}
  body{background:#f59e0b;color:#1c1917;display:flex;align-items:center;
       justify-content:center;gap:%(gap)dpx;white-space:nowrap;
       border-radius:%(radius)dpx;
       font-family:"Segoe UI",system-ui,-apple-system,Arial,sans-serif}
  .face{width:%(face)dpx;height:%(face)dpx;border-radius:50%%;object-fit:cover;
        object-position:top;border:%(ring)dpx solid #fff;flex:none}
  .lead{font-size:%(lead)dpx;font-weight:800;letter-spacing:.10em;color:#7c2d12}
  .dom{font-size:%(dom)dpx;font-weight:800;color:#1c1917;letter-spacing:.01em}
  .tail{font-size:%(tail)dpx;font-weight:600;color:#7c2d12}
</style></head><body>
  <img class="face" src="%(face_src)s" alt="">
  <span class="lead">SEARCH</span>
  <span class="dom">justcordelia.com</span>
  %(tail_html)s
</body></html>"""

# Cordelia herself, on screen in every frame. The site's own image, not a
# redrawn or generated one (web/static/cordelia.jpg — the same file the site
# header and the About page use).
FACE = ROOT / "web" / "static" / "cordelia.jpg"


def banner_html(w, h):
    """The banner at a given size.

    The type is sized from the HEIGHT, so a narrow banner overflows: at 1080 wide
    the tail wrapped to two lines and "SEARCH" was clipped off the left edge
    (2026-08-21, the Short). Below ~1400px the tail is dropped — the domain and
    the instruction to search for it are the part that has to survive.
    """
    wide = w >= 1400
    tail = ('<span class="tail">&mdash; solve the whole puzzle, with built-in '
            'hints and tools</span>') if wide else ""
    if not FACE.exists():
        sys.exit("Cordelia's picture is missing at %s — refusing to build frames "
                 "without it." % FACE)
    return BANNER_HTML % {"w": w, "h": h, "gap": int(h * (0.22 if wide else 0.16)),
                          "lead": int(h * (0.30 if wide else 0.26)),
                          "dom": int(h * (0.42 if wide else 0.38)),
                          "tail": int(h * 0.24), "tail_html": tail,
                          "radius": int(h * 0.14),
                          "face": int(h * 0.74), "ring": max(2, int(h * 0.035)),
                          "face_src": FACE.resolve().as_uri()}


# The opening card. It says which puzzle this is before anything else, so a
# viewer who landed from a search knows in one second they are in the right
# place — and so the video does not open cold on clue 1.
INTRO_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
  html,body{margin:0;padding:0;width:%(w)dpx;height:%(h)dpx;overflow:hidden}
  body{background:#F1F5F9;color:#0f172a;display:flex;flex-direction:column;
       align-items:center;justify-content:center;gap:%(gap)dpx;
       font-family:"Segoe UI",system-ui,-apple-system,Arial,sans-serif}
  .paper{font-size:%(paper)dpx;font-weight:600;letter-spacing:.14em;
         text-transform:uppercase;color:#6366f1}
  .num{font-size:%(num)dpx;font-weight:800;line-height:1.05;text-align:center}
  .sub{font-size:%(sub)dpx;font-weight:600;color:#334155}
  .date{font-size:%(date)dpx;font-weight:500;color:#64748b}
  .face{width:%(face)dpx;height:%(face)dpx;border-radius:50%%;object-fit:cover;
        object-position:top;border:%(ring)dpx solid #fff;
        box-shadow:0 %(sh)dpx %(sh2)dpx rgba(15,23,42,.18)}
</style></head><body>
  <img class="face" src="%(face_src)s" alt="">
  <div class="paper">%(paper_text)s</div>
  <div class="num">%(kind)s<br>%(number)s</div>
  <div class="sub">Every clue explained</div>
  <div class="date">%(date_text)s</div>
</body></html>"""


def intro_html(w, h, paper, kind, number, date_text):
    from html import escape
    return INTRO_HTML % {
        "w": w, "h": h, "gap": int(h * 0.035),
        "paper": int(h * 0.045), "num": int(h * 0.135),
        "sub": int(h * 0.055), "date": int(h * 0.040),
        "paper_text": escape(paper), "kind": escape(kind),
        "number": escape(str(number)), "date_text": escape(date_text or ""),
        "face": int(h * 0.22), "ring": max(3, int(h * 0.008)),
        "sh": int(h * 0.006), "sh2": int(h * 0.020),
        "face_src": FACE.resolve().as_uri(),
    }


def ffmpeg_bin(name):
    """ffmpeg/ffprobe path. winget put them on PATH, but a shell started before
    the install will not see it, so fall back to the known install location."""
    from shutil import which
    found = which(name)
    if found:
        return found
    guess = (Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
             / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
             / "ffmpeg-9.0-full_build/bin" / (name + ".exe"))
    if guess.exists():
        return str(guess)
    sys.exit("%s not found. Install it (winget install Gyan.FFmpeg) or open a new shell."
             % name)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        sys.exit("FAILED: %s\n%s" % (" ".join(str(c) for c in cmd[:6]), r.stderr[-2000:]))
    return r


def latest_capture():
    dirs = [p for p in OUT_ROOT.glob("*-*") if (p / "manifest.json").exists()]
    if not dirs:
        sys.exit("Nothing captured yet — run scripts/youtube_capture.py first.")
    return max(dirs, key=lambda p: (p / "manifest.json").stat().st_mtime)


def render_html(html_text, w, h, path):
    """Render an HTML fragment to a PNG of exactly w x h.

    Written to a file and loaded over file:// rather than a data: URL — a data:
    URL is cut off at the first '#', and the stylesheet is full of hex colours,
    so Chrome received a truncated document and produced a blank 1898x32 strip
    with no error at all.

    The size is ASSERTED, not hoped for: a wrong-sized overlay is exactly the
    kind of failure that produces a finished-looking video with nothing in it.
    """
    src = path.with_suffix(".html")
    src.write_text(html_text, encoding="utf-8")
    opts = Options()
    opts.add_argument("--headless=new")
    # Wider and taller than the target: an element screenshot is clipped to the
    # viewport, and a window sized exactly w gave a viewport 22px narrower
    # (window chrome), silently cropping the render.
    opts.add_argument("--window-size=%d,%d" % (w + 200, h + 260))
    opts.add_argument("--hide-scrollbars")
    opts.add_argument("--force-device-scale-factor=1")
    d = webdriver.Chrome(options=opts)
    try:
        d.get(src.resolve().as_uri())
        # The body's box IS the target, so the file cannot come out a different
        # size from what the overlay expects.
        d.find_element("tag name", "body").screenshot(str(path))
    finally:
        d.quit()
    probe = subprocess.run(
        [ffmpeg_bin("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    got = probe.stdout.strip()
    if got != "%d,%d" % (w, h):
        sys.exit("%s rendered %s, expected %d,%d — refusing to build frames around "
                 "a wrong-sized image." % (path.name, got, w, h))


def render_banner(path, w=None, h=None):
    w = w or W
    h = h or BANNER_H
    render_html(banner_html(w, h), w, h, path)


def png_size(path):
    r = subprocess.run(
        [ffmpeg_bin("ffprobe"), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    ww, hh = r.stdout.strip().split(",")
    return int(ww), int(hh)


def build_frame(ff, src, banner, dst, w, h, banner_h, margin, cw, ch, gap):
    """The card with the call-to-action strip DIRECTLY BENEATH IT, the two
    treated as one block and centred together.

    Not a bar at the frame edge. Pinned to the top of the frame it read as
    browser furniture — dark, out of the eye line, and skipped. Sitting under
    the card in amber it is where the eye already is, at the moment the viewer
    has just finished reading the explanation.

    The card is always height-limited in a 16:9 frame — wider than it is tall,
    but not by 16:9 — so there is space to either side. Filling it would mean
    stretching or cropping the card, and the card is the product.
    """
    sw, sh = png_size(src)
    avail_h = ch - banner_h - gap
    scale = min(cw / sw, avail_h / sh)
    card_w, card_h = int(sw * scale) // 2 * 2, int(sh * scale) // 2 * 2
    block_h = card_h + gap + banner_h
    y0 = margin + (ch - block_h) // 2
    vf = (
        "[0:v]scale=w=%d:h=%d[card];"
        "color=c=%s:s=%dx%d[bg];"
        "[bg][card]overlay=x=(W-w)/2:y=%d[mid];"
        "[mid][2:v]overlay=x=(W-w)/2:y=%d"
        % (card_w, card_h, BG, w, h, y0, y0 + card_h + gap)
    )
    run([ff, "-y", "-loglevel", "error",
         "-i", str(src), "-f", "lavfi", "-i", "color=c=%s:s=%dx%d" % (BG, w, h),
         "-i", str(banner),
         "-filter_complex", vf, "-frames:v", "1", str(dst)])


# Spoken over the opening title card, in Cordelia's voice, on every video.
#
# ADDITIVE, never at the video's expense — the user's correction, 2026-09-04: an
# earlier draft opened "a video can only give you the answers", which sells the site
# by running down the thing the viewer is already watching. The site is offered as
# MORE, not as the better option. It names the domain first and last, because that is
# the one thing the listener has to retain, and the banner on screen says the same
# words.
INTRO_SCRIPT = (
    "Everything you'll see here is on justcordelia.com — and as well as the "
    "explanations you'll find a grid solver with an anagram solver, pattern matcher "
    "and thesaurus built in, plus a hint on any clue while you're still solving. "
    "That's justcordelia.com, and nowhere else."
)
# A breath at the end so the card does not cut the instant she stops speaking.
INTRO_TAIL = 1.0


def narrate_intro(cap, off=False):
    """Synthesise INTRO_SCRIPT to intro_voice.wav; return (path, seconds).

    Returns (None, 0.0) when voice is off OR when synthesis fails for any reason.
    A failed ElevenLabs call must NEVER cost the day's video: the build falls back
    to the silent track it has always used and says so. The deploy step films three
    papers unattended, and a missing key or a 402 is not worth losing a video for.

    `synthesise` is imported lazily because scripts/reel_build.py imports ffmpeg_bin
    and run FROM THIS MODULE — a module-level import here would be circular.
    """
    if off:
        print("Voice off — silent intro.")
        return None, 0.0
    dst = cap / "intro_voice.wav"
    try:
        from scripts.reel_build import synthesise      # lazy: see docstring
        synthesise(INTRO_SCRIPT, dst)
        probe = subprocess.run(
            [ffmpeg_bin("ffprobe"), "-v", "error", "-show_entries",
             "format=duration", "-of", "csv=p=0", str(dst)],
            capture_output=True, text=True)
        secs = float(probe.stdout.strip())
    except SystemExit as e:            # synthesise() exits on a missing key or a 4xx
        print("Voice unavailable (%s) — falling back to a silent intro." % e)
        return None, 0.0
    except Exception as e:
        print("Voice failed (%s: %s) — falling back to a silent intro."
              % (type(e).__name__, e))
        return None, 0.0
    print("Narration: %.2fs" % secs)
    return dst, secs


SOURCE_NAMES = {"telegraph": "Telegraph", "times": "Times", "guardian": "Guardian"}
KIND_WORDS = {"cryptic": "Cryptic Crossword",
              "prize": "Prize Cryptic Crossword",
              "prize-toughie": "Prize Toughie",
              "sunday": "Sunday Crossword",
              "everyman": "Everyman Crossword"}


def pretty_date(iso):
    """2026-08-21 -> 'Friday 21 August 2026'. Falls back to the raw string."""
    try:
        from datetime import date
        y, m, d = (int(x) for x in (iso or "").split("-"))
        dt = date(y, m, d)
        months = ("January February March April May June July August September "
                  "October November December").split()
        days = ("Monday Tuesday Wednesday Thursday Friday Saturday Sunday").split()
        return "%s %d %s %d" % (days[dt.weekday()], d, months[m - 1], y)
    except Exception:
        return iso or ""


def timestamp(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return ("%d:%02d:%02d" % (h, m, s)) if h else ("%d:%02d" % (m, s))


def chapter_title(f):
    """Clue number, clue text, enumeration — never the answer. These become
    'key moments' in Google video results, so they carry the search terms
    someone actually types, not the spoiler."""
    text = (f["clue_text"] or "").strip()
    enum = (f["enumeration"] or "").strip()
    return "%s — %s%s" % (f["label"], text, (" (%s)" % enum) if enum else "")


def main():
    ap = argparse.ArgumentParser(description="Assemble captured frames into a video")
    ap.add_argument("--source", default="telegraph")
    ap.add_argument("--puzzle", default=None)
    ap.add_argument("--seconds", type=float, default=12.0,
                    help="seconds per clue (minimum %d — YouTube's chapter floor)"
                         % MIN_SECONDS)
    ap.add_argument("--dir", default=None, help="capture dir (default: latest)")
    ap.add_argument("--resolution", default="1440p", choices=sorted(RESOLUTIONS),
                    help="output size (default 1440p — YouTube gives a 1440p "
                         "stream a much higher bitrate, and this is all fine text)")
    ap.add_argument("--bitrate", default="24M",
                    help="target video bitrate (default 24M). NOT CRF: a CRF encode "
                         "of near-static text came out at 485 kbps for a 1440p "
                         "stream, and YouTube re-encodes whatever it is given, so a "
                         "thin source stays thin. YouTube's own 1440p guidance is "
                         "~16 Mbps.")
    ap.add_argument("--intro", type=float, default=float(MIN_SECONDS),
                    help="seconds on the opening title card (minimum %d, since it "
                         "is the chapter that must start at 0:00). The narration "
                         "lengthens this when she speaks for longer." % MIN_SECONDS)
    ap.add_argument("--voice-off", action="store_true",
                    help="build without the spoken intro (costs nothing at "
                         "ElevenLabs; the video keeps its silent track)")
    args = ap.parse_args()

    if args.seconds < MIN_SECONDS:
        sys.exit("--seconds %g is below YouTube's %d-second chapter minimum; "
                 "chapters would not render." % (args.seconds, MIN_SECONDS))
    if args.intro < MIN_SECONDS:
        sys.exit("--intro %g is below YouTube's %d-second chapter minimum; the "
                 "opening chapter would not render." % (args.intro, MIN_SECONDS))

    w, h = RESOLUTIONS[args.resolution]
    banner_h, margin, (cw, ch) = geometry(w, h)

    if args.dir:
        cap = Path(args.dir)
    elif args.puzzle:
        cap = OUT_ROOT / ("%s-%s" % (args.source, args.puzzle))
    else:
        cap = latest_capture()
    man_path = cap / "manifest.json"
    if not man_path.exists():
        sys.exit("No manifest.json in %s" % cap)
    man = json.loads(man_path.read_text())
    frames = man["frames"]
    if len(frames) < 3:
        sys.exit("Only %d clue(s) — YouTube needs at least 3 chapters." % len(frames))

    ff = ffmpeg_bin("ffmpeg")

    banner = cap / "banner.png"
    gap = int(round(h * GAP_FRAC))
    # The strip is the width of the content box, so it reads as part of the
    # composition rather than a bar stuck to the frame.
    print("Rendering call-to-action strip (%dx%d)..." % (cw, banner_h))
    render_banner(banner, cw, banner_h)

    label = SOURCE_NAMES.get(man["source"], man["source"].title())
    kind = KIND_WORDS.get(man.get("type_slug") or "", "Cryptic Crossword")
    print("Rendering intro card...")
    intro_src = cap / "intro_src.png"
    intro_h = ch - banner_h - gap
    render_html(intro_html(cw, intro_h, label, kind, man["puzzle_number"],
                           pretty_date(man["publication_date"])),
                cw, intro_h, intro_src)

    # The narration decides how long the title card holds. Chapter timestamps are
    # accumulated from the same `t` this feeds, so lengthening the intro shifts every
    # later chapter automatically — which is why the voice goes OVER the existing
    # opening card rather than being prepended as a segment in front of it.
    voice, voice_secs = narrate_intro(cap, args.voice_off)
    intro_secs = max(args.intro, voice_secs + INTRO_TAIL if voice else args.intro)

    print("Building %d frames at %dx%d..." % (len(frames) + 1, w, h))
    concat_lines, chapters, t = [], [], 0.0

    # The intro is a real chapter, because YouTube requires the first chapter to
    # start at 0:00 — and a title card is a better thing to be at 0:00 than a
    # cold clue.
    intro_frame = cap / "frame_00.png"
    build_frame(ff, intro_src, banner, intro_frame, w, h, banner_h, margin, cw, ch, gap)
    concat_lines.append("file '%s'\nduration %s" % (intro_frame.name, intro_secs))
    chapters.append("%s %s %s — every clue explained"
                    % (timestamp(t), label, man["puzzle_number"]))
    t += intro_secs

    for i, f in enumerate(frames, 1):
        dst = cap / ("frame_%02d.png" % i)
        build_frame(ff, cap / f["frame"], banner, dst, w, h, banner_h, margin, cw, ch, gap)
        # The concat demuxer needs the LAST entry repeated without a duration,
        # or the final image is dropped from the output.
        concat_lines.append("file '%s'\nduration %s" % (dst.name, args.seconds))
        chapters.append("%s %s" % (timestamp(t), chapter_title(f)))
        t += args.seconds
    concat_lines.append("file '%s'" % ("frame_%02d.png" % len(frames)))

    listing = cap / "concat.txt"
    listing.write_text("\n".join(concat_lines) + "\n")

    video = cap / "video.mp4"
    print("Encoding %s (%s)..." % (video.name, timestamp(t)))
    # The audio is either Cordelia's intro padded out with silence, or — when the
    # voice is off or failed — the silent track this has always used. `apad` runs the
    # silence on for ever and -t below cuts it, exactly as the infinite anullsrc was
    # cut; NOT -shortest, which with a concat of stills ends the file at the first
    # image (youtube_short.py, 2026-08-21).
    audio_in = (["-i", str(voice), "-filter_complex", "[1:a]apad[a]",
                 "-map", "0:v", "-map", "[a]"]
                if voice else
                ["-f", "lavfi", "-i",
                 "anullsrc=channel_layout=stereo:sample_rate=44100"])
    run([ff, "-y", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", str(listing)] + audio_in + [
         # The concat demuxer ignores the final entry's duration, so the last
         # image is repeated (above) to make it appear at all — which then holds
         # it for a second helping. -t cuts at the total the chapters were
         # computed from, so timings and video length agree exactly.
         # -t alone. NOT -shortest: with a concat of stills it can end the output
         # at the first image's duration (it did exactly that in youtube_short.py,
         # 2026-08-21). The silent audio is infinite, so -t is the bound.
         "-t", "%.3f" % t,
         # ffmpeg 9 dropped -vsync in favour of -fps_mode. CFR (not VFR) because
         # YouTube re-encodes, and a constant rate is what it expects; the
         # duplicated frames of a still image cost almost nothing at this CRF.
         "-fps_mode", "cfr", "-r", "30",
         # A TARGET BITRATE, not CRF. x264 rightly spends almost nothing on a
         # still image, and the first cut came out at 485 kbps for 2560x1440 —
         # YouTube then re-encodes that thin source and the text goes soft. The
         # fix is to hand YouTube a fat source; the file is bigger, nobody cares.
         "-c:v", "libx264", "-preset", "slow",
         "-b:v", args.bitrate, "-maxrate", args.bitrate, "-bufsize", "40M",
         "-g", "60",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-c:a", "aac", "-b:a", "128k",
         str(video)])

    (cap / "chapters.txt").write_text("\n".join(chapters) + "\n", encoding="utf-8")

    label = man["source"].title()
    desc = [
        "Every clue of %s Cryptic Crossword %s explained — the definition, the "
        "wordplay broken into pieces, and how each piece builds the answer."
        % (label, man["puzzle_number"]),
        "",
        "Published %s." % man["publication_date"],
        "",
        "Stuck on the rest of the puzzle? Search justcordelia.com — every clue "
        "explained, plus a pattern finder, anagram solver and thesaurus built in.",
        "",
        "Chapters:",
    ] + chapters
    (cap / "description.txt").write_text("\n".join(desc) + "\n", encoding="utf-8")

    probe = run([ffmpeg_bin("ffprobe"), "-v", "error", "-show_entries",
                 "format=duration,size", "-of", "default=nw=1", str(video)])
    print("\n%s" % video)
    print(probe.stdout.strip())
    print("chapters.txt / description.txt written (%d chapters)" % len(chapters))
    return 0


if __name__ == "__main__":
    sys.exit(main())
