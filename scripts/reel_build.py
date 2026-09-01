"""Build one daily reel — Cordelia reading a clue over its own answer page.

    python -m scripts.reel_build --clue-id 10086228
    python -m scripts.reel_build --clue-id 10086228 --voice-off   # silent, no API cost

Writes reel.mp4 (1080x1920) and reel_caption.txt into logs/reels/<clue_id>/.

WHAT IT SAYS, and why it is fixed
---------------------------------
Four beats, settled by the user 2026-09-01:

  1. the answers and explanations for today's puzzles are live, on the site and
     on YouTube  — the proof of daily freshness
  2. here's a clue from today's <paper> puzzle, then the clue read out
  3. the answer is <ANSWER>, and it's a <type> clue                — named EARLY
  4. the site has a full grid solver and every tool you need, then the tagline

Beat 3 names the answer up front on purpose. Withholding it to build a moment is
the sermon the whole persona rejects (see the instagram-channel-plan memory), so
the reel spends its time on the mechanism instead of on suspense.

The clue is CHOSEN BY A HUMAN, in the dashboard. Nothing here picks it: the user
wants that judgement, and the reel is built on demand when he makes it — there is
no nightly run.

WHAT IT DOES NOT DO
-------------------
It does not post. Building is safe and repeatable; publishing is a separate
decision and a separate script. Run this, watch the file, then decide.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

from selenium.webdriver.common.by import By

from scripts.youtube_assemble import ffmpeg_bin, run
from scripts.youtube_capture import PANEL, TRIM_JS, Server, make_driver

OUT_ROOT = ROOT / "logs" / "reels"
SW, SH = 1080, 1920           # 9:16, the frame both Instagram and Facebook want
MARGIN = 40
TOP_Y = 300
BG = "0xF1F5F9"

# Emmaline — young British girl. VERIFIED 2026-09-01 by listing the user's own
# voices through the API, not taken from a directory. Overridable by env.
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "nDJIICjR9zfJExIFeSCN")

# THE VOICE SETTINGS ARE FETCHED FROM THE ACCOUNT ON EVERY BUILD, never written
# here.
#
# The user has tuned Emmaline finely and the tone is the point of the channel:
# on 2026-09-01 the voice held stability 0.15 / similarity 1.0, against an
# account default of 0.5 / 0.75. Hard-coding any numbers — even those — would
# freeze a snapshot and silently ignore later retuning; sending none at all would
# rely on the API applying the saved ones implicitly, which is not documented
# behaviour to bet a channel's voice on. So: read them, send them, print them.

PAPER = {"telegraph": "Telegraph", "times": "Times", "guardian": "Guardian"}
TAGLINE = "Just Cordelia — because there's no need to go anywhere else."
SITE = "justcordelia.com"
# Confirmed by the user 2026-09-01: youtube.com/@justcordeliacom — the handle
# matches the Instagram one. My placeholder had been @justcordelia, which was
# wrong, and this string appears ON SCREEN in every reel.
YOUTUBE = os.environ.get("YOUTUBE_CHANNEL_URL", "youtube.com/@justcordeliacom")

# The OPENING frame: where to find it, and what is there today. It runs under
# the first line of narration and nothing else, so it has to be readable in
# about four seconds — which is why it is four things, not a paragraph.
OPENER_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
  html,body{margin:0;padding:0;width:%(w)dpx;height:%(h)dpx;overflow:hidden}
  body{background:#F1F5F9;color:#0f172a;box-sizing:border-box;padding:120px 90px;
       display:flex;flex-direction:column;align-items:center;justify-content:center;
       font-family:"Segoe UI",system-ui,-apple-system,Arial,sans-serif;text-align:center}
  .face{width:300px;height:300px;border-radius:50%%;object-fit:cover;
        border:10px solid #fff;box-shadow:0 10px 40px rgba(15,23,42,.18);margin-bottom:56px}
  .site{font-size:96px;font-weight:800;letter-spacing:-2px;line-height:1.05}
  .yt{font-size:46px;font-weight:600;color:#475569;margin-top:22px}
  .lead{font-size:44px;font-weight:600;color:#0f172a;margin-top:80px;line-height:1.3}
  ul{list-style:none;padding:0;margin:34px 0 0}
  li{font-size:56px;font-weight:700;line-height:1.55;color:#1d4ed8}
</style></head><body>
  %(face_html)s
  <div class="site">%(site)s</div>
  <div class="yt">%(youtube)s</div>
  <div class="lead">Today's puzzles — every clue<br>answered and explained in full</div>
  <ul>%(items)s</ul>
</body></html>"""


def elevenlabs_key():
    """The API key, under either name it may have been saved as.

    The house style is UPPER_SNAKE, but the key was first added as
    `eleven_labs_API`. Accepting both is one line here and avoids a silent
    "no key" failure that looks like a code fault and is really a spelling.
    """
    for name in ("ELEVENLABS_API_KEY", "eleven_labs_API", "ELEVEN_LABS_API"):
        v = os.environ.get(name)
        if v:
            return v.strip()
    return None


def speakable_type(label):
    """"Container + charade + selection" -> "container, charade and selection".

    The label is NOT rebuilt here. It comes from web.wfw_read, the same
    _wordplay_label the site, the clue page and the widget all show, because a
    second labeller would be a fourth mirror to keep in sync and the full clue
    type is a differentiator we get wrong by paraphrasing it.
    """
    parts = [p.strip().lower() for p in (label or "").split("+") if p.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def speakable_answer(answer):
    """A multi-word answer read as words; a solid one left alone. Letters are
    never spelled out — she is reading it, not dictating it."""
    return re.sub(r"\s+", " ", (answer or "").strip())


def load_clue(clue_id):
    import sqlite3
    con = sqlite3.connect(f"file:{ROOT / 'data' / 'clues_master.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            "SELECT id, source, puzzle_number, publication_date, clue_number, "
            "direction, clue_text, enumeration, answer FROM clues WHERE id=?",
            (clue_id,)).fetchone()
    finally:
        con.close()
    if row is None:
        sys.exit("No clue with id %s" % clue_id)
    return dict(row)


def todays_live_puzzles(app, pub_date):
    """The puzzles published on `pub_date` whose every clue is served — i.e. the
    ones a viewer would actually find. Same gate the site uses, so the claim in
    the script cannot promise more than the site delivers."""
    with app.app_context():
        from web.db import get_db
        from web.serving import served_puzzle_numbers
        served = served_puzzle_numbers()
        rows = get_db().execute(
            "SELECT DISTINCT source, puzzle_number FROM clues "
            "WHERE publication_date LIKE ?", (pub_date[:10] + "%",)).fetchall()
    return [(r["source"], str(r["puzzle_number"])) for r in rows
            if (r["source"], str(r["puzzle_number"])) in served]


def build_script(clue, type_label, n_puzzles):
    paper = PAPER.get(clue["source"], (clue["source"] or "").title())
    answer = speakable_answer(clue["answer"])
    kind = speakable_type(type_label)

    puzzles = ("today's puzzles are" if n_puzzles != 1 else "today's puzzle is")

    # TWO BLOCKS, because the picture changes between them: the first plays over
    # the opening banner (address, channel, what is live), the second over the
    # clue's own answer page. Splitting the narration is what lets the cut land
    # on the sentence rather than on a guessed number of seconds.
    opening = ("The answers and full explanations for %s live on %s, "
               "and on YouTube." % (puzzles, SITE))
    body = "\n".join([
        "Here's a clue from today's %s puzzle." % paper,
        (clue["clue_text"] or "").strip().rstrip("."),
        ("The answer is %s, and it's a %s clue." % (answer, kind) if kind
         else "The answer is %s." % answer),
        "%s has a full grid solver, and all the tools you could possibly want "
        "to solve the puzzle." % SITE,
        TAGLINE,
    ])
    return opening, body


def synthesise(text, dst):
    """ElevenLabs -> wav. WAV, not mp3: the audio goes straight into ffmpeg and
    out as the AAC the platforms specify, so an mp3 in the middle would be a
    generation of loss for nothing (and 192kbps mp3 needs the Creator tier)."""
    import requests
    key = elevenlabs_key()
    if not key:
        sys.exit("No ElevenLabs key in .env (ELEVENLABS_API_KEY or "
                 "eleven_labs_API) — add one, or use --voice-off.")
    if not VOICE_ID:
        sys.exit("ELEVENLABS_VOICE_ID is not in .env. There is deliberately no "
                 "default: a guessed id would produce the wrong voice silently.")
    hdr = {"xi-api-key": key}

    # The tuning, straight from the account. Note the LIST endpoint reports
    # settings as null for every voice — only this per-voice endpoint returns
    # them, which is why it is fetched separately.
    s = requests.get("https://api.elevenlabs.io/v1/voices/%s/settings" % VOICE_ID,
                     headers=hdr, timeout=30)
    if s.status_code != 200:
        sys.exit("Could not read the voice settings (%s): %s. Refusing to speak "
                 "with untuned defaults." % (s.status_code, s.text[:200]))
    settings = s.json()
    print("voice %s settings from your account: %s"
          % (VOICE_ID, json.dumps(settings)))

    r = requests.post(
        "https://api.elevenlabs.io/v1/text-to-speech/%s" % VOICE_ID,
        params={"output_format": "pcm_24000"},
        headers={**hdr, "Content-Type": "application/json"},
        json={"text": text, "model_id": "eleven_multilingual_v2",
              "voice_settings": settings},
        timeout=120)
    if r.status_code != 200:
        sys.exit("ElevenLabs %s: %s" % (r.status_code, r.text[:300]))
    raw = dst.with_suffix(".pcm")
    raw.write_bytes(r.content)
    run([ffmpeg_bin("ffmpeg"), "-y", "-loglevel", "error",
         "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", str(raw),
         "-ar", "48000", "-ac", "2", str(dst)])
    raw.unlink(missing_ok=True)
    return dst


def render_opener(live, out_png):
    """The opening frame: the address, the channel, and what is live today."""
    from scripts.youtube_assemble import FACE, render_html
    face_html = ('<img class="face" src="%s">' % FACE.resolve().as_uri()
                 if FACE.exists() else "")
    items = "".join("<li>%s %s</li>" % (PAPER.get(s, s.title()), n)
                    for s, n in live) or "<li>today's puzzles</li>"
    html = OPENER_HTML % {"w": SW, "h": SH, "site": SITE, "youtube": YOUTUBE,
                          "face_html": face_html, "items": items}
    render_html(html, SW, SH, out_png)
    return out_png


def capture_answer_page(slug, out_png):
    """The clue's own page, card and all — the answer and the full breakdown.

    Rendered IN-PROCESS from the same templates the site serves. Never point
    this at the live site: Cloudflare challenges headless Chrome.
    """
    d = make_driver("3")
    try:
        with Server() as srv:
            # ONE RETRY, as scripts/youtube_capture.py does: a miss here is known
            # to be transient (2026-08-21 — a page that serves 200 with panel and
            # card both present returned no-panel once and succeeded on reload).
            # A miss that survives the reload is real, and a wrong frame is worse
            # than no reel, so that stops the build.
            state = None
            for _ in (1, 2):
                d.get("%s/clue/%s" % (srv.base, slug))
                state = d.execute_script(TRIM_JS, PANEL)
                if state == "ok":
                    break
            if state != "ok":
                sys.exit("clue page rendered %r after 2 attempts — refusing to "
                         "build a reel on a frame that is missing the card." % state)
            d.find_element(By.CSS_SELECTOR, PANEL).screenshot(str(out_png))
    finally:
        d.quit()
    return out_png


def audio_seconds(path):
    out = run([ffmpeg_bin("ffprobe"), "-v", "error", "-show_entries",
               "format=duration", "-of", "default=nw=1:nk=1", str(path)])
    return float(out.stdout.strip())


def vertical_frame(ff, card_png, dst):
    """The clue card on the brand background, as a full 1080x1920 still."""
    vf = ("[0:v]scale=w=%d:h=-2[card];"
          "color=c=%s:s=%dx%d[bg];"
          "[bg][card]overlay=x=(W-w)/2:y='min(%d\\,main_h-%d-overlay_h)'"
          % (SW - MARGIN * 2, BG, SW, SH, TOP_Y, MARGIN))
    run([ff, "-y", "-loglevel", "error", "-i", str(card_png),
         "-filter_complex", vf, "-frames:v", "1", str(dst)])
    return dst


def join_audio(ff, parts, dst):
    """The narration blocks end to end, in order."""
    listing = dst.with_suffix(".txt")
    listing.write_text("".join("file '%s'\n" % p.name for p in parts),
                       encoding="utf-8")
    run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-ar", "48000", "-ac", "2", str(dst)])
    return dst


def compose(ff, shots, audio, dst, total):
    """Frames held for their own narration block, cut to one track.

    `shots` is [(png, seconds), ...]. The LAST image is repeated with no
    duration because the concat demuxer ignores the final entry's duration —
    without the repeat the closing frame is dropped entirely.

    -t bounds the output, never -shortest: with a concat of stills, -shortest
    ends the output at the FIRST image's duration (proved 2026-08-21 — a 20s
    Short came out 6.03s, exactly the length of its first frame).
    """
    listing = dst.with_suffix(".concat.txt")
    body = "".join("file '%s'\nduration %.3f\n" % (p.name, s) for p, s in shots)
    body += "file '%s'\n" % shots[-1][0].name
    listing.write_text(body, encoding="utf-8")

    cmd = [ff, "-y", "-loglevel", "error",
           "-f", "concat", "-safe", "0", "-i", str(listing)]
    if audio:
        cmd += ["-i", str(audio)]
    else:
        cmd += ["-f", "lavfi", "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000"]
    cmd += ["-t", "%.3f" % total,
            "-fps_mode", "cfr", "-r", "30",
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", str(dst)]
    run(cmd)


def main():
    ap = argparse.ArgumentParser(description="Build one daily reel for a chosen clue")
    ap.add_argument("--clue-id", type=int, default=None,
                    help="default: the clue picked with the Reel button today")
    ap.add_argument("--voice-off", action="store_true",
                    help="skip ElevenLabs (no API cost); silent 25s reel")
    ap.add_argument("--any-date", action="store_true",
                    help="allow a clue that is not from today (testing only)")
    args = ap.parse_args()

    if args.clue_id is None:
        from core.reel_pick import get_pick
        args.clue_id = get_pick()
        if args.clue_id is None:
            sys.exit("No clue picked for today. Use the Reel button on the "
                     "puzzle page during review, or pass --clue-id.")
        print("using the clue picked today: %s" % args.clue_id)

    clue = load_clue(args.clue_id)

    # THE REEL SAYS "today's puzzle". If the clue is not from today, that is a
    # lie in the narration AND in the banner, which lists the puzzles live on
    # the clue's own date. Caught by eye once on 2026-09-01 (a Guardian clue
    # from 25 August); on a busy morning it would not be.
    if not args.any_date:
        today = date.today().isoformat()
        pub = (clue["publication_date"] or "")[:10]
        if pub != today:
            sys.exit("Clue %s is from %s, not today (%s). The reel would say "
                     "\"today's puzzle\" about an old one. Pass --any-date if "
                     "you are only testing." % (args.clue_id, pub or "unknown", today))
    out_dir = OUT_ROOT / str(args.clue_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    from web import create_app
    from web.routes.clue import generate_clue_slug
    from web.wfw_read import load_breakdown
    app = create_app("development")

    # load_breakdown reads through flask.g, so it needs a real app context —
    # it is the site's own reader, not a standalone query, which is the point.
    with app.app_context():
        breakdown = load_breakdown(args.clue_id)
        slug = generate_clue_slug(clue["clue_text"] or "", clue_id=clue["id"])
    if breakdown is None:
        sys.exit("Clue %s has no passed parse — nothing to explain." % args.clue_id)
    type_label = breakdown.get("operation_label") or ""

    live = todays_live_puzzles(app, clue["publication_date"] or "")

    opening, body = build_script(clue, type_label, len(live))
    (out_dir / "reel_script.txt").write_text(opening + "\n\n" + body,
                                             encoding="utf-8")
    print("script:\n%s\n\n%s\n" % (opening, body))

    ff = ffmpeg_bin("ffmpeg")
    opener = render_opener(live, out_dir / "reel_opener.png")
    card = capture_answer_page(slug, out_dir / "reel_card.png")
    clue_frame = vertical_frame(ff, card, out_dir / "reel_clue.png")

    if args.voice_off:
        audio = None
        shots = [(opener, 5.0), (clue_frame, 20.0)]
    else:
        a1 = synthesise(opening, out_dir / "voice_1.wav")
        a2 = synthesise(body, out_dir / "voice_2.wav")
        d1, d2 = audio_seconds(a1), audio_seconds(a2)
        audio = join_audio(ff, [a1, a2], out_dir / "reel_voice.wav")
        # A held beat after the last word, so the tagline is not clipped by the
        # loop restarting the instant she stops speaking.
        shots = [(opener, d1), (clue_frame, d2 + 0.8)]

    total = sum(s for _, s in shots)
    out = out_dir / "reel.mp4"
    compose(ff, shots, audio, out, total)

    paper = PAPER.get(clue["source"], (clue["source"] or "").title())
    caption = (
        "%s Cryptic %s — %s %s: %s (%s)\n\n"
        "Answers and full explanations for today's puzzles: justcordelia.com\n"
        "%s\n\n"
        "Live today: %s\n\n"
        "#crypticcrossword #crossword #puzzles\n"
        % (paper, clue["puzzle_number"], clue["clue_number"],
           (clue["direction"] or "").title(), clue["clue_text"],
           clue["enumeration"], TAGLINE,
           ", ".join("%s %s" % (PAPER.get(s, s.title()), n) for s, n in live)))
    (out_dir / "reel_caption.txt").write_text(caption, encoding="utf-8")

    probe = run([ffmpeg_bin("ffprobe"), "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height:format=duration",
                 "-of", "default=nw=1", str(out)])
    print(out)
    print(probe.stdout.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
