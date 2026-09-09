"""Build one clue into a NARRATED vertical Short — script, voice, frames, one command.

    python scripts/short_build.py --clue-id 10089910
    python scripts/short_build.py --clue-id 10089910 --voice-off   # free, silent
    python scripts/short_build.py --check --clue-id 10089910       # can it be narrated?

Writes short_narrated.mp4 into the puzzle's logs/youtube/<source>-<puzzle>/ directory.
It does NOT upload. Publishing stays a human act (user, 2026-09-07).

WHY THE SCRIPT IS SPLIT IN TWO
------------------------------
The picture changes once, from the clue to the answer card. Synthesising the narration
as two blocks means the cut lands on a sentence boundary and the frame durations are the
measured audio durations — not a guess. It is the same reason scripts/reel_build splits
its narration, and it is what lets the answer card appear on the exact word "the answer
is".

THE SIXTY-SECOND CAP
--------------------
A Short is capped at 60s. The speech decides the length and cannot be trimmed, so the
HOLD after Cordelia stops talking is the shock absorber: it shrinks to fit. If the speech
alone will not fit, this REFUSES rather than emit a video that is not a Short — an
over-long file is not a smaller problem discovered later, it is a different product.
"""

import argparse
import contextlib
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "logs" / "youtube"

CAP = 59.8          # Shorts limit is 60s; leave a little air for container rounding.
HOLD_MAX = 7.5      # Seconds on the answer card after the narration ends.
HOLD_MIN = 2.0      # Below this the card snaps away the instant she stops.

# The narration splits here: everything before plays over the clue, everything after
# over the answer card. Matches narrate_clue.INTRO_DIFFERENCE's opening words.
SPLIT_AT = "We do this differently"


def _load(name, filename):
    import importlib.util as iu
    spec = iu.spec_from_file_location(name, str(Path(__file__).resolve().parent / filename))
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seconds(path):
    with contextlib.closing(wave.open(str(path))) as w:
        return w.getnframes() / float(w.getframerate())


def script_for(clue_id):
    """(script, row, refusals) — the narration, or the reasons it cannot be spoken."""
    nc = _load("nc", "narrate_clue.py")
    from web import create_app, wfw_read
    app = create_app("development")
    with app.app_context():
        from web.db import get_db
        row = get_db().execute(
            "SELECT id, source, puzzle_number, publication_date, clue_text, answer, "
            "       enumeration, clue_number, direction FROM clues WHERE id = ?",
            (clue_id,)).fetchone()
        if row is None:
            return None, None, ["no such clue"]
        parse = wfw_read._load(clue_id)
        if parse is None:
            return None, dict(row), ["no stored parse — the clue is not solved"]
        paper = nc._SPOKEN_PAPER.get(row["source"], (row["source"] or "").title())
        script, problems = nc.narrate(parse, row["clue_text"], row["answer"] or "",
                                      row["enumeration"] or "", paper)
    return script, dict(row), problems


def build(clue_id, voice_off=False):
    rb = _load("rb", "reel_build.py")
    ya = _load("ya", "youtube_assemble.py")
    ff = ya.ffmpeg_bin("ffmpeg")

    script, row, problems = script_for(clue_id)
    if script is None:
        sys.exit("Cannot narrate this clue: %s" % "; ".join(problems))

    cap = OUT_ROOT / ("%s-%s" % (row["source"], row["puzzle_number"]))
    if not (cap / "manifest.json").exists():
        sys.exit("No capture for %s #%s — run scripts/youtube_capture.py first."
                 % (row["source"], row["puzzle_number"]))
    (cap / "narration.txt").write_text(script, encoding="utf-8")

    i = script.index(SPLIT_AT)
    blocks = (script[:i].strip(), script[i:].strip())

    if voice_off:
        # Silent build for checking the pictures without spending on synthesis.
        secs = [max(4.0, len(b.split()) / 2.6) for b in blocks]
        print("voice off — timings estimated from the word count")
    else:
        # ONE TAKE. Two generations were two cold starts and never matched each
        # other; the boundary now comes from the alignment, not from cutting the
        # recording in two. See reel_build.synthesise_marked.
        _, at, total = rb.synthesise_marked(script, cap / "short_voice.wav", SPLIT_AT)
        secs = [at, total - at]
        print("one take: %.2fs, answer card turns at %.2fs" % (total, at))

    speech = sum(secs)
    hold = min(HOLD_MAX, CAP - speech)
    if hold < HOLD_MIN:
        sys.exit("Narration is %.1fs; with the minimum %.1fs hold that is %.1fs, over "
                 "the %.0fs a Short allows. Pick a shorter clue, or shorten the script."
                 % (speech, HOLD_MIN, speech + HOLD_MIN, CAP + 0.2))
    teaser, reveal, total = secs[0], secs[1] + hold, speech + hold
    print("clue on screen %.2fs | answer card %.2fs (%.2fs of it after she stops) | "
          "total %.2fs" % (teaser, reveal, hold, total))

    label = "%s %s" % (row["clue_number"], (row["direction"] or "").capitalize())
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "youtube_short.py"),
                        "--source", row["source"], "--puzzle", str(row["puzzle_number"]),
                        "--clue", label,
                        "--teaser", "%.2f" % teaser, "--reveal", "%.2f" % reveal],
                       cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit("frame build failed")

    # NAMED BY CLUE. There is one capture directory per PUZZLE, so a fixed filename is
    # overwritten by whichever clue was built last — and the dry run of short_post then
    # showed 4 Across's title above 15 Down's film. A wrong video under a right title is
    # exactly the kind of silent mismatch that only surfaces after it is published.
    out = cap / ("short_%d.mp4" % clue_id)
    desc_src, desc_dst = cap / "short_description.txt", cap / ("short_%d.txt" % clue_id)
    if desc_src.exists():
        desc_dst.write_text(desc_src.read_text(encoding="utf-8"), encoding="utf-8")
    if voice_off:
        (cap / "short.mp4").replace(out)
        print(out)
        return out

    # No concat step any more: synthesise_marked wrote short_voice.wav whole.
    voice = cap / "short_voice.wav"
    held = cap / "short_voice_held.wav"
    # Pad the silence onto the AUDIO so both streams end together; a video longer than
    # its audio is what makes a player look like it has stopped early.
    ya.run([ff, "-y", "-loglevel", "error", "-i", str(voice),
            "-af", "apad=pad_dur=%.2f" % (hold + 0.1), "-t", "%.2f" % total, str(held)])
    ya.run([ff, "-y", "-loglevel", "error", "-i", str(cap / "short.mp4"), "-i", str(held),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
            "-b:a", "160k", "-movflags", "+faststart", str(out)])
    print(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clue-id", type=int, required=True)
    ap.add_argument("--voice-off", action="store_true",
                    help="build the pictures with no narration — free, for checking")
    ap.add_argument("--check", action="store_true",
                    help="say whether this clue can be narrated, build nothing")
    args = ap.parse_args(argv)

    if args.check:
        script, row, problems = script_for(args.clue_id)
        if script is None:
            print("NO  — %s" % "; ".join(problems))
            return 1
        print("YES — %d words" % len(script.split()))
        return 0
    build(args.clue_id, voice_off=args.voice_off)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
