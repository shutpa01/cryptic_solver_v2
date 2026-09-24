"""Cordelia narrating the APPROVED PROSE, as written, for a whole puzzle.

    python scripts/narrate_prose.py --source telegraph --puzzle 31287   # text only
    python scripts/narrate_prose.py --dir logs/youtube/telegraph-31287
    python scripts/narrate_prose.py --source telegraph --puzzle 31287 --build

Text only by default: it prints exactly what she would say and calls nothing, so
the formula can be read cold before a penny is spent. `--build` synthesises.

THE TEXT IS NOT WRITTEN HERE
---------------------------
There is no second script. The page and the video carry the SAME words — the
prose the user ticked on /hs (user, 2026-09-24: "I feel strongly that Cordelia
should narrate the prose as it is, we do not need a separate record version").
`core.prose_store.approved_text` is the only source, which means the video
inherits the tick: an unapproved draft cannot reach a video any more than it can
reach a page.

PRONUNCIATION IS A LAYER, NOT A VARIANT
---------------------------------------
`scripts.narrate_clue._spoken` is imported and applied to the finished text. The
page keeps `NT`; the voice is handed `N-T`. The setter's own words — the clue,
read inside curly quotes — are protected from it by that module's own _VERBATIM
rule, so a clue reading "US author" is not turned into the pronoun. That file is
NOT edited: this is a second source feeding the same voice.

WHAT IT REFUSES TO SAY
----------------------
  * no approved prose            -> the clue is SILENT. The still holds for its
                                    normal time and the video is still complete;
                                    a tick is what makes her speak.
  * INVALID with a comment       -> she reads the user's own note, as written.
  * INVALID with no comment      -> silent (the clue has no public page either).

ONE TAKE FOR THE WHOLE PUZZLE
-----------------------------
Every clue is generated in a SINGLE ElevenLabs call and cut afterwards, not
synthesised clue by clue. Two generations are two cold starts and they do not
match — on 2026-09-09 a 9.9s opening came out in a different accent from the 43s
body of the same Short. The timings come from the character-level alignment the
API returns, exactly as `reel_build.synthesise_marked` does for one mark; here
the offsets are known because this module built the string.

THE BOUNDARIES
--------------
Three explanations read back-to-back sounded like "a mish-mash of two clues"
(user, 2026-09-24). So each clue is announced and READ FIRST — "One across.
'Bed covers medium for cosiness'" — and each still is held for at least the
chapter floor, which leaves a real silence between one clue and the next.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import prose_store                                      # noqa: E402
from core import store                                            # noqa: E402
from scripts.narrate_clue import _spoken                          # noqa: E402
from scripts.youtube_assemble import MIN_SECONDS, ffmpeg_bin, run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "logs" / "youtube"

# Between segments in the string that is sent for synthesis. A blank line is how
# the model is told these are separate thoughts; it is also what makes the offset
# of each segment unambiguous when the alignment comes back.
SEP = "\n\n"

# A breath after she stops speaking, before the still changes. Without it the cut
# lands on the last syllable.
TAIL = 1.2

# eleven_turbo_v2_5 and eleven_flash_v2_5 take 40,000 characters; the other models
# take 5,000. A whole puzzle is 5-8k, so this only ever bites if the model is
# changed in .env — and then it splits into takes rather than failing, with a
# warning, because two takes of a 30-clue walkthrough is still a video.
_BIG_MODELS = ("eleven_turbo_v2_5", "eleven_flash_v2_5")


def _char_limit():
    from scripts.reel_build import MODEL_ID
    return 39000 if MODEL_ID in _BIG_MODELS else 4800


def _solve_status(conn, clue_id):
    row = conn.execute("SELECT status FROM wfw_solve WHERE clue_id = ?",
                       (clue_id,)).fetchone()
    return (row[0] or "").lower() if row else ""


def clue_text_for(frame, conn, data):
    """What she says for ONE clue, or (None, reason).

    The clue is quoted with curly quotes because that is what narrate_clue's
    _VERBATIM protects — inside them the setter's capitals survive the
    pronunciation layer. The quotes themselves are not spoken.
    """
    cid = frame["clue_id"]
    label = (frame.get("label") or "").strip()
    clue = (frame.get("clue_text") or "").strip().rstrip(". ")
    opening = u"%s. “%s”." % (label, clue)

    approved = prose_store.approved_text(cid, data)
    if approved:
        sentence, gloss = approved
        body = "\n\n".join(p for p in (sentence.strip(), gloss.strip()) if p)
        if not body:
            return None, "approved but empty"
        return opening + SEP + body, ""

    if _solve_status(conn, cid) == "invalid":
        note = store.get_note(conn, cid).strip()
        if note:
            # The user's own words, read as written. Not rephrased, not softened.
            return opening + SEP + note, ""
        return None, "INVALID with no comment"

    return None, "no approved prose"


def segments(frames, intro_text):
    """[{clue_id, label, text|None, reason}] — the intro first, then every clue.

    The intro is segment 0 of the SAME take, not a separate call, so one voice
    runs from the title card to the last clue.
    """
    conn = store.connect()
    try:
        data = prose_store.load()
        out = [{"clue_id": None, "label": "Intro", "text": intro_text, "reason": ""}]
        for f in frames:
            text, why = clue_text_for(f, conn, data)
            out.append({"clue_id": f["clue_id"], "label": f.get("label") or "",
                        "text": text, "reason": why})
        return out
    finally:
        conn.close()


def _takes(segs, limit):
    """Group the SPEAKING segments into as few generations as the model allows.

    Returns [[index, ...], ...] over `segs`. Silent segments are not in any take.
    """
    takes, cur, size = [], [], 0
    for i, s in enumerate(segs):
        if not s["text"]:
            continue
        n = len(s["text"]) + len(SEP)
        if cur and size + n > limit:
            takes.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += n
    if cur:
        takes.append(cur)
    return takes


def synthesise_timed(text, dst):
    """One generation of `text`, with the character-level alignment.

    Returns (dst, characters, start_times). Imported pieces are reel_build's own
    — the account's voice settings are fetched there on every build, deliberately,
    and there is no second copy of that here.
    """
    import base64
    import requests
    from scripts.reel_build import (MODEL_ID, VOICE_ID, _pcm_to_wav, _voice_call)

    hdr, settings = _voice_call(" (one take, whole puzzle)")
    r = requests.post(
        "https://api.elevenlabs.io/v1/text-to-speech/%s/with-timestamps" % VOICE_ID,
        params={"output_format": "pcm_24000"},
        headers={**hdr, "Content-Type": "application/json"},
        json={"text": text, "model_id": MODEL_ID, "voice_settings": settings},
        timeout=600)
    if r.status_code != 200:
        raise RuntimeError("ElevenLabs %s: %s" % (r.status_code, r.text[:300]))
    payload = r.json()
    _pcm_to_wav(base64.b64decode(payload["audio_base64"]), dst)

    # `alignment` tracks the TEXT AS SENT; `normalized_alignment` tracks the
    # model's own expansion of it, where the offsets no longer line up with our
    # string. Only the former can be indexed by an offset we computed.
    al = payload.get("alignment") or {}
    chars = al.get("characters") or []
    starts = al.get("character_start_times_seconds") or []
    ends = al.get("character_end_times_seconds") or []
    if not chars or len(chars) != len(starts) or not ends:
        raise RuntimeError("no usable alignment came back")
    if "".join(chars) != text:
        # ASSERT THE SHAPE, DO NOT ASSUME IT. If the returned characters are not
        # the string we sent, every offset below is meaningless and the cuts would
        # land mid-word in a finished-looking video.
        raise RuntimeError("the alignment does not match the text that was sent")
    return dst, starts, float(ends[-1])


def _duration(path):
    r = run([ffmpeg_bin("ffprobe"), "-v", "error", "-show_entries",
             "format=duration", "-of", "csv=p=0", str(path)])
    return float(r.stdout.strip())


def _cut(ff, take, a, b, dst, seconds):
    """[a, b) of `take`, padded out with silence to exactly `seconds`.

    apad runs the silence on for ever and -t cuts it — the same idiom the video
    encoder uses, and NOT -shortest, which with these inputs ends the output
    early (youtube_short.py, 2026-08-21).
    """
    run([ff, "-y", "-loglevel", "error", "-i", str(take),
         "-ss", "%.3f" % a] + (["-to", "%.3f" % b] if b is not None else []) +
        ["-af", "apad", "-t", "%.3f" % seconds, "-c:a", "pcm_s16le", str(dst)])


def _silence(ff, dst, seconds):
    run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         "anullsrc=channel_layout=stereo:sample_rate=48000",
         "-t", "%.3f" % seconds, "-c:a", "pcm_s16le", str(dst)])


def build_track(cap, frames, intro_text, silent_seconds, min_seconds=None):
    """The whole narration as one wav, plus how long each still must hold.

    Returns (track_path, [seconds per segment], report_lines). The list is
    intro-first, so it lines up with [intro] + frames.

    Returns (None, None, report) for ANY failure. A failed narration must never
    cost the video: the caller falls back to the silent build it has always done.
    That is the same rule `youtube_assemble.narrate_intro` already follows.
    """
    floor = MIN_SECONDS if min_seconds is None else min_seconds
    report = []
    segs = segments(frames, intro_text)
    spoken = [dict(s, text=(_spoken(s["text"]) if s["text"] else None)) for s in segs]

    speaking = [s for s in spoken if s["text"]]
    report.append("%d of %d segments have something to say"
                  % (len(speaking), len(spoken)))
    for s in spoken:
        if not s["text"] and s["reason"]:
            report.append("  silent: %s — %s" % (s["label"], s["reason"]))
    if len(speaking) < 2:
        report.append("Nothing to narrate beyond the intro — silent build.")
        return None, None, report

    ff = ffmpeg_bin("ffmpeg")
    takes = _takes(spoken, _char_limit())
    if len(takes) > 1:
        report.append("WARNING: %d generations, not one — the voice may not match "
                      "across them (model character limit)." % len(takes))

    # seconds of SPEECH for each segment index; silent ones stay absent
    speech = {}
    try:
        for t, idxs in enumerate(takes):
            text = SEP.join(spoken[i]["text"] for i in idxs)
            wav = cap / ("narration_take%d.wav" % t)
            _, starts, total = synthesise_timed(text, wav)
            # Where each segment begins in the string we just sent — known,
            # because this module built the string.
            offsets, pos = [], 0
            for i in idxs:
                offsets.append(pos)
                pos += len(spoken[i]["text"]) + len(SEP)
            for k, i in enumerate(idxs):
                a = float(starts[offsets[k]])
                b = float(starts[offsets[k + 1]]) if k + 1 < len(idxs) else None
                speech[i] = (t, a, b, (b if b is not None else total) - a)
    except SystemExit as e:              # _voice_call exits on a missing key
        report.append("Voice unavailable (%s) — silent build." % e)
        return None, None, report
    except Exception as e:
        report.append("Voice failed (%s: %s) — silent build." % (type(e).__name__, e))
        return None, None, report

    # Cut, pad, and write the concat list in segment order.
    durations, lines = [], []
    for i, s in enumerate(spoken):
        dst = cap / ("narration_%02d.wav" % i)
        if i in speech:
            t, a, b, dur = speech[i]
            secs = max(floor, dur + TAIL)
            _cut(ff, cap / ("narration_take%d.wav" % t), a, b, dst, secs)
        else:
            secs = max(floor, silent_seconds)
            _silence(ff, dst, secs)
        durations.append(secs)
        lines.append("file '%s'" % dst.name)

    listing = cap / "narration_concat.txt"
    listing.write_text("\n".join(lines) + "\n")
    track = cap / "narration.wav"
    run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c:a", "pcm_s16le", str(track)])

    got, want = _duration(track), sum(durations)
    if abs(got - want) > 0.25:
        # The concat demuxer has silently produced the wrong length before. If the
        # track and the timeline disagree, every chapter after the drift is wrong.
        report.append("Track is %.2fs but the timeline wants %.2fs — silent build."
                      % (got, want))
        return None, None, report
    report.append("Narration: %.1fs over %d segments" % (got, len(durations)))
    return track, durations, report


# ---------------------------------------------------------------------------
# CLI — text only unless --build is given


def _capture_dir(args):
    if args.dir:
        return Path(args.dir)
    if args.puzzle:
        return OUT_ROOT / ("%s-%s" % (args.source, args.puzzle))
    dirs = [p for p in OUT_ROOT.glob("*-*") if (p / "manifest.json").exists()]
    if not dirs:
        sys.exit("Nothing captured yet — run scripts/youtube_capture.py first.")
    return max(dirs, key=lambda p: (p / "manifest.json").stat().st_mtime)


def main():
    ap = argparse.ArgumentParser(description="What Cordelia would say for a puzzle")
    ap.add_argument("--source", default="telegraph")
    ap.add_argument("--puzzle", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--build", action="store_true",
                    help="synthesise it (costs ElevenLabs characters)")
    ap.add_argument("--seconds", type=float, default=12.0,
                    help="how long a SILENT clue holds")
    args = ap.parse_args()

    cap = _capture_dir(args)
    man_path = cap / "manifest.json"
    if not man_path.exists():
        sys.exit("No manifest.json in %s" % cap)
    frames = json.loads(man_path.read_text())["frames"]

    from scripts.youtube_assemble import INTRO_SCRIPT
    if not args.build:
        segs = segments(frames, INTRO_SCRIPT)
        said = 0
        for s in segs:
            print("=" * 70)
            if s["text"]:
                said += 1
                print("%s\n\n%s" % (s["label"], _spoken(s["text"])))
            else:
                print("%s — SILENT (%s)" % (s["label"], s["reason"]))
        print("=" * 70)
        print("%d of %d segments speak. Nothing was synthesised." % (said, len(segs)))
        return 0

    track, durations, report = build_track(cap, frames, INTRO_SCRIPT, args.seconds)
    for line in report:
        print(line)
    if track:
        print("%s\n%s" % (track, " ".join("%.1f" % d for d in durations)))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
