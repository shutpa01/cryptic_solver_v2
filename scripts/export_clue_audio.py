"""Cut each clue's narration out of a built puzzle video, for the clue page's Listen button.

    python scripts/export_clue_audio.py --source telegraph --puzzle 31357
    python scripts/export_clue_audio.py --dir logs/youtube/telegraph-31357

youtube_assemble.py calls export() itself straight after a narrated build, so the
recording and the words it is stamped with are read in the same minute. Run by
hand only to catch up a build that already exists.

What it reads: narration_NN.wav, which narrate_prose.build_track cut from the ONE
take for frames[NN-1] (segment 00 is the intro and is never exported). Those
files are padded with digital silence to the chapter minimum; the padding is
trimmed here, because on a web page it is just a dead gap after she stops.

What it writes: data/clue_audio/<clue_id>.mp3 (mono speech, small) and a
<clue_id>.json stamped with core.clue_audio.body_hash of the words she read.
The clue page plays the file only while that hash still matches the page.

Costs nothing: no synthesis, only ffmpeg on audio already paid for.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import clue_audio, prose_store, store                   # noqa: E402
from scripts.youtube_assemble import OUT_ROOT, ffmpeg_bin, run    # noqa: E402
from scripts.narrate_prose import clue_text_for, _duration        # noqa: E402

# Below this, after the padding is trimmed, the segment was silent (a clue with
# nothing to say is still given a silent stretch of the video) — no file.
MIN_SPEECH = 1.0

# Trailing silence off, via reverse / strip leading / reverse. The padding is
# exact zeros, so a low threshold finds precisely where she stops.
_TRIM = ("areverse,silenceremove=start_periods=1:start_threshold=-55dB,"
         "areverse,apad=pad_dur=0.4")


def export(cap):
    """Export every speaking clue of one capture dir. Returns report lines."""
    cap = Path(cap)
    man = json.loads((cap / "manifest.json").read_text())
    frames = man["frames"]
    clue_audio.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_bin("ffmpeg")
    data = prose_store.load()
    conn = store.connect()
    done, skipped = 0, []
    try:
        for i, f in enumerate(frames, 1):
            cid = f["clue_id"]
            src = cap / ("narration_%02d.wav" % i)
            if not src.exists():
                skipped.append("%s: no narration file" % f.get("label"))
                continue
            text, why = clue_text_for(f, conn, data)
            if not text:
                skipped.append("%s: %s" % (f.get("label"), why))
                continue
            digest = clue_audio.body_hash(prose_store.approved_text(cid, data),
                                          store.get_note(conn, cid))
            dst = clue_audio.mp3_path(cid)
            tmp = dst.with_suffix(".tmp.mp3")
            run([ff, "-y", "-loglevel", "error", "-i", str(src), "-af", _TRIM,
                 "-ac", "1", "-ar", "24000", "-c:a", "libmp3lame", "-b:a", "48k",
                 str(tmp)])
            secs = _duration(tmp)
            if secs < MIN_SPEECH + 0.4:
                tmp.unlink()
                skipped.append("%s: silent segment" % f.get("label"))
                continue
            tmp.replace(dst)
            clue_audio.write_meta(cid, digest, source=man["source"],
                                  puzzle=str(man["puzzle_number"]),
                                  seconds=round(secs, 2))
            done += 1
    finally:
        conn.close()
    report = ["clue audio: %d of %d clues exported to %s"
              % (done, len(frames), clue_audio.AUDIO_DIR)]
    report += ["  skipped %s" % s for s in skipped]
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=None)
    ap.add_argument("--puzzle", default=None)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    if args.dir:
        cap = Path(args.dir)
    elif args.source and args.puzzle:
        cap = OUT_ROOT / ("%s-%s" % (args.source, args.puzzle))
    else:
        sys.exit("Give --source and --puzzle, or --dir.")
    if not (cap / "manifest.json").exists():
        sys.exit("No manifest.json in %s" % cap)
    for line in export(cap):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
