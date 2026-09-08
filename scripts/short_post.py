"""Post a built Short to YouTube — one button, pressed by a human.

    python scripts/short_post.py --clue-id 10089910 --dry-run
    python scripts/short_post.py --clue-id 10089910

Nothing here decides to publish. The build (scripts/short_build.py) is safe to automate;
this is the step a person triggers, the same shape as the reel's POST TO INSTAGRAM.

It reuses youtube_upload's authenticated service and its upload call — the same account
and the same quota the nightly long-form uploads use, so there is one credential and one
place it can go wrong. An upload costs 1600 of the 10,000 daily units.

WHAT MAKES IT A SHORT: YouTube decides that itself from the file — vertical and under
60 seconds — not from anything we send. The #Shorts tag in the description is a hint for
people, not the mechanism. So a build that came out at 61 seconds would silently post as
an ordinary video; the duration is checked here and refused rather than discovered later.

It is RECORDED in the same ledger as the long videos, with a 'short-' prefix on the
puzzle number so it cannot collide with that puzzle's own film. Without this there is no
record of which clue was shorted on which day, which was a real gap: a hand-posted video
leaves no trace at all.
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "logs" / "youtube"
MAX_SECONDS = 60.0


def _load(name, filename):
    import importlib.util as iu
    spec = iu.spec_from_file_location(name, str(Path(__file__).resolve().parent / filename))
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clue(clue_id):
    from web import create_app
    app = create_app("development")
    with app.app_context():
        from web.db import get_db
        r = get_db().execute(
            "SELECT id, source, puzzle_number, clue_text, answer, enumeration, "
            "       clue_number, direction FROM clues WHERE id = ?", (clue_id,)).fetchone()
    return dict(r) if r else None


def _duration(ya, path):
    r = subprocess.run([ya.ffmpeg_bin("ffprobe"), "-v", "error",
                        "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float((r.stdout or "0").strip())
    except ValueError:
        return 0.0


def title_for(row):
    """The clue itself is the title — it is what a viewer is deciding to watch, and what
    anyone searching for that clue would type. The paper and number follow it."""
    clue = (row["clue_text"] or "").strip().rstrip(".")
    tail = " — %s Cryptic %s" % ((row["source"] or "").title(), row["puzzle_number"])
    room = 100 - len(tail) - len(" #shorts")
    if len(clue) > room:
        clue = clue[:room - 1].rstrip() + "…"
    return "%s%s #shorts" % (clue, tail)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clue-id", type=int, required=True)
    ap.add_argument("--privacy", default="public",
                    choices=["private", "unlisted", "public"])
    ap.add_argument("--dry-run", action="store_true",
                    help="show the title and description, upload nothing")
    args = ap.parse_args(argv)

    row = _clue(args.clue_id)
    if row is None:
        sys.exit("No clue %s." % args.clue_id)
    cap = OUT_ROOT / ("%s-%s" % (row["source"], row["puzzle_number"]))
    # The film is named by CLUE, never by puzzle: one capture directory serves every clue
    # of a puzzle, so a shared filename would let this post the last clue built under
    # this clue's title. Refuse rather than post whatever happens to be lying there.
    video = cap / ("short_%d.mp4" % args.clue_id)
    if not video.exists():
        sys.exit("No short built for clue %s (%s). Build it first — and note a short "
                 "built before this change is not named for its clue, so rebuild it."
                 % (args.clue_id, video.name))

    yu = _load("yu", "youtube_upload.py")
    ya = _load("ya", "youtube_assemble.py")

    secs = _duration(ya, video)
    if secs > MAX_SECONDS:
        sys.exit("The film is %.1fs. Over %.0fs YouTube publishes it as an ordinary "
                 "video, not a Short — rebuild it shorter." % (secs, MAX_SECONDS))

    title = title_for(row)
    desc_file = cap / ("short_%d.txt" % args.clue_id)
    desc = desc_file.read_text(encoding="utf-8") if desc_file.exists() else ""
    tags = ["cryptic crossword", "crossword help", "%s crossword" % row["source"],
            "crossword clue explained"]

    print("clue    %s %s  (%s)" % (row["clue_number"], row["direction"], args.clue_id))
    print("file    %s  %.1fs  %.1f MB" % (video.name, secs, video.stat().st_size / 1e6))
    print("title   %s  (%d chars)" % (title, len(title)))
    print("privacy %s" % args.privacy)
    print("---- description ----")
    print(desc.strip() or "(none)")
    if args.dry_run:
        print("\n[dry-run] nothing uploaded.")
        return 0

    yt = yu.service()
    print("\nUploading…")
    resp = yu.upload(yt, video, title, desc, tags, args.privacy)
    vid = resp["id"]
    url = "https://www.youtube.com/watch?v=%s" % vid

    # Recorded against 'short-<puzzle>' so it cannot collide with the puzzle's own film.
    conn = yu.ledger()
    try:
        yu.record(conn, row["source"], "short-%s" % row["puzzle_number"], vid, title,
                  args.privacy)
    finally:
        conn.close()

    print("\nposted: %s  (%s)" % (url, args.privacy))
    print("recorded in the ledger as %s short-%s" % (row["source"], row["puzzle_number"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
