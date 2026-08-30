"""Read this channel's own YouTube Analytics — where the views actually came from.

WHY THIS EXISTS
---------------
On 2026-08-29 the public view counts showed two videos holding 137 of the channel's
179 views (Times 29628 with 97, Sunday Times 5230 with 40) while the other nineteen
public videos sat between 0 and 6. Age does not explain it — two videos uploaded the
same day as 29628 have 5 and 3.

Studio would answer it, but Studio could not be reached: the browser available was
signed in as the project owner, not the channel account. So the same numbers are pulled
through the API instead, which has the advantage of being repeatable and of covering
every video in one go rather than 21 clicks.

    python scripts/youtube_stats.py                    # totals + per-video views
    python scripts/youtube_stats.py --video qBoMQw9pFSw   # one video's traffic sources
    python scripts/youtube_stats.py --top 5            # traffic sources for the top 5
    python scripts/youtube_stats.py --since 2026-08-01 # window (default: 2026-08-01)

READ-ONLY. This script never writes to YouTube or to any database.

SCOPE
-----
Needs `yt-analytics.readonly`, added to scripts/youtube_auth.py on 2026-08-29. If the
stored token predates that, this exits telling you to re-mint — the token cannot gain a
scope it was not granted, so `python scripts/youtube_auth.py --force` and consent as the
CHANNEL account (justcordelia.com@gmail.com), not the project owner.
"""

import argparse
import sqlite3
import sys
from datetime import date
from pathlib import Path

from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.youtube_auth import load_token, SCOPES          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LEDGER_DB = ROOT / "logs" / "youtube_state.db"

ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"

# The channel's first upload. Nothing exists before this, so it is the natural floor.
DEFAULT_SINCE = "2026-08-21"


def titles():
    """video_id -> "source puzzle" from the upload ledger, for readable output."""
    if not LEDGER_DB.exists():
        return {}
    conn = sqlite3.connect(str(LEDGER_DB))
    try:
        return {r[0]: "%s %s" % (r[1], r[2]) for r in conn.execute(
            "SELECT video_id, source, puzzle_number FROM uploaded_video "
            "WHERE video_id IS NOT NULL")}
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()


def duration_seconds(video_id):
    """The video's length, so a retention point can be read as a timestamp.

    Retention comes back as a FRACTION of the video, which on its own cannot say
    "they left at 40 seconds" — the thing actually worth knowing. Uses the Data API
    (youtube.readonly, already granted), not a guess from the chapter file, because
    the last chapter is not the end of the video.
    """
    creds = load_token()
    yt = build("youtube", "v3", credentials=creds)
    items = yt.videos().list(part="contentDetails", id=video_id).execute().get("items")
    if not items:
        return None
    iso = items[0]["contentDetails"]["duration"]        # e.g. PT6M13S
    import re
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", iso)
    if not m:
        return None
    h, mi, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mi * 60 + s


def chapters_for(video_id):
    """[(seconds, label)] from the chapters.txt the upload was built from, or []."""
    if not LEDGER_DB.exists():
        return []
    conn = sqlite3.connect(str(LEDGER_DB))
    try:
        row = conn.execute("SELECT source, puzzle_number FROM uploaded_video "
                           "WHERE video_id = ?", (video_id,)).fetchone()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    if not row:
        return []
    path = ROOT / "logs" / "youtube" / ("%s-%s" % (row[0], row[1])) / "chapters.txt"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or " " not in line:
            continue
        stamp, label = line.split(" ", 1)
        parts = stamp.split(":")
        if not all(p.isdigit() for p in parts):
            continue
        t = 0
        for p in parts:
            t = t * 60 + int(p)
        out.append((t, label))
    return sorted(out)


def chapter_at(chapters, t):
    """The chapter running at t seconds."""
    label = ""
    for start, name in chapters:
        if start <= t + 0.5:
            label = name
        else:
            break
    return label[:52]


def service():
    creds = load_token()
    if creds is None:
        sys.exit("No token. Run: python scripts/youtube_auth.py")
    granted = set(getattr(creds, "scopes", None) or [])
    if granted and ANALYTICS_SCOPE not in granted:
        sys.exit("The stored token has no analytics scope — it was minted before "
                 "2026-08-29. Re-mint it:\n"
                 "    python scripts/youtube_auth.py --force\n"
                 "and sign in as the CHANNEL account (justcordelia.com@gmail.com).")
    return build("youtubeAnalytics", "v2", credentials=creds)


def query(yta, since, until, metrics, dimensions=None, filters=None, sort=None,
          max_results=None):
    kw = {"ids": "channel==MINE", "startDate": since, "endDate": until,
          "metrics": metrics}
    if dimensions:
        kw["dimensions"] = dimensions
    if filters:
        kw["filters"] = filters
    if sort:
        kw["sort"] = sort
    if max_results:
        kw["maxResults"] = max_results
    return yta.reports().query(**kw).execute()


def table(resp, label_map=None, label_width=34):
    """Print an Analytics response as rows, or say plainly that it is empty.

    An empty `rows` is a real answer ("no views from anything in this window"), not a
    failure, and must not be printed as a blank table that reads like a bug.
    """
    headers = [h["name"] for h in resp.get("columnHeaders", [])]
    rows = resp.get("rows") or []
    if not rows:
        print("  (no data in this window)")
        return
    print("  " + "  ".join(h.ljust(label_width if i == 0 else 12)
                           for i, h in enumerate(headers)))
    for row in rows:
        cells = []
        for i, v in enumerate(row):
            s = str(v)
            if i == 0 and label_map and s in label_map:
                s = "%s  (%s)" % (s, label_map[s])
            cells.append(s.ljust(label_width if i == 0 else 12))
        print("  " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default=DEFAULT_SINCE, help="start date YYYY-MM-DD")
    ap.add_argument("--until", default=None, help="end date; default today")
    ap.add_argument("--video", default=None, help="one video id's traffic sources")
    ap.add_argument("--top", type=int, default=0,
                    help="also break down the N most-viewed videos")
    args = ap.parse_args()

    until = args.until or date.today().isoformat()
    names = titles()

    try:
        yta = service()
    except RefreshError:
        sys.exit("The stored token could not be refreshed. Re-mint it:\n"
                 "    python scripts/youtube_auth.py --force")

    def run(*a, **kw):
        try:
            return query(yta, args.since, until, *a, **kw)
        except HttpError as e:
            body = str(e)
            if "insufficient" in body.lower() or "forbidden" in body.lower():
                sys.exit("Analytics refused the request — the token most likely lacks "
                         "yt-analytics.readonly.\n"
                         "    python scripts/youtube_auth.py --force\n"
                         "and consent as the CHANNEL account.\n\n%s" % body)
            sys.exit("Analytics error: %s" % body)

    print("window: %s .. %s\n" % (args.since, until))

    if args.video:
        print("Traffic sources for %s%s"
              % (args.video, " (%s)" % names[args.video] if args.video in names else ""))
        table(run("views,estimatedMinutesWatched,averageViewDuration",
                  dimensions="insightTrafficSourceType",
                  filters="video==%s" % args.video, sort="-views"))
        print("\nSearch terms that reached it")
        table(run("views", dimensions="insightTrafficSourceDetail",
                  filters="video==%s;insightTrafficSourceType==YT_SEARCH"
                          % args.video,
                  sort="-views", max_results=25))
        print("\nWhere it was suggested from")
        table(run("views", dimensions="insightTrafficSourceDetail",
                  filters="video==%s;insightTrafficSourceType==RELATED_VIDEO"
                          % args.video,
                  sort="-views", max_results=25))
        print("\nExternal sources")
        table(run("views", dimensions="insightTrafficSourceDetail",
                  filters="video==%s;insightTrafficSourceType==EXT_URL" % args.video,
                  sort="-views", max_results=25))

        # Retention curve. audienceWatchRatio is 1.0 where an average viewer watched
        # that slice once; it answers WHERE people leave, which no amount of arguing
        # about presentation style can. Printed coarsely (every 5%) so the shape is
        # readable in a terminal.
        print("\nAudience retention (share still watching)")
        r = run("audienceWatchRatio,relativeRetentionPerformance",
                dimensions="elapsedVideoTimeRatio",
                filters="video==%s;audienceType==ORGANIC" % args.video)
        rows = sorted(r.get("rows") or [], key=lambda x: float(x[0]))
        if not rows:
            print("  (no data — YouTube withholds retention below a view threshold)")
            return 0
        secs = duration_seconds(args.video)
        chaps = chapters_for(args.video)
        if not secs:
            print("  (could not read the video's duration; showing percentages only)")
        prev = None
        for row in rows:
            frac, ratio = float(row[0]), float(row[1])
            t = frac * secs if secs else None
            drop = "" if prev is None else "  %+.2f" % (ratio - prev)
            prev = ratio
            stamp = "%4.0f%%" % (frac * 100)
            if t is not None:
                stamp += "  %2d:%02d" % (int(t) // 60, int(t) % 60)
            label = chapter_at(chaps, t) if (chaps and t is not None) else ""
            print("  %s  %5.2f %-7s %-28s %s"
                  % (stamp, ratio, drop, "#" * int(round(ratio * 26)), label))
        return 0

    print("Channel totals")
    table(run("views,estimatedMinutesWatched,averageViewDuration,subscribersGained"))

    print("\nViews by video")
    by_video = run("views,averageViewDuration", dimensions="video", sort="-views",
                   max_results=50)
    table(by_video, label_map=names)

    print("\nChannel traffic sources")
    table(run("views", dimensions="insightTrafficSourceType", sort="-views"))

    for row in (by_video.get("rows") or [])[:args.top]:
        vid = row[0]
        print("\n--- %s%s" % (vid, "  (%s)" % names[vid] if vid in names else ""))
        table(run("views", dimensions="insightTrafficSourceType",
                  filters="video==%s" % vid, sort="-views"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
