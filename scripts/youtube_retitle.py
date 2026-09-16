"""Bring already-uploaded videos' titles into line with what title_for produces now.

    python scripts/youtube_retitle.py --dry-run      # show what would change
    python scripts/youtube_retitle.py                # retitle today's uploads
    python scripts/youtube_retitle.py --all          # the whole ledger
    python scripts/youtube_retitle.py --video-id XYZ # one video

WHY THIS EXISTS
---------------
Editing youtube_upload.title_for changes what the NEXT videos.insert sends and nothing
else. A title is an argument to the create call (youtube_upload.py:404), not a property
this code owns: once the insert returns, YouTube holds that string in its own database
and never asks us again. So a title fix that is not also pushed with videos.update
leaves every existing video wearing the old title forever. That gap was found the hard
way on 2026-09-16, when a title change was called done while all three of that morning's
uploads still read the old way on the channel.

ONE SOURCE OF TRUTH. The intended title is computed by importing title_for from
youtube_upload — never reimplemented here. Two title builders would drift, and the
drift would show up on the channel rather than in a test.

THE SNIPPET TRAP — READ BEFORE CHANGING THE BODY
------------------------------------------------
videos.update "will override the existing values for all of the mutable properties
that are contained in any parts that the parameter value specifies". A part is
replaced WHOLE. Sending {"snippet": {"title": ...}} therefore does not edit the title
— it wipes the description, the tags and the category of a live video. This script
reads the existing snippet back with videos.list and edits one key of it, and
categoryId is sent explicitly because the API rejects a snippet without it.

`part` is "snippet" alone, deliberately: naming "status" would put the privacy setting
under the same replace-whole rule, and an omitted privacyStatus reverts the video to
the default. Privacy is not this script's business.

SHORTS ARE EXCLUDED. They are titled by scripts/short_post.py:title_for from the clue
itself, and run about twenty seconds. Rewriting one with the puzzle-video title would
have it claim to be a six-minute masterclass. Ledger rows for Shorts carry a
"short-" puzzle_number, which is how they are recognised here.

QUOTA
-----
videos.list costs 1 unit for a batch of up to 50 ids; videos.update costs 50 per video,
against a 10,000/day project ceiling shared with uploads at 1600 each. Retitling a
day's three is ~151 units. The whole ledger of 89 would be ~4,450 — nearly half a day's
quota, which is why --all is not the default and prints its cost before acting.
"""

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from scripts.youtube_upload import (LEDGER_DB, TOKEN_FILE, ledger, title_for)

BATCH = 50               # videos.list takes up to 50 ids for its 1 unit
UPDATE_COST = 50         # per video, documented above


FORCE_SSL = "https://www.googleapis.com/auth/youtube.force-ssl"


def granted_scopes():
    """What the STORED token was actually minted with.

    Read from the token file itself, never from Credentials.scopes. Passing a scope
    list to from_authorized_user_file SETS that attribute rather than verifying it, so
    creds.scopes would report force-ssl as present purely because this script asks for
    it — a check that can only ever say yes is worse than no check.
    """
    try:
        return set(json.loads(TOKEN_FILE.read_text()).get("scopes") or [])
    except Exception:
        return set()


def credentials(need_write):
    """The stored token, refreshed if stale.

    `need_write` is False for --dry-run, which reads live titles with youtube.readonly
    and sends nothing. Refusing to preview on a token that cannot yet write would hide
    the very diff you want before consenting to anything.
    """
    if not TOKEN_FILE.exists():
        sys.exit("No token at %s — run: python scripts/youtube_auth.py" % TOKEN_FILE)
    if need_write and FORCE_SSL not in granted_scopes():
        sys.exit(
            "The stored token predates the youtube.force-ssl scope, so videos.update\n"
            "will be refused. A token does not gain a scope when the constant changes.\n"
            "Re-mint it, signing in as the CHANNEL account (justcordelia.com@gmail.com):\n"
            "    python scripts/youtube_auth.py --force\n"
            "Then re-run this. Use --dry-run meanwhile to see exactly what would change.")
    # NO SCOPES ARGUMENT, and this is not tidiness — it is the difference between a
    # working pipeline and a broken one. Passing a scope list SETS it on the credential
    # rather than checking it, and the refresh then asks Google for whatever was passed.
    # Ask for a scope the user never consented to and the refresh dies with
    # "invalid_scope: Bad Request" — which is exactly what happened here on 2026-09-16,
    # the moment force-ssl was added to the constant while the stored token predated it.
    # Omitting the argument loads the scopes the token actually carries, so a token
    # keeps refreshing no matter what the constant grows to. SCOPES is for MINTING
    # (youtube_auth.mint), never for loading.
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE))
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_FILE.write_text(creds.to_json())
    return creds


def candidates(conn, args):
    """Ledger rows to consider, newest first. Shorts are never included."""
    sql = ("SELECT source, puzzle_number, video_id, title, privacy, uploaded_at "
           "FROM uploaded_video WHERE video_id IS NOT NULL")
    params = []
    if args.video_id:
        sql += " AND video_id = ?"
        params.append(args.video_id)
    elif not args.all:
        sql += " AND date(uploaded_at) = ?"
        params.append(date.today().isoformat())
    if args.source:
        sql += " AND source = ?"
        params.append(args.source)
    sql += " ORDER BY uploaded_at DESC"
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(sql, params)]
    return [r for r in rows if not str(r["puzzle_number"]).startswith("short-")]


def intended_titles(rows):
    """Recompute each row's title through the REAL title_for.

    The ledger stores no type_slug or type_label, so the puzzle has to be classified
    again — by web.models.classify_puzzle, the same function youtube_upload.next_puzzle
    uses (:183). Saturday's prize and Sunday's Toughie get their right names from the
    number and the publication date, so this must not be shortcut.
    """
    from web import create_app
    app = create_app("development")
    out = {}
    with app.app_context():
        from web.db import get_db
        from web.models import classify_puzzle
        for r in rows:
            src, num = r["source"], str(r["puzzle_number"])
            pub = get_db().execute(
                "SELECT MAX(publication_date) AS pub FROM clues "
                "WHERE source = ? AND puzzle_number = ?", (src, num)).fetchone()["pub"]
            slug, label = classify_puzzle(src, num, pub)
            if slug is None:
                print("  skip %s %s — unclassifiable, no title can be built" % (src, num))
                continue
            puzzle = {"number": num, "pub": pub, "type_slug": slug, "type_label": label}
            out[r["video_id"]] = title_for(src, puzzle)
    return out


def live_snippets(yt, video_ids):
    """The current snippet of each video, by id. One unit per batch of 50."""
    snippets = {}
    for i in range(0, len(video_ids), BATCH):
        chunk = video_ids[i:i + BATCH]
        resp = yt.videos().list(part="snippet", id=",".join(chunk)).execute()
        for item in resp.get("items", []):
            snippets[item["id"]] = item["snippet"]
    return snippets


def retitle(yt, video_id, snippet, new_title):
    """Replace ONE key of the existing snippet and send the whole thing back.

    Every field that came from videos.list is preserved deliberately. See the module
    docstring: an omitted field inside `part` is not left alone, it is reset.
    """
    body = dict(snippet)
    body["title"] = new_title
    if not body.get("categoryId"):
        raise RuntimeError("video %s has no categoryId; refusing to send a snippet "
                           "the API would reject" % video_id)
    yt.videos().update(part="snippet", body={"id": video_id, "snippet": body}).execute()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="show every change and send nothing")
    ap.add_argument("--all", action="store_true",
                    help="the whole ledger, not just today's uploads")
    ap.add_argument("--source", help="telegraph | times | guardian")
    ap.add_argument("--video-id", help="one video, by its YouTube id")
    args = ap.parse_args()

    conn = ledger()
    rows = candidates(conn, args)
    if not rows:
        scope = "the whole ledger" if args.all else "today (%s)" % date.today().isoformat()
        print("Nothing to consider in %s." % scope)
        return 0

    print("Considering %d video(s)." % len(rows))
    wanted = intended_titles(rows)
    if not wanted:
        return 0

    creds = credentials(need_write=not args.dry_run)
    yt = build("youtube", "v3", credentials=creds)
    snippets = live_snippets(yt, list(wanted))

    changes = []
    for r in rows:
        vid = r["video_id"]
        new = wanted.get(vid)
        snip = snippets.get(vid)
        if new is None:
            continue
        if snip is None:
            print("  skip %s — not returned by the API (deleted, or not ours)" % vid)
            continue
        if snip.get("title") == new:
            print("  ok   %s — already correct" % vid)
            continue
        changes.append((r, snip, new))

    if not changes:
        print("\nEvery title already matches. Nothing sent.")
        return 0

    print("\n%d title(s) to change — %d quota units:"
          % (len(changes), len(changes) * UPDATE_COST))
    for r, snip, new in changes:
        print("\n  %s (%s %s)" % (r["video_id"], r["source"], r["puzzle_number"]))
        print("    was: %s" % snip.get("title"))
        print("    now: %s" % new)

    if args.dry_run:
        print("\n[dry-run] nothing sent, ledger unchanged.")
        return 0

    print()
    failed = 0
    for r, snip, new in changes:
        try:
            retitle(yt, r["video_id"], snip, new)
        except HttpError as e:
            failed += 1
            body = str(e)
            if "insufficientPermissions" in body or "insufficient" in body.lower():
                print("FAILED %s: the token lacks youtube.force-ssl. Re-mint it:\n"
                      "    python scripts/youtube_auth.py --force" % r["video_id"])
                break
            if "quotaExceeded" in body:
                print("FAILED %s: daily API quota exhausted (an update costs %d)."
                      % (r["video_id"], UPDATE_COST))
                break
            print("FAILED %s: %s" % (r["video_id"], body))
            continue
        # The ledger records what the video is ACTUALLY called now, or it becomes a
        # readout that is quietly false — the same fault record() was fixed for (:108).
        conn.execute("UPDATE uploaded_video SET title = ? WHERE video_id = ?",
                     (new, r["video_id"]))
        conn.commit()
        print("retitled %s -> %s" % (r["video_id"], new))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
