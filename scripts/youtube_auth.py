"""Mint and verify the YouTube upload credential.

The channel is owned by a DIFFERENT Google account (justcordelia.com@gmail.com)
from the one that owns the Cloud project (cordelia-493208). That is fine: the
OAuth client belongs to the project, but the account that CONSENTS decides which
channel the token controls. So sign in as the CHANNEL account when the browser
opens, not the project owner.

    python scripts/youtube_auth.py            # mint if needed, then verify
    python scripts/youtube_auth.py --verify   # verify only, never opens a browser
    python scripts/youtube_auth.py --force    # discard the stored token and re-mint

Files (impressions/ is gitignored — see .gitignore:75):
    impressions/credentials.json     the project's installed-app OAuth client
                                     (shared with search_console_report.py — READ ONLY here)
    impressions/youtube_token.json   this script's own token. Deliberately NOT
                                     token.json, which is Search Console's.
"""

import argparse
import sys
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

ROOT = Path(__file__).resolve().parent.parent
CREDS_FILE = ROOT / "impressions" / "credentials.json"
TOKEN_FILE = ROOT / "impressions" / "youtube_token.json"

# EXACTLY what the code calls, and no more. youtube.upload inserts the video
# (youtube_upload.py:273); youtube.readonly reads our own channel back to prove which
# channel the token controls (:74); yt-analytics.readonly reads this channel's OWN
# traffic-source and view reports (scripts/youtube_stats.py). The full `youtube` scope
# was requested until 2026-08-22 and nothing ever used it.
#
# This is a VERIFICATION requirement, not tidiness. Google's OAuth review states that
# "if the requested scope(s) goes beyond the usage needed, you will be directed to
# request a narrower scope" (support.google.com/cloud/answer/13464321), and a round
# trip through review costs days. Before widening this, read
# documents/YOUTUBE_OAUTH_VERIFICATION_SUBMISSION.md §2 — the cost of a wider scope is
# paid at review, not here.
#
# yt-analytics.readonly was added 2026-08-29 to answer a question Studio could not be
# reached to answer: WHY two videos hold 137 of the channel's 179 views. It is read-only
# and reports on our own channel. NOTE the review caveat above was flagged to the user
# and accepted; if a re-consent is refused, this is the first thing to revert.
#
# The consequence to know: videos.update is still NOT covered. Editing an
# already-uploaded video's title or description remains a Studio job, by hand.
#
# KEEP IN SYNC with scripts/youtube_upload.py — it defines its own copy and loads the
# same token file with it.
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


def load_token():
    """The stored credential, refreshed if it has expired. None if absent."""
    if not TOKEN_FILE.exists():
        return None
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_FILE.write_text(creds.to_json())
    return creds


def mint():
    """Open a browser for consent and store the resulting token."""
    if not CREDS_FILE.exists():
        sys.exit("No OAuth client at %s" % CREDS_FILE)
    print("A browser will open. Sign in as the CHANNEL account "
          "(justcordelia.com@gmail.com) — NOT the project owner.")
    flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_FILE.write_text(creds.to_json())
    print("Token written to %s" % TOKEN_FILE)
    return creds


def verify(creds):
    """Prove the token works AND says which channel it controls.

    A token that authenticates but points at the wrong channel is the failure
    mode worth catching here — it would upload to the project owner's personal
    channel without complaint.
    """
    yt = build("youtube", "v3", credentials=creds)
    resp = yt.channels().list(part="snippet,contentDetails,status",
                              mine=True).execute()
    items = resp.get("items") or []
    if not items:
        print("FAIL: the token authenticates but owns no channel.")
        return False
    ch = items[0]
    print("Channel:   %s" % ch["snippet"]["title"])
    print("Handle:    %s" % ch["snippet"].get("customUrl", "(none)"))
    print("Channel id: %s" % ch["id"])
    uploads = ch["contentDetails"]["relatedPlaylists"].get("uploads")
    print("Uploads playlist: %s" % uploads)
    if "status" in ch:
        print("Long uploads allowed: %s"
              % ch["status"].get("longUploadsStatus", "(not reported)"))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="verify the stored token only; never open a browser")
    ap.add_argument("--force", action="store_true",
                    help="discard the stored token and mint a new one")
    args = ap.parse_args()

    if args.force and TOKEN_FILE.exists():
        TOKEN_FILE.unlink()
        print("Discarded %s" % TOKEN_FILE.name)

    creds = load_token()
    if creds is None:
        if args.verify:
            sys.exit("No token at %s — run without --verify to mint one." % TOKEN_FILE)
        creds = mint()

    try:
        ok = verify(creds)
    except HttpError as e:
        # The two failures worth naming, because they need different fixes.
        body = str(e)
        if "has not been used in project" in body or "accessNotConfigured" in body:
            print("FAIL: YouTube Data API v3 is NOT enabled on project "
                  "cordelia-493208. Enable it in the console as the PROJECT "
                  "OWNER account, then re-run.")
        elif "quotaExceeded" in body:
            print("FAIL: the project's YouTube API quota is exhausted for today.")
        else:
            print("FAIL: %s" % body)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
