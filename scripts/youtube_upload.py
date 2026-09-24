"""Upload an assembled puzzle video to the channel — each puzzle exactly ONCE.

    python scripts/youtube_upload.py --dry-run       # show what would be uploaded
    python scripts/youtube_upload.py                 # upload (PRIVATE by default)
    python scripts/youtube_upload.py --build         # capture + assemble first
    python scripts/youtube_upload.py --privacy public

THE LEDGER
----------
logs/youtube_state.db, table `uploaded_video`, one row per (source, puzzle_number).
Deliberately the same shape as the IndexNow ledger (scripts/indexnow_notify.py:56-58)
and for the same reason, stated there at :8-16: a puzzle is only ever deployed once
every clue is solved, so a puzzle is a complete, stable unit and "have we done this
one?" is the only question worth asking. Recorded ONLY on a successful upload, so a
failure retries next run and nothing is ever half-announced.

WEEKDAY, SATURDAY, SUNDAY
-------------------------
Not a calendar rule. `web/models.classify_puzzle` (web/models.py:136-157) already
decides it from the puzzle number and its publication date:

    telegraph 31xxx on a Saturday  -> prize          "Prize Cryptic"
    telegraph 31xxx otherwise      -> cryptic        "Cryptic"
    telegraph 3000-3999            -> prize          "Prize Cryptic"
    telegraph 1-2999               -> prize-toughie  "Prize Toughie"  (Sunday, weekly)

So the type is read off the puzzle, not off today's date. That matters because a
catch-up upload of Saturday's prize on a Tuesday still gets the right title, and
because Times and Guardian classify by their own ranges with no extra code here.
Titles and descriptions differ by that type; nothing else does.

PRIVACY
-------
Defaults to PRIVATE. Publishing to a public channel is not something to do by
accident, and an unattended first run should not be the thing that makes the
channel live. Pass --privacy public deliberately.

QUOTA — READ THIS BEFORE SCALING UP
-----------------------------------
A videos.insert costs 1600 units against a default 10,000/day project quota, so
roughly SIX uploads a day, total, for the whole project. Three publications plus
a Short each is already at the ceiling. Verify the project's actual quota before
adding publications; this is the first thing that will break silently.
"""

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# The site's own puzzle naming, so a video and its puzzle page target the same
# strings. See puzzle_reference() for why that matters.
from web.routes.clue_seo import puzzle_seo_name

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "logs" / "youtube"
LEDGER_DB = ROOT / "logs" / "youtube_state.db"

# How many videos one source may upload in a single run. A paper publishing two
# puzzles in a day is normal (Sunday: Prize Toughie + Prize Cryptic); a paper
# publishing four is not, and an upload costs 1600 of the 10,000 daily quota
# units, so this stops a runaway sweep from spending the day's quota on one
# paper. What it holds back is still served and unfilmed — --backfill takes it.
PER_SOURCE_CAP = 3
TOKEN_FILE = ROOT / "impressions" / "youtube_token.json"

# Must stay identical to youtube_auth.py:36 — the token is minted there and loaded
# here, and a mismatch is a confusing runtime failure rather than a clear one. See
# that file for why these two and not the full `youtube` scope.
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    # Neither of these is used by THIS script. They are present because it loads the
    # SAME token file, and the list must match scripts/youtube_auth.py or the stored
    # credential looks wrong. yt-analytics belongs to youtube_stats.py; force-ssl
    # belongs to youtube_retitle.py, which fixes titles this script has already sent.
    #
    # THIS LIST IS DECLARATIVE. It says what a token must be MINTED with
    # (youtube_auth.mint). It is never passed when loading one — see service() for
    # what that costs.
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

CATEGORY_EDUCATION = "27"
TITLE_MAX = 100          # YouTube's limit; a longer title is rejected outright
DESC_MAX = 4900          # YouTube's cap is 5000; the margin is for the emoji-free
                         # ASCII assumption and for a title line YouTube may prepend
SITE = "https://justcordelia.com"

SOURCE_NAMES = {"telegraph": "Telegraph", "times": "Times", "guardian": "Guardian"}


# --- ledger -------------------------------------------------------------------------

def ledger():
    LEDGER_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LEDGER_DB))
    conn.execute("CREATE TABLE IF NOT EXISTS uploaded_video ("
                 "source TEXT NOT NULL, puzzle_number TEXT NOT NULL, "
                 "video_id TEXT, title TEXT, privacy TEXT, uploaded_at TEXT, "
                 "PRIMARY KEY (source, puzzle_number))")
    conn.commit()
    return conn


def already_done(conn):
    return {(r[0], r[1]) for r in
            conn.execute("SELECT source, puzzle_number FROM uploaded_video")}


def record(conn, source, number, video_id, title, privacy):
    """Upsert, not INSERT OR IGNORE.

    With IGNORE, a --force re-upload left the ledger pointing at the SUPERSEDED
    video while printing "recorded" — a readout that was quietly false. The row
    must name the video that actually exists now.
    """
    conn.execute(
        "INSERT INTO uploaded_video "
        "(source, puzzle_number, video_id, title, privacy, uploaded_at) "
        "VALUES (?, ?, ?, ?, ?, datetime('now')) "
        "ON CONFLICT(source, puzzle_number) DO UPDATE SET "
        "video_id=excluded.video_id, title=excluded.title, "
        "privacy=excluded.privacy, uploaded_at=excluded.uploaded_at",
        (source, str(number), video_id, title, privacy))
    conn.commit()


# --- what to upload -----------------------------------------------------------------

def next_puzzle(source, done, max_age_days):
    """The most recent fully-served puzzle of `source` not yet uploaded, published
    within `max_age_days` of TODAY. Pass max_age_days=None to lift the guard
    entirely (--backfill).

    Uses the app's own serving truth, so a puzzle can never be filmed before every
    one of its clues has passed review (web/serving.py:143-171).

    THE AGE GUARD EXISTS BECAUSE OF A REAL MISTAKE (2026-08-21). Without it, "the
    most recent puzzle not yet in the ledger" walks backwards through the whole
    served archive — a second run uploaded the previous day's puzzle unbidden. On
    a deploy hook that is a silent quota drain at 1600 units an upload against a
    10,000/day project ceiling.

    IT IS MEASURED AGAINST TODAY, AND THAT IS THE WHOLE POINT (fixed 2026-08-23).
    It used to be measured against each source's own newest SERVED puzzle, on the
    reasoning that a deploy after a quiet weekend should still find something. That
    reasoning was wrong, and it cost real quota: a deploy of Sunday's Telegraph also
    filmed Times 29627 (published the 21st) and Guardian 30090 (the 20th), because
    relative to each of THOSE papers' own newest served puzzle they looked recent.
    The user's verdict was blunt and correct — nobody wants a days-old puzzle, and
    the step exists to film what was just published, not to go hunting the archive.

    Finding NOTHING is the right answer when nothing was published today. A backlog
    is filmed deliberately with --backfill, never as a side effect of a deploy.
    """
    from datetime import date, timedelta
    from web import create_app
    app = create_app("development")
    with app.app_context():
        from web.db import get_db
        from web.serving import served_puzzle_numbers
        from web.models import classify_puzzle
        served = served_puzzle_numbers([source])
        rows = get_db().execute(
            "SELECT puzzle_number, MAX(publication_date) AS pub FROM clues "
            "WHERE source = ? AND puzzle_number IS NOT NULL "
            "GROUP BY puzzle_number ORDER BY pub DESC", (source,)).fetchall()
        served_rows = [r for r in rows if (source, str(r["puzzle_number"])) in served]
        if not served_rows:
            return None
        newest = max((r["pub"] or "") for r in served_rows)
        # None lifts the guard (--backfill). 0 means today only — and 0 must NOT
        # be treated as "no guard", which is what the old truthiness test did.
        cutoff = None
        if max_age_days is not None:
            cutoff = (date.today() - timedelta(days=max_age_days)).isoformat()
        skipped_old = 0
        for r in served_rows:
            num = str(r["puzzle_number"])
            if (source, num) in done:
                continue
            if cutoff and (r["pub"] or "") < cutoff:
                skipped_old += 1
                continue
            slug, label = classify_puzzle(source, num, r["pub"])
            if slug is None:
                continue            # unclassifiable: no title could be built for it
            return {"number": num, "pub": r["pub"], "type_slug": slug,
                    "type_label": label}
        if skipped_old:
            # Never silent: say what the guard held back, so a backlog is a
            # visible decision rather than something that quietly never happens.
            print("%d older %s puzzle(s) are served and unfilmed, held back by the "
                  "%d-day age guard (cutoff %s, newest served %s). This is the guard "
                  "working: use --backfill to film them deliberately."
                  % (skipped_old, source, max_age_days, cutoff, newest))
    return None


def build_video(source, number, narrate=True):
    """Run capture then assemble for this puzzle.

    NARRATED BY DEFAULT. Cordelia reads the prose the user ticked on /hs, clue by
    clue (scripts/narrate_prose.py). The default is on because the fallback is
    today's video, not a worse one: a clue with no tick is simply silent, and a
    puzzle with nothing ticked at all falls straight back to the spoken intro over
    a silent body — which is exactly what this built before. --no-narrate is there
    for a deliberately silent film, not as a safety net.
    """
    py = sys.executable
    assemble = ["--source", source, "--puzzle", number] + (["--narrate"] if narrate else [])
    for script, extra in (("youtube_capture.py", ["--source", source, "--puzzle", number]),
                          ("youtube_assemble.py", assemble)):
        print("--- %s ---" % script)
        r = subprocess.run([py, str(ROOT / "scripts" / script)] + extra,
                           cwd=str(ROOT), text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            sys.exit("%s failed (exit %d)" % (script, r.returncode))


# --- title and description ----------------------------------------------------------

def puzzle_reference(source, puzzle):
    """"DT 31330 — Telegraph Cryptic 31330" — the site's own naming.

    The abbreviation LEADS. It is the string solvers type (measured below), so it
    goes first — it used to sit last, in brackets, on the site and here alike.

    The names and the abbreviation come from `_PUZZLE_SEO_NAMES`
    (web/routes/clue_seo.py:372), which is what justcordelia.com already puts in
    its puzzle-page <title>. There is no second map here on purpose: two maps
    drift, and then the video and the page target different strings for the same
    puzzle.

    Why the abbreviation has to appear at all — measured 2026-08-29 against
    Google's own suggestion engine, which is where this demand lives:

        dt 313      -> dt 31301, 31307, 31305, 31308, 31300, 31302, 31303,
                       31310, 31304, 31306
        toughie 37  -> toughie 3728, 3700, 3717, 3720, 3702, 3716, 3708, ...
        everyman 41 -> everyman 4153, 4160, 4159, 4154, 4152, 4156, ...
        times 296   -> times 29610, 29604, 29609, 29603, 29607, ...

    The same prefixes at YouTube return nothing at all, so this title is aimed
    at Google, not at YouTube search. "DT" previously lived only in tags_for,
    and Google has not read a keywords meta tag since 2009 — so the exact string
    a solver types appeared nowhere that could count.
    """
    type_slug = puzzle["type_slug"]
    name, abbr = puzzle_seo_name(source, type_slug, puzzle["type_label"])
    n = puzzle["number"]
    if abbr:
        return "%s %s — %s %s" % (abbr, n, name.removesuffix(" Crossword"), n)
    return "%s %s" % (name, n)


def title_for(source, puzzle):
    """Puzzle number early and exact — the searches this exists to catch are
    "telegraph cryptic 31324" and "DT 31324", not anything about wordplay —
    then the two things that decide whether a human clicks: what it is, and
    what it costs them.

    Positioned 2026-09-16 against Cracking The Cryptic, whose Times cryptic
    masterclass runs over an hour and takes thousands of views. Their title
    carries a DATE and no puzzle number: with ~670k subscribers they do not
    depend on search. We do, so the number keeps its place at the front for
    the Google-suggest demand measured in puzzle_reference(), and "Masterclass"
    and the six minutes are aimed at the person reading the result.

    "Crossword" is dropped from the reference HERE ONLY. It is redundant beside
    "Cryptic" and it is the cheapest word to spend on the hook. puzzle_reference()
    itself is left alone deliberately: the description and the site's own puzzle
    <title> go on matching each other, and two namings that drift is exactly the
    failure that function's docstring exists to prevent.

    Six minutes is a FLAT claim, not a measurement. These run about six minutes
    (12s a clue against a ~30-clue puzzle) and the point being made is against an
    hour, so rounding a 6.6-minute video up to "7" would be a precision nobody
    asked for and a different number on every video.
    """
    ref = puzzle_reference(source, puzzle).replace(" Crossword ", " ")
    t = "%s Masterclass: Every Clue Explained in 6 Minutes" % ref
    if len(t) > TITLE_MAX:
        t = "%s Masterclass: Explained in 6 Minutes" % ref
    return t[:TITLE_MAX]


def tags_for(source, puzzle):
    paper = SOURCE_NAMES.get(source, source.title())
    n = puzzle["number"]
    tags = [
        "%s %s" % (paper.lower(), n),
        "%s cryptic %s" % (paper.lower(), n),
        "cryptic crossword %s" % n,
        "%s crossword answers" % paper.lower(),
        "cryptic crossword explained",
        "cryptic crossword help",
    ]
    _, abbr = puzzle_seo_name(source, puzzle["type_slug"], puzzle["type_label"])
    if abbr:
        tags += ["%s %s" % (abbr.lower(), n), "%s cryptic %s" % (abbr.lower(), n)]
    return tags[:20]


def puzzle_url(source, puzzle):
    """The puzzle page on the site — /<source>/<type_slug>/<number>, the shape
    web/routes/browse.py serves and the section pages link to."""
    return "%s/%s/%s/%s" % (SITE, source, puzzle["type_slug"], puzzle["number"])


def clue_links(cap_dir):
    """One "23a — https://…/clue/<slug>" line per clue, in clue order.

    The slug comes from the capture manifest, which took it from the site itself,
    so there is no second slug builder to drift from web/routes/clue.py:286.
    Returns [] if the manifest is missing or unreadable — a link block is a bonus,
    never a reason to fail an upload.
    """
    try:
        man = json.loads((cap_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for f in man.get("frames", []):
        slug = f.get("slug")
        if not slug:
            continue
        d = (f.get("direction") or "")[:1] or ""
        out.append("%s%s %s/clue/%s" % (f.get("number", ""), d, SITE, slug))
    return out


def description_for(cap_dir, source, puzzle, title):
    """The assembled description: intro, the puzzle link, the chapters, then a link
    per clue.

    Chapters are why this is worth doing: YouTube surfaces them as separate 'key
    moments' entries in Google video results. They carry the clue text, never the
    answer.

    THE LINKS ARE FOR DISCOVERY, and the honest limit is worth stating where the code
    is: YouTube marks description links nofollow and wraps them in a redirect, so no
    ranking equity passes. What they do is put ~30 same-day URLs on a page Google
    crawls within the hour, while the site's own Discovery crawl sits under 1%
    (Crawl Stats, 2026-09-03). They also give a viewer who found the video one click
    to the clue instead of "go and search", which is what the bare brand mention this
    replaces asked of them.

    ORDER MATTERS. Chapters must survive intact or YouTube renders NO chapters at
    all, so the clue links go LAST and are the only thing trimmed. Lines are added
    whole — the old blunt [:4900] would have cut a URL in half, which is exactly the
    silent-wrong-output failure this pipeline keeps producing.
    """
    chapters = (cap_dir / "chapters.txt").read_text(encoding="utf-8").rstrip("\n")
    # Same reference as the title, so the abbreviation form is in the description
    # too — where Google reads it, unlike the keywords tag.
    head = [
        "%s: every clue explained — the definition, the wordplay broken into "
        "pieces, and how those pieces build the answer."
        % puzzle_reference(source, puzzle),
        "",
        "Published %s." % puzzle["pub"],
        "",
        # Above the "…more" fold, so a viewer sees it without expanding.
        "All the answers, explained: %s" % puzzle_url(source, puzzle),
        "",
        "Every clue explained, plus a pattern finder, anagram solver and thesaurus:"
        " %s" % SITE,
        "",
        "Chapters:",
        chapters,
    ]
    text = "\n".join(head)
    links = clue_links(cap_dir)
    if links:
        block = ["", "Every clue on the site:"]
        for ln in links:
            candidate = "\n".join([text] + block + [ln])
            if len(candidate) > DESC_MAX:
                break                       # whole lines only; never a half URL
            block.append(ln)
        if len(block) > 2:                  # at least one link survived the budget
            text = "\n".join([text] + block)
    assert len(text) <= DESC_MAX, "description %d > %d" % (len(text), DESC_MAX)
    return text


# --- upload -------------------------------------------------------------------------

def service():
    if not TOKEN_FILE.exists():
        sys.exit("No token at %s — run scripts/youtube_auth.py first." % TOKEN_FILE)
    # NO SCOPES ARGUMENT. Passing a scope list SETS it on the credential instead of
    # checking it, and the refresh then asks Google for whatever was passed — so a
    # scope added to the constant but not yet consented to kills the refresh with
    # "invalid_scope: Bad Request". That is not hypothetical: it happened on
    # 2026-09-16, the moment force-ssl was added to SCOPES above, and it would have
    # taken out the 04:00 deploy upload rather than anything anyone was watching.
    # Omitting the argument loads the scopes the token actually carries, so uploads
    # keep working while the constant grows ahead of the stored credential.
    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE))
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_FILE.write_text(creds.to_json())
    return build("youtube", "v3", credentials=creds)


def upload(yt, video_path, title, description, tags, privacy):
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": CATEGORY_EDUCATION,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(video_path), chunksize=4 * 1024 * 1024,
                            resumable=True, mimetype="video/mp4")
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = req.next_chunk()
        if status:
            print("  %d%%" % int(status.progress() * 100))
    return response


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=None,
                    help="one source; omit to do them all (see --sources)")
    ap.add_argument("--sources", default="telegraph,times,guardian",
                    help="sources to run when --source is not given. Each is "
                         "handled independently: a source with nothing served and "
                         "unfilmed today simply reports nothing to do. There is no "
                         "day-of-week rule — Saturday's Guardian prize, Sunday's "
                         "Everyman and the Times Sunday all fall out of "
                         "classify_puzzle on their own.")
    ap.add_argument("--puzzle", default=None, help="override the puzzle chosen")
    ap.add_argument("--privacy", default="private",
                    choices=["private", "unlisted", "public"])
    ap.add_argument("--build", action="store_true",
                    help="run capture + assemble before uploading")
    ap.add_argument("--no-narrate", action="store_true",
                    help="film SILENT — do not read the approved prose over the "
                         "clues. The narration is on by default; an unticked clue "
                         "is silent on its own, so this is only for a film you want "
                         "silent throughout.")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the title, chapters and file; upload nothing")
    ap.add_argument("--max-age-days", type=int, default=0,
                    help="only consider puzzles published within N days of TODAY. "
                         "Default 0 = today's puzzles only, which is what a deploy "
                         "wants: film what was just published, never go hunting the "
                         "archive. Nothing published today means nothing is filmed, "
                         "and that is the correct outcome.")
    ap.add_argument("--force", action="store_true",
                    help="upload even though the ledger says this puzzle is done — "
                         "for a rebuilt video. YouTube cannot replace a video's "
                         "content, so this creates a SECOND video; delete the old "
                         "one by hand. Costs another 1600 quota units.")
    ap.add_argument("--backfill", action="store_true",
                    help="lift the age guard and take the oldest outstanding puzzle "
                         "too — deliberate archive work, watch the quota")
    args = ap.parse_args()

    if args.source is None and not args.puzzle:
        # Every source, one after another. Each failure is contained: a source
        # that errors must not stop the others, or one bad puzzle costs the whole
        # day's uploads.
        rc = 0
        srcs = [s.strip() for s in args.sources.split(",") if s.strip()]
        for i, src in enumerate(srcs):
            print("\n=== %s (%d of %d) ===" % (src, i + 1, len(srcs)))
            args.source = src
            rc |= run_source(args)
        return rc
    args.source = args.source or "telegraph"
    return run_source(args)


def run_source(args):
    """Every puzzle this source published today, not only the first.

    A paper can publish TWO puzzles in a day: Sunday 2026-09-20 brought Telegraph
    Prize Toughie 243 and Prize Cryptic 3387. Taking one per run filmed 243 and
    left 3387 behind, and the next day the age guard (today only) put it out of
    reach of every normal run — so the second puzzle was never filmed unless
    somebody happened to press Deploy twice. Draining the source is the fix; a
    second press was a workaround.

    run_one reports what it did on args._outcome, because its return code cannot:
    it returns 0 both for "uploaded" and for "nothing to upload". A dry run also
    keeps going (that is how you see the whole day's list without spending quota),
    with run_one adding each puzzle it has already described to args._skip.

    A failure stops THIS source and is reported, exactly as before — the sweep in
    main() carries on to the next paper.
    """
    rc = 0
    args._skip = set()
    for _ in range(PER_SOURCE_CAP):
        args._outcome = "nothing"
        try:
            rc |= run_one(args)
        except SystemExit as e:
            print("%s FAILED: %s" % (args.source, e))
            return 1
        except Exception as e:
            print("%s FAILED: %s" % (args.source, e))
            return 1
        if args._outcome not in ("uploaded", "dry"):
            return rc
    print("%s: stopping at %d videos in one run (quota guard) — anything left is "
          "still served and unfilmed, so --backfill can take it."
          % (args.source, PER_SOURCE_CAP))
    return rc


def run_one(args):
    conn = ledger()
    # _skip holds what a dry run has already described this sweep — the ledger
    # cannot, because a dry run records nothing. Without it a dry sweep would
    # name the same puzzle until the cap.
    done = already_done(conn) | getattr(args, "_skip", set())

    if args.puzzle:
        from web import create_app
        app = create_app("development")
        with app.app_context():
            from web.db import get_db
            from web.models import classify_puzzle
            row = get_db().execute(
                "SELECT MAX(publication_date) AS pub FROM clues "
                "WHERE source = ? AND puzzle_number = ?",
                (args.source, args.puzzle)).fetchone()
            slug, label = classify_puzzle(args.source, args.puzzle,
                                          row["pub"] if row else None)
        puzzle = {"number": args.puzzle, "pub": row["pub"] if row else None,
                  "type_slug": slug, "type_label": label}
    else:
        # None, not 0 — 0 now means "today only", so --backfill must pass None to
        # lift the guard. Passing 0 here was the old bug in miniature.
        puzzle = next_puzzle(args.source, done,
                             None if args.backfill else args.max_age_days)
        if puzzle is None:
            print("Nothing to upload for %s." % args.source)
            args._outcome = "nothing"
            return 0

    key = (args.source, str(puzzle["number"]))
    if key in done and args.force:
        print("ledger says %s %s is done — --force given, uploading again as a "
              "NEW video." % key)
        done = done - {key}
    if key in done and not args.dry_run:
        print("%s %s already uploaded — nothing to do." % key)
        args._outcome = "already"
        return 0

    cap_dir = OUT_ROOT / ("%s-%s" % (args.source, puzzle["number"]))
    video = cap_dir / "video.mp4"
    # A dry run must not spend minutes capturing and encoding — it is asked
    # "what would you do", not "do the expensive half of it".
    if not video.exists() and args.dry_run and not args.build:
        print("\n%s #%s (%s, %s)"
              % (args.source, puzzle["number"], puzzle["type_label"], puzzle["pub"]))
        print("title:   %s" % title_for(args.source, puzzle))
        print("[dry-run] no video built yet; would run capture + assemble first.")
        args._outcome = "dry"
        args._skip.add(key)
        return 0
    if args.build or not video.exists():
        build_video(args.source, puzzle["number"], narrate=not args.no_narrate)
    if not video.exists():
        sys.exit("No video at %s" % video)

    title = title_for(args.source, puzzle)
    description = description_for(cap_dir, args.source, puzzle, title)
    tags = tags_for(args.source, puzzle)

    print("\n%s #%s (%s, %s)"
          % (args.source, puzzle["number"], puzzle["type_label"], puzzle["pub"]))
    print("title:   %s  (%d chars)" % (title, len(title)))
    print("file:    %s (%.1f MB)" % (video.name, video.stat().st_size / 1e6))
    print("privacy: %s" % args.privacy)
    print("tags:    %s" % ", ".join(tags[:6]))
    n_chapters = sum(1 for ln in (cap_dir / "chapters.txt")
                     .read_text(encoding="utf-8").splitlines() if ln.strip())
    print("description: %d chars, %d chapters" % (len(description), n_chapters))
    if n_chapters < 3:
        sys.exit("Only %d chapter(s) — YouTube needs at least 3 to render any."
                 % n_chapters)

    if args.dry_run:
        print("\n[dry-run] nothing uploaded, ledger unchanged.")
        args._outcome = "dry"
        args._skip.add(key)
        return 0

    yt = service()
    print("\nUploading...")
    try:
        resp = upload(yt, video, title, description, tags, args.privacy)
    except HttpError as e:
        body = str(e)
        if "quotaExceeded" in body:
            print("FAILED: daily API quota exhausted (an upload costs 1600 of 10,000). "
                  "Not recorded — it will retry next run.")
        elif "youtubeSignupRequired" in body:
            print("FAILED: the authorised account has no YouTube channel.")
        else:
            print("FAILED: %s" % body)
        return 1

    vid = resp.get("id")
    record(conn, args.source, puzzle["number"], vid, title, args.privacy)
    print("\nuploaded: https://www.youtube.com/watch?v=%s  (%s)" % (vid, args.privacy))
    print("recorded in ledger — this puzzle will not be uploaded again.")
    args._outcome = "uploaded"
    return 0


if __name__ == "__main__":
    sys.exit(main())
