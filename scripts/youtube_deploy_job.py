"""The deploy's YouTube step as a JOB THAT OUTLIVES THE PAGE.

    python scripts/youtube_deploy_job.py --privacy public

Why this exists
---------------
The dashboard used to film and upload inside the Streamlit page run: one
subprocess per paper, twenty-minute cap each, output captured and shown only
when that paper finished. Two things went wrong with that, both on 2026-09-20.

  1. Nothing was visible while it worked. A paper takes minutes — ~30 clue pages
     captured in headless Chrome, an ffmpeg encode, then the upload — and the
     page showed a spinner for all of it. The user could not tell a working run
     from a hung one: "I need to know it is running and doing something!"

  2. Work in the page dies with the page. Guardian 4170 produced nothing at all
     that morning — no capture directory, no error line — while running the same
     command by hand worked first time and built a 38 MB video. Anything that
     ends the page run (tab closed, refresh, navigation, a Streamlit restart)
     takes the unfinished papers with it, silently.

So the work moves OUT of the page. The dashboard launches this detached, with
its output going to a log file; the page tails that log. Closing the browser now
costs nothing, and the log is the record either way.

The sentinel
------------
The last line is always

    === job finished (exit N) at <timestamp> ===

so a reader can tell "still working" from "stopped" without hunting for the
process. A log with no sentinel and no recent writes is a job that died — which
is a fact worth being able to see, rather than a spinner that lies.

It runs youtube_upload.py's own sweep, unchanged: every source in --sources, and
within each source every puzzle that paper published today (youtube_upload.
run_source). This file adds no policy of its own — the ledger, the age guard and
the per-source cap all still live there.
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--privacy", default="private",
                    choices=["private", "unlisted", "public"])
    ap.add_argument("--sources", default="telegraph,times,guardian")
    args = ap.parse_args()

    start = datetime.now()
    print("=== youtube job started %s (privacy=%s, sources=%s) ==="
          % (start.strftime("%Y-%m-%d %H:%M:%S"), args.privacy, args.sources),
          flush=True)

    # READ the child and re-print it, rather than letting it inherit this process's
    # stdout. Inheritance looked equivalent and was not: launched detached (no
    # console, stdout redirected to the log file by the caller), the child's output
    # never reached the log — the first detached test produced a header and a
    # sentinel with all three papers' lines missing. Piping is explicit and works
    # the same whether this runs detached, from a terminal, or from a scheduler.
    #
    # -u on the child, flush on every line here: the log is read WHILE it is being
    # written, so a line that sits in a buffer is a line the user cannot see.
    proc = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "scripts" / "youtube_upload.py"),
         "--privacy", args.privacy, "--sources", args.sources],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in proc.stdout:
        print(line.rstrip("\n"), flush=True)
    rc = proc.wait()

    print("=== job finished (exit %d) at %s ==="
          % (rc, datetime.now().strftime("%Y-%m-%d %H:%M:%S")), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
