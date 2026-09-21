#!/usr/bin/env python3
"""Headless DB deploy — the dashboard's Deploy steps 2, 2b, 3 and 3b, no Streamlit.

WHY (user, 2026-09-17). The papers release at 00:00 BST and the nightly chain now
finishes about 00:30, but nothing is public until clues_master.db is on the droplet
and the service has restarted — which until now meant waiting for the 5am manual
deploy. This is the same deploy, runnable unattended, so a puzzle can go live as
soon as its own prefill has finished instead of waiting for the others.

WHAT IT DOES NOT DO, deliberately:
  * NO CODE DEPLOY. Databases only. Code goes up from the dashboard, with eyes on it.
  * NO INDEXNOW AT ALL, since 2026-09-21. Two things kept it out and both still hold.
    The puzzle walk (scripts/indexnow_notify.py) writes `sent_puzzle`, which is
    write-once, so announcing a partly-solved puzzle through it would lock the rest of
    that puzzle's clue pages out for ever. And announcing a partial puzzle ANY other
    way summons bingbot to a puzzle page that is still 410 — measured, twice, see
    Step 4. The complete puzzle is announced by the puzzle walk once it is whole.
  * NO YOUTUBE / REELS. Those are dashboard-only and the user's own call.

WHAT CAN GO LIVE. Only what the serving gate already allows: a stored PASS parse, or
a reviewer INVALID with a comment (web/serving.py:48-57). Prefill files everything as
status='pending' (core/prefill_commit.py:57, "NEVER pass: the user is the only path"),
and pendings do not serve. So an unattended deploy cannot publish an unreviewed
prefill reading — it publishes engine passes, which the nightly's own pass review
(Step 3b) has already been over.

    python scripts/auto_deploy.py --dry-run     # print the exact commands, run none
    python scripts/auto_deploy.py               # do it

Constants below MIRROR dashboard/pages/deploy.py, which stays the source of truth —
importing it would pull in Streamlit. If the droplet or paths move, change both.
"""

import argparse
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- mirrored from dashboard/pages/deploy.py ---------------------------------------
CLUES_DB = ROOT / "data" / "clues_master.db"
CRYPTIC_NEW_DB = ROOT / "data" / "cryptic_new.db"
GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"
CORDELIA_DROPLET = "root@165.232.46.255"

# NEVER call a bare "ssh". MEASURED 2026-09-18, after this script failed every run of
# the first overnight trial: PATH resolves `ssh` to C:\Users\shute\bin\ssh.exe, which
# runs with a different HOME and so has neither the droplet's host key nor the
# identity — "Host key verification failed", exit 255, on every single invocation.
# Give it an explicit known_hosts and it gets one step further and then says
# "Permission denied (publickey)". Git's ssh and Windows OpenSSH both work.
# rsync was never affected because it goes through Git Bash, which sets HOME.
SSH_CANDIDATES = (
    r"C:\Program Files\Git\usr\bin\ssh.exe",
    r"C:\Windows\System32\OpenSSH\ssh.exe",
)


def _ssh_exe():
    for p in SSH_CANDIDATES:
        if Path(p).exists():
            return p
    raise SystemExit("auto_deploy: no usable ssh found. Tried: %s" % ", ".join(SSH_CANDIDATES))


SSH = _ssh_exe()
CORDELIA_REMOTE = "/opt/cordelia"
CORDELIA_JSON_DIRS = [
    ("scraper/telegraph", "scraper/telegraph"),
    ("scraper/times", "scraper/times"),
    ("scraper/guardian", "scraper/guardian"),
]
WARM_SCRIPT = (
    'BASE=http://127.0.0.1:5002; H="Host: justcordelia.com"; '
    'IDX=$(curl -s --max-time 60 -H "$H" "$BASE/sitemap.xml"); '
    'echo "$IDX" | grep -oE "<loc>[^<]+" | sed "s/<loc>//" | while read u; do '
    'p=$(echo "$u" | sed "s#https://justcordelia.com##"); '
    'curl -s -o /dev/null --max-time 120 '
    '-w "%{http_code} %{time_total}s $p\\n" -H "$H" "$BASE$p"; '
    'done'
)
LOG_PATH = ROOT / "logs" / "auto_deploy.log"

# --- rollback ----------------------------------------------------------------------
# Before the FIRST deploy of a night, the droplet keeps a copy of the databases it is
# currently serving. Three deploys in a night take ONE snapshot — of the state before
# any of them — because that is the state we would want back. The 03:00 backup task
# runs after these deploys, so it snapshots the NEW state and is no use for reverting.
# ~1 GB a night against 43 GB free (checked 2026-09-17).
REMOTE_DATA = CORDELIA_REMOTE + "/data"
ROLLBACK_DBS = ("clues_master.db", "cryptic_new.db")


def _snapshot_cmd(tag):
    """Copy each live DB to a dated rollback file, ONLY if that file does not exist."""
    parts = []
    for name in ROLLBACK_DBS:
        live = "%s/%s" % (REMOTE_DATA, name)
        snap = "%s.rollback-%s" % (live, tag)
        parts.append(
            '[ -f "{s}" ] && echo "keep {n}" || {{ [ -f "{l}" ] && cp -p "{l}" "{s}" '
            '&& echo "snap {n}" || echo "missing {n}"; }}'.format(s=snap, l=live, n=name))
    return "; ".join(parts)


def _rollback_cmd(tag):
    parts = []
    for name in ROLLBACK_DBS:
        live = "%s/%s" % (REMOTE_DATA, name)
        snap = "%s.rollback-%s" % (live, tag)
        parts.append(
            '[ -f "{s}" ] && cp -p "{s}" "{l}" && echo "restored {n}" '
            '|| echo "NO SNAPSHOT {n}"'.format(s=snap, l=live, n=name))
    return "; ".join(parts)


def log(msg):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write("[%s] %s\n" % (stamp, msg))
    print("[%s] %s" % (stamp, msg), flush=True)


def _win_to_msys(p):
    s = str(p).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        s = "/" + s[0].lower() + s[2:]
    return s


def _run(cmd, timeout, dry_run, label):
    """Run a command, or print it under --dry-run. Returns (ok, output)."""
    if dry_run:
        log("  [dry-run] %s: %s" % (label, cmd if isinstance(cmd, str) else " ".join(cmd)))
        return True, "(dry run)"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        if r.returncode == 0:
            return True, (r.stdout or "").strip()
        return False, (r.stderr or r.stdout or "failed").strip()
    except Exception as e:
        return False, str(e)


def rsync(local_path, remote_rel, timeout, dry_run, label):
    cmd = "rsync -cz %s %s:%s/%s" % (_win_to_msys(local_path), CORDELIA_DROPLET,
                                     CORDELIA_REMOTE, remote_rel)
    return _run([GIT_BASH, "-c", cmd], timeout, dry_run, label)


def rsync_json_dir(local_dir, remote_rel, timeout, dry_run, label):
    src = _win_to_msys(local_dir).rstrip("/") + "/"
    # EXACTLY the dashboard's _rsync_json_dir command. The dir is flat, so -r plus
    # --exclude='*' just filters to the *.json files; do NOT add --include='*/' or
    # this recurses where the dashboard does not.
    cmd = ("rsync -crz --mkpath --include='*.json' --exclude='*' "
           "%s %s:%s/%s/" % (src, CORDELIA_DROPLET, CORDELIA_REMOTE, remote_rel.rstrip("/")))
    return _run([GIT_BASH, "-c", cmd], timeout, dry_run, label)


def main():
    ap = argparse.ArgumentParser(description="Headless database deploy to the droplet.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every command without running any of them")
    ap.add_argument("--label", default="",
                    help="free text for the log line, e.g. 'after guardian 30115 prefill'")
    ap.add_argument("--source", default=None,
                    help="with --puzzle: names the puzzle in the log line. It no "
                         "longer announces anything — see Step 4 below.")
    ap.add_argument("--puzzle", default=None, help="puzzle number (needs --source)")
    ap.add_argument("--skip-warm", action="store_true",
                    help="skip the sitemap warm (it is SEO plumbing, never fatal)")
    ap.add_argument("--rollback", action="store_true",
                    help="PUT BACK the databases the droplet was serving before this "
                         "night's first deploy, then restart. Uploads nothing.")
    ap.add_argument("--tag", default=None, metavar="YYYYMMDD",
                    help="which night's snapshot to take or restore (default: today)")
    ap.add_argument("--check", action="store_true",
                    help="REALLY talk to the droplet — ssh, rsync and the health check — "
                         "without uploading a database or restarting anything. This is the "
                         "test that --dry-run is not: run it after ANY change here.")
    args = ap.parse_args()

    tag = args.tag or time.strftime("%Y%m%d")

    # --- check: exercise the real connection, change nothing ---
    if args.check:
        log("auto_deploy CHECK — using ssh: %s" % SSH)
        ok, out = _run([SSH, CORDELIA_DROPLET, "echo ssh-ok; df -h /opt | tail -1"],
                       60, False, "ssh")
        log("  ssh: %s" % ("OK — " + (out or "").replace("\n", " | ") if ok else "FAILED: " + out))
        if not ok:
            return 1
        # rsync's own transport, with --dry-run so not one byte of the DB moves.
        cmd = "rsync -cz --dry-run %s %s:%s/data/clues_master.db" % (
            _win_to_msys(CLUES_DB), CORDELIA_DROPLET, CORDELIA_REMOTE)
        ok2, out2 = _run([GIT_BASH, "-c", cmd], 300, False, "rsync")
        log("  rsync transport: %s" % ("OK" if ok2 else "FAILED: " + out2))
        ok3, code = _run([SSH, CORDELIA_DROPLET,
                          'curl -s -o /dev/null -w "%{http_code}" --max-time 30 '
                          '-H "Host: justcordelia.com" http://127.0.0.1:5002/'],
                         60, False, "health check")
        log("  health check: %s" % (code or "?").strip())
        good = ok and ok2 and (code or "").strip() == "200"
        log("auto_deploy CHECK %s" % ("PASSED" if good else "FAILED"))
        return 0 if good else 1

    # --- rollback: restore and restart, upload nothing ---
    if args.rollback:
        log("auto_deploy ROLLBACK to snapshot %s%s" % (tag, "  [DRY RUN]" if args.dry_run else ""))
        ok, out = _run([SSH, CORDELIA_DROPLET, _rollback_cmd(tag)], 300,
                       args.dry_run, "restore snapshot")
        log("  %s" % (out or "(no output)"))
        if not ok or "NO SNAPSHOT" in (out or ""):
            log("  ROLLBACK ABORTED — snapshot missing. Nothing changed.")
            return 1
        ok, out = _run([SSH, CORDELIA_DROPLET, "systemctl restart cordelia"], 120,
                       args.dry_run, "restart cordelia")
        if not ok:
            log("  restart FAILED after restore: %s" % out)
            return 1
        if not args.dry_run:
            time.sleep(3)
            _ok, code = _run([SSH, CORDELIA_DROPLET,
                              'curl -s -o /dev/null -w "%{http_code}" --max-time 30 '
                              '-H "Host: justcordelia.com" http://127.0.0.1:5002/'],
                             60, False, "health check")
            log("  health check after rollback: %s" % (code or "?").strip())
        log("auto_deploy ROLLBACK DONE")
        return 0

    if not CLUES_DB.exists():
        log("auto_deploy: %s missing — nothing to deploy." % CLUES_DB)
        return 1

    what = (" — " + args.label) if args.label else ""
    log("auto_deploy START%s%s" % (what, "  [DRY RUN]" if args.dry_run else ""))

    # Step 0: keep what the droplet is serving NOW, so tonight is reversible. Taken
    # once per night — the second and third deploys find the file and leave it alone,
    # so the snapshot is always the pre-tonight state, not the previous deploy's.
    ok, out = _run([SSH, CORDELIA_DROPLET, _snapshot_cmd(tag)], 300, args.dry_run,
                   "snapshot live DBs")
    if not ok:
        log("  SNAPSHOT FAILED: %s — refusing to deploy. Without a snapshot this "
            "cannot be reverted." % out)
        return 1
    log("  rollback snapshot (%s): %s" % (tag, (out or "").replace("\n", "; ")))

    # Step 2 (a): WAL checkpoint, so every recent write is in the main .db file BEFORE
    # it is copied. Without this the droplet can receive a database missing the last
    # few minutes of work — the very work we are rushing to publish.
    if not args.dry_run:
        for db in (CLUES_DB, CRYPTIC_NEW_DB):
            if db.exists():
                try:
                    conn = sqlite3.connect(str(db))
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    conn.close()
                except Exception as e:
                    log("  WAL checkpoint FAILED for %s: %s — aborting, a half-written "
                        "database must never be copied." % (db.name, e))
                    return 1
        log("  WAL checkpointed.")
    else:
        log("  [dry-run] would WAL-checkpoint clues_master.db and cryptic_new.db")

    # Step 2 (b): the databases. rsync writes to a temp file and renames, so an
    # interrupted transfer leaves the PREVIOUS database serving rather than a
    # truncated one. A failure here must stop everything: no restart, no warm.
    mb = CLUES_DB.stat().st_size / 1024 / 1024
    log("  uploading clues_master.db (%.0f MB)..." % mb)
    ok, out = rsync(CLUES_DB, "data/clues_master.db", 900, args.dry_run, "rsync clues_master.db")
    if not ok:
        log("  UPLOAD FAILED: %s — stopping. The site still serves the previous "
            "database, so nothing is broken; it is just not fresh." % out)
        return 1
    log("  clues_master.db uploaded.")

    if CRYPTIC_NEW_DB.exists():
        ok, out = rsync(CRYPTIC_NEW_DB, "data/cryptic_new.db", 900, args.dry_run,
                        "rsync cryptic_new.db")
        log("  cryptic_new.db %s" % ("uploaded." if ok else "FAILED: %s (not fatal)" % out))

    # Step 2b: grid-structure JSONs. The DB alone does not carry them, so a newly
    # served puzzle has no solve-mode grid without this. Never fatal.
    for local_rel, remote_rel in CORDELIA_JSON_DIRS:
        d = ROOT / local_rel
        if not d.exists():
            continue
        ok, out = rsync_json_dir(d, remote_rel, 600, args.dry_run, "rsync %s" % local_rel)
        if not ok:
            log("  grid JSON sync for %s failed: %s — not fatal, carrying on." % (local_rel, out))

    # Step 3: restart. The unit is Restart=always, so a process that dies comes back;
    # what this does not survive is a persistent startup fault, which is why the
    # health check below exists.
    ok, out = _run([SSH, CORDELIA_DROPLET, "systemctl restart cordelia"], 120,
                   args.dry_run, "restart cordelia")
    if not ok:
        log("  RESTART FAILED: %s" % out)
        return 1
    log("  service restarted.")

    # Health check: prove the site answers before we call this a success. Checked ON
    # the droplet against 127.0.0.1:5002 — Cloudflare 403s a non-browser request to
    # the public URL from here (same reason the warm step works this way).
    if not args.dry_run:
        time.sleep(3)
        ok, out = _run([SSH, CORDELIA_DROPLET,
                        'curl -s -o /dev/null -w "%{http_code}" --max-time 30 '
                        '-H "Host: justcordelia.com" http://127.0.0.1:5002/'],
                       60, False, "health check")
        code = (out or "").strip()
        if code != "200":
            log("  HEALTH CHECK FAILED: home page returned %r. The deploy finished but "
                "the site is NOT answering — needs a human." % code)
            return 1
        log("  health check OK (home page 200).")
    else:
        log("  [dry-run] would health-check http://127.0.0.1:5002/ on the droplet")

    # Step 3b: warm the sitemap cache so a crawler never hits the ~12s cold build.
    # SEO plumbing — never fails the deploy.
    if not args.skip_warm:
        ok, out = _run([SSH, CORDELIA_DROPLET, WARM_SCRIPT], 300, args.dry_run,
                       "warm sitemap")
        log("  sitemap warm %s" % ("done." if ok else "failed: %s (not fatal)" % out))

    # Step 4: DELIBERATELY NOT ANNOUNCED HERE. Dropped 2026-09-21 — read this before
    # putting it back.
    #
    # This step used to announce the puzzle's already-served clue pages the moment a
    # PARTIAL deploy landed. MEASURED two mornings running: IndexNow does not queue,
    # it SUMMONS. Bingbot arrived 13 seconds after the announce on 21 Sep and 40
    # seconds after it on 20 Sep, crawled the clue pages, followed their breadcrumb
    # (web/templates/clue.html:51) to the puzzle page — and the puzzle page was still
    # 410, because the puzzle was half solved and web/routes/puzzle.py:54 serves it
    # only when EVERY clue is served. 410 means "gone, permanently". Bingbot took it
    # at its word and never came back; that one fetch was the whole day's budget for
    # the page. DT 31350 and DT prize 3387 were each absent from Bing for the day
    # they were published, while that day's Times — crawled after its FULL deploy —
    # went straight to #1.
    #
    # The trade was 8 minutes (21 Sep) to 24 minutes (20 Sep) of earlier announcing,
    # paid for with the puzzle page for the whole day. And nothing is stranded by
    # dropping it: every one of the 87 puzzles published since 24 Aug eventually
    # serves in full, so the complete puzzle — its page AND every clue URL — still
    # goes out through scripts/indexnow_notify.py once it is whole.
    #
    # scripts/announce_clues.py is untouched and still available for a deliberate
    # announce by hand. DO NOT re-wire it into this script unless the puzzle page has
    # first been made to serve while partial (answer-only for the clues that have not
    # passed) — otherwise this defect comes straight back.

    log("auto_deploy DONE%s" % what)
    return 0


if __name__ == "__main__":
    sys.exit(main())
