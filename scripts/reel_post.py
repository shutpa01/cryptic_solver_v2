"""Post a built reel to Instagram.

    python -m scripts.reel_post --clue-id 10086228              # dry run, posts NOTHING
    python -m scripts.reel_post --clue-id 10086228 --publish    # actually posts

SAFETY, and it is deliberate: **nothing is published without --publish.** A dry
run does every step except the final call — it uploads the video and reports the
container's status — so a failure shows up before anything is public rather than
after.

THE HONESTY GATE
----------------
The reel's opening frame names today's puzzles and says their answers and full
explanations are live. Before publishing, this re-checks that every puzzle it
names is ACTUALLY SERVED, using the site's own gate. If one is not, it refuses.
Posting a claim the site cannot honour is worse than posting nothing, and the
gap between building and posting is exactly where that can drift.

THE UPLOAD — and why the file has to be hosted
----------------------------------------------
The Instagram-login path we are authenticated on (graph.instagram.com) REFUSES
raw bytes: creating a container without `video_url` fails with "the parameter
video_url is required" (measured 2026-09-01). The resumable endpoint in Meta's
docs belongs to the Facebook-login flow, which is a different configuration.

So the mp4 is copied to the droplet and served from `/static/reels/`, and
Instagram is given that URL. Unlisted, but public — which it is about to be
anyway. robots.txt allows it (`web/routes/seo.py` disallows only /admin/,
/reveal and /explain), and Meta rejects hosted files that robots.txt blocks, so
that ordering matters.

⚠️ UNPROVEN: the site is in Cloudflare-only mode. Whether Cloudflare lets Meta's
fetcher through is the one thing that cannot be established without trying it.
If the container comes back ERROR with a download failure, that is what
happened, and the answer is the Facebook-login configuration instead.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=False)

import requests

GRAPH = "https://graph.instagram.com/v23.0"
OUT_ROOT = ROOT / "logs" / "reels"
# Same droplet and remote root the dashboard DEPLOY page uses.
DROPLET = "root@165.232.46.255"
REMOTE_ROOT = "/opt/cordelia"
SITE_BASE = "https://justcordelia.com"


def token():
    t = os.environ.get("META_ACCESS_TOKEN")
    if not t:
        sys.exit("META_ACCESS_TOKEN is not in .env")
    return t


def me(tok):
    r = requests.get("%s/me" % GRAPH, params={"fields": "id,username",
                                              "access_token": tok}, timeout=30)
    if r.status_code != 200:
        sys.exit("Could not identify the account: %s %s" % (r.status_code, r.text[:200]))
    return r.json()


def still_true(caption_path):
    """Re-check the claim the reel makes, against the site's own serving gate.

    The banner lists puzzles and says their explanations are live. That was true
    when the reel was built; this asks whether it is true NOW, at the moment of
    posting.
    """
    line = ""
    for ln in caption_path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("Live today:"):
            line = ln.split(":", 1)[1].strip()
            break
    if not line:
        return [], []
    named = []
    for chunk in line.split(","):
        parts = chunk.strip().rsplit(" ", 1)
        if len(parts) == 2:
            named.append((parts[0].strip().lower(), parts[1].strip()))

    from web import create_app
    from web.serving import served_puzzle_numbers
    app = create_app("development")
    with app.app_context():
        served = served_puzzle_numbers()
    missing = [(s, n) for s, n in named if (s, n) not in served]
    return named, missing


def ssh_opts():
    """Point ssh at the real key and known_hosts.

    Launched from Python, ssh does not inherit Git Bash's HOME and resolves the
    home directory as the POSIX path /home/<user>, which does not exist on
    Windows. It then finds no known_hosts and no key, and reports "Host key
    verification failed" — which reads as a trust problem and is really a path
    problem (diagnosed 2026-09-01 with ssh -v). Naming both files explicitly
    fixes it WITHOUT weakening host key checking, which stays on.
    """
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "UserKnownHostsFile=%s" % (home / ".ssh" / "known_hosts")]
    key = home / ".ssh" / "id_rsa"
    if key.exists():
        opts += ["-i", str(key)]
    return opts


def host_on_droplet(path, clue_id):
    """Copy the mp4 to the live site's static tree and return its public URL.

    Same transport the dashboard DEPLOY page uses (plain scp over key auth), and
    the same permission fix: scp leaves restrictive modes that nginx will not
    serve, which is a documented trap in dashboard/pages/deploy.py.
    """
    import subprocess
    remote_dir = "%s/web/static/reels" % REMOTE_ROOT
    name = "%s.mp4" % clue_id
    opts = ssh_opts()
    mk = subprocess.run(["ssh"] + opts + [DROPLET, "mkdir -p %s" % remote_dir],
                        capture_output=True, text=True, timeout=60)
    if mk.returncode != 0:
        sys.exit("Could not reach the droplet: %s" % (mk.stderr or "").strip()[:300])
    cp = subprocess.run(["scp"] + opts + [str(path),
                         "%s:%s/%s" % (DROPLET, remote_dir, name)],
                        capture_output=True, text=True, timeout=600)
    if cp.returncode != 0:
        sys.exit("Copy failed: %s" % (cp.stderr or "").strip()[:300])
    subprocess.run(["ssh"] + opts + [DROPLET,
                    "chmod 755 %s && chmod 644 %s/%s" % (remote_dir, remote_dir, name)],
                   capture_output=True, timeout=60)
    return "%s/static/reels/%s" % (SITE_BASE, name)


def reachable(url):
    """Confirm the URL actually serves the file before handing it to Meta —
    a 403 from Cloudflare here is far easier to read than a container ERROR."""
    try:
        r = requests.head(url, timeout=30, allow_redirects=True)
        return r.status_code, r.headers.get("content-type", ""), \
            r.headers.get("content-length", "")
    except Exception as e:
        return None, str(e)[:120], ""


def create_container(tok, ig_id, caption, video_url):
    r = requests.post("%s/%s/media" % (GRAPH, ig_id),
                      data={"media_type": "REELS", "video_url": video_url,
                            "caption": caption, "access_token": tok}, timeout=60)
    if r.status_code != 200:
        sys.exit("Container failed: %s %s" % (r.status_code, r.text[:400]))
    return r.json()["id"]


def wait_ready(tok, container_id, timeout=300):
    """Poll until Meta has finished transcoding. It is NOT ready when the upload
    returns — publishing too early fails with an unhelpful error."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = requests.get("%s/%s" % (GRAPH, container_id),
                         params={"fields": "status_code,status",
                                 "access_token": tok}, timeout=30)
        d = r.json()
        last = d.get("status_code")
        if last == "FINISHED":
            return True, d
        if last == "ERROR":
            return False, d
        time.sleep(5)
    return False, {"status_code": last, "status": "timed out"}


def main():
    ap = argparse.ArgumentParser(description="Post a built reel to Instagram")
    ap.add_argument("--clue-id", type=int, required=True)
    ap.add_argument("--publish", action="store_true",
                    help="actually publish; without it nothing goes public")
    args = ap.parse_args()

    d = OUT_ROOT / str(args.clue_id)
    video, cap_file = d / "reel.mp4", d / "reel_caption.txt"
    for p in (video, cap_file):
        if not p.exists():
            sys.exit("%s is missing — run scripts.reel_build first." % p.name)
    caption = cap_file.read_text(encoding="utf-8").strip()

    named, missing = still_true(cap_file)
    if missing:
        sys.exit("REFUSING TO POST. The reel says these are live and they are "
                 "not: %s. Deploy them, or rebuild the reel."
                 % ", ".join("%s %s" % (s.title(), n) for s, n in missing))
    print("claim checked: %d puzzle(s) named, all served" % len(named))

    tok = token()
    who = me(tok)
    print("account: %s (%s)" % (who.get("username"), who.get("id")))
    print("video: %s (%.1f MB)" % (video.name, video.stat().st_size / 1e6))

    url = host_on_droplet(video, args.clue_id)
    code, ctype, clen = reachable(url)
    print("hosted: %s -> %s %s %s" % (url, code, ctype, clen))
    if code != 200:
        sys.exit("The hosted file does not serve (%s). Instagram fetches this "
                 "URL itself, so it must be publicly readable." % code)

    cid = create_container(tok, who["id"], caption, url)
    print("container: %s" % cid)
    ok, status = wait_ready(tok, cid)
    print("status: %s" % json.dumps(status))
    if not ok:
        sys.exit("Container did not finish — not publishing.")

    if not args.publish:
        print("\nDRY RUN — uploaded and ready, NOT published.")
        print("Re-run with --publish to post it.")
        return 0

    r = requests.post("%s/%s/media_publish" % (GRAPH, who["id"]),
                      data={"creation_id": cid, "access_token": tok}, timeout=120)
    if r.status_code != 200:
        sys.exit("Publish failed: %s %s" % (r.status_code, r.text[:400]))
    media_id = r.json().get("id")
    print("PUBLISHED. media id %s" % media_id)
    (d / "posted.json").write_text(json.dumps(
        {"media_id": media_id, "container": cid, "account": who}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
