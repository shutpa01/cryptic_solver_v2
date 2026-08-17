"""Deploy — push the latest code and databases to the Cordelia droplet."""

import sqlite3
import subprocess
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
GIT_BASH = r'C:\Program Files\Git\bin\bash.exe'


def _rsync(local_path, remote_path, timeout=300):
    """Run rsync via Git Bash (provides full MSYS2 environment including SSH)."""
    s = str(local_path).replace('\\', '/')
    if len(s) >= 2 and s[1] == ':':
        s = '/' + s[0].lower() + s[2:]
    cmd = f'rsync -cz {s} {remote_path}'
    return subprocess.run(
        [GIT_BASH, '-c', cmd],
        capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )


def _rsync_json_dir(local_dir, remote_path, timeout=600):
    """Rsync only *.json CONTENTS of local_dir into remote_path (trailing slashes),
    --mkpath to create it, no --delete. Mirrors the scraper's own sync
    (scraper/orchestrator/puzzle_scraper.py:_rsync_json_dir). These JSONs are the
    AUTHORITATIVE grid structure read by web/grid.py:build_grid_from_json — a newly
    served puzzle has no solve-mode grid on the droplet without them, and the DB
    alone does not carry them. Incremental (-c checksum), so only new/changed files
    transfer. Local-only data, never web-served."""
    s = str(local_dir).replace('\\', '/')
    if len(s) >= 2 and s[1] == ':':
        s = '/' + s[0].lower() + s[2:]
    s = s.rstrip('/') + '/'
    remote = remote_path.rstrip('/') + '/'
    # -r is REQUIRED: without it rsync says "skipping directory ." and transfers nothing
    # (the bug in the scraper's original _rsync_json_dir that left the droplet without new
    # grid JSONs). The dir is flat, so -r + --exclude='*' just filters to the *.json files.
    cmd = f"rsync -crz --mkpath --include='*.json' --exclude='*' {s} {remote}"
    return subprocess.run(
        [GIT_BASH, '-c', cmd],
        capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )


# Scraper JSON dirs shipped with a DB deploy — the grid-structure source the droplet
# needs for solve-mode grids (see _rsync_json_dir). Serving papers only.
CORDELIA_JSON_DIRS = [
    ("scraper/telegraph", "scraper/telegraph"),
    ("scraper/times", "scraper/times"),
    ("scraper/guardian", "scraper/guardian"),
]


def render():
    st.header("Deploy")

    st.subheader("Deploy to Cordelia")
    try:
        _render_cordelia_deploy()
    except Exception as e:
        st.error(f"Error in Cordelia deploy: {e}")


CORDELIA_DROPLET = "root@165.232.46.255"
CORDELIA_REMOTE = "/opt/cordelia"
CRYPTIC_NEW_DB = PROJECT_ROOT / "data" / "cryptic_new.db"
# Directories to deploy to Cordelia (local_dir, remote_dir, glob pattern)
# Uses scp -r for directories, excludes __pycache__
CORDELIA_CODE_DIRS = [
    ("web", "web", "*.py"),
    ("web/routes", "web/routes", "*.py"),
    ("web/templates", "web/templates", "*.html"),
    ("web/templates/partials", "web/templates/partials", "*.html"),
    ("web/static", "web/static", None),  # None = entire directory
    # core/ powers the public clue-page card since the week-only relaunch
    # (web.serving -> core.wfw_card / core.wfw_render; the /solver mount ->
    # core.wfw_web). WITHOUT this the public clue pages 410 on the droplet.
    ("core", "core", "*.py"),
    ("core/atomsig", "core/atomsig", "*.py"),
    ("signature_solver", "signature_solver", "*.py"),
    ("backfill_ai_exp", "backfill_ai_exp", "*.py"),
    ("sonnet_pipeline", "sonnet_pipeline", "*.py"),
    ("scraper/danword", "scraper/danword", "*.py"),
]
# Individual files that don't fit the directory pattern
CORDELIA_EXTRA_FILES = [
    ("data/base_catalog.json", "data/base_catalog.json"),
]


def _render_cordelia_deploy():
    """Deploy databases and/or code to the Cordelia droplet."""
    st.caption("Deploy to justcordelia.com — upload databases, code, or both.")

    col1, col2 = st.columns(2)
    with col1:
        deploy_db = st.checkbox("Deploy databases", value=True, key="co_deploy_db")
        deploy_code = st.checkbox("Deploy code", value=False, key="co_deploy_code")
    with col2:
        if deploy_db:
            clues_size = CLUES_DB.stat().st_size / 1024 / 1024
            ref_size = CRYPTIC_NEW_DB.stat().st_size / 1024 / 1024 if CRYPTIC_NEW_DB.exists() else 0
            st.write(f"**clues_master.db:** {clues_size:.0f} MB")
            st.write(f"**cryptic_new.db:** {ref_size:.0f} MB")

    if not deploy_db and not deploy_code:
        st.info("Select at least one option to deploy.")
        return

    if st.button("Deploy to Cordelia", type="primary", key="deploy_cordelia"):
        steps = []
        failed = False

        # Step 1: Upload code
        if deploy_code and not failed:
            with st.spinner("Uploading code files..."):
                uploaded = 0
                # Upload directories (glob pattern)
                for local_dir, remote_dir, pattern in CORDELIA_CODE_DIRS:
                    local_path = PROJECT_ROOT / local_dir
                    if not local_path.exists():
                        continue
                    # Ensure remote directory exists
                    subprocess.run(
                        ["ssh", CORDELIA_DROPLET, f"mkdir -p {CORDELIA_REMOTE}/{remote_dir}"],
                        capture_output=True, timeout=10,
                    )
                    if pattern is None:
                        # Upload entire directory
                        try:
                            result = subprocess.run(
                                ["scp", "-r", str(local_path) + "/.", f"{CORDELIA_DROPLET}:{CORDELIA_REMOTE}/{remote_dir}/"],
                                capture_output=True, text=True, timeout=60,
                                encoding="utf-8", errors="replace",
                            )
                            if result.returncode == 0:
                                uploaded += 1
                            else:
                                steps.append(("Upload code", False, f"Failed on {local_dir}: {result.stderr}"))
                                failed = True
                                break
                        except Exception as e:
                            steps.append(("Upload code", False, f"Failed on {local_dir}: {e}"))
                            failed = True
                            break
                    else:
                        # Upload matching files in ONE scp connection per directory
                        # (was one SSH handshake per file — 330+ files took ~15 min).
                        # Skip underscore-prefixed dev/analysis one-offs (_ab_*, _seed_*,
                        # _test_*, _diag_*, _mine_* ...): verified 2026-07-17 that no served
                        # file imports any of them (0/83). Keep __init__.py.
                        import glob
                        files = [
                            f for f in glob.glob(str(local_path / pattern))
                            if not (Path(f).name.startswith('_') and Path(f).name != '__init__.py')
                        ]
                        if files:
                            try:
                                result = subprocess.run(
                                    ["scp"] + files + [f"{CORDELIA_DROPLET}:{CORDELIA_REMOTE}/{remote_dir}/"],
                                    capture_output=True, text=True, timeout=300,
                                    encoding="utf-8", errors="replace",
                                )
                                if result.returncode != 0:
                                    steps.append(("Upload code", False, f"Failed on {remote_dir}: {result.stderr}"))
                                    failed = True
                                else:
                                    uploaded += len(files)
                            except Exception as e:
                                steps.append(("Upload code", False, f"Failed on {remote_dir}: {e}"))
                                failed = True
                    if failed:
                        break

                # Upload extra individual files
                if not failed:
                    for local_file, remote_file in CORDELIA_EXTRA_FILES:
                        local_path = PROJECT_ROOT / local_file
                        if not local_path.exists():
                            continue
                        try:
                            result = subprocess.run(
                                ["scp", str(local_path), f"{CORDELIA_DROPLET}:{CORDELIA_REMOTE}/{remote_file}"],
                                capture_output=True, text=True, timeout=30,
                                encoding="utf-8", errors="replace",
                            )
                            if result.returncode == 0:
                                uploaded += 1
                            else:
                                steps.append(("Upload code", False, f"Failed on {remote_file}: {result.stderr}"))
                                failed = True
                                break
                        except Exception as e:
                            steps.append(("Upload code", False, f"Failed on {remote_file}: {e}"))
                            failed = True
                            break

                if not failed:
                    steps.append(("Upload code", True, f"{uploaded} items uploaded."))
                    # Fix permissions — scp sets restrictive modes that block nginx
                    try:
                        subprocess.run(
                            ["ssh", CORDELIA_DROPLET,
                             f"find {CORDELIA_REMOTE}/web/static -type d -exec chmod 755 {{}} \\; && "
                             f"find {CORDELIA_REMOTE}/web/static -type f -exec chmod 644 {{}} \\; && "
                             f"find {CORDELIA_REMOTE}/web/templates -type d -exec chmod 755 {{}} \\; && "
                             f"find {CORDELIA_REMOTE}/web/templates -type f -exec chmod 644 {{}} \\;"],
                            capture_output=True, timeout=15,
                        )
                        steps.append(("Fix permissions", True, "Static/template permissions fixed."))
                    except Exception as e:
                        steps.append(("Fix permissions", False, f"Permission fix failed: {e}"))

        # Step 2: Upload databases
        if deploy_db and not failed:
            # Checkpoint WAL so all recent writes are in the main .db files
            for db in [CLUES_DB, CRYPTIC_NEW_DB]:
                if db.exists():
                    try:
                        conn = sqlite3.connect(str(db))
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                        conn.close()
                    except Exception as e:
                        steps.append(("WAL checkpoint", False, f"{db.name}: {e}"))

            with st.spinner("Uploading clues_master.db..."):
                try:
                    result = _rsync(CLUES_DB, f"{CORDELIA_DROPLET}:{CORDELIA_REMOTE}/data/clues_master.db", timeout=600)
                    if result.returncode == 0:
                        steps.append(("Upload clues_master.db", True, "Done."))
                    else:
                        steps.append(("Upload clues_master.db", False, result.stderr or "Failed."))
                        failed = True
                except subprocess.TimeoutExpired:
                    steps.append(("Upload clues_master.db", False, "Timed out after 10 minutes."))
                    failed = True
                except Exception as e:
                    steps.append(("Upload clues_master.db", False, str(e)))
                    failed = True

            if not failed and CRYPTIC_NEW_DB.exists():
                with st.spinner("Uploading cryptic_new.db..."):
                    try:
                        result = _rsync(CRYPTIC_NEW_DB, f"{CORDELIA_DROPLET}:{CORDELIA_REMOTE}/data/cryptic_new.db", timeout=600)
                        if result.returncode == 0:
                            steps.append(("Upload cryptic_new.db", True, "Done."))
                        else:
                            steps.append(("Upload cryptic_new.db", False, result.stderr or "Failed."))
                            failed = True
                    except subprocess.TimeoutExpired:
                        steps.append(("Upload cryptic_new.db", False, "Timed out after 10 minutes."))
                        failed = True
                    except Exception as e:
                        steps.append(("Upload cryptic_new.db", False, str(e)))
                        failed = True

            # Step 2b: sync the scraper grid-structure JSONs so a newly-served puzzle
            # has its solve-mode grid on the droplet (the DB alone does not carry it).
            # A JSON sync failure NEVER fails the deploy — the DB is already up; report
            # it and carry on to the restart (matches the scraper's own sync behaviour).
            if not failed:
                for local_rel, remote_rel in CORDELIA_JSON_DIRS:
                    local_dir = PROJECT_ROOT / local_rel
                    if not local_dir.exists():
                        continue
                    with st.spinner(f"Syncing {local_rel} grid JSONs..."):
                        try:
                            result = _rsync_json_dir(
                                local_dir,
                                f"{CORDELIA_DROPLET}:{CORDELIA_REMOTE}/{remote_rel}",
                                timeout=600,
                            )
                            if result.returncode == 0:
                                steps.append((f"Sync {local_rel} grids", True, "Done."))
                            else:
                                steps.append((f"Sync {local_rel} grids", False,
                                              (result.stderr or "Failed.")[:200]))
                        except subprocess.TimeoutExpired:
                            steps.append((f"Sync {local_rel} grids", False, "Timed out."))
                        except Exception as e:
                            steps.append((f"Sync {local_rel} grids", False, str(e)))

        # Step 3: Restart service
        if not failed:
            with st.spinner("Restarting Cordelia service..."):
                try:
                    result = subprocess.run(
                        ["ssh", CORDELIA_DROPLET, "systemctl restart cordelia"],
                        capture_output=True, text=True, timeout=120,
                        encoding="utf-8", errors="replace",
                    )
                    if result.returncode == 0:
                        steps.append(("Restart service", True, "Service restarted."))
                    else:
                        steps.append(("Restart service", False, result.stderr or "Restart failed."))
                        failed = True
                except Exception as e:
                    steps.append(("Restart service", False, str(e)))
                    failed = True

        # Step 3b: Warm the sitemap cache. The clue sitemap does a ~12s cold build on the
        # first request after the droplet's /tmp cache is wiped (it renders a WFW card per
        # served clue). Google's BATCH sitemap fetcher times out on that cold build and
        # records "Couldn't fetch" — observed in GSC 2026-08: all three child sitemaps
        # failed to fetch and 0 pages were discovered, even though a live URL-inspection
        # test of the same URL succeeded (so the URL is reachable — it's speed, not a block).
        # Warming right after the restart guarantees the fast cached copy exists before any
        # crawler asks. Done ON the droplet, curling the app on 127.0.0.1:5002 (Host header
        # so Flask routes it): Cloudflare 403s a non-browser request to the public URL from
        # here, and the cache lives in the droplet's /tmp anyway. Children are read from the
        # live index so a future sitemap-clues-2 is picked up automatically. Never fails the
        # deploy (SEO plumbing, not content). See memory www_duplicate_site_redirect_fix.
        if not failed:
            with st.spinner("Warming sitemap cache..."):
                warm_script = (
                    'BASE=http://127.0.0.1:5002; H="Host: justcordelia.com"; '
                    'IDX=$(curl -s --max-time 60 -H "$H" "$BASE/sitemap.xml"); '
                    'echo "$IDX" | grep -oE "<loc>[^<]+" | sed "s/<loc>//" | while read u; do '
                    'p=$(echo "$u" | sed "s#https://justcordelia.com##"); '
                    'curl -s -o /dev/null --max-time 120 '
                    '-w "%{http_code} %{time_total}s $p\\n" -H "$H" "$BASE$p"; '
                    'done'
                )
                try:
                    result = subprocess.run(
                        ["ssh", CORDELIA_DROPLET, warm_script],
                        capture_output=True, text=True, timeout=300,
                        encoding="utf-8", errors="replace",
                    )
                    lines = [ln for ln in (result.stdout or "").splitlines() if ln.strip()]
                    codes = [ln.split()[0] for ln in lines if ln.split()]
                    ok = (result.returncode == 0 and bool(codes)
                          and all(c.startswith("2") for c in codes))
                    summary = (" | ".join(lines) if lines
                               else (result.stderr or "").strip()[:200] or "no output")
                    steps.append(("Warm sitemap", ok, summary))
                except Exception as e:
                    steps.append(("Warm sitemap", False, str(e)))

        # Step 4: IndexNow — notify Bing/Yandex of the newly-live URLs. Only when the DB
        # was deployed (content went live); a code-only deploy serves no new pages. A
        # notification failure NEVER fails the deploy — the deploy itself already succeeded.
        if deploy_db and not failed:
            with st.spinner("Notifying IndexNow (Bing) of new URLs..."):
                try:
                    py = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
                    # Streams one GET per URL (~1.5s each), so runtime scales with URL count.
                    # --max-seconds keeps the script inside the subprocess budget by deferring
                    # any overflow to the next deploy (each puzzle atomic, nothing re-sent);
                    # the 300s ceiling matches the other network steps and leaves ample margin.
                    result = subprocess.run(
                        [py, str(PROJECT_ROOT / "scripts" / "indexnow_notify.py"),
                         "--max-seconds", "200", "--puzzle-pages-only"],
                        capture_output=True, text=True, timeout=300,
                        encoding="utf-8", errors="replace", cwd=str(PROJECT_ROOT),
                    )
                    lines = (result.stdout or "").strip().splitlines()
                    summary = lines[-1] if lines else (result.stderr or "").strip()[:200]
                    steps.append(("IndexNow notify", result.returncode == 0,
                                  summary or "done"))
                except Exception as e:
                    steps.append(("IndexNow notify", False, str(e)))

        # Show results
        for label, ok, msg in steps:
            if ok:
                st.success(f"{label}: {msg}")
            else:
                st.error(f"{label}: {msg}")


# Auto-render when Streamlit runs this file directly (multipage mode)
render()
