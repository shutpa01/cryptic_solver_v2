"""Scraper Control — run and monitor puzzle scrapers."""

import sqlite3
import subprocess
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
PYTHON = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
SCRAPER_SCRIPT = str(PROJECT_ROOT / "scraper" / "orchestrator" / "puzzle_scraper.py")
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


def render():
    st.header("Scraper Control")

    try:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Run scrapers")
            scraper_target = st.selectbox(
                "Target",
                ["All sources", "telegraph", "times", "guardian", "independent", "dailymail"],
            )

            if st.button("Run Scraper", type="primary"):
                cmd = [PYTHON, SCRAPER_SCRIPT]
                if scraper_target != "All sources":
                    cmd += ["--only", scraper_target]

                st.info(f"Running: `{' '.join(cmd)}`")
                with st.spinner("Scraper running..."):
                    try:
                        result = subprocess.run(
                            cmd,
                            cwd=str(PROJECT_ROOT),
                            capture_output=True,
                            text=True,
                            timeout=600,
                            encoding="utf-8",
                            errors="replace",
                        )
                        if result.returncode == 0:
                            st.success("Scraper completed.")
                        else:
                            st.error(f"Scraper exited with code {result.returncode}")
                        with st.expander("Output", expanded=True):
                            st.code(result.stdout[-5000:] if len(result.stdout) > 5000
                                    else result.stdout)
                        if result.stderr:
                            with st.expander("Errors"):
                                st.code(result.stderr[-2000:])
                    except subprocess.TimeoutExpired:
                        st.error("Scraper timed out after 10 minutes.")
                    except Exception as e:
                        st.error(f"Failed to run scraper: {e}")

        with col2:
            st.subheader("Today's puzzle status")
            _show_todays_puzzles()
    except Exception as e:
        st.error(f"Error in scraper section: {e}")

    st.divider()

    st.subheader("Deploy to Cordelia")
    try:
        _render_cordelia_deploy()
    except Exception as e:
        st.error(f"Error in Cordelia deploy: {e}")

    st.divider()

    st.subheader("Scrape detector")
    try:
        _render_scrape_detector()
    except Exception as e:
        st.error(f"Error in scrape detector: {e}")

    st.divider()

    st.subheader("Recent scraper activity")
    try:
        _show_recent_activity()
    except Exception as e:
        st.error(f"Error in activity section: {e}")


def _show_todays_puzzles():
    """Show which of today's expected puzzles are in the DB."""
    today = date.today()
    dow = today.weekday()  # 0=Mon
    today_str = today.isoformat()

    conn = sqlite3.connect(f"file:{CLUES_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # Expected puzzles for today (based on day of week)
    expected = []
    if dow in range(6):  # Mon-Sat
        expected.append(("telegraph", "Telegraph Cryptic"))
    if dow in (1, 2, 3, 4):  # Tue-Fri
        expected.append(("telegraph", "Telegraph Toughie"))
    if dow == 6:  # Sunday
        expected.append(("telegraph", "Telegraph Prize Cryptic"))
        expected.append(("telegraph", "Telegraph Prize Toughie"))
    if dow in range(6):  # Mon-Sat
        expected.append(("times", "Times Cryptic"))
    if dow == 6:
        expected.append(("times", "Sunday Times"))

    for source, label in expected:
        count = conn.execute(
            "SELECT COUNT(*) FROM clues WHERE source = ? AND publication_date = ?",
            (source, today_str),
        ).fetchone()[0]

        has_answers = conn.execute(
            "SELECT COUNT(*) FROM clues WHERE source = ? AND publication_date = ? "
            "AND answer IS NOT NULL AND answer != ''",
            (source, today_str),
        ).fetchone()[0]

        if count == 0:
            st.error(f"{label}: **Missing**")
        elif has_answers < count:
            st.warning(f"{label}: {count} clues, {has_answers} with answers")
        else:
            st.success(f"{label}: {count} clues, all with answers")

    conn.close()


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
    ("signature_solver", "signature_solver", "*.py"),
    ("backfill_ai_exp", "backfill_ai_exp", "*.py"),
    ("sonnet_pipeline", "sonnet_pipeline", "*.py"),
    ("scraper/danword", "scraper/danword", "*.py"),
]
# Individual files that don't fit the directory pattern
CORDELIA_EXTRA_FILES = [
    ("data/base_catalog.json", "data/base_catalog.json"),
]


def _render_scrape_detector():
    """Surface IPs that look like batch /clue/* scrapers in nginx access logs."""
    st.caption(
        "Scan recent nginx access logs from the Cordelia droplet for IPs that "
        "hit many /clue/ pages in a tight time window — the signature of a "
        "daily batch scraper. Note: IPs shown are Cloudflare proxy IPs, not "
        "real client IPs (see scripts/scrape_detector.py docstring)."
    )

    col1, col2 = st.columns(2)
    with col1:
        days = st.number_input(
            "Days to scan", min_value=1, max_value=14, value=2, key="sd_days"
        )
    with col2:
        threshold = st.number_input(
            "Min /clue/ hits to flag", min_value=10, max_value=10000,
            value=50, step=10, key="sd_threshold",
        )

    if not st.button("Run detector", type="primary", key="run_detector"):
        return

    # Import the script's analysis functions. Cached import path: scripts/.
    import sys
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        from scrape_detector import fetch_log_lines, parse_line, analyse
    except ImportError as e:
        st.error(f"Could not import scrape_detector: {e}")
        return

    with st.spinner(f"Pulling last {days} day(s) of logs from droplet..."):
        try:
            lines = fetch_log_lines(int(days))
        except Exception as e:
            st.error(f"SSH/grep failed: {e}")
            return

    if not lines:
        st.error(
            "No log lines retrieved. Check SSH access to "
            f"{CORDELIA_DROPLET} and that the log files exist."
        )
        return

    st.write(f"Pulled **{len(lines):,}** log lines.")
    records = [r for r in (parse_line(l) for l in lines) if r is not None]
    st.write(f"Parsed **{len(records):,}** records.")

    candidates = analyse(records, int(threshold))
    if not candidates:
        st.success(
            f"No IPs above threshold ({threshold} clue hits). "
            "No batch-scrape pattern detected in this window."
        )
        return

    st.write(f"Flagged **{len(candidates)}** IP(s):")
    rows = []
    for c in candidates:
        span_s = c["span_seconds"]
        if span_s < 3600:
            span_str = f"{span_s / 60:.1f}m"
        else:
            span_str = f"{span_s / 3600:.1f}h"
        rows.append({
            "CF IP": c["ip"],
            "Score": c["score"],
            "Clue hits": c["clue_hits"],
            "Rate/min": round(c["rate_per_min"], 1),
            "Span": span_str,
            "Peak hour UTC": c["peak_hour"] if c["peak_hour"] is not None else "-",
            "Peak %": round(c["peak_share"] * 100),
            "UAs": len(c["uas"]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    top = [c for c in candidates if c["score"] >= 5]
    if not top:
        return
    st.markdown(f"**Detail for top {len(top)} candidate(s) (score ≥ 5):**")
    for c in top:
        header = f"{c['ip']} — score {c['score']}, {c['clue_hits']} hits"
        with st.expander(header):
            st.write(f"**First → last:** {c['first_time']} → {c['last_time']}")
            st.write(
                f"**Peak hour (UTC):** {c['peak_hour']} "
                f"({c['peak_share']*100:.0f}% of hits)"
            )
            st.write(f"**Rate per minute:** {c['rate_per_min']:.1f}")
            st.write(f"**User-agents ({len(c['uas'])}):**")
            for ua in c["uas"][:5]:
                st.code(ua)


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
                        # Upload matching files
                        import glob
                        files = glob.glob(str(local_path / pattern))
                        for f in files:
                            fname = Path(f).name
                            try:
                                result = subprocess.run(
                                    ["scp", f, f"{CORDELIA_DROPLET}:{CORDELIA_REMOTE}/{remote_dir}/{fname}"],
                                    capture_output=True, text=True, timeout=30,
                                    encoding="utf-8", errors="replace",
                                )
                                if result.returncode != 0:
                                    steps.append(("Upload code", False, f"Failed on {remote_dir}/{fname}: {result.stderr}"))
                                    failed = True
                                    break
                                uploaded += 1
                            except Exception as e:
                                steps.append(("Upload code", False, f"Failed on {remote_dir}/{fname}: {e}"))
                                failed = True
                                break
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

        # Show results
        for label, ok, msg in steps:
            if ok:
                st.success(f"{label}: {msg}")
            else:
                st.error(f"{label}: {msg}")


def _show_recent_activity():
    """Show most recent puzzles scraped per source."""
    conn = sqlite3.connect(f"file:{CLUES_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT source, puzzle_number, publication_date, COUNT(*) as clue_count,
               SUM(CASE WHEN answer IS NOT NULL AND answer != '' THEN 1 ELSE 0 END) as with_answer
        FROM clues
        WHERE source IN ('telegraph', 'times', 'guardian', 'independent', 'dailymail')
          AND publication_date IS NOT NULL
        GROUP BY source, puzzle_number
        ORDER BY publication_date DESC
        LIMIT 30
    """).fetchall()
    conn.close()

    if rows:
        data = [{
            "Source": r["source"],
            "Puzzle": r["puzzle_number"],
            "Date": r["publication_date"],
            "Clues": r["clue_count"],
            "Answers": r["with_answer"],
            "Complete": "Yes" if r["with_answer"] == r["clue_count"] else "No",
        } for r in rows]
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


# Auto-render when Streamlit runs this file directly (multipage mode)
render()
