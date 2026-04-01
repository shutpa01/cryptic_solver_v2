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


def render():
    st.header("Scraper Control")

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

    st.divider()
    st.subheader("Deploy DB to Honeypot")
    _render_honeypot_deploy()

    st.divider()
    st.subheader("Recent scraper activity")
    _show_recent_activity()


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


DROPLET = "root@134.209.21.34"
HONEYPOT_DB_PATH = "/opt/honeypot/data/clues.db"


def _render_honeypot_deploy():
    """Upload clues_master.db to the honeypot droplet and restart the service."""
    st.caption("Upload the local database to clairesclues.xyz and restart the service.")

    col1, col2 = st.columns(2)
    with col1:
        regen_sitemaps = st.checkbox("Regenerate sitemaps after upload", value=True)
    with col2:
        st.info(f"DB size: {CLUES_DB.stat().st_size / 1024 / 1024:.0f} MB")

    if st.button("Deploy to Honeypot", type="primary", key="deploy_honeypot"):
        steps = []

        # Step 1: Upload DB
        with st.spinner("Uploading database..."):
            try:
                result = subprocess.run(
                    ["scp", str(CLUES_DB), f"{DROPLET}:{HONEYPOT_DB_PATH}"],
                    capture_output=True, text=True, timeout=300,
                    encoding="utf-8", errors="replace",
                )
                if result.returncode == 0:
                    steps.append(("Upload DB", True, "Database uploaded successfully."))
                else:
                    steps.append(("Upload DB", False, result.stderr or "Upload failed."))
            except subprocess.TimeoutExpired:
                steps.append(("Upload DB", False, "Upload timed out after 5 minutes."))
            except Exception as e:
                steps.append(("Upload DB", False, str(e)))

        # Step 2: Restart service
        if steps[-1][1]:
            with st.spinner("Restarting honeypot service..."):
                try:
                    result = subprocess.run(
                        ["ssh", DROPLET, "systemctl restart honeypot"],
                        capture_output=True, text=True, timeout=30,
                        encoding="utf-8", errors="replace",
                    )
                    if result.returncode == 0:
                        steps.append(("Restart service", True, "Service restarted."))
                    else:
                        steps.append(("Restart service", False, result.stderr or "Restart failed."))
                except Exception as e:
                    steps.append(("Restart service", False, str(e)))

        # Step 3: Regenerate sitemaps (optional)
        if regen_sitemaps and steps[-1][1]:
            with st.spinner("Regenerating sitemaps..."):
                try:
                    result = subprocess.run(
                        ["ssh", DROPLET,
                         "cd /opt/honeypot && python3 generate_sitemaps.py --domain https://clairesclues.xyz"],
                        capture_output=True, text=True, timeout=120,
                        encoding="utf-8", errors="replace",
                    )
                    if result.returncode == 0:
                        steps.append(("Regenerate sitemaps", True, result.stdout[-500:] if result.stdout else "Done."))
                    else:
                        steps.append(("Regenerate sitemaps", False, result.stderr or "Failed."))
                except Exception as e:
                    steps.append(("Regenerate sitemaps", False, str(e)))

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
