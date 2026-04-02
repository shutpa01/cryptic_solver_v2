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

    st.subheader("Deploy DB to Honeypot")
    try:
        _render_honeypot_deploy()
    except Exception as e:
        st.error(f"Error in deploy section: {e}")

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


DROPLET = "root@134.209.21.34"
HONEYPOT_DB_PATH = "/opt/honeypot/data/clues.db"


HONEYPOT_LOCAL = PROJECT_ROOT / "honeypot"
HONEYPOT_REMOTE = "/opt/honeypot"
# Files to deploy (relative to honeypot/ directory)
HONEYPOT_CODE_FILES = [
    "app.py",
    "generate_sitemaps.py",
    "generate_slugs.py",
    "templates/base.html",
    "templates/home.html",
    "templates/clue.html",
    "templates/puzzle.html",
    "templates/source.html",
    "templates/search.html",
]


def _render_honeypot_deploy():
    """Deploy database and/or code to the honeypot droplet."""
    st.caption("Deploy to clairesclues.xyz — upload database, code, or both.")

    col1, col2 = st.columns(2)
    with col1:
        deploy_db = st.checkbox("Deploy database", value=True, key="hp_deploy_db")
        deploy_code = st.checkbox("Deploy code", value=False, key="hp_deploy_code")
    with col2:
        regen_sitemaps = st.checkbox("Regenerate sitemaps", value=True, key="hp_regen")
        if deploy_db:
            st.write(f"**DB size:** {CLUES_DB.stat().st_size / 1024 / 1024:.0f} MB")

    if not deploy_db and not deploy_code:
        st.info("Select at least one option to deploy.")
        return

    if st.button("Deploy to Honeypot", type="primary", key="deploy_honeypot"):
        steps = []
        failed = False

        # Step 1: Upload code files
        if deploy_code and not failed:
            with st.spinner("Uploading code files..."):
                uploaded = 0
                for f in HONEYPOT_CODE_FILES:
                    local = HONEYPOT_LOCAL / f
                    remote = f"{DROPLET}:{HONEYPOT_REMOTE}/{f}"
                    if not local.exists():
                        continue
                    try:
                        result = subprocess.run(
                            ["scp", str(local), remote],
                            capture_output=True, text=True, timeout=30,
                            encoding="utf-8", errors="replace",
                        )
                        if result.returncode == 0:
                            uploaded += 1
                        else:
                            steps.append(("Upload code", False, f"Failed on {f}: {result.stderr}"))
                            failed = True
                            break
                    except Exception as e:
                        steps.append(("Upload code", False, f"Failed on {f}: {e}"))
                        failed = True
                        break
                if not failed:
                    steps.append(("Upload code", True, f"{uploaded} files uploaded."))

        # Step 2: Upload DB
        if deploy_db and not failed:
            with st.spinner("Uploading database..."):
                try:
                    result = subprocess.run(
                        ["scp", str(CLUES_DB), f"{DROPLET}:{HONEYPOT_DB_PATH}"],
                        capture_output=True, text=True, timeout=300,
                        encoding="utf-8", errors="replace",
                    )
                    if result.returncode == 0:
                        steps.append(("Upload DB", True, "Database uploaded."))
                    else:
                        steps.append(("Upload DB", False, result.stderr or "Upload failed."))
                        failed = True
                except subprocess.TimeoutExpired:
                    steps.append(("Upload DB", False, "Upload timed out after 5 minutes."))
                    failed = True
                except Exception as e:
                    steps.append(("Upload DB", False, str(e)))
                    failed = True

        # Step 3: Restart service
        if not failed:
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
                        failed = True
                except Exception as e:
                    steps.append(("Restart service", False, str(e)))
                    failed = True

        # Step 4: Regenerate sitemaps (optional)
        if regen_sitemaps and not failed:
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


# Auto-render when Streamlit runs this file directly (multipage mode)
render()
