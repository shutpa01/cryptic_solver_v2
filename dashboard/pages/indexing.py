"""Indexing — submit puzzle + clue URLs to the Google Indexing API on demand.

Lists puzzles published on or after the cutoff (2026-05-13) that have not
already been submitted, with checkboxes. Selected puzzles get their
puzzle-page URL + every clue URL POSTed to the Indexing API, and a row
recorded in indexing_submissions so they fall off the list.

No DB upload happens here — that's the nightly/manual route. This page
is API-only.
"""

import re
import sqlite3
from datetime import date
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
INDEXING_SA_PATH = PROJECT_ROOT / "impressions" / "indexing_service_account.json"
CORDELIA_BASE_URL = "https://justcordelia.com"
INDEXING_DAILY_QUOTA = 200
CUTOFF_DATE = "2026-05-13"


# ---------------------------------------------------------------------------
# Table bootstrap
# ---------------------------------------------------------------------------

def _ensure_table():
    conn = sqlite3.connect(str(CLUES_DB), timeout=10)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS indexing_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            puzzle_number TEXT NOT NULL,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            urls_submitted INTEGER NOT NULL DEFAULT 0,
            urls_failed INTEGER NOT NULL DEFAULT 0
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_indexing_submissions_puzzle "
        "ON indexing_submissions (source, puzzle_number)"
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# URL construction — replicates logic from web.models.classify_puzzle and
# scraper.orchestrator.puzzle_scraper._make_clue_slug without the Flask
# get_db() dependency so it runs cleanly inside Streamlit.
# ---------------------------------------------------------------------------

def _is_saturday(publication_date):
    try:
        return date.fromisoformat(publication_date).weekday() == 5
    except (TypeError, ValueError):
        return False


def _puzzle_type_slug(source, puzzle_number, publication_date):
    try:
        num = int(puzzle_number)
    except (ValueError, TypeError):
        return None

    if source == "telegraph":
        if num == 99999:
            return "cryptic"
        if 3000 <= num <= 3999:
            return "prize"
        if 31000 <= num <= 31999:
            return "prize" if _is_saturday(publication_date) else "cryptic"
        return None
    if source == "times":
        if 5000 <= num <= 9999:
            return "sunday"
        if 26000 <= num <= 39999:
            return "cryptic"
        return None
    if source == "guardian":
        if 4000 <= num <= 5999:
            return "everyman"
        if 20000 <= num <= 39999:
            if _is_saturday(publication_date):
                return None  # Saturday prize not currently a routable slug
            return "cryptic"
        return None
    if source == "independent":
        if 1 <= num <= 19999:
            return "cryptic"
        return None
    if source == "dailymail":
        if 16000 <= num <= 19999:
            return "cryptic"
        return None
    if source == "cordelia":
        return "daily-mashup"
    return None


def _make_clue_slug(clue_id, clue_text):
    text = re.sub(r"[^a-z0-9]+", "-", (clue_text or "").lower().strip()).strip("-")
    if not text:
        return None
    words = text.split("-")[:12]
    return f"{clue_id}-{'-'.join(words)}"


def _build_urls_for_puzzle(conn, source, puzzle_number, publication_date):
    urls = []
    type_slug = _puzzle_type_slug(source, puzzle_number, publication_date)
    if type_slug:
        urls.append(f"{CORDELIA_BASE_URL}/{source}/{type_slug}/{puzzle_number}")
    for clue_id, clue_text in conn.execute(
        "SELECT id, clue_text FROM clues WHERE source = ? AND puzzle_number = ?",
        (source, str(puzzle_number)),
    ):
        slug = _make_clue_slug(clue_id, clue_text)
        if slug:
            urls.append(f"{CORDELIA_BASE_URL}/clue/{slug}")
    return urls


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def _submit_urls(urls):
    """POST each URL to the Indexing API. Returns (submitted, failed, error_lines)."""
    if not INDEXING_SA_PATH.exists():
        return 0, len(urls), [f"Service account file not found at {INDEXING_SA_PATH}"]

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError:
        return 0, len(urls), ["google-api-python-client / google-auth not installed"]

    creds = service_account.Credentials.from_service_account_file(
        str(INDEXING_SA_PATH),
        scopes=["https://www.googleapis.com/auth/indexing"],
    )
    service = build("indexing", "v3", credentials=creds)

    submitted = 0
    failed = 0
    errors = []
    for url in urls:
        try:
            service.urlNotifications().publish(
                body={"url": url, "type": "URL_UPDATED"}
            ).execute()
            submitted += 1
        except Exception as e:
            failed += 1
            errors.append(f"{url} — {e}")
    return submitted, failed, errors


def _quota_used_today(conn):
    """Sum URLs submitted today against indexing_submissions."""
    today = date.today().isoformat()
    row = conn.execute(
        "SELECT COALESCE(SUM(urls_submitted), 0) FROM indexing_submissions "
        "WHERE DATE(submitted_at) = ?",
        (today,),
    ).fetchone()
    return row[0] if row else 0


def _get_unsubmitted_puzzles(conn):
    """Return list of dicts for puzzles published >= CUTOFF_DATE not yet submitted."""
    rows = conn.execute(
        """SELECT c.source, c.puzzle_number, c.publication_date,
                  COUNT(*) AS clue_count
           FROM clues c
           LEFT JOIN indexing_submissions s
             ON s.source = c.source AND s.puzzle_number = c.puzzle_number
           WHERE c.publication_date >= ?
             AND c.answer IS NOT NULL AND c.answer != ''
             AND s.id IS NULL
           GROUP BY c.source, c.puzzle_number, c.publication_date
           ORDER BY c.publication_date DESC, c.source, CAST(c.puzzle_number AS INTEGER)""",
        (CUTOFF_DATE,),
    ).fetchall()
    return [
        {"source": r[0], "puzzle_number": r[1], "publication_date": r[2], "clue_count": r[3]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render():
    st.header("Google Indexing API submissions")
    st.caption(
        "Submit puzzle-page and clue URLs to Google's Indexing API on demand. "
        f"Database upload to the droplet is NOT done here — this page is API-only. "
        f"Daily quota: {INDEXING_DAILY_QUOTA} URLs."
    )

    _ensure_table()
    conn = sqlite3.connect(str(CLUES_DB), timeout=10)

    used = _quota_used_today(conn)
    remaining = INDEXING_DAILY_QUOTA - used
    c1, c2, c3 = st.columns(3)
    c1.metric("Quota used today", used)
    c2.metric("Remaining", remaining)
    c3.metric("Cutoff", CUTOFF_DATE)

    puzzles = _get_unsubmitted_puzzles(conn)

    if not puzzles:
        st.info(f"No unsubmitted puzzles published on or after {CUTOFF_DATE}.")
        conn.close()
        return

    st.subheader(f"Unsubmitted puzzles ({len(puzzles)})")

    selected = []
    for p in puzzles:
        # Estimate URL count: puzzle page (if routable) + every clue
        type_slug = _puzzle_type_slug(p["source"], p["puzzle_number"], p["publication_date"])
        url_estimate = (1 if type_slug else 0) + p["clue_count"]
        label = (
            f"{p['publication_date']}  •  {p['source']:<11} #{p['puzzle_number']:<7}  "
            f"•  {p['clue_count']} clues  →  ~{url_estimate} URLs"
        )
        key = f"chk_{p['source']}_{p['puzzle_number']}"
        if st.checkbox(label, key=key):
            selected.append({**p, "url_estimate": url_estimate})

    if not selected:
        st.write("Tick one or more puzzles, then click submit.")
        conn.close()
        return

    total_estimate = sum(p["url_estimate"] for p in selected)
    st.write(
        f"**{len(selected)} puzzle(s) selected  •  ~{total_estimate} URLs to submit  "
        f"(remaining quota {remaining})**"
    )

    if total_estimate > remaining:
        st.warning(
            f"Selection exceeds remaining quota by {total_estimate - remaining}. "
            "Some URLs will not be submitted."
        )

    if st.button("Submit selected to Google Indexing API", type="primary"):
        st.divider()
        any_ok = False
        for p in selected:
            with st.spinner(f"Submitting {p['source']} #{p['puzzle_number']}…"):
                urls = _build_urls_for_puzzle(
                    conn, p["source"], p["puzzle_number"], p["publication_date"]
                )
                submitted, failed, errors = _submit_urls(urls)

                if submitted > 0:
                    # Only record on at least partial success so failures
                    # don't silently hide the puzzle from the list.
                    wconn = sqlite3.connect(str(CLUES_DB), timeout=10)
                    wconn.execute(
                        "INSERT INTO indexing_submissions "
                        "(source, puzzle_number, urls_submitted, urls_failed) "
                        "VALUES (?, ?, ?, ?)",
                        (p["source"], str(p["puzzle_number"]), submitted, failed),
                    )
                    wconn.commit()
                    wconn.close()
                    any_ok = True

                line = (
                    f"**{p['source']} #{p['puzzle_number']}**: "
                    f"{submitted} submitted, {failed} failed"
                )
                if failed == 0 and submitted > 0:
                    st.success(line)
                elif submitted == 0:
                    st.error(line)
                else:
                    st.warning(line)
                for err in errors[:5]:
                    st.code(err)
                if len(errors) > 5:
                    st.caption(f"…and {len(errors) - 5} more errors")

        if any_ok:
            st.info("Click somewhere on the page to refresh the unsubmitted list.")

    conn.close()


# Auto-render when Streamlit runs this file directly (multipage mode)
render()
