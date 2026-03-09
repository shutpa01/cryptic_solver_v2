"""Cryptic Solver Dashboard — admin interface for managing the site.

Launch: streamlit run dashboard/app.py
"""

import sqlite3
import sys
from pathlib import Path

# Ensure project root is on the path so 'dashboard.pages' imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="Cryptic Solver Dashboard", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
CRYPTIC_DB = PROJECT_ROOT / "data" / "cryptic_new.db"


def get_conn(readonly=True):
    """Get a SQLite connection to clues_master.db."""
    if readonly:
        uri = f"file:{CLUES_DB}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


# Sidebar navigation
st.sidebar.title("Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Review Queue", "Pipeline Runner", "Scraper Control", "Site Analytics", "API Costs"],
)

# Load the appropriate page
if page == "Review Queue":
    from dashboard.pages import review
    review.render()
elif page == "Pipeline Runner":
    from dashboard.pages import pipeline
    pipeline.render()
elif page == "Scraper Control":
    from dashboard.pages import scraper
    scraper.render()
elif page == "Site Analytics":
    from dashboard.pages import analytics
    analytics.render()
elif page == "API Costs":
    from dashboard.pages import costs
    costs.render()
