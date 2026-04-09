"""Cryptic Solver Dashboard — admin interface for managing the site.

Launch: streamlit run dashboard/app.py

Navigation is handled by Streamlit's multipage system — each file in
dashboard/pages/ appears as a sidebar entry automatically.
"""

import sys
from pathlib import Path

# Ensure project root is on the path so 'dashboard.pages' imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="Cryptic Solver Dashboard", layout="wide")

st.header("Cryptic Solver Dashboard")
st.write("Select a page from the sidebar.")
