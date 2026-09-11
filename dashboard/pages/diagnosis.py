"""Diagnosis — post-publish diagnosis of your committed solves, on demand.

Step 4 of the publish-first process: it reads the manual solves you COMMITTED,
re-solves each one cold on snapshot copies of the databases, and works out why
the engines could not do it — filing pending-only signatures and logging engine
gaps. It improves FUTURE puzzles; nothing in today's morning walk waits on it.

It used to run inside the 02:00 nightly, first in the queue. Measured on the
2026-09-11 run it took 8.24M of that night's 22.0M cache-read tokens — 37% —
which left too little budget for the early-morning work (user, 2026-09-11). So
it lives here now, to be run when it suits.

Nothing about the work changed: same prompt, same hard write-scope (pending-only
catalog rows, engine_worklist, the report log — never a pass-tier signature,
never engine code, never wfw_solve, never the reference DB).
"""

import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# The script runs under THIS project's venv, not the dashboard's.
PYTHON_V2 = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
SCRIPT = str(PROJECT_ROOT / "scripts" / "run_diagnosis.py")
LOG_DIR = PROJECT_ROOT / "logs"


def _latest_log():
    files = sorted(LOG_DIR.glob("diagnosis_run_*.log"))
    return files[-1] if files else None


def _start():
    """Launch DETACHED and return the log file.

    The run takes 10-20 minutes. Blocking a Streamlit rerun on it would freeze
    the tab, so it goes to the background exactly as the prefill launcher does.
    """
    LOG_DIR.mkdir(exist_ok=True)
    logf = LOG_DIR / ("diagnosis_run_%s.log" % time.strftime("%Y%m%d_%H%M"))
    handle = open(logf, "w", encoding="utf-8")
    subprocess.Popen(
        [PYTHON_V2, SCRIPT],
        cwd=str(PROJECT_ROOT), stdout=handle, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return logf


def _state():
    """('finished'|'failed'|'running'|'none', tail, usage_line)."""
    logf = _latest_log()
    if logf is None:
        return "none", "", ""
    try:
        text = logf.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "none", "", ""
    tail = "\n".join(text.strip().splitlines()[-15:])
    usage = ""
    m = re.search(r"tokens: .*", text)
    if m:
        usage = m.group(0)
    if "diagnosis complete" in text:
        return "finished", tail, usage
    if "diagnosis failed" in text:
        return "failed", tail, usage
    if time.time() - logf.stat().st_mtime > 3600:
        return "failed", tail + "\n(log silent for an hour)", usage
    return "running", tail, usage


def _report_for(day):
    p = LOG_DIR / ("diagnosis_%s.md" % day)
    return p if p.exists() else None


def render():
    st.header("Post-publish diagnosis")
    st.caption(
        "Why couldn't the engines solve the clues you committed? Files "
        "pending-only signatures and logs engine gaps. **Improves future "
        "puzzles — nothing this morning waits on it**, which is why it is here "
        "and no longer in the 02:00 run."
    )

    state, tail, usage = _state()
    if state == "running":
        st.info("Running — usually 10–20 minutes. Reload this page for progress.")
    elif state == "finished":
        st.success("Last run finished." + (("  \n`%s`" % usage) if usage else ""))
    elif state == "failed":
        st.error("Last run failed — see the log below.")

    st.warning(
        "This is the expensive one. On the night it was measured it used 8.24M "
        "cache-read tokens (37% of the whole nightly run). Run it when you have "
        "budget to spare, not before a morning's work."
    )

    if st.button("Run diagnosis now", type="primary",
                 disabled=(state == "running")):
        logf = _start()
        st.info("Started — log: logs/%s" % logf.name)
        st.rerun()

    if tail:
        with st.expander("Run log", expanded=(state == "failed")):
            st.code(tail)

    st.divider()
    st.subheader("Reports")
    reports = sorted(LOG_DIR.glob("diagnosis_*.md"), reverse=True)[:14]
    if not reports:
        st.info("No diagnosis reports yet.")
        return
    names = [p.name for p in reports]
    default = "diagnosis_%s.md" % date.today().isoformat()
    idx = names.index(default) if default in names else 0
    chosen = st.selectbox("Report", names, index=idx)
    st.markdown((LOG_DIR / chosen).read_text(encoding="utf-8", errors="replace"))


render()
