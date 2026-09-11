"""Custom — paste a clue someone else wrote, then solve it the normal way.

The user answers clue-writing threads on Reddit (2026-09-10). Getting the clue
is the ONLY thing that changes: a pasted clue is filed under source `custom` as
its own one-clue puzzle, and from there the existing chain runs on it untouched.

A clue from a thread normally arrives WITHOUT its answer, so the answer field
here is optional and the first step is Solve, on the clue's own page — which is
the live puzzle page for a one-clue puzzle: solve mode, the answer box, and the
anagram / pattern / similar tools, with no grid. "Save all to DB" there writes
the answer through the same route a prize puzzle uses. Then: cascade, prefill,
Confirm on the WFW clue page or correct it in /hs.

Nothing here is public. `custom` is in neither web.serving.SERVED_SOURCES nor
SERVED_BROWSE, so the Custom section on the site is admin-only and no custom
clue can reach a public page or a sitemap.

This page writes ONE row to `clues` per clue. It never writes a verdict, never
touches the reference DB, and never publishes anything.
"""

import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# The scripts run under THIS project's venv, not the dashboard's (the dashboard
# runs under the AI_Solver venv) — same as web/routes/admin.py does.
PYTHON_V2 = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
CASCADE = str(PROJECT_ROOT / "scripts" / "nightly_cascade.py")
PREFILL = str(PROJECT_ROOT / "scripts" / "run_prefill.py")
LOG_DIR = PROJECT_ROOT / "logs"

# Where the solver surfaces live. The dev site mounts the hand-solver at
# /solver; the standalone WFW server on :5099 is the fallback.
SITE_BASE = "http://127.0.0.1:5001"
SOLVER_BASE = SITE_BASE + "/solver"


def _run_cascade(pnum, timeout=600):
    """Run the WFW cascade on one custom clue. Synchronous — it is one clue,
    so it comes back in seconds, and the same guards apply as every other
    cascade (answer required, no existing parse, never a frozen manual solve)."""
    try:
        p = subprocess.run(
            [PYTHON_V2, CASCADE, "--source", "custom", "--pnum", str(pnum)],
            cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 1, "Cascade timed out after %ds." % timeout


def _prefill_log(pnum):
    files = sorted(LOG_DIR.glob("prefill_ondemand_custom_%s_*.log" % pnum))
    return files[-1] if files else None


def _start_prefill(pnum):
    """Launch the prefill chain DETACHED and return its log file.

    The chain (cascade the remainder, then the headless Claude reading) takes
    minutes. Blocking a Streamlit rerun on it would freeze the tab, so it runs
    in the background exactly as the puzzle page's Prefill button does and the
    log is read back on demand.
    """
    LOG_DIR.mkdir(exist_ok=True)
    logf = LOG_DIR / ("prefill_ondemand_custom_%s_%s.log"
                      % (pnum, time.strftime("%Y%m%d_%H%M")))
    handle = open(logf, "w", encoding="utf-8")
    subprocess.Popen(
        [PYTHON_V2, PREFILL, "--source", "custom", "--pnum", str(pnum)],
        cwd=str(PROJECT_ROOT), stdout=handle, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return logf


def _prefill_state(pnum):
    """('finished'|'failed'|'running'|'none', log tail) for a clue's prefill."""
    logf = _prefill_log(pnum)
    if logf is None:
        return "none", ""
    try:
        text = logf.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "none", ""
    tail = "\n".join(text.strip().splitlines()[-12:])
    if "prefill complete" in text:
        return "finished", tail
    if "prefill failed" in text or "cascade failed" in text:
        return "failed", tail
    if time.time() - logf.stat().st_mtime > 1800:
        return "failed", tail + "\n(log has been silent for 30 minutes)"
    return "running", tail


# ---------------------------------------------------------------------------

def _render_add():
    from core.custom_clues import add_clue, split_enumeration, normalise_answer

    st.subheader("Add a clue")
    st.caption(
        "Paste the clue as it was written. A trailing enumeration — (7), "
        "(3,4), (5-2) — is picked up automatically; otherwise type it in. "
        "**Leave the answer blank if you don't have it** — that is the normal "
        "case. Solve it on the clue's own page, where the anagram, pattern and "
        "synonym tools are, save the answer there, then cascade."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        raw = st.text_area(
            "Clue", key="cc_clue", height=90,
            placeholder="Trouble maker's disturbed rest (7)",
        )
    with col2:
        answer = st.text_input(
            "Answer (optional)", key="cc_answer", placeholder="leave blank",
            help="Only if you already know it. Otherwise solve it on the "
                 "clue's page and save it there.")
        enum_override = st.text_input(
            "Enumeration", key="cc_enum", placeholder="from the clue",
            help="Leave blank to use the enumeration in the clue's brackets.")

    clue_text, parsed_enum = split_enumeration(raw) if raw else ("", None)
    enumeration = (enum_override or "").strip() or parsed_enum
    clean_answer = normalise_answer(answer)

    if raw:
        st.markdown("**Will file:** %s %s &nbsp;&rarr;&nbsp; %s" % (
            clue_text,
            "(%s)" % enumeration if enumeration else "_(no enumeration)_",
            "`%s`" % clean_answer if clean_answer else "_no answer yet_"))

    # Cascading is only possible once there is an answer, so the checkbox is
    # offered only when one was typed — a disabled tick with no explanation
    # would just look broken.
    also_cascade = False
    if clean_answer:
        also_cascade = st.checkbox(
            "Run the cascade straight after adding", value=True,
            key="cc_cascade",
            help="Free and quick — the engines get first pass at the clue.")

    if st.button("Add clue", type="primary", disabled=not raw):
        result = add_clue(raw, answer, enumeration=enum_override or None)
        if not result.get("ok"):
            if "duplicate" in result:
                d = result["duplicate"]
                st.warning("Already here — clue %s, added %s. Nothing written."
                           % (d["id"], d["publication_date"]))
            else:
                for problem in result["problems"]:
                    st.error(problem)
            return
        pnum = result["puzzle_number"]
        st.success("Added as clue %s (custom #%s)." % (result["clue_id"], pnum))
        for warning in result.get("warnings", []):
            st.info(warning)
        if not clean_answer:
            st.markdown("**Solve it here:** [%s/custom/clues/%s](%s/custom/clues/%s)"
                        % (SITE_BASE, pnum, SITE_BASE, pnum))
        elif also_cascade:
            with st.spinner("Cascading…"):
                rc, out = _run_cascade(pnum)
            st.code(out.strip() or "(no output)")
            if rc != 0:
                st.error("Cascade failed — the clue is filed, run it again below.")
        st.rerun()


def _status_chip(status, solved_by, answer):
    """The clue's state in one glance. No answer is a state of its own — it is
    the normal one on arrival, and it is what says the next step is Solve."""
    if not (answer or "").strip():
        return "🔵 NO ANSWER"
    label = (status or "not run").upper()
    colour = {"pass": "🟢", "pending": "🟠", "fail": "🔴",
              "invalid": "🟣"}.get(status, "⚪")
    return "%s %s%s" % (colour, label, " · %s" % solved_by if solved_by else "")


def _render_list():
    from core.custom_clues import list_clues, delete_clue

    st.subheader("Custom clues")
    st.caption(
        "Newest first — the same list the site shows at "
        "[/custom/clues/](%s/custom/clues/) (admin only). Solve a clue on its "
        "own page, then cascade, then prefill; review the reading with Confirm "
        "on the clue page or correct it in the hand-solver." % SITE_BASE)

    rows = list_clues()
    if not rows:
        st.info("No custom clues yet.")
        return

    for r in rows:
        pnum = r["puzzle_number"]
        answer = (r["answer"] or "").strip()
        header = "%s — %s%s  ·  %s  ·  %s" % (
            r["publication_date"] or "—", r["clue_text"],
            " (%s)" % r["enumeration"] if r["enumeration"] else "",
            answer or "—", _status_chip(r["status"], r["solved_by"], answer))
        with st.expander(header, expanded=False):
            # SOLVE first: a custom clue normally arrives without its answer,
            # and its own page is where the anagram / pattern / similar tools
            # and the answer box live (the live puzzle page, one clue, no grid).
            links = ["[**Solve** — answer box + tools ↗](%s/custom/clues/%s)"
                     % (SITE_BASE, pnum)]
            if answer:
                links += ["[Clue page (Confirm) ↗](%s/?id=%s)" % (SOLVER_BASE, r["id"]),
                          "[Hand-solver ↗](%s/hs?id=%s)" % (SOLVER_BASE, r["id"])]
            st.markdown(" &nbsp;|&nbsp; ".join(links))

            if not answer:
                st.info("No answer yet. Solve it on its own page and press "
                        "**Save all to DB** there — the cascade and the "
                        "prefill both refuse a clue with no answer.")
            else:
                cols = st.columns([1, 1, 1, 2])
                if cols[0].button("Cascade", key="casc_%s" % r["id"]):
                    with st.spinner("Cascading…"):
                        rc, out = _run_cascade(pnum)
                    st.code(out.strip() or "(no output)")
                    if rc == 0:
                        st.rerun()
                if cols[1].button("Prefill", key="pre_%s" % r["id"],
                                  help="Cascade the remainder, then the Claude "
                                       "reading — filed as PENDING for you to "
                                       "Confirm. Takes a few minutes."):
                    logf = _start_prefill(pnum)
                    st.info("Prefill started — log: logs/%s" % logf.name)
                state, tail = _prefill_state(pnum)
                if state != "none":
                    cols[2].write({"running": "⏳ running",
                                   "finished": "✅ finished",
                                   "failed": "⚠️ failed"}[state])
                    with st.expander("Prefill log", expanded=(state == "failed")):
                        st.code(tail or "(empty)")

            st.divider()
            confirm = st.checkbox(
                "Delete this clue and everything solved for it",
                key="del_%s" % r["id"])
            if st.button("Delete", key="delbtn_%s" % r["id"],
                         disabled=not confirm):
                out = delete_clue(r["id"])
                if out.get("ok"):
                    st.success("Deleted clue %s." % r["id"])
                    st.rerun()
                else:
                    for problem in out["problems"]:
                        st.error(problem)


def render():
    st.header("Custom clues")
    st.caption(
        "Clues other people wrote — Reddit clue-writing threads. Filed under "
        "the `custom` source, which is admin-only everywhere: not a served "
        "publication, not in any sitemap, no public page."
    )
    _render_add()
    st.divider()
    _render_list()


render()
