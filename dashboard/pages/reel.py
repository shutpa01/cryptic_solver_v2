"""Reel — build today's Instagram reel and post it.

The clue is NOT chosen here. It is chosen during review, with the Reel button on
the puzzle page, because that is the one moment the day's clues are being read
one by one. This page does the rest: build, watch, post.

Nothing posts by itself. Building is free to repeat; posting is one button and
it is deliberately the last thing on the page.
"""

import subprocess
import sys
from datetime import date
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
REELS = PROJECT_ROOT / "logs" / "reels"


def _run(args, timeout=900):
    """Run one of the reel scripts and hand back everything it said.

    Output is shown WHOLE, success or failure. These scripts refuse loudly and
    for good reasons — a clue that is not today's, a puzzle the banner names
    that is not live — and swallowing that would turn a clear refusal into a
    silent nothing-happened.
    """
    try:
        p = subprocess.run([PYTHON, "-m"] + args, cwd=str(PROJECT_ROOT),
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 1, "Timed out after %ds." % timeout


def _clue_row(clue_id):
    import sqlite3
    con = sqlite3.connect(
        f"file:{PROJECT_ROOT / 'data' / 'clues_master.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return con.execute(
            "SELECT id, source, puzzle_number, publication_date, clue_number, "
            "direction, clue_text, answer FROM clues WHERE id=?", (clue_id,)
        ).fetchone()
    finally:
        con.close()


SENSES = PROJECT_ROOT / "logs" / "senses.json"
YT = PROJECT_ROOT / "logs" / "youtube"


def _embed(video, caption=""):
    """A 9:16 video in a phone-shaped frame. st.video forces a landscape box, so a
    vertical film comes out as a postage stamp in a letterbox — see the note below."""
    import base64
    b64 = base64.b64encode(video.read_bytes()).decode()
    st.markdown(
        '<video controls style="width:340px;height:604px;background:#000;'
        'border-radius:14px;display:block;margin:0 auto">'
        '<source src="data:video/mp4;base64,%s" type="video/mp4"></video>' % b64,
        unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def _short_section(pick, row, pub, today):
    """The narrated YouTube Short, built from the SAME pick as the reel.

    Everything that could stall you at publish time happens here BEFORE the build:
    whether the clue can be narrated at all, and — for a double definition — whether the
    usage phrases have been approved. Rejecting is just picking another clue, because
    nothing is built until the button is pressed (user, 2026-09-07: "we do not have time
    to do rework when we are publishing").
    """
    import json

    st.divider()
    st.subheader("YouTube Short")

    # The eligibility answer is cached per pick: the check loads the Flask app, and
    # re-running it on every Streamlit rerun would make the page crawl.
    ck = st.session_state.get("short_check_%s" % pick)
    if ck is None:
        with st.spinner("Can this clue be narrated?"):
            rc, out = _run(["scripts.short_build", "--check", "--clue-id", str(pick)],
                           timeout=300)
        ck = (rc, (out or "").strip())
        st.session_state["short_check_%s" % pick] = ck
    rc, out = ck
    if rc != 0:
        st.error("This clue cannot be narrated — %s"
                 % out.replace("NO  — ", "") or "unknown reason")
        st.caption("Pick a different clue. Nothing has been built, so there is nothing "
                   "to undo.")
        return
    st.success(out or "Narratable.")

    # A double definition needs its usage phrases APPROVED or the narration falls back
    # to "tolerate gives you STICK, criticism gives you STICK", which teaches nothing.
    try:
        senses = json.loads(SENSES.read_text(encoding="utf-8"))
    except Exception:
        senses = {}
    pending = {k: v for k, v in senses.items()
               if isinstance(v, dict) and not v.get("approved")
               and v.get("clue_id") == pick}
    if pending:
        st.warning("This clue's usage phrases have not been approved. Until they are, "
                   "the narration leaves them out.")
        edited = {}
        for k, v in sorted(pending.items()):
            edited[k] = st.text_input(k, value=v.get("phrase", ""), key="sense_%s" % k)
        if st.button("Approve these phrases"):
            for k, phrase in edited.items():
                senses[k] = {**senses[k], "phrase": phrase.strip(), "approved": True}
            SENSES.write_text(json.dumps(senses, indent=2, ensure_ascii=False),
                              encoding="utf-8")
            st.success("Approved.")
            st.rerun()

    cap_dir = YT / ("%s-%s" % (row["source"], row["puzzle_number"]))
    short = cap_dir / "short_narrated.mp4"

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Build the short", type="primary", disabled=(pub != today)):
            with st.spinner("Narrating, calling ElevenLabs, capturing, assembling…"):
                rc, out = _run(["scripts.short_build", "--clue-id", str(pick)],
                               timeout=1800)
            (st.success if rc == 0 else st.error)(
                "Built." if rc == 0 else "Build failed.")
            st.code(out or "(no output)")
    with c2:
        if st.button("Build silent (no voice, free)", key="short_silent",
                     disabled=(pub != today)):
            with st.spinner("Building the pictures only…"):
                rc, out = _run(["scripts.short_build", "--clue-id", str(pick),
                                "--voice-off"], timeout=1800)
            (st.success if rc == 0 else st.error)(
                "Built (silent)." if rc == 0 else "Build failed.")
            st.code(out or "(no output)")

    if not short.exists():
        st.caption("No short built for this clue yet.")
        return

    _embed(short, "%.1f MB" % (short.stat().st_size / 1e6))

    # POSTING. "Manual" means the user presses the button, NOT that they carry the file
    # into YouTube Studio themselves (user, 2026-09-08 — the hand-off is where the first
    # attempt was lost). Same shape as POST TO INSTAGRAM below: dry run, then a confirm,
    # then one button, and it is the last thing in this section.
    st.divider()
    if st.button("Dry run — show the title and description, post nothing",
                 key="short_dry"):
        rc, out = _run(["scripts.short_post", "--clue-id", str(pick), "--dry-run"],
                       timeout=300)
        st.code(out or "(no output)")
    confirm = st.checkbox("I have watched it and it is right", key="short_confirm")
    if st.button("POST TO YOUTUBE", type="primary", disabled=not confirm,
                 key="short_post"):
        with st.spinner("Uploading to YouTube…"):
            rc, out = _run(["scripts.short_post", "--clue-id", str(pick)], timeout=1800)
        (st.success if rc == 0 else st.error)("Posted." if rc == 0 else "Post failed.")
        st.code(out or "(no output)")


def render():
    st.header("Reel")

    from core.reel_pick import get_pick, clear_pick
    pick = get_pick()

    if pick is None:
        st.info("No clue picked for today. Open today's puzzle page and press "
                "**Reel** on the clue you want — the amber button beside HS and WFW.")
        return

    row = _clue_row(pick)
    if row is None:
        st.error("Clue %s is picked but no longer exists." % pick)
        return

    pub = (row["publication_date"] or "")[:10]
    today = date.today().isoformat()

    st.subheader("%s %s — %s %s" % ((row["source"] or "").title(),
                                    row["puzzle_number"], row["clue_number"],
                                    (row["direction"] or "").title()))
    st.write("**%s**" % (row["clue_text"] or ""))
    st.caption("Answer: %s · published %s · clue id %s"
               % (row["answer"], pub or "unknown", pick))

    # The build refuses anything but today's, so say so BEFORE the button is
    # pressed rather than after — the reel claims "today's puzzle" out loud.
    if pub != today:
        st.error("This clue is from %s, not today (%s). The reel would call it "
                 "today's puzzle. Pick a clue from today." % (pub or "unknown", today))
    if st.button("Clear the pick"):
        clear_pick()
        st.rerun()

    st.divider()

    out_dir = REELS / str(pick)
    video = out_dir / "reel.mp4"

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Build the reel", type="primary", disabled=(pub != today)):
            with st.spinner("Capturing the clue page, calling ElevenLabs, assembling…"):
                rc, out = _run(["scripts.reel_build", "--clue-id", str(pick)])
            (st.success if rc == 0 else st.error)(
                "Built." if rc == 0 else "Build failed.")
            st.code(out or "(no output)")
    with c2:
        if st.button("Build silent (no voice, free)", disabled=(pub != today)):
            with st.spinner("Building without narration…"):
                rc, out = _run(["scripts.reel_build", "--clue-id", str(pick),
                                "--voice-off"])
            (st.success if rc == 0 else st.error)(
                "Built (silent)." if rc == 0 else "Build failed.")
            st.code(out or "(no output)")

    _short_section(pick, row, pub, today)

    if not video.exists():
        st.caption("No reel built for this clue yet.")
        return

    st.divider()
    # st.video FORCES A LANDSCAPE BOX, so a 9:16 reel comes out as a black
    # letterbox with a postage stamp in the middle — you cannot see what you are
    # about to post, which is the whole purpose of this page. Narrowing the
    # column does not help; the box keeps its own shape. So the file is embedded
    # directly and given a phone-shaped frame: what you watch here is the shape
    # it will be watched in.
    import base64
    b64 = base64.b64encode(video.read_bytes()).decode()
    st.markdown(
        '<video controls style="width:340px;height:604px;background:#000;'
        'border-radius:14px;display:block;margin:0 auto">'
        '<source src="data:video/mp4;base64,%s" type="video/mp4"></video>' % b64,
        unsafe_allow_html=True)
    st.caption("%.1f MB" % (video.stat().st_size / 1e6))
    caption_file = out_dir / "reel_caption.txt"
    if caption_file.exists():
        st.text_area("Caption", caption_file.read_text(encoding="utf-8"), height=180)

    st.divider()
    st.write("**Post it**")
    st.caption("The dry run uploads to Instagram and stops. It is the only way "
               "to find out whether the video is accepted without publishing it.")

    if st.button("Dry run — upload, do not post"):
        with st.spinner("Hosting the file and handing Instagram the URL…"):
            rc, out = _run(["scripts.reel_post", "--clue-id", str(pick)])
        (st.success if rc == 0 else st.error)(
            "Ready — not posted." if rc == 0 else "Dry run failed.")
        st.code(out or "(no output)")

    st.warning("The next button posts publicly to Instagram. There is no undo "
               "from here.")
    confirm = st.checkbox("I have watched it and I want it posted")
    if st.button("POST TO INSTAGRAM", type="primary", disabled=not confirm):
        with st.spinner("Posting…"):
            rc, out = _run(["scripts.reel_post", "--clue-id", str(pick), "--publish"])
        (st.success if rc == 0 else st.error)(
            "Posted." if rc == 0 else "Posting failed — nothing was published.")
        st.code(out or "(no output)")


render()
