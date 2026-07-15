"""Pipeline Runner — configure and run the Sonnet pipeline from the dashboard."""

import sqlite3
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
PYTHON = r"C:\Users\shute\PycharmProjects\AI_Solver\.venv\Scripts\python.exe"


def _check_tftt_available(puzzle_number):
    """Lightweight HTTP check: does a TFTT blog post exist for this puzzle?

    Uses curl_cffi with Chrome TLS impersonation — TFTT now sits behind
    Cloudflare and plain `requests` returns 403 on every call.
    """
    from curl_cffi import requests as curl_requests

    # Try both URL patterns: recent puzzles use /times-cryptic-{N},
    # older ones moved to /times-cryptic-no-{N}.
    for tmpl in (
        "https://timesforthetimes.co.uk/times-cryptic-{}",
        "https://timesforthetimes.co.uk/times-cryptic-no-{}",
    ):
        url = tmpl.format(puzzle_number)
        try:
            resp = curl_requests.get(url, impersonate='chrome',
                                     timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                return True
        except Exception:
            pass

    try:
        resp = curl_requests.get(
            "https://timesforthetimes.co.uk/wp-json/wp/v2/posts",
            impersonate='chrome', timeout=10,
            params={"search": str(puzzle_number), "per_page": 3, "categories": "11,21"},
        )
        if resp.status_code == 200:
            for post in resp.json():
                if str(puzzle_number) in post.get("slug", ""):
                    return True
    except Exception:
        pass

    return False


def _check_fifteensquared_available(source, puzzle_number):
    """Lightweight HTTP check: does a FifteenSquared blog post exist for this puzzle?"""
    import re
    import requests

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
    }
    category_ids = {'guardian': 7, 'independent': 8}
    cat_id = category_ids.get(source)
    if not cat_id:
        return False

    try:
        resp = requests.get(
            "https://www.fifteensquared.net/wp-json/wp/v2/posts",
            headers=headers, timeout=10,
            params={"search": str(puzzle_number), "categories": str(cat_id), "per_page": 5},
        )
        if resp.status_code == 200:
            pnum_str = str(puzzle_number)
            pnum_pattern = re.compile(r'(?<!\d)' + re.escape(pnum_str) + r'(?!\d)')
            for post in resp.json():
                title = post.get("title", {}).get("rendered", "").replace(",", "")
                slug = post.get("slug", "")
                if pnum_pattern.search(title) or pnum_pattern.search(slug):
                    return True
    except Exception:
        pass

    return False


def _get_unpublished_wfw(cutoff_iso):
    """Puzzles published on/after `cutoff_iso` whose clues are NOT all served —
    so they will NOT appear on the live site yet. 'Served' = a WFW pass, or an
    INVALID with a reviewer comment: the SAME rule as the live puzzle page
    (web.serving.puzzle_is_served). O/S per puzzle = total - served."""
    conn = sqlite3.connect(f"file:{CLUES_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT c.source, c.puzzle_number, MAX(c.publication_date) AS pub,
               COUNT(*) AS total,
               SUM(CASE WHEN w.clue_id IS NOT NULL
                     OR (wi.clue_id IS NOT NULL AND n.note IS NOT NULL
                         AND TRIM(n.note) != '') THEN 1 ELSE 0 END) AS served
        FROM clues c
        LEFT JOIN wfw_solve w  ON w.clue_id = c.id AND w.status = 'pass'
        LEFT JOIN wfw_solve wi ON wi.clue_id = c.id AND wi.status = 'invalid'
        LEFT JOIN wfw_notes  n ON n.clue_id = c.id
        WHERE c.source IN ('telegraph', 'times', 'guardian')
          AND c.publication_date >= ?
        GROUP BY c.source, c.puzzle_number
        HAVING served < total
        ORDER BY pub, c.source
    """, (cutoff_iso,)).fetchall()
    conn.close()
    return rows


def _render_unpublished_section():
    """Which puzzles since the launch floor are NOT yet published because a clue
    is missing a served WFW status — with clue count and clues outstanding."""
    st.subheader("Not yet published (WFW status missing)")
    st.caption(
        "Puzzles that will NOT appear on the live site yet because not every "
        "clue is served — a WFW pass, or an INVALID with a comment. O/S = clues "
        "still outstanding. Same rule as the live puzzle page."
    )
    cutoff = st.date_input(
        "Published on or after", value=date(2026, 7, 11), key="wfw_cutoff",
    )
    rows = _get_unpublished_wfw(cutoff.isoformat())
    if not rows:
        st.success(
            f"All served-source puzzles since {cutoff.isoformat()} are fully "
            "served — nothing outstanding."
        )
        return
    total_os = sum(r["total"] - r["served"] for r in rows)
    st.warning(
        f"{len(rows)} puzzle(s) not yet published — "
        f"{total_os} clue(s) outstanding."
    )
    data = [{
        "Source": r["source"],
        "Puzzle": str(r["puzzle_number"]),
        "Date": r["pub"] or "—",
        "Clues": r["total"],
        "O/S": r["total"] - r["served"],
    } for r in rows]
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


def render():
    st.header("Pipeline Runner")

    _render_unpublished_section()
    st.divider()

    # --- Reset previously run puzzles ---
    st.subheader("Reset Puzzles")
    _render_reset_section()

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Run by puzzle")
        source = st.selectbox(
            "Source",
            ["telegraph", "times", "guardian", "independent", "dailymail"],
            index=["telegraph", "times", "guardian", "independent", "dailymail"].index(
                st.session_state.get("pipe_source", "telegraph")
            ),
        )
        puzzle_number = st.text_input(
            "Puzzle number",
            value=st.session_state.get("pipe_puzzle", ""),
            placeholder="e.g. 31180",
        )
        write_db = st.checkbox("Write to DB", value=True)
        force_api = st.checkbox("Force fresh API calls", value=False)
        partials = st.checkbox("Re-run partials", value=False)

    with col2:
        st.subheader("Run single clue")
        single_clue = st.text_input(
            "Clue text (partial match)",
            placeholder="e.g. Lasting without salary",
        )
        if single_clue:
            # Normalise pasted text — copying from a web page or chat can
            # introduce non-breaking spaces, smart quotes, or trailing
            # whitespace that defeat the LIKE substring match.
            single_clue = (
                single_clue
                .replace(" ", " ")   # NBSP
                .replace("‘", "'").replace("’", "'")  # smart quotes
                .replace("“", '"').replace("”", '"')
                .replace("–", "-").replace("—", "-")  # en/em dash
                .strip()
            )
            # Strip trailing enumeration like "(7)" or "(3,4)" or "(5-2)".
            import re as _re
            single_clue = _re.sub(
                r"\s*\(\d+(?:[\-,]\d+)*\)\s*$", "", single_clue).strip()
            # Auto-detect source and puzzle from DB
            conn = sqlite3.connect(f"file:{CLUES_DB}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            match = conn.execute(
                "SELECT source, puzzle_number, clue_text, answer FROM clues "
                "WHERE clue_text LIKE ? LIMIT 5",
                (f"%{single_clue}%",),
            ).fetchall()
            conn.close()
            if match:
                st.success(f"Found {len(match)} match(es):")
                for m in match:
                    st.text(f"  {m['source']} #{m['puzzle_number']}: "
                            f"{m['clue_text'][:60]} = {m['answer'] or '?'}")
            else:
                st.warning("No clues found matching that text.")

    st.divider()

    # Show current puzzle status if puzzle number entered
    if puzzle_number:
        _show_puzzle_status(source, puzzle_number)

    # Run button
    if st.button("Run Pipeline", type="primary"):
        if not puzzle_number and not single_clue:
            st.error("Enter a puzzle number or single clue text.")
            return

        # Check for blog first — 10x cheaper via blog+Haiku (~$0.10 vs ~$1.30)
        use_blog = False
        blog_cmd = None
        if not single_clue and puzzle_number:
            if source == "times":
                with st.spinner("Checking TFTT blog..."):
                    if _check_tftt_available(puzzle_number):
                        use_blog = True
                        blog_cmd = [PYTHON, "-m", "sonnet_pipeline.tftt_pipeline",
                                    str(puzzle_number), "--write-db"]
                        st.success(f"TFTT blog found — using blog+Haiku pipeline (~$0.10)")
                    else:
                        st.warning("TFTT not posted — using full Sonnet pipeline")
            elif source in ("guardian", "independent"):
                with st.spinner("Checking FifteenSquared blog..."):
                    if _check_fifteensquared_available(source, puzzle_number):
                        use_blog = True
                        blog_cmd = [PYTHON, "-m", "sonnet_pipeline.fifteensquared_pipeline",
                                    source, str(puzzle_number), "--write-db"]
                        st.success(f"FifteenSquared blog found — using blog+Haiku pipeline (~$0.10)")
                    else:
                        st.warning("FifteenSquared not posted — using full Sonnet pipeline")

        if use_blog:
            cmd = blog_cmd
        else:
            cmd = [PYTHON, "-m", "sonnet_pipeline.run", "--mode", "1", "--no-review"]

            if single_clue:
                cmd += ["--single-clue", single_clue]
                if not puzzle_number and match:
                    cmd += ["--source", match[0]["source"], match[0]["puzzle_number"]]
                elif puzzle_number:
                    cmd += ["--source", source, puzzle_number]
            else:
                cmd += ["--source", source, puzzle_number]

            if write_db:
                cmd += ["--write-db"]
            if force_api:
                cmd += ["--force"]
            if partials:
                cmd += ["--partials"]

        st.info(f"Running: `{' '.join(cmd)}`")

        with st.spinner("Pipeline running..."):
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=1200,
                )
                if result.returncode == 0:
                    st.success("Pipeline completed successfully.")
                else:
                    st.error(f"Pipeline exited with code {result.returncode}")
                output = result.stdout or "(no output)"
                with st.expander("Output", expanded=True):
                    st.code(output[-5000:] if len(output) > 5000 else output)
            except subprocess.TimeoutExpired:
                st.error("Pipeline timed out after 20 minutes")
            except Exception as e:
                st.error(f"Failed to run pipeline: {e}")


def _render_reset_section(source_filter=None):
    """Show recently run puzzles with checkboxes for batch reset."""
    conn = sqlite3.connect(f"file:{CLUES_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    where = "WHERE source IN ('telegraph', 'times', 'guardian', 'independent', 'dailymail', 'cordelia')"
    params = []
    if source_filter:
        where = "WHERE source = ?"
        params = [source_filter]
    rows = conn.execute(f"""
        SELECT source, puzzle_number, publication_date,
               COUNT(*) AS total,
               SUM(CASE WHEN has_solution = 1 THEN 1 ELSE 0 END) AS solved,
               SUM(CASE WHEN has_solution = 2 THEN 1 ELSE 0 END) AS partial,
               SUM(CASE WHEN has_solution = 0 AND reviewed IS NOT NULL THEN 1 ELSE 0 END) AS failed
        FROM clues
        {where}
          AND puzzle_number IS NOT NULL
          AND reviewed IS NOT NULL
        GROUP BY source, puzzle_number
        HAVING solved + partial + failed > 0
        ORDER BY publication_date DESC
        LIMIT 30
    """, params).fetchall()
    conn.close()

    if not rows:
        st.info("No previously run puzzles found.")
        return

    selected = []

    with st.expander(f"Previously run puzzles ({len(rows)})", expanded=False):
        cols_header = st.columns([1, 2, 2, 2, 1, 1, 1])
        cols_header[0].markdown("**Select**")
        cols_header[1].markdown("**Source**")
        cols_header[2].markdown("**Puzzle**")
        cols_header[3].markdown("**Date**")
        cols_header[4].markdown("**Solved**")
        cols_header[5].markdown("**Partial**")
        cols_header[6].markdown("**Failed**")
        for i, r in enumerate(rows):
            cols = st.columns([1, 2, 2, 2, 1, 1, 1])
            key = f"rst_{r['source']}_{r['puzzle_number']}"
            if cols[0].checkbox("", key=key, label_visibility="collapsed"):
                selected.append((r["source"], str(r["puzzle_number"])))
            cols[1].write(r["source"])
            cols[2].write(str(r["puzzle_number"]))
            cols[3].write(r["publication_date"] or "—")
            cols[4].write(str(r["solved"] or 0))
            cols[5].write(str(r["partial"] or 0))
            cols[6].write(str(r["failed"] or 0))

    if selected:
        st.warning(f"{len(selected)} puzzle(s) selected for reset")
        if st.button("Reset Selected Puzzles", type="primary", key="batch_reset"):
            for source, puzzle in selected:
                _reset_puzzle(source, puzzle)
            st.success(f"Reset {len(selected)} puzzle(s) — ready for re-run")
            st.rerun()


def _show_puzzle_status(source, puzzle_number):
    """Show current solve status for a puzzle."""
    conn = sqlite3.connect(f"file:{CLUES_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    row = conn.execute("""
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN answer IS NOT NULL AND answer != '' THEN 1 ELSE 0 END) AS has_answer,
               SUM(CASE WHEN has_solution = 1 THEN 1 ELSE 0 END) AS solved,
               SUM(CASE WHEN has_solution = 2 THEN 1 ELSE 0 END) AS partial,
               SUM(CASE WHEN has_solution = 0 AND reviewed IS NOT NULL
                         AND clue_text NOT LIKE 'See %%' THEN 1 ELSE 0 END) AS failed,
               SUM(CASE WHEN reviewed IS NULL AND (has_solution IS NULL OR has_solution = 0)
                         AND clue_text NOT LIKE 'See %%' THEN 1 ELSE 0 END) AS untried,
               SUM(CASE WHEN definition IS NOT NULL AND wordplay_type IS NOT NULL
                         AND ai_explanation IS NOT NULL THEN 1 ELSE 0 END) AS high_tier
        FROM clues WHERE source = ? AND puzzle_number = ?
    """, (source, puzzle_number)).fetchone()
    conn.close()

    if not row or row["total"] == 0:
        st.warning(f"No clues found for {source} #{puzzle_number}")
        return

    cols = st.columns(7)
    cols[0].metric("Total", row["total"])
    cols[1].metric("With answer", row["has_answer"] or 0)
    cols[2].metric("Solved", row["solved"] or 0)
    cols[3].metric("Partial", row["partial"] or 0)
    cols[4].metric("Failed", row["failed"] or 0)
    cols[5].metric("Untried", row["untried"] or 0)
    cols[6].metric("HIGH tier", row["high_tier"] or 0)

    solved_count = (row["solved"] or 0) + (row["partial"] or 0) + (row["failed"] or 0)
    if solved_count > 0:
        if st.button("Reset Puzzle", key=f"reset_{source}_{puzzle_number}",
                     help="Clear all pipeline results so the puzzle can be re-run"):
            _reset_puzzle(source, puzzle_number)
            st.success(f"Reset {source} #{puzzle_number} — {solved_count} clues cleared")
            st.rerun()


def _reset_puzzle(source, puzzle_number):
    """Clear all pipeline results for a puzzle, reverting it to un-run state."""
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)

    # Get clue IDs for this puzzle
    clue_ids = [r[0] for r in conn.execute(
        "SELECT id FROM clues WHERE source = ? AND puzzle_number = ?",
        (source, puzzle_number)
    ).fetchall()]

    if not clue_ids:
        conn.close()
        return

    placeholders = ",".join("?" * len(clue_ids))

    # Delete structured_explanations rows
    conn.execute(
        f"DELETE FROM structured_explanations WHERE clue_id IN ({placeholders})",
        clue_ids
    )

    # Clear pipeline-written fields on clues, revert to scrape-only state
    conn.execute(f"""
        UPDATE clues SET
            has_solution = NULL,
            reviewed = NULL,
            ai_explanation = NULL,
            explanation = NULL,
            definition = NULL,
            wordplay_type = NULL
        WHERE id IN ({placeholders})
    """, clue_ids)

    # Delete any pending enrichments for this puzzle
    conn.execute(
        "DELETE FROM pending_enrichments WHERE source = ? AND puzzle_number = ?",
        (source, puzzle_number)
    )

    conn.commit()
    conn.close()


# Auto-render when Streamlit runs this file directly (multipage mode)
render()
