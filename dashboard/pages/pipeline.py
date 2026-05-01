"""Pipeline Runner — configure and run the Sonnet pipeline from the dashboard."""

import sqlite3
import subprocess
import sys
from datetime import date
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
PYTHON = r"C:\Users\shute\PycharmProjects\AI_Solver\.venv\Scripts\python.exe"


def _check_tftt_available(puzzle_number):
    """Lightweight HTTP check: does a TFTT blog post exist for this puzzle?"""
    import requests

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
    }

    url = f"https://timesforthetimes.co.uk/times-cryptic-{puzzle_number}"
    try:
        resp = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        if resp.status_code == 200:
            return True
    except Exception:
        pass

    try:
        resp = requests.get(
            "https://timesforthetimes.co.uk/wp-json/wp/v2/posts",
            headers=headers, timeout=10,
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


def _get_unrun_puzzles(source_filter=None):
    """Get puzzles that have answers but haven't been fully run through the pipeline."""
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
               SUM(CASE WHEN answer IS NOT NULL AND answer != '' THEN 1 ELSE 0 END) AS with_answer,
               SUM(CASE WHEN reviewed IS NULL AND (has_solution IS NULL OR has_solution = 0)
                         AND clue_text NOT LIKE 'See %%' THEN 1 ELSE 0 END) AS untried,
               SUM(CASE WHEN has_solution = 1 THEN 1 ELSE 0 END) AS solved,
               SUM(CASE WHEN has_solution = 0 AND reviewed IS NOT NULL
                         AND clue_text NOT LIKE 'See %%' THEN 1 ELSE 0 END) AS failed
        FROM clues
        {where}
          AND puzzle_number IS NOT NULL
        GROUP BY source, puzzle_number
        HAVING with_answer > 0 AND (untried > 0 OR failed > 0)
        ORDER BY publication_date DESC
        LIMIT 50
    """, params).fetchall()
    conn.close()
    return rows


def render():
    st.header("Pipeline Runner")

    # --- Unrun puzzles selection ---
    st.subheader("Puzzles awaiting pipeline")
    filter_source = st.selectbox(
        "Filter by source",
        ["all", "telegraph", "times", "guardian", "independent", "dailymail", "cordelia"],
        key="unrun_filter",
    )
    unrun = _get_unrun_puzzles(filter_source if filter_source != "all" else None)

    if unrun:
        today_str = date.today().isoformat()
        today_puzzles = [r for r in unrun if r["publication_date"] == today_str]
        older_puzzles = [r for r in unrun if r["publication_date"] != today_str]

        batch_selected = []

        def _render_unrun_table(rows, prefix):
            selected = []
            cols_header = st.columns([1, 2, 2, 2, 1, 1, 1, 1])
            cols_header[0].markdown("**Select**")
            cols_header[1].markdown("**Source**")
            cols_header[2].markdown("**Puzzle**")
            cols_header[3].markdown("**Date**")
            cols_header[4].markdown("**Total**")
            cols_header[5].markdown("**Answers**")
            cols_header[6].markdown("**Untried**")
            cols_header[7].markdown("**Solved**")
            for i, r in enumerate(rows):
                cols = st.columns([1, 2, 2, 2, 1, 1, 1, 1])
                key = f"{prefix}_{r['source']}_{r['puzzle_number']}"
                if cols[0].checkbox("", key=key, label_visibility="collapsed"):
                    selected.append((r["source"], str(r["puzzle_number"])))
                cols[1].write(r["source"])
                cols[2].write(str(r["puzzle_number"]))
                cols[3].write(r["publication_date"] or "—")
                cols[4].write(str(r["total"]))
                cols[5].write(str(r["with_answer"]))
                cols[6].write(str(r["untried"]))
                cols[7].write(str(r["solved"] or 0))
            return selected

        if today_puzzles:
            st.caption(f"Today's puzzles ({len(today_puzzles)})")
            batch_selected += _render_unrun_table(today_puzzles, "sel")

        if older_puzzles:
            with st.expander(f"Older puzzles ({len(older_puzzles)})", expanded=False):
                batch_selected += _render_unrun_table(older_puzzles, "old")

        if batch_selected:
            st.info(f"{len(batch_selected)} puzzle(s) selected")
            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                batch_write_db = st.checkbox("Write to DB", value=True, key="batch_write_db")
            with bcol2:
                batch_force = st.checkbox("Force fresh API calls", value=False, key="batch_force")
            with bcol3:
                batch_partials = st.checkbox("Re-run partials", value=False, key="batch_partials")

            if st.button("Run Selected Puzzles", type="primary", key="run_batch"):
                _run_batch(batch_selected, batch_write_db, batch_force, batch_partials)
    else:
        st.info("All puzzles with answers have been run through the pipeline.")

    st.divider()

    # --- Reset previously run puzzles ---
    st.subheader("Reset Puzzles")
    _render_reset_section(filter_source if filter_source != "all" else None)

    st.divider()

    # --- FifteenSquared catch-up ---
    st.subheader("FifteenSquared Catch-up")
    st.caption("Parse Guardian/Independent blog explanations (Haiku only, no Sonnet)")

    fs_col1, fs_col2, fs_col3 = st.columns(3)
    with fs_col1:
        fs_source = st.selectbox(
            "Source", ["Both", "guardian", "independent"],
            key="fs_source",
        )
    with fs_col2:
        fs_date = st.date_input("Date", value=date.today(), key="fs_date")
    with fs_col3:
        st.write("")  # spacer
        st.write("")
        fs_run_clicked = st.button("Run Catch-up", type="primary", key="run_fs_catchup")

    # Output at full width, outside the columns
    if fs_run_clicked:
        source_arg = "" if fs_source == "Both" else f"--source {fs_source}"
        date_arg = f"--date {fs_date.isoformat()}"
        cmd = f"{sys.executable} scripts/fifteensquared_catchup.py {source_arg} {date_arg}".split()
        cmd = [c for c in cmd if c]

        with st.spinner("Running FifteenSquared catch-up..."):
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=300,
                )
                if result.returncode == 0:
                    st.success("Catch-up completed.")
                else:
                    st.error(f"Catch-up failed (exit {result.returncode})")
                output = result.stdout or "(no output)"
                with st.expander("Output", expanded=True):
                    st.code(output[-3000:] if len(output) > 3000 else output)
            except Exception as e:
                st.error(f"Error: {e}")

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


def _run_batch(puzzles, write_db, force, partials):
    """Run the pipeline on multiple puzzles sequentially."""
    total = len(puzzles)
    progress = st.progress(0, text=f"Starting batch run: {total} puzzle(s)")
    results = []

    for i, (source, puzzle_number) in enumerate(puzzles):
        progress.progress((i) / total, text=f"Running {source} #{puzzle_number} ({i+1}/{total})")

        # Check for blog first — 10x cheaper via blog+Haiku
        use_blog = False
        blog_cmd = None
        if source == "times":
            try:
                if _check_tftt_available(puzzle_number):
                    use_blog = True
                    blog_cmd = [PYTHON, "-m", "sonnet_pipeline.tftt_pipeline",
                                str(puzzle_number), "--write-db"]
            except Exception:
                pass
        elif source in ("guardian", "independent"):
            try:
                if _check_fifteensquared_available(source, puzzle_number):
                    use_blog = True
                    blog_cmd = [PYTHON, "-m", "sonnet_pipeline.fifteensquared_pipeline",
                                source, str(puzzle_number), "--write-db"]
            except Exception:
                pass

        if use_blog:
            cmd = blog_cmd
        else:
            cmd = [PYTHON, "-m", "sonnet_pipeline.run", "--mode", "1", "--no-review",
                   "--source", source, puzzle_number]
            if write_db:
                cmd += ["--write-db"]
            if force:
                cmd += ["--force"]
            if partials:
                cmd += ["--partials"]

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
            ok = result.returncode == 0
            label = f"[BLOG] {result.stdout or ''}" if use_blog else (result.stdout or "")
            results.append((source, puzzle_number, ok, label))
        except subprocess.TimeoutExpired:
            results.append((source, puzzle_number, False, "TIMEOUT (20 min)"))
        except Exception as e:
            results.append((source, puzzle_number, False, str(e)))

    progress.progress(1.0, text="Batch complete!")

    # Show summary
    successes = sum(1 for _, _, ok, _ in results if ok)
    failures = total - successes
    if failures == 0:
        st.success(f"All {total} puzzle(s) completed successfully.")
    else:
        st.warning(f"{successes} succeeded, {failures} failed.")

    for source, puzzle_number, ok, output in results:
        icon = "+" if ok else "X"
        with st.expander(f"[{icon}] {source} #{puzzle_number}", expanded=not ok):
            st.code(output[-3000:] if len(output) > 3000 else output)


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
