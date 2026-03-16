"""Pipeline Runner — configure and run the Sonnet pipeline from the dashboard."""

import sqlite3
import subprocess
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
PYTHON = r"C:\Users\shute\PycharmProjects\AI_Solver\.venv\Scripts\python.exe"


def _get_unrun_puzzles(source_filter=None):
    """Get puzzles that have answers but haven't been fully run through the pipeline."""
    conn = sqlite3.connect(f"file:{CLUES_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    where = "WHERE source IN ('telegraph', 'times', 'guardian', 'independent')"
    params = []
    if source_filter:
        where = "WHERE source = ?"
        params = [source_filter]
    rows = conn.execute(f"""
        SELECT source, puzzle_number, publication_date,
               COUNT(*) AS total,
               SUM(CASE WHEN answer IS NOT NULL AND answer != '' THEN 1 ELSE 0 END) AS with_answer,
               SUM(CASE WHEN reviewed IS NULL THEN 1 ELSE 0 END) AS untried,
               SUM(CASE WHEN has_solution = 1 THEN 1 ELSE 0 END) AS solved,
               SUM(CASE WHEN has_solution = 0 AND reviewed IS NOT NULL THEN 1 ELSE 0 END) AS failed
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
        ["all", "telegraph", "times", "guardian", "independent"],
        key="unrun_filter",
    )
    unrun = _get_unrun_puzzles(filter_source if filter_source != "all" else None)

    if unrun:
        st.caption(f"Showing {len(unrun)} puzzles with answers but untried clues (most recent first)")
        # Build a selectable table
        cols_header = st.columns([2, 2, 2, 1, 1, 1, 1, 2])
        cols_header[0].markdown("**Source**")
        cols_header[1].markdown("**Puzzle**")
        cols_header[2].markdown("**Date**")
        cols_header[3].markdown("**Total**")
        cols_header[4].markdown("**Answers**")
        cols_header[5].markdown("**Untried**")
        cols_header[6].markdown("**Solved**")
        cols_header[7].markdown("**Action**")

        for i, r in enumerate(unrun):
            cols = st.columns([2, 2, 2, 1, 1, 1, 1, 2])
            cols[0].write(r["source"])
            cols[1].write(str(r["puzzle_number"]))
            cols[2].write(r["publication_date"] or "—")
            cols[3].write(str(r["total"]))
            cols[4].write(str(r["with_answer"]))
            cols[5].write(str(r["untried"]))
            cols[6].write(str(r["solved"] or 0))
            if cols[7].button("Select", key=f"sel_{r['source']}_{r['puzzle_number']}"):
                st.session_state["pipe_source"] = r["source"]
                st.session_state["pipe_puzzle"] = str(r["puzzle_number"])
                st.rerun()
    else:
        st.info("All puzzles with answers have been run through the pipeline.")

    st.divider()

    # --- Reset previously run puzzles ---
    st.subheader("Reset Puzzles")
    _render_reset_section(filter_source if filter_source != "all" else None)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Run by puzzle")
        source = st.selectbox(
            "Source",
            ["telegraph", "times", "guardian", "independent"],
            index=["telegraph", "times", "guardian", "independent"].index(
                st.session_state.get("pipe_source", "telegraph")
            ),
        )
        puzzle_number = st.text_input(
            "Puzzle number",
            value=st.session_state.get("pipe_puzzle", ""),
            placeholder="e.g. 31180",
        )
        write_db = st.checkbox("Write to DB", value=True)
        force_api = st.checkbox("Force fresh API calls", value=True)
        partials = st.checkbox("Re-run partials", value=False)

    with col2:
        st.subheader("Run single clue")
        single_clue = st.text_input(
            "Clue text (partial match)",
            placeholder="e.g. Lasting without salary",
        )
        if single_clue:
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

        cmd = [PYTHON, "-m", "sonnet_pipeline.run", "--mode", "1", "--no-review"]

        if single_clue:
            cmd += ["--single-clue", single_clue]
            # If single clue, we still need a puzzle number for the pipeline
            # but the auto-detect in run.py will handle it
            if not puzzle_number and match:
                # Use the first match's source and puzzle
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
                    timeout=600,
                    encoding="utf-8",
                    errors="replace",
                )
                if result.returncode == 0:
                    st.success("Pipeline completed successfully.")
                else:
                    st.error(f"Pipeline exited with code {result.returncode}")
                output = result.stdout or "(no output)"
                with st.expander("Output", expanded=True):
                    st.code(output[-5000:] if len(output) > 5000 else output)
            except subprocess.TimeoutExpired:
                st.error("Pipeline timed out after 10 minutes.")
            except Exception as e:
                st.error(f"Failed to run pipeline: {e}")


def _render_reset_section(source_filter=None):
    """Show recently run puzzles with checkboxes for batch reset."""
    conn = sqlite3.connect(f"file:{CLUES_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    where = "WHERE source IN ('telegraph', 'times', 'guardian', 'independent')"
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
               SUM(CASE WHEN has_solution = 0 AND reviewed IS NOT NULL THEN 1 ELSE 0 END) AS failed,
               SUM(CASE WHEN reviewed IS NULL THEN 1 ELSE 0 END) AS untried,
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

    # Clear pipeline-written fields on clues, revert to untried state
    conn.execute(f"""
        UPDATE clues SET
            has_solution = NULL,
            reviewed = NULL,
            ai_explanation = NULL
        WHERE id IN ({placeholders})
    """, clue_ids)

    # Delete any pending enrichments for this puzzle
    conn.execute(
        "DELETE FROM pending_enrichments WHERE source = ? AND puzzle_number = ?",
        (source, puzzle_number)
    )

    conn.commit()
    conn.close()
