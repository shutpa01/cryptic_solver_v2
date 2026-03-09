"""Site Analytics — coverage stats, tier breakdown, clue page counts."""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"


def _get_conn():
    conn = sqlite3.connect(f"file:{CLUES_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def render():
    st.header("Site Analytics")

    conn = _get_conn()

    # Overall stats
    st.subheader("Clue inventory")
    overall = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN answer IS NOT NULL AND answer != '' THEN 1 ELSE 0 END) as with_answer,
            SUM(CASE WHEN definition IS NOT NULL AND wordplay_type IS NOT NULL
                      AND ai_explanation IS NOT NULL THEN 1 ELSE 0 END) as high,
            SUM(CASE WHEN definition IS NOT NULL AND wordplay_type IS NOT NULL
                      AND ai_explanation IS NULL THEN 1 ELSE 0 END) as medium,
            SUM(CASE WHEN definition IS NOT NULL AND wordplay_type IS NULL THEN 1 ELSE 0 END) as low,
            SUM(CASE WHEN definition IS NULL AND wordplay_type IS NULL
                      AND ai_explanation IS NULL
                      AND answer IS NOT NULL AND answer != '' THEN 1 ELSE 0 END) as none_with_answer
        FROM clues
        WHERE source IN ('telegraph', 'times')
    """).fetchone()

    cols = st.columns(6)
    cols[0].metric("Total clues", f"{overall['total']:,}")
    cols[1].metric("With answers", f"{overall['with_answer']:,}")
    cols[2].metric("HIGH", f"{overall['high']:,}")
    cols[3].metric("MEDIUM", f"{overall['medium']:,}")
    cols[4].metric("LOW", f"{overall['low']:,}")
    cols[5].metric("NONE (has answer)", f"{overall['none_with_answer']:,}")

    # Indexable pages
    indexable = overall["with_answer"] or 0
    high_pct = 100 * (overall["high"] or 0) / max(indexable, 1)
    st.progress(high_pct / 100, text=f"HIGH tier coverage: {high_pct:.1f}% of {indexable:,} indexable pages")

    st.divider()

    # Breakdown by source
    st.subheader("By source")
    source_rows = conn.execute("""
        SELECT source,
            COUNT(*) as total,
            SUM(CASE WHEN answer IS NOT NULL AND answer != '' THEN 1 ELSE 0 END) as with_answer,
            SUM(CASE WHEN definition IS NOT NULL AND wordplay_type IS NOT NULL
                      AND ai_explanation IS NOT NULL THEN 1 ELSE 0 END) as high,
            SUM(CASE WHEN definition IS NOT NULL AND wordplay_type IS NOT NULL
                      AND ai_explanation IS NULL THEN 1 ELSE 0 END) as medium,
            SUM(CASE WHEN definition IS NOT NULL AND wordplay_type IS NULL THEN 1 ELSE 0 END) as low
        FROM clues
        WHERE source IN ('telegraph', 'times')
        GROUP BY source
    """).fetchall()

    data = []
    for r in source_rows:
        with_ans = r["with_answer"] or 0
        high = r["high"] or 0
        data.append({
            "Source": r["source"].title(),
            "Total": f"{r['total']:,}",
            "With answers": f"{with_ans:,}",
            "HIGH": f"{high:,} ({100 * high / max(with_ans, 1):.0f}%)",
            "MEDIUM": f"{r['medium'] or 0:,}",
            "LOW": f"{r['low'] or 0:,}",
        })
    st.table(pd.DataFrame(data))

    st.divider()

    # Puzzle coverage (recent)
    st.subheader("Recent puzzle coverage")
    recent = conn.execute("""
        SELECT source, puzzle_number, publication_date,
               COUNT(*) as total,
               SUM(CASE WHEN answer IS NOT NULL AND answer != '' THEN 1 ELSE 0 END) as with_answer,
               SUM(CASE WHEN definition IS NOT NULL AND wordplay_type IS NOT NULL
                         AND ai_explanation IS NOT NULL THEN 1 ELSE 0 END) as high
        FROM clues
        WHERE source IN ('telegraph', 'times')
          AND publication_date IS NOT NULL
        GROUP BY source, puzzle_number
        ORDER BY publication_date DESC
        LIMIT 20
    """).fetchall()

    data = []
    for r in recent:
        total = r["total"]
        with_ans = r["with_answer"] or 0
        high = r["high"] or 0
        data.append({
            "Source": r["source"],
            "Puzzle": r["puzzle_number"],
            "Date": r["publication_date"],
            "Clues": total,
            "Answers": with_ans,
            "HIGH": high,
            "Coverage": f"{100 * high / max(total, 1):.0f}%",
        })
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    # Sitemap stats
    st.divider()
    st.subheader("Sitemap")
    st.metric("Indexable clue pages", f"{indexable:,}")
    puzzle_count = conn.execute("""
        SELECT COUNT(DISTINCT puzzle_number) FROM clues
        WHERE source IN ('telegraph', 'times') AND puzzle_number IS NOT NULL
    """).fetchone()[0]
    st.metric("Indexable puzzle pages", f"{puzzle_count:,}")

    conn.close()
