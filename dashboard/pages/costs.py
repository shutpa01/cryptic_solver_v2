"""API Costs — track Sonnet API usage and spending."""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"

# Sonnet pricing (per million tokens)
SONNET_INPUT_COST = 3.00
SONNET_OUTPUT_COST = 15.00


def _get_conn():
    conn = sqlite3.connect(f"file:{CLUES_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def render():
    st.header("API Costs")

    conn = _get_conn()

    # Check table exists
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if "api_explanations" not in tables:
        st.info("No API calls recorded yet.")
        conn.close()
        return

    # Overall totals
    totals = conn.execute("""
        SELECT
            COUNT(*) as total_calls,
            COALESCE(SUM(tokens_in), 0) as total_in,
            COALESCE(SUM(tokens_out), 0) as total_out,
            COALESCE(AVG(total_ms), 0) as avg_ms,
            SUM(CASE WHEN reviewed = 0 THEN 1 ELSE 0 END) as unreviewed,
            SUM(CASE WHEN reviewed = 1 THEN 1 ELSE 0 END) as approved,
            SUM(CASE WHEN reviewed = 2 THEN 1 ELSE 0 END) as rejected
        FROM api_explanations
    """).fetchone()

    total_in = totals["total_in"]
    total_out = totals["total_out"]
    total_cost = (total_in / 1e6 * SONNET_INPUT_COST) + (total_out / 1e6 * SONNET_OUTPUT_COST)

    # Summary metrics
    cols = st.columns(4)
    cols[0].metric("Total API calls", f"{totals['total_calls']:,}")
    cols[1].metric("Total cost", f"${total_cost:.4f}")
    cols[2].metric("Tokens in", f"{total_in:,}")
    cols[3].metric("Tokens out", f"{total_out:,}")

    cols2 = st.columns(4)
    cols2[0].metric("Avg response time", f"{totals['avg_ms']:.0f}ms")
    cols2[1].metric("Unreviewed", f"{totals['unreviewed']:,}")
    cols2[2].metric("Approved", f"{totals['approved']:,}")
    cols2[3].metric("Rejected", f"{totals['rejected']:,}")

    if totals["total_calls"] > 0:
        cost_per_call = total_cost / totals["total_calls"]
        st.caption(f"Average cost per call: ${cost_per_call:.4f}")

    st.divider()

    # Cost per day
    st.subheader("Daily breakdown")
    daily = conn.execute("""
        SELECT DATE(created_at) as day,
               COUNT(*) as calls,
               SUM(tokens_in) as tokens_in,
               SUM(tokens_out) as tokens_out
        FROM api_explanations
        GROUP BY DATE(created_at)
        ORDER BY day DESC
        LIMIT 30
    """).fetchall()

    if daily:
        data = []
        for r in daily:
            day_cost = (r["tokens_in"] / 1e6 * SONNET_INPUT_COST) + \
                       (r["tokens_out"] / 1e6 * SONNET_OUTPUT_COST)
            data.append({
                "Date": r["day"],
                "Calls": r["calls"],
                "Tokens in": f"{r['tokens_in']:,}",
                "Tokens out": f"{r['tokens_out']:,}",
                "Cost": f"${day_cost:.4f}",
            })
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

        # Chart
        chart_data = pd.DataFrame([{
            "Date": r["day"],
            "Calls": r["calls"],
        } for r in daily])
        if len(chart_data) > 1:
            chart_data["Date"] = pd.to_datetime(chart_data["Date"])
            chart_data = chart_data.set_index("Date").sort_index()
            st.bar_chart(chart_data["Calls"])

    st.divider()

    # Cost projection
    st.subheader("Cost projections")
    st.caption("Based on current Sonnet pricing ($3/M input, $15/M output)")

    if totals["total_calls"] > 0:
        avg_in = total_in / totals["total_calls"]
        avg_out = total_out / totals["total_calls"]

        # How much would it cost to process all non-HIGH clues?
        non_high = conn.execute("""
            SELECT COUNT(*) FROM clues
            WHERE source IN ('telegraph', 'times')
              AND answer IS NOT NULL AND answer != ''
              AND NOT (definition IS NOT NULL AND wordplay_type IS NOT NULL
                       AND ai_explanation IS NOT NULL)
        """).fetchone()[0]

        projected_cost = non_high * cost_per_call
        cols = st.columns(3)
        cols[0].metric("Non-HIGH clues remaining", f"{non_high:,}")
        cols[1].metric("Projected cost to process all", f"${projected_cost:.2f}")
        cols[2].metric("Avg tokens per call", f"{avg_in:.0f}in / {avg_out:.0f}out")

    conn.close()
