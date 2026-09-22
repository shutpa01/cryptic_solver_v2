"""Growth — trailing 7-day run rate.

Each point is the average of that day and the six before it. The line
starts on day 7, where a full window first exists: a partial average
would start low and read as growth that never happened.

Feed it with either or both of:
    python3 scripts/ga4_daily_export.py --days 180 --csv data/ga4_daily.csv
    python3 scripts/traffic_from_logs.py --days 180 --csv data/logs_daily.csv

The rolling maths comes from scripts/ga4_daily_export.py rather than being
redone in pandas here, so the chart and the CSV can't disagree.
"""

import importlib.util
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GA4_CSV = DATA_DIR / "ga4_daily.csv"
LOGS_CSV = DATA_DIR / "logs_daily.csv"

# Categorical slots 1 and 2 from the reference palette, light surface.
# Validated: worst adjacent pair ΔE 24.7 (protan), 33.6 normal vision,
# both >= 3:1 against the surface.
SERIES_COLOURS = ["#2a78d6", "#eb6834"]
GRID_COLOUR = "#e6e5e1"
AXIS_INK = "#52514e"

# GA4 column -> label. The log export uses its own names for the same ideas.
GA4_METRICS = {
    "active_users": "Active users",
    "new_users": "New users",
    "sessions": "Sessions",
    "page_views": "Page views",
}
LOG_METRICS = {"visitors": "Active users", "page_views": "Page views"}


@st.cache_resource
def _load_rolling_average():
    """Import rolling_average from scripts/ (no package, so load by path)."""
    path = PROJECT_ROOT / "scripts" / "ga4_daily_export.py"
    spec = importlib.util.spec_from_file_location("ga4_daily_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rolling_average


def _read_daily(path: Path) -> pd.DataFrame | None:
    """Read a daily CSV, or None when it isn't there yet."""
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "date" not in frame.columns or frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame.sort_values("date")


def _run_rate(frame: pd.DataFrame, column: str, window: int,
              since: str | None, label: str) -> pd.DataFrame:
    """Trailing average as a tidy frame, via the tested pure function."""
    rows = frame.to_dict("records")
    if since:
        rows = [r for r in rows if r["date"] >= since]
    rows = [r for r in rows if column in r and pd.notna(r[column])]
    if not rows:
        return pd.DataFrame(columns=["date", "value", "series"])
    points = _load_rolling_average()(rows, column, window)
    out = pd.DataFrame(points)
    if out.empty:
        return pd.DataFrame(columns=["date", "value", "series"])
    out["series"] = label
    return out


def _chart(data: pd.DataFrame, window: int, launch: str | None, multi: bool):
    """Line chart: 2px lines, recessive axes, crosshair tooltip, no legend
    for a single series (the title names it)."""
    import altair as alt

    base = alt.Chart(data)
    hover = alt.selection_point(
        fields=["date"], nearest=True, on="mouseover", empty=False,
    )

    colour = (
        alt.Color("series:N", title=None,
                  scale=alt.Scale(range=SERIES_COLOURS),
                  legend=alt.Legend(orient="top", labelColor=AXIS_INK))
        if multi else alt.value(SERIES_COLOURS[0])
    )

    line = base.mark_line(strokeWidth=2, interpolate="monotone").encode(
        x=alt.X("date:T", title=None,
                axis=alt.Axis(grid=False, labelColor=AXIS_INK,
                              tickColor=GRID_COLOUR, domainColor=GRID_COLOUR)),
        y=alt.Y("value:Q", title=f"{window}-day average",
                scale=alt.Scale(zero=True),
                axis=alt.Axis(gridColor=GRID_COLOUR, labelColor=AXIS_INK,
                              titleColor=AXIS_INK, domain=False, ticks=False)),
        color=colour,
    )

    # Invisible wide target so the tooltip is easy to hit on a sparse line.
    points = base.mark_point(size=120, opacity=0).encode(
        x="date:T", y="value:Q",
        tooltip=[alt.Tooltip("date:T", title="Date"),
                 alt.Tooltip("value:Q", title="Run rate", format=".1f"),
                 alt.Tooltip("series:N", title="Source")],
    ).add_params(hover)

    crosshair = base.mark_rule(color=GRID_COLOUR).encode(
        x="date:T",
    ).transform_filter(hover)

    layers = [crosshair, line, points]

    if launch:
        marker = alt.Chart(pd.DataFrame({"date": [launch]})).mark_rule(
            color=AXIS_INK, strokeDash=[4, 4], strokeWidth=1,
        ).encode(x="date:T")
        layers.insert(0, marker)

    return alt.layer(*layers).properties(height=340).configure_view(
        strokeWidth=0,
    )


def render():
    st.header("Growth — run rate")

    ga4 = _read_daily(GA4_CSV)
    logs = _read_daily(LOGS_CSV)

    if ga4 is None and logs is None:
        st.info("No daily data yet. Generate at least one of these:")
        st.code(
            "python3 scripts/ga4_daily_export.py --days 180 "
            f"--csv {GA4_CSV.relative_to(PROJECT_ROOT)}\n"
            "python3 scripts/traffic_from_logs.py --days 180 "
            f"--csv {LOGS_CSV.relative_to(PROJECT_ROOT)}",
            language="bash",
        )
        return

    earliest = min(
        f["date"].iloc[0] for f in (ga4, logs) if f is not None and not f.empty
    )

    controls = st.columns([2, 1, 2])
    with controls[0]:
        metric = st.selectbox("Metric", list(GA4_METRICS.values()), index=0)
    with controls[1]:
        window = st.number_input("Window (days)", min_value=2, max_value=90,
                                 value=7, step=1)
    with controls[2]:
        use_launch = st.checkbox("Start from relaunch date", value=False)
        launch = None
        if use_launch:
            launch = st.date_input(
                "Relaunch",
                value=datetime.strptime(earliest, "%Y-%m-%d").date(),
            ).isoformat()

    frames = []
    if ga4 is not None:
        column = next((k for k, v in GA4_METRICS.items() if v == metric), None)
        if column and column in ga4.columns:
            frames.append(_run_rate(ga4, column, window, launch, "GA4"))
    if logs is not None:
        column = next((k for k, v in LOG_METRICS.items() if v == metric), None)
        if column and column in logs.columns:
            frames.append(_run_rate(logs, column, window, launch, "Server logs"))

    frames = [f for f in frames if not f.empty]
    if not frames:
        st.warning(
            f"Not enough data for a {window}-day average yet — the line "
            f"starts once {window} days exist"
            + (" after the relaunch date." if launch else ".")
        )
        return

    data = pd.concat(frames, ignore_index=True)
    multi = data["series"].nunique() > 1

    # Hero numbers instead of labelling every point.
    stats = st.columns(len(frames))
    for col, frame in zip(stats, frames):
        series = frame.sort_values("date")
        latest = series["value"].iloc[-1]
        delta = None
        if len(series) > window:
            previous = series["value"].iloc[-1 - window]
            if previous:
                delta = f"{(latest - previous) / previous * 100:+.1f}% vs {window}d ago"
        col.metric(f"{series['series'].iloc[0]} — {metric.lower()}/day",
                   f"{latest:,.1f}", delta)

    st.altair_chart(_chart(data, window, launch, multi), use_container_width=True)
    st.caption(
        f"Each point is the mean of the {window} days ending that day. "
        "The line begins where a full window first exists."
        + (" Dashed rule marks the relaunch date." if launch else "")
    )

    with st.expander("Table view"):
        table = data.pivot_table(index="date", columns="series",
                                 values="value").round(1)
        st.dataframe(table.sort_index(ascending=False),
                     use_container_width=True)


# Auto-render when Streamlit runs this file directly (multipage mode)
render()
