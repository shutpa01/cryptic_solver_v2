"""Growth — trailing 7-day run rate, pulled from GA4 automatically.

Each point is the average of that day and the six before it. The line
starts on day 7, where a full window first exists: a partial average
would start low and read as growth that never happened.

The page fetches GA4 itself — there is no export script to remember. Set
these once in .env and the chart fills on load:

    GA4_PROPERTY_ID=123456789
    GOOGLE_APPLICATION_CREDENTIALS=secrets/ga4-service-account.json

Every successful fetch is written through to data/ga4_daily.csv, so a GA4
outage falls back to the last good data and says so, rather than showing an
empty chart that reads as "no traffic".

The server-log series is the honest second opinion — GA4 misses anyone who
blocks its tag, so the two lines are expected to disagree and the gap is
itself the finding. It still comes from a CSV, because reading it means
reaching the live server:

    python3 scripts/traffic_from_logs.py --days 180 --csv data/logs_daily.csv

The rolling maths comes from scripts/ga4_daily_export.py rather than being
redone in pandas here, so the chart and the CSV can't disagree.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Streamlit execs this file directly in multipage mode, which may happen
# before app.py has put the project root on the path.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard import ga4_live  # noqa: E402

LOGS_CSV = ga4_live.LOGS_CSV

# How much history to pull. A wider window costs nothing and the rolling
# average needs a run-up before it can draw anything.
DEFAULT_DAYS = 180
CACHE_TTL_SECONDS = 3600

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
def _growth_maths():
    """The tested transforms from scripts/ (no package: load by path)."""
    module = ga4_live.export_module()
    return module.rolling_average, module.trim_leading_zeros


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Fetching from GA4…")
def _fetch_ga4(days: int):
    """Live GA4 rows, cached for an hour.

    Returns (rows, error). The error is returned rather than raised so a
    failed fetch degrades to the cached file instead of breaking the page.
    """
    try:
        rows = ga4_live.fetch(days)
    except ga4_live.Ga4Error as error:
        return None, str(error)
    ga4_live.write_cache(rows)
    return rows, None


def _ga4_data(days: int):
    """GA4 rows with their provenance: (frame, label, warning).

    Live when GA4 answers, the written-through cache when it doesn't, and a
    warning naming the reason whenever the live path failed.
    """
    state = ga4_live.ready()
    if not state.ok:
        fallback = ga4_live.cached()
        if fallback:
            return pd.DataFrame(fallback), "GA4 (cached)", state.summary
        return None, None, state.summary

    rows, error = _fetch_ga4(days)
    if rows:
        return pd.DataFrame(rows), "GA4", None
    fallback = ga4_live.cached()
    if fallback:
        return pd.DataFrame(fallback), "GA4 (cached)", error
    return None, None, error


def _setup_panel(problem: str):
    """Say exactly what is missing, and where it has to go."""
    st.info(problem)
    st.markdown("**To connect GA4**, add these to `.env`:")
    st.code(
        "GA4_PROPERTY_ID=123456789\n"
        "GOOGLE_APPLICATION_CREDENTIALS=secrets/ga4-service-account.json",
        language="bash",
    )
    st.markdown(
        "The service account needs **Viewer** on the property "
        "(GA4 → Admin → Property Access Management), granted to the "
        "`client_email` inside that JSON file. The client library installs with:"
    )
    st.code(f"pip install {ga4_live.PACKAGE_NAME}", language="bash")
    st.caption(
        "The server-log line needs no Google access at all: "
        "`python3 scripts/traffic_from_logs.py --days 180 "
        "--csv data/logs_daily.csv`"
    )


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
              label: str) -> pd.DataFrame:
    """Trailing average as a tidy frame, via the tested pure functions.

    Empty days before data starts are dropped first — otherwise the first
    window averages in the silence and the line opens low, showing growth
    that is only an artefact of where the series begins.
    """
    rolling, trim = _growth_maths()
    rows = frame.to_dict("records")
    # A cached CSV read yields strings, and a blank cell is not a zero —
    # int("") would raise, so drop empties rather than invent a value.
    rows = [r for r in rows if column in r and pd.notna(r[column])
            and str(r[column]).strip() != ""]
    rows = trim(rows, column)
    if not rows:
        return pd.DataFrame(columns=["date", "value", "series"])
    points = rolling(rows, column, window)
    out = pd.DataFrame(points)
    if out.empty:
        return pd.DataFrame(columns=["date", "value", "series"])
    out["series"] = label
    return out


def _chart(data: pd.DataFrame, window: int, multi: bool):
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

    return alt.layer(crosshair, line, points).properties(height=340).configure_view(
        strokeWidth=0,
    )


def render():
    st.header("Growth — run rate")

    header = st.columns([2, 1, 1])
    with header[0]:
        days = st.selectbox("History", [90, DEFAULT_DAYS, 365], index=1,
                            format_func=lambda d: f"{d} days")
    with header[2]:
        st.write("")
        if st.button("Refresh from GA4", use_container_width=True):
            _fetch_ga4.clear()
            st.rerun()

    ga4, ga4_label, warning = _ga4_data(days)
    logs = _read_daily(LOGS_CSV)

    if ga4 is None and logs is None:
        _setup_panel(warning or "No daily data yet.")
        return

    if warning:
        age = ga4_live.cache_age_days()
        stale = (f" Showing cached data — its last day is {age} day(s) old."
                 if age is not None else "")
        st.warning(f"{warning}{stale}")

    controls = st.columns([3, 1])
    with controls[0]:
        metric = st.selectbox("Metric", list(GA4_METRICS.values()), index=0)
    with controls[1]:
        window = st.number_input("Window (days)", min_value=2, max_value=90,
                                 value=7, step=1)

    frames = []
    if ga4 is not None:
        column = next((k for k, v in GA4_METRICS.items() if v == metric), None)
        if column and column in ga4.columns:
            frames.append(_run_rate(ga4, column, window, ga4_label))
    if logs is not None:
        column = next((k for k, v in LOG_METRICS.items() if v == metric), None)
        if column and column in logs.columns:
            frames.append(_run_rate(logs, column, window, "Server logs"))

    frames = [f for f in frames if not f.empty]
    if not frames:
        st.warning(
            f"Not enough data for a {window}-day average yet — the line "
            f"starts once {window} days exist from the first day with data."
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

    st.altair_chart(_chart(data, window, multi), use_container_width=True)
    st.caption(
        f"Each point is the mean of the {window} days ending that day. "
        "The line begins one window after the first day with data — "
        "empty days before that are excluded."
    )

    with st.expander("Table view"):
        table = data.pivot_table(index="date", columns="series",
                                 values="value").round(1)
        st.dataframe(table.sort_index(ascending=False),
                     use_container_width=True)


# Auto-render when Streamlit runs this file directly (multipage mode)
render()
