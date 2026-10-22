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
from datetime import date
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
# Where the chart opens. The spring collapse and the empty June-July shrink
# everything after them to a sliver; the averages are still computed over the
# full fetch, so the line starts at this date with a real value.
DEFAULT_SHOW_FROM = date(2026, 8, 1)
CACHE_TTL_SECONDS = 3600

# Categorical slots 1 and 2 from the reference palette, light surface.
# Validated: worst adjacent pair ΔE 24.7 (protan), 33.6 normal vision,
# both >= 3:1 against the surface.
SERIES_COLOURS = ["#2a78d6", "#eb6834"]
GRID_COLOUR = "#e6e5e1"
AXIS_INK = "#52514e"

# GA4 column -> label. The log export uses its own names for the same ideas.
# Engaged sessions comes first, so it is the default: headless bots that land
# and do nothing (a third of GA4 users on 2026-10-21) score zero on it, and
# GA4 has recorded it for every day, so the whole history is restated.
GA4_METRICS = {
    "engaged_sessions": "Engaged sessions",
    "active_users": "Active users",
    "new_users": "New users",
    "sessions": "Sessions",
    "page_views": "Page views",
}
LOG_METRICS = {"visitors": "Active users", "page_views": "Page views"}

# Drawn faint behind Engaged sessions: the gap between the two is the bots.
BOT_SHADOW_LABEL = "GA4 active users (incl. bots)"
SHADOW_COLOUR = "#b9b8b3"


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


def _min_sec(minutes: float) -> str:
    """3.04 -> '3m 02s': decimal minutes read as a rounded whole minute."""
    total = round(minutes * 60)
    return f"{total // 60}m {total % 60:02d}s"


def _chart(data: pd.DataFrame, window: int, multi: bool):
    """Line chart: 2px lines, recessive axes, crosshair tooltip, no legend
    for a single series (the title names it)."""
    import altair as alt

    base = alt.Chart(data)
    hover = alt.selection_point(
        fields=["date"], nearest=True, on="mouseover", empty=False,
    )

    # Fixed series -> colour, so the shadow line is always the grey one
    # whatever order Altair would sort the names into.
    names = list(dict.fromkeys(data["series"]))
    solid = [n for n in names if n != BOT_SHADOW_LABEL]
    domain = solid + [n for n in names if n == BOT_SHADOW_LABEL]
    palette = SERIES_COLOURS[:len(solid)] + (
        [SHADOW_COLOUR] if BOT_SHADOW_LABEL in names else [])
    colour = (
        alt.Color("series:N", title=None,
                  scale=alt.Scale(domain=domain, range=palette),
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
        strokeDash=alt.condition(
            alt.datum.series == BOT_SHADOW_LABEL,
            alt.value([4, 3]), alt.value([1, 0])),
    )

    # Invisible wide target so the tooltip is easy to hit on a sparse line.
    points = base.mark_point(size=120, opacity=0).encode(
        x="date:T", y="value:Q",
        tooltip=[alt.Tooltip("date:T", title="Date"),
                 # A "label" column, when given, is the value already
                 # formatted (e.g. 3m 02s); d3 formats can't do that.
                 alt.Tooltip("label:N", title="Run rate")
                 if "label" in data.columns else
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
    with header[1]:
        show_from = st.date_input("Show from", value=DEFAULT_SHOW_FROM)
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
            label = (f"{ga4_label} engaged sessions"
                     if column == "engaged_sessions" else ga4_label)
            frames.append(_run_rate(ga4, column, window, label))
        if column == "engaged_sessions" and "active_users" in ga4.columns:
            frames.append(_run_rate(ga4, "active_users", window,
                                    BOT_SHADOW_LABEL))
    if logs is not None:
        column = next((k for k, v in LOG_METRICS.items() if v == metric), None)
        if column and column in logs.columns:
            frames.append(_run_rate(logs, column, window, "Server logs"))

    since = show_from.isoformat()
    frames = [f[f["date"] >= since] for f in frames]
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
        name = series["series"].iloc[0]
        title = (f"{name}/day" if name == BOT_SHADOW_LABEL
                 or name.endswith("engaged sessions")
                 else f"{name} — {metric.lower()}/day")
        col.metric(title, f"{latest:,.1f}", delta)

    st.altair_chart(_chart(data, window, multi), use_container_width=True)
    st.caption(
        f"Each point is the mean of the {window} days ending that day. "
        "The line begins one window after the first day with data — "
        "empty days before that are excluded."
        + (" Engaged sessions = sessions of 10s+, 2+ pages, or a key event; "
           "bots that land and leave score zero. The grey dashed line is GA4 "
           "active users including them. Sessions and users are different "
           "units (one person can visit twice), so the gap is only a rough "
           "guide to the bot share."
           if metric == "Engaged sessions" else "")
    )

    # Second chart, same dates: how much each real visitor uses the site.
    # Divided by engaged users (those who fired user_engagement), not all
    # active users, so bots that land and leave don't dilute it.
    if (ga4 is not None and "engagement_seconds" in ga4.columns
            and "engaged_users" in ga4.columns):
        seconds = _run_rate(ga4, "engagement_seconds", window, "seconds")
        users = _run_rate(ga4, "engaged_users", window, "users")
        # Ratio of the two window means = window seconds / window users.
        minutes = seconds.merge(users, on="date", suffixes=("_s", "_u"))
        minutes = minutes[(minutes["date"] >= since)
                          & (minutes["value_u"] > 0)]
        if not minutes.empty:
            minutes = pd.DataFrame({
                "date": minutes["date"],
                "value": minutes["value_s"] / minutes["value_u"] / 60,
                "series": "Minutes per engaged user",
            })
            minutes["label"] = minutes["value"].map(_min_sec)
            st.subheader("Engagement — minutes per engaged user")
            series = minutes.sort_values("date")
            latest = series["value"].iloc[-1]
            delta = None
            if len(series) > window:
                previous = series["value"].iloc[-1 - window]
                if previous:
                    delta = (f"{(latest - previous) / previous * 100:+.1f}% "
                             f"vs {window}d ago")
            st.metric("GA4 time per engaged user", _min_sec(latest), delta)
            st.altair_chart(_chart(minutes, window, False),
                            use_container_width=True)
            st.caption(
                f"Foreground time divided by users who had a page in the "
                f"foreground (GA4 user_engagement event), each a {window}-day "
                f"average. Bots that land and leave are in neither number."
            )

    with st.expander("Table view"):
        # The table is bot-free: the all-users shadow belongs on the chart
        # only, as the visible gap; it has no place among the counts.
        table = data[data["series"] != BOT_SHADOW_LABEL].pivot_table(
            index="date", columns="series", values="value").round(1)
        # Every series above is a run rate; say so, or a 7am row reads as
        # "60 users today".
        table.columns = [f"{name} — {window}-day avg" for name in table.columns]
        if ga4 is not None and "engaged_users" in ga4.columns:
            daily = pd.to_numeric(ga4.set_index("date")["engaged_users"],
                                  errors="coerce")
            table.insert(0, "Engaged users that day (no bots)",
                         daily.reindex(table.index).astype("Int64"))
        st.dataframe(table.sort_index(ascending=False),
                     use_container_width=True)


# Auto-render when Streamlit runs this file directly (multipage mode)
render()
