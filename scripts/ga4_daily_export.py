"""Export GA4 daily metrics to CSV and report growth rates.

Pulls one row per day from the GA4 Data API so growth can be analysed in a
spreadsheet instead of squinting at the Reports snapshot. Pairs with
scripts/traffic_from_logs.py, which produces the same daily shape from the
server's own access logs — run both and compare.

Usage:
    python3 scripts/ga4_daily_export.py --days 90 --csv ga4_daily.csv
    python3 scripts/ga4_daily_export.py --summary            # growth, no file
    python3 scripts/ga4_daily_export.py --by-channel ga4_channels.csv

Setup (one-off):
    pip install google-analytics-data
    export GA4_PROPERTY_ID=123456789
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

The service account needs Viewer on the GA4 property (Admin → Property
Access Management), using the account's client_email from that JSON file.

Two things this script will not pretend away:

  - **The last days are incomplete.** GA4 keeps processing a day for up to
    ~48h, so the most recent rows undercount. They are exported, but the
    growth summary ignores the trailing --provisional-days (default 2) and
    says so. Comparing a part-day against a settled one is how you end up
    reading a -80% that never happened.
  - **Active vs new users is not a retention measure.** new_users counts
    first-ever visits. active_users minus new_users is the number of people
    active in the window whose first visit predates it — it says nothing
    about whether people come back within the window. For that, use GA4's
    Retention report.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# One row per day. Order matters — it's the CSV column order.
METRICS = ("activeUsers", "newUsers", "sessions", "screenPageViews")
CSV_COLUMNS = ("date", "active_users", "new_users", "sessions", "page_views")
CHANNEL_COLUMNS = ("date", "channel", "active_users", "sessions", "page_views")

DEFAULT_PROVISIONAL_DAYS = 2


# ---------------------------------------------------------------------------
# Pure transforms — no API, no I/O. These carry the logic worth testing.
# ---------------------------------------------------------------------------

def parse_ga4_date(value: str) -> str:
    """GA4 returns dates as YYYYMMDD; emit ISO so spreadsheets sort them."""
    return f"{value[0:4]}-{value[4:6]}-{value[6:8]}"


def day_range(start: date, end: date) -> list[str]:
    """Every ISO date from start to end inclusive."""
    days, cursor = [], start
    while cursor <= end:
        days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days


def fill_missing_days(rows: list[dict], start: date, end: date) -> list[dict]:
    """Zero-fill days GA4 omitted.

    GA4 returns no row for a day with no activity. Left as a gap, a growth
    rate computed from the file silently compares non-adjacent days.
    """
    by_date = {row["date"]: row for row in rows}
    filled = []
    for day in day_range(start, end):
        filled.append(by_date.get(day, {
            "date": day, "active_users": 0, "new_users": 0,
            "sessions": 0, "page_views": 0,
        }))
    return filled


def period_totals(rows: list[dict], metric: str, size: int) -> list[tuple[str, str, int]]:
    """Sum `metric` over consecutive blocks of `size` days, newest last.

    Returns (start_date, end_date, total) per complete block. Partial
    leading blocks are dropped rather than compared against full ones.
    """
    blocks = []
    # Walk backwards so the most recent block is always complete.
    for end in range(len(rows), 0, -size):
        start = end - size
        if start < 0:
            break
        chunk = rows[start:end]
        blocks.append((
            chunk[0]["date"], chunk[-1]["date"],
            sum(int(r[metric]) for r in chunk),
        ))
    return list(reversed(blocks))


def rolling_average(rows: list[dict], metric: str, window: int = 7) -> list[dict]:
    """Trailing `window`-day mean of `metric`, one point per day.

    The run rate: each point is the average of that day and the `window`-1
    days before it. Days before a full window exists produce no point —
    a partial average would start the line artificially low and read as
    growth that never happened.

    Returns [{"date": ..., "value": float}], oldest first.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    points = []
    for index in range(window - 1, len(rows)):
        chunk = rows[index - window + 1:index + 1]
        total = sum(int(r[metric]) for r in chunk)
        points.append({"date": rows[index]["date"], "value": total / window})
    return points


def clip_from(rows: list[dict], since: str | None) -> list[dict]:
    """Drop rows before `since` (ISO date). No-op when since is None."""
    if not since:
        return rows
    return [r for r in rows if r["date"] >= since]


def trim_leading_zeros(rows: list[dict], metric: str) -> list[dict]:
    """Drop empty days at the start, before anything was being recorded.

    A run of zeros before the site had traffic (or before tracking was
    installed) drags the first window's average down, so the line opens
    below the real rate and climbs — growth that is an artefact of the
    start date. Zeros *after* data begins are real and are kept.
    """
    for index, row in enumerate(rows):
        if int(row[metric]) > 0:
            return rows[index:]
    return []


def growth_rate(previous: int, current: int) -> float | None:
    """Percentage change, or None when there's no baseline to divide by."""
    if previous == 0:
        return None
    return (current - previous) / previous * 100.0


def summarise(rows: list[dict], metric: str, size: int) -> list[str]:
    """Render period totals and period-on-period growth as text lines."""
    blocks = period_totals(rows, metric, size)
    if len(blocks) < 2:
        return [f"Not enough complete {size}-day periods to show growth."]
    lines = [f"{metric} — {size}-day periods",
             f"{'Period':<26}{'Total':>9}{'Change':>10}"]
    lines.append("-" * 45)
    for index, (start, end, total) in enumerate(blocks):
        if index == 0:
            change = "—"
        else:
            rate = growth_rate(blocks[index - 1][2], total)
            change = "n/a" if rate is None else f"{rate:+.1f}%"
        lines.append(f"{start} to {end:<10}{total:>9,}{change:>10}")
    return lines


# ---------------------------------------------------------------------------
# API access
# ---------------------------------------------------------------------------

def load_client():
    """Import and construct the GA4 client, with a usable error if absent."""
    try:
        from google.analytics.data_v1beta import BetaAnalyticsDataClient
    except ImportError:
        print(
            "google-analytics-data is not installed.\n"
            "  pip install google-analytics-data",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        print(
            "GOOGLE_APPLICATION_CREDENTIALS is not set — point it at the "
            "service account JSON.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return BetaAnalyticsDataClient()


def fetch_daily(client, property_id: str, start: date, end: date) -> list[dict]:
    """One row per day: active users, new users, sessions, page views."""
    from google.analytics.data_v1beta.types import (
        DateRange, Dimension, Metric, RunReportRequest,
    )

    request = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[Dimension(name="date")],
        metrics=[Metric(name=name) for name in METRICS],
        date_ranges=[DateRange(start_date=start.isoformat(),
                               end_date=end.isoformat())],
    )
    response = client.run_report(request)
    rows = []
    for row in response.rows:
        values = [v.value for v in row.metric_values]
        rows.append({
            "date": parse_ga4_date(row.dimension_values[0].value),
            "active_users": int(values[0]),
            "new_users": int(values[1]),
            "sessions": int(values[2]),
            "page_views": int(values[3]),
        })
    rows.sort(key=lambda r: r["date"])
    return rows


def fetch_by_channel(client, property_id: str, start: date, end: date) -> list[dict]:
    """One row per day per acquisition channel."""
    from google.analytics.data_v1beta.types import (
        DateRange, Dimension, Metric, RunReportRequest,
    )

    request = RunReportRequest(
        property=f"properties/{property_id}",
        dimensions=[Dimension(name="date"),
                    Dimension(name="sessionDefaultChannelGroup")],
        metrics=[Metric(name=n) for n in
                 ("activeUsers", "sessions", "screenPageViews")],
        date_ranges=[DateRange(start_date=start.isoformat(),
                               end_date=end.isoformat())],
    )
    response = client.run_report(request)
    rows = []
    for row in response.rows:
        values = [v.value for v in row.metric_values]
        rows.append({
            "date": parse_ga4_date(row.dimension_values[0].value),
            "channel": row.dimension_values[1].value or "(not set)",
            "active_users": int(values[0]),
            "sessions": int(values[1]),
            "page_views": int(values[2]),
        })
    rows.sort(key=lambda r: (r["date"], r["channel"]))
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(path: str, columns: tuple[str, ...], rows: list[dict]) -> None:
    """Write rows to a CSV file, or to stdout when path is '-'."""
    if path == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} row(s) to {path}", file=sys.stderr)


def load_project_env() -> None:
    """Read .env, so the CLI and the dashboard share one configuration.

    Without this the dashboard finds GA4_PROPERTY_ID and this script does
    not, and the same credentials appear to work in one place and fail in
    the other. Real environment variables still win over the file.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def main() -> int:
    # Before argparse, whose defaults read the environment.
    load_project_env()
    parser = argparse.ArgumentParser(
        description="Export GA4 daily metrics and report growth rates.",
    )
    parser.add_argument("--property-id", default=os.environ.get("GA4_PROPERTY_ID"),
                        help="GA4 property ID (or set GA4_PROPERTY_ID).")
    parser.add_argument("--days", type=int, default=90,
                        help="Days back to export, ending today (default: 90).")
    parser.add_argument("--csv", metavar="PATH", default="-",
                        help="Daily CSV output path, - for stdout (default: -).")
    parser.add_argument("--by-channel", metavar="PATH",
                        help="Also write a per-channel daily CSV here.")
    parser.add_argument("--summary", action="store_true",
                        help="Print period totals and growth rates to stderr.")
    parser.add_argument("--period", type=int, default=7, metavar="N",
                        help="Period length in days for growth (default: 7).")
    parser.add_argument("--metric", default="active_users",
                        choices=list(CSV_COLUMNS[1:]),
                        help="Metric to summarise (default: active_users).")
    parser.add_argument("--provisional-days", type=int,
                        default=DEFAULT_PROVISIONAL_DAYS, metavar="N",
                        help="Trailing days GA4 hasn't finished processing; "
                             "excluded from the growth summary (default: 2).")
    parser.add_argument("--complete-only", action="store_true",
                        help="Also drop those provisional days from the CSV.")
    parser.add_argument("--since", metavar="YYYY-MM-DD",
                        help="Ignore data before this date (e.g. relaunch day).")
    parser.add_argument("--from-first-data", action="store_true",
                        help="Start at the first day with any data, dropping "
                             "empty days before tracking began.")
    parser.add_argument("--rolling-window", type=int, default=7, metavar="N",
                        help="Trailing average window in days; adds a "
                             "rolling_N column. 0 disables (default: 7).")
    args = parser.parse_args()

    if not args.property_id:
        print("No property ID — pass --property-id or set GA4_PROPERTY_ID.",
              file=sys.stderr)
        return 2

    end = date.today()
    start = end - timedelta(days=max(args.days, 1) - 1)

    client = load_client()
    rows = fill_missing_days(
        fetch_daily(client, args.property_id, start, end), start, end,
    )

    rows = clip_from(rows, args.since)
    if args.from_first_data:
        rows = trim_leading_zeros(rows, args.metric)
    cutoff = max(args.provisional_days, 0)
    settled = rows[:-cutoff] if cutoff else rows

    out_rows = settled if args.complete_only else rows
    columns = CSV_COLUMNS
    if args.rolling_window > 0:
        column = f"rolling_{args.rolling_window}"
        columns = CSV_COLUMNS + (column,)
        # Blank until a full window exists, so the line starts where the
        # run rate actually becomes meaningful.
        by_date = {p["date"]: p["value"] for p in
                   rolling_average(out_rows, args.metric, args.rolling_window)}
        for row in out_rows:
            value = by_date.get(row["date"])
            row[column] = "" if value is None else f"{value:.2f}"

    write_csv(args.csv, columns, out_rows)

    if args.by_channel:
        write_csv(args.by_channel, CHANNEL_COLUMNS,
                  fetch_by_channel(client, args.property_id, start, end))

    if args.summary:
        if cutoff:
            print(f"Excluding the last {cutoff} day(s) — GA4 is still "
                  f"processing them.\n", file=sys.stderr)
        for line in summarise(settled, args.metric, args.period):
            print(line, file=sys.stderr)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            raise SystemExit(0)
