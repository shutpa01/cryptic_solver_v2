"""Live GA4 access for the dashboard.

The Growth page pulls its own numbers through here, so nobody has to
remember to run an export script. Everything the page needs is one of:

    ready()   -- can we talk to GA4 at all, and if not, exactly what's missing
    fetch()   -- daily rows straight from the API
    cached()  -- the last successful fetch, read back from disk

Configuration lives in .env, like the rest of the project:

    GA4_PROPERTY_ID=123456789
    GOOGLE_APPLICATION_CREDENTIALS=secrets/ga4-service-account.json

A relative credentials path is resolved against the project root, so the
same .env works from the dashboard, a script, or a scheduled task.

Two deliberate choices:

  - **A fetch failure must not blank the chart.** Every successful fetch is
    written through to data/ga4_daily.csv. If GA4 is unreachable the page
    falls back to that file and says it is doing so, rather than showing an
    empty panel that looks like "no traffic".
  - **Nothing here raises SystemExit.** scripts/ga4_daily_export.py exits
    the process when the package or credentials are absent, which is right
    for a CLI and fatal inside Streamlit. This module reports the same
    conditions as values and exceptions the page can render.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "ga4_daily_export.py"
DATA_DIR = PROJECT_ROOT / "data"
GA4_CSV = DATA_DIR / "ga4_daily.csv"
LOGS_CSV = DATA_DIR / "logs_daily.csv"

PACKAGE_NAME = "google-analytics-data"


class Ga4Error(RuntimeError):
    """GA4 could not be reached or refused the request."""


@dataclass
class Readiness:
    """What is configured, and what is missing — in the user's terms."""

    ok: bool
    missing: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.ok:
            return "GA4 is configured."
        return "GA4 is not configured yet: " + "; ".join(self.missing) + "."


def _load_env() -> None:
    """Read .env if python-dotenv is available, matching the project pattern.

    Values already in the environment win, so a shell export still overrides
    the file. Absent dotenv, real environment variables still work.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def export_module():
    """Load scripts/ga4_daily_export.py by path.

    The transforms and the API calls both live there. Importing rather than
    reimplementing is the point: the chart and any CSV export cannot drift
    apart into two different definitions of a run rate.
    """
    spec = importlib.util.spec_from_file_location(
        "ga4_daily_export", EXPORT_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def property_id() -> str | None:
    _load_env()
    value = (os.environ.get("GA4_PROPERTY_ID") or "").strip()
    return value or None


def credentials_path() -> Path | None:
    """The service-account JSON, as an absolute path, if one is configured."""
    _load_env()
    raw = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def package_installed() -> bool:
    """Is the GA4 client library importable?

    find_spec raises rather than returning None when a *parent* package is
    missing — `google` exists as a namespace package here while
    `google.analytics` does not — so the absent case has to be caught, not
    tested for.
    """
    try:
        return importlib.util.find_spec("google.analytics.data_v1beta") is not None
    except (ImportError, ValueError):
        return False


def ready() -> Readiness:
    """Check every precondition, reporting all failures at once.

    Reporting them together matters: finding out about the missing package
    only after installing credentials wastes a round trip.
    """
    missing = []
    if not package_installed():
        missing.append(f"the {PACKAGE_NAME} package is not installed")
    if not property_id():
        missing.append("GA4_PROPERTY_ID is not set")
    path = credentials_path()
    if path is None:
        missing.append("GOOGLE_APPLICATION_CREDENTIALS is not set")
    elif not path.exists():
        missing.append(f"the credentials file {path} does not exist")
    return Readiness(ok=not missing, missing=missing)


def _client():
    """Construct the GA4 client, as an exception rather than an exit."""
    state = ready()
    if not state.ok:
        raise Ga4Error(state.summary)
    # The Google client reads this variable directly, so make sure it sees
    # the resolved absolute path even when .env held a relative one.
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path())
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    return BetaAnalyticsDataClient()


def fetch(days: int = 180) -> list[dict]:
    """Daily rows from GA4, zero-filled and oldest first.

    Raises Ga4Error with a readable message for anything the page should
    show the user rather than swallow.
    """
    module = export_module()
    end = date.today()
    start = end - timedelta(days=max(days, 1) - 1)
    client = _client()
    try:
        rows = module.fetch_daily(client, property_id(), start, end)
    except Ga4Error:
        raise
    except Exception as error:  # noqa: BLE001 - surfaced to the page verbatim
        raise Ga4Error(f"GA4 refused the request: {error}") from error
    return module.fill_missing_days(rows, start, end)


def write_cache(rows: list[dict]) -> None:
    """Write through to data/ga4_daily.csv so a later outage still charts."""
    if not rows:
        return
    module = export_module()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    module.write_csv(str(GA4_CSV), module.CSV_COLUMNS, rows)


def cached() -> list[dict] | None:
    """The last successful fetch, or None when nothing has been cached."""
    if not GA4_CSV.exists():
        return None
    import csv

    with open(GA4_CSV, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows or None


def cache_age_days() -> int | None:
    """How stale the cached file is, in days, or None when there isn't one."""
    rows = cached()
    if not rows:
        return None
    try:
        last = date.fromisoformat(rows[-1]["date"])
    except (KeyError, ValueError):
        return None
    return (date.today() - last).days
