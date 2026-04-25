"""Per-IP rate limiting for content and expensive endpoints.

Cross-worker safe: state lives in a small SQLite DB at
``data/rate_limits.db`` so the limit holds across all gunicorn workers.

Honest scope of this protection:

- SQLite-backed store. State persists across restarts (intentional —
  a scraper restarting their connection should not reset their bucket)
  and is shared across all gunicorn workers via WAL mode + IMMEDIATE
  transactions. Read/write cost is sub-millisecond on local FS for
  the volumes this app sees.
- Fixed window. A determined client can burst up to ~2x the limit
  across a window boundary. Good enough as an anti-scraping measure
  ("make scraping expensive"), not a fairness mechanism.
- Client IP comes from request.remote_addr. ProxyFix in
  web/__init__.py rewrites that to the real client IP using
  X-Forwarded-For from nginx. The nginx config at
  /etc/nginx/sites-enabled/cordelia sets that header (verified
  2026-04-25). If anyone changes the nginx config to drop the
  header, all traffic will appear to come from one IP and the limit
  becomes a global bucket.
- Admin requests bypass via g.is_admin (set by check_admin in
  web/__init__.py).
- Kill switch: app.config["RATE_LIMIT_ENABLED"] = False disables
  all limits without a code change.
- Limits do NOT defend against rotating-IP scrapers (cloud-hosted
  attackers can cycle IPs cheaply). They make naive scraping
  expensive; they are not a wall.

Verifying in production after deploy:
    1. From one client, hit /reveal more than 30 times in 60s.
       Expect 429 with Retry-After header on subsequent calls.
    2. From a second client, the bucket should be independent — no 429.
    3. Restart gunicorn and confirm the bucket persists for ~window
       seconds (not reset by restart).
"""

import sqlite3
import time
from functools import wraps
from pathlib import Path
from threading import Lock

from flask import current_app, g, make_response, render_template, request


_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rate_limits.db"
_init_lock = Lock()
_initialised = False


def _ensure_db() -> None:
    """Create the rate_limits table if missing. Idempotent, thread-safe."""
    global _initialised
    if _initialised:
        return
    with _init_lock:
        if _initialised:
            return
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), timeout=5)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS rate_limits (
                    scope TEXT NOT NULL,
                    ip TEXT NOT NULL,
                    window_start REAL NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY (scope, ip)
                )"""
            )
            conn.commit()
        finally:
            conn.close()
        _initialised = True


def _connect() -> sqlite3.Connection:
    """Open a short-lived connection in WAL mode."""
    conn = sqlite3.connect(str(_DB_PATH), timeout=5, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=2000")
    return conn


def _client_ip() -> str:
    return request.remote_addr or "unknown"


def _is_enabled() -> bool:
    try:
        return bool(current_app.config.get("RATE_LIMIT_ENABLED", True))
    except RuntimeError:
        return True


def _check_and_increment(scope: str, ip: str, limit: int, window: int):
    """Atomically check + increment the bucket. Returns (allowed, retry_after).

    retry_after is only meaningful when allowed is False.
    """
    _ensure_db()
    now = time.time()
    conn = _connect()
    try:
        # IMMEDIATE acquires a write lock immediately, serialising writers.
        # WAL mode lets concurrent readers proceed; we only need exclusive
        # ordering between writers within the same scope+ip pair.
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT window_start, count FROM rate_limits WHERE scope=? AND ip=?",
            (scope, ip),
        ).fetchone()
        if row is None or now - row[0] > window:
            conn.execute(
                "INSERT OR REPLACE INTO rate_limits (scope, ip, window_start, count) "
                "VALUES (?, ?, ?, 1)",
                (scope, ip, now),
            )
            conn.execute("COMMIT")
            return True, 0
        window_start, count = row[0], row[1]
        if count >= limit:
            conn.execute("COMMIT")
            retry_after = max(1, int(window - (now - window_start)) + 1)
            return False, retry_after
        conn.execute(
            "UPDATE rate_limits SET count=count+1 WHERE scope=? AND ip=?",
            (scope, ip),
        )
        conn.execute("COMMIT")
        return True, 0
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        # Fail open — a SQLite hiccup must not take the site down. The
        # tradeoff is that a transient DB failure briefly disables the
        # limiter; combined with WAL mode this should be very rare.
        return True, 0
    finally:
        conn.close()


def _is_htmx_request() -> bool:
    """Detect HTMX-initiated requests via the HX-Request header."""
    return request.headers.get("HX-Request", "").lower() == "true"


def _build_429_response(retry_after: int):
    """Render a friendly 429 — HTML fragment for HTMX, full page otherwise.

    Both responses include the Retry-After header (HTTP standard) so any
    client tooling that respects it can back off automatically.
    """
    if _is_htmx_request():
        body = render_template(
            "partials/rate_limit_fragment.html",
            retry_after=retry_after,
        )
    else:
        body = render_template("429.html", retry_after=retry_after)
    resp = make_response(body, 429)
    resp.headers["Retry-After"] = str(retry_after)
    return resp


def rate_limit(scope: str, limit: int, window: int):
    """Per-IP rate limit decorator (cross-worker safe via SQLite).

    Args:
        scope: namespace string. Endpoints sharing a scope share a bucket.
        limit: max requests per window per IP.
        window: window size in seconds.

    Behaviour:
        - Admin (g.is_admin) bypasses.
        - When app.config['RATE_LIMIT_ENABLED'] is False, bypasses.
        - Over the limit returns 429 with Retry-After header. The body
          is a styled HTML fragment for HTMX requests (so it swaps into
          the same target the user clicked) or a full templated page
          otherwise.
        - On SQLite error: fails open (logs nothing — caller's fault to
          notice). Trade made deliberately to avoid taking the site
          offline if the rate-limit DB is unreachable.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if not _is_enabled() or getattr(g, "is_admin", False):
                return fn(*args, **kwargs)
            allowed, retry_after = _check_and_increment(
                scope, _client_ip(), limit, window
            )
            if not allowed:
                return _build_429_response(retry_after)
            return fn(*args, **kwargs)

        return wrapped

    return decorator


def _testing_reset() -> None:
    """Clear all buckets. Tests only."""
    _ensure_db()
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM rate_limits")
        conn.execute("COMMIT")
    finally:
        conn.close()


def gc_old_entries(max_age: int = 3600) -> int:
    """Delete bucket entries older than max_age seconds. Returns rows deleted.

    Optional housekeeping — call from a cron or admin route. Not required
    for correctness (stale rows are simply overwritten on next request).
    """
    _ensure_db()
    cutoff = time.time() - max_age
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "DELETE FROM rate_limits WHERE window_start < ?", (cutoff,)
        )
        deleted = cursor.rowcount
        conn.execute("COMMIT")
        return deleted
    finally:
        conn.close()
