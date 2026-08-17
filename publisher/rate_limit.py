"""Per-key, per-IP rate limiting for the widget API.

A deliberate near-copy of ``web/rate_limit.py`` rather than an import — see
``publisher_build_decisions``: this package must stay liftable. Two real
differences: buckets are keyed by publisher key *and* client IP (so one
customer's traffic cannot exhaust another's), and over-limit returns JSON,
because every caller here is fetch(), never a browser navigation.

Same honest limits as the original: fixed window, so a client can burst to
about 2x across a boundary; fails open if SQLite is unreachable; and it does
not stop a rotating-IP harvester. It makes bulk extraction slow and visible,
which is what the licensing design asks of it — not a wall.
"""

import sqlite3
import time
from pathlib import Path
from threading import Lock

from flask import current_app, jsonify, request

_init_lock = Lock()
_initialised = False


def _db_path():
    return Path(current_app.config["RATE_LIMIT_DB"])


def _ensure_db():
    global _initialised
    if _initialised:
        return
    with _init_lock:
        if _initialised:
            return
        path = _db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=5)
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


def _connect():
    conn = sqlite3.connect(str(_db_path()), timeout=5, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=2000")
    return conn


def _check_and_increment(scope, ip, limit, window):
    _ensure_db()
    now = time.time()
    conn = _connect()
    try:
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
            return False, max(1, int(window - (now - window_start)) + 1)
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
        # Fail open: a SQLite hiccup must not take a publisher's puzzle page
        # down at 7am. The trade is a brief window with no limiting.
        return True, 0
    finally:
        conn.close()


def check(scope, limit, window):
    """Returns a 429 JSON response to hand back, or None when allowed."""
    if not current_app.config.get("RATE_LIMIT_ENABLED", True):
        return None
    ip = request.remote_addr or "unknown"
    allowed, retry_after = _check_and_increment(scope, ip, limit, window)
    if allowed:
        return None
    response = jsonify({"error": "rate limited", "retry_after": retry_after})
    response.status_code = 429
    response.headers["Retry-After"] = str(retry_after)
    return response
