"""SQLite database access — read-only, per-request connection."""

import sqlite3

from flask import current_app, g


def get_db():
    """Return a read-only SQLite connection for the current request."""
    if "db" not in g:
        db_path = current_app.config["CLUES_DB"]
        uri = f"file:{db_path}?mode=ro"
        g.db = sqlite3.connect(uri, uri=True)
        g.db.row_factory = sqlite3.Row
    return g.db


def get_admin_db():
    """Return a read-write SQLite connection for admin operations."""
    if "admin_db" not in g:
        db_path = current_app.config["CLUES_DB"]
        g.admin_db = sqlite3.connect(db_path)
        g.admin_db.row_factory = sqlite3.Row
    return g.admin_db


def close_db(e=None):
    """Close database connections at the end of the request."""
    for key in ("db", "admin_db"):
        conn = g.pop(key, None)
        if conn is not None:
            conn.close()


def init_app(app):
    """Register teardown hook with the Flask app."""
    app.teardown_appcontext(close_db)
