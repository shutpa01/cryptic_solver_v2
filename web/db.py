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


def close_db(e=None):
    """Close the database connection at the end of the request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_app(app):
    """Register teardown hook with the Flask app."""
    app.teardown_appcontext(close_db)
