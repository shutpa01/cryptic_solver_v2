"""The publisher solver widget — a standalone embeddable app.

Phase one of the licensing product. Served from its own Flask app on its own
port so it can be lifted into its own repository without untangling; it shares
the databases but imports nothing from ``web``.

Shape: the publisher's page frames `/embed/<source>/<number>?k=<key>`. That
page is the whole widget — one engine (grid model, selection, typing, crossing
counts, endpoint calls) under one of three shells (Telegraph first, then Times,
then Guardian). Because it is an iframe on our own origin, every API call is
same-origin and no CORS is involved anywhere.
"""

from flask import Flask

from publisher.config import config_by_name


def create_app(config_name="development"):
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])

    from publisher.routes.embed import bp as embed_bp
    from publisher.routes.api import bp as api_bp

    app.register_blueprint(embed_bp)
    app.register_blueprint(api_bp)

    @app.teardown_appcontext
    def _close_db(_exception=None):
        # web.wfw_read (the full-explanation breakdown) opens its connection
        # through web.db.get_db, which parks it on flask.g and relies on the
        # site's teardown to close it. This app has to do that itself or every
        # explanation request leaks a SQLite handle.
        from flask import g
        connection = g.pop("db", None)
        if connection is not None:
            connection.close()

    @app.after_request
    def _headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response

    return app
