"""Flask application factory."""

import hmac

from flask import Flask, g, request, session

from web.config import config_by_name
from web import db


WORDPLAY_LABELS = {
    "anagram": "Anagram",
    "charade": "Charade",
    "container": "Container",
    "hidden": "Hidden word",
    "reversal": "Reversal",
    "double_definition": "Double definition",
    "cryptic_definition": "Cryptic definition",
    "homophone": "Homophone",
    "deletion": "Deletion",
    "substitution": "Substitution",
    "spoonerism": "Spoonerism",
    "initial_letters": "Initial letters",
    "alternation": "Alternation",
}


def create_app(config_name=None):
    """Create and configure the Flask application."""
    if config_name is None:
        config_name = "development"

    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])

    # Database teardown
    db.init_app(app)

    # Template filter
    @app.template_filter("wordplay_label")
    def wordplay_label_filter(value):
        if not value:
            return ""
        return WORDPLAY_LABELS.get(value, value.replace("_", " ").title())

    # Admin session activation — permanent cookie survives IP changes
    @app.before_request
    def check_admin():
        admin_key = request.args.get("admin", "")
        if admin_key and hmac.compare_digest(admin_key, app.config["ADMIN_KEY"]):
            session["admin"] = True
            session.permanent = True
        g.is_admin = session.get("admin", False)

    # Register blueprints
    from web.routes.browse import bp as browse_bp
    from web.routes.puzzle import bp as puzzle_bp
    from web.routes.hints import bp as hints_bp
    from web.routes.admin import bp as admin_bp
    from web.routes.clue import bp as clue_bp
    from web.routes.seo import bp as seo_bp

    app.register_blueprint(browse_bp)
    app.register_blueprint(puzzle_bp)
    app.register_blueprint(hints_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(clue_bp)
    app.register_blueprint(seo_bp)

    return app
