"""Flask application factory."""

import hmac
import re

from markupsafe import Markup
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

    # Template filters
    @app.template_filter("wordplay_label")
    def wordplay_label_filter(value):
        if not value:
            return ""
        return WORDPLAY_LABELS.get(value, value.replace("_", " ").title())

    @app.template_filter("clickable_words")
    def clickable_words_filter(text, clue_id):
        """Wrap each word in a clue in a clickable span for the helper widget.
        Each span gets a data-idx for multi-word selection support."""
        if not text:
            return ""
        parts = re.split(r'(\s+)', text)
        out = []
        word_idx = 0
        for part in parts:
            if part.strip():
                clean = re.sub(r'[^A-Za-z]', '', part).lower()
                if clean:
                    out.append(
                        '<span class="clue-word cursor-pointer hover:bg-indigo-100 '
                        'hover:rounded px-0.5 -mx-0.5 transition-colors" '
                        'data-idx="%d" data-clean="%s" data-clue="%s" '
                        'onclick="wordHelp(this)">%s</span>'
                        % (word_idx, clean, clue_id, part)
                    )
                    word_idx += 1
                else:
                    out.append(part)
            else:
                out.append(part)
        return Markup("".join(out))

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
    from web.routes.helper import bp as helper_bp
    from web.routes.seo import bp as seo_bp

    app.register_blueprint(browse_bp)
    app.register_blueprint(puzzle_bp)
    app.register_blueprint(hints_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(clue_bp)
    app.register_blueprint(helper_bp)
    app.register_blueprint(seo_bp)

    return app
