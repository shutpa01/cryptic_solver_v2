"""Flask application factory."""

import hmac
import re

from markupsafe import Markup
from flask import Flask, g, request, session
from werkzeug.middleware.proxy_fix import ProxyFix

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

    # nginx (in production) sets X-Forwarded-For. ProxyFix rewrites
    # request.remote_addr to the real client IP so per-IP rate limits
    # (web/rate_limit.py) and helper.py's existing IP throttle work.
    # If the deployment has no proxy in front, set PROXY_HOPS = 0.
    proxy_hops = app.config.get("PROXY_HOPS", 1)
    if proxy_hops:
        app.wsgi_app = ProxyFix(
            app.wsgi_app,
            x_for=proxy_hops,
            x_proto=proxy_hops,
            x_host=proxy_hops,
        )

    # Database teardown
    db.init_app(app)

    # Template filters
    @app.template_filter("wordplay_label")
    def wordplay_label_filter(value):
        if not value:
            return ""
        return WORDPLAY_LABELS.get(value, value.replace("_", " ").title())

    @app.template_filter("source_name")
    def source_name_filter(value):
        """Display name for publication sources (handles compound names)."""
        names = {"dailymail": "Daily Mail", "telegraph-toughie": "Telegraph Toughie"}
        return names.get(value, (value or "").title())

    # Lazy-loaded RefDB for word coverage checks (admin only)
    _word_coverage_db = [None]  # mutable container for closure

    def _get_word_coverage_db():
        if _word_coverage_db[0] is None:
            from signature_solver.db import RefDB
            _word_coverage_db[0] = RefDB()
        return _word_coverage_db[0]

    def invalidate_word_coverage_db():
        """Force RefDB reload on next page load (after DB+ enrichment)."""
        _word_coverage_db[0] = None

    def patch_word_coverage_db(etype, word, value):
        """Patch the in-memory RefDB with a single new entry (instant, no reload)."""
        ref_db = _word_coverage_db[0]
        if ref_db is None:
            return  # not loaded yet, will pick it up on next load
        w = word.lower().strip()
        v = value.upper().strip()
        if etype == "synonym":
            if w not in ref_db.synonyms:
                ref_db.synonyms[w] = []
            if v not in ref_db.synonyms[w]:
                ref_db.synonyms[w].append(v)
        elif etype == "abbreviation":
            if w not in ref_db.abbreviations:
                ref_db.abbreviations[w] = []
            if v not in ref_db.abbreviations[w]:
                ref_db.abbreviations[w].append(v)
        elif etype == "definition":
            # Definitions are stored in synonyms dict (merged during load)
            if w not in ref_db.synonyms:
                ref_db.synonyms[w] = []
            if v not in ref_db.synonyms[w]:
                ref_db.synonyms[w].append(v)
        elif etype == "indicator":
            if w not in ref_db.indicators:
                ref_db.indicators[w] = []
            ref_db.indicators[w].append((v.lower(), None, 'high'))

    _dd_graph_cache = [None]

    def get_shared_ref_db():
        """Get the shared RefDB instance (loads once, reused everywhere)."""
        return _get_word_coverage_db()

    def get_shared_dd_graph():
        """Get the shared DD graph (loads once, reused everywhere)."""
        if _dd_graph_cache[0] is None:
            from backfill_ai_exp.backfill_dd_hidden import build_graph
            _dd_graph_cache[0] = build_graph(_get_word_coverage_db())
        return _dd_graph_cache[0]

    # Expose on the app so routes can access it
    app.invalidate_word_coverage_db = invalidate_word_coverage_db
    app.patch_word_coverage_db = patch_word_coverage_db
    app.get_shared_ref_db = get_shared_ref_db
    app.get_shared_dd_graph = get_shared_dd_graph

    def _word_in_db(word_clean, ref_db):
        """Check if a word has any entry in the reference DB."""
        if len(word_clean) < 2:
            return True  # single letters are always fine
        if ref_db.is_link_word(word_clean):
            return True
        if ref_db.get_abbreviations(word_clean):
            return True
        if ref_db.get_synonyms(word_clean, max_len=15):
            return True
        if ref_db.get_indicator_types(word_clean):
            return True
        return False

    def _def_pair_in_db(definition, answer, ref_db):
        """Check if a definition→answer pair exists in the reference DB."""
        if not definition or not answer:
            return True  # nothing to check
        defn = definition.strip().lower()
        ans = answer.strip().upper()
        if not defn or not ans:
            return True
        # Check definition_answers_augmented
        if ref_db.is_definition_of(defn, ans):
            return True
        # Check synonyms_pairs (both directions)
        syns = ref_db.get_synonyms(defn, max_len=len(ans))
        if ans in [s.upper().replace(" ", "").replace("-", "") for s in syns]:
            return True
        return False

    @app.template_filter("def_missing")
    def def_missing_filter(clue):
        """Return True if the clue has a definition but it's not paired with the answer in the DB."""
        try:
            if not getattr(g, 'is_admin', False):
                return False
            definition = clue.get("definition") if hasattr(clue, 'get') else getattr(clue, 'definition', None)
            answer = clue.get("answer") if hasattr(clue, 'get') else getattr(clue, 'answer', None)
            if not definition or not answer:
                return False
            ref_db = _get_word_coverage_db()
            return not _def_pair_in_db(definition, answer, ref_db)
        except Exception:
            return False

    @app.template_filter("format_answer")
    def format_answer_filter(answer, enumeration):
        """Format answer with spaces based on enumeration.

        E.g. 'AGREATDEAL' with enum '1,5,4' -> 'A GREAT DEAL'
        Handles hyphens too: '4-6' -> 'HALF-NELSON'
        """
        if not answer or not enumeration:
            return answer or ''
        # Already has spaces — return as-is
        if ' ' in answer or '-' in answer:
            return answer
        # Parse enumeration: split on commas, hyphens become joiners
        import re
        parts = re.split(r'([,\-])', enumeration.strip())
        pos = 0
        result = []
        clean = re.sub(r'[^A-Za-z]', '', answer).upper()
        for part in parts:
            part = part.strip()
            if part == ',':
                result.append(' ')
            elif part == '-':
                result.append('-')
            elif part.isdigit():
                n = int(part)
                result.append(clean[pos:pos + n])
                pos += n
        formatted = ''.join(result)
        return formatted if formatted else answer

    @app.template_filter("clickable_words")
    def clickable_words_filter(text, clue_id):
        """Wrap each word in a clue in a clickable span for the helper widget.
        Each span gets a data-idx for multi-word selection support.
        In admin mode, words not found in the reference DB are underlined red."""
        if not text:
            return ""

        is_admin = getattr(g, 'is_admin', False)
        ref_db = _get_word_coverage_db() if is_admin else None

        parts = re.split(r'(\s+)', text)
        out = []
        word_idx = 0
        for part in parts:
            if part.strip():
                clean = re.sub(r'[^A-Za-z]', '', part).lower()
                if clean:
                    # Check DB coverage for admin
                    missing_class = ""
                    if ref_db and not _word_in_db(clean, ref_db):
                        missing_class = " underline decoration-red-400 decoration-2 underline-offset-2"

                    out.append(
                        '<span class="clue-word cursor-pointer hover:bg-indigo-100 '
                        'hover:rounded px-0.5 -mx-0.5 transition-colors%s" '
                        'data-idx="%d" data-clean="%s" data-clue="%s" '
                        'onclick="wordHelp(this)">%s</span>'
                        % (missing_class, word_idx, clean, clue_id, part)
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

    @app.context_processor
    def inject_helper_token():
        """Make helper token available in all templates."""
        from web.routes.helper import generate_helper_token
        return {"helper_token": generate_helper_token()}

    @app.after_request
    def add_no_cache(response):
        """Prevent browser caching of HTML pages and JS files."""
        if response.content_type and ('text/html' in response.content_type or 'javascript' in response.content_type):
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
        return response

    # Register blueprints
    from web.routes.browse import bp as browse_bp
    from web.routes.puzzle import bp as puzzle_bp
    from web.routes.hints import bp as hints_bp
    from web.routes.admin import bp as admin_bp
    from web.routes.clue import bp as clue_bp
    from web.routes.helper import bp as helper_bp
    from web.routes.seo import bp as seo_bp
    from web.routes.learn import bp as learn_bp
    from web.routes.tools import bp as tools_bp

    app.register_blueprint(browse_bp)
    app.register_blueprint(puzzle_bp)
    app.register_blueprint(hints_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(clue_bp)
    app.register_blueprint(helper_bp)
    app.register_blueprint(seo_bp)
    app.register_blueprint(learn_bp)
    app.register_blueprint(tools_bp)

    return app
