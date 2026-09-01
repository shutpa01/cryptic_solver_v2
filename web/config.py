"""Flask configuration."""

import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"

# Load secrets from the project .env into the process environment before the
# config classes below read them. override=False so anything the launcher
# (systemd / gunicorn) already set in the real environment wins over the file.
load_dotenv(PROJECT_ROOT / ".env", override=False)

PUZZLES_PER_PAGE = 30

# Placeholder signing key. Safe ONLY for local development (HTTP on localhost).
# ProductionConfig refuses to boot if the live SECRET_KEY still equals this.
DEV_SECRET_KEY = "dev-secret-change-in-prod"


class Config:
    # Signing key for every cookie/token the app issues: the admin session,
    # the helper `ht` token, the cordelia_session cookie, the /reveal token
    # and the puzzle-match token. MUST be secret in production — a known key
    # lets anyone forge an admin session and bypass every gate. Read from the
    # SECRET_KEY env var / .env; the dev placeholder is used only locally.
    SECRET_KEY = os.environ.get("SECRET_KEY", DEV_SECRET_KEY)
    ADMIN_KEY = os.environ.get("ADMIN_KEY", "dev-admin-key")
    PERMANENT_SESSION_LIFETIME = timedelta(days=30)
    SESSION_COOKIE_SAMESITE = "Lax"
    CLUES_DB = str(CLUES_DB)
    PUZZLES_PER_PAGE = PUZZLES_PER_PAGE
    # Kill switch for per-IP rate limits (web/rate_limit.py).
    # Set False in an emergency to disable without a code change.
    RATE_LIMIT_ENABLED = True
    # ANALYTICS. The Google Analytics and Microsoft Clarity tags are emitted
    # only when this is True, so an excluded visit sends NOTHING — no beacon
    # leaves the browser at all, rather than being filtered after arrival.
    #
    # OWN TRAFFIC. Our own use is not audience behaviour, and a handful of
    # sessions a day from one address distorts a small site's numbers badly.
    # Comma-separated; the env var REPLACES this list, it does not add to it.
    # The address is the one ProxyFix resolves from X-Forwarded-For, i.e. the
    # real client behind Cloudflare and nginx (PROXY_HOPS below).
    ANALYTICS_EXCLUDED_IPS = [
        ip.strip() for ip in os.environ.get(
            "ANALYTICS_EXCLUDED_IPS", "86.149.89.193").split(",") if ip.strip()
    ]
    # Master switch, off everywhere except production — so the dev servers on
    # :5001 and the /solver mount never register a hit at all.
    ANALYTICS_ENABLED = False
    # Trust proxy hops for X-Forwarded-For when reading the client IP.
    # Production chain (2026-04-25 onwards): Cloudflare → nginx → Flask = 2 hops.
    # Each proxy appends one IP to X-Forwarded-For; ProxyFix reads back the
    # configured number of trusted hops to find the real client IP.
    # Set to 0 only in environments with no proxy in front.
    PROXY_HOPS = 2
    # Base URL of the WFW admin solver. Phase 5 mounts it INSIDE this app at
    # /solver (web/solver_mount.py), so the default is same-origin. Set the
    # env var to fall back to the standalone 5099 server if the mount
    # misbehaves (e.g. WFW_ADMIN_BASE=http://127.0.0.1:5099).
    WFW_ADMIN_BASE = os.environ.get("WFW_ADMIN_BASE", "/solver")
    # When True, the FAQPage JSON-LD on /clue/* pages omits the curated
    # definition word and wordplay type, leaving only the answer + a
    # "see page for explanation" teaser. Off by default; flip to True to
    # cut off competitors who are scraping our parses via the structured
    # data leak. The visible page rendering is unaffected (definition and
    # wordplay still come back through /reveal hint chain). See also
    # web/routes/clue_seo.py:generate_faq_schema.
    STRIP_DEFINITION_FROM_JSONLD = False


class DevelopmentConfig(Config):
    DEBUG = True
    # Inherited ANALYTICS_ENABLED = False. The dev server must never appear in
    # the reports: it is the same pages, on the same machine, all day.


class ProductionConfig(Config):
    DEBUG = False
    ANALYTICS_ENABLED = True


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
