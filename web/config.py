"""Flask configuration."""

import os
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"

PUZZLES_PER_PAGE = 30


class Config:
    SECRET_KEY = "dev-secret-change-in-prod"
    ADMIN_KEY = os.environ.get("ADMIN_KEY", "dev-admin-key")
    PERMANENT_SESSION_LIFETIME = timedelta(days=30)
    SESSION_COOKIE_SAMESITE = "Lax"
    CLUES_DB = str(CLUES_DB)
    PUZZLES_PER_PAGE = PUZZLES_PER_PAGE
    # Kill switch for per-IP rate limits (web/rate_limit.py).
    # Set False in an emergency to disable without a code change.
    RATE_LIMIT_ENABLED = True
    # Trust proxy hops for X-Forwarded-For when reading the client IP.
    # Production chain (2026-04-25 onwards): Cloudflare → nginx → Flask = 2 hops.
    # Each proxy appends one IP to X-Forwarded-For; ProxyFix reads back the
    # configured number of trusted hops to find the real client IP.
    # Set to 0 only in environments with no proxy in front.
    PROXY_HOPS = 2
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


class ProductionConfig(Config):
    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
