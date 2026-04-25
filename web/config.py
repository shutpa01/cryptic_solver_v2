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
    # Trust one proxy hop (nginx) for X-Forwarded-For when reading the
    # client IP. Set to 0 in environments with no proxy in front.
    PROXY_HOPS = 1


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
