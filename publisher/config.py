"""Configuration for the publisher widget app.

Deliberately self-contained: this package imports nothing from ``web`` so it
can be lifted into its own repository without untangling. See the memory note
``publisher_build_decisions`` — duplication across that boundary is the
decision, not an oversight.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
REF_DB = PROJECT_ROOT / "data" / "cryptic_new.db"
RATE_LIMIT_DB = PROJECT_ROOT / "data" / "rate_limits.db"

load_dotenv(PROJECT_ROOT / ".env", override=False)

DEV_SECRET_KEY = "publisher-dev-secret-change-in-prod"

# Per-publisher keys. A key is public (it travels in the iframe URL, like a
# Maps API key); it identifies, it does not authenticate. What actually stops a
# non-customer framing the widget is `frame_ancestors` below, which is emitted
# as a Content-Security-Policy header on the embed page, plus the Referer /
# Sec-Fetch-Site origin check in publisher/auth.py.
#
# In production this comes from the PUBLISHER_KEYS env var as JSON of the same
# shape. The demo key is local-only and is refused when DEBUG is off.
DEMO_KEYS = {
    "demo": {
        "name": "Cordelia demo",
        "shell": "telegraph",
        # "*" = frameable anywhere. Only ever acceptable for the demo key.
        "frame_ancestors": ["*"],
        "sources": ["telegraph"],
        # Requests per minute, per key per IP, for the corpus endpoints.
        "corpus_per_min": 240,
    },
}


def _load_keys():
    raw = os.environ.get("PUBLISHER_KEYS", "")
    if not raw:
        return dict(DEMO_KEYS)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raise RuntimeError("PUBLISHER_KEYS is set but is not valid JSON")
    if not isinstance(parsed, dict):
        raise RuntimeError("PUBLISHER_KEYS must be a JSON object keyed by API key")
    return parsed


class Config:
    SECRET_KEY = os.environ.get("PUBLISHER_SECRET_KEY") \
        or os.environ.get("SECRET_KEY", DEV_SECRET_KEY)
    CLUES_DB = str(CLUES_DB)
    REF_DB = str(REF_DB)
    RATE_LIMIT_DB = str(RATE_LIMIT_DB)
    PUBLISHER_KEYS = _load_keys()
    RATE_LIMIT_ENABLED = True

    # Life of the signed API token minted into the embed HTML. Short by
    # design: a leaked token must expire before it is worth reusing. The
    # widget renews itself well inside this window (see engine.js), so a
    # solver sitting on one puzzle for three hours never sees an expiry.
    TOKEN_MAX_AGE = 1800          # 30 minutes
    TOKEN_RENEW_GRACE = 300       # a token up to 5 min stale can still renew

    # The match count stops counting here and reports "over" — the design
    # only shows a number at 99 or fewer, so counting past 100 is wasted work.
    MATCH_COUNT_CEILING = 100

    # How many words an entry is ever offered, and therefore the largest number
    # the grid can show. The paper format cannot carry "67 words fit" — a chip
    # you cannot act on is noise, and the board is only scannable if a number
    # on it means "this one has nearly closed". Above the limit the entry shows
    # nothing at all, unless every crossing letter is already in (see
    # MATCH_OPTIONS_WHEN_CROSSED below).
    MATCH_OPTIONS = 9

    # Once every crossing square of an entry has a letter, the grid can tell
    # the solver nothing more and the count would stay hidden forever. At that
    # point we deliberately offer a shortlist of MATCH_OPTIONS anyway, with the
    # answer among them. This is a decision to help, not a count — the number
    # shown is the length of the shortlist, never the true tally.
    MATCH_OPTIONS_WHEN_CROSSED = True


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}
