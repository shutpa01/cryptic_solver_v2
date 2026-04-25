"""Session cookie issued on page-load, required by /reveal.

Purpose: defence-in-depth between Cloudflare's edge bot management
and the application's content endpoints. A scraper that has somehow
obtained a /reveal token (e.g. by parsing a single page they fetched)
must now also carry a session cookie that the server set on a
prior page-load response. Without it, /reveal returns 403.

This is NOT a strong access control. The cookie is not bound to a
user identity, the IP, or the specific clue. A determined scraper
that maintains session state across requests (which any browser
automation does for free) can still use it. The point is to fail
the trivial "POST /reveal with a token I scraped from someone
else's page" case, and to give Cloudflare's bot scoring an extra
behavioural signal (real browsers do GET-then-XHR; naked POSTs
without a prior GET stand out).

Trust context: documented as part of the 2026-04-25 security plan
after repeated false assurances about content protection in earlier
sessions. Verification evidence lives alongside the commit.
"""

import time
from typing import Optional

from flask import current_app, request
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

COOKIE_NAME = "cordelia_session"
COOKIE_MAX_AGE = 3600  # 1 hour — matches the existing /reveal token TTL


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(
        current_app.config["SECRET_KEY"],
        salt="cordelia-session",
    )


def issue_session_cookie(response):
    """Set the session cookie on a response.

    Call from page-load handlers (clue, puzzle) so subsequent
    /reveal POSTs from the same browser carry it automatically.
    HttpOnly so JS (and therefore XSS) can't read it; SameSite=Lax so
    cross-origin POSTs (e.g. from a scraper's other domain) don't
    include it; Secure when the request is HTTPS.
    """
    value = _serializer().dumps({"issued": int(time.time())})
    response.set_cookie(
        COOKIE_NAME,
        value,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="Lax",
        secure=request.is_secure,
    )
    return response


def has_valid_session() -> bool:
    """True iff the current request carries a valid, unexpired session cookie."""
    cookie_value: Optional[str] = request.cookies.get(COOKIE_NAME)
    if not cookie_value:
        return False
    try:
        _serializer().loads(cookie_value, max_age=COOKIE_MAX_AGE)
        return True
    except (BadSignature, SignatureExpired):
        return False
