"""Publisher key -> short-lived signed token. No cookie anywhere.

Why this exists rather than reusing the site's helper guard: `/helper/*`
requires a `?ht=` token AND a `cordelia_session` cookie set `SameSite=Lax`
(`web/routes/helper.py`). Inside a publisher's iframe that cookie is a
third-party cookie, blocked by default in Safari and Chrome, so the site's
guard cannot work here at all. The replacement is a bearer token with no
cookie component.

The chain:

1. The publisher frames `/embed/<source>/<number>?k=<key>`. The key is public
   — it travels in their page source, like a Maps key. It identifies, it does
   not authenticate.
2. The embed page checks the framing origin against the key's allowlist and
   emits `Content-Security-Policy: frame-ancestors <allowlist>`, so a
   non-customer cannot frame the widget even with a copied key.
3. The embed HTML carries a signed token bound to key + source + puzzle, good
   for 30 minutes.
4. Every API call presents that token. It is scoped: a token for puzzle 3740
   cannot query anything else.
5. The widget renews the token in the background, so a long solve never hits
   the expiry.
"""

import time
from functools import wraps
from urllib.parse import urlparse

from flask import current_app, g, jsonify, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

_SALT = "publisher-api"


class AuthError(Exception):
    def __init__(self, message, status=403):
        super().__init__(message)
        self.message = message
        self.status = status


def _serializer():
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"])


def get_key_config(key):
    """Return the config for a publisher key, or None."""
    keys = current_app.config.get("PUBLISHER_KEYS") or {}
    config = keys.get(key)
    if config is None:
        return None
    # The demo key ships with `frame_ancestors: ["*"]`. That is fine on a dev
    # box and is never acceptable in production, so refuse it there outright
    # rather than leave a wildcard-frameable widget in the wild.
    if not current_app.config.get("DEBUG") and key == "demo":
        return None
    return config


def mint_token(key, source, number):
    """Sign a token scoped to one publisher and one puzzle."""
    return _serializer().dumps(
        {"k": key, "s": source, "n": str(number)}, salt=_SALT
    )


def read_token(token, max_age=None):
    """Validate a token. Returns its payload dict.

    Raises AuthError on anything wrong. `max_age` defaults to the configured
    token life; the renew endpoint passes a longer one so a token that has
    just expired can still be exchanged.
    """
    if not token:
        raise AuthError("missing token", 401)
    if max_age is None:
        max_age = current_app.config["TOKEN_MAX_AGE"]
    try:
        payload = _serializer().loads(token, max_age=max_age, salt=_SALT)
    except SignatureExpired:
        raise AuthError("token expired", 401)
    except BadSignature:
        raise AuthError("bad token", 403)
    if get_key_config(payload.get("k")) is None:
        raise AuthError("unknown publisher key", 403)
    return payload


def _presented_token():
    """Read the bearer token from the Authorization header or the query."""
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.args.get("t", "")


def origin_allowed(config, origin):
    """Is `origin` on this key's frame-ancestors allowlist?

    An empty origin means the request carries no Referer/Origin at all. That is
    what a direct browser hit looks like, and it is allowed only when the key
    permits any ancestor — otherwise there is nothing to check against.
    """
    allowed = config.get("frame_ancestors") or []
    if "*" in allowed:
        return True
    if not origin:
        return False
    return origin.rstrip("/") in {a.rstrip("/") for a in allowed}


def framing_origin():
    """Best available origin of the page doing the framing.

    `Origin` is not sent on a normal document GET, so the Referer is the only
    signal available on the embed request itself. This is a check, not a
    security boundary — `frame-ancestors` is the boundary, enforced by the
    browser. It is here so a copied key fails loudly and early rather than
    rendering a widget that then cannot call anything.
    """
    for header in ("Origin", "Referer"):
        value = request.headers.get(header)
        if value:
            parsed = urlparse(value)
            if parsed.scheme and parsed.netloc:
                return f"{parsed.scheme}://{parsed.netloc}"
    return ""


def csp_frame_ancestors(config):
    allowed = config.get("frame_ancestors") or []
    if "*" in allowed:
        return "frame-ancestors *"
    return "frame-ancestors " + " ".join(allowed) if allowed else "frame-ancestors 'none'"


def require_token(fn):
    """Guard an API view: valid scoped token, then a per-key-per-IP limit.

    On success `g.pub` holds the token payload and `g.pub_config` the key's
    config, so the view never re-reads either.
    """

    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            payload = read_token(_presented_token())
        except AuthError as e:
            return jsonify({"error": e.message}), e.status

        config = get_key_config(payload["k"])
        g.pub = payload
        g.pub_config = config

        from publisher.rate_limit import check
        limit = int(config.get("corpus_per_min", 240))
        blocked = check(f"pub:{payload['k']}", limit, 60)
        if blocked is not None:
            return blocked

        return fn(*args, **kwargs)

    return wrapped


def scoped_to(source, number):
    """True when the presented token is for this exact puzzle.

    Stops a token minted for one puzzle being replayed against another —
    which is the difference between a per-solver credential and a general
    corpus key.
    """
    payload = getattr(g, "pub", None)
    if not payload:
        return False
    return payload.get("s") == source and payload.get("n") == str(number)


def token_age_ok_for_renew(token):
    """Validate a token for renewal, accepting one that has just expired."""
    max_age = (current_app.config["TOKEN_MAX_AGE"]
               + current_app.config["TOKEN_RENEW_GRACE"])
    return read_token(token, max_age=max_age)


def now():
    return int(time.time())
