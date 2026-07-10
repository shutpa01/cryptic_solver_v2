"""Mount the WFW admin solver (core/wfw_web.py) INSIDE the site app.

Live-site plumbing phase 5 (2026-07-10): ONE APP — the WFW clue page and the
hand-solver are served by the site itself under ``/solver/...``, admin-gated by
the site's own session cookie. core/wfw_web.py is NOT modified; its standalone
5099 server keeps working unchanged (the transition fallback — retire it last).

How it works:
- Requests under /solver/* are checked against the SITE session (is_admin);
  anyone else gets a plain 403. Then the path is handed to the wfw Flask app
  with the /solver prefix stripped.
- The wfw app writes root-absolute URLs everywhere (action="/hsresolve",
  fetch('/hscd'), redirect("/hs?...")) — hundreds of literal strings. Rather
  than rewrite that module, the mount rewrites the RESPONSE: every quoted
  root-absolute URL whose first path segment is one of the wfw app's OWN url_map
  segments gets the /solver prefix, and so does a redirect Location header.
  Driven by the url_map, so a new wfw route is picked up automatically.
- The wfw app (and the heavy core solver runtime behind it) is imported lazily
  on the FIRST /solver request, so the site boots fast and non-admin traffic
  never pays for it.
"""

PREFIX = "/solver"


class SolverMount:
    """WSGI wrapper: routes /solver/* to the wfw admin app, everything else on."""

    def __init__(self, flask_app, inner_wsgi):
        self._site = flask_app          # the site app (for session/is_admin)
        self._inner = inner_wsgi        # the site's original wsgi_app
        self._wfw = None                # lazily imported wfw app
        self._patterns = None           # URL-rewrite byte patterns
        self._segments = None           # the wfw app's first path segments

    # -- lazy solver import ---------------------------------------------------

    def _wfw_app(self):
        if self._wfw is None:
            from core.wfw_web import app as wfw_app
            self._wfw = wfw_app
            segs = set()
            for rule in wfw_app.url_map.iter_rules():
                part = rule.rule.lstrip("/").split("/", 1)[0].split("<", 1)[0]
                if part:
                    segs.add(part)
            # each quoted root-absolute URL start: "/hs  '/setstatus  "/worklist ...
            pats = []
            for seg in sorted(segs, key=len, reverse=True):
                for q in ('"', "'"):
                    pats.append(((q + "/" + seg).encode(),
                                 (q + PREFIX + "/" + seg).encode()))
            # the wfw clue page lives at its root: href="/?id=123" and the id
            # navigation form posts to exactly action="/"
            for q in ('"', "'"):
                pats.append(((q + "/?id=").encode(), (q + PREFIX + "/?id=").encode()))
                pats.append(((q + "/" + q).encode(), (q + PREFIX + "/" + q).encode()))
            self._patterns = pats
            self._segments = segs
        return self._wfw

    # -- helpers ---------------------------------------------------------------

    def _is_admin(self, environ):
        """Read the SITE session cookie from this request; True iff admin."""
        try:
            from flask import session
            with self._site.request_context(environ.copy()):
                return bool(session.get("admin"))
        except Exception:
            return False

    def _rewrite_body(self, body):
        for old, new in self._patterns:
            body = body.replace(old, new)
        return body

    def _rewrite_location(self, value):
        if value.startswith("/") and not value.startswith(PREFIX + "/"):
            seg = value.lstrip("/").split("/", 1)[0].split("?", 1)[0]
            if seg in (self._segments or ()) or value.startswith("/?"):
                return PREFIX + value
        return value

    # -- wsgi ------------------------------------------------------------------

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        if path != PREFIX and not path.startswith(PREFIX + "/"):
            return self._inner(environ, start_response)

        if not self._is_admin(environ):
            start_response("403 FORBIDDEN", [("Content-Type", "text/plain")])
            return [b"Admin only."]

        wfw = self._wfw_app()
        env = environ.copy()
        env["SCRIPT_NAME"] = environ.get("SCRIPT_NAME", "") + PREFIX
        env["PATH_INFO"] = path[len(PREFIX):] or "/"

        captured = {}

        def _start(status, headers, exc_info=None):
            captured["status"] = status
            captured["headers"] = headers
            # defer the real start_response until headers are (maybe) rewritten
            return lambda data: None

        chunks = wfw.wsgi_app(env, _start)
        body = b"".join(chunks)
        if hasattr(chunks, "close"):
            chunks.close()

        headers = []
        ctype = ""
        for k, v in captured.get("headers", []):
            if k.lower() == "content-type":
                ctype = v
        is_html = "text/html" in ctype
        if is_html:
            body = self._rewrite_body(body)
        for k, v in captured.get("headers", []):
            if k.lower() == "content-length":
                v = str(len(body))
            elif k.lower() == "location":
                v = self._rewrite_location(v)
            headers.append((k, v))
        start_response(captured.get("status", "200 OK"), headers)
        return [body]
