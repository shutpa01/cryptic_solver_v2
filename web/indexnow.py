"""IndexNow — notify Bing (and Yandex, Seznam, ...) the instant a URL goes live.

IndexNow is a tiny protocol: host a public key file at
https://justcordelia.com/<KEY>.txt, then POST the changed URLs. Participating
engines share submissions, so one call reaches all of them. Unlike Google's
request-indexing (which is ignored), Bing acts on IndexNow.

The KEY is NOT a secret — the key file is public by design, and the whole point
is that anyone can fetch it to prove we own the domain — so it lives in the repo.
"""

import urllib.error
import urllib.parse
import urllib.request

KEY = "c4c9e81fbf21629f7835000e0e13dbd9"
HOST = "justcordelia.com"
KEY_LOCATION = "https://%s/%s.txt" % (HOST, KEY)
ENDPOINT = "https://api.indexnow.org/indexnow"     # generic endpoint; shares to all engines
MAX_URLS = 10000                                   # safety cap on a single run


def _submit_one(url, timeout):
    """Announce ONE url via the single-URL GET endpoint (the 'streaming' method Bing asks
    for). Returns (status_code, body_snippet): 200/202 accepted, 4xx rejected, -1 error."""
    qs = urllib.parse.urlencode({"url": url, "key": KEY, "keyLocation": KEY_LOCATION})
    req = urllib.request.Request(ENDPOINT + "?" + qs, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return (resp.status, resp.read().decode("utf-8", "replace")[:500])
    except urllib.error.HTTPError as e:
        return (e.code, e.read().decode("utf-8", "replace")[:500])
    except Exception as e:                          # noqa: BLE001 - must not propagate
        return (-1, str(e))


def submit(urls, timeout=15):
    """Announce changed URLs to IndexNow, ONE URL PER REQUEST (streaming), not a batch
    urlList — Bing recommends streaming to avoid server-overload/indexing-delay warnings.
    Returns (status_code, body_snippet) with the SAME contract as before so callers are
    unchanged: 200 = all accepted, 0 = nothing, otherwise the first failing (status, body).
    NEVER raises into the caller — a notification failure must never break a deploy."""
    urls = [u for u in dict.fromkeys(u for u in urls if u)][:MAX_URLS]   # dedup, keep order
    if not urls:
        return (0, "no urls")
    ok = 0
    for u in urls:
        status, body = _submit_one(u, timeout)
        if status in (200, 202):
            ok += 1
        else:
            # surface the first failure; caller treats non-200/202 as "do not advance"
            return (status, "%d/%d accepted, then %s failed: %s" % (ok, len(urls), u, body))
    return (200, "%d URL(s) accepted (streamed)" % ok)
