"""IndexNow — notify Bing (and Yandex, Seznam, ...) the instant a URL goes live.

IndexNow is a tiny protocol: host a public key file at
https://justcordelia.com/<KEY>.txt, then POST the changed URLs. Participating
engines share submissions, so one call reaches all of them. Unlike Google's
request-indexing (which is ignored), Bing acts on IndexNow.

The KEY is NOT a secret — the key file is public by design, and the whole point
is that anyone can fetch it to prove we own the domain — so it lives in the repo.
"""

import json
import urllib.error
import urllib.request

KEY = "c4c9e81fbf21629f7835000e0e13dbd9"
HOST = "justcordelia.com"
KEY_LOCATION = "https://%s/%s.txt" % (HOST, KEY)
ENDPOINT = "https://api.indexnow.org/indexnow"     # generic endpoint; shares to all engines
MAX_URLS = 10000                                   # IndexNow's per-request limit


def submit(urls, timeout=15):
    """POST changed URLs to IndexNow. Returns (status_code, body_snippet):
      200/202 = accepted, 4xx = rejected (see body), -1 = network/other error, 0 = nothing.
    NEVER raises into the caller — a notification failure must never break a deploy or a
    commit. Only absolute https://justcordelia.com/... URLs should be passed."""
    urls = [u for u in dict.fromkeys(u for u in urls if u)][:MAX_URLS]   # dedup, keep order
    if not urls:
        return (0, "no urls")
    payload = json.dumps({
        "host": HOST,
        "key": KEY,
        "keyLocation": KEY_LOCATION,
        "urlList": urls,
    }).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT, data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return (resp.status, resp.read().decode("utf-8", "replace")[:500])
    except urllib.error.HTTPError as e:
        return (e.code, e.read().decode("utf-8", "replace")[:500])
    except Exception as e:                          # noqa: BLE001 - must not propagate
        return (-1, str(e))
