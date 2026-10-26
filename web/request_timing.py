"""Per-request timing for the dev server (user, 2026-10-26: Approve-all can take a
minute on the phone and the laptop alike; measure it instead of guessing).

Installed by web/run_dev.py as a WSGI wrapper. One line per request goes to
logs/request_timing.log: start time, total seconds, status, response bytes, method,
path, and — for /solver requests — the seconds spent inside the steps measured in the
sandbox (wiring build, reference-DB adds, Confirm, page re-render). The step timers wrap
module attributes at runtime; no engine or wfw_web code is edited.
"""

import functools
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "logs" / "request_timing.log"

_local = threading.local()
_write_lock = threading.Lock()
_patched = False

# (module, attribute) pairs timed inside a request. Looked up when core.wfw_web is
# first imported (lazily, by the /solver mount).
_STEPS = [
    ("core.engine_registry", "make_db_wiring"),
    ("core.wfw_web", "_do_add"),
    ("core.wfw_web", "apply_add_to_wiring"),
    ("core.wfw_web", "_apply_db_adds"),
    ("core.wfw_web", "_refresh_provisional_flags"),
    ("core.wfw_web", "_confirm_prefill"),
    ("core.wfw_web", "_body"),
    ("core.admin_db", "delete_pending"),
]


def _timed(name, fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        steps = getattr(_local, "steps", None)
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            if steps is not None:
                cnt, tot = steps.get(name, (0, 0.0))
                steps[name] = (cnt + 1, tot + time.perf_counter() - t)
    wrapper._request_timing = True
    return wrapper


def _patch_steps():
    """Wrap the step functions once core.wfw_web has been imported."""
    global _patched
    import sys
    if _patched or "core.wfw_web" not in sys.modules:
        return
    import importlib
    for modname, attr in _STEPS:
        try:
            mod = importlib.import_module(modname)
            fn = getattr(mod, attr)
            if not getattr(fn, "_request_timing", False):
                setattr(mod, attr, _timed(attr, fn))
        except Exception:
            pass
    _patched = True


def _write(line):
    try:
        LOG.parent.mkdir(exist_ok=True)
        with _write_lock, open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


class RequestTiming:
    """WSGI wrapper: times each request end to end (body fully produced)."""

    def __init__(self, inner):
        self._inner = inner

    def __call__(self, environ, start_response):
        _patch_steps()
        _local.steps = {}
        started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        t = time.perf_counter()
        status = {}

        def _start(st, headers, exc_info=None):
            status["s"] = st.split(" ", 1)[0]
            return start_response(st, headers, exc_info)

        size = 0
        try:
            result = self._inner(environ, _start)
            chunks = []
            for c in result:
                size += len(c)
                chunks.append(c)
            if hasattr(result, "close"):
                result.close()
            return chunks
        finally:
            total = time.perf_counter() - t
            _patch_steps()
            steps = getattr(_local, "steps", {}) or {}
            _local.steps = None
            path = environ.get("PATH_INFO", "")
            if not (path.startswith("/static/") or path == "/favicon.ico"):
                step_txt = " ".join(
                    "%s=%.2f%s" % (k, v[1], "x%d" % v[0] if v[0] > 1 else "")
                    for k, v in sorted(steps.items(), key=lambda kv: -kv[1][1])
                    if v[1] >= 0.01)
                _write("%s %6.2fs %s %7dB %s %s%s" % (
                    started, total, status.get("s", "?"), size,
                    environ.get("REQUEST_METHOD", ""), path,
                    ("  [" + step_txt + "]") if step_txt else ""))
