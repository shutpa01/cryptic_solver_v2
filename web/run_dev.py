"""Development server launcher."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `import web` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web import create_app

PORT = 5001


def _kill_stale_listeners(port):
    """Kill any process ALREADY listening on `port` before we bind, so a restart is always
    clean. RECURRING BUG (2026-07-30): with the reloader off, restarting by launching a new
    process WITHOUT killing the old left MULTIPLE dev servers alive on 5001 at once; the
    browser/curl then hit whichever, so a stale server kept serving OLD code and every code
    edit "appeared to do nothing". Windows-only (netstat/taskkill); best-effort — never fatal,
    never kills our own PID. Also clears the AI_Solver venv when it squats 5001."""
    if not sys.platform.startswith("win"):
        return
    try:
        out = subprocess.check_output(["netstat", "-ano"], text=True, errors="ignore")
    except Exception:
        return
    me = os.getpid()
    pids = set()
    for line in out.splitlines():
        parts = line.split()
        # e.g.  TCP    0.0.0.0:5001   0.0.0.0:0   LISTENING   12345
        if len(parts) >= 5 and parts[0].upper() == "TCP" \
                and parts[-2].upper() == "LISTENING" \
                and parts[1].rsplit(":", 1)[-1] == str(port):
            pid = parts[-1]
            if pid.isdigit() and int(pid) != me:
                pids.add(pid)
    for pid in pids:
        try:
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
            print("[run_dev] killed stale listener on :%d (pid %s)" % (port, pid))
        except Exception:
            pass


def _wait_port_free(port, timeout=6.0):
    """Block until `port` is actually bindable (taskkill returns before the OS frees it)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                time.sleep(0.2)
    print("[run_dev] WARNING: port %d still busy after %.0fs — start may fail" % (port, timeout))
    return False


app = create_app("development")

if __name__ == "__main__":
    # Guarantee EXACTLY ONE dev server owns the port: kill any existing listener first, then
    # wait for the OS to release it. Prevents the multi-stale-server trap above.
    _kill_stale_listeners(PORT)
    _wait_port_free(PORT)
    # use_reloader=False (2026-07-12): the auto-reloader KILLED the worker
    # mid-request whenever a render's lazy imports touched the watched tree —
    # the WFW clue page died at ~15-25s every time (reads as an endless hang;
    # proven by the same render completing in 13.5s with the reloader off).
    # debug=True keeps tracebacks + template auto-reload; CODE changes now
    # need a manual restart (which was already the house rule).
    # Port 5001 (2026-07-24): a DEDICATED port for this project's dev server, so it can
    # never clash with the AI_Solver project's dev server (which binds 5000). The old
    # collision let a stale AI_Solver / cryptic process on 5000 answer requests, so code
    # edits here appeared to have no effect. Use http://127.0.0.1:5001/ for this app.
    app.run(debug=True, port=PORT, host="0.0.0.0", threaded=True,
            use_reloader=False)
