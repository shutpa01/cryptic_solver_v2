"""Development server for the publisher widget.

Port 5003: :5000 belongs to the AI_Solver project, :5001 to this project's site
(web/run_dev.py) and :5002 is the live gunicorn port. The reloader is off for
the same reason it is off there — it kills workers mid-request — so a code
change needs a full restart.
"""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from publisher import create_app

PORT = 5003


def _kill_stale_listeners(port):
    """Kill anything already listening on `port`, so a restart is always clean.

    Same trap as the site's dev server: without this you end up with two
    servers on one port, the browser hits whichever, and every code edit
    appears to do nothing.
    """
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
        if (len(parts) >= 5 and parts[0].upper() == "TCP"
                and parts[-2].upper() == "LISTENING"
                and parts[1].rsplit(":", 1)[-1] == str(port)):
            pid = parts[-1]
            if pid.isdigit() and int(pid) != me:
                pids.add(pid)
    for pid in pids:
        try:
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
            print("[publisher] killed stale listener on :%d (pid %s)" % (port, pid))
        except Exception:
            pass


def _wait_port_free(port, timeout=6.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                time.sleep(0.2)
    print("[publisher] WARNING: port %d still busy — start may fail" % port)
    return False


app = create_app("development")

if __name__ == "__main__":
    _kill_stale_listeners(PORT)
    _wait_port_free(PORT)
    print("[publisher] http://127.0.0.1:%d/" % PORT)
    app.run(debug=True, port=PORT, host="0.0.0.0", threaded=True, use_reloader=False)
