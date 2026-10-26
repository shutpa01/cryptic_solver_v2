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


def _normal_priority():
    """Run at NORMAL priority (2026-10-26). The CrypticSolver_FlaskDev scheduled task starts
    this server, and Task Scheduler gives its processes below-normal CPU, low I/O and low
    memory priority. Measured: the same wiring build took 11.3 s at normal priority and 17.0 s
    in this server (27 s while anything else was busy), and Windows kept paging the server out
    between requests. Resets all three. Windows-only, best-effort, never fatal."""
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        h = k32.GetCurrentProcess()
        k32.SetPriorityClass(h, 0x20)                        # NORMAL_PRIORITY_CLASS
        # memory priority: ProcessMemoryPriority (0) = MEMORY_PRIORITY_NORMAL (5)
        mem = ctypes.c_ulong(5)
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                              ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation(h, 0, ctypes.byref(mem), ctypes.sizeof(mem))
        # I/O priority: ProcessIoPriority (33) = IoPriorityNormal (2)
        ntdll = ctypes.WinDLL("ntdll")
        ntdll.NtSetInformationProcess.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                  ctypes.c_void_p, wintypes.ULONG]
        io = ctypes.c_ulong(2)
        ntdll.NtSetInformationProcess(h, 33, ctypes.byref(io), ctypes.sizeof(io))
    except Exception as e:
        print("[run_dev] could not set normal priority: %s" % e)


_normal_priority()

app = create_app("development")

# Per-request timing -> logs/request_timing.log (2026-10-26, Approve-all slowness).
from web.request_timing import RequestTiming
app.wsgi_app = RequestTiming(app.wsgi_app)

def _warm_up():
    """Load the slow one-off things in the background as soon as the server starts, so the
    first click does not pay for them (2026-10-26, measured on the user's own session: first
    puzzle page 21 s = RefDB load; first Approve-all built the solver wiring inside the click).
    The user opens the page BY restarting the server, so before this every session began cold.
    Each loader is locked, so a click that arrives mid-warm waits for that load rather than
    starting a second one. Never fatal."""
    import threading

    # The site starts the word cache on its FIRST request (web/routes/helper.py), where it
    # competed with the two loads below (sandbox: 56 s, and the page render it overlapped
    # slowed 0.3 -> 2.7 s). It only serves the pattern/anagram tools, so it is built LAST,
    # here; marking it as already warming stops the first request starting a second build.
    # A search before it is ready simply builds it on demand (word_cache._get, locked).
    try:
        from web import word_cache
        word_cache._warming = True
    except Exception:
        pass

    def run():
        t0 = time.time()
        steps = []
        # Order = the order the user needs them: puzzle page first, then Approve-all.
        try:
            app.get_shared_ref_db()                     # RefDB: puzzle-page word coverage
            steps.append("RefDB %.1fs" % (time.time() - t0))
        except Exception as e:
            steps.append("RefDB FAILED: %s" % e)
        t1 = time.time()
        try:
            from core import wfw_web                    # solver import + the WFW app
            wfw_web.wiring()                            # the wiring an Approve-all uses
            steps.append("wiring %.1fs" % (time.time() - t1))
        except Exception as e:
            steps.append("wiring FAILED: %s" % e)
        t2 = time.time()
        try:
            from web import word_cache
            word_cache._get()                           # pattern / anagram word list
            steps.append("word cache %.1fs" % (time.time() - t2))
        except Exception as e:
            steps.append("word cache FAILED: %s" % e)
        msg = "warm-up done in %.1fs: %s" % (time.time() - t0, ", ".join(steps))
        print("[run_dev] " + msg, flush=True)
        try:
            from web.request_timing import _write
            from datetime import datetime, timezone
            _write("%s %s" % (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                              "STARTUP " + msg))
        except Exception:
            pass

    threading.Thread(target=run, name="startup-warm", daemon=True).start()


if __name__ == "__main__":
    # Guarantee EXACTLY ONE dev server owns the port: kill any existing listener first, then
    # wait for the OS to release it. Prevents the multi-stale-server trap above.
    _kill_stale_listeners(PORT)
    _wait_port_free(PORT)
    _warm_up()
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
