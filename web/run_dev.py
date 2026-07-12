"""Development server launcher."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `import web` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web import create_app

app = create_app("development")

if __name__ == "__main__":
    # use_reloader=False (2026-07-12): the auto-reloader KILLED the worker
    # mid-request whenever a render's lazy imports touched the watched tree —
    # the WFW clue page died at ~15-25s every time (reads as an endless hang;
    # proven by the same render completing in 13.5s with the reloader off).
    # debug=True keeps tracebacks + template auto-reload; CODE changes now
    # need a manual restart (which was already the house rule).
    app.run(debug=True, port=5000, host="0.0.0.0", threaded=True,
            use_reloader=False)
