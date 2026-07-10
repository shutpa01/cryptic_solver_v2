"""Page-shape contract test for the WFW full-explanation overlay (phase 2).

The PRINCIPLE carried over from the pre-reset contract test (git master
671c5f6e:web/test_clue_wfw_render_contract.py): the rendered breakdown must show
EVERY clue word, and every answer letter, with no admin controls — the page a
site user sees is complete and read-only.

Run directly (read-only against data/clues_master.db):
    .venv\\Scripts\\python.exe web\\test_wfw_overlay_contract.py
"""
import html as html_mod
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SKELETON = [("telegraph", "31285"), ("telegraph", "31286")]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from web import create_app
    from web.routes.hints import generate_token

    app = create_app("development")
    app.config["RATE_LIMIT_ENABLED"] = False   # the test IS one client hitting 60+ clues
    client = app.test_client()
    # a real page-fetch issues the session cookie /wfwfull requires
    client.get("/telegraph/cryptic/31285")

    con = sqlite3.connect(
        "file:%s?mode=ro" % (Path(__file__).resolve().parent.parent
                             / "data" / "clues_master.db"), uri=True)
    con.row_factory = sqlite3.Row
    clues = []
    for src, pnum in SKELETON:
        clues += con.execute(
            "SELECT c.id, c.clue_text, c.answer FROM clues c "
            "JOIN wfw_solve w ON w.clue_id = c.id AND w.status = 'pass' "
            "WHERE c.source = ? AND c.puzzle_number = ?", (src, pnum)).fetchall()
    con.close()
    assert clues, "no wfw-pass clues found for the skeleton puzzles"

    failures = []
    for c in clues:
        with app.test_request_context("/"):
            token = generate_token(c["id"])
        r = client.post("/wfwfull", data={"token": token})
        page = html_mod.unescape(r.data.decode("utf-8", "replace"))
        low = page.lower()
        if r.status_code != 200:
            failures.append((c["id"], "HTTP %d" % r.status_code))
            continue
        # 1. every clue word visible
        words = re.findall(r"[A-Za-z']+", c["clue_text"] or "")
        missing = [w for w in words if w.lower() not in low]
        if missing:
            failures.append((c["id"], "clue words missing: %s" % missing))
        # 2. every answer letter present as a tile
        for ch in set((c["answer"] or "").upper()):
            if ch.isalpha() and (">%s</span>" % ch) not in page:
                failures.append((c["id"], "answer tile missing: %s" % ch))
                break
        # 3. read-only: no forms, no admin controls
        if "<form" in low or "/admin/" in low:
            failures.append((c["id"], "admin/form markup present"))

    print("checked %d wfw clues; %d failures" % (len(clues), len(failures)))
    for cid, why in failures[:20]:
        print("  FAIL clue %s: %s" % (cid, why))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
