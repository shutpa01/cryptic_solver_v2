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
    failures_early = []

    # What each solve RECORDED — the yardstick check 1b measures the rows against.
    con = sqlite3.connect(
        "file:%s?mode=ro" % (Path(__file__).resolve().parent.parent
                             / "data" / "clues_master.db"), uri=True)
    con.row_factory = sqlite3.Row
    ids = tuple(c["id"] for c in clues)
    qmarks = ",".join("?" * len(ids))
    piece_text, link_sound = {}, {}
    for r in con.execute("SELECT clue_id, text FROM wfw_piece "
                         "WHERE clue_id IN (%s)" % qmarks, ids):
        piece_text.setdefault(r["clue_id"], []).append((r["text"] or "").strip())
    for r in con.execute("SELECT DISTINCT clue_id, transform FROM wfw_link "
                         "WHERE clue_id IN (%s) AND transform LIKE 'sounds like%%'"
                         % qmarks, ids):
        tr = r["transform"] or ""
        if '"' in tr:
            link_sound.setdefault(r["clue_id"], []).append(tr.split('"')[1])
    con.close()

    # The same breakdown the overlay renders and the publisher widget consumes,
    # kept beside the page so check 1b can look at ROLES rather than at ink.
    breakdowns = {}
    with app.app_context():
        from web.wfw_read import load_breakdown
        for c in clues:
            try:
                breakdowns[c["id"]] = load_breakdown(c["id"])
            except Exception as exc:                       # noqa: BLE001
                failures_early.append((c["id"], "load_breakdown raised %r" % exc))

    failures = list(failures_early)
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
        # 1b. everything the PARSE recorded reaches a row.
        #
        # Check 1 searches the whole page, and the page prints the clue itself at
        # the top — so a piece with no row at all still satisfies it. That is how
        # this test passed through two real faults on 2026-08-20: the breakdown
        # dropped the deletion rows ("in" -> IN, "good" -> G) and the homophone
        # partner (ROUTES). The stated principle was right; the check could not
        # see the difference.
        #
        # Scope is deliberate. This compares the STORED PARSE against the ROWS,
        # so it fails only when the renderer loses something the solve recorded —
        # the fault this file exists to catch, and the one that keeps recurring
        # because the breakdown is a hand-kept parallel of the admin card. A clue
        # word with no piece at all is an AUTHORING gap, a different problem with
        # its own gates; counted and printed below, never failed here.
        rows = (breakdowns.get(c["id"]) or {}).get("rows", [])
        row_text = " ".join(r.get("detail") or "" for r in rows).lower()
        lost = [p for p in (piece_text.get(c["id"]) or [])
                if p and p.lower() not in row_text]
        if lost:
            failures.append((c["id"], "recorded pieces missing from the rows: %s"
                             % lost))
        # and the sound a homophone claims: recorded on the links as
        # 'sounds like "ROUTES"', and meaningless if the reader never prints it.
        unsaid = [w for w in (link_sound.get(c["id"]) or [])
                  if w.lower() not in row_text]
        if unsaid:
            failures.append((c["id"], "homophone partner never shown: %s" % unsaid))
        # 2. every answer letter present as a tile
        for ch in set((c["answer"] or "").upper()):
            if ch.isalpha() and (">%s</span>" % ch) not in page:
                failures.append((c["id"], "answer tile missing: %s" % ch))
                break
        # 3. read-only: no forms, no admin controls
        if "<form" in low or "/admin/" in low:
            failures.append((c["id"], "admin/form markup present"))

    # Authoring gaps, reported but not failed (see check 1b): a clue word that no
    # piece claims. Printed so the number is visible rather than quietly zero.
    gaps = []
    for c in clues:
        claimed = " ".join(piece_text.get(c["id"]) or []).lower()
        loose = [w for w in re.findall(r"[A-Za-z']+", c["clue_text"] or "")
                 if w.lower() not in claimed]
        if loose:
            gaps.append((c["id"], loose))

    print("checked %d wfw clues; %d failures" % (len(clues), len(failures)))
    print("  (%d clues carry a word no piece claims — authoring, not rendering)"
          % len(gaps))
    for cid, why in failures[:20]:
        print("  FAIL clue %s: %s" % (cid, why))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
