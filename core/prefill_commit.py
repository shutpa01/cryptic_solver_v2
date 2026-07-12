"""File a nightly-prefill reading as a PENDING committed parse (never pass).

The publish-first review used to mean walking every prefilled clue in /hs
(open, check the seeded grid, Commit) — page swapping and scrolling. User
design 2026-07-12 (memory: prefill-pending-commits): the prefill COMMITS its
validated reading, but ONLY EVER as status='pending' with solved_by='prefill',
so the whole puzzle is reviewable from the clue page in one scroll. Confirm
there (/prefillconfirm) = the user's one click: re-validate, promote to a
FROZEN manual pass, and harvest the reusable pieces to the reference DB.

Safety by construction, like pending-only signatures:
- a prefill filing can NEVER be a pass — only the user's Confirm can;
- a pending parse is never served to the public (the site reads pass only);
- NO freeze and NO reference-DB harvest here — both belong to the Confirm;
- never overwrites user state (existing hand-solver assignments or a frozen
  manual solve) and never re-runs anything.

Used by the nightly prefill (scripts/prompts/nightly_prefill.md). Building +
validation are core.wfw_web._build_manual_parse — the SAME gate the user's
own /hs commit goes through (tile/word coverage, fodder rules, selection
derivation), so an invalid reading cannot be filed at all.

CLI (one clue per call, assignments as a JSON file or literal):
    python -m core.prefill_commit <clue_id> <assignments.json | JSON-string>
"""

import json
import sys

from core import store


def file_pending_prefill(clue_id, assignments):
    """File `assignments` (the /hs payload list) as a pending prefill reading.
    Returns {"ok": bool, "msg": str}. On ok: wfw_hs_assignments seeded AND a
    pending manual parse stored (solved_by='prefill'). Never overwrites."""
    conn = store.connect()
    try:
        if store.get_hs_assignments(conn, clue_id):
            return {"ok": False, "msg": "clue %d already has hand-solver state "
                    "— not overwritten" % clue_id}
        if store.is_frozen(conn, clue_id):
            return {"ok": False, "msg": "clue %d is a frozen manual solve "
                    "— not touched" % clue_id}
    finally:
        conn.close()

    from core.wfw_web import _build_manual_parse
    built = _build_manual_parse(clue_id, assignments)
    if not built["ok"]:
        return built                     # the validation gate's own message

    parse = built["parse"]
    parse.status = "pending"             # NEVER pass: the user is the only path
    parse.solved_by = "prefill"          # the clue page badges + gates on this
    parse.warnings = list(parse.warnings or []) + [
        "prefill reading — awaiting your review (Confirm on the clue page)"]

    conn = store.connect()
    try:
        store.set_hs_assignments(conn, clue_id, json.dumps(assignments))
        store.save_parse(conn, clue_id, parse, built["ctx"])
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "msg": "filed PENDING prefill for clue %d (%d piece%s)"
            % (clue_id, built["n_sources"], "" if built["n_sources"] == 1 else "s")}


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    clue_id = int(argv[0])
    raw = argv[1]
    try:
        with open(raw, encoding="utf-8") as f:
            assignments = json.load(f)
    except OSError:
        assignments = json.loads(raw)
    out = file_pending_prefill(clue_id, assignments)
    print(out["msg"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
