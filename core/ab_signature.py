"""Automatic BEFORE/AFTER A/B for a newly-filed PASS-tier signature.

A pass-tier signature can go GREEN and runs in the main cascade, so it can affect OTHER clues.
Before we trust it, we solve the corpus work-list TWICE with the same fresh code path — once
before the signature is filed (baseline) and once after — and diff the two:

  * REGRESSION — a clue that passed in the baseline no longer passes after. Adding a signature
                 is additive, so this should not happen; if it does, do NOT trust it.
  * NEW PASS   — a clue that did not pass in the baseline now passes. These are what the
                 signature unlocked; every one must be inspected for faithfulness (a clean-looking
                 fabrication is the one unforgivable outcome). The triggering clue is expected;
                 passes on OTHER clues are the flag.

Two FRESH solves (not the stored verdicts) are the point: a fresh non-persist solve does not
reproduce every stored pass (acrostic/alternation/AI drift), so diffing against the stored
baseline invents false regressions. Diffing two identical fresh passes cancels that drift — with
no signature change the diff is empty by construction.

The check never files or persists (auto-signature off, clue_id=None) — a pure measurement. The
full corpus is ~10-15 min PER pass; callers run it in the background. pass_limit/fail_limit bound
it for tests.

CLI:  python -m core.ab_signature [pass_limit] [fail_limit]   # self-test: no change -> 0/0
"""

import os
import sqlite3

_CLUES_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "clues_master.db")


def _worklist_rows(pass_limit=None, fail_limit=500):
    con = sqlite3.connect(_CLUES_DB)
    try:
        q = ("SELECT c.id, c.clue_text, c.answer, c.direction "
             "FROM clues c JOIN wfw_solve s ON s.clue_id = c.id WHERE s.status = ? "
             "ORDER BY c.id")
        passes = con.execute(q + ((" LIMIT %d" % pass_limit) if pass_limit else ""),
                             ("pass",)).fetchall()
        fails = con.execute(q + ((" LIMIT %d" % fail_limit) if fail_limit else ""),
                            ("fail",)).fetchall()
    finally:
        con.close()
    return passes, fails


def solve_worklist(rows, progress=None, phase=""):
    """Solve every (id, clue_text, answer, direction) row with a FRESH db-only wiring (auto-
    signature OFF, no persist). Returns {clue_id: (status, solved_by)}."""
    from core.engine_registry import make_db_wiring, db_only, solve_clue_text
    wiring = db_only(make_db_wiring())
    wiring["auto_signature"] = False
    wiring["auto_signature_queue"] = False
    out = {}
    for i, (cid, ct, ans, direction) in enumerate(rows, 1):
        _ctx, parse, name = solve_clue_text(ct, ans, wiring, clue_id=None, direction=direction)
        out[cid] = (parse.status if parse is not None else "fail",
                    name if parse is not None else "")
        if progress and i % 100 == 0:
            progress(phase, i, len(rows))
    return out


def diff(baseline, after, meta):
    """Regressions (was pass, now not) + new passes (was not pass, now pass) between two
    fresh solves. `meta` maps clue_id -> (clue_text, answer) for readable output."""
    regressions, new_passes = [], []
    for cid, (bst, bname) in baseline.items():
        ast = after.get(cid, ("fail", ""))[0]
        if bst == "pass" and ast != "pass":
            regressions.append((cid, meta[cid][0], meta[cid][1], "was pass/%s, now %s"
                                % (bname, ast)))
    for cid, (ast, aname) in after.items():
        bst = baseline.get(cid, ("fail", ""))[0]
        if bst != "pass" and ast == "pass":
            new_passes.append((cid, meta[cid][0], meta[cid][1], "now pass via %s" % aname))
    return {"regressions": regressions, "new_passes": new_passes}


def run_ab(file_fn=None, pass_limit=None, fail_limit=500, progress=None):
    """BEFORE/AFTER A/B around `file_fn` (which files the new signature into the catalog).
    With file_fn=None it is a self-test: baseline and after are identical solves, so the diff
    is empty. Returns {regressions, new_passes, n_pass, n_fail}."""
    passes, fails = _worklist_rows(pass_limit, fail_limit)
    rows = passes + fails
    meta = {r[0]: (r[1], r[2]) for r in rows}
    baseline = solve_worklist(rows, progress=progress, phase="baseline")
    if file_fn is not None:
        file_fn()
    after = solve_worklist(rows, progress=progress, phase="after")
    d = diff(baseline, after, meta)
    d["n_pass"], d["n_fail"] = len(passes), len(fails)
    return d


def verdict(ab, exclude_clue_ids=()):
    """promote / review / hold decision from a run_ab result. `exclude_clue_ids` are the
    clue(s) that triggered the signature — their own new pass is expected, not a flag."""
    excl = set(exclude_clue_ids or ())
    reg = ab["regressions"]
    nps = [n for n in ab["new_passes"] if n[0] not in excl]
    if reg:
        return ("hold", "%d REGRESSION(s) — do not trust: %s"
                % (len(reg), "; ".join("%s (%s)" % (r[0], r[3]) for r in reg[:5])))
    if nps:
        return ("review", "%d other clue(s) now pass — inspect for fabrication: %s"
                % (len(nps), "; ".join("%s %s" % (n[0], n[2]) for n in nps[:8])))
    return ("promote", "clean: 0 regressions, no stray new passes")


def _set_tier(template_id, tier):
    con = sqlite3.connect(_CLUES_DB)
    try:
        con.execute("UPDATE catalog_templates SET tier = ? WHERE id = ?", (tier, template_id))
        con.commit()
    finally:
        con.close()


def try_promote(template_id, trigger_clue_ids=(), pass_limit=None, fail_limit=500,
                progress=None):
    """Run the BEFORE/AFTER A/B around promoting a pending signature to PASS-tier, then decide.
    Baseline = the signature still pending (harmless in the final stage); after = flipped to
    pass (now live in the main cascade). Keep tier='pass' ONLY if the diff is clean; otherwise
    revert to 'pending' — so a signature can never go green unless the check confirms it does
    not regress or fabricate on OTHER clues. Returns (action, summary, ab). Does NOT re-solve
    the trigger clue or reload any wiring — the caller does that on 'promote'."""
    ab = run_ab(file_fn=lambda: _set_tier(template_id, "pass"),
                pass_limit=pass_limit, fail_limit=fail_limit, progress=progress)
    action, summary = verdict(ab, exclude_clue_ids=trigger_clue_ids)
    if action != "promote":
        _set_tier(template_id, "pending")     # not clean -> keep it safe (amber only)
    return action, summary, ab


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "promote":
        tid = int(sys.argv[2])
        clue_ids = [int(x) for x in sys.argv[3:]] if len(sys.argv) > 3 else []
        print("A/B-gated promotion of template %d (trigger clues %s)..." % (tid, clue_ids))
        action, summary, ab = try_promote(tid, trigger_clue_ids=clue_ids,
                                          progress=lambda ph, d, t: print("  [%s] %d/%d"
                                                                          % (ph, d, t)))
        print("checked %d pass + %d fail" % (ab["n_pass"], ab["n_fail"]))
        print("ACTION:", action, "-", summary)
        if action == "promote":
            print("template %d is now PASS-tier. Re-solve its clue(s) to turn them green "
                  "(reload the server or re-run the clue)." % tid)
        else:
            print("template %d stays PENDING-only (safe)." % tid)
        return
    pl = int(sys.argv[1]) if len(sys.argv) > 1 else None
    fl = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    print("A/B self-test (no signature change -> expect 0 regressions, 0 new passes)")
    ab = run_ab(file_fn=None, pass_limit=pl, fail_limit=fl,
                progress=lambda ph, d, t: print("  [%s] %d/%d" % (ph, d, t)))
    print("checked %d pass + %d fail" % (ab["n_pass"], ab["n_fail"]))
    print("regressions:", len(ab["regressions"]), "| new passes:", len(ab["new_passes"]))
    for r in ab["regressions"][:10]:
        print("   REGRESSION", r[0], r[1][:40], "|", r[3])
    print("verdict:", verdict(ab))


if __name__ == "__main__":
    main()
