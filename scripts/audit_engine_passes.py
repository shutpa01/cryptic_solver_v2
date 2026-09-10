"""ENGINE PASSES, laid out for a reader — read-only, judges nothing.

Trial tool (2026-09-10). The STET false pass that morning passed every mechanical
test there is: every clue word accounted for, both indicator types used, the
container geometry sound. What it could not survive was a person reading one line
of it — "is -> PET". No gate can make that judgement; a reader can make it in a
second.

So this prints the CLAIMS an engine pass rests on, in the fewest words that still
let someone say yes or no:

    times 29644 1a   STET   [container_deletion]
      "Don't change way in which core is removed"
      def   "Don't change"
      say   way -> ST          [abbreviation]
      say   is -> PET          [synonym]          <- the line that should die
      ind   "in which"  container
      ind   "core"      deletion
      ind   "removed"   deletion

WHY ONLY ENGINE PASSES. A `manual` solve is the user's own filing and a `prefill`
is already pending — neither is this tool's business. 504 of last week's 695
passes were manual; the engine population is ~27 a night, which is small enough
to read every one.

WHAT IT DOES NOT DO. It writes nothing, judges nothing and touches no verdict.
Demotion, if it is ever wired up, belongs behind a measured false-demotion rate
and moves pass -> pending ONLY — never fail -> pass. That asymmetry is the whole
safety case: its worst failure is extra review, never a false claim on the site.

    python scripts/audit_engine_passes.py --days 7
    python scripts/audit_engine_passes.py --days 1 --out logs/audit.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web import create_app                                        # noqa: E402
from web import wfw_read                                          # noqa: E402

# Solvers whose output is not an engine's claim: the user's own, and prefills that
# are pending by construction.
NOT_ENGINE = ("manual", "prefill")


def rows(days):
    from web.db import get_db
    return get_db().execute(
        "SELECT c.id, c.source, c.puzzle_number, c.clue_number, c.direction, "
        "       c.clue_text, c.answer, s.operation, s.solved_by "
        "FROM wfw_solve s JOIN clues c ON c.id = s.clue_id "
        "WHERE s.status = 'pass' "
        "  AND s.solved_by NOT IN (%s) "
        "  AND c.publication_date >= date('now', ?) "
        "ORDER BY c.publication_date DESC, c.source, c.puzzle_number, c.id"
        % ",".join("?" * len(NOT_ENGINE)),
        (*NOT_ENGINE, "-%d day" % days)).fetchall()


def render(r):
    """One pass as the handful of claims a reader has to agree with."""
    out = []
    head = "%s %s %s%s" % (r["source"], r["puzzle_number"], r["clue_number"],
                           (r["direction"] or "")[:1])
    out.append("%-22s %-16s [%s]" % (head, r["answer"] or "?", r["operation"] or "?"))
    out.append('  "%s"' % (r["clue_text"] or "").strip())
    parse = wfw_read._load(r["id"])
    if parse is None:
        out.append("  (no stored parse)")
        return "\n".join(out)
    for s in parse["sources"]:
        text, val = (s["text"] or "").strip(), (s["value"] or "").strip()
        mech = s["mechanism"] or "?"
        if mech == "definition":
            out.append('  def   "%s"' % text)
            continue
        xf = (s.get("transform") or "").strip()
        out.append("  say   %s -> %s%s   [%s]"
                   % (text or "(no words)", val or "?",
                      ("  " + xf) if xf else "", mech))
    for i in parse["indicators"]:
        out.append('  ind   "%s"   %s' % ((i["text"] or "").strip(),
                                          (i["note"] or "?").strip()))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out")
    args = ap.parse_args()
    app = create_app()
    with app.app_context():
        rs = rows(args.days)
        blocks = [render(r) for r in rs]
    body = ("\n\n".join(blocks) + "\n") if blocks else "(no engine passes)\n"
    header = "%d engine passes in the last %d days\n%s\n\n" % (
        len(rs), args.days, "=" * 60)
    if args.out:
        Path(args.out).write_text(header + body, encoding="utf-8")
        print("%d engine passes -> %s" % (len(rs), args.out))
    else:
        sys.stdout.write(header + body)


if __name__ == "__main__":
    main()
