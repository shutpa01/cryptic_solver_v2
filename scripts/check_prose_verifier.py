"""Assert the prose checker can recognise EVERY answer we hold.

    python scripts/check_prose_verifier.py          # all answers, exit 1 on any failure
    python scripts/check_prose_verifier.py -v       # list the failures

Read-only. Touches no database row, writes nothing, calls no model.

WHY THIS EXISTS
---------------
2026-09-24. The checker refused a correct sentence for 4a of telegraph 31287
because the answer is A BIT MUCH: its capitals scan reads runs of words two
letters or longer, so the article was dropped, the run was read as BIT MUCH, and
BIT was reported as an invention.

The user's objection was the right one, and it was not about that clue: if the
checker can fail on the ANSWER — the one fact held without any doubt — then its
verdicts cannot be trusted anywhere. The fix was to COMPARE the answer rather
than infer it (draft_prose.answer_blanked); this is the test that says so, and
that stops the class coming back silently.

A failure here does NOT mean a clue is wrong. It means the checker would call a
faithful sentence an invention, and would refuse prose it should have filed.
"""

import argparse
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.draft_prose import _CAPS_RUN, answer_blanked, fold   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = ROOT / "data" / "clues_master.db"


def unrecognised(answer):
    """True when the checker cannot account for this answer written as itself.

    The sentence a drafter writes ends with the answer in capitals, so that is
    exactly what is fed in. If anything of it survives `answer_blanked`, the
    capitals scan would go on to judge the leftover on its own and call it an
    invention.
    """
    shouted = fold(answer).upper()
    if not re.search(r"[A-Z]", shouted):
        return False                      # nothing to recognise (digits only)
    return bool(_CAPS_RUN.findall(answer_blanked(shouted, answer)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every answer that fails")
    args = ap.parse_args()

    con = sqlite3.connect(str(CLUES_DB))
    rows = con.execute("SELECT DISTINCT answer FROM clues "
                       "WHERE answer IS NOT NULL AND answer <> ''").fetchall()
    con.close()

    bad = [a for (a,) in rows if unrecognised(a)]
    print("answers checked : %d" % len(rows))
    print("unrecognised    : %d" % len(bad))
    if bad:
        for a in (bad if args.verbose else bad[:20]):
            print("  %s" % a)
        if not args.verbose and len(bad) > 20:
            print("  ... %d more (-v for all)" % (len(bad) - 20))
        print("\nFAIL — the checker would refuse a faithful sentence for these.")
        return 1
    print("\nOK — every answer is recognised.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
