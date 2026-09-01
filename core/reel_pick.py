"""Which clue the day's reel is built from — chosen by hand, during review.

A FILE, not a table. The clue is a daily editorial choice, not reference data,
and the house rule is that database schemas are not modified; a new table would
be a schema change for something that holds one integer a day.

One file per publication date, so a re-pick simply overwrites and the record of
what was chosen on any given day survives.
"""

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PICKS = ROOT / "logs" / "reels" / "picks"


def _path(day=None):
    return PICKS / ("%s.json" % (day or date.today().isoformat()))


def set_pick(clue_id, day=None):
    """Record the chosen clue. Overwrites any earlier choice for that day."""
    PICKS.mkdir(parents=True, exist_ok=True)
    p = _path(day)
    p.write_text(json.dumps({"clue_id": int(clue_id)}), encoding="utf-8")
    return int(clue_id)


def get_pick(day=None):
    """The clue chosen for that day, or None. Never raises — a missing or
    corrupt file means nothing has been chosen, which is a normal state."""
    p = _path(day)
    if not p.exists():
        return None
    try:
        return int(json.loads(p.read_text(encoding="utf-8"))["clue_id"])
    except Exception:
        return None


def clear_pick(day=None):
    p = _path(day)
    if p.exists():
        p.unlink()
