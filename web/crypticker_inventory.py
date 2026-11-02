"""Crypticker quadrant inventory — one small JSON file per puzzle.

    crypticker/v2/inventory/<id>.json      (shipped by the dashboard code deploy)
        {"id", "status": "to_test" | "approved" | "rejected",
         "built_at", "approved_at", "release_date", "game"}

A candidate is built (status to_test), played by the user in admin mode with the
clock off, then APPROVED or REJECTED. Approving fixes its release date: the next
free UK date, never earlier than today. The live site only ever reads files, so
the droplet runs nothing and writes nothing; new approvals go live with a deploy.

Reading needs only the stdlib. Building imports the quadrant builder (MAIN tree,
clues_master.db) lazily, so the public page never loads it.
"""
import datetime, json, os, re
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # project root
BUILDER = os.path.join(ROOT, "crypticker", "v2")                      # build_quadrant (local only)
DIR = os.path.join(BUILDER, "inventory")
UK = ZoneInfo("Europe/London")


def uk_today():
    return datetime.datetime.now(UK).date().isoformat()


def _path(pid):
    if not re.fullmatch(r"[a-z0-9-]+", pid or ""):
        raise ValueError("bad id %r" % pid)
    return os.path.join(DIR, pid + ".json")


def load(pid):
    with open(_path(pid), encoding="utf-8") as f:
        return json.load(f)


def save(item):
    os.makedirs(DIR, exist_ok=True)
    tmp = _path(item["id"]) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _path(item["id"]))


def items():
    if not os.path.isdir(DIR):
        return []
    out = []
    for name in os.listdir(DIR):
        if name.endswith(".json"):
            with open(os.path.join(DIR, name), encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def game_for(date):
    """The approved game released on this date, or None."""
    for it in items():
        if it["status"] == "approved" and it.get("release_date") == date:
            return it["game"]
    return None


def schedule():
    """Approved items by release date (past = released, today/future = inventory)."""
    return sorted((it for it in items() if it["status"] == "approved"),
                  key=lambda it: it["release_date"])


def approve(pid):
    it = load(pid)
    if it["status"] == "approved":
        return it
    # the EARLIEST free date from today: a rejected scheduled puzzle leaves no hole
    taken = {x["release_date"] for x in schedule()}
    nxt = datetime.date.fromisoformat(uk_today())
    while nxt.isoformat() in taken:
        nxt += datetime.timedelta(days=1)
    it.update(status="approved", release_date=nxt.isoformat(),
              approved_at=datetime.datetime.now(UK).isoformat(timespec="seconds"))
    save(it)
    return it


def reject(pid):
    it = load(pid)
    if it["status"] == "approved" and it["release_date"] <= uk_today():
        raise ValueError("already released")
    it.update(status="rejected", release_date=None)
    save(it)
    return it


def build_candidate():
    """Build one new to-test game from a puzzle never used before (any status).
    Must run inside a Flask app context whose CLUES_DB is clues_master.db."""
    import sys
    if BUILDER not in sys.path:
        sys.path.insert(0, BUILDER)
    import build_quadrant as bq                      # heavy: MAIN tree + DB
    every = items()
    used = {(it["game"]["puzzle"]["source"], str(it["game"]["puzzle"]["number"])) for it in every}
    now = datetime.datetime.now(UK)
    game, tried = bq.pick(now.isoformat(), exclude=used, rotation=len(every))
    if game is None:
        return None, tried
    pz = game["puzzle"]
    label = "%s %s %s" % (pz["source"], pz["number"], pz["quadrant"])
    pid = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    game.pop("date", None)
    item = {"id": pid, "status": "to_test", "built_at": now.isoformat(timespec="seconds"),
            "approved_at": None, "release_date": None, "game": game}
    save(item)
    return item, tried
