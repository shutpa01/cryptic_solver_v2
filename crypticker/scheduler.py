"""Build a day from the DB pool, applying the date-relative gates.

Reads crypticker_build.db.clue_pool (seeded by seed_pool.py) + the day/attempt
history, and writes a DRAFT day (day + day_clue rows, payload_json NULL).
Publishing (freezing payloads) is a separate step: publish_day().

Gates applied here (all relative to the issue date; see memory crypticker-db-design):
  - 3-month embargo : pub_date <= issue_date - 3 months
  - six-month repeat: exclude clues ATTEMPTED within the last 6 months
                      ("spent" = attempted, not merely scheduled)
  - companion       : do not reunite a clue with members of its MOST RECENT set
  - rollover        : unattempted clues from the previous day carry forward

Curator status is ADVISORY for now (pending clues are allowed; only 'rejected'
is excluded) — the hard-gate decision is deferred.
"""
import sqlite3, os, sys, hashlib, datetime, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from crypticker.convert import convert_clue, TIME_LIMIT_S, SKIP_PENALTY_S
from crypticker.serve_gate import check as serve_check

HERE = os.path.dirname(__file__)
DB = os.path.join(HERE, "..", "data", "clues_master.db")
REF = os.path.join(HERE, "..", "data", "cryptic_new.db")
BUILD = os.path.join(HERE, "..", "data", "crypticker_build.db")

DAY_SIZE = 20
EXTREME_SLOTS = 3
EXTREME_ANAGRAM_ANCHORS = 2

# 3-month publication embargo: OFF during development (user 2026-10-31) so the
# full-WFW cohort (mostly Jul–Oct 2026 puzzles) is usable. TURN BACK ON AT LAUNCH.
EMBARGO_ON = False

# slot -> difficulty band (5 + 8 + 4 + 3 = 20)
def band_for_slot(slot):                       # slot is 1-based
    if slot <= 5:   return "easy"
    if slot <= 13:  return "medium"
    if slot <= 17:  return "hard"
    return "extreme"


def _shift_months(iso_date, months):
    d = datetime.date.fromisoformat(iso_date)
    m = d.month - 1 - months
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, [31,29 if y%4==0 and (y%100 or y%400==0) else 28,31,30,31,30,31,31,30,31,30,31][m-1])
    return datetime.date(y, m, day).isoformat()


def _seed_for(issue_date):
    return int(hashlib.sha1(issue_date.encode()).hexdigest()[:8], 16)


# ---- companion history -------------------------------------------------------
def _companions(build, clue_id, before_date):
    """clue_ids in this clue's most recent PREVIOUS set (strictly before
    before_date, so rebuilding a day never treats itself as its own predecessor).
    Empty if never used before."""
    row = build.execute("SELECT issue_date FROM day_clue WHERE clue_id=? "
                        "AND issue_date < ? ORDER BY issue_date DESC LIMIT 1",
                        (clue_id, before_date)).fetchone()
    if not row:
        return set()
    return {r[0] for r in build.execute(
        "SELECT clue_id FROM day_clue WHERE issue_date=? AND clue_id<>?",
        (row[0], clue_id))}


# ---- candidate gathering -----------------------------------------------------
def gated_candidates(build, issue_date):
    """Eligible pool after embargo + six-month gates, as dicts sorted by difficulty."""
    embargo = _shift_months(issue_date, 3) if EMBARGO_ON else "9999-12-31"
    six_mo = _shift_months(issue_date, 6)
    rows = build.execute(
        "SELECT p.clue_id, p.kind, p.difficulty "
        "FROM clue_pool p "
        "WHERE p.structurally_eligible=1 "
        "  AND p.curator_status <> 'rejected' "
        "  AND p.pub_date IS NOT NULL AND p.pub_date <= ? "
        "  AND p.clue_id NOT IN ("
        "        SELECT clue_id FROM clue_attempt WHERE attempted=1 AND issue_date > ?) "
        "ORDER BY p.difficulty, p.clue_id", (embargo, six_mo)).fetchall()
    return [dict(clue_id=r[0], kind=r[1], difficulty=r[2]) for r in rows]


def rollover_ids(build, issue_date):
    """Unattempted clues from the most recent published day before issue_date.
    Returns [(clue_id, prior_issue_date, band)]."""
    row = build.execute(
        "SELECT issue_date FROM day WHERE status IN ('published','closed') "
        "AND issue_date < ? ORDER BY issue_date DESC LIMIT 1", (issue_date,)).fetchone()
    if not row:
        return []
    prior = row[0]
    out = []
    for cid, band in build.execute(
            "SELECT dc.clue_id, dc.difficulty_band FROM day_clue dc "
            "WHERE dc.issue_date=? AND dc.clue_id NOT IN ("
            "   SELECT clue_id FROM clue_attempt WHERE issue_date=? AND attempted=1)",
            (prior, prior)):
        out.append((cid, prior, band))
    return out


# ---- selection: an easy->extreme ramp of DAY_SIZE clues ----------------------
def select_day(build, issue_date):
    cands = gated_candidates(build, issue_date)
    if len(cands) < DAY_SIZE:
        raise RuntimeError("only %d gated candidates for %s (need %d)"
                           % (len(cands), issue_date, DAY_SIZE))
    seed = _seed_for(issue_date)

    def rank(cid):                          # per-date deterministic shuffle key
        return int(hashlib.md5(("%d:%d" % (seed, cid)).encode()).hexdigest(), 16)

    by_id = {c["clue_id"]: c for c in cands}
    used, chosen = set(), []

    def clean(cid):                         # unused AND no reunited prior companions
        return cid not in used and not (_companions(build, cid, issue_date) & used)

    # 1. rollover: unattempted prior-day clues keep their place (slotted by difficulty)
    for cid, prior, band in rollover_ids(build, issue_date):
        if cid in by_id and clean(cid):
            used.add(cid); chosen.append(cid)

    # 2. ramp WITH variety: sort the rest by difficulty, carve into `need` contiguous
    #    bins, pick ONE clue per bin by seeded rank. Bins run easy->hard so the ramp
    #    holds; the seeded pick means different DATES draw different clues (not always
    #    the lowest clue_id / oldest Guardian) and spread across all publications.
    remaining = sorted((c for c in cands if c["clue_id"] not in used),
                       key=lambda c: (c["difficulty"], c["clue_id"]))
    need = DAY_SIZE - len(chosen)
    m = len(remaining)
    for b in range(need):
        binslice = remaining[(b * m) // need:((b + 1) * m) // need] or remaining
        for c in sorted(binslice, key=lambda c: rank(c["clue_id"])):
            if clean(c["clue_id"]):
                used.add(c["clue_id"]); chosen.append(c["clue_id"]); break

    # 3. top up if a bin was emptied by companion conflicts
    if len(chosen) < DAY_SIZE:
        for c in sorted(remaining, key=lambda c: rank(c["clue_id"])):
            if len(chosen) >= DAY_SIZE:
                break
            if clean(c["clue_id"]):
                used.add(c["clue_id"]); chosen.append(c["clue_id"])

    # 4. final easy->extreme order
    chosen.sort(key=lambda cid: by_id[cid]["difficulty"])
    return chosen[:DAY_SIZE]


# ---- write a draft day -------------------------------------------------------
def build_day(build, issue_date):
    ordered = select_day(build, issue_date)
    seed = _seed_for(issue_date)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    build.execute("INSERT OR REPLACE INTO day (issue_date, seed, status, built_at) "
                  "VALUES (?,?, 'draft', ?)", (issue_date, seed, now))
    build.execute("DELETE FROM day_clue WHERE issue_date=?", (issue_date,))
    roll_map = {cid: prior for cid, prior, _ in rollover_ids(build, issue_date)}
    for i, cid in enumerate(ordered):
        slot = i + 1
        build.execute(
            "INSERT INTO day_clue (issue_date, slot, clue_id, difficulty_band, "
            " payload_json, rolled_from) VALUES (?,?,?,?, NULL, ?)",
            (issue_date, slot, cid, band_for_slot(slot), roll_map.get(cid)))
    build.commit()
    return ordered


# ---- publish: freeze each clue's converted payload; export the play JSON -----
def publish_day(build, conn, issue_date):
    """Convert every clue with its slot-appropriate mode and FREEZE the payload
    into day_clue.payload_json, then mark the day published. Frozen so a later
    re-parse in clues_master can't mutate a live puzzle."""
    rows = build.execute("SELECT slot, clue_id, difficulty_band FROM day_clue "
                         "WHERE issue_date=? ORDER BY slot", (issue_date,)).fetchall()
    refused = []
    for slot, cid, band in rows:
        mode = "easy" if band == "easy" else "hard"
        max_anchors = EXTREME_ANAGRAM_ANCHORS if band == "extreme" else None
        clue = convert_clue(conn, cid, mode=mode, max_anchors=max_anchors)
        clue["slot"] = slot
        clue["difficulty_band"] = band
        why = serve_check(clue)
        if why:   # pieces as shown can't make the answer -> never frozen, never served
            refused.append((slot, cid, clue.get("answer"), why))
            build.execute("UPDATE day_clue SET payload_json=NULL WHERE issue_date=? AND slot=?",
                          (issue_date, slot))
            continue
        build.execute("UPDATE day_clue SET payload_json=? WHERE issue_date=? AND slot=?",
                      (json.dumps(clue, ensure_ascii=False), issue_date, slot))
    for slot, cid, ans, why in refused:
        print("REFUSED slot %d cid=%s %s: %s" % (slot, cid, ans, why))
    build.execute("UPDATE day SET status='published' WHERE issue_date=?", (issue_date,))
    build.commit()


def export_day(build, issue_date):
    """Assemble the play-page day JSON from the FROZEN payloads."""
    ordinal = build.execute("SELECT count(*) FROM day WHERE issue_date<=?",
                            (issue_date,)).fetchone()[0]
    clues = []
    for (pj,) in build.execute(
            "SELECT payload_json FROM day_clue WHERE issue_date=? ORDER BY slot", (issue_date,)):
        if not pj:
            continue
        clue = json.loads(pj)
        why = serve_check(clue)   # last line: a payload frozen before the gate existed
        if why:
            print("NOT SERVED slot %s %s: %s" % (clue.get("slot"), clue.get("answer"), why))
            continue
        clues.append(clue)
    return {"puzzle_no": ordinal, "date": issue_date, "time_limit_s": TIME_LIMIT_S,
            "skip_penalty_s": SKIP_PENALTY_S, "clues": clues}


def main(issue_date):
    build = sqlite3.connect(BUILD)
    ordered = build_day(build, issue_date)
    print("built draft day %s (%d clues)\n" % (issue_date, len(ordered)))
    rows = build.execute(
        "SELECT dc.slot, dc.difficulty_band, p.kind, p.difficulty, dc.clue_id, dc.rolled_from "
        "FROM day_clue dc JOIN clue_pool p ON p.clue_id=dc.clue_id "
        "WHERE dc.issue_date=? ORDER BY dc.slot", (issue_date,)).fetchall()
    for slot, band, kind, diff, cid, roll in rows:
        r = "  <-rolled %s" % roll if roll else ""
        print("  %2d  %-8s %-14s diff=%-5.1f  cid=%s%s" % (slot, band, kind, diff, cid, r))


if __name__ == "__main__":
    issue = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat()
    main(issue)
