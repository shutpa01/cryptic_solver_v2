"""Crypticker eligibility gate (spec 4.3): decide whether a PASS clue may be
served, and why not. Read-only against both DBs.

Confident mechanical checks are hard rejects. Proper-noun obscurity is NOT
fully mechanical (spec: curator judgement), so we only hard-reject answers with
zero dictionary presence and FLAG thin ones for the human review step.
"""
import sqlite3, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.store import load_parse
from crypticker.serve_data import atoms_for

DB = os.path.join(os.path.dirname(__file__), "..", "data", "clues_master.db")
REF = os.path.join(os.path.dirname(__file__), "..", "data", "cryptic_new.db")

WHITELIST = {"charade", "container", "container_charade", "anagram"}

# defaults (open decisions — tunable)
ARCHIVE_CUTOFF = "2021-07-21"     # ~5 years before 2026-07-21
ASSEMBLY_LEN = (4, 10)            # single-word answer letter count
ANAGRAM_LEN = (7, 15)
ATOM_RANGE = (2, 5)
DICT_FLAG_BELOW = 3              # dictionary entries below this -> curator flag


def _letters(s):
    return "".join(ch for ch in (s or "").upper() if ch.isalpha())


def _dict_count(ref, answer):
    return ref.execute(
        "SELECT count(*) FROM definition_answers_augmented WHERE upper(answer)=?",
        (_letters(answer),)).fetchone()[0]


def check(conn, ref, clue_id):
    """Return (ok, reasons, flags). ok=True means servable (flags may still ask
    for a curator glance)."""
    reasons, flags = [], []
    head = conn.execute(
        "SELECT s.operation, s.answer_text, s.clue_text, c.publication_date "
        "FROM wfw_solve s LEFT JOIN clues c ON c.id=s.clue_id "
        "WHERE s.clue_id=? AND s.status='pass'", (clue_id,)).fetchone()
    if not head:
        return False, ["not a PASS clue"], []
    op, answer, clue_text, pub_date = head

    if op not in WHITELIST:
        reasons.append("device not whitelisted (%s)" % op)
    if not pub_date:
        flags.append("no publication date")
    elif str(pub_date) >= ARCHIVE_CUTOFF:
        reasons.append("too recent (%s >= %s)" % (pub_date, ARCHIVE_CUTOFF))

    multiword = (" " in (answer or "")) or ("-" in (answer or ""))
    n = len(_letters(answer))

    if op == "anagram":
        if multiword:
            reasons.append("anagram answer not a single word")
        if not (ANAGRAM_LEN[0] <= n <= ANAGRAM_LEN[1]):
            reasons.append("anagram length %d outside %s" % (n, ANAGRAM_LEN))
    else:
        if multiword:
            reasons.append("answer not a single word")
        if not (ASSEMBLY_LEN[0] <= n <= ASSEMBLY_LEN[1]):
            reasons.append("length %d outside %s" % (n, ASSEMBLY_LEN))
        parse = load_parse(conn, clue_id)
        if parse:
            atoms, _ = atoms_for(parse)
            if not (ATOM_RANGE[0] <= len(atoms) <= ATOM_RANGE[1]):
                reasons.append("atom count %d outside %s" % (len(atoms), ATOM_RANGE))
            if all(len(a["true"]) <= 2 for a in atoms):
                reasons.append("degenerate (all tiles 1-2 letters)")
            low = (clue_text or "").lower()
            for a in atoms:
                if len(a["true"]) >= 2 and a["true"].lower() in low:
                    reasons.append("leak: '%s' visible in clue" % a["true"])
                    break

    dc = _dict_count(ref, answer)
    if dc == 0:
        reasons.append("answer absent from dictionary (likely proper noun)")
    elif dc < DICT_FLAG_BELOW:
        flags.append("thin dictionary presence (%d) — curator check" % dc)

    return (len(reasons) == 0), reasons, flags


def survey():
    conn, ref = sqlite3.connect(DB), sqlite3.connect(REF)
    import collections
    ok = collections.Counter()
    rej = collections.Counter()
    flagged = 0
    ids = [r[0] for r in conn.execute(
        "SELECT clue_id FROM wfw_solve WHERE status='pass' AND operation IN "
        "('charade','container','container_charade','anagram')")]
    for cid in ids:
        good, reasons, flags = check(conn, ref, cid)
        op = conn.execute("SELECT operation FROM wfw_solve WHERE clue_id=?",
                          (cid,)).fetchone()[0]
        if good:
            ok[op] += 1
            if flags:
                flagged += 1
        else:
            for r in reasons:
                rej[r.split(":")[0].split("(")[0].strip()] += 1
    print("ELIGIBLE (servable) by device:")
    for op in ("charade", "container", "container_charade", "anagram"):
        print("  %-20s %4d" % (op, ok[op]))
    print("  %-20s %4d" % ("TOTAL", sum(ok.values())))
    print("  (of which flagged for curator: %d)" % flagged)
    print("\nREJECTIONS by reason (clue may hit several):")
    for reason, n in rej.most_common():
        print("  %4d  %s" % (n, reason))


if __name__ == "__main__":
    survey()
