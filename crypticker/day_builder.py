"""Build a 20-clue day: a difficulty ramp ending in three extreme clues.

Design (locked 2026-07-21): 20 clues, 2-minute sprint, difficulty rises across
the round. It stops being "can you finish" and becomes "how far can you get" —
the last three are a wall no one should clear, built by cranking the dials we
already have (biggest split-containers, no definition; long anagrams with the
foothold stripped to a minimum). No new mechanics.

Minimal sanity only here (single word, sane length, not degenerate, real word).
The full eligibility gate (leak check, archive-age, curator) is separate/parked.
"""
import sqlite3, os, sys, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.store import load_parse
from crypticker.serve_data import atoms_for
from crypticker.convert import classify, convert_clue, TIME_LIMIT_S, SKIP_PENALTY_S

DB = os.path.join(os.path.dirname(__file__), "..", "data", "clues_master.db")
REF = os.path.join(os.path.dirname(__file__), "..", "data", "cryptic_new.db")

DAY_SIZE = 20
EASY_SLOTS = 5            # slots 1-5: easy jumbled charades (def marked)
EXTREME_SLOTS = 3         # slots 18-20: the wall
EXTREME_ANAGRAM_ANCHORS = 2   # strip the foothold on tail anagrams


def _letters(s):
    return "".join(ch for ch in (s or "").upper() if ch.isalpha())


def difficulty(kind, natoms, ans_len, is_anagram):
    """Heuristic difficulty score (higher = harder). Anagrams score by length;
    assembly clues by kind (charade<reorder<split), piece count and length."""
    if is_anagram:
        return 5 + 0.8 * ans_len
    base = {"charade": 0, "reorder": 3, "split": 6}.get(kind, 3)
    return base + 1.5 * natoms + 0.4 * ans_len


def candidates(conn, ref):
    """Every PASS whitelist clue that passes minimal sanity, with its score."""
    rows = conn.execute(
        "SELECT clue_id, answer_text, operation FROM wfw_solve WHERE status='pass' "
        "AND operation IN ('charade','container','container_charade','anagram')").fetchall()
    out = []
    for cid, answer, op in rows:
        if " " in (answer or "") or "-" in (answer or ""):
            continue                                   # single word only
        n = len(_letters(answer))
        # real word (drops obvious proper nouns cheaply)
        if ref.execute("SELECT count(*) FROM definition_answers_augmented "
                       "WHERE upper(answer)=?", (_letters(answer),)).fetchone()[0] == 0:
            continue
        parse = load_parse(conn, cid)
        if parse is None:
            continue
        is_anag = (op == "anagram")
        if is_anag:
            if not (7 <= n <= 15):
                continue
            kind, natoms = "anagram", n
        else:
            if not (4 <= n <= 10):
                continue
            atoms, order = atoms_for(parse)
            if not (2 <= len(atoms) <= 5):
                continue
            if all(len(a["true"]) <= 2 for a in atoms):
                continue                               # degenerate
            kind, natoms = classify(atoms, order)[0], len(atoms)
        out.append({"cid": cid, "kind": kind, "natoms": natoms, "len": n,
                    "anag": is_anag,
                    "score": difficulty(kind, natoms, n, is_anag)})
    return out


def select_day(conn, ref):
    """Pick 20 clue_ids forming an easy->extreme ramp. Deterministic."""
    cands = candidates(conn, ref)
    cands.sort(key=lambda c: (c["score"], c["cid"]))

    # extreme tail: the hardest 3 (prefer 5-piece splits / long anagrams — these
    # top the score list already). Take from the high end.
    extreme = cands[-EXTREME_SLOTS:]
    pool = cands[:-EXTREME_SLOTS]

    # easy opener: the 5 lowest-score charades (jumble-safe -> easy mode)
    easy = [c for c in pool if c["kind"] == "charade"][:EASY_SLOTS]
    easy_ids = {c["cid"] for c in easy}
    middle_pool = [c for c in pool if c["cid"] not in easy_ids]

    # middle 12: evenly spaced across the remaining score distribution
    n_mid = DAY_SIZE - EASY_SLOTS - EXTREME_SLOTS
    step = max(1, len(middle_pool) / n_mid)
    middle = [middle_pool[min(len(middle_pool) - 1, int(i * step))] for i in range(n_mid)]

    ordered = easy + middle + extreme
    return ordered


def build(conn, ref, puzzle_no=1, date="2026-09-01"):
    chosen = select_day(conn, ref)
    clues = []
    for i, c in enumerate(chosen):
        slot = i + 1
        # mode follows the SLOT, not the clue kind: only the openers are easy
        # (jumbled + def marked). Later charades render hard (clean, no def).
        if c["anag"]:
            mode = None                                 # convert_clue forces "anagram"
        else:
            mode = "easy" if slot <= EASY_SLOTS else "hard"
        max_anchors = EXTREME_ANAGRAM_ANCHORS if (slot > DAY_SIZE - EXTREME_SLOTS and c["anag"]) else None
        clue = convert_clue(conn, c["cid"], mode=mode, max_anchors=max_anchors)
        clue["slot"] = slot
        clue["difficulty"] = round(c["score"], 1)
        clues.append(clue)
    return {"puzzle_no": puzzle_no, "date": date, "time_limit_s": TIME_LIMIT_S,
            "skip_penalty_s": SKIP_PENALTY_S, "clues": clues}


if __name__ == "__main__":
    conn, ref = sqlite3.connect(DB), sqlite3.connect(REF)
    day = build(conn, ref)
    out = os.path.join(os.path.dirname(__file__), "days", "day_20.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(day, f, indent=2, default=str, ensure_ascii=False)
    print("wrote", out, "\n")
    for c in day["clues"]:
        built = "".join(c["atoms"][i]["true"] for i in c["assembly_order"] if i is not None)
        ok = "OK" if built == _letters(c["answer"]) else "MISMATCH"
        extra = ""
        if c["mode"] == "anagram":
            extra = " anchors=%d/%d" % (len(c["anchors"]), len(c["atoms"]))
        print("  %2d  diff=%-5.1f %-8s %-7s %-13s %s%s"
              % (c["slot"], c["difficulty"], c["kind"], c["mode"], c["answer"], ok, extra))
