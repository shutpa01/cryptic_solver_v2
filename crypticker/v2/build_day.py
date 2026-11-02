"""Crypticker v2 — "rebuild the WFW page" (concept agreed with the user 2026-10-31).

The player is shown the clue AND the answer, the clue's words as chips (phrases
pre-grouped, as on the WFW page), and one box per role THIS clue needs. They
assign each chip to its box; a correct chip shows the letters it gives (as on the
WFW page), a wrong one bounces back (costs time only). 10 random reviewed clues,
30 s each; when a clue is done or its time runs out the completed WFW breakdown
shows until Next. Score = clues completed.

THE LIVE WFW PAGE IS THE SPEC. Every row here comes from the live site's own
`web.wfw_read.load_breakdown()` (the data behind partials/wfw_full.html), read from
the MAIN tree, never re-derived. The only additions are (a) the clue words each row
belongs to, gathered in the SAME append order + stable sort load_breakdown uses so
they line up one-to-one, and (b) indicator boxes naming what the indicator does
("Indicator: insert"), from the live `_note_mech`.

    python crypticker/v2/build_day.py [YYYY-MM-DD]   -> crypticker/v2/days/<date>.json
                                                       + crypticker/v2/play.html

Read-only against clues_master.db.
"""
import datetime, hashlib, json, os, random, re, sys

MAIN = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2"   # live code + live data
sys.path.insert(0, MAIN)

from flask import Flask                                  # noqa: E402
from web import wfw_read                                 # noqa: E402  (MAIN tree)

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(MAIN, "data", "clues_master.db")
DAY_SIZE = 10
CLUE_SECONDS = 30

# What an indicator DOES, keyed by the live _note_mech result (user 2026-10-31:
# "say what it does, e.g. insert, reverse").
VERB = {
    "container": "insert", "reversal": "reverse", "anagram": "rearrange",
    "deletion": "delete", "selection": "select letters", "hidden": "hidden word",
    "homophone": "sounds like", "alternation": "alternate letters",
    "acrostic": "first letters", "spoonerism": "swap sounds",
    "letter_shift": "move a letter", "substitution": "substitute",
    "replacement": "replace", "cycling": "cycle letters", "palindrome": "palindrome",
    "repetition": "repeat", "charade": "join",
}


def _indicator_box(note):
    n = (note or "").lower()
    if n.startswith("named/"):
        return "Indicator: " + note.split("/", 1)[1].replace("indicator", "").strip()
    if "positional" in n:
        return "Indicator: position"
    m = wfw_read._note_mech(note)
    return "Indicator: " + VERB[m] if m in VERB else "Indicator"


def _items(parse):
    """(clue_pos, text, box, source_index) for every row, in load_breakdown's exact append order."""
    P = wfw_read._clue_pos
    out = []
    for d in parse["definitions"]:
        out.append((P(d.get("atom_ids")), d["text"], "Definition", None))
    for s in parse["sources"]:
        pos = s.get("clue_pos", wfw_read._UNPLACED)
        if s["mechanism"] == "definition":
            out.append((pos, s["text"], "Definition", None))
            continue
        out.append((pos, s["text"], wfw_read._MECH_LABEL.get(
            s["mechanism"], (s["mechanism"] or "Piece").title()), s["ord"]))
    for ind in parse["indicators"]:
        out.append((P(ind.get("atom_ids")), ind["text"], _indicator_box(ind["note"]), None))
    for p in parse["shifted"]:
        out.append((P(p.get("atom_ids")), p["text"], "Letter moved", None))
    for p in parse["deletions"]:
        out.append((P(p.get("atom_ids")), p["text"], "Deleted", None))
    db = wfw_read.get_db()
    for p in db.execute("SELECT text, atom_ids FROM wfw_piece WHERE clue_id=? AND role='link' "
                        "ORDER BY ord", (parse["_cid"],)):
        out.append((P(p["atom_ids"]), p["text"], "Link word", None))
    return [x for x in sorted(out, key=lambda x: x[0])]


# A clue that points at another clue ("See 10", "mum's 3 Down", "20ac") can't be
# played on its own (user 2026-10-31). A bare number is a reference unless it is a
# quantity: "50 per cent", "25%", "$1", "1950s", "18th", "1/3", "0.9".
_REF = re.compile(r"(?<![$£€\d.,])\b\d+(?:\s*(?:across|down|ac|dn|a|d)\b|\b)"
                  r"(?!\s*(?:per\s*cent|percent|%|st\b|nd\b|rd\b|th\b|s\b)|[.,/]\d)", re.I)


TYPE_CHOICES = (wfw_read._OP_LABEL["cd"], wfw_read._OP_LABEL["double_definition"],
                wfw_read._OP_LABEL["andlit"])


def _cross_reference(text, enum, answer):
    if _REF.search(text or ""):
        return "references another clue"
    # half of a linked answer: CURRY stored under (5,6)
    n = [int(x) for x in re.findall(r"\d+", enum or "")]
    if n and sum(n) != len(re.sub(r"[^A-Za-z]", "", answer or "")):
        return "enumeration covers a linked answer"
    return None


def build_clue(cid):
    """One clue's play payload, or (None, reason)."""
    db = wfw_read.get_db()
    c = db.execute("SELECT source, puzzle_number, clue_number, direction, enumeration, "
                   "clue_text, answer, publication_date FROM clues WHERE id=?", (cid,)).fetchone()
    why = _cross_reference(c["clue_text"], c["enumeration"], c["answer"])
    if why:
        return None, why
    parse = wfw_read._load(cid)
    if parse is None:
        return None, "no pass parse"
    parse["_cid"] = cid
    b = wfw_read.load_breakdown(cid)
    if b is None:
        return None, "no breakdown"
    rows = b["rows"]
    # Clues with no wordplay pieces to assign (CD, DD, &lit): the player picks the
    # clue type from a list instead (user 2026-10-31). Names = the live badge.
    is_type = b["operation_label"] in TYPE_CHOICES
    items = [] if is_type else _items(parse)
    if not is_type and len(items) != len(rows):
        return None, "rows/items misaligned (%d/%d)" % (len(rows), len(items))
    chips, seen = [], set()
    for (pos, text, box, si), r in zip(items, rows):
        key = (pos, (text or "").strip().lower(), box)
        if key in seen:                       # one word, one job, one chip
            continue
        seen.add(key)
        chips.append({"text": (text or "").strip(), "box": box, "pos": pos, "si": si,
                      "detail": r.get("detail", ""), "detail_html": r.get("detail_html"),
                      "fg": r["fg"], "fill": r["fill"]})
    if not chips and not is_type:
        return None, "no chips"
    boxes = []
    for ch in chips:                          # only the boxes this clue needs
        if ch["box"] not in boxes:
            boxes.append(ch["box"])
    return {
        "clue_id": cid,
        "clue": c["clue_text"], "enum": c["enumeration"],
        "answer": parse["answer_text"].upper(),
        "source": {"publication": c["source"], "puzzle": c["puzzle_number"],
                   "position": "%s%s" % (c["clue_number"] or "",
                                         (c["direction"] or "")[:1].upper()),
                   "pub_date": c["publication_date"]},
        "operation_label": b["operation_label"],
        "summary": b["summary"],
        "answer_tiles": b["answer_tiles"],
        "src_fg": b["src_fg"], "src_fill": b["src_fill"],
        "rows": rows,                          # the completed WFW, exactly as the site
        "chips": chips,
        "boxes": boxes,
        "choices": list(TYPE_CHOICES) if is_type else None,
    }, None


def candidates():
    db = wfw_read.get_db()
    return [r[0] for r in db.execute(
        "SELECT s.clue_id FROM wfw_solve s JOIN wfw_frozen f ON f.clue_id=s.clue_id "
        "WHERE s.status='pass' ORDER BY s.clue_id")]


def build_day(date):
    ids = candidates()
    rng = random.Random(int(hashlib.md5(date.encode()).hexdigest()[:8], 16))
    rng.shuffle(ids)
    clues, skipped = [], []
    for cid in ids:
        payload, why = build_clue(cid)
        if payload is None:
            skipped.append((cid, why))
            continue
        clues.append(payload)
        if len(clues) == DAY_SIZE:
            break
    return {"date": date, "clue_seconds": CLUE_SECONDS, "clues": clues}, skipped


def main(date):
    app = Flask(__name__)
    app.config["CLUES_DB"] = DB
    with app.app_context():
        day, skipped = build_day(date)
    os.makedirs(os.path.join(HERE, "days"), exist_ok=True)
    with open(os.path.join(HERE, "days", date + ".json"), "w", encoding="utf-8") as f:
        json.dump(day, f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "play_template.html"), encoding="utf-8") as f:
        tmpl = f.read()
    with open(os.path.join(HERE, "play.html"), "w", encoding="utf-8") as f:
        f.write(tmpl.replace("__DAY_JSON__", json.dumps(day, ensure_ascii=False)))
    print("day %s: %d clues" % (date, len(day["clues"])))
    for c in day["clues"]:
        print("  %-16s %-28s boxes=%s" % (c["answer"], c["operation_label"], c["boxes"]))
    for cid, why in skipped:
        print("  skipped %s: %s" % (cid, why))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat())
