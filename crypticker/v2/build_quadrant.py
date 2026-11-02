"""Crypticker v2 — QUADRANT game (concept agreed with the user 2026-11-01).

One corner of a real puzzle: its whole set of interlocking clues, 6 or 7 (rule
agreed with the user 2026-11-01 after the grid survey). The player plays every
clue but the last in the
words-into-boxes style (build_day.build_clue, unchanged); each one completed
writes its answer into an always-on grid, a missed one leaves its squares blank.
The last clue is the plain clue: 30 s to TYPE the answer, helped by whatever
crossing letters the player earned. Score = 1 per clue + 3 for the last.

  - last clue: picked at RANDOM (some days are hard), but it must cross at
    least 2 of the other clues (user: "we must always have some crossing potential").
  - grid: the live site's own builder (web.grid.build_grid_from_json, falling back
    to the stored solution exactly as routes/puzzle.py:puzzle_grid does).
  - a corner = the entries with most of their squares in that quarter of a 15x15
    grid. 6-7: served whole. 8+: trimmed to 7, fewest crossings dropped first,
    never splitting it. 5 or fewer: never served. Every served clue must be
    playable (reviewed, no reference to another clue, answer fills its squares).
  - the grid shown is always the 8x8 corner (a clue starting outside it shows its
    number on its first square inside); long answers run off the edge.
  - publication rotates day by day: Times, Telegraph, Guardian, Toughie.

    python crypticker/v2/build_quadrant.py [YYYY-MM-DD]
        -> crypticker/v2/quad_days/<date>.json + crypticker/v2/quad.html

Read-only against clues_master.db.
"""
import collections, datetime, hashlib, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_day as bd                                     # noqa: E402  (puts MAIN on sys.path)
from flask import Flask                                    # noqa: E402
from web.grid import build_grid_from_json, parse_grid_solution   # noqa: E402  (MAIN tree)
from web.models import get_puzzle_grid_data, get_puzzle_grid_solution  # noqa: E402

GAME_MIN, GAME_MAX = 6, 7          # clues per game, last included: the whole corner
WINDOW = 8                         # the grid shown: an 8x8 corner, the same size every day
SOURCES = ["times", "telegraph", "guardian", "telegraph-toughie"]   # never Independent / Daily Mail
MIN_CROSSINGS = 2                  # the last clue crosses at least this many others
LAST_SECONDS = 30
LAST_POINTS = 3


def _grid(source, number):
    """Same order as routes/puzzle.py:puzzle_grid — JSON first, stored solution second."""
    g = build_grid_from_json(source, number, get_puzzle_grid_data(source, number))
    if g is None:
        stored = get_puzzle_grid_solution(source, number)
        if stored:
            g = parse_grid_solution(*stored)
    return g


def _entries(grid):
    """{(number, direction): [(r, c), ...]} traced from each numbered square."""
    cells, R, C = grid["cells"], grid["rows"], grid["cols"]
    white = lambda r, c: 0 <= r < R and 0 <= c < C and cells[r][c] is not None
    out = {}
    for r in range(R):
        for c in range(C):
            if not white(r, c) or "number" not in cells[r][c]:
                continue
            n = int(cells[r][c]["number"])
            for d, (dr, dc) in (("across", (0, 1)), ("down", (1, 0))):
                if white(r + dr, c + dc) and not white(r - dr, c - dc):
                    run, rr, cc = [], r, c
                    while white(rr, cc):
                        run.append((rr, cc)); rr += dr; cc += dc
                    out[(n, d)] = run
    return out


def _letters(s):
    return re.sub(r"[^A-Za-z]", "", s or "").upper()


def quadrants(source, number, reviewed):
    """Yield (quadrant_name, corner, grid, (r0, c0)) for each quadrant of the puzzle.

    corner = [(key, cells, clue_id, playable)]: EVERY entry with most of its squares
    in that quarter (by start square alone, Downs starting just above the middle
    line were lost and nothing crossed). (r0, c0) = top-left of the 8x8 window.
    """
    grid = _grid(source, number)
    if grid is None or (grid["rows"], grid["cols"]) != (15, 15):
        return
    ents = _entries(grid)
    db = bd.wfw_read.get_db()
    clue_of = {}
    for r in db.execute("SELECT id, clue_number, direction, answer FROM clues "
                        "WHERE source=? AND puzzle_number=?", (source, str(number))):
        try:
            clue_of[(int(r["clue_number"]), (r["direction"] or "").lower())] = (r["id"], r["answer"])
        except (TypeError, ValueError):
            pass
    R, C = grid["rows"], grid["cols"]
    top, left = (lambda r: r < R / 2), (lambda c: c < C / 2)
    for qname, rows_ok, cols_ok, r0, c0 in (
            ("top left", top, left, 0, 0),
            ("top right", top, lambda c: not left(c), 0, C - WINDOW),
            ("bottom left", lambda r: not top(r), left, R - WINDOW, 0),
            ("bottom right", lambda r: not top(r), lambda c: not left(c), R - WINDOW, C - WINDOW)):
        corner = []
        for k, cells in sorted(ents.items(), key=lambda kv: (kv[0][1] != "across", kv[0][0])):
            if sum(rows_ok(r) and cols_ok(c) for r, c in cells) * 2 <= len(cells):
                continue
            cid, ans = clue_of.get(k, (None, None))
            playable = cid in reviewed and len(_letters(ans)) == len(cells)
            corner.append((k, cells, cid, playable))
        yield qname, corner, grid, (r0, c0)


def _crossings(quad):
    cell_owner = collections.defaultdict(set)
    for q in quad:
        for rc in q[1]:
            cell_owner[rc].add(q[0])
    return {q[0]: {o for rc in q[1] for o in cell_owner[rc] if o != q[0]} for q in quad}


def _connected(quad):
    if not quad:
        return False
    cross = _crossings(quad)
    seen, todo = set(), [quad[0][0]]
    while todo:
        k = todo.pop()
        if k not in seen:
            seen.add(k)
            todo.extend(cross[k])
    return len(seen) == len(quad)


def trim(corner):
    """Corners of 8+ come down to GAME_MAX: drop the entry with the fewest crossings
    first (unplayable, then longer, breaks a tie), never splitting the corner."""
    corner = list(corner)
    while len(corner) > GAME_MAX:
        cross = _crossings(corner)
        can = [q for q in corner if _connected([x for x in corner if x is not q])]
        if not can:
            return None
        drop = min(can, key=lambda q: (len(cross[q[0]]), q[3], -len(q[1])))
        corner.remove(drop)
    return corner


def build_game(source, number, qname, corner, grid, origin, rng):
    if len(corner) < GAME_MIN:
        return None, "only %d clues" % len(corner)
    if not _connected(corner):
        return None, "corner not interlocked"
    quad = trim(corner)
    if quad is None:
        return None, "cannot trim without splitting"
    payloads = {}
    for k, cells, cid, playable in quad:
        if not playable:
            return None, "%s%s not reviewed" % (k[0], k[1][0].upper())
        p, why = bd.build_clue(cid)
        if p is None:
            return None, "%s%s: %s" % (k[0], k[1][0].upper(), why)
        payloads[k] = p
    cross = _crossings(quad)
    last_pool = sorted(k for k in cross if len(cross[k]) >= MIN_CROSSINGS)
    if not last_pool:
        return None, "no clue crosses %d others" % MIN_CROSSINGS
    last = rng.choice(last_pool)
    # the window: always the 8x8 corner. A clue that starts outside it shows its
    # number on its first square inside (stretching to the real start made
    # SUSPENDER BELT's corner 13 wide)
    r0, c0 = origin
    r1, c1 = r0 + WINDOW - 1, c0 + WINDOW - 1
    inside = lambda rc: r0 <= rc[0] <= r1 and c0 <= rc[1] <= c1
    used = {rc for q in quad for rc in q[1]}
    starts = {}
    for q in quad:
        first = next(rc for rc in q[1] if inside(rc))
        starts.setdefault(first, q[0][0])
    window = []
    for r in range(r0, r1 + 1):
        row = []
        for c in range(c0, c1 + 1):
            if grid["cells"][r][c] is None:
                row.append({"t": "b"})                           # black square
            elif (r, c) in used:
                row.append({"t": "w", "n": starts.get((r, c))})  # one of ours
            else:
                row.append({"t": "o"})                           # not in the game
        window.append(row)
    entries = []
    for k, cells, cid, _ in quad:
        p = payloads[k]
        p["label"] = "%d %s" % (k[0], k[1].capitalize())
        # window coordinates; squares that run off the edge keep their place in
        # the answer (the last clue's tiles still need every square)
        p["cells"] = [[r - r0, c - c0] for r, c in cells]
        p["letters"] = _letters(p["answer"])
        p["is_last"] = (k == last)
        p["crosses"] = len(cross[k])
        entries.append(p)
    entries.sort(key=lambda p: p["is_last"])          # stable: puzzle order, last clue at the end
    return {"puzzle": {"source": source, "number": number, "quadrant": qname},
            "window": window, "entries": entries,
            "clue_seconds": bd.CLUE_SECONDS, "last_seconds": LAST_SECONDS,
            "last_points": LAST_POINTS}, None


def reviewed_ids():
    return set(bd.candidates())


def pick(date, exclude=(), rotation=None):
    """Publication rotates day by day; within it, a random puzzle and corner.

    exclude  = {(source, number)} never to use (the inventory's used + rejected).
    rotation = which publication goes first; default = the date's day number.
    """
    exclude = {(s, str(n)) for s, n in exclude}
    reviewed = reviewed_ids()
    db = bd.wfw_read.get_db()
    per = collections.Counter()
    for r in db.execute("SELECT source, puzzle_number, id FROM clues WHERE id IN (%s)"
                        % ",".join(map(str, reviewed))):
        per[(r["source"], r["puzzle_number"])] += 1
    rng = random.Random(int(hashlib.md5(date.encode()).hexdigest()[:8], 16))
    day = rotation if rotation is not None else datetime.date.fromisoformat(date[:10]).toordinal()
    order = SOURCES[day % len(SOURCES):] + SOURCES[:day % len(SOURCES)]
    tried = []
    for src in order:
        puzzles = sorted(k for k, n in per.items() if k[0] == src and n >= GAME_MIN)
        rng.shuffle(puzzles)
        for source, number in puzzles:
            if (source, str(number)) in exclude:
                continue
            corners = list(quadrants(source, number, reviewed))
            rng.shuffle(corners)
            for qname, corner, grid, origin in corners:
                game, why = build_game(source, number, qname, corner, grid, origin, rng)
                if game:
                    game["date"] = date
                    return game, tried
                tried.append("%s %s %s: %s" % (source, number, qname, why))
    return None, tried


def main(date):
    app = Flask(__name__)
    app.config["CLUES_DB"] = bd.DB
    with app.app_context():
        game, tried = pick(date)
    if game is None:
        print("no playable quadrant found")
        return
    os.makedirs(os.path.join(HERE, "quad_days"), exist_ok=True)
    with open(os.path.join(HERE, "quad_days", date + ".json"), "w", encoding="utf-8") as f:
        json.dump(game, f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "quad_template.html"), encoding="utf-8") as f:
        tmpl = f.read()
    with open(os.path.join(HERE, "quad.html"), "w", encoding="utf-8") as f:
        f.write(tmpl.replace("__GAME_JSON__", json.dumps(game, ensure_ascii=False)).replace("__ADMIN_JSON__", "null"))
    pz = game["puzzle"]
    print("%s: %s %s, %s quadrant" % (date, pz["source"], pz["number"], pz["quadrant"]))
    for e in game["entries"]:
        print("  %-10s %-16s %-26s crosses=%d%s" % (e["label"], e["answer"], e["operation_label"],
                                                   e["crosses"], "   <- LAST" if e["is_last"] else ""))
    for t in tried:
        print("  passed over:", t)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat())
