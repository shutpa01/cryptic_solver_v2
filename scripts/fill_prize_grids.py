"""Fill answerless prize grids overnight, so the cascade and prefill can run on them.

User, 2026-10-11: "change the puzzle prize routine so that you fill the grids and then
run the cascade and prefill as normal so that they are ready for me to check when I get
up at 5am". Prize puzzles (Telegraph Prize Cryptic / Prize Toughie, Observer Everyman)
are scraped WITHOUT answers, so the cascade skipped them and the user solved them by
hand in the morning before anything else could run.

For each of the day's serving-paper puzzles that has blank answers:
  1. Read its grid from the scraped JSON (scraper/<source>/*.json, meta.number): the
     cells of every light, so crossings are known, and `solution_hashed` — the md5 of
     the full solution (15 rows of letters, a space for each black square; checked
     against the published previous_solution of Everyman 4172).
  2. Ask headless Claude for every answer (`claude -p`, subscription billing — the
     ANTHROPIC_API_KEY is popped, as at every claude.exe call site).
  3. Check. Every answer must fit its enumeration and agree with every crossing
     answer. If the assembled grid matches the published hash, the whole fill is
     PROVEN and written. If not, Claude is asked once more with the clashes listed;
     after that only answers whose every crossing agrees are written — an answer
     in doubt stays blank, and the cascade skips it as before.
  4. Write blanks only (never overwrite an answer, e.g. one the user typed), to the
     serving row and to any duplicate copy of the puzzle under another source
     (the Prize Toughie is also filed as source='telegraph-toughie').

Usage:
    python -m scripts.fill_prize_grids                       # today's puzzles
    python -m scripts.fill_prize_grids --date 2026-10-11
    python -m scripts.fill_prize_grids --source telegraph --pnum 246
    python -m scripts.fill_prize_grids --check-only ...     # verify stored answers, no Claude
    python -m scripts.fill_prize_grids --dry-run ...        # solve + check, write nothing
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import time
from datetime import date
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = ROOT / "data" / "clues_master.db"
CLAUDE_EXE = r"C:\Users\shute\.local\bin\claude.exe"
CLAUDE_MODEL = "claude-fable-5"          # same pin as scripts/nightly_run.py
SERVE_SOURCES = ["telegraph", "times", "guardian"]
# Other sources a serving puzzle is also filed under (same clues, same numbering).
DUPLICATE_SOURCES = {"telegraph": ["telegraph-toughie"]}


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def letters(s):
    return re.sub(r"[^A-Z]", "", (s or "").upper())


# --------------------------------------------------------------------------- grid

def find_json(source, pnum):
    """Newest scraped JSON for this puzzle (meta.number == pnum), or None."""
    best = None
    for f in glob.glob(str(ROOT / "scraper" / source.split("-")[0] / "*.json")):
        try:
            j = json.load(open(f, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        j = j.get("json") or j.get("data") or j     # Times files wrap it in "data"
        if not isinstance(j, dict) or str((j.get("meta") or {}).get("number")) != str(pnum):
            continue
        if "words" not in (j.get("copy") or {}):
            continue
        if best is None or os.path.getmtime(f) > best[0]:
            best = (os.path.getmtime(f), j)
    return best[1] if best else None


def load_grid(j):
    """{(number, 'across'|'down'): {cells, length, clue, format}}, rows, cols, hash."""
    copy = j["copy"]
    words = {w["id"]: w for w in copy["words"]}

    def cells(w):
        xs = [int(v) for v in str(w["x"]).split("-")]
        ys = [int(v) for v in str(w["y"]).split("-")]
        if len(xs) == 2:
            return [(ys[0], x) for x in range(xs[0], xs[1] + 1)]
        if len(ys) == 2:
            return [(y, xs[0]) for y in range(ys[0], ys[1] + 1)]
        return [(ys[0], xs[0])]

    lights = {}
    for grp in copy["clues"]:
        d = "across" if grp["title"].lower().startswith("across") else "down"
        for c in grp["clues"]:
            lights[(str(c["number"]), d)] = {
                "cells": cells(words[c["word"]]), "clue": unescape(c.get("clue") or ""),
                "format": c.get("format") or ""}
    size = copy.get("gridsize") or {}
    return (lights, int(size.get("rows") or 15), int(size.get("cols") or 15),
            (copy.get("settings") or {}).get("solution_hashed"))


def place(lights, answers):
    """Fill the grid from {(num, dir): ANSWER}; return (grid, clashes)."""
    grid, owner, clashes = {}, {}, []
    for key, ans in answers.items():
        if key not in lights or len(ans) != len(lights[key]["cells"]):
            continue
        for cell, ch in zip(lights[key]["cells"], ans):
            if cell in grid and grid[cell] != ch:
                clashes.append((owner[cell], key, cell, grid[cell], ch))
            else:
                grid[cell], owner[cell] = ch, key
    return grid, clashes


def solution_hash(grid, rows, cols):
    s = "".join(grid.get((r, c), " ") for r in range(1, rows + 1) for c in range(1, cols + 1))
    return hashlib.md5(s.encode()).hexdigest()


def consistent(lights, answers):
    """Answers that fit their light and agree with every crossing answer present."""
    grid, clashes = place(lights, answers)
    bad = {k for a, b, *_ in clashes for k in (a, b)}
    return {k: v for k, v in answers.items()
            if k in lights and len(v) == len(lights[k]["cells"]) and k not in bad}


# --------------------------------------------------------------------------- Claude

def ask_claude(source, pnum, lights, known, feedback=""):
    """Every answer from headless Claude, as {(num, dir): LETTERS}."""
    cross = {}
    for key, L in lights.items():
        for i, cell in enumerate(L["cells"]):
            cross.setdefault(cell, []).append((key, i + 1))
    lines = []
    for key in sorted(lights, key=lambda k: (k[1], int(re.sub(r"\D", "", k[0]) or 0))):
        L = lights[key]
        xs = ["letter %d = %s%s letter %d" % (i, o[0], o[1][0], j)
              for cell in L["cells"] for (o, j) in cross[cell] if o != key
              for i in [L["cells"].index(cell) + 1]]
        given = "  [ANSWER GIVEN: %s]" % known[key] if key in known else ""
        lines.append("%s%s  %s (%s)%s\n    crossings: %s"
                     % (key[0], key[1][0], L["clue"], L["format"], given, "; ".join(xs) or "none"))
    prompt = (
        "Solve this cryptic crossword (%s %s). Clues, enumerations and how the lights "
        "cross are below; answers marked GIVEN are already confirmed. Solve every clue, "
        "check every crossing letter agrees, and fix any clue whose letters clash.\n\n%s\n%s\n"
        "Reply with ONLY one JSON object mapping each clue to its answer in capital "
        "letters, no spaces or hyphens, e.g. {\"1a\": \"ESTHER\", \"2d\": \"STRESSANDSTRAIN\"}. "
        "No other text." % (source, pnum, "\n".join(lines), feedback))
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)   # bill the subscription, never API credits
    r = subprocess.run([CLAUDE_EXE, "-p", prompt, "--model", CLAUDE_MODEL],
                       cwd=tempfile.gettempdir(),   # no project context: a puzzle, not the repo
                       stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=2400, env=env)
    if r.returncode != 0:
        raise RuntimeError("claude exit %s: %s" % (r.returncode, (r.stderr or r.stdout)[-300:]))
    m = re.search(r"\{.*\}", r.stdout or "", re.S)
    if not m:
        raise RuntimeError("no JSON in reply: %s" % (r.stdout or "")[-300:])
    out = {}
    for k, v in json.loads(m.group(0)).items():
        km = re.fullmatch(r"\s*(\d+)\s*([ad])\w*\s*", str(k).lower())
        if km:
            out[(km.group(1), "across" if km.group(2) == "a" else "down")] = letters(v)
    return out


# --------------------------------------------------------------------------- puzzles

def puzzles_to_fill(con, target_date, source=None, pnum=None):
    if source and pnum:
        return [(source, str(pnum))]
    ph = ",".join("?" * len(SERVE_SOURCES))
    return con.execute(
        "SELECT DISTINCT source, puzzle_number FROM clues WHERE publication_date = ? "
        "AND source IN (%s) AND (answer IS NULL OR answer = '') ORDER BY source, puzzle_number"
        % ph, (target_date, *SERVE_SOURCES)).fetchall()


def fill_one(con, source, pnum, check_only=False, dry_run=False, ignore_stored=False):
    rows = con.execute("SELECT id, clue_number, direction, enumeration, answer FROM clues "
                       "WHERE source = ? AND puzzle_number = ?", (source, pnum)).fetchall()
    if ignore_stored:                        # test mode: solve as if every answer were blank
        rows = [(cid, n, d, e, "") for cid, n, d, e, _ in rows]
    j = find_json(source, pnum)
    if not j:
        log("  %s %s: no scraped JSON with its grid - left for the morning" % (source, pnum))
        return
    lights, nrows, ncols, target_hash = load_grid(j)
    known = {(str(n), d): letters(a) for _, n, d, _, a in rows if letters(a)}
    log("  %s %s: %d clues, %d already answered, grid %dx%d, hash %s"
        % (source, pnum, len(rows), len(known), nrows, ncols, "yes" if target_hash else "NO"))

    def proven(ans):
        grid, clashes = place(lights, ans)
        return (not clashes and target_hash and len(ans) == len(lights)
                and solution_hash(grid, nrows, ncols) == target_hash)

    if check_only:
        log("    stored answers %s the published hash" % ("MATCH" if proven(known) else "do NOT match"))
        return
    if len(known) == len(lights):
        log("    nothing blank")
        return

    answers = {**ask_claude(source, pnum, lights, known), **known}
    if not proven(answers):
        _, clashes = place(lights, answers)
        wrong = sorted({k for a, b, *_ in clashes for k in (a, b)} |
                       {k for k in lights if len(answers.get(k, "")) != len(lights[k]["cells"])})
        log("    first fill not proven (%d clashes, %d bad/missing) - asking again" % (
            len(clashes), len(wrong)))
        fb = ("\nYOUR PREVIOUS ANSWERS: %s\nThese clash or do not fit and must be "
              "re-solved: %s. %s" % (
                  json.dumps({k[0] + k[1][0]: v for k, v in answers.items()}),
                  ", ".join(k[0] + k[1][0] for k in wrong) or "none listed",
                  "The grid also fails the published solution check, so at least one "
                  "answer that fits its crossers is still wrong." if target_hash else ""))
        answers = {**ask_claude(source, pnum, lights, known, fb), **known}

    ok = proven(answers)
    to_write = answers if ok else consistent(lights, answers)
    log("    %s: %d of %d answers usable" % (
        "PROVEN by the published hash" if ok else
        ("NOT proven (hash mismatch)" if target_hash else "no hash to prove it"),
        len(to_write), len(lights)))
    if dry_run:
        for k in sorted(to_write, key=lambda k: (k[1], int(k[0]))):
            log("    [dry run] %s%s %s" % (k[0], k[1][0], to_write[k]))
        return

    n = 0
    for src in [source] + DUPLICATE_SOURCES.get(source, []):
        for cid, num, d, enum, ans in con.execute(
                "SELECT id, clue_number, direction, enumeration, answer FROM clues "
                "WHERE source = ? AND puzzle_number = ?", (src, pnum)).fetchall():
            new = to_write.get((str(num), d))
            if not new or letters(ans):
                continue
            if sum(int(x) for x in re.findall(r"\d+", enum or "")) != len(new):
                continue                      # enumeration disagrees with the grid - leave it
            con.execute("UPDATE clues SET answer = ? WHERE id = ? AND (answer IS NULL OR answer = '')",
                        (new, cid))
            n += 1
    con.commit()
    left = [k[0] + k[1][0] for k in lights if k not in to_write]
    log("    wrote %d answers%s" % (n, ("; left blank: " + ", ".join(left)) if left else ""))


def main():
    ap = argparse.ArgumentParser(description="Fill answerless prize grids with Claude")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--source")
    ap.add_argument("--pnum")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ignore-stored", action="store_true",
                    help="test: solve as if all answers were blank (needs --dry-run)")
    a = ap.parse_args()
    if a.ignore_stored and not a.dry_run:
        ap.error("--ignore-stored only runs with --dry-run")
    con = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        todo = puzzles_to_fill(con, a.date, a.source, a.pnum)
        log("Prize fill %s: %d puzzle(s) with blank answers" % (a.date, len(todo)))
        for source, pnum in todo:
            try:
                fill_one(con, source, str(pnum), a.check_only, a.dry_run, a.ignore_stored)
            except Exception as e:      # one puzzle failing never stops the next
                log("  %s %s: ERROR %s - left for the morning" % (source, pnum, e))
    finally:
        con.close()


if __name__ == "__main__":
    main()
