"""Draft the usage phrases a double-definition Short needs — overnight, for review.

    python scripts/draft_senses.py                     # today's served puzzles
    python scripts/draft_senses.py --source telegraph --puzzle 31338
    python scripts/draft_senses.py --list              # what is waiting for approval

WHAT AND WHY
------------
A double definition has no wordplay to take apart, so the only thing worth saying is
what the two senses MEAN — "tolerate, as in I couldn't stick it any longer" against
"criticism, as in he got a lot of stick from his mates". Without those the narration is
"tolerate gives you STICK, criticism gives you STICK", which teaches nothing (user,
2026-09-07: "A=A and A=A!").

Nothing we hold records how a word is used in a sentence, so the phrase has to be
written. This drafts it with the LLM and files it as UNAPPROVED. The narrator speaks
only approved entries, so a draft cannot reach the voice by accident.

WHY OVERNIGHT
-------------
Publishing is not the moment to be writing copy: "we do not have time to do rework when
we are publishing" (user). The nightly run knows the day's puzzles hours ahead, so the
phrases are drafted and waiting; at publish time approval is a glance, not a task.

Model pinned for the same reason nightly_run pins it: with no flag the CLI inherits
whatever the user last chose interactively, and an unattended job must not follow that.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
SENSES = ROOT / "logs" / "senses.json"
CLAUDE = Path.home() / ".local" / "bin" / "claude.exe"
CLAUDE_MODEL = "claude-fable-5"

PROMPT = """The cryptic crossword answer is {answer}. It is a double definition clue: \
"{clue}"
The two definitions are {halves}.

For EACH definition, give ONE short, natural, everyday British English phrase showing \
{answer} used in that exact sense. Six to eight words. No explanation, no preamble.

Output exactly {n} lines, in this format and nothing else:
<definition>|<phrase>
"""


def load():
    try:
        return json.loads(SENSES.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save(data):
    SENSES.parent.mkdir(parents=True, exist_ok=True)
    SENSES.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def key(answer, half):
    return "%s|%s" % (answer.upper(), half.lower())


def dd_clues(source=None, puzzle=None, day=None):
    """[(clue_id, answer, clue_text, [halves])] for every double definition in scope."""
    from web import create_app, wfw_read
    app = create_app("development")
    found = []
    with app.app_context():
        from web.db import get_db
        from web.serving import SERVED_SOURCES
        if source and puzzle:
            sql = ("SELECT id, answer, clue_text FROM clues WHERE source = ? "
                   "AND puzzle_number = ?")
            rows = get_db().execute(sql, (source, str(puzzle))).fetchall()
        else:
            ph = ",".join("?" for _ in SERVED_SOURCES)
            sql = ("SELECT id, answer, clue_text FROM clues WHERE source IN (%s) "
                   "AND publication_date = COALESCE(?, date('now'))" % ph)
            rows = get_db().execute(sql, (*SERVED_SOURCES, day)).fetchall()
        for r in rows:
            parse = wfw_read._load(r["id"])
            if parse is None or parse["operation"] not in ("dd", "double_definition"):
                continue
            halves = [(d["text"] or "").strip() for d in parse["definitions"]
                      if (d["text"] or "").strip()]
            halves += [(s["text"] or "").strip() for s in parse["sources"]
                       if s["mechanism"] == "definition" and (s["text"] or "").strip()
                       and (s["text"] or "").strip() not in halves]
            if len(halves) >= 2:
                found.append((r["id"], r["answer"] or "", r["clue_text"] or "", halves))
    return found


def draft(answer, clue, halves):
    """{half: phrase} from the LLM, or {} when it cannot be parsed. Never raises."""
    prompt = PROMPT.format(answer=answer.upper(), clue=clue, n=len(halves),
                           halves=" and ".join('"%s"' % h for h in halves))
    try:
        r = subprocess.run([str(CLAUDE), "-p", prompt, "--model", CLAUDE_MODEL],
                           capture_output=True, text=True, timeout=180,
                           encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    except Exception as e:
        print("  LLM call failed: %s" % e)
        return {}
    out = {}
    for line in (r.stdout or "").splitlines():
        if "|" not in line:
            continue
        half, phrase = line.split("|", 1)
        half, phrase = half.strip(), phrase.strip().rstrip(".")
        # Only accept a line naming a definition we actually asked about — the model
        # inventing a third sense must not silently become a spoken fact.
        for h in halves:
            if half.lower() == h.lower() and phrase:
                out[h] = phrase
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source")
    ap.add_argument("--puzzle")
    ap.add_argument("--day", help="publication date; default today")
    ap.add_argument("--list", action="store_true", help="show what awaits approval")
    args = ap.parse_args(argv)

    data = load()
    if args.list:
        pending = {k: v for k, v in data.items()
                   if isinstance(v, dict) and not v.get("approved")}
        if not pending:
            print("Nothing awaiting approval.")
            return 0
        print("%d phrase(s) awaiting approval:" % len(pending))
        for k, v in sorted(pending.items()):
            print("  %-28s %s" % (k, v.get("phrase", "")))
        return 0

    clues = dd_clues(args.source, args.puzzle, args.day)
    if not clues:
        print("No double definitions in scope — nothing to draft.")
        return 0
    print("%d double definition(s) in scope." % len(clues))
    added = 0
    for cid, answer, clue, halves in clues:
        missing = [h for h in halves if key(answer, h) not in data]
        if not missing:
            print("  %s (%s) — already filed" % (answer.upper(), cid))
            continue
        print("  %s (%s): drafting %d" % (answer.upper(), cid, len(missing)))
        got = draft(answer, clue, halves)
        for h in missing:
            if h in got:
                data[key(answer, h)] = {"phrase": got[h], "approved": False,
                                        "clue_id": cid}
                added += 1
                print("      %-14s %s" % (h, got[h]))
            else:
                print("      %-14s (no usable draft — left for a human)" % h)
    if added:
        save(data)
    print("%d drafted, awaiting your approval." % added)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
