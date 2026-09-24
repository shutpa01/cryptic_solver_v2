"""Draft the clue page's prose block — overnight, for review.

    python scripts/draft_prose.py                      # today's served puzzles
    python scripts/draft_prose.py --source telegraph --puzzle 31353
    python scripts/draft_prose.py --list               # what is waiting for approval
    python scripts/draft_prose.py --dry-run            # show the facts, call nothing

WHAT AND WHY
------------
A wordhint-style paragraph on each clue page: one sentence explaining the answer,
and a short gloss of the answer word. Today forward, no backfill — "we are not
interested in working on clues that nobody will ever look at again" (user).

It is NOT the narration. `scripts/narrate_clue.py` is a spoken script that walks
the mechanism step by step and MUST NOT be edited. This is compressed prose, and
the decision of 2026-09-24 is that Cordelia narrates this text as written — one
text for the page and the video. That is why the writing rules below ban
parentheses and mid-sentence dashes: "SENT (posted)" stumbles when read aloud.

THE FACTS ARE THE RECORD'S OWN WORDS
------------------------------------
The model is given `web.wfw_read`'s strings — the same summary, definition and
clue-type the card shows — plus every piece as text -> value. It is told those are
the only facts. It is not given the reference DB, the blogs, or its own guesses
about the mechanism.

WHY BATCHED PER PUZZLE. `draft_senses.py` calls the CLI once per clue, which is
the wrong shape at 60-130 clues a day. A puzzle's clues go in ONE call. Measured
2026-09-24 on telegraph 31353: 32/32 clues returned, 0 with an invented piece.

FILED UNAPPROVED. Nothing here reaches a page. Every draft is written with
`"approved": false` and the serving path must show only approved text, exactly as
the narrator speaks only approved senses. The nightly runs at 00:05, BEFORE the
user has approved the readings (`core/prefill_commit.py:57` — a prefill is
`pending`, never `pass`), so a draft is always written against a provisional
parse. The user's tick is the gate that makes that safe.

THE CHECK IS MECHANICAL, AND IT IS THE POINT
--------------------------------------------
A draft is REJECTED, not filed, when it invents. `verify()` holds two rules:

  * every value the record names must appear in the prose — nothing silently
    dropped;
  * every capitalised token in the prose must be a value the record names, the
    answer, or a word from the clue — nothing invented.

The second rule is what stops the failure this feature was nearly shipped with.
Fed the old one-line summary for a homophone ('"net" sounds like -> ERNE') a model
writes "net, said aloud, gives the answer" and means it. That line was false: the
middle, EARN, is recorded on every link of the piece and was simply not being
read. `_summary` now names it, so the facts handed over are true — but the check
stays, because the next lossy string will not announce itself either.

Model pinned for the same reason nightly_run pins it: with no flag the CLI
inherits whatever the user last chose interactively, and an unattended job must
not follow that.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Clue text carries curly apostrophes and drafts carry dashes; under a redirect
# stdout takes the locale encoding (cp1252) and printing one raises. That is how
# every on-demand prefill between 2026-08-30 and 2026-09-10 died AFTER doing its
# work. Same guard as scripts/run_prefill.py.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from core import prose_store                                      # noqa: E402

CLAUDE = Path.home() / ".local" / "bin" / "claude.exe"
CLAUDE_MODEL = "claude-fable-5"

PROMPT_HEAD = """You are writing the explanation paragraph for a cryptic crossword \
answer page. Below are several clues from one puzzle. For EACH, write:

SENTENCE: one sentence saying how the wordplay produces the answer.
GLOSS: a short plain definition of the answer word itself, dictionary-style, \
eight to fifteen words.

RULES — these are absolute:
* The FACTS block under each clue is the only thing you know. Do not add a step, \
a synonym, an abbreviation or a mechanism that is not written there.
* Never name a word in capitals unless it appears in the FACTS block or the clue.
* If the facts do not explain the answer, write SENTENCE: INSUFFICIENT and nothing \
else for that clue. That is a correct answer, not a failure.
* This text is read ALOUD as well as printed. No parentheses, no dashes in the \
middle of a sentence, no semicolons. Plain British English.
* Do not mention the grid, the setter, the puzzle or the solver.

Output format, exactly, and nothing else:

===<clue_id>
SENTENCE: <one sentence>
GLOSS: <short definition>

"""


def facts(parse, clue_text, answer):
    """The record's own words, and nothing else — what the model is allowed to know."""
    from web import wfw_read
    lines = ['CLUE: "%s"' % (clue_text or "").strip(),
             "ANSWER: %s" % (answer or "").upper()]
    label = wfw_read._wordplay_label(parse)
    if label:
        lines.append("CLUE TYPE: %s" % label)
    d = wfw_read._definition(parse)
    if d:
        lines.append("DEFINITION IN THE CLUE: %s" % d)
    s = wfw_read._summary(parse)
    if s:
        lines.append("HOW IT WORKS: %s" % s)
    for p in parse["sources"]:
        if p["mechanism"] == "definition":
            continue
        line = '  piece: "%s" -> %s  [%s]' % ((p["text"] or "").strip(),
                                              (p["value"] or "").upper(),
                                              p["mechanism"] or "?")
        if p["mechanism"] == "homophone":
            snd = wfw_read._homophone_sound(parse, p)
            if snd:
                line += '  (pronounced as "%s")' % snd.upper()
        lines.append(line)
    for i in parse["indicators"]:
        lines.append('  signal: "%s" means %s' % ((i["text"] or "").strip(),
                                                  (i["note"] or "?").strip()))
    return "\n".join(lines)


def facts_hash(block):
    """Fingerprint of the record the prose is written from.

    Stored with the draft so a later commit can ask one question: is this the same
    reading? The user accepts most prefills unchanged, so usually it is, and the
    commit then skips the call entirely instead of spending 8-11 seconds
    re-deriving the same paragraph.
    """
    return hashlib.sha1(block.encode("utf-8", "replace")).hexdigest()


def recorded_values(parse):
    """Every value the record names, uppercased — what the prose must account for."""
    out = set()
    for p in parse["sources"]:
        v = (p["value"] or "").strip().upper()
        if v and p["mechanism"] != "definition":
            out.add(v)
    return out


_CAPS = re.compile(r"\b[A-Z][A-Z'-]{1,}\b")


def verify(text, parse, clue_text, answer):
    """(ok, reason). A draft that invents or drops a piece is REFUSED, never filed."""
    up = text.upper()
    ans = (answer or "").upper()
    values = recorded_values(parse)
    missing = [v for v in values if v.replace(" ", "") not in up.replace(" ", "")]
    if missing:
        return False, "does not account for %s" % ", ".join(sorted(missing))
    # Anything shouted in capitals must be something we actually hold.
    allowed = {ans, ans.replace(" ", "")} | values
    for p in parse["sources"]:
        allowed.add((p["text"] or "").strip().upper())
    from web import wfw_read
    for p in parse["sources"]:
        if p["mechanism"] == "homophone":
            snd = wfw_read._homophone_sound(parse, p)
            if snd:
                allowed.add(snd.upper())
    allowed |= {w.upper() for w in re.split(r"[^A-Za-z']+", clue_text or "") if w}
    allowed |= {"SENTENCE", "GLOSS", "INSUFFICIENT", "A", "I"}
    flat = {a.replace(" ", "") for a in allowed}
    for tok in _CAPS.findall(text):
        if tok not in allowed and tok.replace(" ", "") not in flat:
            return False, "invents %s" % tok
    return True, ""


def clues_in_scope(source=None, puzzle=None, day=None, clue=None, pending=False):
    """[(clue_id, answer, clue_text, parse)] for every served clue with a parse.

    A clue with NO parse is skipped, not drafted: `wfw_read._load` returns a parse
    only when the stored solve is a `pass`, so a pending prefill is invisible here.
    That is the point — there is nothing truthful to say about a reading that has
    not been settled.
    """
    from web import create_app, wfw_read
    app = create_app("development")
    found = []
    with app.app_context():
        from web.db import get_db
        from web.serving import SERVED_SOURCES
        if clue:
            rows = get_db().execute(
                "SELECT id, answer, clue_text FROM clues WHERE id = ?",
                (int(clue),)).fetchall()
        elif source and puzzle:
            rows = get_db().execute(
                "SELECT id, answer, clue_text FROM clues WHERE source = ? "
                "AND puzzle_number = ? ORDER BY id", (source, str(puzzle))).fetchall()
        else:
            ph = ",".join("?" for _ in SERVED_SOURCES)
            rows = get_db().execute(
                "SELECT id, answer, clue_text FROM clues WHERE source IN (%s) "
                "AND publication_date = COALESCE(?, date('now')) ORDER BY id" % ph,
                (*SERVED_SOURCES, day)).fetchall()
        for r in rows:
            parse = wfw_read._load(r["id"], allow_pending=pending)
            if parse is None:
                continue            # nothing settled enough to describe
            found.append((r["id"], r["answer"] or "", r["clue_text"] or "", parse))
    return found


def ask(block):
    """The model's raw reply for one puzzle's clues, or "" — never raises."""
    # Strip any inherited ANTHROPIC_API_KEY so claude bills the SUBSCRIPTION, never
    # prepaid API credits. This file loads the key without meaning to: create_app
    # imports web/config.py, which load_dotenv()s the project .env. claude.exe
    # prefers an API key over the Max login, and on 2026-09-21 that silently spent
    # the user's balance overnight. NEVER remove the key from .env; strip it here.
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    try:
        r = subprocess.run([str(CLAUDE), "-p", block, "--model", CLAUDE_MODEL],
                           capture_output=True, text=True, timeout=600,
                           encoding="utf-8", errors="replace",
                           stdin=subprocess.DEVNULL, env=env)
    except Exception as e:
        print("   call failed: %r" % (e,))
        return ""
    if r.returncode != 0:
        print("   call returned %d: %s" % (r.returncode, (r.stderr or "").strip()[:200]))
        return ""
    return r.stdout or ""


def parse_reply(reply):
    """{clue_id: (sentence, gloss)} from the model's block output."""
    out, cid, sent, gloss = {}, None, None, None
    for line in (reply or "").splitlines():
        line = line.strip()
        if line.startswith("==="):
            if cid and sent:
                out[cid] = (sent, gloss or "")
            cid, sent, gloss = line[3:].strip(), None, None
        elif line.upper().startswith("SENTENCE:"):
            sent = line.split(":", 1)[1].strip()
        elif line.upper().startswith("GLOSS:"):
            gloss = line.split(":", 1)[1].strip()
    if cid and sent:
        out[cid] = (sent, gloss or "")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source")
    ap.add_argument("--puzzle")
    ap.add_argument("--day", help="publication date; default today")
    ap.add_argument("--clue", type=int,
                    help="one clue id — what the /hs Commit button fires")
    ap.add_argument("--list", action="store_true", help="show what awaits approval")
    ap.add_argument("--pending", action="store_true",
                    help="also draft PENDING prefill readings — what the nightly "
                         "uses, so the prose is waiting beside them at 05:00")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the facts block and call nothing")
    args = ap.parse_args(argv)

    data = prose_store.load()
    if args.list:
        pending = {k: v for k, v in data.items()
                   if isinstance(v, dict) and not v.get("approved")}
        if not pending:
            print("Nothing awaiting approval.")
            return 0
        print("%d draft(s) awaiting approval:" % len(pending))
        for k, v in sorted(pending.items()):
            print("  %-10s %s" % (k, v.get("sentence", "")))
            if v.get("gloss"):
                print("  %-10s %s" % ("", v["gloss"]))
        return 0

    clues = clues_in_scope(args.source, args.puzzle, args.day, args.clue,
                           pending=args.pending)
    # A clue is drafted when it has NO draft, or when the reading has CHANGED since
    # the draft was written. An unchanged reading is skipped even though the user
    # has just committed it: the prose already in the box was written from exactly
    # these facts, so re-deriving it would spend a call to produce the same
    # paragraph. This is what makes drafting at the end of the nightly cheap — the
    # user accepts most prefill readings unchanged, and those cost nothing at 05:00.
    todo, blocks, hashes = [], [], {}
    for cid, ans, clue, parse in clues:
        block = facts(parse, clue, ans)
        h = facts_hash(block)
        if prose_store.facts_unchanged(cid, h, data):
            continue
        todo.append((cid, ans, clue, parse))
        blocks.append(block)
        hashes[cid] = h
    print("%d clue(s) with a parse in scope, %d to draft." % (len(clues), len(todo)))
    if not todo:
        print("Nothing changed — the prose already matches every reading.")
        return 0

    body = "\n\n".join("===%s\n%s" % (cid, b)
                       for (cid, _a, _c, _p), b in zip(todo, blocks))
    if args.dry_run:
        print(PROMPT_HEAD + body)
        return 0

    got = parse_reply(ask(PROMPT_HEAD + body))
    print("%d clue(s) came back." % len(got))
    filed = refused = 0
    for cid, ans, clue, parse in todo:
        pair = got.get(str(cid))
        if not pair:
            print("  %-10s no draft returned" % cid)
            continue
        sentence, gloss = pair
        if sentence.strip().upper().startswith("INSUFFICIENT"):
            print("  %-10s model declined — facts do not explain the answer" % cid)
            continue
        ok, why = verify(sentence + " " + gloss, parse, clue, ans)
        if not ok:
            refused += 1
            print("  %-10s REFUSED: %s" % (cid, why))
            print("             %s" % sentence)
            continue
        # save_draft refuses to overwrite a record the user has already ticked, so a
        # re-run cannot undo an approval.
        if prose_store.save_draft(cid, sentence, gloss, ans, hashes.get(cid, "")):
            filed += 1
            print("  %-10s %s" % (cid, sentence))
        else:
            print("  %-10s already approved — left alone" % cid)
    print("\n%d filed unapproved, %d refused." % (filed, refused))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
