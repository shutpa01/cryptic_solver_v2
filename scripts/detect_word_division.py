"""WORD DIVISION candidates, laid out for a reader — read-only, judges nothing.

Built 2026-09-24, after telegraph 31353 11a was served as a pass:

    "Russian leader is installed"   PUTIN (5)
      def  "Russian leader"
      def  "installed"              <- the DD solver's second definition

No row anywhere backs `installed -> PUTIN`. The real device is a WORD DIVISION:
one clue word supplies a TWO-WORD value and the answer is that phrase with the
space closed up — installed = PUT IN -> PUTIN. The label for it exists (a spaced
value in a single-word answer badges "Word division"); what was missing was any
way to NOTICE one that nobody had filed by hand.

WHAT THIS IS NOT. It is not an engine and must never become one. It splits the
answer at every point and asks the reference DB one question — is this spaced
phrase attested? — then prints what the phrase means so a reader can decide
whether a word in the clue means it. It cannot tell a word division from an
ordinary charade, because the difference is in the setter's intention, not in the
letters: LEGIT is LEG IT ("run") in one clue and LEG + IT ("joint" + "just the
thing") in another, and the split looks identical in both. It narrows a list for
a human. It does not classify.

    telegraph 31353 11a  PUTIN (5)  [double_definition/dd]
      "Russian leader is installed"
      splits as PUT IN
        the phrase means   placed, interpolate, plant, inset, laid

THE ENUMERATION IS THE GATE. A word division has a single-number enumeration,
PUTIN (5). An answer stored solid but enumerated (6,4) or (1-5) is a multiword
ANSWER — FAMILYTREE, ALEVEL, TRIPLESEC — and splitting it discovers nothing. Both
look like one word in `clues.answer`; only `clues.enumeration` separates them, so
that column, not the answer, decides. (A handful of clue rows carry a wrong
enumeration — FAMILYTREE as "10" — and those will still get through. A reader
sees it at once; no automatic action rests on this.)

THE SPACE MUST STAY SIGNIFICANT. Lookups key on `norm_word`, which is what the
solver reads (signature_solver.db._normalize_key) and which collapses hyphens and
case but KEEPS the space — so "put in" and "pu tin" remain different keys. Keying
on LOWER(word) instead both misses 39k rows and costs a full scan of 1.35M: the
norm_word and `synonym COLLATE NOCASE` indexes turn a two-minute sweep into one
second.

WHAT IT DOES NOT DO. It writes nothing, demotes nothing and touches no verdict.
In the nightly pass review the only write remains `pass_review.demote_to_pending`,
and the reviewer still has to name the claim that fails. What this adds is the
suggestion that belongs beside the doubt: not "this reading looks wrong" but
"PUTIN (5) splits as PUT IN, an attested phrase meaning placed, inset — is that
the wordplay?"

    python scripts/detect_word_division.py --days 1
    python scripts/detect_word_division.py --days 30 --all
    python scripts/detect_word_division.py --clue 10094461
"""

import argparse
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Clue text carries curly apostrophes and this tool's own output carries em dashes.
# Redirected to a log file, stdout takes the locale encoding (cp1252) and printing
# one raises UnicodeEncodeError — which is exactly how every on-demand prefill
# between 2026-08-30 and 2026-09-10 died AFTER doing its work. Same guard as
# scripts/run_prefill.py.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):      # already-wrapped or closed stream
        pass

from signature_solver.db import _normalize_key                    # noqa: E402

CLUES_DB = ROOT / "data" / "clues_master.db"
REF_DB = ROOT / "data" / "cryptic_new.db"

# Solvers whose output is not an engine's claim: the user's own filings, and
# prefills that are pending by construction. Same exclusion as
# scripts/audit_engine_passes.py, so the two tools read the same population.
NOT_ENGINE = ("manual", "prefill")

# A comma or any dash in the enumeration means the answer is more than one word.
_MULTIWORD_ENUM = re.compile(r"[,‐-―-]")


def single_word_enumeration(enumeration):
    """True when the enumeration names ONE word — the answer PUTIN (5) shape.

    Empty/absent enumeration is not a yes: without it we cannot tell PUTIN (5)
    from FAMILYTREE (6,4), and guessing from the answer is what puts solid
    multiword answers in the list.
    """
    e = (enumeration or "").strip()
    return bool(e) and not _MULTIWORD_ENUM.search(e)


def attestation(ref, phrase):
    """What the reference DB knows about this spaced phrase, both directions.

    `means`    — the phrase is a headword: these are its synonyms.
    `meant_by` — the phrase is a value: these are words that mean it. This is the
                 direction that matters most, because a word division needs a CLUE
                 word that means the phrase.
    """
    means = [r[0] for r in ref.execute(
        "SELECT DISTINCT synonym FROM synonyms_pairs WHERE norm_word = ? LIMIT 12",
        (_normalize_key(phrase),))]
    meant_by = [r[0] for r in ref.execute(
        "SELECT DISTINCT word FROM synonyms_pairs WHERE synonym = ? COLLATE NOCASE "
        "LIMIT 12", (phrase,))]
    return means, meant_by


def clue_words(clue_text):
    """The clue's words as lookup keys, plus every adjacent pair — a word division
    can be clued by a phrase ("put off"), not only a single word."""
    words = [w for w in re.split(r"[^A-Za-z']+", clue_text or "") if w]
    keys = [_normalize_key(w) for w in words]
    keys += [_normalize_key("%s %s" % (a, b)) for a, b in zip(words, words[1:])]
    return [k for k in keys if k]


def clue_word_meaning(ref, clue_text, phrase):
    """Clue words that the DB says mean this phrase.

    When one is found and it is NOT the definition, the reading is nearly written
    for you. When none is, the candidate still stands: `installed -> put in` is
    exactly the row that was missing on 31353 11a, which is WHY the engine got the
    clue wrong.
    """
    hits = []
    for key in dict.fromkeys(clue_words(clue_text)):
        row = ref.execute(
            "SELECT 1 FROM synonyms_pairs WHERE norm_word = ? "
            "  AND synonym = ? COLLATE NOCASE LIMIT 1", (key, phrase)).fetchone()
        if row:
            hits.append(key)
    return hits


def definition_keys(conn, clue_id):
    """The definition spans this pass recorded, as lookup keys.

    A word division is clued by WORDPLAY, so the word that supplies the spaced
    phrase is never the definition. When the only clue word meaning the phrase IS
    the definition, the split is circular — it says the answer means what the
    answer means — and every false positive measured over 30 days had exactly that
    shape: ALFRESCO/"outside", LOGJAM/"deadlock", TAGLINE/"slogan", ABOARD/"on a
    boat", all ordinary charades. So this is reported, loudly, as evidence AGAINST.
    """
    return {_normalize_key(r[0]) for r in conn.execute(
        "SELECT text FROM wfw_piece WHERE clue_id = ? AND role = 'definition'",
        (clue_id,)) if _normalize_key(r[0] or "")}


def splits(answer):
    """Every two-part split of the answer's letters, longest-looking first is not
    a judgement — order is left to right and every split is shown."""
    a = "".join(c for c in (answer or "").upper() if c.isalpha())
    return [("%s %s" % (a[:i], a[i:]), a) for i in range(1, len(a))]


def rows(conn, days, clue_id, engine_only):
    where = ["s.status = 'pass'"]
    params = []
    if clue_id:
        where.append("c.id = ?")
        params.append(clue_id)
    else:
        if engine_only:
            where.append("s.solved_by NOT IN (%s)" % ",".join("?" * len(NOT_ENGINE)))
            params.extend(NOT_ENGINE)
        where.append("c.publication_date >= date('now', ?)")
        params.append("-%d day" % days)
    return conn.execute(
        "SELECT c.id, c.source, c.puzzle_number, c.clue_number, c.direction, "
        "       c.clue_text, c.answer, c.enumeration, s.operation, s.solved_by "
        "FROM wfw_solve s JOIN clues c ON c.id = s.clue_id "
        "WHERE %s "
        "ORDER BY c.publication_date DESC, c.source, c.puzzle_number, c.id"
        % " AND ".join(where), params).fetchall()


def render(conn, ref, r):
    """One clue's candidate splits, or None when it has none."""
    if not single_word_enumeration(r["enumeration"]):
        return None
    out = []
    defs = definition_keys(conn, r["id"])
    for phrase, letters in splits(r["answer"]):
        means, meant_by = attestation(ref, phrase)
        if not (means or meant_by):
            continue
        said_by = clue_word_meaning(ref, r["clue_text"], phrase)
        wordplay = [w for w in said_by if w not in defs]
        definition = [w for w in said_by if w in defs]
        out.append("  splits as %s" % phrase)
        if wordplay:
            out.append("    IN THE CLUE      %s  ->  %s"
                       % (", ".join('"%s"' % w for w in wordplay), phrase))
        if definition:
            out.append("    CIRCULAR         %s is the DEFINITION, not wordplay — "
                       "probably an ordinary charade"
                       % ", ".join('"%s"' % w for w in definition))
        if meant_by:
            out.append("    the phrase is meant by   %s" % ", ".join(meant_by[:8]))
        if means:
            out.append("    the phrase means         %s" % ", ".join(means[:8]))
    if not out:
        return None
    head = "%s %s %s%s" % (r["source"], r["puzzle_number"], r["clue_number"],
                           (r["direction"] or "")[:1])
    return "\n".join(
        ["%-22s %-14s (%s)  [%s/%s]" % (head, (r["answer"] or "?").upper(),
                                        r["enumeration"] or "?",
                                        r["operation"] or "?", r["solved_by"] or "?"),
         '  "%s"' % (r["clue_text"] or "").strip()] + out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1,
                    help="how far back to look (default 1 — today's passes)")
    ap.add_argument("--clue", type=int, help="one clue id, ignoring --days")
    ap.add_argument("--all", action="store_true",
                    help="include manual and prefill solves, not engine passes only")
    ap.add_argument("--out")
    args = ap.parse_args()

    conn = sqlite3.connect("file:%s?mode=ro" % CLUES_DB.as_posix(), uri=True)
    conn.row_factory = sqlite3.Row
    ref = sqlite3.connect("file:%s?mode=ro" % REF_DB.as_posix(), uri=True)

    rs = rows(conn, args.days, args.clue, engine_only=not args.all)
    blocks = [b for b in (render(conn, ref, r) for r in rs) if b]

    scope = ("clue %d" % args.clue if args.clue else
             "%s passes in the last %d day(s)"
             % ("all" if args.all else "engine", args.days))
    header = "%d word-division candidate(s) among %d %s\n%s\n\n" % (
        len(blocks), len(rs), scope, "=" * 60)
    body = ("\n\n".join(blocks) + "\n") if blocks else "(none)\n"
    if args.out:
        Path(args.out).write_text(header + body, encoding="utf-8")
        print("%d candidate(s) -> %s" % (len(blocks), args.out))
    else:
        sys.stdout.write(header + body)


if __name__ == "__main__":
    main()
