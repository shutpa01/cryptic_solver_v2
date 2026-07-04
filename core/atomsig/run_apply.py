"""Measure the trivial verifier (core.atomsig.apply) against the harvested
instances — clues whose TRUE atom-map we already have. For each, run apply_catalog
over the WHOLE catalogue and classify the verifier's output as:

  CORRECT  — produced a clean pass whose parse matches the known atom-map
             PIECE-FOR-PIECE (every piece's value on the same answer positions);
  WRONG    — produced a clean pass, but the parse differs from the known atom-map
             (a letter-correct but false parse, e.g. a junk synonym);
  NONE     — produced no pass at all.

The headline number is CORRECT. A WRONG pass is a FALSE POSITIVE, not a success —
this is the distinction earlier 'reproduction' counting hid.

Run: .venv/Scripts/python.exe -m core.atomsig.run_apply [--limit N]
"""
import argparse
import json
import os
import sqlite3
import time
from collections import defaultdict

from core import engine_registry as er
from core.atomsig.apply import parse_signature, apply_catalog

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_catalog():
    con = sqlite3.connect(os.path.join(ROOT, "data", "atomsig.db"))
    rows = [k for (k,) in con.execute("SELECT key FROM signature")]
    con.close()
    return [parse_signature(k) for k in rows]


def load_instances(limit):
    con = sqlite3.connect(os.path.join(ROOT, "data", "atomsig.db"))
    rows = con.execute(
        "SELECT clue_id, operation, info_json FROM instance").fetchall()
    con.close()
    if limit:
        rows = rows[:limit]
    cm = sqlite3.connect(os.path.join(ROOT, "data", "clues_master.db"))
    out = []
    for cid, op, info in rows:
        r = cm.execute("SELECT clue_text, answer, direction FROM clues WHERE id=?",
                       (cid,)).fetchone()
        if r:
            out.append((cid, op, json.loads(info), r[0], r[1], r[2]))
    cm.close()
    return out


def _letters(s):
    return "".join(c for c in (s or "").upper() if c.isalpha())


def _tclass(transform):
    """Transform class that matters for piece identity. An anagram piece's value
    is a letter MULTISET (order is meaningless), so it is compared sorted; a
    reversed piece is a distinct class (TOP identity != POT reversed), so order
    and class are kept strict for everything else."""
    if transform in ("anagram", "anagram_of"):
        return "anagram"
    if transform == "reversed":
        return "reversed"
    return "plain"            # identity / selection / None


def _piece_key(value, positions, transform):
    cls = _tclass(transform)
    v = "".join(sorted(_letters(value))) if cls == "anagram" else _letters(value)
    return (frozenset(positions), cls, v)


def known_pieces(info):
    """The harvested atom-map's pieces as comparable keys (positions, transform
    class, value — multiset for anagram, exact otherwise)."""
    return frozenset(_piece_key(p["value"], p["answer_positions"], p.get("transform"))
                     for p in info["pieces"])


def produced_pieces(parse):
    pos = defaultdict(set)
    trans = {}
    for l in parse.links:
        pos[l.source_index].add(l.answer_pos)
        trans[l.source_index] = l.transform
    return frozenset(_piece_key(parse.sources[si].value, ps, trans.get(si))
                     for si, ps in pos.items())


def main(limit=None):
    catalog = load_catalog()
    insts = load_instances(limit)
    wiring = er.db_only(er.make_db_wiring())
    print("catalogue: %d signatures   instances: %d" % (len(catalog), len(insts)))

    correct = wrong = none = 0
    by_op = defaultdict(lambda: [0, 0, 0, 0])   # op -> [correct, wrong, none, total]
    wrong_examples, none_examples = [], []
    t0 = time.time()
    for n, (cid, op, info, clue, ans, direction) in enumerate(insts, 1):
        try:
            p = apply_catalog(clue, ans, wiring, catalog, direction=direction)
        except Exception:
            p = None
        rec = by_op[op]
        rec[3] += 1
        if p is None or p.status != "pass":
            none += 1
            rec[2] += 1
            if len(none_examples) < 12:
                none_examples.append("[%s] %s = %s" % (op, clue[:50], ans))
        elif produced_pieces(p) == known_pieces(info):
            correct += 1
            rec[0] += 1
        else:
            wrong += 1
            rec[1] += 1
            if len(wrong_examples) < 12:
                got = ["%s=%s" % (s.text, s.value) for s in p.sources]
                want = ["%s=%s" % (pp["text"], pp["value"]) for pp in info["pieces"]]
                wrong_examples.append("[%s] %s=%s  got %s  want %s"
                                      % (op, clue[:40], ans, got, want))
        if n % 100 == 0:
            print("  %d/%d  (%.1f/s)  correct=%d wrong=%d none=%d"
                  % (n, len(insts), n / (time.time() - t0), correct, wrong, none),
                  flush=True)

    tot = len(insts)
    print("\n" + "=" * 64)
    print("FULL SET: %d instances   in %.1fs" % (tot, time.time() - t0))
    print("  CORRECT (parse matches known atom-map): %4d  %5.1f%%"
          % (correct, 100.0 * correct / tot))
    print("  WRONG   (clean pass, FALSE parse):       %4d  %5.1f%%"
          % (wrong, 100.0 * wrong / tot))
    print("  NONE    (no pass):                       %4d  %5.1f%%"
          % (none, 100.0 * none / tot))
    print("\nBY OPERATION  (correct / wrong / none / total)")
    for op, (c, w, nn, t) in sorted(by_op.items(), key=lambda kv: -kv[1][3]):
        print("  %-24s %4d / %4d / %4d / %4d   correct %5.1f%%"
              % (op, c, w, nn, t, 100.0 * c / t))
    print("\nWRONG-PARSE EXAMPLES (false positives)")
    for e in wrong_examples:
        print("  " + e)
    print("\nNO-PASS EXAMPLES")
    for e in none_examples:
        print("  " + e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(limit=args.limit)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(limit=args.limit)
