"""Convert structured_explanations (confidence >= threshold) into atom-maps.

The structured_explanations table holds, per clue, a decomposition the OLD solvers
produced: piece-lists of (type, fodder, yields) — NOT atom-maps and NOT trustworthy
as-is (mixed provenance, self-reported confidence). This module treats each row as a
CANDIDATE and re-validates it against the answer + DB:

  - RECONSTRUCT: do the pieces' yielded letters actually tile the answer? (charade by
    concatenation, anagram by multiset, a single piece, or a single reversed piece —
    the simple shapes; richer alignment, e.g. containers, is reported as a TODO bucket
    so nothing is silently claimed.)
  - DB-BACK: is each piece's yield a real DB value of its fodder (synonym/abbreviation/
    literal), or a genuine selection from the fodder letters (hidden/first-letter)?

Only candidates that reconstruct AND are fully DB-backed become atom-signatures. This
is a MEASUREMENT first: how much of the >=0.85 corpus converts, by wordplay_type — the
evidence for whether this source extends coverage past the cascade.

Usage: .venv/Scripts/python.exe -m core.atomsig.from_structured --limit 4000 --min-conf 0.85
"""

import argparse
import json
import os
import sqlite3
from collections import Counter

from core import engine_registry as er
from core.atomsig.verifier import verify


def _clues_db():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "data", "clues_master.db")


def _letters(s):
    return "".join(ch for ch in (s or "").upper() if ch.isalpha())


def normalize_components(comp_json):
    """Return [(mechanism, fodder, yields)] from either component format:
    a list of {type,fodder,yields}, or a dict with ai_pieces[{mechanism,clue_word,letters}]."""
    try:
        comp = json.loads(comp_json) if comp_json else None
    except (ValueError, TypeError):
        return None
    out = []
    if isinstance(comp, dict):
        for p in comp.get("ai_pieces", []):
            out.append((p.get("mechanism", ""), p.get("clue_word", ""),
                        _letters(p.get("letters", ""))))
    elif isinstance(comp, list):
        for p in comp:
            if isinstance(p, dict):
                out.append((p.get("type", ""), p.get("fodder", ""),
                            _letters(p.get("yields", ""))))
    return out or None


def run(limit=4000, min_conf=0.85):
    wiring = er.db_only(er.make_db_wiring())

    conn = sqlite3.connect(_clues_db(), timeout=30)
    try:
        rows = conn.execute(
            """SELECT c.answer, se.components, se.wordplay_types, se.confidence,
                      se.model_version
               FROM clues c JOIN structured_explanations se ON se.clue_id = c.id
               WHERE c.has_solution = 1 AND c.answer IS NOT NULL AND c.answer <> ''
                 AND se.confidence >= ? AND se.components IS NOT NULL
                 AND se.components <> '' AND se.components <> '[]'
               LIMIT ?""", (min_conf, limit)).fetchall()
    finally:
        conn.close()

    total = len(rows)
    no_components = validated = 0
    by_type_total = Counter()
    by_type_ok = Counter()
    assembly_counts = Counter()
    transform_counts = Counter()

    for answer, comp_json, wtypes, conf, mv in rows:
        ans = _letters(answer)
        wt = (wtypes or "").strip("[]\" ") or "?"
        by_type_total[wt] += 1
        pieces = normalize_components(comp_json)
        if not pieces:
            no_components += 1
            continue
        res = verify(ans, pieces, wiring)
        if res is None:
            continue
        validated += 1
        by_type_ok[wt] += 1
        assembly_counts[res.assembly] += 1
        for p in res.placed:
            transform_counts[p["transform"]] += 1

    _print_report(total, no_components, validated, assembly_counts, transform_counts,
                  by_type_total, by_type_ok, min_conf)


def _pct(n, d):
    return "%.1f%%" % (100.0 * n / d) if d else "n/a"


def _print_report(total, no_components, validated, assembly_counts, transform_counts,
                  by_type_total, by_type_ok, min_conf):
    print("STRUCTURED_EXPLANATIONS -> ATOM-MAP  (confidence >= %.2f)" % min_conf)
    print("=" * 60)
    print("sampled rows:            %d" % total)
    print("unparseable components:  %d  %s" % (no_components, _pct(no_components, total)))
    print("VALIDATED (coupled verifier: tiled + every piece DB-backed in-orientation):")
    print("  %d  %s of sample   (%s of parseable)"
          % (validated, _pct(validated, total), _pct(validated, total - no_components)))
    print()
    print("by assembly:")
    for a, n in assembly_counts.most_common():
        print("  %-12s %6d" % (a, n))
    print()
    print("by transform (per placed piece):")
    for t, n in transform_counts.most_common():
        print("  %-12s %6d" % (t, n))
    print()
    print("validated yield by wordplay_type (ok / total):")
    for wt, tot in by_type_total.most_common(22):
        print("  %-22s %5d / %-5d  %s" % (wt, by_type_ok[wt], tot, _pct(by_type_ok[wt], tot)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--min-conf", type=float, default=0.85)
    args = ap.parse_args()
    run(limit=args.limit, min_conf=args.min_conf)
