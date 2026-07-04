"""Harvest atom-signatures from the existing cascade's passes.

Runs the existing solver (DB-only, no AI, no persistence) over a corpus of
clue+answer pairs and converts every PASS into an atom-signature (core.atomsig.
signature). The point is to seed a real signature catalogue from already-validated
solves and, in doing so, MEASURE three things with evidence rather than assumption:

  - base pass rate of the existing cascade on the sample;
  - clean-conversion rate (passes that read cleanly into a complete atom-map) —
    this is the base rate the new verifier would inherit;
  - the signature distribution (which shapes, how often) + the reasons passes fail
    to convert + the breakdown of non-passes.

Outputs go to logs/atomsig/ (no DB schema change). Usage:
    .venv/Scripts/python.exe -m core.atomsig.harvest --limit 800 --seed 7
"""

import argparse
import json
import os
import random
import sqlite3
import time
from collections import Counter

from core import engine_registry as er
from core.atomsig.signature import signature_from_parse


def _clues_db():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "data", "clues_master.db")


def _out_dir():
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                     "logs", "atomsig")
    os.makedirs(d, exist_ok=True)
    return d


def load_corpus(limit, seed):
    """A reproducible random sample of solved clues (clue, answer, direction, id)."""
    conn = sqlite3.connect(_clues_db(), timeout=30)
    try:
        rows = conn.execute(
            "SELECT id, clue_text, answer, direction FROM clues "
            "WHERE answer IS NOT NULL AND answer <> '' AND clue_text IS NOT NULL "
            "AND has_solution = 1").fetchall()
    finally:
        conn.close()
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit] if limit else rows


def run(limit=800, seed=7, progress_every=100):
    corpus = load_corpus(limit, seed)
    wiring = er.db_only(er.make_db_wiring())

    status_counts = Counter()          # pass / pending / fail / none / error
    engine_pass = Counter()            # engine name among passes
    convert_ok = 0
    convert_fail_reasons = Counter()
    sig_counts = Counter()             # AtomSignature.key() -> frequency
    sig_examples = {}                  # key -> a sample clue/answer
    instances = []                     # converted instances (seed material)
    nonpass_by_engine = Counter()      # engine name among non-pass (best-effort)

    t0 = time.time()
    for n, (cid, clue, answer, direction) in enumerate(corpus, 1):
        try:
            ctx, parse, name = er.solve_clue_text(clue, answer, wiring,
                                                  direction=direction)
        except Exception as e:                       # never let one clue stop the run
            status_counts["error"] += 1
            convert_fail_reasons["solve-exception: %s" % type(e).__name__] += 1
            continue

        if parse is None:
            status_counts["none"] += 1
            continue
        status_counts[parse.status] += 1

        if parse.status != "pass":
            nonpass_by_engine["%s/%s" % (parse.status, name)] += 1
            continue

        engine_pass[name] += 1
        sig, info = signature_from_parse(parse, ctx)
        if sig is None:
            convert_fail_reasons[info.split(":")[0]] += 1
            continue
        convert_ok += 1
        key = sig.key()
        sig_counts[key] += 1
        sig_examples.setdefault(key, {"clue": clue, "answer": answer})
        instances.append({"id": cid, "signature": key, **info})

        if progress_every and n % progress_every == 0:
            rate = n / (time.time() - t0)
            print("  %d/%d  (%.1f clues/s)  passes=%d convert=%d"
                  % (n, len(corpus), rate, sum(engine_pass.values()), convert_ok),
                  flush=True)

    elapsed = time.time() - t0
    report = _report(corpus, status_counts, engine_pass, convert_ok,
                     convert_fail_reasons, sig_counts, sig_examples,
                     nonpass_by_engine, elapsed, limit, seed)

    out = _out_dir()
    with open(os.path.join(out, "harvest_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(out, "signatures.jsonl"), "w", encoding="utf-8") as f:
        for key, cnt in sig_counts.most_common():
            f.write(json.dumps({"signature": key, "count": cnt,
                                "example": sig_examples.get(key)}) + "\n")
    with open(os.path.join(out, "instances.jsonl"), "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")

    print(report)
    print("\nwrote: %s" % out)
    return report


def _pct(n, d):
    return "%.1f%%" % (100.0 * n / d) if d else "n/a"


def _report(corpus, status_counts, engine_pass, convert_ok, convert_fail_reasons,
            sig_counts, sig_examples, nonpass_by_engine, elapsed, limit, seed):
    total = len(corpus)
    passes = sum(engine_pass.values())
    lines = []
    add = lines.append
    add("ATOM-SIGNATURE HARVEST")
    add("=" * 60)
    add("corpus: %d clues  (limit=%s seed=%s)  in %.1fs" % (total, limit, seed, elapsed))
    add("")
    add("STATUS BREAKDOWN")
    for st in ("pass", "pending", "fail", "none", "error"):
        if status_counts.get(st):
            add("  %-8s %5d   %s" % (st, status_counts[st], _pct(status_counts[st], total)))
    add("")
    add("BASE PASS RATE:        %d / %d   %s" % (passes, total, _pct(passes, total)))
    add("CLEAN-CONVERSION:      %d / %d passes  %s   (= %s of corpus)"
        % (convert_ok, passes, _pct(convert_ok, passes), _pct(convert_ok, total)))
    add("DISTINCT SIGNATURES:   %d" % len(sig_counts))
    add("")
    add("PASSES BY ENGINE")
    for name, cnt in engine_pass.most_common():
        add("  %-18s %5d" % (name, cnt))
    add("")
    add("CONVERSION FAILURES (passes that did NOT read into a clean atom-map)")
    if convert_fail_reasons:
        for reason, cnt in convert_fail_reasons.most_common():
            add("  %-40s %5d" % (reason, cnt))
    else:
        add("  (none)")
    add("")
    add("TOP 30 SIGNATURES")
    for key, cnt in sig_counts.most_common(30):
        ex = sig_examples.get(key, {})
        add("  %4d  %s" % (cnt, key))
        add("        e.g. %s = %s" % (ex.get("clue", "")[:70], ex.get("answer", "")))
    add("")
    add("NON-PASS BY (status/engine)  top 20")
    for k, cnt in nonpass_by_engine.most_common(20):
        add("  %-30s %5d" % (k, cnt))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()
    run(limit=args.limit, seed=args.seed, progress_every=args.progress_every)
