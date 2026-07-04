"""Test the real hypothesis: does the GRAMMAR SIGNATURE of the WHOLE clue
correlate with the WORDPLAY TYPE?

No solving, no answer-awareness. For clues capped at <= MAX_WORDS (the setter's
~8-word budget), tag the full clue, build a coarse POS signature, and ask: can
that signature predict the wordplay type better than chance?

Method: balanced sample per single-label type -> coarse POS signature of the full
clue -> 70/30 train/test -> learn signature->majority-type on train -> evaluate on
held-out test (accuracy vs balanced base rate 1/K), plus signature coverage and
the mutual information between signature and type.

Usage: python -m prototypes.universal_form_v2.runs.sig_type_correlation [N_per_type] [MAX_WORDS]
"""
from __future__ import annotations

import math
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from signature_solver.grammar_triage import _MID_MAP

TYPES = ["anagram", "charade", "container", "reversal", "deletion",
         "double_definition", "hidden", "homophone", "acrostic",
         "cryptic_definition"]


def _words(clue):
    clue = re.sub(r"\(\d+(?:[,\- ]\d+)*\)\s*$", "", clue or "").strip()
    return [w for w in re.findall(r"[A-Za-z']+", clue)]


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    max_words = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    random.seed(12345)  # fixed split (no Date/random-at-runtime surprises)

    master = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    rows_by_type = {}
    for t in TYPES:
        rows = master.execute(
            "SELECT clue_text FROM clues WHERE wordplay_type = ? "
            "AND clue_text IS NOT NULL AND clue_text != '' "
            "ORDER BY RANDOM() LIMIT ?", (t, n_per * 3)).fetchall()
        kept = [r[0] for r in rows if 2 <= len(_words(r[0])) <= max_words][:n_per]
        if kept:
            rows_by_type[t] = kept
    master.close()

    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])

    def signature(clue):
        doc = nlp(" ".join(_words(clue)))
        return tuple(_MID_MAP.get(tok.tag_, "X") for tok in doc if tok.is_alpha)

    # Build (signature, type) dataset, balanced.
    data = []
    print("tagging...", flush=True)
    for t, clues in rows_by_type.items():
        docs = nlp.pipe([" ".join(_words(c)) for c in clues], batch_size=256)
        for doc in docs:
            sig = tuple(_MID_MAP.get(tok.tag_, "X") for tok in doc if tok.is_alpha)
            if sig:
                data.append((sig, t))
    random.shuffle(data)

    types = sorted({t for _, t in data})
    K = len(types)
    cut = int(0.7 * len(data))
    train, test = data[:cut], data[cut:]

    # Learn signature -> type distribution on train.
    sig_counts = defaultdict(Counter)
    for sig, t in train:
        sig_counts[sig][t] += 1
    sig_major = {s: c.most_common(1)[0][0] for s, c in sig_counts.items()}
    global_major = Counter(t for _, t in train).most_common(1)[0][0]

    # Evaluate on held-out test.
    covered = correct_cov = correct_all = 0
    per_type_tot = Counter()
    per_type_hit = Counter()
    for sig, t in test:
        per_type_tot[t] += 1
        if sig in sig_major:
            covered += 1
            pred = sig_major[sig]
            if pred == t:
                correct_cov += 1
                correct_all += 1
                per_type_hit[t] += 1
        else:
            pred = global_major
            if pred == t:
                correct_all += 1
                if t == global_major:
                    per_type_hit[t] += 1

    # Mutual information I(signature; type) on the full dataset.
    N = len(data)
    pt = Counter(t for _, t in data)
    ps = Counter(s for s, _ in data)
    pst = Counter(data)
    H_type = -sum((c / N) * math.log2(c / N) for c in pt.values())
    mi = 0.0
    for (s, t), c in pst.items():
        pxy = c / N
        mi += pxy * math.log2(pxy / ((ps[s] / N) * (pt[t] / N)))

    print("\n" + "=" * 64, flush=True)
    print(f"clues used: {N}   types: {K}   max_words: {max_words}", flush=True)
    print(f"distinct signatures: {len(ps)}   "
          f"(avg {N/len(ps):.1f} clues/sig)", flush=True)
    base = 1.0 / K
    print(f"\nbalanced base rate (guess majority): {base*100:.1f}%", flush=True)
    print(f"held-out signature coverage: {100*covered/len(test):.1f}% "
          f"(test sigs seen in train)", flush=True)
    print(f"accuracy ON COVERED clues: {100*correct_cov/covered:.1f}%"
          if covered else "no coverage", flush=True)
    print(f"accuracy OVERALL (uncovered->global majority): "
          f"{100*correct_all/len(test):.1f}%", flush=True)
    print(f"\nmutual info I(sig;type) = {mi:.2f} bits of "
          f"H(type) = {H_type:.2f} bits  "
          f"({100*mi/H_type:.0f}% of type uncertainty removed)", flush=True)

    print("\nper-type recall on test (pred == true):", flush=True)
    for t in types:
        tot = per_type_tot[t]
        print(f"  {t:18s} {100*per_type_hit[t]/tot:5.1f}%  (n={tot})"
              if tot else f"  {t:18s}   n=0", flush=True)

    print("\ntop signatures by frequency (dominant type / purity):", flush=True)
    for sig, c in ps.most_common(12):
        dist = Counter(tt for ss, tt in data if ss == sig)
        dom, dc = dist.most_common(1)[0]
        print(f"  {'-'.join(sig):28s} n={c:4d}  {dom} {100*dc/c:.0f}%", flush=True)


if __name__ == "__main__":
    main()
