"""Firm up the DD signal with a FEATURE model (not exact-signature lookup).

Trains a logistic-regression DD-vs-rest classifier on grammar/structural features
of the whole clue, and evaluates it as a PRECISION GATE — including precision at
the real DD prevalence (~7%), where a gate actually has to operate.

Features per clue (all from the surface, no answer, no solving):
  - n_words
  - coarse POS counts (via grammar_triage._MID_MAP)
  - frac_content = (N + NP + J) / n_words
  - has_verb, has_prep, has_conj, has_det
  - has_indicator / n_indicator_words : any clue word in the indicators table
    (DD has a surface + two definitions and NO operation indicator)

Usage: python -m prototypes.universal_form_v2.runs.dd_feature_model [N_per_class]
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from signature_solver.grammar_triage import _MID_MAP

COARSE = sorted(set(_MID_MAP.values()))


def _words(clue):
    clue = re.sub(r"\(\d+(?:[,\- ]\d+)*\)\s*$", "", clue or "").strip()
    return re.findall(r"[A-Za-z']+", clue)


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    master = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))

    # Natural DD prevalence among labelled clues (for production-precision calc).
    dd_n = master.execute("SELECT COUNT(*) FROM clues WHERE wordplay_type="
                          "'double_definition'").fetchone()[0]
    lab_n = master.execute("SELECT COUNT(*) FROM clues WHERE wordplay_type "
                           "IS NOT NULL AND wordplay_type!=''").fetchone()[0]
    prevalence = dd_n / lab_n

    pos = [r[0] for r in master.execute(
        "SELECT clue_text FROM clues WHERE wordplay_type='double_definition' "
        "AND clue_text IS NOT NULL AND clue_text!='' ORDER BY RANDOM() LIMIT ?",
        (n_per,)).fetchall()]
    neg = [r[0] for r in master.execute(
        "SELECT clue_text FROM clues WHERE wordplay_type IS NOT NULL "
        "AND wordplay_type NOT IN ('','double_definition') "
        "AND clue_text IS NOT NULL AND clue_text!='' ORDER BY RANDOM() LIMIT ?",
        (n_per,)).fetchall()]

    # Indicator words (any wordplay type) — DD should have none.
    ref = sqlite3.connect(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    indicators = {r[0].lower() for r in ref.execute(
        "SELECT DISTINCT word FROM indicators WHERE word IS NOT NULL") if r[0]}
    ref.close()
    master.close()

    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    feat_names = (["n_words", "frac_content", "has_verb", "has_prep",
                   "has_conj", "has_det", "has_indicator", "n_indicators"]
                  + ["pos_" + c for c in COARSE])

    def featurize(clues):
        rows = []
        texts = [" ".join(_words(c)) for c in clues]
        for doc, clue in zip(nlp.pipe(texts, batch_size=256), clues):
            toks = [t for t in doc if t.is_alpha]
            n = len(toks) or 1
            counts = {c: 0 for c in COARSE}
            for t in toks:
                counts[_MID_MAP.get(t.tag_, "X")] += 1
            content = counts["N"] + counts["NP"] + counts["J"]
            wl = [w.lower() for w in _words(clue)]
            n_ind = sum(1 for w in wl if w in indicators)
            has_verb = 1 if (counts.get("Vb", 0) + counts.get("Vi", 0)) else 0
            rows.append([
                n, content / n, has_verb,
                1 if counts.get("P", 0) else 0,
                1 if counts.get("C", 0) else 0,
                1 if counts.get("D", 0) else 0,
                1 if n_ind else 0, n_ind,
            ] + [counts[c] for c in COARSE])
        return np.array(rows, dtype=float)

    print("featurising...", flush=True)
    X = np.vstack([featurize(pos), featurize(neg)])
    y = np.array([1] * len(pos) + [0] * len(neg))

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42,
                                          stratify=y)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)

    print("\n" + "=" * 64, flush=True)
    print(f"DD positives: {len(pos)}   non-DD: {len(neg)}", flush=True)
    print(f"real DD prevalence among labelled clues: {prevalence*100:.1f}%", flush=True)
    print(f"held-out ROC AUC: {auc:.3f}", flush=True)
    print("\nthreshold table (balanced test); precision_nat = at "
          f"{prevalence*100:.1f}% prevalence:", flush=True)
    print(f"  {'thr':>4} {'recall':>7} {'spec':>6} {'prec_bal':>9} {'prec_nat':>9}", flush=True)
    P = yte.sum(); Nn = len(yte) - P
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        pred = proba >= thr
        tp = int(((pred == 1) & (yte == 1)).sum())
        fp = int(((pred == 1) & (yte == 0)).sum())
        tpr = tp / P if P else 0
        fpr = fp / Nn if Nn else 0
        prec_bal = tp / (tp + fp) if (tp + fp) else 0
        prec_nat = (tpr * prevalence) / (tpr * prevalence + fpr * (1 - prevalence)) \
            if (tpr * prevalence + fpr * (1 - prevalence)) else 0
        print(f"  {thr:>4.2f} {tpr*100:>6.1f}% {(1-fpr)*100:>5.1f}% "
              f"{prec_bal*100:>8.1f}% {prec_nat*100:>8.1f}%", flush=True)

    order = np.argsort(clf.coef_[0])
    print("\nstrongest DD features (+):", flush=True)
    for i in order[::-1][:6]:
        print(f"  {feat_names[i]:16s} {clf.coef_[0][i]:+.2f}", flush=True)
    print("strongest NOT-DD features (-):", flush=True)
    for i in order[:6]:
        print(f"  {feat_names[i]:16s} {clf.coef_[0][i]:+.2f}", flush=True)

    # Interpretable rule gate the DD engine could literally use.
    print("\ninterpretable rule gate "
          "(no indicator AND no verb AND n_words<=5 AND frac_content>=0.6):",
          flush=True)
    nw = Xte[:, 0]; fc = Xte[:, 1]; hv = Xte[:, 2]; hi = Xte[:, 6]
    rule = (hi == 0) & (hv == 0) & (nw <= 5) & (fc >= 0.6)
    tp = int((rule & (yte == 1)).sum()); fp = int((rule & (yte == 0)).sum())
    tpr = tp / P; fpr = fp / Nn
    prec_nat = (tpr * prevalence) / (tpr * prevalence + fpr * (1 - prevalence)) \
        if (tpr * prevalence + fpr * (1 - prevalence)) else 0
    print(f"  recall={tpr*100:.1f}%  prec_bal={100*tp/(tp+fp) if tp+fp else 0:.1f}%"
          f"  prec_nat={prec_nat*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
