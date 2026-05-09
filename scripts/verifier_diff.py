"""Diff two verifier baseline snapshots: report movement per clue.

Usage:
    python scripts/verifier_diff.py data/verifier_baseline_BEFORE.json data/verifier_baseline_AFTER.json
"""

import argparse
import json
import sys
from pathlib import Path


VERDICT_RANK = {"FAIL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "ERROR": -1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=str)
    parser.add_argument("after", type=str)
    args = parser.parse_args()

    before = {e["id"]: e for e in json.loads(Path(args.before).read_text(encoding="utf-8"))}
    after = {e["id"]: e for e in json.loads(Path(args.after).read_text(encoding="utf-8"))}

    common = sorted(set(before) & set(after))
    moved_up = []
    moved_down = []
    same = 0
    new_high_from_low = []  # potential false-HIGH risk
    dropped_high = []  # potential regression

    for cid in common:
        b = before[cid]
        a = after[cid]
        b_score = b.get("score", 0) or 0
        a_score = a.get("score", 0) or 0
        b_verdict = b.get("verdict", "ERROR")
        a_verdict = a.get("verdict", "ERROR")

        if b_score == a_score and b_verdict == a_verdict:
            same += 1
            continue

        delta = a_score - b_score
        record = {
            "id": cid,
            "label": f"{a.get('source','?')[:3]} #{a.get('puzzle','?')} {a.get('label','?')}",
            "answer": a.get("answer", "?"),
            "before": f"{b_verdict} {b_score}",
            "after": f"{a_verdict} {a_score}",
            "delta": delta,
        }

        if delta > 0:
            moved_up.append(record)
            if VERDICT_RANK[b_verdict] <= 1 and VERDICT_RANK[a_verdict] >= 3:
                new_high_from_low.append(record)
        elif delta < 0:
            moved_down.append(record)
            if VERDICT_RANK[b_verdict] >= 3 and VERDICT_RANK[a_verdict] < 3:
                dropped_high.append(record)

    print(f"Compared {len(common)} clues.")
    print(f"  Unchanged: {same}")
    print(f"  Moved up:   {len(moved_up)}")
    print(f"  Moved down: {len(moved_down)}")
    print()

    if dropped_high:
        print(f"!!! REGRESSION RISK: {len(dropped_high)} clues dropped from HIGH:")
        for r in dropped_high[:30]:
            print(f"  {r['label']:30} {r['answer']:20} {r['before']} -> {r['after']}  ({r['delta']:+d})")
        print()

    if new_high_from_low:
        print(f"!!! FALSE-HIGH RISK: {len(new_high_from_low)} clues jumped from FAIL/LOW to HIGH:")
        for r in new_high_from_low[:30]:
            print(f"  {r['label']:30} {r['answer']:20} {r['before']} -> {r['after']}  ({r['delta']:+d})")
        print()

    if moved_up:
        print(f"All upward moves ({len(moved_up)}):")
        for r in sorted(moved_up, key=lambda x: -x["delta"])[:50]:
            print(f"  {r['label']:30} {r['answer']:20} {r['before']:14} -> {r['after']:14}  ({r['delta']:+d})")
        print()

    if moved_down:
        print(f"All downward moves ({len(moved_down)}):")
        for r in sorted(moved_down, key=lambda x: x["delta"])[:50]:
            print(f"  {r['label']:30} {r['answer']:20} {r['before']:14} -> {r['after']:14}  ({r['delta']:+d})")


if __name__ == "__main__":
    main()
