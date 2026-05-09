"""Inspect the strictly-verified forms to inform catalog design.

For each form that strict_verifier says PASSes, dump:
  - clue, answer
  - the form tree
  - a "structural signature" (operations only, no values)
  - a "mechanism signature" (operations + leaf mechanisms)

This is read-only diagnostic — no DB writes, no fixes.
"""
import json
import sys
import sqlite3
from collections import Counter
from pathlib import Path

from .schema import Form, Definition, Node
from .strict_verifier import verify
from .salvage_audit import _form_from_dict, _node_from_dict
from signature_solver.db import RefDB


def signature(node: Node, with_mech=False) -> str:
    """Build a structural signature: operations only.

    Examples:
      charade(synonym, anagram(literal), positional[outer])
      container[anagram(literal), synonym]
    """
    op = node.operation
    is_leaf = op in {"literal", "synonym", "abbreviation", "positional",
                       "homophone", "raw"} and not node.sources
    if is_leaf:
        if op == "positional":
            return f"positional[{node.positional_kind}]"
        return op
    children = [signature(c, with_mech) for c in node.sources or []]
    if op == "deletion":
        return f"deletion[{node.deletion_kind}]({','.join(children)})"
    if op == "acrostic":
        return f"acrostic[{node.acrostic_kind}]({','.join(children)})"
    return f"{op}({','.join(children)})"


def main(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    db = RefDB()
    shadow = Path("data/shadow_blog_v0.db")
    shadow_conn = sqlite3.connect(shadow) if shadow.exists() else None

    pass_records = []
    for r in data["per_clue"]:
        if r.get("form") is None:
            continue
        form = _form_from_dict(r["form"])
        v = verify(form, r["clue_text"], db, shadow_conn)
        if v.verdict != "PASS":
            continue
        pass_records.append((r, form))

    print(f"Strict-PASS forms: {len(pass_records)}")
    print()

    sig_counts = Counter()
    for r, form in pass_records:
        sig = signature(form.tree)
        sig_counts[sig] += 1

    print("Signatures (frequency):")
    for sig, n in sig_counts.most_common(30):
        print(f"  {n:3d}  {sig}")
    print()

    print("Sample of one form per signature (top 10 sigs):")
    seen = set()
    for r, form in pass_records:
        sig = signature(form.tree)
        if sig in seen:
            continue
        if len(seen) >= 10:
            break
        seen.add(sig)
        print(f"  {r['answer']:18}  {sig}")
        print(f"    clue: {r['clue_text']}")
    print()

    return pass_records


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "prototypes/universal_form_v2/runs/shadow_pipeline_random300.json",
        "prototypes/universal_form_v2/runs/shadow_pipeline_29534.json",
    ]
    for p in paths:
        if Path(p).exists():
            print("=" * 60)
            print(p)
            print("=" * 60)
            main(p)
