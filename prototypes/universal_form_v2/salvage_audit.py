"""Re-verify existing run JSONs with the strict verifier.

Reads a shadow_pipeline JSON, reconstructs each form, and runs it through
strict_verifier. Reports the genuine PASS count and categorises failures.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import sqlite3

from .schema import Form, Definition, Node
from .strict_verifier import verify
from signature_solver.db import RefDB


def _node_from_dict(d: dict) -> Node:
    n = Node(
        operation=d["operation"],
        indicator=d.get("indicator"),
        value=d.get("value"),
        source_word=d.get("source_word"),
        positional_kind=d.get("positional_kind"),
        deletion_kind=d.get("deletion_kind"),
        acrostic_kind=d.get("acrostic_kind"),
        flags=d.get("flags") or [],
    )
    n.sources = [_node_from_dict(s) for s in (d.get("sources") or [])]
    return n


def _form_from_dict(d: dict) -> Form:
    return Form(
        tree=_node_from_dict(d["tree"]),
        definition=Definition(
            phrase=d["definition"]["phrase"],
            answer=d["definition"]["answer"],
        ),
        link_words=d.get("link_words") or [],
        is_and_lit=d.get("is_and_lit", False),
        flags=d.get("flags") or [],
    )


def audit(json_path: str):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    db = RefDB()
    shadow_path = Path("data/shadow_blog_v0.db")
    shadow_conn = sqlite3.connect(shadow_path) if shadow_path.exists() else None

    rows = data.get("per_clue", [])
    counts = Counter()
    fail_modes = Counter()
    by_outcome = {"NEW_PASS": [], "DEMOTED": [], "STILL_FAIL": [],
                   "STILL_NO_FORM": []}

    for r in rows:
        old_v = (r.get("verdict_pass2") or {}).get("verdict")
        if r.get("form") is None:
            counts["NO_FORM"] += 1
            by_outcome["STILL_NO_FORM"].append(r)
            continue
        try:
            form = _form_from_dict(r["form"])
        except Exception as e:
            counts["RECONSTRUCT_ERROR"] += 1
            continue
        v = verify(form, r["clue_text"], db, shadow_conn)
        if v.verdict == "PASS":
            counts["STRICT_PASS"] += 1
            if old_v != "PASS":
                by_outcome["NEW_PASS"].append((r, v))
        else:
            counts["STRICT_FAIL"] += 1
            for c in v.checks:
                if c.status == "fail":
                    fail_modes[c.name] += 1
            if old_v == "PASS":
                by_outcome["DEMOTED"].append((r, v))
            else:
                by_outcome["STILL_FAIL"].append((r, v))

    print(f"Audited: {json_path}")
    print(f"  Total rows: {len(rows)}")
    print(f"  Counts: {dict(counts)}")
    print(f"  Strict-fail modes: {dict(fail_modes)}")
    print(f"  Demoted (old PASS, new FAIL): {len(by_outcome['DEMOTED'])}")
    print(f"  New PASS (old FAIL/NO_FORM, now PASS): "
          f"{len(by_outcome['NEW_PASS'])}")

    return counts, fail_modes, by_outcome


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "prototypes/universal_form_v2/runs/shadow_pipeline_29534.json",
        "prototypes/universal_form_v2/runs/shadow_pipeline_random300.json",
    ]
    for p in paths:
        if Path(p).exists():
            print()
            audit(p)
        else:
            print(f"SKIP (not found): {p}")
