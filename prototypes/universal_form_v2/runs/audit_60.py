"""Render the 60 strict-PASS forms with their blogs side-by-side.

For each form, show:
  - Clue, answer, blog
  - Form structure (compact)
  - Per-leaf source/value
  - Highlight templates that may be pathological
"""
import json
import sys
import sqlite3
from collections import defaultdict
from pathlib import Path

from prototypes.universal_form_v2.schema import LEAF_OPERATIONS
from prototypes.universal_form_v2.strict_verifier import verify
from prototypes.universal_form_v2.salvage_audit import _form_from_dict
from prototypes.universal_form_v2.extract_catalog import signature
from signature_solver.db import RefDB

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fmt_form(n, depth=0):
    op = n.get("operation")
    has_sources = bool(n.get("sources"))
    if op in LEAF_OPERATIONS and not has_sources:
        val = n.get("value") or ""
        src = n.get("source_word") or ""
        kind = n.get("positional_kind") or ""
        kind_s = f"[{kind}]" if kind else ""
        return f"{op}{kind_s}({val} ← {src!r})"
    ind = n.get("indicator")
    ind_s = f" [{ind}]" if ind else ""
    children = ", ".join(fmt_form(c, depth + 1) for c in n.get("sources", []))
    return f"{op}{ind_s}({children})"


def main():
    paths = [
        "prototypes/universal_form_v2/runs/shadow_pipeline_random300.json",
        "prototypes/universal_form_v2/runs/shadow_pipeline_29534.json",
    ]
    db = RefDB()
    shadow = Path("data/shadow_blog_v0.db")
    shadow_conn = sqlite3.connect(shadow) if shadow.exists() else None

    by_sig = defaultdict(list)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        for r in data["per_clue"]:
            if r.get("form") is None:
                continue
            form = _form_from_dict(r["form"])
            v = verify(form, r["clue_text"], db, shadow_conn)
            if v.verdict != "PASS":
                continue
            sig = signature(form.tree)
            by_sig[sig].append((r, form))

    # Print grouped by signature
    out_lines = []
    out_lines.append("# Audit: 60 strict-PASS forms")
    out_lines.append("")
    total = sum(len(items) for items in by_sig.values())
    out_lines.append(f"Total: {total} strict-PASSes across "
                      f"{len(by_sig)} signatures")
    out_lines.append("")

    for sig, items in sorted(by_sig.items(),
                              key=lambda kv: (-len(kv[1]), kv[0])):
        out_lines.append(f"\n## {sig}  ({len(items)} examples)\n")
        for r, form in items:
            out_lines.append(f"### {r['answer']}  ({r.get('source','?')} "
                              f"{r.get('puzzle_number','?')} "
                              f"{r.get('clue_number','?')}{r.get('direction','?')[0]})")
            out_lines.append(f"  Clue: {r['clue_text']}")
            blog = (r.get('blog') or '')[:200]
            out_lines.append(f"  Blog: {blog}")
            out_lines.append(f"  Form: {fmt_form(form.tree.to_dict() if hasattr(form.tree, 'to_dict') else form.tree)}")
            # Manual: dump tree dict directly
            try:
                out_lines[-1] = f"  Form: {fmt_form(form.tree.to_dict())}"
            except Exception:
                pass
            out_lines.append("")

    text = "\n".join(out_lines)
    out_path = "prototypes/universal_form_v2/runs/audit_60.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Written {out_path} ({len(text):,} bytes)")


if __name__ == "__main__":
    main()
