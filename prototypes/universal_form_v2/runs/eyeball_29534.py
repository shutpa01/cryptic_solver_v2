"""Run the clipboard verifier on every form we have for Times 29534
and dump a markdown report I can eyeball clue-by-clue against the blog.
"""
import json, sys, sqlite3
sys.stdout.reconfigure(encoding="utf-8")

from prototypes.universal_form_v2.salvage_audit import _form_from_dict
from prototypes.universal_form_v2.clipboard_verifier import verify
from prototypes.universal_form_v2.schema import LEAF_OPERATIONS
from signature_solver.db import RefDB

db = RefDB()

with open("prototypes/universal_form_v2/runs/shadow_pipeline_29534.json",
          encoding="utf-8") as f:
    data = json.load(f)


def fmt(n):
    op = n.get("operation")
    has = bool(n.get("sources"))
    if op in LEAF_OPERATIONS and not has:
        v = n.get("value") or ""
        s = n.get("source_word") or ""
        k = n.get("positional_kind") or ""
        ks = f"[{k}]" if k else ""
        return f"{op}{ks}({v} ← {s!r})"
    ind = n.get("indicator")
    is_ = f" [{ind}]" if ind else ""
    chs = ", ".join(fmt(c) for c in n.get("sources", []))
    return f"{op}{is_}({chs})"


lines = []
lines.append("# Times 29534 — clipboard verifier eyeball\n")
counts = {"PASS": 0, "FAIL": 0, "NO_FORM": 0}
results = data["per_clue"]
results.sort(key=lambda r: (r["direction"], int(r["clue_number"])))

for r in results:
    cn, d = r["clue_number"], r["direction"][0]
    lines.append(f"## {cn}{d}. {r['answer']}\n")
    lines.append(f"- Clue: {r['clue_text']}")
    lines.append(f"- Blog: {r.get('blog') or '(none)'}")
    lines.append(f"- DB definition: `{r.get('definition_db') or '—'}`")
    if r.get("form") is None:
        counts["NO_FORM"] += 1
        lines.append("- **VERDICT: NO_FORM** (no form built)\n")
        continue
    form = _form_from_dict(r["form"])
    v = verify(form, r["clue_text"], db)
    counts[v.verdict] += 1
    lines.append(f"- Form: `{fmt(form.tree.to_dict())}`")
    lines.append(f"- Definition (form): `{form.definition.phrase}`")
    if form.link_words:
        lines.append(f"- Link words: {form.link_words}")
    lines.append(f"- **VERDICT: {v.verdict}**")
    for c in v.checks:
        marker = "✓" if c.status == "pass" else "✗"
        lines.append(f"  - {marker} `{c.name}`: {c.detail}")
    if v.enrichment_candidates:
        lines.append("  - Enrichment candidates:")
        for ec in v.enrichment_candidates:
            lines.append(f"    - `{ec.kind}` {ec.detail}")
    lines.append("")

lines.insert(2, f"**Counts: {counts}**\n")
out = "\n".join(lines)
path = "prototypes/universal_form_v2/runs/eyeball_29534.md"
with open(path, "w", encoding="utf-8") as f:
    f.write(out)
print(f"Written {path} ({len(out):,} bytes)")
print(f"Counts: {counts}")
