"""HTML report listing all 33 catalog templates with origin clue + blog."""
import json
import sys
import sqlite3
import html
from collections import defaultdict

from prototypes.universal_form_v2.schema import LEAF_OPERATIONS
from prototypes.universal_form_v2.strict_verifier import verify
from prototypes.universal_form_v2.salvage_audit import _form_from_dict
from signature_solver.db import RefDB

# Indicator-required ops (per the architecture spec)
_IND_REQUIRED = {"anagram", "reversal", "container", "deletion",
                  "hidden", "homophone", "acrostic"}


def signature(node):
    """Catalog signature, with [+I] markers on ops that require an
    indicator and [+kind] on positional/deletion/acrostic kinds."""
    op = node.operation
    is_leaf = op in {"literal", "synonym", "abbreviation", "positional",
                       "homophone", "raw"} and not node.sources
    if is_leaf:
        if op == "positional":
            return f"positional[{node.positional_kind}+I]"
        return op
    children = [signature(c) for c in node.sources or []]
    extra = ""
    if op == "deletion":
        extra = f"[{node.deletion_kind}+I]"
    elif op == "acrostic":
        extra = f"[{node.acrostic_kind}+I]"
    elif op in _IND_REQUIRED:
        extra = "[+I]"
    return f"{op}{extra}({','.join(children)})"

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fmt_form(n):
    if n is None:
        return ""
    op = n.get("operation")
    has_sources = bool(n.get("sources"))
    if op in LEAF_OPERATIONS and not has_sources:
        val = n.get("value") or ""
        src = n.get("source_word") or ""
        kind = n.get("positional_kind") or ""
        kind_s = f"[{kind}]" if kind else ""
        return f"{op}{kind_s}({html.escape(val)} ← '{html.escape(src)}')"
    ind = n.get("indicator")
    ind_s = f" [{html.escape(ind)}]" if ind else ""
    children = ", ".join(fmt_form(c) for c in (n.get("sources") or []))
    return f"{op}{ind_s}({children})"


def main():
    paths = [
        "prototypes/universal_form_v2/runs/shadow_pipeline_random300.json",
        "prototypes/universal_form_v2/runs/shadow_pipeline_29534.json",
    ]
    db = RefDB()
    by_sig = defaultdict(list)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        for r in data["per_clue"]:
            if r.get("form") is None:
                continue
            form = _form_from_dict(r["form"])
            v = verify(form, r["clue_text"], db, None)
            if v.verdict != "PASS":
                continue
            sig = signature(form.tree)
            by_sig[sig].append((r, form))

    style = """
<style>
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 24px;
       color: #1a1a1a; background: #fafafa; line-height: 1.5; }
h1 { margin: 0 0 8px; }
h2 { margin: 32px 0 12px; padding-bottom: 6px;
     border-bottom: 2px solid #ddd;
     font-family: "SF Mono", Menlo, Consolas, monospace;
     font-size: 1.05em; word-break: break-word; }
h2 .freq { font-family: -apple-system, sans-serif;
            font-size: 0.85em; color: #666; font-weight: normal; }
.warn { background: #fff3cd; border: 1px solid #ffe69c;
        padding: 12px 16px; border-radius: 6px; margin: 12px 0; }
.entry { background: #fff; border: 1px solid #ddd; border-radius: 8px;
         padding: 14px 18px; margin: 10px 0; }
.field { margin: 5px 0; }
.field-label { display: inline-block; font-weight: 600;
                color: #555; min-width: 80px; vertical-align: top; }
code { background: #f0f0f0; padding: 1px 6px; border-radius: 4px;
       font-size: 0.92em; word-break: break-word; }
.form { font-family: "SF Mono", Menlo, Consolas, monospace;
        font-size: 0.85em; background: #f5f5f5; padding: 6px 12px;
        border-radius: 6px; word-break: break-word;
        white-space: pre-wrap; line-height: 1.4;
        border-left: 3px solid #888; }
.toc { background: #fff; padding: 16px 20px; border-radius: 8px;
       border: 1px solid #ddd; margin: 16px 0; }
.toc table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
.toc td { padding: 3px 8px; border-bottom: 1px solid #eee; }
.toc a { text-decoration: none; color: #0050a8;
         font-family: "SF Mono", Menlo, Consolas, monospace; }
.toc a:hover { text-decoration: underline; }
.catalog-tag { background: #e9f2ff; color: #0050a8;
               font-family: "SF Mono", Menlo, Consolas, monospace;
               font-size: 0.9em; padding: 2px 8px; border-radius: 4px; }
</style>
"""

    out = []
    out.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    out.append("<title>v0 catalog — all 33 templates with origin clues</title>")
    out.append(style)
    out.append("</head><body>")
    out.append("<h1>v0 catalog — 33 templates</h1>")
    out.append("<p>Each template was extracted from one or more "
                "blogged Times clues that the strict verifier "
                "marked PASS. For each template the source clue, "
                "blog, and assembled form are shown.</p>")
    out.append('<div class="warn">'
                '<strong>Caveat:</strong> the verifier had a '
                'documented hole on homophone op-nodes (it accepted '
                'any synonym child without checking the homophone DB). '
                'Catalog entries below should be re-verified once that '
                'is fixed; the <code>homophone(synonym)</code> entry '
                'is most exposed.</div>')

    sorted_entries = sorted(by_sig.items(),
                              key=lambda kv: (-len(kv[1]), kv[0]))

    # Index
    out.append("<div class='toc'><h3>Index</h3><table><tr><th>Template</th>"
                "<th>Examples</th></tr>")
    for sig, items in sorted_entries:
        anchor = f"t{abs(hash(sig))}"
        out.append(f"<tr><td><a href='#{anchor}'>"
                    f"{html.escape(sig)}</a></td>"
                    f"<td>{len(items)}</td></tr>")
    out.append("</table></div>")

    # Each catalog entry
    for sig, items in sorted_entries:
        anchor = f"t{abs(hash(sig))}"
        out.append(f"<h2 id='{anchor}'>{html.escape(sig)} "
                    f"<span class='freq'>({len(items)} example"
                    f"{'s' if len(items)!=1 else ''})</span></h2>")
        for r, form in items:
            out.append("<div class='entry'>")
            out.append(f"<div class='field'><span class='field-label'>"
                        f"Answer:</span><strong>{r['answer']}</strong> "
                        f"<span style='color:#888;font-size:0.9em'>"
                        f"({r.get('source','?')} {r.get('puzzle_number','?')} "
                        f"{r.get('clue_number','?')}"
                        f"{r.get('direction','?')[0] if r.get('direction') else ''})"
                        f"</span></div>")
            out.append(f"<div class='field'><span class='field-label'>"
                        f"Catalog:</span><code class='catalog-tag'>"
                        f"{html.escape(sig)}</code></div>")
            out.append(f"<div class='field'><span class='field-label'>"
                        f"Clue:</span>{html.escape(r['clue_text'])}</div>")
            blog = r.get('blog') or ''
            out.append(f"<div class='field'><span class='field-label'>"
                        f"Blog:</span>{html.escape(blog)}</div>")
            out.append("<div class='field'><span class='field-label'>"
                        "Form:</span></div>")
            out.append(f"<div class='form'>{fmt_form(form.tree.to_dict())}"
                        f"</div>")
            def_phrase = form.definition.phrase
            out.append(f"<div class='field'><span class='field-label'>"
                        f"Definition:</span><code>"
                        f"{html.escape(def_phrase)}</code></div>")
            if form.link_words:
                out.append(f"<div class='field'><span class='field-label'>"
                            f"Link words:</span>"
                            f"{html.escape(', '.join(form.link_words))}"
                            f"</div>")
            out.append("</div>")

    out.append("</body></html>")
    path = "prototypes/universal_form_v2/runs/catalog_all_33.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    import os
    print(f"Written {path} ({os.path.getsize(path):,} bytes)")
    print(f"Templates: {len(sorted_entries)}; total forms: "
            f"{sum(len(items) for _, items in sorted_entries)}")


if __name__ == "__main__":
    main()
