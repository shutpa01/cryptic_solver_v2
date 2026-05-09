"""HTML report for the catalog held-out run (50 puzzles, 1194 clues).

Same visual style as report_29534.html. Shows:
  - Summary banner (HIT / MISS, with synonym separated)
  - Index table grouped by template id
  - Per-clue detail for every HIT (excluding synonym-template hits)
  - Sample of MISSes
"""
import json
import sys
import sqlite3
import html
from collections import defaultdict, Counter

from prototypes.universal_form_v2.schema import LEAF_OPERATIONS

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

INPUT = "prototypes/universal_form_v2/runs/catalog_run_50puzzles.json"
OUTPUT = "prototypes/universal_form_v2/runs/report_holdout.html"

with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)

# DB wordplay_type for context
conn = sqlite3.connect("data/clues_master.db")
db_wpt_rows = conn.execute(
    "SELECT puzzle_number, clue_number, direction, wordplay_type "
    "FROM clues WHERE source=? AND CAST(puzzle_number AS INTEGER) "
    "BETWEEN 29400 AND 29449",
    ("times",)).fetchall()
db_wpt = {(p, c, d): w for (p, c, d, w) in db_wpt_rows}


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


# Group hits by template
hits_by_template = defaultdict(list)
all_hits_real = []
all_hits_synonym = []
all_misses = []
for r in data["per_clue"]:
    if r["verdict"] == "HIT":
        if r.get("template") == "synonym":
            all_hits_synonym.append(r)
        else:
            hits_by_template[r["template"]].append(r)
            all_hits_real.append(r)
    elif r["verdict"] == "MISS":
        all_misses.append(r)

n_total = data["n_clues"]
n_real = len(all_hits_real)
n_syn = len(all_hits_synonym)
n_miss = len(all_misses)

style = """
<style>
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 24px;
       color: #1a1a1a; background: #fafafa; line-height: 1.5; }
h1 { margin: 0 0 8px; }
h2 { margin: 32px 0 12px; padding-bottom: 6px;
     border-bottom: 2px solid #ddd; }
h3 { margin: 20px 0 10px; }
.summary { background: #fff; padding: 16px 20px; border-radius: 8px;
           border: 1px solid #ddd; margin: 12px 0 24px;
           font-size: 1.05em; }
.summary span { display: inline-block; margin-right: 24px; }
.summary .hit { color: #1a7f37; font-weight: 600; }
.summary .miss { color: #cf222e; font-weight: 600; }
.summary .syn { color: #9a6700; font-weight: 600; }
.note-box { background: #fffbe7; padding: 12px 16px; border-radius: 6px;
            border-left: 4px solid #d4a72c; margin: 12px 0;
            font-size: 0.95em; }
.clue { background: #fff; border: 1px solid #ddd; border-radius: 8px;
        padding: 16px 20px; margin: 12px 0;
        border-left: 4px solid #1a7f37; }
.clue.miss { border-left: 4px solid #cf222e; }
.clue h4 { margin: 0 0 8px; font-size: 1.05em;
           display: flex; justify-content: space-between; align-items: baseline; }
.clue .verdict-tag { font-size: 0.75em; font-weight: 700;
                     padding: 3px 10px; border-radius: 999px;
                     letter-spacing: 0.5px;
                     background: #d1f4d8; color: #1a7f37; }
.clue.miss .verdict-tag { background: #ffd3d3; color: #cf222e; }
.field { margin: 6px 0; }
.field-label { display: inline-block; font-weight: 600;
                color: #555; min-width: 140px; vertical-align: top; }
.field-value { display: inline-block; max-width: calc(100% - 150px); }
code { background: #f0f0f0; padding: 1px 6px; border-radius: 4px;
       font-size: 0.92em; word-break: break-word; }
.form { font-family: "SF Mono", Menlo, Consolas, monospace;
        font-size: 0.85em; background: #f5f5f5; padding: 8px 12px;
        border-radius: 6px; word-break: break-word;
        white-space: pre-wrap; line-height: 1.4; margin: 6px 0;
        border-left: 3px solid #888; }
.template-id { font-family: "SF Mono", Menlo, Consolas, monospace;
               background: #e9f2ff; color: #0050a8; padding: 2px 8px;
               border-radius: 4px; font-size: 0.85em; }
table.toc { width: 100%; border-collapse: collapse; margin: 12px 0; }
table.toc td, table.toc th { padding: 4px 8px;
                                border-bottom: 1px solid #eee;
                                font-size: 0.9em; }
table.toc th { text-align: left; background: #f0f0f0; }
table.toc a { text-decoration: none; color: inherit; }
table.toc a:hover { text-decoration: underline; }
</style>
"""

out = []
out.append("<!DOCTYPE html><html><head>")
out.append('<meta charset="utf-8">')
out.append("<title>Held-out catalog run — Times 29400-29449</title>")
out.append(style)
out.append("</head><body>")
out.append("<h1>Held-out catalog run — Times 29400-29449</h1>")
out.append('<div class="summary">')
out.append(f'<span class="hit">Real catalog HITs: {n_real}/{n_total} '
            f'({100*n_real/n_total:.1f}%)</span>')
out.append(f'<span class="syn">Degenerate synonym-template hits: {n_syn} (excluded)</span>')
out.append(f'<span class="miss">MISSes: {n_miss}</span>')
out.append('<br><span style="color:#666">v0.1 catalog: 32 templates from '
           '60 strict-PASS forms (29500-29530 + 29534). Held-out: 50 '
           'puzzles 29400-29449 not in training.</span>')
out.append("</div>")

out.append('<div class="note-box">')
out.append("<strong>Note:</strong> the <code>synonym</code> template is "
           "degenerate — it fits any clue where the answer is in the "
           "synonyms DB for some clue span. The 474 hits that fired on "
           "this template are false positives (the audit confirmed this — "
           "they capture no wordplay). They are <strong>excluded</strong> "
           "from the real hit count above.")
out.append("</div>")

# Templates that fired
out.append("<h2>Templates that fired (real templates only)</h2>")
out.append("<table class='toc'><thead><tr>"
            "<th>Template</th><th>Hits</th></tr></thead><tbody>")
sorted_templates = sorted(hits_by_template.items(),
                            key=lambda kv: -len(kv[1]))
for tid, items in sorted_templates:
    out.append(f"<tr><td><code>{html.escape(tid)}</code></td>"
                f"<td>{len(items)}</td></tr>")
out.append("</tbody></table>")

# Per-template detail
out.append("<h2>Per-clue detail (all 130 real hits)</h2>")
for tid, items in sorted_templates:
    out.append(f"<h3 id='{html.escape(tid)}'>"
                f"<span class='template-id'>{html.escape(tid)}</span> "
                f"<span style='color:#666;font-weight:normal'>"
                f"({len(items)} hits)</span></h3>")
    for r in items:
        cn = r["clue_number"]
        d = r["direction"][0]
        db_t = db_wpt.get((r["puzzle_number"], cn, r["direction"]),
                            "—") or "—"
        out.append(f"<div class='clue'>")
        out.append(f"<h4>{r['answer']} "
                    f"<small style='color:#888;font-weight:normal'>"
                    f"(times {r['puzzle_number']} {cn}{d})</small>"
                    f"<span class='verdict-tag'>HIT</span></h4>")
        out.append(f"<div class='field'><span class='field-label'>"
                    f"Clue:</span><span class='field-value'>"
                    f"{html.escape(r['clue_text'])}</span></div>")
        out.append(f"<div class='field'><span class='field-label'>"
                    f"DB wordplay type:</span><span class='field-value'>"
                    f"<code>{html.escape(db_t)}</code></span></div>")
        out.append(f"<div class='field'><span class='field-label'>"
                    f"Template:</span><span class='field-value'>"
                    f"<code>{html.escape(tid)}</code></span></div>")
        if r.get("form"):
            out.append("<div class='field'><span class='field-label'>"
                        "Form:</span></div>")
            out.append(f"<div class='form'>"
                        f"{fmt_form(r['form']['tree'])}</div>")
            out.append(f"<div class='field'><span class='field-label'>"
                        f"Definition:</span><span class='field-value'>"
                        f"<code>"
                        f"{html.escape(r['form']['definition']['phrase'])}"
                        f"</code></span></div>")
            link_words = r['form'].get('link_words', [])
            if link_words:
                out.append(f"<div class='field'><span class='field-label'>"
                            f"Link words:</span><span class='field-value'>"
                            f"{html.escape(', '.join(link_words))}"
                            f"</span></div>")
        out.append("</div>")

# Sample of misses
out.append("<h2>Sample of misses (first 30)</h2>")
out.append('<div class="note-box">No template in the v0.1 catalog '
           'fits these clues. They are the worklist for catalog growth.</div>')
for r in all_misses[:30]:
    cn = r["clue_number"]
    d = r["direction"][0]
    db_t = db_wpt.get((r["puzzle_number"], cn, r["direction"]), "—") or "—"
    out.append(f"<div class='clue miss'>")
    out.append(f"<h4>{r['answer']} "
                f"<small style='color:#888;font-weight:normal'>"
                f"(times {r['puzzle_number']} {cn}{d})</small>"
                f"<span class='verdict-tag'>MISS</span></h4>")
    out.append(f"<div class='field'><span class='field-label'>"
                f"Clue:</span><span class='field-value'>"
                f"{html.escape(r['clue_text'])}</span></div>")
    out.append(f"<div class='field'><span class='field-label'>"
                f"DB wordplay type:</span><span class='field-value'>"
                f"<code>{html.escape(db_t)}</code></span></div>")
    out.append("</div>")

out.append("</body></html>")

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n".join(out))
import os
print(f"Written {OUTPUT} ({os.path.getsize(OUTPUT):,} bytes)")
print(f"Real hits: {n_real}; Synonym (excluded): {n_syn}; Misses: {n_miss}")
