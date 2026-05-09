"""Render run results as a self-contained HTML report.

Usage:
    python -m prototypes.universal_form_v2.html_report > review.html

Reads the JSON files in prototypes/universal_form_v2/runs/ and emits a
single HTML page with one card per clue. Filter buttons + free-text
search at the top. No comparison columns - just the form-based result.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .schema import Form, Node, Definition, LEAF_OPERATIONS
from .renderer import render, wordplay_type

DEFAULT_PUZZLES = ["31132", "31138", "31150"]
RUNS = Path(__file__).resolve().parent / "runs"


def _resolve_run_files(args):
    """Resolve which JSON run files to render.

    Without args: the default DT test bed.
    With args: each arg is either '<basename>.json' (looked up in runs/)
    or just the puzzle id (with .json appended).
    """
    if not args:
        return [(p, RUNS / f"{p}.json") for p in DEFAULT_PUZZLES]
    out = []
    for a in args:
        if a.endswith(".json"):
            path = RUNS / a
            label = a[:-5]
        else:
            path = RUNS / f"{a}.json"
            label = a
        out.append((label, path))
    return out


def esc(s) -> str:
    if s is None:
        return ""
    return html.escape(str(s))


def _form_from_dict(d: dict) -> Form:
    """Reconstruct a Form from the JSON dict (the runner serialised it)."""
    return Form(
        tree=_node_from_dict(d["tree"]),
        definition=Definition(
            phrase=d["definition"]["phrase"],
            answer=d["definition"]["answer"],
        ),
        link_words=d.get("link_words", []),
        is_and_lit=d.get("is_and_lit", False),
        flags=d.get("flags", []),
    )


def _node_from_dict(d: dict) -> Node:
    n = Node(
        operation=d["operation"],
        indicator=d.get("indicator"),
        value=d.get("value"),
        source_word=d.get("source_word"),
        positional_kind=d.get("positional_kind"),
        deletion_kind=d.get("deletion_kind"),
        acrostic_kind=d.get("acrostic_kind"),
        flags=d.get("flags", []),
    )
    n.sources = [_node_from_dict(s) for s in d.get("sources") or []]
    return n


# --- HTML pieces -----------------------------------------------------------

CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
       Helvetica, Arial, sans-serif; margin: 0; padding: 16px;
       background: #fafafa; color: #222; line-height: 1.4; }
h1 { margin: 0 0 4px 0; font-size: 18px; }
.subtitle { color: #666; font-size: 12px; margin-bottom: 16px; }

.summary { background: #fff; padding: 12px; border-radius: 6px;
           border: 1px solid #ddd; margin-bottom: 12px; }
.stat-row { padding: 4px 0; font-size: 13px; }
.stat { display: inline-block; padding: 2px 8px; margin: 2px;
        background: #f0f0f0; border-radius: 4px; font-size: 12px; }
.stat b { color: #000; }

.toolbar { background: #fff; padding: 10px; border-radius: 6px;
           border: 1px solid #ddd; margin-bottom: 12px;
           position: sticky; top: 8px; z-index: 5; }
.toolbar button { padding: 6px 10px; margin: 2px; border: 1px solid #ccc;
                  background: #fff; border-radius: 4px; cursor: pointer;
                  font-size: 12px; font-family: inherit; }
.toolbar button:hover { background: #eef; }
.toolbar button.active { background: #345; color: #fff; border-color: #345; }
.toolbar input { padding: 6px 8px; margin: 2px; border: 1px solid #ccc;
                 border-radius: 4px; font-size: 12px; width: 240px;
                 font-family: inherit; }
.toolbar label { font-size: 12px; color: #666; margin-left: 6px; }

.card { background: #fff; border: 1px solid #ddd; border-radius: 6px;
        margin-bottom: 12px; padding: 12px; }
.card-head { display: flex; justify-content: space-between;
             align-items: flex-start; gap: 8px; margin-bottom: 6px;
             flex-wrap: wrap; }
.card-h-l .puzzle { color: #888; font-size: 11px; margin-right: 8px;
                    font-family: ui-monospace, Menlo, Consolas, monospace; }
.card-h-l .num { font-weight: bold; color: #444; margin-right: 8px; }
.card-h-l .ans { font-family: ui-monospace, Menlo, Consolas, monospace;
                 font-weight: bold; font-size: 14px; }
.card-h-r .pill { display: inline-block; padding: 3px 8px; margin: 2px;
                  border-radius: 12px; font-size: 11px;
                  font-family: ui-monospace, Menlo, Consolas, monospace;
                  border: 1px solid transparent; }
.pill.v-pass { background: #d0f5d6; color: #014a13; border-color: #80d090; }
.pill.v-fail { background: #ffd5d5; color: #6a0000; border-color: #e08080; }
.pill.v-noform { background: #eee; color: #555; border-color: #ccc; }
.pill.wptype { background: #e8e0ff; color: #2a0066; border-color: #b090e0;
               font-weight: bold; }

.clue { font-size: 14px; color: #111; margin: 4px 0; font-style: italic; }
.meta { color: #888; font-size: 11px; margin-bottom: 8px;
        font-family: ui-monospace, Menlo, Consolas, monospace; }

.section { background: #fcfcfc; border: 1px solid #eee; border-radius: 4px;
           padding: 8px; margin-top: 6px; }
.section-label { font-weight: bold; font-size: 12px; color: #444;
                 margin-bottom: 4px; }

pre.rendered { font-family: ui-monospace, Menlo, Consolas, monospace;
               white-space: pre-wrap; word-wrap: break-word;
               margin: 0; padding: 8px; background: #f6f6f6;
               border-radius: 4px; font-size: 12px; line-height: 1.4; }

.checks { display: flex; flex-direction: column; gap: 2px;
          margin-top: 4px; }
.check { display: grid; grid-template-columns: 22px 180px 1fr; gap: 6px;
         padding: 3px 6px; font-size: 11px;
         font-family: ui-monospace, Menlo, Consolas, monospace; }
.check.ck-ok { background: #e8f8ec; }
.check.ck-bad { background: #ffe5e5; }
.ck-mark { font-weight: bold; }
.ck-name { color: #555; }
.ck-detail { color: #222; word-wrap: break-word; }

.tree { font-family: ui-monospace, Menlo, Consolas, monospace;
        font-size: 11px; color: #444; padding: 8px;
        background: #f6f6f6; border-radius: 4px; white-space: pre; }

.flags { font-family: ui-monospace, Menlo, Consolas, monospace;
         font-size: 11px; color: #666; margin-top: 4px; }
.flag { display: inline-block; padding: 1px 6px; margin: 2px;
        background: #fff5cc; color: #5a4400;
        border: 1px solid #f0c060; border-radius: 3px; }
.banner-soft { padding: 6px 10px; background: #f4f4f4; color: #666;
               border-radius: 4px; font-size: 11px; font-style: italic; }
"""

JS = """
function applyFilter() {
  const cat = document.querySelector('button.active')?.dataset.cat || 'all';
  const q = (document.getElementById('q').value || '').toLowerCase();
  let shown = 0;
  for (const card of document.querySelectorAll('.card')) {
    const cardCat = card.dataset.cat || '';
    const text = (card.dataset.text || '').toLowerCase();
    const catOk = (cat === 'all' || cardCat.split(' ').includes(cat));
    const qOk = (q === '' || text.includes(q));
    card.style.display = (catOk && qOk) ? '' : 'none';
    if (catOk && qOk) shown += 1;
  }
  document.getElementById('shown').textContent = String(shown);
}
document.addEventListener('DOMContentLoaded', () => {
  for (const b of document.querySelectorAll('.toolbar button')) {
    b.addEventListener('click', () => {
      for (const x of document.querySelectorAll('.toolbar button'))
        x.classList.remove('active');
      b.classList.add('active');
      applyFilter();
    });
  }
  document.getElementById('q').addEventListener('input', applyFilter);
});
"""


def _tree_to_text(node, depth=0) -> str:
    """Compact tree text for the diagnostic section."""
    indent = "  " * depth
    op = node.operation
    if op in LEAF_OPERATIONS:
        val_repr = repr(node.value or "")
        line = f"{op}({val_repr}"
        if node.source_word and node.source_word != node.value:
            line += f" src={node.source_word!r}"
        line += ")"
        if node.positional_kind:
            line += f" kind={node.positional_kind}"
        return indent + line
    line = op
    if node.indicator:
        line += f" [{node.indicator!r}]"
    if node.deletion_kind:
        line += f" kind={node.deletion_kind}"
    if node.acrostic_kind:
        line += f" kind={node.acrostic_kind}"
    out = [indent + line]
    for c in node.sources or []:
        out.append(_tree_to_text(c, depth + 1))
    return "\n".join(out)


def render_card(c: dict) -> str:
    status = c.get("status") or "NO_FORM"
    status_class = {"PASS": "v-pass", "FAIL": "v-fail",
                    "NO_FORM": "v-noform"}.get(status, "v-noform")

    form = None
    if c.get("form"):
        form = _form_from_dict(c["form"])

    wp_type = wordplay_type(form) if form else "-"
    rendered = render(form) if form else "(no form built)"

    cat = status.lower().replace("_", "-")
    text_for_search = " ".join([
        c["clue_text"] or "", c["answer"] or "", wp_type,
        " ".join(c.get("flags") or []),
    ])

    parts = []
    parts.append(f'<div class="card" data-cat="{cat}" '
                 f'data-text="{esc(text_for_search)}">')
    parts.append('<div class="card-head">')
    parts.append('<div class="card-h-l">')
    parts.append(f'<span class="puzzle">{esc(c["puzzle"])}</span>')
    parts.append(f'<span class="num">'
                 f'{esc(c["clue_number"])}{esc(c["direction"][:1])}</span>')
    parts.append(f'<span class="ans">{esc(c["answer"])}</span>')
    parts.append('</div>')
    parts.append('<div class="card-h-r">')
    parts.append(f'<span class="pill wptype">{esc(wp_type)}</span>')
    parts.append(f'<span class="pill {status_class}">{esc(status)}</span>')
    parts.append('</div>')
    parts.append('</div>')

    parts.append(f'<div class="clue">{esc(c["clue_text"])}</div>')

    if form:
        parts.append('<div class="section">')
        parts.append('<div class="section-label">Explanation</div>')
        parts.append(f'<pre class="rendered">{esc(rendered)}</pre>')
        parts.append('</div>')
        parts.append('<div class="section">')
        parts.append('<div class="section-label">Form tree</div>')
        parts.append(f'<div class="tree">{esc(_tree_to_text(form.tree))}</div>')
        if form.definition.phrase:
            parts.append('<div class="meta" style="margin-top:6px;">'
                         f'definition: "{esc(form.definition.phrase)}" -> '
                         f'{esc(form.definition.answer)}</div>')
        if form.link_words:
            parts.append('<div class="meta">link_words: '
                         f'{esc(form.link_words)}</div>')
        parts.append('</div>')

    if c.get("verdict"):
        parts.append('<div class="section">')
        parts.append('<div class="section-label">Verifier checks</div>')
        parts.append('<div class="checks">')
        for chk in c["verdict"]["checks"]:
            css = "ck-ok" if chk["status"] == "pass" else "ck-bad"
            mark = "+" if chk["status"] == "pass" else "X"
            parts.append(f'<div class="check {css}">'
                         f'<span class="ck-mark">{mark}</span>'
                         f'<span class="ck-name">{esc(chk["name"])}</span>'
                         f'<span class="ck-detail">{esc(chk.get("detail",""))}</span>'
                         f'</div>')
        parts.append('</div>')
        parts.append('</div>')

    if not form:
        notes = c.get("notes") or ""
        parts.append(f'<div class="banner-soft">{esc(notes)}</div>')

    if c.get("flags"):
        parts.append('<div class="flags">flags: ')
        for f in sorted(set(c["flags"])):
            parts.append(f'<span class="flag">{esc(f)}</span>')
        parts.append('</div>')

    parts.append('</div>')
    return "\n".join(parts)


def render_page(run_files=None) -> str:
    if run_files is None:
        run_files = _resolve_run_files([])
    cards = []
    counts = {"PASS": 0, "FAIL": 0, "NO_FORM": 0}
    wp_counts = {}
    labels = []
    for label, path in run_files:
        if not path.exists():
            continue
        labels.append(label)
        d = json.loads(path.read_text())
        for c in d["per_clue"]:
            c["puzzle"] = label
            cards.append(render_card(c))
            counts[c["status"]] = counts.get(c["status"], 0) + 1
            if c.get("form"):
                form = _form_from_dict(c["form"])
                wp = wordplay_type(form)
                wp_counts[wp] = wp_counts.get(wp, 0) + 1

    total = sum(counts.values())
    summary_pills = "".join(
        f'<span class="stat">{k}: <b>{v}</b></span>'
        for k, v in counts.items() if v
    )

    wp_pills = "".join(
        f'<span class="stat">{esc(k)}: <b>{v}</b></span>'
        for k, v in sorted(wp_counts.items(), key=lambda x: -x[1])
    )

    toolbar_buttons = "".join([
        '<button data-cat="all" class="active">All</button>',
        '<button data-cat="pass">PASS</button>',
        '<button data-cat="fail">FAIL</button>',
        '<button data-cat="no-form">NO_FORM</button>',
    ])

    label_str = ", ".join(labels)
    out = []
    out.append('<!DOCTYPE html><html><head><meta charset="utf-8">')
    out.append(f'<title>Universal form v0 - {label_str}</title>')
    out.append(f'<style>{CSS}</style>')
    out.append('</head><body>')
    out.append(f'<h1>Universal form v0 - {total} clues '
               f'({label_str})</h1>')
    out.append('<div class="subtitle">Per-clue review. '
               'Showing <span id="shown">'
               f'{total}</span>. Read-only.</div>')
    out.append('<div class="summary">')
    out.append(f'<div class="stat-row"><b>Verdicts:</b> {summary_pills}</div>')
    out.append(f'<div class="stat-row"><b>Wordplay type paths:</b> '
               f'{wp_pills}</div>')
    out.append('</div>')
    out.append('<div class="toolbar">')
    out.append(toolbar_buttons)
    out.append('<input id="q" type="search" '
               'placeholder="Search clue / answer / wordplay-type / flag...">')
    out.append('</div>')
    out.extend(cards)
    out.append(f'<script>{JS}</script>')
    out.append('</body></html>')
    return "\n".join(out)


def main():
    args = sys.argv[1:]
    run_files = _resolve_run_files(args)
    page = render_page(run_files)
    sys.stdout.buffer.write(page.encode("utf-8"))


if __name__ == "__main__":
    main()
