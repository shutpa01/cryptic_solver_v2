"""Build an HTML report for Times 29534."""
import json
import sqlite3
import html

with open('prototypes/universal_form_v2/runs/shadow_pipeline_29534.json') as f:
    data = json.load(f)

conn = sqlite3.connect('data/clues_master.db')
db_rows = conn.execute(
    'SELECT clue_number, direction, wordplay_type '
    'FROM clues WHERE source=? AND puzzle_number=?',
    ('times', '29534')).fetchall()
db_wpt = {(r[0], r[1]): r[2] for r in db_rows}

results = data['per_clue']
results.sort(key=lambda r: (r['direction'], int(r['clue_number'])))

v_counts = {"PASS": 0, "FAIL": 0, "NO_FORM": 0}
for r in results:
    v = r.get('verdict_pass2')
    if v is None:
        v_counts['NO_FORM'] += 1
    else:
        v_counts[v.get('verdict')] += 1


def fmt_form(n):
    if n is None:
        return ''
    op = n.get('operation')
    has_sources = bool(n.get('sources'))
    if op in ('literal', 'synonym', 'abbreviation', 'positional', 'raw') and not has_sources:
        val = n.get('value') or ''
        src = n.get('source_word') or ''
        kind = n.get('positional_kind') or ''
        kind_s = f"[{kind}]" if kind else ""
        return f"{op}{kind_s}({html.escape(val)} ← '{html.escape(src)}')"
    if op == 'homophone' and not has_sources:
        val = n.get('value') or ''
        src = n.get('source_word') or ''
        return f"homophone({html.escape(val)} ← '{html.escape(src)}')"
    ind = n.get('indicator')
    ind_s = f" [{html.escape(ind)}]" if ind else ""
    children = ", ".join(fmt_form(c) for c in (n.get('sources') or []))
    return f"{op}{ind_s}({children})"


style = """
<style>
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 24px;
       color: #1a1a1a; background: #fafafa; line-height: 1.5; }
h1 { margin: 0 0 8px; }
h2 { margin: 32px 0 12px; padding-bottom: 6px;
     border-bottom: 2px solid #ddd; }
.summary { background: #fff; padding: 16px 20px; border-radius: 8px;
           border: 1px solid #ddd; margin: 12px 0 24px;
           font-size: 1.05em; }
.summary span { display: inline-block; margin-right: 24px; }
.summary .pass { color: #1a7f37; font-weight: 600; }
.summary .fail { color: #cf222e; font-weight: 600; }
.summary .none { color: #9a6700; font-weight: 600; }
.clue { background: #fff; border: 1px solid #ddd; border-radius: 8px;
        padding: 16px 20px; margin: 12px 0; }
.clue.pass { border-left: 4px solid #1a7f37; }
.clue.fail { border-left: 4px solid #cf222e; }
.clue.no_form { border-left: 4px solid #9a6700; }
.clue h3 { margin: 0 0 8px; font-size: 1.1em;
           display: flex; justify-content: space-between; align-items: baseline; }
.clue .verdict-tag { font-size: 0.75em; font-weight: 700;
                     padding: 3px 10px; border-radius: 999px;
                     letter-spacing: 0.5px; }
.verdict-tag.pass { background: #d1f4d8; color: #1a7f37; }
.verdict-tag.fail { background: #ffd3d3; color: #cf222e; }
.verdict-tag.no_form { background: #ffe9b3; color: #9a6700; }
.field { margin: 6px 0; }
.field-label { display: inline-block; font-weight: 600;
                color: #555; min-width: 160px; vertical-align: top; }
.field-value { display: inline-block; max-width: calc(100% - 170px); }
code { background: #f0f0f0; padding: 1px 6px; border-radius: 4px;
       font-size: 0.92em; word-break: break-word; }
.form { font-family: "SF Mono", Menlo, Consolas, monospace;
        font-size: 0.85em; background: #f5f5f5; padding: 8px 12px;
        border-radius: 6px; word-break: break-word;
        white-space: pre-wrap; line-height: 1.4; margin: 6px 0;
        border-left: 3px solid #888; }
.checks { margin: 8px 0 4px; padding-left: 0; list-style: none; }
.checks li { font-size: 0.92em; padding: 2px 0;
             font-family: "SF Mono", Menlo, Consolas, monospace; }
.check-pass::before { content: "✓ "; color: #1a7f37; font-weight: 700; }
.check-fail::before { content: "✗ "; color: #cf222e; font-weight: 700; }
.wpt-match { color: #1a7f37; font-weight: 600; }
.wpt-mismatch { color: #cf222e; font-weight: 600; }
.note { font-style: italic; color: #555; margin-top: 6px;
        background: #fff8dd; padding: 6px 10px; border-radius: 4px;
        border-left: 3px solid #d4a72c; font-size: 0.92em; }
table.toc { width: 100%; border-collapse: collapse; margin: 12px 0; }
table.toc td, table.toc th { padding: 4px 8px; border-bottom: 1px solid #eee;
                                font-size: 0.9em; }
table.toc th { text-align: left; background: #f0f0f0; }
table.toc a { text-decoration: none; color: inherit; }
table.toc a:hover { text-decoration: underline; }
table.toc tr.pass td:first-child { color: #1a7f37; font-weight: 600; }
table.toc tr.fail td:first-child { color: #cf222e; font-weight: 600; }
table.toc tr.no_form td:first-child { color: #9a6700; font-weight: 600; }
</style>
"""

out = []
out.append('<!DOCTYPE html><html><head>')
out.append('<meta charset="utf-8">')
out.append('<title>Times 29534 - Form Pipeline Report</title>')
out.append(style)
out.append('</head><body>')
out.append('<h1>Times 29534 - Form Pipeline Report</h1>')
out.append('<div class="summary">')
out.append(f'<span class="pass">PASS: {v_counts["PASS"]}/28 ({100*v_counts["PASS"]/28:.0f}%)</span>')
out.append(f'<span class="fail">FAIL: {v_counts["FAIL"]}</span>')
out.append(f'<span class="none">NO_FORM: {v_counts["NO_FORM"]}</span>')
out.append('<br><span style="color:#666">Run: 2026-05-05 - Held-out puzzle (not in training sample 29500-29530)</span>')
out.append('</div>')

out.append('<h2>Index</h2>')
out.append('<table class="toc"><thead><tr>'
            '<th>Verdict</th><th>Clue</th><th>Answer</th>'
            '<th>DB wordplay type</th><th>Pipeline pattern</th></tr></thead><tbody>')
for r in results:
    cn = r['clue_number']
    d = r['direction'][0]
    v = r.get('verdict_pass2')
    verdict = 'NO_FORM' if v is None else v.get('verdict')
    db_t = db_wpt.get((cn, r['direction']), '-') or '-'
    pat = r.get('pattern') or '-'
    cls = verdict.lower()
    out.append(f'<tr class="{cls}"><td>{verdict}</td>'
               f'<td><a href="#c{cn}{d}">{cn}{d}</a></td>'
               f'<td>{r["answer"]}</td>'
               f'<td><code>{db_t}</code></td>'
               f'<td><code>{pat}</code></td></tr>')
out.append('</tbody></table>')

out.append('<h2>Clue-by-clue detail</h2>')

notes_map = {
    ('16', 'across'): 'Should be a head-deletion of M from MARCH. The {m} curly notation was identified but the deletion node never got built into the form.',
    ('22', 'down'): '"leaders of" was tagged as acrostic indicator but the form ended up as plain charade - the acrostic op did not fire because pieces W, H were already typed as literals.',
    ('20', 'across'): 'Caret marker in HO⁁RSE was not stripped, splitting HORSE into two pieces. Should be container(HORSE, A) = HOARSE.',
    ('25', 'across'): 'Caret marker in TOR⁁Y splits TORY into TOR + Y.',
    ('3', 'down'): 'Caret marker in REEL⁁S splits REELS into REEL + S.',
    ('6', 'down'): 'Caret marker in S⁁T splits ST into S + T.',
    ('8', 'down'): 'Caret marker in T⁁OILER splits TOILER into T + OILER.',
    ('18', 'down'): 'Caret marker in PL⁁ANT splits PLANT into PL + ANT. "cuts" wrongly tagged as deletion - it is the container indicator here.',
    ('15', 'down'): 'Blog has M1 (uppercase letter + digit). Parser only matches uppercase letter sequences for PIECEs, so it picked up M but dropped the 1. The 1 reverses to I (numeral-as-letter trick).',
    ('9', 'across'): 'Pure double-definition. Blog says "Cryptic with reference to tennis" - DD detector requires literal "Double definition" or "DD".',
    ('23', 'across'): 'Same as above - "Two meanings" not detected as DD.',
    ('11', 'across'): 'Unusual instruction-prose blog. Even hard for a human to parse mechanically.',
    ('13', 'across'): 'Blog notation "P AND A" with literal AND between letters not parsed as charade(P, AND, A).',
    ('7', 'down'): 'Blog typo: "CAMERAMAN USED AT" instead of "CAMERAMEN USED AT" (the clue says CAMERAMEN). Letter sets do not match - assembly fails.',
}

for r in results:
    cn = r['clue_number']
    d = r['direction'][0]
    v_obj = r.get('verdict_pass2')
    verdict = 'NO_FORM' if v_obj is None else v_obj.get('verdict')
    cls = verdict.lower()
    db_t = db_wpt.get((cn, r['direction']), '-') or '-'
    pipe_t = r.get('wordplay_type') or '-'
    pat = r.get('pattern') or '-'
    wpt_match_class = ''
    if db_t != '-' and pipe_t != '-':
        if db_t == pipe_t or (db_t in pipe_t) or (pipe_t in db_t):
            wpt_match_class = 'wpt-match'
        else:
            wpt_match_class = 'wpt-mismatch'

    out.append(f'<div class="clue {cls}" id="c{cn}{d}">')
    out.append(f'<h3>{cn}{d}. {r["answer"]}'
               f'<span class="verdict-tag {cls}">{verdict}</span></h3>')
    out.append(f'<div class="field"><span class="field-label">Clue:</span>'
               f'<span class="field-value">{html.escape(r["clue_text"])}</span></div>')
    out.append(f'<div class="field"><span class="field-label">Blog:</span>'
               f'<span class="field-value">{html.escape(r.get("blog") or "")}</span></div>')
    out.append(f'<div class="field"><span class="field-label">Definition (DB):</span>'
               f'<span class="field-value"><code>{html.escape(r.get("definition_db") or "-")}</code></span></div>')
    out.append(f'<div class="field"><span class="field-label">Wordplay type (DB):</span>'
               f'<span class="field-value {wpt_match_class}"><code>{html.escape(db_t)}</code></span></div>')
    out.append(f'<div class="field"><span class="field-label">Pipeline classification:</span>'
               f'<span class="field-value {wpt_match_class}"><code>{html.escape(pipe_t)}</code> '
               f'(trial fired: <code>{html.escape(pat)}</code>)</span></div>')

    if r.get('form'):
        form_str = fmt_form(r['form']['tree'])
        out.append('<div class="field"><span class="field-label">Form:</span></div>')
        out.append(f'<div class="form">{form_str}</div>')

    m = r.get('mapping') or {}
    pieces = [t for t in m.get('tags', []) if t['role'] == 'piece']
    inds = [t for t in m.get('tags', []) if t['role'] == 'indicator']
    if pieces:
        ps = "; ".join(f"<code>{html.escape(t.get('value') or '?')}</code> ← '{html.escape(' '.join(t['words']))}'" for t in pieces)
        out.append(f'<div class="field"><span class="field-label">Pieces:</span>'
                    f'<span class="field-value">{ps}</span></div>')
    if inds:
        isr = "; ".join(f"'<code>{html.escape(' '.join(t['words']))}</code>' [{t.get('operation') or '?'}]" for t in inds)
        out.append(f'<div class="field"><span class="field-label">Indicators:</span>'
                    f'<span class="field-value">{isr}</span></div>')

    if v_obj:
        out.append('<ul class="checks">')
        for c in v_obj.get('checks', []):
            cls2 = 'check-pass' if c['status'] == 'pass' else 'check-fail'
            out.append(f'<li class="{cls2}"><code>{c["name"]}</code>: '
                       f'{html.escape(c["detail"]) or "(no detail)"}</li>')
        out.append('</ul>')

    note = notes_map.get((cn, r['direction']))
    if note:
        out.append(f'<div class="note">{html.escape(note)}</div>')

    out.append('</div>')

out.append('</body></html>')

path = 'prototypes/universal_form_v2/runs/report_29534.html'
with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))

import os
size = os.path.getsize(path)
print(f"Written {path} ({size:,} bytes)")
