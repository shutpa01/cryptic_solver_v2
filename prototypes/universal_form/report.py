"""HTML report — self-contained, browser-friendly view of a simulation run.

Takes the list of ClueResult dataclasses produced by harness.run_query and
emits a single HTML file with:

  - Top bar: summary stats + filter buttons
  - Per-clue cards with:
      headline (clue#/answer + NEW vs OLD verdict pills + translation flag)
      clue text
      side-by-side: NEW (rendered explanation + check list)
                    OLD (original ai_explanation + check list)
      translation notes (when present)

The filter bar slices to: all / translatable only / prose-only / wins for new
/ losses for new / disagreements / translation gaps.

No external CSS or JS — everything is inline so the file works offline and
can be saved/shared as a single artifact.
"""
from __future__ import annotations

import html
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .harness import ClueResult


# --- HTML helpers ----------------------------------------------------------

def _esc(s) -> str:
    if s is None:
        return ""
    return html.escape(str(s))


def _verdict_class(v) -> str:
    if not v:
        return "v-none"
    vl = v.lower()
    # PASS -> green, FAIL -> red, OLD's HIGH/MEDIUM/LOW kept for reference
    return f"v-{vl}"


def _flag_class(flag: str) -> str:
    return {
        "clean": "flag-clean",
        "partial_translation": "flag-partial",
        "requires_re_derivation": "flag-noform",
        "malformed": "flag-bad",
        "unknown_op": "flag-unknown",
    }.get(flag, "flag-other")


def _flag_label(flag: str) -> str:
    return {
        "clean": "translatable",
        "partial_translation": "translatable (partial)",
        "requires_re_derivation": "prose-only — no form",
        "malformed": "components malformed",
        "unknown_op": "translatable (unknown op)",
    }.get(flag, flag)


def _check_row(c: dict) -> str:
    status = c.get("status", "?")
    name = c.get("check") or c.get("name") or "?"
    detail = c.get("detail", "")
    cls = {"pass": "ck-ok", "fail": "ck-bad",
           "verified": "ck-ok", "wrong": "ck-bad",
           "unverifiable": "ck-meh"}.get(status, "ck-other")
    mark = {"pass": "+", "fail": "X",
            "verified": "+", "wrong": "X",
            "unverifiable": "?"}.get(status, "?")
    return (f'<div class="check {cls}">'
            f'<span class="ck-mark">{mark}</span>'
            f'<span class="ck-name">{_esc(name)}</span>'
            f'<span class="ck-detail">{_esc(detail)}</span>'
            f'</div>')


def _prod_cycle_block(r) -> str:
    """The third panel — production-cycle simulation."""
    if r.new_verdict is None:
        return ('<div class="prod-cycle">'
                '<div class="rendered-label">Production cycle:</div>'
                '<div class="banner-soft">Not applicable - no NEW form was '
                'built.</div></div>')
    if r.new_verdict == "PASS":
        return ('<div class="prod-cycle">'
                '<div class="rendered-label">Production cycle:</div>'
                '<div class="prod-pass">No enrichments needed - form '
                'already PASSES.</div></div>')

    cf_label = (f'<span class="pill {_verdict_class(r.counterfactual_verdict)}">'
                f'after enrichment: {_esc(r.counterfactual_verdict or "?")}</span>')

    if not r.proposals:
        body = ('<div class="banner-soft">FAIL but no DB-gap proposals - '
                'failures are structural (assembly mismatch, residue '
                'unaccounted, mis-tagged literal, etc.). Production '
                'would not auto-enrich; the parse needs human re-work.'
                '</div>')
    else:
        rows = []
        for p in r.proposals:
            status = p.get("dedupe_status", "new")
            cls = {"new": "prop-new",
                   "already_in_db": "prop-dup",
                   "already_pending": "prop-pending"}.get(status, "")
            status_label = {"new": "NEW (would queue)",
                            "already_in_db": "ALREADY IN DB",
                            "already_pending": "ALREADY PENDING"
                            }.get(status, status)
            row = (f'<div class="proposal {cls}">'
                   f'<span class="prop-status">{_esc(status_label)}</span>'
                   f'<span class="prop-type">{_esc(p.get("type", "?"))}</span>'
                   f'<span class="prop-word">"{_esc(p.get("word", ""))}"</span>'
                   f'<span class="prop-arrow">-&gt;</span>'
                   f'<span class="prop-letters">{_esc(p.get("letters", ""))}</span>'
                   f'</div>')
            rows.append(row)
        body = '<div class="proposals">' + "".join(rows) + '</div>'

    cf_explainer = ""
    if r.counterfactual_verdict == "PASS":
        cf_explainer = ('<div class="prod-pass">If approved, the form '
                        'would PASS on the next run.</div>')
    elif r.counterfactual_verdict == "FAIL":
        # Find what's still failing in the counterfactual
        still_fails = [c for c in r.counterfactual_checks
                       if c.get("status") == "fail"
                       and not c.get("enrichment_proposal")]
        if still_fails:
            failures = [f'{c.get("check")}: {c.get("detail")}'
                        for c in still_fails[:5]]
            cf_explainer = ('<div class="prod-fail">Even after enrichment, '
                            'the form would still FAIL on:'
                            '<ul>'
                            + "".join(f'<li>{_esc(f)}</li>' for f in failures)
                            + '</ul></div>')
        else:
            cf_explainer = ('<div class="prod-fail">After enrichment the '
                            'form still FAILS (no further DB-gap fixes '
                            'available).</div>')

    return (f'<div class="prod-cycle">'
            f'<div class="rendered-label">'
            f'Production cycle: proposed enrichments {cf_label}</div>'
            f'{body}'
            f'{cf_explainer}'
            f'</div>')


def _categorise(r: "ClueResult") -> str:
    """Tags applied to each card for the filter buttons (data-cat)."""
    cats = []
    if r.translation_flag in ("clean", "partial_translation",
                              "unknown_op"):
        cats.append("translatable")
    if r.translation_flag == "requires_re_derivation":
        cats.append("prose-only")
    if r.translation_flag in ("partial_translation", "unknown_op"):
        cats.append("gaps")
    if r.new_verdict == "PASS":
        cats.append("new-pass")
    elif r.new_verdict == "FAIL":
        cats.append("new-fail")
    if r.new_verdict and r.old_verdict:
        new_p = (r.new_verdict == "PASS")
        old_p = (r.old_verdict == "HIGH")
        if new_p and old_p:
            cats.append("both-pass")
        elif not new_p and not old_p:
            cats.append("both-fail")
        elif new_p and not old_p:
            cats.append("new-only")
            cats.append("disagree")
        else:
            cats.append("old-only")
            cats.append("disagree")
    else:
        cats.append("incomparable")
    if r.counterfactual_verdict == "PASS" and r.new_verdict == "FAIL":
        cats.append("would-pass-after-enrichment")
    elif r.counterfactual_verdict == "FAIL" and r.new_verdict == "FAIL":
        cats.append("still-fail-after-enrichment")
    return " ".join(cats)


# --- Per-clue card ---------------------------------------------------------

def _card(r: "ClueResult", idx: int) -> str:
    rendered = r.rendered_explanation or ""
    new_pill = (f'<span class="pill {_verdict_class(r.new_verdict)}">'
                f'NEW &middot; {_esc(r.new_verdict or "—")}'
                f'</span>')
    old_pill = (f'<span class="pill {_verdict_class(r.old_verdict)}">'
                f'OLD &middot; {_esc(r.old_verdict or "—")}'
                f' {_esc(r.old_score if r.old_score is not None else "")}'
                f'</span>')
    flag_pill = (f'<span class="pill {_flag_class(r.translation_flag)}">'
                 f'{_esc(_flag_label(r.translation_flag))}</span>')

    notes_html = ""
    if r.translation_notes:
        notes_html = ('<div class="notes-block">'
                      '<div class="notes-label">Translator notes</div>'
                      '<ul>'
                      + "".join(f'<li>{_esc(n)}</li>'
                                for n in r.translation_notes)
                      + '</ul></div>')

    if r.new_verdict is None and r.translation_flag == "requires_re_derivation":
        new_block = ('<div class="banner">'
                     '<b>NEW verifier did not run.</b> The DB row stores '
                     'an explanation in prose only — '
                     '<code>components.ai_pieces</code> is empty. The '
                     'universal form requires a structured tree from the '
                     'generator; this row would need to be re-emitted in '
                     'form-format before the new verifier can score it.'
                     '</div>')
    elif r.new_verdict is None:
        new_block = ('<div class="banner">'
                     f'<b>NEW verifier did not run.</b> Reason: '
                     f'<code>{_esc(r.translation_flag)}</code>.</div>')
    else:
        rendered_html = (f'<pre class="rendered">{_esc(rendered)}</pre>'
                         if rendered else
                         '<div class="banner-soft">No rendered output.</div>')
        checks_html = "".join(_check_row(c) for c in r.new_checks) \
            or '<div class="banner-soft">No checks recorded.</div>'
        new_block = (
            '<div class="rendered-label">Rendered explanation (from form):</div>'
            + rendered_html
            + '<div class="checks-label">NEW verifier checks:</div>'
            + '<div class="checks">' + checks_html + '</div>'
        )

    if not r.ai_explanation:
        old_explain = '<div class="banner-soft">No ai_explanation stored.</div>'
    else:
        old_explain = f'<pre class="rendered">{_esc(r.ai_explanation)}</pre>'

    if not r.old_checks:
        old_checks_html = ('<div class="banner-soft">No old verifier checks '
                           '(verifier failed or returned none).</div>')
    else:
        old_checks_html = "".join(_check_row(c) for c in r.old_checks)

    old_block = (
        '<div class="rendered-label">Original ai_explanation '
        '(prose, scored by old verifier):</div>'
        + old_explain
        + '<div class="checks-label">OLD verifier checks:</div>'
        + '<div class="checks">' + old_checks_html + '</div>'
    )

    # Production-cycle panel
    prod_block = _prod_cycle_block(r)

    cat = _categorise(r)
    meta = (f'{_esc(r.model_version or "?")} &middot; '
            f'wp_types: {_esc(r.wordplay_types or "—")} &middot; '
            f'def: {_esc(r.definition_text or "—")} &middot; '
            f'conf: {_esc(r.confidence)}')

    return f"""
<div class="card" data-cat="{cat}" id="c-{idx}">
  <div class="card-head">
    <div class="card-h-l">
      <span class="num">{_esc(r.clue_number)}{_esc(r.direction[:1])}</span>
      <span class="ans">{_esc(r.answer)}</span>
    </div>
    <div class="card-h-r">
      {flag_pill} {new_pill} {old_pill}
    </div>
  </div>
  <div class="clue">{_esc(r.clue_text)}</div>
  <div class="meta">{meta}</div>
  {notes_html}
  <div class="grid">
    <div class="col col-new">{new_block}</div>
    <div class="col col-old">{old_block}</div>
  </div>
  {prod_block}
</div>
"""


# --- Top bar / summary -----------------------------------------------------

def _summary_html(results: list) -> str:
    flag_counts = Counter(r.translation_flag for r in results)
    new_v = Counter(r.new_verdict or "?" for r in results)
    old_v = Counter(r.old_verdict or "?" for r in results)

    both_pass = both_fail = new_only = old_only = incomp = 0
    for r in results:
        if r.new_verdict in ("PASS", "FAIL") and r.old_verdict:
            new_p = (r.new_verdict == "PASS")
            old_p = (r.old_verdict == "HIGH")
            if new_p and old_p:
                both_pass += 1
            elif not new_p and not old_p:
                both_fail += 1
            elif new_p and not old_p:
                new_only += 1
            else:
                old_only += 1
        else:
            incomp += 1

    def _row(label, items):
        cells = "".join(
            f'<span class="stat">{_esc(k)} <b>{v}</b></span>'
            for k, v in items.items())
        return f'<div class="stat-row"><b>{label}</b> {cells}</div>'

    cf_pass = sum(1 for r in results
                  if r.counterfactual_verdict == "PASS")
    cf_fail = sum(1 for r in results
                  if r.counterfactual_verdict == "FAIL")
    cf_would_pass = sum(1 for r in results
                        if r.new_verdict == "FAIL"
                        and r.counterfactual_verdict == "PASS")
    cf_still_fail = sum(1 for r in results
                        if r.new_verdict == "FAIL"
                        and r.counterfactual_verdict == "FAIL")
    total_proposals = sum(len(r.proposals) for r in results)
    new_proposals = sum(1 for r in results for p in r.proposals
                        if p.get("dedupe_status") == "new")
    dup_proposals = sum(1 for r in results for p in r.proposals
                        if p.get("dedupe_status") == "already_in_db")

    return f"""
<div class="summary">
  {_row("Translation:", flag_counts)}
  {_row("NEW verdicts:", new_v)}
  {_row("OLD verdicts (HIGH = pass):", old_v)}
  <div class="stat-row"><b>Comparison:</b>
    <span class="stat">both PASS <b>{both_pass}</b></span>
    <span class="stat">both FAIL <b>{both_fail}</b></span>
    <span class="stat">new only <b>{new_only}</b></span>
    <span class="stat">old only <b>{old_only}</b></span>
    <span class="stat">incomparable <b>{incomp}</b></span>
  </div>
  <div class="stat-row"><b>Production cycle:</b>
    <span class="stat">would PASS after enrichment <b>{cf_would_pass}</b></span>
    <span class="stat">still FAIL after enrichment <b>{cf_still_fail}</b></span>
    <span class="stat">total proposals <b>{total_proposals}</b></span>
    <span class="stat">new <b>{new_proposals}</b></span>
    <span class="stat">already in DB <b>{dup_proposals}</b></span>
  </div>
</div>
"""


_FILTER_BUTTONS = [
    ("all", "All"),
    ("translatable", "Translatable only"),
    ("prose-only", "Prose-only"),
    ("new-pass", "NEW: PASS"),
    ("new-fail", "NEW: FAIL"),
    ("would-pass-after-enrichment", "Would PASS after enrichment"),
    ("still-fail-after-enrichment", "Still FAIL after enrichment"),
    ("disagree", "Disagreements"),
    ("new-only", "NEW pass / OLD fail"),
    ("old-only", "OLD pass / NEW fail"),
    ("gaps", "Translation gaps"),
]


# --- Page -----------------------------------------------------------------

CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
       Helvetica, Arial, sans-serif; margin: 0; padding: 16px;
       background: #fafafa; color: #222; }
h1 { margin: 0 0 8px 0; font-size: 18px; }
.context { color: #666; font-size: 12px; margin-bottom: 16px; }

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
                  font-size: 12px; }
.toolbar button:hover { background: #eef; }
.toolbar button.active { background: #345; color: #fff;
                          border-color: #345; }
.toolbar input { padding: 6px 8px; margin: 2px; border: 1px solid #ccc;
                 border-radius: 4px; font-size: 12px; width: 220px; }

.card { background: #fff; border: 1px solid #ddd; border-radius: 6px;
        margin-bottom: 12px; padding: 12px; }
.card-head { display: flex; justify-content: space-between;
             align-items: center; gap: 8px; margin-bottom: 6px; }
.card-h-l .num { font-weight: bold; color: #444; margin-right: 8px; }
.card-h-l .ans { font-family: ui-monospace, Menlo, Consolas, monospace;
                 font-weight: bold; font-size: 14px; }
.card-h-r .pill { display: inline-block; padding: 3px 8px; margin: 2px;
                  border-radius: 12px; font-size: 11px;
                  font-family: ui-monospace, Menlo, Consolas, monospace;
                  border: 1px solid transparent; }
.pill.v-pass { background: #d0f5d6; color: #014a13;
               border-color: #80d090; }
.pill.v-high { background: #d0f5d6; color: #014a13;
               border-color: #80d090; }
.pill.v-medium { background: #fff5cc; color: #5a4400;
                 border-color: #f0c060; }
.pill.v-low { background: #ffe5cc; color: #6a2a00;
              border-color: #f08850; }
.pill.v-fail { background: #ffd5d5; color: #6a0000;
               border-color: #e08080; }
.pill.v-err  { background: #ffd5d5; color: #6a0000;
               border-color: #e08080; }
.pill.v-none { background: #eee; color: #555; border-color: #ccc; }
.pill.flag-clean { background: #e0f0ff; color: #003a6a;
                   border-color: #80b0e0; }
.pill.flag-partial { background: #fff0e0; color: #5a2a00;
                     border-color: #e0a060; }
.pill.flag-noform { background: #e8e0ff; color: #2a0066;
                    border-color: #b090e0; }
.pill.flag-bad { background: #ffd5d5; color: #6a0000;
                 border-color: #e08080; }
.pill.flag-unknown { background: #fff5cc; color: #5a4400;
                     border-color: #f0c060; }

.clue { font-size: 14px; color: #111; margin: 4px 0;
        font-style: italic; }
.meta { color: #888; font-size: 11px; margin-bottom: 8px;
        font-family: ui-monospace, Menlo, Consolas, monospace; }

.notes-block { background: #fff8e0; border: 1px solid #f0c060;
               border-radius: 4px; padding: 6px 10px; margin: 6px 0;
               font-size: 12px; }
.notes-label { font-weight: bold; color: #5a4400; margin-bottom: 4px; }
.notes-block ul { margin: 0; padding-left: 20px; }

.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
        margin-top: 8px; }
.col { background: #fcfcfc; border: 1px solid #eee; border-radius: 4px;
       padding: 8px; min-width: 0; }
.col-new { border-left: 3px solid #4080c0; }
.col-old { border-left: 3px solid #888; }

.rendered-label, .checks-label { font-weight: bold; font-size: 12px;
                                  color: #444; margin: 4px 0; }
pre.rendered { font-family: ui-monospace, Menlo, Consolas, monospace;
               white-space: pre-wrap; word-wrap: break-word;
               margin: 0 0 8px 0; padding: 8px; background: #f6f6f6;
               border-radius: 4px; font-size: 12px; line-height: 1.4; }

.checks { display: flex; flex-direction: column; gap: 2px; }
.check { display: grid; grid-template-columns: 22px 180px 1fr; gap: 6px;
         padding: 3px 6px; font-size: 11px;
         font-family: ui-monospace, Menlo, Consolas, monospace; }
.check.ck-ok { background: #e8f8ec; }
.check.ck-bad { background: #ffe5e5; }
.check.ck-meh { background: #fff5cc; }
.check.ck-other { background: #f0f0f0; }
.ck-mark { font-weight: bold; }
.ck-name { color: #555; }
.ck-detail { color: #222; word-wrap: break-word; }

.banner { padding: 10px; background: #f0e8ff; border: 1px solid #b090e0;
          border-radius: 4px; font-size: 12px; color: #2a0066; }
.banner-soft { padding: 6px 10px; background: #f4f4f4; color: #888;
               border-radius: 4px; font-size: 11px; font-style: italic; }

.prod-cycle { margin-top: 10px; padding: 8px 10px;
              background: #f8f4ff; border: 1px solid #d0c0e8;
              border-radius: 4px; }
.prod-cycle .rendered-label { color: #2a0066; }
.proposals { display: flex; flex-direction: column; gap: 3px;
             margin: 4px 0; }
.proposal { display: grid; grid-template-columns: 130px 90px 1fr 20px 1fr;
            gap: 6px; padding: 3px 6px; font-size: 11px;
            font-family: ui-monospace, Menlo, Consolas, monospace;
            border-radius: 3px; align-items: center; }
.proposal.prop-new { background: #fff5d0; border: 1px solid #f0c060; }
.proposal.prop-dup { background: #f0f0f0; color: #666;
                     border: 1px solid #ccc; }
.proposal .prop-status { font-weight: bold; color: #5a4400; }
.proposal.prop-dup .prop-status { color: #666; font-weight: normal; }
.proposal .prop-type { color: #555; }
.proposal .prop-arrow { text-align: center; color: #888; }
.prod-pass { padding: 6px 10px; background: #e0f5e0; color: #014a13;
             border-radius: 4px; font-size: 12px; margin-top: 6px;
             font-weight: bold; }
.prod-fail { padding: 6px 10px; background: #ffe0e0; color: #6a0000;
             border-radius: 4px; font-size: 12px; margin-top: 6px; }
.prod-fail ul { margin: 4px 0 0 0; padding-left: 20px; }

details.guide { background: #f0f6fa; border: 1px solid #c0d0e0;
                border-radius: 6px; padding: 8px 12px; margin-bottom: 12px;
                font-size: 12px; }
details.guide summary { cursor: pointer; font-weight: bold;
                         color: #003a6a; }
details.guide p, details.guide ul { margin: 6px 0; line-height: 1.4; }
details.guide ul { padding-left: 20px; }
details.guide code { background: #fff; padding: 1px 4px; border-radius: 3px;
                     font-size: 11px; }
"""

JS = """
(function() {
  var buttons = document.querySelectorAll('.toolbar button[data-cat]');
  var search = document.getElementById('q');
  var cards = document.querySelectorAll('.card');

  function applyFilter(cat, query) {
    cards.forEach(function(c) {
      var cardCats = (c.getAttribute('data-cat') || '').split(/\\s+/);
      var catOk = (cat === 'all') || cardCats.indexOf(cat) >= 0;
      var qOk = !query || c.textContent.toLowerCase()
                          .indexOf(query.toLowerCase()) >= 0;
      c.style.display = (catOk && qOk) ? '' : 'none';
    });
    document.getElementById('shown').textContent =
      Array.from(cards).filter(function(c) {
        return c.style.display !== 'none';
      }).length;
  }

  var current = 'all';
  buttons.forEach(function(b) {
    b.addEventListener('click', function() {
      current = b.getAttribute('data-cat');
      buttons.forEach(function(x) { x.classList.remove('active'); });
      b.classList.add('active');
      applyFilter(current, search.value);
    });
  });
  search.addEventListener('input', function() {
    applyFilter(current, search.value);
  });
})();
"""


def write_html_report(results: list, out_path: str,
                      title: str = "Universal-form prototype — simulation") \
        -> None:
    """Write a single-file HTML report. `results` is a list of ClueResult."""
    cards = "\n".join(_card(r, i) for i, r in enumerate(results))
    summary = _summary_html(results)
    btns = "".join(
        f'<button data-cat="{cat}" class="{ "active" if cat=="all" else ""}">'
        f'{label}</button>'
        for cat, label in _FILTER_BUTTONS)
    page = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{_esc(title)}</title>
<style>{CSS}</style>
</head><body>
<h1>{_esc(title)}</h1>
<div class="context">
  <b>{len(results)}</b> clues processed.
  Showing <span id="shown">{len(results)}</span>.
  Read-only simulation — no live tables modified.
</div>
<details class="guide">
<summary>How to read this report</summary>
<p>Each card below is one clue, with the original clue text and answer
   in the header. Three pills tell you the status:</p>
<ul>
  <li><b>Translation flag</b>: <i>translatable</i> means the prototype's
      translator built a tree from the stored <code>components</code>;
      <i>translatable (partial)</i> means it built one but had to
      interpret some pieces (notes shown in the yellow box on the card);
      <i>prose-only — no form</i> means the stored row has no usable
      structured data, so the new verifier could not run.</li>
  <li><b>NEW</b>: verdict and score from the form-based verifier
      (assembly + bridge + mechanism + residue checks).</li>
  <li><b>OLD</b>: verdict and score from the live prose-based verifier
      (the regex-based one in <code>sonnet_pipeline/verify_explanation.py</code>).</li>
</ul>
<p>The two columns inside each card show: on the LEFT, the NEW renderer's
   prose explanation built from the form, and the per-check breakdown
   from the new verifier. On the RIGHT, the original
   <code>ai_explanation</code> stored in the DB and the per-check
   breakdown from the OLD verifier. They use the same data — what
   differs is what each verifier sees.</p>
<p>Use the filter buttons to slice: <i>Disagreements</i> finds clues
   where the two verifiers disagreed; <i>NEW wins</i> finds clues where
   the new verifier scored higher; <i>Translation gaps</i> finds rows
   where the translator had to make interpretations.</p>
</details>
{summary}
<div class="toolbar">
  {btns}
  <input id="q" type="search" placeholder="Search clue / answer / notes&hellip;">
</div>
{cards}
<script>{JS}</script>
</body></html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
