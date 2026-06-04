"""Render a WFW Parse as the clue page — the shared BASE screen for every type.

Visual tone: a clean, high-contrast "modern app" card. A header (coloured
clue-type badge + PASS/PENDING/FAIL), the clue line with the used letters lit in place,
bold rounded answer tiles, and a word-by-word breakdown laid out as an ALIGNED
two-column grid (a solid colour-coded role pill, then the content) so every row
lines up. Pure: takes a core.wfw_model.Parse and returns an HTML fragment.

Colour: in `coloured` mode each wordplay source gets its own palette colour and
the answer tiles it produced wear it. Hidden runs uncoloured — there is a single
host source, so it (and its tiles) wear one accent (amber), matching the lit
letters in the clue line. The role pills (definition / indicator / link) are
colour-coded by role in both modes.
"""

from html import escape

# One stable colour (text, fill) per source index, used in coloured mode.
PALETTE = [
    ("#1d6fb8", "#e3f0fb"),   # blue
    ("#2e8b57", "#e4f4ea"),   # green
    ("#d2691e", "#fbeadf"),   # orange
    ("#c2185b", "#fbe4ee"),   # pink
    ("#0097a7", "#e0f5f7"),   # cyan
    ("#b8860b", "#f8efd6"),   # gold
    ("#c62828", "#fbe4e4"),   # red
    ("#6a1b9a", "#f0e4f7"),   # purple
]

# The hidden accent — tiles + host pill wear it; ties to the lit clue letters.
HIDDEN_FG = "#92600a"
HIDDEN_FILL = "#fde68a"
HIDDEN_BORDER = "#f59e0b"

# Friendly clue-type labels for the header badge.
_TYPE_LABEL = {
    "hidden": "Hidden word",
    "hidden_reversed": "Hidden word (reversed)",
    "dd": "Double definition",
    "cd": "Cryptic definition",
    "charade": "Charade",
    "container": "Container",
    "anagram": "Anagram",
    "anagram_charade": "Anagram + charade",
    "reversal": "Reversal",
    "deletion": "Deletion",
    "acrostic": "Acrostic",
    "homophone": "Homophone",
}

# Friendly role labels for a wordplay piece, by mechanism.
_MECH_LABEL = {
    "hidden": "Hidden in",
    "hidden_reversed": "Hidden in (rev.)",
    "synonym": "Synonym",
    "abbreviation": "Abbrev.",
    "raw": "Literal",
    "first_letter": "Initial",
    "last_letter": "Last letter",
    "outer": "Outer letters",
    "homophone": "Sounds like",
    "anagram_fodder": "Anagram of",
}


def _colour(i):
    return PALETTE[i % len(PALETTE)]


def render_parse(parse, ctx=None, clue_line_html=None, coloured=True):
    """Return the HTML fragment for one solved clue (the shared base screen).

    A clue type customises only:
    - `clue_line_html`: a pre-rendered clue line (hidden lights the host letters);
    - `coloured`: per-source palette colour on or off (hidden runs uncoloured and
      uses the single amber accent instead).
    """
    if coloured:
        src_fg = {i: _colour(i)[0] for i in range(len(parse.sources))}
        src_fill = {i: _colour(i)[1] for i in range(len(parse.sources))}
        tile_fg = src_fg
        tile_border = {i: _colour(i)[0] for i in range(len(parse.sources))}
        tile_fill = src_fill
    else:
        src_fg = {i: "#ffffff" for i in range(len(parse.sources))}
        src_fill = {i: HIDDEN_FG for i in range(len(parse.sources))}
        tile_fg = {i: HIDDEN_FG for i in range(len(parse.sources))}
        tile_border = {i: HIDDEN_BORDER for i in range(len(parse.sources))}
        tile_fill = {i: HIDDEN_FILL for i in range(len(parse.sources))}

    # --- header: clue-type badge + verdict ---
    type_label = _TYPE_LABEL.get(parse.operation or "", (parse.operation or "—"))
    header = (
        '<div class="wfw-head">'
        '<span class="wfw-type"><span class="wfw-dot"></span>%s</span>%s</div>'
        % (escape(type_label.upper()), _verdict_badge(parse)))

    # --- clue line ---
    enum = parse.enumeration()
    clue_html = escape(parse.clue_text) if clue_line_html is None else clue_line_html
    if enum:
        clue_html += ' <span class="wfw-enum">%s</span>' % escape(enum)

    # --- answer tiles ---
    letters = parse.answer_letters()
    by_pos = {l.answer_pos: l for l in parse.links}
    tiles = []
    for pos in range(1, len(letters) + 1):
        link = by_pos.get(pos)
        si = link.source_index if link else None
        if si is not None and si in tile_fill:
            style = ("background:%s;border-color:%s;color:%s"
                     % (tile_fill[si], tile_border[si], tile_fg[si]))
        else:
            style = "background:#f1f5f9;border-color:#cbd5e1;color:#94a3b8"
        tiles.append('<span class="wfw-tile" style="%s">%s</span>'
                     % (style, escape(letters[pos - 1])))
    tiles_html = "".join(tiles)

    breakdown = _render_breakdown(parse, src_fg, src_fill)

    prov_def = (parse.definition is not None
                and getattr(parse.definition, "source", "db") == "pending")
    prov_ann = any(getattr(a, "source", "db") == "pending"
                   for a in parse.annotations)
    prov_src = any(getattr(s, "source", "db") == "pending"
                   for s in parse.sources)
    prov = ""
    if prov_def or prov_ann or prov_src:
        prov = ('<div class="wfw-banner wfw-banner-prov">A provisional piece '
                '(highlighted "provisional") is queued for verification &mdash; '
                'provisional until enriched.</div>')

    warns = ""
    status = getattr(parse, "status", "pass")
    if status != "pass" and parse.warnings:
        items = "".join("<li>%s</li>" % escape(w) for w in parse.warnings)
        cls = "wfw-banner-warn" if status == "fail" else "wfw-banner-pending"
        warns = '<div class="wfw-banner %s"><ul>%s</ul></div>' % (cls, items)

    return ('<div class="wfw-card">%s'
            '<div class="wfw-clue">%s</div>'
            '<div class="wfw-tiles">%s</div>'
            '<div class="wfw-rows">%s</div>'
            '%s%s</div>'
            % (header, clue_html, tiles_html, breakdown, prov, warns))


def _verdict_badge(parse):
    status = getattr(parse, "status", "pass")
    if status == "pass":
        return '<span class="wfw-verdict pass">&#10003; PASS</span>'
    if status == "pending":
        return '<span class="wfw-verdict pending">&#8226; PENDING</span>'
    return '<span class="wfw-verdict fail">&#10007; FAIL</span>'


def _first_index(atom_ids):
    """Sort key: the position of a span's first clue character."""
    nums = []
    for a in atom_ids:
        tail = a.rsplit("_", 1)[-1]
        if tail.isdigit():
            nums.append(int(tail))
    return min(nums) if nums else 1_000_000


def _row(sort_i, role_label, role_style, content):
    """One breakdown row: a colour-coded role pill + aligned content. Uses
    display:contents so the two cells join the parent grid and columns align."""
    return (sort_i,
            '<div class="wfw-row">'
            '<span class="wfw-role" style="%s">%s</span>'
            '<span class="wfw-content">%s</span></div>'
            % (role_style, escape(role_label.upper()), content))


def _render_breakdown(parse, src_fg, src_fill):
    """One aligned row per clue word/span, in clue order, each with its role."""
    rows = []

    for si, s in enumerate(parse.sources):
        label = _MECH_LABEL.get(s.mechanism, s.mechanism)
        style = "background:%s;color:%s" % (src_fill[si], src_fg[si])
        content = ('%s <span class="wfw-arrow">&rarr;</span> '
                   '<strong class="wfw-val">%s</strong>'
                   % (escape(s.text), escape(s.value)))
        if getattr(s, "source", "db") == "pending":
            content += ' <span class="wfw-prov">provisional</span>'
        rows.append(_row(_first_index(s.clue_atom_ids), label, style, content))

    if parse.definition:
        prov = ""
        if getattr(parse.definition, "source", "db") == "pending":
            prov = ' <span class="wfw-prov">provisional</span>'
        rows.append(_row(_first_index(parse.definition.clue_atom_ids),
                         "Definition", "background:#2563eb;color:#fff",
                         escape(parse.definition.text) + prov))

    for a in parse.annotations:
        if a.role == "indicator" and getattr(a, "note", "") == "definition by example":
            style = "background:#2563eb;color:#fff"
            label = "By example"
        elif a.role == "indicator":
            style = "background:#7c3aed;color:#fff"
            label = "Indicator"
        else:
            style = "background:#64748b;color:#fff"
            label = "Link"
        content = escape(a.text)
        if getattr(a, "source", "db") == "pending":
            content += ' <span class="wfw-prov">provisional</span>'
        rows.append(_row(_first_index(a.clue_atom_ids), label, style, content))

    rows.sort(key=lambda r: r[0])
    return "".join(html for _, html in rows)


PAGE_CSS = """
  :root { --ink:#0f172a; --muted:#475569; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
         Helvetica, Arial, sans-serif; margin: 2rem auto; max-width: 680px;
         color: var(--ink); background:#f1f5f9; padding:0 1rem; }
  .wfw-tag { color:#64748b; font-size:.8rem; letter-spacing:.04em;
             text-transform:uppercase; font-weight:600; }
  .wfw-card { background:#fff; border:1px solid #e2e8f0; border-radius:16px;
              padding:1.4rem 1.5rem; margin:1.25rem 0;
              box-shadow:0 4px 16px rgba(15,23,42,.06); }
  .wfw-head { display:flex; align-items:center; justify-content:space-between;
              margin-bottom:1rem; gap:.75rem; }
  .wfw-type { display:inline-flex; align-items:center; gap:.45rem;
              background:#0f172a; color:#fff; font-size:.72rem; font-weight:700;
              letter-spacing:.08em; padding:.35rem .7rem; border-radius:999px; }
  .wfw-dot { width:.5rem; height:.5rem; border-radius:50%;
             background:#f59e0b; display:inline-block; }
  .wfw-verdict { font-size:.78rem; font-weight:800; letter-spacing:.05em;
                 padding:.35rem .7rem; border-radius:999px; color:#fff; }
  .wfw-verdict.pass { background:#16a34a; }
  .wfw-verdict.pending { background:#d97706; }
  .wfw-verdict.fail { background:#dc2626; }
  .wfw-clue { font-size:1.4rem; line-height:1.6; margin-bottom:1.1rem;
              color:var(--ink); font-weight:500; }
  .wfw-enum { color:#94a3b8; font-weight:600; }
  .wfw-lit { background:#fde68a; border-radius:4px; padding:0 .06em;
             box-shadow:inset 0 -2px 0 #f59e0b; font-weight:800; color:#7a4f00; }
  .wfw-tiles { display:flex; gap:.45rem; flex-wrap:wrap; margin:.25rem 0 1.3rem; }
  .wfw-tile { display:inline-flex; align-items:center; justify-content:center;
              width:2.7rem; height:2.7rem; border:2px solid; border-radius:10px;
              font-size:1.3rem; font-weight:800;
              font-family:'SF Mono','Courier New',monospace; }
  .wfw-rows { display:grid; grid-template-columns:max-content 1fr;
              gap:.55rem .9rem; align-items:center; }
  .wfw-row { display:contents; }
  .wfw-role { justify-self:start; font-size:.68rem; font-weight:800;
              letter-spacing:.06em; padding:.28rem .6rem; border-radius:7px;
              white-space:nowrap; text-align:center; min-width:5.5rem; }
  .wfw-content { font-size:1.05rem; color:var(--ink); line-height:1.4; }
  .wfw-arrow { color:#94a3b8; margin:0 .15rem; font-weight:700; }
  .wfw-val { font-family:'SF Mono','Courier New',monospace; letter-spacing:.04em;
             color:#0f172a; }
  .wfw-prov { display:inline-block; margin-left:.4rem; padding:.05rem .45rem;
              border-radius:6px; background:#fef3c7; color:#92600a;
              border:1px solid #fcd34d; font-size:.72rem; font-weight:700;
              text-transform:uppercase; letter-spacing:.04em; }
  .wfw-banner { margin-top:1rem; border-radius:10px; padding:.65rem .85rem;
                font-size:.92rem; line-height:1.45; }
  .wfw-banner-prov { background:#fffbeb; border:1px solid #fcd34d; color:#92600a; }
  .wfw-banner-pending { background:#fffbeb; border:1px solid #fcd34d; color:#92600a; }
  .wfw-banner-warn { background:#fef2f2; border:1px solid #fecaca; color:#b91c1c; }
  .wfw-banner-pending ul, .wfw-banner-warn ul { margin:0; padding-left:1.2rem; }
  .wfw-banner-pending li, .wfw-banner-warn li { margin:.15rem 0; }
"""


def render_page(parses, title="WFW preview"):
    """Wrap one or more rendered parses in a full standalone HTML page."""
    cards = "".join(render_parse(p) for p in parses)
    return ('<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>%s</title><style>%s</style></head><body>'
            '<p class="wfw-tag">word-for-word preview</p>%s</body></html>'
            % (escape(title), PAGE_CSS, cards))
