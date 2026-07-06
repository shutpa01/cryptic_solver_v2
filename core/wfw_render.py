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
    "andlit": "All-in-one (&lit)",
    "charade": "Charade",
    "container": "Container",
    "anagram": "Anagram",
    "anagram_charade": "Anagram + charade",
    "anagram_container": "Anagram + container",
    "reversal": "Reversal",
    "reversal_deletion": "Reversal + deletion",
    "deletion": "Deletion",
    "charade_alternation": "Charade + alternation",
    "acrostic": "Acrostic",
    "homophone": "Homophone",
}

# Friendly role labels for a wordplay piece, by mechanism.
_MECH_LABEL = {
    "hidden": "Hidden in",
    "hidden_reversed": "Hidden in (rev.)",
    "synonym": "Synonym",
    "abbreviation": "Substitution",   # wordplay-table values (abbrevs, Roman numerals, compass
                                      # points, symbols) — "substitution" is the accurate umbrella
    "raw": "Literal",
    "first_letter": "Initial",
    "last_letter": "Last letter",
    "outer": "Outer letters",
    "homophone": "Sounds like",
    "anagram_fodder": "Anagram of",
    "alternate": "Alternate letters",
    "definition": "Definition",
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

    # --- header: clue-type badge + solving-engine tag + verdict ---
    type_label = _TYPE_LABEL.get(parse.operation or "", (parse.operation or "—"))
    engine = (getattr(parse, "solved_by", "") or "").strip()
    # Always show the SPECIFIC solving engine (parse.solved_by) so it is clear at a glance
    # which engine produced the parse — no DB query, no stack-trace hunting.
    eng_tag = ""
    if engine:
        eng_tag = '<span class="wfw-engine" title="solving engine">%s</span>' % escape(engine)
    header = (
        '<div class="wfw-head">'
        '<span class="wfw-type"><span class="wfw-dot"></span>%s</span>%s%s</div>'
        % (escape(type_label.upper()), eng_tag, _verdict_badge(parse)))

    # --- clue line ---
    enum = parse.enumeration()
    clue_html = escape(parse.clue_text) if clue_line_html is None else clue_line_html
    if enum:
        clue_html += ' <span class="wfw-enum">%s</span>' % escape(enum)

    # --- answer tiles --- walk the ENUMERATED answer so multi-word answers keep their word
    # gaps and hyphens (IN THE RAW, not INTHERAW). Letters are tiles coloured by their source;
    # spaces become a word gap and hyphens a separator. `pos` counts letters only (the link key).
    letters = parse.answer_letters()
    by_pos = {l.answer_pos: l for l in parse.links}
    tiles, pos = [], 0
    for ch in (parse.answer_text or letters):
        if ch.isalpha():
            pos += 1
            link = by_pos.get(pos)
            si = link.source_index if link else None
            if si is not None and si in tile_fill:
                style = ("background:%s;border-color:%s;color:%s"
                         % (tile_fill[si], tile_border[si], tile_fg[si]))
            else:
                style = "background:#f1f5f9;border-color:#cbd5e1;color:#94a3b8"
            tiles.append('<span class="wfw-tile" style="%s">%s</span>'
                         % (style, escape(ch.upper())))
        elif ch in "-–—":
            tiles.append('<span class="wfw-tile-sep">&ndash;</span>')
        elif ch.isspace():
            tiles.append('<span class="wfw-tile-gap"></span>')
    tiles_html = "".join(tiles)

    renderer = _TYPE_RENDERERS.get(parse.operation or "", _render_generic_breakdown)
    breakdown = renderer(parse, ctx, src_fg, src_fill)

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
            '%s'                                   # breakdown brings its own structure
            '%s%s</div>'
            % (header, clue_html, tiles_html, breakdown, prov, warns))


def _verdict_badge(parse):
    from core import review_gate
    status = getattr(parse, "status", "pass")
    if status == "pending" and review_gate.is_review(parse):
        # a high-risk engine's full solve, held for human confirmation (distinct from an
        # ordinary provisional-definition pending).
        return '<span class="wfw-verdict review">&#9873; REVIEW</span>'
    if status == "pass":
        return '<span class="wfw-verdict pass">&#10003; PASS</span>'
    if status == "pending":
        return '<span class="wfw-verdict pending">&#8226; PENDING</span>'
    if status == "invalid":
        return '<span class="wfw-verdict invalid">&#9888; INVALID</span>'
    return '<span class="wfw-verdict fail">&#10007; FAIL</span>'


# The indicator's precise type, read off the note each engine records, with a colour.
_IND_TYPES = ("anagram", "container", "insertion", "reversal", "deletion",
              "hidden", "homophone", "acrostic", "palindrome", "spoonerism",
              "selection", "substitution", "alternation")
_IND_COLOUR = {"anagram": "#7c3aed", "container": "#0e7490", "reversal": "#b45309",
               "deletion": "#be185d", "hidden": "#92600a", "homophone": "#4d7c0f",
               "acrostic": "#5b21b6", "letter-shift": "#0369a1", "indicator": "#7c3aed"}


# Friendly detail for an indicator's sub-type code, per type (a manual "type/subtype" note like
# "selection/first" or "deletion/head" must READ well, not show a bare code). Keyed by type then
# sub-type; "middle" means different things for selection vs deletion, hence the per-type nesting.
_SUBTYPE_DETAIL = {
    "selection": {"first": "first letter(s)", "last": "last letter(s)", "outer": "outer letters",
                  "middle": "middle letter(s)", "alternate": "alternate letters"},
    "deletion": {"head": "remove first letter", "tail": "remove last letter",
                 "ends": "remove outer letters", "middle": "remove middle letter",
                 "empty": "remove inner letters", "general": "named letter(s)"},
    "letter_shift": {"last_front": "move last letter to front",
                     "first_end": "move first letter to end"},
}


def _indicator_label(note):
    """(pill label, extra detail) for an indicator annotation, from its note. Gives the
    PRECISE type — 'Container indicator', 'Reversal indicator', 'Deletion indicator' (+ the
    deletion sub-type as detail) — instead of a bare 'Indicator'."""
    n = (note or "").lower()
    if n.startswith("deletion:"):                  # "deletion: behead (drops the ...)"
        return "Deletion indicator", note.split(":", 1)[1].strip()
    if n.startswith("spoonerism:"):                # "spoonerism: BARRED HACK -> HARDBACK"
        return "Spoonerism indicator", note.split(":", 1)[1].strip()
    if n.startswith("substitution:"):              # "substitution: love (O) replaces one (I)..."
        return "Substitution indicator", note.split(":", 1)[1].strip()
    if n.startswith("reversal:"):                  # "reversal: BALS -> SLAB"
        return "Reversal indicator", note.split(":", 1)[1].strip()
    if n.startswith("alternation:"):               # "alternation: alternate letters of near -> ER"
        return "Alternation indicator", note.split(":", 1)[1].strip()
    if n.startswith("selection ("):               # "selection (first)"
        return "Selection indicator", note[note.find("(") + 1:note.find(")")].strip()
    if n.startswith("letter_shift") or n.startswith("letter-shift") \
            or n.startswith("letter shift"):       # "letter_shift/last_front indicator"
        sub = n.split("/", 1)[1].replace("indicator", "").strip() if "/" in n else ""
        detail = {"last_front": "move last letter to front",
                  "first_end": "move first letter to end"}.get(sub, sub)
        return "Letter-shift indicator", detail
    for t in _IND_TYPES:
        if t in n:
            disp = "Container" if t == "insertion" else t.capitalize()
            # manual "type/subtype indicator" format (e.g. "selection/first", "deletion/head") —
            # SHOW the sub-type, friendly-mapped, so it is not a bare "Selection indicator".
            sub = n.split("/", 1)[1].replace("indicator", "").strip() if "/" in n else ""
            return disp + " indicator", _SUBTYPE_DETAIL.get(t, {}).get(sub, sub)
    return "Indicator", ""


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


# ---- reusable row builders (shared by every renderer) ----------------------------

def _source_row(parse, si, src_fg, src_fill):
    """One source piece row: 'wood -> BALSA' coloured by source, with the homophone-via-
    synonym aside and the provisional badge."""
    s = parse.sources[si]
    label = _MECH_LABEL.get(s.mechanism, s.mechanism)
    style = "background:%s;color:%s" % (src_fill[si], src_fg[si])
    content = ('%s <span class="wfw-arrow">&rarr;</span> '
               '<strong class="wfw-val">%s</strong>'
               % (escape(s.text), escape(s.value)))
    # Show HOW the piece's letters reached the answer (reversed / minus a deleted run), so a
    # piece that supplies IS but lands as SI reads "is -> IS reversed" here too — not a bare IS
    # whose order isn't in the answer. Matches the assembly build line (same _transform_note).
    if s.mechanism != "anagram_fodder":
        al = parse.answer_letters()
        positions = sorted(l.answer_pos for l in parse.links if l.source_index == si)
        got = "".join(al[p - 1] for p in positions if 1 <= p <= len(al))
        content += _transform_note(s.value, got)
    if s.mechanism == "homophone":
        tr = next((l.transform for l in parse.links
                   if l.source_index == si and l.transform), None)
        snd = tr.split('"')[1] if tr and '"' in tr else None
        if snd and snd.lower() != s.text.lower():
            content += (' <span class="wfw-emuted">&mdash; via &ldquo;%s&rdquo;'
                        ' (synonym)</span>' % escape(snd))
    if getattr(s, "source", "db") == "pending":
        content += ' <span class="wfw-prov">provisional</span>'
    return _row(_first_index(s.clue_atom_ids), label, style, content)


def _definition_row(parse):
    """The definition row, or None. A guessed (pending) edge is labelled 'unidentified'."""
    if not parse.definition:
        return None
    _dsrc = getattr(parse.definition, "source", "db")
    def_label, def_style, prov = "Definition", "background:#2563eb;color:#fff", ""
    if getattr(parse.definition, "mechanism", "") == "definition_by_example":
        def_label = "Definition by example"      # same layer as a definition; label only
    if _dsrc == "pending":
        def_label = "Unidentified definition"
        def_style = "background:#64748b;color:#fff"
        prov = ' <span class="wfw-prov">not confirmed</span>'
    elif _dsrc == "manual":
        prov = ' <span class="wfw-prov">manual (not in DB)</span>'
    return _row(_first_index(parse.definition.clue_atom_ids), def_label, def_style,
                escape(parse.definition.text) + prov)


def _annotation_row(parse, a):
    """An indicator / deletion / link annotation row, with its precise label + detail."""
    note = getattr(a, "note", "") or ""
    content = escape(a.text)
    if a.role == "indicator" and note == "definition by example":
        style, label = "background:#2563eb;color:#fff", "By example"
    elif a.role == "indicator" and note.lower().startswith("deleted letters:"):
        # a word that SUPPLIES the removed letter(s) (Pound -> L): show the letter with the
        # word, like a named deletion. Render-only — the engine keeps its own note convention.
        style, label = "background:#b91c1c;color:#fff", "Deleted"
        content += (' <span class="wfw-arrow">&rarr;</span> '
                    '<strong class="wfw-val">%s</strong>'
                    % escape(note.split(":", 1)[1].strip()))
    elif a.role == "indicator":
        label, detail = _indicator_label(note)
        style = "background:%s;color:#fff" % _IND_COLOUR.get(label.split()[0].lower(),
                                                            "#7c3aed")
        if detail:
            content += ' <span class="wfw-emuted">&mdash; %s</span>' % escape(detail)
    elif a.role == "deletion":
        style, label = "background:#b91c1c;color:#fff", "Deleted"
        if "→" in note:
            removed = note.split("→")[-1].strip()
        elif note.lower().startswith("deleted letters:"):    # "deleted letters: A" -> show the A
            removed = note.split(":", 1)[1].strip()
        else:
            removed = ""
        if removed:
            content += (' <span class="wfw-arrow">&rarr;</span> '
                        '<strong class="wfw-val">%s</strong>' % escape(removed))
    else:
        style, label = "background:#64748b;color:#fff", "Link"
    if getattr(a, "source", "db") == "pending":
        content += ' <span class="wfw-prov">provisional</span>'
    return _row(_first_index(a.clue_atom_ids), label, style, content)


def _all_rows(parse, src_fg, src_fill):
    """Every row (sources + definition + annotations), unsorted: [(sort_i, html), ...]."""
    rows = [_source_row(parse, si, src_fg, src_fill) for si in range(len(parse.sources))]
    d = _definition_row(parse)
    if d:
        rows.append(d)
    rows += [_annotation_row(parse, a) for a in parse.annotations]
    return rows


def _grid(rows):
    """Sort rows by clue position and join into the aligned 2-column grid block. The grid
    div must contain ONLY .wfw-row children (each display:contents) or the columns break."""
    body = "".join(html for _, html in sorted(rows, key=lambda r: r[0]))
    return '<div class="wfw-rows">%s</div>' % body


def _build_line(inner):
    """A type's one-line 'build' summary, shown above the rows."""
    return '<div class="wfw-build">%s</div>' % inner


def _pval(parse, si, src_fg):
    """The coloured value strong-tag for source `si` (used in summary chains)."""
    return ('<strong class="wfw-val" style="color:%s">%s</strong>'
            % (src_fg.get(si, "#0f172a"), escape(parse.sources[si].value)))


# ---- per-type explanation renderers ----------------------------------------------
# A clue type registers ONE renderer here; it renders the wordplay in its NATURAL shape.
# Types with no renderer fall back to the generic clue-order rows, so nothing regresses.

_TYPE_RENDERERS = {}


def renders(*ops):
    def deco(fn):
        for op in ops:
            _TYPE_RENDERERS[op] = fn
        return fn
    return deco


def _render_generic_breakdown(parse, ctx, src_fg, src_fill):
    """Fallback: every clue word/span in clue order with its role (the original format)."""
    return _grid(_all_rows(parse, src_fg, src_fill))


@renders("charade", "charade_alternation")
def _render_charade(parse, ctx, src_fg, src_fill):
    """A + B + C -> ANSWER, pieces in clue order, then the detailed rows."""
    order = sorted(range(len(parse.sources)),
                   key=lambda si: _first_index(parse.sources[si].clue_atom_ids))
    chain = ' <span class="wfw-plus">+</span> '.join(_pval(parse, si, src_fg) for si in order)
    summ = ('%s <span class="wfw-arrow">&rarr;</span> '
            '<strong class="wfw-val">%s</strong>' % (chain, escape((parse.answer_text or "").upper())))
    return _build_line(summ) + _grid(_all_rows(parse, src_fg, src_fill))


@renders("anagram")
def _render_anagram(parse, ctx, src_fg, src_fill):
    """anagram of FODDER [- removed letters] -> ANSWER, then the detailed rows."""
    from collections import Counter
    fodder = [(s.value or "").upper() for s in parse.sources if s.mechanism == "anagram_fodder"]
    pool = "".join(fodder)                              # letters only — for the - removed math
    removed = Counter(pool) - Counter(parse.answer_letters())
    # display the fodder words spaced (IN ON WAGER), not run together (INONWAGER)
    summ = 'anagram of <strong class="wfw-val">%s</strong>' % escape(" ".join(fodder))
    if pool and sum(removed.values()):
        removed_str = "".join(sorted(removed.elements()))
        reduced = list(pool)
        for ch in removed.elements():
            if ch in reduced:
                reduced.remove(ch)
        summ += (' &minus; <strong class="wfw-val">%s</strong> '
                 '(&rarr; <strong class="wfw-val">%s</strong>)'
                 % (escape(removed_str), escape("".join(reduced))))
    summ += (' <span class="wfw-arrow">&rarr;</span> '
             '<strong class="wfw-val">%s</strong>' % escape((parse.answer_text or "").upper()))
    return _build_line(summ) + _grid(_all_rows(parse, src_fg, src_fill))


@renders("reversal_deletion")
def _render_reversal_deletion(parse, ctx, src_fg, src_fill):
    """SYNONYM -> (after the cut) -> ANSWER, read off the structured notes, then the rows."""
    val = parse.sources[0].value if parse.sources else ""
    mid = ""
    for a in parse.annotations:
        note = getattr(a, "note", "") or ""
        if note.lower().startswith("deletion:") and "→" in note:
            mid = note.rsplit("→", 1)[-1].strip()
    steps = ['<strong class="wfw-val">%s</strong>' % escape(val)]
    if mid:
        steps.append('<strong class="wfw-val">%s</strong>' % escape(mid))
    steps.append('<strong class="wfw-val">%s</strong>' % escape((parse.answer_text or "").upper()))
    summ = ' <span class="wfw-arrow">&rarr;</span> '.join(steps)
    return _build_line(summ) + _grid(_all_rows(parse, src_fg, src_fill))


def _ans(parse):
    """The answer, uppercased but keeping its enumeration spacing/hyphens."""
    return escape((parse.answer_text or "").upper())


def _arrow_ans(parse):
    return ('<span class="wfw-arrow">&rarr;</span> <strong class="wfw-val">%s</strong>'
            % _ans(parse))


@renders("double_definition")
def _render_dd(parse, ctx, src_fg, src_fill):
    """"def 1" = "def 2" -> ANSWER (two definitions of the same word)."""
    defs = [s for s in parse.sources if s.mechanism == "definition"]
    if len(defs) >= 2:
        chain = ' <span class="wfw-eq">=</span> '.join(
            '&ldquo;%s&rdquo;' % escape(s.text) for s in defs)
        return _build_line('%s %s' % (chain, _arrow_ans(parse))) \
            + _grid(_all_rows(parse, src_fg, src_fill))
    return _grid(_all_rows(parse, src_fg, src_fill))


@renders("container")
def _render_container(parse, ctx, src_fg, src_fill):
    """OUTER around INNER -> ANSWER. The OUTER source is the one whose answer letters are
    split (non-contiguous) around the INNER; derived from the links, so it is not guessed."""
    pos = {}
    for l in parse.links:
        pos.setdefault(l.source_index, []).append(l.answer_pos)

    def contig(ps):
        ps = sorted(ps)
        return bool(ps) and ps[-1] - ps[0] + 1 == len(ps)
    outer = [si for si, ps in pos.items() if not contig(ps)]
    inner = [si for si, ps in pos.items() if contig(ps)]
    if len(outer) == 1 and inner:
        inners = ' <span class="wfw-plus">+</span> '.join(
            _pval(parse, si, src_fg) for si in sorted(inner))
        summ = ('%s <span class="wfw-around">around</span> %s %s'
                % (_pval(parse, outer[0], src_fg), inners, _arrow_ans(parse)))
        return _build_line(summ) + _grid(_all_rows(parse, src_fg, src_fill))
    return _grid(_all_rows(parse, src_fg, src_fill))


@renders("charade_container_acrostic")
def _render_charade_container_acrostic(parse, ctx, src_fg, src_fill):
    """A charade whose pieces include ONE container (OUTER around INNER) and >=1 acrostic run.
    Renders the pieces in answer order, e.g. INDIA around C + TES -> INDICATES, then the rows.
    Piece membership is read from the per-link operation + answer positions, not guessed."""
    pos, op = {}, {}
    for l in parse.links:
        pos.setdefault(l.source_index, []).append(l.answer_pos)
        op[l.source_index] = l.operation

    def contig(ps):
        ps = sorted(ps)
        return bool(ps) and ps[-1] - ps[0] + 1 == len(ps)

    pieces, consumed = [], set()                         # pieces: (min_answer_pos, html)
    # container piece: the OUTER source's letters are split (non-contiguous) around the INNER
    con = [si for si in pos if op.get(si) == "container"]
    outer = [si for si in con if not contig(pos[si])]
    inner = [si for si in con if contig(pos[si])]
    if len(outer) == 1 and inner:
        inners = ' <span class="wfw-plus">+</span> '.join(
            _pval(parse, si, src_fg) for si in sorted(inner, key=lambda s: min(pos[s])))
        html = ('%s <span class="wfw-around">around</span> %s'
                % (_pval(parse, outer[0], src_fg), inners))
        minp = min(pos[outer[0]] + [p for si in inner for p in pos[si]])
        pieces.append((minp, html))
        consumed = {outer[0]} | set(inner)
    # acrostic runs: contiguous single-letter first/last-letter sources -> one initials piece
    acro = sorted((si for si in pos
                   if parse.sources[si].mechanism in ("first_letter", "last_letter")),
                  key=lambda s: min(pos[s]))
    run, prev = [], None
    def flush(run):
        if not run:
            return
        letters = "".join((parse.sources[si].value or "") for si in run)
        col = src_fg.get(run[0], "#0f172a")
        pieces.append((min(min(pos[si]) for si in run),
                       '<strong class="wfw-val" style="color:%s">%s</strong>'
                       % (col, escape(letters))))
    for si in acro:
        p0 = min(pos[si])
        if prev is not None and p0 != prev + 1:
            flush(run); run = []
        run.append(si); prev = max(pos[si]); consumed.add(si)
    flush(run)
    # remaining value pieces (ordinary charade tiles)
    for si in pos:
        if si not in consumed:
            pieces.append((min(pos[si]), _pval(parse, si, src_fg)))
    pieces.sort(key=lambda x: x[0])
    if pieces:
        chain = ' <span class="wfw-plus">+</span> '.join(h for _, h in pieces)
        summ = '%s %s' % (chain, _arrow_ans(parse))
        return _build_line(summ) + _grid(_all_rows(parse, src_fg, src_fill))
    return _grid(_all_rows(parse, src_fg, src_fill))


@renders("charade_container_selection")
def _render_charade_container_selection(parse, ctx, src_fg, src_fill):
    """A charade with one container piece whose inner is a letter-selection, e.g.
    TENDER + HEATED around R -> TENDERHEARTED. Outer = the source whose answer letters are
    split around the selection inner; read from per-link op + positions, not guessed."""
    pos, op = {}, {}
    for l in parse.links:
        pos.setdefault(l.source_index, []).append(l.answer_pos)
        op[l.source_index] = l.operation
    outer = [si for si in pos if op.get(si) == "container"]
    inner = [si for si in pos if op.get(si) == "selection"]
    pieces, consumed = [], set()
    if len(outer) == 1 and len(inner) == 1:
        html = ('%s <span class="wfw-around">around</span> %s'
                % (_pval(parse, outer[0], src_fg), _pval(parse, inner[0], src_fg)))
        minp = min(pos[outer[0]] + pos[inner[0]])
        pieces.append((minp, html))
        consumed = {outer[0], inner[0]}
    for si in pos:
        if si not in consumed:
            pieces.append((min(pos[si]), _pval(parse, si, src_fg)))
    pieces.sort(key=lambda x: x[0])
    if pieces:
        chain = ' <span class="wfw-plus">+</span> '.join(h for _, h in pieces)
        return _build_line('%s %s' % (chain, _arrow_ans(parse))) \
            + _grid(_all_rows(parse, src_fg, src_fill))
    return _grid(_all_rows(parse, src_fg, src_fill))


@renders("container_deletion_selection")
def _render_container_built(parse, ctx, src_fg, src_fill):
    """OUTER (a deleted synonym) around INNER (a selection) -> ANSWER. The OUTER's letters are
    split (non-contiguous) around the inner; its DB value and its post-deletion letters (read
    from the links) are both shown (FRIEND -> RIEN around V -> RIVEN)."""
    pos = {}
    for l in parse.links:
        pos.setdefault(l.source_index, []).append(l.answer_pos)
    al = parse.answer_letters()

    def contig(ps):
        ps = sorted(ps)
        return bool(ps) and ps[-1] - ps[0] + 1 == len(ps)
    outer = [si for si, ps in pos.items() if not contig(ps)]
    inner = [si for si, ps in pos.items() if contig(ps)]
    if len(outer) == 1 and inner:
        osi = outer[0]
        oval = (parse.sources[osi].value or "").upper()
        oget = "".join(al[p - 1] for p in sorted(pos[osi]) if 1 <= p <= len(al))
        col = '<strong class="wfw-val" style="color:%s">%s</strong>' % (_src_colour(osi), escape(oval))
        outer_disp = col if oget == oval else (
            '%s <span class="wfw-arrow">&rarr;</span> <strong class="wfw-val">%s</strong>'
            % (col, escape(oget)))
        inners = ' <span class="wfw-plus">+</span> '.join(
            _pval(parse, si, src_fg) for si in sorted(inner))
        summ = ('%s <span class="wfw-around">around</span> %s %s'
                % (outer_disp, inners, _arrow_ans(parse)))
        return _build_line(summ) + _grid(_all_rows(parse, src_fg, src_fill))
    return _grid(_all_rows(parse, src_fg, src_fill))


@renders("hidden")
def _render_hidden(parse, ctx, src_fg, src_fill):
    """hidden in "host phrase" -> ANSWER."""
    host = parse.sources[0].text if parse.sources else ""
    return _build_line('hidden in &ldquo;%s&rdquo; %s' % (escape(host), _arrow_ans(parse))) \
        + _grid(_all_rows(parse, src_fg, src_fill))


@renders("hidden_reversed")
def _render_hidden_rev(parse, ctx, src_fg, src_fill):
    """hidden (reversed) in "host phrase" -> ANSWER."""
    host = parse.sources[0].text if parse.sources else ""
    return _build_line('hidden (reversed) in &ldquo;%s&rdquo; %s'
                       % (escape(host), _arrow_ans(parse))) \
        + _grid(_all_rows(parse, src_fg, src_fill))


@renders("reversal")
def _render_reversal(parse, ctx, src_fg, src_fill):
    """VALUE reversed -> ANSWER."""
    if parse.sources:
        v = escape((parse.sources[0].value or "").upper())
        summ = ('<strong class="wfw-val">%s</strong> <span class="wfw-emuted">reversed</span> %s'
                % (v, _arrow_ans(parse)))
        return _build_line(summ) + _grid(_all_rows(parse, src_fg, src_fill))
    return _grid(_all_rows(parse, src_fg, src_fill))


@renders("deletion")
def _render_deletion(parse, ctx, src_fg, src_fill):
    """VALUE - removed -> ANSWER, when there is a single value source (positional deletion)."""
    from collections import Counter
    vals = [s for s in parse.sources if s.mechanism in ("synonym", "abbreviation", "raw")]
    if len(vals) == 1:
        v = (vals[0].value or "").upper()
        removed = "".join(sorted((Counter(v) - Counter(parse.answer_letters())).elements()))
        if removed:
            summ = ('<strong class="wfw-val">%s</strong> &minus; '
                    '<strong class="wfw-val">%s</strong> %s'
                    % (escape(v), escape(removed), _arrow_ans(parse)))
            return _build_line(summ) + _grid(_all_rows(parse, src_fg, src_fill))
    return _grid(_all_rows(parse, src_fg, src_fill))


@renders("homophone")
def _render_homophone(parse, ctx, src_fg, src_fill):
    """"host phrase" sounds like -> ANSWER."""
    if parse.sources:
        return _build_line('&ldquo;%s&rdquo; sounds like %s'
                           % (escape(parse.sources[0].text), _arrow_ans(parse))) \
            + _grid(_all_rows(parse, src_fg, src_fill))
    return _grid(_all_rows(parse, src_fg, src_fill))


# ---- compound assembly (batch 3) --------------------------------------------------
# Reconstruct the assembly from the LINKS: each source's answer positions tell whether it
# is a plain piece (contiguous), a CONTAINER (its letters split around an inner block), and
# whether its letters are NORMAL / REVERSED / an ANAGRAM / a DELETION of its DB value. Built
# generically so one renderer serves every compound (anagram+container, container+charade,
# charade+deletion, reversal+charade, ...). Falls back to plain rows if the map is incomplete.

def _transform_note(value, got):
    """The 'reversed' / '&minus;deleted-run' marker for a piece whose DB value is `value` and
    whose answer letters IN ANSWER READING ORDER are `got`. '' when they match plainly. Does
    NOT cover anagram (the caller flags anagram_fodder itself). Shared by the assembly build
    line (_piece_label) and the per-piece breakdown rows (_source_row) so they never diverge —
    e.g. a piece that supplies IS but lands as SI shows 'reversed' in BOTH places."""
    v = (value or "").upper()
    if got == v:
        return ""
    if not got:                                          # piece contributes no answer letters
        return ""                                        #   here -> not a reversal/deletion of it
    if got == v[::-1]:
        return ' <span class="wfw-emuted">reversed</span>'
    if len(v) >= 3 and len(got) == len(v):               # a single-letter cyclic shift (rotation)
        if got == v[-1] + v[:-1]:                         #   TERNS -> STERN (tail to the front)
            return ' <span class="wfw-emuted">last&rarr;front</span>'
        if got == v[1:] + v[0]:                          #   STERN -> TERNS (head to the back)
            return ' <span class="wfw-emuted">first&rarr;end</span>'
    for i in range(len(v)):                              # a single contiguous deletion of v,
        for j in range(i + 1, len(v) + 1):               #   leaving a NON-empty survivor (got)
            if v[:i] + v[j:] == got:
                return ' <span class="wfw-emuted">&minus;%s</span>' % escape(v[i:j])
    return ""


def _piece_label(parse, si, positions, answer_letters):
    """The coloured value for source `si`, marked with how its letters reached the answer:
    reversed / anagram / minus-deleted-run, derived from positions vs the DB value."""
    s = parse.sources[si]
    v = (s.value or "").upper()
    got = "".join(answer_letters[p - 1] for p in positions if 1 <= p <= len(answer_letters))
    col = '<strong class="wfw-val" style="color:%s">%s</strong>' % (_src_colour(si), escape(v))
    if s.mechanism == "anagram_fodder":
        from collections import Counter                   # fodder longer than what it fills => a
        removed = "".join(sorted((Counter(v) - Counter(got)).elements()))  # deletion before the
        if removed:                                       # anagram (10 fodder letters -> 9 tiles)
            return col + ' <span class="wfw-emuted">anagram &minus;%s</span>' % escape(removed)
        return col + ' <span class="wfw-emuted">anagram</span>'
    return col + _transform_note(v, got)


def _src_colour(si):
    return PALETTE[si % len(PALETTE)][0]


def _assembly_expr(parse, answer_letters):
    """An HTML expression for the assembly (pieces joined by + / nested with 'around'), or
    None when the links do not cover every answer letter (then the caller falls back)."""
    seq = {}
    for l in parse.links:
        seq[l.answer_pos] = l.source_index
    n = len(answer_letters)
    if any(p not in seq for p in range(1, n + 1)):
        return None

    def render(lo, hi, depth=0):
        if depth > 8:
            return None
        parts, i = [], lo
        while i <= hi:
            si = seq[i]
            sip = [p for p in range(lo, hi + 1) if seq[p] == si]
            start, end = sip[0], sip[-1]
            if end - start + 1 == len(sip):              # contiguous -> a plain piece
                parts.append(_piece_label(parse, si, sip, answer_letters))
                i = end + 1
            else:                                        # split -> a container around the gap
                inner = [p for p in range(start, end + 1) if seq[p] != si]
                sub = render(inner[0], inner[-1], depth + 1)
                if sub is None:
                    return None
                parts.append('%s <span class="wfw-around">around</span> (%s)'
                             % (_piece_label(parse, si, sip, answer_letters), sub))
                i = end + 1
        return ' <span class="wfw-plus">+</span> '.join(parts)

    return render(1, n)


@renders("anagram_container", "container_charade", "charade_deletion", "anagram_charade",
         "container_deletion", "reversal_charade", "reversal_container", "container_outer_charade",
         "container_inner_deletion", "container_inner_alternation", "charade_multi_deletion",
         "manual")
def _render_assembly(parse, ctx, src_fg, src_fill):
    expr = _assembly_expr(parse, parse.answer_letters())
    if expr:
        return _build_line('%s %s' % (expr, _arrow_ans(parse))) \
            + _grid(_all_rows(parse, src_fg, src_fill))
    return _grid(_all_rows(parse, src_fg, src_fill))


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
  .wfw-engine { display:inline-flex; align-items:center; font-size:.62rem;
                font-weight:700; letter-spacing:.04em; color:#64748b;
                background:#f1f5f9; border:1px solid #e2e8f0; border-radius:999px;
                padding:.2rem .55rem; margin-left:.4rem; font-family:monospace; }
  .wfw-type { display:inline-flex; align-items:center; gap:.45rem;
              background:#0f172a; color:#fff; font-size:.72rem; font-weight:700;
              letter-spacing:.08em; padding:.35rem .7rem; border-radius:999px; }
  .wfw-dot { width:.5rem; height:.5rem; border-radius:50%;
             background:#f59e0b; display:inline-block; }
  .wfw-verdict { font-size:.78rem; font-weight:800; letter-spacing:.05em;
                 padding:.35rem .7rem; border-radius:999px; color:#fff; }
  .wfw-verdict.pass { background:#16a34a; }
  .wfw-verdict.pending { background:#d97706; }
  .wfw-verdict.review { background:#7c3aed; }
  .wfw-verdict.fail { background:#dc2626; }
  .wfw-verdict.invalid { background:#475569; }
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
  .wfw-tile-gap { width:1rem; }                       /* word break in a multi-word answer */
  .wfw-tile-sep { display:inline-flex; align-items:center; color:#94a3b8;
                  font-weight:800; font-size:1.3rem; }   /* hyphen in a hyphenated answer */
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
  .wfw-build { font-family:'SF Mono','Courier New',monospace; font-size:1.02rem;
               background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
               padding:.6rem .85rem; margin:0 0 1rem; line-height:1.7; color:#0f172a; }
  .wfw-plus { color:#94a3b8; font-weight:800; margin:0 .15rem; }
  .wfw-eq { color:#94a3b8; font-weight:800; margin:0 .25rem; }
  .wfw-around { color:#0e7490; font-weight:700; font-style:italic; margin:0 .2rem;
                font-family:-apple-system,'Segoe UI',sans-serif; }
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
