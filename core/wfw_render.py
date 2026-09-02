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
    "double_definition": "Double definition",
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
    "reverse_anagram": "Reverse anagram",
    "double_homophone": "Double homophone",
}

# CLUE TYPES whose reviewer comment IS the explanation: there is no piece assembly to
# print, so the comment leads the breakdown and the definition row follows it. No banner —
# the clue is sound and the clue-type badge already names it (user, 2026-08-20).
# Mirrors the same set in core/wfw_web._TYPE_VERDICTS and web/wfw_read.
_COMMENT_LED_OPS = frozenset(("reverse_anagram", "double_homophone"))

# Mechanism words recognised inside an engine operation name ("anagram_container")
# or a manual solve's indicator notes. Mirrors web/wfw_read._MECH_WORDS — keep in
# sync by hand; the site must not import core and vice versa.
_MECH_WORDS = ("anagram", "hidden", "container", "reversal", "deletion",
               "charade", "homophone", "alternation", "selection",
               "palindrome", "spoonerism", "acrostic", "replacement",
               "cycling", "substitution")

# Canonical reading order for a composite clue type — outer operation first,
# inner transforms last. We name EVERY mechanism a clue uses (the full clue-type
# is a differentiator), so the order has to be deterministic. Mirrors
# web/wfw_read._MECH_ORDER — keep in sync by hand.
_MECH_ORDER = ("container", "charade", "anagram", "reversal", "deletion",
               "selection", "hidden", "acrostic", "alternation", "homophone",
               "spoonerism", "palindrome", "cycling", "letter_shift",
               "substitution", "replacement")

# Ops whose curated label is authoritative and must NOT be enriched (no enumerable
# wordplay, or enrichment would drop a defining designation). Mirrors
# web/wfw_read._ATOMIC_OPS.
_ATOMIC_OPS = frozenset((
    "dd", "double_definition", "cd", "andlit", "continuation",
    "hidden", "hidden_reversed", "reverse_anagram", "double_homophone"))

# Ops whose extra sources are NOT charade pieces, so a charade must NOT be inferred
# when one is present (gather ops fold several words into one gestalt; substitution
# swaps letters in place).
# HOMOPHONE is NOT here (user-reported 2026-08-20, EYEBALLING guardian 30090 11a =
# EYE ["vote in favour", sounds like AYE] + BALLING ["outcry", sounds like BAWLING]).
# A homophone piece is a WHOLE PIECE that sounds like something else, so two of them —
# or one beside a synonym — sit side by side exactly as charade parts do; suppressing
# the charade labelled the clue "Homophone" and misled any reader who took the
# clue-type hint. Each sound belongs to ONE source (its own 'sounds like' transform),
# so the join count cannot invent a charade out of one sound: measured over all 6,590
# stored solves, 37 labels gain the charade and every one has genuinely separate
# pieces. Mirrors web/wfw_read._CHARADE_SUPPRESS.
# ANAGRAM IS NOT HERE EITHER (user-reported 2026-08-25, POTATO BLIGHT telegraph 31323
# 14a = TOPBOAT anagrammed by "at sea" + dawn=LIGHT, which read "Anagram"). Same fault as
# the homophone one above and the same reasoning: a multi-word anagram's fodder is stored
# as ONE source ("top boat" -> TOPBOAT), so it contributes ONE placed piece and cannot
# inflate the join count. The gestalt argument was about several clue WORDS, not several
# PIECES, and the two were conflated. Measured over all 5,298 stored passes: 577 labels
# gain the charade and EVERY ONE has 2+ placed sources — zero false charades.
# Still suppressed and NOT measured: acrostic, alternation, spoonerism, hidden,
# palindrome, cycling, substitution, replacement. Whether the same conflation hides a
# charade in those is an open question — measure before removing any of them.
_CHARADE_SUPPRESS = frozenset((
    "acrostic", "alternation", "spoonerism",
    "hidden", "palindrome", "cycling", "substitution", "replacement"))


def _note_mech(note):
    """The single mechanism an indicator note denotes, or None to skip. Handles the
    note-vocabulary variants that don't spell the mechanism verbatim (insertion =>
    container; 'first-letter indicator' => selection; 'deleted letters' =>
    deletion). Mirrors web/wfw_read._note_mech."""
    n = (note or "").lower()
    if (not n or "definition by example" in n or "positional" in n
            or "surface" in n or n == "wordplay indicator"):
        return None
    if "insertion" in n or "container" in n:
        return "container"
    if ("selection" in n or "first-letter" in n or "last-letter" in n
            or "middle-letter" in n or "outer-letter" in n):
        return "selection"
    if "acrostic" in n:
        return "acrostic"
    if "letter_shift" in n:
        return "letter_shift"
    if "deletion" in n or "deleted" in n:
        return "deletion"
    for w in _MECH_WORDS:
        if w in n:
            return w
    return None


def _note_mechs(parse):
    """The mechanism set named by a manual/engine solve's indicator notes, plus the
    count of container/insertion joins. Charade is added by the caller from the
    join count (it carries no indicator)."""
    found = set()
    container_joins = 0
    for a in (parse.annotations or []):
        if getattr(a, "role", "") != "indicator":
            continue
        m = _note_mech(getattr(a, "note", ""))
        if m is None:
            continue
        found.add(m)
        if m == "container":
            container_joins += 1
    return found, container_joins


def _has_charade(placed_pieces, container_joins, mechs):
    """Charade (side-by-side concatenation) carries no indicator, so we infer it by
    counting joins: PLACED pieces (sources that actually contribute answer letters —
    a deleted/removed source places nothing) need placed-1 binary joins, each
    container nesting is one join, any leftover join is a charade. Suppressed when a
    _CHARADE_SUPPRESS op accounts for the extra sources. Mirrors web/wfw_read."""
    if mechs & _CHARADE_SUPPRESS:
        return False
    return placed_pieces - 1 > container_joins


def _order_mechs(found):
    """Canonical outer→inner ordering of a mechanism set into 'Container + charade +
    selection'. Mirrors web/wfw_read._order_mechs."""
    ordered = [m for m in _MECH_ORDER if m in found]
    ordered += [m for m in found if m not in _MECH_ORDER]
    return (" + ".join(ordered)).replace("_", " ").capitalize()


def _placed_pieces(parse):
    return len({l.source_index for l in (parse.links or [])})


def _manual_type_label(parse):
    """The clue type of a manual/prefill parse, naming EVERY mechanism it uses —
    mirrors web/wfw_read._manual_label so the clue page badge and the live-site
    hint label always agree."""
    found, container_joins = _note_mechs(parse)
    if _has_charade(_placed_pieces(parse), container_joins, found):
        found.add("charade")
    if found:
        return _order_mechs(found)
    srcs = [s for s in (parse.sources or []) if s.mechanism != "definition"]
    if len(srcs) >= 2:
        return "Charade"
    if len(srcs) == 1:
        return {"synonym": "Synonym", "abbreviation": "Abbreviation",
                "hidden": "Hidden word"}.get(srcs[0].mechanism, "Word building")
    return "Word building"


def _engine_type_label(parse):
    """The clue type of an engine-solved parse, naming EVERY mechanism it uses. The
    engine op-name flattens composites (op='container' hides an inner acrostic +
    charade), and the rich indicator notes are more complete, so we read the
    mechanisms from notes+pieces and, when that reveals MORE than the op name names,
    build the full clue-type; otherwise keep the curated op-name label. Mirrors
    web/wfw_read._wordplay_label's engine branch."""
    op = parse.operation or ""
    if op in _ATOMIC_OPS:
        return _TYPE_LABEL.get(op) or op.replace("_", " ").capitalize()
    op_mechs = set(w for w in op.split("_") if w in _MECH_WORDS)
    if "insertion" in op_mechs:
        op_mechs.discard("insertion"); op_mechs.add("container")
    found, container_joins = _note_mechs(parse)
    if "container" in op_mechs:
        container_joins = max(container_joins, 1)   # op name declares the nesting
    allm = op_mechs | found
    if _has_charade(_placed_pieces(parse), container_joins, allm):
        allm.add("charade")
    # A composite (2+ mechanisms) always renders in canonical order; a single
    # mechanism that discovered something new does too. Only a single-mechanism op
    # with nothing new keeps its curated (nicer) label. Keeps badge and hint
    # identical regardless of which curated compound entries each map carries.
    if allm and (len(allm) >= 2 or allm != op_mechs):
        return _order_mechs(allm)
    if op in _TYPE_LABEL:
        return _TYPE_LABEL[op]
    if op_mechs:
        return _order_mechs(op_mechs)
    return op.replace("_", " ").capitalize() if op else "—"

# Friendly role labels for a wordplay piece, by mechanism.
_MECH_LABEL = {
    "hidden": "Hidden in",
    "hidden_reversed": "Hidden in (rev.)",
    "synonym": "Synonym",
    "abbreviation": "Substitution",   # wordplay-table values (abbrevs, Roman numerals, compass
                                      # points, symbols) — "substitution" is the accurate umbrella
    "raw": "Literal",
    "replacement_letter": "New letter",   # unclued replacement ("with new leader"): the
                                          # letter comes from the answer, not a clue word
    "first_letter": "Initial",
    "last_letter": "Last letter",
    "outer": "Outer letters",
    "homophone": "Sounds like",
    "spoonerism": "Spoonerism of",   # vetted pair (spoonerisms table): value = source phrase
    "anagram_fodder": "Anagram of",
    "alternate": "Alternate letters",
    "definition": "Definition",
}


def _colour(i):
    return PALETTE[i % len(PALETTE)]


def render_parse(parse, ctx=None, clue_line_html=None, coloured=True, comment=""):
    """Return the HTML fragment for one solved clue (the shared base screen).

    A clue type customises only:
    - `clue_line_html`: a pre-rendered clue line (hidden lights the host letters);
    - `coloured`: per-source palette colour on or off (hidden runs uncoloured and
      uses the single amber accent instead);
    - `comment`: the reviewer's own words. A REVERSE ANAGRAM has no assemblable
      wordplay to render — the answer read as wordplay produces a phrase in the
      clue — so the comment IS the explanation and is rendered above the
      definition. Ignored for every other clue type.
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
    # A manual/prefill parse stores the useless operation 'manual' — derive its
    # REAL clue type from the indicator notes instead, exactly as the live-site
    # hints do (web/wfw_read._manual_label, phase 1). Regression caught by the
    # user 2026-07-12: the clue page is now the prefill REVIEW surface, so the
    # badge must read "ANAGRAM", not "MANUAL".
    _op = parse.operation or ""
    _status = (getattr(parse, "status", "") or "")
    # A FAILED parse HAS NO CLUE TYPE, and the badge is omitted entirely.
    #
    # An engine that fails still leaves its own name in `operation` as
    # fail-evidence — the anagram engine's "no anagram signature matched this
    # clue" leaves operation='anagram'. Badging that asserts a clue type nobody
    # established, and it survives: GUARDIAN 30101 19d is "See 8", a
    # continuation stub with no wordplay at all, and it badged ANAGRAM
    # (2026-09-02). A type we could not derive must be absent, not guessed.
    # ...and neither has a parse with NOTHING IN IT. A shared-enumeration stub
    # ("See 8", GUARDIAN 30101 19d) has no sources and no definition: there is
    # no wordplay to name, whatever its status, so any badge is a leftover from
    # an engine that tried and failed. Derived from the parse, not guessed.
    # INVALID IS TESTED FIRST and keeps "unsound". A rejected reading is a
    # verdict about the reading, and it must not be silently swallowed by the
    # empty-parse rule below.
    _empty = not getattr(parse, "sources", None) and not getattr(parse, "definition", None)
    if _status == "invalid":
        type_label = "unsound"          # do not badge the rejected reading's clue-type
    elif _status == "fail" or _empty:
        type_label = ""
    elif _op == "manual":
        type_label = _manual_type_label(parse)
    else:
        type_label = _engine_type_label(parse)
    engine = (getattr(parse, "solved_by", "") or "").strip()
    # Always show the SPECIFIC solving engine (parse.solved_by) so it is clear at a glance
    # which engine produced the parse — no DB query, no stack-trace hunting.
    eng_tag = ""
    # ...but NOT on an empty parse. `solved_by` there names the engine that
    # FAILED on it, and a faint pill reading "anagram" beside a stub is read as
    # a clue type by anyone looking at the card — which is exactly what it must
    # not say. No parse, no engine credit.
    if engine and not _empty:
        eng_tag = '<span class="wfw-engine" title="solving engine">%s</span>' % escape(engine)
    # No label => no badge. An empty pill would still show the dot and the
    # colour, which reads as a type the viewer cannot make out rather than as
    # the absence of one.
    type_html = ("" if not type_label else
                 '<span class="wfw-type"><span class="wfw-dot"></span>%s</span>'
                 % escape(type_label.upper()))
    header = ('<div class="wfw-head">%s%s%s</div>'
              % (type_html, eng_tag, _verdict_badge(parse)))

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

    # INVALID = the reviewer judged the stored wordplay UNSOUND. Never render that wordplay
    # breakdown (it is exactly the wrong parse the user does not want shown) — show an honest
    # note instead. The answer tiles + the INVALID badge stay; the reviewer's comment is shown
    # by the page around this card. Systemic: covers every invalid clue, not one at a time.
    if (getattr(parse, "status", "") or "") == "invalid":
        breakdown = ('<div class="wfw-row"><em>Marked INVALID — the stored wordplay is '
                     'unsound, so it is not shown. See the comment for why.</em></div>')
    elif _empty:
        # NOTHING IN THE PARSE => NOTHING TO LAY OUT. The operation renderers are
        # dispatched on `parse.operation`, which on a stub still holds the name of
        # whichever engine last failed on it — so "See 15" was dispatched to the
        # ANAGRAM renderer and produced the line "anagram of  -> RECORD", with
        # empty fodder (user, 2026-09-02). A shared-enumeration stub has no
        # wordplay; the tiles and the verdict say everything there is to say.
        breakdown = ""
    else:
        renderer = _TYPE_RENDERERS.get(parse.operation or "", _render_generic_breakdown)
        breakdown = renderer(parse, ctx, src_fg, src_fill)
        # REVERSE ANAGRAM: the answer itself, read as wordplay, produces a phrase in the
        # clue — there is no chain of pieces to lay out, so the reviewer's comment carries
        # the mechanism and leads the breakdown. The definition row still follows it (every
        # clue ends with a definition). No banner: the clue is sound, and the clue-type
        # badge already names it (user, 2026-08-20).
        if (parse.operation or "") in _COMMENT_LED_OPS and (comment or "").strip():
            breakdown = ('<div class="wfw-comment">%s</div>'
                         % escape(comment.strip()).replace("\n", "<br>")) + breakdown

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
               "acrostic": "#5b21b6", "letter-shift": "#0369a1", "positional": "#475569",
               "indicator": "#7c3aed"}


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
                     "first_end": "move first letter to end",
                     "move_left": "move letter left",
                     "move_right": "move letter right",
                     "named": "the named letters change places"},
    "charade_positional": {"after": "this piece goes after its neighbour",
                           "before": "this piece goes before its neighbour"},
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
        return "Letter-shift indicator", _SUBTYPE_DETAIL["letter_shift"].get(sub, sub)
    if "charade_positional" in n:                  # "charade_positional/after indicator"
        # a positional/charade indicator ("after", "before") — tells the reader WHERE the
        # piece sits relative to its neighbour, not a bare "Indicator".
        sub = n.split("/", 1)[1].replace("indicator", "").strip() if "/" in n else ""
        return "Positional indicator", _SUBTYPE_DETAIL["charade_positional"].get(sub, sub)
    if n.startswith("named/"):
        # "named/French indicator" — the human typed what this one does, because
        # no fixed type fits (GUARDIAN 30101 22a: "in Le Mans" = read the next
        # words in French).
        # MIRROR: web/wfw_read.py has the matching branch — keep them in step.
        sub = note.split("/", 1)[1].replace("indicator", "").strip()
        return (sub[:1].upper() + sub[1:] + " indicator") if sub else "Indicator", ""
    for t in _IND_TYPES:
        if t in n:
            disp = "Container" if t == "insertion" else t.capitalize()
            # manual "type/subtype indicator" format (e.g. "selection/first", "deletion/head") —
            # SHOW the sub-type, friendly-mapped, so it is not a bare "Selection indicator".
            sub = n.split("/", 1)[1].replace("indicator", "").strip() if "/" in n else ""
            return disp + " indicator", _SUBTYPE_DETAIL.get(t, {}).get(sub, sub)
    # A note this function does not recognise USED TO BE DISCARDED, and the row
    # rendered as a bare "Indicator" — so a human-named type vanished from the
    # card with no sign it had ever been there (2026-09-02, "French Translation
    # indicator" on GUARDIAN 30101 22a). If the note names something and calls
    # itself an indicator, SHOW IT: whatever the author wrote is more
    # informative than the word "Indicator" on its own, and this also repairs
    # rows stored before the named type existed.
    if n.endswith("indicator"):
        named = note[: -len("indicator")].strip().strip("/")
        if named:
            return named[:1].upper() + named[1:] + " indicator", ""
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


# ---- letter-selection highlighting ------------------------------------------------
# For a piece whose letters are SELECTED from the fodder (initials, alternate letters,
# outer letters, ...) show WHICH letters were taken instead of a bare "address -> DRS".
# Render-only: every candidate pattern is verified to actually spell the value, so we
# never highlight letters that don't reproduce it — if none verifies we return None and
# the caller falls back to the plain fodder text (no regression). Mechanism == the rule
# for the named ones; the generic "selection" bucket is resolved by trying the patterns.
_SELECTION_MECHS = {"selection", "first_letter", "last_letter",
                    "outer", "middle", "alternate", "acrostic"}


def _alpha_indices(text):
    """Indices of the alphabetic characters in `text`, in order."""
    return [i for i, ch in enumerate(text) if ch.isalpha()]


def _cand_initials(text):
    return [i for i, ch in enumerate(text)
            if ch.isalpha() and (i == 0 or not text[i - 1].isalpha())]


def _cand_finals(text):
    return [i for i, ch in enumerate(text)
            if ch.isalpha() and (i == len(text) - 1 or not text[i + 1].isalpha())]


def _cand_outer(text):
    a = _alpha_indices(text)
    return sorted(set([a[0], a[-1]])) if a else []


def _cand_alt(text, start):
    return _alpha_indices(text)[start::2]


def _cand_middle(text, n):
    a = _alpha_indices(text)
    if n <= 0 or n > len(a):
        return []
    off = (len(a) - n) // 2
    return a[off:off + n]


def _cand_greedy(text, want):
    picks, wi = [], 0
    for i, ch in enumerate(text):
        if wi < len(want) and ch.isalpha() and ch.upper() == want[wi]:
            picks.append(i)
            wi += 1
    return picks if wi == len(want) else []


def _cand_greedy_right(text, want):
    """Like _cand_greedy but matches from the RIGHT — the last occurrence of the value's
    letters. This is what a 'last letter' selection means: athlete's -> E is the SECOND e
    (athletE's), not the first (athlEte's)."""
    picks, wi = [], len(want) - 1
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if wi >= 0 and ch.isalpha() and ch.upper() == want[wi]:
            picks.append(i)
            wi -= 1
    return sorted(picks) if wi < 0 else []


_SELECTION_RULE_SUBS = ("first", "last", "outer", "middle", "alternate")


def _selection_rule(parse):
    """The selection sub-rule (first/last/outer/middle/alternate) declared by the clue's
    selection indicator, or None. The generic 'selection' mechanism doesn't carry the rule
    on the source piece — the human's assignment stores it on the indicator (e.g. note
    'selection/last indicator'), so read it there rather than re-guessing which letters."""
    for a in getattr(parse, "annotations", []):
        if getattr(a, "role", "") != "indicator":
            continue
        n = (getattr(a, "note", "") or "").lower()
        if "selection" not in n:
            continue
        for sub in _SELECTION_RULE_SUBS:
            if sub in n:
                return sub
    return None


def _selection_picks(text, value, mechanism, rule=None):
    """Indices of the fodder letters that spell `value` under the selection rule, or None.

    `rule` is the sub-type from the human's assignment (first/last/outer/middle/alternate)
    when the mechanism is the generic 'selection'; the named mechanisms carry their own rule.
    Applies that rule DIRECTIONALLY — 'last' matches the rightmost letters, 'first' the
    leftmost — so a duplicated letter (athlete's -> last E) is taken from the correct end.
    Only returns a candidate whose highlighted letters exactly equal the value, so the
    highlight can never contradict the answer."""
    want = [c.upper() for c in value if c.isalpha()]
    if not want:
        return None
    n = len(want)

    def ok(idxs):
        return idxs if [text[i].upper() for i in idxs] == want else None

    eff = rule or {
        "first_letter": "first", "acrostic": "first",
        "last_letter": "last", "outer": "outer",
        "middle": "middle", "alternate": "alternate",
    }.get(mechanism)

    order = {
        "first":     [_cand_initials, lambda t: _cand_greedy(t, want)],
        "last":      [_cand_finals, lambda t: _cand_greedy_right(t, want)],
        "outer":     [_cand_outer],
        "alternate": [lambda t: _cand_alt(t, 0), lambda t: _cand_alt(t, 1)],
        "middle":    [lambda t: _cand_middle(t, n), lambda t: _cand_greedy(t, want)],
    }.get(eff, [
        _cand_initials, _cand_finals,
        lambda t: _cand_alt(t, 0), lambda t: _cand_alt(t, 1),
        _cand_outer, lambda t: _cand_middle(t, n),
    ])
    for gen in order:
        picks = ok(gen(text))
        if picks:
            return picks
    # Universal last resort — a valid subsequence from whichever end the rule prefers.
    if eff == "last":
        return ok(_cand_greedy_right(text, want)) or ok(_cand_greedy(text, want))
    return ok(_cand_greedy(text, want)) or ok(_cand_greedy_right(text, want))


def _selection_fodder_html(text, value, mechanism, rule=None):
    """The fodder text with selected letters highlighted and the rest dimmed, or None
    when the selection can't be reproduced (caller then shows the plain fodder)."""
    picks = _selection_picks(text, value, mechanism, rule)
    if not picks:
        return None
    pick = set(picks)
    out = []
    for i, ch in enumerate(text):
        e = escape(ch)
        if not ch.isalpha():
            out.append(e)
        elif i in pick:
            out.append('<span class="wfw-sel">%s</span>' % e)
        else:
            out.append('<span class="wfw-unsel">%s</span>' % e)
    return "".join(out)


# ---- reusable row builders (shared by every renderer) ----------------------------

def _source_row(parse, si, src_fg, src_fill):
    """One source piece row: 'wood -> BALSA' coloured by source, with the homophone-via-
    synonym aside and the provisional badge."""
    s = parse.sources[si]
    label = _MECH_LABEL.get(s.mechanism, s.mechanism)
    style = "background:%s;color:%s" % (src_fill[si], src_fg[si])
    # For a letter-selection piece, highlight WHICH fodder letters were taken. For the
    # generic 'selection' mechanism the rule (first/last/...) lives on the indicator, so
    # read it from the assignment rather than re-guessing which of a repeated letter to take.
    fodder = None
    if s.mechanism in _SELECTION_MECHS:
        rule = _selection_rule(parse) if s.mechanism == "selection" else None
        fodder = _selection_fodder_html(s.text, s.value, s.mechanism, rule)
    content = ('%s <span class="wfw-arrow">&rarr;</span> '
               '<strong class="wfw-val">%s</strong>'
               % (fodder if fodder else escape(s.text), escape(s.value)))
    # Show HOW the piece's letters reached the answer (reversed / minus a deleted run), so a
    # piece that supplies IS but lands as SI reads "is -> IS reversed" here too — not a bare IS
    # whose order isn't in the answer. Matches the assembly build line (same _transform_note).
    # Spoonerism pieces are a SOUND pair — their value never letter-matches the tiles, and a
    # DEFINITION source (both halves of a double definition) places no letters at all: neither
    # has letters that could disagree with its value, so neither goes through this at all.
    if s.mechanism not in ("anagram_fodder", "spoonerism",
                           "definition", "definition_by_example"):
        al = parse.answer_letters()
        positions = sorted(l.answer_pos for l in parse.links if l.source_index == si)
        got = "".join(al[p - 1] for p in positions if 1 <= p <= len(al))
        # The RECORD first (what the solve itself says happened), then the anagram
        # indicator, then — for rows written before pieces recorded anything — the old
        # letter-derivation, and finally an explicit "not accounted for" rather than the
        # silence that used to read as "landed unchanged".
        rec = _recorded_note(s, got)
        ana = _anagram_note(parse, (s.value or "").upper(), got) if rec is None else None
        # A PROVEN named shift accounts for this piece at ASSEMBLY level: its own
        # letters are right, they simply land displaced because two letters traded
        # places across the join. Judging the piece alone called that "not accounted
        # for" — the very complaint that started this (user, 2026-08-18): the label
        # ignored what the assembly needed in order to work.
        _shifted = (rec is None and ana is None
                    and _named_shift_proof(parse) is not None
                    and sorted(got) == sorted(_letters_of(s.value)))
        content += (rec if rec is not None else
                    (ana if ana else
                     (' <span class="wfw-emuted">moved by the exchange</span>' if _shifted
                      else (_transform_note(s.value, got, parse)
                            or _unexplained_note(s.value, got)))))
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
        # Just "manual": the commit DOES save the definition to the reference DB
        # (wfw_web /hsmanualcommit db_adds), so the old "(not in DB)" claim was false.
        prov = ' <span class="wfw-prov">manual</span>'
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
    elif a.role == "shifted":
        # a word that NAMES a letter which moves ("tense" -> T). Letterless: the letter is
        # already on the board inside the shifted piece — this row says WHICH one moved.
        style, label = "background:#c2410c;color:#fff", "Letter moved"
        moved = note.split(":", 1)[1].strip() if ":" in note else ""
        if moved:
            content += (' <span class="wfw-arrow">&rarr;</span> '
                        '<strong class="wfw-val">%s</strong>' % escape(moved))
    elif a.role == "link" and note == "synonym by example":
        # a perhaps/maybe word marking a by-example synonym — accounted, letterless, no
        # validity (the wordplay twin of definition-by-example); its own pill, not "Link".
        style, label = "background:#0891b2;color:#fff", "Synonym by example"
    else:
        style, label = "background:#64748b;color:#fff", "Link"
    if getattr(a, "source", "db") == "pending":
        content += ' <span class="wfw-prov">provisional</span>'
    return _row(_first_index(a.clue_atom_ids), label, style, content)


def _all_rows(parse, src_fg, src_fill):
    """Every row (sources + definition + annotations), unsorted: [(sort_i, html), ...].

    ONE WORD, ONE ROW. An indicator that governs several pieces is recorded once
    per piece, so WISTERIA ("Climber regularly waits at sea, wind rising") stored
    "regularly" three times — against waits->WIS, at->T and sea->E — and the card
    printed the identical "Selection indicator: regularly" row three times over
    (user, 2026-08-29). The reader learns nothing from the repeats: one word did
    one job. Rows identical in BOTH clue position and rendered html are therefore
    collapsed to the first. Nothing else can collide — same position plus same
    html means the same clue word annotated the same way — so a genuinely
    different role, sub-type or value still gets its own row.

    Display only. The pieces keep their per-piece indicator record, which is what
    the assembly and the letter highlighting read.
    """
    rows = [_source_row(parse, si, src_fg, src_fill) for si in range(len(parse.sources))]
    d = _definition_row(parse)
    if d:
        rows.append(d)
    seen = set()
    for a in parse.annotations:
        row = _annotation_row(parse, a)
        if row in seen:
            continue
        seen.add(row)
        rows.append(row)
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


def _has_named_shift(parse):
    """True when this parse carries a letter-shift indicator whose sub-type is
    'named' — the clue NAMES the letters that move, rather than rotating an end."""
    for a in getattr(parse, "annotations", None) or []:
        n = (getattr(a, "note", "") or "").lower()
        if n.startswith(("letter_shift", "letter-shift", "letter shift")) \
                and "/named" in n:
            return True
    return False


def _shift_named_letters(parse):
    """The letters the clue names as moving, in clue order — read from the
    'shifted' annotations ("tense" -> T, "Romeo" -> R). These place no tiles;
    they say WHICH letters move."""
    out = []
    for a in sorted((getattr(parse, "annotations", None) or []),
                    key=lambda x: _first_index(x.clue_atom_ids)):
        if getattr(a, "role", "") != "shifted":
            continue
        note = getattr(a, "note", "") or ""
        val = note.split(":", 1)[1].strip().upper() if ":" in note else ""
        if len(val) == 1:
            out.append(val)
    return out


def _letter_shift_detail(parse):
    """Readable direction of a letter-shift indicator on this parse, or None. A letter-shift
    re-orders letters WITHIN a piece, so it contributes no chain segment; the summary must
    name it explicitly or a charade + letter-shift reads as a plain charade. Label-only."""
    for a in getattr(parse, "annotations", None) or []:
        n = (getattr(a, "note", "") or "").lower()
        if (n.startswith("letter_shift") or n.startswith("letter-shift")
                or n.startswith("letter shift")):
            sub = n.split("/", 1)[1].replace("indicator", "").strip() if "/" in n else ""
            return _SUBTYPE_DETAIL["letter_shift"].get(sub, sub or "letter shift")
    return None


@renders("charade", "charade_alternation")
def _render_charade(parse, ctx, src_fg, src_fill):
    """A + B + C -> ANSWER, pieces in clue order, then the detailed rows."""
    order = sorted(range(len(parse.sources)),
                   key=lambda si: _first_index(parse.sources[si].clue_atom_ids))
    chain = ' <span class="wfw-plus">+</span> '.join(_pval(parse, si, src_fg) for si in order)
    summ = ('%s <span class="wfw-arrow">&rarr;</span> '
            '<strong class="wfw-val">%s</strong>' % (chain, escape((parse.answer_text or "").upper())))
    _ls = _letter_shift_detail(parse)
    if _ls:
        summ += ' <span class="wfw-emuted">(%s)</span>' % escape(_ls)
    return _build_line(summ) + _grid(_all_rows(parse, src_fg, src_fill))


@renders("anagram")
def _render_anagram(parse, ctx, src_fg, src_fill):
    """anagram of FODDER [- removed letters] -> ANSWER, then the detailed rows.

    EVERY PIECE THE ANAGRAM EATS APPEARS IN THE LINE. The fodder used to be
    "sources whose mechanism is anagram_fodder", which silently dropped any
    piece that reaches the anagram by another route — an abbreviation, a first
    or last letter, a synonym. WYOMING ("Leader of government on the right, I
    own my rogue state") read "anagram of I OWN MY -> WYOMING": six letters
    producing seven, with the G from "Leader of government" nowhere in the
    summary, though its own row was there below (user, 2026-08-30).

    The links are the authority, not the mechanism label: a source whose letters
    land in the answer under an `anagram` link IS fodder, whatever produced its
    value. 47 of 439 stored anagram passes were understating themselves this way.

    Clue order, per the display convention — verified not to reorder any
    existing line (source order already matched clue order in all 439).
    """
    from collections import Counter
    eaten = {l.source_index for l in (parse.links or [])
             if (l.operation or "") == "anagram"}
    fodder_si = [si for si, s in enumerate(parse.sources)
                 if s.mechanism == "anagram_fodder" or si in eaten]
    fodder_si.sort(key=lambda si: _first_index(parse.sources[si].clue_atom_ids))
    fodder = [(parse.sources[si].value or "").upper() for si in fodder_si]
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

def _recorded_note(src, got):
    """The marker for a piece that RECORDS what happened to its value (the transform
    stored on it at authoring, 2026-08-17) — or None when it records nothing.

    This is the whole point of storing it: the card READS the change instead of
    working it out from the letters, so a composed change (cut AND reversed) is
    stated rather than falling through the guesses to silence. The record is still
    checked against the letters it claims to place — a stored transform that does
    not spell the tiles is ignored, never shown."""
    from core import piece_transform
    t = piece_transform.loads(getattr(src, "transform", "") or "")
    if piece_transform.empty(t):
        return None
    if not piece_transform.places(src.value, t, got):
        return None                                      # record disagrees with the tiles
    return ' <span class="wfw-emuted">%s</span>' % escape(piece_transform.short(src.value, t))


def _unexplained_note(value, got):
    """The marker for a piece whose letters are NOT its value and where nothing —
    no stored record, no anagram indicator, no single recognisable change — accounts
    for the difference. Saying so is the point: the old code returned '' here, which
    renders as 'the value landed unchanged' and produced assembly lines that do not
    spell the answer (EPHESUS 10085630, OMELETTE 10081158). An explanation we cannot
    stand behind must LOOK like one.

    It applies ONLY to a piece that actually places letters. A source contributing
    NONE — both halves of a double definition, a source whose letters are removed —
    has no letters to disagree with its value, and flagging it said every double
    definition was unaccounted for (user-reported 2026-08-18: "Got plastered?" =
    RENDERED). Same guard _transform_note has always had."""
    from core import piece_transform
    if not piece_transform.letters_only(got):
        return ""                                # places nothing here — nothing to explain
    if piece_transform.letters_only(value) == piece_transform.letters_only(got):
        return ""
    return ' <span class="wfw-unex">not accounted for</span>'


def _transform_note(value, got, parse=None):
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
    idx = v.find(got)                                    # 'peeled': got survives as a contiguous
    if 0 < idx and idx + len(got) < len(v):              #   interior run of v, with BOTH a removed
        found = _note_mechs(parse)[0] if parse is not None else set()
        if "deletion" in found:                          #   prefix AND suffix — claimed ONLY when
            pre, suf = v[:idx], v[idx + len(got):]        #   the clue names a deletion (SQUID->QUI),
            return (' <span class="wfw-emuted">&minus;%s &minus;%s</span>'  # never on a hidden word
                    % (escape(pre), escape(suf)))
    return ""


def _anagram_note(parse, v, got):
    """'anagram [&minus;X]' when the clue names an anagram indicator and `got` (the piece's
    answer letters, in answer order) is a re-ordering of the piece value `v` — a sub-multiset
    of it. Lets a source roled selection/synonym still read as anagram fodder when the
    assignment names an anagram indicator, so the per-piece line agrees with the type badge
    instead of guessing a rotation/reversal off the incidental letter order. Returns None to
    fall through to _transform_note. A real reversal (reversal indicator + exact reverse)
    keeps priority. This renderer IS the clue-page card on BOTH the admin solver and the
    public site (via core/wfw_card.stored_card), so the fix covers both. The separate
    web/wfw_read._summary (hints/overlay text) groups multi-piece anagrams its own way."""
    from collections import Counter
    if not got or got == v:                              # unchanged order => not this piece
        return None
    found, _ = _note_mechs(parse)
    if "anagram" not in found:
        return None
    if Counter(got) - Counter(v):                        # got uses letters v doesn't have
        return None                                      #   => not an anagram OF v
    it = iter(v)                                          # got is v with letters dropped, order
    if all(ch in it for ch in got):                      #   KEPT (an in-order survivor) => a
        return None                                      #   plain deletion, not a re-ordering
    if "reversal" in found and len(v) > 1 and got == v[::-1]:
        return None                                      # a real reversal keeps priority
    removed = "".join(sorted((Counter(v) - Counter(got)).elements()))
    if removed:                                          # value longer than the tiles: a
        return ' <span class="wfw-emuted">anagram &minus;%s</span>' % escape(removed)  # deletion
    return ' <span class="wfw-emuted">anagram</span>'    #   before the anagram


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
    rec = _recorded_note(s, got)                    # what the piece RECORDS (never derived)
    if rec is not None:
        return col + rec
    ana = _anagram_note(parse, v, got)
    return col + (ana if ana else
                  (_transform_note(v, got, parse) or _unexplained_note(v, got)))


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


def _letters_of(text):
    return "".join(c for c in (text or "").upper() if c.isalpha())


def _named_shift_proof(parse):
    """(order, values, joined, answer, letterA, letterB) when a NAMED letter shift
    really does account for this whole assembly — the pieces joined in clue order,
    with the two named letters exchanged, spelling the answer EXACTLY. None
    otherwise. This is the arithmetic; every caller relies on it rather than
    trusting that a clue mentioning an exchange has one."""
    if not _has_named_shift(parse):
        return None
    letters = _shift_named_letters(parse)          # the letters the clue NAMES
    if len(letters) != 2:
        return None
    order = sorted(range(len(parse.sources)),
                   key=lambda si: _first_index(parse.sources[si].clue_atom_ids))
    vals = [(parse.sources[si].value or "").upper() for si in order]
    if not all(vals):
        return None
    joined = "".join(c for v in vals for c in v if c.isalpha())
    answer = "".join(c for c in (parse.answer_text or "").upper() if c.isalpha())
    if sorted(joined) != sorted(answer):
        return None
    a, b = letters
    i, j = joined.find(a), joined.find(b)
    if i < 0 or j < 0:
        return None
    swapped = list(joined)
    swapped[i], swapped[j] = swapped[j], swapped[i]
    if "".join(swapped) != answer:                 # the exchange must SPELL the answer
        return None
    return order, vals, joined, answer, a, b


def _named_shift_line(parse, src_fg):
    """The assembly line for a NAMED letter shift, or None.

    A named shift ("tense exchanges with Romeo") operates on the WHOLE assembly,
    not on one piece: the T comes from `met` and the R from `Curio`, and they
    trade places across the join. Read per-piece, each piece's letters look
    scattered, and _assembly_expr describes the interleaving as two pieces
    containing each other — MET around (RCUIO -IO) + RCUIO around (MET -ME),
    which is nonsense (user-reported 2026-08-18).

    So: join the pieces IN CLUE ORDER, apply the exchange the clue names, and
    print that. PROVEN, never asserted — the joined letters must actually become
    the answer under the named exchange, or this returns None and the ordinary
    renderer runs. A card may not claim an assembly that does not spell the
    answer."""
    proof = _named_shift_proof(parse)
    if proof is None:
        return None
    order, vals, joined, answer, a, b = proof
    chain = ' <span class="wfw-plus">+</span> '.join(
        '<strong class="wfw-val" style="color:%s">%s</strong>' % (_src_colour(si), escape(v))
        for si, v in zip(order, vals))
    return ('%s <span class="wfw-arrow">&rarr;</span> '
            '<strong class="wfw-val">%s</strong> '
            '<span class="wfw-emuted">%s&harr;%s exchanged</span> '
            '<span class="wfw-arrow">&rarr;</span> '
            '<strong class="wfw-val">%s</strong>'
            % (chain, escape(joined), escape(a), escape(b), escape(answer)))


@renders("anagram_container", "container_charade", "charade_deletion", "anagram_charade",
         "container_deletion", "reversal_charade", "reversal_container", "container_outer_charade",
         "container_inner_deletion", "container_inner_alternation", "charade_multi_deletion",
         "manual")
def _render_assembly(parse, ctx, src_fg, src_fill):
    named = _named_shift_line(parse, src_fg)      # the exchange happens AFTER the join
    if named:
        return _build_line(named) + _grid(_all_rows(parse, src_fg, src_fill))
    expr = _assembly_expr(parse, parse.answer_letters())
    if expr:
        line = '%s %s' % (expr, _arrow_ans(parse))
        _ls = _letter_shift_detail(parse)
        if _ls:
            line += ' <span class="wfw-emuted">(%s)</span>' % escape(_ls)
        return _build_line(line) + _grid(_all_rows(parse, src_fg, src_fill))
    return _grid(_all_rows(parse, src_fg, src_fill))


# CARD_CSS is the card's own styles WITHOUT page-shell rules (body/:root/*) so
# the public site can embed the identical card inside its own layout
# (week-only relaunch 2026-07-13). PAGE_CSS = shell + CARD_CSS, unchanged look
# for the solver's standalone pages.
CARD_CSS = """
  .wfw-card { --ink:#0f172a; --muted:#475569; }
  .wfw-card, .wfw-card * { box-sizing: border-box; }
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
  .wfw-sel { background:#fde68a; border-radius:3px; padding:0 .05em;
             box-shadow:inset 0 -2px 0 #f59e0b; font-weight:800; color:#7a4f00; }
  .wfw-unsel { color:#cbd5e1; }
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
  .wfw-unex { display:inline-block; margin-left:.35rem; padding:.02rem .4rem;
              border-radius:6px; background:#fef2f2; color:#b91c1c;
              border:1px solid #fecaca; font-size:.72rem; font-weight:700;
              font-family:-apple-system,'Segoe UI',sans-serif; }
  .wfw-prov { display:inline-block; margin-left:.4rem; padding:.05rem .45rem;
              border-radius:6px; background:#fef3c7; color:#92600a;
              border:1px solid #fcd34d; font-size:.72rem; font-weight:700;
              text-transform:uppercase; letter-spacing:.04em; }
  .wfw-comment { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
                 padding:.65rem .85rem; margin:0 0 1rem; line-height:1.55;
                 font-size:.98rem; color:#0f172a; }   /* the reviewer's own explanation */
  .wfw-banner { margin-top:1rem; border-radius:10px; padding:.65rem .85rem;
                font-size:.92rem; line-height:1.45; }
  .wfw-banner-prov { background:#fffbeb; border:1px solid #fcd34d; color:#92600a; }
  .wfw-banner-pending { background:#fffbeb; border:1px solid #fcd34d; color:#92600a; }
  .wfw-banner-warn { background:#fef2f2; border:1px solid #fecaca; color:#b91c1c; }
  .wfw-banner-pending ul, .wfw-banner-warn ul { margin:0; padding-left:1.2rem; }
  .wfw-banner-pending li, .wfw-banner-warn li { margin:.15rem 0; }
"""

PAGE_CSS = """
  :root { --ink:#0f172a; --muted:#475569; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
         Helvetica, Arial, sans-serif; margin: 2rem auto; max-width: 680px;
         color: var(--ink); background:#f1f5f9; padding:0 1rem; }
""" + CARD_CSS


def render_page(parses, title="WFW preview"):
    """Wrap one or more rendered parses in a full standalone HTML page."""
    cards = "".join(render_parse(p) for p in parses)
    return ('<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>%s</title><style>%s</style></head><body>'
            '<p class="wfw-tag">word-for-word preview</p>%s</body></html>'
            % (escape(title), PAGE_CSS, cards))
