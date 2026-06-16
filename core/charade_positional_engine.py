"""Charade-positional engine — a charade whose pieces are RE-ORDERED by a positional
indicator (design: the indicator is the operator; it pivots GROUPS, not single pieces).

The plain charade engine (core.charade_signature_engine) assembles pieces strictly in
clue (left-to-right) order. Many charades are not in clue order: a positional indicator
tells the solver to place one group of pieces before or after another. E.g.

    "School following second-class old" = BOSCH
      LEFT group  = {School -> SCH}        (the words before the indicator)
      indicator   = "following"            (subtype 'after')
      RIGHT group = {second-class -> B, old -> O}   (the words after the indicator)
      'after'  =>  answer = RIGHT then LEFT  =  B O SCH  =  BOSCH

This engine is ISOLATED per the 13-engines rule: its own assembly + verifier, no shared
substrate, the plain charade engine untouched. It is GATED — it only fires when a
charade-positional indicator (typed in the indicators table) actually appears in the
wordplay — and ANSWER-DRIVEN — the re-ordered pieces must concatenate to the EXACT answer,
so it cannot fabricate. The definition is decided upstream; gap words are classified as
links LAST (never pre-stripped).

Indicator subtypes (in the indicators table, wordplay_type 'charade_positional'):
  after        — LEFT group placed AFTER the RIGHT group   (answer = RIGHT+LEFT); any clue
  before       — LEFT group placed BEFORE the RIGHT group  (answer = LEFT+RIGHT); any clue
  after_down   — as 'after',  but only in a DOWN clue (vertical 'support' metaphor:
                 supporting / bearing / holding / beneath — the supporter sits lower)
  before_down  — as 'before', but only in a DOWN clue (vertical 'top' metaphor:
                 on / above / over — the topper sits higher)
Orientation gating uses ctx.direction; when it is unknown (None) the *_down subtypes do
not fire, so an across-only run never mis-reads a down-metaphor word.
"""

from core import literals
from core.wfw_model import Source, Link, Annotation, Parse

# A piece's role -> the displayed mechanism (and the ONLY lookup allowed to fill it).
ROLE_MECHANISM = {"SYN_F": "synonym", "ABR_F": "abbreviation", "LIT_F": "raw"}

# subtype -> ('swap'|'natural', needs_down). 'swap' => RIGHT+LEFT; 'natural' => LEFT+RIGHT.
_SUBTYPE_ORDER = {
    "after": ("swap", False),
    "before": ("natural", False),
    "after_down": ("swap", True),
    "before_down": ("natural", True),
}

_MAX_PIECE_WORDS = 3        # a charade piece spans at most this many clue words
_MAX_IND_WORDS = 2          # a positional indicator spans at most this many words


def _candidates(phrase, answer, lookup):
    """Role-pure fill values for `phrase`, drawn ONLY from the DB lookup (synonym /
    abbreviation, both already substring-filtered to the answer) plus a curated literal
    (a short function word read as its own letters, e.g. 'in' -> IN). No free raw-tiling:
    a literal is allowed only for the curated LITERAL_WORDS, exactly as lookup_all does."""
    out, seen = [], set()
    for value, mech in lookup(phrase, answer):
        role = {"synonym": "SYN_F", "abbreviation": "ABR_F"}.get(mech)
        v = (value or "").upper()
        if role and v and (v, role) not in seen:
            seen.add((v, role))
            out.append((v, role))
    lit = literals.literal_value(phrase)
    if lit:
        lit = lit.upper()
        if (lit, "LIT_F") not in seen:
            out.append((lit, "LIT_F"))
    return out


def _tile(words, lo, hi, target, lookup, answer, is_link):
    """Tile the word run words[lo:hi] so its pieces concatenate to EXACTLY `target`,
    left-to-right, with gap words classified as links (a non-link gap word -> reject).
    Returns (pieces, link_idxs) with ABSOLUTE word indices, or None. `pieces` is
    [(a, b, role, value)]. Requires at least one piece (a group is real wordplay, not
    all links). Pieces are tried before gaps, so the result accounts the most words."""
    n = hi - lo

    def dfs(rel, pos, pieces, gaps):
        if rel == n:
            if pos != len(target):
                return None
            for k in gaps:
                if not is_link(words[k].text):
                    return None             # a content word left unaccounted -> reject
            return (pieces, gaps) if pieces else None
        wi = lo + rel
        # try this word as the start of a piece (1.._MAX_PIECE_WORDS words) FIRST
        for nw in range(1, min(_MAX_PIECE_WORDS, n - rel) + 1):
            phrase = " ".join(words[lo + rel + k].text for k in range(nw))
            for val, role in _candidates(phrase, answer, lookup):
                if target.startswith(val, pos):
                    r = dfs(rel + nw, pos + len(val),
                            pieces + [(wi, wi + nw, role, val)], gaps)
                    if r:
                        return r
        # else treat this word as a gap (must be a link, checked at the end)
        return dfs(rel + 1, pos, pieces, gaps + [wi])

    return dfs(0, 0, [], [])


def _find_indicators(words, positional_subtypes, direction):
    """Yield (i, j, subtype, order) for every positional indicator span words[i:j]
    (1.._MAX_IND_WORDS words) whose subtype applies in this clue's orientation."""
    down = (direction or "").lower() == "down"
    n = len(words)
    for i in range(n):
        for j in range(i + 1, min(i + _MAX_IND_WORDS, n) + 1):
            phrase = " ".join(words[k].text for k in range(i, j))
            for sub in positional_subtypes(phrase):
                order_needs = _SUBTYPE_ORDER.get(sub)
                if order_needs is None:
                    continue
                order, needs_down = order_needs
                if needs_down and not down:
                    continue                # a down-metaphor word, not a down clue
                yield i, j, sub, order


def _verify(ctx, parse):
    """Three-state verdict: pass / pending (provisional definition) / fail."""
    warnings = []
    if not parse.is_complete():
        warnings.append("the answer letters are not fully tiled by the pieces")
    missing = parse.unexplained_words(ctx)
    if missing:
        warnings.append("these clue words are unaccounted for: "
                        + ", ".join(repr(m) for m in missing))
    if parse.definition is None:
        warnings.append("no definition found")
    elif getattr(parse.definition, "source", "db") == "pending":
        warnings.append("the definition is provisional (queued for enrichment)")
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing:
        parse.status = "fail"
    else:
        parse.status = "pending"


def _build(ctx, split, words, ordered_pieces, ind_span, subtype, link_idxs):
    """Assemble the Parse. `ordered_pieces` is [(a, b, role, value)] in ANSWER order
    (already pivoted); `ind_span` is (i, j) the indicator words; `link_idxs` the gap
    word indices. Sources/links are emitted in answer order so each tile lights the
    answer letters its piece produced."""
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    sources, links, pos = [], [], 0
    for si, (a, b, role, value) in enumerate(ordered_pieces):
        toks = words[a:b]
        sources.append(Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks), value=value,
            mechanism=ROLE_MECHANISM[role], source="db"))
        for _ in value:
            pos += 1
            links.append(Link(answer_pos=pos, source_index=si,
                              operation="charade", clue_atom_id=None))
    i, j = ind_span
    annotations = [Annotation(
        clue_atom_ids=tuple(aid for k in range(i, j) for aid in words[k].atom_ids),
        text=" ".join(words[k].text for k in range(i, j)),
        role="indicator", note="positional indicator (%s)" % subtype)]
    for k in link_idxs:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link",
                                      note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="charade", solved_by="catalog")
    parse.positional_subtype = subtype
    _verify(ctx, parse)
    return parse


def solve_charade_positional(ctx, defines, lookup, is_link, positional_subtypes,
                             define_fallback=None, is_dbe=None):
    """Solve a re-ordered charade. GATED on a positional indicator; ANSWER-DRIVEN.

    For each candidate definition split, find a positional indicator in the wordplay,
    split the wordplay into a LEFT group (before it) and a RIGHT group (after it), and
    try to tile the two groups so that — in the order the indicator dictates — they
    concatenate to the exact answer. Among complete solves prefer the FEWEST link
    (residue) words. Returns a PASS parse, else fail-evidence (an indicator fired but
    the pieces could not be assembled), else None (no positional indicator -> abstain)."""
    from core.definition_engine import find_definitions

    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 2:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    direction = getattr(ctx, "direction", None)

    best_pass = None
    best_pass_links = None
    best_fail = None
    saw_indicator = False

    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 3:                      # >=1 piece each side + >=1 indicator word
            continue
        for i, j, subtype, order in _find_indicators(words, positional_subtypes,
                                                      direction):
            left = (0, i)                       # words[:i]
            right = (j, len(words))             # words[j:]
            if left[1] - left[0] < 1 or right[1] - right[0] < 1:
                continue                        # both groups must be non-empty
            saw_indicator = True
            # Pivot: 'swap' (after) => answer = RIGHT+LEFT; 'natural' (before) => LEFT+RIGHT.
            first, second = (right, left) if order == "swap" else (left, right)
            for p in range(1, len(answer)):     # answer split: first->[:p], second->[p:]
                t_first = _tile(words, first[0], first[1], answer[:p],
                                lookup, answer, is_link)
                if t_first is None:
                    continue
                t_second = _tile(words, second[0], second[1], answer[p:],
                                 lookup, answer, is_link)
                if t_second is None:
                    continue
                pieces_first, gaps_first = t_first
                pieces_second, gaps_second = t_second
                ordered_pieces = pieces_first + pieces_second
                link_idxs = sorted(gaps_first + gaps_second)
                parse = _build(ctx, split, words, ordered_pieces, (i, j), subtype,
                               link_idxs)
                if parse.status == "pass":
                    nlinks = len(link_idxs)
                    if best_pass is None or nlinks < best_pass_links:
                        best_pass, best_pass_links = parse, nlinks
                elif best_fail is None and parse.status in ("pending", "fail"):
                    best_fail = parse

    if best_pass is not None:
        return best_pass
    if best_fail is not None:
        return best_fail
    if saw_indicator:
        # A positional indicator fired but no assembly reached the answer — preserve the
        # evidence (definition + the indicator) as a fail so the indicator survives and
        # the clue is shown as the positional charade it is, not dropped.
        split = splits[0]
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        for i, j, subtype, order in _find_indicators(words, positional_subtypes,
                                                      direction):
            # No pieces, no fabricated links: the wordplay content words stay
            # UNACCOUNTED, so the verdict is an honest fail (never a role-by-elimination
            # pass). Only the definition + the indicator are shown.
            parse = _build(ctx, split, words, [], (i, j), subtype, [])
            return parse
    return None
