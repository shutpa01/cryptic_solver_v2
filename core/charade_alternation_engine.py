"""Charade with an ALTERNATION-selected piece — a charade read in clue order where ONE
piece is the EVERY-OTHER (alternate) letters of a single word, licensed by an alternation
indicator; the other pieces are ordinary synonym/abbreviation charade pieces:

    "Smooth seed oddly ignored near bird" = SANDPIPER
      def "bird"; SAND[Smooth] + PIP[seed] + ER[even letters of "near", "oddly ignored"]

A NEW bespoke stage — it never edits the working charade or alternation engines. It
LEVERAGES two existing primitives rather than duplicating them:
  - the alternate-letter extraction (core.selection 'alternate' rule, both alignments), and
  - the alternation-indicator vocabulary (the `alternation`/`alternating`/`alternate`
    wordplay types, plus any word the selection layer licenses for the 'alternate' rule).

ANSWER-DRIVEN: the pieces in clue order must concatenate to the answer EXACTLY. GATED: an
alternation indicator must be present AND there must be >= 1 ordinary charade piece, so it
cannot intercept a plain charade (no alternation piece) or a whole-answer alternation (no
ordinary piece). Pure: a wfw_atoms context + injected predicates. A clean PASS is filed as a
catalog signature by the cascade's auto-file path.
"""

from core import literals
from core.selection import select_span
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation")
MAX_PIECE_WORDS = 4
# wordplay types that mean "take alternate letters" (mirrors alternation_engine._ALT_TYPES,
# minus the loose 'parts' — for a SINGLE piece we want the precise alternate licence, which
# selection_rules supplies for parts/even|odd|alternate).
_ALT_IND_TYPES = {"alternation", "alternating", "alternate"}


def _alt_licensed(text, indicator_types, selection_rules):
    """True if `text` is an alternation indicator: a DB alternation wordplay type, or a word
    the selection layer licenses for the 'alternate' rule (parts/even|odd|alternate)."""
    try:
        if set(indicator_types(text) or ()) & _ALT_IND_TYPES:
            return True
    except Exception:
        pass
    try:
        if "alternate" in (selection_rules(text) or ()):
            return True
    except Exception:
        pass
    return False


def _alt_targets(words, n, is_link, indicator_types, selection_rules):
    """[(indicator_run_indices, fodder_word_index), ...] — each alternation indicator run
    (1..4 contiguous words) paired with the next CONTENT word it selects from (links
    skipped). Longer indicator runs first."""
    out, seen = [], set()
    for L in range(min(4, n), 0, -1):
        for i in range(n - L + 1):
            phrase = " ".join(words[j].text for j in range(i, i + L))
            if not _alt_licensed(phrase, indicator_types, selection_rules):
                continue
            fodder = None
            for j in range(i + L, n):
                if is_link and is_link(words[j].text):
                    continue
                fodder = j
                break
            if fodder is None:
                continue
            key = (tuple(range(i, i + L)), fodder)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def _assemble(answer, words, n, ind_set, fodder_idx, alt_val, lookup, is_link):
    """Tile `answer` in clue order: indicator words carry no letters, the fodder word is the
    alternation piece (value `alt_val`), every other word is a DB value piece; leftover words
    are links. Requires the alternation piece AND >= 1 ordinary piece. Returns a placement
    {pieces, links} or None. pieces: [(start, end, mechanism, value)]."""
    N = len(answer)

    def residue_link(k):
        return bool(is_link and is_link(words[k].text))

    def finalize(pieces, skipped, used_alt):
        if not used_alt:
            return None
        if sum(1 for p in pieces if p[2] != "alternate") < 1:
            return None                              # need >= 1 ORDINARY charade piece
        if len(pieces) < 2:
            return None                              # a charade is >= 2 pieces
        links = []
        for k in skipped:
            if k in ind_set:
                continue                             # indicator words: accounted, no letters
            if residue_link(k):
                links.append(k)
            else:
                return None                          # a content word unaccounted
        return {"pieces": pieces, "links": links}

    def dfs(i, pos, pieces, skipped, used_alt):
        if pos == N:
            return finalize(pieces, skipped + list(range(i, n)), used_alt)
        if i >= n:
            return None
        if i in ind_set:                             # indicator word — no letters
            return dfs(i + 1, pos, pieces, skipped, used_alt)
        if i == fodder_idx:                          # the alternation piece (must be used)
            if alt_val and answer.startswith(alt_val, pos):
                r = dfs(i + 1, pos + len(alt_val),
                        pieces + [(i, i + 1, "alternate", alt_val)], skipped, True)
                if r:
                    return r
            return None
        for k in range(1, min(MAX_PIECE_WORDS, n - i) + 1):
            if any((i + off) in ind_set or (i + off) == fodder_idx for off in range(k)):
                break                                # a piece run can't span ind / fodder
            phrase = " ".join(words[j].text for j in range(i, i + k))
            for val, mech in lookup(phrase, answer):
                v = (val or "").upper()
                if mech in _VALUE_MECH and v and answer.startswith(v, pos):
                    r = dfs(i + k, pos + len(v),
                            pieces + [(i, i + k, mech, v)], skipped, used_alt)
                    if r:
                        return r
            lit = literals.literal_value(phrase)
            if lit and answer.startswith(lit, pos):
                r = dfs(i + k, pos + len(lit),
                        pieces + [(i, i + k, "raw", lit)], skipped, used_alt)
                if r:
                    return r
        return dfs(i + 1, pos, pieces, skipped + [i], used_alt)

    return dfs(0, 0, [], [], False)


def solve_charade_alternation(ctx, defines, lookup, is_link, indicator_types,
                              selection_rules, define_fallback=None, is_dbe=None):
    """Solve a charade one of whose pieces is the alternate letters of a word. Returns the
    first clean PASS, else the best parse, else None (abstain)."""
    from core.definition_engine import find_definitions
    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 3 or indicator_types is None:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None

    best = None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        n = len(words)
        if n < 3:                                    # indicator + fodder + >= 1 piece
            continue
        for ind_run, fodder_idx in _alt_targets(words, n, is_link, indicator_types,
                                                 selection_rules):
            for alt_val, _atoms in select_span(ctx, words[fodder_idx], "alternate"):
                av = "".join(c for c in alt_val.upper() if c.isalpha())
                if not av:
                    continue
                placement = _assemble(answer, words, n, set(ind_run), fodder_idx, av,
                                      lookup, is_link)
                if placement is None:
                    continue
                parse = _build(ctx, split, words, placement, ind_run, fodder_idx)
                if parse.status == "pass":
                    return parse
                if best is None:
                    best = parse
    return best


def _build(ctx, split, words, placement, ind_run, fodder_idx):
    """Assemble the Parse: one Source per piece (the alternation piece wears mechanism
    'alternate'), the alternation indicator + link words as annotations."""
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition", source=split.source)
    sources, links, pos = [], [], 0
    for si, (a, b, mech, value) in enumerate(placement["pieces"]):
        toks = words[a:b]
        sources.append(Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks), value=value, mechanism=mech))
        for _ in value:
            pos += 1
            links.append(Link(answer_pos=pos, source_index=si,
                              operation="charade", clue_atom_id=None))

    ind_toks = [words[k] for k in ind_run]
    annotations = [Annotation(
        clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in ind_toks), role="indicator",
        note="alternation indicator (alternate letters of %s)" % words[fodder_idx].text)]
    for k in placement["links"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                                      role="link", note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="charade_alternation", solved_by="catalog")
    parse.template_id = None
    parse.matched_signature = None
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    """Three-state verdict: pass / pending (provisional def) / fail (a clue word
    unaccounted, or a role the DB does not back)."""
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
    from core import role_validity
    bad = role_validity.unbacked_roles(parse)
    if bad:
        parse.warnings = warnings + bad
        parse.status = "fail"
        return
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing:
        parse.status = "fail"
    else:
        parse.status = "pending"
