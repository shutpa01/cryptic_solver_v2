"""Anagram+charade engine — a charade where ONE piece is an anagram (compound).

EVIDENCE-DRIVEN, links classified LAST (memory: feedback-never-preassign-links).
It does NOT strip link words up front. It tiles the answer left-to-right by finding
PIECES from positive evidence:
  - an anagram piece: a contiguous run of words whose letters (contraction-stripped
    fallback) anagram to a span of the answer at the current position;
  - synonym/abbreviation pieces: a DB lookup of a contiguous run of 1+ words
    (multi-word phrases included, e.g. "kind of square" -> T, which may contain a
    function word like "of") whose value sits at the current position.
Words consumed by no piece are RESIDUE, classified ONLY at the end: a confirmed
anagram indicator is the indicator; a function/connective word (is_link, or POS
ADP/PART/AUX/DET/CCONJ/SCONJ/VERB/ADV) is a link; anything else leaves the parse
unaccounted (no match on that branch).

Example: VIOLET = anag(OLIVE) + "kind of square"(->T), def "This woman", indicator
"cocktail", with "wants … in a" as connectives.

Definition decided upstream (def_pos). Per-piece colour. Pure and DB-decoupled.
"""

from core import contractions, grammar
from core.wordplay import GLUE_POS, raw, is_anagram_indicator, adjacent_run
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation")
MAX_VALUE_WORDS = 4        # a synonym/abbreviation piece spans up to this many words
MAX_ANAG_WORDS = 6         # an anagram fodder spans up to this many words


def _assemble(answer, words, postags, lookup, is_link, indicator_types):
    """Tile `answer` with pieces; classify residue last. Returns a placement
    dict {pieces, indicator, links} or None. pieces: [(start, end, mechanism,
    value)]; one piece must be the anagram and at least one a value piece."""
    n, N = len(words), len(answer)

    def is_ind(k):
        return is_anagram_indicator(words[k].text, indicator_types)

    def residue_link(k):
        return (is_link and is_link(words[k].text)) or (postags[k] in GLUE_POS)

    def finalize(pieces, skipped):
        if not any(m == "anagram_fodder" for _, _, m, _ in pieces):
            return None
        if not any(m != "anagram_fodder" for _, _, m, _ in pieces):
            return None                              # need a non-anagram piece too
        indicator = [k for k in skipped if is_ind(k)]
        source = "db"
        if not indicator:
            # MISSING-INDICATOR FALLBACK: the anagram piece is proven by its letters,
            # so the residue word(s) ADJACENT to it must be the indicator even if the
            # DB doesn't know it yet — accept provisionally and queue for enrichment.
            anag = next(((a, b) for a, b, m, _ in pieces if m == "anagram_fodder"),
                        None)
            indicator = adjacent_run(skipped, anag[0], anag[1]) if anag else []
            if not indicator:
                return None
            source = "pending"
        links = []
        for k in skipped:
            if k in indicator:
                continue
            if residue_link(k):
                links.append(k)
            else:
                return None                          # a content word unaccounted
        return {"pieces": pieces, "indicator": sorted(indicator), "links": links,
                "indicator_source": source}

    def anag_letter_forms(a, b):
        as_written = "".join(raw(words[k].text) for k in range(a, b))
        stripped = "".join(raw(contractions.strip_suffixes(words[k].text))
                           for k in range(a, b))
        return [f for f in (as_written, stripped) if f]

    def dfs(i, pos, anag_used, pieces, skipped):
        if pos == N:
            return finalize(pieces, skipped + list(range(i, n)))
        if i >= n:
            return None
        # 1. value piece (synonym/abbreviation), multi-word phrases included
        for k in range(1, min(MAX_VALUE_WORDS, n - i) + 1):
            phrase = " ".join(words[j].text for j in range(i, i + k))
            for val, mech in lookup(phrase, answer):
                v = (val or "").upper()
                if mech in _VALUE_MECH and v and answer.startswith(v, pos):
                    r = dfs(i + k, pos + len(v), anag_used,
                            pieces + [(i, i + k, mech, v)], skipped)
                    if r:
                        return r
        # 2. anagram piece (once)
        if not anag_used:
            for k in range(1, min(MAX_ANAG_WORDS, n - i) + 1):
                for fl in anag_letter_forms(i, i + k):
                    span = answer[pos:pos + len(fl)]
                    if len(span) == len(fl) and sorted(span) == sorted(fl) \
                            and span[::-1] != fl:        # exact reversal isn't anagram
                        r = dfs(i + k, pos + len(fl), True,
                                pieces + [(i, i + k, "anagram_fodder", span)],
                                skipped)
                        if r:
                            return r
        # 3. skip word i (candidate residue — validated LAST in finalize)
        return dfs(i + 1, pos, anag_used, pieces, skipped + [i])

    return dfs(0, 0, False, [], [])


def _build(ctx, split, words, placement):
    """Assemble the Parse from a placement. One Source per piece (anagram span
    included), per-letter links coloured by piece, indicator + link words and the
    by-example marker as annotations."""
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    sources, links, pos = [], [], 0
    for si, (a, b, mech, value) in enumerate(placement["pieces"]):
        toks = words[a:b]
        sources.append(Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks), value=value, mechanism=mech))
        for _ in value:
            pos += 1
            links.append(Link(answer_pos=pos, source_index=si,
                              operation="anagram_charade", clue_atom_id=None,
                              transform="anagram_of" if mech == "anagram_fodder"
                              else None))
    ind_toks = [words[k] for k in placement["indicator"]]
    annotations = [Annotation(
        clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in ind_toks),
        role="indicator", note="anagram indicator",
        source=placement.get("indicator_source", "db"))]
    for k in placement["links"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link",
                                      note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="anagram_charade",
                  solved_by="catalog")
    parse.template_id = None
    parse.matched_signature = None
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
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
    if any(a.role == "indicator" and getattr(a, "source", "db") == "pending"
           for a in parse.annotations):
        warnings.append("the anagram indicator is provisional (queued for enrichment)")
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing or parse.definition is None:
        parse.status = "fail"
    else:
        parse.status = "pending"


def solve_anagram_charade(ctx, defines, lookup, is_link, indicator_types,
                          templates=None, define_fallback=None, is_dbe=None):
    """Full anagram+charade solve — evidence-driven (no preassigned links).
    `templates` accepted for call-site compatibility but unused. Returns the first
    clean PASS, else the best parse found, else None."""
    from core.definition_engine import find_definitions

    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 3:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None

    best = None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 2:
            continue
        postags = grammar.pos_tags([t.text for t in words]) or [None] * len(words)
        placement = _assemble(answer, words, postags, lookup, is_link,
                              indicator_types)
        if placement is None:
            continue
        parse = _build(ctx, split, words, placement)
        if parse.status == "pass":
            return parse
        if best is None:
            best = parse
    return best
