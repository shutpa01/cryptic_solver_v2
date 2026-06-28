"""Charade-positional LOCAL engine — a charade where a positional indicator swaps ONLY the
two pieces immediately FLANKING it; every other piece tiles in normal clue order.

Sibling of core.charade_positional_engine, which does the GLOBAL group pivot (everything
before the indicator swaps with everything after — "School following second-class old" =
B+O+SCH = BOSCH). That global pivot cannot reach a clue where the indicator binds only its
adjacent pair:

    "Cover county show after parking" = BEDSPREAD
      def "Cover"; county -> BEDS ; (show AFTER parking -> parking+show = P+READ)
      => BEDS + P + READ = BEDSPREAD
      county is a normal LEADING piece; "after" swaps only show <-> parking.

Two distinct SHAPES (local does not solve BOSCH, global does not solve BEDSPREAD), so this
is a bespoke sibling per the engine-grain rule — NOT a branch inside the global engine. It
reuses the global engine's primitives (_tile, _candidates, _find_indicators, _verify) so it
stays thin; the bespoke part is the local-pivot assembly + the indicator's render note.

Gated on a SWAP positional indicator only (subtypes 'after' / 'after_down'; the latter only
in a down clue). 'before' / 'before_down' are clue-order locally, i.e. a plain charade, so
this engine abstains on them. ANSWER-DRIVEN: prefix + after-piece + before-piece + suffix
must concatenate to the EXACT answer, so it cannot fabricate.
"""

from core.wfw_model import Source, Link, Annotation, Parse
from core.charade_positional_engine import (
    _tile, _candidates, _find_indicators, _verify, _MAX_PIECE_WORDS, ROLE_MECHANISM)


def _tile_run(words, lo, hi, target, lookup, answer, is_link):
    """Tile words[lo:hi] to EXACTLY `target`, returning (pieces, gaps) or None. An EMPTY
    run (lo==hi) is valid ONLY for the empty target (no prefix/suffix), returning ([], [])
    — the underlying _tile rejects an empty run, so handle that boundary here."""
    if lo >= hi:
        return ([], []) if target == "" else None
    return _tile(words, lo, hi, target, lookup, answer, is_link)


def _build_local(ctx, split, words, ordered_pieces, ind_span, subtype, link_idxs):
    """Assemble the Parse. `ordered_pieces` is [(a, b, role, value)] in ANSWER order
    (prefix pieces, then the after-piece, then the before-piece, then suffix pieces).
    Mirrors charade_positional_engine._build but records a LOCAL-swap indicator note so the
    explanation is faithful to the adjacent-pair pivot."""
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
        role="indicator",
        note="positional indicator (%s — swaps the adjacent pair)" % subtype)]
    for k in link_idxs:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link", note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="charade", solved_by="catalog")
    parse.positional_subtype = subtype
    _verify(ctx, parse)
    return parse


def solve_charade_positional_local(ctx, defines, lookup, is_link, positional_subtypes,
                                   define_fallback=None, is_dbe=None):
    """Solve a charade whose positional indicator swaps only its adjacent pair. GATED on a
    SWAP positional indicator; ANSWER-DRIVEN. Returns a PASS/PENDING parse, else None
    (abstains — the global positional engine / plain charade keep their own fail-evidence)."""
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
    best_pending = None

    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        n = len(words)
        if n < 3:                       # before-piece + indicator + after-piece, minimum
            continue
        for i, j, subtype, order in _find_indicators(words, positional_subtypes, direction):
            if order != "swap":         # local pivot is only meaningful for a swap subtype
                continue
            if i < 1 or j > n - 1:      # need >=1 word before AND >=1 word after the indicator
                continue
            # BEFORE-piece occupies words[bp:i]; AFTER-piece occupies words[j:ae].
            for bp in range(max(0, i - _MAX_PIECE_WORDS), i):
                before_cands = _candidates(
                    " ".join(words[k].text for k in range(bp, i)), answer, lookup)
                if not before_cands:
                    continue
                for ae in range(j + 1, min(n, j + _MAX_PIECE_WORDS) + 1):
                    after_cands = _candidates(
                        " ".join(words[k].text for k in range(j, ae)), answer, lookup)
                    if not after_cands:
                        continue
                    for va, arole in after_cands:
                        for vb, brole in before_cands:
                            mid = va + vb
                            lm = len(mid)
                            # answer = prefix + va + vb + suffix; find where the pivot sits
                            for x in range(0, len(answer) - lm + 1):
                                if answer[x:x + lm] != mid:
                                    continue
                                t_pre = _tile_run(words, 0, bp, answer[:x],
                                                  lookup, answer, is_link)
                                if t_pre is None:
                                    continue
                                t_suf = _tile_run(words, ae, n, answer[x + lm:],
                                                  lookup, answer, is_link)
                                if t_suf is None:
                                    continue
                                pre_pieces, pre_gaps = t_pre
                                suf_pieces, suf_gaps = t_suf
                                ordered = (pre_pieces
                                           + [(j, ae, arole, va)]
                                           + [(bp, i, brole, vb)]
                                           + suf_pieces)
                                link_idxs = sorted(pre_gaps + suf_gaps)
                                parse = _build_local(ctx, split, words, ordered,
                                                     (i, j), subtype, link_idxs)
                                if parse.status == "pass":
                                    nlinks = len(link_idxs)
                                    if best_pass is None or nlinks < best_pass_links:
                                        best_pass, best_pass_links = parse, nlinks
                                elif best_pending is None and parse.status == "pending":
                                    best_pending = parse

    return best_pass or best_pending
