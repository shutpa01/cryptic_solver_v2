"""Charade engine — EVIDENCE-DRIVEN, links classified LAST.

A charade reads in clue order: each piece contributes letters (a synonym or an
abbreviation from the DB) and the pieces join end-to-end to spell the answer. There
is no operation indicator and no reordering.

It does NOT pre-assign link words (memory: feedback-never-preassign-links). It tiles
the answer left-to-right by positive evidence: each piece is a DB lookup of a
contiguous run of 1..N words (multi-word phrases included) whose synonym/abbreviation
value sits at the current answer position. NEVER the word's own raw letters — that
wildcard is the free-tiling this engine exists to replace. Words consumed by no
piece are RESIDUE, classified ONLY at the end: a function/connective word (is_link,
or POS ADP/PART/AUX/DET/CCONJ/SCONJ/VERB/ADV) is a link; anything else is left
unaccounted and that branch fails. A charade is >= 2 pieces.

(This replaces the earlier catalog-signature version, which stripped link words up
front to map content onto slots — the pre-assignment the user repeatedly rejected.
The DB lookup still constrains pieces to real synonyms/abbreviations, so this is not
the old raw-letter free-tiler.)

Definition decided upstream (def_pos / find_definitions). Per-piece colour. Pure and
DB-decoupled; evidence-preserving on a fail (no roles assigned).
"""

from core import grammar
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation")
FUNCTION_POS = {"ADP", "PART", "AUX", "DET", "CCONJ", "SCONJ"}
GLUE_POS = FUNCTION_POS | {"VERB", "ADV"}
MAX_PIECE_WORDS = 4


def _assemble(answer, words, postags, lookup, is_link):
    """Tile `answer` with DB-value pieces; classify residue last. Returns a
    placement {pieces, links} or None. pieces: [(start, end, mechanism, value)]."""
    n, N = len(words), len(answer)

    def residue_link(k):
        return (is_link and is_link(words[k].text)) or (postags[k] in GLUE_POS)

    def finalize(pieces, skipped):
        if len(pieces) < 2:
            return None                              # a charade is >= 2 pieces
        links = []
        for k in skipped:
            if residue_link(k):
                links.append(k)
            else:
                return None                          # a content word unaccounted
        return {"pieces": pieces, "links": links}

    def dfs(i, pos, pieces, skipped):
        if pos == N:
            return finalize(pieces, skipped + list(range(i, n)))
        if i >= n:
            return None
        # value piece (synonym/abbreviation), multi-word phrases included — tried
        # BEFORE skipping, so a word that is both a link and a piece ("with" -> W)
        # is used as a piece when the letters need it.
        for k in range(1, min(MAX_PIECE_WORDS, n - i) + 1):
            phrase = " ".join(words[j].text for j in range(i, i + k))
            for val, mech in lookup(phrase, answer):
                v = (val or "").upper()
                if mech in _VALUE_MECH and v and answer.startswith(v, pos):
                    r = dfs(i + k, pos + len(v), pieces + [(i, i + k, mech, v)],
                            skipped)
                    if r:
                        return r
        # skip word i (residue candidate — validated LAST in finalize)
        return dfs(i + 1, pos, pieces, skipped + [i])

    return dfs(0, 0, [], [])


def _build(ctx, split, words, placement):
    """Assemble the Parse from a placement: one Source per piece, per-letter links
    coloured by piece, link words and the by-example marker as annotations."""
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
                              operation="charade", clue_atom_id=None))
    annotations = [Annotation(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                              role="link", note="link word")
                   for k in placement["links"]]
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="charade", solved_by="catalog")
    parse.template_id = None
    parse.matched_signature = None
    _verify_charade(ctx, parse)
    return parse


def _verify_charade(ctx, parse):
    """Three-state verdict: pass / pending (provisional def or piece) / fail (a clue
    word unaccounted)."""
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
    if any(getattr(s, "source", "db") == "pending" for s in parse.sources):
        warnings.append("a wordplay piece is provisional (queued for enrichment)")
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing:
        parse.status = "fail"
    else:
        parse.status = "pending"


def solve_charade(ctx, defines, lookup, is_link, templates=None,
                  define_fallback=None, is_dbe=None):
    """Full charade solve — evidence-driven (no preassigned links). `templates`
    accepted for call-site compatibility but unused (this engine tiles directly).
    Returns the first clean PASS; else the best parse; else a fail-evidence parse
    when a definition was found but nothing tiled; else None."""
    from core.definition_engine import find_definitions

    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 2:
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
        placement = _assemble(answer, words, postags, lookup, is_link)
        if placement is None:
            continue
        parse = _build(ctx, split, words, placement)
        if parse.status == "pass":
            return parse
        if best is None:
            best = parse
    if best is not None:
        return best

    return _build_fail_evidence(ctx, answer, splits[0], lookup)


def _build_fail_evidence(ctx, answer, split, lookup):
    """Preserve the evidence when nothing tiled: keep the definition and show, for
    each wordplay word, a candidate synonym/abbreviation value it COULD contribute.
    NO roles assigned (memory: feedback-no-role-assignment-on-fail) — function words
    are left unaccounted, never relabelled as links by elimination."""
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    sources, unresolved = [], []
    for token in split.wordplay_tokens:
        cands = []
        for value, mech in lookup(token.text, answer):
            v = (value or "").upper()
            if v and v in answer and v not in {c for c, _ in cands}:
                cands.append((v, mech))
        if cands:
            value, mech = cands[0]
            sources.append(Source(clue_atom_ids=token.atom_ids, text=token.text,
                                  value=value, mechanism=mech))
        else:
            unresolved.append(token.text)
    warnings = ["no charade tiling matched this clue "
                "(pieces below are candidates, not a placement)"]
    if unresolved:
        warnings.append("these clue words are unaccounted for: "
                        + ", ".join(repr(u) for u in unresolved))
    from core.definition_engine import dbe_annotation
    dbe = dbe_annotation(split)
    annotations = [dbe] if dbe is not None else []
    return Parse(
        clue_text=ctx.clue_text, answer_text=ctx.answer_text,
        sources=sources, links=[], annotations=annotations,
        definition=definition, operation="charade", solved_by="catalog",
        status="fail", warnings=warnings)
