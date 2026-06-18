"""Charade + hollowing engine — a charade where ONE piece is a HOLLOWED word: its outer
shell only (first + last letter), the inside emptied out.

    "In the morning exercise drained king, Middle Eastern ruler" = AMEER
      def   = "Middle Eastern ruler"
      AM    = "In the morning"
      EE    = "exercise" drained  -> E[xercis]E  (keep first + last)
      R     = "king" (Rex)
      AM + EE + R = AMEER

core.deletion already defines the 'empty' op (keep first+last, BUGATTI -> BI) and the
standalone deletion engine can use it — but the charade+deletion engine cannot: its design
reconstructs the pre-deletion value BACKWARD from the answer segment by restoring a bounded
affix (behead/curtail/outer/heartless), which cannot express an arbitrary-length removed
middle. This stage does it FORWARD instead: a hollow indicator licenses taking the outer
shell of an adjacent word, matched answer-driven via selection.match_span(rule='outer').

A NEW stage (per the project rule: never edit a working engine to add a case). Gated on a
hollow indicator — a DB deletion indicator whose sub-type is 'empty' (e.g. drained, gutted).
ANSWER-DRIVEN and per-letter sourced. Pure and DB-decoupled.
"""

from core import selection
from core.wordplay import raw
from core.wfw_model import Source, Link, Annotation, Parse

MAX_RUN = 4
_MECH_PRI = {"literal": 0, "raw": 0, "abbreviation": 1, "synonym": 2}


def _ordered_values(lookup_all, phrase):
    """(value, mechanism) for a phrase, LITERAL first then abbreviation then synonym."""
    rows = [((v or "").upper(), m) for v, m in lookup_all(phrase) if v]
    lit = raw(phrase)
    if lit:
        rows.append((lit, "literal"))
    rows.sort(key=lambda x: (_MECH_PRI.get(x[1], 3), len(x[0])))
    out, seen = [], set()
    for vu, m in rows:
        if vu and vu not in seen:
            seen.add(vu)
            out.append((vu, m))
    return out


def solve_charade_hollow(ctx, defines, lookup_all, is_link, indicator_types,
                         deletion_subtypes, define_fallback=None, is_dbe=None):
    """Full charade+hollow solve. First clean PASS, else best parse, else None."""
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
        parse = _assemble(ctx, answer, split, words, lookup_all, is_link,
                          indicator_types, deletion_subtypes)
        if parse is not None:
            if parse.status == "pass":
                return parse
            if best is None:
                best = parse
    return best


def _assemble(ctx, answer, split, words, lookup_all, is_link, indicator_types,
              deletion_subtypes):
    n, N = len(words), len(answer)

    def types(k):
        try:
            return indicator_types(words[k].text) or set()
        except Exception:
            return set()

    def is_hollow_ind(k):
        subs = deletion_subtypes(words[k].text) if deletion_subtypes else set()
        return "empty" in (subs or set())

    def is_glue(k):
        # a leftover word is acceptable as charade glue if it is a link or ANY indicator,
        # never a bare content word.
        return bool((is_link and is_link(words[k].text)) or types(k))

    if not any(is_hollow_ind(k) for k in range(n)):
        return None                              # GATE: a hollow indicator is required

    vcache = {}

    def ordered(a, b):
        if (a, b) not in vcache:
            vcache[(a, b)] = _ordered_values(
                lookup_all, " ".join(words[k].text for k in range(a, b)))
        return vcache[(a, b)]

    def dfs(wi, pos, pieces, gaps, used_h):
        if pos == N:
            return _finalize(ctx, answer, split, words, pieces,
                             gaps + list(range(wi, n)), is_hollow_ind, is_glue,
                             is_link, indicator_types)
        if wi >= n:
            return None
        # skip wi as a gap (glue / link / indicator)
        r = dfs(wi + 1, pos, pieces, gaps + [wi], used_h)
        if r is not None:
            return r
        # plain pieces: a run value that is a prefix of the remaining answer
        for b in range(wi + 1, min(wi + MAX_RUN, n) + 1):
            for V, mech in ordered(wi, b):
                if V and answer.startswith(V, pos):
                    r = dfs(b, pos + len(V),
                            pieces + [(wi, b, V, "plain", mech, None)], gaps, used_h)
                    if r is not None:
                        return r
        # hollow piece (only one): the outer shell (first+last) of the SINGLE word wi
        # equals the next two answer letters.
        if not used_h and pos + 2 <= N:
            seg = answer[pos:pos + 2]
            aids = selection.match_span(ctx, words[wi], "outer", seg)
            if aids:
                r = dfs(wi + 1, pos + 2,
                        pieces + [(wi, wi + 1, seg, "hollow", "outer", aids)],
                        gaps, True)
                if r is not None:
                    return r
        return None

    return dfs(0, 0, [], [], False)


def _finalize(ctx, answer, split, words, pieces, gaps, is_hollow_ind, is_glue,
              is_link, indicator_types):
    if not any(p[3] == "hollow" for p in pieces) or len(pieces) < 2:
        return None
    if not any(is_hollow_ind(g) for g in gaps):
        return None                              # the hollow indicator must be a leftover
    # Leftover words are charade glue: a link or an indicator. Check each contiguous
    # leftover RUN at the PHRASE level too (so "put on" is recognised as a whole).
    from core.engine_common import contiguous_groups
    for run in contiguous_groups(sorted(gaps)):
        phrase = " ".join(words[g].text for g in run)
        try:
            phrase_typed = bool(indicator_types(phrase))
        except Exception:
            phrase_typed = False
        if (is_link and is_link(phrase)) or phrase_typed:
            continue
        if all(is_glue(g) for g in run):
            continue
        return None                              # a bare content word is unaccounted

    sources, links = [], []
    pos = 0
    for (a, b, seg, kind, op, aids) in pieces:
        toks = words[a:b]
        mech = "deletion" if kind == "hollow" else op
        src = Source(clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                     text=" ".join(t.text for t in toks), value=seg, mechanism=mech,
                     source="db")
        si = len(sources)
        sources.append(src)
        for j in range(len(seg)):
            # hollow letters carry their exact source atom (per-letter provenance); plain
            # synonym/abbreviation letters come from the DB value, not a clue character.
            caid = aids[j] if kind == "hollow" else None
            links.append(Link(answer_pos=pos + 1, source_index=si,
                              operation=("deletion" if kind == "hollow" else "charade"),
                              clue_atom_id=caid))
            pos += 1

    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    annotations = []
    for g in gaps:
        if is_hollow_ind(g):
            role, note = "indicator", "deletion indicator"
        elif is_link and is_link(words[g].text):
            role, note = "link", "link word"
        else:
            role, note = "indicator", "charade indicator"
        annotations.append(Annotation(clue_atom_ids=words[g].atom_ids,
                                      text=words[g].text, role=role, note=note))

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="charade", solved_by="catalog")
    parse.template_id = None
    parse.matched_signature = None
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    warnings = []
    if not parse.is_complete():
        warnings.append("the answer letters are not fully covered by the pieces")
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
    elif missing or parse.definition is None:
        parse.status = "fail"
    else:
        parse.status = "pending"
