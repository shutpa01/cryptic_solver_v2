"""Charade + multiple deletions — a charade where TWO OR MORE pieces are each a deletion.

The single-deletion case is core.charade_deletion_engine (exactly ONE deletion piece, the
rest plain values). This is its bespoke SIBLING for the distinct shape where >= 2 pieces are
each a deletion, typically governed by ONE shared deletion indicator, e.g.

    "Magical character stripped off robes? Wrong!" = OBERON
      def    = "Magical character"   (Oberon, king of the fairies)
      OBE    = "robes" with both outer letters removed (ROBES - R,S)   [ends -> outer]
      RON    = "wrong" with both outer letters removed (WRONG - W,G)   [ends -> outer]
      "stripped off" = the shared deletion indicator

ANSWER-DRIVEN, exactly like charade_deletion: it tiles the answer left-to-right; a plain
piece is a run value that is a prefix of the remaining answer; a DELETION piece reconstructs
its pre-deletion value FROM the answer segment (per a DB-licensed deletion sub-type) and
checks it is a value of the run by MEMBERSHIP (literal first, then abbreviation, then
synonym). GATED on a deletion indicator and REQUIRES >= 2 deletion pieces, so it never
intercepts the plain charade (no deletions) or the single-deletion charade (exactly one).
Leftover words are the deletion indicator(s) + charade glue/links. Pure; definition decided
upstream; its own verifier (role_validity-gated). A fresh stage; it edits no other engine.
"""

from core import deletion
from core.wordplay import raw
from core.wfw_model import Source, Link, Annotation, Parse

MAX_RUN = 4
_OP_ORDER = ["behead", "curtail", "outer", "heartless"]


def solve_charade_multi_deletion(ctx, defines, lookup_all, is_link, indicator_types,
                                 deletion_subtypes, templates=None, define_fallback=None,
                                 is_dbe=None, loc_rules=None):
    """Full charade + multi-deletion solve. First clean PASS, else best parse, else None.

    `loc_rules` (selection_rules) supports the location/operation split, identically to
    charade_deletion: a letter-location word that sits with a genuine deletion operation
    word pins which letters drop; it never licenses a deletion alone."""
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
                          indicator_types, deletion_subtypes, loc_rules)
        if parse is not None:
            if parse.status == "pass":
                return parse
            if best is None:
                best = parse
    return best


def _apply_del(lit, op):
    """Apply deletion `op` to the literal value `lit`, returning the kept segment (or None
    if the op can't apply). LITERAL-ONLY, so the segment is fully determined by the word's
    own letters — no exhaustive letter restoration is needed (that was charade_deletion's
    approach for indirect values; here the source IS the clue word)."""
    if op == "behead":
        return lit[1:] if len(lit) >= 2 else None
    if op == "curtail":
        return lit[:-1] if len(lit) >= 2 else None
    if op == "outer":
        return lit[1:-1] if len(lit) >= 3 else None
    if op == "heartless":                         # remove the single central letter
        if len(lit) >= 3 and len(lit) % 2 == 1:
            m = len(lit) // 2
            return lit[:m] + lit[m + 1:]
        return None
    return None


def _assemble(ctx, answer, split, words, lookup_all, is_link, indicator_types,
              deletion_subtypes, loc_rules=None):
    n, N = len(words), len(answer)

    def types(k):
        try:
            return indicator_types(words[k].text) or set()
        except Exception:
            return set()

    def is_del(k):
        return "deletion" in types(k)

    def loc_ops(k):
        if not loc_rules:
            return set()
        try:
            return deletion.loc_drop_ops(loc_rules(words[k].text))
        except Exception:
            return set()

    def is_loc(k):
        return bool(loc_ops(k))

    def is_glue(k):
        # a leftover word is acceptable as charade glue if it is a link or ANY indicator
        # (the charade-assembly / deletion words), never a bare content word.
        return bool((is_link and is_link(words[k].text)) or types(k))

    if not any(is_del(k) for k in range(n)):
        return None                              # GATE: a deletion indicator is required

    # ops the clue's deletion indicators license (from DB sub-types)
    ops = set()
    for k in range(n):
        if not is_del(k):
            continue
        subs = deletion_subtypes(words[k].text) if deletion_subtypes else set()
        named = False
        for s in subs:
            op = deletion.SUBTYPE_OP.get(s)
            if op:
                ops.add(op); named = True
        if not named:
            ops |= {"behead", "curtail"}
    for k in range(n):
        if not is_del(k):
            ops |= loc_ops(k)
    ops = [o for o in _OP_ORDER if o in ops]
    if not ops:
        return None

    def dfs(wi, pos, pieces, gaps):
        if pos == N:
            return _finalize(ctx, answer, split, words, pieces,
                             gaps + list(range(wi, n)), is_del, is_glue,
                             is_link, indicator_types, is_loc)
        if wi >= n:
            return None
        # skip wi as a gap (glue / link / indicator)
        r = dfs(wi + 1, pos, pieces, gaps + [wi])
        if r is not None:
            return r
        # place a DELETION piece starting at wi. EVERY piece in this engine is a deletion
        # (the distinguishing shape: OBE + RON, ALL + URE — no plain pieces). LITERAL-ONLY:
        # the kept segment is the clue word(s)' OWN letters with the op's letters removed
        # (you strip letters off the word SHOWN, never an indirect synonym — the fair rule).
        # No DB value lookup at all here, so it is both faithful AND fast (the broad
        # synonym lookup is what made the plain-piece variant fabricate and crawl).
        for b in range(wi + 1, min(wi + MAX_RUN, n) + 1):
            lit = raw(" ".join(words[k].text for k in range(wi, b)))
            if not lit:
                continue
            for op in ops:
                seg = _apply_del(lit, op)
                if seg and answer.startswith(seg, pos):
                    r = dfs(b, pos + len(seg),
                            pieces + [(wi, b, seg, "deletion", op, lit)], gaps)
                    if r is not None:
                        return r
        return None

    return dfs(0, 0, [], [])


def _finalize(ctx, answer, split, words, pieces, gaps, is_del, is_glue, is_link,
              indicator_types, is_loc=None):
    # REQUIRE >= 2 deletion pieces (the distinguishing shape; one deletion is
    # charade_deletion's job, zero is the plain charade's).
    if sum(1 for p in pieces if p[3] == "deletion") < 2:
        return None
    if len(pieces) < 2:
        return None
    del_gaps = [g for g in gaps if is_del(g)]
    if not del_gaps:
        return None
    # Leftover words are charade glue: a link or an indicator. Check each contiguous
    # leftover RUN at the PHRASE level too, so a multi-word connective is recognised even
    # when a single word of it is not typed on its own.
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
    for (a, b, seg, kind, op, vrec) in pieces:
        toks = words[a:b]
        mech = "deletion" if kind == "deletion" else op
        src = Source(clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                     text=" ".join(t.text for t in toks), value=seg, mechanism=mech,
                     source="db")
        si = len(sources)
        sources.append(src)
        for _ in range(len(seg)):
            links.append(Link(answer_pos=pos + 1, source_index=si,
                              operation="charade_multi_deletion", clue_atom_id=None))
            pos += 1

    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    annotations = []
    for g in gaps:
        if is_del(g):
            note, role = "deletion indicator", "indicator"
        elif is_loc and is_loc(g):
            note, role = "deletion location indicator", "indicator"
        elif is_link and is_link(words[g].text):
            note, role = "link word", "link"
        else:
            note, role = "charade indicator", "indicator"
        annotations.append(Annotation(clue_atom_ids=words[g].atom_ids,
                                      text=words[g].text, role=role, note=note))

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="charade_multi_deletion",
                  solved_by="catalog")
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
    from core import role_validity
    bad = role_validity.unbacked_roles(parse)
    if bad:
        parse.warnings = warnings + bad
        parse.status = "fail"
        return
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing or parse.definition is None:
        parse.status = "fail"
    else:
        parse.status = "pending"
