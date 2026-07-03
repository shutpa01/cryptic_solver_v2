"""Charade+deletion engine — a charade where ONE piece is a deletion.

A two-operation compound: pieces concatenate to the answer (a charade), but one piece is
not a plain value — it is a DB value with letters removed by a deletion. e.g.

    "Former lover put on garment with no top, generating urges" = EXHORTS
      def    = "urges"   (exhort = urge)
      EX     = "Former lover"
      HORTS  = "garment" (SHORTS) with "no top" -> first letter removed (behead)
      EX + HORTS = EXHORTS,  "put on" the charade

ANSWER-DRIVEN: it tiles the answer left-to-right with pieces over contiguous clue-word
runs. A plain piece is a run value that is a prefix of the remaining answer; a DELETION
piece reconstructs the pre-deletion value FROM the answer segment (per the deletion
indicator's DB sub-type) and checks it is a value of the run by MEMBERSHIP — literal letters
first, then abbreviations, then synonyms only if needed, never scanned in a loop. Exactly
one deletion piece, GATED on a deletion indicator. Leftover words are the deletion indicator
+ charade glue/links. Pure; definition decided upstream.
"""

import string

from core import deletion
from core.wordplay import raw
from core.wfw_model import Source, Link, Annotation, Parse

MAX_RUN = 4
_ALPHA = string.ascii_uppercase
_MECH_PRI = {"literal": 0, "raw": 0, "abbreviation": 1, "synonym": 2}
_OP_ORDER = ["behead", "curtail", "outer", "heartless"]


def solve_charade_deletion(ctx, defines, lookup_all, is_link, indicator_types,
                           deletion_subtypes, templates=None, define_fallback=None,
                           is_dbe=None, loc_rules=None):
    """Full charade+deletion solve. First clean PASS, else best parse, else None.

    `loc_rules` (selection_rules) supports the location/operation split: a letter-location
    word ("opening"/"leader" = first letter) that sits with a genuine deletion operation
    word ("shed"/"demolished") is accounted as part of the deletion and pins which letters
    drop — it never licenses a deletion alone."""
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


def _reconstruct(seg, op):
    """Pre-deletion values whose `op` yields the answer segment `seg` (exhaustive over the
    restored letter[s])."""
    if op == "behead":
        return [c + seg for c in _ALPHA]
    if op == "curtail":
        return [seg + c for c in _ALPHA]
    if op == "outer":
        return [c1 + seg + c2 for c1 in _ALPHA for c2 in _ALPHA]
    if op == "heartless":
        m = (len(seg) + 1) // 2
        return [seg[:m] + c + seg[m:] for c in _ALPHA]
    return []


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
        """The deletion ops the word's letter-location rules license (first->behead, ...);
        empty for a non-location word or when no rules provider is wired."""
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
    # LOCATION/OPERATION split: a location word pins which letters drop, accounted only
    # because a genuine deletion operation word is present (the gate guarantees one). Fold
    # its op in so "after opening shed" (general 'shed' + location 'opening') can behead.
    for k in range(n):
        if not is_del(k):
            ops |= loc_ops(k)
    ops = [o for o in _OP_ORDER if o in ops]
    if not ops:
        return None

    vcache, scache = {}, {}

    def ordered(a, b):
        if (a, b) not in vcache:
            vcache[(a, b)] = _ordered_values(
                lookup_all, " ".join(words[k].text for k in range(a, b)))
        return vcache[(a, b)]

    def vset(a, b):
        if (a, b) not in scache:
            scache[(a, b)] = {v for v, _ in ordered(a, b)}
        return scache[(a, b)]

    def dfs(wi, pos, pieces, gaps, used_del):
        if pos == N:
            return _finalize(ctx, answer, split, words, pieces,
                             gaps + list(range(wi, n)), is_del, is_glue,
                             is_link, indicator_types, is_loc)
        if wi >= n:
            return None
        # skip wi as a gap (glue / link / indicator)
        r = dfs(wi + 1, pos, pieces, gaps + [wi], used_del)
        if r is not None:
            return r
        # place a piece starting at wi
        for b in range(wi + 1, min(wi + MAX_RUN, n) + 1):
            # plain piece: a run value that is a prefix of the remaining answer
            for V, mech in ordered(wi, b):
                if V and answer.startswith(V, pos):
                    r = dfs(b, pos + len(V),
                            pieces + [(wi, b, V, "plain", mech, None)], gaps, used_del)
                    if r is not None:
                        return r
            # deletion piece (only one): the segment's pre-deletion value is a run value
            if not used_del:
                vs = vset(wi, b)
                for k in range(1, N - pos + 1):
                    seg = answer[pos:pos + k]
                    for op in ops:
                        for vrec in _reconstruct(seg, op):
                            if vrec in vs:
                                r = dfs(b, pos + k,
                                        pieces + [(wi, b, seg, "deletion", op, vrec)],
                                        gaps, True)
                                if r is not None:
                                    return r
        return None

    return dfs(0, 0, [], [], False)


def _finalize(ctx, answer, split, words, pieces, gaps, is_del, is_glue, is_link,
              indicator_types, is_loc=None):
    if not any(p[3] == "deletion" for p in pieces) or len(pieces) < 2:
        return None
    del_gaps = [g for g in gaps if is_del(g)]
    if not del_gaps:
        return None

    # ADJACENCY: a deletion indicator must ATTACH to the piece it deletes from. From the piece's
    # word-run, walk outward through deletion-expression words only — links (of, with), other
    # deletion-indicator words, and deletion-LOCATION words (start, initially, opening) — and it
    # is bound iff that reaches a deletion indicator or location word. A content word or a
    # foreign-structure indicator that is NOT a link (SUBSCRIBER: "collecting" the container)
    # blocks the path, so a far-off "snubbed" can no longer license a deletion on "taxi". Genuine
    # charade+deletions keep their indicator adjacent (INERT "no start of better", UTTERABLE), so
    # they are unaffected.
    n_words = len(words)
    covered = {w for p in pieces for w in range(p[0], p[1])}

    def _expr(w):
        return (w not in covered) and (
            is_del(w) or bool(is_loc and is_loc(w)) or bool(is_link and is_link(words[w].text)))

    def _del_bound(da, db):
        j = db
        while j < n_words and _expr(j):
            if is_del(j) or (is_loc and is_loc(j)):
                return True
            j += 1
        j = da - 1
        while j >= 0 and _expr(j):
            if is_del(j) or (is_loc and is_loc(j)):
                return True
            j -= 1
        return False

    for p in pieces:
        if p[3] == "deletion" and not _del_bound(p[0], p[1]):
            return None

    # Leftover words are charade glue: a link or an indicator. Check each contiguous
    # leftover RUN at the PHRASE level too, so a multi-word connective ("put on") is
    # recognised even when a single word of it is not typed on its own.
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
        return None                              # a bare content word is unaccounted                          # a bare content word is unaccounted

    sources, links = [], []
    pos = 0
    for (a, b, seg, kind, op, vrec) in pieces:
        toks = words[a:b]
        if kind == "deletion":
            mech = "deletion"
        else:
            mech = op                            # plain piece carries its mechanism
        src = Source(clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                     text=" ".join(t.text for t in toks), value=seg, mechanism=mech,
                     source="db")
        si = len(sources)
        sources.append(src)
        for _ in range(len(seg)):
            links.append(Link(answer_pos=pos + 1, source_index=si,
                              operation="charade_deletion", clue_atom_id=None))
            pos += 1

    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    annotations = []
    for g in gaps:
        if is_del(g):
            note, role = "deletion indicator", "indicator"
        elif is_loc and is_loc(g):
            # a location word is part of the deletion expression (it names which letters
            # the operation word removes), accounted because a deletion indicator is present
            note, role = "deletion location indicator", "indicator"
        elif is_link and is_link(words[g].text):
            note, role = "link word", "link"
        else:
            note, role = "charade indicator", "indicator"
        annotations.append(Annotation(clue_atom_ids=words[g].atom_ids,
                                      text=words[g].text, role=role, note=note))

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="charade_deletion",
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
