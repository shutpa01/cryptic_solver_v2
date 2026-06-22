"""Anagram + deletion engine — an anagram whose fodder has letters REMOVED before the
anagram, the mirror of the substitution engines (fodder letters minus removed letters =
answer, vs fodder letters plus substituted letters = answer). Two deletion forms:

  NAMED-LETTER  a word names the removed letter(s) by its abbreviation, gated by a
                deletion indicator:
    "Plant seeds while missing hard ground" = EDELWEISS
      def "Plant"; "ground" = anagram indicator; "missing" = deletion indicator;
      "hard" = H (removed); anagram(SEEDS+WHILE - H) = EDELWEISS

  CURTAILMENT   a fodder word loses its LAST letter, gated by a deletion indicator:
    "Bra seems almost undone being dirty" = BESMEAR
      def "dirty"; "undone" = anagram indicator; "almost" = deletion indicator curtails
      "seems" -> SEEM; "being" link; anagram(BRA + SEEM) = BESMEAR

A NEW stage (never edit a working engine). EVIDENCE/ANSWER-DRIVEN: the fodder pool, after
the deletion (and any optional substitution), must equal the answer EXACTLY as a multiset.
GATED on BOTH an anagram indicator and a deletion indicator, and the deleted/curtailed word
must be ADJACENT to the deletion indicator, so it cannot fabricate. Fodder words may
optionally substitute (a short abbreviation value) so a combined sub+deletion clue works;
the standalone substitution engines are untouched. Pure and DB-decoupled.
"""

from collections import Counter

from core.wordplay import raw, is_anagram_indicator
from core.wfw_model import Source, Link, Annotation, Parse

_MAX_DEL_LEN = 3          # a named-deletion value: a letter or short abbreviation
_MAX_SUB_LEN = 3          # an optional substituted value
_MAX_SUBS = 1             # at most one substituted fodder word (combos are rare)
_MAX_WORDS = 9            # bound the search per clue
_SUB_MECH = ("abbreviation", "substitution")


def _answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _del_values(word, value_lookup):
    """Short, GENUINE letter values for a named-deletion word — abbreviation/substitution
    only (never a synonym: an indirect deletion would let any word delete any letters)."""
    own = Counter(raw(word))
    out, seen = [], set()
    for val, mech in value_lookup(word):
        if mech not in _SUB_MECH:
            continue
        vv = "".join(ch for ch in (val or "").upper() if ch.isalpha())
        if vv and len(vv) <= _MAX_DEL_LEN and Counter(vv) != own and vv not in seen:
            seen.add(vv)
            out.append(vv)
    return out


def solve_anagram_deletion(ctx, defines, value_lookup, indicator_types, is_link,
                           deletion_subtypes=None, define_fallback=None, is_dbe=None):
    from core.definition_engine import find_definitions

    answer = _answer(ctx)
    if len(answer) < 3:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    ans_c = Counter(answer)
    best, best_key = None, None

    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        n = len(words)
        if not (3 <= n <= _MAX_WORDS):
            continue

        def types(k):
            try:
                return indicator_types(words[k].text) or set()
            except Exception:
                return set()

        def is_anag(k):
            return is_anagram_indicator(words[k].text, indicator_types)

        def is_del(k):
            return "deletion" in types(k)

        def is_link_w(k):
            return bool(is_link and is_link(words[k].text))

        anag_words = [k for k in range(n) if is_anag(k)]
        del_words = [k for k in range(n) if is_del(k)]
        if not anag_words or not del_words:
            continue                              # gate: need both indicators

        for d in del_words:
            # the deletion indicator word `d`; its neighbours are the deleted/curtailed word
            for nbr in (d - 1, d + 1):
                # a neighbour may name the deleted letter or be the curtailed fodder word;
                # it MAY also be anagram-typed (e.g. "new" = both an anagram word and the
                # abbreviation N), so do NOT exclude it for that — the answer-driven match
                # and the anag-indicator exclusion disambiguate.
                if nbr < 0 or nbr >= n or nbr in del_words:
                    continue
                # --- MECHANISM A: nbr NAMES the removed letters (its abbreviation) ---
                for rv in _del_values(words[nbr].text, value_lookup):
                    pl = _attempt(ctx, answer, ans_c, split, words, n, is_anag, is_del,
                                  is_link_w, value_lookup, del_idx=d,
                                  named_idx=nbr, removed=rv, curtail_idx=None)
                    best, best_key = _keep(best, best_key, pl)
                # --- MECHANISM B: nbr is a fodder word CURTAILED (drop last letter) ---
                wl = raw(words[nbr].text)
                if len(wl) >= 2:
                    pl = _attempt(ctx, answer, ans_c, split, words, n, is_anag, is_del,
                                  is_link_w, value_lookup, del_idx=d,
                                  named_idx=None, removed=None, curtail_idx=nbr)
                    best, best_key = _keep(best, best_key, pl)
    return best


def _keep(best, best_key, parse):
    if parse is None or parse.status not in ("pass", "pending"):
        return best, best_key
    rank = 2 if parse.status == "pass" else 1
    key = (rank,)
    if best_key is None or key > best_key:
        return parse, key
    return best, best_key


def _attempt(ctx, answer, ans_c, split, words, n, is_anag, is_del, is_link_w,
             value_lookup, del_idx, named_idx, removed, curtail_idx):
    """Assemble fodder = all content words except the indicators / named-deletion word,
    apply the deletion, allow <= _MAX_SUBS substitutions, and require the resulting
    multiset to equal the answer EXACTLY. Returns a PASS/pending parse or None."""
    from itertools import combinations
    # an anagram indicator run is required somewhere; pick the first contiguous anag run that
    # is NOT the deletion indicator or the named/curtailed word (a word like "new" can be
    # typed both 'anagram' AND give an abbreviation N — when it is the deleted letter it must
    # not also be claimed as the anagram indicator).
    excl = {del_idx, named_idx, curtail_idx}
    anag_run = _first_run(range(n), lambda k: is_anag(k) and k not in excl)
    if anag_run is None:
        return None
    reserved = set(anag_run) | {del_idx}
    if named_idx is not None:
        reserved.add(named_idx)
    cand = [k for k in range(n) if k not in reserved]
    # FORCED fodder = content words (must contribute letters); the curtailed word is forced
    # fodder. OPTIONAL = link-eligible words, which may be fodder OR inert links (a link
    # word can still be anagram fodder, e.g. "while" in EDELWEISS) — searched below.
    forced = [k for k in cand if (k == curtail_idx) or not is_link_w(k)]
    optional = [k for k in cand if is_link_w(k) and k != curtail_idx]

    def raw_of(k):
        wl = raw(words[k].text)
        if k == curtail_idx:
            wl = wl[:-1]                           # last-letter curtailment
        return wl

    for r in range(len(optional) + 1):
        for extra in combinations(optional, r):
            fodder_all = sorted(forced + list(extra))
            if not fodder_all:
                continue
            # try 0.._MAX_SUBS of the fodder words substituting (a short abbreviation value)
            for ns in range(0, _MAX_SUBS + 1):
                for subs in combinations(fodder_all, ns):
                    bulk = [k for k in fodder_all if k not in set(subs)]
                    bulk_c = Counter("".join(raw_of(k) for k in bulk))
                    if removed is not None:
                        rc = Counter(removed)
                        if any(rc[c] > bulk_c.get(c, 0) for c in rc):
                            continue               # removed letters must be in the pool
                        bulk_c = bulk_c - rc
                    chosen = _match_subs(list(subs), ans_c - bulk_c, words, value_lookup)
                    if chosen is None:
                        continue
                    total = bulk_c + Counter("".join(v for _, v, _ in chosen))
                    if total != ans_c:
                        continue
                    return _build(ctx, answer, split, words, bulk, chosen, anag_run,
                                  del_idx, named_idx, removed, curtail_idx)
    return None


def _match_subs(sub_words, residual, words, value_lookup):
    if not sub_words:
        return [] if not +residual else None       # +Counter drops zero/negatives
    if any(v < 0 for v in residual.values()):
        return None
    k = sub_words[0]
    own = Counter(raw(words[k].text))
    for val, mech in value_lookup(words[k].text):
        if mech not in _SUB_MECH:
            continue
        vv = "".join(ch for ch in (val or "").upper() if ch.isalpha())
        if not vv or len(vv) > _MAX_SUB_LEN or Counter(vv) == own:
            continue
        c = Counter(vv)
        if any(c[ch] > residual.get(ch, 0) for ch in c):
            continue
        rest = _match_subs(sub_words[1:], residual - c, words, value_lookup)
        if rest is not None:
            return [(k, vv, mech)] + rest
    return None


def _first_run(idxs, pred):
    run = []
    for k in idxs:
        if pred(k):
            run.append(k)
        elif run:
            break
    return run or None


def _build(ctx, answer, split, words, bulk_idx, chosen, anag_run, del_idx,
           named_idx, removed, curtail_idx):
    from core.definition_engine import dbe_annotation

    sources, remaining = [], []
    for k in bulk_idx:
        wl = raw(words[k].text)
        if k == curtail_idx:
            wl = wl[:-1]                            # last-letter curtailment
        remaining.append([len(sources), Counter(wl)])
        sources.append(Source(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                              value=wl, mechanism="anagram_fodder"))
    for (k, val, mech) in chosen:
        remaining.append([len(sources), Counter(val)])
        sources.append(Source(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                              value=val, mechanism=mech))

    links = []
    for pos_i, ch in enumerate(answer, start=1):
        si = 0
        for entry in remaining:
            if entry[1].get(ch, 0) > 0:
                entry[1][ch] -= 1
                si = entry[0]
                break
        links.append(Link(answer_pos=pos_i, source_index=si, operation="anagram",
                          clue_atom_id=None, transform="anagram_of"))

    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    anag_toks = [words[k] for k in anag_run]
    annotations = [Annotation(
        clue_atom_ids=tuple(aid for t in anag_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in anag_toks), role="indicator",
        note="anagram indicator")]
    annotations.append(Annotation(
        clue_atom_ids=words[del_idx].atom_ids, text=words[del_idx].text,
        role="indicator",
        note="deletion indicator (removes %s)" % (removed or "last letter")))
    if named_idx is not None:
        annotations.append(Annotation(
            clue_atom_ids=words[named_idx].atom_ids, text=words[named_idx].text,
            role="indicator", note="deleted letters: %s" % removed))
    used = {del_idx, named_idx, curtail_idx} | set(anag_run) | set(bulk_idx) \
        | {k for k, _, _ in chosen}
    for k in range(len(words)):
        if k not in used:
            annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                          text=words[k].text, role="link",
                                          note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="anagram", solved_by="catalog")
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    warnings = []
    if not parse.is_complete():
        warnings.append("the answer letters are not fully covered by the fodder")
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
    elif missing or not parse.is_complete():
        parse.status = "fail"
    else:
        parse.status = "pending"
