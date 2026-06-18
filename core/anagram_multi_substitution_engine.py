"""Anagram multi-substitution engine — an anagram whose fodder contains TWO OR MORE
substituted words (the single-substitution engine handles exactly one).

    "Changing circs surrounding Old Firm? It's just hot air" = SCIROCCO
      def       = "It's just hot air"
      indicator = "Changing"  (anagram)
      fodder    = CIRCS (literal "circs")  +  O (Old)  +  CO (Firm)
      "surrounding" = an inert container word (surface only): the whole pool is
                      anagrammed, so the Old Firm letters cannot be a literal
                      contiguous insertion — the answer's geometry forbids it.
      anagram(CIRCS + O + CO) = SCIROCCO

METHOD — the same residual deduction as the single-substitution engine, generalised.
Split the (non-indicator) words into BULK (contribute raw letters), SUBSTITUTED (each
contributes a short DB value, <= 3 letters), and inert GLUE (a link or another indicator,
e.g. the container word "surrounding"). The bulk raw letters must be a sub-multiset of the
answer; the deduced residual = answer - bulk must be supplied EXACTLY by a combination of
short, GENUINE substitutions (>= 2 of them). Each substitution is source-guarded exactly as
the single-sub engine: a multi-letter value only from the abbreviation/substitution tables,
never a synonym (no indirect anagrams).

A NEW stage (never edit a working engine). Gated on an anagram indicator, answer-driven,
definition decided upstream. Pure and DB-decoupled.
"""

from collections import Counter
from itertools import combinations

from core.wordplay import raw, is_anagram_indicator
from core.wfw_model import Source, Link, Annotation, Parse

_MAX_SUB_LEN = 3          # each substituted value: a letter or short abbreviation
_MAX_SUBS = 3             # cap the number of substituted words (bounds the search)
_MAX_CONTENT = 8          # bound the role search per clue
_MECH_PREF = {"abbreviation": 0, "substitution": 1, "raw": 2, "synonym": 3}


_SUB_MECH = ("abbreviation", "substitution")


def _sub_values(word, value_lookup):
    """Short, GENUINE substitution values for a word (<= _MAX_SUB_LEN letters), drawn ONLY
    from the curated abbreviation/substitution tables — never a synonym. With two or three
    free substitutions the combinatorial slack is large, and a single-letter SYNONYM from
    any word ("head"->A) lets the engine manufacture a letter-fit (it conjured BACCHUS that
    way). Conventional substituted anagram fodder is an abbreviation anyway (Oscar->O,
    times->X, firm->CO); restricting to that keeps the engine honest. A value equal to the
    word's own raw letters is not a substitution and is dropped."""
    own = Counter(raw(word))
    out = []
    seen = set()
    for val, mech in value_lookup(word):
        if mech not in _SUB_MECH:
            continue
        vv = "".join(ch for ch in (val or "").upper() if ch.isalpha())
        if not vv or len(vv) > _MAX_SUB_LEN:
            continue
        if Counter(vv) == own:                 # not a genuine substitution
            continue
        key = (vv, mech)
        if key not in seen:
            seen.add(key)
            out.append((vv, mech))
    return out


def _match_subs(sub_words, residual, words, value_lookup):
    """Choose one genuine value per substituted word (given by index in `words`) so their
    letters sum EXACTLY to `residual`. Returns [(word_index, value, mechanism), ...] or
    None."""
    if not sub_words:
        return [] if not residual else None
    k = sub_words[0]
    for val, mech in _sub_values(words[k].text, value_lookup):
        c = Counter(val)
        if any(c[ch] > residual.get(ch, 0) for ch in c):
            continue
        rest = _match_subs(sub_words[1:], residual - c, words, value_lookup)
        if rest is not None:
            return [(k, val, mech)] + rest
    return None


def solve_anagram_multi_substitution(ctx, defines, value_lookup, indicator_types,
                                     is_link, define_fallback=None, is_dbe=None):
    from core.definition_engine import find_definitions

    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 3:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    ans_counter = Counter(answer)
    best, best_key = None, None

    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if not (3 <= len(words) <= _MAX_CONTENT + 3):
            continue

        def types(k):
            try:
                return indicator_types(words[k].text) or set()
            except Exception:
                return set()

        def is_glue(k):
            return bool((is_link and is_link(words[k].text)) or types(k))

        ind_flag = {k for k in range(len(words))
                    if is_anagram_indicator(words[k].text, indicator_types)}
        if not ind_flag:
            continue

        # try each contiguous run of anagram-flagged words AS the indicator
        for ia in range(len(words)):
            for ib in range(ia + 1, len(words) + 1):
                ind_idx = list(range(ia, ib))
                if not all(k in ind_flag for k in ind_idx):
                    continue
                rest = [k for k in range(len(words)) if k not in set(ind_idx)]
                if len(rest) > _MAX_CONTENT:
                    continue
                forced = [k for k in rest if not is_glue(k)]    # must be fodder
                optional = [k for k in rest if is_glue(k)]      # fodder OR inert glue

                # choose which glue-eligible words also act as fodder; the rest are glue
                for r in range(len(optional) + 1):
                    for extra in combinations(optional, r):
                        fodder = forced + list(extra)
                        glue = [k for k in optional if k not in set(extra)]
                        if len(fodder) < 2:
                            continue
                        # PRECISION GUARD: a multi-sub anagram may absorb at most ONE inert
                        # surface indicator (the single container/"surrounding" word the full
                        # anagram makes mechanically dead). Discarding TWO+ operator words to
                        # force a letter-fit is too loose (it manufactured BACCHUS by dropping
                        # both "head" and "absorbed") — abstain. Link words are unrestricted.
                        inert_ind = sum(1 for k in glue
                                        if not (is_link and is_link(words[k].text)))
                        if inert_ind > 1:
                            continue
                        # within fodder, pick the substituted words (>= 2); rest are bulk
                        for ns in range(2, min(_MAX_SUBS, len(fodder)) + 1):
                            for subs in combinations(fodder, ns):
                                bulk = [k for k in fodder if k not in set(subs)]
                                bulk_c = Counter("".join(raw(words[k].text) for k in bulk))
                                if any(bulk_c[c] > ans_counter[c] for c in bulk_c):
                                    continue
                                residual = ans_counter - bulk_c
                                if sum(residual.values()) > _MAX_SUBS * _MAX_SUB_LEN:
                                    continue
                                chosen = _match_subs(list(subs), residual, words,
                                                     value_lookup)
                                if chosen is None:
                                    continue
                                parse = _build(ctx, answer, split, words, bulk,
                                               chosen, ind_idx, glue,
                                               indicator_types, is_link)
                                if parse.status not in ("pass", "pending"):
                                    continue
                                rank = 2 if parse.status == "pass" else 1
                                key = (rank, -len(glue), -ns)
                                if best_key is None or key > best_key:
                                    best, best_key = parse, key
    return best


def _build(ctx, answer, split, words, bulk_idx, chosen, ind_idx, glue_idx,
           indicator_types=None, is_link=None):
    """bulk_idx: raw fodder words; chosen: [(word_idx, value, mech), ...] substituted
    words; ind_idx: anagram indicator run; glue_idx: inert link/indicator words."""
    from core.definition_engine import dbe_annotation

    sources, remaining = [], []
    for k in bulk_idx:
        wl = raw(words[k].text)
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
    ind_toks = [words[k] for k in ind_idx]
    annotations = [Annotation(
        clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in ind_toks), role="indicator",
        note="anagram indicator", source="db")]
    for k in glue_idx:
        link_word = bool(is_link and is_link(words[k].text))
        try:
            typed = bool(indicator_types and indicator_types(words[k].text))
        except Exception:
            typed = False
        if link_word or not typed:
            role, note = "link", "link word"
        else:
            role, note = "indicator", "surface indicator (no letters)"
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role=role, note=note))
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
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing or not parse.is_complete():
        parse.status = "fail"
    else:
        parse.status = "pending"
