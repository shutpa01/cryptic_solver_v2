"""Anagram-substitution engine — an anagram whose fodder contains a SUBSTITUTED word.

A common device the plain anagram engine cannot see: one fodder word is replaced by a
short value (a single letter or short abbreviation) BEFORE the anagram. E.g.

    "Safe, Oscar shut away from rioting" = OUT OF HARM'S WAY
      def       = "Safe"
      indicator = "rioting"  (anagram)
      fodder    = O (Oscar) + SHUTAWAYFROM   ->  anagram -> OUTOFHARMSWAY

The plain anagram engine builds fodder from RAW clue letters only, so it tried
raw("Oscar shut away from") = OSCARSHUTAWAYFROM (16 letters) which cannot anagram the
13-letter answer, and failed.

METHOD — residual deduction (be smart; KNOW what you are looking for). The answer's
letters are fixed. Hold out one fodder word as the SUBSTITUTED one; the rest contribute
their raw letters (the "bulk"). If the bulk is a sub-multiset of the answer, the DEDUCED
RESIDUAL = answer_letters - bulk_letters is EXACTLY what the held-out word must supply.
Then look for a value of that word — from ANY table (abbreviation, substitution, synonym,
single-letter) — whose letters equal the residual. For the example: bulk "shut away from"
= 12 letters, answer 13, residual = {O}, and "Oscar" yields O. Conclusive.

The residual is capped at 3 letters: substituted anagram fodder is conventionally a single
letter (Oscar->O, fifty->L) or a short abbreviation (times->X, artists->RA). A long
multi-letter substitution would be an "indirect anagram" (anagramming a synonym), which is
unfair by convention and would explode into false positives — the cap forbids it while the
exact-residual match keeps every accepted substitution honest. Source table is permissive;
the deduction is strict.

ISOLATED per the 13-engines rule: own assembly + verifier; the plain anagram engine is
untouched. Gated on an anagram indicator, answer-driven, definition decided upstream. The
same residual-deduction idea extends naturally to anagram+DELETION (a contiguous fodder
that over-covers the answer by exactly the letters a 'without TIME' word removes) — a
sibling engine, not built here.
"""

from collections import Counter

from core.wordplay import raw, is_anagram_indicator
from core.wfw_model import Source, Link, Annotation, Parse

_MAX_RESIDUAL = 3        # substituted fodder is a single letter / short abbreviation only

# Prefer a conventional single-letter source over a synonym when several values match the
# residual (lower = preferred). Label-only ordering; the letters are identical either way.
_MECH_PREF = {"abbreviation": 0, "substitution": 1, "raw": 2, "synonym": 3}


def _build(ctx, answer, split, words, bulk_idx, sub_i, sub_value, sub_mech,
           ind_idx):
    """Assemble the Parse. `bulk_idx` are the raw fodder words; `sub_i` the substituted
    word with value `sub_value` (mechanism `sub_mech`); `ind_idx` the anagram indicator
    words. Each answer letter is attributed (anagram_of) to whichever source still has it."""
    from core.definition_engine import dbe_annotation

    sources, remaining = [], []
    for k in bulk_idx:
        wl = raw(words[k].text)
        remaining.append([len(sources), Counter(wl)])
        sources.append(Source(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                              value=wl, mechanism="anagram_fodder"))
    # the substituted word: its VALUE (not its raw letters) enters the fodder
    remaining.append([len(sources), Counter(sub_value)])
    sources.append(Source(clue_atom_ids=words[sub_i].atom_ids, text=words[sub_i].text,
                          value=sub_value, mechanism=sub_mech))

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


def solve_anagram_substitution(ctx, defines, value_lookup, indicator_types, is_link,
                               define_fallback=None, is_dbe=None):
    """Solve an anagram whose fodder has one substituted word. GATED on an anagram
    indicator, ANSWER-DRIVEN (residual deduction). Returns the best PASS/PENDING parse,
    else None (abstain). All non-indicator wordplay words are treated as fodder (no
    interior-link exclusion in this version), so it abstains on clues with a true link
    word inside the fodder rather than mis-reading it."""
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
        if len(words) < 3:                             # indicator + substituted + >=1 bulk
            continue
        ind_words = {k for k, t in enumerate(words)
                     if is_anagram_indicator(t.text, indicator_types)}
        if not ind_words:
            continue                                   # gated: needs an anagram indicator
        def_words = len([t for t in split.def_tokens if t.kind == "word"]) \
            if getattr(split, "def_tokens", None) else 1
        # An indicator-flagged word may actually be FODDER ("from" in "shut away from" is
        # both a flagged indicator AND fodder letters). So don't exclude all of them: try
        # each contiguous run of flagged words AS the indicator, leaving the rest as fodder;
        # the residual deduction + exact anagram pick the reading that works.
        for ia in range(len(words)):
            for ib in range(ia + 1, len(words) + 1):
                ind_idx = list(range(ia, ib))
                if not all(k in ind_words for k in ind_idx):
                    continue
                content = [k for k in range(len(words)) if k not in set(ind_idx)]
                if len(content) < 2:                   # need substituted + >=1 bulk word
                    continue
                for sub_i in content:
                    bulk_idx = [k for k in content if k != sub_i]
                    bulk = "".join(raw(words[k].text) for k in bulk_idx)
                    bc = Counter(bulk)
                    if any(bc[c] > ans_counter[c] for c in bc):
                        continue                       # bulk not contained in the answer
                    residual = ans_counter - bc
                    rlen = sum(residual.values())
                    if rlen < 1 or rlen > _MAX_RESIDUAL:
                        continue                       # short substitution only
                    raw_sub = Counter(raw(words[sub_i].text))
                    for val, mech in value_lookup(words[sub_i].text):
                        vv = "".join(ch for ch in (val or "").upper() if ch.isalpha())
                        # SOURCE GUARD against indirect anagrams. A SINGLE letter may come
                        # from any table (Oscar->O, love->O, 50->L are all conventional and
                        # the residual pins it). A MULTI-letter value is accepted ONLY from
                        # the curated abbreviation/substitution tables (church->CH, artists
                        # ->RA); a multi-letter SYNONYM anagrammed (punk->RUN, one->ACE) is
                        # an indirect anagram — unfair by convention and a frequent false
                        # attribution (often really an anagram+charade) — so it is refused.
                        if mech == "synonym" and len(vv) > 1:
                            continue
                        # A GENUINE substitution changes the letters. If the value's letters
                        # equal the word's own raw letters (a->A, war->RAW), the word is just
                        # plain anagram fodder, not a substitution — reject so we never
                        # mislabel (or steal) a plain anagram.
                        if vv and Counter(vv) == residual and Counter(vv) != raw_sub:
                            parse = _build(ctx, answer, split, words, bulk_idx, sub_i,
                                           vv, mech, ind_idx)
                            if parse.status not in ("pass", "pending"):
                                continue
                            status_rank = 2 if parse.status == "pass" else 1
                            key = (status_rank, -def_words, -_MECH_PREF.get(mech, 9))
                            if best_key is None or key > best_key:
                                best, best_key = parse, key
    return best
