"""Anagram engine — CATALOG-DRIVEN with links-as-residue (design §4).

Signature-driven sibling of the charade signature engine, for pure anagrams. It walks
the mined anagram signatures (ANA_F fodder + optional ANA_I indicator, in various
orders/word-counts, def at one edge) in priority order and instantiates each:

  for each anagram signature (priority order), for each definition split at its edge:
    - PLACE the typed slots on the wordplay words in clue order, GAPS ALLOWED (a slot
      consumes its n_words consecutive words);
    - ANA_F slot: the fodder words, whose letters (as written, or contraction-stripped
      "Lionel's"->LIONEL) must ANAGRAM to the whole answer (sorted-equal, not a
      self-anagram, not an exact reversal);
    - ANA_I slot: the indicator words, which must be a CONFIRMED anagram indicator;
    - the GAP words are classified LAST: function/connective (is_link or POS function)
      -> link; anything else leaves the parse unaccounted and the placement is rejected.
      Links are NEVER pre-stripped.

Among passing placements, prefer the FEWEST residue (link) words, then signature
priority. Records the matched signature (template_id) — design §10. The definition is
the shared stage (find_definitions); the engine only explains the wordplay.

ISOLATED per the 13-engines decision: own assembly + own verifier; the colour/letter
assignment mirrors the evidence anagram engine (per-fodder-word colour). Pure and
DB-decoupled.
"""

from collections import Counter
from itertools import combinations

from core import contractions, grammar
from core.wordplay import GLUE_POS, raw, is_anagram_indicator
from core.wfw_model import Source, Link, Annotation, Parse


def _fodder_forms(words, a, b):
    """The fodder letters of words[a:b], as written and contraction-stripped."""
    as_written = "".join(raw(words[k].text) for k in range(a, b))
    stripped = "".join(raw(contractions.strip_suffixes(words[k].text))
                       for k in range(a, b))
    return [f for f in (as_written, stripped) if f]


def _place(slots, words, answer, postags, is_link, indicator_types):
    """Place ANA_F / ANA_I slots on the words (gaps allowed); classify gaps as links
    LAST. Returns {fodder:(a,b,letters), indicator:[idx], links:[idx]} or None."""
    n, N = len(words), len(answer)
    nslots = len(slots)

    def residue_link(k):
        return (is_link and is_link(words[k].text)) or (postags[k] in GLUE_POS)

    def is_ind_word(k):
        return is_anagram_indicator(words[k].text, indicator_types)

    def finalize(placed, gap_idxs):
        fodder = next((p for p in placed if p[2] == "ANA_F"), None)
        if fodder is None:
            return None
        a, b, _ = fodder
        # INTERIOR-LINK EXCLUSION: a link word inside the fodder span may be a joiner
        # whose letters are NOT fodder ("tea AND chips" -> TEACHIPS). Try excluding
        # subsets of interior link words, FEWEST exclusions first (so a link that IS
        # fodder is kept by default); the exact letter-match decides. The excluded
        # words become links (residue), never silently dropped.
        span = list(range(a, b))
        link_in = [k for k in span if is_link and is_link(words[k].text)]
        kept_idx, letters, excl_idx = None, None, []
        for r in range(0, len(link_in) + 1):
            for excl in combinations(link_in, r):
                es = set(excl)
                kept = [k for k in span if k not in es]
                if not kept:
                    continue
                aw = "".join(raw(words[k].text) for k in kept)
                st = "".join(raw(contractions.strip_suffixes(words[k].text))
                             for k in kept)
                f = next((x for x in (aw, st) if len(x) == N
                          and sorted(x) == sorted(answer) and x[::-1] != answer),
                         None)
                if f is not None:
                    kept_idx, letters, excl_idx = kept, f, list(excl)
                    break
            if kept_idx is not None:
                break
        if kept_idx is None:
            return None                              # fodder does not anagram answer
        ind_slots = [p for p in placed if p[2] == "ANA_I"]
        indicator = []
        for (ia, ib, _) in ind_slots:
            if not any(is_ind_word(k) for k in range(ia, ib)):
                return None                          # ANA_I run not a real indicator
            indicator.extend(range(ia, ib))
        links = []
        for k in list(gap_idxs) + excl_idx:
            if residue_link(k):
                links.append(k)
            else:
                return None                          # content word unaccounted
        return {"fodder_idx": kept_idx, "letters": letters,
                "indicator": sorted(indicator), "links": sorted(links)}

    def dfs(si, wi, placed, gaps):
        if si == nslots:
            return finalize(placed, gaps + list(range(wi, n)))
        nw = slots[si].n_words
        role = slots[si].role
        for j in range(wi, n - nw + 1):
            r = dfs(si + 1, j + nw, placed + [(j, j + nw, role)],
                    gaps + list(range(wi, j)))
            if r:
                return r
        return None

    return dfs(0, 0, [], [])


def _try_template(ctx, answer, template, split, words, postags, is_link,
                  indicator_types):
    if split.where != template.def_pos:
        return None
    roles = {s.role for s in template.slots}
    if not roles <= {"ANA_F", "ANA_I"} or "ANA_F" not in roles:
        return None
    if sum(s.n_words for s in template.slots) > len(words):
        return None
    placement = _place(template.slots, words, answer, postags, is_link,
                       indicator_types)
    if placement is None:
        return None
    return _build(ctx, answer, split, words, placement, template)


def _build(ctx, answer, split, words, placement, template):
    """Assemble the Parse: one Source per fodder word (colour), each answer letter
    attributed to a fodder word that supplied it; indicator + links as annotations."""
    from core.definition_engine import dbe_annotation
    letters = placement["letters"]
    fodder_tokens = [words[k] for k in placement["fodder_idx"]]
    # per-fodder-word letters (split the matched letters back across the words by the
    # same as-written/stripped choice that produced `letters`)
    per_word, used = [], 0
    for t in fodder_tokens:
        w_letters = raw(t.text)
        if used + len(w_letters) > len(letters) or \
                letters[used:used + len(w_letters)] != w_letters:
            w_letters = raw(contractions.strip_suffixes(t.text))
        per_word.append(w_letters)
        used += len(w_letters)

    sources, remaining = [], []
    for t, wl in zip(fodder_tokens, per_word):
        remaining.append([len(sources), Counter(wl)])
        sources.append(Source(clue_atom_ids=t.atom_ids, text=t.text, value=wl,
                              mechanism="anagram_fodder"))
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

    ind_toks = [words[k] for k in placement["indicator"]]
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    annotations = []
    if ind_toks:
        annotations.append(Annotation(
            clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
            text=" ".join(t.text for t in ind_toks), role="indicator",
            note="anagram indicator", source="db"))
    for k in placement["links"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link",
                                      note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="anagram", solved_by="catalog")
    parse.template_id = template.id
    parse.matched_signature = template.signature
    _verify(ctx, parse, bool(ind_toks))
    return parse


def _verify(ctx, parse, has_indicator):
    warnings = []
    if not parse.is_complete():
        warnings.append("the answer letters are not fully covered by the fodder")
    missing = parse.unexplained_words(ctx)
    if missing:
        warnings.append("these clue words are unaccounted for: "
                        + ", ".join(repr(m) for m in missing))
    if not has_indicator:
        warnings.append("no anagram indicator found")
    if parse.definition is None:
        warnings.append("no definition found")
    elif getattr(parse.definition, "source", "db") == "pending":
        warnings.append("the definition is provisional (queued for enrichment)")
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing or not has_indicator or not parse.is_complete():
        parse.status = "fail"
    else:
        parse.status = "pending"


def solve_anagram(ctx, defines, is_link, indicator_types, templates,
                  define_fallback=None, is_dbe=None):
    """Full anagram solve — catalog-driven. Returns the first clean PASS (fewest
    residue), else the best parse, else None."""
    from core.definition_engine import find_definitions

    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 3 or not templates:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    prepared = []
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 1:
            continue
        postags = grammar.pos_tags([t.text for t in words]) or [None] * len(words)
        prepared.append((split, words, postags))
    if not prepared:
        return None

    best_pass, best_key, best_other = None, None, None
    for template in templates:                       # priority order
        for split, words, postags in prepared:
            parse = _try_template(ctx, answer, template, split, words, postags,
                                  is_link, indicator_types)
            if parse is None:
                continue
            if parse.status == "pass":
                residue = sum(1 for a in parse.annotations if a.role == "link")
                if best_key is None or residue < best_key:
                    best_pass, best_key = parse, residue
                    if residue == 0:
                        return parse
            elif best_other is None:
                best_other = parse
    return best_pass or best_other
