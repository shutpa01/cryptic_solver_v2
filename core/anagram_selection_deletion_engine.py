"""Anagram + selection-deletion engine — an anagram whose fodder has a letter (or short
run) REMOVED, where the removed letters are a SELECTION of an adjacent word
(last/first/outer/middle/...). It is the sibling of anagram_deletion: that engine sources
the removed letters from an ABBREVIATION ("hard" = H); this one sources them from a
SELECTION indicator + word ("last of alluring" = G).

  "Naked, removing last of alluring loose nightwear" = IN THE RAW
    def "Naked"; "loose" = anagram indicator; "removing" = deletion indicator;
    "last" = selection indicator (last letter); "alluring" -> G (its last letter);
    anagram(NIGHTWEAR - G) = INTHERAW

A NEW stage — it never edits the working anagram_deletion engine. ANSWER-DRIVEN: the fodder
multiset MINUS the selected letters must equal the answer EXACTLY. GATED on THREE indicators
together — an anagram indicator, a deletion indicator AND a selection indicator — and the
selected word must FOLLOW the selection indicator (link words skipped), so it cannot
fabricate. Pure: a wfw_atoms context + injected predicates, no DB lookup of its own. A clean
PASS is filed as a catalog signature by the cascade's auto-file path, so a repeat of this
shape matches the signature directly (fast) instead of re-searching.
"""

from collections import Counter
from itertools import combinations

from core.wordplay import raw, is_anagram_indicator
from core.wfw_model import Source, Link, Annotation, Parse
from core.selection import select_span

_MAX_WORDS = 10
_MAX_SEL_LEN = 3          # a selection removes a letter or a short run


def _answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def solve_anagram_selection_deletion(ctx, defines, indicator_types, is_link,
                                     selection_rules, define_fallback=None, is_dbe=None):
    """Solve an anagram whose fodder loses a SELECTED letter/run of an adjacent word.
    Returns a PASS/pending Parse or None. `selection_rules(word)` -> the set of selection
    rules the DB licenses for that word (first/last/outer/...)."""
    from core.definition_engine import find_definitions

    answer = _answer(ctx)
    if len(answer) < 3:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    ans_c = Counter(answer)
    best = None

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

        def sel_rules(k):
            try:
                return selection_rules(words[k].text) or set()
            except Exception:
                return set()

        anag_words = [k for k in range(n) if is_anag(k)]
        del_words = [k for k in range(n) if is_del(k)]
        sel_words = [k for k in range(n) if sel_rules(k)]
        if not anag_words or not del_words or not sel_words:
            continue                              # gate: need all three indicators

        for s in sel_words:
            # the selected word: the next CONTENT word after the selection indicator,
            # skipping link words and other indicators ("last of alluring" -> alluring).
            sel_word = None
            for j in range(s + 1, n):
                if is_link_w(j) or j in anag_words or j in del_words or j in sel_words:
                    continue
                sel_word = j
                break
            if sel_word is None:
                continue
            for rule in sel_rules(s):
                for sel_str, sel_atoms in select_span(ctx, words[sel_word], rule):
                    sel_str = "".join(c for c in sel_str.upper() if c.isalpha())
                    if not sel_str or len(sel_str) > _MAX_SEL_LEN:
                        continue
                    parse = _attempt(ctx, answer, ans_c, split, words, n, is_anag,
                                     is_link_w, anag_words, del_words, sel_words,
                                     s, sel_word, sel_str, sel_atoms, rule)
                    if parse is not None and parse.status == "pass":
                        return parse
                    if parse is not None and best is None:
                        best = parse
    return best


def _first_run(idxs, pred):
    run = []
    for k in idxs:
        if pred(k):
            run.append(k)
        elif run:
            break
    return run or None


def _attempt(ctx, answer, ans_c, split, words, n, is_anag, is_link_w, anag_words,
             del_words, sel_words, sel_idx, sel_word_idx, removed, sel_atoms, rule):
    """Fodder = the content words that are NOT indicators and NOT the selected word; the
    selected letters are removed from the pool; the result must equal the answer EXACTLY."""
    # the anagram-indicator run (first contiguous run of anag words)
    anag_run = _first_run(range(n), is_anag)
    if anag_run is None:
        return None
    excl = set(anag_words) | set(del_words) | set(sel_words) | {sel_word_idx}
    cand = [k for k in range(n) if k not in excl]
    forced = [k for k in cand if not is_link_w(k)]       # content words -> fodder
    optional = [k for k in cand if is_link_w(k)]         # links: fodder OR inert
    rc = Counter(removed)

    for r in range(len(optional) + 1):
        for extra in combinations(optional, r):
            fodder = sorted(forced + list(extra))
            if not fodder:
                continue
            bulk_c = Counter("".join(raw(words[k].text) for k in fodder))
            if any(rc[c] > bulk_c.get(c, 0) for c in rc):
                continue                          # removed letters must be in the pool
            if bulk_c - rc == ans_c:              # exact anagram after the deletion
                return _build(ctx, answer, split, words, fodder, anag_run, del_words,
                              sel_idx, sel_word_idx, removed, sel_atoms, rule)
    return None


def _build(ctx, answer, split, words, fodder, anag_run, del_words, sel_idx,
           sel_word_idx, removed, sel_atoms, rule):
    from core.definition_engine import dbe_annotation

    sources, remaining = [], []
    for k in fodder:
        wl = raw(words[k].text)
        remaining.append([len(sources), Counter(wl)])
        sources.append(Source(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                              value=wl, mechanism="anagram_fodder"))

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

    # Account for EVERY wordplay word exactly once (no role by elimination): the anagram
    # indicator run, the selection indicator, the selected word, the deletion indicator(s)
    # that are not the selection word, and any leftover LINK words. Fodder words are the
    # sources above.
    accounted = set(fodder) | set(anag_run) | {sel_idx, sel_word_idx}
    anag_toks = [words[k] for k in anag_run]
    annotations = [Annotation(
        clue_atom_ids=tuple(aid for t in anag_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in anag_toks), role="indicator",
        note="anagram indicator")]
    annotations.append(Annotation(
        clue_atom_ids=words[sel_idx].atom_ids, text=words[sel_idx].text, role="indicator",
        note="selection indicator (%s letter)" % rule))
    annotations.append(Annotation(
        clue_atom_ids=words[sel_word_idx].atom_ids, text=words[sel_word_idx].text,
        role="deletion", note="%s letter → %s" % (rule, removed)))
    for d in del_words:                           # the genuine deletion indicator(s), e.g.
        if d in accounted:                        # "removing"; skip "last" if it doubles as
            continue                              # both the selection and a deletion word
        accounted.add(d)
        annotations.append(Annotation(
            clue_atom_ids=words[d].atom_ids, text=words[d].text, role="indicator",
            note="deletion indicator (removes the selected %s)" % removed))
    for k in range(len(words)):                    # leftover words are link glue ("of");
        if k in accounted:                         # forced fodder is already consumed, so a
            continue                               # leftover can only be a link, never content
        accounted.add(k)
        annotations.append(Annotation(
            clue_atom_ids=words[k].atom_ids, text=words[k].text, role="link",
            note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="anagram", solved_by="catalog")
    parse.template_id = None
    parse.matched_signature = None
    # verdict: full letter coverage already guaranteed by the exact-multiset match; a
    # provisional definition makes it pending, else a clean pass.
    if split.source == "pending":
        parse.warnings = ["the definition is provisional (queued for enrichment)"]
        parse.status = "pending"
    else:
        parse.warnings = []
        parse.status = "pass"
    return parse
