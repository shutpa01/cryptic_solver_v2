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

        def is_link_w(k):
            return bool(is_link and is_link(words[k].text))

        # PHRASE-AWARE indicators: deletion and selection rows may be multi-word.
        from core.engine_common import typed_runs
        _del_pos = {k for r in typed_runs(words, range(n), indicator_types, "deletion")
                    for k in r}
        sel_runs, _sel_pos = [], set()
        for a in range(n):
            for b in range(a + 1, min(a + 4, n) + 1):
                phrase = " ".join(words[k].text for k in range(a, b))
                try:
                    rules = selection_rules(phrase) or set()
                except Exception:
                    rules = set()
                if rules:
                    sel_runs.append((list(range(a, b)), rules))
                    _sel_pos.update(range(a, b))

        anag_words = [k for k in range(n) if is_anag(k)]
        del_words = sorted(_del_pos)
        if not anag_words or not del_words or not sel_runs:
            continue                              # gate: need all three indicators

        for srun, srules in sel_runs:
            # the selected fodder: the next CONTENT word after the selection indicator
            # (links/other indicators skipped), extended to a RUN of 1..3 words from
            # there ("last of alluring" -> alluring; multi-word fodder now reachable).
            sel_word = None
            for j in range(srun[-1] + 1, n):
                if is_link_w(j) or j in anag_words or j in _del_pos or j in _sel_pos:
                    continue
                sel_word = j
                break
            if sel_word is None:
                continue
            from core.selection import select_span_run
            for sw_end in range(sel_word + 1, min(sel_word + 3, n) + 1):
                sw_run = list(range(sel_word, sw_end))
                if any(k in anag_words or k in _del_pos or k in _sel_pos
                       for k in sw_run):
                    break
                for rule in srules:
                    for sel_str, sel_atoms in select_span_run(ctx, words[sel_word:sw_end],
                                                              rule):
                        sel_str = "".join(c for c in sel_str.upper() if c.isalpha())
                        if not sel_str or len(sel_str) > _MAX_SEL_LEN:
                            continue
                        parse = _attempt(ctx, answer, ans_c, split, words, n, is_anag,
                                         is_link_w, anag_words, del_words,
                                         sorted(_sel_pos), srun, sw_run, sel_str,
                                         sel_atoms, rule)
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
             del_words, sel_words, sel_run, sel_word_run, removed, sel_atoms, rule):
    """Fodder = the content words that are NOT indicators and NOT the selected word(s);
    the selected letters are removed from the pool; the result must equal the answer
    EXACTLY. `sel_run` = the selection indicator's positions; `sel_word_run` = the
    selected fodder's positions (each a single word or a run)."""
    # the anagram-indicator run (first contiguous run of anag words)
    anag_run = _first_run(range(n), is_anag)
    if anag_run is None:
        return None
    excl = set(anag_words) | set(del_words) | set(sel_words) | set(sel_word_run)
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
                              sel_run, sel_word_run, removed, sel_atoms, rule)
    return None


def _build(ctx, answer, split, words, fodder, anag_run, del_words, sel_run,
           sel_word_run, removed, sel_atoms, rule):
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
    accounted = set(fodder) | set(anag_run) | set(sel_run) | set(sel_word_run)
    anag_toks = [words[k] for k in anag_run]
    annotations = [Annotation(
        clue_atom_ids=tuple(aid for t in anag_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in anag_toks), role="indicator",
        note="anagram indicator")]
    annotations.append(Annotation(
        clue_atom_ids=tuple(aid for k in sel_run for aid in words[k].atom_ids),
        text=" ".join(words[k].text for k in sel_run), role="indicator",
        note="selection indicator (%s letter)" % rule))
    annotations.append(Annotation(
        clue_atom_ids=tuple(aid for k in sel_word_run for aid in words[k].atom_ids),
        text=" ".join(words[k].text for k in sel_word_run),
        role="deletion", note="%s letter → %s" % (rule, removed)))
    # the genuine deletion indicator(s) — grouped per contiguous RUN (joined phrase),
    # skipping any position that doubles as the selection indicator / selected word.
    from core.engine_common import contiguous_groups
    for drun in contiguous_groups([d for d in del_words if d not in accounted]):
        accounted.update(drun)
        annotations.append(Annotation(
            clue_atom_ids=tuple(aid for k in drun for aid in words[k].atom_ids),
            text=" ".join(words[k].text for k in drun), role="indicator",
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
