"""Substitution engine — a base value with one clued letter replaced by another.

INSOLENCE = IN SILENCE ("saying nothing") with one[I] replaced by love[O]. The base is a
DB value of a clue span; the removed and inserted letters BOTH come from the wordplay
table via lookup (one->I, love->O) — never fabricated. Gated by a substitution indicator
("acquiring", "replacing", "becomes", "instead of", "ousting", ...). Answer-driven: the
base with the swap must EQUAL the answer exactly.

Whole-answer form for now (the substituted base IS the whole answer); the multi-piece
charade-with-a-substitution-constituent form is a later extension. Every clue word is
accounted: base span, the two letter words, the indicator, and links. Paired with the
base screen (the substitution detail is carried on the indicator).
"""

from core import engine_common
from core.wfw_model import Source, Annotation, Parse

def _norm(text):
    return "".join(c for c in text.lower() if c.isalpha())


def solve_substitution(ctx, defines, lookup, synonyms_of, is_link, indicator_types,
                       define_fallback=None, is_dbe=None):
    """Whole-answer substitution. Abstains (None) unless a substitution indicator is
    present and some base value, with one clued letter swapped for another clued letter,
    equals the answer. Returns a Parse (pass / pending) or None.

    The substitution indicator is read from the DB via `indicator_types` (formerly a
    hardcoded set here). The base comes from `synonyms_of` (UNFILTERED — the
    pre-substitution form is not a substring of the answer, so the answer-filtered
    `lookup` would miss it); the swapped single letters come from `lookup` (they do
    appear in the answer)."""
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
        parse = _try_split(ctx, answer, split, words, lookup, synonyms_of, is_link,
                           indicator_types)
        if parse is None:
            continue
        if parse.status == "pass":
            return parse
        if best is None:
            best = parse
    return best


def _single_letters(word, answer, lookup):
    """Single-letter values the word can clue (wordplay-table letters: one->I, love->O)."""
    return {(v or "").upper() for v, m in lookup(word, answer) if len(v or "") == 1}


def _try_split(ctx, answer, split, words, lookup, synonyms_of, is_link, indicator_types):
    N = len(answer)
    ind = [i for i, t in enumerate(words)
           if "substitution" in (indicator_types(t.text) or set())]
    if not ind:
        return None                                  # gated: needs a substitution indicator
    letter = {i: _single_letters(words[i].text, answer, lookup)
              for i in range(len(words))}
    letter = {i: v for i, v in letter.items() if v}
    n = len(words)
    for ai in letter:
        for bi in letter:
            if ai == bi:
                continue
            for bstart in range(n):                  # base span (contiguous), not ai/bi/ind
                for bend in range(bstart, n):
                    span = set(range(bstart, bend + 1))
                    if ai in span or bi in span or span & set(ind):
                        continue
                    phrase = " ".join(words[k].text for k in range(bstart, bend + 1))
                    for bval in (synonyms_of(phrase) or []):
                        B = "".join(c for c in (bval or "").upper() if c.isalpha())
                        if len(B) != N:
                            continue
                        for La in letter[ai]:
                            for Lb in letter[bi]:
                                for (X, ax), (Y, by) in (((La, ai), (Lb, bi)),
                                                         ((Lb, bi), (La, ai))):
                                    for pos in range(len(B)):
                                        if B[pos] == X and B[:pos] + Y + B[pos + 1:] == answer:
                                            used = span | {ax, by} | set(ind)
                                            rest = [k for k in range(n) if k not in used]
                                            if all(is_link and is_link(words[k].text)
                                                   for k in rest):
                                                return _build(ctx, split, words,
                                                              (bstart, bend), bval, B,
                                                              (ax, X), (by, Y), ind, rest)
    return None


def _build(ctx, split, words, base_span, base_val, B, removed, inserted, ind, links):
    """removed/inserted = (word_index, letter). The base value with `removed`->`inserted`
    spells the answer; show the base + the two clued letters + the substitution indicator."""
    bs, be = base_span
    base_toks = words[bs:be + 1]
    rx_i, rx_l = removed
    in_i, in_l = inserted
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    sources = [
        Source(clue_atom_ids=tuple(aid for t in base_toks for aid in t.atom_ids),
               text=" ".join(t.text for t in base_toks), value=base_val.upper(),
               mechanism="synonym", source="db"),
        Source(clue_atom_ids=words[in_i].atom_ids, text=words[in_i].text, value=in_l,
               mechanism="abbreviation", source="db"),
        Source(clue_atom_ids=words[rx_i].atom_ids, text=words[rx_i].text, value=rx_l,
               mechanism="abbreviation", source="db"),
    ]
    ind_toks = [words[i] for i in ind]
    annotations = [Annotation(
        clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in ind_toks), role="indicator",
        note="substitution: %s (%s) replaces %s (%s) in %s"
             % (words[in_i].text, in_l, words[rx_i].text, rx_l, base_val.upper()))]
    for k in links:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link", note="link word"))
    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=[], annotations=annotations,
                  definition=definition, operation="substitution",
                  solved_by="substitution")
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    """pass: every clue word accounted + DB definition. pending: provisional definition."""
    warnings = []
    missing = parse.unexplained_words(ctx)
    w_unaccounted = engine_common.unaccounted_words_warning(ctx, parse)
    if w_unaccounted:
        warnings.append(w_unaccounted)
    w_def = engine_common.definition_warning(parse)
    if w_def:
        warnings.append(w_def)
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
