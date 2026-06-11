"""Plain reversal — CATALOG-DRIVEN (design §4). Walks the mined 'reversal' signatures; for
each, for each definition split at its edge, assigns the slots (one SYN_F/ABR_F fodder +
REV_I indicator) to clue runs in clue order (gaps -> links), validates the reversal indicator,
and reconstructs ANSWER-DRIVEN: the single fodder run's DB value must equal reverse(answer).

Reuses reversal_engine._build / _verify so the Parse is byte-identical to the evidence engine
(this is the signature-driven replacement for that engine in the cascade). Pure, DB-decoupled,
no enumeration -> no memory cost.
"""

from core import grammar
from core.wordplay import GLUE_POS
from core.reversal_engine import _build, _verify

_VALUE_MECH = ("synonym", "abbreviation")


def _value_candidates(words, a, b, lookup_all):
    phrase = " ".join(words[k].text for k in range(a, b))
    out, seen = [], set()
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in seen:
            seen.add(v)
            out.append((v, mech))
    return out


def _is_rev_indicator(text, indicator_types):
    try:
        return "reversal" in (indicator_types(text) or set())
    except Exception:
        return False


def _answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _place(slots, words, answer, postags, lookup_all, is_link, indicator_types):
    n = len(words)
    nslots = len(slots)
    target = answer[::-1]
    if target == answer:
        return None

    def residue_link(k):
        return (is_link and is_link(words[k].text))

    def finalize(assigned, gap_idxs):
        fodder = [r for r, role in assigned if role in ("SYN_F", "ABR_F")]
        if len(fodder) != 1:
            return None
        rev_i = [k for r, role in assigned if role == "REV_I" for k in range(*r)]
        links = []
        for k in gap_idxs:
            if residue_link(k):
                links.append(k)
            else:
                return None
        a, b = fodder[0]
        mech = None
        for v, m in _value_candidates(words, a, b, lookup_all):
            if v == target:
                mech = m
                break
        if mech is None:
            return None
        return {"run": fodder[0], "value": target, "mech": mech,
                "rev": sorted(rev_i), "links": sorted(links)}

    def dfs(si, wi, assigned, gaps):
        if si == nslots:
            return finalize(assigned, gaps + list(range(wi, n)))
        slot = slots[si]
        nw = slot.n_words
        for j in range(wi, n - nw + 1):
            run = (j, j + nw)
            if slot.role == "REV_I" and not any(
                    _is_rev_indicator(words[k].text, indicator_types)
                    for k in range(*run)):
                continue
            r = dfs(si + 1, j + nw, assigned + [(run, slot.role)],
                    gaps + list(range(wi, j)))
            if r:
                return r
        return None

    return dfs(0, 0, [], [])


def _try_template(ctx, answer, template, split, words, postags, lookup_all, is_link,
                  indicator_types):
    if split.where != template.def_pos:
        return None
    if template.fodder_word_count > len(words):
        return None
    roles = [s.role for s in template.slots]
    if not set(roles) <= {"SYN_F", "ABR_F", "REV_I"}:
        return None
    if sum(roles.count(r) for r in ("SYN_F", "ABR_F")) != 1:
        return None
    if roles.count("REV_I") < 1:
        return None
    placement = _place(template.slots, words, answer, postags, lookup_all, is_link,
                       indicator_types)
    if placement is None:
        return None
    parse = _build(ctx, split, words, answer, placement)
    parse.matched_signature = template.signature
    parse.template_id = template.id
    return parse


def solve_reversal(ctx, defines, lookup_all, is_link, indicator_types,
                   templates=None, define_fallback=None, is_dbe=None):
    """Full plain-reversal solve — catalog-driven. First clean PASS (fewest residue links),
    else best non-pass, else None."""
    from core.definition_engine import find_definitions

    answer = _answer(ctx)
    if len(answer) < 3 or not templates:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    prepared = []
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 2:
            continue
        postags = grammar.wordplay_pos_tags(ctx, words)
        prepared.append((split, words, postags))
    if not prepared:
        return None

    best_pass, best_key, best_other = None, None, None
    for template in templates:
        for split, words, postags in prepared:
            parse = _try_template(ctx, answer, template, split, words, postags,
                                  lookup_all, is_link, indicator_types)
            if parse is None:
                continue
            if parse.status == "pass":
                residue = sum(1 for a in parse.annotations if a.role == "link")
                if best_pass is None or residue < best_key:
                    best_pass, best_key = parse, residue
            elif best_other is None:
                best_other = parse
    return best_pass or best_other
