"""Reversal+charade — CATALOG-DRIVEN (design §4). Walks the mined 'reversal_charade'
signatures; for each, for each definition split at its edge, assigns the slots to clue runs
in clue order (gaps -> links), validates the reversal indicator, and reconstructs: tile the
answer with the value pieces, each placed FORWARD (SYN_F/ABR_F slot) or REVERSED (REV_F slot)
as its slot dictates. ANSWER-DRIVEN (answer.startswith(value[::-1], pos) for reversed pieces)
-> bounded by value count, no product, no memory cost.

Reuses reversal_charade_engine._build / _verify so the Parse is byte-identical to the
evidence engine (this is its signature-driven replacement in the cascade). Pure, DB-decoupled.
"""

from core import grammar
from core.wordplay import GLUE_POS
from core.reversal_charade_engine import _build, _verify

_VALUE_MECH = ("synonym", "abbreviation")


def _value_candidates(words, a, b, lookup_all):
    phrase = " ".join(words[k].text for k in range(a, b))
    out, seen = [], set()
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _is_rev_indicator(text, indicator_types):
    try:
        return "reversal" in (indicator_types(text) or set())
    except Exception:
        return False


def _answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _reconstruct(answer, pieces_spec, words, lookup_all):
    """pieces_spec: list of ((a, b), is_reversed). Tile the answer in answer order, each
    piece placed forward or reversed per its spec. Returns answer-order [(kind, run, v)]."""
    N = len(answer)
    npieces = len(pieces_spec)
    cand = []
    for (a, b), is_rev in pieces_spec:
        placed = []
        for v in _value_candidates(words, a, b, lookup_all):
            if is_rev:
                if v[::-1] == v:
                    continue            # a 1-letter/palindrome "reversal" is a no-op — not real
                s = v[::-1]
            else:
                s = v
            if s:
                placed.append((s, v))
        cand.append(placed)

    def dfs(pos, used, arrangement):
        if pos == N:
            return arrangement if len(used) == npieces else None
        for i in range(npieces):
            if i in used:
                continue
            for s, v in cand[i]:
                if answer.startswith(s, pos):
                    r = dfs(pos + len(s), used | {i}, arrangement + [(i, v)])
                    if r:
                        return r
        return None

    res = dfs(0, frozenset(), [])
    if res is None:
        return None
    out = []
    for i, v in res:
        run, is_rev = pieces_spec[i]
        out.append(("rev" if is_rev else "fwd", run, v))
    return out


def _place(slots, words, answer, postags, lookup_all, is_link, indicator_types):
    n = len(words)
    nslots = len(slots)

    def residue_link(k):
        return (is_link and is_link(words[k].text)) or (postags[k] in GLUE_POS)

    def finalize(assigned, gap_idxs):
        pieces_spec = [(r, role == "REV_F") for r, role in assigned
                       if role in ("SYN_F", "ABR_F", "REV_F")]
        if not any(is_rev for _, is_rev in pieces_spec):
            return None
        rev_i = [k for r, role in assigned if role == "REV_I" for k in range(*r)]
        if not rev_i:
            return None
        links = []
        for k in gap_idxs:
            if residue_link(k):
                links.append(k)
            else:
                return None
        arrangement = _reconstruct(answer, pieces_spec, words, lookup_all)
        if arrangement is None:
            return None
        return {"pieces": arrangement, "rev": sorted(rev_i), "links": sorted(links)}

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
    if not set(roles) <= {"SYN_F", "ABR_F", "REV_F", "REV_I"}:
        return None
    if roles.count("REV_F") < 1:
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


def solve_reversal_charade(ctx, defines, lookup_all, is_link, indicator_types,
                           templates=None, define_fallback=None, is_dbe=None):
    """Full reversal+charade solve — catalog-driven. First clean PASS (fewest residue links),
    else best non-pass, else None."""
    from core.definition_engine import find_definitions

    answer = _answer(ctx)
    if len(answer) < 4 or not templates:
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
