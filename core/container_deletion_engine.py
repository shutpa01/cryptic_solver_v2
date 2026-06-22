"""Container+deletion engine — ANSWER-DRIVEN.

A two-operation compound: build a CONTAINER (one value inserted into another), then a
DELETION trims the result to the answer. e.g.

    "old spades in the last shed" = TOSH
      O (old) + S (spades) = OS  inserted into the LITERAL  THE  ->  TOSHE
      'shed' the last letter (curtail)  ->  TOSH      def = "Rubbish"

ANSWER-DRIVEN, like the plain container engine — it does NOT enumerate a word's values.
It reconstructs the pre-deletion string FROM THE ANSWER (the answer with the letter[s] the
deletion removed restored, per the deletion indicator's DB sub-type), splits that into
outer + inner as substrings, and checks each is a value of a clue-word run by a membership
/charade test that tries the LITERAL letters first, then abbreviations, then (only if those
fail) synonyms. So a literal like THE is read in one step and a word's synonyms are never
scanned in a loop — no value enumeration, no arbitrary caps.

GATED on BOTH a container/insertion indicator AND a deletion indicator. The inner may be a
small charade (consecutive words' values, e.g. old+spades = OS). Isolated engine on the
wfw_model substrate; definition decided upstream; links classified last; pure.
"""

import string

from core import deletion
from core.wordplay import raw
from core.wfw_model import Source, Link, Annotation, Parse

MAX_RUN = 4
_ALPHA = string.ascii_uppercase
_MECH_PRI = {"literal": 0, "raw": 0, "abbreviation": 1, "synonym": 2}
# cheapest reconstruction first (one restored letter), the 26x26 'outer' last
_OP_ORDER = ["curtail", "behead", "heartless", "outer"]


def solve_container_deletion(ctx, defines, lookup_all, is_link, indicator_types,
                             deletion_subtypes, templates=None, define_fallback=None,
                             is_dbe=None, loc_rules=None):
    """Full container+deletion solve. First clean PASS, else best parse, else None.

    `loc_rules` (selection_rules) supports the location/operation split: a letter-location
    word ("capital"/"opening" = first letter) that sits with a genuine deletion operation
    word ("short"/"shed") is accounted as part of the deletion and pins which letters drop —
    it never licenses a deletion alone (that is enforced by removing such words' deletion
    typing from the DB)."""
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
        if len(words) < 3:
            continue
        parse = _assemble(ctx, answer, split, words, lookup_all, is_link,
                          indicator_types, deletion_subtypes, loc_rules)
        if parse is not None:
            if parse.status == "pass":
                return parse
            if best is None:
                best = parse
    return best


def _reconstruct(answer, op):
    """Pre-deletion candidates I with `op` applied giving the answer: the answer with the
    removed letter(s) restored at the op's position, over every letter choice (bounded,
    exhaustive — not a heuristic)."""
    N = len(answer)
    if op == "curtail":                          # removed I's last letter
        return [answer + c for c in _ALPHA]
    if op == "behead":                           # removed I's first letter
        return [c + answer for c in _ALPHA]
    if op == "outer":                            # removed I's first AND last
        return [c1 + answer + c2 for c1 in _ALPHA for c2 in _ALPHA]
    if op == "heartless":                        # removed I's single central letter
        m = (N + 1) // 2
        return [answer[:m] + c + answer[m:] for c in _ALPHA]
    return []


def _del_ops(words, deletion_subtypes, is_del):
    """The deletion ops the clue's deletion indicators license (from their DB sub-types).
    A generic/unnamed removal -> the common positional ops."""
    ops = set()
    for k in range(len(words)):
        if not is_del(k):
            continue
        subs = deletion_subtypes(words[k].text) if deletion_subtypes else set()
        named = False
        for s in subs:
            op = deletion.SUBTYPE_OP.get(s)
            if op:
                ops.add(op)
                named = True
        if not named:
            ops |= {"curtail", "behead"}         # unnamed removal: common positional ops
    return ops


def _ordered_values(lookup_all, phrase):
    """A word/phrase's candidate values, LITERAL first then abbreviation then synonym, so
    a literal/abbreviation is found without scanning synonyms. (value, mechanism)."""
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


def _can_make(words, a, b, target, vals, memo):
    """A charade split of `target` across words[a:b] — each word a contiguous value,
    LITERAL/abbreviation preferred (vals(a) is pre-ordered). Returns the split
    [(word_index, value, mechanism), ...] or None. Stops at the first working split."""
    if a >= b:
        return [] if target == "" else None
    key = (a, b, target)
    if key in memo:
        return memo[key]
    res = None
    for v, mech in vals(a):
        if v and target.startswith(v):
            rest = _can_make(words, a + 1, b, target[len(v):], vals, memo)
            if rest is not None:
                res = [(a, v, mech)] + rest
                break
    memo[key] = res
    return res


def _del_list(op, items):
    """Apply deletion op to a LIST, so per-letter provenance survives the trim (parallel
    to deletion.apply_op on a string)."""
    n = len(items)
    if op == "behead":
        return items[1:] if n >= 2 else None
    if op == "curtail":
        return items[:-1] if n >= 2 else None
    if op == "outer":
        return items[1:-1] if n >= 3 else None
    if op == "heartless":
        if n >= 3 and n % 2 == 1:
            m = n // 2
            return items[:m] + items[m + 1:]
        return None
    return None


def _assemble(ctx, answer, split, words, lookup_all, is_link, indicator_types,
              deletion_subtypes, loc_rules=None):
    n = len(words)

    def types(k):
        try:
            return indicator_types(words[k].text) or set()
        except Exception:
            return set()

    def is_con(k):
        return bool({"container", "insertion"} & types(k))

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

    def residue_link(k):
        return bool(is_link and is_link(words[k].text))

    if not any(is_con(k) for k in range(n)) or not any(is_del(k) for k in range(n)):
        return None                              # GATE: both ops' indicators required
    ops = _del_ops(words, deletion_subtypes, is_del)
    # LOCATION/OPERATION split: a location word pins which letters drop, but only because a
    # genuine deletion operation word is present (the gate above guarantees one). Fold its
    # op in so "short of capital" (general 'short' + location 'capital') can behead.
    for k in range(n):
        if not is_con(k) and not is_del(k):
            ops |= loc_ops(k)
    if not ops:
        return None

    vcache = {}

    def vals(a):
        if a not in vcache:
            vcache[a] = _ordered_values(lookup_all, words[a].text)
        return vcache[a]

    memo = {}
    runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)]

    best, seen_I = None, set()
    # cheap ops first (one restored letter = 26 candidates) before the costly 'outer'
    # (two = 676), and return on the first PASS — so a clue solved by curtail never pays
    # for the outer reconstruction.
    for op in [o for o in _OP_ORDER if o in ops]:
        for I in _reconstruct(answer, op):
            if I in seen_I:
                continue
            seen_I.add(I)
            LI = len(I)
            for p in range(0, LI):
                for L in range(1, LI - p + 1):
                    if p == 0 and p + L == LI:
                        continue                  # outer must be non-empty
                    inner, outer = I[p:p + L], I[:p] + I[p + L:]
                    if not outer:
                        continue
                    for (ia, ib) in runs:
                        if _can_make(words, ia, ib, inner, vals, memo) is None:
                            continue
                        for (oa, ob) in runs:
                            if not (ob <= ia or oa >= ib):
                                continue
                            if _can_make(words, oa, ob, outer, vals, memo) is None:
                                continue
                            parse = _build(ctx, answer, split, words, (oa, ob), outer,
                                           vals, (ia, ib), inner, I, p, L, op,
                                           is_con, is_del, residue_link, is_loc)
                            if parse is None:
                                continue
                            if parse.status == "pass":
                                return parse
                            if best is None:
                                best = parse
    return best


def _build(ctx, answer, split, words, outer_run, outer, vals, inner_run, inner, I, p, L,
           op, is_con, is_del, residue_link, is_loc=None):
    n = len(words)
    oa, ob = outer_run
    ia, ib = inner_run
    used = set(range(oa, ob)) | set(range(ia, ib))
    residue = [k for k in range(n) if k not in used]
    con = [k for k in residue if is_con(k)]
    dele = [k for k in residue if is_del(k) and k not in con]
    if not con or not dele:
        return None
    # A letter-location word in the residue ("capital"/"opening") is part of the deletion
    # expression (it names which letters the operation word removes), not unaccounted
    # content. Accounted only because a genuine deletion operation word (dele) is present.
    loc = [k for k in residue if is_loc and is_loc(k) and k not in con and k not in dele]
    loc_set = set(loc)
    links = []
    for k in residue:
        if k in con or k in dele or k in loc_set:
            continue
        if residue_link(k):
            links.append(k)
        else:
            return None                          # unaccounted content word

    # tagged pre-deletion string -> apply the op -> answer letters with their source tags
    tagged = [["outer", c] for c in I]
    for j in range(p, p + L):
        tagged[j][0] = "inner"
    survived = _del_list(op, tagged)
    if survived is None or len(survived) != len(answer):
        return None
    if "".join(c for _t, c in survived) != answer:
        return None

    def mech_of(run, value):
        sp = _can_make(words, run[0], run[1], value, vals, {})
        return sp[0][2] if sp and len(sp) == 1 else "charade"

    outer_toks = words[oa:ob]
    inner_toks = words[ia:ib]
    outer_src = Source(
        clue_atom_ids=tuple(aid for t in outer_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in outer_toks), value=outer,
        mechanism=mech_of(outer_run, outer))
    inner_src = Source(
        clue_atom_ids=tuple(aid for t in inner_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in inner_toks), value=inner,
        mechanism=mech_of(inner_run, inner))
    if oa < ia:
        sources = [outer_src, inner_src]; OUT, IN = 0, 1
    else:
        sources = [inner_src, outer_src]; OUT, IN = 1, 0

    links_out = [Link(answer_pos=i + 1, source_index=(IN if tag == "inner" else OUT),
                      operation="container_deletion", clue_atom_id=None)
                 for i, (tag, _c) in enumerate(survived)]
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    annotations = []
    for k in con:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="indicator",
                                      note="container indicator"))
    for k in dele:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="indicator",
                                      note="deletion indicator (%s)" % op))
    for k in loc:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="indicator",
                                      note="deletion location indicator (%s)" % op))
    for k in links:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link",
                                      note="link word"))
    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links_out, annotations=annotations,
                  definition=definition, operation="container_deletion",
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
