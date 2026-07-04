"""Design-faithful explainer — builds the atom-map DIRECTLY, no catalogue.

The mechanic (exactly the agreed design):
  1. Atomise the answer into letter slots.
  2. Peel the definition (a DB lookup that defines the whole answer).
  3. For each remaining clue word, pull the letter-strings it can become from
     the DB (itself / a synonym / an abbreviation) or from its OWN letters.
  4. Land those letters on the answer. The OPERATION is read off the landing:
       letters in order            -> identity (plain charade piece)
       letters reversed            -> reversal
       a permutation (same multiset)-> anagram      (uses the word's own letters)
       a contiguous run inside word -> hidden        (uses the word's own letters)
  5. A non-identity landing is LICENSED only if an indicator of that type is
     present in the clue (indicators VALIDATE, they do not initiate).
  6. Completeness: every answer letter sourced, every clue word given a role
     (piece / definition / indicator / link), every used operation licensed.

This never consults the signature catalogue. The catalogue is for recognising a
shape after the fact, not for solving.
"""

from core.wfw_atoms import build_wfw_atom_context
from core.wfw_model import Source, Link, Annotation, Parse
from core.definition_engine import find_definitions

# operation -> the indicator type (in the DB) that licenses it
OP_LICENCE = {"reversed": "reversal", "anagram": "anagram", "hidden": "hidden"}


def _answer_letters(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _run_text(words, a, b):
    return " ".join(t.text for t in words[a:b])


def _run_letters(words, a, b):
    return "".join(c for t in words[a:b] for c in t.text.upper() if c.isalpha())


def _run_atoms(words, a, b):
    return tuple(aid for t in words[a:b] for aid in t.atom_ids)


def _indicator_map(words, indicator_types):
    """{indicator_type -> set(word indices)} for every clue word that the DB
    records as an indicator of that type. This is the set of LICENSED operations."""
    out = {}
    for i, w in enumerate(words):
        try:
            for t in (indicator_types(w.text) or ()):
                out.setdefault(t, set()).add(i)
        except Exception:
            pass
    return out


def _landings(words, a, b, A, apos, licensed):
    """Ways the word-run words[a:b] can land on the answer starting at apos.
    Yields (operation, value, mechanism, consumed_len). Indicator-using landings
    are tried FIRST (a present indicator should do work), identity last."""
    run_letters = _run_letters(words, a, b)
    # --- landings that use the word's OWN letters ---
    if "anagram" in licensed:                       # scramble of the run's letters
        L = len(run_letters)
        if L and sorted(A[apos:apos + L]) == sorted(run_letters):
            yield "anagram", run_letters, "anagram", L
    if "hidden" in licensed:                         # a run hidden inside the letters
        for L in range(len(A) - apos, 0, -1):
            seg = A[apos:apos + L]
            if seg and seg in run_letters:
                yield "hidden", seg, "hidden", L
                break
    # --- landings that use a DB letter-string (synonym / abbreviation / itself) ---
    for val, mech in _DB(words, a, b):
        L = len(val)
        seg = A[apos:apos + L]
        if len(seg) < L:
            continue
        if "reversal" in licensed and seg == val[::-1]:
            yield "reversed", val, mech, L
        if seg == val:
            yield "identity", val, mech, L


_WIRING = {}        # filled by solve() so _DB can reach lookup_all


def _DB(words, a, b):
    return _WIRING["lookup_all"](_run_text(words, a, b))


def solve(clue_text, answer, wiring, direction=None):
    """Return (Parse, trace_lines) for the first complete, validated atom-map,
    or (None, trace_lines) if none is found."""
    global _WIRING
    _WIRING = wiring
    ctx = build_wfw_atom_context(clue_text, answer, direction=direction)
    A = _answer_letters(ctx)
    trace = ["ANSWER atomised: " + " ".join("%s(%d)" % (c, i + 1)
                                            for i, c in enumerate(A))]

    splits = [s for s in find_definitions(ctx, wiring["defines"],
                                          is_dbe=wiring.get("is_dbe"))
              if s.source == "db"]
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        licensed = _indicator_map(words, wiring["indicator_types"])
        trace2 = list(trace)
        trace2.append("DEFINITION: %r = %s  (def at %s)"
                      % (split.phrase, answer, split.where))
        trace2.append("INDICATORS present: " + (", ".join(
            "%s->%s" % (",".join(words[i].text for i in idx), t)
            for t, idx in licensed.items()) or "(none)"))
        pieces = _cover(words, A, licensed, wiring)
        if pieces is not None:
            return _build(ctx, split, words, pieces, A, licensed, wiring, trace2)
        trace2.append("  -> no complete validated cover for this definition")
        trace = trace[:1]  # keep only the atomisation header for the next split
        trace_last = trace2
    return None, (trace_last if splits else
                  trace + ["DEFINITION: none found in DB"])


def _cover(words, A, licensed, wiring):
    """DFS: tile the answer left-to-right. Each piece is a contiguous run of
    unused words landing on the answer by a licensed operation. Returns the list
    of pieces (a,b,op,value,mech,apos,clen) or None."""
    n = len(words)
    result = [None]

    def leftover_ok(used, used_ops):
        # every used non-identity op must have its indicator present
        for op in used_ops:
            if OP_LICENCE.get(op) and OP_LICENCE[op] not in licensed:
                return False
        # every unused word must be a licensed indicator or a link word
        for i in range(n):
            if i in used:
                continue
            is_ind = any(i in idx for idx in licensed.values())
            if not is_ind and not wiring["is_link"](words[i].text):
                return False
        return True

    def dfs(apos, used, used_ops, pieces):
        if apos == len(A):
            if leftover_ok(used, used_ops):
                result[0] = pieces
                return True
            return False
        for a in range(n):
            if a in used:
                continue
            for b in range(a + 1, min(a + 3, n) + 1):
                if any(i in used for i in range(a, b)):
                    break
                for op, val, mech, clen in _landings(words, a, b, A, apos, licensed):
                    if dfs(apos + clen, used | set(range(a, b)),
                           used_ops | ({op} if op != "identity" else set()),
                           pieces + [(a, b, op, val, mech, apos, clen)]):
                        return True
        return False

    dfs(0, set(), set(), [])
    return result[0]


_LINK_TRANSFORM = {"identity": None, "reversed": "reversed",
                   "anagram": "anagram_of", "hidden": None}


def _build(ctx, split, words, pieces, A, licensed, wiring, trace):
    sources, links = [], []
    used_ops = set()
    for si, (a, b, op, val, mech, apos, clen) in enumerate(pieces):
        sources.append(Source(clue_atom_ids=_run_atoms(words, a, b),
                              text=_run_text(words, a, b), value=val,
                              mechanism=mech))
        for k in range(clen):
            links.append(Link(answer_pos=apos + k + 1, source_index=si,
                              operation=op, transform=_LINK_TRANSFORM.get(op)))
        if op != "identity":
            used_ops.add(op)
        lic = ("  [validated by indicator '%s']" % OP_LICENCE[op]
               if op in OP_LICENCE else "")
        trace.append("PIECE: %-14s -> %-10s [%s] %s -> positions %s%s"
                     % (_run_text(words, a, b), val, mech, op,
                        list(range(apos + 1, apos + clen + 1)), lic))

    annotations = []
    used_word_idx = {i for a, b, *_ in pieces for i in range(a, b)}
    for i, w in enumerate(words):
        if i in used_word_idx:
            continue
        ind_types = wiring["indicator_types"](w.text) or set()
        used_here = ind_types & {OP_LICENCE[o] for o in used_ops if o in OP_LICENCE}
        if used_here:
            annotations.append(Annotation(clue_atom_ids=w.atom_ids, text=w.text,
                                          role="indicator",
                                          note="%s indicator" % sorted(used_here)[0]))
            trace.append("INDICATOR: %r validates %s" % (w.text, sorted(used_here)[0]))
        else:
            annotations.append(Annotation(clue_atom_ids=w.atom_ids, text=w.text,
                                          role="link", note="link word"))
            trace.append("LINK: %r" % w.text)

    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition,
                  operation=pieces[0][2] if len(pieces) == 1 else "charade",
                  solved_by="atomsig")
    complete = parse.is_complete() and not parse.unexplained_words(ctx)
    parse.status = "pass" if complete else "fail"
    trace.append("COMPLETE: %s" % ("all answer letters sourced, all words roled"
                                   if complete else "NO — " + str(parse.warnings)))
    return parse, trace


if __name__ == "__main__":
    from core import engine_registry as er
    w = er.db_only(er.make_db_wiring())
    cases = [
        ("Card from company doctor", "COMB", None),
        ("Agent on vessel lifted hat", "TOPPER", None),
        ("Runner-up in close race?", "LOSER", None),
    ]
    for clue, ans, d in cases:
        print("=" * 70)
        print("CLUE: %s   ANSWER: %s" % (clue, ans))
        parse, trace = solve(clue, ans, w, direction=d)
        for line in trace:
            print("  " + line)
        print("  STATUS:", parse.status if parse else "no parse")
