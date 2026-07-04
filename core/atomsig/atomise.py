"""Increment 1 — the ATOMISATION GATE.

Given an answer and a set of candidate DB bits (word -> value), find EVERY way the
bits' atoms tile the answer's atoms as a TWO-WAY UNIQUE map (a bijection: each
answer atom sourced once, each source atom consumed once). For each tiling the
MECHANISM is read off the geometry alone:

    pieces' atoms in order            -> charade (identity pieces)
    a piece's atoms reversed          -> that piece is reversed
    a piece's atoms non-contiguous    -> container (it is split around another)

Only AFTER a mechanism is detected is an indicator consulted, purely to legitimise
it (indicators confirm, they do not initiate). The gate returns ALL valid maps, so
NON-UNIQUENESS is exposed, never silently resolved.

This is the gate only — no search proposes the bits yet (that is increment 2).
The bits are supplied; the gate verifies.
"""

from itertools import combinations


def atomise(s):
    """The answer as a list of (letter, position) atoms, 1-based."""
    return [(c, i + 1) for i, c in enumerate(s)]


def _cover(seg, idxs, bits, offset):
    """Every way to tile the answer-segment `seg` using EXACTLY the bits in `idxs`,
    as a two-way map. Recursive, so it NESTS: a charade piece may itself be a
    container, a container's inner may itself be a charade, etc. Positions are
    global (offset + local). Returns a list of piece-lists."""
    n = len(seg)
    if not idxs:
        return []
    results = []

    # A single bit fills the whole segment: identity, reversed, or anagram. Deletion
    # (a too-long piece dropping letters) is NOT realised here — that is a manual
    # hand-solve operation, not the auto-search.
    if len(idxs) == 1:
        lb, v = bits[idxs[0]]
        pos = list(range(offset + 1, offset + n + 1))
        if len(v) == n:
            if seg == v:
                results.append([(lb, v, pos, "identity")])
            elif seg == v[::-1]:
                results.append([(lb, v, pos, "reversed")])
            elif sorted(seg) == sorted(v):
                results.append([(lb, v, pos, "anagram")])
        return results

    # CONTAINER: one outer bit split around the rest (which tile the middle).
    for oi in idxs:
        lb, oval = bits[oi]
        rest = tuple(x for x in idxs if x != oi)
        for k in range(1, len(oval)):
            head, tail = oval[:k], oval[k:]
            if seg[:k] == head and seg[n - len(tail):] == tail:
                gap = seg[k:n - len(tail)]
                if not gap:
                    continue
                for sub in _cover(gap, rest, bits, offset + k):
                    opos = (list(range(offset + 1, offset + k + 1)) +
                            list(range(offset + n - len(tail) + 1, offset + n + 1)))
                    results.append([(lb, oval, opos, "split")] + sub)

    # CHARADE: a leftmost block covered by a non-empty proper subset, the rest by
    # the complement. The block length is the total letters that subset places.
    for r in range(1, len(idxs)):
        for S in combinations(idxs, r):
            comp = tuple(x for x in idxs if x not in S)
            # the subset's letters fill the block exactly (no overshoot/deletion —
            # deletion is a manual hand-solve operation, not the auto-search).
            p = sum(len(bits[i][1]) for i in S)
            if p < 1 or p >= n:
                continue
            left = _cover(seg[:p], S, bits, offset)
            if not left:
                continue
            right = _cover(seg[p:], comp, bits, offset + p)
            for l in left:
                for rr in right:
                    results.append(l + rr)
    return results


def find_maps(answer, bits):
    """bits: list of (label, value). Every two-way-unique tiling of the answer,
    NESTING allowed. Each map is (kind, pieces) with pieces = [(label, value,
    positions, mechanism)]; kind is derived from the structure."""
    bits = list(bits)
    seen, out = set(), []
    for pieces in _cover(answer, tuple(range(len(bits))), bits, 0):
        key = tuple(sorted((lb, v, tuple(ps), m) for lb, v, ps, m in pieces))
        if key in seen:
            continue
        seen.add(key)
        if any(m == "split" for *_, m in pieces):
            kind = "container"
        elif len(pieces) > 1:
            kind = "charade"
        else:
            kind = "single"
        out.append((kind, pieces))
    return out


def analyse(answer, bits, indicator_words, def_phrase, wiring):
    """Run the gate AND emit a full evidence record — so that when it does not cleanly
    pass, the next person (hand-solving) inherits everything it found instead of a bare
    'no'. Returns a dict: atoms, bits (with DB-backing), definition (with DB-backing),
    every atom-map found, the indicator confirmation, the gaps, and a verdict."""
    lookup_all, indicator_types, defines = (wiring["lookup_all"],
                                            wiring["indicator_types"], wiring["defines"])

    def db_backed(value, label):
        return any(v == value for v, _ in lookup_all(label))

    ev = {
        "answer": answer,
        "atoms": atomise(answer),
        "bits": [{"word": lb, "value": v, "db_backed": db_backed(v, lb)}
                 for lb, v in bits],
        "definition": {"phrase": def_phrase,
                       "db_backed": bool(def_phrase) and defines(def_phrase, answer)},
        "indicators_present": {w: sorted(indicator_types(w) or []) for w in indicator_words},
        "maps": [],
    }
    for kind, pieces in find_maps(answer, bits):
        ok, conf = confirm_indicators(pieces, indicator_words, indicator_types)
        ev["maps"].append({"kind": kind,
                           "pieces": [{"word": lb, "value": v, "positions": ps,
                                       "mechanism": m} for lb, v, ps, m in pieces],
                           "indicator_ok": ok, "indicator": conf})

    legit = [m for m in ev["maps"] if m["indicator_ok"]]
    gaps = []
    gaps += ["bit not in DB: %s=%s" % (b["word"], b["value"])
             for b in ev["bits"] if not b["db_backed"]]
    if def_phrase and not ev["definition"]["db_backed"]:
        gaps += ["definition not in DB: %r" % def_phrase]
    for m in legit:
        gaps += ["indicator missing for: %s" % t for t in m["indicator"]["missing"]]
    ev["gaps"] = gaps

    if not ev["maps"]:
        ev["verdict"] = "NONE — no two-way atom-map; evidence above for hand-solving"
    elif len(legit) > 1:
        ev["verdict"] = "AMBIGUOUS — %d legitimate maps; do not fabricate" % len(legit)
    elif not legit:
        ev["verdict"] = "REJECTED — map(s) found but no indicator legitimises the mechanism"
    elif gaps:
        ev["verdict"] = "PENDING — unique map, but %d DB gap(s) to enrich" % len(gaps)
    else:
        ev["verdict"] = "PASS — unique, fully DB-backed, indicator-confirmed"
    return ev


def print_evidence(ev):
    print("ANSWER %s   atoms: %s" % (ev["answer"],
          " ".join("%s(%d)" % (c, i) for c, i in ev["atoms"])))
    print("  definition: %r  db_backed=%s" % (ev["definition"]["phrase"],
                                              ev["definition"]["db_backed"]))
    print("  bits found:")
    for b in ev["bits"]:
        print("    %-12s = %-9s db_backed=%s" % (b["word"], b["value"], b["db_backed"]))
    print("  indicators present: %s" % ev["indicators_present"])
    print("  atom-maps found:")
    for m in ev["maps"]:
        desc = " + ".join("%s=%s[%s@%s]" % (p["word"], p["value"], p["mechanism"],
                                            p["positions"]) for p in m["pieces"])
        print("    [%s] %s  indicator_ok=%s %s" % (m["kind"], desc, m["indicator_ok"],
              m["indicator"]["confirmed"] or m["indicator"]["missing"]))
    if ev["gaps"]:
        print("  GAPS (for hand-solving): " + "; ".join(ev["gaps"]))
    print("  VERDICT: %s" % ev["verdict"])


def _viable(value, answer):
    """A DB value can only contribute if its letters could land on the answer:
    contiguous (identity), reversed, or split around something (a container outer,
    possibly nested inside the answer). Cheap pre-filter so the search stays
    bounded; the gate does the real check."""
    from collections import Counter
    if value in answer or value[::-1] in answer:
        return True
    # a shorter value whose letters are all available could be a container outer
    # (split somewhere inside the answer) or an inner piece.
    if len(value) <= len(answer) and not (Counter(value) - Counter(answer)):
        return True
    # overshoot: a piece may carry up to TWO letters that get deleted, so (value
    # minus a 1- or 2-letter block), forward or reversed, need only appear IN the
    # answer — RIOT->RIO (whole) or FILM->LIF as part of LIFER.
    for extra in (1, 2):
        if extra >= len(value):
            break
        for k in range(len(value) - extra + 1):
            rest = value[:k] + value[k + extra:]
            if rest and (rest in answer or rest[::-1] in answer):
                return True
    return False


def _removed_letters(value, ps, mech, answer):
    """The contiguous block a deletion piece drops (the letters of `value` not
    landed). Recomputed from the piece so it can be named by a clue word."""
    seg = "".join(answer[p - 1] for p in sorted(ps))
    extra = len(value) - len(seg)
    if extra <= 0:
        return ""
    target = seg[::-1] if "reversed" in mech else seg
    for k in range(len(value) - extra + 1):
        if value[:k] + value[k + extra:] == target:
            return value[k:k + extra]
    return ""


def _confirm_roles(pieces, nonpiece_words, indicator_types, is_link,
                   answer="", lookup_all=None, src=None):
    """Every detected non-identity mechanism must be legitimised by a DB indicator
    among the leftover words, AND every leftover word must itself be a confirming
    indicator or a DB link word — else it is unaccounted and the placement is
    rejected (the completeness invariant). Deletion (and naming a deleted letter)
    is a manual hand-solve operation, not part of the auto-search.

    A piece whose value came from the clue's SURFACE letters (src[(label,value)] ==
    'surface') is a hidden: its placement geometry additionally requires a `hidden`
    indicator — so a hidden / reverse-hidden is licensed without any hidden-specific
    code, it is just the surface-source rule layered on the normal geometry."""
    need = set()
    for lb, v, _, m in pieces:
        need |= _needs(m)
        if src and src.get((lb, v)) == "surface":
            need.add("hidden")
    have, roled = {}, []
    for w in nonpiece_words:
        types = set(indicator_types(w) or ())
        match = types & need
        if match:
            for t in match:
                have.setdefault(t, []).append(w)
            roled.append((w, "indicator(%s)" % sorted(match)[0]))
        elif is_link(w):
            roled.append((w, "link"))
        else:
            return None                      # unaccounted word -> reject
    if any(t not in have for t in need):
        return None                          # a mechanism with no indicator -> reject
    return {"confirmed": {t: have[t][0] for t in need}, "roled": roled}


def search(clue, answer, wiring, direction=None, max_combos=300000, max_cand=60):
    """Feed ALL DB candidates through the gate. Strips the definition, gathers every
    viable DB value per wordplay word, tries every combination through find_maps, and
    collects every legitimate two-way atom-map. Returns a full evidence record + a
    verdict; non-uniqueness surfaces on its own."""
    from itertools import product
    from core.wfw_atoms import build_wfw_atom_context
    from core.definition_engine import find_definitions

    ctx = build_wfw_atom_context(clue, answer, direction=direction)
    A = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    lookup_all, indicator_types = wiring["lookup_all"], wiring["indicator_types"]
    is_link, defines = wiring["is_link"], wiring["defines"]

    # DOUBLE DEFINITION first: the whole clue splits into two parts that BOTH define
    # the answer, with no wordplay. A standalone check, not the tiler ("Fight a bit"
    # = SCRAP: Fight=SCRAP, a bit=SCRAP).
    clue_words = [t.text for t in ctx.clue_tokens if t.kind == "word"]
    n = len(clue_words)
    for i in range(1, n):
        left = " ".join(clue_words[:i])
        if not defines(left, answer):
            continue
        for j in range(i, n):                       # j==i: adjacent; j>i: link gap
            gap = clue_words[i:j]
            right = " ".join(clue_words[j:])
            if defines(right, answer) and all(is_link(g) for g in gap):
                return {"clue": clue, "answer": answer, "atoms": atomise(A),
                        "dd": (left, right), "dd_link": " ".join(gap),
                        "maps": [], "primary": None,
                        "verdict": "PASS — double definition"}

    db_splits = [s for s in find_definitions(ctx, defines, is_dbe=wiring.get("is_dbe"))
                 if s.source == "db"]
    if db_splits:
        splits = [(s.phrase, [t for t in s.wordplay_tokens if t.kind == "word"], True)
                  for s in db_splits]
    else:                                    # no DB definition: tentative edge splits
        allw = [t for t in ctx.clue_tokens if t.kind == "word"]
        splits = []
        for k in range(1, min(5, len(allw))):
            splits.append((" ".join(t.text for t in allw[:k]), allw[k:], False))
            splits.append((" ".join(t.text for t in allw[len(allw) - k:]),
                           allw[:len(allw) - k], False))

    legit, split_ev = [], {}
    for def_phrase, words, def_backed in splits:
        cand = []
        for w in words:
            seen, vals = set(), []
            for v, mech in lookup_all(w.text):
                v = (v or "").upper()
                if v and v not in seen and _viable(v, A):
                    seen.add(v)
                    vals.append((v, mech))
            cand.append(vals[:max_cand])
        split_ev[def_phrase] = (def_backed,
                          [{"word": w.text, "values": [v for v, _ in cand[i]],
                            "indicator": sorted(indicator_types(w.text) or []),
                            "link": is_link(w.text), "has_db_value": bool(lookup_all(w.text))}
                           for i, w in enumerate(words)])
        # A piece is a contiguous RUN of up to MAXRUN words, looked up as one phrase
        # — so a multi-word synonym ("sea at Cannes" -> MED) is a single piece, not
        # three unaccounted words. (The single-word-only lookup was the bug.)
        n = len(words)
        MAXRUN = 4
        run_cands = {}
        for a in range(n):
            for b in range(a + 1, min(a + MAXRUN, n) + 1):
                phrase = " ".join(words[i].text for i in range(a, b))
                seen, vals = set(), []
                for v, mech in lookup_all(phrase):
                    v = (v or "").upper()
                    if v and v not in seen and _viable(v, A):
                        seen.add(v)
                        vals.append((v, mech))
                # LITERAL fodder: the run's own letters (anagram / charade of letters).
                # Without this the search can never see an anagram.
                raw = "".join(c for i in range(a, b)
                              for c in words[i].text.upper() if c.isalpha())
                if raw and raw not in seen and _viable(raw, A):
                    seen.add(raw)
                    vals.append((raw, "literal"))
                # SURFACE source: any contiguous slice of these words' letters — it may
                # cross the internal word boundary, which the whole-value lookups never
                # can. A hidden / reverse-hidden then falls out of the SAME tiler as an
                # identity / reversed placement of such a slice; the only special rule is
                # in _confirm_roles, which makes a surface-sourced piece require a
                # `hidden` indicator. No mechanism-specific code.
                for i0 in range(len(raw)):
                    for j0 in range(i0 + 2, len(raw) + 1):
                        sl = raw[i0:j0]
                        if sl != raw and sl not in seen and _viable(sl, A):
                            seen.add(sl)
                            vals.append((sl, "surface"))
                if vals:
                    run_cands[(a, b)] = vals[:max_cand]

        def _selections(i, chosen):
            if i == n:
                if chosen:
                    yield chosen
                return
            yield from _selections(i + 1, chosen)            # word i: not a piece
            for b in range(i + 1, min(i + MAXRUN, n) + 1):     # word i: start a piece run
                for v, m in run_cands.get((i, b), ()):
                    yield from _selections(b, chosen + [(i, b, v, m)])

        for cnt, chosen in enumerate(_selections(0, [])):
            if cnt > max_combos:
                break
            bits = [(" ".join(words[i].text for i in range(a, b)), v)
                    for (a, b, v, m) in chosen]
            src = {(" ".join(words[i].text for i in range(a, b)), v): m
                   for (a, b, v, m) in chosen}      # (label,value) -> source type
            covered = {i for (a, b, _, _) in chosen for i in range(a, b)}
            nonpiece = [words[i].text for i in range(n) if i not in covered]
            for kind, pieces in find_maps(A, bits):
                conf = _confirm_roles(pieces, nonpiece, indicator_types, is_link,
                                      A, lookup_all, src)
                if conf is not None:
                    legit.append((def_phrase, def_backed, kind, pieces, conf))

    # de-duplicate by (definition, the pieces themselves)
    seen, uniq = set(), []
    for dp, db, kind, pieces, conf in legit:
        key = (dp, tuple(sorted((lb, v, tuple(ps), m) for lb, v, ps, m in pieces)))
        if key not in seen:
            seen.add(key)
            uniq.append((dp, db, kind, pieces, conf))

    if not uniq:
        verdict = "NONE — no two-way atom-map; evidence below for hand-solving"
    elif len(uniq) > 1:
        verdict = "AMBIGUOUS — %d legitimate maps; do not fabricate, hand-solve" % len(uniq)
    elif not uniq[0][1]:
        verdict = "PENDING — unique map, but definition not DB-backed"
    else:
        verdict = "PASS — unique, fully DB-backed, indicator-confirmed"

    # evidence to display: the WINNING split when there is a map, else the split
    # that got furthest (most words with a viable bit).
    if uniq:
        dp = uniq[0][0]
        primary = (dp,) + split_ev.get(dp, (False, []))
    elif split_ev:
        dp = max(split_ev, key=lambda k: sum(1 for wd in split_ev[k][1] if wd["values"]))
        primary = (dp,) + split_ev[dp]
    else:
        primary = None
    return {"clue": clue, "answer": answer, "atoms": atomise(A),
            "primary": primary, "maps": uniq, "verdict": verdict}


def print_search(r):
    print("CLUE: %s   ANSWER: %s" % (r["clue"], r["answer"]))
    print("  atoms: " + " ".join("%s(%d)" % (c, i) for c, i in r["atoms"]))
    if r["primary"]:
        dp, db, words = r["primary"]
        print("  definition tried: %r  db_backed=%s" % (dp, db))
        print("  wordplay evidence (viable DB bits per word):")
        for wd in words:
            flag = "" if wd["has_db_value"] else "  <-- NO DB VALUE"
            print("    %-12s values=%s ind=%s link=%s%s"
                  % (wd["word"], wd["values"], wd["indicator"], wd["link"], flag))
    print("  legitimate atom-maps:")
    for dp, db, kind, pieces, conf in r["maps"]:
        desc = " + ".join("%s=%s[%s@%s]" % (lb, v, m, ps) for lb, v, ps, m in pieces)
        print("    [%s] %s   def=%r  indicators=%s" % (kind, desc, dp, conf["confirmed"]))
    print("  VERDICT: %s" % r["verdict"])


def _needs(mech):
    """The DB indicator type(s) a detected mechanism must be legitimised by. A
    contiguous removal is a deletion; a removal that splits the piece around a new
    letter is a substitution (its own indicator), not a container."""
    s = set()
    if "deletion_split" in mech:
        s.add("substitution")
    elif "deletion" in mech:
        s.add("deletion")
    elif "split" in mech:
        s.add("container")
    if "reversed" in mech:
        s.add("reversal")
    if mech == "anagram":
        s.add("anagram")
    return s


def verify_placement(answer, placed, nonpiece_words, wiring):
    """Hand-solve check: the human has linked each answer atom to a piece (placed =
    [(label, value, positions)]). Derive each piece's mechanism FROM the geometry of
    the link the human made, then hold it to the SAME gate — two-way uniqueness +
    indicator confirmation + every leftover word roled. Returns a dict with the
    derived pieces, the verdict, and any gaps (for enrichment)."""
    n = len(answer)
    problems, pieces = [], []
    cover = [p for _, _, ps in placed for p in ps]
    if sorted(cover) != list(range(1, n + 1)):
        problems.append("answer atoms not each linked exactly once (got %s)"
                        % sorted(cover))
    for label, value, ps in placed:
        ps = sorted(ps)
        seg = "".join(answer[p - 1] for p in ps)
        contiguous = ps == list(range(ps[0], ps[-1] + 1))
        extra = len(value) - len(seg)        # source atoms NOT landed = removed
        mech, removed = None, ""
        if extra < 0:
            problems.append("%s=%s links %d atoms but has only %d letters"
                            % (label, value, len(ps), len(value)))
            continue
        if extra == 0:
            if contiguous and seg == value:
                mech = "identity"
            elif contiguous and seg == value[::-1]:
                mech = "reversed"
            elif contiguous and sorted(seg) == sorted(value):
                mech = "anagram"
            elif not contiguous and seg == value:
                mech = "split"
            elif not contiguous and seg == value[::-1]:
                mech = "split_reversed"
        else:
            # some source atoms removed: value with a contiguous block of length
            # `extra` taken out spells the linked atoms (handles deletion, and the
            # split case = substitution: original split around the new letter).
            for k in range(0, len(value) - extra + 1):
                rest = value[:k] + value[k + extra:]
                if rest == seg:
                    removed = value[k:k + extra]
                    mech = "deletion" if contiguous else "deletion_split"
                    break
                if rest == seg[::-1]:
                    removed = value[k:k + extra]
                    mech = "deletion_reversed" if contiguous else "deletion_split_reversed"
                    break
        if mech is None:
            problems.append("%s=%s does not match the atoms you linked (%s = %s)"
                            % (label, value, ps, seg))
            continue
        pieces.append((label, value, ps, mech, removed))

    need = set()
    for _, _, _, m, _ in pieces:
        need |= _needs(m)
    removed_blocks = [rm for *_, rm in pieces if rm]
    confirmed, missing_ind, unaccounted, named = {}, [], [], []
    have = {}
    for w in nonpiece_words:
        types = set(wiring["indicator_types"](w) or ())
        match = types & need
        if match:
            for t in match:
                have.setdefault(t, w)
        elif wiring["is_link"](w):
            pass
        elif any(rm == (v or "").upper()
                 for rm in removed_blocks for v, _ in wiring["lookup_all"](w)):
            named.append(w)          # spells the removed letters (named deletion source)
        else:
            unaccounted.append(w)
    for t in need:
        if t in have:
            confirmed[t] = have[t]
        else:
            missing_ind.append(t)

    gaps = []
    gaps += ["unaccounted word (mark it as a link or an indicator): %s" % w
             for w in unaccounted]
    gaps += ["no DB indicator present for: %s" % t for t in missing_ind]
    ok = not problems and not gaps
    return {"pieces": pieces, "problems": problems, "confirmed": confirmed,
            "named": named, "gaps": gaps, "ok": ok,
            "verdict": ("PASS — valid atom-map" if ok else
                        ("INVALID — " + "; ".join(problems)) if problems else
                        "PENDING — map valid but " + "; ".join(gaps))}


def build_handsolve_parse(ctx, answer_text, def_phrase, pieces, nonpiece_words, wiring):
    """Turn a validated hand placement into a wfw_model.Parse so it renders on the
    WFW page exactly like an auto-solve. pieces = [(label, value, positions, mech)]."""
    from core.wfw_model import Source, Link, Annotation, Parse
    def _transform(m):
        return "reversed" if "reversed" in m else ("anagram_of" if m == "anagram" else None)

    def _op(m):
        if "deletion" in m:
            return "deletion"
        if "split" in m:
            return "container"
        if m == "reversed":
            return "reversal"
        if m == "anagram":
            return "anagram"
        return "charade"
    used = set()

    def word_atoms(text):
        for t in ctx.clue_tokens:
            if t.kind == "word" and t.text == text and id(t) not in used:
                used.add(id(t))
                return t.atom_ids
        return ()

    def phrase_atoms(phrase):
        ids = ()
        for wd in phrase.split():
            ids = ids + word_atoms(wd)
        return ids

    sources, links = [], []
    for si, (label, value, ps, mech, removed) in enumerate(pieces):
        val_label = value if not removed else "%s-%s" % (value, removed)
        sources.append(Source(clue_atom_ids=phrase_atoms(label), text=label,
                              value=val_label, mechanism="piece"))
        for p in ps:
            links.append(Link(answer_pos=p, source_index=si,
                              operation=_op(mech), transform=_transform(mech)))
    need = set()
    for _, _, _, m, _ in pieces:
        need |= _needs(m)
    annotations = []
    for wtext in nonpiece_words:
        atoms = word_atoms(wtext)
        match = set(wiring["indicator_types"](wtext) or ()) & need
        if match:
            annotations.append(Annotation(clue_atom_ids=atoms, text=wtext,
                                          role="indicator", note="%s indicator" % sorted(match)[0]))
        elif wiring["is_link"](wtext):
            annotations.append(Annotation(clue_atom_ids=atoms, text=wtext,
                                          role="link", note="link word"))
    def_atoms = ()
    for dt in def_phrase.split():
        def_atoms = def_atoms + word_atoms(dt)
    definition = Source(clue_atom_ids=def_atoms, text=def_phrase, value=answer_text,
                        mechanism="definition")
    return Parse(clue_text=ctx.clue_text, answer_text=answer_text, sources=sources,
                 links=links, annotations=annotations, definition=definition,
                 operation=(pieces[0][3] if len(pieces) == 1 else "charade"),
                 solved_by="handsolve", status="pass")


def confirm_indicators(pieces, indicator_words, indicator_types):
    """For each detected non-identity mechanism, confirm a DB indicator legitimises
    it. Returns (ok, evidence). The indicator is consulted ONLY here, after the
    mechanism is already known from the atoms."""
    need = set()
    for _, _, _, mech in pieces:
        if mech == "reversed":
            need.add("reversal")
        elif mech == "split":
            need.add("container")
        elif mech == "anagram":
            need.add("anagram")
    have = {}
    for w in indicator_words:
        for t in (indicator_types(w) or ()):
            have.setdefault(t, []).append(w)
    missing = [t for t in need if t not in have]
    confirmations = {t: have[t][0] for t in need if t in have}
    return (not missing), {"need": sorted(need), "confirmed": confirmations,
                           "missing": missing}
