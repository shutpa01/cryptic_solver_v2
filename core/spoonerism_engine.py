"""Spoonerism engine — transpose the leading sounds of the answer's two syllables.

A spoonerism clue (flagged by the Reverend Spooner) defines the answer one way and,
the other way, gives a two-word phrase whose initial sounds, swapped, sound like the
answer: HAIRSHIRT <-> SHARE HURT (commiserate); COLEY <-> LOW KEY (subdued).

Everything is done in PHONEMES (ARPABET via the pronunciations table), which dissolves
the homophone problem: the swapped answer-sounds are matched against the source words'
pronunciations directly, so the un-spellable "SHAIR"/"HIRT" never need looking up — they
match share/hurt by sound. The answer's pronunciation is taken whole when the dictionary
has it (COLEY), else built by splitting the spelling into two known sub-words (HAIR +
SHIRT) whose pronunciations are concatenated.

Answer-driven and gated on a Spooner indicator; the source is a wordplay phrase or its
DB synonym; the rest of the wordplay is links. No per-letter clue provenance (sound
match). Isolated per-type engine; paired with core/spoonerism_screen.
"""

from core import engine_common
from core import spoonerism_indicators
from core.wfw_model import Source, Link, Annotation, Parse

_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
           "IH", "IY", "OW", "OY", "UH", "UW"}


def _key(phonemes):
    """Stress-stripped phoneme key for sound comparison (BAT -> 'B AE T')."""
    return " ".join(ph.rstrip("012") for ph in phonemes)


def _has_vowel(phonemes):
    return any(ph.rstrip("012") in _VOWELS for ph in phonemes)


def _split_onset(phonemes):
    """(onset, rest): leading consonant phonemes, then from the first vowel on."""
    for i, ph in enumerate(phonemes):
        if ph.rstrip("012") in _VOWELS:
            return phonemes[:i], phonemes[i:]
    return phonemes, []


def _spoonerise(p1, p2):
    """Swap the onsets of two phoneme runs: (onset2+rest1, onset1+rest2)."""
    o1, r1 = _split_onset(p1)
    o2, r2 = _split_onset(p2)
    return o2 + r1, o1 + r2


def _answer_pronunciations(answer_text, pronounce):
    """Phoneme sequences for the whole answer: the dictionary pronunciation when it has
    the word/compound, plus every split of the spelling into two known sub-words (so a
    compound the dictionary lacks, like HAIRSHIRT = HAIR + SHIRT, is still pronounced),
    plus an explicit two-word answer split on its space."""
    whole = "".join(c for c in answer_text.lower() if c.isalpha())
    out = []
    out += [p.split() for p in pronounce(whole)]
    for i in range(2, len(whole) - 1):
        pa, pb = pronounce(whole[:i]), pronounce(whole[i:])
        for x in pa:
            for y in pb:
                out.append(x.split() + y.split())
    parts = [w for w in answer_text.split() if any(c.isalpha() for c in w)]
    if len(parts) == 2:
        for x in pronounce(parts[0]):
            for y in pronounce(parts[1]):
                out.append(x.split() + y.split())
    return out


def _swap_pairs(answer_text, pronounce):
    """The set of (sound1, sound2) keys the answer yields when split into two syllables
    and the onsets transposed — every plausible un-spoonerised source sound pair."""
    pairs = set()
    for ph in _answer_pronunciations(answer_text, pronounce):
        for k in range(1, len(ph)):
            p1, p2 = ph[:k], ph[k:]
            if not _has_vowel(p1) or not _has_vowel(p2):
                continue
            sw1, sw2 = _spoonerise(p1, p2)
            pairs.add((_key(sw1), _key(sw2)))
    return pairs


def solve_spoonerism(ctx, defines, is_link, pronounce, synonyms_of,
                     define_fallback=None, is_dbe=None):
    """Full spoonerism solve. Abstains (None) unless a Spooner indicator is present and
    a two-word source (a wordplay phrase or its DB synonym) transposes to sound like the
    answer. Returns a Parse (pass / pending / fail)."""
    from core.definition_engine import find_definitions
    # Phoneme swap pairs when the answer is pronounceable (handles homophone sources like
    # barred->hard); may be EMPTY for answers the dictionary lacks (SPARKLER) — the letter
    # fallback in _build covers those, so we never bail here. A Spooner clue is never
    # abstained: _build always returns at least a fail carrying the indicator.
    pairs = _swap_pairs(ctx.answer_text, pronounce)
    # Answer-driven definition: try EVERY edge window (no AI fallback — its single greedy
    # guess can swallow the spoonerism source into the definition).
    splits = list(find_definitions(ctx, defines, is_dbe=is_dbe))
    if not splits:
        return None
    best, best_key = None, None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        ind_pos = spoonerism_indicators.find_indicator(words)
        if ind_pos is None:
            continue                                 # gated: Spooner not named here
        parse = _build(ctx, pairs, split, words, set(ind_pos), is_link,
                       pronounce, synonyms_of)
        if parse is None:
            continue
        key = (0 if parse.status == "pass" else 1 if parse.status == "pending" else 2,
               len(split.phrase.split()))
        if best_key is None or key < best_key:
            best, best_key = parse, key
    return best


def _source_matches(s1, s2, pairs, pronounce):
    """True if source words s1,s2 (real words) sound like one of the answer's swap pairs."""
    k1 = {_key(p.split()) for p in pronounce(s1)}
    k2 = {_key(p.split()) for p in pronounce(s2)}
    return any((a, b) in pairs for a in k1 for b in k2)


def _sound_synonyms(word, targets, synonyms_of, pronounce):
    """[(synonym, [matching_keys])] for the word itself and its SINGLE-WORD synonyms
    whose pronunciation key is in `targets` (a set of transposed answer-half sounds).
    No cap — every synonym is considered, only those that sound right are kept."""
    out = []
    for s in [word] + [x for x in (synonyms_of(word) or []) if len(x.split()) == 1]:
        hit = [k for k in {_key(p.split()) for p in pronounce(s)} if k in targets]
        if hit:
            out.append((s, hit))
    return out


_LVOWELS = frozenset("AEIOU")


def _onset_rest_letters(s):
    """(onset, rest) by LETTERS: the leading consonants, then from the first vowel on."""
    s = "".join(c for c in s.upper() if c.isalpha())
    i = 0
    while i < len(s) and s[i] not in _LVOWELS:
        i += 1
    return s[:i], s[i:]


def _skel(s):
    """Consonant skeleton (vowels dropped) — the letter-level homophone proxy."""
    return "".join(c for c in s.upper() if c.isalpha() and c not in _LVOWELS)


def _letter_splits(answer):
    """[(src1, src2)] — split the answer LETTERS at each point and transpose the leading
    consonant clusters (SPARK|LER -> LARK + SPER). Needs no pronunciation, so it works for
    answers the dictionary lacks. Both halves must have an onset consonant and a vowel."""
    al = "".join(c for c in answer.upper() if c.isalpha())
    out = []
    for k in range(1, len(al)):
        o1, r1 = _onset_rest_letters(al[:k])
        o2, r2 = _onset_rest_letters(al[k:])
        if o1 and r1 and o2 and r2:
            out.append((o2 + r1, o1 + r2))
    return out


def _letter_match(cand, word):
    """`word` equals the transposed source string `cand` up to homophone VOWELS: same
    consonant skeleton and same first letter. Length is NOT required equal — the vowels
    are exactly the homophone-variable part (sped ~ SPEAD, here ~ HEAR), so forcing equal
    length would reject genuine sound matches. The consonant skeleton + first letter keep
    it tight."""
    w = "".join(c for c in word.upper() if c.isalpha())
    return bool(w) and w[:1] == cand[:1] and _skel(w) == _skel(cand)


def _grow_definition(ctx, def_atom_ids, role_atom_ids):
    """Fold adjacent words that are grammatical DEPENDENTS of the definition span into
    the definition, so a noun phrase like "little gem" (little is an amod of gem) stays
    one definition. Walks outward from the def edge and STOPS at the first word that is a
    role word (source / indicator / link) or not a dependent — so it can never run into
    the wordplay. Returns the grown set of clue-word atom_ids."""
    from core import grammar
    allwords = [t for t in ctx.clue_tokens if t.kind == "word"]
    try:
        doc = grammar._parse(" ".join(t.text for t in allwords))
    except Exception:
        return set(def_atom_ids)
    if len(doc) != len(allwords):
        return set(def_atom_ids)
    defset, roleset = set(def_atom_ids), set(role_atom_ids)
    atoms = [set(t.atom_ids) for t in allwords]
    def_idx = {i for i, a in enumerate(atoms) if a & defset}
    role_idx = {i for i, a in enumerate(atoms) if a & roleset}
    if not def_idx:
        return set(def_atom_ids)
    keep = set(def_idx)
    changed = True
    while changed:
        changed = False
        lo, hi = min(keep), max(keep)
        for adj in (lo - 1, hi + 1):
            if 0 <= adj < len(doc) and adj not in keep and adj not in role_idx \
                    and doc[adj].head.i in keep:
                keep.add(adj)
                changed = True
    out = set()
    for i in keep:
        out |= atoms[i]
    return out


def _region_spans(region, maxlen=3):
    """Contiguous runs of region word positions, up to `maxlen` words — so a source can be
    a multi-word phrase ("coastal city"), not just a single word."""
    rs = sorted(region)
    out = []
    for a in range(len(rs)):
        for b in range(a, min(a + maxlen, len(rs))):
            if rs[b] - rs[a] == b - a:                 # positions consecutive
                out.append((rs[a], rs[b]))             # inclusive
    return out


def _build(ctx, pairs, split, words, ind_set, is_link, pronounce, synonyms_of):
    """Answer-driven. We KNOW the answer, so we don't parse clue structure — the answer's
    transposed sounds/letters tell us the source, and we find the clue SPAN(s) whose
    synonyms supply it, in any order. A source unit can be a multi-word phrase ("coastal
    city" -> HULL). PHONEME match first (handles homophone sources like barred->hard); a
    LETTER split+transpose fallback covers answers the dictionary can't pronounce
    (SPARKLER->larkspur). Connecting words are links. A Spooner clue NEVER abstains.
    Returns a Parse (pass / pending / fail)."""
    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    region = [p for p in range(len(words)) if p not in ind_set]
    spans = _region_spans(region)

    def _phrase(s, e):
        return " ".join(words[k].text for k in range(s, e + 1))

    def _span_syns(s, e):
        """The span's own letters (if one word) plus its single-word synonyms."""
        ph = _phrase(s, e)
        return ([ph] if len(ph.split()) == 1 else []) + \
               [x for x in (synonyms_of(ph) or []) if len(x.split()) == 1]

    def _overlap(a, b):
        return not (a[1] < b[0] or b[1] < a[0])

    def _unaccounted(span_positions):
        """Region words not covered by the source spans and not in the link list — a
        clean parse leaves none, so prefer the assignment that minimises this (covers
        "moved fast" rather than leaving "fast" over)."""
        return sum(1 for p in region if p not in span_positions
                   and not (is_link and is_link(words[p].text)))

    pieces = None              # [((s, e), source_value)] -> breakdown rows
    detail = None              # "SOURCE -> ANSWER" for the indicator row
    literal_spans = set()      # span(s) shown as a literal (the fixed middle "A")
    tall = {x for ab in pairs for x in ab}

    # 1) PHONEME: two disjoint spans whose single-word synonyms' sounds form a swap pair.
    sound = {}                 # (s,e) -> [(value, [keys])]
    for sp in spans:
        lst = []
        for v in _span_syns(*sp):
            ks = [k for k in {_key(p.split()) for p in pronounce(v)} if k in tall]
            if ks:
                lst.append((v, ks))
        if lst:
            sound[sp] = lst
    best = None        # (key, pieces, detail) — prefer fewest unaccounted, then coverage
    for A in sound:
        for B in sound:
            if A == B or _overlap(A, B):
                continue
            mv = next(((v1, v2) for v1, k1s in sound[A] for v2, k2s in sound[B]
                       if any((a, b) in pairs for a in k1s for b in k2s)), None)
            if mv:
                sp = set(range(A[0], A[1] + 1)) | set(range(B[0], B[1] + 1))
                key = (_unaccounted(sp), -len(sp))
                if best is None or key < best[0]:
                    best = (key, [(A, mv[0].upper()), (B, mv[1].upper())],
                            "%s %s → %s" % (mv[0].upper(), mv[1].upper(), answer))
    if best is not None:
        pieces, detail = best[1], best[2]

    # 2) PHONEME phrase: one span whose two-word synonym transposes (commiserate ->
    #    "share hurt").
    if pieces is None:
        for sp in spans:
            for syn in (synonyms_of(_phrase(*sp)) or []):
                sw = syn.split()
                if len(sw) == 2 and _source_matches(sw[0], sw[1], pairs, pronounce):
                    pieces = [(sp, syn.upper())]
                    detail = "%s → %s" % (syn.upper(), answer)
                    break
            if pieces:
                break

    # 3) LETTER fallback: split the answer letters, transpose leading consonants, match a
    #    source by consonant skeleton (answers the pronunciation dictionary lacks).
    if pieces is None:
        for src1, src2 in _letter_splits(answer):
            whole = src1 + src2
            for sp in spans:                          # single-source (-> larkspur)
                hit = next((v for v in _span_syns(*sp) if _letter_match(whole, v)), None)
                if hit:
                    pieces = [(sp, hit.upper())]
                    detail = "%s → %s" % (hit.upper(), answer)
                    break
            if pieces:
                break
            m1 = {sp: v for sp in spans
                  if (v := next((x for x in _span_syns(*sp)
                                 if _letter_match(src1, x)), None))}
            m2 = {sp: v for sp in spans
                  if (v := next((x for x in _span_syns(*sp)
                                 if _letter_match(src2, x)), None))}
            best = None
            for A in m1:
                for B in m2:
                    if A == B or _overlap(A, B):
                        continue
                    sp = set(range(A[0], A[1] + 1)) | set(range(B[0], B[1] + 1))
                    key = (_unaccounted(sp), -len(sp))
                    if best is None or key < best[0]:
                        best = (key, [(A, m1[A].upper()), (B, m2[B].upper())],
                                "%s %s → %s" % (m1[A].upper(), m2[B].upper(), answer))
            if best is not None:
                pieces, detail = best[1], best[2]
                break

    # 4) FIXED-MIDDLE "A": the answer is P1 + "A" + P2 and the onsets of P1/P2 swap around
    #    a fixed middle "A" (a common form: BUST·A·GUT <-> GUST·A·BUTT). The middle "a" is a
    #    literal supplied by a clue "a"; the two outer sources match spans either side.
    if pieces is None:
        a_words = [p for p in region
                   if "".join(c for c in words[p].text.lower() if c.isalpha()) == "a"]
        for k in range(1, len(answer) - 1):
            if answer[k] != "A":
                continue
            o1, r1 = _onset_rest_letters(answer[:k])
            o2, r2 = _onset_rest_letters(answer[k + 1:])
            if not (o1 and r1 and o2 and r2):
                continue
            s1, s2 = o2 + r1, o1 + r2
            m1 = {sp: v for sp in spans
                  if (v := next((x for x in _span_syns(*sp) if _letter_match(s1, x)), None))}
            m2 = {sp: v for sp in spans
                  if (v := next((x for x in _span_syns(*sp) if _letter_match(s2, x)), None))}
            for A in m1:
                for B in m2:
                    if A == B or _overlap(A, B):
                        continue
                    ap = next((p for p in a_words
                               if not (A[0] <= p <= A[1]) and not (B[0] <= p <= B[1])), None)
                    if ap is None:
                        continue
                    pieces = [(A, m1[A].upper()), ((ap, ap), "A"), (B, m2[B].upper())]
                    literal_spans = {(ap, ap)}
                    detail = "%s A %s → %s" % (m1[A].upper(), m2[B].upper(), answer)
                    break
                if pieces:
                    break
            if pieces:
                break

    ind_toks = [words[k] for k in sorted(ind_set)]
    if pieces is None:                                # never abstain on a Spooner clue
        return Parse(
            clue_text=ctx.clue_text, answer_text=ctx.answer_text, sources=[], links=[],
            annotations=[Annotation(
                clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
                text=" ".join(t.text for t in ind_toks), role="indicator",
                note="spoonerism indicator")],
            definition=Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                              value=ctx.answer_text, mechanism="definition",
                              source=split.source),
            operation="spoonerism", solved_by="spoonerism", status="fail",
            warnings=['spoonerism indicator present but no source in the clue transposes '
                      'to sound like the answer'])

    pieces.sort(key=lambda pv: pv[0][0])             # display in clue order
    src_pos = {k for (s, e), _ in pieces for k in range(s, e + 1)}
    link_pos = [p for p in region
                if p not in src_pos and is_link and is_link(words[p].text)]

    # Grow the definition to absorb an adjacent grammatically-bound modifier (little ->
    # "little gem"), so it is part of the definition, not relabelled a link by elimination.
    role_atoms = set()
    for k in src_pos:
        role_atoms.update(words[k].atom_ids)
    for t in ind_toks:
        role_atoms.update(t.atom_ids)
    for p in link_pos:
        role_atoms.update(words[p].atom_ids)
    grown = _grow_definition(ctx, split.def_atom_ids, role_atoms)
    allwords = [t for t in ctx.clue_tokens if t.kind == "word"]
    def_words = [t for t in allwords if set(t.atom_ids) & grown]
    definition = Source(
        clue_atom_ids=tuple(aid for t in def_words for aid in t.atom_ids),
        text=" ".join(t.text for t in def_words), value=ctx.answer_text,
        mechanism="definition", source=split.source)

    ind_ann = Annotation(
        clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in ind_toks), role="indicator",
        note="spoonerism: %s" % detail)
    sources = [Source(
        clue_atom_ids=tuple(aid for k in range(s, e + 1) for aid in words[k].atom_ids),
        text=_phrase(s, e), value=v,
        mechanism="raw" if (s, e) in literal_spans else "synonym", source="db")
        for (s, e), v in pieces]
    annotations = [ind_ann]
    for p in link_pos:
        annotations.append(Annotation(clue_atom_ids=words[p].atom_ids,
                                      text=words[p].text, role="link",
                                      note="link word"))
    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=[], annotations=annotations,
                  definition=definition, operation="spoonerism",
                  solved_by="spoonerism")
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    """pass: source found + every clue word accounted + DB definition. pending:
    provisional (residue-edge) definition. fail: unaccounted word / no definition."""
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
