"""Homophone engine — the whole answer SOUNDS like the wordplay.

Read the way a solver reads it:
  1. A homophone indicator is present -> it is a homophone clue.
  2. The definition is taken off the edge (decided upstream).
  3. What is left is the SOURCE: the word(s) that, read aloud, give the answer --
     directly, or via a synonym. The source may be SEVERAL words, so it is looked
     up as a PHRASE, not one word at a time:
        SITES = sounds like CITES;  "refers to" = cites;  indicator "In speech".
        REST  = sounds like WREST;  "take forcible control" = wrest;  indicator "broadcast".
        LISZT = sounds like LIST;   "register" = list;  indicator "vocal".

So: among the leftover wordplay, find the source phrase whose sound (its own, or a
synonym's) equals the answer, with the indicator accounted and any remaining
leftover words being links. Sound-equality is judged by core.live_db.sounds_alike
(pronunciation dictionary, with the curated homophones table as a supplement).

SPAN-LEVEL provenance (§5.5): the whole answer is the sound of the source phrase, so
every answer letter links to that one source (clue_atom_id=None — meaning-based, not
letter-selection). Isolated per-type engine; definition decided upstream; links from
the link-list only; NO role by elimination. Paired with core/homophone_screen.py.
"""
from core import engine_common
from core.wfw_model import Source, Link, Annotation, Parse


def solve_homophone(ctx, defines, is_link, indicator_types, sounds_alike,
                    synonyms_of, define_fallback=None, is_dbe=None, pronounce=None):
    """Full homophone solve. Returns a Parse (pass / pending) or None (abstain).

    `sounds_alike(word_or_phrase, target) -> bool` judges sound-equality.
    `synonyms_of(word_or_phrase) -> [synonyms]` looks up a phrase as a whole.
    `pronounce(word) -> [phoneme strings]` enables the MULTI-WORD homophone (PISCES =
    pie[Patty] + sees[understands]), where the sound-matcher can't pronounce the joined
    phrase, so per-word pronunciations are concatenated and compared to the answer."""
    from core.definition_engine import find_definitions
    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 3 or indicator_types is None or sounds_alike is None:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    best_pending = None
    best_fail = None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        parse = _try_split(ctx, answer, split, words, is_link, indicator_types,
                           sounds_alike, synonyms_of, pronounce)
        if parse is None:
            continue
        if parse.status == "pass":
            return parse
        if parse.status == "pending":
            if best_pending is None:
                best_pending = parse
        elif best_fail is None:                  # preserved fail-evidence (no source)
            best_fail = parse
    # pass > pending > fail: a near-miss (indicator + definition found, source missing)
    # is preserved as fail-evidence rather than discarded by abstaining.
    return best_pending or best_fail


def _letters(s):
    return "".join(c for c in (s or "").upper() if c.isalpha())


def _source_sound(phrase, answer, sounds_alike, synonyms_of):
    """Does this phrase sound like the answer, directly or via a synonym?
    Returns (kind, sound_source) — the word/phrase actually pronounced as the answer
    (the phrase itself, or its synonym) — or None.

    A homophone needs a DIFFERENT spelling that sounds the same. A sound-source whose
    letters ARE the answer (a word sounds like itself) is not a homophone — it is a
    plain definition (e.g. "school of painting" = GENRE), so it is rejected here."""
    if _letters(phrase) != answer and sounds_alike(phrase, answer):
        return ("direct", phrase)
    for syn in (synonyms_of(phrase) or []):
        if _letters(syn) != answer and sounds_alike(syn, answer):
            return ("synonym", syn)
    return None


def _pron_key(phonemes):
    return " ".join(p.rstrip("012") for p in phonemes)


def _multiword_source(content_toks, answer, pronounce, synonyms_of):
    """MULTI-WORD homophone: each content word -> a synonym; the synonyms read together
    SOUND like the answer (Patty->pie + understands->sees = "pie sees" ~ PISCES). The
    sound-matcher can't pronounce the joined phrase, so per-word pronunciations are
    concatenated and compared to the answer's pronunciation (incl. two-sub-word splits
    for compounds). Two-word form. Returns the spoken phrase (e.g. "pie sees") or None."""
    if len(content_toks) != 2 or pronounce is None:
        return None
    whole = answer.lower()
    akeys = {_pron_key(p.split()) for p in pronounce(whole)}
    for i in range(2, len(whole) - 1):
        for x in pronounce(whole[:i]):
            for y in pronounce(whole[i:]):
                akeys.add(_pron_key(x.split() + y.split()))
    if not akeys:
        return None
    w1, w2 = content_toks
    o1 = ([w1.text] + list(synonyms_of(w1.text) or []))[:60]
    o2 = ([w2.text] + list(synonyms_of(w2.text) or []))[:60]
    for s1 in o1:
        k1 = [_pron_key(p.split()) for p in pronounce(s1)]
        if not k1:
            continue
        for s2 in o2:
            for p2 in pronounce(s2):
                k2 = _pron_key(p2.split())
                if any(a + " " + k2 in akeys for a in k1):
                    cand = "%s %s" % (s1, s2)
                    if _letters(cand) != answer:
                        return cand
    return None


def _try_split(ctx, answer, split, words, is_link, indicator_types, sounds_alike,
               synonyms_of, pronounce=None):
    """In this split's wordplay: gate on a homophone indicator, then look for the
    SOURCE — the longest leftover phrase whose sound (own or via a synonym) is the
    answer, with every other leftover word a link. Prefers a clean PASS."""
    n = len(words)
    if n < 2:                                    # need the source AND >=1 indicator
        return None
    ind_pos = engine_common.find_typed_run(words, list(range(n)), indicator_types,
                                           "homophone", min_length=1)
    if ind_pos is None:
        return None                              # gated: no homophone indicator
    ind_set = set(ind_pos)
    region = [i for i in range(n) if i not in ind_set]   # the source + any links
    if not region:
        return None

    best_pending = None
    # Try contiguous runs of leftover words as the source phrase, LONGEST first (a
    # multi-word source is the whole phrase: "take forcible control"). Whatever is
    # not the source must be a link word.
    for src_idxs in _contiguous_subruns(region):
        phrase = " ".join(words[i].text for i in src_idxs)
        mech = _source_sound(phrase, answer, sounds_alike, synonyms_of)
        if mech is None:
            continue
        leftover = [i for i in region if i not in set(src_idxs)]
        if not _leftover_all_links(words, leftover, is_link):
            continue                             # an unaccounted content word
        parse = _build(ctx, answer, split, words, src_idxs, mech, ind_pos, leftover)
        if parse is None:
            continue
        if parse.status == "pass":
            return parse
        if best_pending is None:
            best_pending = parse
    if best_pending is not None:
        return best_pending
    # MULTI-WORD homophone: the source is two clue words EACH read as a synonym, the two
    # synonyms together sounding like the answer (PISCES = Patty[pie] + understands[sees]).
    content = [i for i in region if not (is_link and is_link(words[i].text))]
    if len(content) == 2:
        cand = _multiword_source([words[i] for i in content], answer, pronounce,
                                 synonyms_of)
        if cand is not None:
            leftover = [i for i in region if i not in set(content)]
            if all(is_link and is_link(words[i].text) for i in leftover):
                parse = _build(ctx, answer, split, words, content, ("synonym", cand),
                               ind_pos, leftover)
                if parse is not None and parse.status in ("pass", "pending"):
                    return parse
    # A homophone INDICATOR is present (DB-typed, found above) but no leftover phrase
    # sounds like the answer — the commonest failure. Preserve that evidence instead of
    # abstaining (returning None would store nothing and lose the near-miss).
    return _build_fail_evidence(ctx, split, words, ind_pos, region)


def _build_fail_evidence(ctx, split, words, ind_pos, region):
    """Preserve the evidence when the homophone indicator is present but no leftover
    phrase — itself or via a synonym — sounds like the answer. A FAIL commits NO roles
    by elimination (feedback-no-role-on-fail): it keeps only the definition (decided
    upstream) and NAMES, in a warning, the present indicator and the leftover source
    phrase whose sound did not match, so the gap can be investigated. No fabricated
    source value, no links."""
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    ind_text = " ".join(words[i].text for i in ind_pos)
    src_text = " ".join(words[i].text for i in region)
    warnings = ['homophone indicator present ("%s") but no wordplay phrase sounds like '
                'the answer, directly or via a synonym — the homophone source is '
                'missing (leftover wordplay: "%s")' % (ind_text, src_text)]
    dbe = dbe_annotation(split)
    annotations = [dbe] if dbe is not None else []
    return Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                 sources=[], links=[], annotations=annotations,
                 definition=definition, operation="homophone",
                 solved_by="homophone", status="fail", warnings=warnings)


def _leftover_all_links(words, leftover, is_link):
    """Every leftover word is a link — checked at PHRASE level over contiguous runs, so a
    multi-word connector the DB types as a link ("to get") is recognised even when a single
    word of it ("get") is not a link on its own. Mirrors dd_engine / charade_deletion."""
    if not is_link:
        return not leftover
    for run in engine_common.contiguous_groups(sorted(leftover)):
        phrase = " ".join(words[i].text for i in run)
        if is_link(phrase) or all(is_link(words[i].text) for i in run):
            continue
        return False
    return True


def _contiguous_subruns(positions):
    """Every contiguous sub-run of the given (sorted) positions, LONGEST first, so the
    fullest source phrase is preferred. positions need not be contiguous overall; runs
    are formed within each contiguous stretch."""
    out = []
    for run in engine_common.contiguous_groups(sorted(positions)):
        L = len(run)
        for size in range(L, 0, -1):
            for s in range(0, L - size + 1):
                out.append(run[s:s + size])
    out.sort(key=len, reverse=True)
    return out


def _build(ctx, answer, split, words, src_idxs, mech, ind_pos, link_pos):
    """Assemble the Parse: the source phrase, span-level links, the indicator, and any
    link words. (Gating + link/leftover checks already done by _try_split.)"""
    kind, sound_source = mech
    src_toks = [words[i] for i in src_idxs]
    transform = 'sounds like "%s"' % sound_source
    source = Source(clue_atom_ids=tuple(aid for t in src_toks for aid in t.atom_ids),
                    text=" ".join(t.text for t in src_toks),
                    value=answer, mechanism="homophone", source="db")
    links = [Link(answer_pos=i + 1, source_index=0, operation="homophone",
                  clue_atom_id=None, transform=transform)
             for i in range(len(answer))]
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    ind_toks = [words[i] for i in ind_pos]
    annotations = [Annotation(
        clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in ind_toks), role="indicator",
        note="homophone indicator")]
    for p in link_pos:
        annotations.append(Annotation(clue_atom_ids=words[p].atom_ids,
                                      text=words[p].text, role="link",
                                      note="link word"))
    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=[source], links=links, annotations=annotations,
                  definition=definition, operation="homophone",
                  solved_by="homophone")
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    """Three-state verdict. pass: every answer letter sourced, every clue word
    accounted, DB indicator (by construction) and DB definition. pending: provisional
    definition. fail: a structural gap (should not occur given _try_split's checks)."""
    warnings = []
    n_letters = sum(1 for a in ctx.answer_atoms if a.kind == "letter")
    if len(parse.links) != n_letters:
        warnings.append("not every answer letter has a source")
    w_unaccounted = engine_common.unaccounted_words_warning(ctx, parse)
    if w_unaccounted:
        warnings.append(w_unaccounted)
    missing = parse.unexplained_words(ctx)
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
