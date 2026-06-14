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
    pairs = _swap_pairs(ctx.answer_text, pronounce)
    if not pairs:
        return None                                  # cannot pronounce/split the answer
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


def _build(ctx, pairs, split, words, ind_set, is_link, pronounce, synonyms_of):
    """Find the two-word source (clue phrase or its DB synonym) that transposes to the
    answer; classify the rest as links. Returns a Parse, or None if the leftovers do not
    resolve to a clean source + links."""
    region = [p for p in range(len(words)) if p not in ind_set]
    link_pos = [p for p in region if is_link and is_link(words[p].text)]
    content = [p for p in region if p not in link_pos]
    if not content:
        return None
    content_toks = [words[p] for p in content]
    cwords = [t.text for t in content_toks]
    source_phrase = " ".join(cwords)
    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")

    # Find the two source words (each a content word or its single-word synonym) whose
    # transposed onsets sound like the answer, and RECORD which clue word maps to which,
    # so the breakdown can show the derivation (excluded->BARRED, journalist->HACK).
    pieces = None                 # [(tokens, source_value)] — the displayed synonyms
    s1 = s2 = None
    if len(cwords) == 2:
        # Filter each word's synonyms by the answer's target sounds (NO arbitrary cap —
        # the needed synonym can be the 120th, so capping silently loses it). Each side
        # keeps only synonyms that already sound like a transposed half of the answer,
        # then the pair is accepted iff together they form one of the swap pairs.
        t1 = {a for a, _ in pairs}
        t2 = {b for _, b in pairs}
        c1cands = _sound_synonyms(cwords[0], t1, synonyms_of, pronounce)
        c2cands = _sound_synonyms(cwords[1], t2, synonyms_of, pronounce)
        for s1v, k1s in c1cands:
            for s2v, k2s in c2cands:
                if any((a, b) in pairs for a in k1s for b in k2s):
                    pieces = [([content_toks[0]], s1v.upper()),
                              ([content_toks[1]], s2v.upper())]
                    s1, s2 = s1v, s2v
                    break
            if pieces:
                break
    if pieces is None:            # one-word content whose synonym is a two-word phrase
        for syn in [source_phrase] + list(synonyms_of(source_phrase) or []):
            sw = syn.split()
            if len(sw) == 2 and _source_matches(sw[0], sw[1], pairs, pronounce):
                pieces = [(content_toks, ("%s %s" % (sw[0], sw[1])).upper())]
                s1, s2 = sw
                break

    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    ind_toks = [words[i] for i in sorted(ind_set)]

    if pieces is None:
        warnings = ['spoonerism indicator present but no two-word source (the wordplay '
                    'or its synonym) transposes to sound like the answer']
        ind_ann = Annotation(
            clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
            text=" ".join(t.text for t in ind_toks), role="indicator",
            note="spoonerism indicator")
        return Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                     sources=[], links=[], annotations=[ind_ann],
                     definition=definition, operation="spoonerism",
                     solved_by="spoonerism", status="fail", warnings=warnings)

    # The indicator carries the transposition so the breakdown reads
    # "Spoonerism indicator — BARRED HACK -> HARDBACK".
    ind_ann = Annotation(
        clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in ind_toks), role="indicator",
        note="spoonerism: %s %s → %s" % (s1.upper(), s2.upper(), answer))
    sources = [Source(clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                      text=" ".join(t.text for t in toks), value=val,
                      mechanism="synonym", source="db")
               for toks, val in pieces]
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
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing or parse.definition is None:
        parse.status = "fail"
    else:
        parse.status = "pending"
