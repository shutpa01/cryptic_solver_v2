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
    """`word` equals the transposed source string `cand` up to homophone vowels: same
    consonant skeleton, same first letter, same length."""
    w = "".join(c for c in word.upper() if c.isalpha())
    return len(w) == len(cand) and w[:1] == cand[:1] and _skel(w) == _skel(cand)


def _build(ctx, pairs, split, words, ind_set, is_link, pronounce, synonyms_of):
    """Answer-driven. We KNOW the answer, so we don't parse clue structure — the answer's
    transposed sounds/letters tell us the source, and we find the clue word(s) whose
    synonyms supply it, in any order. PHONEME match first (handles homophone sources like
    barred->hard); a LETTER split+transpose fallback covers answers the dictionary can't
    pronounce (SPARKLER->larkspur). Connecting words are links. A Spooner clue NEVER
    abstains: with no source found it still returns a fail carrying the indicator, so no
    other engine can mislabel it. Returns a Parse (pass / pending / fail)."""
    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    region = [p for p in range(len(words)) if p not in ind_set]

    def _single_word_opts(p):
        return [words[p].text] + [x for x in (synonyms_of(words[p].text) or [])
                                  if len(x.split()) == 1]

    pieces = None              # [(clue_word_pos, source_value)] -> breakdown rows
    detail = None              # "SOURCE -> ANSWER" for the indicator row

    # 1) PHONEME: two region words whose synonyms' sounds form a swap pair (either order).
    tall = {x for ab in pairs for x in ab}
    cand = {p: c for p in region
            if (c := _sound_synonyms(words[p].text, tall, synonyms_of, pronounce))}
    ks = list(cand)
    for i in ks:
        for j in ks:
            if i == j:
                continue
            for vi, kis in cand[i]:
                for vj, kjs in cand[j]:
                    if any((a, b) in pairs for a in kis for b in kjs):
                        pieces = sorted([(i, vi.upper()), (j, vj.upper())])
                        detail = "%s %s → %s" % (vi.upper(), vj.upper(), answer)
                        break
                if pieces:
                    break
            if pieces:
                break
        if pieces:
            break

    # 2) PHONEME phrase: one region word whose two-word synonym transposes (commiserate
    #    -> "share hurt").
    if pieces is None:
        for p in region:
            for syn in (synonyms_of(words[p].text) or []):
                sw = syn.split()
                if len(sw) == 2 and _source_matches(sw[0], sw[1], pairs, pronounce):
                    pieces = [(p, syn.upper())]
                    detail = "%s → %s" % (syn.upper(), answer)
                    break
            if pieces:
                break

    # 3) LETTER fallback: split the answer letters, transpose leading consonants, match a
    #    source by consonant skeleton (answers the pronunciation dictionary lacks).
    if pieces is None:
        for src1, src2 in _letter_splits(answer):
            whole = src1 + src2
            for p in region:                          # single-word source (-> larkspur)
                hit = next((s for s in _single_word_opts(p)
                            if _letter_match(whole, s)), None)
                if hit:
                    pieces = [(p, hit.upper())]
                    detail = "%s → %s" % (hit.upper(), answer)
                    break
            if pieces:
                break
            half = {p: (next((s for s in _single_word_opts(p) if _letter_match(src1, s)),
                             None),
                        next((s for s in _single_word_opts(p) if _letter_match(src2, s)),
                             None)) for p in region}
            for i in region:                          # two single words (either order)
                for j in region:
                    if i != j and half[i][0] and half[j][1]:
                        pieces = sorted([(i, half[i][0].upper()), (j, half[j][1].upper())])
                        detail = "%s %s → %s" % (half[i][0].upper(), half[j][1].upper(),
                                                 answer)
                        break
                if pieces:
                    break
            if pieces:
                break

    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    ind_toks = [words[k] for k in sorted(ind_set)]

    if pieces is None:                                # never abstain on a Spooner clue
        return Parse(
            clue_text=ctx.clue_text, answer_text=ctx.answer_text, sources=[], links=[],
            annotations=[Annotation(
                clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
                text=" ".join(t.text for t in ind_toks), role="indicator",
                note="spoonerism indicator")],
            definition=definition, operation="spoonerism", solved_by="spoonerism",
            status="fail",
            warnings=['spoonerism indicator present but no source in the clue transposes '
                      'to sound like the answer'])

    src_pos = {p for p, _ in pieces}
    ind_ann = Annotation(
        clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in ind_toks), role="indicator",
        note="spoonerism: %s" % detail)
    sources = [Source(clue_atom_ids=words[p].atom_ids, text=words[p].text, value=v,
                      mechanism="synonym", source="db") for p, v in pieces]
    annotations = [ind_ann]
    for p in region:
        if p not in src_pos:
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
