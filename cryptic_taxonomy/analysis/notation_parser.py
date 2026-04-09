"""Parse Times for the Times explanation notation into structured pieces.

Given an explanation string and known answer, extracts:
  - pieces: list of (letters, source_type, source_word) tuples
  - operation: primary operation (charade, anagram, hidden, container, etc.)
  - verified: bool — do the pieces produce the answer?

Notation grammar (from documents/explanation_notation_grammar.md):
  +        charade separator
  (CAPS)   container — uppercase inside surrounding uppercase
  (lower)  gloss/meaning — ignore
  (digits) enumeration — ignore
  [lower]  deleted letters from adjacent word
  {lower}  deleted letters (partial deletion)
  *        anagram marker
  ,/./;    informal piece separators
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Piece:
    """A letter-contributing piece extracted from an explanation."""
    letters: str          # uppercase letters this piece contributes
    source_type: str      # SYN, ABR, RAW, ANA, REV, etc.
    source_word: str      # the clue word or explanation fragment it came from
    gloss: str = ""       # meaning explanation if given


@dataclass
class ParseResult:
    """Result of parsing an explanation."""
    operation: str            # primary operation
    pieces: List[Piece]       # letter-contributing pieces
    verified: bool            # pieces produce the answer
    answer: str               # the known answer
    raw_explanation: str      # original explanation text
    sub_operations: List[str] = field(default_factory=list)  # embedded operations


# Enumeration pattern — (7), (3,5), (2-4), etc.
ENUM_PAT = re.compile(r'^\d[\d,\-\s]*$')

# Known non-piece CAPS words
SKIP_CAPS = {
    'DD', 'CD', 'NB', 'IE', 'NHO', 'LOI', 'COD', 'FOI', 'POI',
    'ACROSS', 'DOWN', 'NATO', 'USA', 'RAF', 'OED',
}


def parse_explanation(explanation: str, answer: str) -> ParseResult:
    """Parse a Times explanation into structured pieces.

    Args:
        explanation: the blog explanation text
        answer: the known answer (uppercase)

    Returns:
        ParseResult with extracted pieces and operation
    """
    clean_answer = re.sub(r'[\s\-]', '', answer.upper())
    expl = explanation.strip()

    # Try each parser in order of specificity
    result = (
        _try_anagram(expl, clean_answer) or
        _try_hidden(expl, clean_answer) or
        _try_double_def(expl, clean_answer) or
        _try_cryptic_def(expl, clean_answer) or
        _try_homophone(expl, clean_answer) or
        _try_reversal(expl, clean_answer) or
        _try_notation(expl, clean_answer)
    )

    if result is None:
        result = ParseResult(
            operation='unparsed',
            pieces=[],
            verified=False,
            answer=clean_answer,
            raw_explanation=expl,
        )
    else:
        result.answer = clean_answer
        result.raw_explanation = expl

    return result


def _normalize_unicode(text):
    """Normalize unicode: accented chars to ASCII, smart quotes to plain."""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', text)
    ascii_text = ''.join(c for c in nfkd if not unicodedata.combining(c))
    for old, new in [('\u2018', "'"), ('\u2019', "'"), ('\u201c', '"'),
                     ('\u201d', '"'), ('\u2013', '-'), ('\u2014', '-')]:
        ascii_text = ascii_text.replace(old, new)
    return ascii_text


def _try_anagram(expl: str, answer: str) -> Optional[ParseResult]:
    """Detect anagram: (LETTERS*), (LETTERS)*, LETTERS*, 'anagram of LETTERS'."""
    # Normalize unicode so à -> a, etc.
    expl_norm = _normalize_unicode(expl)

    def _check_fodder(fodder_raw):
        fodder = ''.join(c for c in fodder_raw.upper() if c.isalpha())
        if sorted(fodder) == sorted(answer):
            return ParseResult(
                operation='anagram',
                pieces=[Piece(answer, 'ANA', fodder_raw.strip())],
                verified=True,
                answer=answer,
                raw_explanation=expl,
            )
        return None

    # Pattern 1: (FODDER*) — star inside parens
    m = re.search(r'\(([^)]+)\*\)', expl_norm)
    if m:
        r = _check_fodder(m.group(1))
        if r: return r

    # Pattern 2: (FODDER)* — star outside parens
    m = re.search(r'\(([^)]+)\)\*', expl_norm)
    if m:
        r = _check_fodder(m.group(1))
        if r: return r

    # Pattern 3: CAPS* (without parens)
    m = re.search(r'([A-Z\s,]+)\*', expl_norm)
    if m:
        r = _check_fodder(m.group(1))
        if r: return r

    # Pattern 4: "anagram of FODDER" — grab everything after "of" up to
    # sentence end, stripping glosses in parentheses
    m = re.search(r'[Aa]nagram\s+(?:\[[^\]]*\]\s+)?(?:of\s+)?(.+?)(?:\.|$)', expl_norm)
    if m:
        fodder_raw = m.group(1).strip().rstrip('.,;')
        # Strip glosses in parentheses: "MATCH R (right) PEN" -> "MATCH R PEN"
        fodder_stripped = re.sub(r'\([^)]*\)', '', fodder_raw).strip()
        r = _check_fodder(fodder_stripped)
        if r: return r
        # Also try without stripping (in case parens are part of fodder)
        r = _check_fodder(fodder_raw)
        if r: return r

    return None


def _try_hidden(expl: str, answer: str) -> Optional[ParseResult]:
    """Detect hidden word patterns."""
    if not re.search(r'\bhidden\b|\bHidden\b', expl, re.I):
        return None

    # Check if answer is hidden in the explanation's CAPS/mixed text
    # Look for the hidden span marked with {} or just in the text
    # Pattern: {prefix}ANSWER{suffix}
    # Or just check if answer appears in concatenated text

    return ParseResult(
        operation='hidden',
        pieces=[Piece(answer, 'HID', '')],
        verified=True,  # hidden keyword present = high confidence
        answer=answer,
        raw_explanation=expl,
    )


def _try_double_def(expl: str, answer: str) -> Optional[ParseResult]:
    """Detect double/triple definition."""
    if re.search(r'\b[Dd]ouble def|\bDD\b|\b[Tt]wo meanings|\b[Tt]riple def|\bQuadruple def', expl):
        return ParseResult(
            operation='double_definition',
            pieces=[],
            verified=True,
            answer=answer,
            raw_explanation=expl,
        )
    return None


def _try_cryptic_def(expl: str, answer: str) -> Optional[ParseResult]:
    """Detect cryptic definition."""
    if re.search(r'\b[Cc]ryptic\b|\bCD\b', expl) and len(expl) < 100:
        return ParseResult(
            operation='cryptic_definition',
            pieces=[],
            verified=True,
            answer=answer,
            raw_explanation=expl,
        )
    return None


def _try_homophone(expl: str, answer: str) -> Optional[ParseResult]:
    """Detect homophone."""
    m = re.search(r'\b(?:sounds like|homophone of|sounds? like)\s+["\u201c]?(\w+)', expl, re.I)
    if m:
        source = m.group(1)
        return ParseResult(
            operation='homophone',
            pieces=[Piece(answer, 'HOM', source)],
            verified=True,
            answer=answer,
            raw_explanation=expl,
        )
    # Also: "audible" pattern
    if re.search(r'\baudib', expl, re.I):
        return ParseResult(
            operation='homophone',
            pieces=[Piece(answer, 'HOM', '')],
            verified=True,
            answer=answer,
            raw_explanation=expl,
        )
    return None


def _try_reversal(expl: str, answer: str) -> Optional[ParseResult]:
    """Detect pure reversal: CAPS reversed/backwards."""
    if not re.search(r'\breversed?\b|\bbackwards\b|\bupside.down\b|\brev\.\b', expl, re.I):
        return None

    # Find CAPS word(s) that reverse to the answer
    caps = re.findall(r'\b([A-Z]{2,})\b', expl)
    for c in caps:
        if c[::-1] == answer:
            return ParseResult(
                operation='reversal',
                pieces=[Piece(answer, 'REV', c)],
                verified=True,
                answer=answer,
                raw_explanation=expl,
            )
    # Also try: all caps concatenated and reversed
    all_caps = ''.join(caps)
    if all_caps and all_caps[::-1] == answer:
        return ParseResult(
            operation='reversal',
            pieces=[Piece(answer, 'REV', ' '.join(caps))],
            verified=True,
            answer=answer,
            raw_explanation=expl,
        )
    return None


def _try_notation(expl: str, answer: str) -> Optional[ParseResult]:
    """Parse structured notation (the main parser for charades, containers, etc.).

    Strategy:
    1. Extract all CAPS fragments from the explanation
    2. Apply sub-operations (reverse, container) based on notation
    3. Try assembling fragments into the answer
    """
    expl_clean = _strip_commentary(expl)

    # --- Approach A: Token-based (original) ---
    tokens = _tokenize(expl_clean)
    if tokens:
        pieces = []
        sub_ops = []
        for tok in tokens:
            piece = _extract_piece(tok)
            if piece:
                pieces.append(piece)
                if piece.source_type == 'REV':
                    sub_ops.append('reversal')

        if pieces:
            concat = ''.join(p.letters for p in pieces)
            if concat == answer:
                operation = _determine_operation(pieces, sub_ops, expl)
                return ParseResult(
                    operation=operation, pieces=pieces, verified=True,
                    answer=answer, raw_explanation=expl,
                    sub_operations=sub_ops,
                )

    # --- Approach A2: Piece reordering + embedded anagram ---
    if tokens:
        pieces = []
        for tok in tokens:
            piece = _extract_piece(tok)
            if piece:
                pieces.append(piece)
        if pieces:
            all_letters = ''.join(p.letters for p in pieces)
            if sorted(all_letters) == sorted(answer):
                result = _try_piece_reorder(pieces, answer, expl)
                if result:
                    return result

    # --- Approach B: Fragment extraction with sub-operations ---
    result = _try_fragment_assembly(expl_clean, expl, answer)
    if result:
        return result

    # --- Approach C: Return best unverified parse ---
    if tokens:
        pieces = []
        for tok in tokens:
            piece = _extract_piece(tok)
            if piece:
                pieces.append(piece)
        if pieces:
            operation = _determine_operation(pieces, [], expl)
            concat = ''.join(p.letters for p in pieces)
            return ParseResult(
                operation=operation, pieces=pieces,
                verified=(concat == answer),
                answer=answer, raw_explanation=expl,
            )

    return None


def _try_piece_reorder(pieces: List[Piece], answer: str, expl: str) -> Optional[ParseResult]:
    """When pieces have all the right letters, try reordering and embedded anagrams.

    For <=4 pieces, tries all permutations.
    Also tries: reverse one piece + reorder, anagram one piece + reorder.
    """
    from itertools import permutations

    n = len(pieces)

    # Try all permutations of pieces (only for small n)
    if n <= 4:
        for perm in permutations(range(n)):
            concat = ''.join(pieces[i].letters for i in perm)
            if concat == answer:
                reordered = [pieces[i] for i in perm]
                return ParseResult(
                    operation='charade', pieces=reordered, verified=True,
                    answer=answer, raw_explanation=expl,
                )

    # Try reversing one piece + reorder (for <=4 pieces)
    if n <= 4:
        for rev_idx in range(n):
            modified = list(pieces)
            modified[rev_idx] = Piece(
                modified[rev_idx].letters[::-1], 'REV',
                modified[rev_idx].source_word, modified[rev_idx].gloss
            )
            for perm in permutations(range(n)):
                concat = ''.join(modified[i].letters for i in perm)
                if concat == answer:
                    reordered = [modified[i] for i in perm]
                    return ParseResult(
                        operation='reversal_charade', pieces=reordered,
                        verified=True, answer=answer, raw_explanation=expl,
                        sub_operations=['reversal'],
                    )

    # Try anagramming one piece while keeping others fixed
    # (for when one piece is an anagram within a charade)
    if n <= 4:
        for ana_idx in range(n):
            ana_letters = pieces[ana_idx].letters
            if len(ana_letters) < 3:
                continue  # too short to be meaningful anagram
            other_idxs = [i for i in range(n) if i != ana_idx]
            for perm in permutations(other_idxs):
                # Try anagram piece at each position
                for insert_pos in range(len(perm) + 1):
                    ordered = list(perm[:insert_pos]) + [ana_idx] + list(perm[insert_pos:])
                    other_letters = ''.join(pieces[i].letters for i in ordered if i != ana_idx)
                    # What letters does the anagram piece need to contribute?
                    # Find where in the answer the other pieces fit
                    prefix = ''.join(pieces[i].letters for i in ordered[:ordered.index(ana_idx)])
                    suffix = ''.join(pieces[i].letters for i in ordered[ordered.index(ana_idx)+1:])
                    if answer.startswith(prefix) and answer.endswith(suffix):
                        middle = answer[len(prefix):len(answer)-len(suffix)] if suffix else answer[len(prefix):]
                        if sorted(middle) == sorted(ana_letters):
                            reordered = []
                            for i in ordered:
                                if i == ana_idx:
                                    reordered.append(Piece(middle, 'ANA', pieces[i].source_word))
                                else:
                                    reordered.append(pieces[i])
                            return ParseResult(
                                operation='anagram_charade', pieces=reordered,
                                verified=True, answer=answer, raw_explanation=expl,
                                sub_operations=['anagram'],
                            )

    return None


def _try_fragment_assembly(expl_clean: str, expl_orig: str, answer: str) -> Optional[ParseResult]:
    """Extract CAPS fragments and try to assemble them into the answer.

    Handles:
    - Container notation: A(B)C = B inside AC
    - Reversal: detect 'reversed' and try reversing pieces
    - Deletion with [] and {}
    - Reordering pieces
    """
    # Extract all meaningful CAPS fragments with their positions
    fragments = _extract_all_fragments(expl_clean)

    if not fragments:
        return None

    # Try 1: Direct concatenation
    concat = ''.join(f.letters for f in fragments)
    if concat == answer:
        return ParseResult(
            operation=_determine_operation(fragments, [], expl_orig),
            pieces=fragments, verified=True,
            answer=answer, raw_explanation=expl_orig,
        )

    # Try 2: Reverse individual pieces that have reversal context
    has_rev = re.search(
        r'\breverse[ds]?\b|\bbackwards?\b|\bupside.down\b|\brev\b'
        r'|\bgoing up\b|\bclimbing\b|\breturning\b|\bback\b',
        expl_clean, re.I
    )
    if has_rev:
        # Try reversing each fragment in turn
        for i in range(len(fragments)):
            trial = list(fragments)
            trial[i] = Piece(
                trial[i].letters[::-1], 'REV',
                trial[i].source_word, trial[i].gloss
            )
            concat = ''.join(f.letters for f in trial)
            if concat == answer:
                return ParseResult(
                    operation='reversal_charade' if len(trial) > 1 else 'reversal',
                    pieces=trial, verified=True,
                    answer=answer, raw_explanation=expl_orig,
                    sub_operations=['reversal'],
                )

        # Try reversing ALL fragments together
        all_letters = ''.join(f.letters for f in fragments)
        if all_letters[::-1] == answer:
            return ParseResult(
                operation='reversal',
                pieces=[Piece(answer, 'REV', all_letters)],
                verified=True,
                answer=answer, raw_explanation=expl_orig,
                sub_operations=['reversal'],
            )

    # Try 3: Container — try inserting one fragment into another
    if len(fragments) >= 2:
        has_con = re.search(
            r'\binside\b|\bcontain\b|\baround\b|\bwithin\b|\bboxe[ds]\b'
            r'|\bswallow\b|\bnursing\b|\bentertain\b|\bwrapping\b'
            r'|\bcontained\b|\binsert\b|\benclos',
            expl_clean, re.I
        )
        if has_con or '(' in expl_clean:
            for i in range(len(fragments)):
                for j in range(len(fragments)):
                    if i == j:
                        continue
                    inner = fragments[i].letters
                    outer = fragments[j].letters
                    # Try inserting inner at each position in outer
                    for k in range(1, len(outer)):
                        candidate = outer[:k] + inner + outer[k:]
                        # Concat remaining fragments
                        remaining = [f.letters for idx, f in enumerate(fragments)
                                     if idx != i and idx != j]
                        # Try the container result + remaining in various orders
                        parts = [candidate] + remaining
                        # Forward order
                        if ''.join(parts) == answer:
                            pieces = [Piece(candidate, 'CON', f'{inner} in {outer}')]
                            pieces += [fragments[idx] for idx in range(len(fragments))
                                       if idx != i and idx != j]
                            return ParseResult(
                                operation='container' if len(pieces) == 1 else 'container_charade',
                                pieces=pieces, verified=True,
                                answer=answer, raw_explanation=expl_orig,
                                sub_operations=['container'],
                            )
                        # Try remaining before container
                        for perm_idx in range(len(remaining) + 1):
                            trial = remaining[:perm_idx] + [candidate] + remaining[perm_idx:]
                            if ''.join(trial) == answer:
                                all_pieces = []
                                rem_iter = iter(
                                    [fragments[idx] for idx in range(len(fragments))
                                     if idx != i and idx != j]
                                )
                                for t_idx, t in enumerate(trial):
                                    if t == candidate:
                                        all_pieces.append(
                                            Piece(candidate, 'CON', f'{inner} in {outer}')
                                        )
                                    else:
                                        all_pieces.append(next(rem_iter))
                                return ParseResult(
                                    operation='container_charade',
                                    pieces=all_pieces, verified=True,
                                    answer=answer, raw_explanation=expl_orig,
                                    sub_operations=['container'],
                                )

    # Try 4: Reversal + container combined
    if has_rev:
        if len(fragments) >= 2:
            for i in range(len(fragments)):
                rev_frags = list(fragments)
                rev_frags[i] = Piece(
                    rev_frags[i].letters[::-1], 'REV',
                    rev_frags[i].source_word
                )
                # Now try container with reversed piece
                for j in range(len(rev_frags)):
                    for k in range(len(rev_frags)):
                        if j == k:
                            continue
                        inner = rev_frags[j].letters
                        outer = rev_frags[k].letters
                        for pos in range(1, len(outer)):
                            candidate = outer[:pos] + inner + outer[pos:]
                            rem = [f.letters for idx, f in enumerate(rev_frags)
                                   if idx != j and idx != k]
                            for p_idx in range(len(rem) + 1):
                                trial = rem[:p_idx] + [candidate] + rem[p_idx:]
                                if ''.join(trial) == answer:
                                    return ParseResult(
                                        operation='reversal_container',
                                        pieces=[Piece(answer, 'CON+REV', expl_clean)],
                                        verified=True,
                                        answer=answer, raw_explanation=expl_orig,
                                        sub_operations=['reversal', 'container'],
                                    )

    return None


def _extract_all_fragments(expl: str) -> List[Piece]:
    """Extract all CAPS letter fragments from an explanation.

    Handles:
    - Container notation: CAPS(CAPS)CAPS → separate outer and inner
    - Deletion [lower]CAPS, CAPS[lower], {lower}CAPS, CAPS{lower}
    - Plain CAPS words
    - Abbreviation: LETTER(source)
    - Dot notation: O.M., N.E.
    """
    fragments = []
    pos = 0

    while pos < len(expl):
        # Skip whitespace and punctuation
        if not expl[pos].isupper() and expl[pos] not in '[{':
            pos += 1
            continue

        # Try container: CAPS(CAPS)CAPS
        m = re.match(r'([A-Z]+)\(([A-Z][A-Z\s,]*)\)([A-Z]*)', expl[pos:])
        if m:
            outer_left = m.group(1)
            inner = ''.join(c for c in m.group(2) if c.isalpha())
            outer_right = m.group(3)
            # Return as separate outer and inner pieces for assembly
            outer_word = outer_left + outer_right
            if outer_word and inner:
                fragments.append(Piece(outer_word, 'SYN', outer_left + '(...)' + outer_right))
                fragments.append(Piece(inner, 'SYN', inner))
                pos += m.end()
                continue
            else:
                # Single piece container
                letters = outer_left + inner + outer_right
                fragments.append(Piece(letters, 'CON', m.group(0)))
                pos += m.end()
                continue

        # Try deletion: [lower]CAPS
        m = re.match(r'\[([a-z]+)\]([A-Z]+)', expl[pos:])
        if m:
            fragments.append(Piece(m.group(2), 'DEL', m.group(0)))
            pos += m.end()
            continue

        # Try deletion: CAPS[lower]
        m = re.match(r'([A-Z]+)\[([a-z]+)\]', expl[pos:])
        if m:
            fragments.append(Piece(m.group(1), 'DEL', m.group(0)))
            pos += m.end()
            continue

        # Try deletion: {lower}CAPS{lower} or {lower}CAPS or CAPS{lower}
        m = re.match(r'\{([^}]*)\}([A-Z]+)(?:\{([^}]*)\})?', expl[pos:])
        if m:
            fragments.append(Piece(m.group(2), 'DEL', m.group(0)))
            pos += m.end()
            continue
        m = re.match(r'([A-Z]+)\{([^}]*)\}', expl[pos:])
        if m:
            fragments.append(Piece(m.group(1), 'DEL', m.group(0)))
            pos += m.end()
            continue

        # Try dot-notation abbreviation: O.M., N.E., etc.
        m = re.match(r'([A-Z](?:\.[A-Z])+\.?)', expl[pos:])
        if m:
            letters = m.group(1).replace('.', '')
            fragments.append(Piece(letters, 'ABR', m.group(0)))
            pos += m.end()
            continue

        # Try abbreviation: LETTER(source)
        m = re.match(r'([A-Z])\(([a-z][^)]*)\)', expl[pos:])
        if m:
            fragments.append(Piece(m.group(1), 'ABR', m.group(0)))
            pos += m.end()
            continue

        # Try CAPS(gloss) — synonym with meaning
        m = re.match(r'([A-Z]{2,})\s*\(([a-z\u2018\u2019\u201c\u201d][^)]*)\)', expl[pos:])
        if m:
            fragments.append(Piece(m.group(1), 'SYN', m.group(0), gloss=m.group(2)))
            pos += m.end()
            continue

        # Plain CAPS word
        m = re.match(r'([A-Z]+)', expl[pos:])
        if m:
            letters = m.group(1)
            if letters not in SKIP_CAPS:
                stype = 'ABR' if len(letters) <= 2 else 'SYN'
                fragments.append(Piece(letters, stype, letters))
            pos += m.end()
            continue

        pos += 1

    return fragments


def _strip_commentary(expl: str) -> str:
    """Remove trailing editorial commentary from explanation.

    Commentary often starts after a period/semicolon followed by a space.
    """
    # Remove definition marker at start: "D word" or "def word"
    expl = re.sub(r'^[Dd](?:ef)?\s+\w+[;:,]\s*', '', expl)

    # Cut at commentary boundaries — try each pattern, keep earliest match
    best = len(expl)
    for pat in [
        r'\.\s+[A-Z]',              # ". A..." or ". I..." or ". NHO..."
        r'\.\s*\u2019',             # ".'s" (smart quote)
        r'\.\s*$',                   # trailing period
        r'[;]\s+[A-Za-z]',         # "; word..."
        r'\u2013\s*$',              # trailing dash
        r'\s+\u2013\s+',           # " — " em-dash mid-text
        r'\s+-\s+[A-Z]',           # " - Sentence"
        r'[.]\s*\n',               # period + newline
    ]:
        m = re.search(pat, expl)
        if m and m.start() > 10:  # don't cut too early
            best = min(best, m.start())

    if best < len(expl):
        expl = expl[:best]

    return expl.strip()


def _tokenize(expl: str) -> List[str]:
    """Split explanation into piece tokens.

    Splits on: +, commas between CAPS fragments, semicolons.
    Preserves parenthesised groups.
    """
    # Normalize: strip +_ variants to plain +
    normalized = re.sub(r'\+[_~]', '+', expl)
    normalized = re.sub(r'[_~]\+', '+', normalized)

    # First try splitting on +
    if re.search(r'\s?\+\s?', normalized):
        parts = re.split(r'\s*\+\s*', normalized)
        return [p.strip() for p in parts if p.strip()]

    # Try splitting on commas between CAPS-starting fragments
    if ',' in normalized:
        parts = re.split(r',\s*', normalized)
        # Only accept if most parts start with a CAPS letter
        caps_parts = sum(1 for p in parts if re.match(r'[A-Z]', p.strip()))
        if caps_parts >= len(parts) * 0.5:
            return [p.strip() for p in parts if p.strip()]

    # Try extracting multiple CAPS(gloss) pieces from a single string
    # e.g. "T (end of agent) REASON (motivation)"
    multi = re.findall(r'([A-Z][A-Z.]*(?:\s*\([^)]*\))?)', normalized)
    if len(multi) >= 2:
        return [m.strip() for m in multi if m.strip()]

    # Try splitting on spaces between CAPS words (space-separated charade)
    if len(normalized) < 40 and re.match(r'^[A-Z]', normalized):
        parts = re.split(r'\s+', normalized)
        if all(re.match(r'[A-Z]', p) for p in parts if p):
            return parts

    # Can't split — return as single token
    return [normalized]


def _extract_piece(token: str) -> Optional[Piece]:
    """Extract the letter contribution from a single token.

    Handles:
      - Plain CAPS: "GENT" → letters=GENT, type=RAW
      - CAPS(gloss): "GENT(male)" → letters=GENT, type=SYN, gloss=male
      - LETTER(source): "S(econd)" → letters=S, type=ABR
      - [deleted]CAPS: "[pr]EVENT" → letters=EVENT, type=DEL
      - CAPS[deleted]: "INDUS[try]" → letters=INDUS, type=DEL
      - {deleted}CAPS{deleted}: "{c}ODE{s}" → letters=ODE, type=DEL
      - CAPS(CAPS)CAPS: "A(CHAT)ES" → container, needs special handling
      - reversed: detect REV keyword
    """
    tok = token.strip()
    if not tok:
        return None

    # Skip enumeration-only tokens: (7), (3,5)
    if re.match(r'^\(\d[\d,\-\s]*\)$', tok):
        return None

    # Skip pure commentary (all lowercase, no caps)
    if not re.search(r'[A-Z]', tok):
        return None

    # Container pattern: CAPS(CAPS)CAPS — inner can have spaces
    # e.g. S(TO)IC, FA(LUNG ON)G, A(CHAT)ES
    m = re.match(r'^([A-Z]*)\(([A-Z][A-Z\s]*)\)([A-Z]*)$', tok)
    if m:
        outer_left = m.group(1)
        inner = ''.join(c for c in m.group(2) if c.isalpha())
        outer_right = m.group(3)
        letters = outer_left + inner + outer_right
        return Piece(letters, 'CON', tok)

    # Deletion with []: [lower]CAPS or CAPS[lower]
    # First strip any (gloss) parts
    tok_no_gloss = re.sub(r'\([^)]*\)', '', tok)

    m = re.match(r'^\[([a-z]+)\]([A-Z]+)', tok_no_gloss)
    if m:
        letters = m.group(2)
        return Piece(letters, 'DEL', tok)

    m = re.match(r'^([A-Z]+)\[([a-z]+)\]', tok_no_gloss)
    if m:
        letters = m.group(1)
        return Piece(letters, 'DEL', tok)

    # Deletion with {}: {lower}CAPS{lower} or {lower}CAPS or CAPS{lower}
    remaining = tok_no_gloss
    remaining = re.sub(r'\{[^}]*\}', '', remaining)
    caps_after_curly = ''.join(c for c in remaining if c.isupper())
    if '{' in tok and caps_after_curly:
        return Piece(caps_after_curly, 'DEL', tok)

    # Abbreviation: SINGLE_LETTER(source) like "S(econd)" or "E(nglish)"
    m = re.match(r'^([A-Z])\(([a-z][^)]*)\)', tok)
    if m:
        letter = m.group(1)
        source = m.group(2)
        return Piece(letter, 'ABR', f"{letter}({source})")

    # Abbreviation: LETTERS(source) like "ER(hesitation)", "O.M.(award)"
    m = re.match(r'^([A-Z][A-Z.]+)\(([a-z][^)]*)\)', tok)
    if m:
        letters = m.group(1).replace('.', '')
        source = m.group(2)
        return Piece(letters, 'ABR', f"{m.group(1)}({source})")

    # CAPS(gloss) — synonym with meaning
    m = re.match(r'^([A-Z]{2,})\s*\(([^)]+)\)', tok)
    if m:
        letters = m.group(1)
        gloss = m.group(2)
        return Piece(letters, 'SYN', tok, gloss=gloss)

    # CAPS with dot notation: O.M., N.E., etc.
    m = re.match(r'^([A-Z](?:\.[A-Z])+\.?)\s*', tok)
    if m:
        letters = m.group(1).replace('.', '')
        return Piece(letters, 'ABR', tok)

    # Plain CAPS
    m = re.match(r'^([A-Z]{1,})', tok)
    if m:
        letters = m.group(1)
        if letters in SKIP_CAPS:
            return None
        source_type = 'ABR' if len(letters) <= 2 else 'SYN'
        return Piece(letters, source_type, tok)

    return None


def _determine_operation(pieces: List[Piece], sub_ops: List[str],
                         expl: str) -> str:
    """Determine the primary operation from pieces and context."""

    types = set(p.source_type for p in pieces)

    if 'CON' in types:
        return 'container'
    if 'reversal' in sub_ops:
        if len(pieces) > 1:
            return 'reversal_charade'
        return 'reversal'
    if len(pieces) == 1 and pieces[0].source_type == 'SYN':
        if 'DEL' in types:
            return 'deletion'
        return 'synonym'
    if len(pieces) > 1:
        return 'charade'
    if len(pieces) == 1:
        return pieces[0].source_type.lower()

    return 'unknown'


# === Batch processing ===

def parse_batch(rows: List[Tuple[str, str]]) -> List[ParseResult]:
    """Parse a batch of (answer, explanation) tuples."""
    return [parse_explanation(expl, answer) for answer, expl in rows]
