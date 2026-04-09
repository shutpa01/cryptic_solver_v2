"""Mine word-span-aware signatures from parsed Times explanations.

For each verified parse:
1. Extract wordplay window (clue minus definition)
2. Map each parsed piece back to clue words via gloss matching
3. Record the span-aware signature: e.g. SYN_F(2w)+ABR_F(1w) charade
4. Count frequencies

Output: signature frequency table that can directly generate catalog.py entries.
"""

import sqlite3
import sys
import re
from collections import Counter
from typing import List, Optional, Tuple

sys.path.insert(0, '.')
from cryptic_taxonomy.analysis.notation_parser import parse_explanation, Piece


def extract_wordplay_window(clue_text, definition):
    """Remove definition from clue to get wordplay words."""
    clue = clue_text.strip()
    defn = definition.strip()

    if clue.lower().startswith(defn.lower()):
        wp = clue[len(defn):].strip()
    elif clue.lower().endswith(defn.lower()):
        wp = clue[:-len(defn)].strip()
    else:
        return None

    wp = wp.strip(" .,;:!?'\"")
    return wp.split() if wp else None


def _clean_text(text):
    """Normalize unicode and strip punctuation for matching."""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', text)
    ascii_text = ''.join(c for c in nfkd if not unicodedata.combining(c))
    # Replace em-dash, en-dash, smart quotes with space/nothing
    for old, new in [('\u2013', ' '), ('\u2014', ' '), ('\u2018', ''),
                     ('\u2019', ''), ('\u201c', ''), ('\u201d', ''),
                     ('–', ' '), ('—', ' ')]:
        ascii_text = ascii_text.replace(old, new)
    return re.sub(r'[^\w\s]', '', ascii_text).strip().lower()


def _clean_word(w):
    """Clean a single clue word for matching."""
    return w.lower().strip(".,;:!?'\"()-\u2018\u2019\u201c\u201d")


def match_gloss_to_words(gloss, wp_words, start_from=0):
    """Try to match a piece's gloss to consecutive clue words.

    Returns (start_idx, span) if matched, or None.
    """
    if not gloss:
        return None

    gloss_clean = _clean_text(gloss)
    gloss_words = gloss_clean.split()
    if not gloss_words:
        return None

    # Take only the first N meaningful gloss words (glosses can have extra commentary)
    # e.g. "smell  body odour" -> just match "smell"
    # e.g. "harbour areas" -> match "harbour areas"
    # Try progressively shorter prefixes of the gloss
    for n_gloss in range(len(gloss_words), 0, -1):
        gw = gloss_words[:n_gloss]
        for i in range(start_from, len(wp_words)):
            for span in range(1, len(wp_words) - i + 1):
                candidate = [_clean_word(w) for w in wp_words[i:i+span]]
                if _words_match(gw, candidate):
                    return (i, span)

    return None


def _words_match(gloss_words, candidate_words):
    """Check if gloss words match candidate clue words."""
    if not candidate_words:
        return False

    # Exact match
    if gloss_words == candidate_words:
        return True

    # Single gloss word matching single candidate (most common)
    if len(gloss_words) == 1 and len(candidate_words) == 1:
        g, c = gloss_words[0], candidate_words[0]
        # Exact, prefix, or suffix match
        if g == c or c.startswith(g) or g.startswith(c):
            return True
        # Handle possessives: "parliament's" matches "parliament"
        if c.rstrip("s").rstrip("'") == g or g.rstrip("s").rstrip("'") == c:
            return True
        return False

    # Multi-word: check if all gloss words appear in candidate
    if all(any(gw == cw or cw.startswith(gw) or gw.startswith(cw)
               for cw in candidate_words) for gw in gloss_words):
        return True

    # Check first word match with same word count
    if (len(gloss_words) == len(candidate_words) and
            gloss_words[0] == candidate_words[0]):
        return True

    return False


def _extract_gloss_from_source(source):
    """Extract matching hints from source_word string.

    Handles patterns like:
    - "BO (smell — body odour )" -> "smell"
    - "DOCKS (harbour areas)" -> "harbour areas"
    - "R(right)" -> "right"
    - "HT (height)" -> "height"
    - "PA" -> None (bare abbreviation)
    - "{market}S" -> "market"
    - "{truc}K [back of]" -> "truck"
    """
    if not source:
        return None

    # Multi-letter with gloss: "CAPS (gloss)"
    m = re.match(r'^[A-Z]+\s*\(([^)]+)\)', source)
    if m:
        raw = m.group(1).strip()
        # Take text before em-dash or slash (commentary separator)
        raw = re.split(r'[—\u2013/]', raw)[0].strip()
        return _clean_text(raw)

    # Single letter with gloss: "R(right)"
    m = re.match(r'^[A-Z]\(([^)]+)\)$', source)
    if m:
        return _clean_text(m.group(1))

    # Deletion with curly braces: "{market}S" or "S{omething}"
    m = re.search(r'\{([^}]+)\}', source)
    if m:
        return m.group(1).lower()

    # Deletion with square brackets: "[pr]EVENT" -> "prevent"
    m = re.match(r'\[([a-z]+)\]([A-Z]+)', source)
    if m:
        return (m.group(1) + m.group(2)).lower()
    m = re.match(r'([A-Z]+)\[([a-z]+)\]', source)
    if m:
        return (m.group(1) + m.group(2)).lower()

    return None


def match_piece_to_clue(piece, wp_words, used_indices):
    """Match a single piece to clue word(s), avoiding used indices.

    Tries multiple strategies:
    1. Gloss-based matching
    2. Source word extraction
    3. Letter-based matching against clue words
    """
    # Strategy 1: Use gloss
    if piece.gloss:
        result = match_gloss_to_words(piece.gloss, wp_words)
        if result:
            start, span = result
            indices = set(range(start, start + span))
            if not indices & used_indices:
                return (start, span)

    # Strategy 2: Extract hint from source_word
    hint = _extract_gloss_from_source(piece.source_word)
    if hint:
        hint_words = hint.split()
        if hint_words:
            result = match_gloss_to_words(hint, wp_words)
            if result:
                start, span = result
                indices = set(range(start, start + span))
                if not indices & used_indices:
                    return (start, span)

    # Strategy 3: Match piece letters against clue words
    letters = piece.letters.lower()
    for i in range(len(wp_words)):
        if i in used_indices:
            continue
        w = _clean_word(wp_words[i])
        # Direct match: letters == word or word starts with letters
        if w == letters or w.startswith(letters) or letters.startswith(w):
            return (i, 1)
        # For short abbreviations: match if letters are initials/parts of the word
        if len(letters) <= 2 and len(w) >= 3 and w[0] == letters[0]:
            return (i, 1)

    # Strategy 4: For multi-word glosses, try matching just the first gloss word
    if piece.gloss:
        first_word = _clean_text(piece.gloss).split()[0] if _clean_text(piece.gloss) else None
        if first_word:
            for i in range(len(wp_words)):
                if i in used_indices:
                    continue
                w = _clean_word(wp_words[i])
                if w == first_word or w.startswith(first_word) or first_word.startswith(w):
                    return (i, 1)

    return None


def map_pieces_to_words(pieces, wp_words):
    """Map each piece to its clue word(s), returning spans.

    Returns list of (piece, start_idx, span) or None if mapping fails.
    """
    mappings = []
    used_indices = set()

    for piece in pieces:
        result = match_piece_to_clue(piece, wp_words, used_indices)
        if result is None:
            return None  # Can't map this piece

        start, span = result
        used_indices.update(range(start, start + span))
        mappings.append((piece, start, span))

    return mappings


def piece_to_token(piece, operation):
    """Map a Piece source_type to our token vocabulary."""
    t = piece.source_type
    if t == 'ABR':
        return 'ABR_F'
    elif t == 'SYN':
        return 'SYN_F'
    elif t == 'ANA':
        return 'ANA_F'
    elif t == 'DEL':
        # Deletion pieces: short = ABR_F, long = SYN_F
        if len(piece.letters) <= 2:
            return 'ABR_F'
        else:
            return 'SYN_F'
    elif t == 'REV':
        return 'SYN_F'  # reversed synonym
    elif t == 'CON':
        return 'SYN_F'  # container result (two parts)
    elif t == 'HOM':
        return 'HOM_F'
    elif t == 'HID':
        return 'HID_F'
    elif t == 'RAW':
        return 'RAW'
    else:
        return t


def build_span_signature(mappings, operation):
    """Build a span-aware signature string from piece mappings.

    Returns e.g. "SYN_F(2w)+ABR_F(1w) charade"
    """
    # Sort by position in clue (left to right)
    sorted_mappings = sorted(mappings, key=lambda x: x[1])

    parts = []
    for piece, start, span in sorted_mappings:
        token = piece_to_token(piece, operation)
        parts.append(f"{token}({span}w)")

    return "+".join(parts) + f" {operation}"


def run():
    conn = sqlite3.connect('data/times_explanations.db', timeout=30)
    rows = conn.execute('''
        SELECT clue_text, answer, definition, explanation
        FROM clues
        WHERE explanation IS NOT NULL AND explanation != ''
        AND definition IS NOT NULL AND definition != ''
        AND clue_text IS NOT NULL AND clue_text != ''
    ''').fetchall()
    conn.close()

    print(f"Total rows with clue+definition+explanation: {len(rows)}")

    sig_counts = Counter()
    mapped_count = 0
    verified_count = 0
    failed_map = 0
    no_wp = 0

    # Also track non-span signatures for comparison
    coarse_counts = Counter()

    for clue_text, answer, definition, expl in rows:
        answer_clean = re.sub(r'[\s\-]', '', answer.upper())
        result = parse_explanation(expl, answer_clean)

        if not result.verified:
            continue
        verified_count += 1

        # Skip whole-clue types (no wordplay decomposition)
        if result.operation in ('double_definition', 'cryptic_definition',
                                 'hidden', 'homophone'):
            sig = f"{result.operation}"
            sig_counts[sig] += 1
            coarse_counts[sig] += 1
            mapped_count += 1
            continue

        # Get wordplay window
        wp_words = extract_wordplay_window(clue_text, definition)
        if not wp_words:
            no_wp += 1
            continue

        if not result.pieces:
            continue

        # Map pieces to clue words
        mappings = map_pieces_to_words(result.pieces, wp_words)
        if mappings is None:
            failed_map += 1
            # Still count coarse signature
            tokens = [piece_to_token(p, result.operation) for p in result.pieces]
            coarse_sig = "+".join(tokens) + f" {result.operation}"
            coarse_counts[coarse_sig] += 1
            continue

        # Build span-aware signature
        sig = build_span_signature(mappings, result.operation)
        sig_counts[sig] += 1
        mapped_count += 1

        # Also count coarse
        tokens = [piece_to_token(p, result.operation) for p in result.pieces]
        coarse_sig = "+".join(tokens) + f" {result.operation}"
        coarse_counts[coarse_sig] += 1

    print(f"Verified parses: {verified_count}")
    print(f"Successfully mapped to spans: {mapped_count}")
    print(f"Failed to map: {failed_map}")
    print(f"No wordplay window: {no_wp}")
    print()

    # Show span-aware signatures
    print(f"{'Span-Aware Signature':60s} {'Count':>7s} {'%':>6s} {'Cum%':>6s}")
    print('-' * 80)
    cum = 0
    total_mapped = sum(sig_counts.values())
    for sig, cnt in sig_counts.most_common(100):
        pct = 100 * cnt / total_mapped
        cum += pct
        print(f"{sig:60s} {cnt:7d} {pct:5.1f}% {cum:5.1f}%")

    print()
    print(f"Total unique span-aware signatures: {len(sig_counts)}")
    print(f"  With count >= 50: {sum(1 for c in sig_counts.values() if c >= 50)}")
    print(f"  With count >= 20: {sum(1 for c in sig_counts.values() if c >= 20)}")
    print(f"  With count >= 10: {sum(1 for c in sig_counts.values() if c >= 10)}")
    print(f"  With count >= 5:  {sum(1 for c in sig_counts.values() if c >= 5)}")

    print()
    print(f"Coarse signatures (no spans): {len(coarse_counts)}")


if __name__ == '__main__':
    run()
