"""Improved piece-to-word mapping using DB reverse lookups.

The current mapper in mine_span_signatures.py relies on gloss text matching,
which fails 85% of the time because many parsed pieces have no gloss.

This mapper uses the reference DB to match pieces to clue words:
- ABR piece with letters "S" → which clue word has "S" as an abbreviation?
- SYN piece with letters "BO" → which clue word has "BO" as a synonym?
- ANA piece → which consecutive clue words have the right total letter count?
- CON piece → parse source to find inner/outer, match via DB
- REV piece → check synonym reversals
- DEL piece → parse source to find original word, match via DB
"""

import re
import sqlite3
from typing import List, Optional, Tuple, Set, Dict
from cryptic_taxonomy.analysis.notation_parser import Piece


class MappingDB:
    """Lightweight DB loader for reverse lookups during mining."""

    def __init__(self, db_path='data/cryptic_new.db'):
        conn = sqlite3.connect(db_path, timeout=30)

        # word → list of abbreviation values (uppercase)
        self.abbreviations = {}
        for indicator, substitution in conn.execute(
            "SELECT indicator, substitution FROM wordplay"
        ):
            w = indicator.lower().strip()
            sub = substitution.strip().upper()
            if sub:
                self.abbreviations.setdefault(w, []).append(sub)

        # word → list of synonyms (uppercase)
        self.synonyms = {}
        for word, synonym in conn.execute(
            "SELECT word, synonym FROM synonyms_pairs"
        ):
            w = word.lower().strip()
            self.synonyms.setdefault(w, []).append(synonym.strip().upper())

        # Also load definition_answers_augmented into synonyms
        for definition, answer in conn.execute(
            "SELECT definition, answer FROM definition_answers_augmented"
            " WHERE definition IS NOT NULL AND answer IS NOT NULL"
        ):
            w = definition.lower().strip()
            val = answer.strip().upper()
            if w and val:
                if w not in self.synonyms:
                    self.synonyms[w] = []
                if val not in self.synonyms[w]:
                    self.synonyms[w].append(val)

        conn.close()

        # Build reverse indexes: value → set of words that produce it
        self.abbr_reverse = {}  # "S" → {"second", "south", "small", ...}
        for word, vals in self.abbreviations.items():
            for v in vals:
                self.abbr_reverse.setdefault(v, set()).add(word)

        self.syn_reverse = {}  # "BO" → {"smell", "body odour", ...}
        for word, vals in self.synonyms.items():
            for v in vals:
                if len(v) <= 15:  # don't index very long synonyms
                    self.syn_reverse.setdefault(v, set()).add(word)

    def word_produces_abbr(self, word, letters):
        """Check if this clue word can produce these letters as an abbreviation."""
        w = word.lower().strip(".,;:!?\"'()-")
        variants = [w]
        if w.endswith("'s"):
            variants.append(w[:-2])
        if w.endswith("s'"):
            variants.append(w[:-2])
        if len(w) >= 4 and w.endswith("s") and not w.endswith("ss"):
            variants.append(w[:-1])

        for v in variants:
            if v in self.abbreviations:
                if letters in self.abbreviations[v]:
                    return True
        # Also check first letter (common abbreviation pattern)
        if len(letters) == 1 and w and w[0].upper() == letters:
            return True
        return False

    def word_produces_syn(self, word, letters):
        """Check if this clue word can produce these letters as a synonym."""
        w = word.lower().strip(".,;:!?\"'()-")
        variants = [w]
        if w.endswith("'s"):
            variants.append(w[:-2])
        if w.endswith("s'"):
            variants.append(w[:-2])
        if len(w) >= 4 and w.endswith("s") and not w.endswith("ss"):
            variants.append(w[:-1])

        for v in variants:
            if v in self.synonyms:
                if letters in self.synonyms[v]:
                    return True
        return False

    def phrase_produces_syn(self, words, letters):
        """Check if a multi-word phrase can produce these letters as a synonym."""
        phrase = " ".join(w.lower().strip(".,;:!?\"'()-") for w in words)
        if phrase in self.synonyms:
            if letters in self.synonyms[phrase]:
                return True
        return False

    def phrase_produces_abbr(self, words, letters):
        """Check if a multi-word phrase can produce these letters as an abbreviation."""
        phrase = " ".join(w.lower().strip(".,;:!?\"'()-") for w in words)
        if phrase in self.abbreviations:
            if letters in self.abbreviations[phrase]:
                return True
        return False


def _clean(w):
    return w.lower().strip(".,;:!?\"'()-\u2018\u2019\u201c\u201d")


def _normalize_letters(text):
    """Extract ASCII letters from text, normalizing unicode (à→A, é→E etc)."""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join(c.upper() for c in nfkd if c.isascii() and c.isalpha())


def improved_map_pieces_to_words(pieces, wp_words, db):
    """Map each piece to clue word(s) using DB reverse lookups.

    Returns list of (piece, start_idx, span) or None if mapping fails.
    Uses a multi-pass approach:
      Pass 1: High-confidence matches (exact gloss, DB-confirmed abbreviations)
      Pass 2: DB-confirmed synonyms
      Pass 3: Anagram fodder (letter count matching)
      Pass 4: Container/reversal/deletion (source string parsing)
      Pass 5: Fallback strategies
    """
    n = len(wp_words)
    n_pieces = len(pieces)
    used_indices = set()
    mappings = [None] * n_pieces  # index-aligned with pieces

    # === Pass 1: Text-based matches (hint first, then full gloss) ===
    for pi, piece in enumerate(pieces):
        if mappings[pi] is not None:
            continue

        # Try extracted hint FIRST (shorter, more precise than full gloss)
        hint = _extract_hint(piece.source_word)
        if hint:
            result = _gloss_match(hint, wp_words, used_indices)
            if result:
                start, span = result
                mappings[pi] = (start, span)
                used_indices.update(range(start, start + span))
                continue

        # Then try full gloss (may be longer/noisier)
        if piece.gloss:
            result = _gloss_match(piece.gloss, wp_words, used_indices)
            if result:
                start, span = result
                mappings[pi] = (start, span)
                used_indices.update(range(start, start + span))
                continue

    # === Pass 2: DB-backed abbreviation matching ===
    # Process pieces WITH hint/gloss first (they have text to disambiguate),
    # then bare pieces (no text — rely on DB + positional heuristics)
    abr_pieces = [pi for pi, p in enumerate(pieces)
                  if mappings[pi] is None and p.source_type == 'ABR']
    abr_with_text = [pi for pi in abr_pieces
                     if _extract_hint(pieces[pi].source_word) or pieces[pi].gloss]
    abr_bare = [pi for pi in abr_pieces if pi not in abr_with_text]

    for pi in abr_with_text + abr_bare:
        if mappings[pi] is not None:
            continue
        piece = pieces[pi]

        letters = piece.letters.upper()
        candidates = []
        for i in range(n):
            if i in used_indices:
                continue
            if db.word_produces_abbr(wp_words[i], letters):
                candidates.append(i)

        # Also try multi-word phrases
        for span in (2, 3):
            for i in range(n - span + 1):
                indices = set(range(i, i + span))
                if indices & used_indices:
                    continue
                if db.phrase_produces_abbr(wp_words[i:i+span], letters):
                    candidates.append((i, span))

        if len(candidates) == 1:
            c = candidates[0]
            if isinstance(c, tuple):
                start, span = c
            else:
                start, span = c, 1
            mappings[pi] = (start, span)
            used_indices.update(range(start, start + span))
        elif len(candidates) > 1:
            # Multiple candidates — pick the one closest to expected position
            # (pieces are roughly left-to-right in clue)
            best = _pick_best_positional(candidates, pi, n_pieces, n, used_indices)
            if best is not None:
                if isinstance(best, tuple):
                    start, span = best
                else:
                    start, span = best, 1
                mappings[pi] = (start, span)
                used_indices.update(range(start, start + span))

    # === Pass 3: DB-backed synonym matching ===
    # Process pieces WITH hint/gloss first, then bare pieces
    syn_pieces = [pi for pi, p in enumerate(pieces)
                  if mappings[pi] is None and p.source_type in ('SYN', 'DEL', 'REV')]
    syn_with_text = [pi for pi in syn_pieces
                     if _extract_hint(pieces[pi].source_word) or pieces[pi].gloss]
    syn_bare = [pi for pi in syn_pieces if pi not in syn_with_text]

    for pi in syn_with_text + syn_bare:
        if mappings[pi] is not None:
            continue
        piece = pieces[pi]

        letters = piece.letters.upper()
        # For REV, the clue word's synonym should match the REVERSED letters
        lookup_letters = letters[::-1] if piece.source_type == 'REV' else letters

        candidates = []
        # Single word
        for i in range(n):
            if i in used_indices:
                continue
            if db.word_produces_syn(wp_words[i], lookup_letters):
                candidates.append(i)

        # Multi-word phrases
        for span in (2, 3, 4):
            for i in range(n - span + 1):
                indices = set(range(i, i + span))
                if indices & used_indices:
                    continue
                if db.phrase_produces_syn(wp_words[i:i+span], lookup_letters):
                    candidates.append((i, span))

        # Also check abbreviation DB for short pieces (1-2 letters might be ABR not SYN)
        if len(letters) <= 2:
            for i in range(n):
                if i in used_indices and i not in [c for c in candidates if not isinstance(c, tuple)]:
                    continue
                if i not in used_indices and db.word_produces_abbr(wp_words[i], lookup_letters):
                    if i not in candidates:
                        candidates.append(i)

        if len(candidates) == 1:
            c = candidates[0]
            if isinstance(c, tuple):
                start, span = c
            else:
                start, span = c, 1
            mappings[pi] = (start, span)
            used_indices.update(range(start, start + span))
        elif len(candidates) > 1:
            best = _pick_best_positional(candidates, pi, n_pieces, n, used_indices)
            if best is not None:
                if isinstance(best, tuple):
                    start, span = best
                else:
                    start, span = best, 1
                mappings[pi] = (start, span)
                used_indices.update(range(start, start + span))

    # === Pass 4: Anagram fodder (letter count matching) ===
    for pi, piece in enumerate(pieces):
        if mappings[pi] is not None:
            continue
        if piece.source_type != 'ANA':
            continue

        target_letters = _normalize_letters(piece.letters)
        target_sorted = sorted(target_letters)

        # Try consecutive unused word spans
        unused = [i for i in range(n) if i not in used_indices]
        for span in range(1, len(unused) + 1):
            for start_idx in range(len(unused) - span + 1):
                indices = unused[start_idx:start_idx + span]
                # Check contiguity
                if any(indices[j+1] != indices[j] + 1 for j in range(len(indices) - 1)):
                    continue
                combined = _normalize_letters(
                    "".join(wp_words[idx] for idx in indices)
                )
                if sorted(combined) == target_sorted:
                    mappings[pi] = (indices[0], span)
                    used_indices.update(indices)
                    break
            if mappings[pi] is not None:
                break

    # === Pass 5: Container pieces — parse source for inner/outer ===
    for pi, piece in enumerate(pieces):
        if mappings[pi] is not None:
            continue
        if piece.source_type != 'CON':
            continue

        # Try to parse "X in Y" from source_word
        source = piece.source_word or ''
        m = re.search(r'(\w+)\s+in\s+(\w+)', source, re.I)
        if m:
            inner_letters = m.group(1).upper()
            outer_letters = m.group(2).upper()
            # Find clue words that produce inner and outer
            inner_cands = []
            outer_cands = []
            for i in range(n):
                if i in used_indices:
                    continue
                if db.word_produces_syn(wp_words[i], inner_letters) or \
                   db.word_produces_abbr(wp_words[i], inner_letters):
                    inner_cands.append(i)
                if db.word_produces_syn(wp_words[i], outer_letters) or \
                   db.word_produces_abbr(wp_words[i], outer_letters):
                    outer_cands.append(i)

            # Find a non-overlapping pair
            for ic in inner_cands:
                for oc in outer_cands:
                    if ic != oc:
                        # Map the container to span covering both
                        lo = min(ic, oc)
                        hi = max(ic, oc)
                        span = hi - lo + 1
                        if not (set(range(lo, hi + 1)) - {ic, oc}) & used_indices:
                            # Only assign if intervening words are unused
                            mappings[pi] = (lo, span)
                            used_indices.update(range(lo, lo + span))
                            break
                if mappings[pi] is not None:
                    break

    # === Pass 6: Letter-based fallback for remaining pieces ===
    for pi, piece in enumerate(pieces):
        if mappings[pi] is not None:
            continue

        letters = piece.letters.upper()
        # Try direct letter match: clue word's letters == piece letters
        for i in range(n):
            if i in used_indices:
                continue
            w_alpha = "".join(c for c in wp_words[i].upper() if c.isalpha())
            if w_alpha == letters:
                mappings[pi] = (i, 1)
                used_indices.update({i})
                break

    # === Pass 7: First/last-letter abbreviation for single-letter pieces ===
    for pi, piece in enumerate(pieces):
        if mappings[pi] is not None:
            continue
        if len(piece.letters) != 1:
            continue

        letter = piece.letters.upper()
        candidates = []
        for i in range(n):
            if i in used_indices:
                continue
            w = _clean(wp_words[i])
            if w and (w[0].upper() == letter or w[-1].upper() == letter):
                candidates.append(i)

        if len(candidates) == 1:
            mappings[pi] = (candidates[0], 1)
            used_indices.add(candidates[0])
        elif len(candidates) > 1:
            best = _pick_best_positional(candidates, pi, n_pieces, n, used_indices)
            if best is not None:
                idx = best if not isinstance(best, tuple) else best[0]
                mappings[pi] = (idx, 1)
                used_indices.add(idx)

    # === Pass 8: Process of elimination ===
    # If exactly one unmapped piece and one unused word remain, assign it
    unmapped = [pi for pi in range(n_pieces) if mappings[pi] is None]
    unused = [i for i in range(n) if i not in used_indices]
    if len(unmapped) == 1 and len(unused) == 1:
        mappings[unmapped[0]] = (unused[0], 1)
        used_indices.add(unused[0])

    # More general: if all unmapped pieces are single-letter and we have
    # exactly that many unused words, assign by position
    unmapped = [pi for pi in range(n_pieces) if mappings[pi] is None]
    unused = sorted(i for i in range(n) if i not in used_indices)
    if unmapped and len(unmapped) == len(unused):
        # Check all unmapped are single-letter (ABR or similar)
        all_single = all(len(pieces[pi].letters) <= 2 for pi in unmapped)
        if all_single:
            # Sort unmapped by piece index to maintain left-to-right ordering
            unmapped_sorted = sorted(unmapped)
            for pi, wi in zip(unmapped_sorted, unused):
                mappings[pi] = (wi, 1)
                used_indices.add(wi)

    # Check if all pieces mapped
    if any(m is None for m in mappings):
        return None

    return [(pieces[i], mappings[i][0], mappings[i][1]) for i in range(n_pieces)]


def _gloss_match(gloss, wp_words, used_indices):
    """Try to match gloss text against clue words.

    Prefers SHORTEST matching span to avoid greedy over-consumption.
    Tries progressively longer gloss prefixes, but for each, finds
    the shortest clue word span that matches.
    """
    import unicodedata

    def _clean_text(text):
        nfkd = unicodedata.normalize('NFKD', text)
        ascii_text = ''.join(c for c in nfkd if not unicodedata.combining(c))
        for old, new in [('\u2013', ' '), ('\u2014', ' '), ('\u2018', ''),
                         ('\u2019', ''), ('\u201c', ''), ('\u201d', '')]:
            ascii_text = ascii_text.replace(old, new)
        return re.sub(r'[^\w\s]', '', ascii_text).strip().lower()

    gloss_clean = _clean_text(gloss)
    gloss_words = gloss_clean.split()
    if not gloss_words:
        return None

    # Try shortest gloss prefix first (1 word, then 2, etc.)
    # For each, find shortest matching clue span
    for n_gloss in range(1, len(gloss_words) + 1):
        gw = gloss_words[:n_gloss]
        # Try shortest span first
        for span in range(1, len(wp_words) + 1):
            for i in range(len(wp_words) - span + 1):
                indices = set(range(i, i + span))
                if indices & used_indices:
                    continue
                candidate = [_clean(w) for w in wp_words[i:i+span]]
                if _words_match_fuzzy(gw, candidate):
                    return (i, span)
    return None


def _words_match_fuzzy(gloss_words, candidate_words):
    """Fuzzy word matching.

    Handles: exact match, prefix match, plural stripping,
    candidate words being a subsequence of gloss words.
    """
    if not candidate_words:
        return False
    if gloss_words == candidate_words:
        return True
    if len(gloss_words) == 1 and len(candidate_words) == 1:
        g, c = gloss_words[0], candidate_words[0]
        if g == c or c.startswith(g) or g.startswith(c):
            return True
        if c.rstrip("s").rstrip("'") == g or g.rstrip("s").rstrip("'") == c:
            return True
        return False
    if all(any(gw == cw or cw.startswith(gw) or gw.startswith(cw)
               for cw in candidate_words) for gw in gloss_words):
        return True
    if (len(gloss_words) == len(candidate_words) and
            gloss_words[0] == candidate_words[0]):
        return True
    # Candidate words are a subsequence of gloss words
    # e.g. gloss "take legal action" matches candidate ["take", "action"]
    if len(candidate_words) < len(gloss_words):
        gi = 0
        for cw in candidate_words:
            while gi < len(gloss_words):
                gw = gloss_words[gi]
                gi += 1
                if gw == cw or cw.startswith(gw) or gw.startswith(cw):
                    break
            else:
                return False
        return True
    return False


def _extract_hint(source):
    """Extract matching hints from source_word string.

    Prioritizes (gloss) parenthesized text over {deleted} braces,
    since {x} represents deleted letters, not clue word hints.
    Also handles truncated parentheses (missing close paren).
    """
    if not source:
        return None

    # Pattern 1: LETTERS(hint) — e.g. "GR(oss)" or "M(ass)" or "JAy" mixed case
    m = re.match(r'^[A-Za-z]+\(([^)]+)\)', source)
    if m:
        return m.group(1).strip()

    # Pattern 2: LETTERS (hint) or LETTERS (hint — extra) — with space before paren
    # Also handle {deleted} LETTERS (hint) for DEL pieces
    # Allow mixed case letters (e.g. "JAy (bird)")
    m = re.search(r'[A-Za-z]+\s+\(([^)]+)\)', source)
    if m:
        raw = m.group(1).strip()
        raw = re.split(r'[\u2014\u2013/]', raw)[0].strip()
        return raw

    # Pattern 3: LETTERS (hint  — truncated, missing close paren
    m = re.search(r'[A-Za-z]+\s+\(([^)]{2,})', source)
    if m:
        raw = m.group(1).strip()
        raw = re.split(r'[\u2014\u2013/\[]', raw)[0].strip()
        if raw:
            return raw

    # Pattern 4: LETTERS, hint — e.g. "ME, myself," or "NOT, never"
    m = re.match(r'^[A-Z]+,\s*(.+)', source)
    if m:
        raw = m.group(1).strip().rstrip('.,;:')
        if raw and len(raw) > 1:
            return raw

    # Pattern 5: {deleted} — only if no (gloss) was found
    m = re.search(r'\{([^}]+)\}', source)
    if m:
        return m.group(1)

    return None


def _pick_best_positional(candidates, piece_idx, n_pieces, n_words, used_indices):
    """Pick the candidate closest to the expected position.

    Pieces are roughly left-to-right in the clue, so piece_idx/n_pieces
    should map to approximately the same fraction of n_words.
    """
    expected_pos = (piece_idx / max(n_pieces, 1)) * n_words
    best = None
    best_dist = float('inf')

    for c in candidates:
        if isinstance(c, tuple):
            pos = c[0]
        else:
            pos = c
        dist = abs(pos - expected_pos)
        if dist < best_dist:
            best_dist = dist
            best = c
    return best
