"""Batch V1 mechanical solver — adapts archived V1 stages to solve unsolved clues.

Runs definition finding, anagram detection, compound wordplay, and secondary
helpers on all unsolved clues. Zero API cost — pure DB lookups and string ops.

Writes results to JSONL for upload via batch_upload.py.

Usage:
    python scripts/batch_v1_solver.py                    # all unsolved
    python scripts/batch_v1_solver.py --limit 1000       # test run
    python scripts/batch_v1_solver.py --source guardian   # one source
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
import itertools
from itertools import combinations, permutations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from signature_solver.db import RefDB

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
RESULTS_PATH = os.path.join(ROOT, "data", "batch_v1_results.jsonl")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def norm_letters(s):
    """Normalize string to lowercase letters only."""
    return re.sub(r"[^A-Za-z]", "", s or "").lower()


def parse_enumeration(enumeration):
    """Parse enumeration string to total letter count. '(5,3)' -> 8"""
    if isinstance(enumeration, int):
        return enumeration
    if isinstance(enumeration, str):
        digits = re.findall(r'\d+', enumeration)
        return sum(int(d) for d in digits) if digits else 0
    return 0


def strip_enumeration(clue_text):
    """Remove trailing enumeration from clue text."""
    return re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text).strip()


# ---------------------------------------------------------------------------
# Stage 1: Definition finder
# ---------------------------------------------------------------------------

def generate_definition_windows(clue_text):
    """Generate candidate definition substrings from both ends of the clue."""
    text = strip_enumeration(clue_text)
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')

    words = text.split()
    windows = set()

    for i in range(len(words)):
        window1 = " ".join(words[:i + 1])
        window2 = " ".join(words[-(i + 1):])
        if window1:
            windows.add(window1.strip())
        if window2:
            windows.add(window2.strip())

    # Apostrophe variants
    expanded = set(windows)
    for w in windows:
        if "\u2019s" in w or "'s" in w:
            expanded.add(w.replace("\u2019s", "s").replace("'s", "s"))

    return list(expanded)


def find_definition(clue_text, answer, ref_db, max_words=4):
    """Find the definition window — a substring from start or end that defines the answer.

    Returns (definition_text, remaining_words) or (None, None).
    Prefers longer matches. Definition must be from start or end of clue.
    """
    text = strip_enumeration(clue_text)
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    words = text.split()

    if not words:
        return None, None

    best_def = None
    best_remaining = None
    best_len = 0

    max_window = min(max_words, len(words) - 1)  # must leave at least 1 word for wordplay

    for n in range(1, max_window + 1):
        # From start
        window = " ".join(words[:n])
        if ref_db.is_definition_of(window, answer):
            if n > best_len:
                best_def = window
                best_remaining = words[n:]
                best_len = n

        # From end
        window = " ".join(words[-n:])
        if ref_db.is_definition_of(window, answer):
            if n > best_len:
                best_def = window
                best_remaining = words[:-n]
                best_len = n

    return best_def, best_remaining


def solve_without_definition(clue_text, answer, ref_db, max_def_words=4):
    """Try to solve the wordplay without a known definition.

    For each possible definition window (1 to max_def_words words from
    start or end of clue), try all mechanical solvers on the remaining words.
    If the wordplay assembles to the answer, the definition is confirmed.

    Returns (definition, wordplay_type, pieces) or (None, None, None).
    """
    text = strip_enumeration(clue_text)
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    words = text.split()

    if len(words) < 3:
        return None, None, None

    answer_clean = norm_letters(answer).upper()
    max_window = min(max_def_words, len(words) - 2)  # leave at least 2 words for wordplay

    candidates = []

    for n in range(1, max_window + 1):
        # Definition from start
        candidates.append((" ".join(words[:n]), words[n:]))
        # Definition from end
        candidates.append((" ".join(words[-n:]), words[:-n]))

    for candidate_def, remaining in candidates:
        if len(remaining) < 2:
            continue

        # Try each mechanical solver
        # Charade
        result = try_charade(remaining, answer_clean, ref_db)
        if result:
            return candidate_def, "charade", result["pieces"]

        # Container
        result = try_container(remaining, answer_clean, ref_db)
        if result:
            return candidate_def, "container", result["pieces"]

        # Reversal
        result = try_reversal(remaining, answer_clean, ref_db)
        if result:
            return candidate_def, "reversal", result["pieces"]

        # Anagram
        result = try_anagram(clue_text, answer_clean, ref_db,
                             definition_words=candidate_def.split())
        if result:
            pieces = [{"clue_word": w,
                       "letters": norm_letters(w).upper(),
                       "mechanism": "anagram_fodder"}
                      for w in result["fodder_words"]]
            return candidate_def, "anagram", pieces

        # Acrostic
        result = try_acrostic(remaining, answer_clean, ref_db)
        if result:
            return candidate_def, "acrostic", result["pieces"]

        # Homophone
        result = try_homophone(remaining, answer_clean, ref_db)
        if result:
            return candidate_def, "homophone", result["pieces"]

    return None, None, None


# ---------------------------------------------------------------------------
# Stage 2: Anagram solver (from V1 anagram.py)
# ---------------------------------------------------------------------------

def try_anagram(clue_text, answer, ref_db, definition_words=None):
    """Find combinations of clue words that anagram to the answer.

    Returns dict with fodder_words, wordplay_type, etc. or None.
    """
    answer_clean = norm_letters(answer)
    enum_len = len(answer_clean)
    if enum_len < 3:
        return None

    text = strip_enumeration(clue_text)
    words = text.split()
    excluded = {norm_letters(w) for w in definition_words} if definition_words else set()

    answer_counter = Counter(answer_clean)

    # Build word counters, excluding definition words
    word_counters = []
    for w in words:
        wn = norm_letters(w)
        if wn and wn not in excluded:
            word_counters.append((w, Counter(wn)))

    # Cap combinatorial explosion
    if len(word_counters) > 12:
        return None

    for r in range(1, len(word_counters) + 1):
        for idxs in combinations(range(len(word_counters)), r):
            combined = Counter()
            for i in idxs:
                combined += word_counters[i][1]

            if sum(combined.values()) != enum_len:
                continue

            if combined == answer_counter:
                # Reject self-anagram (answer appears verbatim in clue)
                fodder_text = " ".join(word_counters[i][0] for i in idxs)
                if answer.lower() in text.lower():
                    continue

                fodder_words = [word_counters[i][0] for i in idxs]
                unused_words = [word_counters[i][0] for i in range(len(word_counters)) if i not in idxs]

                return {
                    "wordplay_type": "anagram",
                    "fodder_words": fodder_words,
                    "unused_words": unused_words,
                }

    return None


# ---------------------------------------------------------------------------
# Stage 3: Charade solver (concatenation of abbreviations/synonyms)
# ---------------------------------------------------------------------------

def try_charade(remaining_words, answer, ref_db):
    """Try to build the answer by concatenating abbreviation/synonym values.

    For each word, get all possible letter contributions. Try all ordered
    combinations that concatenate to the answer.
    Returns dict or None.
    """
    answer_clean = norm_letters(answer).upper()
    if not answer_clean or len(remaining_words) < 2:
        return None

    # For each word, collect possible letter contributions
    word_values = []  # list of (original_word, [(value, mechanism), ...])
    for w in remaining_words:
        values = []
        wn = norm_letters(w)

        # Abbreviations
        for abbr in ref_db.get_abbreviations(wn):
            if abbr.upper() in answer_clean:
                values.append((abbr.upper(), "abbreviation"))

        # Short synonyms (up to answer length)
        for syn in ref_db.get_synonyms(wn, max_len=len(answer_clean)):
            s = syn.upper().replace(" ", "").replace("-", "")
            if s and s in answer_clean and len(s) <= len(answer_clean):
                values.append((s, "synonym"))

        # Raw letters of the word itself
        raw = wn.upper()
        if raw and raw in answer_clean:
            values.append((raw, "literal"))

        # First letter
        if wn:
            fl = wn[0].upper()
            if fl in answer_clean:
                values.append((fl, "first_letter"))

        word_values.append((w, values))

    # Filter to words that have at least one possible value
    active = [(w, vals) for w, vals in word_values if vals]
    if len(active) < 2:
        return None

    # Try all combinations of value assignments
    # Cap at 7 active words to avoid explosion
    if len(active) > 7:
        return None

    def try_build(idx, remaining_answer, pieces):
        if not remaining_answer:
            return pieces if idx >= len(active) or all(
                norm_letters(active[j][0]) in {norm_letters(p[0]) for p in pieces}
                or ref_db.is_link_word(norm_letters(active[j][0]))
                for j in range(idx, len(active))
            ) else None
        if idx >= len(active):
            return None

        w, vals = active[idx]

        # Try each possible value for this word
        for val, mech in vals:
            if remaining_answer.startswith(val):
                result = try_build(idx + 1, remaining_answer[len(val):], pieces + [(w, val, mech)])
                if result is not None:
                    return result

        # Try skipping this word (if it's a link word or indicator)
        if ref_db.is_link_word(norm_letters(w)) or ref_db.get_indicator_types(norm_letters(w)):
            result = try_build(idx + 1, remaining_answer, pieces)
            if result is not None:
                return result

        return None

    pieces = try_build(0, answer_clean, [])
    if pieces and len(pieces) >= 2:
        return {
            "wordplay_type": "charade",
            "pieces": [{"clue_word": w, "letters": v, "mechanism": m} for w, v, m in pieces],
        }

    return None


# ---------------------------------------------------------------------------
# Stage 4: Reversal solver
# ---------------------------------------------------------------------------

def try_reversal(remaining_words, answer, ref_db):
    """Check if reversing a synonym/abbreviation of any word gives the answer."""
    answer_clean = norm_letters(answer).upper()

    for w in remaining_words:
        wn = norm_letters(w)
        if ref_db.get_indicator_types(wn):
            continue  # skip indicators

        # Check synonyms
        for syn in ref_db.get_synonyms(wn, max_len=len(answer_clean)):
            s = syn.upper().replace(" ", "").replace("-", "")
            if s[::-1] == answer_clean:
                return {
                    "wordplay_type": "reversal",
                    "pieces": [{"clue_word": w, "letters": s, "mechanism": "reversal"}],
                    "source_word": s,
                }

        # Check raw word reversed
        raw = wn.upper()
        if len(raw) == len(answer_clean) and raw[::-1] == answer_clean:
            return {
                "wordplay_type": "reversal",
                "pieces": [{"clue_word": w, "letters": raw, "mechanism": "reversal"}],
                "source_word": raw,
            }

    return None


# ---------------------------------------------------------------------------
# Stage 4b: Container solver
# ---------------------------------------------------------------------------

def _get_word_values(words, answer_clean, ref_db):
    """For each word, gather all possible letter contributions.

    Returns list of (original_word, [(value, mechanism), ...]).
    Skips container indicators, reversal indicators, and link words.
    Detects reversal indicators and adds reversed values for adjacent words.
    """
    max_piece_len = len(answer_clean) - 1

    # Identify reversal indicators
    rev_indicator_idxs = set()
    for i, w in enumerate(words):
        wn = norm_letters(w)
        ind_types = ref_db.get_indicator_types(wn)
        if ind_types and any(t[0] == 'reversal' for t in ind_types):
            rev_indicator_idxs.add(i)

    word_values = []
    for i, w in enumerate(words):
        wn = norm_letters(w)
        if not wn:
            continue

        # Skip container indicators and reversal indicators
        ind_types = ref_db.get_indicator_types(wn)
        if ind_types and any(t[0] == 'container' for t in ind_types):
            continue
        if i in rev_indicator_idxs:
            continue
        # Skip link words
        if ref_db.is_link_word(wn):
            continue

        values = []
        has_rev = rev_indicator_idxs and (i - 1 in rev_indicator_idxs or i + 1 in rev_indicator_idxs)

        # Abbreviations
        for abbr in ref_db.get_abbreviations(wn):
            a = abbr.upper()
            if len(a) <= max_piece_len:
                values.append((a, "abbreviation"))

        # All synonyms that fit
        for syn in ref_db.get_synonyms(wn, max_len=max_piece_len):
            s = syn.upper().replace(" ", "").replace("-", "")
            if s and len(s) <= max_piece_len:
                values.append((s, "synonym"))
                if has_rev:
                    values.append((s[::-1], "reversal"))

        # Raw letters
        raw = wn.upper()
        if len(raw) <= max_piece_len:
            values.append((raw, "literal"))
            if has_rev:
                values.append((raw[::-1], "reversal"))

        # First letter
        if wn:
            values.append((wn[0].upper(), "first_letter"))

        if values:
            word_values.append((w, values))

    return word_values


def _try_build_string(word_values, target):
    """Recursively try to build target string by concatenating word values.

    Same approach as the charade solver — prunes early when the remaining
    string can't be matched. Words can be skipped (indicators, link words
    already filtered out).

    Returns list of (word, value, mechanism) or None.
    """
    if not word_values:
        return [] if not target else None

    def build(idx, remaining, pieces):
        if not remaining:
            return pieces
        if idx >= len(word_values):
            return None

        w, vals = word_values[idx]

        # Try each value for this word
        for val, mech in vals:
            if remaining.startswith(val):
                result = build(idx + 1, remaining[len(val):], pieces + [(w, val, mech)])
                if result is not None:
                    return result

        # Skip this word (it may be an indicator or irrelevant)
        result = build(idx + 1, remaining, pieces)
        if result is not None:
            return result

        return None

    return build(0, target, [])


def try_container(remaining_words, answer, ref_db):
    """Try to solve as a container: inner letters inserted into outer letters.

    Algorithm:
    1. Require a container indicator in the remaining words
    2. Gather all word values (abbreviations, synonyms, reversals, etc.)
    3. For each possible split point in the answer (where inner is inserted):
       - Extract candidate inner string and outer string
       - Try to build the inner from one subset of words
       - Try to build the outer from the remaining words
    4. Return first valid solution

    Returns dict or None.
    """
    answer_clean = norm_letters(answer).upper()
    if not answer_clean or len(answer_clean) < 4 or len(remaining_words) < 3:
        return None

    # Must have a container indicator
    has_container_ind = False
    for w in remaining_words:
        wn = norm_letters(w)
        ind_types = ref_db.get_indicator_types(wn)
        if ind_types and any(t[0] == 'container' for t in ind_types):
            has_container_ind = True
            break

    if not has_container_ind:
        return None

    word_values = _get_word_values(remaining_words, answer_clean, ref_db)

    if len(word_values) < 2 or len(word_values) > 8:
        return None

    # For each possible inner substring position in the answer:
    # answer = outer_left + inner + outer_right
    # outer = outer_left + outer_right
    for inner_start in range(1, len(answer_clean)):
        for inner_end in range(inner_start + 1, len(answer_clean)):
            inner_target = answer_clean[inner_start:inner_end]
            outer_target = answer_clean[:inner_start] + answer_clean[inner_end:]

            if len(outer_target) < 2:
                continue

            # Try every partition of words into inner/outer groups
            indices = list(range(len(word_values)))
            for inner_size in range(1, len(word_values)):
                for inner_idxs in combinations(indices, inner_size):
                    outer_idxs = tuple(i for i in indices if i not in inner_idxs)
                    if not outer_idxs:
                        continue

                    inner_wv = [word_values[i] for i in inner_idxs]
                    outer_wv = [word_values[i] for i in outer_idxs]

                    # Try to build inner string
                    inner_pieces = _try_build_string(inner_wv, inner_target)
                    if inner_pieces is None:
                        continue

                    # Try to build outer string
                    outer_pieces = _try_build_string(outer_wv, outer_target)
                    if outer_pieces is None:
                        continue

                    # Both matched — solution found
                    pieces = []
                    for w, v, m in outer_pieces:
                        pieces.append({"clue_word": w, "letters": v, "mechanism": m})
                    for w, v, m in inner_pieces:
                        pieces.append({"clue_word": w, "letters": v, "mechanism": m})

                    return {
                        "wordplay_type": "container",
                        "pieces": pieces,
                        "inner": inner_target,
                        "outer": outer_target,
                        "insert_pos": inner_start,
                    }

    return None


# ---------------------------------------------------------------------------
# Stage 5: Acrostic solver
# ---------------------------------------------------------------------------

def try_acrostic(remaining_words, answer, ref_db):
    """Check if first letters of consecutive words spell the answer."""
    answer_clean = norm_letters(answer).upper()
    if len(answer_clean) < 3:
        return None

    # Need at least as many words as answer letters
    words_norm = [norm_letters(w) for w in remaining_words if norm_letters(w)]

    if len(words_norm) < len(answer_clean):
        return None

    # Try contiguous runs of words whose first letters spell the answer
    for start in range(len(words_norm) - len(answer_clean) + 1):
        first_letters = "".join(w[0].upper() for w in words_norm[start:start + len(answer_clean)])
        if first_letters == answer_clean:
            return {
                "wordplay_type": "acrostic",
                "pieces": [{"clue_word": remaining_words[start + i],
                            "letters": words_norm[start + i][0].upper(),
                            "mechanism": "first_letter"}
                           for i in range(len(answer_clean))],
            }

    # Also try last letters
    for start in range(len(words_norm) - len(answer_clean) + 1):
        last_letters = "".join(w[-1].upper() for w in words_norm[start:start + len(answer_clean)])
        if last_letters == answer_clean:
            return {
                "wordplay_type": "acrostic",
                "pieces": [{"clue_word": remaining_words[start + i],
                            "letters": words_norm[start + i][-1].upper(),
                            "mechanism": "last_letter"}
                           for i in range(len(answer_clean))],
            }

    return None


# ---------------------------------------------------------------------------
# Stage 6: Homophone solver
# ---------------------------------------------------------------------------

def try_homophone(remaining_words, answer, ref_db):
    """Check if a homophone of any word matches the answer."""
    answer_clean = norm_letters(answer).upper()

    for w in remaining_words:
        wn = norm_letters(w)
        homos = ref_db.get_homophones(wn)
        for h in homos:
            if h.upper().replace(" ", "") == answer_clean:
                return {
                    "wordplay_type": "homophone",
                    "pieces": [{"clue_word": w, "letters": h.upper(), "mechanism": "sound_of"}],
                }

    return None


# ---------------------------------------------------------------------------
# Build payload for batch_upload.py
# ---------------------------------------------------------------------------

def build_payload(definition, wordplay_type, pieces, explanation_text,
                  row, confidence=1.0):
    """Build payload matching batch_upload.py format."""
    def_start = None
    def_end = None
    if definition:
        idx = row["clue_text"].lower().find(definition.lower())
        if idx >= 0:
            def_start = idx
            def_end = idx + len(definition)

    return {
        "definition": definition,
        "wordplay_type": wordplay_type,
        "ai_explanation": explanation_text,
        "has_solution": 1,
        "reviewed": 1,
        "confidence": confidence,
        "components": {
            "ai_pieces": pieces,
            "assembly": {"op": wordplay_type},
            "wordplay_types": [wordplay_type],
        },
        "wordplay_types": [wordplay_type],
        "definition_start": def_start,
        "definition_end": def_end,
        "model_version": "mechanical_parse",
        "source": row["source"],
        "puzzle_number": row["puzzle_number"],
        "clue_number": row["clue_number"],
    }


def build_explanation_text(wordplay_type, pieces, definition, answer):
    """Build human-readable explanation."""
    if wordplay_type == "anagram":
        fodder = " + ".join(p["clue_word"].upper() for p in pieces)
        expl = 'anagram of %s = %s' % (fodder, answer.upper())
    elif wordplay_type == "container":
        parts = []
        for p in pieces:
            parts.append('%s (%s="%s")' % (p["letters"], p["mechanism"], p["clue_word"]))
        expl = " + ".join(parts) + " = " + answer.upper()
    elif wordplay_type == "charade":
        parts = []
        for p in pieces:
            parts.append('%s (%s="%s")' % (p["letters"], p["mechanism"], p["clue_word"]))
        expl = " + ".join(parts) + " = " + answer.upper()
    elif wordplay_type == "reversal":
        expl = 'reverse of %s = %s' % (pieces[0]["letters"], answer.upper())
    elif wordplay_type == "acrostic":
        words = " ".join(p["clue_word"] for p in pieces)
        mech = pieces[0]["mechanism"] if pieces else "first_letter"
        label = "first letters" if mech == "first_letter" else "last letters"
        expl = '%s of "%s" = %s' % (label, words, answer.upper())
    elif wordplay_type == "homophone":
        expl = 'sounds like %s = %s' % (pieces[0]["clue_word"], answer.upper())
    else:
        expl = answer.upper()

    if definition:
        expl += '; definition: "%s"' % definition

    return expl


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

def load_clues(limit, source_filter):
    """Load unsolved clues from DB, then close connection."""
    conn = sqlite3.connect(CLUES_DB, timeout=30)
    conn.row_factory = sqlite3.Row

    where = "answer IS NOT NULL AND answer != '' AND clue_text IS NOT NULL AND clue_text != ''"
    where += " AND (has_solution IS NULL OR has_solution = 0)"
    params = []
    if source_filter:
        where += " AND source = ?"
        params.append(source_filter)

    if limit:
        rows = conn.execute("""
            SELECT id, source, puzzle_number, clue_number, clue_text, answer, enumeration
            FROM clues WHERE %s
            ORDER BY publication_date DESC
            LIMIT ?
        """ % where, params + [limit]).fetchall()
    else:
        rows = conn.execute("""
            SELECT id, source, puzzle_number, clue_number, clue_text, answer, enumeration
            FROM clues WHERE %s
            ORDER BY publication_date DESC
        """ % where, params).fetchall()

    clues = []
    for r in rows:
        clues.append({
            "id": r["id"],
            "source": r["source"],
            "puzzle_number": r["puzzle_number"],
            "clue_number": r["clue_number"],
            "clue_text": r["clue_text"],
            "answer": r["answer"],
            "enumeration": r["enumeration"],
        })

    conn.close()
    return clues


def run_batch(clues, ref_db, results_path):
    """Run V1 stages on all clues, write results to JSONL."""
    stats = {
        "definition_found": 0,
        "anagram": 0,
        "charade": 0,
        "reversal": 0,
        "acrostic": 0,
        "homophone": 0,
        "unsolved": 0,
    }
    t0 = time.time()

    with open(results_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(clues):
            answer = row["answer"]
            answer_clean = norm_letters(answer).upper()
            if not answer_clean or len(answer_clean) < 2:
                stats["unsolved"] += 1
                continue

            clue_text = row["clue_text"]

            # Stage 1: Find definition
            definition, remaining = find_definition(clue_text, answer_clean, ref_db)
            if definition:
                stats["definition_found"] += 1

            # If no definition found, use full clue words as remaining
            if remaining is None:
                text = strip_enumeration(clue_text)
                remaining = text.split()

            # Stage 2: Try anagram
            result = try_anagram(clue_text, answer_clean, ref_db,
                                 definition_words=definition.split() if definition else None)
            if result:
                pieces = [{"clue_word": w, "letters": norm_letters(w).upper(),
                           "mechanism": "anagram_fodder"} for w in result["fodder_words"]]
                expl = build_explanation_text("anagram", pieces, definition, answer)
                payload = build_payload(definition, "anagram", pieces, expl, row)
                f.write(json.dumps({"clue_id": row["id"], "action": "high_solve", "payload": payload}) + "\n")
                stats["anagram"] += 1
                continue

            # Stage 3: Try charade
            result = try_charade(remaining, answer_clean, ref_db)
            if result:
                expl = build_explanation_text("charade", result["pieces"], definition, answer)
                payload = build_payload(definition, "charade", result["pieces"], expl, row)
                f.write(json.dumps({"clue_id": row["id"], "action": "high_solve", "payload": payload}) + "\n")
                stats["charade"] += 1
                continue

            # Stage 4: Try reversal
            result = try_reversal(remaining, answer_clean, ref_db)
            if result:
                expl = build_explanation_text("reversal", result["pieces"], definition, answer)
                payload = build_payload(definition, "reversal", result["pieces"], expl, row)
                f.write(json.dumps({"clue_id": row["id"], "action": "high_solve", "payload": payload}) + "\n")
                stats["reversal"] += 1
                continue

            # Stage 5: Try acrostic
            result = try_acrostic(remaining, answer_clean, ref_db)
            if result:
                expl = build_explanation_text("acrostic", result["pieces"], definition, answer)
                payload = build_payload(definition, "acrostic", result["pieces"], expl, row)
                f.write(json.dumps({"clue_id": row["id"], "action": "high_solve", "payload": payload}) + "\n")
                stats["acrostic"] += 1
                continue

            # Stage 6: Try homophone
            result = try_homophone(remaining, answer_clean, ref_db)
            if result:
                expl = build_explanation_text("homophone", result["pieces"], definition, answer)
                payload = build_payload(definition, "homophone", result["pieces"], expl, row)
                f.write(json.dumps({"clue_id": row["id"], "action": "high_solve", "payload": payload}) + "\n")
                stats["homophone"] += 1
                continue

            stats["unsolved"] += 1

            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                solved = sum(v for k, v in stats.items() if k != "unsolved" and k != "definition_found")
                print(f"  {i+1}/{len(clues)} ({elapsed:.0f}s) - {solved} solved, {stats['definition_found']} defs found")

    elapsed = time.time() - t0
    solved = sum(v for k, v in stats.items() if k != "unsolved" and k != "definition_found")

    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Definitions found: {stats['definition_found']:,}")
    print(f"  Anagram: {stats['anagram']:,}")
    print(f"  Charade: {stats['charade']:,}")
    print(f"  Reversal: {stats['reversal']:,}")
    print(f"  Acrostic: {stats['acrostic']:,}")
    print(f"  Homophone: {stats['homophone']:,}")
    print(f"  Total solved: {solved:,}")
    print(f"  Unsolved: {stats['unsolved']:,}")
    print(f"  Results written to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch V1 mechanical solver")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clues")
    parser.add_argument("--source", type=str, default=None, help="Filter by source")
    args = parser.parse_args()

    print("=" * 60)
    print("BATCH V1 MECHANICAL SOLVER")
    if args.limit:
        print(f"  Limit: {args.limit}")
    if args.source:
        print(f"  Source: {args.source}")
    print("=" * 60)

    print("\nLoading clues from DB...")
    clues = load_clues(args.limit, args.source)
    print(f"Loaded {len(clues):,} clues. DB connection closed.")

    print("Loading RefDB into memory...")
    ref_db = RefDB()
    print("RefDB loaded. All DB connections closed.\n")

    run_batch(clues, ref_db, RESULTS_PATH)


if __name__ == "__main__":
    main()
