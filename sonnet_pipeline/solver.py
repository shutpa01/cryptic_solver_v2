"""Sonnet pipeline solver — core logic for parsing cryptic clues.

Extracted from pipeline_tiered.py. Contains:
- Assembly operations (charade, container, reversal, deletion, anagram, etc.)
- Homophone engine
- API calls to Claude Sonnet
- Mechanism validation (check_mechanism)
- DB writer (store_result)
- High-level solve_clue() entry point
"""

import itertools
import json
import re
import sqlite3
import time

from anthropic import Anthropic
from dotenv import load_dotenv
from sonnet_pipeline.enricher import _is_fractured_substring

load_dotenv()
client = Anthropic()

SONNET_MODEL = "claude-sonnet-4-20250514"


def _is_container_outer(syn, answer_clean):
    """Check if answer = syn with a contiguous block inserted (container outer).

    E.g. HEARTY is a container outer for HENPARTY because
    HE + NP + ARTY = HENPARTY (NP inserted at position 2 of HEARTY).
    """
    gap = len(answer_clean) - len(syn)
    if gap < 1 or len(syn) < 3:
        return False
    for i in range(1, len(syn) + 1):
        if answer_clean[:i] == syn[:i] and answer_clean[i + gap:] == syn[i:]:
            return True
    return False


def clean(s):
    return re.sub(r"[^A-Z]", "", s.upper())


# -- Cross-reference resolver -------------------------------------------------

def resolve_cross_references(clue_text, puzzle_answers):
    substitutions = {}
    for m in re.finditer(r'\b(\d+)\b', clue_text):
        num = m.group(1)
        if num in puzzle_answers:
            substitutions[num] = puzzle_answers[num]
    return substitutions


# -- Homophone engine ---------------------------------------------------------

class HomophoneEngine:
    def __init__(self, db_path="data/cryptic_new.db"):
        conn = sqlite3.connect(db_path, timeout=30)
        rows = conn.execute("SELECT word, homophone FROM homophones").fetchall()
        conn.close()
        self.sounds_like = {}
        for word, homo in rows:
            self.sounds_like.setdefault(word.lower(), set()).add(homo.lower())
        print("  Loaded %d homophone keys" % len(self.sounds_like))

    def get_homophones(self, word):
        return self.sounds_like.get(word.lower(), set())

    def sounds_similar(self, word1, word2):
        w1, w2 = word1.lower(), word2.lower()
        return w2 in self.sounds_like.get(w1, set()) or w1 in self.sounds_like.get(w2, set())


# -- Assembly operations ------------------------------------------------------

def try_charade(pieces, target):
    if len(pieces) > 7:
        return None
    for perm in itertools.permutations(pieces):
        if "".join(perm) == target:
            return {"op": "charade", "order": list(perm)}
    return None


def try_container(pieces, target):
    for i, outer in enumerate(pieces):
        if len(outer) < 2:
            continue
        for j, inner in enumerate(pieces):
            if i == j:
                continue
            for pos in range(1, len(outer)):
                combined = outer[:pos] + inner + outer[pos:]
                remaining = [p for k, p in enumerate(pieces) if k != i and k != j]
                all_p = [combined] + remaining
                for perm in itertools.permutations(all_p):
                    if "".join(perm) == target:
                        return {"op": "container", "inner": inner, "outer": outer,
                                "pos": pos, "combined": combined, "order": list(perm)}
    return None


def try_reversal(pieces, target):
    # Single piece reversal with charade
    for i, piece in enumerate(pieces):
        if len(piece) < 2:
            continue
        rev = piece[::-1]
        new = pieces[:i] + [rev] + pieces[i+1:]
        for perm in itertools.permutations(new):
            if "".join(perm) == target:
                return {"op": "reversal", "reversed": piece, "gives": rev, "order": list(perm)}
    # Full reversal: reverse the concatenation of all pieces (or a subset charade)
    if len(pieces) >= 2:
        for perm in itertools.permutations(pieces):
            combined = "".join(perm)
            if combined[::-1] == target:
                return {"op": "reversal", "reversed": combined, "gives": target,
                        "order": [target], "reversed_parts": list(perm)}
    return None


def try_deletion(pieces, target):
    for i, piece in enumerate(pieces):
        for del_len in range(1, min(4, len(piece))):
            for start_pos in [0, len(piece) - del_len]:
                shortened = piece[del_len:] if start_pos == 0 else piece[:start_pos]
                if not shortened:
                    continue
                new = pieces[:i] + [shortened] + pieces[i+1:]
                for perm in itertools.permutations(new):
                    if "".join(perm) == target:
                        deleted = piece[:del_len] if start_pos == 0 else piece[start_pos:]
                        return {"op": "deletion", "from": piece, "deleted": deleted,
                                "gives": shortened, "order": list(perm)}
            if del_len == 1:
                for pos in range(1, len(piece) - 1):
                    shortened = piece[:pos] + piece[pos+1:]
                    new = pieces[:i] + [shortened] + pieces[i+1:]
                    for perm in itertools.permutations(new):
                        if "".join(perm) == target:
                            return {"op": "deletion", "from": piece, "deleted": piece[pos],
                                    "gives": shortened, "order": list(perm)}
    return None


def try_outer_deletion(pieces, target):
    for i, piece in enumerate(pieces):
        if len(piece) >= 4:
            stripped = piece[1:-1]
            new = pieces[:i] + [stripped] + pieces[i+1:]
            for perm in itertools.permutations(new):
                if "".join(perm) == target:
                    return {"op": "outer_deletion", "from": piece, "gives": stripped, "order": list(perm)}
    return None


def try_anagram(pieces, target):
    combined = "".join(pieces)
    if sorted(combined) == sorted(target):
        return {"op": "anagram", "fodder": pieces, "gives": target}
    for size in range(1, len(pieces)):
        for combo in itertools.combinations(range(len(pieces)), size):
            sub = "".join(pieces[k] for k in combo)
            rest = [pieces[k] for k in range(len(pieces)) if k not in combo]
            for rperm in itertools.permutations(rest):
                rest_str = "".join(rperm)
                if target.startswith(rest_str) and sorted(sub) == sorted(target[len(rest_str):]):
                    return {"op": "charade+anagram", "charade": list(rperm),
                            "anagram": [pieces[k] for k in combo]}
                if target.endswith(rest_str) and sorted(sub) == sorted(target[:-len(rest_str)]):
                    return {"op": "anagram+charade", "anagram": [pieces[k] for k in combo],
                            "charade": list(rperm)}
    return None


def try_reversal_container(pieces, target):
    for i, piece in enumerate(pieces):
        if len(piece) < 2:
            continue
        rev = piece[::-1]
        new = pieces[:i] + [rev] + pieces[i+1:]
        # Try plain container first, then merged container (for multi-piece inner)
        result = try_container(new, target)
        if not result:
            result = try_merged_container(new, target)
        # Try concatenating the reversed piece with other small pieces as inner
        if not result and len(new) >= 3:
            others = [p for p in new if p != rev]
            for outer in others:
                if len(outer) < 3:
                    continue
                inner_pieces = [p for p in new if p != outer]
                for perm in itertools.permutations(inner_pieces):
                    merged_inner = "".join(perm)
                    for pos in range(1, len(outer)):
                        combined = outer[:pos] + merged_inner + outer[pos:]
                        if combined == target:
                            result = {
                                "op": "reversal_container",
                                "inner": merged_inner, "outer": outer,
                                "pos": pos, "combined": combined,
                                "merged_inner": merged_inner,
                                "pre_reversal": piece,
                                "order": [combined],
                            }
                            return result
        if result:
            result["op"] = "reversal_container"
            result["pre_reversal"] = piece
            return result
    return None


def try_container_reversal(pieces, target):
    """Try container then reverse the entire result.

    E.g. [PEWS, E] → E inside PEWS = PEEWS → reverse = SWEEP.
    """
    for i, outer in enumerate(pieces):
        if len(outer) < 2:
            continue
        for j, inner in enumerate(pieces):
            if i == j:
                continue
            for pos in range(1, len(outer)):
                combined = outer[:pos] + inner + outer[pos:]
                remaining = [p for k, p in enumerate(pieces) if k != i and k != j]
                # Reverse the combined piece, then charade with remaining
                rev_combined = combined[::-1]
                all_p = [rev_combined] + remaining
                for perm in itertools.permutations(all_p):
                    if "".join(perm) == target:
                        return {
                            "op": "container_reversal",
                            "inner": inner, "outer": outer,
                            "pos": pos, "combined": combined,
                            "reversed_combined": rev_combined,
                            "order": list(perm),
                        }
    return None


def try_homophone(pieces, target, homo_engine):
    for i, piece in enumerate(pieces):
        piece_lower = piece.lower()
        for homo in homo_engine.get_homophones(piece_lower):
            homo_upper = clean(homo)
            if not homo_upper:
                continue
            new_pieces = pieces[:i] + [homo_upper] + pieces[i+1:]
            for perm in itertools.permutations(new_pieces):
                if "".join(perm) == target:
                    return {"op": "homophone", "sounds_like": piece,
                            "gives": homo_upper, "order": list(perm)}
    # Try concatenation of pieces (with and without spaces) as homophone key
    for concat in ("".join(pieces).lower(), " ".join(p.lower() for p in pieces)):
        for homo in homo_engine.get_homophones(concat):
            if clean(homo) == target:
                return {"op": "homophone", "sounds_like": concat.upper(), "gives": target}
    # Reverse: check if target sounds like the pieces
    target_lower = target.lower()
    for homo in homo_engine.get_homophones(target_lower):
        if clean(homo) == "".join(pieces):
            return {"op": "homophone", "sounds_like": "".join(pieces), "gives": target}
    return None


def try_spoonerism(clue_text, target, enricher):
    """Try spoonerism: swap initial consonant clusters of two words.

    E.g. HEADBANGING → HEAD|BANGING → swap H↔B → BEAD + HANGING.
    Only triggers when 'Spooner' appears in the clue.
    """
    if "spooner" not in clue_text.lower():
        return None

    VOWELS = set("AEIOU")

    def split_cluster(word):
        """Split word into initial consonant cluster and remainder."""
        for i, c in enumerate(word):
            if c in VOWELS:
                return word[:i], word[i:]
        return word, ""  # all consonants

    # Try splitting target at every position (min 2 chars each side)
    matches = []
    for pos in range(2, len(target) - 1):
        w1, w2 = target[:pos], target[pos:]
        c1, r1 = split_cluster(w1)
        c2, r2 = split_cluster(w2)
        # Both halves need a consonant cluster and a vowel remainder to swap
        if not c1 or not c2 or not r1 or not r2:
            continue
        src1 = c2 + r1
        src2 = c1 + r2
        if src1.lower() in enricher.synonyms and src2.lower() in enricher.synonyms:
            matches.append({
                "op": "spoonerism",
                "source_words": [src1, src2],
                "result_words": [w1, w2],
                "swapped_clusters": [c1, c2],
            })

    # Return best match (prefer longer first word — more natural word boundary)
    if matches:
        return max(matches, key=lambda m: len(m["result_words"][0]))
    return None


def try_hidden(clue_text, target):
    words = clue_text.split()
    pos = 0
    boundaries = []
    for w in words:
        wc = clean(w)
        boundaries.append((pos, pos + len(wc), w))
        pos += len(wc)
    concat = "".join(clean(w) for w in words)

    # Try forwards first, then reversed
    for candidate, reversed_flag in ((target, False), (target[::-1], True)):
        idx = concat.find(candidate)
        if idx >= 0:
            sw = ew = None
            for wi, (ws, we, _) in enumerate(boundaries):
                if ws <= idx < we:
                    sw = wi
                if ws < idx + len(candidate) <= we:
                    ew = wi
            if sw is not None and ew is not None:
                if sw != ew:
                    op = "hidden_reversed" if reversed_flag else "hidden"
                    return {"op": op, "words": " ".join(words[sw:ew+1])}
                word_clean = clean(words[sw])
                if len(candidate) < len(word_clean):
                    if reversed_flag:
                        return {"op": "hidden_reversed", "words": words[sw]}
                    return {"op": "hidden_in_word", "word": words[sw]}
    return None


def try_merged_container(pieces, target):
    """Try merging pieces into a single inner for container.

    Fast path: merge tiny pieces (len <= 2) — e.g. [HEARTY, N, P] -> NP inside HEARTY.
    Broad path: merge any non-largest subset — e.g. [P, RIS, SURE] -> PRIS inside SURE.
    """
    if len(pieces) < 3:
        return None

    # Fast path: merge only tiny pieces (original logic)
    small = [(i, p) for i, p in enumerate(pieces) if len(p) <= 2]
    if len(small) >= 2:
        for n in range(2, len(small) + 1):
            for combo in itertools.combinations(small, n):
                idxs = {i for i, _ in combo}
                merged_parts = [p for _, p in combo]
                remaining = [p for i, p in enumerate(pieces) if i not in idxs]
                for perm in itertools.permutations(merged_parts):
                    merged = "".join(perm)
                    trial = remaining + [merged]
                    result = try_container(trial, target)
                    if result:
                        result["merged_inner"] = merged
                        return result

    # Broad path: try merging any subset as inner, keeping one piece as outer.
    # Pick each piece as the candidate outer, merge the rest.
    max_len = max(len(p) for p in pieces)
    indexed = list(enumerate(pieces))
    for outer_idx, outer_p in indexed:
        if len(outer_p) < max_len:
            continue  # only the longest piece(s) can be outer
        others = [(i, p) for i, p in indexed if i != outer_idx]
        if len(others) < 2:
            continue
        other_parts = [p for _, p in others]
        for perm in itertools.permutations(other_parts):
            merged = "".join(perm)
            trial = [outer_p, merged]
            result = try_container(trial, target)
            if result:
                result["merged_inner"] = merged
                return result
    return None


def try_deletion_anagram(pieces, target):
    """Try deleting from one piece, then check if all pieces form an anagram of target.

    E.g. [BRITISH, CUT] -> delete T from CUT -> [BRITISH, CU],
    anagram(BRITISHCU) = HUBRISTIC.

    Iterates shortest pieces first — in cryptic crosswords the deletion target
    is almost always the shorter word, not the main anagram fodder.
    """
    total = sum(len(p) for p in pieces)
    excess = total - len(target)
    if excess < 1 or excess > 3:
        return None
    # Try shorter pieces first (more likely deletion targets)
    order = sorted(range(len(pieces)), key=lambda i: len(pieces[i]))
    for i in order:
        piece = pieces[i]
        if len(piece) <= excess:
            continue
        # Delete from end
        shortened = piece[:len(piece) - excess]
        new = pieces[:i] + [shortened] + pieces[i+1:]
        if sorted("".join(new)) == sorted(target):
            return {"op": "deletion+anagram", "from": piece,
                    "deleted": piece[len(piece) - excess:], "gives": shortened,
                    "fodder": new}
        # Delete from start
        shortened = piece[excess:]
        new = pieces[:i] + [shortened] + pieces[i+1:]
        if sorted("".join(new)) == sorted(target):
            return {"op": "deletion+anagram", "from": piece,
                    "deleted": piece[:excess], "gives": shortened,
                    "fodder": new}
        # Internal single deletion
        if excess == 1:
            for pos in range(1, len(piece) - 1):
                shortened = piece[:pos] + piece[pos+1:]
                new = pieces[:i] + [shortened] + pieces[i+1:]
                if sorted("".join(new)) == sorted(target):
                    return {"op": "deletion+anagram", "from": piece,
                            "deleted": piece[pos], "gives": shortened,
                            "fodder": new}
    return None


def assemble(clue_text, answer, pieces, max_pieces=6, homo_engine=None,
             ai_wtype=None):
    target = clean(answer)
    if not pieces:
        return try_hidden(clue_text, target)
    if len(pieces) > max_pieces:
        return None

    # If AI identified a specific non-anagram type, suppress anagram catch-all
    # to avoid producing misleading explanations
    suppress_anagram = (ai_wtype is not None and ai_wtype != "anagram")

    # Only try deletion/outer_deletion when AI explicitly said deletion type,
    # or when pieces are from the enrichment fallback (ai_wtype=None).
    suppress_deletion = (ai_wtype is not None and ai_wtype not in
                         ("deletion", "substitution"))

    methods = [
        lambda: try_charade(pieces, target),
        lambda: try_container(pieces, target),
        lambda: try_merged_container(pieces, target),
        lambda: try_reversal(pieces, target),
    ]
    if not suppress_deletion:
        methods.extend([
            lambda: try_deletion(pieces, target),
            lambda: try_outer_deletion(pieces, target),
        ])
    if not suppress_anagram:
        methods.extend([
            lambda: try_anagram(pieces, target),
            lambda: try_deletion_anagram(pieces, target),
        ])
    methods.append(lambda: try_reversal_container(pieces, target))
    methods.append(lambda: try_container_reversal(pieces, target))
    if homo_engine:
        methods.append(lambda: try_homophone(pieces, target, homo_engine))
    for method in methods:
        result = method()
        if result:
            return result
    return None


# -- Gap filler ----------------------------------------------------------------

def try_gap_fill(clue_text, answer, pieces, enricher, target, ai_wtype=None):
    total_len = sum(len(p) for p in pieces)
    gap = len(target) - total_len
    if gap < 1 or gap > 4:
        return None

    words = clue_text.split()
    fills = []
    for w in words:
        w_lower = w.lower().strip(".,;:!?\"'()-")
        if not w_lower:
            continue
        for a in enricher.lookup_abbreviations(w_lower):
            a_clean = clean(a)
            if a_clean and 1 <= len(a_clean) <= gap:
                fills.append((w_lower, a_clean, "abbreviation"))
        syns = enricher.lookup_synonyms(w_lower, max_results=10, max_len=gap, answer=answer)
        for s in syns:
            if 1 <= len(s) <= gap:
                fills.append((w_lower, s, "synonym"))
        w_clean = clean(w)
        if w_clean and gap >= 1:
            fills.append((w_lower, w_clean[0], "first_letter"))

    fills = list(set(fills))
    for num_fills in range(1, min(3, len(fills) + 1)):
        for combo in itertools.combinations(fills, num_fills):
            fill_letters = [yld for _, yld, _ in combo]
            fill_total = sum(len(f) for f in fill_letters)
            if fill_total != gap:
                continue
            all_pieces = pieces + fill_letters
            result = assemble(clue_text, answer, all_pieces, ai_wtype=ai_wtype)
            if result:
                result["gap_fill"] = [(w, yld, mech) for w, yld, mech in combo]
                return result
    return None


# -- Enrichment fallback -------------------------------------------------------

def enrichment_fallback(clue_text, answer, enricher, target, definition=None):
    words = clue_text.split()
    answer_clean = clean(answer)
    _deletion_meta = {}  # (word, reduced) -> {source_syn, deleted, deleted_word}

    # Detect reversal/deletion indicators in clue
    has_reversal_indicator = False
    has_deletion_indicator = False
    for w in words:
        ind_types = enricher.lookup_indicators(w.lower().strip(".,;:!?\"'()-"))
        for t in ind_types:
            t_clean = t.rstrip("?")
            if t_clean == "reversal":
                has_reversal_indicator = True
            if t_clean == "deletion":
                has_deletion_indicator = True

    # Pre-collect all abbreviations across clue words (for deletion matching)
    all_abbrevs = {}  # word -> set of abbreviation strings
    if has_deletion_indicator:
        for w in words:
            wl = w.lower().strip(".,;:!?\"'()-")
            if wl:
                for a in enricher.lookup_abbreviations(wl):
                    a_clean = clean(a)
                    if a_clean:
                        all_abbrevs.setdefault(wl, set()).add(a_clean)

    word_candidates = {}
    for w in words:
        w_lower = w.lower().strip(".,;:!?\"'()-")
        if not w_lower:
            continue
        cands = []
        w_clean = clean(w)
        if w_clean and len(w_clean) >= 1:
            cands.append((w_clean, "literal"))
        # Allow slightly longer synonyms when deletion indicator present
        # (e.g. ADDER=5 for EDDA=4: ADDER - R = ADDE → anagram = EDDA)
        syn_max_len = len(answer_clean) + 2 if has_deletion_indicator else len(answer_clean)
        syns = enricher.lookup_synonyms(w_lower, max_results=30, max_len=syn_max_len, answer=answer)
        for s in syns:
            if s in answer_clean and len(s) >= 2:
                cands.append((s, "synonym"))
            elif _is_container_outer(s, answer_clean):
                cands.append((s, "synonym"))
            elif _is_fractured_substring(s, answer_clean):
                cands.append((s, "synonym"))
            elif has_reversal_indicator and len(s) >= 3 and s[::-1] in answer_clean:
                cands.append((s, "synonym"))
            elif has_deletion_indicator and len(s) >= 3:
                # Try deleting abbreviations from OTHER clue words
                for other_w, abbr_set in all_abbrevs.items():
                    if other_w == w_lower:
                        continue
                    for a in abbr_set:
                        if a in s:
                            reduced = s.replace(a, "", 1)
                            if len(reduced) >= 2 and reduced in answer_clean:
                                cands.append((reduced, "deletion"))
                # Try removing any single letter from slightly longer synonyms
                # and check if the result is an anagram of the answer
                # (e.g. ADDER-R=ADDE→EDDA, ALIAS-I=ALAS)
                if len(s) == len(answer_clean) + 1:
                    for di in range(len(s)):
                        reduced = s[:di] + s[di+1:]
                        if sorted(reduced) == sorted(answer_clean):
                            cands.append((reduced, "deletion"))
                            break
        # For deletion clues, scan ALL synonyms (bypassing max_results/max_len)
        # for cases where synonym minus abbreviation = answer
        if has_deletion_indicator:
            from sonnet_pipeline.enricher import _word_variants
            all_syns = []
            for variant in _word_variants(w_lower):
                all_syns.extend(enricher.synonyms.get(variant, []))
            for s in all_syns:
                if len(s) <= len(answer_clean) or len(s) > len(answer_clean) + 4:
                    continue
                for other_w, abbr_set in all_abbrevs.items():
                    if other_w == w_lower:
                        continue
                    for a in abbr_set:
                        if a in s:
                            reduced = s.replace(a, "", 1)
                            if reduced == answer_clean:
                                cands.append((reduced, "deletion"))
                                # Store metadata for description
                                _deletion_meta[(w_lower, reduced)] = {
                                    "source_syn": s, "deleted": a,
                                    "deleted_word": other_w,
                                }
                # Also try removing any single letter to get the answer directly
                # (e.g. ALIAS - I = ALAS, where I isn't an abbreviation of another word)
                if len(s) == len(answer_clean) + 1:
                    for di in range(len(s)):
                        reduced = s[:di] + s[di+1:]
                        if reduced == answer_clean:
                            cands.append((reduced, "deletion"))
                            _deletion_meta[(w_lower, reduced)] = {
                                "source_syn": s, "deleted": s[di],
                                "deleted_word": "(single letter)",
                            }
                            break
        abbrevs = enricher.lookup_abbreviations(w_lower)
        for a in abbrevs:
            a_clean = clean(a)
            if a_clean and a_clean in answer_clean:
                cands.append((a_clean, "abbreviation"))
        # When deletion indicator present, try halves of literal words
        # (e.g. "half of yard" → YA or YA)
        if has_deletion_indicator and w_clean and len(w_clean) >= 4:
            mid = len(w_clean) // 2
            first_half = w_clean[:mid]
            last_half = w_clean[mid:]
            if first_half in answer_clean and len(first_half) >= 2:
                cands.append((first_half, "deletion"))
            if last_half in answer_clean and len(last_half) >= 2:
                cands.append((last_half, "deletion"))
        if w_clean:
            if w_clean[0] in answer_clean:
                cands.append((w_clean[0], "first_letter"))
            if len(w_clean) > 1 and w_clean[-1] in answer_clean:
                cands.append((w_clean[-1], "last_letter"))
            if len(w_clean) >= 4:
                alt_even = "".join(w_clean[i] for i in range(0, len(w_clean), 2))
                alt_odd = "".join(w_clean[i] for i in range(1, len(w_clean), 2))
                if alt_even in answer_clean and len(alt_even) >= 2:
                    cands.append((alt_even, "alternate_letters"))
                if alt_odd in answer_clean and len(alt_odd) >= 2:
                    cands.append((alt_odd, "alternate_letters"))
        if cands:
            word_candidates[w_lower] = list(set(cands))

    for i in range(len(words) - 1):
        phrase = words[i].lower().strip(".,;:!?\"'()") + " " + words[i+1].lower().strip(".,;:!?\"'()")
        syns = enricher.lookup_synonyms(phrase, max_results=10, max_len=len(answer_clean), answer=answer)
        for s in syns:
            if s in answer_clean and len(s) >= 2:
                word_candidates.setdefault(phrase, []).append((s, "synonym"))

    all_cands = list(set(
        (w, yld, mech) for w, cands in word_candidates.items() for yld, mech in cands
    ))

    # Exclude candidates whose clue word is part of the definition
    # (a word can't be both definition and wordplay)
    if definition:
        def_words = set(definition.lower().split())
        all_cands = [(w, yld, mech) for w, yld, mech in all_cands
                     if w not in def_words]

    # Special case: single candidate that assembles to the answer
    # (deletion to exact match, or deletion to anagram/reversal)
    for w, yld, mech in all_cands:
        if len(yld) == len(target) and yld != target:
            result = assemble(clue_text, answer, [yld])
            if result:
                result["source"] = "enrichment_fallback"
                result["pieces_detail"] = [(w, yld, mech)]
                return result

    # Special case: deletion candidate that equals the full answer
    for w, yld, mech in all_cands:
        if mech == "deletion" and yld == target:
            meta = _deletion_meta.get((w, yld), {})
            source_syn = meta.get("source_syn", "?")
            deleted = meta.get("deleted", "?")
            deleted_word = meta.get("deleted_word", "?")
            return {
                "op": "deletion", "source": "enrichment_fallback",
                "from": source_syn, "deleted": deleted,
                "gives": target,
                "pieces_detail": [
                    (w, source_syn, "synonym"),
                    (deleted_word, deleted, "abbreviation"),
                ],
            }

    for num in range(2, min(6, len(all_cands) + 1)):
        for combo in itertools.combinations(all_cands, num):
            # One-function-per-word: each clue word can only contribute one piece
            # (exception: definition words may overlap, but that's handled elsewhere)
            combo_words = [w for w, _, _ in combo]
            if len(combo_words) != len(set(combo_words)):
                continue
            pieces = [yld for _, yld, _ in combo]
            total_len = sum(len(p) for p in pieces)
            if total_len < len(target) or total_len > len(target) * 2:
                continue
            if any(p == target for p in pieces):
                continue
            result = assemble(clue_text, answer, pieces)
            if result:
                result["source"] = "enrichment_fallback"
                result["pieces_detail"] = [(w, yld, mech) for w, yld, mech in combo]
                return result

    # Try substitution: gather ALL synonyms close to answer length from every
    # clue word (not just the filtered candidates — substitution bases like SAVE
    # won't be substrings of the answer SANE)
    sub_cands = []
    for w in words:
        w_lower = w.lower().strip(".,;:!?\"'()-")
        if not w_lower or len(w_lower) < 2:
            continue
        if definition and w_lower in definition.lower().split():
            continue
        syns = enricher.lookup_synonyms(w_lower, max_results=30,
                                        max_len=len(target) + 2, answer=answer)
        for s in syns:
            if len(s) >= 3 and abs(len(s) - len(target)) <= 2:
                sub_cands.append(s)
    for s in sub_cands:
        result = try_substitution([s], target, clue_text, enricher)
        if result:
            result["source"] = "enrichment_fallback"
            return result

    return None


# -- API calls -----------------------------------------------------------------

# ---- Pass 1: Reasoning (free-text explanation, no DB menu) ----

REASONING_PROMPT = """You are an expert cryptic crossword analyst. Given a clue and its answer, explain how the wordplay works.

Think step by step:
1. Identify the definition part (always at the start or end of the clue).
2. Identify any indicator words (anagram indicators, reversal indicators, container indicators, deletion indicators, etc.). Indicator words signal the wordplay mechanism but do NOT contribute letters to the answer. Do not assign letters to indicator words.
3. For each remaining wordplay word, explain what letters it contributes and why (synonym, abbreviation, first letter, literal letters for anagram, etc.).
4. Show how the letters combine to spell the answer.

IMPORTANT: If indicator words are listed below the clue, those words signal the mechanism — they do not produce letters. For example, if "various" is flagged as an anagram indicator, it tells you the other words are anagram fodder, but "various" itself contributes no letters.

Be precise about letter-level mechanics. For example:
- "father" = PA (informal word for father)
- "attempt" = TRY (synonym)
- PA + S (possessive 's) + TRY = PASTRY

Wordplay types: charade, container, anagram, deletion, hidden, reversal, homophone, double_definition, cryptic_definition, acrostic, spoonerism, substitution.

IMPORTANT: If "Known lookups" are provided below the clue, these are confirmed synonym/abbreviation mappings where the letters appear in the answer. If your initial interpretation doesn't work, TRY THESE LOOKUPS as letter contributors before giving up. A word flagged as an indicator may actually be a synonym instead — e.g. "unfortunately" could be an anagram indicator OR it could contribute ALAS as a synonym.

If a number appears in the clue, consider it in its WORD FORM as potential letter fodder (e.g. "18" = EIGHTEEN, "100" = HUNDRED or C).

If you cannot make the wordplay produce the answer, consider whether the clue is a double definition or cryptic definition — the entire clue (or two halves) may define the answer with no wordplay at all.

Keep your explanation concise — focus on the mechanics, not on restating the clue."""

REASONING_EXAMPLES = [
    {
        "input": "Clue: Patterned plate for various clients (7)\nAnswer: STENCIL",
        "output": "Definition: \"Patterned plate\" (a stencil creates patterned plates).\nWordplay: anagram. \"various\" is the anagram indicator.\nFodder: CLIENTS → rearranged → STENCIL."
    },
    {
        "input": "Clue: Juliet in sober group with kiss for hero (4)\nAnswer: AJAX",
        "output": "Definition: \"hero\" (Ajax is a Greek hero).\nWordplay: charade. A = sober (teetotal, AA abbreviated — but here just A for a single \"sober\" abbreviation? No: AA = Alcoholics Anonymous (sober group), J = Juliet (NATO alphabet), X = kiss. But AAJX ≠ AJAX.\nRethinking: A = first letter? No. The answer is A-J-A-X. So: A (from group=AA split), J (Juliet), A (second A from AA), X (kiss). AA split around J: A + J + A + X = AJAX. Actually: J inside AA = AJA, then + X = AJAX. Container: J inside AA gives AJA, plus X = AJAX."
    },
    {
        "input": "Clue: Unwilling to forgo large bond (4)\nAnswer: OATH",
        "output": "Definition: \"bond\" (an oath is a bond).\nWordplay: deletion. \"Unwilling\" = LOATH (synonym). \"forgo large\" = remove L (standard abbreviation for large). LOATH − L = OATH."
    },
    {
        "input": "Clue: Father's attempt to make small cake (6)\nAnswer: PASTRY",
        "output": "Definition: \"cake\" (a pastry is a cake).\nWordplay: charade. \"Father's\" = PA + S (possessive). \"attempt\" = TRY (synonym). PAS + TRY = PASTRY."
    },
]

# ---- Pass 2: Structuring (extract JSON from free-text explanation) ----

STRUCTURING_PROMPT = """Extract structured data from a cryptic crossword explanation.

Given a clue, its answer, and a free-text explanation of the wordplay, output JSON with:
- "definition": the exact substring of the clue that defines the answer
- "wordplay_type": one of: charade, container, anagram, deletion, hidden, reversal, homophone, double_definition, cryptic_definition, acrostic, spoonerism, substitution
- "pieces": array of objects, each with:
  - "clue_word": the word(s) from the clue
  - "letters": the uppercase letters this produces
  - "mechanism": synonym, abbreviation, literal, anagram_fodder, first_letter, last_letter, reversal, sound_of, alternate_letters, core_letters, deletion, hidden

Rules:
- The pieces' letters, when assembled via the wordplay_type, MUST spell the full answer.
- For anagrams: pieces are the raw fodder letters BEFORE rearrangement.
- For containers: show outer and inner pieces separately.
- For hidden words: one piece with the spanning clue words and mechanism "hidden".
- For double/cryptic definitions: no pieces needed.
- Definition is always at the start or end of the clue.
- Indicator words are NOT pieces — they signal the operation type.

Return ONLY valid JSON."""

STRUCTURING_EXAMPLES = [
    {
        "input": """Clue: Father's attempt to make small cake (6)
Answer: PASTRY
Explanation: Definition: "cake" (a pastry is a cake). Wordplay: charade. "Father's" = PA + S (possessive). "attempt" = TRY (synonym). PAS + TRY = PASTRY.""",
        "output": json.dumps({
            "definition": "cake",
            "wordplay_type": "charade",
            "pieces": [
                {"clue_word": "Father's", "letters": "PAS", "mechanism": "synonym"},
                {"clue_word": "attempt", "letters": "TRY", "mechanism": "synonym"}
            ]
        })
    },
    {
        "input": """Clue: Unwilling to forgo large bond (4)
Answer: OATH
Explanation: Definition: "bond" (an oath is a bond). Wordplay: deletion. "Unwilling" = LOATH (synonym). "forgo large" = remove L (abbreviation for large). LOATH − L = OATH.""",
        "output": json.dumps({
            "definition": "bond",
            "wordplay_type": "deletion",
            "pieces": [
                {"clue_word": "Unwilling", "letters": "LOATH", "mechanism": "synonym"},
                {"clue_word": "large", "letters": "L", "mechanism": "abbreviation"}
            ]
        })
    },
]


def build_example_messages():
    """Build few-shot examples for the reasoning pass."""
    msgs = []
    for ex in REASONING_EXAMPLES:
        msgs.append({"role": "user", "content": ex["input"]})
        msgs.append({"role": "assistant", "content": ex["output"]})
    return msgs


def _build_structuring_examples():
    """Build few-shot examples for the structuring pass."""
    msgs = []
    for ex in STRUCTURING_EXAMPLES:
        msgs.append({"role": "user", "content": ex["input"]})
        msgs.append({"role": "assistant", "content": ex["output"]})
    return msgs


def _parse_json_response(raw):
    """Extract JSON from a model response, tolerating preamble/postamble."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        if "{" in raw:
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                return json.loads(raw[start:end])
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def _extract_indicator_hints(enrichment):
    """Extract indicator types from enrichment string for the AI prompt.

    Returns a list like ['broken: anagram', 'around: container'].
    """
    import re
    hints = []
    for line in enrichment.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        # Lines look like: "broken: syn=X abbr=Y ind=anagram,reversal"
        word = line.split(":")[0].strip()
        ind_match = re.search(r"ind=([^\s;]+)", line)
        if ind_match:
            ind_types = ind_match.group(1).replace("?", "")
            for t in ind_types.split(","):
                t = t.strip()
                if t:
                    hints.append("%s: %s" % (word, t))
    return hints


def _extract_starred_lookups(enrichment):
    """Extract answer-relevant synonyms and abbreviations from enrichment.

    Starred entries (e.g. syn=TRY*,GO or abbr=L*) indicate the lookup
    matches a substring of the answer. These are the most useful hints
    for the AI.

    Filters: synonyms must be 3+ letters (short ones are often noise),
    skip words that are also indicators (they signal mechanism, not letters).

    Returns a list like ['attempt → synonym TRY', 'student → abbrev L'].
    """
    import re

    # First pass: collect indicator words so we can skip them
    indicator_words = set()
    for line in enrichment.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        if "ind=" in line:
            indicator_words.add(line.split(":")[0].strip())

    lookups = []
    for line in enrichment.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        word = line.split(":")[0].strip()

        # Skip words that are indicators — they signal mechanism, not letters
        if word in indicator_words:
            continue

        entries_part = line.split(":", 1)[1]

        # Extract starred synonyms (3+ letters to avoid noise like RH, IS, etc.)
        syn_match = re.search(r"syn=([^\s;]+)", entries_part)
        if syn_match:
            for s in syn_match.group(1).split(","):
                if s.endswith("*") and len(s) >= 4:  # 4 = 3 letters + *
                    lookups.append("%s → synonym %s" % (word, s[:-1]))

        # Extract starred abbreviations (1-2 letters, these are reliable)
        abbr_match = re.search(r"abbr=([^\s;]+)", entries_part)
        if abbr_match:
            for a in abbr_match.group(1).split(","):
                if a.endswith("*") and len(a) <= 3:  # 3 = 2 letters + *
                    lookups.append("%s → abbrev %s" % (word, a[:-1]))

    return lookups


def call_reasoning(model, clue_text, answer, example_messages,
                    indicator_hints=None, starred_lookups=None):
    """Pass 1: Get free-text explanation of how the wordplay works.

    Uses extended thinking to allow the model to explore hypotheses
    privately before committing to a visible explanation.
    """
    enum_len = len(answer.replace(" ", "").replace("-", ""))
    user_msg = "Clue: %s (%d)\nAnswer: %s" % (clue_text, enum_len, answer)
    if indicator_hints:
        user_msg += "\nIndicator words (these signal mechanism, do NOT contribute letters): %s" % "; ".join(indicator_hints)
    if starred_lookups:
        user_msg += "\nKnown lookups (confirmed in reference DB): %s" % "; ".join(starred_lookups)

    # Skip few-shot examples when thinking is enabled — they cause the API
    # to silently disable thinking (no thinking blocks returned).
    messages = [{"role": "user", "content": user_msg}]

    with client.messages.stream(
        model=model,
        max_tokens=16000,
        temperature=1,  # required for extended thinking
        thinking={
            "type": "enabled",
            "budget_tokens": 10000,
        },
        system="You are an expert cryptic crossword solver. Given a clue and its answer, explain how the wordplay works.\n\nBe precise about letter-level mechanics. Show how the letters combine to spell the answer.\n\nKeep your explanation concise.",
        messages=messages,
    ) as stream:
        response = stream.get_final_message()

    explanation = ""
    thinking_text = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            explanation = block.text.strip()

    if thinking_text:
        print("      [thinking: %d chars]" % len(thinking_text))
    else:
        print("      [WARNING: no thinking block returned]")

    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens
    return explanation, tokens_in, tokens_out


def call_structuring(model, clue_text, answer, explanation):
    """Pass 2: Extract structured JSON from free-text explanation."""
    enum_len = len(answer.replace(" ", "").replace("-", ""))
    user_msg = "Clue: %s (%d)\nAnswer: %s\nExplanation: %s" % (
        clue_text, enum_len, answer, explanation)

    messages = _build_structuring_examples() + [{"role": "user", "content": user_msg}]

    response = client.messages.create(
        model=model,
        max_tokens=400,
        temperature=0,
        system=STRUCTURING_PROMPT,
        messages=messages,
    )
    raw = response.content[0].text.strip()
    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens

    parsed = _parse_json_response(raw)
    return parsed, tokens_in, tokens_out


THINKING_PROMPT = """You are an expert cryptic crossword solver. Given a clue and its answer, work out how the wordplay produces the answer.

Use your thinking to reason through the clue. Then output ONLY valid JSON with:
- "definition": the exact substring of the clue that defines the answer
- "wordplay_type": one of: charade, container, anagram, deletion, hidden, reversal, homophone, double_definition, cryptic_definition, acrostic, spoonerism, substitution
- "pieces": array of objects, each with:
  - "clue_word": the word(s) from the clue
  - "letters": the uppercase letters this produces AFTER any operations (deletions, reversals, etc.)
  - "mechanism": synonym, abbreviation, literal, anagram_fodder, first_letter, last_letter, reversal, sound_of, alternate_letters, core_letters, deletion, hidden
- "_reasoning": a one-line summary of the wordplay logic

Rules:
- FUNDAMENTAL: Every word in a cryptic clue has exactly ONE role — it is definition, indicator, link word, or wordplay fodder. A word CANNOT serve two purposes.
- FUNDAMENTAL: The definition words MUST NOT also appear as wordplay fodder. If "Spring's past" is the definition, then "Spring" and "past" CANNOT be pieces. The only exception is double_definition or cryptic_definition where the whole clue IS the definition.
- FUNDAMENTAL: Every non-definition, non-link word in the clue should have a role in the wordplay. If your explanation leaves words unaccounted for, reconsider.
- The pieces' letters, when concatenated (for charade) or assembled via the wordplay_type, MUST spell the full answer.
- Each piece must map to the SMALLEST unit of the clue: split into individual words or short phrases, not lumped together. E.g. "to" and "American" are separate pieces, never "to American" as one piece.
- Indicator words (anagram indicators, reversal indicators, container indicators) are NEVER pieces. They signal the operation but contribute NO letters. Do not include them in any piece's "clue_word".
- For reversals within a charade: use mechanism "reversal" for the reversed piece, not "anagram_fodder". E.g. "to" reversed = OT should be {"clue_word": "to", "letters": "OT", "mechanism": "reversal"}.
- For deletions: show the RESULT after deletion, not the deleted letter. E.g. if CORPSE minus P = CORSE, the piece is {"clue_word": "stiff", "letters": "CORSE", "mechanism": "deletion"}.
- For anagrams: pieces are the raw fodder letters BEFORE rearrangement.
- For containers: show outer and inner pieces separately.
- For hidden words: one piece with the spanning clue words and mechanism "hidden".
- For double/cryptic definitions: no pieces needed.
- Definition is always at the start or end of the clue.

Return ONLY valid JSON, no other text."""


def call_api(model, clue_text, answer, enrichment, example_messages, extra_context=""):
    """Single-pass API call with extended thinking.

    Thinking handles reasoning privately, visible output is structured JSON.
    Eliminates the lossy two-pass reasoning→structuring translation.
    """
    enum_len = len(answer.replace(" ", "").replace("-", ""))
    user_msg = "Clue: %s (%d)\nAnswer: %s" % (clue_text, enum_len, answer)

    messages = [{"role": "user", "content": user_msg}]

    with client.messages.stream(
        model=model,
        max_tokens=16000,
        temperature=1,  # required for extended thinking
        thinking={
            "type": "enabled",
            "budget_tokens": 10000,
        },
        system=THINKING_PROMPT,
        messages=messages,
    ) as stream:
        response = stream.get_final_message()

    thinking_text = ""
    raw_output = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            raw_output = block.text.strip()

    if thinking_text:
        print("      [thinking: %d chars]" % len(thinking_text))
    else:
        print("      [WARNING: no thinking block returned]")

    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens

    parsed = _parse_json_response(raw_output)

    # Attach reasoning from thinking block for debugging/reporting
    if parsed and isinstance(parsed, dict):
        if not parsed.get("_reasoning"):
            # Use first 500 chars of thinking as reasoning summary
            parsed["_reasoning"] = thinking_text[:500] if thinking_text else raw_output[:500]

    return parsed, tokens_in, tokens_out


def extract_pieces(api_output):
    if not api_output:
        return [], None, None
    pieces = []
    for comp in api_output.get("pieces", []):
        letters = clean(comp.get("letters", ""))
        if letters:
            pieces.append(letters)
    return pieces, api_output.get("wordplay_type"), api_output.get("definition")


def extract_literal_fodder(api_output):
    """Extract literal clue words from anagram_fodder pieces as fallback pieces.

    When the model identifies the right clue words but pre-processes the letters
    incorrectly (e.g. "British cut short" -> HUBRISTC instead of BRITISH + CUT),
    this extracts the raw words as individual pieces.
    """
    if not api_output:
        return []
    literals = []
    for comp in api_output.get("pieces", []):
        if comp.get("mechanism") in ("anagram_fodder", "literal"):
            clue_word = comp.get("clue_word", "")
            for word in clue_word.split():
                cleaned = clean(word)
                if cleaned and len(cleaned) >= 2:
                    literals.append(cleaned)
    return literals


def _try_truncation_from_db(clue, answer, target, enricher, homo_engine):
    """Build pieces from DB for truncation clues when AI pieces fail.

    For each non-indicator clue word, find the best synonym/abbreviation.
    Try truncating each synonym (len >= 4) by 1 letter and assemble.
    Returns (assembly, method) or None.
    """
    TRUNC_WORDS = {"short", "shortly", "brief", "briefly", "cut",
                   "curtailed", "clipped", "trimmed", "truncated",
                   "shortened", "cropped", "docked"}
    words = clue.split()
    answer_clean = clean(answer)

    # For each non-indicator word, collect candidate pieces (short list)
    word_pieces = {}  # word -> list of (letters, mechanism)
    for w in words:
        wl = w.lower().strip(".,;:!?\"'()-")
        if not wl or len(wl) < 2 or wl in TRUNC_WORDS:
            continue
        cands = []
        # Abbreviations (usually 1-2 chars — very useful)
        for a in enricher.lookup_abbreviations(wl):
            a_clean = clean(a)
            if a_clean and a_clean in answer_clean:
                cands.append((a_clean, "abbreviation"))
        # Synonyms: only those that are substrings of the answer
        syns = enricher.lookup_synonyms(wl, max_results=50,
                                        max_len=len(answer_clean) + 2,
                                        answer=answer)
        for s in syns:
            if len(s) >= 2 and s in answer_clean:
                cands.append((s, "synonym"))
        # Also keep long synonyms (4+) as truncation sources
        for s in syns:
            if len(s) >= 4 and s not in answer_clean:
                cands.append((s, "trunc_source"))
        # Literal word
        wc = clean(w)
        if wc and wc in answer_clean:
            cands.append((wc, "literal"))
        if cands:
            word_pieces[wl] = cands

    if not word_pieces:
        return None

    # Collect truncation sources: synonyms that, when truncated by 1,
    # could participate in assembly
    trunc_sources = []  # (word, full_synonym, truncated)
    for wl, cands in word_pieces.items():
        for letters, mech in cands:
            if mech == "trunc_source":
                truncated = letters[:-1]
                if len(truncated) >= 3:
                    trunc_sources.append((wl, letters, truncated))

    # For each truncation source, collect pieces from other words and try assembly
    for trunc_word, full_syn, truncated in trunc_sources:
        # Get usable pieces from other words with word-to-piece mapping
        other_items = []  # (letters, word, mechanism)
        seen = set()
        for wl, cands in word_pieces.items():
            if wl == trunc_word:
                continue
            for letters, mech in cands:
                if mech in ("synonym", "abbreviation", "literal") and letters not in seen:
                    other_items.append((letters, wl, mech))
                    seen.add(letters)

        if len(other_items) > 15:
            other_items.sort(key=lambda x: len(x[0]))
            other_items = other_items[:15]

        # Try combinations of 2-5 other pieces + truncated
        for num in range(2, min(6, len(other_items) + 1)):
            for combo in itertools.combinations(other_items, num):
                combo_pieces = [letters for letters, _, _ in combo]
                all_pieces = [truncated] + combo_pieces
                total = sum(len(p) for p in all_pieces)
                if total < len(target) or total > len(target) + 3:
                    continue
                result = assemble(clue, answer, all_pieces,
                                  homo_engine=homo_engine, ai_wtype=None)
                if result:
                    result["truncated"] = {
                        "from": full_syn, "to": truncated, "removed": 1
                    }
                    result["source"] = "truncation_from_db"
                    # Build pieces_detail for report
                    detail = [(trunc_word, truncated, "truncation")]
                    for letters, word, mech in combo:
                        detail.append((word, letters, mech))
                    result["pieces_detail"] = detail
                    return result, "truncation_db"

    return None


# -- Full assembly attempt with fallbacks --------------------------------------

def _validate_double_definition(clue, answer, enricher):
    """Validate DD by checking two separate clue windows both define the answer.

    Splits the clue at every possible point. For each split, generates
    definition windows from each half independently. If both halves
    produce the answer via DB lookup with at most 1 uncovered link word,
    it's a valid DD. Returns assembly dict or None.
    """
    # Strip enumeration
    text = re.sub(r'\(\d+(?:,\d+)*\)\s*$', '', clue).strip()
    words = text.split()
    if len(words) < 2:
        return None

    for split_point in range(1, len(words)):
        left_words = words[:split_point]
        right_words = words[split_point:]

        # Generate windows from each half (sub-phrases anchored to the split boundary)
        left_windows = []
        for i in range(len(left_words)):
            left_windows.append(" ".join(left_words[i:]))  # right-anchored
        right_windows = []
        for i in range(len(right_words)):
            right_windows.append(" ".join(right_words[:i + 1]))  # left-anchored

        # Check if any window from each half defines the answer
        best_left = None
        for w in left_windows:
            phrase = w.lower().strip(".,;:!?\"'()-")
            if enricher.lookup_definition(phrase, answer):
                best_left = w
                break
        if not best_left:
            continue

        best_right = None
        for w in right_windows:
            phrase = w.lower().strip(".,;:!?\"'()-")
            if enricher.lookup_definition(phrase, answer):
                best_right = w
                break
        if not best_right:
            continue

        # Coverage check: at most 1 uncovered link word
        left_covered = len(best_left.split())
        right_covered = len(best_right.split())
        uncovered = (split_point - left_covered) + (len(right_words) - right_covered)
        if uncovered > 1:
            continue

        return {
            "op": "double_definition",
            "left_window": best_left,
            "right_window": best_right,
            "split_point": split_point,
        }

    return None


def full_assembly_attempt(clue, answer, pieces, wtype, enricher, homo_engine, target):
    """Try assembling pieces with all fallback strategies."""
    # Cryptic definition — no mechanical validation possible, accept AI's label
    if wtype == "cryptic_definition":
        return {"op": wtype}, "dd_cd"

    # Double definition — validate that two separate windows both define the answer
    if wtype == "double_definition":
        dd_asm = _validate_double_definition(clue, answer, enricher)
        if dd_asm:
            return dd_asm, "dd_cd"
        # DD validation failed — still return DD assembly (unvalidated) so that
        # evidence is preserved. Scoring handles it: no validated windows = low score.
        # Fall-through strategies can still override if they produce a better result.
        # But first, try other strategies — if they work, use them instead.

    # Hidden word check FIRST — if the answer is contiguously hidden spanning
    # multiple clue words, that's a guaranteed correct match and takes priority
    # over any AI-suggested assembly (e.g. DISCO hidden in "legendISCOunting")
    hidden = try_hidden(clue, target)
    if hidden:
        return hidden, "hidden"

    # First attempt: restricted by AI type
    assembly = assemble(clue, answer, pieces, homo_engine=homo_engine,
                        ai_wtype=wtype) if pieces else None

    # Reject degenerate case: single piece that equals the answer
    # (model just echoed the answer instead of breaking it down)
    if assembly and len(pieces) == 1 and pieces[0] == target:
        assembly = None

    if not assembly:
        spoon = try_spoonerism(clue, target, enricher)
        if spoon:
            return spoon, "spoonerism"

    # Progressive truncation: when "short/brief/cut" indicator is present,
    # try removing 1-2 letters from the end of the longest piece and re-assemble
    # e.g. SISTER (nurse) → SISTE (short nurse), then container/charade works
    TRUNC_INDICATORS = {"short", "shortly", "brief", "briefly", "cut",
                        "curtailed", "clipped", "trimmed", "truncated",
                        "shortened", "cropped", "docked"}
    clue_words_lower = {w.lower().strip(".,;:!?\"'()-") for w in clue.split()}
    has_trunc = bool(clue_words_lower & TRUNC_INDICATORS)

    # First try DB-backed truncation: build pieces from DB lookups with one
    # synonym truncated by 1. This produces correct pieces even when the AI
    # returns garbage (e.g. nurse→SIS instead of nurse→SISTER).
    if not assembly and has_trunc and enricher:
        result = _try_truncation_from_db(clue, answer, target, enricher, homo_engine)
        if result:
            return result

    # Then try truncating AI pieces directly
    if not assembly and pieces and has_trunc:
            longest_idx = max(range(len(pieces)), key=lambda i: len(pieces[i]))
            longest = pieces[longest_idx]
            if len(longest) >= 4:
                for trim in range(1, min(3, len(longest) - 2)):
                    truncated = longest[:-trim]
                    trial = pieces[:longest_idx] + [truncated] + pieces[longest_idx+1:]
                    result = assemble(clue, answer, trial, homo_engine=homo_engine,
                                      ai_wtype=wtype)
                    if result:
                        result["truncated"] = {"from": longest, "to": truncated, "removed": trim}
                        return result, "truncation"

    if not assembly and len(pieces) >= 2:
        for skip in range(len(pieces)):
            subset = pieces[:skip] + pieces[skip+1:]
            assembly = assemble(clue, answer, subset, homo_engine=homo_engine,
                                ai_wtype=wtype)
            if assembly:
                assembly["note"] = "dropped piece %d" % skip
                return assembly, "drop_piece"

    # Also try truncation AFTER dropping pieces
    # (AI may return extra junk pieces alongside the correct ones)
    # Loop order: trim=1 across ALL drop combos first, then trim=2.
    # This prefers minimal truncation (SISTE over SIST).
    if not assembly and pieces and len(pieces) >= 3 and has_trunc:
        # Drop 1 piece + truncate
        for trim in range(1, 3):
            for skip in range(len(pieces)):
                subset = pieces[:skip] + pieces[skip+1:]
                longest_idx = max(range(len(subset)), key=lambda i: len(subset[i]))
                longest = subset[longest_idx]
                if len(longest) >= 4 and trim < len(longest) - 1:
                    truncated = longest[:-trim]
                    trial = subset[:longest_idx] + [truncated] + subset[longest_idx+1:]
                    result = assemble(clue, answer, trial, homo_engine=homo_engine, ai_wtype=wtype)
                    if result:
                        result["truncated"] = {"from": longest, "to": truncated, "removed": trim}
                        result["note"] = "dropped piece %d" % skip
                        return result, "truncation+drop"
        # Drop 2 pieces + truncate (for clues with extra junk pieces)
        if len(pieces) >= 5:
            for trim in range(1, 3):
                for i in range(len(pieces)):
                    for j in range(i + 1, len(pieces)):
                        subset = [p for k, p in enumerate(pieces) if k != i and k != j]
                        longest_idx = max(range(len(subset)), key=lambda k: len(subset[k]))
                        longest = subset[longest_idx]
                        if len(longest) >= 4 and trim < len(longest) - 1:
                            truncated = longest[:-trim]
                            trial = subset[:longest_idx] + [truncated] + subset[longest_idx+1:]
                            result = assemble(clue, answer, trial, homo_engine=homo_engine, ai_wtype=wtype)
                            if result:
                                result["truncated"] = {"from": longest, "to": truncated, "removed": trim}
                                return result, "truncation+drop2"

    if not assembly and pieces:
        assembly = try_gap_fill(clue, answer, pieces, enricher, target, ai_wtype=wtype)
        if assembly:
            return assembly, "gap_fill"

    if not assembly and pieces:
        total_len = sum(len(p) for p in pieces)
        if total_len == len(target) - 1:
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                result = assemble(clue, answer, pieces + [letter], homo_engine=homo_engine, ai_wtype=wtype)
                if result:
                    result["brute_gap"] = letter
                    return result, "brute_1letter"

    if not assembly and pieces:
        assembly = try_substitution(pieces, target, clue, enricher)
        if assembly:
            return assembly, "substitution"

    if assembly:
        return assembly, "direct"

    # Last resort: if AI said DD but validation failed and no other strategy
    # worked, return an unvalidated DD so evidence is preserved (scores low).
    if wtype == "double_definition":
        return {"op": "double_definition"}, "dd_cd"

    return None, None


# -- Substitution operation ----------------------------------------------------

def try_substitution(pieces, target, clue_text, enricher):
    """Try substitution: delete letters from a piece, replace with letters from clue words.

    Handles patterns like "deer leaving motorway for lake":
    MOOSE - M (motorway) + L (lake) → LOOSE
    """
    if not pieces:
        return None

    # Gather short yields (abbreviations, short synonyms, first letters) for each clue word
    # Priority: abbreviations > synonyms > first letters (for better add_word attribution)
    words = clue_text.split()
    word_yields = {}
    for w in words:
        w_lower = w.lower().strip(".,;:!?\"'()-")
        if not w_lower:
            continue
        yields = []  # ordered list — abbreviations first
        seen = set()
        for a in enricher.lookup_abbreviations(w_lower):
            a_clean = clean(a)
            if a_clean and 1 <= len(a_clean) <= 3 and a_clean not in seen:
                yields.append(a_clean)
                seen.add(a_clean)
        syns = enricher.lookup_synonyms(w_lower, max_results=10, max_len=3, answer="")
        for s in syns:
            if 1 <= len(s) <= 3 and s not in seen:
                yields.append(s)
                seen.add(s)
        w_clean = clean(w)
        if w_clean and w_clean[0] not in seen:
            yields.append(w_clean[0])
        if yields:
            word_yields[w_lower] = yields

    # Invert: letters -> list of source words
    # Words where the letter is an abbreviation come first (more authoritative)
    all_yields = {}
    for w, yields in word_yields.items():
        abbr_set = set()
        for a in enricher.lookup_abbreviations(w):
            a_clean = clean(a)
            if a_clean:
                abbr_set.add(a_clean)
        for y in yields:
            lst = all_yields.setdefault(y, [])
            if y in abbr_set:
                lst.insert(0, w)  # abbreviation source first
            else:
                lst.append(w)

    # Strategy 1: One Sonnet piece IS the deletion target inside another piece
    # e.g. pieces=[MOOSE, M] — delete M from MOOSE, add L (lake) → LOOSE
    for i, piece in enumerate(pieces):
        if len(piece) < 3:
            continue
        for j, del_piece in enumerate(pieces):
            if i == j or len(del_piece) < 1 or len(del_piece) > 3:
                continue
            pos = 0
            while True:
                idx = piece.find(del_piece, pos)
                if idx < 0:
                    break
                pos = idx + 1
                shortened = piece[:idx] + piece[idx + len(del_piece):]
                if not shortened:
                    continue
                other = [p for k, p in enumerate(pieces) if k != i and k != j]
                needed = len(target) - len(shortened) - sum(len(p) for p in other)
                if needed < 1 or needed > 3:
                    continue
                for add_str, add_words in all_yields.items():
                    if len(add_str) != needed:
                        continue
                    # Try inserting at the deletion position (true substitution)
                    substituted = piece[:idx] + add_str + piece[idx + len(del_piece):]
                    trial_sub = [substituted] + other
                    if "".join(trial_sub) == target:
                        return {
                            "op": "substitution",
                            "from": piece, "deleted": del_piece,
                            "added": add_str, "add_word": add_words[0],
                            "gives": substituted,
                            "order": trial_sub,
                        }
                    # Also try as separate charade pieces
                    trial = [shortened] + other + [add_str]
                    if len(trial) > 6:
                        continue
                    for perm in itertools.permutations(trial):
                        if "".join(perm) == target:
                            return {
                                "op": "substitution",
                                "from": piece, "deleted": del_piece,
                                "added": add_str, "add_word": add_words[0],
                                "gives": shortened,
                                "order": list(perm),
                            }

    # Strategy 2: Deletion target not in pieces — try all clue word yields as deletion
    for i, piece in enumerate(pieces):
        if len(piece) < 3:
            continue
        remaining = [p for j, p in enumerate(pieces) if j != i]
        remaining_len = sum(len(p) for p in remaining)

        for del_str, del_words in all_yields.items():
            if len(del_str) < 1 or len(del_str) > 3:
                continue
            pos = 0
            while True:
                idx = piece.find(del_str, pos)
                if idx < 0:
                    break
                pos = idx + 1
                shortened = piece[:idx] + piece[idx + len(del_str):]
                if not shortened:
                    continue
                needed = len(target) - len(shortened) - remaining_len
                if needed < 1 or needed > 3:
                    continue
                for add_str, add_words in all_yields.items():
                    if len(add_str) != needed:
                        continue
                    if add_words[0] == del_words[0]:
                        continue
                    # Try inserting at the deletion position (true substitution)
                    substituted = piece[:idx] + add_str + piece[idx + len(del_str):]
                    trial_sub = [substituted] + remaining
                    if "".join(trial_sub) == target:
                        return {
                            "op": "substitution",
                            "from": piece, "deleted": del_str,
                            "del_word": del_words[0],
                            "added": add_str, "add_word": add_words[0],
                            "gives": substituted,
                            "order": trial_sub,
                        }
                    # Also try as separate charade pieces
                    trial = [shortened] + remaining + [add_str]
                    if len(trial) > 6:
                        continue
                    for perm in itertools.permutations(trial):
                        if "".join(perm) == target:
                            return {
                                "op": "substitution",
                                "from": piece, "deleted": del_str,
                                "del_word": del_words[0],
                                "added": add_str, "add_word": add_words[0],
                                "gives": shortened,
                                "order": list(perm),
                            }

    return None


# -- Mechanism checker (post-assembly validation) ------------------------------

OP_TO_TYPE = {
    "charade": "charade",
    "container": "container",
    "reversal": "reversal",
    "deletion": "deletion",
    "outer_deletion": "deletion",
    "anagram": "anagram",
    "charade+anagram": "anagram",
    "anagram+charade": "anagram",
    "homophone": "homophone",
    "hidden": "hidden",
    "hidden_in_word": "hidden",
    "hidden_reversed": "hidden",
    "deletion+anagram": "anagram",
    "double_definition": "double_definition",
    "cryptic_definition": "cryptic_definition",
    "reversal_container": "container",
    "container_reversal": "reversal",
    "substitution": "substitution",
    "spoonerism": "spoonerism",
}


def _try_decompose_piece(words, target_letters, enricher):
    """Try to decompose target_letters into per-word contributions.

    For each word, checks synonyms and abbreviations. Returns a list of
    replacement piece dicts if all words contribute and letters concatenate
    to target_letters, else None.
    """
    def _candidates(word):
        cands = []
        for s in enricher.lookup_synonyms(word, max_results=50):
            cands.append((clean(s), "synonym"))
        for a in enricher.lookup_abbreviations(word):
            cands.append((clean(a), "abbreviation"))
        return cands

    def _decompose(pos, word_idx):
        if pos == len(target_letters) and word_idx == len(words):
            return []
        if pos >= len(target_letters) or word_idx >= len(words):
            return None
        for letters, mech in _candidates(words[word_idx]):
            if not letters:
                continue
            if target_letters[pos:pos + len(letters)] == letters:
                rest = _decompose(pos + len(letters), word_idx + 1)
                if rest is not None:
                    return [{"clue_word": words[word_idx], "letters": letters,
                             "mechanism": mech}] + rest
        return None

    return _decompose(0, 0)


def refine_pieces(ai_output, clue_text, enricher):
    """Post-process AI pieces to add deletion/truncation detail.

    Handles three patterns Sonnet often gets wrong:
    1. Multi-word "synonym" pieces that are really truncation/deletion:
       e.g. "Short emperor" → NER  should be  emperor → NERO, short → truncate, NER
    2. Bare "deletion" pieces missing source/deleted detail:
       e.g. "lady" → OMAN  should show  WOMAN - W(whiskey) = OMAN
    3. Multi-word "synonym" pieces that should be split into individual pieces:
       e.g. "on loch" → REL  should be  RE(on) + L(loch)
    """
    if not ai_output:
        return
    pieces = ai_output.get("pieces", [])
    if not pieces:
        return

    clue_words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", clue_text.lower())

    # Track pieces that should be replaced with multiple pieces (for Case 3)
    replacements = {}  # idx -> list of replacement piece dicts

    for idx, p in enumerate(pieces):
        mech = p.get("mechanism", "")
        clue_word = (p.get("clue_word") or "").strip()
        letters = clean(p.get("letters") or "")
        if not clue_word or not letters:
            continue

        # Case 1: "synonym" piece with multi-word clue_word — check for
        # truncation pattern like "Short emperor" → NER (really NERO truncated)
        if mech == "synonym":
            words_in_piece = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", clue_word.lower())
            if len(words_in_piece) >= 2:
                found = False
                for i, w in enumerate(words_in_piece):
                    indicators = enricher.lookup_indicators(w)
                    if "deletion" not in indicators and "parts" not in indicators:
                        continue
                    # This word is a deletion/truncation indicator
                    remaining = [words_in_piece[j] for j in range(len(words_in_piece)) if j != i]
                    for r_word in remaining:
                        syns = enricher.lookup_synonyms(r_word, max_results=200)
                        for s in syns:
                            sc = clean(s)
                            if len(sc) <= len(letters) or sc == letters:
                                continue
                            # Check truncation: letters is prefix or suffix of synonym
                            if sc.startswith(letters):
                                deleted = sc[len(letters):]
                                p["mechanism"] = "deletion"
                                p["clue_word"] = r_word
                                p["source"] = s.upper()
                                p["indicator"] = w
                                p["deleted"] = deleted.upper()
                                found = True
                                break
                            elif sc.endswith(letters):
                                deleted = sc[:len(sc) - len(letters)]
                                p["mechanism"] = "deletion"
                                p["clue_word"] = r_word
                                p["source"] = s.upper()
                                p["indicator"] = w
                                p["deleted"] = deleted.upper()
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

                # Case 3: If truncation didn't match, try decomposing into
                # individual word contributions
                # e.g. "on loch" → REL  becomes  RE(on) + L(loch)
                if not found:
                    decomp = _try_decompose_piece(words_in_piece, letters, enricher)
                    if decomp:
                        replacements[idx] = decomp

        # Case 2: "deletion" piece without source detail — find what synonym
        # was used and what was deleted
        # e.g. "lady" → OMAN: find WOMAN(lady) - W(whiskey) = OMAN
        # Also handles truncation: "emperor" → NER: NERO - O, "short" = indicator
        elif mech == "deletion" and "source" not in p:
            syns = enricher.lookup_synonyms(clue_word.lower(), max_results=200)
            for s in syns:
                sc = clean(s)
                if len(sc) <= len(letters) or sc == letters:
                    continue
                # Check if removing some substring from sc leaves letters
                if sc.endswith(letters):
                    deleted_part = sc[:len(sc) - len(letters)]
                elif sc.startswith(letters):
                    deleted_part = sc[len(letters):]
                else:
                    continue
                if len(deleted_part) < 1 or len(deleted_part) > 3:
                    continue
                # First: check if deleted_part is an abbreviation of another clue word
                for cw in clue_words:
                    if cw != clue_word.lower():
                        abbrs = [clean(a) for a in enricher.lookup_abbreviations(cw)]
                        if deleted_part in abbrs:
                            p["source"] = s.upper()
                            p["deleted"] = deleted_part.upper()
                            p["deleted_word"] = cw
                            break
                if "source" in p:
                    break
                # Second: check for truncation via deletion indicator in clue
                for cw in clue_words:
                    if cw != clue_word.lower():
                        indicators = enricher.lookup_indicators(cw)
                        if "deletion" in indicators or "parts" in indicators:
                            p["source"] = s.upper()
                            p["indicator"] = cw
                            p["deleted"] = deleted_part.upper()
                            break
                if "source" in p:
                    break

    # Apply piece replacements (Case 3: split multi-word pieces)
    if replacements:
        new_pieces = []
        for idx, p in enumerate(pieces):
            if idx in replacements:
                new_pieces.extend(replacements[idx])
            else:
                new_pieces.append(p)
        ai_output["pieces"] = new_pieces


def check_mechanism(clue_text, answer, ai_output, assembly, enricher, tier,
                    enrichment="", ref_db=None):
    """Score how useful this result is to a user seeking progressive hints.

    Scoring reflects user value:
      Definition (30pts)  — "What am I looking for?"
      Wordplay type (20pts) — "What technique is used?"
      Explanation (50pts) — "How do the pieces work?"
        - yields check (20) — pieces mechanically produce the answer
        - pieces validated (15) — each piece traces to a real synonym/abbreviation
        - fodder in clue (10) — source words actually appear in the clue
        - base assembled (5) — we have something to show

    Returns score 0 for failed assembly. Callers use None for not-attempted.
    """
    if not assembly:
        return {"confidence": "none", "score": 0, "checks": {}}

    checks = {}
    score = 5  # base: we assembled something
    answer_clean = clean(answer)

    if ai_output:
        ai_type = ai_output.get("wordplay_type")
        ai_def = ai_output.get("definition")
        ai_pieces = ai_output.get("pieces", [])
    else:
        ai_type = None
        ai_def = None
        ai_pieces = []

    asm_op = assembly.get("op", "")
    asm_type = OP_TO_TYPE.get(asm_op, asm_op)

    # --- Definition (30pts) ---
    if ai_def:
        def_clean = ai_def.lower().strip(".,;:!?\"'()-")
        # Check definition_answers_augmented first, then synonyms_pairs as fallback
        if enricher.lookup_definition(def_clean, answer):
            checks["definition"] = "confirmed in DB"
            score += 30
        elif answer_clean in [s.upper() for s in enricher.lookup_synonyms(def_clean, max_results=200)]:
            checks["definition"] = "confirmed in DB (synonym)"
            score += 30
        else:
            clue_lower = clue_text.lower()
            if clue_lower.startswith(def_clean) or clue_lower.endswith(def_clean):
                checks["definition"] = "position OK (start/end of clue)"
                score += 15
            else:
                checks["definition"] = "not in DB, odd position"
                score += 5
    else:
        checks["definition"] = "none identified"

    # --- Wordplay type (20pts, evidence-based) ---
    # Points awarded based on convergent evidence, not just having a label.
    #   Base for having a type:           +5
    #   AI and assembler agree:           +5
    #   Indicator word for type in DB:    +5
    #   AI reasoning text supports type:  +5
    type_score = 0
    type_evidence = []

    COMPATIBLE_TYPES = {
        "reversal_container": {"container", "reversal"},
        "container_reversal": {"container", "reversal"},
        "charade+anagram": {"charade", "anagram"},
        "anagram+charade": {"charade", "anagram"},
        "deletion+anagram": {"deletion", "anagram"},
    }

    if asm_type:
        checks["wordplay_type"] = asm_type
        type_score += 5
        type_evidence.append("assembled")

        # Evidence 1: AI agrees with assembler
        if ai_type:
            compat_set = COMPATIBLE_TYPES.get(asm_type, set())
            if ai_type == asm_type or ai_type in compat_set:
                type_score += 5
                type_evidence.append("AI agrees")
            else:
                checks["type_mismatch"] = "AI=%s, assembled=%s" % (ai_type, asm_type)
                score -= 10  # disagreement penalty on top of missed bonus

        # Evidence 2: Indicator word for this type in DB enrichment
        if enrichment:
            # Map assembled type to indicator DB labels
            type_to_ind = {
                "anagram": {"anagram"},
                "container": {"container", "insertion"},
                "hidden": {"hidden"},
                "reversal": {"reversal"},
                "deletion": {"deletion"},
                "homophone": {"homophone"},
                "acrostic": {"acrostic"},
                "charade+anagram": {"anagram"},
                "anagram+charade": {"anagram"},
                "deletion+anagram": {"anagram", "deletion"},
                "reversal_container": {"container", "reversal"},
                "container_reversal": {"container", "reversal"},
                "substitution": {"deletion"},
            }
            ind_labels = type_to_ind.get(asm_type, set())
            if ind_labels:
                ind_hints = _extract_indicator_hints(enrichment)
                ind_types_found = {h.split(": ")[1] for h in ind_hints if ": " in h}
                if ind_labels & ind_types_found:
                    type_score += 5
                    type_evidence.append("indicator in DB")

        # Evidence 3: AI reasoning text contains type-specific language
        reasoning = (ai_output or {}).get("_reasoning", "")
        if reasoning:
            reasoning_lower = reasoning.lower()
            TYPE_KEYWORDS = {
                "anagram": ["anagram", "rearrange", "scramble", "mixed", "shuffle"],
                "container": ["inside", "within", "around", "containing", "goes in",
                              "placed in", "wrapping", "surrounding"],
                "hidden": ["hidden in", "concealed", "lurking", "embedded",
                           "contained within"],
                "reversal": ["reverse", "reversed", "back", "returned", "flipped",
                             "reflected"],
                "deletion": ["remove", "without", "minus", "drop", "losing",
                             "deleted", "shed"],
                "homophone": ["sounds like", "heard", "spoken", "pronounced",
                              "audibly", "say"],
                "charade": ["followed by", "next to", "then", "after", "before",
                            "plus"],
                "double_definition": ["double definition", "two meanings",
                                      "two definitions"],
                "cryptic_definition": ["cryptic definition", "whole clue",
                                       "entire clue", "playful"],
                "acrostic": ["first letter", "initial", "leading letter",
                             "head of"],
                "substitution": ["replace", "substitut", "swap", "exchange"],
            }
            # Check assembled type and its compatible parent types
            check_types = {asm_type} | COMPATIBLE_TYPES.get(asm_type, set())
            for ct in check_types:
                keywords = TYPE_KEYWORDS.get(ct, [])
                if any(kw in reasoning_lower for kw in keywords):
                    type_score += 5
                    type_evidence.append("reasoning supports")
                    break

    else:
        checks["wordplay_type"] = "unknown"

    checks["type_evidence"] = ", ".join(type_evidence) if type_evidence else "none"
    score += type_score

    # Penalty: anagram fallback used despite AI suggesting a different type
    if assembly.get("anagram_fallback"):
        checks["anagram_fallback"] = "AI suggested %s, fell back to anagram" % ai_type
        score -= 15

    # --- Explanation: Yields check (20pts) ---
    YIELDS_OPS = {
        "charade", "container", "merged_container", "reversal", "anagram",
        "reversal_container", "container_reversal",
        "charade+anagram", "anagram+charade",
    }
    if asm_op in YIELDS_OPS and ai_pieces:
        piece_letters = "".join(
            clean(p.get("letters") or "") for p in ai_pieces
        )
        brute_gap = clean(assembly.get("brute_gap", "") or "")
        piece_letters += brute_gap
        for _, yld, _ in assembly.get("gap_fill", []):
            piece_letters += clean(yld)
        if sorted(piece_letters) == sorted(answer_clean):
            checks["yields_check"] = "pass"
            score += 20
        else:
            checks["yields_check"] = "FAIL: pieces=%s answer=%s" % (
                piece_letters, answer_clean)
    elif asm_op in ("hidden", "hidden_in_word", "hidden_reversed"):
        # Hidden words are self-evidently correct — the answer is in the clue
        checks["yields_check"] = "hidden (self-evident)"
        score += 20
    elif asm_op in ("deletion", "outer_deletion", "deletion+anagram"):
        # Deletion: source minus deleted letters should yield the answer
        source_word = clean(assembly.get("from", "") or "")
        deleted = clean(assembly.get("deleted", "") or "")
        gives = clean(assembly.get("gives", "") or "")
        order = assembly.get("order", [])
        if source_word and gives:
            # Check: remaining letters after deletion, combined with other pieces
            combined = "".join(clean(p) for p in order) if order else gives
            if combined == answer_clean or sorted(combined) == sorted(answer_clean):
                checks["yields_check"] = "pass (deletion verified)"
                score += 20
            else:
                checks["yields_check"] = "FAIL: deletion gives=%s answer=%s" % (combined, answer_clean)
        elif source_word and deleted:
            # Fallback: just check source - deleted = answer
            remaining = source_word
            for ch in deleted:
                remaining = remaining.replace(ch, "", 1)
            if remaining == answer_clean:
                checks["yields_check"] = "pass (deletion verified)"
                score += 20
            else:
                checks["yields_check"] = "FAIL: %s - %s = %s, expected %s" % (
                    source_word, deleted, remaining, answer_clean)
        else:
            checks["yields_check"] = "deletion: insufficient assembly data"
    elif asm_op == "double_definition":
        # DD validated by _validate_double_definition — both windows confirmed in DB
        if assembly.get("left_window") and assembly.get("right_window"):
            checks["yields_check"] = "DD validated: '%s' + '%s' both define %s" % (
                assembly["left_window"], assembly["right_window"], answer)
            score += 20
        else:
            checks["yields_check"] = "n/a for double_definition"
    elif asm_op == "homophone":
        # Homophone: the assembler found a sounds-like match
        sounds_like = clean(assembly.get("sounds_like", "") or "")
        if sounds_like:
            checks["yields_check"] = "pass (homophone: %s sounds like %s)" % (sounds_like, answer_clean)
            score += 20
        else:
            checks["yields_check"] = "homophone: insufficient assembly data"
    elif asm_op == "substitution":
        # Substitution: one piece replaced by another within a word
        checks["yields_check"] = "pass (substitution assembled)"
        score += 20
    elif asm_op == "spoonerism":
        checks["yields_check"] = "pass (spoonerism assembled)"
        score += 20
    elif asm_op == "cryptic_definition":
        # CD is inherently unverifiable — no free points
        checks["yields_check"] = "n/a for cryptic_definition"

    # --- Explanation: Pieces validated (15pts) ---
    validated_pieces = 0
    unverified_syns = 0
    total_pieces = 0
    for p in ai_pieces:
        mech = p.get("mechanism", "")
        clue_word = (p.get("clue_word") or "").lower().strip()
        letters = clean(p.get("letters") or "")
        if not clue_word or not letters:
            continue
        total_pieces += 1

        if mech == "synonym":
            syns = enricher.lookup_synonyms(clue_word, max_results=200)
            if letters in syns:
                validated_pieces += 1
            else:
                unverified_syns += 1
        elif mech == "abbreviation":
            abbrs = enricher.lookup_abbreviations(clue_word)
            if letters in [clean(a) for a in abbrs]:
                validated_pieces += 1
            else:
                unverified_syns += 1
        elif mech in ("literal", "anagram_fodder"):
            if clean(clue_word) == letters or letters in clean(clue_word):
                validated_pieces += 1
        elif mech == "first_letter":
            if clean(clue_word) and clean(clue_word)[0] == letters:
                validated_pieces += 1
        elif mech == "last_letter":
            if clean(clue_word) and clean(clue_word)[-1] == letters:
                validated_pieces += 1
        elif mech == "deletion" and p.get("source"):
            # Refined deletion piece: source synonym already validated
            validated_pieces += 1
        elif mech == "hidden":
            # Hidden words: the letters are literally in the clue text
            if letters in clean(clue_text):
                validated_pieces += 1
        else:
            validated_pieces += 0.5

    if total_pieces > 0:
        piece_pct = validated_pieces / total_pieces
        piece_score = int(15 * piece_pct)
        checks["pieces_validated"] = "%d/%d (%.0f%%)" % (
            validated_pieces, total_pieces, 100 * piece_pct)
        score += piece_score
    elif asm_op == "double_definition":
        checks["pieces_validated"] = "n/a for double_definition"
    elif asm_op == "cryptic_definition":
        checks["pieces_validated"] = "n/a for cryptic_definition"
    elif asm_op in ("hidden", "hidden_in_word", "hidden_reversed"):
        checks["pieces_validated"] = "n/a for hidden"
        score += 10
    else:
        checks["pieces_validated"] = "no pieces to validate"

    # --- Penalty: unverified synonym/abbreviation claims ---
    if unverified_syns > 0:
        syn_penalty = unverified_syns * 20
        checks["unverified_claims"] = "%d synonym/abbr not in DB" % unverified_syns
        score -= syn_penalty

    # --- Penalty: nonsense words in pieces (not real English words) ---
    if ref_db is not None:
        nonsense_pieces = []
        for p in ai_pieces:
            mech = p.get("mechanism", "")
            letters = clean(p.get("letters") or "")
            # Only check pieces that claim to be words (synonym, abbreviation)
            # Skip: literal/anagram_fodder (raw letters), first/last_letter (single chars),
            # hidden (span of clue text), reversal (checked as its source word),
            # deletion (fragments like CROS from CROSS are valid, not standalone words)
            if mech not in ("synonym", "abbreviation"):
                continue
            # Short abbreviations (1-2 chars) are expected to not be words
            if mech == "abbreviation" and len(letters) <= 2:
                continue
            if letters and not ref_db.is_real_word(letters):
                nonsense_pieces.append(letters)
        if nonsense_pieces:
            nonsense_penalty = len(nonsense_pieces) * 60
            checks["nonsense_words"] = "not real words: %s" % ", ".join(nonsense_pieces)
            score -= nonsense_penalty

    # --- Explanation: Fodder in clue (10pts) ---
    clue_lower = clue_text.lower()
    fodder_ok = True
    fodder_issues = []
    for p in ai_pieces:
        clue_word = (p.get("clue_word") or "").lower().strip()
        if clue_word and clue_word not in clue_lower:
            clue_stripped = re.sub(r"[^a-z ]", "", clue_lower)
            word_stripped = re.sub(r"[^a-z ]", "", clue_word)
            if word_stripped not in clue_stripped:
                fodder_ok = False
                fodder_issues.append(clue_word)
    if fodder_ok:
        checks["fodder_in_clue"] = "all present"
        score += 10
    else:
        checks["fodder_in_clue"] = "missing: %s" % fodder_issues

    # --- Penalty: gap fill or brute gap (unexplained letters) ---
    gap_fill = assembly.get("gap_fill", [])
    brute_gap = assembly.get("brute_gap", "")
    unexplained = len(gap_fill) + len(clean(brute_gap or ""))
    if unexplained:
        penalty = min(unexplained * 10, 25)
        checks["gap_fill"] = "%d unexplained letter(s)" % unexplained
        score -= penalty

    # --- Penalty: circular assembly (answer used in own explanation) ---
    circular = False
    for p in ai_pieces:
        if clean(p.get("letters") or "") == answer_clean:
            circular = True
            break
    if not circular and assembly.get("from"):
        if clean(assembly["from"]) == answer_clean:
            circular = True
    if circular:
        checks["circular_assembly"] = "answer used in own explanation"
        score = min(score, 65)

    # --- Penalty: single piece producing full answer (no real breakdown) ---
    if (asm_op not in ("double_definition", "cryptic_definition", "hidden", "hidden_in_word", "hidden_reversed")
            and total_pieces == 1 and ai_pieces):
        solo_letters = clean(ai_pieces[0].get("letters") or "")
        if solo_letters == answer_clean:
            checks["single_piece_answer"] = "single piece equals full answer"
            score -= 20

    # --- Penalty: zero pieces validated ---
    if total_pieces > 0 and validated_pieces == 0:
        checks["zero_validation"] = "no pieces verified in DB"
        score -= 15

    # --- Penalty: AI expressed uncertainty in reasoning ---
    reasoning = (ai_output or {}).get("_reasoning", "")
    if reasoning:
        reasoning_lower = reasoning.lower()
        hedge_phrases = [
            "likely", "probably", "perhaps", "might be", "could be",
            "not sure", "unclear", "not certain", "i think",
            "let me reconsider", "rethinking", "wait,",
            "this doesn't work", "doesn't work cleanly",
            "still not", "not enough",
        ]
        hedge_count = sum(1 for h in hedge_phrases if h in reasoning_lower)
        if hedge_count >= 3:
            checks["ai_uncertain"] = "AI expressed significant uncertainty (%d hedges)" % hedge_count
            score -= 15
        elif hedge_count >= 1:
            checks["ai_uncertain"] = "AI expressed some uncertainty (%d hedges)" % hedge_count
            score -= 5

    # --- Floor: hidden words are mechanically verified ---
    # If the assembler found a hidden word and the answer is literally in the clue,
    # guarantee HIGH confidence — this is as certain as it gets.
    if asm_op in ("hidden", "hidden_in_word", "hidden_reversed"):
        if answer_clean in clean(clue_text):
            score = max(score, 80)
            checks["hidden_verified"] = "answer confirmed in clue text"

    # Final confidence bands
    if score >= 70:
        confidence = "high"
    elif score >= 40:
        confidence = "medium"
    else:
        confidence = "low"

    return {"confidence": confidence, "score": score, "checks": checks}


# -- DB gap extraction ---------------------------------------------------------

def extract_db_gaps(results, enricher):
    """Extract DB gaps from high-confidence results.

    Returns (auto_inserts, suggestions):
      auto_inserts: synonyms_pairs entries safe to insert automatically
        - synonyms only, min 3 letters each side, high confidence, no type mismatch
      suggestions: abbreviation/wordplay entries for user review (not auto-inserted)
    """
    auto_inserts = []
    suggestions = []
    for r in results:
        if r.get("status") != "ASSEMBLED":
            continue
        if r.get("score", 0) < 80:
            continue
        if r.get("checks", {}).get("type_mismatch"):
            continue

        ai = r.get("ai_output") or {}
        answer_clean = clean(r.get("answer", ""))

        for p in ai.get("pieces", []):
            mech = p.get("mechanism", "")
            clue_word = (p.get("clue_word") or "").strip()
            letters = clean(p.get("letters") or "")

            if not clue_word or not letters:
                continue
            if mech not in ("synonym", "abbreviation"):
                continue
            if letters == answer_clean:
                continue
            if letters not in answer_clean:
                continue
            if " " in clue_word.strip():
                continue

            clue_lower = clue_word.lower().strip(".,;:!?\"'()-")
            if not clue_lower:
                continue

            if mech == "synonym":
                if len(clue_lower) < 3 or len(letters) < 3:
                    continue
                existing = enricher.lookup_synonyms(clue_lower, max_results=200)
                if letters in existing:
                    continue
                auto_inserts.append(("synonyms_pairs", clue_lower, letters, r["answer"]))
            elif mech == "abbreviation":
                existing = enricher.lookup_abbreviations(clue_lower)
                if letters in [clean(a) for a in existing]:
                    continue
                suggestions.append(("wordplay", clue_lower, letters, r["answer"]))

    return list(set(auto_inserts)), list(set(suggestions))


# -- DB writer -----------------------------------------------------------------

def _ensure_structured_table(conn):
    """Create structured_explanations table if it doesn't exist."""
    conn.execute("""CREATE TABLE IF NOT EXISTS structured_explanations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        clue_id INTEGER NOT NULL UNIQUE,
        definition_text TEXT,
        definition_start INTEGER,
        definition_end INTEGER,
        wordplay_types TEXT,
        components TEXT,
        model_version TEXT,
        confidence REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source TEXT,
        puzzle_number TEXT,
        clue_number TEXT,
        FOREIGN KEY (clue_id) REFERENCES clues(id)
    )""")


def store_result(conn, clue_id, ai_output, assembly, validation, tier):
    """Store pipeline result into clues and structured_explanations tables.

    Stores assembler-corrected breakdown (not raw AI pieces).
    Only persists results with score > 0.
    Sets has_solution=1 when score >= 80 and no type mismatch.
    """
    _ensure_structured_table(conn)
    score = validation.get("score", 0)

    # Don't persist failures
    if not assembly or score <= 0:
        return

    ai_type = None
    ai_def = None
    ai_pieces = []

    if ai_output:
        ai_type = ai_output.get("wordplay_type")
        ai_def = ai_output.get("definition")
        for p in ai_output.get("pieces", []):
            piece = {
                "mechanism": p.get("mechanism", "unknown"),
                "clue_word": p.get("clue_word", ""),
                "letters": p.get("letters", ""),
            }
            # Preserve refinement fields (source, indicator, deleted, deleted_word)
            for key in ("source", "indicator", "deleted", "deleted_word"):
                if p.get(key):
                    piece[key] = p[key]
            ai_pieces.append(piece)

    asm_op = assembly.get("op", "")
    asm_type = OP_TO_TYPE.get(asm_op, asm_op)

    # Assembler type FIRST (it's the verified one), AI type as fallback
    wordplay_types = []
    if asm_type:
        wordplay_types.append(asm_type)
    if ai_type and ai_type not in wordplay_types:
        wordplay_types.append(ai_type)
    if not wordplay_types:
        wordplay_types = ["unknown"]

    # Store full corrected picture: AI pieces for reference + assembler result
    components = {
        "ai_pieces": ai_pieces,
        "assembly": assembly,
        "wordplay_type": wordplay_types[0],
    }

    # Build human-readable explanation from assembly
    from .report import _describe_assembly
    # Fetch answer and clue text for verification
    clue_row = conn.execute("SELECT clue_text, answer FROM clues WHERE id = ?", (clue_id,)).fetchone()
    clue_answer = clue_row[1] if clue_row else None
    clue_text = clue_row[0] if clue_row else ""
    explanation_text = _describe_assembly(assembly, ai_pieces, answer=clue_answer) if assembly else None

    # Run mechanical verifier for confidence score
    if explanation_text:
        from .verify_explanation import ExplanationVerifier
        _verifier = getattr(store_result, '_verifier', None)
        if _verifier is None:
            _verifier = ExplanationVerifier()
            store_result._verifier = _verifier
        v_result = _verifier.verify(
            clue_text, clue_answer or "", ai_def or "",
            wordplay_types[0], explanation_text,
        )
        confidence = v_result["score"] / 100.0 if v_result else score / 100.0
    else:
        confidence = score / 100.0

    # Update definition only if not already set
    if ai_def:
        conn.execute("""
            UPDATE clues SET definition = ?
            WHERE id = ? AND (definition IS NULL OR definition = '')
        """, (ai_def, clue_id))
    # Always overwrite wordplay_type and explanation with assembler result
    conn.execute("""
        UPDATE clues SET wordplay_type = ?
        WHERE id = ?
    """, (wordplay_types[0], clue_id))
    if explanation_text:
        conn.execute("""
            UPDATE clues SET ai_explanation = ?
            WHERE id = ?
        """, (explanation_text, clue_id))

    # Determine solved status based on actual content, not score
    # 1 = all three hint fields present, 2 = partial
    row = conn.execute(
        "SELECT definition, wordplay_type, ai_explanation FROM clues WHERE id = ?",
        (clue_id,)
    ).fetchone()
    has_def = bool(row[0]) if row else bool(ai_def)
    has_type = bool(row[1]) if row else bool(wordplay_types[0])
    # Explanation: either ai_explanation in clues, or components we're about to write
    has_expl = bool(row[2]) if row else False
    has_expl = has_expl or bool(ai_pieces)  # pipeline components count as explanation

    # Auto-approve high-confidence solves (score >= 80), flag others for review
    # But NEVER overwrite a manual review (reviewed = 1 approved, 2 rejected)
    current_reviewed = conn.execute(
        "SELECT reviewed FROM clues WHERE id = ?", (clue_id,)
    ).fetchone()
    already_reviewed = current_reviewed and current_reviewed[0] in (1, 2)

    if not already_reviewed:
        auto_reviewed = 1 if score >= 80 else 0
    else:
        auto_reviewed = current_reviewed[0]

    if has_def and has_type and has_expl:
        conn.execute("UPDATE clues SET has_solution = 1, reviewed = ? WHERE id = ?", (auto_reviewed, clue_id))
    elif has_type and has_expl and score >= 80:
        # High-confidence solve with type + explanation but missing definition —
        # still auto-approve (definition absence is a minor gap, not a quality issue)
        conn.execute("UPDATE clues SET has_solution = 1, reviewed = ? WHERE id = ?", (auto_reviewed, clue_id))
    elif has_def or has_type:
        if not already_reviewed:
            conn.execute("UPDATE clues SET has_solution = 2, reviewed = 0 WHERE id = ?", (clue_id,))
        else:
            conn.execute("UPDATE clues SET has_solution = 2 WHERE id = ?", (clue_id,))

    def_start = None
    def_end = None
    if ai_def:
        clue_text = conn.execute(
            "SELECT clue_text FROM clues WHERE id = ?", (clue_id,)
        ).fetchone()
        if clue_text:
            idx = clue_text[0].lower().find(ai_def.lower())
            if idx >= 0:
                def_start = idx
                def_end = idx + len(ai_def)

    # Fetch source/puzzle/clue metadata for structured_explanations
    clue_meta = conn.execute(
        "SELECT source, puzzle_number, clue_number FROM clues WHERE id = ?", (clue_id,)
    ).fetchone()
    src, pnum, cnum = clue_meta if clue_meta else (None, None, None)

    existing = conn.execute(
        "SELECT id FROM structured_explanations WHERE clue_id = ?", (clue_id,)
    ).fetchone()

    model_version = "haiku_sonnet_tiered_v1"

    if existing:
        conn.execute("""
            UPDATE structured_explanations
            SET definition_text = ?, definition_start = ?, definition_end = ?,
                wordplay_types = ?, components = ?,
                model_version = ?, confidence = ?,
                source = ?, puzzle_number = ?, clue_number = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE clue_id = ?
        """, (
            ai_def, def_start, def_end,
            json.dumps(wordplay_types), json.dumps(components),
            model_version, confidence,
            src, pnum, cnum, clue_id
        ))
    else:
        conn.execute("""
            INSERT INTO structured_explanations
            (clue_id, definition_text, definition_start, definition_end,
             wordplay_types, components, model_version, confidence,
             source, puzzle_number, clue_number, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (
            clue_id, ai_def, def_start, def_end,
            json.dumps(wordplay_types), json.dumps(components),
            model_version, confidence,
            src, pnum, cnum
        ))


# -- High-level solve_clue entry point ----------------------------------------

def solve_clue(clue_text, answer, enrichment, enricher, homo_engine,
               example_messages, cached_ai=None, ref_db=None):
    """Solve a single cryptic clue using Sonnet + assembler + fallback.

    If cached_ai is provided, skip the API call and re-run assembler + scoring
    on the cached Sonnet output. Use this for previously-solved clues to test
    assembler improvements without paying for API calls.

    Returns a dict with:
        ai_output, assembly, tier, validation,
        tokens_in, tokens_out, sonnet_pieces, sonnet_wtype, sonnet_def,
        fallback_method
    """
    target = clean(answer)

    # Extract indicator hints for AI prompt (not used for assembly suppression)
    ind_hints = _extract_indicator_hints(enrichment) if enrichment else []

    # Use cached AI output or call Sonnet
    if cached_ai:
        sonnet_out = cached_ai
        tokens_in, tokens_out = 0, 0
    else:
        sonnet_out, tokens_in, tokens_out = call_api(
            SONNET_MODEL, clue_text, answer, enrichment, example_messages
        )
    sonnet_pieces, sonnet_wtype, sonnet_def = extract_pieces(sonnet_out)

    # Try assembly
    assembly, fallback_method = full_assembly_attempt(
        clue_text, answer, sonnet_pieces, sonnet_wtype,
        enricher, homo_engine, target
    )

    # If assembly is poor (brute_gap or failed), try literal clue words as pieces.
    # Handles cases where model identified correct fodder words but pre-processed
    # the letters wrong (e.g. "British cut short" -> HUBRISTC instead of BRITISH+CUT).
    if (not assembly or (assembly and assembly.get("brute_gap"))) and sonnet_wtype == "anagram":
        literal_pieces = extract_literal_fodder(sonnet_out)
        if literal_pieces and literal_pieces != sonnet_pieces:
            alt_assembly, alt_method = full_assembly_attempt(
                clue_text, answer, literal_pieces, sonnet_wtype,
                enricher, homo_engine, target
            )
            if alt_assembly and not alt_assembly.get("brute_gap"):
                assembly = alt_assembly
                fallback_method = alt_method
                sonnet_pieces = literal_pieces

    tier = None
    if assembly:
        tier = "Sonnet"
        asm_op = assembly.get("op", "")
        # When assembler found hidden but Sonnet suggested different type,
        # override pieces — Sonnet's pieces are meaningless for hidden words
        if asm_op in ("hidden", "hidden_in_word", "hidden_reversed") and sonnet_wtype != "hidden":
            source_phrase = assembly.get("words") or assembly.get("word") or ""
            sonnet_out = sonnet_out or {}
            sonnet_out["pieces"] = [
                {"clue_word": source_phrase, "letters": target, "mechanism": "hidden"}
            ]
            sonnet_out["wordplay_type"] = "hidden"
        # When truncation_db built pieces from scratch, override AI pieces
        elif assembly.get("source") == "truncation_from_db" and assembly.get("pieces_detail"):
            tier = "Fallback"
            sonnet_out = sonnet_out or {}
            sonnet_out["pieces"] = [
                {"clue_word": w, "letters": yld, "mechanism": mech}
                for w, yld, mech in assembly["pieces_detail"]
            ]
            asm_type = OP_TO_TYPE.get(asm_op, "")
            if asm_type:
                sonnet_out["wordplay_type"] = asm_type
        # When assembler dropped a piece, remove it from AI pieces so scoring
        # and report reflect only the pieces actually used
        elif assembly.get("note", "").startswith("dropped piece") and sonnet_out:
            used_order = assembly.get("order", [])
            if used_order:
                used_set = set(used_order)
                # For reversals, map pre-reversal form to post-reversal
                rev_from = assembly.get("reversed", "")
                rev_to = assembly.get("gives", "")
                new_pieces = []
                for p in sonnet_out.get("pieces", []):
                    pl = clean(p.get("letters", ""))
                    if pl in used_set:
                        new_pieces.append(p)
                    elif rev_from and pl == rev_from:
                        # Keep reversed piece, update letters to post-reversal
                        p = dict(p)
                        p["letters"] = rev_to
                        new_pieces.append(p)
                sonnet_out["pieces"] = new_pieces

    else:
        assembly = enrichment_fallback(clue_text, answer, enricher, target,
                                       definition=sonnet_def)
        if assembly:
            tier = "Fallback"
            # Override AI pieces with the actual pieces the fallback used
            if assembly.get("pieces_detail"):
                sonnet_out = sonnet_out or {}
                sonnet_out["pieces"] = [
                    {"clue_word": w, "letters": yld, "mechanism": mech}
                    for w, yld, mech in assembly["pieces_detail"]
                ]
                asm_type = OP_TO_TYPE.get(assembly.get("op", ""), "")
                if asm_type:
                    sonnet_out["wordplay_type"] = asm_type

    # Refine piece annotations (add deletion/truncation detail)
    refine_pieces(sonnet_out, clue_text, enricher)

    validation = check_mechanism(
        clue_text, answer, sonnet_out, assembly, enricher, tier or "FAIL",
        enrichment=enrichment, ref_db=ref_db
    )

    return {
        "ai_output": sonnet_out,
        "assembly": assembly,
        "tier": tier,
        "validation": validation,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "sonnet_pieces": sonnet_pieces,
        "sonnet_wtype": sonnet_wtype,
        "sonnet_def": sonnet_def,
        "fallback_method": fallback_method,
    }
