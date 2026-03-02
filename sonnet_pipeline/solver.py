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
        conn = sqlite3.connect(db_path)
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
    idx = concat.find(target)
    if idx >= 0:
        sw = ew = None
        for wi, (ws, we, _) in enumerate(boundaries):
            if ws <= idx < we:
                sw = wi
            if ws < idx + len(target) <= we:
                ew = wi
        if sw is not None and ew is not None:
            if sw != ew:
                return {"op": "hidden", "words": " ".join(words[sw:ew+1])}
            word_clean = clean(words[sw])
            if len(target) < len(word_clean):
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


def assemble(clue_text, answer, pieces, max_pieces=6, homo_engine=None, ai_wtype=None):
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
    # Without this, wrong pieces that total more letters than the answer
    # get force-fitted via deletion, masking errors.
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

SYSTEM_PROMPT = """You are parsing cryptic crossword clues. You are given a clue, its answer, and DB lookups showing known synonyms, abbreviations, and indicators for each clue word.

CRITICAL: The DB lookups are your primary source. Each clue word has:
- syn= synonyms (e.g. drink: syn=BELT,ALE means "drink" can give letters BELT or ALE)
- abbr= abbreviations (e.g. stone: abbr=ST means "stone" gives letters ST)
- ind= indicator types (e.g. in: ind=container means "in" signals a container operation)
- sounds= homophones

Entries marked with * (e.g. syn=PIE*,TART) are substrings of the answer -- these are the most likely wordplay components. PRIORITIZE starred entries when building your solution.

Your job: select the right synonym or abbreviation for each wordplay word from the DB lookups, then show how they combine to spell the answer.

Output JSON with:
- "definition": the exact substring of the clue that defines the answer (always at the start or end of the clue)
- "wordplay_type": charade, container, anagram, deletion, hidden, reversal, homophone, double_definition, cryptic_definition, acrostic, spoonerism
- "pieces": array of objects, each with:
  - "clue_word": the word(s) from the clue
  - "letters": the uppercase letters this produces (MUST come from a DB lookup syn/abbr, or be the literal letters of the clue word for anagram fodder)
  - "mechanism": synonym, abbreviation, literal, anagram_fodder, first_letter, last_letter, reversal, sound_of, alternate_letters, core_letters, deletion, hidden

Rules:
- SELECT synonyms and abbreviations from the DB lookups. Do not invent synonyms not listed.
- PRIORITIZE entries marked with * -- they are substrings of the answer and most likely to be correct.
- The pieces' letters, when assembled via the wordplay_type, MUST spell the full answer.
- NEVER return the full answer as a single piece. Every answer must be broken into 2+ pieces from separate clue words, UNLESS the wordplay_type is double_definition, cryptic_definition, or hidden.
- Indicator words (ind=) are NOT part of the answer -- they tell you the operation type.
- Definition is always at the start or end of the clue, never in the middle.
- For containers: show the outer piece and inner piece separately. The inner goes inside the outer.
- For anagrams: pieces are the raw fodder letters before rearrangement.
- For hidden words: the answer appears as a literal substring spanning consecutive clue words. Look at the actual letters. "Some", "in part", "partly" are common indicators.
- For hidden reversed: the substring is reversed. "Brought back in/through" signals this.
- For double definitions: two separate definitions, no pieces.
- For spoonerisms: "Spooner's" signals swapping initial consonants of two words.

Return ONLY valid JSON."""

FEW_SHOT_EXAMPLES = [
    {
        "input": """Clue: Patterned plate for various clients (7)
Answer: STENCIL
DB lookups:
  patterned: ind=anagram
  plate: syn=DISH,DISC,PAN,SLAB,TILE; abbr=P
  various: ind=anagram
  clients: syn=USERS""",
        "output": json.dumps({
            "definition": "Patterned plate",
            "wordplay_type": "anagram",
            "pieces": [
                {"clue_word": "clients", "letters": "CLIENTS", "mechanism": "anagram_fodder"}
            ]
        })
    },
    {
        "input": """Clue: Juliet in sober group with kiss for hero (4)
Answer: AJAX
DB lookups:
  juliet: abbr=J
  in: ind=container,hidden,insertion
  sober: ind=deletion
  group: syn=AA,BAND,GANG,SET,SIDE
  kiss: abbr=X; syn=BUSS,PECK
  hero: syn=ACE,GOD,IDOL,LION""",
        "output": json.dumps({
            "definition": "hero",
            "wordplay_type": "container",
            "pieces": [
                {"clue_word": "sober group", "letters": "AA", "mechanism": "abbreviation"},
                {"clue_word": "Juliet", "letters": "J", "mechanism": "abbreviation"},
                {"clue_word": "kiss", "letters": "X", "mechanism": "abbreviation"}
            ]
        })
    },
    {
        "input": """Clue: Unwilling to forgo large bond (4)
Answer: OATH
DB lookups:
  unwilling: syn=LOATH,AVERSE; ind=deletion
  forgo: ind=deletion
  large: abbr=L; syn=BIG,DEEP,FULL,HUGE
  bond: syn=BAIL,BAND,CORD,GLUE,KNOT,LINK,OATH,SEAL,WORD""",
        "output": json.dumps({
            "definition": "bond",
            "wordplay_type": "deletion",
            "pieces": [
                {"clue_word": "Unwilling", "letters": "LOATH", "mechanism": "synonym"},
                {"clue_word": "large", "letters": "L", "mechanism": "abbreviation"}
            ]
        })
    },
    {
        "input": """Clue: Father's attempt to make small cake (6)
Answer: PASTRY
DB lookups:
  father: syn=DAD,PA,POP,SIRE; abbr=FR
  attempt: syn=BID,GO,TRY; ind=anagram
  make: ind=anagram; syn=BRAND,BUILD,EARN,FORM
  small: abbr=S; syn=LOW,TINY,WEE
  cake: syn=BUN,GATEAU,TART,TORTE""",
        "output": json.dumps({
            "definition": "cake",
            "wordplay_type": "charade",
            "pieces": [
                {"clue_word": "Father's", "letters": "PAS", "mechanism": "synonym"},
                {"clue_word": "attempt", "letters": "TRY", "mechanism": "synonym"}
            ]
        })
    }
]


def build_example_messages():
    msgs = []
    for ex in FEW_SHOT_EXAMPLES:
        msgs.append({"role": "user", "content": ex["input"]})
        msgs.append({"role": "assistant", "content": ex["output"]})
    return msgs


def call_api(model, clue_text, answer, enrichment, example_messages, extra_context=""):
    """Call a Claude model to identify pieces."""
    enum_len = len(answer.replace(" ", "").replace("-", ""))
    user_msg = "Clue: %s (%d)\nAnswer: %s" % (clue_text, enum_len, answer)
    if enrichment:
        user_msg += "\n" + enrichment
    if extra_context:
        user_msg += "\n\n" + extra_context

    messages = example_messages + [{"role": "user", "content": user_msg}]

    response = client.messages.create(
        model=model,
        max_tokens=400,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    raw = response.content[0].text.strip()
    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        if "{" in raw:
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                parsed = json.loads(raw[start:end])
            except (json.JSONDecodeError, ValueError):
                parsed = None
        else:
            parsed = None

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

def full_assembly_attempt(clue, answer, pieces, wtype, enricher, homo_engine, target):
    """Try assembling pieces with all fallback strategies."""
    # First attempt: restricted by AI type (suppresses anagram when AI says otherwise)
    assembly = assemble(clue, answer, pieces, homo_engine=homo_engine, ai_wtype=wtype) if pieces else None

    # Reject degenerate case: single piece that equals the answer
    # (model just echoed the answer instead of breaking it down)
    if assembly and len(pieces) == 1 and pieces[0] == target:
        assembly = None

    if not assembly and wtype in ("double_definition", "cryptic_definition"):
        return {"op": wtype}, "dd_cd"

    if not assembly:
        spoon = try_spoonerism(clue, target, enricher)
        if spoon:
            return spoon, "spoonerism"

    if not assembly:
        hidden = try_hidden(clue, target)
        if hidden:
            return hidden, "hidden"

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
                    result = assemble(clue, answer, trial, homo_engine=homo_engine, ai_wtype=wtype)
                    if result:
                        result["truncated"] = {"from": longest, "to": truncated, "removed": trim}
                        return result, "truncation"

    if not assembly and len(pieces) >= 2:
        for skip in range(len(pieces)):
            subset = pieces[:skip] + pieces[skip+1:]
            assembly = assemble(clue, answer, subset, homo_engine=homo_engine, ai_wtype=wtype)
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

    # Last resort: try unrestricted assembly (anagram allowed) with penalty flag
    # Better to give a flagged anagram than nothing, but the score will be penalised
    if not assembly and pieces and wtype and wtype != "anagram":
        assembly = assemble(clue, answer, pieces, homo_engine=homo_engine, ai_wtype=None)
        if assembly and assembly.get("op") in ("anagram", "charade+anagram",
                                                "anagram+charade", "deletion+anagram"):
            assembly["anagram_fallback"] = True
            return assembly, "anagram_fallback"
        elif assembly:
            return assembly, "direct"

    if assembly:
        return assembly, "direct"
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


def check_mechanism(clue_text, answer, ai_output, assembly, enricher, tier):
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
        if enricher.lookup_definition(def_clean, answer):
            checks["definition"] = "confirmed in DB"
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

    # --- Wordplay type (20pts) ---
    # The assembler's op is the verified type; award full points for it
    if asm_type:
        checks["wordplay_type"] = asm_type
        score += 20
    else:
        checks["wordplay_type"] = "unknown"

    # Penalty: assembler type doesn't match AI type
    if ai_type and asm_type and ai_type != asm_type:
        # Some types are compatible (e.g. reversal_container matches both)
        compatible = {
            "reversal_container": {"container", "reversal"},
            "container_reversal": {"container", "reversal"},
            "charade+anagram": {"charade", "anagram"},
            "anagram+charade": {"charade", "anagram"},
            "deletion+anagram": {"deletion", "anagram"},
        }
        compat_set = compatible.get(asm_type, set())
        if ai_type not in compat_set:
            checks["type_mismatch"] = "AI=%s, assembled=%s" % (ai_type, asm_type)
            score -= 10

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
    elif asm_op in ("hidden", "hidden_in_word"):
        # Hidden words are self-evidently correct — the answer is in the clue
        checks["yields_check"] = "hidden (self-evident)"
        score += 20
    elif asm_op in ("double_definition", "cryptic_definition"):
        # DD/CD have no pieces to check
        checks["yields_check"] = "n/a for %s" % asm_type
        score += 15

    # --- Explanation: Pieces validated (15pts) ---
    validated_pieces = 0
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
        elif mech == "abbreviation":
            abbrs = enricher.lookup_abbreviations(clue_word)
            if letters in [clean(a) for a in abbrs]:
                validated_pieces += 1
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
    elif asm_op in ("double_definition", "cryptic_definition"):
        checks["pieces_validated"] = "n/a for %s" % asm_type
        score += 10
    else:
        checks["pieces_validated"] = "no pieces to validate"

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

def store_result(conn, clue_id, ai_output, assembly, validation, tier):
    """Store pipeline result into clues and structured_explanations tables.

    Stores assembler-corrected breakdown (not raw AI pieces).
    Only persists results with score > 0.
    Sets has_solution=1 when score >= 80 and no type mismatch.
    """
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

    confidence = score / 100.0

    conn.execute("""
        UPDATE clues SET definition = ?, wordplay_type = ?
        WHERE id = ? AND (definition IS NULL OR definition = '')
    """, (ai_def, wordplay_types[0], clue_id))

    # Determine solved status: score >= 80
    is_solved = score >= 80

    if is_solved:
        conn.execute("UPDATE clues SET has_solution = 1 WHERE id = ?", (clue_id,))

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
               example_messages, cached_ai=None):
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
        if asm_op in ("hidden", "hidden_in_word") and sonnet_wtype != "hidden":
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
        clue_text, answer, sonnet_out, assembly, enricher, tier or "FAIL"
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
