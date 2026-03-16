"""Match clue words against the signature catalog.

Filtering-first approach:
1. Eliminate signatures impossible for this word count
2. Eliminate signatures missing required indicators
3. Eliminate signatures where letter budget is implausible
4. Only THEN try word→token assignments
"""

from itertools import permutations
from .tokens import *
from .catalog import CatalogEntry


def _is_container_outer(syn, answer):
    """Check if answer = syn with a contiguous block inserted."""
    gap = len(answer) - len(syn)
    if gap < 1 or len(syn) < 2:
        return False
    for i in range(1, len(syn) + 1):
        if answer[:i] == syn[:i] and answer[i + gap:] == syn[i:]:
            return True
    return False


def match_signatures(words, analyses, phrases, catalog, answer, db):
    """Yield (catalog_entry, assignment) pairs in priority order.

    assignment is a dict: word_index -> (token, value)
    Plus 'indicator_indices': set of word indices used as indicators
    Plus 'lnk_indices': set of word indices used as link words
    Plus 'fodder_order': list of (word_index, token, value) in signature order
    """
    n = len(words)
    answer_len = len(answer)

    # Pre-compute per-word role sets for fast filtering
    word_indicator_tokens = []  # index -> set of indicator tokens this word has
    for wa in analyses:
        ind_toks = set()
        for tok in wa.roles:
            if tok in INDICATOR_TOKENS:
                ind_toks.add(tok)
        word_indicator_tokens.append(ind_toks)

    # Pre-compute phrase indicator tokens for multi-word indicator matching
    phrase_indicator_tokens = {}  # (i, j) -> set of indicator tokens
    for (pi, pj), pwa in phrases.items():
        ind_toks = set()
        for tok in pwa.roles:
            if tok in INDICATOR_TOKENS:
                ind_toks.add(tok)
        if ind_toks:
            phrase_indicator_tokens[(pi, pj)] = ind_toks

    for entry in catalog:
        # === FILTER 1: Word count ===
        if n < entry.min_words:
            continue
        # Max words: span-derived fodder words + indicators + LNK allowance
        fodder_words = sum(entry.word_spans) if entry.word_spans else len(entry.tokens)
        max_words = fodder_words + len(entry.indicators) + 4
        if not entry.allow_extra_lnk:
            max_words = fodder_words + len(entry.indicators)
        if n > max_words:
            continue

        # === FILTER 2: Required indicators present ===
        if entry.indicators:
            missing = False
            for req_ind in entry.indicators:
                # Check single words
                found = any(req_ind in wit for wit in word_indicator_tokens)
                # Also check multi-word phrases
                if not found:
                    found = any(req_ind in ptoks
                                for ptoks in phrase_indicator_tokens.values())
                if not found:
                    missing = True
                    break
            if missing:
                continue

        # === FILTER 3: Letter budget (cheap arithmetic) ===
        if not _letter_budget_plausible(entry, words, analyses, answer_len):
            continue

        # === ASSIGNMENT: Try to map words to signature tokens ===
        yield from _try_assignments(entry, words, analyses, phrases,
                                     word_indicator_tokens,
                                     phrase_indicator_tokens, answer, db)


def _letter_budget_plausible(entry, words, analyses, answer_len):
    """Quick check: could fodder tokens plausibly produce answer_len letters?"""
    op = entry.operation

    # For hidden words, answer must be shorter than combined fodder
    if op in ("hidden", "hidden_reversed"):
        total_letters = sum(
            len("".join(c for c in w if c.isalpha()))
            for w in words
        )
        return answer_len < total_letters

    # For anagram types, total fodder letters must equal answer length
    # (checked more precisely during assignment)
    if op.startswith("anagram"):
        return True  # checked during execution

    # For charade: sum of min piece sizes <= answer_len <= sum of max piece sizes
    # ABR_F: 1-3 letters, SYN_F: 1-answer_len, RAW: word length, POS_F: 1-2
    min_total = 0
    max_total = 0
    for tok in entry.tokens:
        if tok == ABR_F:
            min_total += 1
            max_total += 3
        elif tok == SYN_F:
            min_total += 1
            max_total += answer_len
        elif tok == RAW:
            min_total += 1
            max_total += answer_len
        elif tok == POS_F:
            min_total += 1
            max_total += answer_len // 2 + 1
        elif tok == HOM_F:
            min_total += answer_len  # homophone ≈ same length
            max_total += answer_len + 2
        elif tok == HID_F:
            return True  # handled above
        else:
            min_total += 1
            max_total += answer_len

    if op in ("container",):
        # Container: inner inserted into outer, total = answer_len
        return min_total <= answer_len <= max_total
    if op in ("deletion",):
        # Deletion: base minus removed = answer; base > answer
        return True
    if op.startswith("trim"):
        return True  # trim removes 1-2 letters, hard to bound tightly
    if op in ("reversal", "reversal_charade"):
        return min_total <= answer_len <= max_total

    # Charade: pieces concatenate to answer
    return min_total <= answer_len <= max_total


def _try_assignments(entry, words, analyses, phrases, word_indicator_tokens,
                      phrase_indicator_tokens, answer, db):
    """Generate valid word→token assignments for this catalog entry."""
    n = len(words)
    op = entry.operation

    # Step 1: Identify which words/phrases CAN fill each indicator slot
    # Candidates are either a single word index (int) or a phrase tuple (i, j)
    indicator_candidates = {}  # indicator_token -> list of candidates
    for ind_tok in entry.indicators:
        cands = []
        # Single-word candidates
        for i in range(n):
            if ind_tok in word_indicator_tokens[i]:
                cands.append(i)
        # Multi-word phrase candidates
        for (pi, pj), ptoks in phrase_indicator_tokens.items():
            if ind_tok in ptoks:
                cands.append((pi, pj))
        if not cands:
            return  # impossible
        indicator_candidates[ind_tok] = cands

    # Step 2: For each indicator assignment, try fodder assignments
    for ind_assignment in _indicator_combos(indicator_candidates):
        # Collect all word indices consumed by indicators
        ind_indices = set()
        for val in ind_assignment.values():
            if isinstance(val, tuple):
                ind_indices.update(range(val[0], val[1]))
            else:
                ind_indices.add(val)
        remaining = [i for i in range(n) if i not in ind_indices]

        # All entries have word_spans — unified span-guided matching
        if entry.word_spans:
            yield from _assign_with_spans(entry, words, analyses,
                                           remaining, ind_assignment,
                                           answer, db)


def _indicator_combos(indicator_candidates):
    """Yield all ways to assign indicator tokens to distinct words/phrases.

    indicator_candidates: {token: [candidates]}
    Each candidate is either an int (single word index) or a tuple (i, j)
    representing a phrase spanning words[i:j].
    Yields: {token: candidate} dicts where no word indices overlap.
    """
    tokens = list(indicator_candidates.keys())
    if not tokens:
        yield {}
        return

    candidates_list = [indicator_candidates[t] for t in tokens]

    def _candidate_indices(cand):
        """Return the set of word indices consumed by a candidate."""
        if isinstance(cand, tuple):
            return set(range(cand[0], cand[1]))
        return {cand}

    def recurse(idx, used, assignment):
        if idx == len(tokens):
            yield dict(assignment)
            return
        for cand in candidates_list[idx]:
            cand_idx = _candidate_indices(cand)
            if not (cand_idx & used):
                assignment[tokens[idx]] = cand
                yield from recurse(idx + 1, used | cand_idx, assignment)
                del assignment[tokens[idx]]

    yield from recurse(0, set(), {})


def _remaining_are_valid(remaining_indices, words, analyses, db,
                         assigned_indicator_types=None):
    """Check that unassigned words are link words or redundant indicators.

    Accepts: known link words, and words that are indicators of the same
    type already assigned (e.g. "radio" is HOM_I when "heard" is already
    the assigned HOM_I — both signal the same operation).
    """
    for i in remaining_indices:
        w = words[i].lower().strip(".,;:!?\"'()-")
        if db.is_link_word(w):
            continue
        # Accept if word is an indicator of a type already assigned
        if assigned_indicator_types:
            wa = analyses[i]
            word_ind_types = {tok for tok in wa.roles if tok in INDICATOR_TOKENS}
            if word_ind_types & assigned_indicator_types:
                continue
        return False
    return True


def _build_assignment(entry, fodder_order, ind_assignment, lnk_indices):
    """Build the standard (entry, assignment) tuple."""
    assignment = {
        'fodder_order': fodder_order,  # [(word_idx, token, value), ...]
        'indicator_indices': ind_assignment,  # {ind_token: word_idx_or_phrase_tuple}
        'lnk_indices': lnk_indices,
    }
    return (entry, assignment)




# ============================================================
# Span-guided matching (charade, container, reversal)
# ============================================================

def _assign_with_spans(entry, words, analyses, remaining,
                       ind_assignment, answer, db):
    """Span-guided matching: the catalog entry specifies how many contiguous
    clue words each token slot consumes.  We place the spans left-to-right
    with gaps for link/indicator words, look up each phrase in the DB, and
    verify the answer using the operation-specific check.

    Works for: charade, container, reversal_charade, reversal.
    """
    from .word_analyzer import clean_word

    op = entry.operation
    n_tokens = len(entry.tokens)
    spans = entry.word_spans
    total_span = sum(spans)

    if len(remaining) < total_span:
        return

    answer_len = len(answer)
    is_reversal = op in ("reversal", "reversal_charade")

    # Generate all valid placements of N contiguous spans within `remaining`.
    def generate_placements(slot_idx, min_start):
        """Recursively place slot_idx at some start position >= min_start."""
        if slot_idx == n_tokens:
            yield []
            return
        span_size = spans[slot_idx]
        max_start = len(remaining) - span_size
        for k in range(slot_idx + 1, n_tokens):
            max_start -= spans[k]

        for start in range(min_start, max_start + 1):
            slot_indices = remaining[start:start + span_size]
            is_contiguous = all(
                slot_indices[j + 1] == slot_indices[j] + 1
                for j in range(len(slot_indices) - 1)
            )
            if not is_contiguous:
                continue
            for rest in generate_placements(slot_idx + 1,
                                             start + span_size):
                yield [start] + rest

    for placement in generate_placements(0, 0):
        covered = set()
        slot_word_groups = []
        for i, start in enumerate(placement):
            indices = remaining[start:start + spans[i]]
            covered.update(indices)
            slot_word_groups.append((indices, entry.tokens[i], spans[i]))

        leftover = [idx for idx in remaining if idx not in covered]
        assigned_ind_types = set(ind_assignment.keys()) if ind_assignment else None
        if leftover and not _remaining_are_valid(
                leftover, words, analyses, db, assigned_ind_types):
            continue

        # Look up each slot's phrase in the DB
        slot_values = []
        failed = False
        for indices, token_type, span_size in slot_word_groups:
            vals = _lookup_slot(indices, token_type, span_size, words,
                                analyses, answer, answer_len, is_reversal,
                                db, clean_word, entry)
            if not vals:
                failed = True
                break
            slot_values.append(vals)

        if failed:
            continue

        # Verify using operation-specific logic
        result = _verify_combo(op, entry, slot_values, slot_word_groups,
                               ind_assignment, leftover, answer, words)
        if result:
            yield result
            return


def _lookup_slot(indices, token_type, span_size, words, analyses,
                 answer, answer_len, is_reversal, db, clean_word,
                 entry=None):
    """Look up possible values for a token slot.

    Handles all token types: SYN_F, ABR_F, ANA_F, HID_F, HOM_F, POS_F.
    Operation-aware: deletion/trim get broader synonym search.
    For reversals, also accept values whose reverse is in the answer.
    """
    from . import executor

    vals = []
    op = entry.operation if entry else None

    # --- ANA_F: raw letters for anagramming ---
    if token_type == ANA_F:
        # Include all words in span, but also try excluding LNK words
        # (link words embedded in fodder span shouldn't contribute letters)
        all_letters = "".join(c for idx in indices
                              for c in words[idx].upper() if c.isalpha())
        non_lnk_letters = "".join(
            c for idx in indices
            if LNK not in analyses[idx].roles
            for c in words[idx].upper() if c.isalpha()
        )
        if all_letters:
            vals.append(all_letters)
        if non_lnk_letters and non_lnk_letters != all_letters:
            vals.append(non_lnk_letters)
        return vals

    # --- HID_F: raw letters for hidden word check ---
    if token_type == HID_F:
        letters = "".join(c for idx in indices
                          for c in words[idx].upper() if c.isalpha())
        if letters and len(letters) > answer_len:
            vals.append(letters)
        return vals

    # --- HOM_F: homophone lookup ---
    if token_type == HOM_F:
        if span_size == 1:
            wa = analyses[indices[0]]
            # Direct homophone of this word
            if HOM_F in wa.roles:
                for h in wa.roles[HOM_F]:
                    if isinstance(h, str) and h == answer:
                        vals.append(h)
            # Synonym → homophone chain
            w_clean = clean_word(words[indices[0]])
            syns = db.get_synonyms(w_clean)
            for syn in syns:
                homophones = db.get_homophones(syn.lower())
                for h in homophones:
                    if h == answer and h not in vals:
                        vals.append(h)
            # Abbreviation → homophone chain (e.g. one→I→EYE)
            if ABR_F in wa.roles:
                for abbr in wa.roles[ABR_F]:
                    if isinstance(abbr, str):
                        homophones = db.get_homophones(abbr.lower())
                        for h in homophones:
                            if h == answer and h not in vals:
                                vals.append(h)
        else:
            phrase = " ".join(words[idx] for idx in indices)
            phrase_clean = clean_word(phrase)
            # Direct phrase homophone
            homophones = db.get_homophones(phrase_clean)
            for h in homophones:
                if h == answer and h not in vals:
                    vals.append(h)
            # Phrase synonym → homophone chain
            syns = db.get_synonyms(phrase_clean)
            for syn in syns:
                homophones = db.get_homophones(syn.lower())
                for h in homophones:
                    if h == answer and h not in vals:
                        vals.append(h)
        return vals

    # --- POS_F: positional letter extraction ---
    if token_type == POS_F:
        if op == "acrostic":
            firsts = "".join(
                words[idx][0].upper() for idx in indices
                if words[idx] and words[idx][0].isalpha()
            )
            if firsts:
                vals.append(firsts)
        elif op == "alternate":
            text = "".join(c for idx in indices
                           for c in words[idx].upper() if c.isalpha())
            if text:
                vals.append(text[::2])   # odd positions (1st, 3rd, 5th)
                vals.append(text[1::2])  # even positions (2nd, 4th, 6th)
        else:
            # positional_charade, container_positional — extract using indicator type
            # Collect all positional types to try (use + trim indicators)
            pos_types_to_try = []
            pos_type = _get_pos_type(entry)
            if pos_type:
                pos_types_to_try.append(pos_type)
            trim_types = _get_trim_types(entry)
            if trim_types:
                pos_types_to_try.extend(trim_types)
                # TRIM_MIDDLE often means "empty" = keep outer shell
                if POS_I_TRIM_MIDDLE in trim_types and POS_I_OUTER not in pos_types_to_try:
                    pos_types_to_try.append(POS_I_OUTER)
            for pt in pos_types_to_try:
                if span_size == 1:
                    # Strip possessive 's before extraction (e.g. "Argument's" → "Argument")
                    raw = words[indices[0]]
                    if raw.endswith("'s") or raw.endswith("\u2019s"):
                        raw = raw[:-2]
                    extracted = executor.extract_positional(raw, pt)
                else:
                    text = "".join(c for idx in indices
                                   for c in words[idx].upper() if c.isalpha())
                    extracted = executor.extract_positional(text, pt)
                if extracted and extracted not in vals:
                    vals.append(extracted)
        return vals

    # --- SYN_F for container: also accept container-outer synonyms ---
    if token_type == SYN_F and op in ("container", "container_charade",
                                       "anagram_container", "container_positional"):
        is_multi_piece = op == "container_charade"
        if span_size == 1:
            wa = analyses[indices[0]]
            if SYN_F in wa.roles:
                for v in wa.roles[SYN_F]:
                    if isinstance(v, str):
                        if v in answer:
                            vals.append(v)
                        elif _is_container_outer(v, answer):
                            vals.append(v)
                        elif is_multi_piece and len(v) >= 2 and len(v) < answer_len:
                            # For container_charade, the container covers a substring
                            # of the answer, so accept any reasonable-length synonym
                            vals.append(v)
            # Also do a broader synonym lookup for container outers
            w_clean = clean_word(words[indices[0]])
            for s in db.get_synonyms(w_clean, max_len=answer_len):
                if s not in vals and _is_container_outer(s, answer):
                    vals.append(s)
        else:
            phrase = " ".join(words[idx] for idx in indices)
            phrase_clean = clean_word(phrase)
            for s in db.get_synonyms(phrase_clean, max_len=answer_len):
                if s not in vals:
                    if s in answer or _is_container_outer(s, answer):
                        vals.append(s)
        if vals:
            return vals
        # Fall through to standard handling if nothing found

    # --- SYN_F for deletion/trim: broader synonym search ---
    if token_type == SYN_F and op in ("deletion", "trim", "trim_charade"):
        max_syn_len = answer_len + 3
        if span_size == 1:
            wa = analyses[indices[0]]
            if SYN_F in wa.roles:
                for v in wa.roles[SYN_F]:
                    if isinstance(v, str) and v not in vals:
                        vals.append(v)
            w_clean = clean_word(words[indices[0]])
            for s in db.get_synonyms(w_clean, max_len=max_syn_len):
                if s not in vals:
                    vals.append(s)
        else:
            phrase = " ".join(words[idx] for idx in indices)
            phrase_clean = clean_word(phrase)
            for s in db.get_synonyms(phrase_clean, max_len=max_syn_len):
                if s not in vals:
                    vals.append(s)
        return vals

    # --- Standard SYN_F / ABR_F handling ---
    if span_size == 1:
        wa = analyses[indices[0]]
        if token_type in wa.roles:
            for v in wa.roles[token_type]:
                if isinstance(v, str):
                    if v in answer or (is_reversal and v[::-1] in answer):
                        vals.append(v)
        # Also check RAW letters
        w_alpha = "".join(c for c in words[indices[0]].upper()
                          if c.isalpha())
        if w_alpha and token_type == SYN_F:
            if w_alpha in answer or (is_reversal and w_alpha[::-1] in answer):
                if w_alpha not in vals:
                    vals.append(w_alpha)
    else:
        phrase = " ".join(words[idx] for idx in indices)
        phrase_clean = clean_word(phrase)
        if token_type == SYN_F:
            syns = db.get_synonyms(phrase_clean, max_len=answer_len)
            for s in syns:
                if s in answer or (is_reversal and s[::-1] in answer):
                    vals.append(s)
        elif token_type == ABR_F:
            abbrs = db.get_abbreviations(phrase_clean)
            for a in abbrs:
                if a in answer or (is_reversal and a[::-1] in answer):
                    vals.append(a)
    return vals


def _verify_combo(op, entry, slot_values, slot_word_groups,
                  ind_assignment, leftover, answer, words=None):
    """Try all value combinations and verify against the operation."""
    from . import executor

    for combo in _product_capped(slot_values, cap=200):
        result = None

        if op == "charade":
            if "".join(combo) == answer:
                result = combo
            elif len(combo) <= 4:
                for perm in permutations(combo):
                    if "".join(perm) == answer:
                        result = combo
                        break
        elif op in ("reversal", "reversal_charade"):
            result = _verify_reversal_combo(combo, answer)
        elif op == "container":
            result = _verify_container_combo(combo, answer)
        elif op == "container_charade":
            result = _verify_container_charade_combo(combo, answer)
        elif op == "anagram":
            if len(combo) == 1 and sorted(combo[0]) == sorted(answer):
                result = combo
        elif op == "anagram_plus":
            combined = "".join(combo)
            if sorted(combined) == sorted(answer):
                result = combo
        elif op == "anagram_charade":
            result = _verify_anagram_charade_combo(combo, slot_word_groups, answer)
        elif op == "anagram_container":
            result = _verify_anagram_container_combo(combo, slot_word_groups, answer)
        elif op == "hidden":
            if len(combo) == 1 and answer in combo[0]:
                result = combo
        elif op == "hidden_reversed":
            if len(combo) == 1 and answer[::-1] in combo[0]:
                result = combo
        elif op == "homophone":
            if len(combo) == 1 and combo[0] == answer:
                result = combo
        elif op == "alternate":
            if len(combo) == 1 and combo[0] == answer:
                result = combo
        elif op == "acrostic":
            if len(combo) == 1 and combo[0] == answer:
                result = combo
        elif op == "deletion":
            result = _verify_deletion_combo(combo, slot_word_groups, answer)
        elif op == "trim":
            result = _verify_trim_combo(combo, entry, answer)
        elif op == "trim_charade":
            result = _verify_trim_charade_combo(combo, slot_word_groups, entry, answer)
        elif op == "positional_charade":
            if "".join(combo) == answer:
                result = combo
            elif len(combo) <= 4:
                for perm in permutations(combo):
                    if "".join(perm) == answer:
                        result = combo
                        break
        elif op == "container_positional":
            result = _verify_container_combo(combo, answer)

        if result is not None:
            fodder = []
            for i, (indices, token_type, _) in enumerate(slot_word_groups):
                word_key = (indices[0] if len(indices) == 1
                            else (indices[0], indices[-1] + 1))
                fodder.append((word_key, token_type, combo[i]))
            return _build_assignment(entry, fodder, ind_assignment,
                                     set(leftover))
    return None


def _verify_reversal_combo(combo, answer):
    """Try reversing each piece in turn; check if concatenation = answer."""
    # Try each piece reversed
    for rev_idx in range(len(combo)):
        parts = list(combo)
        parts[rev_idx] = parts[rev_idx][::-1]
        if "".join(parts) == answer:
            return combo
    # Try reversing the entire concatenation (e.g. ART+X+E → ARTXE → EXTRA)
    if "".join(combo)[::-1] == answer:
        return combo
    # Also try all pieces reversed individually (pure multi-piece reversal)
    if len(combo) > 1:
        parts = [p[::-1] for p in combo]
        if "".join(parts) == answer:
            return combo
    # Single piece pure reversal
    if len(combo) == 1 and combo[0][::-1] == answer:
        return combo
    return None


def _verify_container_combo(combo, answer):
    """Two pieces: try inserting one inside the other at every position."""
    if len(combo) != 2:
        return None
    a, b = combo
    # Try a=outer, b=inner
    for pos in range(1, len(a)):
        if a[:pos] + b + a[pos:] == answer:
            return combo
    # Try b=outer, a=inner
    for pos in range(1, len(b)):
        if b[:pos] + a + b[pos:] == answer:
            return combo
    return None


def _verify_container_charade_combo(combo, answer):
    """3+ pieces: two form a container, rest charade'd alongside."""
    n = len(combo)
    # Try each pair as container, remaining pieces concatenated around it
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            outer, inner = combo[i], combo[j]
            remaining_pieces = [combo[k] for k in range(n) if k != i and k != j]
            # Try inserting inner at every position in outer
            for pos in range(1, len(outer)):
                container_result = outer[:pos] + inner + outer[pos:]
                # Now try container_result + remaining in every order position
                all_parts = remaining_pieces + [container_result]
                # Try all orderings (small n, capped)
                from itertools import permutations
                if len(all_parts) <= 4:
                    for perm in permutations(all_parts):
                        if "".join(perm) == answer:
                            return combo
    return None


def _verify_anagram_charade_combo(combo, slot_word_groups, answer):
    """Verify anagram + charade: fixed pieces at positions, ANA gaps anagrammed."""
    n = len(combo)
    is_ana = [slot_word_groups[i][1] == ANA_F for i in range(n)]

    def try_split(seg_idx, pos):
        if seg_idx == n:
            if pos == len(answer):
                yield []
            return
        if is_ana[seg_idx]:
            seg_len = len(combo[seg_idx])
            end = pos + seg_len
            if end <= len(answer):
                for rest in try_split(seg_idx + 1, end):
                    yield [answer[pos:end]] + rest
        else:
            val = combo[seg_idx]
            end = pos + len(val)
            if end <= len(answer) and answer[pos:end] == val:
                for rest in try_split(seg_idx + 1, end):
                    yield [val] + rest

    for split in try_split(0, 0):
        ana_gap = ""
        ana_fodder = ""
        for i in range(n):
            if is_ana[i]:
                ana_gap += split[i]
                ana_fodder += combo[i]
        if sorted(ana_gap) == sorted(ana_fodder):
            return combo
    return None


def _verify_anagram_container_combo(combo, slot_word_groups, answer):
    """Verify anagram + container: one piece inside the other, ANA part rearranged."""
    if len(combo) != 2:
        return None
    is_ana = [slot_word_groups[i][1] == ANA_F for i in range(2)]

    for inner_idx in range(2):
        outer_idx = 1 - inner_idx
        inner_val = combo[inner_idx]
        outer_val = combo[outer_idx]

        if is_ana[inner_idx]:
            # Inner is anagram fodder: try inserting rearranged letters into outer
            for pos in range(1, len(outer_val)):
                needed = answer[len(outer_val[:pos]):len(answer) - len(outer_val[pos:])]
                if (len(needed) == len(inner_val) and
                        outer_val[:pos] + needed + outer_val[pos:] == answer and
                        sorted(needed) == sorted(inner_val)):
                    return combo
        elif is_ana[outer_idx]:
            # Outer is anagram fodder: outer wraps around inner
            for pos in range(len(answer)):
                end = pos + len(inner_val)
                if end <= len(answer) and answer[pos:end] == inner_val:
                    outer_parts = answer[:pos] + answer[end:]
                    if sorted(outer_parts) == sorted(outer_val):
                        return combo
        else:
            # Both fixed: standard container check
            for pos in range(1, len(outer_val)):
                if outer_val[:pos] + inner_val + outer_val[pos:] == answer:
                    return combo
    return None


def _verify_deletion_combo(combo, slot_word_groups, answer):
    """Verify deletion: SYN minus ABR = answer."""
    from . import executor
    syn_val = None
    abr_val = None
    for i in range(len(combo)):
        if slot_word_groups[i][1] == SYN_F:
            syn_val = combo[i]
        elif slot_word_groups[i][1] == ABR_F:
            abr_val = combo[i]
    if syn_val and abr_val and executor.try_deletion(syn_val, abr_val, answer):
        return combo
    return None


def _verify_trim_combo(combo, entry, answer):
    """Verify trim: trimmed SYN = answer."""
    from . import executor
    trim_types = _get_trim_types(entry)
    if not trim_types or len(combo) != 1:
        return None
    for trim_type in trim_types:
        trimmed = executor.extract_positional(combo[0], trim_type)
        if trimmed and trimmed == answer:
            return combo
    return None


def _verify_trim_charade_combo(combo, slot_word_groups, entry, answer):
    """Verify trim_charade: one SYN trimmed + other pieces = answer."""
    from . import executor
    trim_types = _get_trim_types(entry)
    if not trim_types:
        return None
    n = len(combo)
    # Try trimming each SYN_F slot with each possible trim type
    for trim_type in trim_types:
        for trim_idx in range(n):
            if slot_word_groups[trim_idx][1] != SYN_F:
                continue
            trimmed = executor.extract_positional(combo[trim_idx], trim_type)
            if not trimmed:
                continue
            parts = list(combo)
            parts[trim_idx] = trimmed
            # Try all permutations for charade assembly
            if "".join(parts) == answer:
                return combo
            if n <= 4:
                from itertools import permutations
                for perm in permutations(parts):
                    if "".join(perm) == answer:
                        return combo
    return None


def _get_pos_type(entry):
    """Get positional extraction type from entry's indicators."""
    if entry is None:
        return None
    for ind_tok in entry.indicators:
        if ind_tok in (POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
                       POS_I_ALTERNATE, POS_I_HALF):
            return ind_tok
    return None


def _get_trim_types(entry):
    """Get trim type(s) from entry's indicators.

    Returns a list of trim types to try. For specific POS_I_TRIM_* indicators,
    returns just that one. For generic DEL_I, returns all trim variants.
    """
    if entry is None:
        return None
    for ind_tok in entry.indicators:
        if ind_tok in (POS_I_TRIM_FIRST, POS_I_TRIM_LAST,
                       POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER):
            return [ind_tok]
    # DEL_I is a generic deletion — try all trim positions
    if DEL_I in entry.indicators:
        return [POS_I_TRIM_FIRST, POS_I_TRIM_LAST,
                POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER]
    return None


def _product_capped(lists, cap=500):
    """Cartesian product with a cap on total combinations."""
    if not lists:
        yield ()
        return
    count = 0
    first, *rest = lists
    for item in first:
        for combo in _product_capped(rest, cap):
            yield (item,) + combo
            count += 1
            if count >= cap:
                return


