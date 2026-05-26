# Phase 2 Anagram Solver Fix — Codex Instruction

## Task

Fix _try_anagram and _build_anagram_result in
signature_solver/grammar_triage.py so that:

1. Words that substitute their abbreviation or synonym value into the
   anagram pool are tagged ABR_F or SYN_F, not ANA_F.
2. Two-word phrase substitutions (e.g. "in charge" -> IC) are tried
   before single-word substitutions and rank higher.
3. Candidates are collected and ranked, not returned on first fit.

Only this one file changes. Do not touch any other file.


---

## Background

The current _try_anagram has a "substitution" section (after the
plain exclusion attempts) that replaces a word's raw letters with a
short DB value (1-2 chars) obtained from _get_word_values. The value
is correct, but the substituted word is still tagged ANA_F in
_build_anagram_result. ANA_F means "raw letters unchanged", so the
tag is wrong for any word whose value came from a DB lookup.

The new derivation verifier in stage_two_casefile.py (being
implemented separately) checks ANA_F tokens by requiring value ==
full cleaned source text. A substituted word tagged ANA_F therefore
fails that check as "unlicensed_partial". That is the correct
outcome for a genuinely bad parse. However, it is too blunt: a word
that legitimately contributes its abbreviation (e.g. "charge" -> C)
should be tagged ABR_F, not ANA_F. When tagged ABR_F the verifier
checks db.get_abbreviations("charge") and accepts C. When tagged
SYN_F it checks db.get_synonyms(...). If the DB entry does not
support the value, the verifier correctly returns REVIEW.

Additionally, the current code never tries phrase abbreviations such
as "in charge" -> IC. This means the solver misses better parses and
falls back to fragmented single-word substitutions instead.


---

## Change 1: _build_anagram_result

### Current signature

    def _build_anagram_result(wp_words, word_letters, excluded, answer, db):

### New signature

    def _build_anagram_result(wp_words, word_letters, excluded, answer, db,
                              word_overrides=None):

word_overrides is an optional dict mapping word index (int) to a
(value, token_type) tuple. When an index is present in word_overrides,
use that token and value instead of (ANA_F, word_letters[k]).

### Replace the full function body with

    def _build_anagram_result(wp_words, word_letters, excluded, answer, db,
                              word_overrides=None):
        """Build SolveResult for an anagram."""
        word_roles = []
        fodder = ''.join(
            (word_overrides[k][0]
             if word_overrides and k in word_overrides
             else word_letters[k])
            for k in range(len(wp_words))
            if k not in excluded
        )

        for k, word in enumerate(wp_words):
            if k in excluded:
                ind_types = db.get_indicator_types(_clean(word))
                is_ana_ind = any(t == 'anagram' for t, _, _ in ind_types)
                if is_ana_ind:
                    word_roles.append((word, ANA_I, None))
                else:
                    word_roles.append((word, LNK, None))
            elif word_overrides and k in word_overrides:
                override_val, override_token = word_overrides[k]
                word_roles.append((word, override_token, override_val))
            else:
                word_roles.append((word, ANA_F, word_letters[k]))

        has_indicator = any(t == ANA_I for _, t, _ in word_roles)
        non_ana_tokens = list(dict.fromkeys(
            t for _, t, _ in word_roles
            if t not in (ANA_F, ANA_I, LNK)
        ))
        operation = 'anagram_charade' if non_ana_tokens else 'anagram'
        if has_indicator:
            sig_tokens = [ANA_I, ANA_F] + non_ana_tokens
        else:
            sig_tokens = [ANA_F] + non_ana_tokens
        sig_tokens = list(dict.fromkeys(sig_tokens))
        explanation = 'Anagram of "%s" = %s' % (fodder, answer)
        sig = SignatureResult(sig_tokens, word_roles, [explanation])
        confidence = 90 if has_indicator else 70
        return SolveResult(sig, confidence, [(operation, 0)], [], {})

Key points:
- Existing callers (plain exclusion paths at lines 284, 290, 298)
  pass no word_overrides, so their behaviour is unchanged.
- The fodder string uses the override value when present, so the
  explanation reflects what the solver actually used.
- non_ana_tokens drives both the operation label and the signature.


---

## Change 2: _try_anagram — replace the substitution section

### Current code (lines 300-356)

    # Anagram with abbreviation substitution: some words contribute their
    # abbreviation/synonym value instead of raw letters (e.g. "western"=W, "area"=A)
    # Try substituting 1-2 words with their short DB values
    answer_len = len(answer)
    short_vals = {}  # word_idx -> list of short values (1-2 chars)
    for k in range(n):
        vals = []
        for val, src in _get_word_values(wp_words[k], db, answer_len):
            if len(val) <= 2:
                vals.append((val, src))
        if vals:
            short_vals[k] = vals

    if short_vals:
        from itertools import combinations
        sub_indices = list(short_vals.keys())

        # Try 1 substitution
        for si in sub_indices:
            for sub_val, sub_src in short_vals[si]:
                others = [k for k in range(n) if k != si]
                for n_exc in range(0, len(others) + 1):
                    for exc_combo in combinations(others, n_exc):
                        remaining = sub_val + ''.join(
                            word_letters[k] for k in range(n)
                            if k != si and k not in exc_combo
                        )
                        if sorted(remaining) == answer_sorted:
                            # Build result with substituted word
                            modified_letters = list(word_letters)
                            modified_letters[si] = sub_val
                            return _build_anagram_result(
                                wp_words, modified_letters, set(exc_combo), answer, db
                            )

        # Try 2 substitutions
        if len(sub_indices) >= 2:
            for s1, s2 in combinations(sub_indices, 2):
                for v1, src1 in short_vals[s1]:
                    for v2, src2 in short_vals[s2]:
                        others = [k for k in range(n) if k not in (s1, s2)]
                        for n_exc in range(0, len(others) + 1):
                            for exc_combo in combinations(others, n_exc):
                                remaining = v1 + v2 + ''.join(
                                    word_letters[k] for k in range(n)
                                    if k not in (s1, s2) and k not in exc_combo
                                )
                                if sorted(remaining) == answer_sorted:
                                    modified_letters = list(word_letters)
                                    modified_letters[s1] = v1
                                    modified_letters[s2] = v2
                                    return _build_anagram_result(
                                        wp_words, modified_letters,
                                        set(exc_combo), answer, db
                                    )

    return None

### Replace with

    # Substitution section: collect candidates by priority, return best.
    # Priority 2: two adjacent words replaced by a single phrase DB value.
    # Priority 1: one word replaced by its DB abbreviation or synonym.
    # Priority 0: two words each replaced by their DB abbreviation or synonym.
    # Plain exclusion paths above already returned if they fit, so we only
    # reach here when raw letter totals do not match.
    from itertools import combinations

    answer_len = len(answer)
    candidates = []  # list of (SolveResult, priority)

    # --- Priority 2: phrase substitution ---
    # Try each adjacent pair as a single phrase DB lookup before any
    # single-word substitutions.  _get_phrase_values handles multi-word
    # synonym and abbreviation lookups.
    for pi in range(n - 1):
        pj = pi + 1
        phrase_vals = _get_phrase_values(
            [wp_words[pi], wp_words[pj]], db, answer_len)
        for val, src in phrase_vals:
            if len(val) > 3:
                continue
            others = [k for k in range(n) if k not in (pi, pj)]
            for n_exc in range(0, len(others) + 1):
                for exc_combo in combinations(others, n_exc):
                    remaining = val + ''.join(
                        word_letters[k] for k in range(n)
                        if k not in (pi, pj) and k not in exc_combo
                    )
                    if sorted(remaining) != answer_sorted:
                        continue
                    # Build word_roles directly: phrase collapses to one entry.
                    token = ABR_F if src == 'abbreviation' else SYN_F
                    phrase_text = wp_words[pi] + ' ' + wp_words[pj]
                    phrase_fodder = val + ''.join(
                        word_letters[k] for k in range(n)
                        if k not in (pi, pj) and k not in exc_combo
                    )
                    # Build word_roles in clue order.
                    # At position pi insert the phrase entry; skip pj.
                    word_roles = []
                    for k in range(n):
                        if k == pi:
                            word_roles.append((phrase_text, token, val))
                        elif k == pj:
                            continue
                        elif k in exc_combo:
                            ind_types = db.get_indicator_types(_clean(wp_words[k]))
                            is_ana = any(t == 'anagram' for t, _, _ in ind_types)
                            if is_ana:
                                word_roles.append((wp_words[k], ANA_I, None))
                            else:
                                word_roles.append((wp_words[k], LNK, None))
                        else:
                            word_roles.append((wp_words[k], ANA_F, word_letters[k]))
                    has_indicator = any(t == ANA_I for _, t, _ in word_roles)
                    non_ana_tokens = list(dict.fromkeys(
                        t for _, t, _ in word_roles
                        if t not in (ANA_F, ANA_I, LNK)
                    ))
                    if has_indicator:
                        sig_tokens = [ANA_I, ANA_F] + non_ana_tokens
                    else:
                        sig_tokens = [ANA_F] + non_ana_tokens
                    sig_tokens = list(dict.fromkeys(sig_tokens))
                    explanation = 'Anagram of "%s" = %s' % (phrase_fodder, answer)
                    sig = SignatureResult(sig_tokens, word_roles, [explanation])
                    confidence = 85 if has_indicator else 70
                    result = SolveResult(
                        sig, confidence, [('anagram_charade', 0)], [], {})
                    candidates.append((result, 2))

    # If any phrase substitution found, skip lower-priority searches.
    if not candidates:
        # --- Build single-word short value lookup ---
        short_vals = {}
        for k in range(n):
            vals = []
            for val, src in _get_word_values(wp_words[k], db, answer_len):
                if len(val) <= 2:
                    vals.append((val, src))
            if vals:
                short_vals[k] = vals

        if short_vals:
            sub_indices = list(short_vals.keys())

            # --- Priority 1: one word substituted ---
            for si in sub_indices:
                for sub_val, sub_src in short_vals[si]:
                    others = [k for k in range(n) if k != si]
                    for n_exc in range(0, len(others) + 1):
                        for exc_combo in combinations(others, n_exc):
                            remaining = sub_val + ''.join(
                                word_letters[k] for k in range(n)
                                if k != si and k not in exc_combo
                            )
                            if sorted(remaining) == answer_sorted:
                                modified_letters = list(word_letters)
                                modified_letters[si] = sub_val
                                override_token = (
                                    ABR_F if sub_src == 'abbreviation' else SYN_F)
                                result = _build_anagram_result(
                                    wp_words, modified_letters,
                                    set(exc_combo), answer, db,
                                    word_overrides={si: (sub_val, override_token)},
                                )
                                candidates.append((result, 1))

            # --- Priority 0: two words substituted ---
            if len(sub_indices) >= 2:
                for s1, s2 in combinations(sub_indices, 2):
                    for v1, src1 in short_vals[s1]:
                        for v2, src2 in short_vals[s2]:
                            others = [k for k in range(n) if k not in (s1, s2)]
                            for n_exc in range(0, len(others) + 1):
                                for exc_combo in combinations(others, n_exc):
                                    remaining = v1 + v2 + ''.join(
                                        word_letters[k] for k in range(n)
                                        if k not in (s1, s2)
                                        and k not in exc_combo
                                    )
                                    if sorted(remaining) == answer_sorted:
                                        modified_letters = list(word_letters)
                                        modified_letters[s1] = v1
                                        modified_letters[s2] = v2
                                        tok1 = (
                                            ABR_F if src1 == 'abbreviation'
                                            else SYN_F)
                                        tok2 = (
                                            ABR_F if src2 == 'abbreviation'
                                            else SYN_F)
                                        result = _build_anagram_result(
                                            wp_words, modified_letters,
                                            set(exc_combo), answer, db,
                                            word_overrides={
                                                s1: (v1, tok1),
                                                s2: (v2, tok2),
                                            },
                                        )
                                        candidates.append((result, 0))

    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    return candidates[0][0]

Key points:
- The phrase loop runs before the short_vals loop and is gated with
  `if not candidates` so lower-priority paths are skipped once a
  phrase candidate is found.
- phrase_text collapses "in" + "charge" into "in charge" — one role
  entry carries the phrase, its token (ABR_F/SYN_F), and its value.
- word_overrides carries the source token for single/two-word subs,
  so _build_anagram_result stores ABR_F/SYN_F not ANA_F.
- candidates.sort is stable so within a priority tier the first fit
  is preserved.
- The existing plain exclusion paths (lines 283-298, unchanged) still
  return immediately before this section is reached.


---

## What not to change

Do not change _try_anagram_with_positional.
Do not change _build_anagram_result callers at lines 284, 290, 298
(the plain exclusion paths — they pass no word_overrides).
Do not change any other function.
Do not modify any other file.


---

## After writing

Paste _build_anagram_result and the full replacement substitution
section of _try_anagram (from the "Substitution section" comment to
the final return statement) so the result can be audited before
anything is run.


---

## Verification

Run the focused no-write pipeline for UNDEMOCRATIC (clue id 10069315).

Expected:

If the correct parse (courted + man + in charge anagrammed to
UNDEMOCRATIC, with "in charge" -> IC as ABR_F) is now found:
  word_roles include ("in charge", ABR_F, "IC")
  word_roles include ("man", ANA_F, "MAN")
  Stage Two assembly kind = "anagram" (build_stage_two_from_solve_result
    derives kind from indicator tokens only; ANA_I sets "anagram"
    regardless of whether ABR_F source parts are also present;
    the SolveResult operation field will say "anagram_charade" but
    Stage Two does not read that field for kind assignment)
  Stage Three source_evidence = PASS for all parts
  Stage Three overall status = PASS or REVIEW depending on other checks

If the bad parse (man -> MA) is still produced at lower priority:
  It is either not returned (outranked by the correct parse), or
  Stage Three source_evidence = REVIEW because MA tagged ABR_F is
  not in db.get_abbreviations("man").

In either case the bad parse must not reach wfw_proven.

Run the focused no-write pipeline for STIPULATION (clue id 10068568).

Expected:
  Stage Three overall status = PASS.
  Display status = wfw_proven.
  This proves the plain exclusion path and _build_anagram_result
  unchanged paths are not broken.
