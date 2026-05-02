"""Catalog-driven signature solver.

Simple loop: analyze words → match against catalog → execute → done.
The complexity lives in the catalog (data) and matcher (logic),
not in ad-hoc strategy functions.
"""

from .tokens import *
from .db import RefDB
from .word_analyzer import analyze_words, analyze_phrases, clean_word
from .catalog import CATALOG
from .matcher import match_signatures
from .positional_catalog import POSITIONAL_CATALOG
from .positional_matcher import match_positional
from .base_catalog import BASE_CATALOG
from .base_matcher import match_base
from . import executor
from .confidence import score_result


class SignatureResult:
    """Result of a successful signature match."""

    def __init__(self, signature, word_roles, explanation_parts):
        self.signature = signature          # list of tokens
        self.word_roles = word_roles        # list of (word, token, value)
        self.explanation_parts = explanation_parts  # human-readable breakdown

        # Promote LNK -> DBE_MARKER for clue words that ARE DBE markers
        # (maybe / perhaps / say / for example / ...). The role lets the
        # explanation render "via 'maybe'" annotations and stops the
        # marker disappearing into anonymous filler.
        try:
            from .grammar_triage import _annotate_dbe_markers
            wp_words = [r[0] for r in word_roles]
            _annotate_dbe_markers(wp_words, self.word_roles)
        except Exception:
            pass

    def signature_str(self):
        return " · ".join(self.signature)

    def __repr__(self):
        return f"<SignatureResult: {self.signature_str()}>"


class SolveResult:
    """Full result from solve(), including evidence for API fallback."""

    def __init__(self, result, confidence, confidence_reasons, analyses, phrases):
        self.result = result              # SignatureResult or None
        self.confidence = confidence      # 0-100
        self.confidence_reasons = confidence_reasons
        self.analyses = analyses          # word analyses for API evidence
        self.phrases = phrases            # phrase analyses

    @property
    def solved(self):
        return self.result is not None

    @property
    def high_confidence(self):
        return self.confidence >= 80

    @property
    def medium_confidence(self):
        return 50 <= self.confidence < 80

    def evidence_summary(self):
        """Format word analyses as structured evidence for an API call."""
        lines = []
        for wa in self.analyses:
            roles = []
            for tok, vals in wa.roles.items():
                if tok in (HID_F, ANA_F, POS_F):
                    continue
                if vals == [True]:
                    roles.append(tok)
                else:
                    short = vals[:5]
                    roles.append(f"{tok}({','.join(str(v) for v in short)})")
            if roles:
                lines.append(f"  {wa.text}: {' | '.join(roles)}")
        return "\n".join(lines)


def extract_definition_candidates(clue_words, answer, db, max_def_words=4):
    """Try definition candidates from both ends of the clue.

    For each candidate, check if it's a synonym of the answer using
    definition_answers_augmented and synonyms_pairs.

    Returns list of (definition_phrase, wordplay_words) tuples,
    ordered by likelihood (exact DB matches first, shorter defs first).
    """
    candidates = []

    for n in range(1, min(max_def_words + 1, len(clue_words))):
        # Try n words from the start
        def_phrase = " ".join(clue_words[:n])
        wp_words = clue_words[n:]
        if wp_words and db.is_definition_of(def_phrase, answer):
            candidates.append((def_phrase, wp_words))

        # Try n words from the end
        def_phrase = " ".join(clue_words[-n:])
        wp_words = clue_words[:-n]
        if wp_words and db.is_definition_of(def_phrase, answer):
            candidates.append((def_phrase, wp_words))

    return candidates


def _normalize_clue(text):
    """Normalize clue text: strip smart quotes, accented chars, etc."""
    import unicodedata
    # Normalize unicode (e.g. à -> a, é -> e)
    nfkd = unicodedata.normalize('NFKD', text)
    ascii_text = ''.join(c for c in nfkd if not unicodedata.combining(c))
    # Replace smart quotes/dashes with plain equivalents
    for old, new in [('\u2018', "'"), ('\u2019', "'"), ('\u201c', '"'),
                     ('\u201d', '"'), ('\u2013', '-'), ('\u2014', '-')]:
        ascii_text = ascii_text.replace(old, new)
    return ascii_text


def solve_clue(clue_text, answer, db, min_confidence=0, extra_catalog=None,
               extra_synonyms=None, extra_indicators=None,
               _dbe_already_attempted=False):
    """Solve a raw clue: extract definition candidates, then solve each.

    Tries grammar-guided triage first (fast, high precision) on each
    definition candidate. Falls through to catalog-based solve if
    grammar triage returns nothing.

    Args:
        clue_text: the full clue text (string)
        answer: the known answer
        db: RefDB instance
        min_confidence: minimum confidence score to accept (0-100)
        extra_catalog: optional list of additional CatalogEntry objects
                       (injected from P's discoveries in Phase 3)
        extra_synonyms: optional dict mapping clue-word (lowercase) ->
                       list of additional uppercase synonym candidates.
                       Used for DBE-Haiku-derived suggestions. None /
                       empty dict is a no-op.
        extra_indicators: optional dict mapping clue-word -> list of
                       (wordplay_type, subtype, confidence) tuples to
                       inject as additional indicator entries. Used by
                       the indicator-enrichment retry path. No-op when
                       None or empty.

    Returns:
        SolveResult (best result across all definition candidates)
    """
    # Wrap the DB with overlays if extras were supplied. No-op when empty.
    if extra_synonyms:
        db = db.with_extra_synonyms(extra_synonyms)
    if extra_indicators:
        db = db.with_extra_indicators(extra_indicators)

    clue_words = _normalize_clue(clue_text).strip().split()
    answer_clean = answer.upper().replace(" ", "").replace("-", "")

    candidates = extract_definition_candidates(clue_words, answer_clean, db)

    # Track whether the Haiku definition fallback has been consulted.
    # Two invocation points: (1) here, when RefDB returns no candidates
    # at all, and (2) below, as a second-chance after every candidate
    # fails to produce a confident solve. The flag prevents redundant
    # calls and makes the gating explicit.
    haiku_tried = False

    # --- Haiku definition fallback (near-free) ---
    # If no definition candidates found in RefDB, ask Haiku
    if not candidates:
        haiku_tried = True
        try:
            from .haiku_definition import find_definition as haiku_find_def
            haiku_result = haiku_find_def(clue_text, answer)
            if haiku_result:
                def_phrase, wp_words = haiku_result
                candidates.append((def_phrase, wp_words))
        except Exception:
            pass

    best_sr = None

    # --- Grammar-guided triage (fast path) ---
    try:
        from .grammar_triage import grammar_triage

        # Try with each definition candidate
        for def_phrase, wp_words in candidates:
            gt_result = grammar_triage(clue_text, answer_clean, db,
                                       def_phrase=def_phrase, wp_words=wp_words)
            if gt_result and gt_result.confidence >= 80:
                gt_result.definition = def_phrase
                if best_sr is None or gt_result.confidence > best_sr.confidence:
                    best_sr = gt_result
                if gt_result.confidence >= 90:
                    return best_sr

        # No definition found or grammar failed with all candidates —
        # try full clue as wordplay (definition unknown)
        if best_sr is None or best_sr.confidence < 80:
            gt_result = grammar_triage(clue_text, answer_clean, db,
                                       wp_words=clue_words)
            if gt_result and gt_result.confidence >= 80:
                # If candidates exist (DB- or Haiku-supplied), preserve the
                # first candidate's def_phrase rather than discarding to None.
                # Without this, a confident full-clue parse silently overrides
                # a known-correct definition (verified for ABBA, CLOWNISH).
                # Wordplay parse may double-use def words; the user gets a
                # semantic definition rather than nothing.
                gt_result.definition = candidates[0][0] if candidates else None
                if best_sr is None or gt_result.confidence > best_sr.confidence:
                    best_sr = gt_result
    except Exception:
        pass  # Grammar triage unavailable — continue with catalog solver

    # If grammar triage found a high-confidence result, return it
    if best_sr is not None and best_sr.confidence >= 80:
        return best_sr

    # --- Catalog-based solve (existing path) ---
    for def_phrase, wp_words in candidates:
        # Determine def position: if wp_words are after def, def is at start
        if clue_words[:len(clue_words) - len(wp_words)] == clue_words[:len(def_phrase.split())]:
            def_pos = 'start'
        else:
            def_pos = 'end'

        sr = solve(wp_words, answer_clean, db, min_confidence, def_pos=def_pos,
                   extra_catalog=extra_catalog)
        if sr.solved:
            sr.definition = def_phrase
            if best_sr is None or sr.confidence > best_sr.confidence:
                best_sr = sr
            if sr.high_confidence:
                break

    # --- DBE-Haiku fallback ---
    # If nothing high-confidence yet AND the clue has DBE-marked words AND
    # we haven't already retried with extras, ask Haiku for category-mate /
    # famous-bearer suggestions for each marked word, then recurse with them
    # injected as extra synonyms. The recursion path skips this block
    # because extra_synonyms will be non-empty on the second pass.
    #
    # Every valid Haiku candidate is collected for enrichment review,
    # regardless of whether it enables a successful solve. We paid for the
    # call; the candidate already passed the answer-substring filter; it
    # deserves human review either way. The pipeline reads the attribute
    # `dbe_haiku_candidates` and queues each pair via queue_dbe_enrichment.
    dbe_haiku_candidates = {}
    if ((best_sr is None or not best_sr.high_confidence)
            and not extra_synonyms and not _dbe_already_attempted):
        try:
            from .grammar_triage import _dbe_marked_indices
            from .haiku_dbe import find_dbe_candidates
            marked_idx = _dbe_marked_indices(clue_words)
            if marked_idx:
                for idx in marked_idx:
                    word = clue_words[idx]
                    cands = find_dbe_candidates(word, answer_clean, clue_text)
                    if cands:
                        dbe_haiku_candidates[word] = cands
                if dbe_haiku_candidates:
                    new_sr = solve_clue(clue_text, answer, db, min_confidence,
                                         extra_catalog=extra_catalog,
                                         extra_synonyms=dbe_haiku_candidates)
                    if (new_sr is not None and new_sr.solved
                            and (best_sr is None
                                 or new_sr.confidence > best_sr.confidence)):
                        new_sr.dbe_haiku_candidates = dbe_haiku_candidates
                        return new_sr
        except Exception:
            pass  # Any failure here just falls through to the existing fallback

    # --- Indicator-enrichment fallback ---
    # If still nothing high-confidence AND we haven't already retried with
    # a proposed indicator, see if there's a missing positional indicator
    # we could suggest. Runs WITH the DBE-augmented synonym pool so the two
    # enrichment passes can chain (e.g. NOMADIC needs DBE→DAMON AND
    # indicator→exhausted to complete).
    suggested_indicator = None  # (word, type, subtype) of any one we tried

    def _suggestion_actually_used(sr_obj, sugg_word, sugg_wp_type, sugg_subtype):
        """Confirm the proposed indicator word ended up with the matching
        indicator role in the winning parse. Otherwise the matcher solved
        the clue via some unrelated path and the suggestion didn't help —
        we shouldn't queue it as if it did."""
        if sr_obj is None or sr_obj.result is None:
            return False
        # Map (wp_type, subtype) -> the POS_I_* token the role would carry.
        from .tokens import (
            POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
            POS_I_ALTERNATE, POS_I_TRIM_FIRST, POS_I_TRIM_LAST,
            POS_I_TRIM_OUTER, POS_I_TRIM_MIDDLE,
            ANA_I, REV_I, CON_I, DEL_I, HID_I, HOM_I,
        )
        SUBTYPE_TO_TOKEN = {
            'first_use': POS_I_FIRST, 'last_use': POS_I_LAST,
            'outer_use': POS_I_OUTER, 'center_use': POS_I_MIDDLE,
            'inner_use': POS_I_MIDDLE, 'alternate': POS_I_ALTERNATE,
            'odd': POS_I_ALTERNATE, 'even': POS_I_ALTERNATE,
            'first_delete': POS_I_TRIM_FIRST,
            'last_delete': POS_I_TRIM_LAST,
            'tail_delete': POS_I_TRIM_LAST,
            'outer_delete': POS_I_TRIM_OUTER,
            'center_delete': POS_I_TRIM_MIDDLE,
        }
        TYPE_TO_TOKEN = {
            'anagram': ANA_I, 'reversal': REV_I, 'container': CON_I,
            'deletion': DEL_I, 'hidden': HID_I, 'homophone': HOM_I,
            'acrostic': POS_I_FIRST,
        }
        if sugg_wp_type == 'parts':
            target_token = SUBTYPE_TO_TOKEN.get(sugg_subtype)
        else:
            target_token = TYPE_TO_TOKEN.get(sugg_wp_type)
        if not target_token:
            return False
        sw_clean = sugg_word.lower().strip(",.;:!?\"'()-")
        for word, tok, _val, *_ in sr_obj.result.word_roles:
            if (word.lower().strip(",.;:!?\"'()-") == sw_clean
                    and tok == target_token):
                return True
        return False

    if (best_sr is None or not best_sr.high_confidence) and not extra_indicators:
        try:
            from .indicator_detect import detect_missing_indicator
            from .haiku_indicator import find_indicator_candidate
            wp_for_detection = candidates[0][1] if candidates else clue_words

            # Compute used word indices from any partial best_sr.
            used_indices = set()
            if best_sr is not None and best_sr.result is not None:
                from .tokens import LNK
                for i, (w, tok, _v) in enumerate(best_sr.result.word_roles):
                    if tok != LNK:
                        used_indices.add(i)

            # Pass 1: rule-based suggestions. We try them in score order;
            # the first that produces a verified HIGH solve wins. AFTER
            # success, if there were tied alternatives (same indicator
            # word + same extract value, different subtype), ask Haiku to
            # pick the semantically right subtype — that's what we queue.
            rule_suggs = detect_missing_indicator(
                wp_for_detection, answer_clean, db, used_indices)

            for s in rule_suggs[:3]:
                inj = {s['indicator_word']:
                       [(s['indicator_type'], s['subtype'], 'medium')]}
                new_sr = solve_clue(clue_text, answer, db, min_confidence,
                                     extra_catalog=extra_catalog,
                                     extra_synonyms=(extra_synonyms or
                                                       dbe_haiku_candidates),
                                     extra_indicators=inj,
                                     _dbe_already_attempted=True)
                if (new_sr is not None and new_sr.high_confidence
                        and (best_sr is None
                             or new_sr.confidence > best_sr.confidence)
                        and _suggestion_actually_used(
                            new_sr, s['indicator_word'],
                            s['indicator_type'], s['subtype'])):
                    # Verified: 'exhausted' (or whatever) IS the indicator.
                    # If there are tied alternatives in rule_suggs, ask
                    # Haiku to pick the semantically right subtype. If
                    # Haiku picks a DIFFERENT subtype AND that subtype
                    # also operationally verifies, use it — so the parse
                    # mechanism matches the queued subtype.
                    tied_subtypes = sorted({
                        x['subtype'] for x in rule_suggs
                        if x['indicator_word'] == s['indicator_word']
                        and x['extract_value'] == s['extract_value']
                    })
                    final_sr = new_sr
                    final_subtype = s['subtype']
                    if len(tied_subtypes) > 1:
                        from .haiku_indicator import disambiguate_subtype as _dis
                        pick = _dis(s['indicator_word'], tied_subtypes,
                                     clue_text)
                        if pick and pick in tied_subtypes and pick != s['subtype']:
                            inj_pick = {s['indicator_word']:
                                        [(s['indicator_type'], pick, 'medium')]}
                            haiku_sr = solve_clue(
                                clue_text, answer, db, min_confidence,
                                extra_catalog=extra_catalog,
                                extra_synonyms=(extra_synonyms or
                                                  dbe_haiku_candidates),
                                extra_indicators=inj_pick,
                                _dbe_already_attempted=True)
                            if (haiku_sr is not None
                                    and haiku_sr.high_confidence
                                    and _suggestion_actually_used(
                                        haiku_sr, s['indicator_word'],
                                        s['indicator_type'], pick)):
                                # Haiku's subtype works AND verifies — prefer
                                # it; the parse mechanism now matches the
                                # subtype we'll queue.
                                final_sr = haiku_sr
                                final_subtype = pick
                            elif pick:
                                # Haiku picked but matcher couldn't verify.
                                # Stick with the operationally-verified one.
                                final_subtype = s['subtype']
                    suggested_indicator = (
                        s['indicator_word'], s['indicator_type'],
                        final_subtype)
                    final_sr.suggested_indicators = [suggested_indicator]
                    if dbe_haiku_candidates:
                        final_sr.dbe_haiku_candidates = dbe_haiku_candidates
                    return final_sr

            # Pass 2: Haiku fallback. Only fire if rule-based produced
            # at least one fodder candidate (tells us what gap to fill).
            if rule_suggs:
                top = rule_suggs[0]
                hk_suggs = find_indicator_candidate(
                    clue_text, answer_clean,
                    top['fodder_word'], top['extract_value'])
                for s in hk_suggs[:3]:
                    inj = {s['indicator_word']:
                           [(s['indicator_type'], s['subtype'], 'medium')]}
                    new_sr = solve_clue(clue_text, answer, db, min_confidence,
                                         extra_catalog=extra_catalog,
                                         extra_synonyms=(extra_synonyms or
                                                           dbe_haiku_candidates),
                                         extra_indicators=inj,
                                         _dbe_already_attempted=True)
                    if (new_sr is not None and new_sr.high_confidence
                            and (best_sr is None
                                 or new_sr.confidence > best_sr.confidence)
                            and _suggestion_actually_used(
                                new_sr, s['indicator_word'],
                                s['indicator_type'], s['subtype'])):
                        suggested_indicator = (
                            s['indicator_word'], s['indicator_type'],
                            s['subtype'])
                        new_sr.suggested_indicators = [suggested_indicator]
                        if dbe_haiku_candidates:
                            new_sr.dbe_haiku_candidates = dbe_haiku_candidates
                        return new_sr
        except Exception:
            pass  # Any failure here just falls through

    def _attach_and_return(sr):
        if sr is not None:
            if dbe_haiku_candidates:
                sr.dbe_haiku_candidates = dbe_haiku_candidates
            if suggested_indicator:
                sr.suggested_indicators = [suggested_indicator]
        return sr

    # --- Haiku definition second-chance ---
    # If we got here without a confident solve AND Haiku hasn't been
    # consulted (because DB candidates were non-empty at the top of
    # this function), give Haiku a chance now. The DB candidates may
    # have been weak fuzzy matches that prevented the initial Haiku
    # gate from firing yet failed to produce a parse. A successful
    # Haiku candidate is prepended to `candidates` so the unsolved-
    # fallback path below uses its def_phrase rather than the failed
    # DB one; we also retry grammar-triage and catalog-solve with it
    # in case the better definition unlocks a parse.
    if (best_sr is None or not best_sr.high_confidence) and not haiku_tried:
        haiku_tried = True
        try:
            from .haiku_definition import find_definition as haiku_find_def
            haiku_result = haiku_find_def(clue_text, answer)
            if haiku_result:
                h_def, h_wp = haiku_result
                already_tried = any(dp == h_def for dp, _ in candidates)
                if not already_tried:
                    candidates.insert(0, (h_def, h_wp))
                    # Retry grammar triage with the new candidate
                    try:
                        from .grammar_triage import grammar_triage
                        gt_result = grammar_triage(
                            clue_text, answer_clean, db,
                            def_phrase=h_def, wp_words=h_wp)
                        if gt_result and gt_result.confidence >= 80:
                            gt_result.definition = h_def
                            if (best_sr is None
                                    or gt_result.confidence > best_sr.confidence):
                                best_sr = gt_result
                    except Exception:
                        pass
                    # Retry catalog solve with the new candidate
                    if best_sr is None or not best_sr.high_confidence:
                        if (clue_words[:len(clue_words) - len(h_wp)]
                                == clue_words[:len(h_def.split())]):
                            def_pos = 'start'
                        else:
                            def_pos = 'end'
                        sr_h = solve(h_wp, answer_clean, db, min_confidence,
                                     def_pos=def_pos,
                                     extra_catalog=extra_catalog)
                        if sr_h.solved:
                            sr_h.definition = h_def
                            if (best_sr is None
                                    or sr_h.confidence > best_sr.confidence):
                                best_sr = sr_h
        except Exception:
            pass

    if best_sr is not None:
        return _attach_and_return(best_sr)

    # No definition candidate worked — fall back to best unsolved result
    if candidates:
        def_phrase, wp_words = candidates[0]
        sr = solve(wp_words, answer_clean, db, min_confidence=0,
                   extra_catalog=extra_catalog)
        sr.definition = def_phrase
        return _attach_and_return(sr)

    # No definition candidates found at all — try full clue as wordplay
    sr = solve(clue_words, answer_clean, db, min_confidence=0,
               extra_catalog=extra_catalog)
    sr.definition = None
    return _attach_and_return(sr)


def solve(wordplay_words, answer, db, min_confidence=0, def_pos=None,
          extra_catalog=None):
    """Solve using catalog-driven signature matching.

    Args:
        wordplay_words: list of words from the wordplay window
        answer: the known answer (uppercase, no spaces/hyphens)
        db: RefDB instance
        min_confidence: minimum confidence score to accept (0-100)
        def_pos: "start" or "end" (where definition is) or None
        extra_catalog: optional list of additional CatalogEntry objects

    Returns:
        SolveResult with .result, .confidence, .confidence_reasons, .analyses
    """
    answer_clean = answer.upper().replace(" ", "").replace("-", "")

    # Step 1: Analyse all words
    analyses, phrases = analyze_phrases(wordplay_words, answer, db)

    # Step 2: Match against catalogs and execute
    best_result = None
    best_confidence = -1
    best_reasons = []

    def _process_match(entry, assignment):
        """Process a verified match — returns (result, confidence, reasons) or None."""
        # Execute (builds explanation)
        success, explanation, pieces = executor.execute_signature(
            entry, assignment, wordplay_words, answer_clean
        )
        if not success:
            return None

        # Build SignatureResult
        sig_tokens = [tok for _, tok, _ in pieces]
        for ind_tok, ind_idx in assignment['indicator_indices'].items():
            sig_tokens.append(ind_tok)
        for lnk_idx in assignment.get('lnk_indices', set()):
            sig_tokens.append(LNK)

        word_roles = list(pieces)
        for ind_tok, ind_idx in assignment['indicator_indices'].items():
            if isinstance(ind_idx, tuple):
                phrase = " ".join(wordplay_words[k] for k in range(ind_idx[0], ind_idx[1]))
                word_roles.append((phrase, ind_tok, None))
            else:
                word_roles.append((wordplay_words[ind_idx], ind_tok, None))
        for lnk_idx in assignment.get('lnk_indices', set()):
            word_roles.append((wordplay_words[lnk_idx], LNK, None))

        sig_result = SignatureResult(sig_tokens, word_roles, [explanation])
        confidence, reasons = score_result(
            sig_result, wordplay_words, answer_clean, analyses, db
        )
        return sig_result, confidence, reasons

    # Try base matcher first (flexible spans, collapsed patterns)
    for entry, assignment in match_base(
        wordplay_words, analyses, phrases, BASE_CATALOG,
        answer_clean, db, def_pos=def_pos
    ):
        result = _process_match(entry, assignment)
        if result is None:
            continue

        sig_result, confidence, reasons = result
        if confidence > best_confidence:
            best_result = sig_result
            best_confidence = confidence
            best_reasons = reasons

        if confidence >= 80:
            break

    # If base matcher found high confidence, return immediately
    if best_confidence >= 80:
        return SolveResult(best_result, best_confidence, best_reasons,
                           analyses, phrases)

    # Fall back to positional matcher (fixed spans, more entries)
    for entry, assignment in match_positional(
        wordplay_words, analyses, phrases, POSITIONAL_CATALOG,
        answer_clean, db, def_pos=def_pos
    ):
        result = _process_match(entry, assignment)
        if result is None:
            continue

        sig_result, confidence, reasons = result
        if confidence > best_confidence:
            best_result = sig_result
            best_confidence = confidence
            best_reasons = reasons

        if confidence >= 80:
            break

    if best_confidence >= 80:
        return SolveResult(best_result, best_confidence, best_reasons,
                           analyses, phrases)

    # Fall back to old catalog matcher for patterns not in other catalogs
    catalog_to_use = CATALOG + extra_catalog if extra_catalog else CATALOG
    if extra_catalog:
        print("      [DEBUG] Old catalog matcher: %d base + %d extra entries" % (
            len(CATALOG), len(extra_catalog)))
        print("      [DEBUG] Wordplay words: %s" % wordplay_words)
        print("      [DEBUG] Extra entries: %s" % [
            (e.label, e.operation, e.tokens, e.word_spans) for e in extra_catalog])
    for entry, assignment in match_signatures(
        wordplay_words, analyses, phrases, catalog_to_use, answer_clean, db
    ):
        result = _process_match(entry, assignment)
        if result is None:
            continue

        sig_result, confidence, reasons = result
        if confidence > best_confidence:
            best_result = sig_result
            best_confidence = confidence
            best_reasons = reasons

        if confidence >= 80:
            break

    # Return best result
    if best_result is not None and best_confidence >= min_confidence:
        return SolveResult(best_result, best_confidence, best_reasons,
                           analyses, phrases)

    return SolveResult(None, max(0, best_confidence), best_reasons,
                       analyses, phrases)
