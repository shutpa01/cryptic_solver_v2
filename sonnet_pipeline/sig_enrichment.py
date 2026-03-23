"""Enrichment bridge: P's discoveries → S's RefDB + catalog.

Used in Phase 3 of the integrated pipeline: after P runs, collect the pieces
it discovered and inject them into a cloned RefDB and catalog so that S can
re-run with enriched data AND new signature patterns.
"""

import copy

from signature_solver.tokens import (
    SYN_F, ABR_F, ANA_F, RAW, HID_F, HOM_F, POS_F,
    ANA_I, REV_I, CON_I, DEL_I, HID_I, HOM_I,
    POS_I_FIRST, POS_I_LAST, POS_I_ALTERNATE,
    LINK_WORDS,
)
from signature_solver.catalog import CatalogEntry


def collect_gaps_from_results(results):
    """Extract synonym/abbreviation/definition pieces from P's result dicts.

    Returns list of dicts with keys: word, letters, role.
    These are formatted for enrich_refdb().
    """
    gaps = []
    seen = set()

    for r in results:
        if r.get("status") != "ASSEMBLED":
            continue
        # Inject ALL of P's discovered pieces — S's mechanical matching
        # is the quality gate (signature match + letters must equal answer).
        # Wrong pieces simply won't produce a valid S solve.

        ai = r.get("ai_output") or {}
        pieces = ai.get("pieces", [])
        answer = (r.get("answer") or "").upper().replace(" ", "").replace("-", "")
        definition = ai.get("definition")

        for p in pieces:
            mech = p.get("mechanism", "")
            word = (p.get("clue_word") or "").strip()
            letters = (p.get("letters") or "").strip().upper().replace(" ", "").replace("-", "")

            if not word or not letters:
                continue
            # Skip pieces that are the full answer (definition, not a wordplay piece)
            if letters == answer:
                continue
            # Skip non-enrichable mechanisms
            if mech in ("literal", "anagram_fodder", "first_letter", "last_letter",
                        "hidden", "alternate_letters", "core_letters"):
                continue

            if mech == "synonym":
                key = ("synonym", word.lower(), letters)
                if key not in seen:
                    seen.add(key)
                    gaps.append({"word": word, "letters": letters, "role": "synonym"})
            elif mech == "abbreviation":
                key = ("abbreviation", word.lower(), letters)
                if key not in seen:
                    seen.add(key)
                    gaps.append({"word": word, "letters": letters, "role": "abbreviation"})
            elif mech == "sound_of":
                key = ("homophone", word.lower(), letters)
                if key not in seen:
                    seen.add(key)
                    gaps.append({"word": word, "letters": letters, "role": "homophone"})

        # Also inject definition → answer if P identified one
        if definition and answer:
            key = ("definition", definition.lower(), answer)
            if key not in seen:
                seen.add(key)
                gaps.append({"word": definition, "letters": answer, "role": "synonym"})

    return gaps


def enrich_refdb(ref_db, gaps):
    """Clone RefDB and inject gaps. Returns (enriched_db, injected_list).

    This is a thin wrapper around test_integration.enrich_refdb's logic,
    adapted to work with P's result format.
    """
    enriched = copy.copy(ref_db)
    # Deep-copy only the dicts we'll mutate
    enriched.synonyms = dict(ref_db.synonyms)
    enriched.abbreviations = dict(ref_db.abbreviations)

    injected = []

    for gap in gaps:
        word = gap["word"].lower().strip(".,;:!?\"'()-")
        letters = gap["letters"].upper().replace(" ", "").replace("-", "")
        role = gap["role"]

        if not word or not letters:
            continue

        if role in ("synonym", "homophone"):
            if word not in enriched.synonyms:
                enriched.synonyms[word] = []
            else:
                enriched.synonyms[word] = list(enriched.synonyms[word])
            if letters not in enriched.synonyms[word]:
                enriched.synonyms[word].append(letters)
                injected.append("SYN: %s -> %s" % (word, letters))

        elif role == "abbreviation":
            if word not in enriched.abbreviations:
                enriched.abbreviations[word] = []
            else:
                enriched.abbreviations[word] = list(enriched.abbreviations[word])
            if letters not in enriched.abbreviations[word]:
                enriched.abbreviations[word].append(letters)
                injected.append("ABR: %s -> %s" % (word, letters))

    return enriched, injected


# ============================================================
# Indicator enrichment — infer indicator words from P's solves
# ============================================================

# Map P's wordplay_type to indicator DB type
_WTYPE_TO_INDICATOR_TYPE = {
    "anagram": "anagram",
    "container": "container",
    "reversal": "reversal",
    "deletion": "deletion",
    "hidden": "hidden",
    "homophone": "homophone",
}


def collect_indicators_from_results(results, ref_db=None):
    """Infer indicator words from P's successful solves.

    Logic: for each piece that needs an indicator (anagram fodder, container
    pieces, etc.), check whether a known indicator is already adjacent to it.
    If not — the fodder is "orphaned" — look at non-consumed words adjacent
    to the fodder (within 1 link-word gap) as candidate indicators.

    Returns list of dicts: {word: str, wordplay_type: str}
    """
    import re
    indicators = []
    seen = set()

    # Mechanisms whose fodder requires an adjacent indicator
    _NEEDS_INDICATOR = {
        "anagram": "anagram_fodder",
        "container": None,       # any piece in a container needs CON_I nearby
        "reversal": None,        # any piece in a reversal needs REV_I nearby
        "deletion": None,        # deletion needs DEL_I nearby
        "hidden": "hidden",
        "homophone": "sound_of",
    }

    for r in results:
        if r.get("status") != "ASSEMBLED":
            continue
        tier = r.get("tier") or ""
        if tier.startswith("Signature"):
            continue

        ai = r.get("ai_output") or {}
        wtype = ai.get("wordplay_type", "")
        pieces = ai.get("pieces", [])
        definition = (ai.get("definition") or "").lower().strip()
        clue_text = (r.get("clue") or "").lower()

        ind_type = _WTYPE_TO_INDICATOR_TYPE.get(wtype)
        if not ind_type or not pieces:
            continue

        # Strip enumeration
        clue_text = re.sub(r'\(\d+(?:[,-]\d+)*\)\s*$', '', clue_text).strip()
        clue_words = clue_text.split()
        n_words = len(clue_words)

        # Map each clue word index to what consumes it
        consumed = {}  # index -> "piece" | "definition"

        # Mark words consumed by pieces, and track fodder positions
        fodder_indices = set()  # indices of words that are fodder needing an indicator
        target_mech = _NEEDS_INDICATOR.get(wtype)

        for p in pieces:
            mech = p.get("mechanism", "")
            clue_word = (p.get("clue_word") or "").lower().strip()
            if not clue_word:
                continue
            piece_words = clue_word.split()
            for i in range(n_words - len(piece_words) + 1):
                window = [w.strip(".,;:!?\"'()-") for w in clue_words[i:i + len(piece_words)]]
                piece_clean = [w.strip(".,;:!?\"'()-") for w in piece_words]
                if window == piece_clean:
                    for j in range(i, i + len(piece_words)):
                        consumed[j] = "piece"
                        # Mark as fodder if this piece type needs an indicator
                        if target_mech is None or mech == target_mech:
                            fodder_indices.add(j)
                    break

        # Mark definition words
        if definition:
            def_words = definition.split()
            for i in range(n_words - len(def_words) + 1):
                window = [w.strip(".,;:!?\"'()-") for w in clue_words[i:i + len(def_words)]]
                def_clean = [w.strip(".,;:!?\"'()-") for w in def_words]
                if window == def_clean:
                    for j in range(i, i + len(def_words)):
                        consumed[j] = "definition"
                    break

        if not fodder_indices:
            continue

        # Find the boundary of fodder (leftmost and rightmost fodder word)
        fodder_min = min(fodder_indices)
        fodder_max = max(fodder_indices)

        # Check if there's already a known indicator adjacent to fodder
        def _is_known_indicator(idx):
            if idx < 0 or idx >= n_words or idx in consumed:
                return False
            w = clue_words[idx].strip(".,;:!?\"'()-").lower()
            if ref_db:
                types = ref_db.get_indicator_types(w)
                return any(t == ind_type for t, _, _ in types)
            return False

        # Look in the zone adjacent to fodder: 1 word before fodder_min,
        # 1 word after fodder_max, plus allow 1 link-word gap
        adjacent_zone = set()
        for offset in [fodder_min - 1, fodder_min - 2,
                       fodder_max + 1, fodder_max + 2]:
            if 0 <= offset < n_words and offset not in consumed:
                adjacent_zone.add(offset)

        # Check if any adjacent word is already a known indicator for this type
        has_known = any(_is_known_indicator(idx) for idx in adjacent_zone)
        if has_known:
            continue  # Fodder already has an indicator — nothing to infer

        # No known indicator adjacent — infer from adjacent non-consumed words
        for idx in adjacent_zone:
            w_clean = clue_words[idx].strip(".,;:!?\"'()-").lower()
            if not w_clean or w_clean in LINK_WORDS:
                continue
            if len(w_clean) < 3:
                continue
            # If there's a link word between this candidate and the fodder,
            # that's OK (gap of 1). But if there are 2 non-link words between
            # the candidate and fodder, skip it.
            gap_to_fodder = min(abs(idx - fodder_min), abs(idx - fodder_max))
            if gap_to_fodder > 2:
                continue

            key = (w_clean, ind_type)
            if key not in seen:
                seen.add(key)
                indicators.append({"word": w_clean, "wordplay_type": ind_type})

    return indicators


def enrich_indicators(ref_db, new_indicators):
    """Inject inferred indicator words into RefDB's indicators dict.

    Mutates ref_db.indicators in place (already a cloned copy).
    Returns list of injected descriptions.
    """
    injected = []
    for ind in new_indicators:
        word = ind["word"]
        wtype = ind["wordplay_type"]

        # Check if already known
        existing = ref_db.indicators.get(word, [])
        already_has = any(t == wtype for t, _, _ in existing)
        if already_has:
            continue

        if word not in ref_db.indicators:
            ref_db.indicators[word] = []
        else:
            ref_db.indicators[word] = list(ref_db.indicators[word])
        ref_db.indicators[word].append((wtype, None, 0.5))
        injected.append("IND: %s -> %s" % (word, wtype))

    return injected


# ============================================================
# Catalog enrichment — new signature patterns from P's solves
# ============================================================

# Map P's piece mechanism to S's token type
_MECHANISM_TO_TOKEN = {
    "synonym": SYN_F,
    "abbreviation": ABR_F,
    "anagram_fodder": ANA_F,
    "first_letter": POS_F,
    "last_letter": POS_F,
    "alternate_letters": POS_F,
    "core_letters": POS_F,
    "hidden": HID_F,
    "sound_of": HOM_F,
    "literal": RAW,
    "reversal": RAW,       # word used as-is then reversed — REV_I handles the operation
    "deletion": RAW,       # literal word with letters removed — DEL_I handles the operation
}

# Map P's piece mechanism to required indicator (if any)
_MECHANISM_TO_INDICATOR = {
    "first_letter": POS_I_FIRST,
    "last_letter": POS_I_LAST,
    "alternate_letters": POS_I_ALTERNATE,
    "hidden": HID_I,
    "sound_of": HOM_I,
    "reversal": REV_I,
    "deletion": DEL_I,
}

# Map P's wordplay_type to S operation and required indicators
_WTYPE_TO_OPERATION = {
    "charade": ("charade", set()),
    "container": ("container", {CON_I}),
    "reversal": ("reversal_charade", {REV_I}),
    "anagram": ("anagram", {ANA_I}),
    "deletion": ("deletion", {DEL_I}),
    "hidden": ("hidden", {HID_I}),
    "homophone": ("homophone", {HOM_I}),
    "acrostic": ("acrostic", {POS_I_FIRST}),
}

# Types we cannot create signatures for
_SKIP_TYPES = {"double_definition", "cryptic_definition", "spoonerism",
               "substitution", "unknown", "", None}


def collect_signatures_from_results(results):
    """Extract catalog-ready signature patterns from P's successful solves.

    Returns list of dicts:
      {tokens: tuple, word_spans: tuple, operation: str, indicators: set}
    """
    signatures = []
    seen = set()

    for r in results:
        if r.get("status") != "ASSEMBLED":
            continue
        # Skip clues already solved by S
        tier = r.get("tier") or ""
        if tier.startswith("Signature"):
            continue

        ai = r.get("ai_output") or {}
        wtype = ai.get("wordplay_type", "")
        pieces = ai.get("pieces", [])

        if wtype in _SKIP_TYPES or not pieces:
            continue
        if wtype not in _WTYPE_TO_OPERATION:
            continue

        # Map each piece to a token + word_span
        tokens = []
        word_spans = []
        indicators = set()
        skip = False

        for p in pieces:
            mech = p.get("mechanism", "")
            clue_word = (p.get("clue_word") or "").strip()
            letters = (p.get("letters") or "").strip()

            if not clue_word or not letters:
                skip = True
                break

            token = _MECHANISM_TO_TOKEN.get(mech)
            if token is None:
                skip = True
                break

            tokens.append(token)
            word_spans.append(len(clue_word.split()))

            # Collect per-piece indicators (e.g. first_letter needs POS_I_FIRST)
            piece_ind = _MECHANISM_TO_INDICATOR.get(mech)
            if piece_ind:
                indicators.add(piece_ind)

        if skip or not tokens:
            continue

        # Get operation and type-level indicators
        operation, type_indicators = _WTYPE_TO_OPERATION[wtype]
        indicators |= type_indicators

        # Refine operation based on actual token/piece composition
        has_ana = ANA_F in tokens
        has_non_ana = any(t in (SYN_F, ABR_F, POS_F, RAW) for t in tokens)
        has_reversal_piece = any(p.get("mechanism") == "reversal" for p in pieces)

        if wtype == "anagram" and has_ana and has_non_ana:
            operation = "anagram_charade"
        elif wtype == "reversal" and len(tokens) == 1:
            operation = "reversal"
        elif wtype == "charade" and has_reversal_piece:
            operation = "reversal_charade"
            indicators.add(REV_I)

        tokens_tuple = tuple(tokens)
        spans_tuple = tuple(word_spans)

        # Dedup
        key = (tokens_tuple, spans_tuple, operation)
        if key in seen:
            continue
        seen.add(key)

        signatures.append({
            "tokens": tokens_tuple,
            "word_spans": spans_tuple,
            "operation": operation,
            "indicators": indicators,
        })

    return signatures


def enrich_catalog(existing_catalog, new_signatures):
    """Create CatalogEntry objects from P's signature patterns.

    Returns (extra_entries, count_added).
    extra_entries is a NEW list (does not modify existing_catalog).
    """
    # Build set of existing keys for dedup
    existing_keys = set()
    for entry in existing_catalog:
        existing_keys.add((entry.tokens, entry.word_spans, entry.operation))

    extra = []
    for sig in new_signatures:
        key = (sig["tokens"], sig["word_spans"], sig["operation"])
        if key in existing_keys:
            continue

        label_parts = ["%s(%dw)" % (t, s)
                       for t, s in zip(sig["tokens"], sig["word_spans"])]
        entry = CatalogEntry(
            tokens=sig["tokens"],
            operation=sig["operation"],
            tier=4,  # lowest priority — enriched/speculative
            indicators=sig["indicators"],
            word_spans=sig["word_spans"],
            label="P:%s" % " · ".join(label_parts),
        )
        extra.append(entry)
        existing_keys.add(key)

    return extra, len(extra)
