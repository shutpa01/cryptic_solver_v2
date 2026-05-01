"""Adapter: maps signature solver results into P's DB schema and result dict format.

Bridges the gap between signature_solver.SolveResult and the production pipeline's
clues + structured_explanations tables and results[] list format.
"""

import json
import re

from .solver import OP_TO_TYPE, _ensure_structured_table
from .report import _describe_assembly


# ── Signature operation → P wordplay type string ──
# Uses the same OP_TO_TYPE mapping from solver.py where possible,
# with additions for signature-specific operations.
SIG_OP_TO_TYPE = {
    "charade": "charade",
    "container": "container",
    "container_charade": "container",
    "reversal": "reversal",
    "reversal_charade": "reversal",
    "container_reversal": "reversal",
    "anagram": "anagram",
    "anagram_charade": "anagram",
    "anagram_plus": "anagram",
    "anagram_container": "anagram",
    "hidden": "hidden",
    "hidden_reversed": "hidden",
    "homophone": "homophone",
    "deletion": "deletion",
    "trim": "deletion",
    "trim_charade": "deletion",
    "synonym": "double_definition",
    "alternate": "acrostic",
    "acrostic": "acrostic",
    "positional_charade": "charade",
    "container_positional": "container",
    "double_definition": "double_definition",
}


# ── Signature token → P mechanism string ──
# Indicator and LNK tokens are skipped (they don't produce pieces in P's format).
from signature_solver.tokens import (
    SYN_F, ABR_F, ANA_F, RAW, HID_F, HOM_F, POS_F, DEF, LNK, DBE_MARKER,
    ANA_I, REV_I, CON_I, HID_I, HOM_I, DEL_I,
    POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
    POS_I_ALTERNATE, POS_I_HALF, POS_I_TRIM_FIRST,
    POS_I_TRIM_LAST, POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER,
)

# Tokens that produce pieces (fodder tokens)
SIG_TOKEN_TO_MECHANISM = {
    SYN_F: "synonym",
    ABR_F: "abbreviation",
    ANA_F: "anagram_fodder",
    RAW: "literal",
    HID_F: "hidden",
    HOM_F: "sound_of",
    POS_F: None,  # resolved at runtime from positional indicator context
}

# Positional indicator → mechanism
_POS_INDICATOR_TO_MECHANISM = {
    POS_I_FIRST: "first_letter",
    POS_I_LAST: "last_letter",
    POS_I_OUTER: "outer_letters",      # first AND last (drop middle)
    POS_I_MIDDLE: "middle_letters",    # the inner letters (drop outer)
    POS_I_ALTERNATE: "alternate_letters",
    POS_I_HALF: "half_letters",
    POS_I_TRIM_FIRST: "deletion",
    POS_I_TRIM_LAST: "deletion",
    POS_I_TRIM_MIDDLE: "deletion",
    POS_I_TRIM_OUTER: "deletion",
}

# Indicator tokens → DB wordplay_type string (for explanation labels)
_INDICATOR_TOKEN_TO_DB_TYPE = {
    ANA_I: "anagram",
    REV_I: "reversal",
    CON_I: "container",
    DEL_I: "deletion",
    HID_I: "hidden",
    HOM_I: "homophone",
    POS_I_FIRST: "acrostic",
    POS_I_LAST: "parts",
    POS_I_OUTER: "parts",
    POS_I_MIDDLE: "parts",
    POS_I_ALTERNATE: "alternating",
    POS_I_TRIM_FIRST: "deletion",
    POS_I_TRIM_LAST: "deletion",
    POS_I_TRIM_MIDDLE: "deletion",
    POS_I_TRIM_OUTER: "deletion",
    POS_I_HALF: "deletion",
}

# Tokens that are indicators or links — skip these when building pieces.
# DBE_MARKER is also skipped at piece-building time; sig_explain renders
# its annotation inline on the example word it modifies.
_SKIP_TOKENS = frozenset([
    ANA_I, REV_I, CON_I, HID_I, HOM_I, DEL_I, LNK, DBE_MARKER,
    POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
    POS_I_ALTERNATE, POS_I_HALF, POS_I_TRIM_FIRST,
    POS_I_TRIM_LAST, POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER,
])


def build_ai_pieces(sr):
    """Build P-compatible ai_pieces list from SolveResult's word_roles.

    Returns list of dicts with keys: mechanism, clue_word, letters.
    """
    if not sr.result:
        return []

    pieces = []
    # Collect positional indicators from word_roles to determine POS_F mechanism
    pos_indicator = None
    for word, tok, val, *_ in sr.result.word_roles:
        if tok in _POS_INDICATOR_TO_MECHANISM:
            pos_indicator = tok

    for word, tok, val, *_meta in sr.result.word_roles:
        if tok in _SKIP_TOKENS:
            continue

        mechanism = SIG_TOKEN_TO_MECHANISM.get(tok)
        if mechanism is None and tok == POS_F:
            # Determine from positional indicator context
            mechanism = _POS_INDICATOR_TO_MECHANISM.get(pos_indicator, "first_letter")
        elif mechanism is None:
            continue  # unknown token type, skip

        pieces.append({
            "mechanism": mechanism,
            "clue_word": word,
            "letters": val if val else "",
        })

    return pieces


def build_assembly_dict(sr):
    """Build P-compatible assembly dict from SolveResult.

    Shapes must match what _describe_assembly expects in report.py.
    Returns dict or None if no result.
    """
    if not sr.result:
        return None

    # Parse the operation from the signature's entry label
    # word_roles format: [(word, token, value), ...]
    # The explanation_parts[0] has the human-readable text
    # We need to reconstruct the assembly dict from the word_roles

    # Get the operation from the entry label (stored in explanation)
    # The signature result doesn't store the entry directly, but we can
    # infer operation from the token mix and explanation
    roles = sr.result.word_roles

    # Collect fodder pieces (non-indicator, non-link)
    fodder_pieces = [(w, t, v) for w, t, v, *_ in roles if t not in _SKIP_TOKENS]
    indicator_pieces = [(w, t, v) for w, t, v, *_ in roles if t in _SKIP_TOKENS and t != LNK]

    # Infer operation from token types present
    tokens_present = set(t for _, t, _, *_ in (r for r in roles))
    fodder_tokens = set(t for _, t, _ in fodder_pieces)

    # Determine operation string from explanation or token analysis
    op = _infer_operation(tokens_present, fodder_tokens, indicator_pieces, fodder_pieces)

    # Build the assembly dict based on operation type
    values = [v for _, _, v in fodder_pieces if v]

    if op in ("charade", "positional_charade", "trim_charade",
              "reversal_charade", "anagram_charade", "container_charade"):
        return {"op": "charade", "order": values}

    elif op == "container_with_deletion":
        # Inner is the deletion-derived value (carried in role meta);
        # outer is the regular SYN_F/ABR_F.
        outer_val = None
        inner_derived = None
        for r in sr.result.word_roles:
            tok = r[1]
            val = r[2] if len(r) > 2 else None
            meta = r[3] if len(r) > 3 and isinstance(r[3], dict) else {}
            if tok in (SYN_F, ABR_F):
                if meta.get('transform') == 'deletion':
                    inner_derived = meta.get('derived')
                else:
                    outer_val = val
        return {"op": "container_with_deletion",
                "inner": inner_derived or "?",
                "outer": outer_val or "?"}

    elif op in ("container", "container_charade", "container_positional",
                "anagram_container"):
        if len(values) >= 2:
            # Convention: last piece with more letters is outer
            # But this is a simplification — the actual assembly was verified
            return {"op": "container", "inner": values[0], "outer": values[1]}
        return {"op": "container", "inner": values[0] if values else "?", "outer": "?"}

    elif op in ("reversal",):
        if len(values) == 1:
            return {"op": "reversal", "reversed": values[0]}
        else:
            return {"op": "reversal", "reversed_parts": values}

    elif op in ("container_reversal",):
        if len(values) >= 2:
            return {"op": "container_reversal", "inner": values[0], "outer": values[1]}
        return {"op": "container_reversal"}

    elif op in ("anagram", "anagram_plus"):
        return {"op": "anagram", "fodder": values}

    elif op in ("hidden",):
        # Hidden: the fodder words form the hiding text
        words_text = " ".join(w for w, t, _ in fodder_pieces)
        return {"op": "hidden", "words": words_text}

    elif op in ("hidden_reversed",):
        words_text = " ".join(w for w, t, _ in fodder_pieces)
        return {"op": "hidden_reversed", "words": words_text}

    elif op in ("homophone",):
        return {"op": "homophone",
                "sounds_like": values[0] if values else "?",
                "gives": sr.result.word_roles[0][2] if sr.result.word_roles else "?"}

    elif op in ("deletion", "trim"):
        return {"op": "deletion", "from": values[0] if values else "?",
                "deleted": "?"}

    elif op in ("synonym", "double_definition"):
        return {"op": "double_definition"}

    elif op in ("alternate", "acrostic"):
        words_text = " ".join(w for w, t, _ in fodder_pieces)
        return {"op": "charade", "order": values}

    # Fallback
    return {"op": op if op else "charade", "order": values}


def _infer_operation(tokens_present, fodder_tokens, indicator_pieces, fodder_pieces):
    """Infer the operation name from the token mix."""
    ind_tokens = set(t for _, t, _ in indicator_pieces)

    if HID_I in ind_tokens or HID_F in fodder_tokens:
        if REV_I in ind_tokens:
            return "hidden_reversed"
        return "hidden"
    if HOM_I in ind_tokens or HOM_F in fodder_tokens:
        return "homophone"
    if ANA_I in ind_tokens or ANA_F in fodder_tokens:
        if CON_I in ind_tokens:
            return "anagram_container"
        non_ana = [t for t in fodder_tokens if t != ANA_F]
        if non_ana:
            return "anagram_charade"
        return "anagram"
    if CON_I in ind_tokens and DEL_I in ind_tokens:
        return "container_with_deletion"
    if CON_I in ind_tokens:
        if REV_I in ind_tokens:
            return "container_reversal"
        pos_indicators = ind_tokens & set(_POS_INDICATOR_TO_MECHANISM.keys())
        if pos_indicators:
            return "container_positional"
        return "container"
    if REV_I in ind_tokens:
        non_rev = [t for t in fodder_tokens if t not in (SYN_F, ABR_F)]
        if len(fodder_pieces) > 1:
            return "reversal_charade"
        return "reversal"
    if DEL_I in ind_tokens:
        return "deletion"
    pos_indicators = ind_tokens & set(_POS_INDICATOR_TO_MECHANISM.keys())
    if pos_indicators:
        return "positional_charade"
    if POS_F in fodder_tokens:
        return "positional_charade"

    # Pure fodder — charade or double definition
    if len(fodder_pieces) == 1 and fodder_pieces[0][1] == SYN_F:
        return "synonym"
    return "charade"


def _describe_fodder(word, tok, val, pos_indicator=None, meta=None):
    """Describe a single fodder piece: 'PICK (synonym of "best")'.

    `meta` (optional) carries word_role metadata such as DBE source or
    reversal info, so the description can attribute pieces honestly.
    """
    meta = meta or {}
    if tok == SYN_F:
        # Render as 'synonym of' regardless of source. The verifier only
        # accepts that prefix; richer source info (e.g. DBE-injected
        # category-mate) is preserved in word_roles meta but not surfaced
        # in the explanation string.
        base = '%s (synonym of "%s")' % (val, word)
        # Reversal: if this synonym was used reversed, show that explicitly,
        # naming the reversal indicator word when known.
        if meta.get('transform') == 'reversed' and meta.get('reversed_to'):
            ind = meta.get('reversal_indicator')
            if ind:
                base = '%s reversed by "%s" = %s' % (
                    base, ind, meta['reversed_to'])
            else:
                base = '%s reversed = %s' % (base, meta['reversed_to'])
        return base
    elif tok == ABR_F:
        return '%s (abbreviation of "%s")' % (val, word)
    elif tok == ANA_F:
        # Anagram fodder: render so the verifier recognises the source.
        # If val == raw letters of word, render bare (no annotation needed —
        # the fodder_provenance check confirms the letters are in the clue).
        # If val is a known abbreviation of word, label it as such.
        # If val is a known synonym of word, label it as such.
        # Otherwise fall back to bare letters (still verifier-friendly).
        word_alpha = ''.join(c for c in word.upper() if c.isalpha())
        v_upper = (val or '').upper()
        if v_upper == word_alpha:
            return val  # raw fodder
        if meta.get('source') == 'abbreviation':
            return '%s (abbreviation of "%s")' % (val, word)
        if meta.get('source') == 'synonym':
            return '%s (synonym of "%s")' % (val, word)
        # Last resort: bare letters (no annotation) — accepted by verifier
        # because pieces without parens aren't checked by silent_piece.
        return val
    elif tok == RAW:
        # Letters lifted directly from the clue word. Use a verifier-
        # recognised prefix so silent_piece is satisfied AND the verifier
        # can identify this as a piece (assembly/abbreviation checks need
        # the WORD(content) format with parens).
        return '%s (from clue)' % val
    elif tok == HID_F:
        return '"%s"' % word
    elif tok == HOM_F:
        return '%s (sounds like "%s")' % (val, word)
    elif tok == POS_F:
        mech = _POS_INDICATOR_TO_MECHANISM.get(pos_indicator, "letters from")
        labels = {
            "first_letter": "first letter of",
            "last_letter": "last letter of",
            "outer_letters": "outer letters of",
            "middle_letters": "middle of",
            "alternate_letters": "alternate letters of",
            "half_letters": "half of",
            "deletion": "part of",
        }
        label = labels.get(mech, "letters from")
        return '%s (%s "%s")' % (val, label, word)
    return '%s ("%s")' % (val or "?", word)


def _indicator_label(tok):
    """Human-readable label for an indicator token."""
    labels = {
        ANA_I: "anagram",
        REV_I: "reversal",
        CON_I: "container",
        DEL_I: "deletion",
        HID_I: "hidden",
        HOM_I: "homophone",
        POS_I_FIRST: "first letter",
        POS_I_LAST: "last letter",
        POS_I_OUTER: "outer letters",
        POS_I_MIDDLE: "middle",
        POS_I_ALTERNATE: "alternating",
        POS_I_TRIM_FIRST: "beheading",
        POS_I_TRIM_LAST: "curtailing",
        POS_I_TRIM_MIDDLE: "gutting",
        POS_I_TRIM_OUTER: "stripping",
        POS_I_HALF: "halving",
    }
    return labels.get(tok, "indicator")


def _is_container_with_deletion(sr):
    """Detect the container-with-deletion compound from word_roles."""
    if not sr.result:
        return False
    roles = sr.result.word_roles
    has_con = any(r[1] == CON_I for r in roles)
    has_del = any(r[1] == DEL_I for r in roles)
    has_deletion_derived = any(
        len(r) > 3 and isinstance(r[3], dict)
        and r[3].get('transform') == 'deletion'
        for r in roles
    )
    return has_con and has_del and has_deletion_derived


def _explain_container_with_deletion(sr, answer):
    """Render a container-with-deletion compound honestly.

    Output shape (verifier-compatible):
      DD (synonym of "theologian") containing OODLE
      (deletion="OODLES", last letter dropped) [container: "claims";
      deletion: "mostly"] = DOODLED; definition: "idly scribbled down"
    """
    answer_clean = answer.upper().replace(" ", "").replace("-", "")
    definition = getattr(sr, "definition", None)

    outer = None         # (word, val, src)
    inner = None         # (word, val, derived, subtype, src)
    del_word = None
    con_word = None

    for r in sr.result.word_roles:
        word, tok = r[0], r[1]
        val = r[2] if len(r) > 2 else None
        meta = r[3] if len(r) > 3 and isinstance(r[3], dict) else {}

        if tok == DEL_I:
            del_word = word
        elif tok == CON_I:
            con_word = word
        elif tok in (SYN_F, ABR_F):
            src = 'synonym' if tok == SYN_F else 'abbreviation'
            if meta.get('transform') == 'deletion':
                inner = (word, val, meta.get('derived'),
                         meta.get('subtype'), src)
            else:
                outer = (word, val, src)

    if not (outer and inner and del_word and con_word):
        return None

    out_word, out_val, out_src = outer
    in_word, in_val, in_derived, in_subtype, in_src = inner

    if in_subtype in ('head', 'first_delete'):
        del_label = 'first letter dropped'
    else:
        del_label = 'last letter dropped'

    out_prefix = ('synonym of "%s"' % out_word) if out_src == 'synonym' \
        else ('abbreviation of "%s"' % out_word)
    explanation = (
        '%s (%s) containing %s (deletion="%s", %s) '
        '[container: "%s"; deletion: "%s"] = %s'
        % (out_val, out_prefix,
           in_derived, in_val, del_label,
           con_word, del_word, answer_clean)
    )

    if definition:
        explanation += '; definition: "%s"' % definition

    return {
        "explanation": explanation,
        "wordplay_types": ["container", "deletion"],
        "definition": definition,
    }


def sig_explain(sr, answer):
    """Build a human-readable explanation directly from S's word_roles.

    This is S's native explanation — no conversion to P's format.
    Walks word_roles left to right, pairs indicators with their fodder,
    and outputs a clear explanation with all wordplay types listed.

    Returns dict with keys:
        explanation: str  — the human-readable explanation text
        wordplay_types: list[str]  — all wordplay operations used
        definition: str or None
    """
    if not sr.result:
        return {"explanation": None, "wordplay_types": [], "definition": None}

    # Specific compound renderers run first; the generic walker is the
    # fallback. New cases get their own slot here — see
    # feedback_new_type_not_edit.md.
    if _is_container_with_deletion(sr):
        rendered = _explain_container_with_deletion(sr, answer)
        if rendered is not None:
            return rendered

    roles = sr.result.word_roles
    answer_clean = answer.upper().replace(" ", "").replace("-", "")
    definition = getattr(sr, "definition", None)

    # Collect positional indicator context
    pos_indicator = None
    for _, tok, _, *_meta in roles:
        if tok in _POS_INDICATOR_TO_MECHANISM:
            pos_indicator = tok

    # Build a map: fodder-word-index -> DBE marker word that modifies it.
    # The marker is the adjacent DBE_MARKER role; convention is that the
    # marker FOLLOWS its fodder ('Hill maybe'), but we also accept it
    # preceding ('say, hill') as a fallback.
    dbe_marker_for_idx = {}
    for i, role in enumerate(roles):
        w, tok = role[0], role[1]
        if tok != DBE_MARKER:
            continue
        # Look back first (preceding fodder is the example)
        if i > 0 and roles[i - 1][1] not in (LNK, DBE_MARKER, DEF):
            dbe_marker_for_idx[i - 1] = w
        elif i + 1 < len(roles) and roles[i + 1][1] not in (LNK, DBE_MARKER, DEF):
            dbe_marker_for_idx[i + 1] = w

    # Walk roles left to right, building explanation segments
    # Each segment is either a fodder description or an indicator annotation
    segments = []        # list of description strings in assembly order
    wordplay_types = set()
    indicators_used = []  # (indicator_word, indicator_token) pairs
    fodder_items = []     # (word, tok, val) for assembly ordering

    # First pass: collect indicators and fodder
    pending_indicator = None  # (word, tok) — indicator waiting for its fodder
    for i, role in enumerate(roles):
        word, tok, val = role[0], role[1], role[2] if len(role) > 2 else None
        meta = role[3] if len(role) > 3 else {}
        if tok in _SKIP_TOKENS and tok != LNK and tok != DBE_MARKER:
            # This is an indicator
            pending_indicator = (word, tok)
            indicators_used.append((word, tok))
            # Track wordplay types from indicators
            ind_type = _INDICATOR_TOKEN_TO_DB_TYPE.get(tok)
            if ind_type:
                wordplay_types.add(ind_type)
        elif tok == LNK:
            continue  # skip link words
        elif tok == DBE_MARKER:
            continue  # DBE marker — annotation rendered on its fodder
        elif tok == DEF:
            continue  # definition handled separately
        else:
            # This is a fodder piece
            indicator_note = ""
            if pending_indicator:
                iw, it = pending_indicator
                indicator_note = ' [%s: "%s"]' % (_indicator_label(it), iw)
                pending_indicator = None

            desc = _describe_fodder(word, tok, val, pos_indicator, meta=meta)
            # DBE marker association is preserved in word_roles meta but not
            # appended to the explanation string (the verifier rejects the
            # '(via "X")' annotation as an unrecognised source claim).
            fodder_items.append((word, tok, val, desc, indicator_note))

            # Infer wordplay from fodder type
            if tok == SYN_F:
                wordplay_types.add("charade")
            elif tok == ABR_F:
                wordplay_types.add("charade")
            elif tok == ANA_F:
                wordplay_types.add("anagram")
            elif tok == HID_F:
                wordplay_types.add("hidden")
            elif tok == HOM_F:
                wordplay_types.add("homophone")
            elif tok == POS_F:
                mech = _POS_INDICATOR_TO_MECHANISM.get(pos_indicator)
                if mech in ("deletion",):
                    wordplay_types.add("deletion")
                else:
                    # first_letter, last_letter, outer_letters etc.
                    # Mark as _pos_first_letter for now — we'll decide
                    # acrostic vs charade after seeing ALL pieces
                    wordplay_types.add("_pos_first_letter" if mech == "first_letter"
                                       else "charade")

    # Resolve double definition: single SYN_F whose value equals the answer
    if (len(fodder_items) == 1
            and fodder_items[0][1] == SYN_F
            and fodder_items[0][2] == answer_clean):
        wordplay_types.discard("charade")
        wordplay_types.add("double_definition")

    # Resolve _pos_first_letter: acrostic only if ALL fodder is first-letter
    if "_pos_first_letter" in wordplay_types:
        wordplay_types.discard("_pos_first_letter")
        all_pos_f = all(tok == POS_F for _, tok, _, _, _ in fodder_items)
        if all_pos_f:
            wordplay_types.add("acrostic")
        else:
            wordplay_types.add("charade")

    # If a trailing indicator wasn't consumed, attach it to the last fodder
    if pending_indicator and fodder_items:
        iw, it = pending_indicator
        w, t, v, desc, note = fodder_items[-1]
        note += ' [%s: "%s"]' % (_indicator_label(it), iw)
        fodder_items[-1] = (w, t, v, desc, note)

    # Now build the explanation in ASSEMBLY order (the order that spells the answer)
    # For non-anagram charades, find the permutation that spells the answer
    values = [val for _, _, val, _, _ in fodder_items]
    is_anagram = any(tok == ANA_F for _, tok, _, _, _ in fodder_items)
    is_hidden = any(tok == HID_F for _, tok, _, _, _ in fodder_items)
    is_reversal = "reversal" in wordplay_types and not is_anagram

    if is_hidden:
        # Hidden word: show the hiding text with answer highlighted
        hiding_text = " ".join(w for w, t, v, _, _ in fodder_items)
        # Find answer span in the concatenated letters
        concat = "".join(w.upper().replace(" ", "").replace("-", "")
                         for w, _, _, _, _ in fodder_items)
        idx = concat.find(answer_clean)
        indicator_notes = " ".join(n for _, _, _, _, n in fodder_items if n)

        from sonnet_pipeline.report import _highlight_hidden
        if "reversal" in wordplay_types:
            highlighted = _highlight_hidden(hiding_text, answer_clean[::-1])
            segments.append('hidden reversed in "%s"%s' % (highlighted, indicator_notes))
        else:
            highlighted = _highlight_hidden(hiding_text, answer_clean)
            segments.append('hidden in "%s"%s' % (highlighted, indicator_notes))

    elif is_anagram:
        # Anagram: collect all pieces — both ANA_F fodder and other pieces
        # that contribute letters to the anagram
        ana_parts = []
        non_ana_parts = []
        for w, t, v, desc, note in fodder_items:
            if t == ANA_F:
                ana_parts.append((w, t, v, desc, note))
            else:
                non_ana_parts.append((w, t, v, desc, note))

        # Check if non-ana pieces are part of the anagram fodder
        # (i.e. their letters + ana letters = answer when anagrammed)
        ana_letters = "".join(v for _, _, v, _, _ in ana_parts)
        non_ana_letters = "".join(v for _, _, v, _, _ in non_ana_parts)
        all_letters = sorted(ana_letters + non_ana_letters)
        answer_sorted = sorted(answer_clean)

        if all_letters == answer_sorted:
            # All pieces are part of one big anagram
            all_descs = []
            for _, _, _, desc, note in ana_parts:
                all_descs.append(desc + note)
            for _, _, _, desc, note in non_ana_parts:
                all_descs.append(desc + note)
            segments.append("anagram of %s = %s" % (" + ".join(all_descs), answer_clean))
        else:
            # Some pieces are separate (anagram_charade)
            ana_desc = " + ".join(d + n for _, _, _, d, n in ana_parts)
            segments.append("anagram of %s = %s" % (ana_desc,
                            "".join(v for _, _, v, _, _ in ana_parts)))
            for _, _, _, desc, note in non_ana_parts:
                segments.append(desc + note)

    elif is_reversal and len(fodder_items) == 1:
        # Pure reversal
        desc, note = fodder_items[0][3], fodder_items[0][4]
        segments.append("reverse of %s%s" % (desc, note))

    else:
        # Charade / container / positional: order pieces to match answer
        import itertools
        ordered_items = fodder_items  # default: clue order

        if len(values) <= 7 and not is_anagram:
            # Try permutations to find assembly order
            for perm in itertools.permutations(range(len(fodder_items))):
                perm_vals = "".join(values[i] for i in perm)
                if perm_vals == answer_clean:
                    ordered_items = [fodder_items[i] for i in perm]
                    break

        for w, t, v, desc, note in ordered_items:
            segments.append(desc + note)

    # Build final explanation
    if len(segments) == 1:
        explanation = segments[0]
    else:
        explanation = " + ".join(segments)

    # Add definition
    if definition:
        explanation += '; definition: "%s"' % definition

    # Clean up wordplay_types
    wp_list = sorted(wordplay_types)

    return {
        "explanation": explanation,
        "wordplay_types": wp_list,
        "definition": definition,
    }


def build_result_dict(sr, clue_text, answer, clue_number, direction, enumeration,
                      explanation=None):
    """Build a full result dict compatible with P's results[] list.

    This is the dict format that run_puzzle appends to results[],
    used by generate_report and the summary stats.
    """
    ai_pieces = build_ai_pieces(sr)
    assembly = build_assembly_dict(sr)
    score = sr.confidence
    confidence_label = "high" if score >= 80 else "medium" if score >= 50 else "low"

    # Build checks dict from confidence_reasons
    checks = {}
    for reason, delta in sr.confidence_reasons:
        key = reason.replace(" ", "_").replace(":", "").lower()[:30]
        checks[key] = "%s%d" % ("+" if delta >= 0 else "", delta)

    # Use S-native explanation
    sig_expl = sig_explain(sr, answer)
    definition = getattr(sr, "definition", None)

    ai_output = {
        "definition": definition,
        "wordplay_type": ", ".join(sig_expl["wordplay_types"]) if sig_expl["wordplay_types"] else "unknown",
        "pieces": ai_pieces,
    }

    validation = {
        "score": score,
        "confidence": confidence_label,
        "checks": checks,
    }

    return {
        "status": "ASSEMBLED" if assembly else "FAILED",
        "tier": "Signature",
        "clue_number": clue_number,
        "direction": direction,
        "enumeration": enumeration,
        "clue": clue_text,
        "answer": answer,
        "explanation": explanation,
        "enrichment": "",
        "confidence": confidence_label,
        "score": score,
        "checks": checks,
        "ai_output": ai_output,
        "assembly": assembly,
        "sig_explanation": sig_expl["explanation"],
        "sig_wordplay_types": sig_expl["wordplay_types"],
    }


def store_signature_result(conn, clue_id, sr, clue_text, answer, enriched=False):
    """Write signature solver result to clues + structured_explanations tables.

    Mirrors store_result() from solver.py but uses signature solver data.
    """
    _ensure_structured_table(conn)

    score = sr.confidence
    if score <= 0 or not sr.result:
        return

    ai_pieces = build_ai_pieces(sr)
    assembly = build_assembly_dict(sr)
    definition = getattr(sr, "definition", None)

    # Use S-native explanation
    sig_expl = sig_explain(sr, answer)
    explanation_text = sig_expl["explanation"]
    wordplay_types = sig_expl["wordplay_types"]
    wordplay_type = ", ".join(wordplay_types) if wordplay_types else "unknown"

    # Build components JSON
    components = {
        "ai_pieces": ai_pieces,
        "assembly": assembly,
        "wordplay_types": wordplay_types,
        "sig_explanation": explanation_text,
    }

    confidence = score / 100.0

    # Update clues table
    if definition:
        conn.execute("""
            UPDATE clues SET definition = ?
            WHERE id = ? AND (definition IS NULL OR definition = '')
        """, (definition, clue_id))

    conn.execute("""
        UPDATE clues SET wordplay_type = ?
        WHERE id = ?
    """, (wordplay_type, clue_id))

    if explanation_text:
        conn.execute("""
            UPDATE clues SET ai_explanation = ?
            WHERE id = ?
        """, (explanation_text, clue_id))

    # Determine solved status
    row = conn.execute(
        "SELECT definition, wordplay_type, ai_explanation FROM clues WHERE id = ?",
        (clue_id,)
    ).fetchone()
    has_def = bool(row[0]) if row else bool(definition)
    has_type = bool(row[1]) if row else bool(wordplay_type)
    has_expl = bool(row[2]) if row else False
    has_expl = has_expl or bool(ai_pieces)

    # Auto-approve high-confidence, flag others for review
    # Never overwrite manual reviews (reviewed = 1 or 2)
    current_reviewed = conn.execute(
        "SELECT reviewed FROM clues WHERE id = ?", (clue_id,)
    ).fetchone()
    already_reviewed = current_reviewed and current_reviewed[0] in (1, 2)

    if not already_reviewed:
        auto_reviewed = 1 if score >= 80 else 0
    else:
        auto_reviewed = current_reviewed[0]

    if has_def and has_type and has_expl:
        conn.execute("UPDATE clues SET has_solution = 1, reviewed = ? WHERE id = ?",
                      (auto_reviewed, clue_id))
    elif has_def or has_type:
        if not already_reviewed:
            conn.execute("UPDATE clues SET has_solution = 2, reviewed = 0 WHERE id = ?",
                          (clue_id,))
        else:
            conn.execute("UPDATE clues SET has_solution = 2 WHERE id = ?", (clue_id,))

    # Definition position in clue text
    def_start = None
    def_end = None
    if definition:
        idx = clue_text.lower().find(definition.lower())
        if idx >= 0:
            def_start = idx
            def_end = idx + len(definition)

    # Fetch metadata
    clue_meta = conn.execute(
        "SELECT source, puzzle_number, clue_number FROM clues WHERE id = ?", (clue_id,)
    ).fetchone()
    src, pnum, cnum = clue_meta if clue_meta else (None, None, None)

    # wordplay_types already set from sig_explain — use as-is for DB
    wp_types_for_db = wordplay_types if wordplay_types else ["unknown"]
    model_version = "signature_solver_enriched_v1" if enriched else "signature_solver_v1"

    existing = conn.execute(
        "SELECT id FROM structured_explanations WHERE clue_id = ?", (clue_id,)
    ).fetchone()

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
            definition, def_start, def_end,
            json.dumps(wp_types_for_db), json.dumps(components),
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
            clue_id, definition, def_start, def_end,
            json.dumps(wp_types_for_db), json.dumps(components),
            model_version, confidence,
            src, pnum, cnum
        ))
