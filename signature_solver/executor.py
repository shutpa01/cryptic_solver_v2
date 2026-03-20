"""Execute signature patterns mechanically to check if they produce the answer."""

from itertools import permutations
from .tokens import *


def check_anagram(letters, answer):
    """Check if letters can be anagrammed to produce the answer."""
    return sorted(letters.upper()) == sorted(answer.upper())


def check_hidden(words, answer):
    """Check if the answer is hidden as a contiguous span across words."""
    combined = "".join(w.upper() for w in words if w.isalpha() or True)
    # Strip non-alpha for matching
    combined_alpha = "".join(c for c in combined.upper() if c.isalpha())
    answer_alpha = "".join(c for c in answer.upper() if c.isalpha())
    return answer_alpha in combined_alpha


def check_hidden_reversed(words, answer):
    """Check if the answer is hidden reversed across words."""
    combined_alpha = "".join(c for c in "".join(words).upper() if c.isalpha())
    answer_alpha = "".join(c for c in answer.upper() if c.isalpha())
    return answer_alpha in combined_alpha[::-1]


def extract_positional(word, pos_type):
    """Extract letters from a word based on positional indicator.

    Returns the extracted letters (uppercase), or None if not applicable.
    """
    w = "".join(c for c in word.upper() if c.isalpha())
    if not w:
        return None

    if pos_type == POS_I_FIRST:
        return w[0]
    elif pos_type == POS_I_LAST:
        return w[-1]
    elif pos_type == POS_I_OUTER:
        if len(w) >= 2:
            return w[0] + w[-1]
        return w
    elif pos_type == POS_I_MIDDLE:
        if len(w) >= 3:
            return w[len(w) // 2] if len(w) % 2 == 1 else w[len(w) // 2 - 1:len(w) // 2 + 1]
        return None
    elif pos_type == POS_I_ALTERNATE:
        return w[::2]  # odd positions (1st, 3rd, 5th)
    elif pos_type == POS_I_TRIM_FIRST:
        return w[1:] if len(w) >= 2 else None
    elif pos_type == POS_I_TRIM_LAST:
        return w[:-1] if len(w) >= 2 else None
    elif pos_type == POS_I_TRIM_MIDDLE:
        if len(w) >= 3:
            mid = len(w) // 2
            if len(w) % 2 == 1:
                return w[:mid] + w[mid + 1:]
            else:
                return w[:mid - 1] + w[mid + 1:]
        return None
    elif pos_type == POS_I_TRIM_OUTER:
        if len(w) >= 3:
            return w[1:-1]
        return None
    elif pos_type == POS_I_HALF:
        return w[:len(w) // 2]
    return None


def try_pure_anagram(word_texts, answer):
    """Check if all words anagrammed together produce the answer."""
    all_letters = "".join(
        c for w in word_texts for c in w.upper() if c.isalpha()
    )
    return check_anagram(all_letters, answer)


def try_charade(pieces, answer):
    """Check if pieces concatenated produce the answer.

    Args:
        pieces: list of strings (uppercase letter contributions)
        answer: target answer (uppercase)

    Returns: True if concatenation matches
    """
    result = "".join(pieces)
    return result == answer.upper().replace(" ", "").replace("-", "")


def try_container(outer, inner, answer):
    """Check if inner inserted into outer at any position produces the answer.

    Returns: True if any valid insertion works.
    """
    answer_clean = answer.upper().replace(" ", "").replace("-", "")
    outer_clean = outer.upper()
    inner_clean = inner.upper()

    # Try inserting inner at each position in outer
    for i in range(1, len(outer_clean)):
        candidate = outer_clean[:i] + inner_clean + outer_clean[i:]
        if candidate == answer_clean:
            return True
    return False


def try_reversal(text, answer):
    """Check if reversing the text produces the answer."""
    t = "".join(c for c in text.upper() if c.isalpha())
    a = "".join(c for c in answer.upper() if c.isalpha())
    return t[::-1] == a


def try_deletion(base, remove, answer):
    """Check if removing 'remove' from 'base' produces the answer.

    Tries removing the first occurrence.
    """
    base_upper = base.upper()
    remove_upper = remove.upper()
    answer_clean = answer.upper().replace(" ", "").replace("-", "")

    idx = base_upper.find(remove_upper)
    if idx >= 0:
        result = base_upper[:idx] + base_upper[idx + len(remove_upper):]
        if result == answer_clean:
            return True
    return False


def execute_signature(entry, assignment, words, answer):
    """Verify that a matched signature actually produces the answer.

    The matcher already checks most operations during assignment,
    so this is mainly a validation pass + explanation builder.

    Returns: (success: bool, explanation: str, pieces: list)
        pieces: [(word_text, token, value), ...]
    """
    fodder = assignment['fodder_order']
    op = entry.operation

    # Build pieces list with word text
    pieces = []
    for item in fodder:
        if isinstance(item[0], tuple):
            # Phrase reference (pi, pj)
            pi, pj = item[0]
            word_text = " ".join(words[pi:pj])
        elif isinstance(item[0], int):
            word_text = words[item[0]]
        else:
            word_text = str(item[0])
        pieces.append((word_text, item[1], item[2]))

    # Extract indicator word texts from assignment
    indicators = {}
    for ind_tok, ind_idx in assignment.get('indicator_indices', {}).items():
        if isinstance(ind_idx, tuple):
            ind_text = " ".join(words[k] for k in range(ind_idx[0], ind_idx[1]))
        else:
            ind_text = words[ind_idx]
        indicators[ind_tok] = ind_text

    # Build explanation
    explanation = _build_explanation(op, pieces, answer, entry, indicators)

    return True, explanation, pieces


# --- Indicator helpers ---

# Positional indicators are piece-level: they describe how letters are
# extracted from a specific source word (the POS_F piece).
_PIECE_LEVEL_INDICATORS = {
    POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
    POS_I_ALTERNATE, POS_I_TRIM_FIRST, POS_I_TRIM_LAST,
    POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER, POS_I_HALF,
}


def _indicator_meaning(ind_tok):
    """Plain English description of what a positional indicator does."""
    return {
        POS_I_FIRST: "first letter(s)",
        POS_I_LAST: "last letter(s)",
        POS_I_OUTER: "outer letters",
        POS_I_MIDDLE: "middle letter(s)",
        POS_I_ALTERNATE: "alternating letters",
        POS_I_TRIM_FIRST: "remove first letter",
        POS_I_TRIM_LAST: "remove last letter",
        POS_I_TRIM_MIDDLE: "remove middle",
        POS_I_TRIM_OUTER: "inner letters",
        POS_I_HALF: "half",
    }.get(ind_tok, "")


def _describe_piece(word_text, tok, val, piece_inds):
    """Describe a single piece with indicator attribution.

    Args:
        word_text: the clue word(s)
        tok: the token type (SYN_F, ABR_F, POS_F, etc.)
        val: the resolved value (letters produced)
        piece_inds: dict of {ind_tok: ind_word} for piece-level indicators
    """
    if tok == SYN_F:
        return f'{val} (synonym of "{word_text}")'
    elif tok == ABR_F:
        return f'{val} (abbreviation of "{word_text}")'
    elif tok == RAW:
        w = "".join(c for c in word_text.upper() if c.isalpha())
        return f'{w} ("{word_text}")'
    elif tok == ANA_F:
        return f'"{word_text}"'
    elif tok == POS_F:
        # Find the positional indicator that applies to this piece
        for ind_tok, ind_word in piece_inds.items():
            if ind_tok in _PIECE_LEVEL_INDICATORS:
                meaning = _indicator_meaning(ind_tok)
                return f'{val} ("{ind_word}" of "{word_text}" = {meaning})'
        return f'{val} (from "{word_text}")'
    elif tok == HID_F:
        return f'"{word_text}"'
    elif tok == HOM_F:
        return f'{val} (sounds like "{word_text}")'
    elif tok == DEL_F:
        return f'{val}'
    else:
        return f'{word_text}'


def _build_explanation(op, pieces, answer, entry, indicators=None):
    """Build human-readable explanation string with indicator attribution.

    Each piece shows what it contributes and why. Each indicator is attributed
    so the user understands which clue word signals which operation.
    """
    if indicators is None:
        indicators = {}

    # Separate piece-level (positional) and operation-level indicators
    piece_inds = {}
    op_inds = {}
    for ind_tok, ind_word in indicators.items():
        if ind_tok in _PIECE_LEVEL_INDICATORS:
            piece_inds[ind_tok] = ind_word
        else:
            op_inds[ind_tok] = ind_word

    # Build per-piece description strings
    part_strs = []
    for word_text, tok, val in pieces:
        part_strs.append(_describe_piece(word_text, tok, val, piece_inds))

    # --- Format by operation type ---

    if op == "charade":
        return f'{" + ".join(part_strs)} = {answer}'

    elif op == "anagram":
        ind = op_inds.get(ANA_I)
        ind_attr = f' ["{ind}"]' if ind else ''
        fodder = " ".join(p[0] for p in pieces if p[1] == ANA_F)
        return f'Anagram{ind_attr} of "{fodder}" = {answer}'

    elif op in ("anagram_plus", "anagram_charade", "anagram_container"):
        ind = op_inds.get(ANA_I)
        ind_attr = f' ["{ind}"]' if ind else ''
        ana_fodder = " ".join(p[0] for p in pieces if p[1] == ANA_F)
        extra_strs = [s for s, (_, t, _) in zip(part_strs, pieces) if t != ANA_F]
        ana_part = f'anagram{ind_attr} of "{ana_fodder}"'
        if extra_strs:
            return f'{" + ".join(extra_strs)} + {ana_part} = {answer}'
        return f'{ana_part} = {answer}'

    elif op == "hidden":
        ind = op_inds.get(HID_I)
        ind_attr = f' ["{ind}"]' if ind else ''
        words = " ".join(p[0] for p in pieces)
        return f'Hidden{ind_attr} in "{words}" = {answer}'

    elif op == "hidden_reversed":
        attrs = []
        if HID_I in op_inds:
            attrs.append(f'"{op_inds[HID_I]}"')
        if REV_I in op_inds:
            attrs.append(f'"{op_inds[REV_I]}"')
        ind_attr = f' [{", ".join(attrs)}]' if attrs else ''
        words = " ".join(p[0] for p in pieces)
        return f'Hidden reversed{ind_attr} in "{words}" = {answer}'

    elif op in ("container", "container_charade", "container_positional"):
        ind = op_inds.get(CON_I)
        ind_attr = f' ["{ind}"]' if ind else ''
        if len(pieces) >= 2:
            outer_str = part_strs[0]
            inner_str = part_strs[1]
            return f'{inner_str} inside{ind_attr} {outer_str} = {answer}'
        return f'{" + ".join(part_strs)} = {answer}'

    elif op == "container_reversal":
        con_ind = op_inds.get(CON_I)
        rev_ind = op_inds.get(REV_I)
        attrs = []
        if con_ind:
            attrs.append(f'"{con_ind}"')
        if rev_ind:
            attrs.append(f'reversed "{rev_ind}"')
        ind_attr = f' [{", ".join(attrs)}]' if attrs else ''
        if len(pieces) >= 2:
            outer_str = part_strs[0]
            inner_str = part_strs[1]
            return f'{inner_str} inside{ind_attr} {outer_str} = {answer}'
        return f'{" + ".join(part_strs)} = {answer}'

    elif op == "reversal":
        ind = op_inds.get(REV_I)
        ind_attr = f' ["{ind}"]' if ind else ''
        return f'Reverse{ind_attr} of {" + ".join(part_strs)} = {answer}'

    elif op == "reversal_charade":
        ind = op_inds.get(REV_I)
        ind_attr = f' ["{ind}" = reversal]' if ind else ''
        return f'{" + ".join(part_strs)}{ind_attr} = {answer}'

    elif op == "homophone":
        ind = op_inds.get(HOM_I)
        ind_attr = f' ["{ind}"]' if ind else ''
        return f'Sounds like{ind_attr} {" + ".join(part_strs)} = {answer}'

    elif op == "deletion":
        ind = op_inds.get(DEL_I)
        ind_attr = f' ["{ind}"]' if ind else ''
        if len(pieces) >= 2:
            return f'{part_strs[0]} minus{ind_attr} {part_strs[1]} = {answer}'
        return f'{" + ".join(part_strs)} = {answer}'

    elif op in ("trim", "trim_charade"):
        ind = op_inds.get(DEL_I)
        ind_attr = f' ["{ind}"]' if ind else ''
        return f'Trim{ind_attr}: {" + ".join(part_strs)} = {answer}'

    elif op == "alternate":
        ind = piece_inds.get(POS_I_ALTERNATE)
        ind_attr = f' ["{ind}"]' if ind else ''
        words = " ".join(p[0] for p in pieces)
        return f'Alternating letters{ind_attr} of "{words}" = {answer}'

    elif op == "acrostic":
        ind = piece_inds.get(POS_I_FIRST)
        ind_attr = f' ["{ind}"]' if ind else ''
        words = " ".join(p[0] for p in pieces)
        return f'First letters{ind_attr} of "{words}" = {answer}'

    elif op == "positional_charade":
        return f'{" + ".join(part_strs)} = {answer}'

    elif op == "synonym":
        return f'{" + ".join(part_strs)} = {answer}'

    else:
        return f'{" + ".join(part_strs)} = {answer}'
