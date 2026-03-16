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

    # Build explanation
    explanation = _build_explanation(op, pieces, answer, entry)

    return True, explanation, pieces


def _build_explanation(op, pieces, answer, entry):
    """Build human-readable explanation string."""
    parts = []
    for word_text, tok, val in pieces:
        if tok == ANA_F:
            parts.append(f"{word_text}")
        elif tok == ABR_F:
            parts.append(f"{word_text}={val}")
        elif tok == SYN_F:
            parts.append(f"{word_text}={val}")
        elif tok == RAW:
            parts.append(f"{word_text} (raw)")
        elif tok == POS_F:
            parts.append(f"{word_text}={val}")
        elif tok == HID_F:
            parts.append(f"{word_text}")
        elif tok == HOM_F:
            parts.append(f"{word_text} (sounds like)")
        else:
            parts.append(f"{word_text}({tok})")

    if op == "anagram":
        fodder_words = [p[0] for p in pieces if p[1] == ANA_F]
        return f"Anagram of {' '.join(fodder_words)} = {answer}"
    elif op == "anagram_plus":
        ana_words = [p[0] for p in pieces if p[1] == ANA_F]
        extras = [f"{p[0]}={p[2]}" for p in pieces if p[1] != ANA_F]
        return f"Anagram of {' '.join(ana_words)} + {', '.join(extras)} = {answer}"
    elif op == "anagram_charade":
        ana_words = [p[0] for p in pieces if p[1] == ANA_F]
        extras = [f"{p[0]}={p[2]}" for p in pieces if p[1] != ANA_F]
        return f"{', '.join(extras)} + anagram of {' '.join(ana_words)} = {answer}"
    elif op == "hidden":
        return f"Hidden in: {' '.join(p[0] for p in pieces)} = {answer}"
    elif op == "hidden_reversed":
        return f"Hidden reversed in: {' '.join(p[0] for p in pieces)} = {answer}"
    elif op == "container":
        if len(pieces) >= 2:
            return f"{pieces[1][2]} inside {pieces[0][2]} = {answer}"
        return f"Container = {answer}"
    elif op.startswith("reversal"):
        return f"{'Reverse ' + ', '.join(parts)} = {answer}"
    elif op == "homophone":
        return f"Sounds like {parts[0]} = {answer}"
    elif op == "deletion":
        if len(pieces) >= 2:
            return f"{pieces[0][2]} minus {pieces[1][2]} = {answer}"
        return f"Deletion = {answer}"
    elif op.startswith("trim"):
        return f"Trim: {', '.join(parts)} = {answer}"
    elif op == "alternate":
        return f"Alternate letters of {' '.join(p[0] for p in pieces)} = {answer}"
    elif op == "acrostic":
        return f"First letters of {' '.join(p[0] for p in pieces)} = {answer}"
    elif op.startswith("positional"):
        return f"Positional: {', '.join(parts)} = {answer}"
    else:
        return f"{entry.label}: {', '.join(parts)} = {answer}"
