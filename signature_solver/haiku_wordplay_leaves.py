"""Haiku-powered wordplay-leaf decomposition.

When the cascade fails to assemble a clue's wordplay, ask Haiku
(near-free) to decompose the wordplay window into pieces given the
known answer and the definition phrase. Each piece carries a clue
word or contiguous phrase, a role (synonym, abbreviation, anagram
indicator/fodder, container/deletion/reversal/hidden/homophone
indicator/source, link, literal), and a value (letters contributed,
or null for indicators/links).

Output feeds the leftover dashboard's diagnostic-candidate miner so
the human reviewer can accept the missing DB rows. Pieces with a
DB-actionable role (synonym, abbreviation, *_indicator) become
enrichment candidates; the rest are explanation only.
"""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
load_dotenv()


_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from anthropic import Anthropic
        _CLIENT = Anthropic()
    return _CLIENT


HAIKU_MODEL = "claude-haiku-4-5-20251001"


_VALID_ROLES = {
    "synonym", "abbreviation",
    "anagram_indicator", "anagram_fodder",
    "container_indicator", "containment_inner", "containment_outer",
    "deletion_indicator", "deletion_source",
    "reversal_indicator", "reversal_source",
    "hidden_indicator", "hidden_source",
    "homophone_indicator", "homophone_source",
    "acrostic_indicator", "acrostic_source",
    "link_word", "literal",
}


def find_wordplay_leaves(clue_text, answer, def_phrase, wp_words):
    """Decompose the wordplay window into role-tagged pieces.

    Args:
        clue_text: full clue text (for surface context).
        answer: known answer, uppercase, no spaces/hyphens.
        def_phrase: the identified definition phrase.
        wp_words: list of wordplay-window words (definition removed).

    Returns:
        List of piece dicts, or None if Haiku failed / output was
        malformed. Each dict has keys: word, role, value. Phrases
        spanning multiple clue words appear as one element with the
        phrase joined by spaces.
    """
    if not wp_words or not answer:
        return None

    answer_letters = "".join(c for c in answer.upper() if c.isalpha())
    prompt = (
        f"Clue: {clue_text}\n"
        f"Answer: {answer} ({len(answer_letters)} letters)\n"
        f"Definition (already identified): {def_phrase}\n"
        f"Wordplay window: {' '.join(wp_words)}\n\n"
        "Decompose the wordplay window. Allowed roles: synonym, "
        "abbreviation, anagram_indicator, anagram_fodder, "
        "container_indicator, containment_inner, containment_outer, "
        "deletion_indicator, deletion_source, reversal_indicator, "
        "reversal_source, hidden_indicator, hidden_source, "
        "homophone_indicator, homophone_source, acrostic_indicator, "
        "acrostic_source, link_word, literal.\n\n"
        "Rules:\n"
        "1. Every wordplay word appears exactly once. Combine "
        "consecutive words into one element if the role spans them.\n"
        "2. value = uppercase letters the piece contributes (source "
        "letters for anagram_fodder / reversal_source). For "
        "indicators and link_word use null.\n"
        "3. Concatenate all non-null values; the multiset of letters "
        f"must equal {sorted(answer_letters)} (anagram_fodder "
        "contributes its letters as a multiset; other pieces "
        "contribute their letters in order).\n"
        "4. If no decomposition adds up to the answer, reply [].\n\n"
        "Example: clue 'Good man helping' answer HANDYMAN def "
        "'helping' window 'Good man' ->\n"
        '[{"word":"Good","role":"synonym","value":"HANDY"},'
        '{"word":"man","role":"synonym","value":"MAN"}]\n\n'
        "Reply with ONLY a JSON array. No prose, no fences."
    )

    try:
        client = _get_client()
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=600,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
    except Exception:
        return None

    # Robustly extract the first balanced [...] JSON array from the
    # response. Tolerates code fences and trailing prose (Haiku
    # sometimes adds a "Wait, let me reconsider..." paragraph after
    # the array, which would break a naive json.loads of the whole
    # string).
    pieces = _extract_first_array(raw)
    if pieces is None:
        return None

    out = []
    for p in pieces:
        if not isinstance(p, dict):
            continue
        word = str(p.get("word") or "").strip()
        role = str(p.get("role") or "").strip().lower()
        value = p.get("value")
        if not word or role not in _VALID_ROLES:
            continue
        if isinstance(value, str):
            value = value.strip().upper() or None
        else:
            value = None
        out.append({"word": word, "role": role, "value": value})

    return out or None


def _extract_first_array(text):
    """Return the first balanced JSON array found in `text`, parsed.

    Walks the string character by character respecting JSON string
    delimiters and escapes, so square brackets inside string values
    don't throw the bracket counter off. Returns None if no balanced
    array parses cleanly.
    """
    if not text:
        return None
    start = text.find("[")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    return None
                return parsed if isinstance(parsed, list) else None
    return None
