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
        f"Definition (do NOT include in your output): {def_phrase}\n"
        f"Wordplay window: {' '.join(wp_words)}\n\n"
        "Assign each WORDPLAY word a role. ONLY the wordplay words "
        f"appear in your output: {' / '.join(wp_words)}. The "
        "definition word(s) above must NOT appear. Allowed roles: "
        "synonym, abbreviation, anagram_indicator, anagram_fodder, "
        "container_indicator, containment_inner, containment_outer, "
        "deletion_indicator, deletion_source, reversal_indicator, "
        "reversal_source, hidden_indicator, hidden_source, "
        "homophone_indicator, homophone_source, acrostic_indicator, "
        "acrostic_source, link_word, literal.\n\n"
        "Rules: every wordplay word appears exactly once. "
        "Indicators (anagram_indicator, reversal_indicator, "
        "container_indicator, deletion_indicator, hidden_indicator, "
        "homophone_indicator, acrostic_indicator) and link_words are "
        "ALWAYS single words. Only synonym / abbreviation / "
        "anagram_fodder / source roles may combine consecutive "
        "words into one element, and only when the phrase has a "
        "meaning together (e.g. 'some beer' meaning HALF). "
        "value = uppercase letters the piece contributes (source "
        "letters for anagram_fodder / reversal_source), null for "
        "indicators and link_word. The non-null values together "
        "(anagram_fodder as a letter multiset, others in order) "
        f"must produce the answer letters {sorted(answer_letters)}. "
        "If you cannot find a decomposition that produces the "
        "answer, reply [].\n\n"
        "Example A: clue 'Good man helping with simple tasks' "
        "answer HANDYMAN def 'helping with simple tasks' wordplay "
        "'Good man' ->\n"
        '[{"word":"Good","role":"synonym","value":"HANDY"},'
        '{"word":"man","role":"synonym","value":"MAN"}]\n\n'
        "Example B: clue 'Volley troubled by lovely shot' answer "
        "VOLLEY def 'shot' wordplay 'Volley troubled by lovely' ->\n"
        '[{"word":"Volley","role":"link_word","value":null},'
        '{"word":"troubled","role":"anagram_indicator","value":null},'
        '{"word":"by","role":"link_word","value":null},'
        '{"word":"lovely","role":"anagram_fodder","value":"LOVELY"}]\n\n'
        "Reply ONLY with a JSON array. No prose, no fences."
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

    # Build the set of valid word strings: every individual wp_word
    # plus every space-joined contiguous run. This lets Haiku label
    # multi-word phrases like "some beer" as one piece while still
    # rejecting any word it invents that isn't in the window.
    valid_phrases: set = set()
    cleaned_wp = [w.strip() for w in wp_words]
    for i in range(len(cleaned_wp)):
        for j in range(i + 1, len(cleaned_wp) + 1):
            valid_phrases.add(
                " ".join(cleaned_wp[i:j]).lower())

    out = []
    for p in pieces:
        if not isinstance(p, dict):
            continue
        word = str(p.get("word") or "").strip()
        role = str(p.get("role") or "").strip().lower()
        value = p.get("value")
        if not word or role not in _VALID_ROLES:
            continue
        # Drop pieces that name words outside the wordplay window —
        # Haiku occasionally hallucinates the definition word back in
        # or invents new tokens that violate the input.
        if word.lower() not in valid_phrases:
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
