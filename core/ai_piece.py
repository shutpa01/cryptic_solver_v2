"""AI wordplay-piece assist — Haiku suggests a missing charade/wordplay piece.

The companion to ai_definition: when the reference DB cannot supply a wordplay
piece a clue needs (a synonym/abbreviation a clue word stands for), this asks
Haiku for it. It NEVER discovers the answer and never reasons about whole-clue
structure — it is given the known answer and ONE clue word/phrase, and asked what
letters of the answer that phrase produces. The result is used PROVISIONALLY: the
engine marks the piece source='pending', the parse becomes pending, and the
piece is queued to the normal enrichment dashboard for human Accept/Reject.

Haiku only (claude-haiku-4-5). Sonnet is never used (expensive + unreliable).

suggest_piece(phrase, answer) -> the letters (str, uppercase) or None.
"""

import os

from dotenv import load_dotenv
load_dotenv()

HAIKU_MODEL = "claude-haiku-4-5-20251001"

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from anthropic import Anthropic
        _CLIENT = Anthropic()
    return _CLIENT


_PROMPT = (
    "In a cryptic crossword the ANSWER is built from wordplay pieces. You are "
    "given the answer and ONE word or phrase from the clue. Say what letters of "
    "the answer that word/phrase stands for, as a synonym or abbreviation.\n"
    "- The letters MUST be a contiguous run that appears in the answer.\n"
    "- Reply with ONLY those letters in uppercase, nothing else.\n"
    "- If the word/phrase does not plausibly stand for any run of the answer, "
    "reply exactly NONE.\n\n"
    "Answer: %s\nWord/phrase: \"%s\""
)


def suggest_piece(phrase, answer):
    """Ask Haiku what letters of `answer` the clue `phrase` produces. Returns the
    uppercase letters, or None (no plausible piece / any failure / not a run of
    the answer). The substring guard keeps a hallucinated value out: a suggestion
    that is not actually in the answer is discarded."""
    phrase = (phrase or "").strip()
    answer = (answer or "").strip().upper()
    if not phrase or not answer:
        return None
    try:
        resp = _client().messages.create(
            model=HAIKU_MODEL,
            max_tokens=12,
            temperature=0,
            messages=[{"role": "user", "content": _PROMPT % (answer, phrase)}],
        )
        text = resp.content[0].text.strip().upper()
    except Exception:
        return None
    value = "".join(ch for ch in text if ch.isalpha())
    if not value or value == "NONE":
        return None
    if value not in answer:                  # guard against a hallucinated piece
        return None
    return value


_PRODUCE_PROMPT = (
    "In a cryptic crossword's wordplay a word or phrase stands for some letters via a "
    "synonym or a standard abbreviation. Answer strictly YES or NO: can \"%s\" stand "
    "for the letters \"%s\" (as a synonym or abbreviation)? Consider all common senses. "
    "Reply with ONLY YES or NO."
)


def could_produce(phrase, value):
    """YES/NO: can `phrase` stand for the KNOWN letters `value` (synonym/abbreviation)?

    Used where the target is already fixed — e.g. a container's value component, whose
    letters are determined by the insertion, so we cannot ask 'what letters?' (the outer
    is split around the inner and is not even a substring of the answer). Returns True
    only on a definite YES; any failure / NO / unparseable reply is False."""
    phrase = (phrase or "").strip()
    value = (value or "").strip().upper()
    if not phrase or not value:
        return False
    try:
        resp = _client().messages.create(
            model=HAIKU_MODEL,
            max_tokens=4,
            temperature=0,
            messages=[{"role": "user",
                       "content": _PRODUCE_PROMPT % (phrase, value)}],
        )
        text = resp.content[0].text.strip().upper()
    except Exception:
        return False
    return text.startswith("YES")


_HOM_PROMPT = (
    "In a cryptic crossword's wordplay, some answer letters are a HOMOPHONE: they "
    "sound like a different word, and that word is a synonym of a clue word. The answer "
    "letters \"%s\" sound like a word that means \"%s\". Give that single word.\n"
    "- It must SOUND like \"%s\" and be able to mean \"%s\".\n"
    "- Reply with ONLY the word in uppercase, or exactly NONE if there is none."
)


def suggest_homophone_source(phrase, span):
    """Homophone-aware piece assist. Given a clue word/phrase and the real-word answer
    letters `span` it must sound like, ask Haiku for the SOUND SOURCE — the word that
    sounds like `span` AND can mean `phrase` (e.g. phrase='went', span='ROAD' -> RODE).

    The companion to suggest_piece for the HOM_F slot, whose missing DB fact is the
    synonym phrase->source (the homophone source->span is dictionary-confirmed by the
    caller, never taken on the model's word). Returns the uppercase word, or None. The
    word is NOT required to be a substring of the answer (it is a different spelling —
    that is the whole point of a homophone)."""
    phrase = (phrase or "").strip()
    span = (span or "").strip().upper()
    if not phrase or not span:
        return None
    try:
        resp = _client().messages.create(
            model=HAIKU_MODEL,
            max_tokens=12,
            temperature=0,
            messages=[{"role": "user",
                       "content": _HOM_PROMPT % (span, phrase, span, phrase)}],
        )
        text = resp.content[0].text.strip().upper()
    except Exception:
        return None
    word = "".join(ch for ch in text if ch.isalpha())
    if not word or word == "NONE":
        return None
    return word
