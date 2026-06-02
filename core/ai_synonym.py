"""AI synonym/definition check — the narrow fallback for the DD engine.

Used only when one half of a candidate double definition is confirmed by the DB
but the other half is not. Asks Haiku a single yes/no question: does this phrase
define the known answer? It never discovers the answer and never reasons about
structure — one phrase, one answer, YES or NO. The narrow, operation-specific
call pattern, same as ai_definition.

is_definition_of(phrase, answer) -> True | False | None
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
    "In a cryptic crossword, a DOUBLE DEFINITION clue is two separate "
    "definitions of the same answer placed side by side.\n"
    "Answer strictly YES or NO: could the phrase \"%s\" be a definition or "
    "synonym of the answer \"%s\" (i.e. could it clue that answer)? Consider "
    "all common senses of the words. Reply with ONLY YES or NO."
)


def is_definition_of(phrase, answer):
    """Ask Haiku whether `phrase` could define `answer`.

    Returns True (a definite yes), False (a definite no), or None when the check
    could not be made — empty input, any API/network error, or an unparseable
    reply. The None case is load-bearing: a swallowed error is "unknown", not
    "no", and the DD engine must never record a confident FAIL on a verdict it
    never actually obtained. Only a real True/False is a considered verdict."""
    phrase = (phrase or "").strip()
    if not phrase or not answer:
        return None
    try:
        resp = _client().messages.create(
            model=HAIKU_MODEL,
            max_tokens=4,
            temperature=0,
            messages=[{"role": "user", "content": _PROMPT % (phrase, answer)}],
        )
        text = resp.content[0].text.strip().upper()
    except Exception:
        return None
    if text.startswith("YES"):
        return True
    if text.startswith("NO"):
        return False
    return None
