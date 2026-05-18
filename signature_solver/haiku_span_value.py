"""Narrow Haiku verifier for answer-constrained span/value pairs."""
from __future__ import annotations

import json

from dotenv import load_dotenv

load_dotenv()


HAIKU_MODEL = "claude-haiku-4-5-20251001"
_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from anthropic import Anthropic
        _CLIENT = Anthropic()
    return _CLIENT


def verify_span_value(clue_text, span, value, answer, role_hint=None):
    """Return True if `span` can clue `value` in this cryptic context.

    This is intentionally a yes/no semantic check, not a whole-clue solver.
    The mechanical parse must already have derived `value` from the answer.
    """
    span = (span or "").strip()
    value = "".join(c for c in (value or "").upper() if c.isalpha())
    answer = "".join(c for c in (answer or "").upper() if c.isalpha())
    if not span or not value or not answer:
        return False

    hint = f"\nLikely role: {role_hint}." if role_hint else ""
    prompt = (
        f"Clue: {clue_text}\n"
        f"Known answer: {answer}\n"
        f"Candidate clue phrase: {span}\n"
        f"Candidate letters: {value}{hint}\n\n"
        "Question: in this cryptic crossword clue, can the candidate clue "
        "phrase fairly clue those candidate letters? Judge only this "
        "span/value pair; do not solve the whole clue.\n\n"
        'Reply with ONLY JSON: {"valid": true/false, "reason": "..."}'
    )

    try:
        response = _get_client().messages.create(
            model=HAIKU_MODEL,
            max_tokens=180,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < start:
            return False
        data = json.loads(raw[start:end + 1])
    except Exception:
        return False

    return bool(data.get("valid"))
