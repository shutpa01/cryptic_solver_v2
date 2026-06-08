"""AI definition assist — the clearly-separated fallback for stage 3.

Called ONLY when the reference DB cannot define a clue. It is not part of the
pure core: core/definition.py takes it as an injected callable, so the stages
stay testable without an API. This is the one AI touch-point in the solving
path, deliberately isolated here.

Lesson baked in: the previous definition Haiku call truncated multi-word
definitions ("Stuff" instead of "Stuff from the dairy") because its prompt
asked for a "word or short phrase". This prompt demands the COMPLETE phrase.

define(clue_text, answer) -> the definition phrase (str) or None.
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
    "You are given a cryptic crossword clue and its ANSWER. Identify the "
    "DEFINITION: the run of words, at the START or the END of the clue, that "
    "defines the answer.\n"
    "- Return the COMPLETE definition phrase, exactly as it appears in the "
    "clue, including EVERY word that is part of it. It may be one word or "
    "several. For example, for \"Stuff from the dairy, ...\" the definition is "
    "\"Stuff from the dairy\", NOT just \"Stuff\". Do not shorten it.\n"
    "- It must sit at the very start or the very end of the clue, with no real "
    "words left outside it at that edge.\n"
    "- The definition may be given in PARENTHESES at the end of the clue, e.g. "
    "for \"A little antimacassar on Gainsborough (piece of cloth)\" the "
    "definition is \"piece of cloth\". Return the words inside the parentheses, "
    "without the brackets.\n"
    "- If the whole clue defines the answer with no separate wordplay (a "
    "cryptic definition), reply exactly NONE.\n"
    "CRITICAL: Output the definition phrase ALONE on a single line — nothing "
    "else. No explanation, no reasoning, no working, no 'Let me', no labels, no "
    "quotation marks. Just the phrase, or the single word NONE.\n\n"
    "Clue: %s\nAnswer: %s"
)


def define(clue_text, answer):
    """Ask Haiku for the full definition phrase. Returns the phrase or None
    (None also for a cryptic definition / any failure)."""
    try:
        resp = _client().messages.create(
            model=HAIKU_MODEL,
            max_tokens=64,
            temperature=0,
            messages=[{"role": "user",
                       "content": _PROMPT % (clue_text, answer)}],
        )
        text = resp.content[0].text.strip()
    except Exception:
        return None
    # Robustness: if the model rambled despite the instruction, keep only the
    # first non-empty line and strip any "Label: " preamble / quotes.
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if ":" in line and len(line.split(":", 1)[0]) <= 12:
        line = line.split(":", 1)[1].strip()
    line = line.strip().strip('"\'').strip()
    if not line or line.upper() == "NONE":
        return None
    return line
