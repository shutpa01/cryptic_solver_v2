"""Haiku vs Sonnet comparison with extended thinking.

Standalone script — no pipeline, no DB writes, no puzzle state.
Tests 10 clues from Times 29493 on both Haiku and Sonnet with thinking,
comparing quality, speed, and cost.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-20250514"

SIMPLE_PROMPT = """You are an expert cryptic crossword solver. Given a clue and its answer, explain how the wordplay works.

Be precise about letter-level mechanics. Show how the letters combine to spell the answer.

Keep your explanation concise."""

# 10 clues from Times 29493
TEST_CLUES = [
    ("Covering page in proofreading mark", "CARPET", 6),
    ("Start to tuck into sandwich and roly-poly", "ROTUND", 6),
    ("Spots litter leader's left", "RASH", 4),
    ("Lake bed holds solidified matter", "CLOT", 4),
    ("One miracle involved splendid ritual", "CEREMONIAL", 10),
    ("Have high opinion of endless desert", "RAT", 3),
    ("Be confronted by heartless travesty", "FACE", 4),
    ("Almost terrifying disfigurement", "SCAR", 4),
    ("Tremble when delivering note", "QUAVER", 6),
    ("Time without tea for so many? (A number)", "TWO", 3),
]


def test_with_thinking(model, clue, answer, enum):
    """Run a single clue with extended thinking enabled."""
    user_msg = "Clue: %s (%d)\nAnswer: %s" % (clue, enum, answer)

    start = time.time()
    with client.messages.stream(
        model=model,
        max_tokens=16000,
        temperature=1,  # required for extended thinking
        thinking={
            "type": "enabled",
            "budget_tokens": 10000,
        },
        messages=[{"role": "user", "content": user_msg}],
        system=SIMPLE_PROMPT,
    ) as stream:
        response = stream.get_final_message()
    elapsed = time.time() - start

    thinking_text = ""
    answer_text = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            answer_text = block.text.strip()

    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens
    return thinking_text, answer_text, tokens_in, tokens_out, elapsed


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    for model, model_name in [(SONNET, "Sonnet")]:
        print("\n" + "=" * 80)
        print("%s WITH EXTENDED THINKING" % model_name.upper())
        print("=" * 80)

        total_time = 0
        total_in = 0
        total_out = 0

        for clue, answer, enum in TEST_CLUES:
            print("\n--- %s (%d) = %s ---" % (clue[:60], enum, answer))
            try:
                thinking, response, tok_in, tok_out, elapsed = test_with_thinking(
                    model, clue, answer, enum
                )
                total_time += elapsed
                total_in += tok_in
                total_out += tok_out

                print("  Time: %.1fs | Thinking: %d chars | Tokens: %d in, %d out" % (
                    elapsed, len(thinking), tok_in, tok_out))
                print("  Response: %s" % response.replace("\n", " | ")[:300])
            except Exception as e:
                print("  ERROR: %s" % e)
            time.sleep(0.5)

        print("\n--- %s TOTALS ---" % model_name.upper())
        print("  Total time: %.1fs (avg %.1fs/clue)" % (total_time, total_time / 10))
        print("  Total tokens: %d in, %d out" % (total_in, total_out))


if __name__ == "__main__":
    main()
