"""API test with extended thinking enabled.

Tests whether extended thinking changes the model's ability to solve
cryptic clues — the hypothesis is that private reasoning (as in
conversation) vs visible reasoning (standard API) is the key difference.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

OPUS = "claude-opus-4-20250514"
SONNET = "claude-sonnet-4-20250514"

SIMPLE_PROMPT = """You are an expert cryptic crossword solver. Given a clue and its answer, explain how the wordplay works.

Be precise about letter-level mechanics. Show how the letters combine to spell the answer.

Keep your explanation concise."""

TEST_CLUES = [
    ("Program's content lacking boundaries", "APP", 3),
    ("Garment's stiff — pressure's released by assistant at the back", "CORSET", 6),
    ("Fish grabbing snail in the middle, cracking shell", "CARAPACE", 8),
    ("New Ikea seat becoming centrepiece of trendy lounge", "TAKEITEASY", 10),
    ("One's extremely lucky with tiny cut during boxing match?", "FLYWEIGHT", 9),
    ("Well-built enclosure receiving go-ahead to accommodate 100", "STOCKY", 6),
    ("Heinous violent assault — officer's jacket torn apart", "ABHORRENT", 9),
    ("Rescuers firm as wind from the east loses strength, finally", "COASTGUARD", 10),
    ("Stewed dish I'd mentioned", "PIEEYED", 7),
    ("King Henry meets an Eastern ruler", "KHAN", 4),
]


def test_with_thinking(model, model_name, clue, answer, enum):
    """Run a single clue with extended thinking enabled."""
    user_msg = "Clue: %s (%d)\nAnswer: %s" % (clue, enum, answer)

    thinking_text = ""
    answer_text = ""
    tokens_in = 0
    tokens_out = 0

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

    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            answer_text = block.text.strip()

    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens
    return thinking_text, answer_text, tokens_in, tokens_out


def test_without_thinking(model, model_name, clue, answer, enum):
    """Run the same clue WITHOUT extended thinking for comparison."""
    user_msg = "Clue: %s (%d)\nAnswer: %s" % (clue, enum, answer)

    response = client.messages.create(
        model=model,
        max_tokens=600,
        temperature=0,
        messages=[{"role": "user", "content": user_msg}],
        system=SIMPLE_PROMPT,
    )

    answer_text = response.content[0].text.strip()
    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens
    return answer_text, tokens_in, tokens_out


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    results = {}

    # Run with extended thinking
    print("=" * 80)
    print("OPUS WITH EXTENDED THINKING")
    print("=" * 80)

    for clue, answer, enum in TEST_CLUES:
        print("\n--- %s (%d) = %s ---" % (clue[:60], enum, answer))
        try:
            thinking, response, tok_in, tok_out = test_with_thinking(
                OPUS, "Opus+Thinking", clue, answer, enum
            )
            print("THINKING (%d chars):" % len(thinking))
            print(thinking[:500])
            if len(thinking) > 500:
                print("... [truncated, %d total chars]" % len(thinking))
            print("\nRESPONSE:")
            print(response)
            print("[tokens: %d in, %d out]" % (tok_in, tok_out))
            results[answer] = {
                "thinking": thinking,
                "response": response,
            }
        except Exception as e:
            print("ERROR: %s" % e)
            results[answer] = {"thinking": "", "response": "ERROR: %s" % e}
        time.sleep(1)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for clue, answer, enum in TEST_CLUES:
        r = results.get(answer, {})
        resp = r.get("response", "N/A")
        preview = resp.replace("\n", " | ")[:200]
        print("\n%s = %s" % (clue[:60], answer))
        print("  Response: %s" % preview)


if __name__ == "__main__":
    main()
