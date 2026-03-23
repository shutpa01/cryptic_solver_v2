"""Bare API test: Sonnet vs Opus on failed clues, no enrichment.

Tests whether the model can solve cryptic clues with just the clue,
answer, and a simple prompt — no DB lookups, no indicator hints.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

SONNET = "claude-sonnet-4-20250514"
OPUS = "claude-opus-4-20250514"

# Simple prompt — just solve the clue, no framework
SIMPLE_PROMPT = """You are an expert cryptic crossword solver. Given a clue and its answer, explain how the wordplay works.

Be precise about letter-level mechanics. Show how the letters combine to spell the answer.

Keep your explanation concise."""

# Failed clues from puzzle 29492
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


def test_clue(model, model_name, clue, answer, enum):
    """Run a single clue through the bare API."""
    user_msg = "Clue: %s (%d)\nAnswer: %s" % (clue, enum, answer)

    response = client.messages.create(
        model=model,
        max_tokens=600,
        temperature=0,
        system=SIMPLE_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    explanation = response.content[0].text.strip()
    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens
    return explanation, tokens_in, tokens_out


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    models = [(SONNET, "Sonnet"), (OPUS, "Opus")]
    results = {}

    for model_id, model_name in models:
        print("=" * 80)
        print("MODEL: %s" % model_name)
        print("=" * 80)
        total_in = 0
        total_out = 0

        for clue, answer, enum in TEST_CLUES:
            print("\n--- %s (%d) = %s ---" % (clue[:50], enum, answer))
            try:
                explanation, tok_in, tok_out = test_clue(
                    model_id, model_name, clue, answer, enum
                )
                total_in += tok_in
                total_out += tok_out
                print(explanation)
                print("[tokens: %d in, %d out]" % (tok_in, tok_out))
                results.setdefault(answer, {})[model_name] = explanation
            except Exception as e:
                print("ERROR: %s" % e)
                results.setdefault(answer, {})[model_name] = "ERROR: %s" % e
            time.sleep(0.5)

        print("\n%s total tokens: %d in, %d out" % (model_name, total_in, total_out))

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    for clue, answer, enum in TEST_CLUES:
        print("\n%s = %s" % (clue[:60], answer))
        for model_name in ["Sonnet", "Opus"]:
            expl = results.get(answer, {}).get(model_name, "N/A")
            # Show first 200 chars
            preview = expl.replace("\n", " | ")[:200]
            print("  %s: %s" % (model_name, preview))


if __name__ == "__main__":
    main()
