#!/usr/bin/env python3
"""Test unified_parse_builder on a single clue with debug output."""

import sys
sys.path.insert(0, r"C:\Users\shute\PycharmProjects\AI_Solver\Solver\orchestrator")

from unified_parse_builder import UnifiedParseBuilder

DB_PATH = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"

# Test clue
clue = "More beloved member embraced by the German"
answer = "dearer"

print("=" * 60)
print(f"CLUE: {clue}")
print(f"ANSWER: {answer}")
print("=" * 60)

builder = UnifiedParseBuilder(DB_PATH)

# Run with debug on
result = builder.parse(clue, answer, definition_words=None, debug=True)

print("\n" + "=" * 60)
if result:
    print(f"RESULT: {result.derivation}")
    print(f"Complete: {result.is_complete}")
    print(f"Contributions: {len(result.contributions)}")
    for c in result.contributions:
        print(f"  - {c.source_words} → {c.letters} ({c.operation})")
else:
    print("RESULT: None (no parse found)")

builder.close()