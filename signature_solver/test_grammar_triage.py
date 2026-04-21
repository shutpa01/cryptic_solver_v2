"""Regression tests for grammar_triage.py.

Every clue we've confirmed solving correctly goes here.
Run after every change to grammar_triage.py to ensure
new mechanisms don't break existing solves.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.solver import solve_clue
from signature_solver.db import RefDB


# (clue_text, answer, expected_min_confidence, description)
KNOWN_SOLVES = [
    # Pure anagrams
    ("Learn the story about what may bring snow", "NORTHEASTERLY", 80,
     "anagram: LEARN THE STORY"),
    ("Electronic device disturbed mother's nap", "SMARTPHONE", 80,
     "anagram: MOTHERS NAP"),

    # Simple charades
    ("Completely keen on Tokyo's outskirts", "INTOTO", 80,
     "charade: INTO + T + O"),
    ("Unlocked shed is mine", "OPENCAST", 80,
     "charade: OPEN + CAST"),
    ("Leave with most of fruit and veg", "SPLITPEA", 80,
     "charade: SPLIT + PEA"),
    ("Sink India before noon, or capitulate?", "GIVEIN", 80,
     "charade: GIVE + I + N"),
    ("Drone perhaps beginning to film grouse", "BEEF", 80,
     "charade: BEE + F"),

    # Charade with positional pieces
    ("Son off to visit Mike and Iris regularly in US state", "MISSOURI", 80,
     "charade: M + I + S + SOUR + I (positional)"),
    ("Goes to court entrance, finding drunk is outside", "LITIGATES", 80,
     "charade: LIT + I + GATE + S"),
    ("Get these gardening tools as temperature lowers unexpectedly", "TROWELS", 80,
     "charade: TROWEL + S"),
    ("A lot of land trespasser calmly goes around to the west", "ACRES", 80,
     "charade: A + C + R + E + S"),

    # Container
    ("Groom horse, taking minutes", "COMB", 80,
     "container: COB containing M"),
    ("Divine woodland inhabited by heartless cobra", "FORECAST", 80,
     "container: FOREST containing CA (outer letters)"),

    # Container + charade
    ("Stir up Conservative amid departure close to here", "EXCITE", 80,
     "container+charade: EXIT containing C + E"),

    # Anagram with positional feed
    ("At first, Harry Potter got embroiled with emotion of greater relevance",
     "MORETOTHEPOINT", 80,
     "anagram+positional: H + anagram(POTTER+EMOTION)"),

    # Hidden reversed
    ("Turned over some carriages, salvaging weapon", "ASSEGAI", 80,
     "hidden reversed"),

    # DM 17861 — charade with positional
    ("Jenny, say, given varied roses is sort to calculate prices", "ASSESSOR", 80,
     "charade: ASS + ES + S + OR"),
    # Anagram with abbreviation substitution
    ("Underpass is unusually busy around western area", "SUBWAY", 80,
     "anagram: BUSY + W(western) + A(area)"),
]


def run_tests():
    print("Loading RefDB...")
    db = RefDB()

    passed = 0
    failed = 0
    errors = []

    for clue, answer, min_conf, desc in KNOWN_SOLVES:
        answer_clean = answer.upper().replace(" ", "").replace("-", "")
        try:
            sr = solve_clue(clue, answer_clean, db)
        except Exception as e:
            errors.append((answer, desc, "ERROR: %s" % e))
            failed += 1
            continue

        if sr.solved and sr.confidence >= min_conf:
            passed += 1
            print("  PASS  %s (%d) — %s" % (answer, sr.confidence, desc))
        else:
            failed += 1
            conf = sr.confidence if sr.solved else 0
            errors.append((answer, desc, "conf=%d (need %d)" % (conf, min_conf)))
            print("  FAIL  %s (conf=%d, need %d) — %s" % (answer, conf, min_conf, desc))

    print("\n%d passed, %d failed out of %d tests" % (passed, failed, len(KNOWN_SOLVES)))

    if errors:
        print("\nFAILURES:")
        for answer, desc, reason in errors:
            print("  %s: %s — %s" % (answer, desc, reason))
        return False
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
