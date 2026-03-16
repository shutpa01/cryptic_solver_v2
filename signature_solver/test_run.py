"""Test the signature solver against known clues from our puzzle analysis."""

from .db import RefDB
from .solver import solve


# Test clues from our 12-puzzle analysis
# Format: (wordplay_words, answer, expected_type, description)
TEST_CLUES = [
    # Pure anagrams
    (["Large", "pet", "damaged", "hotel"], "TELEGRAPH",
     "anagram+abbr", "9A P1 — anagram + abbreviation"),
    (["in", "glass", "smeared"], "SIGNALS",
     "anagram", "3D P1 — pure anagram"),
    (["I", "search", "desperately"], "CASHIER",
     "anagram", "18A P2 — pure anagram"),
    (["Fracture", "below"], "ELBOW",
     "anagram", "17A P12 — pure anagram 2 words"),

    # Hidden
    (["plant", "sap", "covered", "in"], "ANTS",
     "hidden", "19D P1 — hidden word"),
    (["Chin-chin", "gently", "swallows"], "INCHING",
     "hidden", "13A P1 — hidden in hyphenated"),

    # Charade: ABR + SYN
    (["Former", "lover", "mentioned"], "EXCITED",
     "charade", "12A P1 — ABR_F + SYN_F"),
    (["style", "our", "ships", "must", "follow"], "MODERN",
     "charade", "28A P1 — SYN_F + ABR_F"),
    (["Son", "to", "take", "out"], "SKILL",
     "charade", "7D P8 — ABR_F + SYN_F"),

    # Charade: SYN + SYN
    (["goat", "maybe", "then", "run", "away"], "BUTTERFLY",
     "charade", "1D P1 — SYN_F + SYN_F"),

    # Container: SYN inside SYN
    (["Fuss", "about", "a"], "STAIR",
     "container", "21A P9 — SYN contains RAW"),

    # Container: ABR inside SYN
    (["Victor", "entering", "only"], "SOLVE",
     "container", "2D P1 — ABR inside SYN"),

    # Reversal
    (["correct", "Backing"], "TIDE",
     "reversal", "29A P3 — reverse synonym"),

    # Deletion
    (["Swears", "at", "adult", "leaving"], "BUSES",
     "deletion", "1A P1 — synonym minus abbreviation"),
    (["flowers", "but", "not", "the", "first", "one"], "RISES",
     "deletion", "29A P1 — trim first from synonym"),

    # Homophone
    (["end", "is", "heard"], "TALE",
     "homophone", "1D P3 — homophone of synonym"),

    # Alternate letters
    (["The", "grasses", "regularly"], "TERSE",
     "alternate", "15A P8 — alternate letters"),

    # Double definition — can't really test mechanically, skip

    # Positional trim
    (["muffler", "end", "off"], "SCAR",
     "trim", "6A P2 — synonym trim last"),
    (["dull", "without", "leader"], "OFTEN",
     "trim", "4D P8 — synonym trim first"),
]


def run_tests():
    print("Loading reference database...")
    db = RefDB()
    print()

    solved = 0
    failed = 0

    for wordplay_words, answer, expected_type, description in TEST_CLUES:
        sr = solve(wordplay_words, answer, db)

        if sr.solved:
            r = sr.result
            conf = sr.confidence
            tier = "HIGH" if sr.high_confidence else "MED " if sr.medium_confidence else "LOW "
            print(f"{tier} [{conf:3d}]  {description}")
            print(f"        Answer: {answer}")
            print(f"        Signature: {r.signature_str()}")
            print(f"        {r.explanation_parts[0]}")
            print()
            solved += 1
        else:
            print(f"FAILED  {description}")
            print(f"        Answer: {answer}")
            print(f"        Words: {wordplay_words}")
            print()
            failed += 1

    print(f"{'='*60}")
    print(f"Results: {solved}/{solved+failed} solved ({100*solved/(solved+failed):.0f}%)")
    print(f"Solved: {solved}, Failed: {failed}")


if __name__ == "__main__":
    run_tests()
