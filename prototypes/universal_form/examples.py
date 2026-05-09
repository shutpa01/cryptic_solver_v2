"""Hand-built example trees — the proof that the 11 basic types are
expressive enough.

Every example below is a real clue. The trees are built using the
short aliases from schema.py. The main block prints each one's JSON
serialisation and validates it.

Coverage of the 11 basic types:

    leaves:     literal, synonym, abbreviation, positional
    operations: charade, container, anagram, reversal, deletion,
                hidden, double_definition

Plus a compound (anagram-of-deletion) and a reversal-of-charade — the two
shapes the universal form was designed to make trivially representable.
"""
from __future__ import annotations

from .schema import (
    Form, Definition, validate,
    lit, syn, abbr, pos,
    charade, container, anagram, reversal, deletion, hidden, double_definition,
)


# 1. Pure charade — Republican + fish = RIDE
#    Clue: "Republican fish (4)"
#    Two leaves of different types (abbreviation + synonym) joined by charade.
RIDE = Form(
    tree=charade(
        abbr("Republican", "R"),
        syn("fish", "IDE"),
    ),
    definition=Definition(phrase="fish", answer="RIDE"),  # placeholder
    link_words=[],
)

# 2. Container — REPAST. PA inside REST.
#    Clue: "Old man, among others, gets a meal" (6)
REPAST = Form(
    tree=container(
        outer=syn("others", "REST"),
        inner=syn("old man", "PA"),
        indicator="among",
    ),
    definition=Definition(phrase="meal", answer="REPAST"),
    link_words=["gets", "a"],
)

# 3. Reversal-of-charade — REVOLTING. Lover + nit + good, reversed.
#    Outer-op-first: reversal at the root, charade as child.
#    NOTE: the charade children are listed in ASSEMBLY order, not surface
#    order. To produce REVOLTING by reversing, the charade must build
#    GNITLOVER (G + NIT + LOVER); reversed = REVOLTING. Surface order
#    "lover nit good" appears in residue via source_word matching but the
#    tree records the order the pieces actually concatenate in.
REVOLTING = Form(
    tree=reversal(
        charade(
            abbr("good", "G"),
            lit("nit"),
            syn("lover", "LOVER"),
        ),
        indicator="back",
    ),
    definition=Definition(phrase="revolting", answer="REVOLTING"),
    link_words=["and"],
)

# 4. Deletion-anagram — NEAPOLITAN. Anagram of (Antonio mostly + pale).
#    "deletion anagram" by the user's naming convention (deletion first,
#    then anagram). In tree form: anagram outer, deletion as one child.
#    Clue: "Cook Antonio, mostly pale native of southern Italy" (10)
NEAPOLITAN = Form(
    tree=anagram(
        deletion(
            lit("Antonio", "ANTONIO"),
            indicator="mostly",
            kind="tail",
        ),
        lit("pale", "PALE"),
        indicator="Cook",
    ),
    definition=Definition(phrase="native of southern Italy",
                          answer="NEAPOLITAN"),
    link_words=[],
)

# 5. Pure anagram — FEAST. Cook fate's = FEAST.
#    Clue: "Twist of fate's causing blow-out" (5)
#    (Real clue — appears as "tutorial" model_version in the live DB.)
FEAST = Form(
    tree=anagram(
        lit("fate's", "FATES"),
        indicator="Twist",
    ),
    definition=Definition(phrase="blow-out", answer="FEAST"),
    link_words=["of", "causing"],
)

# 6. Hidden — TAR. "that a rainstorm" → TAR.
#    Clue: "Pitch that a rainstorm covers" (3)
#    Hidden's children are the literals of the spanning clue words.
TAR = Form(
    tree=hidden(
        lit("that"), lit("a"), lit("rainstorm"),
        indicator="covers",
    ),
    definition=Definition(phrase="Pitch", answer="TAR"),
    link_words=[],
)

# 7. Double definition — SULKY.
#    Clue: "Light carriage being put out?" (5)
DD_SULKY = Form(
    tree=double_definition(
        syn("Light carriage", "SULKY"),
        syn("being put out", "SULKY"),
    ),
    definition=Definition(phrase="Light carriage", answer="SULKY"),
    link_words=[],
)

# 8. Positional + charade — VINDICATION.
#    "Start of victory sign for defence" (11)
#    V (start of victory) + INDICATION (synonym for sign) = VINDICATION
VINDICATION = Form(
    tree=charade(
        pos("victory", "V", kind="first", indicator="Start of"),
        syn("sign", "INDICATION"),
    ),
    definition=Definition(phrase="defence", answer="VINDICATION"),
    link_words=["for"],
)


ALL_EXAMPLES = {
    "RIDE": RIDE,
    "REPAST": REPAST,
    "REVOLTING": REVOLTING,
    "NEAPOLITAN": NEAPOLITAN,
    "FEAST": FEAST,
    "TAR": TAR,
    "SULKY (DD)": DD_SULKY,
    "VINDICATION": VINDICATION,
}


if __name__ == "__main__":
    import json
    failures = 0
    for name, form in ALL_EXAMPLES.items():
        problems = validate(form)
        marker = "OK " if not problems else "BAD"
        print(f"--- {marker}  {name} ({form.definition.answer}) ---")
        if problems:
            for p in problems:
                print(f"  PROBLEM: {p}")
            failures += 1
        # Round-trip JSON to prove the schema is self-consistent
        round_tripped = Form.from_json(form.to_json())
        if round_tripped.to_dict() != form.to_dict():
            print("  PROBLEM: JSON round-trip changed the form")
            failures += 1
        print(form.to_json())
        print()
    print(f"=== {len(ALL_EXAMPLES) - failures}/{len(ALL_EXAMPLES)} examples valid ===")
