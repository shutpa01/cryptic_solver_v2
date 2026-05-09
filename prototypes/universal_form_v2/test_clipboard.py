"""Hand-built test cases for the clipboard verifier.

Each case: a form (correct or deliberately wrong) + the clue + the
expected verdict + the failure reasons we expect to see if FAIL.

Run: python -m prototypes.universal_form_v2.test_clipboard
"""
import sys
from prototypes.universal_form_v2.schema import (
    Form, Definition, Node, charade, anagram, reversal, container,
    positional, syn, abbr, lit
)
from prototypes.universal_form_v2.clipboard_verifier import verify
from signature_solver.db import RefDB

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def case(name, form, clue, expected, expect_failed_checks=None,
         expect_enrichment_kinds=None):
    """Run verifier and report pass/fail.

    expect_failed_checks: set of check names expected to fail.
    expect_enrichment_kinds: set of enrichment-candidate kinds expected.
    """
    print(f"\n=== {name} ===")
    print(f"  clue: {clue}")
    v = verify(form, clue, db)
    print(f"  expected={expected}  got={v.verdict}")
    for c in v.checks:
        marker = "✓" if c.status == "pass" else "✗"
        print(f"    {marker} {c.name}: {c.detail[:90]}")
    if v.enrichment_candidates:
        print("  enrichment candidates:")
        for ec in v.enrichment_candidates:
            print(f"    - {ec.kind}: {ec.detail}")
    correct = (v.verdict == expected)
    if correct and expect_failed_checks:
        actual_fails = {c.name for c in v.checks if c.status == "fail"}
        missing = expect_failed_checks - actual_fails
        if missing:
            print(f"  ! expected check names to fail: {missing}")
            correct = False
    if correct and expect_enrichment_kinds:
        actual_kinds = {ec.kind for ec in v.enrichment_candidates}
        missing = expect_enrichment_kinds - actual_kinds
        if missing:
            print(f"  ! expected enrichment kinds: {missing}")
            correct = False
    print(f"  {'OK' if correct else 'BUG'}")
    return correct


db = RefDB()
results = []

# ---- 1. SHARP correct ----------------------------------------------------
# charade(positional[first](S ← shrill, ind=principally), syn(HARP ← instrument))
s_leaf = positional(source_word="shrill", value="S", kind="first")
s_leaf.indicator = "principally"
form = Form(
    tree=charade(s_leaf, syn(source_word="instrument", value="HARP")),
    definition=Definition(phrase="Piercing", answer="SHARP"),
    link_words=[],
)
results.append(case(
    "SHARP — correct form",
    form,
    "Piercing, principally shrill instrument",
    expected="PASS",
))


# ---- 2. SHARP wrong (literal S from "principally shrill") -----------------
form = Form(
    tree=charade(
        lit(source_word="principally shrill", value="S"),
        syn(source_word="instrument", value="HARP"),
    ),
    definition=Definition(phrase="Piercing", answer="SHARP"),
    link_words=[],
)
results.append(case(
    "SHARP — wrong form (literal S from full phrase)",
    form,
    "Piercing, principally shrill instrument",
    expected="FAIL",
    expect_failed_checks={"mechanism.leaves"},
))


# ---- 3. LIVERWORT wrong (anagram with synonym/positional children) -------
ana = anagram(
    syn(source_word="On air", value="LIVE"),
    lit(source_word="row", value="ROW"),
    positional(source_word="exhausted Robert", value="RT", kind="outer"),
    indicator="traumatised",
)
form = Form(
    tree=ana,
    definition=Definition(phrase="Plant", answer="LIVERWORT"),
    link_words=["and"],
)
results.append(case(
    "LIVERWORT — wrong form (anagram absorbing all)",
    form,
    "On air row traumatised and exhausted Robert Plant",
    expected="FAIL",
    expect_failed_checks={"mechanism.indicators"},
))


# ---- 4. MILDEW correct ---------------------------------------------------
# charade(syn(MILD←beer), reversal(literal(WE←we), ind=knocked back))
form = Form(
    tree=charade(
        syn(source_word="beer", value="MILD"),
        reversal(lit(source_word="we", value="WE"),
                 indicator="knocked back"),
    ),
    definition=Definition(phrase="Disease", answer="MILDEW"),
    link_words=["from"],
)
results.append(case(
    "MILDEW — correct form",
    form,
    "Disease from beer we knocked back",
    expected="PASS",
))


# ---- 5. Bogus homophone (BABY → PSYCHE) ----------------------------------
# homophone op wrapping syn(BABY ← exhale): no homophone DB row for
# baby↔psyche
form = Form(
    tree=Node(
        operation="homophone",
        indicator="audibly",
        sources=[syn(source_word="exhale", value="BABY")],
    ),
    definition=Definition(phrase="spirit",
                            answer="PSYCHE"),
    link_words=["in", "relief", "vital", "for"],
)
results.append(case(
    "PSYCHE — bogus homophone (BABY → PSYCHE not in DB)",
    form,
    "Audibly exhale in relief vital for spirit",
    expected="FAIL",
    expect_failed_checks={"mechanism.indicators"},
    expect_enrichment_kinds={"homophone"},
))


# ---- 6. Missing DB row (deliberately invented synonym) -------------------
# charade(syn(GROAT ← whatsit), abbr(X ← y))  — invented pieces.
# Coverage will fail too because surface words aren't really there;
# but we just want to see the missing-DB-row case fail with an
# enrichment candidate.
form = Form(
    tree=charade(
        syn(source_word="floozle", value="GROAT"),
        abbr(source_word="bibble", value="X"),
    ),
    definition=Definition(phrase="thing", answer="GROATX"),
    link_words=[],
)
results.append(case(
    "Invented words — missing DB rows produce enrichment candidates",
    form,
    "thing floozle bibble",
    expected="FAIL",
    expect_failed_checks={"mechanism.leaves"},
    expect_enrichment_kinds={"synonym", "abbreviation"},
))


# ---- 7. Residue: leftover content word ----------------------------------
# Take MILDEW but add an extra unaccounted word "elephant" in the clue.
# Verifier should fail residue.
form = Form(
    tree=charade(
        syn(source_word="beer", value="MILD"),
        reversal(lit(source_word="we", value="WE"),
                 indicator="knocked back"),
    ),
    definition=Definition(phrase="Disease", answer="MILDEW"),
    link_words=["from"],
)
results.append(case(
    "MILDEW + stray content word — residue should fail",
    form,
    "Disease elephant from beer we knocked back",
    expected="FAIL",
    expect_failed_checks={"residue"},
))


# ---- 8. Container correct: CORD --------------------------------------------
# "Republican in fish" → R inside COD = CORD
# container[in](syn(COD←fish), abbr(R←Republican))
form = Form(
    tree=container(
        syn(source_word="fish", value="COD"),
        abbr(source_word="Republican", value="R"),
        indicator="in",
    ),
    definition=Definition(phrase="Rope", answer="CORD"),
    link_words=[],
)
results.append(case(
    "CORD — container[in](syn COD, abbr R)",
    form,
    "Rope Republican in fish",
    expected="PASS",
))


# ---- 9. Container WRONG: outer/inner swapped -------------------------------
# Same clue, but form claims R is the outer and COD inner.
# Verifier should FAIL because outer (R, 1 char) cannot wrap inner (COD).
form = Form(
    tree=container(
        abbr(source_word="Republican", value="R"),
        syn(source_word="fish", value="COD"),
        indicator="in",
    ),
    definition=Definition(phrase="Rope", answer="CORD"),
    link_words=[],
)
results.append(case(
    "CORD — wrong order (R as outer)",
    form,
    "Rope Republican in fish",
    expected="FAIL",
    expect_failed_checks={"assembly"},
))


# ---- 10. Hidden correct: EVITA ---------------------------------------------
# "Show some incredible vitality" → EVITA hidden in "incredible vitality"
# hidden[some](literal("incredible"←"incredible"), literal("vitality"←"vitality"))
form = Form(
    tree=Node(
        operation="hidden",
        indicator="some",
        sources=[
            lit(source_word="incredible", value="INCREDIBLE"),
            lit(source_word="vitality", value="VITALITY"),
        ],
    ),
    definition=Definition(phrase="Show", answer="EVITA"),
    link_words=[],
)
results.append(case(
    "EVITA — hidden[some](literal incredible, literal vitality)",
    form,
    "Show some incredible vitality",
    expected="PASS",
))


# ---- 11. Deletion correct: ARCH --------------------------------------------
# "Chief taking off first of the month"
# deletion[head](syn(MARCH←month), indicator="taking off first")
# MARCH minus first letter = ARCH
form = Form(
    tree=Node(
        operation="deletion",
        deletion_kind="head",
        indicator="taking off first",
        sources=[syn(source_word="month", value="MARCH")],
    ),
    definition=Definition(phrase="Chief", answer="ARCH"),
    link_words=["of", "the"],
)
results.append(case(
    # Form is structurally correct; "taking off first" isn't in the
    # indicators DB, so the verifier FAILs the form and queues an
    # indicator enrichment candidate. After enrichment a re-run would
    # PASS.
    "ARCH — deletion form correct, indicator missing in DB",
    form,
    "Chief taking off first of the month",
    expected="FAIL",
    expect_failed_checks={"mechanism.indicators"},
    expect_enrichment_kinds={"indicator"},
))


# ---- 12. Compound: charade with reversal child ----------------------------
# MILDEW already covered, but let me also test that the verifier passes
# a charade where the reversal child has a synonym leaf inside.
# NUTMEG: "Fanatic with stone climbing tree" → NUT + GEM-reversed = NUTMEG
# charade(syn(NUT←Fanatic), reversal(syn(GEM←stone), ind=climbing))
form = Form(
    tree=charade(
        syn(source_word="Fanatic", value="NUT"),
        reversal(
            syn(source_word="stone", value="GEM"),
            indicator="climbing",
        ),
    ),
    definition=Definition(phrase="tree", answer="NUTMEG"),
    link_words=["with"],
)
results.append(case(
    "NUTMEG — charade(syn, reversal(syn))",
    form,
    "Fanatic with stone climbing tree",
    expected="PASS",
))


# ---- Summary -------------------------------------------------------------
print(f"\n\n{'='*50}")
n_ok = sum(1 for r in results if r)
print(f"Tests: {n_ok}/{len(results)} behaved as expected")
print('='*50)
sys.exit(0 if n_ok == len(results) else 1)
