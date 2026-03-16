"""Format word analyses as structured evidence for API calls.

Instead of asking the API to reason about a cryptic clue from scratch,
we give it pre-computed evidence: for each word in the wordplay window,
what roles it could play (synonym, abbreviation, indicator, etc.) with
specific values from our reference DB.

This turns the API's task from open-ended reasoning into constrained
selection from verified options.
"""

from .tokens import *

# Compact labels for display
_TOKEN_LABELS = {
    SYN_F: "synonym",
    ABR_F: "abbreviation",
    ANA_F: "anagram fodder",
    RAW: "literal letters",
    HID_F: "hidden fodder",
    HOM_F: "homophone",
    DEL_F: "deletion fodder",
    POS_F: "positional fodder",
    ANA_I: "anagram indicator",
    REV_I: "reversal indicator",
    CON_I: "container indicator",
    DEL_I: "deletion indicator",
    HID_I: "hidden indicator",
    HOM_I: "homophone indicator",
    POS_I_FIRST: "first-letter indicator",
    POS_I_LAST: "last-letter indicator",
    POS_I_OUTER: "outer-letters indicator",
    POS_I_MIDDLE: "middle-letter indicator",
    POS_I_ALTERNATE: "alternate-letters indicator",
    POS_I_TRIM_FIRST: "beheadment indicator",
    POS_I_TRIM_LAST: "curtailment indicator",
    POS_I_TRIM_MIDDLE: "disembowelment indicator",
    POS_I_TRIM_OUTER: "shelling indicator",
    POS_I_HALF: "halving indicator",
    LNK: "link word",
}

# Tokens worth showing to the API (skip noise)
_SKIP_TOKENS = {HID_F, ANA_F, POS_F, DEL_F}


def format_evidence(analyses, phrases, answer, clue_text=None, definition=None):
    """Format word analyses as a structured evidence block for the API.

    Args:
        analyses: list of WordAnalysis objects (one per word)
        phrases: dict of (i, j) -> WordAnalysis for multi-word phrases
        answer: the known answer (uppercase, no spaces)
        clue_text: optional full clue text
        definition: optional definition text

    Returns:
        str: formatted evidence block for inclusion in API prompt
    """
    lines = []

    if clue_text:
        lines.append(f"Clue: {clue_text}")
    if definition:
        lines.append(f"Definition: \"{definition}\"")
    lines.append(f"Answer: {answer} ({len(answer)} letters)")
    lines.append("")
    lines.append("=== Word Analysis (from reference database) ===")
    lines.append("")

    for i, wa in enumerate(analyses):
        word = wa.text
        roles = []

        for tok, vals in wa.roles.items():
            if tok in _SKIP_TOKENS:
                continue
            label = _TOKEN_LABELS.get(tok, tok)

            if tok == RAW:
                # Only show RAW for short words (1-2 chars) — these are
                # commonly used as literal letter contributions
                w_alpha = "".join(c for c in word.upper() if c.isalpha())
                if len(w_alpha) <= 2:
                    roles.append(f"literal: {w_alpha}")
                continue

            if tok == LNK:
                roles.append("link word (ignorable)")
                continue

            if vals == [True]:
                roles.append(label)
            else:
                # Show values — cap at 8 to avoid overwhelming
                display_vals = vals[:8]
                if tok in (SYN_F, HOM_F):
                    # For synonyms, show the ones that are substrings of
                    # the answer first (most relevant)
                    in_answer = [v for v in vals if v in answer]
                    not_in = [v for v in vals if v not in answer and v != answer]
                    # Also flag exact-length matches
                    exact_len = [v for v in not_in if len(v) == len(answer)]
                    short = [v for v in not_in if v not in exact_len]
                    ordered = in_answer + exact_len + short
                    display_vals = ordered[:8]

                val_str = ", ".join(str(v) for v in display_vals)
                if len(vals) > 8:
                    val_str += f" (+{len(vals)-8} more)"
                roles.append(f"{label}: {val_str}")

        if roles:
            lines.append(f"  {word}: {' | '.join(roles)}")
        else:
            lines.append(f"  {word}: (no known roles)")

    # Show phrase analyses
    if phrases:
        lines.append("")
        lines.append("=== Multi-word phrases ===")
        for (pi, pj), pwa in phrases.items():
            roles = []
            for tok, vals in pwa.roles.items():
                label = _TOKEN_LABELS.get(tok, tok)
                if vals == [True]:
                    roles.append(label)
                else:
                    val_str = ", ".join(str(v) for v in vals[:5])
                    roles.append(f"{label}: {val_str}")
            if roles:
                lines.append(f"  \"{pwa.text}\": {' | '.join(roles)}")

    return "\n".join(lines)


def format_failed_solve_context(solve_result, answer):
    """Format context about a failed mechanical solve for the API.

    If the solver found a low-confidence result, include it as a
    candidate (but flagged as unverified).
    """
    lines = []

    if solve_result.result and solve_result.confidence > 0:
        r = solve_result.result
        lines.append(f"Candidate solution (unverified, confidence {solve_result.confidence}/100):")
        lines.append(f"  Signature: {r.signature_str()}")
        lines.append(f"  {r.explanation_parts[0]}")
        lines.append("")

    return "\n".join(lines)
