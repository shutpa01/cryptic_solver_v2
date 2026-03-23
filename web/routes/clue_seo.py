"""SEO helpers for individual clue pages — meta descriptions, JSON-LD schemas."""

import json
import re


def generate_meta_description(clue):
    """Build a dynamic meta description for a clue page.

    Aims for 150-160 chars. Includes clue text, source, and a hint at
    what the user will find (definition, wordplay type, or just "answer").

    Args:
        clue: dict with clue_text, enumeration, source, puzzle_number,
              type_label, definition, wordplay_type, answer.
    """
    clue_text = clue.get("clue_text", "")
    enum = clue.get("enumeration", "")
    source = (clue.get("source") or "").title()
    type_label = clue.get("type_label") or ""
    puzzle_number = clue.get("puzzle_number", "")
    definition = clue.get("definition")
    wordplay_type = clue.get("wordplay_type")

    # Core: clue text with enumeration
    core = clue_text
    if enum:
        core += f" ({enum})"

    # Origin line
    origin = f"{source}"
    if type_label:
        origin += f" {type_label}"
    origin += f" #{puzzle_number}"

    # What we can offer
    if definition and wordplay_type:
        offer = "Step-by-step hints: definition, wordplay type, full explanation, and answer."
    elif definition:
        offer = "Hints available: definition and answer."
    else:
        offer = "Answer and hints for this cryptic crossword clue."

    desc = f'Cryptic crossword clue: "{core}" from {origin}. {offer}'

    # Truncate to ~160 chars if needed
    if len(desc) > 160:
        desc = desc[:157] + "..."

    return desc


def generate_faq_schema(clue, steps):
    """Build FAQPage JSON-LD schema for a clue.

    Each available hint step becomes a question/answer pair.
    Google shows FAQ rich results for pages with this markup.

    Args:
        clue: dict with clue_text, enumeration, answer, definition,
              wordplay_type, and explanation content.
        steps: list of step dicts from get_hint_steps.

    Returns:
        JSON string ready for a <script type="application/ld+json"> tag.
    """
    clue_text = clue.get("clue_text", "")
    enum = clue.get("enumeration", "")
    clue_display = clue_text
    if enum:
        clue_display += f" ({enum})"

    faq_entries = []

    # Q1: What does [clue] mean?
    faq_entries.append({
        "@type": "Question",
        "name": f'What does the cryptic crossword clue "{clue_display}" mean?',
        "acceptedAnswer": {
            "@type": "Answer",
            "text": _build_meaning_answer(clue),
        },
    })

    # Q2: What is the answer?
    answer = clue.get("answer", "")
    if answer:
        faq_entries.append({
            "@type": "Question",
            "name": f'What is the answer to "{clue_display}"?',
            "acceptedAnswer": {
                "@type": "Answer",
                "text": f"The answer is {answer}.",
            },
        })

    # Q3: What type of wordplay is used?
    wordplay_type = clue.get("wordplay_type")
    if wordplay_type:
        wp_label = _wordplay_label(wordplay_type)
        faq_entries.append({
            "@type": "Question",
            "name": f'What type of wordplay is used in "{clue_display}"?',
            "acceptedAnswer": {
                "@type": "Answer",
                "text": f"This clue uses {wp_label} as its wordplay technique.",
            },
        })

    schema = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": faq_entries,
    }

    return json.dumps(schema, ensure_ascii=False)


def generate_breadcrumb_schema(clue):
    """Build BreadcrumbList JSON-LD schema for a clue page.

    Breadcrumb: Home > Source Type > Puzzle #N > Clue

    Args:
        clue: dict with source, type_slug, type_label, puzzle_number,
              puzzle_url, clue_text, enumeration.

    Returns:
        JSON string ready for a <script type="application/ld+json"> tag.
    """
    source = (clue.get("source") or "").title()
    type_label = clue.get("type_label") or ""
    type_slug = clue.get("type_slug") or ""
    puzzle_number = clue.get("puzzle_number", "")
    clue_text = clue.get("clue_text", "")
    enum = clue.get("enumeration", "")

    clue_display = clue_text
    if enum:
        clue_display += f" ({enum})"
    # Truncate long clue text for breadcrumb
    if len(clue_display) > 60:
        clue_display = clue_display[:57] + "..."

    items = [
        {
            "@type": "ListItem",
            "position": 1,
            "name": "Home",
            "item": "/",
        },
    ]

    if source and type_slug:
        items.append({
            "@type": "ListItem",
            "position": 2,
            "name": f"{source} {type_label}",
            "item": f"/{clue.get('source')}/{type_slug}/",
        })

    if puzzle_number:
        items.append({
            "@type": "ListItem",
            "position": len(items) + 1,
            "name": f"#{puzzle_number}",
            "item": clue.get("puzzle_url") or f"/{clue.get('source')}/{type_slug}/{puzzle_number}",
        })

    items.append({
        "@type": "ListItem",
        "position": len(items) + 1,
        "name": clue_display,
    })

    schema = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": items,
    }

    return json.dumps(schema, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_meaning_answer(clue):
    """Build a rich answer for the 'what does this clue mean' FAQ entry."""
    parts = []
    definition = clue.get("definition")
    wordplay_type = clue.get("wordplay_type")
    answer = clue.get("answer", "")

    if definition:
        parts.append(f'The definition part of the clue is "{definition}".')
    if wordplay_type:
        wp_label = _wordplay_label(wordplay_type)
        parts.append(f"The wordplay technique is {wp_label}.")
    if answer:
        parts.append(f"The answer is {answer}.")

    if not parts:
        return "Visit the page for progressive hints and the full answer."

    parts.append("Visit the page for the full step-by-step explanation.")
    return " ".join(parts)


_WORDPLAY_LABELS = {
    "anagram": "an anagram",
    "charade": "a charade (building blocks)",
    "container": "a container (one word inside another)",
    "hidden": "a hidden word",
    "reversal": "a reversal",
    "double_definition": "a double definition",
    "cryptic_definition": "a cryptic definition",
    "homophone": "a homophone (sounds like)",
    "deletion": "a deletion",
    "substitution": "a substitution",
    "spoonerism": "a spoonerism",
    "initial_letters": "initial letters",
    "alternation": "alternating letters",
}


def _wordplay_label(wordplay_type):
    """Return a human-friendly label for a wordplay type."""
    return _WORDPLAY_LABELS.get(wordplay_type, wordplay_type.replace("_", " "))
