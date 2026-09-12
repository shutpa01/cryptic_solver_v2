"""SEO helpers for individual clue pages — meta descriptions, JSON-LD schemas."""

import json
import re

from flask import current_app


MAX_META_DESCRIPTION = 160

# Never shorten the clue text below this — past it the snippet stops
# matching what the searcher typed.
_MIN_CLUE_CHARS = 40

# Publication display names. `source.title()` alone gives "Dailymail".
# Matches web/routes/browse.py and web/routes/seo.py.
_SOURCE_DISPLAY = {
    "dailymail": "Daily Mail",
    "telegraph-toughie": "Telegraph Toughie",
    "telegraph": "Telegraph",
    "times": "Times",
    "guardian": "Guardian",
    "independent": "Independent",
}

# Short wordplay names for the meta description. Deliberately terser than
# _WORDPLAY_LABELS (which carries parentheticals like "a charade (building
# blocks)") because every character here competes with the clue text.
_WORDPLAY_SHORT = {
    "anagram": "anagram",
    "charade": "charade",
    "container": "container",
    "hidden": "hidden-word",
    "reversal": "reversal",
    "double_definition": "double-definition",
    "cryptic_definition": "cryptic-definition",
    "homophone": "homophone",
    "deletion": "deletion",
    "substitution": "substitution",
    "spoonerism": "spoonerism",
    "initial_letters": "initial-letters",
    "alternation": "alternating-letters",
}


def _source_display(source):
    """Human-readable publication name for a source slug."""
    source = (source or "").strip()
    if not source:
        return ""
    return _SOURCE_DISPLAY.get(source.lower(), source.replace("-", " ").title())


def _truncate_words(text, limit):
    """Cut text to at most `limit` chars on a word boundary, with an ellipsis.

    Never leaves a half-word. The previous blunt `desc[:157] + "..."` was
    producing snippets that ended "...and how every word builds the
    explanat..." — the one word the snippet exists to say.
    """
    if len(text) <= limit:
        return text
    cut = text[:limit - 1].rstrip()
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:-—") + "…"


def generate_meta_description(clue):
    """Build a dynamic meta description for a clue page.

    Shape: `"CLUE (enum)" from Source Type #N — answer plus a full
    explanation of ...`

    The explanation is the differentiator, so it is named in every variant
    and it is never the part that gets cut. When the line is over budget we
    degrade in this order: drop the puzzle type label, shorten the clue
    text, drop the origin entirely. The offer clause stays whole.

    Args:
        clue: dict with clue_text, enumeration, source, puzzle_number,
              type_label, definition, wordplay_type, answer.
    """
    clue_text = (clue.get("clue_text") or "").strip()
    enum = clue.get("enumeration") or ""
    source = _source_display(clue.get("source"))
    type_label = clue.get("type_label") or ""
    puzzle_number = clue.get("puzzle_number") or ""
    definition = clue.get("definition")
    wordplay_type = clue.get("wordplay_type")

    # Clue text as it reads in the paper.
    core = clue_text + (f" ({enum})" if enum else "")

    # What the page actually gives you. Naming the wordplay type when we
    # know it also keeps descriptions distinct across clue pages.
    if definition and wordplay_type:
        wp = _WORDPLAY_SHORT.get(wordplay_type, str(wordplay_type).replace("_", "-"))
        offer = (
            "answer plus a full explanation of the definition and the "
            f"{wp} wordplay."
        )
    elif definition:
        offer = "answer plus a full explanation of the definition and the wordplay."
    else:
        offer = "answer plus a full explanation of how the wordplay works, word by word."

    def _origin(with_type):
        bits = [source] + ([type_label] if with_type and type_label else [])
        text = " ".join(b for b in bits if b)
        if puzzle_number:
            text = f"{text} #{puzzle_number}".strip()
        return text

    def _assemble(core_text, origin):
        lead = f'"{core_text}"' if core_text else ""
        if origin:
            lead = f"{lead} from {origin}".strip() if lead else origin
        if not lead:
            return offer[0].upper() + offer[1:]
        return f"{lead} — {offer}"

    for candidate in (_assemble(core, _origin(True)), _assemble(core, _origin(False))):
        if len(candidate) <= MAX_META_DESCRIPTION:
            return candidate

    # Only the clue text is still elastic; shorten that, not the offer.
    over = len(_assemble(core, _origin(False))) - MAX_META_DESCRIPTION
    short_core = _truncate_words(core, max(len(core) - over, _MIN_CLUE_CHARS))
    desc = _assemble(short_core, _origin(False))
    if len(desc) > MAX_META_DESCRIPTION:
        desc = _assemble(short_core, "")
    if len(desc) > MAX_META_DESCRIPTION:
        desc = _truncate_words(desc, MAX_META_DESCRIPTION)
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

    answer = clue.get("answer", "")
    definition = clue.get("definition")
    wordplay_type = clue.get("wordplay_type")
    confidence = clue.get("confidence")
    is_high = confidence is not None and confidence >= 0.7
    is_medium = confidence is not None and confidence >= 0.4 and not is_high

    # Q1: What does [clue] mean?
    #
    # We deliberately do NOT include the full ai_explanation step-by-step
    # text in this block, even for HIGH-confidence clues. That text is the
    # proprietary content the solver pipeline produces; embedding it in
    # public JSON-LD made it scrapable from one HTTP GET. The structured
    # data still tells Google what the page is about (definition + wordplay
    # type + answer) which is enough for SERP context, without handing
    # over the breakdown that's behind the hint reveal flow.
    #
    # The STRIP_DEFINITION_FROM_JSONLD config flag (web/config.py) replaces
    # the definition+wordplay sentences with a teaser, leaving only the
    # answer in JSON-LD. Use this to cut off scrapers harvesting our parses
    # without removing the answer Google needs for "{clue} crossword answer"
    # queries.
    strip_def = bool(current_app.config.get("STRIP_DEFINITION_FROM_JSONLD", False))
    if strip_def:
        meaning_parts = []
        if answer:
            meaning_parts.append(f"The answer is {answer}.")
        meaning_parts.append(
            "Visit the page for the full step-by-step explanation, "
            "including which word in the clue serves as the definition."
        )
        meaning_text = " ".join(meaning_parts)
    elif (is_high or is_medium) and (definition or wordplay_type):
        meaning_parts = []
        if definition:
            meaning_parts.append(f'The definition is "{definition}".')
        if wordplay_type:
            wp_label = _wordplay_label(wordplay_type)
            meaning_parts.append(f"The wordplay uses {wp_label}.")
        if answer:
            meaning_parts.append(f"The answer is {answer}.")
        meaning_parts.append("Visit the page for the full step-by-step explanation.")
        meaning_text = " ".join(meaning_parts)
    else:
        # LOW/FAIL/PENDING — teaser only
        meaning_text = "This cryptic clue uses wordplay to arrive at the answer. Visit the page for progressive hints — definition, wordplay type, and a full step-by-step explanation."

    faq_entries.append({
        "@type": "Question",
        "name": f'What does the cryptic crossword clue "{clue_display}" mean?',
        "acceptedAnswer": {
            "@type": "Answer",
            "text": meaning_text,
        },
    })

    # Q2: What is the answer?
    if answer:
        faq_entries.append({
            "@type": "Question",
            "name": f'What is the answer to "{clue_display}"?',
            "acceptedAnswer": {
                "@type": "Answer",
                "text": f"The answer is {answer}.",
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

    base = "https://justcordelia.com"
    items = [
        {
            "@type": "ListItem",
            "position": 1,
            "name": "Home",
            "item": f"{base}/",
        },
    ]

    if source and type_slug:
        items.append({
            "@type": "ListItem",
            "position": 2,
            "name": f"{source} {type_label}",
            "item": f"{base}/{clue.get('source')}/{type_slug}/",
        })

    if puzzle_number:
        puzzle_path = clue.get("puzzle_url") or f"/{clue.get('source')}/{type_slug}/{puzzle_number}"
        items.append({
            "@type": "ListItem",
            "position": len(items) + 1,
            "name": f"#{puzzle_number}",
            "item": f"{base}{puzzle_path}",
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


def generate_word_roles_schema(clue, role_groups, mechanism_label=None):
    """Build a DefinedTermSet JSON-LD block describing the word-by-word
    breakdown. Each piece in role_groups becomes a DefinedTerm with its
    role + letters as the term's name and description.

    The visible HTML already shows the same data; this just hands the
    classifier a machine-readable view of it. Returns an empty string
    when there are no role_groups (the structured-data block then
    omits the script tag entirely).

    Respects STRIP_DEFINITION_FROM_JSONLD: when set, we still emit a
    minimal stub so the page advertises that the analysis exists, but
    the per-word breakdown is replaced with a teaser. This matches the
    posture in generate_faq_schema — don't hand the parses to scrapers
    via JSON-LD when the flag is on.
    """
    if not role_groups:
        return ""

    clue_text = clue.get("clue_text", "")
    enum = clue.get("enumeration", "")
    clue_display = clue_text + (f" ({enum})" if enum else "")
    name = f"Wordplay analysis for the cryptic crossword clue {clue_display!r}"

    strip_def = bool(current_app.config.get("STRIP_DEFINITION_FROM_JSONLD", False))
    if strip_def:
        # Emit a presence marker without the actual breakdown.
        schema = {
            "@context": "https://schema.org",
            "@type": "DefinedTermSet",
            "name": name,
            "description": (
                "Cordelia analyses every word in the clue and records its role "
                "(definition, indicator, letters-producing piece). Visit the "
                "page for the full word-by-word breakdown."
            ),
        }
        return json.dumps(schema, ensure_ascii=False)

    terms = []
    for idx, grp in enumerate(role_groups, start=1):
        words = " ".join(grp.get("words") or []).strip()
        if not words:
            continue
        role = (grp.get("role") or "").replace("_", " ")
        letters = grp.get("letters")
        if letters:
            description = f"{role} → {letters}"
        else:
            description = role
        terms.append({
            "@type": "DefinedTerm",
            "termCode": words.lower(),
            "name": words,
            "description": description,
            "inDefinedTermSet": "https://justcordelia.com/learn",
        })

    description_text = (
        "Every word in the clue is accounted for. Each piece has been "
        "verified against Cordelia's reference database to ensure the "
        "wordplay produces the answer."
    )
    if mechanism_label:
        description_text = (
            f"Wordplay type: {mechanism_label}. " + description_text
        )

    schema = {
        "@context": "https://schema.org",
        "@type": "DefinedTermSet",
        "name": name,
        "description": description_text,
        "hasDefinedTerm": terms,
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


# ---------------------------------------------------------------------------
# Puzzle page schemas
# ---------------------------------------------------------------------------

def generate_puzzle_breadcrumb_schema(source, type_slug, type_label, puzzle_number):
    """BreadcrumbList JSON-LD for a puzzle page: Home > Source Type > #N."""
    base = "https://justcordelia.com"
    schema = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {"@type": "ListItem", "position": 1, "name": "Home", "item": f"{base}/"},
            {"@type": "ListItem", "position": 2, "name": f"{source.title()} {type_label}",
             "item": f"{base}/{source}/{type_slug}/"},
            {"@type": "ListItem", "position": 3, "name": f"#{puzzle_number}"},
        ],
    }
    return json.dumps(schema, ensure_ascii=False)


def generate_puzzle_faq_schema(source, type_label, puzzle_number, clue_count, publication_date):
    """FAQPage JSON-LD for a puzzle page."""
    source_display = source.title().replace("Dailymail", "Daily Mail")
    puzzle_display = f"{source_display} {type_label} #{puzzle_number}"

    entries = []

    # Q1: What are the answers?
    a1 = f"Cordelia has answers, hints, and step-by-step explanations for all {clue_count} clues in {puzzle_display}."
    if publication_date:
        a1 += f" Published {publication_date}."
    entries.append({
        "@type": "Question",
        "name": f"What are the answers to {puzzle_display}?",
        "acceptedAnswer": {"@type": "Answer", "text": a1},
    })

    # Q2: When was it published?
    if publication_date:
        entries.append({
            "@type": "Question",
            "name": f"When was {puzzle_display} published?",
            "acceptedAnswer": {"@type": "Answer", "text": f"{puzzle_display} was published on {publication_date}."},
        })

    schema = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": entries,
    }
    return json.dumps(schema, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Learn page schemas
# ---------------------------------------------------------------------------

def generate_learn_faq_schema():
    """FAQPage JSON-LD for the learn index page."""
    entries = [
        {
            "@type": "Question",
            "name": "How do cryptic crosswords work?",
            "acceptedAnswer": {"@type": "Answer", "text": (
                "Every cryptic clue has two parts: a straight definition (always at the start or end) "
                "and wordplay instructions that build the answer from pieces. You get two routes to the "
                "same answer — that's not harder than a regular crossword, it's easier."
            )},
        },
        {
            "@type": "Question",
            "name": "What are the types of cryptic crossword clue?",
            "acceptedAnswer": {"@type": "Answer", "text": (
                "The main types are: anagram (letters rearranged), charade (pieces joined end to end), "
                "container (one word inside another), hidden word (answer hiding in the clue text), "
                "reversal (word spelled backwards), double definition (two meanings, one answer), "
                "homophone (sounds like another word), deletion (letters removed), "
                "acrostic (first letters spell the answer), and cryptic definition (the whole clue is a tricky definition)."
            )},
        },
        {
            "@type": "Question",
            "name": "Are cryptic crosswords hard?",
            "acceptedAnswer": {"@type": "Answer", "text": (
                "No — they're different, not harder. In a regular crossword you get one definition. "
                "In a cryptic you get a definition plus wordplay instructions. Once you learn to spot "
                "the common patterns, you have two ways to find every answer instead of one."
            )},
        },
    ]
    schema = {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": entries}
    return json.dumps(schema, ensure_ascii=False)


def generate_learn_breadcrumb_schema():
    """BreadcrumbList JSON-LD for the learn index page."""
    base = "https://justcordelia.com"
    schema = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {"@type": "ListItem", "position": 1, "name": "Home", "item": f"{base}/"},
            {"@type": "ListItem", "position": 2, "name": "Learn"},
        ],
    }
    return json.dumps(schema, ensure_ascii=False)


def generate_learn_type_faq_schema(label, short_desc, total):
    """FAQPage JSON-LD for a learn type page."""
    entries = [
        {
            "@type": "Question",
            "name": f"What is a {label.lower()} clue in a cryptic crossword?",
            "acceptedAnswer": {"@type": "Answer", "text": (
                f"{label}: {short_desc}. Cordelia has {total} example clues with "
                f"colour-coded visual breakdowns showing exactly how each one works."
            )},
        },
        {
            "@type": "Question",
            "name": f"How do I spot a {label.lower()} clue?",
            "acceptedAnswer": {"@type": "Answer", "text": (
                f"Look for indicator words in the clue that signal {label.lower()} wordplay. "
                f"Visit the page for real examples with visual breakdowns — you'll start spotting "
                f"the pattern after just a few."
            )},
        },
    ]
    schema = {"@context": "https://schema.org", "@type": "FAQPage", "mainEntity": entries}
    return json.dumps(schema, ensure_ascii=False)


def generate_learn_type_breadcrumb_schema(label):
    """BreadcrumbList JSON-LD for a learn type page."""
    base = "https://justcordelia.com"
    schema = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": [
            {"@type": "ListItem", "position": 1, "name": "Home", "item": f"{base}/"},
            {"@type": "ListItem", "position": 2, "name": "Learn", "item": f"{base}/learn"},
            {"@type": "ListItem", "position": 3, "name": label},
        ],
    }
    return json.dumps(schema, ensure_ascii=False)
