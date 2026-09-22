"""SEO helpers for individual clue pages — meta descriptions, JSON-LD schemas."""

import json
import re

from flask import current_app


# Bing showed a 69-character rival title WHOLE on the SERP for "Troubled medico
# cares about a fattening drink?" (measured 2026-09-17) - three of them, in the
# top three places: "<clue> (3-5,4) Crossword Clue". 69 is therefore known to be
# displayable, and it is the budget we spend. Our own 66-character title came
# back as 53 characters plus " ..." on that same page; the one thing it had that
# no rival title had was an em dash separator, so the separator is gone (see
# generate_title) and our selling word is SHORTER than the rivals' "Crossword
# Clue" - 10 characters against 15.
TITLE_VISIBLE_CHARS = 69


def generate_title(clue, explained):
    """Build the clue-page title: the searcher's clue first, then the offer.

    The clue and enumeration lead, so the searcher sees at once that this is
    their clue (Bing bolds the words that match the query). The offer follows
    IN THE SAME LINE, because that is the whole point of the row: every rival
    result fits "<clue> (<enum>) Crossword Clue" into 69 characters and Bing
    shows all 69. Ours says Explained where theirs says Crossword Clue, and
    Explained is five characters shorter, so we fit what they fit and sell
    something they cannot.

    No em dash. Every rival title that displayed whole used a plain space; the
    one title on that page that Bing chopped early was ours, and the separator
    was the one thing that distinguished it. A separator also costs two
    characters that buy nothing.

    The offer is the longest rung of a ladder that still fits. The clue and
    enumeration are never shortened, and the offer is only dropped when even
    the shortest rung would push the line past the budget and break the
    enumeration open - which Bing renders as "(3-5,4 ...", the thing that
    makes us look broken next to results that are not.

    Args:
        clue: dict with clue_text, enumeration.
        explained: True when the served card is a PASS parse.
    """
    enum = clue.get("enumeration", "")
    head = clue.get("clue_text", "") + (f" ({enum})" if enum else "")
    if explained:
        ladder = (" Explained: every word, and the answer",
                  " Explained, with the answer",
                  " Explained")
    else:
        ladder = (" The answer, and why this clue fails",
                  " Why this clue fails",
                  " Why it fails")
    for offer in ladder:
        if len(head) + len(offer) <= TITLE_VISIBLE_CHARS:
            return head + offer
    return head


def generate_meta_description(clue, explained):
    """Build the meta description for a clue page.

    A clue page is served for exactly two cards (web/serving.get_card): a PASS
    parse, which is the full word-by-word explanation, or a reviewer INVALID
    verdict with a comment, which is the answer plus why the clue is unsound.
    The offer names what that card is. It is keyed on the served card, never on
    the legacy clues.definition / clues.wordplay_type columns, which the WFW
    system does not fill (a pass with both empty was described as "hints").

    Order is what the searcher needs: the clue and enumeration first (is this
    my clue?), the offer straight after, the puzzle last. Aims for <= 160
    chars; when over, the puzzle is dropped — never the clue, never the offer.

    Args:
        clue: dict with clue_text, enumeration, source, puzzle_number, type_label.
        explained: True when the served card is a PASS parse.
    """
    clue_text = clue.get("clue_text", "")
    enum = clue.get("enumeration", "")
    source = (clue.get("source") or "").title()
    type_label = clue.get("type_label") or ""
    puzzle_number = clue.get("puzzle_number", "")

    # Origin line
    origin = f"{source}"
    if type_label:
        origin += f" {type_label}"
    origin += f" #{puzzle_number}"

    # What the served card is
    if explained:
        offer = "full explanation: what each word does, why, and the answer."
    else:
        offer = "the answer, and why this clue breaks the cryptic rules."

    enum_part = f" ({enum})" if enum else ""
    head = f'"{clue_text}{enum_part}" — {offer}'

    full = f"{head} {origin}."
    return full if len(full) <= 160 else head


def generate_faq_schema(clue, steps, explained):
    """Build FAQPage JSON-LD schema for a clue.

    Each available hint step becomes a question/answer pair.
    Google shows FAQ rich results for pages with this markup.

    Args:
        clue: dict with clue_text, enumeration, answer, definition,
              wordplay_type, and explanation content.
        steps: list of step dicts from get_hint_steps.
        explained: True when the served card is a PASS parse (else INVALID).

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
    # What the served card is (see generate_meta_description): a PASS parse is the
    # full word-by-word explanation; an INVALID card is the answer plus why the
    # clue is unsound. Never "hints" — that is what the competition sells.
    if explained:
        more = ("Visit the page for the full word-by-word explanation: "
                "what every word in the clue does, and why.")
    else:
        more = "Visit the page to see why this clue doesn't work by the standard cryptic rules."

    strip_def = bool(current_app.config.get("STRIP_DEFINITION_FROM_JSONLD", False))
    if strip_def:
        meaning_parts = []
        if answer:
            meaning_parts.append(f"The answer is {answer}.")
        meaning_parts.append(more)
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
        meaning_parts.append(more)
        meaning_text = " ".join(meaning_parts)
    else:
        # No legacy definition/type (every WFW-only clue) — teaser only
        lead = ("This cryptic clue uses wordplay to arrive at the answer. " if explained
                else "")
        meaning_text = lead + more

    faq_entries.append({
        "@type": "Question",
        "name": f'What does "{clue_display}" mean?',
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


# Public, search-friendly names + common abbreviation per puzzle type, for SEO
# titles/descriptions. Keyed (source, puzzle_type). The name is how solvers
# actually search ("Sunday Times", not "Times Sunday"); the abbreviation is the
# short form people type ("DT 31205").
_PUZZLE_SEO_NAMES = {
    ("telegraph", "cryptic"):       ("Telegraph Cryptic Crossword", "DT"),
    ("telegraph", "prize"):         ("Telegraph Prize Cryptic Crossword", "DT"),
    ("telegraph", "prize-toughie"): ("Telegraph Prize Toughie Crossword", None),
    ("times", "cryptic"):           ("Times Cryptic Crossword", None),
    ("times", "sunday"):            ("Sunday Times Cryptic Crossword", None),
    ("guardian", "cryptic"):        ("Guardian Cryptic Crossword", None),
    ("guardian", "everyman"):       ("Everyman Crossword", None),
    ("independent", "cryptic"):     ("Independent Cryptic Crossword", None),
    ("dailymail", "cryptic"):       ("Daily Mail Cryptic Crossword", None),
    # Custom = one clue someone else wrote, admin-only and never indexed. Named
    # only so the page does not head itself "Custom Custom Crossword" from the
    # source+label fallback below.
    ("custom", "clues"):            ("Custom clue", None),
}


def puzzle_seo_name(source, puzzle_type, type_label):
    """Return (search_name, abbreviation) for a puzzle. Falls back to a sensible
    built name for any (source, type) not in the map."""
    name, abbr = _PUZZLE_SEO_NAMES.get((source, puzzle_type), (None, None))
    if name is None:
        src = source.title().replace("Dailymail", "Daily Mail")
        name = f"{src} {type_label} Crossword"
    return name, abbr


def generate_puzzle_title(source, puzzle_type, type_label, puzzle_number):
    """SEO <title> for a puzzle page. Where an abbreviation exists it LEADS
    ('DT 31205 — Telegraph Cryptic 31205: …'): it is the phrase solvers type,
    so it goes first, not in trailing brackets. The offer is the explanation,
    never "hints" — hints are what the competition sells."""
    name, abbr = puzzle_seo_name(source, puzzle_type, type_label)
    if abbr:
        short = name.removesuffix(" Crossword")
        return f"{abbr} {puzzle_number} — {short} {puzzle_number}: Answers & Explanations"
    return f"{name} {puzzle_number} — Answers & Explanations"


def generate_puzzle_heading(source, puzzle_type, type_label, puzzle_number):
    """Keyword-rich H1 for a puzzle page — the abbreviation form leads where one
    exists ('DT 31205 — Telegraph Cryptic Crossword')."""
    name, abbr = puzzle_seo_name(source, puzzle_type, type_label)
    if abbr:
        return f"{abbr} {puzzle_number} — {name}"
    return f"{name} {puzzle_number}"


def generate_puzzle_meta_description(source, puzzle_type, type_label,
                                     puzzle_number, clue_count, publication_date):
    """SEO meta description for a puzzle page."""
    name, abbr = puzzle_seo_name(source, puzzle_type, type_label)
    ref = f"{abbr} {puzzle_number} — {name}" if abbr else f"{name} {puzzle_number}"
    desc = (f"Answers and full word-by-word explanations for the clues in {ref}: "
            f"what every word does, and why.")
    if publication_date:
        desc += f" Published {publication_date}."
    return desc


def generate_puzzle_faq_schema(source, type_label, puzzle_number, clue_count,
                               publication_date, puzzle_type=None):
    """FAQPage JSON-LD for a puzzle page."""
    if puzzle_type is not None:
        name, abbr = puzzle_seo_name(source, puzzle_type, type_label)
        # The abbreviation form leads where one exists — it is what solvers type.
        puzzle_display = (f"{abbr} {puzzle_number} ({name})" if abbr
                          else f"{name} {puzzle_number}")
    else:
        source_display = source.title().replace("Dailymail", "Daily Mail")
        puzzle_display = f"{source_display} {type_label} #{puzzle_number}"

    entries = []

    # Q1: What are the answers?
    a1 = (f"Cordelia has the answers to all {clue_count} clues in {puzzle_display}, "
          f"with full word-by-word explanations: what every word does, and why.")
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
