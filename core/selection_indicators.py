"""Selection indicators — which indicator words license which letter-selection rule.

A selection piece (core.selection.select_span) is only legitimate when an indicator
LICENSES the rule: "house originally" -> H is a first-letter selection ONLY because
"originally" is present and means "take the first letter". Without the indicator a
selection is a fabrication (any word yields a first/middle/alternate run), so the
charade/container engines must find a matching indicator before filling a SEL slot.

DB-DRIVEN (was a hardcoded word list — the same anti-pattern deletion/substitution had
before their conversion). The wiring registers a `rules` provider (see
engine_registry.selection_rules) that reads the indicators table; SUBTYPE_RULE maps a DB
(wordplay_type, subtype) to a selection rule. Add/curate selection indicators in the DB,
never here.
"""

# DB (wordplay_type, subtype) -> selection rule. Only SELECTION (keep) subtypes — the
# matching deletion subtypes (first_delete, last_delete, outer_delete, center_delete,
# tail_delete) are REMOVALS and deliberately excluded.
SUBTYPE_RULE = {
    ("acrostic", "initial"):        "first",
    ("acrostic", "first"):          "first",
    ("parts", "first_use"):         "first",
    ("parts", "last_use"):          "last",
    ("parts", "last"):              "last",
    ("parts", "outer_use"):         "outer",
    ("parts", "center_use"):        "middle",
    ("parts", "alternate"):         "alternate",
    ("parts", "even"):              "alternate",
    ("parts", "odd"):               "alternate",
    ("selection", "last_letter"):   "last",
    ("selection", "outside_letters"): "outer",
}

# Provider set by the wiring: rules_for(text) -> set of selection rules the DB licenses
# for that word/phrase (inflection-aware). None until wired (find_indicators then empty).
_RULES_PROVIDER = None


def set_rules_provider(fn):
    global _RULES_PROVIDER
    _RULES_PROVIDER = fn


def find_indicators(words, max_run=4):
    """[(rule, (idx, ...)), ...] — every selection indicator in `words` (token objects
    with .text), each with the word indices it occupies. Contiguous runs (phrases) and
    single words are matched against the DB-driven provider. Empty when none / unwired."""
    provider = _RULES_PROVIDER
    if provider is None:
        return []
    texts = [t.text for t in words]
    n = len(texts)
    out, seen = [], set()
    for L in range(min(max_run, n), 0, -1):           # longer phrases first
        for i in range(n - L + 1):
            phrase = " ".join(texts[i:i + L])
            for rule in (provider(phrase) or ()):
                key = (rule, i, i + L)
                if key not in seen:
                    seen.add(key)
                    out.append((rule, tuple(range(i, i + L))))
    return out
