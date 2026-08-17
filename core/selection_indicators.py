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
    # The alternation TYPE means "take alternate letters" by definition; its rows carry
    # no sub-type. Without these entries a word/phrase typed ONLY as alternation (e.g.
    # the phrase row 'oddly rejected') licensed nothing for the SEL engines — the gap
    # that rolled back TERMINAL's hand-solved signature (2026-07-07). Mirrors the
    # engines' own _ALT_IND_TYPES = {alternation, alternating, alternate}.
    ("alternation", ""):            "alternate",
    ("alternating", ""):            "alternate",
    ("alternate", ""):              "alternate",
    ("selection", "last_letter"):   "last",
    ("selection", "outside_letters"): "outer",
    # Clean canonical subtype set for the `selection` type — one subtype per rule, so a
    # selection indicator added via the clue page (admin_db.add_indicator) maps directly to
    # its rule. A selection with NO subtype (or an unknown one) has no rule and is a
    # fabrication, so the write layer rejects it; only these five are offered/accepted.
    ("selection", "first"):         "first",
    ("selection", "last"):          "last",
    ("selection", "outer"):         "outer",
    ("selection", "middle"):        "middle",
    ("selection", "alternate"):     "alternate",
    # NAMED positions ("second and third in Gleneagles"). Deliberately maps to NO rule:
    # the setter names the positions, so there is nothing for an engine to derive and the
    # combinations cannot be enumerated. Present as a KEY so the write layer accepts the
    # subtype, but valued None so `if rule:` in engine_registry.selection_rules skips it and
    # admin_db's backing check does not count it. A label the human can record, nothing more.
    ("selection", "named"):         None,
}

# The canonical selection subtypes (the keys of SUBTYPE_RULE under the `selection` type)
# that the clue-page Add panel offers and the write layer accepts. Order = display order.
CLUE_PAGE_SUBTYPES = ("first", "last", "outer", "middle", "alternate", "named")

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
