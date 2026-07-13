"""THE serving rule for the public site (week-only, no legacy — user decision
2026-07-13).

A clue is SERVED if and only if:
  1. its source is a served publication (telegraph, times, guardian —
     scope decision 2026-07-04), AND
  2. its stored WFW pass parse renders a card (engine pass, frozen manual
     solve, or a prefill the user confirmed — all human-verified by the
     publish-first process).

Everything else stays in the database as data but gets NO public page: the
clue URL answers 410 Gone, and the page returns at the SAME URL if the clue
is later solved (resurrection). Sitemaps, cross-links and browse pages must
apply the same rule so we never link to a 410.

The card itself is core/wfw_card.stored_card — the SAME renderer the solver
uses (user 2026-07-13: "base it on what we have"), imported lazily so the
site boots without core. The import chain is light (store/screens/render
only, no engine wiring); the /solver mount already set the precedent of core
inside the site process.
"""

import re

from flask import g

SERVED_SOURCES = ("telegraph", "times", "guardian")

# The (source, type_slug) pairs a PUBLIC visitor can browse from the home /
# puzzles pages (user 2026-07-13). Admin sees the full BROWSE_SOURCES list.
# Everyman stays visible pending the user's Everyman discussion.
SERVED_BROWSE = {
    ("telegraph", "cryptic"), ("telegraph", "prize"),
    ("times", "cryptic"), ("times", "sunday"),
    ("guardian", "cryptic"), ("guardian", "everyman"),
}

# Solver-workflow chips stripped from the PUBLIC card (user 2026-07-13): the
# verdict badge (PASS), the solving-engine tag (manual/dd/...) and the
# provenance chip a manual piece wears. They are review-surface furniture; a
# public page only exists BECAUSE the parse passed review, so they say nothing.
# These spans contain plain text only (no nesting) — the regex is safe.
_INTERNAL_CHIPS = re.compile(
    r'<span class="wfw-(?:engine|verdict|prov)[^"]*"[^>]*>.*?</span>')


def get_card(clue_id):
    """The rendered WFW card HTML for this clue, or None. Memoised per request."""
    cache = getattr(g, "_wfw_card_cache", None)
    if cache is None:
        cache = g._wfw_card_cache = {}
    if clue_id not in cache:
        try:
            from core.wfw_card import stored_card
            html = stored_card(clue_id)
            cache[clue_id] = (_INTERNAL_CHIPS.sub("", html)
                              if html is not None else None)
        except Exception:
            cache[clue_id] = None
    return cache[clue_id]


def card_css():
    """The card's embeddable stylesheet (no page-shell rules)."""
    from core.wfw_render import CARD_CSS
    return CARD_CSS


def is_served(source, clue_id):
    """True when this clue gets a public page — ONE truth for the clue route,
    the sitemap and every internal link."""
    if source not in SERVED_SOURCES:
        return False
    return get_card(clue_id) is not None
