"""Per-clue manual role overrides — the single shared point that applies a clue's human
FORCE decisions to the solver wiring, on EVERY solve path.

Three overrides, all stored per-clue in the store DB (never in the shared reference DBs):
  - FILLER          (wfw_filler)            -> those words count like a link, this clue only
  - FORCED DEFINITION (wfw_forced_def)      -> `defines` confirms ONLY the pinned edge phrase
  - FORCED INDICATOR  (wfw_forced_indicator) -> a span reports the pinned indicator TYPE,
                                                looked up as a PHRASE

`apply_forced_overrides(wiring, clue_id)` returns a shallow-wrapped copy of the wiring with
those predicates adjusted. CRITICAL: when a clue has NO overrides it returns the SAME wiring
object unchanged — so for the overwhelming majority of clues this is a strict no-op and the
auto-solver behaves exactly as before. The cached global wiring is never mutated.

This consolidates (and is the single source of truth for) what wfw_web previously did inline
via _filler_wiring / _forced_wiring, and closes the gap that the batch solver and the A/B
harness did not apply forces (so a forced clue could look like a fail on a non-page re-run).
"""

import re


def _norm_phrase(s):
    """Loose match key: lowercase, non-alphanumeric -> space, collapse whitespace.
    Mirrors wfw_web._norm_phrase so a typed phrase matches a clue edge/word."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", (s or "").lower()).split())


def load_overrides(clue_id):
    """(filler_set, forced_def_or_None, [(phrase, wptype), ...]) for the clue. Empty/None
    when the store is unavailable — overrides are best-effort, never fatal to a solve."""
    from core import store
    conn = store.connect()
    try:
        filler = store.get_clue_filler(conn, clue_id)
        forced_def = store.get_forced_definition(conn, clue_id)
        forced_ind = store.get_forced_indicators(conn, clue_id)
    except Exception:
        filler, forced_def, forced_ind = set(), None, []
    finally:
        conn.close()
    return filler, forced_def, forced_ind


def apply_forced_overrides(wiring, clue_id):
    """Wrap `wiring` with this clue's forced overrides. No-op (returns `wiring` unchanged)
    when the clue has none. Never mutates the input wiring."""
    if wiring is None or clue_id is None:
        return wiring
    try:
        filler, forced_def, forced_ind = load_overrides(clue_id)
    except Exception:
        return wiring
    if not filler and not forced_def and not forced_ind:
        return wiring                                  # strict no-op — identical behaviour

    w = dict(wiring)

    if filler:
        fil = {(x or "").strip().lower() for x in filler}
        orig_is_link = wiring["is_link"]

        def is_link(word, _fil=fil, _orig=orig_is_link):
            return (word or "").strip().lower() in _fil or bool(_orig(word))
        w["is_link"] = is_link

    if forced_def:
        target = _norm_phrase(forced_def)

        def defines(*a, _t=target):                    # called as defines(phrase, answer)
            return bool(a) and _norm_phrase(a[0]) == _t
        w["defines"] = defines

    if forced_ind:
        # A forced indicator's type may be a plain wordplay_type (e.g. 'reversal') OR a
        # positional one encoded 'charade_positional:after' — split off the subtype.
        norm_forced = []                       # (norm_phrase, base_type, subtype_or_None)
        for p, t in forced_ind:
            base, _, sub = (t or "").partition(":")
            norm_forced.append((_norm_phrase(p), base, sub or None))
        orig_types = wiring["indicator_types"]

        def indicator_types(text, _forced=norm_forced, _orig=orig_types):
            types = set(_orig(text) or ())
            key = _norm_phrase(text)
            for p, base, _sub in _forced:
                if key == p:
                    types.add(base)            # the base wordplay_type (DB-recognised)
            return types
        w["indicator_types"] = indicator_types

        # Positional indicators are read via charade_positional_subtypes, not
        # indicator_types — so a forced 'charade_positional:after' must also surface there.
        pos_forced = [(p, sub) for p, base, sub in norm_forced
                      if base == "charade_positional" and sub]
        if pos_forced and wiring.get("charade_positional_subtypes") is not None:
            orig_pos = wiring["charade_positional_subtypes"]

            def charade_positional_subtypes(text, _forced=pos_forced, _orig=orig_pos):
                subs = set(_orig(text) or ())
                key = _norm_phrase(text)
                for p, sub in _forced:
                    if key == p:
                        subs.add(sub)
                return subs
            w["charade_positional_subtypes"] = charade_positional_subtypes

    return w
