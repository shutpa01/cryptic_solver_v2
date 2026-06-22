"""DB-validity of a parse's recorded roles — the shared check each engine's own _verify
calls before it may return a PASS.

This is NOT a generic re-solver (that was the mistake last time). It does one thing: confirm
every role the engine RECORDED is DB-backed, never assigned by elimination. Each engine still
owns its _verify and its piece checks; they call unbacked_roles() for the part that is
identical everywhere — that an indicator is a real DB indicator OF THE RIGHT TYPE, and a link
is a real DB link word.

The DB predicates (indicator_types, is_link) are set ONCE per solve by engine_registry.solve
via set_predicates(); the per-engine _verify functions just call unbacked_roles(parse). If the
predicates were never set, unbacked_roles FAILS CLOSED (reports a violation) rather than
silently passing — a missing check must never look like a clean solve.
"""

_INDICATOR_TYPES = None
_IS_LINK = None

# Map a recorded indicator NOTE to the DB indicator type(s) it must have. A note that does
# not match (e.g. a selection/location indicator, or a bare "charade indicator") is checked
# only for being SOME DB indicator (non-empty) — the baseline db-backed test.
_NOTE_TYPES = (
    ("location", None),                       # location/selection word: any DB indicator
    ("anagram", {"anagram"}),
    ("container", {"container", "insertion"}),
    ("insertion", {"container", "insertion"}),
    ("deletion", {"deletion"}),
    ("reversal", {"reversal"}),
    ("homophone", {"homophone"}),
    ("hidden", {"hidden"}),
    ("alternation", {"alternation", "alternating"}),
    ("alternat", {"alternation", "alternating"}),
)


def set_predicates(indicator_types, is_link):
    """Install the DB predicates for the current solve (called by engine_registry.solve)."""
    global _INDICATOR_TYPES, _IS_LINK
    _INDICATOR_TYPES = indicator_types
    _IS_LINK = is_link


def _expected_types(note):
    n = (note or "").lower()
    for key, types in _NOTE_TYPES:
        if key in n:
            return types
    return None                               # unknown note -> any DB indicator


def unbacked_roles(parse):
    """Recorded roles in `parse` that are NOT DB-backed. Empty == every role is DB-backed.
    A non-empty result means the parse rests on a fabricated role and must not be a PASS.
    Fails closed if the predicates were not installed for this solve."""
    it, il = _INDICATOR_TYPES, _IS_LINK
    if it is None or il is None:
        return ["role-validity predicates not initialised (failing closed)"]
    bad = []
    for a in getattr(parse, "annotations", None) or []:
        role = getattr(a, "role", "")
        text = getattr(a, "text", "") or ""
        if role == "indicator":
            types = set(it(text) or ())
            want = _expected_types(getattr(a, "note", ""))
            if not types:
                bad.append("indicator %r is not a DB indicator" % text)
            elif want is not None and not (types & want):
                bad.append("indicator %r is not a DB %s indicator (it is %s)"
                           % (text, "/".join(sorted(want)), "/".join(sorted(types))))
        elif role == "link":
            if not il(text):
                bad.append("link word %r is not a DB link word" % text)
    return bad
