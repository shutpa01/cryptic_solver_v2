"""Per-mechanism backing — the verifier core.

Given one piece `(mechanism, fodder, yields)` decide whether `yields` is a LEGITIMATE
product of `fodder` under that mechanism, using the reference DB. This is the precision
gate of the atom-map model: a piece is accepted only if its mechanism genuinely backs
its letters — no DB value, no acceptance — which is what stops fabrication.

    backs(mechanism, fodder, yields, wiring) -> (ok: bool, detail: str|None)

`detail` records HOW it backed (e.g. 'synonym', 'anagram', 'deletion:behead') for later
provenance. Each mechanism has its own check because each derives letters differently:
synonym/abbreviation/literal come from the DB value; selection draws letters from the
fodder; anagram is the fodder's own letters re-ordered; homophone sounds like the fodder
(or a synonym); deletion is a DB value of the fodder with letters removed.
"""

from collections import Counter

from core import deletion

_SELECTION = {"hidden", "hidden_word", "hidden_in_word", "first_letter", "first letter",
              "initial", "initials", "last_letter", "selection", "acrostic",
              "alternation", "telescopic"}
_ANAGRAM = {"anagram", "anagram_fodder", "anag", "fodder"}
_HOMOPHONE = {"homophone", "homophones", "sounds_like", "sounds like", "sound"}
_DELETION = {"deletion", "delete", "subtraction", "removal", "minus"}
_LITERAL = {"literal", "raw", "letters", "literally"}

_DEL_OPS = tuple(deletion._OP_FUNCS.keys()) if hasattr(deletion, "_OP_FUNCS") else \
    ("behead", "curtail", "outer", "heartless", "empty")


def _letters(s):
    return "".join(ch for ch in (s or "").upper() if ch.isalpha())


def _is_subsequence(sub, whole):
    """Letters of `sub` appear in `whole` in order (selection from the fodder)."""
    it = iter(whole)
    return all(c in it for c in sub)


def _db_values(fodder, wiring):
    """Set of DB (synonym/abbreviation/literal) value strings for the fodder phrase."""
    out = set()
    try:
        for v, _ in wiring["lookup_all"](fodder):
            if v:
                out.add(_letters(v))
    except Exception:
        pass
    try:
        for s in wiring["synonyms_of"](fodder):       # phrase/inflection-aware synonyms
            if s:
                out.add(_letters(s))
    except Exception:
        pass
    return {v for v in out if v}


def _value_mech(fodder, yld, wiring):
    """Was yld a synonym or an abbreviation of fodder? Best-effort label."""
    try:
        for v, m in wiring["lookup_all"](fodder):
            if _letters(v) == yld:
                return m
    except Exception:
        pass
    return "synonym"


def _homophone_ok(yld, fodder, wiring):
    sounds = wiring.get("sounds_alike")
    if sounds is None:
        return False
    cands = [fodder]
    try:
        cands += list(wiring["synonyms_of"](fodder))
    except Exception:
        pass
    for cand in cands:
        try:
            if sounds(yld, cand) or sounds(cand, yld):
                return True
        except Exception:
            continue
    return False


def _deletion_ok(yld, fodder, wiring):
    """yld is a DB value of fodder with letters removed (positional op or one run)."""
    for base in _db_values(fodder, wiring):
        if len(base) <= len(yld):
            continue
        for op in _DEL_OPS:
            try:
                if deletion.apply_op(op, base) == yld:
                    return "deletion:%s" % op
            except Exception:
                pass
        try:
            if deletion.removed_runs(base, yld):
                return "deletion:run"
        except Exception:
            pass
    return None


def backs(mechanism, fodder, yields, wiring):
    """Return (ok, detail). See module docstring."""
    yld = _letters(yields)
    if not yld:
        return False, None
    mech = (mechanism or "").lower().strip()
    fl = _letters(fodder)

    if mech in _SELECTION:
        if yld in fl or _is_subsequence(yld, fl):
            return True, "selection"
        return False, None

    if mech in _ANAGRAM:
        if Counter(fl) == Counter(yld) and fl:
            return True, "anagram"
        return False, None

    if mech in _HOMOPHONE:
        return (True, "homophone") if _homophone_ok(yld, fodder, wiring) else (False, None)

    if mech in _DELETION:
        d = _deletion_ok(yld, fodder, wiring)
        return (True, d) if d else (False, None)

    if mech in _LITERAL:
        if yld == fl or yld in _db_values(fodder, wiring):
            return True, "literal"
        return False, None

    # synonym / abbreviation / unknown -> must be a DB value of the fodder
    if yld in _db_values(fodder, wiring):
        return True, _value_mech(fodder, yld, wiring)
    return False, None
