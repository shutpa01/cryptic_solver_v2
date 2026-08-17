"""What happened to a piece's value on its way to the answer squares — RECORDED,
never re-derived (user rule 2026-08-17).

A wordplay piece stores a VALUE (the synonym/abbreviation/literal in full, e.g.
SUPER) and the answer tiles it fills (E, P, U, S). Until now nothing recorded the
steps between the two, so every renderer had to work them out again from the
letters — a guess that succeeded for one simple change (a deletion OR a reversal)
and fell silent on anything composed, printing "SUPER around HES -> EPHESUS" for
EPHESUS and "LEMON + E + TT + E -> OMELETTE" for OMELETTE. Silence read as "the
value landed unchanged", so the card asserted an assembly that does not spell the
answer.

This module is the vocabulary and the arithmetic. A transform is a small dict:

    {"cuts": [{"letters": "R", "at": 4}], "rev": true, "shift": null}

  cuts   letters removed from the value, applied LEFT TO RIGHT. Each cut names
         the run AND the position it was taken from, because "BALSA minus A" is
         ambiguous (BLSA or BALS) and an ambiguous record is just another guess.
         `at` is the 0-based start index in the letters as they stand when that
         cut is applied.
  rev    the surviving letters were laid on the tiles backwards.
  shift  a single-letter rotation of the survivor: 'last_front' | 'first_end'.

Order is fixed and total: cut(s), then shift, then reverse. Anagram, selection
and homophone are NOT here — those pieces already carry their mechanism, which
says what happened to them.

apply() is the authority: a recorded transform is accepted only when applying it
to the value reproduces the placed letters EXACTLY, so a transform can never
claim a change that does not spell the tiles (the same rule the selection role
has enforced since 2026-08-11).
"""

import json

_SHIFTS = ("last_front", "first_end")


def letters_only(text):
    """The comparable letters of a value or span — folds spaces, punctuation and
    case, so a multi-word value ('A TAD', "SET IN") compares like any other."""
    return "".join(c for c in (text or "").upper() if c.isalpha())


def empty(t):
    """True when this transform records no change at all."""
    if not t:
        return True
    return not (t.get("cuts") or t.get("rev") or t.get("shift"))


def cut(letters, at):
    """One recorded deletion: the run `letters` taken from index `at`."""
    return {"letters": letters_only(letters), "at": int(at)}


def make(cuts=(), rev=False, shift=None):
    """A transform dict from its parts, normalised. Returns {} for 'no change'.
    Each cut must be a {'letters', 'at'} mapping — see cut()."""
    out = []
    for c in (cuts or ()):
        letters = letters_only(c.get("letters"))
        if not letters:
            continue
        try:
            at = int(c.get("at"))
        except (TypeError, ValueError):
            continue                          # a cut with no position is not a record
        out.append({"letters": letters, "at": at})
    shift = shift if shift in _SHIFTS else None
    if not out and not rev and not shift:
        return {}
    return {"cuts": out, "rev": bool(rev), "shift": shift}


def apply(value, t):
    """The letters `value` places on the answer after transform `t`, or None when
    the transform cannot be applied (a cut whose run is not at the position
    recorded, or nothing left to place). Pure arithmetic — no search, no
    guessing: a cut is lifted from exactly where the record says it was."""
    v = letters_only(value)
    if not v:
        return None
    if empty(t):
        return v
    for c in (t.get("cuts") or ()):
        run, at = c.get("letters") or "", c.get("at")
        if not isinstance(at, int) or at < 0 or not run:
            return None
        if v[at:at + len(run)] != run:
            return None                       # not the run the record claims
        v = v[:at] + v[at + len(run):]
    if not v:
        return None                           # a piece must place something
    shift = t.get("shift")
    if shift == "last_front":
        v = v[-1] + v[:-1]
    elif shift == "first_end":
        v = v[1:] + v[0]
    if t.get("rev"):
        v = v[::-1]
    return v


def places(value, t, got):
    """True when the recorded transform really does turn `value` into `got`."""
    return apply(value, t) == letters_only(got)


_SHIFT_WORDS = {"last_front": "last letter to the front",
                "first_end": "first letter to the end"}


def _cut_words(v, c):
    """One cut in plain words, naming the END when it is taken from one — the way
    a solver says it ("SUPER without its last letter"), not a bare 'minus R'."""
    run, at = c.get("letters") or "", c.get("at")
    n = len(run)
    if at == 0:
        return ("without its first letter %s" % run if n == 1
                else "without its first %d letters %s" % (n, run))
    if isinstance(at, int) and at + n == len(v):
        return ("without its last letter %s" % run if n == 1
                else "without its last %d letters %s" % (n, run))
    return "minus %s" % run


def describe(value, t):
    """The transform as plain words, in the order it happens, or '' for no change.
    Needs the value so a cut can be named as an end ('without its last letter R')."""
    if empty(t):
        return ""
    v = letters_only(value)
    bits = []
    for c in (t.get("cuts") or ()):
        bits.append(_cut_words(v, c))
        run, at = c.get("letters") or "", c.get("at")
        if isinstance(at, int) and 0 <= at <= len(v):
            v = v[:at] + v[at + len(run):]
    if t.get("shift"):
        bits.append(_SHIFT_WORDS[t["shift"]])
    if t.get("rev"):
        bits.append("reversed")
    return ", ".join(bits)


def short(value, t):
    """The compact form for the assembly line — '&minus;R reversed' in words a
    monospace expression can carry, with no HTML of its own."""
    if empty(t):
        return ""
    bits = ["−%s" % (c.get("letters") or "")   # a real minus sign, so the card reads
            for c in (t.get("cuts") or ())]          #   the same as its older "&minus;X" rows
    if t.get("shift"):
        bits.append(_SHIFT_WORDS[t["shift"]])
    if t.get("rev"):
        bits.append("reversed")
    return " ".join(bits)


def dumps(t):
    """The transform as stored (a compact JSON string; '' when there is none)."""
    return "" if empty(t) else json.dumps(t, separators=(",", ":"), sort_keys=True)


def coerce(obj):
    """A transform from whatever the caller has — the dict an /hs payload carries,
    a stored JSON string, or nothing. Always normalised through make(), so a
    malformed record (a cut with no position) becomes {} rather than a half-truth."""
    if isinstance(obj, dict):
        return make(obj.get("cuts"), obj.get("rev"), obj.get("shift"))
    if isinstance(obj, str):
        return loads(obj)
    return {}


def loads(s):
    """A stored transform back into a dict; {} when absent or unreadable."""
    if not s:
        return {}
    try:
        t = json.loads(s)
    except (ValueError, TypeError):
        return {}
    if not isinstance(t, dict):
        return {}
    return make(t.get("cuts"), t.get("rev"), t.get("shift"))
