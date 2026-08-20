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
  swap   two letters of the survivor exchanged: {"i": 2, "j": 5}, 0-based
         positions. ONE exchange only — two letters trading places is the
         device ("tense exchanges with Romeo": METCURIO -> MERCUTIO). A record
         that swapped several pairs would be an anagram wearing a disguise, and
         anagram pieces carry their own mechanism.
  move   ONE letter lifted out and re-inserted elsewhere: {"from": 4, "to": 0},
         0-based positions, `to` read in the letters AFTER the lift. This is the
         other half of the named-letter device: an exchange keeps both letters
         where the other stood, a move relocates one and lets the rest close up
         ("moving miles to the west"). A swap of ADJACENT letters and a move by
         one place are the same thing; the swap wins, because "these two changed
         places" is the more specific claim.
  rev    the surviving letters were laid on the tiles backwards.
  shift  a single-letter rotation of the survivor: 'last_front' | 'first_end'.
         last_front/first_end are the two moves common enough to have names of
         their own; they are kept because older records use them.

Order is fixed and total: cut(s), then swap, then move, then shift, then
reverse. Anagram, selection and homophone are NOT here — those pieces already
carry their mechanism, which says what happened to them.

Swap and move are recorded by POSITION, not by letter, for the same reason a cut
carries `at`: naming "T and R" is ambiguous the moment the value repeats a
letter, and an ambiguous record is a guess.

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
    return not (t.get("cuts") or t.get("swap") or t.get("move")
                or t.get("rev") or t.get("shift"))


def cut(letters, at):
    """One recorded deletion: the run `letters` taken from index `at`."""
    return {"letters": letters_only(letters), "at": int(at)}


def swap(i, j):
    """One recorded exchange: the letters at positions `i` and `j` trade places."""
    return {"i": int(i), "j": int(j)}


def move(frm, to):
    """One recorded relocation: the letter at `frm` lifted out and re-inserted at
    `to` (read in the letters after the lift)."""
    return {"from": int(frm), "to": int(to)}


def _norm_move(mv):
    """A move normalised to {'from','to'}, or None when it is not a real
    relocation (missing positions, or a letter put back where it came from)."""
    if not isinstance(mv, dict):
        return None
    try:
        f, t = int(mv.get("from")), int(mv.get("to"))
    except (TypeError, ValueError):
        return None                           # a move with no positions is not a record
    if f < 0 or t < 0 or f == t:
        return None
    return {"from": f, "to": t}


def _norm_swap(s):
    """A swap normalised to {'i','j'} with i < j, or None when it is not a real
    exchange (missing/undecipherable positions, or a letter with itself)."""
    if not isinstance(s, dict):
        return None
    try:
        i, j = int(s.get("i")), int(s.get("j"))
    except (TypeError, ValueError):
        return None                           # a swap with no positions is not a record
    if i < 0 or j < 0 or i == j:
        return None
    return {"i": min(i, j), "j": max(i, j)}


def make(cuts=(), rev=False, shift=None, swap=None, move=None):
    """A transform dict from its parts, normalised. Returns {} for 'no change'.
    Each cut must be a {'letters', 'at'} mapping — see cut(); the swap a
    {'i', 'j'} mapping — see swap(); the move a {'from', 'to'} — see move()."""
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
    sw, mv = _norm_swap(swap), _norm_move(move)
    if not out and not rev and not shift and not sw and not mv:
        return {}
    return {"cuts": out, "rev": bool(rev), "shift": shift, "swap": sw, "move": mv}


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
    sw = t.get("swap")
    if sw:
        i, j = sw.get("i"), sw.get("j")
        if not isinstance(i, int) or not isinstance(j, int):
            return None
        if i >= len(v) or j >= len(v):
            return None                       # the record points outside the letters
        b = list(v)
        b[i], b[j] = b[j], b[i]
        v = "".join(b)
    mv = t.get("move")
    if mv:
        f, to = mv.get("from"), mv.get("to")
        if not isinstance(f, int) or not isinstance(to, int):
            return None
        if f >= len(v):
            return None                       # the record points outside the letters
        b = list(v)
        ch = b.pop(f)                         # lift the letter out, then put it back
        if to > len(b):                       #   `to` is read AFTER the lift
            return None
        b.insert(to, ch)
        v = "".join(b)
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


def _swap_words(v, sw):
    """One exchange in plain words, naming the two letters as a solver would
    ("with T and R exchanged"). `v` is the letters as they stand when the swap
    is applied, so the positions can be read back as letters."""
    i, j = sw.get("i"), sw.get("j")
    if (isinstance(i, int) and isinstance(j, int)
            and 0 <= i < len(v) and 0 <= j < len(v)):
        return "with %s and %s exchanged" % (v[i], v[j])
    return "with two letters exchanged"


def _move_words(v, mv):
    """One relocation in plain words, naming the LETTER and where it went ("with
    M moved to the front"). `v` is the letters as they stand when the move is
    applied, so the position can be read back as a letter."""
    f, to = mv.get("from"), mv.get("to")
    if not (isinstance(f, int) and isinstance(to, int) and 0 <= f < len(v)):
        return "with one letter moved"
    ch, last = v[f], len(v) - 1
    if to == 0:
        return "with %s moved to the front" % ch
    if to >= last:
        return "with %s moved to the end" % ch
    return "with %s moved %s" % (ch, "left" if to < f else "right")


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
    sw = t.get("swap")
    if sw:                                    # name the LETTERS — the positions are how the
        bits.append(_swap_words(v, sw))       #   record stays unambiguous, not how it reads
        i, j = sw.get("i"), sw.get("j")
        if isinstance(i, int) and isinstance(j, int) and i < len(v) and j < len(v):
            b = list(v); b[i], b[j] = b[j], b[i]; v = "".join(b)
    mv = t.get("move")
    if mv:
        bits.append(_move_words(v, mv))
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
    v = letters_only(value)                    # walk the cuts so the swapped/moved letters
    for c in (t.get("cuts") or ()):            #   can be NAMED, exactly as describe() does
        run, at = c.get("letters") or "", c.get("at")
        if isinstance(at, int) and 0 <= at <= len(v):
            v = v[:at] + v[at + len(run):]
    sw = t.get("swap")
    if sw:
        i, j = sw.get("i"), sw.get("j")
        if (isinstance(i, int) and isinstance(j, int)
                and 0 <= i < len(v) and 0 <= j < len(v)):
            bits.append("%s↔%s exchanged" % (v[i], v[j]))
            b = list(v); b[i], b[j] = b[j], b[i]; v = "".join(b)
        else:
            bits.append("two letters exchanged")
    mv = t.get("move")
    if mv:
        f, to = mv.get("from"), mv.get("to")
        if isinstance(f, int) and isinstance(to, int) and 0 <= f < len(v):
            where = ("to the front" if to == 0 else
                     "to the end" if to >= len(v) - 1 else
                     ("left" if to < f else "right"))
            bits.append("%s moved %s" % (v[f], where))
        else:
            bits.append("one letter moved")
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
        return make(obj.get("cuts"), obj.get("rev"), obj.get("shift"),
                    obj.get("swap"), obj.get("move"))
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
    return make(t.get("cuts"), t.get("rev"), t.get("shift"), t.get("swap"),
                t.get("move"))
