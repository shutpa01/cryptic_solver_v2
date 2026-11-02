"""Serve gate: never serve a clue whose pieces, AS THE PLAYER SEES THEM, cannot
make the answer.

The play page checks the player's build against each atom's TRUE letters, but it
shows the DISPLAY letters (tileText in play_template.html). If those differ — a
jumbled charade piece, "ANNE" shown as "ANEN" (2026-07-23 slot 2, ANNEAL) — the
player is asked to spell the answer from pieces that visibly can't spell it.

Same idea as the main solver's tile gate (core/wfw_web.py `_build_manual_parse`:
"a piece must spell the tiles it lands on"): what is placed must make the answer.

Each check mirrors the page's own code path for that board, so a pass here means
the page's own check can be satisfied by tapping what is on screen.

    check(payload) -> None if servable, else a reason string.
"""
from collections import Counter


def _answer_letters(p):                       # play_template.html: answerLetters
    return "".join(c for c in (p.get("answer") or "").upper() if "A" <= c <= "Z")


def _shown(p, a):                             # play_template.html: tileText
    if p.get("mode") == "easy" and a.get("display"):
        return a["display"]
    return a.get("true") or ""


def _first_alpha(s):                          # play_template.html: firstAlpha
    for c in (s or "").upper():
        if "A" <= c <= "Z":
            return c
    return ""


def _check_assembly(p, ans):
    atoms = p.get("atoms") or []
    if not atoms:
        return "no pieces"
    order = p.get("assembly_order") or []
    if sorted(order) != list(range(len(atoms))):
        return "assembly_order %r is not a permutation of the %d pieces" % (order, len(atoms))
    shown = "".join(_shown(p, atoms[i]) for i in order)
    if shown != ans:
        pieces = " ".join(_shown(p, a) for a in atoms)
        return ("pieces as shown (%s) placed in the answer order make %s, not %s"
                % (pieces, shown, ans))
    true = "".join(atoms[i].get("true") or "" for i in order)
    if true != ans:                           # the page checks .true — must agree too
        return "true letters in answer order make %s, not %s" % (true, ans)
    return None


def _check_anagram(p, ans):
    atoms = p.get("atoms") or []
    shown = [_shown(p, a) for a in atoms]     # anagram tray shows a.true; same here
    if any(len(s) != 1 for s in shown):
        return "anagram board has a multi-letter piece (%s)" % " ".join(shown)
    if Counter(shown) != Counter(ans):
        over = Counter(shown) - Counter(ans)
        short = Counter(ans) - Counter(shown)
        return ("anagram letters %s cannot make %s (extra %s, missing %s)"
                % ("".join(shown), ans, "".join(sorted(over.elements())) or "-",
                   "".join(sorted(short.elements())) or "-"))
    for an in p.get("anchors") or []:
        pos, ai = an.get("pos"), an.get("atom_index")
        if not (0 <= pos < len(ans)) or not (0 <= ai < len(atoms)):
            return "anchor %r out of range" % (an,)
        if atoms[ai].get("true") != ans[pos]:
            return ("anchor locks %s at position %d, answer has %s"
                    % (atoms[ai].get("true"), pos + 1, ans[pos]))
    return None


def _check_hidden(p, ans):
    h = p.get("hidden") or {}
    target = h.get("target") or ""
    if target not in (ans, ans[::-1]):
        return "hidden target %s is not the answer %s (or its reversal)" % (target, ans)
    letters = "".join(L.get("ch") or "" for L in h.get("letters") or [])
    if target not in letters:
        return "hidden run %s is not in the shown letters %s" % (target, letters)
    return None


def _check_acrostic(p, ans):
    ac = p.get("acrostic") or {}
    target = ac.get("target") or ""
    if target != ans:
        return "acrostic target %s is not the answer %s" % (target, ans)
    initials = "".join(_first_alpha(W.get("text")) for W in ac.get("words") or [])
    if target not in initials:
        return "no run of shown words has initials %s (initials %s)" % (target, initials)
    return None


def check(p):
    ans = _answer_letters(p)
    if not ans:
        return "no answer"
    board = p.get("board")
    if board == "hidden":
        return _check_hidden(p, ans)
    if board == "acrostic":
        return _check_acrostic(p, ans)
    if p.get("mode") == "anagram":
        return _check_anagram(p, ans)
    return _check_assembly(p, ans)
