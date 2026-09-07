"""Clue-of-the-day NARRATION SCRIPT — text only, for the reel and the short.

Trial tool (2026-09-07). It prints what Cordelia would SAY for one clue. It writes
nothing, uploads nothing and changes no existing behaviour: read the output cold and
judge the formula before any of it reaches a voice.

    python scripts/narrate_clue.py --clue 10089910
    python scripts/narrate_clue.py --source telegraph --puzzle 31338
    python scripts/narrate_clue.py --source telegraph --puzzle 31338 --only-fails

THE RULE THIS IS BUILT ON. Every sentence comes from the STORED PARSE — the same
reading the website and the WFW card show (web.wfw_read._load). Nothing here re-derives
cryptic grammar. If the narrator worked it out for itself, the video and the site would
eventually disagree, and the video is the one you cannot correct after publishing.

THE FORMULA. A piece is narrated one of two ways, depending on where its letters come
from:

  * A LOOKUP (synonym, abbreviation, literal) is self-contained, so the clue word is the
    subject:            "Pressure gives us P."
  * A DERIVATION (selection, anagram, hidden, repetition) only exists because an
    indicator instructed it, so the INDICATOR is the subject and the word is its object:
                        "Unrestricted tells us to take the inside of SOME, giving OM."

That distinction is the whole design. Saying "some gives us OM" would assert that the
word means those letters, which it does not.

WHERE IT REFUSES. An honest blank is a result, not a failure:
  * a derivation whose indicator is not recorded — there is nothing truthful to say
    about why those letters were taken, so the clue is held back;
  * a cryptic definition, double definition or &lit — no enumerable wordplay to narrate;
  * an unparsed or unsolved clue.
Refusals print with a reason so the trial can count them.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The narration's own copy of nothing: every fact comes from these two.
from web import create_app                                        # noqa: E402
from web import wfw_read                                          # noqa: E402


# --- the fixed phrases ---------------------------------------------------------------

# Selection rules -> how a human says them. The rule list is closed (core.selection
# SPAN_RULES, mirrored in the /hs picker), so this map is complete by construction; an
# unknown rule falls through to a refusal rather than a guess.
_SELECTION_PHRASE = {
    "first":         "the first letter of",
    "last":          "the last letter of",
    "outer":         "the outer letters of",
    "middle":        "the middle of",
    "alternate":     "alternate letters of",
    "remove_first":  "all but the first letter of",
    "remove_last":   "all but the last letter of",
    "remove_outer":  "the inside of",
    "remove_middle": "the outside letters of",
    "named":         "the letters named in",
    "initial":       "the first letters of",
    "acrostic":      "the first letters of",
}

# Mechanisms that stand on their own — the clue word simply means those letters.
_LOOKUP_MECHS = {"synonym", "abbreviation", "raw", "literal", "replacement_letter",
                 "definition_by_example", "charade", "manual"}

# Mechanisms that NAME THEIR OWN RULE. The parse already records what was done, so these
# can be narrated whether or not an indicator is bound: we are reporting the recorded
# mechanism, not inventing a reason. The indicator is named as well when there is one.
_SELF_DESCRIBING = {
    "first_letter":   "the first letter of",
    "last_letter":    "the last letter of",
    "middle":         "the middle of",
    "outer":          "the outer letters of",
    "alternate":      "alternate letters of",
    "alternation":    "alternate letters of",
}

# The generic 'selection' mechanism does NOT name its rule — the rule lives on the
# indicator's subtype. No indicator, no rule, no honest sentence.
# Mechanism -> note words that mark an indicator as governing it.
_MECH_NOTE_KEYS = {
    "selection":      ("selection", "acrostic", "first-letter", "last-letter"),
    "anagram_fodder": ("anagram",),
    "hidden":         ("hidden",),
    "hidden_reversed": ("hidden",),
    "repetition":     ("repetition",),
    "deletion":       ("deletion", "deleted"),
    "deletion_removed": ("deletion", "deleted"),
    "homophone":      ("homophone",),
    "spoonerism":     ("spoonerism",),
    "first_letter":   ("selection", "acrostic", "first-letter"),
    "last_letter":    ("selection", "last-letter"),
    "middle":         ("selection",),
    "outer":          ("selection",),
    "alternate":      ("alternation", "alternat"),
    "alternation":    ("alternation", "alternat"),
}

# Operations with no piece-by-piece wordplay to narrate. Double definitions are NOT
# here: they have no pieces, but they DO have something to say (user, 2026-09-07 — "these
# are often nice clues"), and dd_script below says it without inventing anything.
_NO_WORDPLAY_OPS = {"cd", "andlit", "continuation", "reverse_anagram",
                    "double_homophone"}
_DD_OPS = {"dd", "double_definition"}

_NUMBER_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
                12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen"}

# What to say about an indicator that no piece bound to. Keyed on the indicator's own
# recorded type, so nothing here is inferred from the clue's wording.
_INDICATOR_PHRASE = {
    "container":  "%s is the container indicator — it tells us one piece goes inside another.",
    "insertion":  "%s is the container indicator — it tells us one piece goes inside another.",
    "reversal":   "%s tells us to read it backwards.",
    "hidden":     "%s tells us the answer is hiding in the clue's own words.",
    "deletion":   "%s tells us something is taken away.",
    "homophone":  "%s tells us to listen to it rather than read it.",
    "anagram":    "%s tells us the letters are jumbled.",
    "link":       "%s is just a link word.",
    "acrostic":   "%s tells us to take first letters.",
    "alternation": "%s tells us to take alternate letters.",
    "palindrome": "%s tells us it reads the same both ways.",
    "spoonerism": "%s tells us to swap the opening sounds.",
    "letter_shift": "%s tells us a letter moves.",
    "charade_positional": "%s tells us where the pieces sit relative to each other.",
    "repetition": "%s tells us to use a piece twice.",
}

INTRO_DIFFERENCE = (
    "Now, we do this differently from everyone else who explains a single clue. "
    "They start with the clue and hunt for the answer. We start with the answer and "
    "work backwards, so you can see exactly how the setter built it."
)

OUTRO = ("If any step of that is unclear, put it in the comments and I'll answer you "
         "there.")


# --- helpers -------------------------------------------------------------------------

def _subtype(note):
    """The indicator's sub-type, in either note shape the renderers produce:

        'selection/remove_outer indicator'   -> 'remove_outer'
        'selection indicator (outer)'        -> 'outer'

    Both are live — the second turned up in the first trial run and was read as having
    no rule at all. '' when there genuinely isn't one.
    """
    n = (note or "").lower()
    m = re.match(r"^\s*([a-z_]+)\s*/\s*([a-z_]+)", n)
    if m:
        return m.group(2)
    m = re.search(r"\(\s*([a-z_]+)\s*\)", n)
    return m.group(1) if m else ""


def _kind(note):
    """The indicator's own type word: 'selection/remove_outer indicator' -> 'selection'."""
    n = (note or "").lower()
    m = re.match(r"^\s*([a-z_]+)", n)
    return m.group(1) if m else ""


def _positions(parse, ord_):
    """Answer positions this source places, in answer order."""
    return sorted(l["answer_pos"] for l in parse["links"] if l["source_index"] == ord_)


def _landed(parse, ord_):
    """The letters this piece actually puts in the answer, in answer order.

    NOT the piece's value: an anagram fodder's value IS the fodder, so 'they make
    <value>' said the letters rearrange into themselves (ACCELERATING was narrated as
    making TRIAGECANCEL). What they make is what lands on the grid.
    """
    letters = "".join(c for c in (parse["answer_text"] or "").upper() if c.isalpha())
    return "".join(letters[p - 1] for p in _positions(parse, ord_)
                   if 1 <= p <= len(letters))


def _transform_words(src):
    """The piece's RECORDED transform in words ('reversed', 'less R'), or ''.

    Read from the parse, never derived. Without this a reversed piece is narrated as a
    plain lookup — AGNOSTIC was read as 'explicit song appropriate gives us CITSONGA'
    with no hint that it goes in backwards.
    """
    xf = wfw_read._xf_load(src.get("transform"))
    if not xf:
        return ""
    words = wfw_read._xf_words((src["value"] or "").upper(), xf) or ""
    # "less V less L" is the card's written shorthand and reads badly aloud. Reworded
    # HERE only: the card and the site keep their own vocabulary untouched, since this
    # is a speech nicety, not a difference of meaning.
    words = re.sub(r"less (\w+) less (\w+)", r"less \1 and \2", words)
    return words


def _is_split(pos):
    """True when a piece's letters are NOT one run — it wraps around something."""
    return bool(pos) and pos != list(range(pos[0], pos[0] + len(pos)))


def _find_indicator(parse, mech):
    """The indicator that licenses this mechanism, or None.

    Matched by TYPE, never by proximity. ONE indicator may govern MANY pieces — "leaders
    of trade unions talking" is a single instruction over three words — so this does NOT
    consume the indicator it returns. Several candidates of the same type means we cannot
    say which governs what, so we return None rather than pick one.
    """
    keys = _MECH_NOTE_KEYS.get(mech)
    if not keys:
        return None
    cands = [i for i in parse["indicators"]
             if any(k in (i["note"] or "").lower() for k in keys)]
    return cands[0] if len(cands) == 1 else None


def _join(items):
    """['A', 'B', 'C'] -> 'A, B and C' — spoken lists, not comma soup."""
    items = [i for i in items if i]
    if len(items) <= 1:
        return items[0] if items else ""
    return "%s and %s" % (", ".join(items[:-1]), items[-1])


def _container_indicator(parse):
    for i in parse["indicators"]:
        if _kind(i["note"]) in ("container", "insertion"):
            return i
    return None


# --- the narration -------------------------------------------------------------------

def dd_script(parse, answer):
    """The body for a double (or triple) definition, or (None, reason).

    A DD has no pieces to take apart, so the piece formula has nothing to work on — but
    the clue is not thin, it is two meanings sharing one spelling, and that IS the
    explanation. Everything said here comes from the parse: the definition halves and
    the answer. NO usage examples are invented — "I can't stick him" is a fact about
    English that nothing in our data holds, so it is not spoken. If those are wanted they
    have to be written and stored, never generated at build time and read out as fact.
    """
    # Gathered EXACTLY as web/wfw_read._definition does: a double definition stores its
    # second half as a SOURCE with mechanism 'definition', not in the definitions list.
    # Reading only one of the two places found no halves at all on the first attempt.
    halves = [(d["text"] or "").strip() for d in parse["definitions"]
              if (d["text"] or "").strip()]
    halves += [(s["text"] or "").strip() for s in parse["sources"]
               if s["mechanism"] == "definition" and (s["text"] or "").strip()
               and (s["text"] or "").strip() not in halves]
    if len(halves) < 2:
        return None, "double definition with fewer than two definition halves recorded"
    a = answer.upper()
    n = len(re.sub(r"[^A-Za-z]", "", answer))
    count = _NUMBER_WORD.get(n, str(n))
    kind = "double definition" if len(halves) == 2 else "triple definition"
    many = "two meanings" if len(halves) == 2 else "three meanings"

    lines = ["This one's a %s — there's no wordplay to unpick here, just %s sitting "
             "side by side." % (kind, many)]
    for h in halves:
        lines.append("“%s” gives you %s." % (h, a))
    lines.append("Same %s letters, %s completely different senses. That's the whole "
                 "trick, and it's why these are often the most satisfying clues in the "
                 "puzzle." % (count, len(halves) == 2 and "two" or "three"))
    return "\n".join(lines), None


def group_sentence(parse, mech, pieces, used_inds=None):
    """One sentence for every piece sharing a mechanism, or (None, reason).

    Pieces are grouped because ONE indicator commonly governs several words. Narrating
    them separately would repeat the instruction three times and, worse, would need to
    bind the indicator three times — the bug the first trial run exposed on TUT-TUT.
    """
    words = [(p["text"] or "").strip() for p in pieces]
    values = [(p["value"] or "").strip().upper() for p in pieces]
    ind = _find_indicator(parse, mech)
    if ind is not None and used_inds is not None:
        used_inds.add(id(ind))
    iword = ((ind["text"] or "").strip() if ind else "")
    lead = ("%s tells us to take" % iword.capitalize()) if ind else "We take"

    if mech in _LOOKUP_MECHS:
        out = []
        for p, w, v in zip(pieces, words, values):
            xfw = _transform_words(p)
            landed = _landed(parse, p["ord"])
            # Say what the transform PRODUCES. "gives us CITSONGA, reversed" is true but
            # leaves the listener to do the reversing in their head; the point of the
            # sentence is the letters that end up in the grid.
            if xfw and landed and landed != v:
                out.append("%s gives us %s — %s, that's %s."
                           % (w.capitalize(), v, xfw, landed))
            elif xfw:
                out.append("%s gives us %s, %s." % (w.capitalize(), v, xfw))
            else:
                out.append("%s gives us %s." % (w.capitalize(), v))
        return " ".join(out), None

    if mech in _SELF_DESCRIBING:
        phrase = _SELF_DESCRIBING[mech]
        return ("%s %s %s, giving us %s."
                % (lead, phrase, _join([w.upper() for w in words]),
                   _join(values))), None

    if mech == "selection":
        # The rule is the indicator's subtype; without one there is nothing to say.
        if ind is None:
            return None, ("selection over %s, but no single indicator in the clue is "
                          "recorded as governing it" % _join(words))
        rule = _subtype(ind["note"])
        phrase = _SELECTION_PHRASE.get(rule)
        if not phrase:
            return None, "selection rule %r has no spoken phrase" % (rule or "(none)")
        return ("%s tells us to take %s %s, giving us %s."
                % (iword.capitalize(), phrase, _join([w.upper() for w in words]),
                   _join(values))), None

    if mech == "anagram_fodder":
        # What the fodder MAKES is what it lands on in the answer — never its own value,
        # which is the fodder itself.
        made = "".join(_landed(parse, p["ord"]) for p in pieces)
        if ind:
            return ("%s tells us to rearrange the letters of %s, and they make %s."
                    % (iword.capitalize(), _join([w.upper() for w in words]), made)), None
        return ("The letters of %s rearrange to make %s."
                % (_join([w.upper() for w in words]), made)), None

    if mech in ("hidden", "hidden_reversed"):
        back = " — and it runs backwards" if mech == "hidden_reversed" else ""
        lead = ("%s tells us the answer is hidden inside" % iword.capitalize()) if ind \
            else "The answer is hidden inside"
        return ("%s %s%s, where we find %s."
                % (lead, _join([w.upper() for w in words]), back, _join(values))), None

    if mech in ("deletion", "deletion_removed"):
        return ("%s tells us to drop %s."
                % (iword.capitalize() if ind else "The clue", _join(values))), None

    if mech == "repetition":
        return ("%s tells us to use %s a second time."
                % (iword.capitalize() if ind else "The clue", _join(values))), None

    if mech == "homophone":
        return ("%s sounds like %s." % (_join([w.upper() for w in words]),
                                        _join(values))), None

    if mech == "spoonerism":
        return ("Swap the opening sounds of %s and you get %s."
                % (_join([w.upper() for w in words]), _join(values))), None

    return None, "mechanism %r has no narration rule" % mech


def narrate(parse, clue_text, answer, enumeration, paper_label):
    """The full spoken script, or (None, [reasons]) when the clue must be held back."""
    problems = []
    op = parse["operation"]
    if op in _NO_WORDPLAY_OPS:
        return None, ["%s has no piece-by-piece wordplay to narrate" % op]

    is_dd = op in _DD_OPS
    wordplay = [s for s in parse["sources"] if s["mechanism"] != "definition"]
    if not wordplay and not is_dd:
        return None, ["no wordplay pieces recorded"]

    lines = []
    lines.append("Today's clue of the day is from %s." % paper_label)
    lines.append("")
    lines.append("Here it is: %s%s"
                 % (clue_text.rstrip(". "), (" — %s" % enumeration) if enumeration else ""))
    lines.append("")
    lines.append(INTRO_DIFFERENCE)
    lines.append("")
    lines.append("The answer is %s." % answer.upper())
    lines.append("")

    if is_dd:
        # A DD has no "definition plus wordplay" split to announce — both halves ARE the
        # definition, so the usual line would be wrong here.
        body, why = dd_script(parse, answer)
        if why:
            return None, [why]
        lines.append(body)
        lines.append("")
        lines.append(OUTRO)
        return "\n".join(lines), []

    definition = wfw_read._definition(parse)
    if definition:
        lines.append("The definition is “%s”. Everything else is the wordplay."
                     % definition)
        lines.append("")

    # Group by mechanism, keeping the clue's own order of first appearance, so one
    # indicator governing several words becomes one instruction rather than three.
    groups, order = {}, []
    for src in wordplay:
        mech = src["mechanism"] or ""
        if mech not in groups:
            groups[mech] = []
            order.append(mech)
        groups[mech].append(src)

    used_inds = set()
    body = []
    for mech in order:
        sentence, why = group_sentence(parse, mech, groups[mech], used_inds)
        if why:
            problems.append(why)
            continue
        if not sentence:
            continue
        # A split piece does NOT sit in the answer as one run — it opens up and something
        # goes inside. Read as a plain list it would be a lie.
        if any(_is_split(_positions(parse, p["ord"])) for p in groups[mech]):
            sentence += (" That one is split — it opens up and another piece goes "
                         "inside it.")
        body.append(sentence)

    if problems:
        return None, problems
    lines.extend(body)
    lines.append("")

    # EVERY indicator the clue records gets said. One that no piece claimed was simply
    # dropped before, so AGNOSTIC lost both "part of" (hidden) and "on reflection"
    # (reversal) and read as though the words just meant CITSONGA.
    for ind in parse["indicators"]:
        if id(ind) in used_inds:
            continue
        word = (ind["text"] or "").strip()
        kind = _kind(ind["note"])
        phrase = _INDICATOR_PHRASE.get(kind)
        if phrase:
            lines.append(phrase % word.capitalize())

    lines.append("")
    lines.append("Put it together and you get %s." % answer.upper())
    lines.append("")
    lines.append(OUTRO)
    return "\n".join(lines), []


# --- driver --------------------------------------------------------------------------

def one(clue_id, verbose=True, only_fails=False):
    from web.db import get_db
    from web.models import classify_puzzle
    row = get_db().execute(
        "SELECT id, source, puzzle_number, publication_date, clue_text, answer, "
        "       enumeration, clue_number, direction FROM clues WHERE id = ?",
        (clue_id,)).fetchone()
    if row is None:
        print("  %s: no such clue" % clue_id)
        return False
    head = "%s %s %s%s" % (row["source"], row["puzzle_number"],
                           row["clue_number"], (row["direction"] or "")[:1])
    parse = wfw_read._load(clue_id)
    if parse is None:
        if verbose:
            print("REFUSED  %-24s %-10s  no stored parse" % (head, clue_id))
        return False

    # The PAPER's name, not the puzzle type: "Cryptic" alone is not a publication.
    # Reuse the uploader's map rather than writing a second one — its own docstring
    # says two maps drift and then the video and the page name the puzzle differently.
    slug, type_label = classify_puzzle(row["source"], row["puzzle_number"],
                                       row["publication_date"])
    try:
        import importlib.util as _iu
        _s = _iu.spec_from_file_location(
            "_yu", str(Path(__file__).resolve().parent / "youtube_upload.py"))
        _yu = _iu.module_from_spec(_s)
        _s.loader.exec_module(_yu)
        label = _yu.puzzle_reference(row["source"],
                                     {"number": row["puzzle_number"],
                                      "type_slug": slug, "type_label": type_label,
                                      "pub": row["publication_date"]})
    except Exception:
        label = "%s %s" % (row["source"].title(), type_label or "")
    script, problems = narrate(parse, row["clue_text"], row["answer"] or "",
                               row["enumeration"] or "", label or row["source"].title())
    if script is None:
        print("REFUSED  %-24s %-10s  %s" % (head, clue_id, "; ".join(problems)))
        return False
    if only_fails:
        return True
    print("=" * 78)
    print("%s   clue_id %s   [%s]" % (head, clue_id, wfw_read._wordplay_label(parse)))
    print("=" * 78)
    print(script)
    print()
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clue", type=int, help="one clue id")
    ap.add_argument("--source", help="paper, with --puzzle")
    ap.add_argument("--puzzle", help="puzzle number, with --source")
    ap.add_argument("--only-fails", action="store_true",
                    help="print only the refusals and a tally — the trial's real output")
    args = ap.parse_args(argv)

    app = create_app("development")
    with app.app_context():
        from web.db import get_db
        if args.clue:
            one(args.clue, only_fails=args.only_fails)
            return 0
        if not (args.source and args.puzzle):
            ap.error("give --clue, or --source and --puzzle")
        rows = get_db().execute(
            "SELECT id FROM clues WHERE source = ? AND puzzle_number = ? ORDER BY id",
            (args.source, args.puzzle)).fetchall()
        ok = sum(1 for r in rows if one(r["id"], only_fails=args.only_fails))
        print("-" * 78)
        print("%d of %d clues narrated; %d held back."
              % (ok, len(rows), len(rows) - ok))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
