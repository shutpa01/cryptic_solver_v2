"""Double-definition engine — built on the shared character atomiser.

A double definition (DD) is two separate definitions of the same answer placed
side by side, with no wordplay: "Bound | spring" -> LEAP. This engine finds the
split where BOTH halves define the answer, records each half as a definition of
the whole answer (span-level provenance — DD has no per-letter sourcing), and
verifies itself. Both halves are always shown.

Grounded design (measured 2026-06-01): the clue's grammar shape is a strong DD
signal but can't stand alone at DD's ~7% prevalence, so it is used as a recall
gate, and a narrow Haiku check carries the precision:

  1. Find a split (left | optional link words | right) covering ALL clue words.
  2. If BOTH halves are confirmed by the DB -> a double definition (no AI).
  3. Else, if exactly one half is DB-confirmed AND the clue is DD-shaped (no
     operative wordplay indicator in the link region between the halves — an
     indicator inside a definition like "in" in "fat in pastry" does not count),
     ask Haiku whether the OTHER half defines
     the answer. If yes -> a double definition; the AI-confirmed half is flagged
     provisional and queued for enrichment so next time it is a pure DB solve.

Pure: takes a wfw_atoms context + injected predicates, returns a wfw_model.Parse
or None. Paired with core/dd_screen.py.

Injected, DB-decoupled:
  defines(phrase, answer_letters) -> bool
  is_link(word) -> bool                                     (optional)
  indicator_types(word) -> set of wordplay-type strings     (optional)
  ai_is_definition(phrase, answer_letters) -> bool          (optional)
"""

from core.wfw_model import Source, Annotation, Parse

MAX_AI_CALLS = 2       # cap Haiku spend per clue

# Words that join the two halves of a double definition but are not general link words
# (so they are NOT added to the link table, which would shift the definition stage). Used
# ONLY here, in the DD gap between the two halves: "criticism, LIKE soldiers..." Both
# halves must still DB-define and DD runs last, so this cannot fabricate a solve.
_DD_CONNECTORS = {"like", "as"}


def _dd_gap_ok(token, is_link):
    t = (token.text or "").strip().lower()
    return t in _DD_CONNECTORS or bool(is_link and is_link(token.text))


def _answer_letters(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _link_region_has_indicator(mid, indicator_types):
    """True if a word in the LINK region between the two halves is a known
    wordplay indicator — the only place an OPERATIVE indicator can sit in a DD.

    An indicator word swallowed inside a definition half (e.g. "in" inside "fat in
    pastry") is operating on nothing; it is just a function word in the definition.
    Only an indicator sitting between the halves signals an actual operation, so
    that is the sole disqualifier. Checking the whole clue (as before) wrongly
    blocked every DD whose definition merely contained a common word like "in"."""
    if indicator_types is None:
        return False
    # PHRASE-AWARE (2026-07-08): an operative indicator stored as a multi-word DB row
    # ("put up") sitting between the halves disqualifies the DD too — checking words
    # one at a time could not see it. NOTE: this is a DISQUALIFIER, so phrase-awareness
    # makes DD stricter; the A/B run judges the effect.
    n = len(mid)
    for L in range(1, min(4, n) + 1):
        for i in range(n - L + 1):
            phrase = " ".join(t.text for t in mid[i:i + L])
            try:
                if indicator_types(phrase):
                    return True
            except Exception:
                pass
    return False


def _half_defines(tokens, answer, defines, is_dbe):
    """Does this token span define the answer — directly, or with a single
    definition-by-example indicator ("possibly", "say", "perhaps") stripped from its
    leading or trailing edge? Returns (ok, def_tokens, dbe_tokens): the tokens that
    form the actual definition, and the DBE indicator token(s) peeled off (empty when
    none).

    The DIRECT match is tried FIRST, so a half that already defines is never
    re-interpreted — existing DD behaviour is unchanged. This only adds a path for a
    half whose definition is by example, e.g. SIDE = "team" + "Possibly left" ("left"
    is possibly a SIDE). A leading OR trailing single DBE word is handled (multi-word
    DBE phrases like "for example" are a later extension)."""
    phrase = " ".join(t.text for t in tokens)
    if defines(phrase, answer):
        return True, tokens, []
    if is_dbe is None or len(tokens) < 2:
        return False, tokens, []
    if is_dbe(tokens[0].text):                          # "Possibly left" -> "left"
        rest = tokens[1:]
        if defines(" ".join(t.text for t in rest), answer):
            return True, rest, [tokens[0]]
    if is_dbe(tokens[-1].text):                         # "left say" -> "left"
        rest = tokens[:-1]
        if defines(" ".join(t.text for t in rest), answer):
            return True, rest, [tokens[-1]]
    return False, tokens, []


def solve_dd(ctx, defines, is_link=None, indicator_types=None,
             ai_is_definition=None, is_dbe=None):
    words = [t for t in ctx.clue_tokens if t.kind == "word"]
    n = len(words)
    if n < 2:
        return None
    answer = _answer_letters(ctx)
    if not answer:
        return None

    # All full-coverage splits: left = words[:i], optional link words, right. Each
    # half "defines" either directly OR via a definition-by-example indicator stripped
    # from its edge (so "Possibly left" counts as a definition, with "Possibly" a DBE
    # marker and "left" the definition).
    both = []        # (mid, l_def, l_dbe, r_def, r_dbe) where both halves define
    one = []         # (mid, db_side, other_toks, def_toks, dbe_toks) one defines
    for i in range(1, n):
        left = words[:i]
        l_ok, l_def, l_dbe = _half_defines(left, answer, defines, is_dbe)
        for j in range(i, n):
            mid = words[i:j]
            if mid and not all(_dd_gap_ok(t, is_link) for t in mid):
                continue                      # gap must be link words / DD connectors only
            right = words[j:]
            if not right:
                continue
            r_ok, r_def, r_dbe = _half_defines(right, answer, defines, is_dbe)
            if l_ok and r_ok:
                both.append((mid, l_def, l_dbe, r_def, r_dbe))
            elif l_ok != r_ok:
                if l_ok:
                    one.append((mid, "left", right, l_def, l_dbe))
                else:
                    one.append((mid, "right", left, r_def, r_dbe))

    if both:
        # Prefer the cleanest: fewest link words, then the most balanced split.
        mid, l_def, l_dbe, r_def, r_dbe = min(
            both, key=lambda c: (len(c[0]), abs(len(c[1]) - len(c[3]))))
        return _build(ctx, mid, (l_def, l_dbe, "db"), (r_def, r_dbe, "db"))

    # One half confirmed. Keep only DD-shaped candidates: those with no operative
    # wordplay indicator in the link region between the halves. If none remain the
    # clue is not a plain DD — abstain (None) and let the cascade try other engines.
    if not one:
        return None
    shaped = [c for c in one
              if not _link_region_has_indicator(c[0], indicator_types)]
    if not shaped:
        return None
    # Try the most promising candidates (fewest link words, longest other half),
    # asking Haiku about the unconfirmed half. Cap the spend.
    shaped.sort(key=lambda c: (len(c[0]), -len(c[2])))
    if ai_is_definition is None:
        return None                     # cannot consult Haiku -> make no claim
    asked = set()
    calls = 0
    considered = False                  # got at least one definite YES/NO back
    for mid, db_side, other_toks, def_toks, dbe_toks in shaped:
        phrase = " ".join(t.text for t in other_toks)
        if phrase in asked:
            continue
        if calls >= MAX_AI_CALLS:
            break
        asked.add(phrase)
        calls += 1
        verdict = ai_is_definition(phrase, answer)   # True | False | None
        if verdict:
            confirmed = (def_toks, dbe_toks, "db")
            other = (other_toks, [], "pending")
            lhalf, rhalf = ((confirmed, other) if db_side == "left"
                            else (other, confirmed))
            return _build(ctx, mid, lhalf, rhalf)
        if verdict is False:
            considered = True           # a definite NO -> enrichment was considered
        # verdict is None -> the check could not be made; not a considered verdict.
    # FAIL only when we genuinely CONSIDERED enrichment (a definite NO): one real
    # definition, no confirmable second. If every check errored / was unknown we
    # never considered it, so we ABSTAIN (None) rather than persist a durable FAIL
    # on a verdict we never obtained — a later run resolves it.
    if considered:
        mid, db_side, other_toks, def_toks, dbe_toks = shaped[0]
        return _build_one_def_fail(ctx, mid, def_toks, dbe_toks)
    return None


def _dbe_annotations(dbe_tokens):
    """One 'definition by example' indicator annotation per stripped DBE word, so the
    word is accounted (unexplained_words) and rendered as a By-example marker, exactly
    like the wordplay path's dbe_annotation."""
    return [Annotation(clue_atom_ids=t.atom_ids, text=t.text, role="indicator",
                       note="definition by example") for t in dbe_tokens]


def _build_one_def_fail(ctx, mid, def_toks, dbe_toks):
    """Build the FAIL parse for a DD-shaped clue where exactly one half is a confirmed
    definition and the other cannot be confirmed. Record the one real definition (and
    any DBE marker it carried), mark joining words, and leave the unconfirmable half
    unaccounted so _verify_dd surfaces it honestly as a FAIL."""
    src = Source(clue_atom_ids=tuple(a for t in def_toks for a in t.atom_ids),
                 text=" ".join(t.text for t in def_toks), value=ctx.answer_text,
                 mechanism="definition", source="db")
    annotations = [Annotation(clue_atom_ids=t.atom_ids, text=t.text,
                              role="link", note="link word") for t in mid]
    annotations += _dbe_annotations(dbe_toks)
    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=[src], links=[], annotations=annotations,
                  definition=None, operation="double_definition", solved_by="dd")
    _verify_dd(ctx, parse)
    return parse


def _build(ctx, mid, lhalf, rhalf):
    """Build a DD parse. Each half is (def_tokens, dbe_tokens, flag): the definition
    Source covers def_tokens only; any DBE marker peeled from the half is recorded as a
    by-example annotation so it is accounted and labelled."""
    def _src(half):
        def_toks, _dbe, flag = half
        return Source(clue_atom_ids=tuple(a for t in def_toks for a in t.atom_ids),
                      text=" ".join(t.text for t in def_toks), value=ctx.answer_text,
                      mechanism="definition", source=flag)
    annotations = [Annotation(clue_atom_ids=t.atom_ids, text=t.text,
                              role="link", note="link word") for t in mid]
    annotations += _dbe_annotations(lhalf[1]) + _dbe_annotations(rhalf[1])
    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=[_src(lhalf), _src(rhalf)],
                  links=[], annotations=annotations, definition=None,
                  operation="double_definition", solved_by="dd")
    _verify_dd(ctx, parse)
    return parse


def _verify_dd(ctx, parse):
    """Engine-level verification — rules SPECIFIC to double definition.

    Three-state verdict (2026-06-02 cascade rules):
      - 'pass'    both halves DB-confirmed, exactly two non-overlapping
                  definitions, every clue word accounted for.
      - 'pending' structurally a clean DD but one half is provisional
                  (AI-confirmed, queued as a definition for enrichment) — the
                  queueable case; accepting it makes the clue a pure-DB pass.
      - 'fail'    a structural problem (not two clean halves / overlap / words
                  unaccounted): substantial evidence (one real definition) but no
                  confirmable second, not reachable by enrichment.

    A half defining the answer is guaranteed by construction (DB- or AI-confirmed);
    the source flag records which. No score."""
    warnings = []
    defs = [s for s in parse.sources if s.mechanism == "definition"]
    if len(defs) != 2:
        warnings.append("a double definition needs exactly two definitions")

    missing = parse.unexplained_words(ctx)
    if missing:
        warnings.append("these clue words are unaccounted for: "
                        + ", ".join(repr(m) for m in missing))

    ids = [set(s.clue_atom_ids) for s in defs]
    if len(ids) == 2 and ids[0] & ids[1]:
        warnings.append("the two definitions overlap")

    from core import role_validity
    warnings += role_validity.unbacked_roles(parse)   # links/indicators must be DB-backed

    if warnings:
        parse.warnings = warnings
        parse.status = "fail"
        return
    # Structurally a clean DD. A provisional (AI-confirmed, queued) half -> pending;
    # both halves DB-confirmed -> pass.
    if any(getattr(s, "source", "db") == "pending" for s in defs):
        parse.warnings = ["one definition is provisional (queued for enrichment)"]
        parse.status = "pending"
    else:
        parse.warnings = []
        parse.status = "pass"
