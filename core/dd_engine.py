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
    for t in mid:
        try:
            if indicator_types(t.text):
                return True
        except Exception:
            pass
    return False


def solve_dd(ctx, defines, is_link=None, indicator_types=None,
             ai_is_definition=None):
    words = [t for t in ctx.clue_tokens if t.kind == "word"]
    n = len(words)
    if n < 2:
        return None
    answer = _answer_letters(ctx)
    if not answer:
        return None

    # All full-coverage splits: left = words[:i], optional link words, right.
    both = []        # (i, j, left, right, mid) where both halves DB-define
    one = []         # (i, j, left, right, mid, db_side) exactly one DB-defines
    for i in range(1, n):
        left = words[:i]
        lp = " ".join(t.text for t in left)
        l_db = defines(lp, answer)
        for j in range(i, n):
            mid = words[i:j]
            if mid and not all(is_link and is_link(t.text) for t in mid):
                continue                      # gap must be link words only
            right = words[j:]
            if not right:
                continue
            rp = " ".join(t.text for t in right)
            r_db = defines(rp, answer)
            if l_db and r_db:
                both.append((i, j, left, right, mid))
            elif l_db != r_db:
                one.append((i, j, left, right, mid, "left" if l_db else "right"))

    if both:
        # Prefer the cleanest: fewest link words, then the most balanced split.
        i, j, left, right, mid = min(
            both, key=lambda c: (len(c[4]), abs(len(c[2]) - len(c[3]))))
        return _build(ctx, answer, left, right, mid, "db", "db")

    # One half confirmed. Keep only DD-shaped candidates: those with no operative
    # wordplay indicator in the link region between the halves. If none remain the
    # clue is not a plain DD — abstain (None) and let the cascade try other engines.
    if not one:
        return None
    shaped = [c for c in one
              if not _link_region_has_indicator(c[4], indicator_types)]
    if not shaped:
        return None
    # Try the most promising candidates (fewest link words, longest other half),
    # asking Haiku about the unconfirmed half. Cap the spend.
    shaped.sort(key=lambda c: (len(c[4]),
                               -len((c[3] if c[5] == "left" else c[2]))))
    if ai_is_definition is None:
        return None                     # cannot consult Haiku -> make no claim
    asked = set()
    calls = 0
    considered = False                  # got at least one definite YES/NO back
    for i, j, left, right, mid, db_side in shaped:
        other = right if db_side == "left" else left
        phrase = " ".join(t.text for t in other)
        if phrase in asked:
            continue
        if calls >= MAX_AI_CALLS:
            break
        asked.add(phrase)
        calls += 1
        verdict = ai_is_definition(phrase, answer)   # True | False | None
        if verdict:
            lflag = "db" if db_side == "left" else "pending"
            rflag = "db" if db_side == "right" else "pending"
            return _build(ctx, answer, left, right, mid, lflag, rflag)
        if verdict is False:
            considered = True           # a definite NO -> enrichment was considered
        # verdict is None -> the check could not be made; not a considered verdict.
    # FAIL only when we genuinely CONSIDERED enrichment (a definite NO): one real
    # definition, no confirmable second. If every check errored / was unknown we
    # never considered it, so we ABSTAIN (None) rather than persist a durable FAIL
    # on a verdict we never obtained — a later run resolves it.
    if considered:
        return _build_one_def_fail(ctx, answer, shaped[0])
    return None


def _build_one_def_fail(ctx, answer, candidate):
    """Build the FAIL parse for a DD-shaped clue where exactly one half is a
    DB-confirmed definition and the other cannot be confirmed. Record the one real
    definition, mark any joining words, and leave the unconfirmable half
    unaccounted so _verify_dd surfaces it honestly as a FAIL."""
    i, j, left, right, mid, db_side = candidate
    confirmed = left if db_side == "left" else right
    src = Source(clue_atom_ids=tuple(a for t in confirmed for a in t.atom_ids),
                 text=" ".join(t.text for t in confirmed), value=ctx.answer_text,
                 mechanism="definition", source="db")
    annotations = [Annotation(clue_atom_ids=t.atom_ids, text=t.text,
                              role="link", note="link word") for t in mid]
    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=[src], links=[], annotations=annotations,
                  definition=None, operation="double_definition", solved_by="dd")
    _verify_dd(ctx, parse)
    return parse


def _build(ctx, answer, left, right, mid, lflag, rflag):
    def _src(toks, flag):
        return Source(clue_atom_ids=tuple(a for t in toks for a in t.atom_ids),
                      text=" ".join(t.text for t in toks), value=ctx.answer_text,
                      mechanism="definition", source=flag)
    annotations = [Annotation(clue_atom_ids=t.atom_ids, text=t.text,
                              role="link", note="link word") for t in mid]
    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=[_src(left, lflag), _src(right, rflag)],
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
