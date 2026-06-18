"""Cryptic-definition engine — the whole clue IS the definition, no wordplay.

A cryptic definition (CD) has no decomposable wordplay: the entire clue is a single,
playful definition of the answer (e.g. "Flower of London?" = THAMES — a thing that
flows). There is nothing to reconstruct letter-by-letter, so it CANNOT be machine-
verified. This engine therefore only fires when the WHOLE clue is recorded as a
definition of the answer — either scraped into the reference DB or entered by hand —
and it is deliberately conservative about the verdict:

  * Sourced from the DB  -> ALWAYS 'pending'. A CD can never be auto-confirmed; a human
    must read it and agree. (The per-clue verdict override in the UI is how a human
    promotes it to 'pass', optionally with a comment.)

It is the LAST resort in the cascade — tried only after every wordplay engine and the
double-definition engine have failed — so it never intercepts a clue with real,
verifiable wordplay. Pure and DB-decoupled: it takes the same injected `defines`
predicate the other engines use, applied to the whole clue.
"""

from core.wfw_model import Source, Annotation, Parse


def solve_cryptic_definition(ctx, defines, comment=None):
    """If the WHOLE clue is a recorded definition of the answer, return a cryptic-
    definition Parse (status 'pending' — a CD is never machine-confirmed). Else None.

    `comment` is an optional human note to display alongside the definition."""
    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if not answer:
        return None
    word_toks = [t for t in ctx.clue_tokens if t.kind == "word"]
    if not word_toks:
        return None
    whole = " ".join(t.text for t in word_toks)
    try:
        if not defines(whole, answer):
            return None
    except Exception:
        return None

    def_atom_ids = tuple(aid for t in word_toks for aid in t.atom_ids)
    definition = Source(clue_atom_ids=def_atom_ids, text=ctx.clue_text,
                        value=ctx.answer_text, mechanism="definition", source="db")
    annotations = []
    if comment:
        annotations.append(Annotation(clue_atom_ids=def_atom_ids,
                                      text=str(comment), role="indicator",
                                      note="cryptic-definition note"))

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=[], links=[], annotations=annotations,
                  definition=definition, operation="cd", solved_by="cd")
    parse.template_id = None
    parse.matched_signature = None
    # A cryptic definition is never machine-confirmed: it has no wordplay to check, so the
    # whole-clue DB match is only evidence. A human must agree (UI verdict override) to pass.
    parse.warnings = ["cryptic definition — the whole clue defines the answer; "
                      "needs human confirmation to pass"]
    parse.status = "pending"
    return parse
