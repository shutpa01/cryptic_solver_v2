"""Double-definition enrichment — queue an AI-confirmed half for verification.

When the DD engine confirms one half via the DB and the OTHER via Haiku, that
other half is a definition of the answer the DB does not yet know. This routes it
to the live enrichment queue (as a definition — a DD half IS a definition of the
answer), so once you accept it, next time both halves are pure DB hits. If a
reviewer already rejected it, the DD is dropped to a fail. Mirrors
indicator_enrichment / definition_fallback.finalize.
"""


def finalize_dd(parse, ctx, store, source=None, puzzle_number=None):
    if store is None or parse is None or getattr(parse, "solved_by", "") != "dd":
        return
    for s in parse.sources:
        if s.mechanism != "definition" or getattr(s, "source", "db") != "pending":
            continue
        if store.is_rejected_definition(s.text, parse.answer_text):
            msg = "definition %r was rejected by a reviewer" % s.text
            if msg not in parse.warnings:
                parse.warnings = list(parse.warnings) + [msg]
            parse.status = "fail"
        else:
            store.queue_definition(s.text, parse.answer_text, ctx.clue_text,
                                   source, puzzle_number)
