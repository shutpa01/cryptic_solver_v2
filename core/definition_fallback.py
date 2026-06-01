"""Haiku definition fallback — used only when the reference DB cannot define a
clue. Pure and DB-decoupled: it receives the Haiku caller and the enrichment
store injected, so the engines and their tests stay free of any API or database.

How it fits the universal definition stage:

  1. find_definitions() tries the reference DB first (unchanged). Only when it
     finds NOTHING does the engine consult this fallback.
  2. The fallback first reuses any definition already queued for this answer (so
     a re-run never pays Haiku twice); otherwise it asks Haiku for the phrase.
  3. Haiku may return a TRUNCATED phrase ("mountains"). We locate it among the
     clue's word tokens — punctuation/brackets are separate tokens, so a
     parenthesised definition like SARONG's "(piece of cloth)" is found
     naturally — and hand back a DefinitionSplit flagged source='pending'.
  4. The engine then runs its normal grammar-EXTENT pass, which grows the
     truncated phrase to the full correct definition ("in the mountains") using
     the clue's own bound function words — the same step every DB definition gets.
  5. After the engine has the final, grown phrase, finalize() queues THAT phrase
     to pending_enrichments for human verification (or, if a reviewer already
     rejected it, drops it and fails honestly). Nothing is written to the
     reference table here; Accept in the dashboard does that.

So the provisional definition is used this run and flagged provisional; once you
verify it, the next run finds it in the DB as an ordinary definition.
"""

from core.definition_engine import _split_from_indices


def _norm(text):
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _word_tokens(ctx):
    return [t for t in ctx.clue_tokens if t.kind == "word"]


def provisional_split(ctx, phrase):
    """Locate `phrase` as a contiguous run of clue WORD tokens at an edge and
    return a DefinitionSplit (source='pending'), or None if it cannot be
    placed at an edge. Edge-anchored by cryptic convention; a mid-clue match is
    rejected. Punctuation/bracket tokens are ignored because they are not word
    tokens."""
    words = _word_tokens(ctx)
    n = len(words)
    if n < 2:
        return None
    norm_words = [_norm(t.text) for t in words]
    phrase_words = [w for w in (_norm(p) for p in phrase.split()) if w]
    if not phrase_words:
        return None
    length = len(phrase_words)
    if length >= n:            # must leave at least one wordplay word
        return None

    edge_starts = []
    for start in range(0, n - length + 1):
        if norm_words[start:start + length] == phrase_words:
            if start == 0 or start + length == n:
                edge_starts.append(start)
    if not edge_starts:
        return None
    # Prefer an end-edge match (definitions parenthesised/added at the end are
    # the common fallback case); else the start edge.
    start = max(edge_starts)
    where = "start" if start == 0 else "end"
    idx = set(range(start, start + length))
    return _split_from_indices(words, idx, where, source="pending")


def make_fallback(ai_define, store):
    """Build the `define_fallback(ctx) -> DefinitionSplit | None` callable the
    engine consults on a DB miss. `ai_define(clue_text, answer) -> phrase|None`
    is the Haiku caller; `store` is a PendingStore (or a fake). Returns None if
    no Haiku caller is available, so the fallback is simply absent."""
    if ai_define is None:
        return None

    def define_fallback(ctx):
        answer = ctx.answer_text
        phrase = store.pending_definition(answer) if store else None
        if phrase is None:
            phrase = ai_define(ctx.clue_text, answer)
        if not phrase:
            return None
        return provisional_split(ctx, phrase)

    return define_fallback


def finalize(parse, ctx, store, source=None, puzzle_number=None):
    """Run AFTER the engine has produced its parse (definition grown to full
    extent). If the presented definition is a provisional Haiku one, either queue
    its final phrase for verification, or — if a reviewer already rejected it —
    drop it and mark the parse a fail. No-op for DB-confirmed definitions."""
    if store is None or parse is None or parse.definition is None:
        return
    if getattr(parse.definition, "source", "db") != "pending":
        return
    final = parse.definition.text
    answer = parse.answer_text
    if store.is_rejected_definition(final, answer):
        parse.definition = None
        msg = "definition %r was rejected by a reviewer" % final
        if msg not in parse.warnings:
            parse.warnings = list(parse.warnings) + [msg]
        parse.status = "fail"
        return
    store.queue_definition(final, answer, ctx.clue_text, source, puzzle_number)
