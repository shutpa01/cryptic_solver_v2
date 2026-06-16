"""Haiku wordplay-piece fallback — used only when the reference DB cannot supply a
piece a clue needs (a synonym/abbreviation a clue word stands for).

Mirrors core/definition_fallback.py exactly, one layer down: a Haiku suggestion is
used PROVISIONALLY (the engine marks the piece source='pending'), so the parse
becomes pending — never a silent pass — and finalize_pieces queues the piece to the
normal enrichment dashboard (type='synonym', which Accept inserts into
synonyms_pairs). Pure and DB-decoupled: the engine receives the suggest callable and
the store injected, never importing either.

Haiku only (claude-haiku-4-5). Sonnet is never used. The pending state IS the design
for AI-suggested material (feedback-haiku-in-enrichment).
"""


def make_piece_fallback(ai_suggest, store):
    """Build `suggest_piece(phrase, answer) -> value|None`, consulted by an engine
    only when the DB has no piece for a clue word it is trying to place. Reuses any
    value already queued for (phrase, answer) first (a re-run never re-pays Haiku),
    else asks Haiku; caches in memory for the wiring's lifetime. Returns None if no
    Haiku caller is available, so the fallback is simply absent."""
    if ai_suggest is None:
        return None
    cache = {}

    def suggest_piece(phrase, answer):
        key = ((phrase or "").lower().strip(), (answer or "").upper())
        if key in cache:
            return cache[key]
        val = store.pending_synonym(phrase, answer) if store is not None else None
        if not val:
            val = ai_suggest(phrase, answer)
        v = (val or "").upper() or None
        if v and v not in (answer or "").upper():     # must be a run of the answer
            v = None
        cache[key] = v
        return v

    return suggest_piece


def make_homophone_piece_fallback(ai_suggest_hom, store):
    """Build `suggest_hom(phrase, span) -> source_word|None` for the HOM_F slot: the
    word that sounds like the real-word answer letters `span` AND can mean `phrase`
    (went, ROAD -> RODE). The engine RE-VERIFIES the sound against the pronunciation
    dictionary before using it, so this never asserts the sound itself. A word a
    reviewer already rejected as a synonym of `phrase` is suppressed. Returns None if
    no Haiku caller is available, so the fallback is simply absent."""
    if ai_suggest_hom is None:
        return None
    cache = {}

    def suggest_hom(phrase, span):
        key = ((phrase or "").lower().strip(), (span or "").upper())
        if key in cache:
            return cache[key]
        w = (ai_suggest_hom(phrase, span) or "").upper() or None
        if w and store is not None and store.is_rejected_synonym(phrase, w):
            w = None
        cache[key] = w
        return w

    return suggest_hom


def finalize_homophone_pieces(parse, ctx, store, source=None, puzzle_number=None):
    """Run AFTER the parse is final. Queue each PROVISIONAL homophone piece's missing
    synonym (its .enrich_synonym = (phrase, source_word)) to the enrichment dashboard as
    type='synonym' — the homophone source->span sound is already dictionary-confirmed,
    so the synonym phrase->source is the only gap. If a reviewer already rejected it,
    fail the parse honestly. No-op for DB-confirmed pieces."""
    if store is None or parse is None:
        return
    for src in list(getattr(parse, "sources", [])):
        if getattr(src, "source", "db") != "pending" or src.mechanism != "homophone":
            continue
        pair = getattr(src, "enrich_synonym", None)
        if not pair:
            continue
        phrase, word = pair
        if store.is_rejected_synonym(phrase, word):
            msg = "homophone source %r -> %s was rejected by a reviewer" % (phrase, word)
            if msg not in parse.warnings:
                parse.warnings = list(parse.warnings) + [msg]
            parse.status = "fail"
        else:
            store.queue_synonym(phrase, word, parse.answer_text,
                                ctx.clue_text, source, puzzle_number)


def _norm(text):
    return " ".join((text or "").lower().split())


def augmented_lookup(lookup, suggest_piece):
    """Wrap an engine's `lookup(phrase, answer)` so a phrase the DB cannot resolve is
    offered to Haiku. Returns (aug_lookup, ai_values): aug_lookup behaves exactly like
    `lookup` whenever the DB has a hit, and otherwise adds a single provisional synonym
    value from Haiku; ai_values collects the (phrase, VALUE) pairs it supplied, so the
    caller can mark those pieces provisional after assembly.

    For the free-tiling evidence engines (anagram+charade, ...), which consult lookup
    directly during their DFS — the parallel of threading suggest_piece into the
    signature engine's _role_candidates. Only on a DB miss, so the DB-only pass is
    untouched; gate the AI pass on a DB-only failure at the call site."""
    ai_values = set()

    def aug(phrase, answer):
        hits = lookup(phrase, answer)
        if hits:
            return hits
        if suggest_piece is None:
            return []
        v = suggest_piece(phrase, answer)
        if not v:
            return []
        ai_values.add((_norm(phrase), v.upper()))
        return [(v.upper(), "synonym")]

    return aug, ai_values


def make_value_check(ai_could_produce, store):
    """Build `value_check(phrase, value) -> bool`: can `phrase` stand for the KNOWN
    letters `value`? For engines whose value target is fixed (the container's value
    component), where suggest_piece's 'guess the letters' model does not apply. Caches
    in memory and honours an existing reviewer rejection. Returns None if no caller."""
    if ai_could_produce is None:
        return None
    cache = {}

    def value_check(phrase, value):
        key = (_norm(phrase), (value or "").upper())
        if key in cache:
            return cache[key]
        if store is not None and store.is_rejected_synonym(phrase, value):
            cache[key] = False
            return False
        ok = bool(ai_could_produce(phrase, value))
        cache[key] = ok
        return ok

    return value_check


def mark_provisional(parse, ai_values):
    """Mark every Source the Haiku fallback supplied (its (text, value) is in
    ai_values) source='pending', so the engine's verifier downgrades the parse to
    pending and finalize_pieces queues it. Returns True if any was marked."""
    marked = False
    for s in getattr(parse, "sources", []):
        if (_norm(s.text), (s.value or "").upper()) in ai_values:
            s.source = "pending"
            marked = True
    return marked


def finalize_pieces(parse, ctx, store, source=None, puzzle_number=None):
    """Run AFTER the engine's parse is final. Queue each PROVISIONAL wordplay piece
    (a Haiku suggestion the engine actually used) to the enrichment dashboard as
    type='synonym', or — if a reviewer already rejected it — fail the parse honestly.
    No-op for DB-confirmed pieces. Nothing is written to the reference table here;
    Accept in the dashboard does that."""
    if store is None or parse is None:
        return
    for src in list(parse.sources):
        if getattr(src, "source", "db") != "pending":
            continue
        if src.mechanism not in ("synonym", "abbreviation"):
            continue
        if store.is_rejected_synonym(src.text, src.value):
            msg = "piece %r -> %s was rejected by a reviewer" % (src.text, src.value)
            if msg not in parse.warnings:
                parse.warnings = list(parse.warnings) + [msg]
            parse.status = "fail"
        else:
            store.queue_synonym(src.text, src.value, parse.answer_text,
                                ctx.clue_text, source, puzzle_number)
