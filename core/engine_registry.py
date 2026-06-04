"""Engine registry + orchestrator — run a clue through every engine that exists.

This is the true-test entry point: given a clue and answer, try each clue-type
engine currently built, in order, and return the first Parse that solves. As new
engines are added (charade, container, anagram, ...) they are registered here and
the same runner picks them up — so "run it through all the engines that exist"
stays true automatically.

DB wiring is injected once (RefDB), kept out of the pure engines.
"""

from core.wfw_atoms import build_wfw_atom_context
from core import inflect, contractions


def _match_variants(text):
    """All forms to try when matching a clue word/phrase against the DB: its regular
    inflections (plural/verb), plus contraction/possessive base + expansions (each
    also inflected). Deduped, original first. The single place the inflection and
    contraction rules combine (memory: feedback-inflection-match)."""
    out = []

    def add(t):
        for v in inflect.phrase_variants(t):
            if v not in out:
                out.append(v)

    add(text)
    for f in contractions.forms(text):
        add(f)
    return out


def make_db_wiring():
    """Build the injected predicates from the live RefDB. Returns a dict the
    engines consume. Imported lazily so pure tests need no DB."""
    import os
    import sqlite3
    from signature_solver.db import RefDB
    db = RefDB()

    cryptic_db = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "data", "cryptic_new.db")

    # Memo caches (per wiring). The definition stage probes the DB O(windows) times
    # per clue and inflection multiplies that, while the live definition query is a
    # full-table scan; without caching the page becomes unusably slow. Trade-off: a
    # dashboard add is not seen until the wiring (server) is rebuilt — acceptable
    # for the true-test tool, which already restarts on code change.
    _cache_def, _cache_ind, _cache_look, _cache_lookall = {}, {}, {}, {}

    # Live indexes, built ONCE at wiring time by a single scan of each table, so the
    # per-clue lookups (probed O(windows) times, multiplied by inflection variants)
    # are O(1) dict hits instead of full-table scans — the scan-per-call was making
    # a single clue take tens of seconds. Same startup-snapshot staleness as RefDB.
    def _norm_def(text):
        return (text or "").lower().strip(".,;:!?\"'()-").strip()

    def _norm_ans(text):
        return (text or "").upper().replace(" ", "").replace("-", "")

    _live_def_index, _live_ind_index = {}, {}
    try:
        _c = sqlite3.connect(cryptic_db, timeout=30)
        try:
            for ans, dfn in _c.execute(
                    "SELECT answer, definition FROM definition_answers_augmented"):
                if ans and dfn:
                    _live_def_index.setdefault(_norm_ans(ans), set()).add(_norm_def(dfn))
            for wd, wt in _c.execute(
                    "SELECT word, wordplay_type FROM indicators"):
                if wd and wt:
                    _live_ind_index.setdefault(wd.lower().strip(), set()).add(wt)
        finally:
            _c.close()
    except Exception:
        _live_def_index, _live_ind_index = {}, {}

    def _live_defines(phrase, answer):
        """O(1) check against the prebuilt definition index (same data and
        normalisation as the old per-call scan of definition_answers_augmented)."""
        return _norm_def(phrase) in _live_def_index.get(_norm_ans(answer), ())

    def _defines_exact(phrase, answer):
        key = (phrase, answer)
        if key in _cache_def:
            return _cache_def[key]
        r = False
        try:
            r = bool(db.is_definition_of(phrase, answer))
        except Exception:
            r = False
        if not r:
            try:
                r = _live_defines(phrase, answer)
            except Exception:
                r = False
        _cache_def[key] = r
        return r

    def defines(phrase, answer):
        # Inflection- and contraction-aware: a clue word matches its regular
        # inflections and its contraction/possessive base+expansions.
        return any(_defines_exact(v, answer) for v in _match_variants(phrase))

    def _live_indicator_types(word):
        """O(1) check against the prebuilt indicator index (same data as the old
        per-call scan of the indicators table)."""
        return set(_live_ind_index.get(word.lower().strip(), ()))

    def _indicator_types_exact(word):
        if word in _cache_ind:
            return _cache_ind[word]
        types = set()
        try:
            types = {t for t, _, _ in db.get_indicator_types(word)}
        except Exception:
            types = set()
        if not types:                       # only pay the live query on a miss
            try:
                types |= _live_indicator_types(word)
            except Exception:
                pass
        _cache_ind[word] = types
        return types

    def indicator_types(word):
        # Inflection- and contraction-aware. Union the types over the variants.
        types = set()
        for v in _match_variants(word):
            types |= _indicator_types_exact(v)
        return types

    def is_dbe(word):
        """A definition-by-example indicator ('perhaps', 'possibly', 'maybe', ...).
        Inflection/contraction-aware via indicator_types."""
        return "definition by example" in (indicator_types(word) or set())

    def is_link(word):
        try:
            return db.is_link_word(word)
        except Exception:
            return False

    def _lookup_exact(word, answer):
        key = (word, answer)
        if key in _cache_look:
            return _cache_look[key]
        out = []
        try:
            for s in db.get_synonyms_substring_of(word, answer):
                out.append((s, "synonym"))
        except Exception:
            pass
        try:
            au = answer.upper()
            for a in db.get_abbreviations(word):
                if a and a in au:
                    out.append((a, "abbreviation"))
        except Exception:
            pass
        _cache_look[key] = out
        return out

    def _lookup_all_exact(word):
        out = []
        try:
            for s in db.get_synonyms(word):
                v = (s or "").upper()
                if v:
                    out.append((v, "synonym"))
        except Exception:
            pass
        try:
            for a in db.get_abbreviations(word):
                v = (a or "").upper()
                if v:
                    out.append((v, "abbreviation"))
        except Exception:
            pass
        return out

    def lookup_all(word):
        """(value, mechanism) options for a word with NO substring filter — the
        container engine needs an OUTER value (e.g. SANDS) that is split around the
        inner and so is NOT a contiguous substring of the answer. Inflection/
        contraction-aware; cached."""
        if word in _cache_lookall:
            return _cache_lookall[word]
        out, seen = [], set()
        for v in _match_variants(word):
            for val, mech in _lookup_all_exact(v):
                if (val, mech) not in seen:
                    seen.add((val, mech))
                    out.append((val, mech))
        _cache_lookall[word] = out
        return out

    def lookup(word, answer):
        """(value, mechanism) options for a wordplay word that are substrings of
        the answer — the answer-aware, UNCAPPED candidate set the catalog engines
        consume. Synonyms and abbreviations from RefDB; the word's own raw letters
        are added by each engine. No cap (the answer itself is the filter).

        Inflection- and contraction-aware: variants of the word are looked up too,
        so a DB synonym stored under a plural/verb/base form is found.
        """
        out, seen = [], set()
        for v in _match_variants(word):
            for val, mech in _lookup_exact(v, answer):
                if (val, mech) not in seen:
                    seen.add((val, mech))
                    out.append((val, mech))
        return out

    # AI definition fallback (the one AI touch-point), built only if available —
    # its absence simply means no fallback, never a crash. The store is the live
    # enrichment queue; the engines never see it directly. (Indicators need no AI
    # fallback: they are read off the clue's own leftover words and queued.)
    store = None
    define_fallback = None
    ai_is_definition = None
    try:
        from core.pending_store import PendingStore
        store = PendingStore()
        from core import ai_definition, definition_fallback
        define_fallback = definition_fallback.make_fallback(
            ai_definition.define, store)
        from core import ai_synonym
        ai_is_definition = ai_synonym.is_definition_of   # narrow DD half-check
    except Exception:
        pass

    # Catalog signatures, read once into memory (design §4: the engines are
    # catalog-DRIVEN). Loaded here with the rest of the wiring and injected, so the
    # engines stay pure and DB-decoupled. Absence simply means no catalog match.
    try:
        from core.catalog_loader import (load_charade_templates,
                                         load_anagram_templates,
                                         load_anagram_charade_templates)
        charade_templates = load_charade_templates()
        anagram_templates = load_anagram_templates()
        anagram_charade_templates = load_anagram_charade_templates()
    except Exception:
        charade_templates = anagram_templates = anagram_charade_templates = []

    return {"db": db, "defines": defines, "lookup": lookup,
            "indicator_types": indicator_types, "is_link": is_link,
            "define_fallback": define_fallback,
            "ai_is_definition": ai_is_definition, "store": store,
            "is_dbe": is_dbe, "lookup_all": lookup_all,
            "charade_templates": charade_templates,
            "anagram_templates": anagram_templates,
            "anagram_charade_templates": anagram_charade_templates}


def solve(ctx, wiring, source=None, puzzle_number=None, clue_id=None):
    """Run the clue through every engine that exists, return (parse, engine_name)
    for the first that solves, or (None, None).

    `source` / `puzzle_number` (the clue's own, when known) are carried only so a
    provisional piece can be queued under them for your review-by-source.
    `clue_id`, when given, makes the solve DURABLE: the final Parse (PASS or FAIL)
    is persisted as the substrate of record (core.store), so it survives the call
    and the screen can render straight from the DB."""
    # HIDDEN — triggered by the hidden run; simplest, tried first. Finding the
    # answer as a contiguous run is conclusive that the clue is hidden, so hidden
    # is TERMINAL whenever it fires (verdict pass or pending — it never fails). No
    # run -> it returns None and the cascade falls through.
    from core.hidden_engine import solve_hidden
    ph = solve_hidden(ctx, wiring["defines"],
                      indicator_types=wiring["indicator_types"],
                      is_link=wiring["is_link"],
                      define_fallback=wiring.get("define_fallback"))
    if ph is not None:
        return _finish(ph, "hidden", ctx, wiring, source, puzzle_number, clue_id)

    # DOUBLE DEFINITION — two definitions, no wordplay; gated grammar shape +
    # DB-confirm-one-half + narrow Haiku on the other. A pass (both DB) or pending
    # (one half queued) STOPS here. A DD fail (one real definition, no confirmable
    # second) does not stop the cascade.
    from core.dd_engine import solve_dd
    pd = solve_dd(ctx, wiring["defines"], is_link=wiring["is_link"],
                  indicator_types=wiring["indicator_types"],
                  ai_is_definition=wiring.get("ai_is_definition"))
    if pd is not None and pd.status in ("pass", "pending"):
        return _finish(pd, "dd", ctx, wiring, source, puzzle_number, clue_id)

    # ANAGRAM — catalog-driven, WORDPLAY-ONLY. The definition stage (here) decides
    # the split; the engine is handed only the wordplay and never sees the
    # definition. Indicator-gated and exact-letter, so high precision — tried before
    # charade. A pass or pending stops here.
    from core.anagram_engine import solve_anagram
    pa = _solve_wordplay_engine(ctx, wiring, solve_anagram,
                                wiring.get("anagram_templates") or [])
    if pa is not None and pa.status in ("pass", "pending"):
        return _finish(pa, "catalog", ctx, wiring, source, puzzle_number, clue_id)

    # CHARADE — the catalog spine, tried after DD. Catalog-DRIVEN: it walks the
    # mined charade signatures (injected as wiring["charade_templates"]) in
    # priority order. A pass or pending stops here. Definition (incl. the shared
    # Haiku fallback) and the synonym/abbreviation lookup come from the wiring; a
    # provisional definition makes it pending and is queued by _finish.
    from core.charade_engine import solve_charade
    pc = solve_charade(ctx, wiring["defines"], wiring["lookup"], wiring["is_link"],
                       wiring.get("charade_templates") or [],
                       define_fallback=wiring.get("define_fallback"),
                       is_dbe=wiring.get("is_dbe"))
    if pc is not None and pc.status in ("pass", "pending"):
        return _finish(pc, "catalog", ctx, wiring, source, puzzle_number, clue_id)

    # ANAGRAM+CHARADE — compound: a charade with one anagram piece. Tried after the
    # pure engines (it is more specific). A pass or pending stops here.
    from core.anagram_charade_engine import solve_anagram_charade
    pac = solve_anagram_charade(ctx, wiring["defines"], wiring["lookup"],
                                wiring["is_link"], wiring["indicator_types"],
                                wiring.get("anagram_charade_templates") or [],
                                define_fallback=wiring.get("define_fallback"),
                                is_dbe=wiring.get("is_dbe"))
    if pac is not None and pac.status in ("pass", "pending"):
        return _finish(pac, "catalog", ctx, wiring, source, puzzle_number, clue_id)

    # ANAGRAM+CONTAINER — compound: a container where one component is an anagram
    # (SANDWICHES, EXHORT). Gated on BOTH a container and an anagram indicator, so it
    # fires rarely and does not contend with the simpler engines.
    from core.anagram_container_engine import solve_anagram_container
    paco = solve_anagram_container(ctx, wiring["defines"], wiring["lookup_all"],
                                   wiring["is_link"], wiring["indicator_types"],
                                   define_fallback=wiring.get("define_fallback"),
                                   is_dbe=wiring.get("is_dbe"))
    if paco is not None and paco.status in ("pass", "pending"):
        return _finish(paco, "catalog", ctx, wiring, source, puzzle_number, clue_id)

    # Nothing produced a clean stop. Return the genuinely MOST COMPLETE fail so the
    # richest evidence is shown — measured (status, answer letters explained, clue
    # words accounted, fewest warnings), NOT by engine order.
    candidates = [(p, n) for p, n in ((pd, "dd"), (pa, "catalog"), (pc, "catalog"),
                                      (pac, "catalog"), (paco, "catalog"))
                  if p is not None]
    if candidates:
        parse, name = _most_complete(candidates, ctx)
        return _finish(parse, name, ctx, wiring, source, puzzle_number, clue_id)
    return None, None


def _solve_wordplay_engine(ctx, wiring, engine_fn, templates):
    """Run a WORDPLAY-ONLY catalog engine under the definition stage.

    The definition is decided HERE, never by the engine: for each candidate edge
    definition the definition stage proposes, hand the engine ONLY the wordplay
    tokens. When the engine returns a clean wordplay parse, attach that definition
    and fold it into the verdict (a provisional definition makes it pending). The
    engine never sees or chooses the definition. Returns the first PASS, else the
    best parse found, else None.
    """
    from core.definition_engine import find_definitions, dbe_annotation
    from core.wfw_model import Source
    if not templates:
        return None
    splits = list(find_definitions(ctx, wiring["defines"],
                                   define_fallback=wiring.get("define_fallback"),
                                   is_dbe=wiring.get("is_dbe")))
    if not splits:
        return None
    best = None
    for split in splits:
        parse = engine_fn(ctx, split.wordplay_tokens, wiring["is_link"],
                          wiring["indicator_types"], templates)
        if parse is None:
            continue
        parse.definition = Source(clue_atom_ids=split.def_atom_ids,
                                  text=split.phrase, value=ctx.answer_text,
                                  mechanism="definition", source=split.source)
        dbe = dbe_annotation(split)            # by-example marker, kept out of wordplay
        if dbe is not None:
            parse.annotations = list(parse.annotations) + [dbe]
        if parse.status != "fail":                  # wordplay clean -> fold in def
            if split.source == "pending":
                parse.status = "pending"
                parse.warnings = list(parse.warnings) + [
                    "the definition is provisional (queued for enrichment)"]
            else:
                parse.status = "pass"
        if parse.status == "pass":
            return parse
        if best is None:
            best = parse
    return best


def _most_complete(candidates, ctx):
    """Pick the parse that explains the most, by a measured key (not engine order):
    status rank, then answer letters linked, then clue words accounted, then fewest
    warnings. `candidates` is a non-empty list of (parse, engine_name)."""
    word_total = sum(1 for t in ctx.clue_tokens if t.kind == "word")

    def key(item):
        parse, _ = item
        status_rank = {"pass": 2, "pending": 1, "fail": 0}.get(parse.status, 0)
        linked = len({l.answer_pos for l in parse.links})
        accounted = word_total - len(parse.unexplained_words(ctx))
        return (status_rank, linked, accounted, -len(parse.warnings))

    return max(candidates, key=key)


def _finish(parse, name, ctx, wiring, source, puzzle_number, clue_id):
    """Queue any provisional pieces, persist the final Parse, return it."""
    _finalize_provisional(parse, ctx, wiring.get("store"), source, puzzle_number)
    if clue_id is not None:
        from core import store as wfw_store
        wfw_store.persist(clue_id, parse, ctx)    # preserve the parse + atomisation
    return parse, name


def _finalize_provisional(parse, ctx, store, source, puzzle_number):
    """Queue (or reject-drop) any provisional piece once the engine's parse is
    final. Central here so every engine gets it for free."""
    if store is None:
        return
    from core import definition_fallback, indicator_enrichment, dd_enrichment
    definition_fallback.finalize(parse, ctx, store, source, puzzle_number)
    indicator_enrichment.finalize_indicators(parse, ctx, store, source,
                                             puzzle_number)
    dd_enrichment.finalize_dd(parse, ctx, store, source, puzzle_number)


def solve_clue_text(clue_text, answer, wiring, source=None, puzzle_number=None,
                    clue_id=None):
    ctx = build_wfw_atom_context(clue_text, answer)
    parse, name = solve(ctx, wiring, source=source, puzzle_number=puzzle_number,
                        clue_id=clue_id)
    return ctx, parse, name
