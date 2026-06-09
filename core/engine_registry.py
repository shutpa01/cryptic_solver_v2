"""Engine registry + orchestrator — run a clue through every engine that exists.

This is the true-test entry point: given a clue and answer, try each clue-type
engine currently built, in order, and return the first Parse that solves. As new
engines are added (charade, container, anagram, ...) they are registered here and
the same runner picks them up — so "run it through all the engines that exist"
stays true automatically.

DB wiring is injected once (RefDB), kept out of the pure engines.
"""

from core.wfw_atoms import build_wfw_atom_context
from core import inflect, contractions, literals


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
        contraction-aware; cached.

        Adds a curated LITERAL value (mechanism 'raw') when `word` is a short
        function word read as its own letters (it -> IT): an additive source the
        DB lacks, gated to core.literals.LITERAL_WORDS. Only the container family
        accepts 'raw'; other consumers filter it out, so this is inert for them."""
        if word in _cache_lookall:
            return _cache_lookall[word]
        out, seen = [], set()
        for v in _match_variants(word):
            for val, mech in _lookup_all_exact(v):
                if (val, mech) not in seen:
                    seen.add((val, mech))
                    out.append((val, mech))
        lit = literals.literal_value(word)
        if lit and (lit, "raw") not in seen:
            seen.add((lit, "raw"))
            out.append((lit, "raw"))
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
    suggest_piece = None
    value_check = None
    try:
        from core.pending_store import PendingStore
        store = PendingStore()
        from core import ai_definition, definition_fallback
        define_fallback = definition_fallback.make_fallback(
            ai_definition.define, store)
        from core import ai_synonym
        ai_is_definition = ai_synonym.is_definition_of   # narrow DD half-check
        from core import ai_piece, piece_fallback
        # Haiku wordplay-piece fallback: suggests a missing synonym piece, used
        # PROVISIONALLY (-> pending) and queued for enrichment. The pending state is
        # the design for AI-suggested material (feedback-haiku-in-enrichment).
        suggest_piece = piece_fallback.make_piece_fallback(
            ai_piece.suggest_piece, store)
        # Haiku value check (yes/no against a KNOWN target) for engines whose value
        # target is fixed — the container's value component (its outer is not a
        # substring of the answer, so suggest_piece's 'guess letters' model can't apply).
        value_check = piece_fallback.make_value_check(ai_piece.could_produce, store)
    except Exception:
        pass

    # Catalog signatures, read once into memory (design §4: the engines are
    # catalog-DRIVEN). Loaded here with the rest of the wiring and injected, so the
    # engines stay pure and DB-decoupled. Absence simply means no catalog match.
    try:
        from core.catalog_loader import (load_charade_templates,
                                         load_anagram_templates,
                                         load_anagram_charade_templates,
                                         load_anagram_container_templates,
                                         load_container_templates,
                                         load_container_charade_templates,
                                         load_reversal_templates,
                                         load_reversal_charade_templates)
        charade_templates = load_charade_templates()
        anagram_templates = load_anagram_templates()
        anagram_charade_templates = load_anagram_charade_templates()
        anagram_container_templates = load_anagram_container_templates()
        container_templates = load_container_templates()
        container_charade_templates = load_container_charade_templates()
        reversal_templates = load_reversal_templates()
        reversal_charade_templates = load_reversal_charade_templates()
    except Exception:
        charade_templates = anagram_templates = anagram_charade_templates = []
        anagram_container_templates = []
        container_templates = []
        container_charade_templates = []
        reversal_templates = []
        reversal_charade_templates = []

    return {"db": db, "defines": defines, "lookup": lookup,
            "indicator_types": indicator_types, "is_link": is_link,
            "define_fallback": define_fallback,
            "ai_is_definition": ai_is_definition, "store": store,
            "is_dbe": is_dbe, "lookup_all": lookup_all,
            "suggest_piece": suggest_piece, "value_check": value_check,
            "charade_templates": charade_templates,
            "anagram_templates": anagram_templates,
            "anagram_charade_templates": anagram_charade_templates,
            "anagram_container_templates": anagram_container_templates,
            "container_templates": container_templates,
            "container_charade_templates": container_charade_templates,
            "reversal_templates": reversal_templates,
            "reversal_charade_templates": reversal_charade_templates}


def solve(ctx, wiring, source=None, puzzle_number=None, clue_id=None,
          charade_solve=None, anagram_solve=None):
    """Run the clue through every engine that exists, return (parse, engine_name)
    for the first that solves, or (None, None).

    `source` / `puzzle_number` (the clue's own, when known) are carried only so a
    provisional piece can be queued under them for your review-by-source.
    `clue_id`, when given, makes the solve DURABLE: the final Parse (PASS or FAIL)
    is persisted as the substrate of record (core.store), so it survives the call
    and the screen can render straight from the DB.
    `charade_solve` / `anagram_solve` override those engines (default = the
    catalog-driven signature engines); the A/B harness passes the legacy evidence
    engine here to compare the full cascade both ways."""
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

    # ANAGRAM — catalog-DRIVEN: walks the mined anagram signatures (ANA_F fodder +
    # optional ANA_I indicator) in priority order, placing slots on the wordplay with
    # interior-link exclusion in the fodder and gaps -> links classified last. Tried
    # before charade (indicator-gated, exact-letter, high precision). A pass/pending
    # stops here. Default = core.anagram_signature_engine; the A/B harness can pass
    # the legacy evidence engine via `anagram_solve`.
    if anagram_solve is None:
        from core.anagram_signature_engine import solve_anagram as anagram_solve
    pa = anagram_solve(ctx, wiring["defines"], wiring["is_link"],
                       wiring["indicator_types"], wiring.get("anagram_templates") or [],
                       define_fallback=wiring.get("define_fallback"),
                       is_dbe=wiring.get("is_dbe"))
    if pa is not None and pa.status in ("pass", "pending"):
        return _finish(pa, "catalog", ctx, wiring, source, puzzle_number, clue_id)

    # CHARADE — the catalog spine. Catalog-DRIVEN: it walks the mined charade
    # signatures (injected as wiring["charade_templates"]) in priority order, places
    # the typed slots on the wordplay (gaps -> links classified last), fills role-pure
    # (SYN_F/ABR_F/LIT_F), and verifies the pieces concatenate to the answer. A pass
    # or pending stops here. Default = core.charade_signature_engine; the A/B harness
    # can pass the legacy evidence engine via `charade_solve`.
    if charade_solve is None:
        from core.charade_signature_engine import solve_charade as charade_solve
    pc = charade_solve(ctx, wiring["defines"], wiring["lookup"], wiring["is_link"],
                       wiring.get("charade_templates") or [],
                       define_fallback=wiring.get("define_fallback"),
                       is_dbe=wiring.get("is_dbe"))   # DB-only; AI recovery runs later
    if pc is not None and pc.status in ("pass", "pending"):
        return _finish(pc, "catalog", ctx, wiring, source, puzzle_number, clue_id)

    # ANAGRAM+CHARADE — compound: a charade with one anagram piece. Tried after the
    # pure engines (it is more specific). A pass or pending stops here. Catalog-DRIVEN
    # (signature engine); catalog gaps (multi-word slots not yet mined) become preserved
    # fail-evidence for the separate signature-creation process, not free-tiled.
    from core.anagram_charade_signature_engine import solve_anagram_charade
    pac = solve_anagram_charade(ctx, wiring["defines"], wiring["lookup"],
                                wiring["is_link"], wiring["indicator_types"],
                                wiring.get("anagram_charade_templates") or [],
                                define_fallback=wiring.get("define_fallback"),
                                is_dbe=wiring.get("is_dbe"))   # DB-only; AI later
    if pac is not None and pac.status in ("pass", "pending"):
        return _finish(pac, "catalog", ctx, wiring, source, puzzle_number, clue_id)

    # ANAGRAM+CONTAINER — compound: a container where one component is an anagram
    # (SANDWICHES, EXHORT). Catalog-DRIVEN (signature engine, seeded from working
    # solves); insertion-aware. Gaps -> preserved fail-evidence for the separate
    # signature process. (The deferred AI recovery below still routes through the
    # evidence engine's yes/no value check.)
    from core.anagram_container_signature_engine import solve_anagram_container
    paco = solve_anagram_container(ctx, wiring["defines"], wiring["lookup_all"],
                                   wiring["is_link"], wiring["indicator_types"],
                                   wiring.get("anagram_container_templates") or [],
                                   define_fallback=wiring.get("define_fallback"),
                                   is_dbe=wiring.get("is_dbe"))
    if paco is not None and paco.status in ("pass", "pending"):
        return _finish(paco, "catalog", ctx, wiring, source, puzzle_number, clue_id)

    # CONTAINER — plain insertion: one DB value inserted into another (BREAM=BEAM around
    # R, TACTICS=TICS around ACT). Catalog-DRIVEN (signature engine, seeded from working
    # solves); insertion-aware (the verifier resolves which value run is outer vs inner).
    # Gated on a container indicator + exact reconstruction. Gaps -> preserved
    # fail-evidence for the separate signature process.
    from core.container_signature_engine import solve_container
    pcon = solve_container(ctx, wiring["defines"], wiring["lookup_all"],
                           wiring["is_link"], wiring["indicator_types"],
                           wiring.get("container_templates") or [],
                           define_fallback=wiring.get("define_fallback"),
                           is_dbe=wiring.get("is_dbe"))
    if pcon is not None and pcon.status in ("pass", "pending"):
        return _finish(pcon, "catalog", ctx, wiring, source, puzzle_number, clue_id)

    # CONTAINER+CHARADE — a charade where one piece is a container (LURCHER = LURE around
    # CH + R). Catalog-DRIVEN (signature engine; the container pair is marked CNT_F in the
    # signature, charade pieces SYN_F); the reconstructor tiles the answer with the charade
    # pieces + one container span in any order. Tried after the plain container (more
    # general); gated on a container indicator. Gaps -> preserved fail-evidence.
    from core.container_charade_signature_engine import solve_container_charade
    pccc = solve_container_charade(ctx, wiring["defines"], wiring["lookup_all"],
                                   wiring["is_link"], wiring["indicator_types"],
                                   wiring.get("container_charade_templates") or [],
                                   define_fallback=wiring.get("define_fallback"),
                                   is_dbe=wiring.get("is_dbe"))
    if pccc is not None and pccc.status in ("pass", "pending"):
        return _finish(pccc, "catalog", ctx, wiring, source, puzzle_number, clue_id)

    # REVERSAL — a plain reversal (the whole answer is one DB value, reversed: SMART =
    # rev(TRAMS)). Catalog-DRIVEN (signature engine); single-piece, answer-driven (the fodder
    # run's DB value must equal reverse(answer)). Tried after the container family; gated on a
    # reversal indicator. Multi-piece reverse-of-charade is reversal_charade, a separate engine.
    from core.reversal_signature_engine import solve_reversal
    prev = solve_reversal(ctx, wiring["defines"], wiring["lookup_all"],
                          wiring["is_link"], wiring["indicator_types"],
                          wiring.get("reversal_templates") or [],
                          define_fallback=wiring.get("define_fallback"),
                          is_dbe=wiring.get("is_dbe"))
    if prev is not None and prev.status in ("pass", "pending"):
        return _finish(prev, "catalog", ctx, wiring, source, puzzle_number, clue_id)

    # REVERSAL+CHARADE — a charade where the reversal applies (one piece reversed, e.g.
    # AFAR = A + rev(RAF); or the whole charade reversed, e.g. ERATO = rev(ARE)+rev(OT)).
    # Catalog-DRIVEN (signature engine): the slots fix which pieces are reversed (REV_F) vs
    # forward (SYN_F); the reconstructor tiles the answer answer-driven. Tried after the plain
    # reversal (more general / multi-piece); gated on a reversal indicator.
    from core.reversal_charade_signature_engine import solve_reversal_charade
    prevc = solve_reversal_charade(ctx, wiring["defines"], wiring["lookup_all"],
                                   wiring["is_link"], wiring["indicator_types"],
                                   wiring.get("reversal_charade_templates") or [],
                                   define_fallback=wiring.get("define_fallback"),
                                   is_dbe=wiring.get("is_dbe"))
    if prevc is not None and prevc.status in ("pass", "pending"):
        return _finish(prevc, "catalog", ctx, wiring, source, puzzle_number, clue_id)

    # DOUBLE DEFINITION — run LAST, not first. Its second-definition check (esp. the
    # Haiku half) is softer than the catalog engines, which reconstruct the answer
    # exactly; running it first let it intercept catalog clues. So the precise engines
    # claim first and DD catches only the residue. A pass/pending stops here.
    from core.dd_engine import solve_dd
    pd = solve_dd(ctx, wiring["defines"], is_link=wiring["is_link"],
                  indicator_types=wiring["indicator_types"],
                  ai_is_definition=wiring.get("ai_is_definition"))
    if pd is not None and pd.status in ("pass", "pending"):
        return _finish(pd, "dd", ctx, wiring, source, puzzle_number, clue_id)

    # AI PIECE RECOVERY — runs ONLY now that every DB-grounded engine above has failed,
    # so a provisional Haiku piece can never intercept a clue a later DB engine solves
    # cleanly (the same reason DD runs last). Re-try the piece-capable engines WITH the
    # Haiku suggester; a solve here is PENDING (provisional pieces) and queued for
    # enrichment. Skipped when no suggester is wired (tests / batch set it to None).
    sp = wiring.get("suggest_piece")
    if sp is not None:
        from core.charade_signature_engine import solve_charade as _sig_charade
        pcr = _sig_charade(ctx, wiring["defines"], wiring["lookup"],
                           wiring["is_link"], wiring.get("charade_templates") or [],
                           define_fallback=wiring.get("define_fallback"),
                           is_dbe=wiring.get("is_dbe"), suggest_piece=sp)
        if pcr is not None and pcr.status in ("pass", "pending"):
            return _finish(pcr, "catalog", ctx, wiring, source, puzzle_number, clue_id)
        pacr = solve_anagram_charade(ctx, wiring["defines"], wiring["lookup"],
                                     wiring["is_link"], wiring["indicator_types"],
                                     wiring.get("anagram_charade_templates") or [],
                                     define_fallback=wiring.get("define_fallback"),
                                     is_dbe=wiring.get("is_dbe"), suggest_piece=sp)
        if pacr is not None and pacr.status in ("pass", "pending"):
            return _finish(pacr, "catalog", ctx, wiring, source, puzzle_number,
                           clue_id)
        vc = wiring.get("value_check")
        if vc is not None:
            # AI recovery uses the EVIDENCE container engine's yes/no value check (the
            # signature engine's AI path is a follow-up); only fires after all DB
            # engines failed, so it never intercepts a clean DB solve.
            from core.anagram_container_engine import \
                solve_anagram_container as _ev_container
            pacor = _ev_container(
                ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
                wiring["indicator_types"], define_fallback=wiring.get("define_fallback"),
                is_dbe=wiring.get("is_dbe"), could_produce=vc)
            if pacor is not None and pacor.status in ("pass", "pending"):
                return _finish(pacor, "catalog", ctx, wiring, source, puzzle_number,
                               clue_id)

    # Nothing produced a clean stop. Return the genuinely MOST COMPLETE fail so the
    # richest evidence is shown — measured (status, answer letters explained, clue
    # words accounted, fewest warnings), NOT by engine order.
    candidates = [(p, n) for p, n in ((pd, "dd"), (pa, "catalog"), (pc, "catalog"),
                                      (pac, "catalog"), (paco, "catalog"),
                                      (pcon, "catalog"), (pccc, "catalog"))
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
            # Pending if EITHER the wordplay is provisional (a missing-indicator
            # fallback left the indicator pending) OR the definition is provisional.
            wordplay_pending = parse.status == "pending"
            if split.source == "pending" or wordplay_pending:
                parse.status = "pending"
                if split.source == "pending":
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
    from core import (definition_fallback, indicator_enrichment, dd_enrichment,
                      piece_fallback)
    definition_fallback.finalize(parse, ctx, store, source, puzzle_number)
    indicator_enrichment.finalize_indicators(parse, ctx, store, source,
                                             puzzle_number)
    piece_fallback.finalize_pieces(parse, ctx, store, source, puzzle_number)
    dd_enrichment.finalize_dd(parse, ctx, store, source, puzzle_number)


def solve_clue_text(clue_text, answer, wiring, source=None, puzzle_number=None,
                    clue_id=None, charade_solve=None, anagram_solve=None):
    ctx = build_wfw_atom_context(clue_text, answer)
    parse, name = solve(ctx, wiring, source=source, puzzle_number=puzzle_number,
                        clue_id=clue_id, charade_solve=charade_solve,
                        anagram_solve=anagram_solve)
    return ctx, parse, name
