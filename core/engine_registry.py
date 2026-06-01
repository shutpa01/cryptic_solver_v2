"""Engine registry + orchestrator — run a clue through every engine that exists.

This is the true-test entry point: given a clue and answer, try each clue-type
engine currently built, in order, and return the first Parse that solves. As new
engines are added (charade, container, anagram, ...) they are registered here and
the same runner picks them up — so "run it through all the engines that exist"
stays true automatically.

DB wiring is injected once (RefDB), kept out of the pure engines.
"""

from core.wfw_atoms import build_wfw_atom_context


def make_db_wiring():
    """Build the injected predicates from the live RefDB. Returns a dict the
    engines consume. Imported lazily so pure tests need no DB."""
    import os
    import sqlite3
    from signature_solver.db import RefDB
    db = RefDB()

    cryptic_db = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "data", "cryptic_new.db")

    def _live_defines(phrase, answer):
        """Direct lookup of definition_answers_augmented, so a definition just
        accepted in the dashboard is seen WITHOUT reloading the in-memory RefDB
        (which is a startup snapshot). Same normalisation as is_definition_of."""
        ac = answer.upper().replace(" ", "").replace("-", "")
        pc = phrase.lower().strip(".,;:!?\"'()-").strip()
        conn = sqlite3.connect(cryptic_db, timeout=30)
        try:
            rows = conn.execute(
                "SELECT definition FROM definition_answers_augmented "
                "WHERE UPPER(REPLACE(REPLACE(answer,' ',''),'-','')) = ?",
                (ac,)).fetchall()
        finally:
            conn.close()
        for (d,) in rows:
            if d and d.lower().strip(".,;:!?\"'()-").strip() == pc:
                return True
        return False

    def defines(phrase, answer):
        try:
            if db.is_definition_of(phrase, answer):
                return True
        except Exception:
            pass
        try:
            return _live_defines(phrase, answer)
        except Exception:
            return False

    def _live_indicator_types(word):
        """Direct lookup of the indicators table, so an indicator just accepted in
        the dashboard is seen WITHOUT reloading the in-memory RefDB snapshot."""
        w = word.lower().strip()
        conn = sqlite3.connect(cryptic_db, timeout=30)
        try:
            rows = conn.execute(
                "SELECT DISTINCT wordplay_type FROM indicators WHERE LOWER(word)=?",
                (w,)).fetchall()
        finally:
            conn.close()
        return {r[0] for r in rows if r[0]}

    def indicator_types(word):
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
        return types

    def is_link(word):
        try:
            return db.is_link_word(word)
        except Exception:
            return False

    # AI definition fallback (the one AI touch-point), built only if available —
    # its absence simply means no fallback, never a crash. The store is the live
    # enrichment queue; the engines never see it directly. (Indicators need no AI
    # fallback: they are read off the clue's own leftover words and queued.)
    store = None
    define_fallback = None
    try:
        from core.pending_store import PendingStore
        store = PendingStore()
        from core import ai_definition, definition_fallback
        define_fallback = definition_fallback.make_fallback(
            ai_definition.define, store)
    except Exception:
        pass

    return {"db": db, "defines": defines,
            "indicator_types": indicator_types, "is_link": is_link,
            "define_fallback": define_fallback, "store": store}


def solve(ctx, wiring, source=None, puzzle_number=None, clue_id=None):
    """Run the clue through every engine that exists, return (parse, engine_name)
    for the first that solves, or (None, None).

    `source` / `puzzle_number` (the clue's own, when known) are carried only so a
    provisional piece can be queued under them for your review-by-source.
    `clue_id`, when given, makes the solve DURABLE: the final Parse (PASS or FAIL)
    is persisted as the substrate of record (core.store), so it survives the call
    and the screen can render straight from the DB."""
    store = wiring.get("store")

    # HIDDEN — built. Triggered by the hidden run; simplest, tried first.
    from core.hidden_engine import solve_hidden
    parse = solve_hidden(ctx, wiring["defines"],
                         indicator_types=wiring["indicator_types"],
                         is_link=wiring["is_link"],
                         define_fallback=wiring.get("define_fallback"))
    if parse is not None:
        _finalize_provisional(parse, ctx, store, source, puzzle_number)
        if clue_id is not None:
            from core import store as wfw_store
            wfw_store.persist(clue_id, parse)     # preserve the evidence
        return parse, "hidden"

    # (future engines register below, in clue-flow order)
    return None, None


def _finalize_provisional(parse, ctx, store, source, puzzle_number):
    """Queue (or reject-drop) any provisional piece once the engine's parse is
    final. Central here so every engine gets it for free."""
    if store is None:
        return
    from core import definition_fallback, indicator_enrichment
    definition_fallback.finalize(parse, ctx, store, source, puzzle_number)
    indicator_enrichment.finalize_indicators(parse, ctx, store, source,
                                             puzzle_number)


def solve_clue_text(clue_text, answer, wiring, source=None, puzzle_number=None,
                    clue_id=None):
    ctx = build_wfw_atom_context(clue_text, answer)
    parse, name = solve(ctx, wiring, source=source, puzzle_number=puzzle_number,
                        clue_id=clue_id)
    return ctx, parse, name
