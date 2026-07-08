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


def _norm_apostrophe(text):
    """Curly apostrophes/quotes -> straight, so a clue's "that's"/"Spooner's" (typeset
    with U+2019) matches the DB, which stores the straight form. Telegraph clues use the
    curly form throughout, so without this every possessive/contraction silently misses."""
    return (text or "").replace("’", "'").replace("‘", "'")


def _match_variants(text):
    """All forms to try when matching a clue word/phrase against the DB: its regular
    inflections (plural/verb), plus contraction/possessive base + expansions (each
    also inflected). Deduped, original first. The single place the inflection and
    contraction rules combine (memory: feedback-inflection-match)."""
    text = _norm_apostrophe(text)
    out = []

    def add(t):
        for v in inflect.phrase_variants(t):
            if v not in out:
                out.append(v)

    add(text)
    for f in contractions.forms(text):
        add(f)
    return out


# Possessive-S: a possessive clue word "X's" can contribute a literal S to its value
# (gang's = POSSE + S = POSSES). Module-level toggle so a before/after regression can
# isolate its effect; ON by default.
POSSESSIVE_S = True


def make_db_wiring():
    """Build the injected predicates from the reference DB. Returns a dict the
    engines consume. Imported lazily so pure tests need no DB.

    Uses core.live_db.LiveDB — every lookup is a live indexed query (via the
    normalized-key columns) instead of preloading the whole ~1.7M-row reference DB
    into RAM. Start is ~instant, nothing big is held in memory (no long-session
    crash), and a freshly-added entry is seen at once. Verified 40/40 identical to
    the old RefDB preload (core/_run_live_batch.py)."""
    import os
    import sqlite3
    from core.live_db import LiveDB
    db = LiveDB()

    # Single source of truth for the reference-DB path: reuse the one LiveDB resolved
    # (core.live_db._default_path). Same value as before; sharing it means a test can point
    # the whole wiring at a temp DB copy by patching _default_path, never the real DB.
    cryptic_db = db.path

    # Fill any blank normalized-key columns (e.g. rows hand-added in DB Browser),
    # so the live lookups — which key off that tidied column for speed — can see them
    # after a reload. Only NULL rows are touched, so it is cheap; the fast-lookup
    # design is unchanged. Best-effort: never fatal to wiring build.
    try:
        from core.norm_backfill import backfill_null_norm_keys
        backfill_null_norm_keys(cryptic_db)
    except Exception:
        pass

    # Memo caches (per wiring). The definition stage probes the DB O(windows) times
    # per clue and inflection multiplies that, while the live definition query is a
    # full-table scan; without caching the page becomes unusably slow. Trade-off: a
    # dashboard add is not seen until the wiring (server) is rebuilt — acceptable
    # for the true-test tool, which already restarts on code change.
    _cache_def, _cache_ind, _cache_look, _cache_lookall = {}, {}, {}, {}
    _cache_sel = {}

    # Live indexes, built ONCE at wiring time by a single scan of each table, so the
    # per-clue lookups (probed O(windows) times, multiplied by inflection variants)
    # are O(1) dict hits instead of full-table scans — the scan-per-call was making
    # a single clue take tens of seconds. Same startup-snapshot staleness as RefDB.
    def _norm_def(text):
        return (text or "").lower().strip(".,;:!?\"'()-").strip()

    def _norm_ans(text):
        return (text or "").upper().replace(" ", "").replace("-", "")

    _live_def_index, _live_ind_index, _subst_index = {}, {}, {}
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
            # The small dedicated substitutions table (original_word -> substitution,
            # e.g. times->X, artists->RA): folded into all_values so the anagram-
            # substitution engine can draw a fodder letter from ANY table, not just one.
            for ow, sub in _c.execute(
                    "SELECT original_word, substitution FROM substitutions"):
                if ow and sub:
                    _subst_index.setdefault(ow.lower().strip(), []).append(
                        (sub.strip().upper(), "substitution"))
        finally:
            _c.close()
    except Exception:
        _live_def_index, _live_ind_index, _subst_index = {}, {}, {}

    # The curated literal lexicon (short function words a setter may use as their own
    # letters: it->IT). Was a hardcoded frozenset in core.literals; now the live
    # literal_words table (cryptic_new.db), editable via the clue-page admin panel. Its
    # OWN try so a missing table on an old DB falls back to the seed, never wiping the
    # indexes above. The set is installed as core.literals' words provider below.
    _live_literal_index = set(literals.LITERAL_WORDS)
    try:
        _c = sqlite3.connect(cryptic_db, timeout=30)
        try:
            rows = _c.execute("SELECT word FROM literal_words").fetchall()
            _live_literal_index = {w.lower().strip() for (w,) in rows if w and w.strip()}
        finally:
            _c.close()
    except Exception:
        _live_literal_index = set(literals.LITERAL_WORDS)

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
        #
        # WORD-INTEGRITY GUARD: a variant may only confirm the phrase if it still
        # contains ALL the phrase's words. A variant that DROPS a word (fewer words
        # than the phrase) can never confirm it — otherwise a multi-word block gets
        # stamped DB-confirmed off a shorter match ("Virtuoso's vocal" via "virtuoso").
        # Changing a word's ending (suffers->suffer) keeps the count and is allowed;
        # a contraction EXPANSION (he's->he is) adds words and is allowed.
        nwords = len(phrase.split())
        return any(_defines_exact(v, answer) for v in _match_variants(phrase)
                   if len(v.split()) >= nwords)

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

    def deletion_subtypes(word):
        """DB deletion subtypes for a word/phrase — the deletion indicator vocabulary,
        read live from the `indicators` table (formerly a hardcoded set in deletion.py).
        A NULL/blank subtype maps to 'general' (a plain removal). Inflection/contraction-
        aware so 'releases' finds a 'release' row and vice versa."""
        out = set()
        for v in _match_variants(word):
            try:
                for wtype, sub, _ in db.get_indicator_types(v):
                    if wtype == "deletion":
                        out.add(((sub or "").strip().lower()) or "general")
            except Exception:
                pass
        return out

    def charade_positional_subtypes(word):
        """Charade-positional subtypes for a word/phrase — the re-ordering indicator
        vocabulary, read live from the `indicators` table (wordplay_type
        'charade_positional'): 'after'/'before' (any clue) and 'after_down'/'before_down'
        (down clues only). A NULL/blank subtype is skipped (order cannot be inferred), so
        a legacy untyped row never drives a re-order. Inflection/contraction-aware."""
        out = set()
        for v in _match_variants(word):
            try:
                for wtype, sub, _ in db.get_indicator_types(v):
                    if wtype == "charade_positional" and sub and sub.strip():
                        out.add(sub.strip().lower())
            except Exception:
                pass
        return out

    def selection_rules(word):
        """DB-licensed letter-SELECTION rules for a word/phrase (first/last/outer/middle/
        alternate), read live from the indicators table via selection_indicators.
        SUBTYPE_RULE. Inflection/contraction-aware. Replaces the old hardcoded word list
        in core.selection_indicators (the same anti-pattern deletion/substitution shed)."""
        if word in _cache_sel:
            return _cache_sel[word]
        from core import selection_indicators
        out = set()
        for v in _match_variants(word):
            try:
                for wtype, sub, _ in db.get_indicator_types(v):
                    rule = selection_indicators.SUBTYPE_RULE.get(
                        (wtype, (sub or "").strip().lower()))
                    if rule:
                        out.add(rule)
            except Exception:
                pass
        _cache_sel[word] = out
        return out

    # selection_indicators.find_indicators reads this provider instead of a hardcoded list
    from core import selection_indicators as _sel_ind
    _sel_ind.set_rules_provider(selection_rules)

    # core.literals.literal_value reads this provider (the live literal_words table)
    # instead of its seed frozenset. A clue-level add patches _live_literal_index via
    # invalidate("literal"), so the provider closes over the live set.
    literals.set_words_provider(lambda: _live_literal_index)

    # Palindrome indicator vocabulary: load from the indicators table (wordplay_type
    # 'palindrome') and install into core.palindrome_indicators — was a hardcoded word list,
    # now curated through the DB gate. Bucketed by subtype: 'opp_pair' (two words, both must
    # appear) / a multi-word row = 'phrase' / a single word = 'single'.
    from core import palindrome_indicators as _pal
    _pn = _pal._norm
    _pal_singles, _pal_phrases, _pal_pairs = set(), [], []
    try:
        _c = sqlite3.connect(cryptic_db, timeout=30)
        try:
            for wd, sub in _c.execute(
                    "SELECT word, subtype FROM indicators WHERE wordplay_type='palindrome'"):
                parts = [p for p in (_pn(x) for x in (wd or "").split()) if p]
                if not parts:
                    continue
                if (sub or "").strip().lower() == "opp_pair" and len(parts) == 2:
                    _pal_pairs.append(frozenset(parts))
                elif len(parts) > 1:
                    _pal_phrases.append(tuple(parts))
                else:
                    _pal_singles.add(parts[0])
        finally:
            _c.close()
    except Exception:
        pass
    _pal.set_vocab(_pal_singles, _pal_phrases, _pal_pairs)

    def is_dbe(word):
        """A definition-by-example indicator ('perhaps', 'possibly', 'maybe', ...).
        Inflection/contraction-aware via indicator_types."""
        return "definition by example" in (indicator_types(word) or set())

    def is_link(word):
        try:
            return db.is_link_word(_norm_apostrophe(word))
        except Exception:
            return False

    def phrase_synonyms(phrase):
        # Inflection-aware synonyms for the homophone source lookup: try the phrase
        # AND its per-word singular/plural variants, so a clue's "refers to" also
        # finds a "refer to" row and vice versa (the singular/plural rule, which the
        # legacy synonym lookup applies only to the whole string). Gated downstream by
        # the sound match, so widening here cannot pass a non-homophone.
        out, seen = [], set()
        for v in _match_variants(phrase):
            try:
                for s in db.get_synonyms(v):
                    if s not in seen:
                        seen.add(s)
                        out.append(s)
            except Exception:
                pass
        return out

    # NOTE: abbreviations are NOT looked up here — they are form-specific (an abbreviation
    # is a fixed convention tied to an exact word, e.g. us->US, starting->S), so inflecting
    # them only leaks a different word's value onto the clue word (start -> "starting" -> S).
    # Synonyms ARE inflected (semantic equivalence across forms). The abbreviation lookup
    # runs ONCE on the original word in lookup()/lookup_all().
    def _lookup_syn_exact(word, answer):
        key = (word, answer)
        if key in _cache_look:
            return _cache_look[key]
        out = []
        try:
            for s in db.get_synonyms_substring_of(word, answer):
                out.append((s, "synonym"))
        except Exception:
            pass
        _cache_look[key] = out
        return out

    def _lookup_all_syn_exact(word):
        out = []
        try:
            for s in db.get_synonyms(word):
                v = (s or "").upper()
                if v:
                    out.append((v, "synonym"))
        except Exception:
            pass
        return out

    def _abbreviations(word, answer=None):
        """Abbreviations for the EXACT word (apostrophe-normalised, NO inflection variants).
        When `answer` is given, keep only those that appear in it (the substring filter)."""
        out = []
        au = (answer or "").upper()
        try:
            for a in db.get_abbreviations(_norm_apostrophe(word)):
                v = (a or "").upper()
                if v and (answer is None or v in au):
                    out.append((v, "abbreviation"))
        except Exception:
            pass
        return out

    def _is_possessive(word):
        """True for a possessive 'X's' — the apostrophe-s can contribute a literal S to
        the value (gang's = POSSE + S = POSSES). Only the singular possessive adds an S;
        a plural possessive (ladies') already ends in s, so it is excluded."""
        w = (word or "").replace("’", "'").strip().lower()
        return len(w) > 2 and w.endswith("'s")

    def _add_possessive_s(word, pairs, seen, answer=None):
        """Append val+'S' for each synonym/abbreviation in `pairs` when `word` is a
        possessive. When `answer` is given (the substring-filtered lookup), only keep a
        variant that is a substring of the answer. Mutates/extends a copy and returns it."""
        if not POSSESSIVE_S or not _is_possessive(word):
            return pairs
        out = list(pairs)
        au = (answer or "").upper()
        for val, mech in list(pairs):
            if mech not in ("synonym", "abbreviation"):
                continue
            sv = (val + "S", mech)
            if sv in seen:
                continue
            if answer is not None and sv[0] not in au:
                continue
            seen.add(sv)
            out.append(sv)
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
        for v in _match_variants(word):                 # synonyms: inflection-aware
            for val, mech in _lookup_all_syn_exact(v):
                if (val, mech) not in seen:
                    seen.add((val, mech))
                    out.append((val, mech))
        for val, mech in _abbreviations(word):          # abbreviations: exact word only
            if (val, mech) not in seen:
                seen.add((val, mech))
                out.append((val, mech))
        lit = literals.literal_value(word)
        if lit and (lit, "raw") not in seen:
            seen.add((lit, "raw"))
            out.append((lit, "raw"))
        out = _add_possessive_s(word, out, seen)   # gang's -> ... + POSSES
        _cache_lookall[word] = out
        return out

    def all_values(word):
        """Every (value, mechanism) a word can take, across ALL tables — synonyms,
        abbreviations and curated literals (via lookup_all) PLUS the substitutions table.
        Used by the anagram-substitution engine, which deduces the EXACT residual letters
        it needs, so a permissive multi-table source is safe (the residual match filters)."""
        out, seen = [], set()
        for val, mech in lookup_all(word):
            if (val, mech) not in seen:
                seen.add((val, mech))
                out.append((val, mech))
        for v in _match_variants(word):
            for pair in _subst_index.get(v.lower().strip(), ()):
                if pair not in seen:
                    seen.add(pair)
                    out.append(pair)
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
        for v in _match_variants(word):                 # synonyms: inflection-aware
            for val, mech in _lookup_syn_exact(v, answer):
                if (val, mech) not in seen:
                    seen.add((val, mech))
                    out.append((val, mech))
        for val, mech in _abbreviations(word, answer):  # abbreviations: exact word only
            if (val, mech) not in seen:
                seen.add((val, mech))
                out.append((val, mech))
        out = _add_possessive_s(word, out, seen, answer)   # gang's -> ... + POSSES
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
    suggest_hom = None
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
        # Homophone-aware piece fallback for the charade+homophone HOM_F slot: a
        # missing-synonym homophone source named for enrichment (the SOUND is
        # re-verified in the engine, never taken on the model's word).
        suggest_hom = piece_fallback.make_homophone_piece_fallback(
            ai_piece.suggest_homophone_source, store)
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
                                         load_reversal_charade_templates,
                                         load_charade_homophone_templates,
                                         load_deletion_templates,
                                         load_template_tiers)
        # The MAIN cascade loads only PASS-tier signatures (they may produce a PASS).
        # PENDING-only signatures are loaded separately for the final pending stage
        # (signature-tiers build §5 Step 3/4) so they can never pass in the main cascade.
        charade_templates = load_charade_templates(tier='pass')
        anagram_templates = load_anagram_templates(tier='pass')
        anagram_charade_templates = load_anagram_charade_templates(tier='pass')
        anagram_container_templates = load_anagram_container_templates(tier='pass')
        container_templates = load_container_templates(tier='pass')
        container_charade_templates = load_container_charade_templates(tier='pass')
        reversal_templates = load_reversal_templates(tier='pass')
        reversal_charade_templates = load_reversal_charade_templates(tier='pass')
        charade_homophone_templates = load_charade_homophone_templates(tier='pass')
        deletion_templates = load_deletion_templates(tier='pass')
        # PENDING-only signatures, loaded once for the final pending-only stage (Step 4).
        # They are NOT in the main cascade above, so they can never produce a PASS.
        pending_templates = {
            "charade": load_charade_templates(tier='pending'),
            "anagram": load_anagram_templates(tier='pending'),
            "anagram_charade": load_anagram_charade_templates(tier='pending'),
            "anagram_container": load_anagram_container_templates(tier='pending'),
            "container": load_container_templates(tier='pending'),
            "container_charade": load_container_charade_templates(tier='pending'),
            "reversal": load_reversal_templates(tier='pending'),
            "reversal_charade": load_reversal_charade_templates(tier='pending'),
            "charade_homophone": load_charade_homophone_templates(tier='pending'),
            "deletion": load_deletion_templates(tier='pending'),
        }
        # {template_id: tier} over ALL templates — feeds the defence-in-depth cap in _finish.
        template_tier = load_template_tiers()
    except Exception:
        charade_templates = anagram_templates = anagram_charade_templates = []
        anagram_container_templates = []
        container_templates = []
        container_charade_templates = []
        reversal_templates = []
        reversal_charade_templates = []
        charade_homophone_templates = []
        deletion_templates = []
        pending_templates = {}
        template_tier = {}

    def invalidate(kind=None, word=None, synonym=None, definition=None, answer=None,
                   wordplay_type=None):
        """Fold a SINGLE reference-DB add into this live wiring WITHOUT a full rebuild.
        The adders (core.admin_db) write the normalized-key columns the live queries
        match on, so all that is stale after an add is the in-memory caching: drop the
        memo + per-word caches (the next lookup re-reads the just-written row) and patch
        the two prebuilt indexes with just the new entry. No 643k-row rescan, no LiveDB
        reconnect — a clue-level add costs the same as solving one clue."""
        for _c in (_cache_def, _cache_ind, _cache_look, _cache_lookall, _cache_sel):
            _c.clear()
        try:
            db.clear_caches()
        except Exception:
            pass
        if kind == "definition" and answer and definition:
            _live_def_index.setdefault(_norm_ans(answer), set()).add(
                _norm_def(definition))
        elif kind == "indicator" and word and wordplay_type:
            _live_ind_index.setdefault(word.lower().strip(), set()).add(
                (wordplay_type or "").strip().lower())
        elif kind == "link" and word:
            try:
                db.note_link_word(word)
            except Exception:
                pass
        elif kind == "literal" and word:
            _live_literal_index.add(word.lower().strip())
        elif kind == "homophone":
            pass        # homophones are read live (db.get_homophones); the cache clear
                        # above is all that's needed for the new pair to be seen

    return {"db": db, "defines": defines, "lookup": lookup, "invalidate": invalidate,
            "indicator_types": indicator_types, "deletion_subtypes": deletion_subtypes,
            "charade_positional_subtypes": charade_positional_subtypes,
            "selection_rules": selection_rules,
            "is_link": is_link,
            "sounds_like": db.get_homophones, "synonyms_of": phrase_synonyms,
            "sounds_alike": db.sounds_alike, "pronounce": db.get_pronunciation,
            "define_fallback": define_fallback,
            "ai_is_definition": ai_is_definition, "store": store,
            "try_andlit": True,        # offer the &lit (all-in-one) reading to opted-in
                                       #   engines (acrostic ...): whole clue = def = wordplay
            "is_dbe": is_dbe, "lookup_all": lookup_all, "all_values": all_values,
            "suggest_piece": suggest_piece, "value_check": value_check,
            "suggest_hom": suggest_hom,
            "charade_homophone_templates": charade_homophone_templates,
            "charade_templates": charade_templates,
            "anagram_templates": anagram_templates,
            "anagram_charade_templates": anagram_charade_templates,
            "anagram_container_templates": anagram_container_templates,
            "container_templates": container_templates,
            "container_charade_templates": container_charade_templates,
            "reversal_templates": reversal_templates,
            "reversal_charade_templates": reversal_charade_templates,
            "deletion_templates": deletion_templates,
            "pending_templates": pending_templates,
            "template_tier": template_tier}


# The wiring keys that drive an AI (Haiku/Sonnet) call. A batch run nulls these
# so a whole-puzzle solve makes ZERO network calls and is instant; AI fires only on
# an explicit per-clue re-run (see db_only).
_AI_KEYS = ("define_fallback", "suggest_piece", "value_check", "ai_is_definition",
            "suggest_hom")


def db_only(wiring):
    """A DB-only VIEW of a wiring: same DB predicates and caches, every AI
    touch-point nulled. Shallow copy, so it shares the live DB connection and the
    memo caches with the full wiring — building it is free. Used for batch solving
    so a whole puzzle never makes an AI call; the full wiring (AI on) is used only
    when the user re-runs a single clue on demand."""
    if wiring is None:
        return None
    w = dict(wiring)
    for k in _AI_KEYS:
        w[k] = None
    return w


def solve(ctx, wiring, source=None, puzzle_number=None, clue_id=None,
          charade_solve=None, anagram_solve=None, deletion_solve=None):
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
    # Install the DB predicates for this solve so every engine's _verify can DB-validate
    # the roles it records (no role assigned by elimination — core.role_validity).
    from core import role_validity
    role_validity.set_predicates(wiring["indicator_types"], wiring["is_link"])

    # HIDDEN — triggered by the hidden run; simplest, tried first. Finding the
    # answer as a contiguous run is conclusive that the clue is hidden, so hidden
    # is TERMINAL whenever it fires (verdict pass or pending — it never fails). No
    # run -> it returns None and the cascade falls through.
    from core.hidden_engine import solve_hidden
    ph = solve_hidden(ctx, wiring["defines"],
                      indicator_types=wiring["indicator_types"],
                      is_link=wiring["is_link"],
                      define_fallback=wiring.get("define_fallback"))
    hidden_fallback = None
    if ph is not None:
        if ph.status == "pass":
            return _finish(ph, "hidden", ctx, wiring, source, puzzle_number, clue_id)
        # A hidden PENDING is NOT terminal (it was: a coincidental contiguous run with no
        # confirmed hidden indicator / unaccounted words — e.g. INN inside "sINNing" —
        # used to pre-empt every later engine and block a forced alternation). Hold it as a
        # LAST-RESORT fallback and let the cascade continue, so a more specific engine (a
        # forced/indicator-gated alternation, acrostic, anagram, ...) can claim the clue.
        # Returned at the end only if nothing better solves it, so out-of-box behaviour is
        # unchanged (the hidden pending is still what shows when no other engine fires).
        hidden_fallback = ph

    # ACROSTIC — initial/final-letter selection. Answer-driven (the selected letters
    # must EXACTLY spell the answer) and indicator-gated, so it is highly specific and
    # cannot intercept another type (0 false positives measured over 1,500 non-acrostic
    # clues). Tried right after hidden — both are precise letter-level mechanisms — so a
    # genuine acrostic (even one mis-tagged charade in the corpus) is claimed before the
    # looser charade engine. Abstains (None) on anything that is not a clean acrostic.
    from core.acrostic_engine import solve_acrostic
    pacro = solve_acrostic(ctx, wiring["defines"], wiring["is_link"],
                           wiring["indicator_types"],
                           define_fallback=wiring.get("define_fallback"),
                           is_dbe=wiring.get("is_dbe"),
                           andlit=wiring.get("try_andlit", False))
    if pacro is not None and pacro.status in ("pass", "pending"):
        return _finish(pacro, "acrostic", ctx, wiring, source, puzzle_number, clue_id)

    # ALTERNATION — every-other-letter selection over a contiguous word run read as one
    # letter stream ("sordid play" -> ODDLY). Answer-driven (the selected letters must
    # EXACTLY spell the answer) and indicator-gated, the sibling of acrostic; tried right
    # after it. Abstains (None) on anything that is not a clean alternation.
    from core.alternation_engine import solve_alternation
    palt = solve_alternation(ctx, wiring["defines"], wiring["is_link"],
                             wiring["indicator_types"],
                             define_fallback=wiring.get("define_fallback"),
                             is_dbe=wiring.get("is_dbe"))
    if palt is not None and palt.status in ("pass", "pending"):
        return _finish(palt, "alternation", ctx, wiring, source, puzzle_number, clue_id)

    # PALINDROME — the whole answer reads the same both ways (SAGAS, ROTOR). Answer-driven
    # (answer == reverse) and gated on a palindrome indicator; the wordplay produces no
    # letters of its own. Tried with the other precise gated mechanisms. A palindrome
    # answer cannot be a meaningful reversal of a DIFFERENT word, so it cannot intercept a
    # genuine reversal clue. Abstains (None) on anything that is not a clean palindrome.
    from core.palindrome_engine import solve_palindrome
    ppal = solve_palindrome(ctx, wiring["defines"], wiring["is_link"],
                            define_fallback=wiring.get("define_fallback"),
                            is_dbe=wiring.get("is_dbe"))
    if ppal is not None and ppal.status in ("pass", "pending"):
        return _finish(ppal, "palindrome", ctx, wiring, source, puzzle_number, clue_id)

    # SPOONERISM — the answer's two syllables with their onsets transposed sound like a
    # source phrase (HAIRSHIRT <-> SHARE HURT). Phonetic + answer-driven, gated on a
    # Spooner indicator, so it only fires on Spooner clues and cannot intercept anything
    # else. Two-word content pairs each word's synonyms (class+stadium -> rank+bowl);
    # otherwise abstains. Needs pronunciations, so it sits after the other gated mechanisms.
    from core.spoonerism_engine import solve_spoonerism
    pspoon = solve_spoonerism(ctx, wiring["defines"], wiring["is_link"],
                              wiring["pronounce"], wiring["synonyms_of"],
                              is_dbe=wiring.get("is_dbe"))
    if pspoon is not None and pspoon.status in ("pass", "pending"):
        return _finish(pspoon, "spoonerism", ctx, wiring, source, puzzle_number, clue_id)

    # HOMOPHONE — the whole answer SOUNDS like a source word (or its synonym), gated on
    # a homophone indicator. Answer-driven (the source's homophone must EQUAL the answer)
    # and indicator-gated, so highly specific. Span-level provenance ("sounds like X").
    # Tried with the other precise gated mechanisms (hidden/acrostic), before the catalog
    # spine. Only the plain whole-answer form; the compound/phonetic ("I lash"->EYELASH)
    # form is a separate, signature-based engine not built. Abstains (None) otherwise.
    from core.homophone_engine import solve_homophone
    phom = solve_homophone(ctx, wiring["defines"], wiring["is_link"],
                           wiring["indicator_types"], wiring["sounds_alike"],
                           wiring["synonyms_of"],
                           define_fallback=wiring.get("define_fallback"),
                           is_dbe=wiring.get("is_dbe"), pronounce=wiring.get("pronounce"))
    if phom is not None and phom.status in ("pass", "pending"):
        return _finish(phom, "homophone", ctx, wiring, source, puzzle_number, clue_id)

    # CHARADE + HOMOPHONE — a charade with one homophone piece (MIDDLEWEIGHT = MIDDLE +
    # sounds-like WAIT). Catalog-driven and homophone-indicator gated; the HOM_F slot
    # resolves an answer span by sound (the other pieces are ordinary SYN_F/ABR_F). A
    # missing-synonym homophone source is named PENDING for enrichment (suggest_hom,
    # AI-off in batch). Tried after the plain homophone, before the catalog spine.
    from core.charade_homophone_signature_engine import solve_charade_homophone
    phomc = solve_charade_homophone(
        ctx, wiring["defines"], wiring["lookup"], wiring["is_link"],
        wiring["indicator_types"], wiring.get("charade_homophone_templates") or [],
        wiring["sounds_alike"], wiring["synonyms_of"],
        define_fallback=wiring.get("define_fallback"), is_dbe=wiring.get("is_dbe"),
        suggest_hom=wiring.get("suggest_hom"))
    if phomc is not None and phomc.status in ("pass", "pending"):
        return _finish(phomc, "charade_homophone", ctx, wiring, source,
                       puzzle_number, clue_id)

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
    if pa is not None and pa.status in ("pass", "pending") and not _anagram_degenerate(pa):
        return _finish(pa, "anagram", ctx, wiring, source, puzzle_number, clue_id)

    # ANAGRAM (SUBSTITUTION) — an anagram whose fodder has one SUBSTITUTED word (Oscar->O,
    # then anagram). The plain anagram above uses raw clue letters only, so it cannot see
    # this; here the residual (answer letters minus the raw bulk) is deduced and matched to
    # a short value of the held-out word from ANY table. Gated on an anagram indicator,
    # answer-driven. Tried right after the plain anagram (same family), before charade.
    from core.anagram_substitution_engine import solve_anagram_substitution
    pasub = solve_anagram_substitution(
        ctx, wiring["defines"], wiring["all_values"], wiring["indicator_types"],
        wiring["is_link"], define_fallback=wiring.get("define_fallback"),
        is_dbe=wiring.get("is_dbe"))
    if (pasub is not None and pasub.status in ("pass", "pending")
            and not _anagram_degenerate(pasub)):
        return _finish(pasub, "anagram_substitution", ctx, wiring, source, puzzle_number, clue_id)

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
        return _finish(pc, "charade", ctx, wiring, source, puzzle_number, clue_id)

    # CHARADE (POSITIONAL) — a charade whose pieces are RE-ORDERED by a positional
    # indicator ("School following second-class old" = B+O+SCH = BOSCH). The plain charade
    # above assembles only in clue order, so it cannot reach a re-ordered answer; this
    # engine fires only when it has not, is GATED on a charade-positional indicator, and is
    # ANSWER-DRIVEN (the pivoted pieces must concatenate to the exact answer). Isolated
    # engine; the plain charade engine is untouched. A pass/pending stops here; a fail
    # (indicator fired but no assembly) is preserved as evidence below.
    from core.charade_positional_engine import solve_charade_positional
    ppos = solve_charade_positional(
        ctx, wiring["defines"], wiring["lookup"], wiring["is_link"],
        wiring["charade_positional_subtypes"],
        define_fallback=wiring.get("define_fallback"), is_dbe=wiring.get("is_dbe"))
    if ppos is not None and ppos.status in ("pass", "pending"):
        return _finish(ppos, "charade_positional", ctx, wiring, source, puzzle_number, clue_id)

    # CHARADE (POSITIONAL, LOCAL) — sibling of the above. The positional indicator swaps ONLY
    # its ADJACENT PAIR ("Cover county show after parking" = county + (show after parking ->
    # P+READ) = BEDS+P+READ = BEDSPREAD), leaving other pieces in clue order. The global pivot
    # above cannot reach this, and this cannot reach a global pivot (BOSCH); two distinct
    # shapes, so a bespoke sibling (not a branch). Gated on a SWAP indicator; answer-driven.
    from core.charade_positional_local_engine import solve_charade_positional_local
    pposl = solve_charade_positional_local(
        ctx, wiring["defines"], wiring["lookup"], wiring["is_link"],
        wiring["charade_positional_subtypes"],
        define_fallback=wiring.get("define_fallback"), is_dbe=wiring.get("is_dbe"))
    if pposl is not None and pposl.status in ("pass", "pending"):
        return _finish(pposl, "charade_positional_local", ctx, wiring, source, puzzle_number, clue_id)

    # CHARADE + ACROSTIC — a charade where ONE piece is a multi-word acrostic (DOCTORS =
    # DOC[first of Ducks Observed Crossing, "firstly"] + TORS[rocky hills]). The acrostic
    # engine only spells the WHOLE answer; the charade's SEL_F takes letters from a single
    # word. This builds the first/last letter of a RUN of words as one charade piece.
    # Requires >=1 acrostic piece AND >=1 plain piece (can't intercept a pure acrostic or a
    # plain charade); gated on an acrostic indicator; answer-driven; PASS-only.
    from core.charade_acrostic_engine import solve_charade_acrostic
    pcac2 = solve_charade_acrostic(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], define_fallback=wiring.get("define_fallback"),
        is_dbe=wiring.get("is_dbe"))
    if pcac2 is not None and pcac2.status in ("pass", "pending"):
        return _finish(pcac2, "charade_acrostic", ctx, wiring, source, puzzle_number, clue_id)

    # CHARADE + CONTAINER + ACROSTIC — a charade combining ONE container piece and >=1
    # acrostic piece (INDICATES = [INDIA around C] + TES[initially try erotic strip]). Falls
    # between charade_acrostic (no container piece) and container_charade (no acrostic piece);
    # neither can reach it. Requires >=1 container piece AND >=1 acrostic piece, double-gated
    # on a container/insertion AND an acrostic indicator, answer-driven, PASS-only.
    from core.charade_container_acrostic_engine import solve_charade_container_acrostic
    pcca = solve_charade_container_acrostic(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], define_fallback=wiring.get("define_fallback"),
        is_dbe=wiring.get("is_dbe"))
    if pcca is not None and pcca.status in ("pass", "pending"):
        return _finish(pcca, "charade_container_acrostic", ctx, wiring, source, puzzle_number, clue_id)

    # CHARADE + ALTERNATION — a charade where ONE piece is the alternate (every-other)
    # letters of a word ("oddly ignored near" = ER), the others ordinary pieces (SANDPIPER =
    # SAND + PIP + ER). Gated on an alternation indicator AND >= 1 ordinary piece, so it
    # cannot intercept a plain charade (no alternation piece) or a whole-answer alternation
    # (no ordinary piece). Answer-driven; a fresh stage that never edits the charade or
    # alternation engines.
    from core.charade_alternation_engine import solve_charade_alternation
    pcalt = solve_charade_alternation(
        ctx, wiring["defines"], wiring["lookup"], wiring["is_link"],
        wiring["indicator_types"], wiring["selection_rules"],
        define_fallback=wiring.get("define_fallback"), is_dbe=wiring.get("is_dbe"))
    if pcalt is not None and pcalt.status in ("pass", "pending"):
        return _finish(pcalt, "charade_alternation", ctx, wiring, source, puzzle_number, clue_id)

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
        return _finish(pac, "anagram_charade", ctx, wiring, source, puzzle_number, clue_id)

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
        return _finish(paco, "anagram_container", ctx, wiring, source, puzzle_number, clue_id)

    # CHARADE + ANAGRAM-CONTAINER — a charade with one anagram-container piece (DOGFIGHT =
    # D["Day"] + [anag("fog hit") "awful" containing G "hiding"], G = first of "Germany's"
    # via "leader"). The plain anagram-container above builds the whole answer as one
    # [anag ∋ value]; it has no charade prefix/suffix. This requires >=1 plain charade piece
    # AND exactly one anagram-container span, so it cannot intercept a plain charade or a
    # plain anagram-container. Gated on BOTH an anagram and a container indicator;
    # answer-driven. The inserted value may itself be a first/last-letter selection.
    from core.charade_anagram_container_engine import solve_charade_anagram_container
    pcac = solve_charade_anagram_container(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], wiring["selection_rules"],
        define_fallback=wiring.get("define_fallback"), is_dbe=wiring.get("is_dbe"))
    if pcac is not None and pcac.status in ("pass", "pending"):
        return _finish(pcac, "charade_anagram_container", ctx, wiring, source, puzzle_number, clue_id)

    # CONTAINER+DELETION — a compound: build a container (one DB value inserted into
    # another, the inner optionally a small charade), then a deletion trims the result to
    # the answer ("old spades in the last shed" = OS in THE -> TOSHE, shed last -> TOSH).
    # Answer-driven (final string must EQUAL the answer) and gated on BOTH a container and
    # a deletion indicator, so it fires only on the genuine compound — tried before the
    # plain container/deletion engines, which cannot reach it.
    from core.container_deletion_engine import solve_container_deletion
    pcd = solve_container_deletion(ctx, wiring["defines"], wiring["lookup_all"],
                                   wiring["is_link"], wiring["indicator_types"],
                                   wiring["deletion_subtypes"],
                                   define_fallback=wiring.get("define_fallback"),
                                   is_dbe=wiring.get("is_dbe"),
                                   loc_rules=wiring.get("selection_rules"))
    if pcd is not None and pcd.status in ("pass", "pending"):
        return _finish(pcd, "container_deletion", ctx, wiring, source, puzzle_number, clue_id)

    # CHARADE+DELETION — a charade where ONE piece is a deletion (EX + HORTS, where HORTS
    # = SHORTS 'no top'). Answer-driven; the deletion piece's pre-deletion value is
    # reconstructed from the answer and checked by membership (literal first). Gated on a
    # deletion indicator; tried before the plain charade/deletion engines.
    from core.charade_deletion_engine import solve_charade_deletion
    pchd = solve_charade_deletion(ctx, wiring["defines"], wiring["lookup_all"],
                                  wiring["is_link"], wiring["indicator_types"],
                                  wiring["deletion_subtypes"],
                                  define_fallback=wiring.get("define_fallback"),
                                  is_dbe=wiring.get("is_dbe"),
                                  loc_rules=wiring.get("selection_rules"))
    if pchd is not None and pchd.status in ("pass", "pending"):
        return _finish(pchd, "charade_deletion", ctx, wiring, source, puzzle_number, clue_id)

    # CHARADE + MULTI-DELETION — the sibling shape: a charade where TWO OR MORE pieces are
    # each a deletion, governed by one shared deletion indicator (OBERON = robes[-ends] +
    # wrong[-ends] = OBE + RON, "stripped off"). charade_deletion above does exactly one
    # deletion piece, so it cannot reach this; this requires >= 2 deletion pieces, so it
    # never intercepts that engine or the plain charade. Answer-driven; deletion-gated.
    from core.charade_multi_deletion_engine import solve_charade_multi_deletion
    pcmd = solve_charade_multi_deletion(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], wiring["deletion_subtypes"],
        define_fallback=wiring.get("define_fallback"), is_dbe=wiring.get("is_dbe"),
        loc_rules=wiring.get("selection_rules"))
    if pcmd is not None and pcmd.status in ("pass", "pending"):
        return _finish(pcmd, "charade_multi_deletion", ctx, wiring, source, puzzle_number, clue_id)

    # (charade_synonym_multi_deletion was trialled here for MENDELEEV but SHELVED 2026-07-02:
    #  its safety A/B fabricated 10/600 false passes — indirect synonym-deletion over 2+ pieces
    #  tiles almost any answer into plausible-but-wrong deletions, the exact failure the
    #  literal-only rule avoids. Engine file kept, inactive, pending a tighter adjacency-gated
    #  redesign. Do NOT re-wire without a clean A/B.)

    # CHARADE + NAMED-LETTER DELETION — a charade where ONE piece is a value with a SPECIFIC
    # named letter removed, the removed letter a VERIFIED wordplay-table value of another
    # word (CAREEN = CAR + EVEN["even"] losing V["volume"]). Unlike charade_deletion (a
    # positional first/last/outer/middle removal), the deleted letter is named, so nothing is
    # invented. Gated on a deletion indicator; answer-driven. Tried after positional
    # charade+deletion.
    from core.charade_named_deletion_engine import solve_charade_named_deletion
    pcnd = solve_charade_named_deletion(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["all_values"],
        wiring["is_link"], wiring["indicator_types"],
        define_fallback=wiring.get("define_fallback"), is_dbe=wiring.get("is_dbe"))
    if pcnd is not None and pcnd.status in ("pass", "pending"):
        return _finish(pcnd, "charade_named_deletion", ctx, wiring, source, puzzle_number, clue_id)

    # CHARADE + MULTI-WORD NAMED DELETION (sibling of the above) — the deleted string is a
    # CHARADE of named letters from a CONTIGUOUS RUN of >=2 clue words, removed from a REAL DB
    # value (DANTE = ANDANTE["slowish"] minus "a name" = A+N). The single-namer engine above
    # cannot do this. Named + exact + adjacent + fully-accounted, so nothing is invented.
    from core.charade_multi_named_deletion_engine import solve_charade_multi_named_deletion
    pcmnd = solve_charade_multi_named_deletion(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["all_values"],
        wiring["is_link"], wiring["indicator_types"],
        define_fallback=wiring.get("define_fallback"), is_dbe=wiring.get("is_dbe"))
    if pcmnd is not None and pcmnd.status in ("pass", "pending"):
        return _finish(pcmnd, "charade_multi_named_deletion", ctx, wiring, source,
                       puzzle_number, clue_id)

    # CHARADE+HOLLOW — a charade where ONE piece is a hollowed word: its outer shell only
    # (first+last, the inside emptied), AMEER = AM + E[xercis]E + R. The charade+deletion
    # engine cannot do this (its backward reconstruction restores only a bounded affix, not
    # an arbitrary removed middle); this stage matches the shell FORWARD. Gated on a hollow
    # ('empty' sub-type) deletion indicator. Tried with the charade+deletion family.
    from core.charade_hollow_engine import solve_charade_hollow
    pchh = solve_charade_hollow(ctx, wiring["defines"], wiring["lookup_all"],
                                wiring["is_link"], wiring["indicator_types"],
                                wiring["deletion_subtypes"],
                                define_fallback=wiring.get("define_fallback"),
                                is_dbe=wiring.get("is_dbe"))
    if pchh is not None and pchh.status in ("pass", "pending"):
        return _finish(pchh, "charade_hollow", ctx, wiring, source, puzzle_number, clue_id)

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
        return _finish(pcon, "container", ctx, wiring, source, puzzle_number, clue_id)

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
        return _finish(pccc, "container_charade", ctx, wiring, source, puzzle_number, clue_id)

    # CONTAINER-OF-CHARADE — an OUTER DB value wrapped around an INNER that is itself a
    # charade of 2+ DB values (LIMESTONE = LONE around IM+EST). The mirror of the
    # container+charade above: there the charade is outside one container piece; here the
    # whole answer is one container whose inner is the charade. Tried after both, gated on a
    # container indicator + exact reconstruction. Gaps -> preserved fail-evidence.
    from core.container_inner_charade_engine import solve_container_inner_charade
    pcic = solve_container_inner_charade(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], define_fallback=wiring.get("define_fallback"),
        is_dbe=wiring.get("is_dbe"))
    if pcic is not None and pcic.status in ("pass", "pending"):
        return _finish(pcic, "container_inner_charade", ctx, wiring, source, puzzle_number, clue_id)

    # CONTAINER-WITH-CHARADE-OUTER — the MIRROR of container-of-charade: an OUTER that is
    # itself a charade of 2+ DB values, wrapped around a single INNER value (DUB = (D+B)
    # around U; "Germany"=D, "Britain"=B, "university"=U, "in"=insertion). The plain
    # container needs a single-value outer and container_inner_charade puts the charade on
    # the INNER; neither covers a two-piece outer. Answer-driven, gated on a container
    # indicator. Tried with the container family. Gaps -> preserved fail-evidence.
    from core.container_outer_charade_engine import solve_container_outer_charade
    pcoc = solve_container_outer_charade(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], define_fallback=wiring.get("define_fallback"),
        is_dbe=wiring.get("is_dbe"))
    if pcoc is not None and pcoc.status in ("pass", "pending"):
        return _finish(pcoc, "container_outer_charade", ctx, wiring, source, puzzle_number, clue_id)

    # CONTAINER-OF-ACROSTIC — an OUTER DB value wrapped around an INNER formed by acrostic
    # letter-selection (MESCAL = MEAL around S,C = initials of "Served Cold"). The container
    # engines insert a DB value and the acrostic engine spells the whole answer; neither
    # inserts an acrostic. Answer-driven, gated on BOTH a container and an acrostic
    # indicator. Tried with the container family. Gaps -> preserved fail-evidence.
    from core.container_acrostic_engine import solve_container_acrostic
    pca = solve_container_acrostic(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], define_fallback=wiring.get("define_fallback"),
        is_dbe=wiring.get("is_dbe"))
    if pca is not None and pca.status in ("pass", "pending"):
        return _finish(pca, "container_acrostic", ctx, wiring, source, puzzle_number, clue_id)

    # CONTAINER-WITH-DELETED-INNER — an OUTER DB value wrapped around an INNER that is a DB
    # value with a POSITIONAL deletion applied before insertion (ASPIC = AC around SPI[SPIN
    # "briefly"->curtail]; LIMBURGER = LIMBER around URG[URGE "for the most part"->curtail]).
    # container_deletion deletes from the WHOLE assembled string, not from the inner piece, so
    # it cannot reach these. Answer-driven (outer exact + inner == op(V) exact), gated on BOTH
    # a container and a deletion indicator. Returns PASS-only so it never displaces a simpler
    # engine's pending/fail. Tried with the container family.
    from core.container_inner_deletion_engine import solve_container_inner_deletion
    pcid = solve_container_inner_deletion(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], wiring["deletion_subtypes"],
        define_fallback=wiring.get("define_fallback"), is_dbe=wiring.get("is_dbe"))
    if pcid is not None and pcid.status in ("pass", "pending"):
        return _finish(pcid, "container_inner_deletion", ctx, wiring, source, puzzle_number, clue_id)

    # CONTAINER-WITH-ALTERNATION-INNER — an OUTER DB value wrapped around an INNER that is the
    # alternate (every-other) letters of a single clue word (PRISONER = PRIER around SON, where
    # SON = "scor[n] at intervals"). The alternation sibling of container_inner_deletion; none
    # of the other container engines inserts an alternation selection. Answer-driven (outer
    # exact + inner == a word's alternate letters exactly), gated on BOTH a container and an
    # alternation indicator. PASS-only. Tried with the container family.
    from core.container_inner_alternation_engine import solve_container_inner_alternation
    pcia = solve_container_inner_alternation(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], define_fallback=wiring.get("define_fallback"),
        is_dbe=wiring.get("is_dbe"), selection_rules=wiring.get("selection_rules"))
    if pcia is not None and pcia.status in ("pass", "pending"):
        return _finish(pcia, "container_inner_alternation", ctx, wiring, source, puzzle_number, clue_id)

    # CONTAINER (DELETION OUTER + SELECTION INNER) — the hardest container: BOTH pieces built.
    # OUTER is a positional deletion of a synonym, INNER is a letter-selection of a word
    # (RIVEN = RIEN[FRIEND "discovered"=remove ends] around V["valuables primarily"]). No other
    # container engine builds both pieces. Tightly gated: requires container + deletion +
    # selection indicators (all distinct), both pieces answer-driven exact, every word
    # accounted; PASS-only. Tried last in the container family (most specific/expensive).
    from core.container_deletion_selection_engine import solve_container_deletion_selection
    pcds = solve_container_deletion_selection(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], wiring["deletion_subtypes"], wiring["selection_rules"],
        define_fallback=wiring.get("define_fallback"), is_dbe=wiring.get("is_dbe"))
    if pcds is not None and pcds.status in ("pass", "pending"):
        # REVIEW-GATED: this two-built-piece engine is the riskiest in the family, so a clean
        # pass is downgraded to a review-pending (human confirms before it is trusted).
        from core import review_gate
        review_gate.gate(pcds, "container_deletion_selection")
        return _finish(pcds, "container_deletion_selection", ctx, wiring, source, puzzle_number, clue_id)

    # CHARADE with a CONTAINER piece whose INNER is a letter-SELECTION — the insertion mirror of
    # the selection-DELETION engines (TENDERHEARTED = TENDER + [HEATED around R["our","last"]];
    # COPSHOP = [COSH around P] + OP). Plain outer (vs container_deletion_selection's deleted
    # outer) but composed inside a charade. Tightly gated: container/insertion indicator + a
    # selection indicator ADJACENT to its word (links may intervene), inner = exact rule letters,
    # outer = exact DB value, every word accounted; answer-driven; PASS-only.
    from core.charade_container_selection_engine import solve_charade_container_selection
    pccs = solve_charade_container_selection(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], wiring["selection_rules"],
        define_fallback=wiring.get("define_fallback"), is_dbe=wiring.get("is_dbe"))
    if pccs is not None and pccs.status in ("pass", "pending"):
        return _finish(pccs, "charade_container_selection", ctx, wiring, source, puzzle_number, clue_id)

    # NESTED CONTAINER — a container inside a container (VACUUM = VAM["5am"] around
    # [CU["Copper"] around U["university"]]; FALLENANGEL = FL["Florida"] around
    # [ALLEGE["claim"] around NAN["relative"]]). The plain container engines insert ONE
    # value into another; this is a DOUBLE insertion (three DB values, two genuine wraps).
    # Requires >=2 container indicators, so it cannot intercept a single-container clue.
    # Answer-driven; returns PASS-only so it never displaces a simpler engine's pending/
    # fail. Tried with the container family.
    from core.nested_container_engine import solve_nested_container
    pnc = solve_nested_container(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], define_fallback=wiring.get("define_fallback"),
        is_dbe=wiring.get("is_dbe"))
    if pnc is not None and pnc.status in ("pass", "pending"):
        return _finish(pnc, "nested_container", ctx, wiring, source, puzzle_number, clue_id)

    # REVERSED-OUTER CONTAINER — a container whose OUTER is a reversed synonym, wrapped
    # around a charade inner (EMPEROR = EOR[caviar->ROE, "flipping"] around MPER[MP+ER],
    # "tucked into"). reversal_container reverses the INNER; container_inner_charade has a
    # plain outer; neither reverses the outer. Gated on BOTH a reversal and a container
    # indicator (multi-word aware), answer-driven, PASS-only.
    from core.reversed_outer_container_engine import solve_reversed_outer_container
    proc = solve_reversed_outer_container(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], define_fallback=wiring.get("define_fallback"),
        is_dbe=wiring.get("is_dbe"))
    if proc is not None and proc.status in ("pass", "pending"):
        return _finish(proc, "reversed_outer_container", ctx, wiring, source, puzzle_number, clue_id)

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
        return _finish(prev, "reversal", ctx, wiring, source, puzzle_number, clue_id)

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
        return _finish(prevc, "reversal_charade", ctx, wiring, source, puzzle_number, clue_id)

    # REVERSAL+CHARADE (evidence-driven) — the WHOLE charade reversed, where the piece order
    # flips and the signature form above cannot encode it (NARRATIVE = rev(EVITA+RR+A+N) =
    # rev(musical) tail + repeatedly-runs + a + new). The order-free tiler handles it; it was
    # unwired when the reversal family went signature-driven (e5c726f6), but the signature
    # never encodes this sub-form. Tried AFTER the signature engine (which still claims what it
    # can); gated on a reversal indicator, answer-driven, >=1 piece reversed. The signature
    # engine is untouched.
    from core.reversal_charade_engine import solve_reversal_charade as solve_reversal_charade_ev
    prevce = solve_reversal_charade_ev(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], define_fallback=wiring.get("define_fallback"),
        is_dbe=wiring.get("is_dbe"))
    if prevce is not None and prevce.status in ("pass", "pending"):
        return _finish(prevce, "reversal_charade_evidence", ctx, wiring, source, puzzle_number, clue_id)

    # SELECTION + REVERSAL CHARADE — a charade mixing a letter-SELECTION piece and a
    # REVERSED piece (TRAIL = T[end of "account"] + RAIL[reverse of LIAR "storyteller"]).
    # charade_signature does selection (SEL_F) and reversal_charade does reversal (REV_F),
    # but neither combines them in one charade. Requires >=1 selection AND >=1 reversal, so
    # it cannot intercept a plain charade, reversal_charade, or selection charade (the
    # earlier engines claim those). Gated on BOTH a reversal and a selection indicator,
    # answer-driven. Tried right after the reversal_charade family.
    from core.selection_reversal_charade_engine import solve_selection_reversal_charade
    psrc = solve_selection_reversal_charade(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], define_fallback=wiring.get("define_fallback"),
        is_dbe=wiring.get("is_dbe"))
    if psrc is not None and psrc.status in ("pass", "pending"):
        return _finish(psrc, "selection_reversal_charade", ctx, wiring, source, puzzle_number, clue_id)

    # REVERSE-OF-CHARADE — the WHOLE assembled charade is reversed (TRAIN = reverse(new->N +
    # international->I + art->ART) = reverse(NIART)). reversal_charade reverses ONE piece in
    # place and tiles in clue order; it cannot reach a whole-charade reversal, which flips the
    # piece order. Requires >=2 pieces + a reversal indicator; answer-driven (forward fodder
    # must spell reverse(answer)). Tried after the reversal_charade family (more general).
    from core.reverse_charade_engine import solve_reverse_charade
    prevwc = solve_reverse_charade(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], define_fallback=wiring.get("define_fallback"),
        is_dbe=wiring.get("is_dbe"))
    if prevwc is not None and prevwc.status in ("pass", "pending"):
        return _finish(prevwc, "reverse_charade", ctx, wiring, source, puzzle_number, clue_id)

    # REVERSAL + CONTAINER — an outer DB value wrapping an inner DB value that is REVERSED
    # before insertion (ARABS = AS around reverse(BAR)). The plain container inserts values
    # as-is and the reversal engines concatenate; neither covers a reversed inner inside a
    # container. Evidence-driven, gated on BOTH a container and a reversal indicator, and
    # answer-driven (exact reconstruction), so it cannot intercept another type. Tried after
    # the reversal family, before deletion.
    from core.reversal_container_engine import solve_reversal_container
    prevcon = solve_reversal_container(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], define_fallback=wiring.get("define_fallback"),
        is_dbe=wiring.get("is_dbe"))
    if prevcon is not None and prevcon.status in ("pass", "pending"):
        return _finish(prevcon, "reversal_container", ctx, wiring, source, puzzle_number, clue_id)

    # REVERSAL + DELETION — the whole answer is ONE synonym with letters removed and reversed
    # (SLAB = reverse(curtail(BALSA)): "wood"=BALSA, "cut"=deletion, "after turning"=reversal).
    # The plain reversal reverses a whole DB value and the plain deletion deletes from one;
    # neither composes the two on a single piece. Gated on BOTH a reversal and a deletion
    # indicator, answer-driven (delete+reverse, either order, == answer exactly), so it cannot
    # intercept a plain reversal or deletion. A fresh stage; never edits those engines.
    from core.reversal_deletion_engine import solve_reversal_deletion
    prevdel = solve_reversal_deletion(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["is_link"],
        wiring["indicator_types"], wiring["deletion_subtypes"],
        define_fallback=wiring.get("define_fallback"), is_dbe=wiring.get("is_dbe"))
    if prevdel is not None and prevdel.status in ("pass", "pending"):
        return _finish(prevdel, "reversal_deletion", ctx, wiring, source, puzzle_number, clue_id)

    # DELETION — a plain deletion (the whole answer is one DB value with letters removed).
    # EVIDENCE-DRIVEN, two honestly-attributed forms: POSITIONAL (a fused/position-noun
    # indicator fixes which letters go: TAU = curtail(TAUT)) and NAMED (the removed letters
    # are a DB value of another word: LOTTO = BLOTTO - B[bishop]). Answer-driven, gated on a
    # genuine deletion indicator. Tried after the reversal family; not yet signature-driven.
    # CATALOG-DRIVEN by the GENERIC verifier (core.signature_verifier): it reads each
    # deletion recipe's persisted assembly+structure and drives the composable operation
    # engines (pos_delete / named_delete) + the 'single' assembly. This is the first clue
    # type on the operation/assembly architecture (documents/OPERATION_ASSEMBLY_SCHEMA.md);
    # it reproduced the deletion signature engine 3000/3000 identical, which is itself
    # proven == the bespoke engine. The bespoke core.deletion_engine is retired from the
    # cascade (kept for its _build/_verify helpers + as the A/B override via deletion_solve);
    # core.deletion_signature_engine is superseded (kept for _del_ops, used by the verifier).
    if deletion_solve is None:
        from core.signature_verifier import solve_deletion as deletion_solve
    pdel = deletion_solve(ctx, wiring["defines"], wiring["lookup_all"],
                          wiring["is_link"], wiring["deletion_subtypes"],
                          templates=wiring.get("deletion_templates"),
                          define_fallback=wiring.get("define_fallback"),
                          is_dbe=wiring.get("is_dbe"),
                          loc_rules=wiring.get("selection_rules"))
    if pdel is not None and pdel.status in ("pass", "pending"):
        return _finish(pdel, "deletion", ctx, wiring, source, puzzle_number, clue_id)

    # SUBSTITUTION — a base value with one clued letter replaced by another (INSOLENCE =
    # IN SILENCE with one[I] -> love[O]). Both letters come from the wordplay table; gated
    # on a substitution indicator and answer-driven, so it cannot fabricate. Whole-answer
    # form only for now. Tried after the deletion family, before DD.
    from core.substitution_engine import solve_substitution
    psub = solve_substitution(ctx, wiring["defines"], wiring["lookup"],
                              wiring["synonyms_of"], wiring["is_link"],
                              wiring["indicator_types"],
                              define_fallback=wiring.get("define_fallback"),
                              is_dbe=wiring.get("is_dbe"))
    if psub is not None and psub.status in ("pass", "pending"):
        return _finish(psub, "substitution", ctx, wiring, source, puzzle_number, clue_id)

    # LOCATED SUBSTITUTION — a synonym base with its FIRST or LAST letter replaced by a
    # clued value (DRESSES = TRESSES["Hair"], initial letter cut, D["diamonds"] instead).
    # The plain substitution above swaps two clued letters anywhere; it cannot reach a
    # removal at a LOCATED position whose removed letter is named only by a position
    # indicator ("initially"). Gated on BOTH a substitution indicator and a first/last
    # selection indicator, answer-driven. Tried right after the plain substitution.
    from core.located_substitution_engine import solve_located_substitution
    plsub = solve_located_substitution(
        ctx, wiring["defines"], wiring["lookup_all"], wiring["synonyms_of"],
        wiring["is_link"], wiring["indicator_types"], wiring["selection_rules"],
        define_fallback=wiring.get("define_fallback"), is_dbe=wiring.get("is_dbe"))
    if plsub is not None and plsub.status in ("pass", "pending"):
        return _finish(plsub, "substitution", ctx, wiring, source, puzzle_number, clue_id)

    # (No free-tiling fallback. The cascade is signature-first: a clue with no matching
    # signature falls through to DD and then to the most-complete fail — it is NOT
    # free-tiled. The free-tiling evidence pass was removed because it produced
    # letter-correct but fabricated attributions, e.g. MANITOBA's "I <- on" — pinning a
    # value onto a word that does not produce it, which a signature must never allow.)

    # ANAGRAM + MULTIPLE SUBSTITUTIONS — an anagram whose fodder mixes a literal with TWO OR
    # MORE substituted words (SCIROCCO = anag of CIRCS + O[ld] + [firm]CO). The single-sub
    # engine holds out one word; this one supplies the residual from >=2 short, genuine
    # substitutions, treating a leftover container/link word as inert. Run LATE (here, with
    # DD): it is a LOOSER reading than the precise compound engines (anagram+charade,
    # anagram+container), so they must claim first or it mis-attributes their clues as plain
    # anagrams. Gated on an anagram indicator, answer-driven.
    from core.anagram_multi_substitution_engine import solve_anagram_multi_substitution
    pamus = solve_anagram_multi_substitution(
        ctx, wiring["defines"], wiring["all_values"], wiring["indicator_types"],
        wiring["is_link"], define_fallback=wiring.get("define_fallback"),
        is_dbe=wiring.get("is_dbe"))
    if (pamus is not None and pamus.status in ("pass", "pending")
            and not _anagram_degenerate(pamus)):
        return _finish(pamus, "anagram_multi_substitution", ctx, wiring, source, puzzle_number, clue_id)

    # ANAGRAM + DELETION — an anagram whose fodder has letters REMOVED before the anagram:
    # a NAMED letter (EDELWEISS = anag of SEEDS+WHILE - H["hard","missing"]) or a CURTAILED
    # fodder word (BESMEAR = anag of BRA + SEEM["seems almost"]). The mirror of the
    # substitution engines (fodder minus removed = answer). Gated on BOTH an anagram and a
    # deletion indicator, the deleted/curtailed word ADJACENT to the deletion indicator, and
    # answer-driven (exact). Run LATE with the other modified-fodder anagram readings.
    from core.anagram_deletion_engine import solve_anagram_deletion
    pad = solve_anagram_deletion(
        ctx, wiring["defines"], wiring["all_values"], wiring["indicator_types"],
        wiring["is_link"], deletion_subtypes=wiring.get("deletion_subtypes"),
        define_fallback=wiring.get("define_fallback"), is_dbe=wiring.get("is_dbe"))
    if pad is not None and pad.status in ("pass", "pending") and not _anagram_degenerate(pad):
        return _finish(pad, "anagram_deletion", ctx, wiring, source, puzzle_number, clue_id)

    # ANAGRAM + SELECTION-DELETION — like anagram+deletion, but the removed letter is a
    # SELECTION of an adjacent word, not an abbreviation (IN THE RAW = anag(NIGHTWEAR - G),
    # G = "last of alluring"). Triple-gated (anagram + deletion + selection indicator),
    # answer-driven (fodder minus the selected letters == answer exactly), so it cannot
    # intercept a plain anagram. A fresh stage — never edits anagram_deletion.
    from core.anagram_selection_deletion_engine import solve_anagram_selection_deletion
    pasd = solve_anagram_selection_deletion(
        ctx, wiring["defines"], wiring["indicator_types"], wiring["is_link"],
        wiring["selection_rules"], define_fallback=wiring.get("define_fallback"),
        is_dbe=wiring.get("is_dbe"))
    if (pasd is not None and pasd.status in ("pass", "pending")
            and not _anagram_degenerate(pasd)):
        return _finish(pasd, "anagram_selection_deletion", ctx, wiring, source, puzzle_number, clue_id)

    # ANAGRAM CONTAINING A SELECTED LETTER — an anagram of a fodder run with a single first/
    # last letter inserted (OYSTER = anag(STORY) containing E["beginning to emerge"], "in").
    # Triple-gated (anagram + container + first/last-letter indicator), answer-driven, so it
    # cannot intercept a plain anagram. Run with the other modified-fodder anagram readings.
    from core.anagram_insert_letter_engine import solve_anagram_insert_letter
    pail = solve_anagram_insert_letter(
        ctx, wiring["defines"], wiring["indicator_types"], wiring["is_link"],
        define_fallback=wiring.get("define_fallback"), is_dbe=wiring.get("is_dbe"))
    if pail is not None and pail.status in ("pass", "pending") and not _anagram_degenerate(pail):
        return _finish(pail, "anagram_insert_letter", ctx, wiring, source, puzzle_number, clue_id)

    # DOUBLE DEFINITION — run LAST, not first. Its second-definition check (esp. the
    # Haiku half) is softer than the catalog engines, which reconstruct the answer
    # exactly; running it first let it intercept catalog clues. So the precise engines
    # claim first and DD catches only the residue. A pass/pending stops here.
    from core.dd_engine import solve_dd
    pd = solve_dd(ctx, wiring["defines"], is_link=wiring["is_link"],
                  indicator_types=wiring["indicator_types"],
                  ai_is_definition=wiring.get("ai_is_definition"),
                  is_dbe=wiring.get("is_dbe"))
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
            return _finish(pcr, "charade", ctx, wiring, source, puzzle_number, clue_id)
        pacr = solve_anagram_charade(ctx, wiring["defines"], wiring["lookup"],
                                     wiring["is_link"], wiring["indicator_types"],
                                     wiring.get("anagram_charade_templates") or [],
                                     define_fallback=wiring.get("define_fallback"),
                                     is_dbe=wiring.get("is_dbe"), suggest_piece=sp)
        if pacr is not None and pacr.status in ("pass", "pending"):
            return _finish(pacr, "anagram_charade", ctx, wiring, source, puzzle_number,
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
                return _finish(pacor, "anagram_container", ctx, wiring, source, puzzle_number,
                               clue_id)

    # PENDING-ONLY SIGNATURES — the final signature stage (signature-tiers build §5 Step 4,
    # the core). Runs the pending-tier signatures AFTER every PASS-capable engine; any match
    # is CAPPED to 'pending' (never a pass). Reached only when everything above failed, so it
    # can only ever turn FAIL -> PENDING — it can never touch a pass, pre-empt a better
    # engine, or mint a false pass (safety by construction). Placed before the
    # cryptic-definition fallback so a wordplay-based provisional beats a bare CD pending.
    ppend = _solve_pending_signatures(ctx, wiring, source, puzzle_number, clue_id)
    if ppend is not None:
        return ppend

    # CRYPTIC DEFINITION — the LAST resort: no wordplay engine and not DD. If the WHOLE
    # clue is a recorded definition of the answer (scraped or hand-entered), treat it as a
    # cryptic definition. A CD has nothing to reconstruct, so it is NEVER auto-confirmed —
    # always 'pending', promoted to 'pass' only by a human (UI verdict override).
    from core.cryptic_definition_engine import solve_cryptic_definition
    pcd = solve_cryptic_definition(ctx, wiring["defines"],
                                   comment=wiring.get("cd_comment"))
    if pcd is not None and pcd.status in ("pass", "pending"):
        return _finish(pcd, "cd", ctx, wiring, source, puzzle_number, clue_id)

    # Nothing produced a clean stop. The GATED engines (spoonerism / palindrome /
    # acrostic) only return a parse when their indicator actually fired, so a non-None
    # fail from one means the clue IS that type — its evidence (the indicator + the
    # attempted reading) is the relevant thing to SHOW, ahead of a generic anagram/charade
    # fodder guess that knows nothing about the indicator. Preserve it (design: never drop
    # fail evidence; the indicator must survive).
    # A demoted hidden PENDING is preserved as the result UNLESS a pass/pending engine
    # already claimed the clue earlier (in which case we returned before reaching here).
    # Placed BEFORE the gated fail-evidence so a hidden pending can only be SUPERSEDED by a
    # genuine pass/pending, never DOWNGRADED to a fail; for clues no between-engine claims,
    # behaviour is identical to the old hidden-first cascade (the hidden pending still shows).
    if hidden_fallback is not None:
        return _finish(hidden_fallback, "hidden", ctx, wiring, source, puzzle_number, clue_id)

    for p, n in ((pspoon, "spoonerism"), (ppal, "palindrome"), (pacro, "acrostic")):
        if p is not None:
            return _finish(p, n, ctx, wiring, source, puzzle_number, clue_id)

    # Otherwise return the genuinely MOST COMPLETE fail so the richest evidence is shown
    # — measured (status, answer letters explained, clue words accounted, fewest
    # warnings), NOT by engine order.
    # ALL engines' fails are eligible (the free-tiling guesser was removed, so every remaining
    # engine is answer-/signature-driven — its fail is an honest partial, safe to rank by
    # completeness). Excludes pcd (that variable is REUSED for cryptic_definition below, and a
    # CD is pending-or-None, never a fail). _most_complete picks the richest evidence.
    candidates = [(p, n) for p, n in (
        (pd, "dd"), (pa, "anagram"), (pasub, "anagram_substitution"),
        (pamus, "anagram_multi_substitution"), (pad, "anagram_deletion"),
        (pasd, "anagram_selection_deletion"), (pail, "anagram_insert_letter"),
        (pc, "charade"), (ppos, "charade_positional"),
        (pposl, "charade_positional_local"), (pcac2, "charade_acrostic"),
        (pcca, "charade_container_acrostic"), (pcalt, "charade_alternation"),
        (pac, "anagram_charade"), (paco, "anagram_container"),
        (pcac, "charade_anagram_container"), (pchd, "charade_deletion"),
        (pcmd, "charade_multi_deletion"), (pcnd, "charade_named_deletion"),
        (pcmnd, "charade_multi_named_deletion"), (pchh, "charade_hollow"),
        (pcon, "container"), (pccc, "container_charade"),
        (pcic, "container_inner_charade"), (pcoc, "container_outer_charade"),
        (pca, "container_acrostic"), (pcid, "container_inner_deletion"),
        (pcia, "container_inner_alternation"),
        (pcds, "container_deletion_selection"),
        (pccs, "charade_container_selection"), (pnc, "nested_container"),
        (proc, "reversed_outer_container"), (prev, "reversal"),
        (prevc, "reversal_charade"), (prevce, "reversal_charade_evidence"),
        (psrc, "selection_reversal_charade"), (prevwc, "reverse_charade"),
        (prevcon, "reversal_container"), (prevdel, "reversal_deletion"),
        (pdel, "deletion"), (psub, "substitution"), (plsub, "substitution"),
        (phom, "homophone"), (phomc, "charade_homophone"), (palt, "alternation"))
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


def _anagram_degenerate(parse):
    """True if an 'anagram' parse is not really an anagram: the wordplay letters (in CLUE
    order) equal the answer FORWARD (identity — a charade of literal letters, no rearrange)
    or REVERSED (a reversal — belongs to the reversal engine). A genuine anagram rearranges
    the fodder into something that is NEITHER. E.g. TRAIN from "new international art" =
    N·I·ART, whose reverse IS TRAIN -> a reversal, not an anagram. Gated to
    operation=='anagram' so compound ops are untouched; returning True lets the cascade fall
    through to the reversal family instead of minting a false anagram PASS."""
    if parse is None or (parse.operation or "") != "anagram":
        return False
    a = (parse.answer_letters() or "").upper().replace(" ", "").replace("-", "")
    srcs = sorted(parse.sources,
                  key=lambda s: min(s.clue_atom_ids) if s.clue_atom_ids else 0)
    src = "".join((s.value or "") for s in srcs).upper().replace(" ", "").replace("-", "")
    return bool(src) and (src == a or src == a[::-1])


def _solve_pending_signatures(ctx, wiring, source, puzzle_number, clue_id):
    """FINAL PENDING-only signature stage (signature-tiers build §5 Step 4 — the core).

    Runs the template-driven signature engines against the PENDING-tier templates only, and
    CAPS any match to status='pending'. Placed AFTER every PASS-capable engine and reached
    only when they all failed (the cascade returns on any earlier pass/pending), so a match
    here can only ever convert FAIL -> PENDING: it can never touch an existing pass, pre-empt
    a better engine, or mint a false pass. The ordering IS the safety guarantee (soundness by
    construction). Returns (parse, name) or None.

    With no pending-tier templates loaded, every `if t` guard is false and the whole stage is
    a fast no-op — so behaviour is UNCHANGED until a pending signature is filed."""
    pend = wiring.get("pending_templates") or {}
    if not pend:
        return None

    df = wiring.get("define_fallback")
    dbe = wiring.get("is_dbe")

    def _cap(parse, name):
        # Only a clean reconstruction (pass) or an otherwise-clean pending is surfaced; a
        # fail near-miss is dropped (return None) so the cascade falls through to the CD
        # fallback. A surfaced parse is FORCED to 'pending' — a pending-only signature can
        # never mint a pass.
        if parse is not None and parse.status in ("pass", "pending"):
            parse.status = "pending"
            parse.warnings = list(parse.warnings) + [
                "matched a NEW signature awaiting review — check this solve; if it is "
                "right, set Status to PASS. The triage page's 'Regression-check pending "
                "signatures' button promotes reviewed signatures for future puzzles"]
            return _finish(parse, name, ctx, wiring, source, puzzle_number, clue_id)
        return None

    t = pend.get("charade")
    if t:
        from core.charade_signature_engine import solve_charade
        r = _cap(solve_charade(ctx, wiring["defines"], wiring["lookup"], wiring["is_link"],
                               t, define_fallback=df, is_dbe=dbe), "charade")
        if r is not None:
            return r

    t = pend.get("anagram")
    if t:
        from core.anagram_signature_engine import solve_anagram
        r = _cap(solve_anagram(ctx, wiring["defines"], wiring["is_link"],
                               wiring["indicator_types"], t,
                               define_fallback=df, is_dbe=dbe), "anagram")
        if r is not None:
            return r

    t = pend.get("anagram_charade")
    if t:
        from core.anagram_charade_signature_engine import solve_anagram_charade
        r = _cap(solve_anagram_charade(ctx, wiring["defines"], wiring["lookup"],
                                       wiring["is_link"], wiring["indicator_types"], t,
                                       define_fallback=df, is_dbe=dbe), "anagram_charade")
        if r is not None:
            return r

    t = pend.get("anagram_container")
    if t:
        from core.anagram_container_signature_engine import solve_anagram_container
        r = _cap(solve_anagram_container(ctx, wiring["defines"], wiring["lookup_all"],
                                         wiring["is_link"], wiring["indicator_types"], t,
                                         define_fallback=df, is_dbe=dbe), "anagram_container")
        if r is not None:
            return r

    t = pend.get("container")
    if t:
        from core.container_signature_engine import solve_container
        r = _cap(solve_container(ctx, wiring["defines"], wiring["lookup_all"],
                                 wiring["is_link"], wiring["indicator_types"], t,
                                 define_fallback=df, is_dbe=dbe), "container")
        if r is not None:
            return r

    t = pend.get("container_charade")
    if t:
        from core.container_charade_signature_engine import solve_container_charade
        r = _cap(solve_container_charade(ctx, wiring["defines"], wiring["lookup_all"],
                                         wiring["is_link"], wiring["indicator_types"], t,
                                         define_fallback=df, is_dbe=dbe), "container_charade")
        if r is not None:
            return r

    t = pend.get("reversal")
    if t:
        from core.reversal_signature_engine import solve_reversal
        r = _cap(solve_reversal(ctx, wiring["defines"], wiring["lookup_all"],
                                wiring["is_link"], wiring["indicator_types"], t,
                                define_fallback=df, is_dbe=dbe), "reversal")
        if r is not None:
            return r

    t = pend.get("reversal_charade")
    if t:
        from core.reversal_charade_signature_engine import solve_reversal_charade
        r = _cap(solve_reversal_charade(ctx, wiring["defines"], wiring["lookup_all"],
                                        wiring["is_link"], wiring["indicator_types"], t,
                                        define_fallback=df, is_dbe=dbe), "reversal_charade")
        if r is not None:
            return r

    t = pend.get("charade_homophone")
    if t:
        from core.charade_homophone_signature_engine import solve_charade_homophone
        r = _cap(solve_charade_homophone(ctx, wiring["defines"], wiring["lookup"],
                                         wiring["is_link"], wiring["indicator_types"], t,
                                         wiring["sounds_alike"], wiring["synonyms_of"],
                                         define_fallback=df, is_dbe=dbe,
                                         suggest_hom=wiring.get("suggest_hom")),
                 "charade_homophone")
        if r is not None:
            return r

    t = pend.get("deletion")
    if t:
        from core.signature_verifier import solve_deletion
        r = _cap(solve_deletion(ctx, wiring["defines"], wiring["lookup_all"],
                                wiring["is_link"], wiring["deletion_subtypes"],
                                templates=t, define_fallback=df, is_dbe=dbe,
                                loc_rules=wiring.get("selection_rules")), "deletion")
        if r is not None:
            return r

    return None


def _finish(parse, name, ctx, wiring, source, puzzle_number, clue_id):
    """Queue any provisional pieces, persist the final Parse, return it.

    Records the SPECIFIC solving engine: every cascade site passes its engine name here
    (no longer the generic "catalog"), and we stamp it onto parse.solved_by so the store's
    wfw_solve.solved_by column says exactly which engine solved each clue — no stack-trace
    hunting. Render keys off parse.operation first, so this stamp is identification only."""
    if parse is not None and name:
        parse.solved_by = name
    # DEFENCE-IN-DEPTH TIER CAP (signature-tiers build §5 Step 5): a PENDING-only signature
    # must never produce a PASS. The final pending stage already caps its own matches, but
    # this is the belt-and-braces backstop — if a pending-tier template EVER reaches the main
    # cascade and passes (e.g. a future loader-filter regression), force the verdict down to
    # 'pending' here, keyed on the parse's template_id -> tier. Runs BEFORE the auto-signature
    # filing below (which is gated on status=='pass'), so a capped parse is never auto-filed.
    if parse is not None and parse.status == "pass":
        _tid = getattr(parse, "template_id", None)
        if _tid is not None and (wiring.get("template_tier") or {}).get(_tid) == "pending":
            parse.status = "pending"
            parse.warnings = list(parse.warnings) + [
                "matched a NEW signature awaiting review — check this solve; if it is "
                "right, set Status to PASS. The triage page's 'Regression-check pending "
                "signatures' button promotes reviewed signatures for future puzzles"]
    # FLOOR GUARD: a GUESSED definition (source='pending' — the no-definition floor edge
    # guess or the Haiku fallback) must NEVER be shown on a FAIL. On a fail the wordplay did
    # not reconstruct the answer, so a guessed edge is a "forced definition with no wordplay"
    # (e.g. ROSEOLA: half the clue claimed as the definition). Drop it — and its now-stale
    # "provisional" warning — BEFORE _finalize_provisional, so the nonsense def is neither
    # shown nor queued for enrichment. A DB ('db') or hand-set ('manual') definition is kept;
    # only the guess is removed, and only on a fail (pending/pass near-solves keep theirs).
    if (parse is not None and parse.status == "fail" and parse.definition is not None
            and getattr(parse.definition, "source", "db") == "pending"):
        parse.definition = None
        parse.warnings = [w for w in parse.warnings
                          if "definition is provisional" not in w]
    _finalize_provisional(parse, ctx, wiring.get("store"), source, puzzle_number)
    # AUTO SIGNATURE-CREATION: a clue that fully PASSED but matched NO catalog signature
    # was solved by the fallback — its decomposition is a shape the catalog is missing.
    # File it so the catalog matches this shape directly next time. Gated on the wiring
    # flag (only interactive/page routes opt in; internal verify/regression solves do
    # not) and on a full pass (the rule: never file a provisional/pending parse). Never
    # let filing break a solve.
    if (wiring.get("auto_signature") and parse is not None
            and parse.status == "pass"
            and getattr(parse, "matched_signature", None) is None):
        try:
            from core.catalog_creator import auto_file_signature
            auto_file_signature(ctx.clue_text, ctx.answer_text, wiring)
        except Exception:
            pass
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
    piece_fallback.finalize_homophone_pieces(parse, ctx, store, source, puzzle_number)
    dd_enrichment.finalize_dd(parse, ctx, store, source, puzzle_number)


def solve_clue_text(clue_text, answer, wiring, source=None, puzzle_number=None,
                    clue_id=None, charade_solve=None, anagram_solve=None,
                    deletion_solve=None, direction=None):
    ctx = build_wfw_atom_context(clue_text, answer, direction=direction)
    parse, name = solve(ctx, wiring, source=source, puzzle_number=puzzle_number,
                        clue_id=clue_id, charade_solve=charade_solve,
                        anagram_solve=anagram_solve, deletion_solve=deletion_solve)
    # GENERAL AUTO-SIGNATURE LOOP: the catalog could not solve it, but if the clue's bits
    # PROVABLY assemble to the answer, create the missing signature and solve. The signature
    # is written ONLY when it produces a verified clean pass (see auto_discover_and_file).
    # Gated on the wiring flag; never lets discovery break a solve.
    if (wiring.get("auto_signature")
            and (parse is None or parse.status not in ("pass", "pending"))):
        try:
            from core.catalog_creator import auto_discover_and_file
            result = auto_discover_and_file(
                ctx, wiring,
                lambda c, wr: solve(c, wr, source=source, puzzle_number=puzzle_number,
                                    charade_solve=charade_solve,
                                    anagram_solve=anagram_solve,
                                    deletion_solve=deletion_solve))   # no clue_id: no persist
            if result is not None:
                parse, name = result
                if clue_id is not None:
                    from core import store as wfw_store
                    wfw_store.persist(clue_id, parse, ctx)
        except Exception:
            pass

    # AUTO-SIGNATURE QUEUE: like the loop above, but instead of FILING the discovered
    # signature it QUEUES the proven shape for human approval (signature_queue). The
    # catalog is untouched and the clue stays unsolved until the signature is approved.
    # Gated on its own flag so interactive routes can opt in without auto-writing the
    # catalog. Never lets discovery break a solve.
    if (wiring.get("auto_signature_queue")
            and (parse is None or parse.status not in ("pass", "pending"))):
        try:
            from core.catalog_creator import auto_discover_and_queue
            auto_discover_and_queue(
                ctx, wiring,
                lambda c, wr: solve(c, wr, source=source, puzzle_number=puzzle_number,
                                    charade_solve=charade_solve,
                                    anagram_solve=anagram_solve,
                                    deletion_solve=deletion_solve),
                clue_id=clue_id)
        except Exception:
            pass
    return ctx, parse, name
