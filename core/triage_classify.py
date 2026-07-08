"""DETERMINISTIC triage classifier.

Given a clue and the current DB/catalog state, decide WHY it did not cleanly solve using a
FIXED set of mechanical rules — the same clue in the same state always yields the same category
and evidence, regardless of who (or what) runs it. This is the rigid half of the triage process;
the specific missing VALUE (e.g. "French river" = OISE) is NOT decided here — that needs crossword
knowledge and is carried as a separately-labelled human/AI suggestion.

Categories (a clue may collect more than one):
  * missing_data      — a needed fact is absent (no definition / an unresolved word / a homophone
                        entry / one side of a double definition).
  * missing_signature — the pieces assemble via a shape that is NOT in the catalog (+ its tier).
  * missing_engine    — a recognised mechanism with no engine, or material present that nothing
                        known assembles.
  * solves_now        — the pieces, shape and definition are all present, so it should solve; the
                        failed mark is most likely stale. (The tool re-runs these to confirm.)

Instruments, all deterministic: the reference-DB predicates (defines / synonyms / abbreviations /
literals / indicator types); catalog_creator.discover(...) for charade/anagram/container/reversal
shapes; and cheap mechanical detectors for the shapes discover does not try — double definition
(both edges define), acrostic (initial letters spell the answer), hidden (answer sits inside the
clue letters), homophone (a homophone indicator is present).
"""

import os
import sqlite3

from core.diagnose import _hit, _answer_letters

_CLUES_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "clues_master.db")


def _catalog_signatures():
    con = sqlite3.connect(_CLUES_DB)
    try:
        return {r[0] for r in con.execute("SELECT signature FROM catalog_templates")}
    finally:
        con.close()


def _content_words(ctx):
    return [t.text for t in ctx.clue_tokens if t.kind == "word"]


def _defs(words, answer, defines):
    """Confirmed edge definitions as (edge, phrase) — edge in {'start','end'}."""
    n = len(words)
    out = []
    for k in range(1, min(4, n) + 1):
        for edge, phrase in (("start", " ".join(words[:k])), ("end", " ".join(words[n - k:]))):
            try:
                if defines(phrase, answer):
                    out.append((edge, phrase))
            except Exception:
                pass
    return out


def _word_resolves(word, ans, db, indicator_types, is_link):
    """(resolves_to_in_answer_value, is_indicator, is_link) for one clue word."""
    from core import literals
    found = False
    try:
        for s in db.get_synonyms(word)[:60]:
            if _hit(s, ans):
                found = True
                break
    except Exception:
        pass
    if not found:
        try:
            for s in db.get_abbreviations(word):
                if _hit(s, ans):
                    found = True
                    break
        except Exception:
            pass
    if not found:
        lv = literals.literal_value(word)
        if lv and _hit(lv, ans):
            found = True
    try:
        is_ind = bool(indicator_types(word))
    except Exception:
        is_ind = False
    try:
        lk = bool(is_link(word))
    except Exception:
        lk = False
    return found, is_ind, lk


# --- mechanical detectors for shapes discover() does not try ------------------------------

def _acrostic_hit(words, ans):
    """The initial letters of some run of consecutive words spell the answer."""
    if len(ans) < 3:
        return False
    firsts = "".join(w[0].upper() for w in words if w)
    return ans.upper() in firsts


def _hidden_hit(words, ans):
    """The answer sits as a contiguous run inside the clue's letters (and is not itself a whole
    clue word — that would be a coincidence, not a hidden word)."""
    if len(ans) < 3:
        return False
    letters = "".join(w.upper() for w in words)
    if ans.upper() not in letters:
        return False
    return not any(ans.upper() == w.upper() for w in words)


def _homophone_indicator(words, indicator_types):
    for w in words:
        try:
            if "homophone" in (indicator_types(w) or set()):
                return True
        except Exception:
            pass
    return False


def classify(clue_text, answer, wiring, run_discover=True):
    """Deterministic classification: {reasons: [...], detail: {...}}."""
    from core.wfw_atoms import build_wfw_atom_context
    from core import catalog_creator as CC
    ctx = build_wfw_atom_context(clue_text, answer)
    ans = _answer_letters(ctx)
    words = _content_words(ctx)
    db = wiring["db"]
    defines = wiring["defines"]
    indicator_types = wiring["indicator_types"]
    is_link = wiring["is_link"]

    defs = _defs(words, answer, defines)
    edges = {e for e, _p in defs}
    def_words = set()
    for _e, phrase in defs:
        def_words.update(phrase.lower().split())

    reasons, detail = [], {}

    # discover: charade / anagram / container / reversal shapes
    novel, known, cands = [], [], []
    if run_discover:
        have = _catalog_signatures()
        try:
            cands = list(CC.discover(clue_text, answer, wiring))
        except Exception:
            cands = []
        for c in cands:
            try:
                sig = CC._signature_str(c)
            except Exception:
                continue
            (novel if sig not in have else known).append((c, sig))
    if novel:
        reasons.append("missing_signature")
        detail["novel_shapes"] = [(sig, CC.rubric_tier(c, answer)) for c, sig in novel]

    # mechanical detectors for the shapes discover does not try
    dd = ("start" in edges and "end" in edges)          # both edges define -> double definition
    acr = _acrostic_hit(words, ans)
    hid = _hidden_hit(words, ans)
    hom = _homophone_indicator(words, indicator_types)
    if dd:
        detail["double_definition"] = True
    if acr:
        detail["acrostic"] = True
    if hid:
        detail["hidden"] = True
    if hom:
        detail["homophone"] = True

    mechanism_found = bool(known) or dd or acr or hid
    # a definition is present for the purpose of "no definition" if an edge defines OR the clue is
    # a hidden/acrostic whose whole-clue reading is the definition (no clean edge def expected).
    has_def = bool(defs)

    if not novel:
        if mechanism_found and (has_def or hid or acr):
            reasons.append("solves_now")            # pieces+shape+def present -> should solve
        elif not cands:
            # nothing discover-assembles and no complete known mechanism: find the gap
            if not has_def and not (acr or hid):
                reasons.append("missing_data")
                detail["no_definition"] = True
            unresolved = []
            for w in words:
                if w.lower() in def_words:
                    continue
                res, is_ind, lk = _word_resolves(w, ans, db, indicator_types, is_link)
                if not res and not is_ind and not lk:
                    unresolved.append(w)
            if unresolved:
                if "missing_data" not in reasons:
                    reasons.append("missing_data")
                detail["unresolved_words"] = unresolved
            elif has_def and len(defs) == 1 and len(words) <= 5:
                # one side defines, the rest isn't wordplay and doesn't resolve -> the OTHER side
                # is probably a second definition that is missing (a double definition gap).
                reasons.append("missing_data")
                only_edge = defs[0][0]
                detail["possible_double_definition"] = ("other side of %s" % only_edge)
            elif hom:
                reasons.append("missing_data")
                detail["homophone_entry_maybe_missing"] = True
            elif has_def:
                reasons.append("missing_engine")
                detail["material_present_no_assembly"] = True

    if not reasons:
        reasons.append("missing_engine")
        detail["unclassified"] = True
    # de-dup preserving order
    seen, ordered = set(), []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    return {"reasons": ordered, "detail": detail}
