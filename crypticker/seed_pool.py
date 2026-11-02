"""Seed crypticker_build.db.clue_pool from the verified WFW corpus.

Vets every PASS parse against the decided presentation map (memory:
crypticker-presentation, 2026-07-23) and records its kind + difficulty +
static eligibility. DATE-RELATIVE gates (3-month embargo, six-month repeat,
companions) are NOT applied here — they are computed at build time. This pass
records only static facts + curator status.

Read-only against clues_master.db / cryptic_new.db; writes ONLY crypticker_build.db.
Idempotent upsert keyed on clue_id; preserves curator_status + added_at on re-runs.
"""
import sqlite3, os, sys, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.store import load_parse
from crypticker.serve_data import atoms_for
from crypticker.mech_kind import effective_op
from crypticker.convert import convert_clue
from crypticker.serve_gate import check as serve_check
from crypticker.convert import classify

HERE = os.path.dirname(__file__)
DB = os.path.join(HERE, "..", "data", "clues_master.db")
REF = os.path.join(HERE, "..", "data", "cryptic_new.db")
BUILD = os.path.join(HERE, "..", "data", "crypticker_build.db")

# --- presentation map (see memory crypticker-presentation) --------------------
# Operations excluded outright.
EXCLUDE_OPS = {"cd"}                       # cryptic definition: no wordplay
SOUND_MARKERS = ("homophone", "spoonerism")  # any op containing these -> excluded

# Operations that map to a fixed kind (not shape-classified).
FIXED_KIND = {
    "anagram": "anagram",
    "anagram_charade": "anagram_hybrid",
    "anagram_container": "anagram_hybrid",
    "hidden": "hidden",
    "hidden_reversed": "hidden",
    "acrostic": "acrostic",
    "alternation": "alternation",       # alternate letters of a word — different mechanic
    "double_definition": "dd",
}

# static length / shape bounds
ANAGRAM_LEN = (7, 15)
ASSEMBLY_LEN = (4, 12)
ATOM_RANGE = (2, 6)


def _letters(s):
    return "".join(ch for ch in (s or "").upper() if ch.isalpha())


def _dict_count(ref, answer):
    return ref.execute(
        "SELECT count(*) FROM definition_answers_augmented WHERE upper(answer)=?",
        (_letters(answer),)).fetchone()[0]


def _score(kind, natoms, ans_len):
    """Difficulty heuristic (higher = harder). Extends day_builder.difficulty
    across the wider kind set."""
    if kind == "anagram":
        return 5 + 0.8 * ans_len
    if kind == "anagram_hybrid":
        return 8 + 0.5 * ans_len + 1.0 * natoms
    base = {"charade": 0, "reorder": 3, "split": 6,
            "dd": 2, "hidden": 4, "acrostic": 5}.get(kind, 3)
    return base + 1.5 * natoms + 0.4 * ans_len


def vet(conn, ref, clue_id, op, answer):
    """Return dict(kind, mode, difficulty, eligible, reason). eligible=False
    rows are still stored (with reason) so the survey is transparent."""
    ans_letters = _letters(answer)
    n = len(ans_letters)
    multiword = (" " in (answer or "")) or ("-" in (answer or ""))

    # /hs-filed clues are op 'manual': read the kind from the pieces instead
    # (same function convert_clue uses, so pool and board always agree).
    _p = load_parse(conn, clue_id)
    if _p is not None:
        op = effective_op(_p)

    # --- hard exclusions by device ---
    if op in EXCLUDE_OPS:
        return dict(kind=None, mode=None, difficulty=None,
                    eligible=False, reason="excluded_device:%s" % op)
    if any(m in (op or "") for m in SOUND_MARKERS):
        return dict(kind=None, mode=None, difficulty=None,
                    eligible=False, reason="excluded_sound:%s" % op)

    # --- dictionary presence (cheap proper-noun screen; single word only) ---
    if not multiword and _dict_count(ref, answer) == 0:
        return dict(kind=None, mode=None, difficulty=None,
                    eligible=False, reason="not_in_dictionary")

    # --- fixed-kind devices (anagram / hybrids / hidden / acrostic / dd) ---
    if op in FIXED_KIND:
        kind = FIXED_KIND[op]
        if kind == "alternation":       # presentation not built yet — defer
            return dict(kind="alternation", mode=None, difficulty=None,
                        eligible=False, reason="alternation_todo")
        if kind == "dd":
            natoms = max(1, len((answer or "").split()))
            return dict(kind="dd", mode="hard",
                        difficulty=round(_score("dd", natoms, n), 1),
                        eligible=True, reason=None)
        if kind in ("anagram", "anagram_hybrid"):
            if multiword:
                return dict(kind=None, mode=None, difficulty=None,
                            eligible=False, reason="anagram_multiword")
            if not (ANAGRAM_LEN[0] <= n <= ANAGRAM_LEN[1]):
                return dict(kind=None, mode=None, difficulty=None,
                            eligible=False, reason="anagram_len_%d" % n)
            natoms = n
            return dict(kind=kind, mode="hard",
                        difficulty=round(_score(kind, natoms, n), 1),
                        eligible=True, reason=None)
        # hidden / acrostic: atomised + labelled; the mechanism IS revealed, so
        # the answer letters appearing in the clue is expected (leak-exempt).
        parse = load_parse(conn, clue_id)
        if parse is None:
            return dict(kind=None, mode=None, difficulty=None,
                        eligible=False, reason="no_parse")
        atoms, _ = atoms_for(parse)
        natoms = max(1, len(atoms))
        return dict(kind=kind, mode="hard",
                    difficulty=round(_score(kind, natoms, n), 1),
                    eligible=True, reason=None)

    # --- constructive letter devices (charade/container/deletion/reversal/...,
    #     and 'manual'): classified by SHAPE via the link-map ---
    if not (ASSEMBLY_LEN[0] <= n <= ASSEMBLY_LEN[1]):
        return dict(kind=None, mode=None, difficulty=None,
                    eligible=False, reason="len_%d" % n)
    parse = load_parse(conn, clue_id)
    if parse is None:
        return dict(kind=None, mode=None, difficulty=None,
                    eligible=False, reason="no_parse")
    atoms, order = atoms_for(parse)
    if not (ATOM_RANGE[0] <= len(atoms) <= ATOM_RANGE[1]):
        return dict(kind=None, mode=None, difficulty=None,
                    eligible=False, reason="atoms_%d" % len(atoms))
    if all(len(a["true"]) <= 2 for a in atoms):
        return dict(kind=None, mode=None, difficulty=None,
                    eligible=False, reason="degenerate_tiles")
    # leak: a placeable tile whose letters sit verbatim in the clue gives it away
    low = (parse.clue_text or "").lower()
    for a in atoms:
        if len(a["true"]) >= 3 and a["true"].lower() in low:
            return dict(kind=None, mode=None, difficulty=None,
                        eligible=False, reason="leak:%s" % a["true"])
    kind = classify(atoms, order)[0]
    # Charade quality gate: a charade's tiles sit in answer order, so a 2-piece
    # (esp. single-letter-piece) charade is a giveaway. Require >=3 pieces AND at
    # least one piece long enough to jumble (>=3 letters) so it isn't trivial.
    if kind == "charade":
        if len(atoms) < 3:
            return dict(kind=None, mode=None, difficulty=None,
                        eligible=False, reason="charade_lt3_pieces")
        if not any(len(a["true"]) >= 3 for a in atoms):
            return dict(kind=None, mode=None, difficulty=None,
                        eligible=False, reason="charade_no_jumbleable_piece")
    mode = "easy" if kind == "charade" else "hard"
    return dict(kind=kind, mode=mode,
                difficulty=round(_score(kind, len(atoms), n), 1),
                eligible=True, reason=None)


def seed(build_conn, conn, ref, verbose=True):
    today = datetime.date.today().isoformat()
    rows = conn.execute(
        "SELECT clue_id, operation, answer_text, publication_date, "
        "       s.clue_id IN (SELECT clue_id FROM wfw_frozen) "
        "FROM wfw_solve s LEFT JOIN clues c ON c.id=s.clue_id "
        "WHERE s.status='pass'").fetchall()
    import collections
    kept = collections.Counter()
    rej = collections.Counter()
    n_written = 0
    for clue_id, op, answer, pub_date, frozen in rows:
        # FULL WFW PROCESS ONLY (user 2026-10-31): a clue is sourced only if a human
        # confirmed it in /hs (wfw_frozen). Unreviewed engine passes — e.g. ANNEAL,
        # a 2015 archive clue passed by the catalog engine — are never served.
        if not frozen:
            v = dict(kind=None, mode=None, difficulty=None,
                     eligible=False, reason="not_full_wfw")
        else:
            try:
                v = vet(conn, ref, clue_id, op, answer)
            except Exception as e:
                v = dict(kind=None, mode=None, difficulty=None,
                         eligible=False, reason="error:%s" % type(e).__name__)
            if v["eligible"]:
                # Serve gate at seed time: the board as built must be makeable
                # (e.g. a subtractive anagram — TEENTOLD minus D = NOTELET — shows
                # surplus letters and can never be completed). Checked in 'hard'
                # mode: the easy-mode jumble is a presentation choice, gated again
                # at publish.
                try:
                    why = serve_check(convert_clue(conn, clue_id, mode="hard"))
                except Exception as e:
                    why = "convert_error %s" % type(e).__name__
                if why:
                    v = dict(kind=v["kind"], mode=None, difficulty=None, eligible=False,
                             reason="serve_gate:%s" % why[:120])
        if v["eligible"]:
            kept[v["kind"]] += 1
        else:
            rej[v["reason"].split(":")[0]] += 1
        build_conn.execute(
            "INSERT INTO clue_pool "
            "(clue_id, kind, mode, difficulty, pub_date, structurally_eligible, "
            " ineligible_reason, curator_status, added_at) "
            "VALUES (?,?,?,?,?,?,?, 'pending', ?) "
            "ON CONFLICT(clue_id) DO UPDATE SET "
            "  kind=excluded.kind, mode=excluded.mode, difficulty=excluded.difficulty, "
            "  pub_date=excluded.pub_date, structurally_eligible=excluded.structurally_eligible, "
            "  ineligible_reason=excluded.ineligible_reason",
            (clue_id, v["kind"], v["mode"], v["difficulty"], pub_date,
             1 if v["eligible"] else 0, v["reason"], today))
        n_written += 1
    build_conn.commit()
    if verbose:
        print("scanned %d passes, wrote %d pool rows\n" % (len(rows), n_written))
        print("ELIGIBLE by kind:")
        for k, c in kept.most_common():
            print("  %-15s %4d" % (k, c))
        print("  %-15s %4d" % ("TOTAL", sum(kept.values())))
        print("\nREJECTED by reason:")
        for r, c in rej.most_common():
            print("  %4d  %s" % (c, r))


if __name__ == "__main__":
    build_conn = sqlite3.connect(BUILD)
    conn, ref = sqlite3.connect(DB), sqlite3.connect(REF)
    seed(build_conn, conn, ref)
