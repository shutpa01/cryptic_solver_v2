"""The §8 catalog-creation method — discover and add the signature a failing clue needs.

Design §8 operationalised. For a clue the catalog cannot yet solve, this tool:

  1. DISCOVERS candidate signatures — it tries every mechanically-valid decomposition
     of the answer over the wordplay (NOT limited to existing signatures): for charade,
     role-pure pieces (synonym / abbreviation / literal) that concatenate to the answer
     with the leftover words as links; for anagram, a fodder run (interior links
     excluded) that anagrams to the answer plus a confirmed indicator. Each candidate
     is the role-sequence + n_words + def-edge it implies, shown with the exact parse
     it would produce, so a human can confirm the intended one (§8 step 1-2).
  2. CLASSIFIES the gap: if the only decompositions found lean on literals / have no
     DB-backed pieces, the gap is likely a missing DB entry, not a missing signature —
     flagged, not silently added.
  3. CREATES the chosen signature in catalog_templates (+ slots), origin 'hand_added',
     after backing up the catalog tables (§8 step 3).
  4. VERIFIES the clue now solves through the REAL cascade, parse shown (§8 step 4).
  5. REGRESSION-CHECKS a sample of the same operation — pass count must not drop (§8 5).
  6. RECORDS the signature (its row + note is the record) (§8 step 6).

Adding a signature is additive and isolated — it cannot change how any other clue is
decomposed, only enable a shape that did not exist. That is the determinism dividend:
a fix here never regresses a working solve, by construction.

CLI:
  python -m core.catalog_creator <clue_id>            # discover + show candidates
  python -m core.catalog_creator <clue_id> --add N    # add candidate N, verify, regress
"""

import os
import sqlite3
import sys

from core.wordplay import raw, is_anagram_indicator, GLUE_POS
from core import contractions, grammar

_CLUES_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                         "clues_master.db")
_MECH_ROLE = {"synonym": "SYN_F", "abbreviation": "ABR_F"}
# (the former MAX_PIECE_WORDS=4 cap on a discovered piece's span was removed — it hid
#  long-phrase pieces like "a female in the family"->AUNT; pieces are now bound only by the
#  wordplay span, and lookup + the exact answer-tiling are the real filters)
MAX_RUN = 5


# ---------------------------------------------------------------- clue loading

def load_clue(clue_id, db_path=None):
    con = sqlite3.connect(db_path or _CLUES_DB)
    try:
        r = con.execute("SELECT clue_text, answer, source, puzzle_number "
                        "FROM clues WHERE id=?", (clue_id,)).fetchone()
    finally:
        con.close()
    return r


# --------------------------------------------------------------- discovery

def _answer_letters(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _prepare(ctx, defines, define_fallback, is_dbe):
    """Definition splits with their wordplay words + POS, as the engines see them."""
    from core.definition_engine import find_definitions
    out = []
    for split in find_definitions(ctx, defines, define_fallback=define_fallback,
                                  is_dbe=is_dbe):
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if not words:
            continue
        postags = grammar.wordplay_pos_tags(ctx, words)
        out.append((split, words, postags))
    return out


def _discover_charade(answer, words, postags, split, lookup, is_link):
    """Yield (roles, n_words, pieces, links) for every role-pure tiling of `answer`
    by the words, leftover words as links. roles like ['SYN_F','LIT_F','ABR_F']."""
    n, N = len(words), len(answer)

    def residue_link(k):
        return (is_link and is_link(words[k].text))

    def role_values(a, b):
        phrase = " ".join(words[k].text for k in range(a, b))
        out, seen = [], set()
        for val, mech in lookup(phrase, answer):
            role = _MECH_ROLE.get(mech)
            v = (val or "").upper()
            if role and v and v in answer and (role, v) not in seen:
                out.append((role, v))
                seen.add((role, v))
        lit = raw(phrase)
        if lit and (("LIT_F", lit) not in seen):
            out.append(("LIT_F", lit))
        return out

    found = []

    def dfs(wi, pos, pieces, gaps):
        if pos == N:
            allgaps = gaps + list(range(wi, n))
            if len(pieces) >= 2 and all(residue_link(k) for k in allgaps):
                found.append((list(pieces), sorted(allgaps)))
            return
        if wi >= n:
            return
        for k in range(1, (n - wi) + 1):     # no artificial cap: bound by the wordplay span
            for role, val in role_values(wi, wi + k):
                if answer.startswith(val, pos):
                    dfs(wi + k, pos + len(val),
                        pieces + [(wi, wi + k, role, val)], gaps)
        dfs(wi + 1, pos, pieces, gaps + [wi])      # skip wi as a gap

    dfs(0, 0, [], [])
    for pieces, gaps in found:
        roles = [r for _, _, r, _ in pieces]
        nwords = [b - a for a, b, _, _ in pieces]
        yield {"operation": "charade", "roles": roles, "n_words": nwords,
               "def_pos": split.where, "pieces": pieces, "links": gaps,
               "split": split, "words": words}


def _discover_anagram(answer, words, postags, split, is_link, indicator_types):
    """Yield anagram candidates: a contiguous fodder run (interior links excluded)
    whose letters anagram to `answer`, plus a confirmed indicator run, in either
    order, leftover words as links."""
    from itertools import combinations
    n, N = len(words), len(answer)

    def residue_link(k):
        return (is_link and is_link(words[k].text))

    def is_ind(k):
        return is_anagram_indicator(words[k].text, indicator_types)

    out = []
    for fa in range(n):
        for fb in range(fa + 1, min(fa + MAX_RUN, n) + 1):
            span = list(range(fa, fb))
            link_in = [k for k in span if is_link and is_link(words[k].text)]
            kept = None
            for r in range(len(link_in) + 1):
                for excl in combinations(link_in, r):
                    es = set(excl)
                    ks = [k for k in span if k not in es]
                    if not ks:
                        continue
                    aw = "".join(raw(words[k].text) for k in ks)
                    st = "".join(raw(contractions.strip_suffixes(words[k].text))
                                 for k in ks)
                    f = next((x for x in (aw, st) if len(x) == N
                              and sorted(x) == sorted(answer) and x[::-1] != answer),
                             None)
                    if f is not None:
                        kept = (ks, list(excl))
                        break
                if kept:
                    break
            if not kept:
                continue
            fodder_idx, fodder_excl = kept
            # indicator run: a contiguous run outside the fodder containing a
            # confirmed indicator; the rest must be links.
            for ia in range(n):
                for ib in range(ia + 1, min(ia + MAX_RUN, n) + 1):
                    irun = list(range(ia, ib))
                    if any(fa <= k < fb for k in irun):
                        continue                    # disjoint from fodder span
                    if not (any(is_ind(k) for k in irun)
                            or is_anagram_indicator(
                                " ".join(words[k].text for k in irun), indicator_types)):
                        continue
                    used = set(span) | set(irun)
                    rest = [k for k in range(n) if k not in used]
                    if not all(residue_link(k) for k in rest):
                        continue
                    links = sorted(set(fodder_excl) | set(rest))
                    roles = (["ANA_F", "ANA_I"] if fa < ia else ["ANA_I", "ANA_F"])
                    nwf, nwi = fb - fa, ib - ia
                    nwords = [nwf, nwi] if fa < ia else [nwi, nwf]
                    out.append({"operation": "anagram", "roles": roles,
                                "n_words": nwords, "def_pos": split.where,
                                "fodder_idx": fodder_idx, "indicator_idx": irun,
                                "links": links, "split": split, "words": words})
    return out


def _discover_reversal_charade(answer, words, split, lookup_all, is_link,
                               indicator_types):
    """Yield reversal_charade candidates: a left-to-right tiling of `answer` by role-pure
    pieces (SYN_F/ABR_F forward, exactly ONE REV_F reversed), with a reversal indicator
    (REV_I) among the leftovers and the remaining leftovers as links."""
    from core.engine_common import find_typed_run
    n, N = len(words), len(answer)

    def is_rev(k):
        try:
            return "reversal" in (indicator_types(words[k].text) or set())
        except Exception:
            return False

    def residue_link(k):
        return bool(is_link and is_link(words[k].text))

    if not any(is_rev(k) for k in range(n)):
        return

    def run_roles(a, b):
        phrase = " ".join(words[k].text for k in range(a, b))
        out, seen = [], set()
        for val, mech in lookup_all(phrase):
            role = _MECH_ROLE.get(mech)
            v = (val or "").upper()
            if role and v and (role, v) not in seen:
                seen.add((role, v))
                out.append((role, v))
        return out

    found = []

    def dfs(wi, pos, pieces, gaps, used_rev):
        if pos == N:
            found.append((list(pieces), sorted(gaps + list(range(wi, n))), used_rev))
            return
        if wi >= n:
            return
        for k in range(1, (n - wi) + 1):     # no artificial cap: bound by the wordplay span
            for role, val in run_roles(wi, wi + k):
                if answer.startswith(val, pos):
                    dfs(wi + k, pos + len(val),
                        pieces + [(wi, wi + k, role, val)], gaps, used_rev)
                if not used_rev:
                    rv = val[::-1]
                    if rv != val and answer.startswith(rv, pos):
                        dfs(wi + k, pos + len(rv),
                            pieces + [(wi, wi + k, "REV_F", val)], gaps, True)
        dfs(wi + 1, pos, pieces, gaps + [wi], used_rev)

    dfs(0, 0, [], [], False)
    for pieces, gaps, used_rev in found:
        if not used_rev or len(pieces) < 2:
            continue
        rev_run = find_typed_run(words, gaps, indicator_types, "reversal", min_length=1)
        if rev_run is None:
            continue
        rev_set = set(rev_run)
        if not all(residue_link(g) for g in gaps if g not in rev_set):
            continue
        slot_items = [(a, b, role) for (a, b, role, _v) in pieces]
        slot_items.append((rev_run[0], rev_run[-1] + 1, "REV_I"))
        slot_items.sort()
        yield {"operation": "reversal_charade",
               "roles": [r for _, _, r in slot_items],
               "n_words": [b - a for a, b, _ in slot_items],
               "def_pos": split.where, "pieces": pieces, "rev_run": rev_run,
               "links": [g for g in gaps if g not in rev_set],
               "split": split, "words": words}


def _discover_anagram_charade(answer, words, split, lookup_all, is_link, indicator_types):
    """Yield anagram_charade candidates: a left-to-right tiling of `answer` by role-pure
    pieces (SYN_F/ABR_F forward, exactly ONE ANA_F anagram piece — a run whose letters
    anagram an answer span), with an anagram indicator (ANA_I) among the leftovers and the
    remaining leftovers as links."""
    from core.engine_common import find_typed_run
    from core.wordplay import raw
    n, N = len(words), len(answer)

    def is_ana(k):
        try:
            return "anagram" in (indicator_types(words[k].text) or set())
        except Exception:
            return False

    def residue_link(k):
        return bool(is_link and is_link(words[k].text))

    if not any(is_ana(k) for k in range(n)):
        return

    def run_roles(a, b):
        phrase = " ".join(words[k].text for k in range(a, b))
        out, seen = [], set()
        for val, mech in lookup_all(phrase):
            role = _MECH_ROLE.get(mech)
            v = (val or "").upper()
            if role and v and (role, v) not in seen:
                seen.add((role, v))
                out.append((role, v))
        return out

    found = []

    def dfs(wi, pos, pieces, gaps, used_ana):
        if pos == N:
            found.append((list(pieces), sorted(gaps + list(range(wi, n))), used_ana))
            return
        if wi >= n:
            return
        for k in range(1, (n - wi) + 1):     # no artificial cap: bound by the wordplay span
            for role, val in run_roles(wi, wi + k):
                if answer.startswith(val, pos):
                    dfs(wi + k, pos + len(val),
                        pieces + [(wi, wi + k, role, val)], gaps, used_ana)
            if not used_ana:
                fl = raw(" ".join(words[x].text for x in range(wi, wi + k)))
                if len(fl) >= 2 and pos + len(fl) <= N:
                    seg = answer[pos:pos + len(fl)]
                    if sorted(seg) == sorted(fl) and seg != fl:    # anagram, not identity
                        dfs(wi + k, pos + len(fl),
                            pieces + [(wi, wi + k, "ANA_F", seg)], gaps, True)
        dfs(wi + 1, pos, pieces, gaps + [wi], used_ana)

    dfs(0, 0, [], [], False)
    for pieces, gaps, used_ana in found:
        if not used_ana or len(pieces) < 2:
            continue
        ana_run = find_typed_run(words, gaps, indicator_types, "anagram", min_length=1)
        if ana_run is None:
            continue
        ana_set = set(ana_run)
        if not all(residue_link(g) for g in gaps if g not in ana_set):
            continue
        # The anagram indicator is RESIDUE in this engine (found among the gaps in
        # finalize), NOT a slot — so the signature is just the typed pieces (no ANA_I).
        slot_items = sorted((a, b, role) for (a, b, role, _v) in pieces)
        yield {"operation": "anagram_charade",
               "roles": [r for _, _, r in slot_items],
               "n_words": [b - a for a, b, _ in slot_items],
               "def_pos": split.where, "pieces": pieces, "ana_run": ana_run,
               "links": [g for g in gaps if g not in ana_set],
               "split": split, "words": words}


def _discover_container(answer, words, split, lookup_all, is_link, indicator_types, ctx):
    """Yield container candidates: TWO value runs and a CON_I (container/insertion)
    indicator run such that one value inserted into the other spells `answer`.

    A value run is a DB synonym/abbreviation (SYN_F / ABR_F) OR a single-word letter
    SELECTION (SEL_F) licensed by a selection indicator ("Hops finally" -> S). Leftover
    words must be links or the selection indicator. Mirrors container_signature_engine so
    only shapes the engine can instantiate are proposed; auto_discover_and_file then
    verifies each through the real engine before anything is filed."""
    from core.selection import select_span
    from core.selection_indicators import find_indicators
    from core.container_signature_engine import _is_con_indicator, _verify_insertion
    n, N = len(words), len(answer)

    def residue_link(k):
        return bool(is_link and is_link(words[k].text))

    sel_inds = find_indicators(words)                 # [(rule, (idx, ...)), ...]
    sel_ind_pos = {i for _r, idxs in sel_inds for i in idxs}

    # Candidate value runs: (a, b, role, [value strings]).
    vruns = []
    for a in range(n):
        for b in range(a + 1, n + 1):        # no artificial cap: bound by the wordplay span
            phrase = " ".join(words[k].text for k in range(a, b))
            by_role = {}
            for val, mech in lookup_all(phrase):
                role = _MECH_ROLE.get(mech)
                v = (val or "").upper()
                if role and v:
                    by_role.setdefault(role, [])
                    if v not in by_role[role]:
                        by_role[role].append(v)
            for role, vs in by_role.items():
                vruns.append((a, b, role, vs))
            if b - a == 1:                            # SEL_F: one word, licensed elsewhere
                for rule, idxs in sel_inds:
                    if a in idxs:
                        continue                      # the source word is not the indicator
                    svals = [s for s, _ in select_span(ctx, words[a], rule)]
                    if svals:
                        vruns.append((a, b, "SEL_F", svals))

    seen = set()
    for i in range(len(vruns)):
        for j in range(i + 1, len(vruns)):
            a1, b1, role1, vals1 = vruns[i]
            a2, b2, role2, vals2 = vruns[j]
            if not (b1 <= a2 or b2 <= a1):            # value runs must be disjoint
                continue
            if _verify_insertion((a1, b1), (a2, b2), vals1, vals2, answer) is None:
                continue
            used_val = set(range(a1, b1)) | set(range(a2, b2))
            for ca in range(n):
                for cb in range(ca + 1, min(ca + MAX_RUN, n) + 1):
                    crun = range(ca, cb)
                    if any(k in used_val for k in crun):
                        continue
                    if not any(_is_con_indicator(words[k].text, indicator_types)
                               for k in crun):
                        continue
                    used = used_val | set(crun)
                    residue = [k for k in range(n) if k not in used]
                    if not all(residue_link(k) or k in sel_ind_pos for k in residue):
                        continue
                    if "SEL_F" in (role1, role2) and not (sel_ind_pos & set(residue)):
                        continue                      # a SEL_F needs its licensing indicator
                    slot_items = sorted([(a1, b1, role1), (a2, b2, role2),
                                         (ca, cb, "CON_I")])
                    roles = tuple(r for _, _, r in slot_items)
                    nwords = tuple(b - a for a, b, _ in slot_items)
                    key = (roles, nwords, split.where)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield {"operation": "container", "roles": list(roles),
                           "n_words": list(nwords), "def_pos": split.where,
                           "pieces": [(a1, b1, role1, vals1[0]),
                                      (a2, b2, role2, vals2[0])],
                           "links": [k for k in residue if k not in sel_ind_pos],
                           "split": split, "words": words}


def _signature_str(cand):
    parts = []
    for role, nw in zip(cand["roles"], cand["n_words"]):
        parts.append(role + ("(%dw)" % nw if nw > 1 else ""))
    return "+".join(parts) + " %s def:%s" % (cand["operation"], cand["def_pos"])


def _score(cand):
    """Rank a candidate signature (lower is better): prefer DB-backed pieces (fewest
    literals), then fewest leftover links (account for the most words / fullest
    indicator), then tightest spans (smallest total n_words), then fewest pieces."""
    roles = cand["roles"]
    return (sum(1 for r in roles if r == "LIT_F"), len(cand["links"]),
            sum(cand["n_words"]), len(roles))


def discover(clue_text, answer, wiring):
    """All candidate signatures for the clue, de-duplicated by signature, ranked."""
    from core.wfw_atoms import build_wfw_atom_context
    ctx = build_wfw_atom_context(clue_text, answer)
    answer_l = _answer_letters(ctx)
    prepared = _prepare(ctx, wiring["defines"], wiring.get("define_fallback"),
                        wiring.get("is_dbe"))
    cands = []
    for split, words, postags in prepared:
        if len(words) >= 2:
            cands += list(_discover_charade(answer_l, words, postags, split,
                                            wiring["lookup"], wiring["is_link"]))
            cands += list(_discover_reversal_charade(
                answer_l, words, split, wiring["lookup_all"], wiring["is_link"],
                wiring["indicator_types"]))
            cands += list(_discover_anagram_charade(
                answer_l, words, split, wiring["lookup_all"], wiring["is_link"],
                wiring["indicator_types"]))
            cands += list(_discover_container(
                answer_l, words, split, wiring["lookup_all"], wiring["is_link"],
                wiring["indicator_types"], ctx))
        cands += _discover_anagram(answer_l, words, postags, split,
                                   wiring["is_link"], wiring["indicator_types"])
    by_sig = {}
    for c in cands:
        sig = _signature_str(c)
        if sig not in by_sig or _score(c) < _score(by_sig[sig]):
            by_sig[sig] = c
    return sorted(by_sig.values(), key=_score)


def render_candidate(cand):
    """Human-readable parse the candidate implies."""
    words = cand["words"]
    lines = ["  definition: %r (%s)" % (cand["split"].phrase, cand["def_pos"])]
    if cand["operation"] == "charade":
        for a, b, role, val in cand["pieces"]:
            text = " ".join(words[k].text for k in range(a, b))
            lines.append("  %-6s %r -> %s" % (role, text, val))
    elif cand["operation"] == "reversal_charade":
        for a, b, role, val in cand["pieces"]:
            text = " ".join(words[k].text for k in range(a, b))
            arrow = "%s (reversed)" % val if role == "REV_F" else val
            lines.append("  %-6s %r -> %s" % (role, text, arrow))
        lines.append("  REV_I  %r" % " ".join(words[k].text for k in cand["rev_run"]))
    else:
        fod = " ".join(words[k].text for k in cand["fodder_idx"])
        ind = " ".join(words[k].text for k in cand["indicator_idx"])
        lines.append("  ANA_F  %r -> (anagram)" % fod)
        lines.append("  ANA_I  %r" % ind)
    if cand["links"]:
        lines.append("  links: %s" % ", ".join(repr(words[k].text)
                                               for k in cand["links"]))
    return "\n".join(lines)


# --------------------------------------------------------------- create / verify

def backup_catalog(con):
    for t in ("catalog_templates", "catalog_template_slots"):
        con.execute("DROP TABLE IF EXISTS %s_bak_creator" % t)
        con.execute("CREATE TABLE %s_bak_creator AS SELECT * FROM %s" % (t, t))


def add_signature(cand, note, db_path=None, backup=True, origin="hand_added"):
    """Insert the candidate's signature (+ slots) into the catalog. Returns the
    new template id, or None if it already exists. Backs the catalog up first unless
    backup=False (the auto path files many in succession; git is the backup there).
    `origin` tags the row ('hand_added' for the CLI, 'auto' for the solve hook)."""
    sig = _signature_str(cand)
    con = sqlite3.connect(db_path or _CLUES_DB)
    try:
        if con.execute("SELECT 1 FROM catalog_templates WHERE signature=?",
                       (sig,)).fetchone():
            return None
        if backup:
            backup_catalog(con)
        tid = con.execute("SELECT COALESCE(MAX(id),0)+1 "
                          "FROM catalog_templates").fetchone()[0]
        sid = con.execute("SELECT COALESCE(MAX(id),0)+1 "
                          "FROM catalog_template_slots").fetchone()[0]
        pri = con.execute("SELECT COALESCE(MAX(priority),0)+1 FROM catalog_templates "
                          "WHERE operation=?", (cand["operation"],)).fetchone()[0]
        con.execute(
            "INSERT INTO catalog_templates(id,operation,signature,def_pos,count,"
            "priority,origin,active,version,created_at,notes) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (tid, cand["operation"], sig, cand["def_pos"], 1, pri, origin, 1,
             1, "2026-06-05", note))
        for pos, (role, nw) in enumerate(zip(cand["roles"], cand["n_words"])):
            con.execute("INSERT INTO catalog_template_slots(id,template_id,position,"
                        "role,n_words) VALUES(?,?,?,?,?)", (sid + pos, tid, pos,
                                                            role, nw))
        con.commit()
        return tid
    finally:
        con.close()


def auto_file_signature(clue_text, answer, wiring,
                        note="auto-filed from a solved clue"):
    """File the best DB-backed signature a SOLVED clue implies, if the catalog lacks it.
    Additive and de-duped by signature string, so it can never alter an existing
    decomposition or yield a wrong answer (a catalog match must still spell the answer).
    Returns the filed signature string, or None (nothing new, or only literal-leaning
    candidates — those usually mean a missing DB entry, not a missing shape). Safe to
    call on every fully-passed clue; used by the auto signature-creation hook."""
    try:
        cands = discover(clue_text, answer, wiring)
    except Exception:
        return None
    for cand in cands:                       # best (lowest _score) first
        if cand["operation"] == "charade" and all(r == "LIT_F" for r in cand["roles"]):
            continue                         # wholly literal -> likely a DB gap, not a shape
        tid = add_signature(cand, note=note, backup=False, origin="auto")
        if tid is not None:
            return _signature_str(cand)      # filed a new shape
    return None                              # every candidate already in the catalog


_OP_TEMPLATE_KEY = {
    "charade": "charade_templates",
    "anagram": "anagram_templates",
    "reversal_charade": "reversal_charade_templates",
    "anagram_charade": "anagram_charade_templates",
    "container": "container_templates",
}


def auto_discover_and_file(ctx, wiring, solve_fn):
    """The GENERAL loop, STRICT — only record a signature that is PROVEN correct.

    For a clue the catalog could not solve, discover the decompositions its bits imply; for
    each, append the implied signature to the IN-MEMORY catalog (not the DB) and re-solve
    through the real cascade. ONLY if that yields a CLEAN PASS *via this signature* — every
    piece a confirmed DB value, a recognised indicator, a DB-confirmed definition, the
    letters spelling the answer exactly, every word accounted — is the signature written to
    the catalog. Otherwise nothing is filed and the in-memory trial is removed: the clue
    stays unsolved. No false signature is ever created, even briefly.

    `solve_fn(ctx, wiring)` re-solves without persisting. Returns (parse, engine_name) on a
    verified solve, else None. Extends to a new operation via its discoverer + templates key."""
    try:
        cands = discover(ctx.clue_text, ctx.answer_text, wiring)
    except Exception:
        return None
    from core.catalog_loader import Template, Slot
    for cand in cands:
        if cand["operation"] == "charade" and all(r == "LIT_F" for r in cand["roles"]):
            continue                             # wholly literal -> likely a DB gap
        key = _OP_TEMPLATE_KEY.get(cand["operation"])
        if not (key and isinstance(wiring.get(key), list)):
            continue
        sig = _signature_str(cand)
        slots = tuple(Slot(position=i, role=r, n_words=nw)
                      for i, (r, nw) in enumerate(zip(cand["roles"], cand["n_words"])))
        tmpl = Template(id=-1, operation=cand["operation"], signature=sig,
                        def_pos=cand["def_pos"], count=1, priority=10 ** 6, slots=slots)
        wiring[key].append(tmpl)                 # TRIAL: in-memory only, DB untouched
        try:
            rparse, rname = solve_fn(ctx, wiring)
        except Exception:
            rparse, rname = None, None
        # Keep ONLY a clean PASS produced by THIS signature (a pending/fail proves nothing).
        if (rparse is not None and rparse.status == "pass"
                and getattr(rparse, "matched_signature", None) == sig):
            add_signature(cand, note="auto: verified clean solve the catalog lacked",
                          backup=False, origin="auto")   # NOW it is proven -> commit
            return rparse, rname
        wiring[key].pop()                        # unproven -> leave the catalog untouched
    return None


def _parse_summary(parse):
    """One-line-per-piece review text of a verified parse (for the approval queue)."""
    lines = []
    if parse.definition is not None:
        lines.append("def: %s" % parse.definition.text)
    for s in parse.sources:
        lines.append("%s -> %s (%s)" % (s.text, s.value, s.mechanism))
    for a in parse.annotations:
        if getattr(a, "role", "") == "indicator":
            lines.append("[%s: %s]" % (a.note, a.text))
    return "\n".join(lines)


def auto_discover_and_queue(ctx, wiring, solve_fn, clue_id=None, created_at="2026-06-22"):
    """Like auto_discover_and_file, but QUEUE the proven signature for human approval
    instead of filing it. Same strict gate: a candidate is only queued if, trialled
    in-memory, it produces a CLEAN PASS via its own signature through the real cascade.
    The catalog is NOT changed (the clue stays unsolved until the signature is approved).
    Returns the queued signature string, or None. De-duped against catalog + queue +
    rejected via signature_queue.is_known."""
    from core import signature_queue
    try:
        cands = discover(ctx.clue_text, ctx.answer_text, wiring)
    except Exception:
        return None
    from core.catalog_loader import Template, Slot
    for cand in cands:
        if cand["operation"] == "charade" and all(r == "LIT_F" for r in cand["roles"]):
            continue                             # wholly literal -> likely a DB gap
        key = _OP_TEMPLATE_KEY.get(cand["operation"])
        if not (key and isinstance(wiring.get(key), list)):
            continue
        sig = _signature_str(cand)
        if signature_queue.is_known(sig):        # already catalogued / queued / rejected
            continue
        slots = tuple(Slot(position=i, role=r, n_words=nw)
                      for i, (r, nw) in enumerate(zip(cand["roles"], cand["n_words"])))
        tmpl = Template(id=-1, operation=cand["operation"], signature=sig,
                        def_pos=cand["def_pos"], count=1, priority=10 ** 6, slots=slots)
        wiring[key].append(tmpl)                  # TRIAL: in-memory only, DB untouched
        try:
            rparse, _ = solve_fn(ctx, wiring)
        except Exception:
            rparse = None
        wiring[key].pop()                         # always remove the trial; we only QUEUE
        if (rparse is not None and rparse.status == "pass"
                and getattr(rparse, "matched_signature", None) == sig):
            cand_min = {"signature": sig, "operation": cand["operation"],
                        "roles": cand["roles"], "n_words": cand["n_words"],
                        "def_pos": cand["def_pos"]}
            qid = signature_queue.queue(
                cand_min, clue_id, ctx.clue_text,
                ctx.answer_text, _parse_summary(rparse), created_at)
            if qid is not None:
                return sig
    return None


def verify(clue_text, answer, wiring):
    """Re-solve through the REAL cascade; return (status, engine, signature, parse)."""
    from core.engine_registry import solve_clue_text
    wiring = dict(wiring); wiring["store"] = None
    ctx, p, name = solve_clue_text(clue_text, answer, wiring)
    if p is None:
        return ("none", None, None, None)
    return (p.status, name, getattr(p, "matched_signature", None), p)


def regression_sample(operation, wiring, n=120, seed=42):
    """Pass count over a random sample of `operation` clues (for before/after)."""
    import random
    from core.engine_registry import solve_clue_text
    wt = {"charade": "charade", "anagram": "anagram"}.get(operation, operation)
    con = sqlite3.connect(_CLUES_DB)
    ids = [r[0] for r in con.execute(
        "SELECT id FROM clues WHERE wordplay_type=? AND answer!='' AND clue_text!='' "
        "AND definition IS NOT NULL AND definition!=''", (wt,))]
    con.close()
    random.Random(seed).shuffle(ids); ids = ids[:n]
    w = dict(wiring); w["store"] = None
    passes = 0
    for cid in ids:
        con = sqlite3.connect(_CLUES_DB)
        ct, ans = con.execute("SELECT clue_text,answer FROM clues WHERE id=?",
                              (cid,)).fetchone()
        con.close()
        try:
            _, p, _ = solve_clue_text(ct, ans, w)
        except Exception:
            continue
        if p is not None and p.status in ("pass", "pending"):
            passes += 1
    return passes, len(ids)


# --------------------------------------------------------------- CLI

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    clue_id = int(sys.argv[1])
    add_idx = None
    if "--add" in sys.argv:
        add_idx = int(sys.argv[sys.argv.index("--add") + 1])

    row = load_clue(clue_id)
    if not row:
        print("no clue", clue_id)
        return
    clue_text, answer, source, puzzle = row
    from core.engine_registry import make_db_wiring
    w = make_db_wiring()

    print("CLUE %d: %s  =  %s\n" % (clue_id, clue_text, answer))
    st, name, sig, _ = verify(clue_text, answer, w)
    print("current cascade: %s (%s) sig=%s\n" % (st, name, sig))

    cands = discover(clue_text, answer, w)
    if not cands:
        print("No mechanically-valid decomposition found. The gap is NOT a missing "
              "signature — likely a missing DB synonym/abbreviation/indicator, or a "
              "different clue type. (§8 step 2: classify the gap.)")
        return
    print("Discovered %d candidate signature(s) (best first):\n" % len(cands))
    for i, c in enumerate(cands):
        lit = sum(1 for r in c["roles"] if r == "LIT_F")
        flag = "  [leans on literals — check a DB piece isn't missing]" if (
            c["operation"] == "charade" and lit and lit >= len(c["roles"]) - 0) else ""
        print("[%d] %s%s" % (i, _signature_str(c), flag))
        print(render_candidate(c))
        print()

    if add_idx is None:
        print("To add one: python -m core.catalog_creator %d --add <N>" % clue_id)
        return

    cand = cands[add_idx]
    sig = _signature_str(cand)
    print("=== adding [%d] %s ===" % (add_idx, sig))
    before, ntot = regression_sample(cand["operation"], w)
    tid = add_signature(cand, note="§8 catalog-creation for clue %d" % clue_id)
    if tid is None:
        print("signature already present — nothing added.")
        return
    print("inserted template id %d (catalog backed up: *_bak_creator)" % tid)
    w2 = make_db_wiring()                       # reload so the new signature is live
    st, name, vsig, p = verify(clue_text, answer, w2)
    print("re-solve: %s (%s) sig=%s" % (st, name, vsig))
    after, _ = regression_sample(cand["operation"], w2)
    print("regression sample (%s, n=%d): before %d -> after %d  %s" %
          (cand["operation"], ntot, before, after,
           "OK" if after >= before else "REGRESSION!"))
    print("\nrecorded as catalog_templates.id=%d origin=hand_added." % tid)


if __name__ == "__main__":
    main()
