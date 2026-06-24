"""Plain deletion — CATALOG-DRIVEN (design §4). The recipe-driven replacement for the
bespoke core.deletion_engine.

A deletion recipe records the STRUCTURE of the clue as typed slots, and the verifier
EXECUTES it (it does not search for the shape):

  Form A — positional deletion:   base + DEL_I
      base  = SYN_F/ABR_F  (the pre-deletion DB value V, longer than the answer)
      DEL_I = the deletion indicator; its DB sub-type names the op (behead/curtail/
              outer/heartless/empty) via core.deletion.SUBTYPE_OP
      verify: apply_op(op, V) == answer      e.g. TAU = curtail(TAUT)

  Form B — named deletion:         base + DEL_I + REM_F
      base  = SYN_F/ABR_F  (V, longer than the answer)
      DEL_I = a GENERIC removal indicator (a sub-type with no specific op)
      REM_F = SYN_F/ABR_F naming the removed letters R (a DB value of another word)
      verify: V with the run R removed == answer   e.g. LOTTO = BLOTTO - B("bishop")

Placement is recipe-driven (a DFS assigns each recipe slot to a contiguous clue-word run,
in clue order, leftover words classified as links LAST). The deletion op is DATA — read
from the indicator's DB sub-type — not hand-coded per clue. Reuses
deletion_engine._build / _verify so the emitted Parse is byte-identical to the engine this
replaces. Pure, DB-decoupled, no value-product enumeration -> no memory cost.
"""

from core import deletion
from core.deletion_engine import _build, _verify, _run_values

_BASE_ROLES = ("SYN_F", "ABR_F")
_RECIPE_ROLES = {"SYN_F", "ABR_F", "DEL_I", "REM_F"}


def _answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _del_ops(words, run, del_subtypes):
    """(specific_ops, is_generic, typed_idx) for the deletion-indicator run
    words[run[0]:run[1]]. specific_ops are canonical ops its DB sub-types pin (behead/
    curtail/...); is_generic is True when a sub-type carries a generic removal (general /
    removal / deletion / NULL) — the Form B marker; typed_idx is the set of word indices
    that fall in ANY DB-typed sub-run. Empty spec + not generic => not a usable indicator.

    Scans EVERY contiguous sub-run of the indicator (not just the whole phrase), because a
    multi-word indicator's op is often named by a component word — "removing top" types the
    op via "top" (behead), not via the 2-word phrase — and a split indicator ("cut at the
    front" = cut...front) carries glue words between its typed parts. typed_idx lets the
    verifier mark those interior glue words as LINKS, not indicator, so the Parse matches
    deletion_engine's _indicator_runs attribution exactly."""
    if not del_subtypes:
        return set(), False, set()
    a, b = run
    spec, generic, typed = set(), False, set()
    for s in range(a, b):
        for e in range(s + 1, b + 1):
            subs = del_subtypes(" ".join(words[k].text for k in range(s, e)))
            if not subs:
                continue
            spec |= {deletion.SUBTYPE_OP[x] for x in subs if x in deletion.SUBTYPE_OP}
            if any(x not in deletion.SUBTYPE_OP for x in subs):
                generic = True
            typed |= set(range(s, e))
    return spec, generic, typed


def _place(slots, words, answer, lookup_all, is_link, del_subtypes):
    """Assign the recipe slots to contiguous word-runs in clue order (gaps -> links),
    then EXECUTE the deletion. Returns a placement dict (the same shape deletion_engine._build
    consumes) or None. DFS mirrors reversal_signature_engine._place."""
    n = len(words)
    nslots = len(slots)

    def residue_link(k):
        return bool(is_link and is_link(words[k].text))

    def base_value_and_mech(run):
        a, b = run
        for v, m in _run_values(words, a, b, lookup_all):
            if len(v) > len(answer):
                yield v, m

    def finalize(assigned, gap_idxs):
        base = [(r, role) for r, role in assigned if role in _BASE_ROLES]
        dels = [r for r, role in assigned if role == "DEL_I"]
        rems = [r for r, role in assigned if role == "REM_F"]
        if len(base) != 1 or len(dels) != 1 or len(rems) > 1:
            return None
        base_run = base[0][0]
        del_run = dels[0]
        links = []
        for k in gap_idxs:
            if residue_link(k):
                links.append(k)
            else:
                return None
        spec, generic, typed = _del_ops(words, del_run, del_subtypes)
        # The indicator is the TYPED words of the slot; any interior glue word the slot
        # spans (e.g. "at the" in "cut at the front") is a link, not part of the indicator,
        # but must itself be a link word to stand.
        ind_idx = sorted(typed) if typed else sorted(range(del_run[0], del_run[1]))
        interior = [k for k in range(del_run[0], del_run[1]) if k not in typed]
        for k in interior:
            if not residue_link(k):
                return None
        all_links = sorted(links + interior)

        if not rems:
            # FORM A — positional: the indicator pins the op.
            ops = set(spec)
            if {"behead", "curtail"} <= ops:
                ops.add("outer")
            if not ops:
                return None
            for V, mech in base_value_and_mech(base_run):
                for op in ops:
                    if deletion.apply_op(op, V) == answer:
                        return {"form": "A", "run": base_run, "value": V, "mech": mech,
                                "op": op, "ind": ind_idx, "links": all_links}
            return None

        # FORM B — named: a generic removal marks the cut; REM_F names the removed run.
        if not generic:
            return None
        rem_run = rems[0]
        rem_vals = {v for v, _ in _run_values(words, rem_run[0], rem_run[1], lookup_all)}
        if not rem_vals:
            return None
        for V, mech in base_value_and_mech(base_run):
            cut_runs = deletion.removed_runs(V, answer)
            R = next((r for r in cut_runs if r in rem_vals), None)
            if R is None:
                continue
            return {"form": "B", "run": base_run, "value": V, "mech": mech,
                    "removed": R, "named": rem_run, "ind": ind_idx,
                    "links": all_links}
        return None

    def dfs(si, wi, assigned, gaps):
        if si == nslots:
            return finalize(assigned, gaps + list(range(wi, n)))
        slot = slots[si]
        nw = slot.n_words
        for j in range(wi, n - nw + 1):
            run = (j, j + nw)
            if slot.role == "DEL_I":
                spec, generic, _ = _del_ops(words, run, del_subtypes)
                if not spec and not generic:
                    continue                      # must be a real DB deletion indicator
            r = dfs(si + 1, j + nw, assigned + [(run, slot.role)],
                    gaps + list(range(wi, j)))
            if r:
                return r
        return None

    return dfs(0, 0, [], [])


def _try_template(ctx, answer, template, split, words, lookup_all, is_link, del_subtypes):
    if split.where != template.def_pos:
        return None
    if template.fodder_word_count > len(words):
        return None
    roles = [s.role for s in template.slots]
    if not set(roles) <= _RECIPE_ROLES:
        return None
    if sum(roles.count(r) for r in _BASE_ROLES) != 1:
        return None
    if roles.count("DEL_I") != 1 or roles.count("REM_F") > 1:
        return None
    placement = _place(template.slots, words, answer, lookup_all, is_link, del_subtypes)
    if placement is None:
        return None
    parse = _build(ctx, split, words, answer, placement)
    parse.matched_signature = template.signature
    parse.template_id = template.id
    return parse


def solve_deletion(ctx, defines, lookup_all, is_link, del_subtypes,
                   templates=None, define_fallback=None, is_dbe=None):
    """Full plain-deletion solve — catalog-driven. First clean PASS (fewest residue links),
    else best non-pass, else None. Signature of the bespoke solve_deletion plus `templates`
    (the deletion recipes) so the cascade can swap the two engines for an A/B comparison."""
    from core.definition_engine import find_definitions

    answer = _answer(ctx)
    if len(answer) < 3 or not templates:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    prepared = []
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 2:
            continue
        prepared.append((split, words))
    if not prepared:
        return None

    best_pass, best_key, best_other = None, None, None
    for template in templates:
        for split, words in prepared:
            parse = _try_template(ctx, answer, template, split, words, lookup_all,
                                  is_link, del_subtypes)
            if parse is None:
                continue
            if parse.status == "pass":
                residue = sum(1 for a in parse.annotations if a.role == "link")
                if best_pass is None or residue < best_key:
                    best_pass, best_key = parse, residue
            elif best_other is None:
                best_other = parse
    return best_pass or best_other
