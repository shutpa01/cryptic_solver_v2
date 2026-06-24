"""Generic signature verifier — one verifier driven by a signature's assembly + per-piece
operations, replacing the per-type engines (documents/OPERATION_ASSEMBLY_SCHEMA.md §7).

It places the signature's flat slots onto the clue (gaps -> links), resolves each
contributing piece through the operation engines (core.operations), lets the assembly
engine (core.assemblies) tile the answer, and builds a Parse.

STAGE 1 covers the deletion family only: assembly 'single' with a pos_delete or named_delete
operation. The structure is derived from the existing deletion recipes here (it is persisted
in a later step). The Parse is built by reusing deletion_engine._build so the output is
byte-identical to the proven deletion signature engine — the A/B on known-good ground that
validates the layering before any other operation/assembly is added.
"""

from core import operations, assemblies
from core.deletion_engine import _build, _run_values
from core.deletion_signature_engine import _del_ops

_BASE_ROLES = ("SYN_F", "ABR_F")

# A letter-LOCATION rule (selection_indicators) -> the deletion op that DROPS that letter.
# Used to honour the location/operation split: a location indicator names WHICH letters go,
# the operation indicator (off/losing/...) triggers the removal. (Selection 'outer' KEEPS the
# ends, but in a deletion the located ends are what's dropped, hence 'outer'.)
_LOC_DELOP = {"first": "behead", "last": "curtail", "outer": "outer", "middle": "heartless"}


def _answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


# --------------------------------------------------------------- structure derivation

def deletion_structure(template):
    """Derive the (assembly, piece) structure for a deletion recipe from its flat slots.
    REM_F present -> named_delete (the recipe's DEL_I slot is really the REM_I signpost);
    otherwise -> pos_delete. Returns a dict, or None if the slots are not a deletion shape."""
    roles = [s.role for s in template.slots]
    base_i = next((i for i, r in enumerate(roles) if r in _BASE_ROLES), None)
    # the indicator slot is DEL_I (positional) or REM_I (the named-removal signpost)
    del_i = next((i for i, r in enumerate(roles) if r in ("DEL_I", "REM_I")), None)
    rem_i = next((i for i, r in enumerate(roles) if r == "REM_F"), None)
    if base_i is None or del_i is None:
        return None
    if rem_i is not None:
        op = {"kind": "named_delete", "indicator_slot": del_i, "operand_slots": [rem_i]}
    else:
        op = {"kind": "pos_delete", "indicator_slot": del_i, "operand_slots": []}
    return {"assembly": "single",
            "pieces": [{"base_slots": [base_i], "operation": op}]}


# --------------------------------------------------------------- placement

def _placements(slots, words, is_link):
    """Yield (assign, links): assign[i] = (a, b) run for slot i (clue order, gaps allowed),
    links = the gap word indices (each must be a link word). DFS, mirrors the per-type
    engines' placement; bounded by the small slot count."""
    n = len(words)
    nslots = len(slots)

    def residue_link(k):
        return bool(is_link and is_link(words[k].text))

    def dfs(si, wi, assign, gaps):
        if si == nslots:
            tail = list(range(wi, n))
            allgaps = gaps + tail
            if all(residue_link(k) for k in allgaps):
                yield list(assign), sorted(allgaps)
            return
        nw = slots[si].n_words
        for j in range(wi, n - nw + 1):
            yield from dfs(si + 1, j + nw, assign + [(j, j + nw)],
                           gaps + list(range(wi, j)))

    yield from dfs(0, 0, [], [])


# --------------------------------------------------------------- resolve a piece

def _base_candidates(run, words, lookup_all, target):
    """(value, mechanism) base candidates for a run that are longer than the target."""
    out = []
    for v, m in _run_values(words, run[0], run[1], lookup_all):
        if len(v) > len(target):
            out.append((v, m))
    return out


def _resolve_deletion_piece(piece, assign, words, answer, target, lookup_all, del_subtypes,
                            loc_rules=None):
    """Resolve a deletion piece against `target`. Returns
    (base_value, mech, op_kind, op_prov, ind_typed_idx, interior_links, operand_run) or None."""
    op = piece["operation"]
    base_run = assign[piece["base_slots"][0]]
    del_run = assign[op["indicator_slot"]]
    spec, generic, typed = _del_ops(words, del_run, del_subtypes)
    # LOCATION/OPERATION split: a letter-location indicator (start/first/head -> first letter;
    # end/last -> last) inside the deletion slot is part of the deletion expression
    # ("start off" = location 'start' + operation 'off'). Account it and let it pin WHICH
    # letters drop -- but ONLY when a genuine deletion operation word is also present in the
    # slot (a DB-typed deletion word or a generic removal). A location word NEVER licenses a
    # deletion on its own, so "start of X" stays a selection rather than a beheading.
    has_del_word = bool(typed) or generic
    loc = set()
    if loc_rules and has_del_word:
        for k in range(del_run[0], del_run[1]):
            if k in typed:
                continue
            drop = {_LOC_DELOP[r] for r in (loc_rules(words[k].text) or ()) if r in _LOC_DELOP}
            if drop:
                loc.add(k)
                spec = spec | drop
    typed = typed | loc
    ind_typed = sorted(typed) if typed else list(range(del_run[0], del_run[1]))
    interior = [k for k in range(del_run[0], del_run[1]) if k not in typed]

    engine = operations.get(op["kind"])
    if op["kind"] == "pos_delete":
        ops_set = set(spec)
        if {"behead", "curtail"} <= ops_set:
            ops_set.add("outer")
        for base, mech in _base_candidates(base_run, words, lookup_all, target):
            prov = engine.verify(base, {"ops": ops_set}, [], target)
            if prov is not None:
                return base, mech, "pos_delete", prov, ind_typed, interior, None
        return None
    # named_delete
    if not generic:
        return None
    operand_run = assign[op["operand_slots"][0]]
    operands = [v for v, _ in _run_values(words, operand_run[0], operand_run[1], lookup_all)]
    for base, mech in _base_candidates(base_run, words, lookup_all, target):
        prov = engine.verify(base, {}, operands, target)
        if prov is not None:
            return base, mech, "named_delete", prov, ind_typed, interior, operand_run
    return None


# --------------------------------------------------------------- the verifier

def _try_split(ctx, answer, template, structure, split, words, lookup_all, is_link,
               del_subtypes, loc_rules=None):
    assembly = assemblies.get(structure["assembly"])
    if assembly is None or structure["assembly"] != "single":
        return None                                   # stage 1: single only
    piece = structure["pieces"][0]
    for assign, gap_links in _placements(template.slots, words, is_link):
        resolved = {}

        def resolve(pc, target):
            r = _resolve_deletion_piece(pc, assign, words, answer, target, lookup_all,
                                        del_subtypes, loc_rules=loc_rules)
            if r is None:
                return None
            resolved["r"] = r
            return (r[0], None)                        # (base_value, op_prov placeholder)

        placed = assembly.solve([piece], resolve, answer)
        if placed is None:
            continue
        base, mech, kind, prov, ind_typed, interior, operand_run = resolved["r"]
        links = sorted(gap_links + [k for k in interior])
        for k in interior:
            if not (is_link and is_link(words[k].text)):
                links = None
                break
        if links is None:
            continue
        if kind == "pos_delete":
            pl = {"form": "A", "run": piece_run(piece, assign), "value": base, "mech": mech,
                  "op": prov["op"], "ind": ind_typed, "links": links}
        else:
            pl = {"form": "B", "run": piece_run(piece, assign), "value": base, "mech": mech,
                  "removed": prov["removed"], "named": operand_run, "ind": ind_typed,
                  "links": links}
        parse = _build(ctx, split, words, answer, pl)
        parse.matched_signature = template.signature
        parse.template_id = template.id
        return parse
    return None


def piece_run(piece, assign):
    return assign[piece["base_slots"][0]]


# =================================================================== CHARADE assembly
# A charade is pieces (each a role-pure value, optionally operated) concatenated to the
# answer. STAGE 2 covers IDENTITY pieces (plain charade) driven through the Charade assembly
# + the generic placement, reusing the proven charade engine's role-candidate lookup and
# Parse builder so the output is identical. (Operated charade pieces — charade_deletion etc.
# — are the composition step that follows, and need the general provenance builder.)

_CHARADE_FILLABLE = ("SYN_F", "ABR_F", "LIT_F")


def charade_structure(template):
    """Derive a plain-charade structure: assembly 'charade', one identity piece per slot
    (clue order = assembly order). Returns None if a slot is not a plain charade role
    (SEL_F selection charades are deferred)."""
    if any(s.role not in _CHARADE_FILLABLE for s in template.slots):
        return None
    pieces = [{"base_slots": [i], "operation": None}
              for i, _ in enumerate(template.slots)]
    return {"assembly": "charade", "pieces": pieces}


def _try_charade(ctx, answer, template, structure, split, words, lookup, is_link):
    from core.assemblies import Charade
    from core.charade_signature_engine import _role_candidates, _build
    roles = [s.role for s in template.slots]

    for assign, gap_links in _placements(template.slots, words, is_link):
        def resolve(piece, target):
            si = piece["base_slots"][0]
            a, b = assign[si]
            phrase = " ".join(words[k].text for k in range(a, b))
            for val in _role_candidates(roles[si], phrase, answer, lookup):
                if val == target:
                    return (val, None)
            return None

        placed = Charade().solve(structure["pieces"], resolve, answer)
        if placed is None:
            continue
        # rebuild the charade engine's placement dict (pieces in clue/assembly order)
        pieces = []
        for piece, span, value, _prov in placed:
            a, b = assign[piece["base_slots"][0]]
            pieces.append((a, b, roles[piece["base_slots"][0]], value, None))
        placement = {"pieces": pieces, "links": sorted(gap_links), "indicator": []}
        parse = _build(ctx, split, words, placement, template, lookup)
        return parse
    return None


def build_anagram_parse(ctx, definition_source, words, fodder_idx, ana_ind_idx,
                        del_ind_idx, removed_value, removed_idx, link_idx, answer):
    """Focused provenance builder for an anagram (optionally with a removed letter).
    Modelled on anagram_signature_engine._build: one Source per fodder word, each answer
    letter attributed to a fodder word that supplied it (span-level, transform 'anagram_of').
    The removed letter is an ANNOTATION (role 'deletion'), like the deletion engine's named
    removal — it is consumed, not placed, so it never enters the answer tiling. A step toward
    the general builder; covers the anagram operation under the single assembly."""
    from collections import Counter
    from core.wordplay import raw
    from core.wfw_model import Source, Link, Annotation, Parse

    fodder_tokens = [words[k] for k in fodder_idx]
    sources, remaining = [], []
    for t in fodder_tokens:
        wl = raw(t.text)
        remaining.append([len(sources), Counter(wl)])
        sources.append(Source(clue_atom_ids=t.atom_ids, text=t.text, value=wl,
                              mechanism="anagram_fodder"))
    links = []
    for pos_i, ch in enumerate(answer, start=1):
        si = 0
        for entry in remaining:
            if entry[1].get(ch, 0) > 0:
                entry[1][ch] -= 1
                si = entry[0]
                break
        links.append(Link(answer_pos=pos_i, source_index=si, operation="anagram",
                          clue_atom_id=None, transform="anagram_of"))

    annotations = []
    if ana_ind_idx:
        annotations.append(Annotation(
            clue_atom_ids=tuple(aid for k in ana_ind_idx for aid in words[k].atom_ids),
            text=" ".join(words[k].text for k in ana_ind_idx),
            role="indicator", note="anagram indicator"))
    if del_ind_idx:
        annotations.append(Annotation(
            clue_atom_ids=tuple(aid for k in del_ind_idx for aid in words[k].atom_ids),
            text=" ".join(words[k].text for k in del_ind_idx),
            role="indicator", note="deletion indicator"))
    if removed_value and removed_idx is not None:
        annotations.append(Annotation(
            clue_atom_ids=words[removed_idx].atom_ids, text=words[removed_idx].text,
            role="deletion",
            note="lost %s (%r)" % (removed_value, words[removed_idx].text)))
    for k in link_idx:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link", note="link word"))

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition_source, operation="anagram", solved_by="catalog")
    return parse


def solve_charade(ctx, defines, lookup, is_link, templates=None, define_fallback=None,
                  is_dbe=None):
    """Plain charade via the GENERIC verifier (Charade assembly + identity pieces). Same
    call signature as the charade signature engine for the A/B. Best pass = fewest residue
    links, then most literal pieces (the charade engine's key)."""
    from core.definition_engine import find_definitions
    answer = _answer(ctx)
    if len(answer) < 2 or not templates:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    prepared = [(s, [t for t in s.wordplay_tokens if t.kind == "word"]) for s in splits]
    prepared = [(s, w) for s, w in prepared if len(w) >= 2]
    if not prepared:
        return None

    # Best by the charade engine's key (fewest residue links, then most literals) — applied
    # to PASS and PENDING alike (a pending solve must still pick its best placement, not the
    # first one seen; this matches charade_signature_engine._search).
    best_pass = best_pending = best_other = None
    pass_key = pend_key = None
    for template in templates:
        structure = template.structure if (template.structure and
                                           template.assembly == "charade") \
            else charade_structure(template)
        if structure is None or structure.get("assembly") != "charade":
            continue
        for split, words in prepared:
            if split.where != template.def_pos:
                continue
            parse = _try_charade(ctx, answer, template, structure, split, words,
                                 lookup, is_link)
            if parse is None:
                continue
            residue = sum(1 for a in parse.annotations if a.role == "link")
            literals = sum(1 for s in parse.sources if s.mechanism == "raw")
            key = (residue, -literals)
            if parse.status == "pass":
                if pass_key is None or key < pass_key:
                    best_pass, pass_key = parse, key
            elif parse.status == "pending":
                if pend_key is None or key < pend_key:
                    best_pending, pend_key = parse, key
            elif best_other is None:
                best_other = parse
    return best_pass or best_pending or best_other


def solve_deletion(ctx, defines, lookup_all, is_link, del_subtypes,
                   templates=None, define_fallback=None, is_dbe=None, loc_rules=None):
    """Drop-in for the deletion cascade slot, but solved by the GENERIC verifier driving
    operation + assembly engines. Same signature as the deletion signature engine so the
    A/B harness can compare the two. `loc_rules` (selection_rules) lets a letter-location
    indicator inside a deletion slot pin which letters drop (the location/operation split)."""
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
        # Prefer the PERSISTED structure (data); derive only for a pre-migration catalog.
        structure = template.structure or deletion_structure(template)
        if structure is None:
            continue
        for split, words in prepared:
            if split.where != template.def_pos:
                continue
            parse = _try_split(ctx, answer, template, structure, split, words,
                               lookup_all, is_link, del_subtypes, loc_rules=loc_rules)
            if parse is None:
                continue
            if parse.status == "pass":
                residue = sum(1 for a in parse.annotations if a.role == "link")
                if best_pass is None or residue < best_key:
                    best_pass, best_key = parse, residue
            elif best_other is None:
                best_other = parse
    return best_pass or best_other
