"""Apply a harvested atom-signature to a clue — the TRIVIAL verifier.

Given a clue+answer and a signature (assembly + ordered typed slots, def edge,
required indicator types), this PLACES the signature on the clue words and FITS
DB values to the answer. The signature pins the structure, so there is NO search
over operations or piece orderings — only (a) which words fill each slot and
(b) which DB value each slot takes. The completeness invariant is the precision
guard: every answer letter must be sourced, every content word must be a slot /
the definition / a DB-licensed indicator / a link, or the placement is rejected.

It builds a wfw_model.Parse identical in shape to what the cascade produces, so it
renders through the existing WFW page unchanged.

Roles handled in this version: synonym, abbreviation, raw, anagram_fodder, hidden,
first_letter. Assemblies: single, charade, container. Homophone is NOT yet handled
(reported, not faked).
"""

from dataclasses import dataclass

from core.wfw_atoms import build_wfw_atom_context
from core.wfw_model import Source, Link, Annotation, Parse
from core.definition_engine import find_definitions


# --------------------------------------------------------------- signature spec

@dataclass(frozen=True)
class Slot:
    role: str
    transform: str
    placement: str


@dataclass(frozen=True)
class SigSpec:
    assembly: str
    slots: tuple          # tuple[Slot] in answer order
    def_pos: str          # start | end | mixed
    indicators: tuple     # required indicator types


def parse_signature(key):
    """Parse a catalogue key back into a SigSpec.
    'charade | synonym:identity:contiguous + synonym:identity:contiguous | def:end | ind:'
    """
    parts = [p.strip() for p in key.split("|")]
    assembly = parts[0]
    slots = []
    for ps in parts[1].split(" + "):
        role, transform, placement = ps.split(":")
        slots.append(Slot(role, transform, placement))
    def_pos = parts[2].split(":", 1)[1].strip()
    ind = parts[3].split(":", 1)[1].strip()
    indicators = tuple(t for t in (x.strip() for x in ind.split(",")) if t)
    return SigSpec(assembly, tuple(slots), def_pos, indicators)


# --------------------------------------------------------------- small helpers

def _answer_letters(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _run_text(words, a, b):
    return " ".join(t.text for t in words[a:b])


def _run_letters(words, a, b):
    return "".join(ch for t in words[a:b] for ch in t.text.upper() if ch.isalpha())


def _run_atoms(words, a, b):
    return tuple(aid for t in words[a:b] for aid in t.atom_ids)


_LINK_OP = {"identity": None, "reversed": "reversed",
            "anagram": "anagram_of", "selection": None}


def _consume(val, transform, A, apos):
    """If `val` under `transform` fits the answer starting at apos, return the
    number of answer letters consumed, else None."""
    L = len(val)
    seg = A[apos:apos + L]
    if len(seg) < L or L == 0:
        return None
    if transform == "identity":
        return L if seg == val else None
    if transform == "reversed":
        return L if seg == val[::-1] else None
    if transform == "anagram":
        return L if sorted(seg) == sorted(val) else None
    if transform == "selection":
        return L if seg == val else None
    return None


def _slot_candidates(slot, words, a, b, A, apos, wiring):
    """Yield (value, mechanism, consumed_len) for filling slot with words[a:b],
    matched against the answer at apos."""
    role, transform = slot.role, slot.transform
    if role in ("synonym", "abbreviation", "raw"):
        phrase = _run_text(words, a, b)
        for val, mech in wiring["lookup_all"](phrase):
            val = (val or "").upper()
            clen = _consume(val, transform, A, apos)
            if clen:
                yield val, mech, clen
    elif role == "anagram_fodder":
        val = _run_letters(words, a, b)
        clen = _consume(val, "anagram", A, apos)
        if clen:
            yield val, "anagram_fodder", clen
    elif role == "hidden":
        # selection: a contiguous substring of the run's letters that matches the
        # answer at apos (single-piece hidden consumes the whole answer).
        letters = _run_letters(words, a, b)
        for L in range(len(A) - apos, 0, -1):
            seg = A[apos:apos + L]
            if seg in letters:
                yield seg, "hidden", L
                break
    elif role == "first_letter":
        val = "".join(t.text[0].upper() for t in words[a:b]
                      if t.text and t.text[0].isalpha())
        clen = _consume(val, "selection", A, apos)
        if clen:
            yield val, "first_letter", clen


# --------------------------------------------------------------- finalisation

def _mechanism(role, mech):
    if role in ("anagram_fodder", "hidden", "first_letter"):
        return role
    return mech


def _account_gaps(words, gap_idx, sig, wiring):
    """Turn gap words into Annotations. Every gap word must be a DB-licensed
    indicator (of a type the signature requires) or a DB link word, else the
    placement is invalid (returns None). Also every required indicator type must
    be present. Returns (annotations, ) or None."""
    needed = set(sig.indicators)
    covered = set()
    anns = []
    for g in gap_idx:
        w = words[g]
        types = set()
        try:
            types = wiring["indicator_types"](w.text) or set()
        except Exception:
            types = set()
        match = types & needed if needed else set()
        if match:
            t = sorted(match)[0]
            covered |= match
            anns.append(Annotation(clue_atom_ids=w.atom_ids, text=w.text,
                                   role="indicator", note="%s indicator" % t))
        elif wiring["is_link"](w.text):
            anns.append(Annotation(clue_atom_ids=w.atom_ids, text=w.text,
                                   role="link", note="link word"))
        else:
            return None
    if needed - covered:
        return None
    return anns


def _build_parse(ctx, sig, split, sources, links, annotations):
    A = _answer_letters(ctx)
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation=sig.assembly,
                  solved_by="atomsig")
    parse.matched_signature = sig
    warnings = []
    if not parse.is_complete():
        warnings.append("answer letters not fully covered")
    missing = parse.unexplained_words(ctx)
    if missing:
        warnings.append("unaccounted words: " + ", ".join(missing))
    if parse.definition is None:
        warnings.append("no definition")
    parse.status = "pass" if (not warnings and split.source == "db") else (
        "pending" if not warnings else "fail")
    parse.warnings = warnings
    return parse


def _finalize_linear(ctx, sig, words, assigned, gap_idx, split, wiring):
    anns = _account_gaps(words, gap_idx, sig, wiring)
    if anns is None:
        return None
    sources, links = [], []
    for si, (a, b, slot, val, mech, apos, clen) in enumerate(assigned):
        sources.append(Source(clue_atom_ids=_run_atoms(words, a, b),
                              text=_run_text(words, a, b), value=val,
                              mechanism=_mechanism(slot.role, mech)))
        for k in range(clen):
            links.append(Link(answer_pos=apos + k + 1, source_index=si,
                              operation=sig.assembly,
                              transform=_LINK_OP.get(slot.transform)))
    return _build_parse(ctx, sig, split, sources, links, anns)


def _finalize_container(ctx, sig, words, outer, inner, split, wiring):
    """outer/inner = dict(run=(a,b), val, mech, positions=[1-based answer pos])."""
    gap_idx = outer["gaps"]
    anns = _account_gaps(words, gap_idx, sig, wiring)
    if anns is None:
        return None
    # sources in ANSWER order: the outer piece starts at position 1, so it is first.
    sources = [
        Source(clue_atom_ids=_run_atoms(words, *outer["run"]),
               text=_run_text(words, *outer["run"]), value=outer["val"],
               mechanism=_mechanism("synonym", outer["mech"])),
        Source(clue_atom_ids=_run_atoms(words, *inner["run"]),
               text=_run_text(words, *inner["run"]), value=inner["val"],
               mechanism=_mechanism("synonym", inner["mech"])),
    ]
    links = []
    for pos in outer["positions"]:
        links.append(Link(answer_pos=pos, source_index=0,
                          operation="container", transform=None))
    for pos in inner["positions"]:
        links.append(Link(answer_pos=pos, source_index=1,
                          operation="container", transform=None))
    return _build_parse(ctx, sig, split, sources, links, anns)


# --------------------------------------------------------------- assemblers

def _assemble_linear(ctx, sig, words, A, split, wiring):
    """single / charade: the pieces TILE the answer left-to-right (the answer side
    is monotonic), but each piece is located on the clue by VALUE-FIT and may sit
    anywhere among the clue words — the clue order need not match the answer order
    (e.g. a reversed charade flips it). No assumption, no ordering search: the
    signature fixes the piece sequence and transforms; we only locate known pieces,
    pinned by exact letter-match to fixed answer positions. Unused words must all be
    DB-licensed indicators / links, or the placement is rejected."""
    n = len(words)
    nslots = len(sig.slots)
    result = [None]

    def dfs(pi, apos, used, assigned):
        if pi == nslots:
            if apos == len(A):
                gaps = [i for i in range(n) if i not in used]
                p = _finalize_linear(ctx, sig, words, assigned, gaps, split, wiring)
                if p and p.status in ("pass", "pending"):
                    result[0] = p
                    return True
            return False
        slot = sig.slots[pi]
        for a in range(n):
            if a in used:
                continue
            for b in range(a + 1, n + 1):
                if any(i in used for i in range(a, b)):
                    break
                for val, mech, clen in _slot_candidates(slot, words, a, b,
                                                        A, apos, wiring):
                    used2 = used | set(range(a, b))
                    if dfs(pi + 1, apos + clen, used2,
                           assigned + [(a, b, slot, val, mech, apos, clen)]):
                        return True
        return False

    dfs(0, 0, set(), [])
    return result[0]


def _assemble_container(ctx, sig, words, A, wiring, split):
    """container: an identity outer piece split around a contiguous inner piece."""
    n = len(words)
    lookup_all = wiring["lookup_all"]
    for i in range(n):
        for j in range(i + 1, n + 1):
            for p in range(j, n):
                for q in range(p + 1, n + 1):
                    run1, run2 = (i, j), (p, q)
                    gaps = sorted(set(range(0, i)) | set(range(j, p)) | set(range(q, n)))
                    for outer_run, inner_run in ((run1, run2), (run2, run1)):
                        for Vo, mo in lookup_all(_run_text(words, *outer_run)):
                            Vo = (Vo or "").upper()
                            for Vi, mi in lookup_all(_run_text(words, *inner_run)):
                                Vi = (Vi or "").upper()
                                if len(Vo) + len(Vi) != len(A):
                                    continue
                                for k in range(1, len(Vo)):
                                    if A == Vo[:k] + Vi + Vo[k:]:
                                        outer = {"run": outer_run, "val": Vo,
                                                 "mech": mo, "gaps": gaps,
                                                 "positions": list(range(1, k + 1)) +
                                                 list(range(k + len(Vi) + 1, len(A) + 1))}
                                        inner = {"run": inner_run, "val": Vi,
                                                 "mech": mi,
                                                 "positions": list(range(k + 1, k + len(Vi) + 1))}
                                        res = _finalize_container(ctx, sig, words,
                                                                  outer, inner, split, wiring)
                                        if res and res.status in ("pass", "pending"):
                                            return res
    return None


# --------------------------------------------------------------- public entry

def apply_catalog(clue_text, answer, wiring, catalog, direction=None,
                  accept_pending=False):
    """Try every catalogue SigSpec on the clue; return the first Parse whose
    status is 'pass' (or 'pending' if accept_pending), else None.

    `catalog` is a list of SigSpec. The clue's definition is found once per
    distinct def edge and reused across signatures with that edge.
    """
    ctx = build_wfw_atom_context(clue_text, answer, direction=direction)
    A = _answer_letters(ctx)
    if not A:
        return None

    # definitions per edge, computed lazily
    defines = wiring["defines"]
    is_dbe = wiring.get("is_dbe")
    _defs = {}

    def defs_for(edge):
        if edge not in _defs:
            try:
                splits = find_definitions(ctx, defines, is_dbe=is_dbe)
            except Exception:
                splits = []
            _defs[edge] = [s for s in splits
                           if (edge in ("mixed",) or s.where == edge)
                           and (accept_pending or s.source == "db")]
        return _defs[edge]

    best_pending = None
    for sig in catalog:
        for split in defs_for(sig.def_pos):
            words = [t for t in split.wordplay_tokens if t.kind == "word"]
            if len(words) < len(sig.slots):
                continue
            if sig.assembly in ("single", "charade"):
                p = _assemble_linear(ctx, sig, words, A, split, wiring)
            elif sig.assembly == "container":
                p = _assemble_container(ctx, sig, words, A, wiring, split)
            else:
                continue
            if p is None:
                continue
            if p.status == "pass":
                return p
            if p.status == "pending" and accept_pending and best_pending is None:
                best_pending = p
    return best_pending
