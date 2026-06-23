"""Charade + acrostic — a charade where ONE piece is a multi-word acrostic (Times 31273
DOCTORS).

  DOCTORS = DOC ([D]ucks [O]bserved [C]rossing, "firstly") + TORS ("rocky hills"), def
  "Quacks".

The plain acrostic engine (core.acrostic_engine) only fires when the selected first/last
letters spell the WHOLE answer; the charade engine's letter-selection (SEL_F) takes letters
from a SINGLE word. Neither builds an acrostic that spans several words AND forms only PART
of the answer alongside other charade pieces. This engine does exactly that, and only that:
the answer is tiled left-to-right by plain charade pieces (DB synonym/abbreviation values)
PLUS at least one ACROSTIC piece — the first (or last) letter of each of a run of >=2
consecutive words. It requires >=1 acrostic piece AND >=1 plain piece, so it cannot intercept
a pure acrostic (no plain piece) or a plain charade (no acrostic piece).

ANSWER-DRIVEN: the acrostic letters must land exactly at their answer position (like a
synonym landing in the answer). Gated on a DB acrostic indicator in the residue (the same
gate as the acrostic engine — the noisy indicator table is backed by the exact-letter match).
Per-letter provenance on the acrostic piece (each answer letter <- the clue letter taken).
Returns ONLY a clean PASS (a conservative compound engine — never displaces a simpler
engine's pending/fail). _verify calls role_validity. Definition decided upstream. Pure.
"""

from core import selection, engine_common
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation")
_MODE_MECHANISM = {"first": "first_letter", "last": "last_letter"}
MAX_RUN = 5            # max words in a plain piece
MAX_ACRO = 6           # max words spanned by the acrostic piece


def _answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _run_values(words, a, b, lookup):
    phrase = " ".join(words[k].text for k in range(a, b))
    out = []
    for val, mech in lookup(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in out:
            out.append(v)
    return out


def solve_charade_acrostic(ctx, defines, lookup_all, is_link, indicator_types,
                           define_fallback=None, is_dbe=None):
    """Charade with one multi-word acrostic piece. Returns ONLY a clean PASS, else None."""
    from core.definition_engine import find_definitions
    answer = _answer(ctx)
    if len(answer) < 4 or indicator_types is None:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 3:
            continue
        parse = _try_split(ctx, answer, split, words, lookup_all, is_link,
                           indicator_types)
        if parse is not None and parse.status == "pass":
            return parse
    return None


def _try_split(ctx, answer, split, words, lookup_all, is_link, indicator_types):
    n, N = len(words), len(answer)
    val_cache = {}

    def values(a, b):
        if (a, b) not in val_cache:
            val_cache[(a, b)] = _run_values(words, a, b, lookup_all)
        return val_cache[(a, b)]

    # DFS tiling: plain pieces (SYN/ABR) + >=1 acrostic piece (first/last of a word run).
    def dfs(pos, used, pieces, n_plain, n_acro):
        if pos == N:
            if n_acro < 1 or n_plain < 1:
                return None
            return _finalize(ctx, split, words, answer, pieces, used, is_link,
                            indicator_types)
        # plain charade piece
        for a in range(n):
            if a in used:
                continue
            for b in range(a + 1, min(a + MAX_RUN, n) + 1):
                if any(k in used for k in range(a, b)):
                    break
                for v in values(a, b):
                    if answer.startswith(v, pos):
                        r = dfs(pos + len(v), used | set(range(a, b)),
                                pieces + [("plain", (a, b), v, None)], n_plain + 1, n_acro)
                        if r:
                            return r
        # acrostic piece: first/last letter of each of a run of >=2 consecutive words
        for mode in ("first", "last"):
            for a in range(n):
                if a in used:
                    continue
                for b in range(a + 2, min(a + MAX_ACRO, n) + 1):
                    if any(k in used for k in range(a, b)):
                        break
                    sel = selection.selected(ctx, words[a:b], mode)
                    if sel is None:
                        continue
                    s = "".join(c for c, _ in sel)
                    if s and answer.startswith(s, pos):
                        r = dfs(pos + len(s), used | set(range(a, b)),
                                pieces + [("acro", (a, b), s, (mode, sel))],
                                n_plain, n_acro + 1)
                        if r:
                            return r
        return None

    return dfs(0, set(), [], 0, 0)


def _finalize(ctx, split, words, answer, pieces, used, is_link, indicator_types):
    n = len(words)
    residue = [k for k in range(n) if k not in used]
    # Gate: an acrostic indicator must sit in the residue (same gate as acrostic_engine).
    ind_pos = engine_common.find_typed_run(words, set(residue), indicator_types,
                                            "acrostic", min_length=1)
    if ind_pos is None:
        return None
    ind_set = set(ind_pos)
    links = []
    for k in residue:
        if k in ind_set:
            continue
        if is_link and is_link(words[k].text):
            links.append(k)
        else:
            return None                              # unaccounted content word -> reject
    return _build(ctx, split, words, answer, pieces, ind_pos, links)


def _build(ctx, split, words, answer, pieces, ind_pos, links):
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    sources, links_out, pos = [], [], 0
    for kind, (a, b), value, extra in pieces:
        if kind == "plain":
            toks = words[a:b]
            si = len(sources)
            sources.append(Source(
                clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                text=" ".join(t.text for t in toks), value=value, mechanism="synonym"))
            for _ in range(len(value)):
                pos += 1
                links_out.append(Link(answer_pos=pos, source_index=si,
                                      operation="charade", clue_atom_id=None))
        else:  # acrostic: one Source per selected letter, with per-letter provenance
            mode, sel = extra
            mech = _MODE_MECHANISM[mode]
            for (char, aid), tok in zip(sel, words[a:b]):
                si = len(sources)
                sources.append(Source(clue_atom_ids=tok.atom_ids, text=tok.text,
                                      value=char, mechanism=mech, source="db"))
                pos += 1
                links_out.append(Link(answer_pos=pos, source_index=si,
                                      operation="acrostic", clue_atom_id=aid))

    ind_toks = [words[i] for i in ind_pos]
    annotations = [Annotation(
        clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in ind_toks), role="indicator",
        note="acrostic indicator")]
    for k in links:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link", note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links_out, annotations=annotations,
                  definition=definition, operation="charade", solved_by="catalog")
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    warnings = []
    if not parse.is_complete():
        warnings.append("the answer letters are not fully covered by the pieces")
    missing = parse.unexplained_words(ctx)
    if missing:
        warnings.append("these clue words are unaccounted for: "
                        + ", ".join(repr(m) for m in missing))
    if parse.definition is None:
        warnings.append("no definition found")
    elif getattr(parse.definition, "source", "db") == "pending":
        warnings.append("the definition is provisional (queued for enrichment)")
    from core import role_validity
    bad = role_validity.unbacked_roles(parse)
    if bad:
        parse.warnings = warnings + bad
        parse.status = "fail"
        return
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing or parse.definition is None:
        parse.status = "fail"
    else:
        parse.status = "pending"
