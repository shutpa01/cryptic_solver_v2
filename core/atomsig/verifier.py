"""Coupled verifier — placement + per-mechanism backing in ONE pass.

The two crude steps (reconstruct, then back each piece) are joined here: the verifier
tiles the answer with the pieces, and a piece may fill a target span ONLY if its fodder
genuinely produces that span's letters under its mechanism AND a transform read off the
fit. So the transform (identity/reversed/anagram/selection/deletion/homophone) and the
DB-backing are decided together, in the correct orientation — which is what the separate
steps could not do (a reversed piece sits backwards; a homophone sounds, not spells).

    verify(answer, pieces, wiring) -> Result | None

`pieces` is [(mechanism, fodder, yields)] (yields ignored except as a hint — the source
value is re-derived from the fodder + DB). Result.placed is one entry per contributing
piece: {span, transform, source_value, mechanism, detail}; Result.assembly is 'single' |
'charade' | 'container'. None means the pieces cannot validly tile the answer.

Answer-driven: every candidate span is checked against the KNOWN answer, so the search is
tightly bounded and cannot fabricate (no DB value for the fodder -> the branch dies).
"""

from collections import Counter
from dataclasses import dataclass

from core import deletion

_SELECTION = {"hidden", "hidden_word", "hidden_in_word", "first_letter", "first letter",
              "initial", "initials", "last_letter", "selection", "acrostic",
              "alternation", "telescopic", "core_letters", "alternate_letters",
              "outer_letters", "middle_letters", "end_letter"}
_ANAGRAM = {"anagram", "anagram_fodder", "anag", "fodder"}
_HOMOPHONE = {"homophone", "homophones", "sounds_like", "sounds like", "sound",
              "sound_of", "homophone_of"}
_DELETION = {"deletion", "delete", "subtraction", "removal", "minus"}
_LITERAL = {"literal", "raw", "letters", "literally"}

_DEL_OPS = tuple(deletion._OP_FUNCS.keys()) if hasattr(deletion, "_OP_FUNCS") else \
    ("behead", "curtail", "outer", "heartless", "empty")

_MAX_PIECES = 6          # placement is bounded; absurd piece counts are skipped
_DEL_SCAN = 400          # cap DB values scanned for a deletion base
_HOM_SCAN = 200          # cap DB values scanned for a homophone source


def _letters(s):
    return "".join(ch for ch in (s or "").upper() if ch.isalpha())


def _is_subsequence(sub, whole):
    it = iter(whole)
    return all(c in it for c in sub)


def _db_values(fodder, wiring):
    out = set()
    try:
        for v, _ in wiring["lookup_all"](fodder):
            if v:
                out.add(_letters(v))
    except Exception:
        pass
    try:
        for s in wiring["synonyms_of"](fodder):
            if s:
                out.add(_letters(s))
    except Exception:
        pass
    return {v for v in out if v}


def _deletion_produces(base, target):
    for op in _DEL_OPS:
        try:
            if deletion.apply_op(op, base) == target:
                return "deletion:%s" % op
        except Exception:
            pass
    try:
        if deletion.removed_runs(base, target):
            return "deletion:run"
    except Exception:
        pass
    return None


class PieceFiller:
    """Precomputes a piece's DB-backed source values once; `fill(target)` decides whether
    the piece can produce `target` (a contiguous answer span) and under which transform."""

    def __init__(self, mechanism, fodder, yields, wiring):
        self.mech = (mechanism or "").lower().strip()
        self.wiring = wiring
        self.fl = _letters(fodder)
        self.yld = _letters(yields)
        self.db = _db_values(fodder, wiring)

    def fill(self, target):
        if not target:
            return None
        m = self.mech
        if m in _ANAGRAM:
            return ("anagram", self.fl, "anagram") \
                if self.fl and Counter(self.fl) == Counter(target) else None
        if m in _SELECTION:
            if _is_subsequence(target, self.fl):
                return ("selection", target, "selection")
            rev = target[::-1]                       # reversed selection (hidden_reversed)
            if rev != target and _is_subsequence(rev, self.fl):
                return ("reversed", rev, "selection_reversed")
            return None
        if m in _HOMOPHONE:
            return self._homophone(target)
        if m in _DELETION:
            return self._deletion(target)

        # value-based: synonym / abbreviation / literal / unknown
        if target in self.db:
            return ("identity", target, "value")
        rev = target[::-1]
        if rev != target and rev in self.db:
            return ("reversed", rev, "value")
        if m in _LITERAL and target == self.fl:
            return ("identity", self.fl, "literal")
        if not m or m == "?":                       # unknown mech: allow anagram/selection
            if self.fl and Counter(self.fl) == Counter(target):
                return ("anagram", self.fl, "anagram")
            if _is_subsequence(target, self.fl):
                return ("selection", target, "selection")
        return None

    def _deletion(self, target):
        for i, base in enumerate(self.db):
            if i >= _DEL_SCAN:
                break
            if len(base) > len(target):
                d = _deletion_produces(base, target)
                if d:
                    return ("deletion", base, d)
        return None

    def _homophone(self, target):
        sounds = self.wiring.get("sounds_alike")
        if sounds is None:
            return None
        cands = list(self.db)[:_HOM_SCAN] + [self.fl]
        for sv in cands:
            if not sv:
                continue
            try:
                if sounds(target, sv) or sounds(sv, target):
                    return ("homophone", sv, "homophone")
            except Exception:
                continue
        return None


@dataclass
class Result:
    assembly: str
    placed: list          # [{piece, span, transform, source_value, mechanism, detail}]


def _place_charade(answer, fillers, idxs):
    """Tile `answer` left-to-right, each piece in `idxs` filling one contiguous span (any
    order). Returns [(piece_idx, (a,b), transform, source_value, detail)] or None."""
    N = len(answer)
    out = []
    used = [False] * len(idxs)

    def rec(pos):
        if pos == N:
            return all(used)
        for k, pi in enumerate(idxs):
            if used[k]:
                continue
            for end in range(pos + 1, N + 1):
                fit = fillers[pi].fill(answer[pos:end])
                if fit is None:
                    continue
                used[k] = True
                out.append((pi, (pos, end)) + fit)
                if rec(end):
                    return True
                out.pop()
                used[k] = False
        return False

    return out if rec(0) else None


def _place_container(answer, fillers, idxs):
    """One piece is the OUTER (its value fills the two end spans, split around the middle);
    the rest tile the middle as a charade. Returns placement or None."""
    N = len(answer)
    for oi in range(len(idxs)):
        outer_pi = idxs[oi]
        rest = [idxs[j] for j in range(len(idxs)) if j != oi]
        if not rest:
            continue
        for p in range(1, N):
            for q in range(p + 1, N + 1):
                ends = answer[:p] + answer[q:]
                if not ends:
                    continue
                ofit = fillers[outer_pi].fill(ends)
                if ofit is None:
                    continue
                inner = _place_charade(answer[p:q], fillers, rest)
                if inner is None:
                    continue
                placed = [(outer_pi, (0, p, q)) + ofit]      # split span marker (0,p,q)
                for (pi, (a, b), tf, sv, det) in inner:
                    placed.append((pi, (p + a, p + b), tf, sv, det))
                return placed
    return None


def _to_result(assembly, placement, pieces):
    out = []
    for entry in placement:
        pi, span = entry[0], entry[1]
        tf, sv, det = entry[2], entry[3], entry[4]
        out.append({"piece": pi, "span": span, "transform": tf,
                    "source_value": sv, "mechanism": pieces[pi][0], "detail": det})
    return Result(assembly=assembly, placed=out)


def _merge_anagram(pieces):
    """Merge a run of adjacent anagram-fodder pieces into ONE piece — a multi-word
    anagram's fodder words jointly fill a single span (their letters interleave), so
    they must not be placed as separate contiguous pieces."""
    out, i = [], 0
    while i < len(pieces):
        m = (pieces[i][0] or "").lower().strip()
        if m in _ANAGRAM:
            fs, ys, j = [pieces[i][1]], [pieces[i][2]], i + 1
            while j < len(pieces) and (pieces[j][0] or "").lower().strip() in _ANAGRAM:
                fs.append(pieces[j][1])
                ys.append(pieces[j][2])
                j += 1
            out.append(("anagram_fodder", " ".join(fs), "".join(ys)))
            i = j
        else:
            out.append(pieces[i])
            i += 1
    return out


def verify(answer, pieces, wiring):
    """Return a Result (validated atom-map) or None. See module docstring."""
    ans = _letters(answer)
    # drop zero-yield pieces (indicators/operands contribute no answer letters), then
    # merge multi-word anagram fodder into one piece.
    contributing = _merge_anagram([(m, f, y) for (m, f, y) in pieces if _letters(y)])
    if not ans or not contributing or len(contributing) > _MAX_PIECES:
        return None
    fillers = [PieceFiller(m, f, y, wiring) for (m, f, y) in contributing]
    idxs = list(range(len(contributing)))

    charade = _place_charade(ans, fillers, idxs)
    if charade is not None:
        assembly = "single" if len(contributing) == 1 else "charade"
        return _to_result(assembly, charade, contributing)

    if len(contributing) >= 2:
        cont = _place_container(ans, fillers, idxs)
        if cont is not None:
            return _to_result("container", cont, contributing)
    return None
