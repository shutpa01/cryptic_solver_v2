"""Assembly engines — compose contributing pieces into the answer.

Per documents/OPERATION_ASSEMBLY_SCHEMA.md §4. An assembly decides the TARGET span each
contributing piece must produce, then asks a `resolve(piece, target)` callback to satisfy
that piece (the callback bridges to the operation engines + DB lookups). The assembly knows
nothing about what operated on each piece.

    assembly.solve(pieces, resolve, answer) -> [(piece, target, prov), ...] | None

`resolve(piece, target)` returns (chosen_base_value, op_prov) | None. A returned list (one
entry per piece, in answer order) is enough for the verifier to build the provenance map;
None means the pieces cannot tile the answer under this assembly.

Starts with `single`; charade and container follow once the deletion layering is proven.
"""


class Single:
    """One contributing piece whose produced value IS the whole answer."""
    kind = "single"

    def solve(self, pieces, resolve, answer):
        if len(pieces) != 1:
            return None
        r = resolve(pieces[0], answer)
        if r is None:
            return None
        base_value, op_prov = r
        return [(pieces[0], (0, len(answer)), base_value, op_prov)]


class Charade:
    """≥2 contributing pieces whose produced values concatenate, in order, to the answer.
    Enumerates the ordered split points of the answer and resolves each piece against its
    part. (Built but not wired until the charade migration step.)"""
    kind = "charade"

    def solve(self, pieces, resolve, answer):
        n = len(pieces)
        if n < 2:
            return None
        N = len(answer)
        out = []

        def rec(pi, pos):
            if pi == n:
                return pos == N
            # the last piece takes the whole remainder; others take a prefix of >=1
            ends = [N] if pi == n - 1 else range(pos + 1, N - (n - pi - 1) + 1)
            for end in ends:
                target = answer[pos:end]
                if not target:
                    continue
                r = resolve(pieces[pi], target)
                if r is None:
                    continue
                base_value, op_prov = r
                out.append((pieces[pi], (pos, end), base_value, op_prov))
                if rec(pi + 1, end):
                    return True
                out.pop()
            return False

        return out if rec(0, 0) else None


REGISTRY = {a.kind: a for a in (Single(), Charade())}


def get(kind):
    return REGISTRY.get(kind)
