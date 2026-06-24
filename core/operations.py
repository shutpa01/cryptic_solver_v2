"""Operation engines — composable, answer-driven value transforms.

Per documents/OPERATION_ASSEMBLY_SCHEMA.md §3. An operation transforms a base value and
is verified against the TARGET value the piece must produce (the assembly supplies the
target — the whole answer for a `single` assembly, a sub-span for a charade/container).
Operations know nothing about how pieces are combined.

    operation.verify(base_value, params, operands, target) -> prov | None

`prov` is a small dict of operation-specific provenance (e.g. {'op': 'behead'} or
{'removed': 'M'}); None means this base does not produce the target under this operation.
Each operation declares its provenance granularity ('atom' or 'span'), the indicator role
that signals it, and how many operand values it consumes.

This module starts with the deletion family (pos_delete, named_delete) — the two operations
the role split separates — and grows one operation at a time as each is A/B-gated.
"""

from collections import Counter

from core import deletion


class Operation:
    kind = None
    granularity = "atom"        # 'atom' (per-letter sourced) | 'span' (sourced as a whole)
    indicator_role = None       # the slot role that signals this operation
    operand_arity = 0           # how many operand values the operation consumes

    def verify(self, base_value, params, operands, target):
        raise NotImplementedError


class PosDelete(Operation):
    """Positional deletion — the indicator names which letters go (behead/curtail/outer/
    heartless/empty). params['ops'] is the candidate op set resolved from the indicator's
    DB sub-type; the operation tries each. The indicator IS the operation."""
    kind = "pos_delete"
    indicator_role = "DEL_I"

    def verify(self, base_value, params, operands, target):
        if not base_value or len(base_value) <= len(target):
            return None
        for op in params.get("ops", ()):
            if deletion.apply_op(op, base_value) == target:
                return {"op": op}
        return None


class NamedDelete(Operation):
    """Named deletion — a named value (operand) is removed from the base, position resolved
    answer-driven. The indicator (REM_I) is only a 'remove' signpost; the operation is
    carried by the operand value. `operands` is the list of candidate values for the one
    operand slot."""
    kind = "named_delete"
    indicator_role = "REM_I"
    operand_arity = 1

    def verify(self, base_value, params, operands, target):
        if not base_value or len(base_value) <= len(target):
            return None
        cuts = deletion.removed_runs(base_value, target)
        if not cuts:
            return None
        for r in operands or ():
            if r in cuts:
                return {"removed": r}
        return None


class Anagram(Operation):
    """Anagram — the fodder letters rearranged to the target span. Span-level provenance
    (the individual letters are not separately sourced). The same operation covers the
    PLAIN anagram (no operands) and the anagram whose fodder has letters REMOVED first
    (operands = the removed letter-values, e.g. 'one' -> I 'having lost'): the removal is
    multiset subtraction on the pool, so 'remove then permute' == 'permute (pool - removed)'.
    Gated on an anagram indicator (ANA_I); the deletion, when present, is gated on its own
    deletion indicator at the structure level."""
    kind = "anagram"
    granularity = "span"
    indicator_role = "ANA_I"

    def verify(self, base_value, params, operands, target):
        if not base_value or not target:
            return None
        pool = Counter(base_value)
        removed = Counter()
        for r in operands or ():
            removed += Counter(r)
        pool -= removed
        if sum(pool.values()) != len(target) or pool != Counter(target):
            return None
        if not operands and base_value == target:
            return None                       # identity is not an anagram
        return {"fodder": base_value, "removed": "".join(sorted(removed.elements()))}


REGISTRY = {op.kind: op for op in (PosDelete(), NamedDelete(), Anagram())}


def get(kind):
    return REGISTRY.get(kind)
