"""Universal Form schema — recursive operation tree for cryptic wordplay.

The form has four fields:

    Form
      tree         the recursive wordplay tree (root = outermost operation)
      definition   {phrase, answer} — phrase from the clue that defines the answer
      link_words   surface words that join clue parts but carry no role
      is_and_lit   whole-clue-also-defines flag (capped at LOW unless DB-backed)

The tree is made of two kinds of node:

    Leaf — terminal letter-producer. Operation is one of:
        literal       letters are the source word's letters as-is
        synonym       source word means a word with these letters (DB)
        abbreviation  source word maps to letters by convention (DB)
        positional    extracts letters from the source word (first/last/etc.)

    Op — combines or transforms one or more children. Operation is one of:
        charade       concatenate children left-to-right
        container     children[0] outer wraps children[1] inner
        anagram       rearrange the letters of all children
        reversal      reverse the letters of the single child
        deletion      remove letters from the single child (head/tail/outer/heart)
        hidden        the answer is contiguous letters across a span
        double_definition   two child definitions both map to the answer

Every operation that touches the result is its own node. The outermost
operation sits at the top — a tree of `reversal { charade { ... } }` reads
"reverse the result of the charade".

Naming convention for compounds (used in reports and the unknown-op tag):
the chronologically EARLIEST operation comes first. So an anagram of the
result of a deletion is a "deletion anagram" — the deletion happened first,
then the anagram. The tree's outer-op-first rule and the name's
earliest-op-first rule are the same statement read from opposite ends.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Union, Any
import json


# --- Allowed operation vocabularies -----------------------------------------

LEAF_OPERATIONS = {"literal", "synonym", "abbreviation", "positional"}

OP_OPERATIONS = {
    "charade", "container", "anagram", "reversal", "deletion",
    "hidden", "double_definition",
    # Acrostic — take the first (or last/middle/etc.) letter of each
    # child's source_word and concatenate. Distinct from a charade of
    # positional leaves because the licensing indicator ("at first",
    # "initially", etc.) sits on the acrostic node, not on each leaf.
    "acrostic",
}

ALL_OPERATIONS = LEAF_OPERATIONS | OP_OPERATIONS

# Sentinel for anything the translator can't yet name.
UNKNOWN_OPERATION = "unknown"

# Positional kinds — the seven extractions covered by the live verifier.
POSITIONAL_KINDS = {
    "first", "last", "middle", "outer",
    "initial", "final", "odd", "even", "alternate",
}

# Deletion kinds — head/tail/outer/heart, matching the live verifier's
# check_deletion_mechanism().
DELETION_KINDS = {"head", "tail", "outer", "heart"}

# Operation arity — number of direct children expected. None means "any".
OP_ARITY = {
    "charade":           None,   # any number >= 2 in practice; >=1 admitted
    "container":         2,      # [outer, inner]
    "anagram":           None,   # one or more children whose letters are pooled
    "reversal":          1,
    "deletion":          1,
    "hidden":            None,   # one or more child literals spanning the source
    "double_definition": 2,      # [defn_phrase_left, defn_phrase_right]
    "acrostic":          None,   # one or more children — first letter of each
}


# --- Leaf and Op dataclasses ------------------------------------------------

@dataclass
class Leaf:
    """Terminal node — produces letters from a single source word."""
    operation: str               # literal | synonym | abbreviation | positional
    source_word: str             # the clue word the leaf was derived from
    value: str                   # the produced letters (UPPERCASE)

    # For positional leaves: which slice (first/last/middle/etc.).
    # For other leaves: None.
    positional_kind: Optional[str] = None

    # For positional leaves: the indicator word in the clue (e.g. "head of"
    # for a "first" extraction). For other leaves: None.
    positional_indicator: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"kind": "leaf", "operation": self.operation,
             "source_word": self.source_word, "value": self.value}
        if self.positional_kind is not None:
            d["positional_kind"] = self.positional_kind
        if self.positional_indicator is not None:
            d["positional_indicator"] = self.positional_indicator
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Leaf":
        return cls(
            operation=d["operation"],
            source_word=d["source_word"],
            value=d["value"],
            positional_kind=d.get("positional_kind"),
            positional_indicator=d.get("positional_indicator"),
        )


@dataclass
class Op:
    """Operation node — combines or transforms one or more children."""
    operation: str               # one of OP_OPERATIONS (or UNKNOWN_OPERATION)
    indicator: Optional[str]     # the clue word(s) that license the op
    sources: list                # list of Leaf | Op (children)

    # Deletion-only: which letters were dropped (head/tail/outer/heart).
    deletion_kind: Optional[str] = None

    # Acrostic-only: which letter slice of each child's source_word
    # (first/last/middle/etc.). Defaults to "first" — the most common.
    acrostic_kind: Optional[str] = None

    # For UNKNOWN_OPERATION: keep the original flat-form op name so the
    # report can surface it for naming.
    flat_op: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "kind": "op",
            "operation": self.operation,
            "indicator": self.indicator,
            "sources": [s.to_dict() for s in self.sources],
        }
        if self.deletion_kind is not None:
            d["deletion_kind"] = self.deletion_kind
        if self.acrostic_kind is not None:
            d["acrostic_kind"] = self.acrostic_kind
        if self.flat_op is not None:
            d["flat_op"] = self.flat_op
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Op":
        return cls(
            operation=d["operation"],
            indicator=d.get("indicator"),
            sources=[node_from_dict(s) for s in d.get("sources", [])],
            deletion_kind=d.get("deletion_kind"),
            acrostic_kind=d.get("acrostic_kind"),
            flat_op=d.get("flat_op"),
        )


Node = Union[Leaf, Op]


def node_from_dict(d: dict) -> Node:
    """Deserialise either a Leaf or an Op from its dict form."""
    kind = d.get("kind")
    if kind == "leaf":
        return Leaf.from_dict(d)
    if kind == "op":
        return Op.from_dict(d)
    raise ValueError(f"Unknown node kind: {kind!r}")


# --- Definition + Form ------------------------------------------------------

@dataclass
class Definition:
    phrase: str                  # the clue-text phrase that defines the answer
    answer: str                  # the answer (uppercase, no spaces)

    def to_dict(self) -> dict:
        return {"phrase": self.phrase, "answer": self.answer}

    @classmethod
    def from_dict(cls, d: dict) -> "Definition":
        return cls(phrase=d["phrase"], answer=d["answer"])


@dataclass
class Form:
    """Universal form for a single clue."""
    tree: Node
    definition: Definition
    link_words: list = field(default_factory=list)
    is_and_lit: bool = False

    def to_dict(self) -> dict:
        return {
            "tree": self.tree.to_dict(),
            "definition": self.definition.to_dict(),
            "link_words": list(self.link_words),
            "is_and_lit": self.is_and_lit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Form":
        return cls(
            tree=node_from_dict(d["tree"]),
            definition=Definition.from_dict(d["definition"]),
            link_words=list(d.get("link_words", [])),
            is_and_lit=bool(d.get("is_and_lit", False)),
        )

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Form":
        return cls.from_dict(json.loads(s))


# --- Validation -------------------------------------------------------------

class FormValidationError(ValueError):
    """Raised when a Form fails structural validation."""


def validate(form: Form) -> list[str]:
    """Return a list of validation problems. Empty list = valid form.

    Catches structural mistakes — wrong operation names, wrong arity,
    leaf with operation that requires children, op with no children, etc.
    Does NOT do semantic checks (assembly, bridge, residue) — those are the
    verifier's job.
    """
    problems: list[str] = []
    _walk_validate(form.tree, "tree", problems)
    if not form.definition.phrase:
        problems.append("definition.phrase is empty")
    if not form.definition.answer:
        problems.append("definition.answer is empty")
    return problems


def _walk_validate(node: Node, path: str, problems: list[str]) -> None:
    if isinstance(node, Leaf):
        if node.operation not in LEAF_OPERATIONS:
            problems.append(
                f"{path}: leaf has non-leaf operation {node.operation!r} "
                f"(allowed: {sorted(LEAF_OPERATIONS)})")
        if not node.source_word:
            problems.append(f"{path}: leaf has empty source_word")
        if not node.value:
            problems.append(f"{path}: leaf has empty value")
        if node.operation == "positional":
            if node.positional_kind not in POSITIONAL_KINDS:
                problems.append(
                    f"{path}: positional leaf has invalid kind "
                    f"{node.positional_kind!r}")
        else:
            if node.positional_kind is not None:
                problems.append(
                    f"{path}: non-positional leaf has positional_kind set")
        return

    if isinstance(node, Op):
        if node.operation == UNKNOWN_OPERATION:
            # Unknowns are allowed (the prototype surfaces them rather
            # than crashing) but must carry their flat_op for the report.
            if not node.flat_op:
                problems.append(f"{path}: unknown op missing flat_op tag")
        elif node.operation not in OP_OPERATIONS:
            problems.append(
                f"{path}: op has unknown operation {node.operation!r} "
                f"(allowed: {sorted(OP_OPERATIONS)} or 'unknown')")

        if node.operation in OP_ARITY:
            expected = OP_ARITY[node.operation]
            if expected is not None and len(node.sources) != expected:
                problems.append(
                    f"{path}: {node.operation} expects {expected} children, "
                    f"got {len(node.sources)}")
            if expected is None and len(node.sources) < 1:
                problems.append(
                    f"{path}: {node.operation} expects >=1 children, got 0")

        if node.operation == "deletion":
            if node.deletion_kind is not None and \
                    node.deletion_kind not in DELETION_KINDS:
                problems.append(
                    f"{path}: deletion has invalid kind "
                    f"{node.deletion_kind!r}")
        else:
            if node.deletion_kind is not None:
                problems.append(
                    f"{path}: non-deletion op has deletion_kind set")

        for i, child in enumerate(node.sources):
            _walk_validate(child, f"{path}.sources[{i}]", problems)
        return

    problems.append(f"{path}: object is neither Leaf nor Op")


# --- Convenience constructors ----------------------------------------------
# Short aliases for hand-building example trees in examples.py and tests.
# Production translators don't need them but they make the examples readable.

def lit(source_word: str, value: Optional[str] = None) -> Leaf:
    """Literal leaf — letters are the source_word's own letters."""
    if value is None:
        value = source_word.upper().replace(" ", "")
    return Leaf("literal", source_word, value)


def syn(source_word: str, value: str) -> Leaf:
    return Leaf("synonym", source_word, value.upper())


def abbr(source_word: str, value: str) -> Leaf:
    return Leaf("abbreviation", source_word, value.upper())


def pos(source_word: str, value: str, kind: str,
        indicator: Optional[str] = None) -> Leaf:
    return Leaf("positional", source_word, value.upper(),
                positional_kind=kind, positional_indicator=indicator)


def charade(*sources: Node, indicator: Optional[str] = None) -> Op:
    return Op("charade", indicator, list(sources))


def container(outer: Node, inner: Node, indicator: Optional[str] = None) -> Op:
    return Op("container", indicator, [outer, inner])


def anagram(*sources: Node, indicator: str) -> Op:
    return Op("anagram", indicator, list(sources))


def reversal(source: Node, indicator: str) -> Op:
    return Op("reversal", indicator, [source])


def deletion(source: Node, indicator: str, kind: str) -> Op:
    return Op("deletion", indicator, [source], deletion_kind=kind)


def hidden(*sources: Node, indicator: Optional[str] = None) -> Op:
    return Op("hidden", indicator, list(sources))


def double_definition(left: Node, right: Node) -> Op:
    return Op("double_definition", None, [left, right])


def acrostic(*sources: Node, indicator: str, kind: str = "first") -> Op:
    """First-letter (or last/middle/etc.) of each child's source_word."""
    return Op("acrostic", indicator, list(sources), acrostic_kind=kind)


def unknown(flat_op: str, sources: Optional[list] = None,
            indicator: Optional[str] = None) -> Op:
    """Sentinel for any operation the translator can't yet name."""
    return Op(UNKNOWN_OPERATION, indicator, sources or [], flat_op=flat_op)
