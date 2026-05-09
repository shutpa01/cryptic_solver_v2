"""Universal form schema (v2 — fresh implementation from the spec).

Per `memory/project_universal_form_schema.md` the form has four fields:

    {
      "tree":        <recursive node>,
      "definition":  {"phrase": "...", "answer": "..."},
      "link_words":  [...],
      "is_and_lit":  bool
    }

A node has:
    operation:  one of LEAF_OPERATIONS or NON_LEAF_OPERATIONS
    indicator:  the surface word(s) that signal the operation, or None
                (always None for leaves)
    sources:    list of child nodes, OR for leaves the empty list

A leaf additionally carries:
    value:        the letters the leaf contributes
    source_word:  the clue word(s) the leaf's letters came from

Some nodes carry a "kind" (positional kind, deletion kind, acrostic kind).
These are typed sub-discriminators kept on the node as plain string fields.

The adapter may attach `flags` to a node (or the form) to surface gaps in
what the production solver supplied — these are part of the worklist.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


LEAF_OPERATIONS = frozenset({
    "literal", "synonym", "abbreviation", "positional", "homophone",
    # raw-letter sources from the clue (not in DB-as-synonym form)
    "raw",
})

NON_LEAF_OPERATIONS = frozenset({
    "charade", "anagram", "reversal", "container", "deletion",
    "hidden", "double_definition", "acrostic", "cryptic_definition",
    "homophone", "substitution", "spoonerism",
    # marker for unrecognised operation — the form is built but flagged
    "unknown",
})


@dataclass
class Node:
    operation: str
    indicator: Optional[str] = None
    sources: list = field(default_factory=list)
    # leaf-only fields (None on internal nodes):
    value: Optional[str] = None
    source_word: Optional[str] = None
    # operation-specific kind (None when not applicable):
    positional_kind: Optional[str] = None  # first/last/outer/middle/odd/even/alternate
    deletion_kind: Optional[str] = None    # head/tail/outer/heart
    acrostic_kind: Optional[str] = None    # first/last
    # adapter flags — TODO entries for production solver:
    flags: list = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.operation in LEAF_OPERATIONS

    def to_dict(self) -> dict:
        d = {"operation": self.operation}
        if self.indicator is not None:
            d["indicator"] = self.indicator
        if self.is_leaf and not self.sources:
            d["value"] = self.value
            d["source_word"] = self.source_word
        else:
            d["sources"] = [s.to_dict() for s in self.sources]
            if self.value is not None:
                d["value"] = self.value
            if self.source_word is not None:
                d["source_word"] = self.source_word
        if self.positional_kind:
            d["positional_kind"] = self.positional_kind
        if self.deletion_kind:
            d["deletion_kind"] = self.deletion_kind
        if self.acrostic_kind:
            d["acrostic_kind"] = self.acrostic_kind
        if self.flags:
            d["flags"] = list(self.flags)
        return d


@dataclass
class Definition:
    phrase: str
    answer: str

    def to_dict(self) -> dict:
        return {"phrase": self.phrase, "answer": self.answer}


@dataclass
class Form:
    tree: Node
    definition: Definition
    link_words: list = field(default_factory=list)
    is_and_lit: bool = False
    flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tree": self.tree.to_dict(),
            "definition": self.definition.to_dict(),
            "link_words": list(self.link_words),
            "is_and_lit": self.is_and_lit,
            "flags": list(self.flags),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# --- Construction helpers (concise factories) ------------------------------

def lit(source_word: str, value: str) -> Node:
    return Node(operation="literal", value=value, source_word=source_word)


def syn(source_word: str, value: str) -> Node:
    return Node(operation="synonym", value=value, source_word=source_word)


def abbr(source_word: str, value: str) -> Node:
    return Node(operation="abbreviation", value=value, source_word=source_word)


def raw(source_word: str, value: str) -> Node:
    return Node(operation="raw", value=value, source_word=source_word)


def positional(source_word: str, value: str, kind: str) -> Node:
    return Node(operation="positional", value=value,
                source_word=source_word, positional_kind=kind)


def homophone_leaf(source_word: str, value: str) -> Node:
    return Node(operation="homophone", value=value, source_word=source_word)


def charade(*children: Node, indicator: Optional[str] = None) -> Node:
    return Node(operation="charade", indicator=indicator,
                sources=list(children))


def anagram(*children: Node, indicator: Optional[str] = None) -> Node:
    return Node(operation="anagram", indicator=indicator,
                sources=list(children))


def reversal(child: Node, indicator: Optional[str] = None) -> Node:
    return Node(operation="reversal", indicator=indicator, sources=[child])


def container(outer: Node, inner: Node,
              indicator: Optional[str] = None) -> Node:
    return Node(operation="container", indicator=indicator,
                sources=[outer, inner])


def deletion(source: Node, indicator: Optional[str] = None,
             kind: str = "tail") -> Node:
    return Node(operation="deletion", indicator=indicator,
                sources=[source], deletion_kind=kind)


def hidden(*children: Node, indicator: Optional[str] = None) -> Node:
    return Node(operation="hidden", indicator=indicator,
                sources=list(children))


def double_definition(left: Node, right: Node) -> Node:
    return Node(operation="double_definition", sources=[left, right])


def acrostic(*children: Node, indicator: Optional[str] = None,
             kind: str = "first") -> Node:
    return Node(operation="acrostic", indicator=indicator,
                sources=list(children), acrostic_kind=kind)


def unknown(flat_op: Optional[str] = None,
            sources: Optional[list] = None,
            indicator: Optional[str] = None) -> Node:
    n = Node(operation="unknown", indicator=indicator,
             sources=list(sources or []))
    if flat_op:
        n.flags.append(f"flat_op={flat_op}")
    return n
