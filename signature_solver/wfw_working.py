"""Working blocks and transformations for WFW proof records.

These structures sit above character atoms.  They record produced cryptic
material, explicit transformations, and answer-letter placement without trying
to solve a clue by themselves.
"""
from __future__ import annotations

from dataclasses import dataclass

from .wfw_atoms import CharAtom, atomize_characters


@dataclass(frozen=True)
class WorkingBlock:
    block_id: str
    kind: str
    text: str
    value: str
    char_atoms: tuple[CharAtom, ...]
    source_atom_ids: tuple[str, ...] = ()
    source_token_ids: tuple[str, ...] = ()
    mechanism: str | None = None
    parent_block_ids: tuple[str, ...] = ()

    def as_dict(self):
        return {
            "block_id": self.block_id,
            "kind": self.kind,
            "text": self.text,
            "value": self.value,
            "char_atoms": [atom.as_dict() for atom in self.char_atoms],
            "source_atom_ids": list(self.source_atom_ids),
            "source_token_ids": list(self.source_token_ids),
            "mechanism": self.mechanism,
            "parent_block_ids": list(self.parent_block_ids),
        }


@dataclass(frozen=True)
class WFWTransformation:
    transform_id: str
    operation: str
    input_block_ids: tuple[str, ...]
    output_block_id: str
    controller_atom_ids: tuple[str, ...] = ()
    removed_char_atom_ids: tuple[str, ...] = ()
    detail: str = ""

    def as_dict(self):
        return {
            "transform_id": self.transform_id,
            "operation": self.operation,
            "input_block_ids": list(self.input_block_ids),
            "output_block_id": self.output_block_id,
            "controller_atom_ids": list(self.controller_atom_ids),
            "removed_char_atom_ids": list(self.removed_char_atom_ids),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AnswerPlacement:
    answer_atom_id: str
    answer_position: int
    answer_letter: str
    source_block_id: str
    source_char_atom_id: str
    source_index: int

    def as_dict(self):
        return {
            "answer_atom_id": self.answer_atom_id,
            "answer_position": self.answer_position,
            "answer_letter": self.answer_letter,
            "source_block_id": self.source_block_id,
            "source_char_atom_id": self.source_char_atom_id,
            "source_index": self.source_index,
        }


def produced_block(block_id, value, kind="working", text=None,
                   mechanism=None, source_tokens=(), source_atoms=(),
                   parent_blocks=()):
    """Create a produced working block with its own stable character atoms."""
    clean_value = _clean_letters(value)
    return WorkingBlock(
        block_id=block_id,
        kind=kind,
        text=text if text is not None else clean_value,
        value=clean_value,
        char_atoms=atomize_characters(block_id, clean_value,
                                      number_letters=True),
        source_atom_ids=tuple(source_atoms),
        source_token_ids=tuple(source_tokens),
        mechanism=mechanism,
        parent_block_ids=tuple(parent_blocks),
    )


def source_block_from_tokens(context, block_id, token_indices, value,
                             mechanism):
    """Create a produced block from original clue token provenance."""
    tokens = tuple(context.clue_tokens[i] for i in token_indices)
    source_atom_ids = tuple(
        atom_id for token in tokens for atom_id in token.atom_ids)
    return produced_block(
        block_id=block_id,
        value=value,
        kind="source_piece",
        text=" ".join(token.text for token in tokens),
        mechanism=mechanism,
        source_tokens=tuple(token.token_id for token in tokens),
        source_atoms=source_atom_ids,
    )


def controller_atom_ids(context, token_indices):
    return tuple(
        atom_id
        for idx in token_indices
        for atom_id in context.clue_tokens[idx].atom_ids
    )


def reverse_block(block, output_id, controller_atoms=()):
    value = block.value[::-1]
    out = produced_block(
        output_id,
        value,
        kind="transformed_piece",
        text=value,
        mechanism="reversal",
        source_atoms=block.source_atom_ids,
        source_tokens=block.source_token_ids,
        parent_blocks=(block.block_id,),
    )
    transform = WFWTransformation(
        transform_id="%s_reversal" % output_id,
        operation="reversal",
        input_block_ids=(block.block_id,),
        output_block_id=out.block_id,
        controller_atom_ids=tuple(controller_atoms),
        detail="%s reversed = %s" % (block.value, out.value),
    )
    return out, transform


def trim_block(block, output_id, side, controller_atoms=()):
    """Trim first or last letter from a working block."""
    if side not in {"first", "last"}:
        raise ValueError("side must be 'first' or 'last'")
    if not block.char_atoms:
        raise ValueError("cannot trim an empty block")
    if side == "first":
        kept_atoms = block.char_atoms[1:]
        removed = block.char_atoms[:1]
    else:
        kept_atoms = block.char_atoms[:-1]
        removed = block.char_atoms[-1:]
    value = "".join(atom.char for atom in kept_atoms)
    out = produced_block(
        output_id,
        value,
        kind="transformed_piece",
        text=value,
        mechanism="trim_%s" % side,
        source_atoms=block.source_atom_ids,
        source_tokens=block.source_token_ids,
        parent_blocks=(block.block_id,),
    )
    transform = WFWTransformation(
        transform_id="%s_trim_%s" % (output_id, side),
        operation="trim_%s" % side,
        input_block_ids=(block.block_id,),
        output_block_id=out.block_id,
        controller_atom_ids=tuple(controller_atoms),
        removed_char_atom_ids=tuple(atom.atom_id for atom in removed),
        detail="%s without %s letter = %s" % (
            block.value, side, out.value),
    )
    return out, transform


def place_charade(answer_atoms, blocks):
    """Map ordered working-block letters onto answer letter atoms.

    Raises ValueError rather than silently publishing an unproven placement.
    """
    answer_letters = tuple(
        atom for atom in answer_atoms if atom.letter_position is not None)
    produced = "".join(block.value for block in blocks)
    target = "".join(atom.normalized for atom in answer_letters)
    if produced != target:
        raise ValueError("charade mismatch: %s != %s" % (produced, target))

    placements = []
    answer_index = 0
    for block in blocks:
        for source_index, source_atom in enumerate(block.char_atoms):
            answer_atom = answer_letters[answer_index]
            placements.append(AnswerPlacement(
                answer_atom_id=answer_atom.atom_id,
                answer_position=answer_atom.letter_position,
                answer_letter=answer_atom.normalized,
                source_block_id=block.block_id,
                source_char_atom_id=source_atom.atom_id,
                source_index=source_index,
            ))
            answer_index += 1
    return tuple(placements)


def container_block(outer, inner_blocks, output_id, target,
                    controller_atoms=()):
    """Create a container output block if outer + inner can make target."""
    inner_value = "".join(block.value for block in inner_blocks)
    target = _clean_letters(target)
    insert_pos = _container_insert_pos(outer.value, inner_value, target)
    if insert_pos is None:
        raise ValueError("container mismatch: %s around %s != %s" % (
            outer.value, inner_value, target))
    out = produced_block(
        output_id,
        target,
        kind="assembled_piece",
        text=target,
        mechanism="container",
        source_atoms=outer.source_atom_ids + tuple(
            atom_id
            for block in inner_blocks
            for atom_id in block.source_atom_ids
        ),
        source_tokens=outer.source_token_ids + tuple(
            token_id
            for block in inner_blocks
            for token_id in block.source_token_ids
        ),
        parent_blocks=(outer.block_id,) + tuple(
            block.block_id for block in inner_blocks),
    )
    transform = WFWTransformation(
        transform_id="%s_container" % output_id,
        operation="container",
        input_block_ids=(outer.block_id,) + tuple(
            block.block_id for block in inner_blocks),
        output_block_id=out.block_id,
        controller_atom_ids=tuple(controller_atoms),
        detail="%s contains %s = %s" % (outer.value, inner_value, target),
    )
    return out, transform, insert_pos


def place_container(answer_atoms, outer, inner_blocks, insert_pos):
    """Map a container result onto answer atoms from original source blocks."""
    answer_letters = tuple(
        atom for atom in answer_atoms if atom.letter_position is not None)
    inner_chars = [
        (block, atom)
        for block in inner_blocks
        for atom in block.char_atoms
    ]
    source_chars = []
    for idx, atom in enumerate(outer.char_atoms):
        if idx == insert_pos:
            source_chars.extend(inner_chars)
        source_chars.append((outer, atom))
    if insert_pos == len(outer.char_atoms):
        source_chars.extend(inner_chars)
    if len(source_chars) != len(answer_letters):
        raise ValueError("container placement length mismatch")
    produced = "".join(atom.char for _, atom in source_chars)
    target = "".join(atom.normalized for atom in answer_letters)
    if produced != target:
        raise ValueError("container placement mismatch: %s != %s" % (
            produced, target))

    placements = []
    source_index_by_block = {}
    for answer_atom, (block, source_atom) in zip(answer_letters, source_chars):
        source_index = source_index_by_block.get(block.block_id, 0)
        source_index_by_block[block.block_id] = source_index + 1
        placements.append(AnswerPlacement(
            answer_atom_id=answer_atom.atom_id,
            answer_position=answer_atom.letter_position,
            answer_letter=answer_atom.normalized,
            source_block_id=block.block_id,
            source_char_atom_id=source_atom.atom_id,
            source_index=source_index,
        ))
    return tuple(placements)


def anagram_block(fodder_blocks, output_id, target, controller_atoms=()):
    """Create an anagram output block if fodder letters match target."""
    fodder = "".join(block.value for block in fodder_blocks)
    target = _clean_letters(target)
    if sorted(fodder) != sorted(target):
        raise ValueError("anagram mismatch: sorted(%s) != sorted(%s)" % (
            fodder, target))
    out = produced_block(
        output_id,
        target,
        kind="assembled_piece",
        text=target,
        mechanism="anagram",
        source_atoms=tuple(
            atom_id
            for block in fodder_blocks
            for atom_id in block.source_atom_ids
        ),
        source_tokens=tuple(
            token_id
            for block in fodder_blocks
            for token_id in block.source_token_ids
        ),
        parent_blocks=tuple(block.block_id for block in fodder_blocks),
    )
    transform = WFWTransformation(
        transform_id="%s_anagram" % output_id,
        operation="anagram",
        input_block_ids=tuple(block.block_id for block in fodder_blocks),
        output_block_id=out.block_id,
        controller_atom_ids=tuple(controller_atoms),
        detail="anagram of %s = %s" % (fodder, target),
    )
    return out, transform


def hidden_block(source, output_id, target, controller_atoms=()):
    """Create a hidden-word output block if target appears inside source."""
    target = _clean_letters(target)
    start = source.value.find(target)
    reversed_hidden = False
    if start < 0:
        reversed_target = target[::-1]
        start = source.value.find(reversed_target)
        reversed_hidden = start >= 0
    if start < 0:
        raise ValueError("hidden mismatch: %s not in %s" % (
            target, source.value))
    hidden_value = (
        source.value[start:start + len(target)]
        if not reversed_hidden else target
    )
    out = produced_block(
        output_id,
        hidden_value,
        kind="transformed_piece",
        text=hidden_value,
        mechanism="hidden_reversed" if reversed_hidden else "hidden",
        source_atoms=source.source_atom_ids,
        source_tokens=source.source_token_ids,
        parent_blocks=(source.block_id,),
    )
    detail = "%s hidden in %s" % (target, source.value)
    if reversed_hidden:
        detail = "%s hidden backwards in %s" % (target, source.value)
    transform = WFWTransformation(
        transform_id="%s_hidden" % output_id,
        operation="hidden_reversed" if reversed_hidden else "hidden",
        input_block_ids=(source.block_id,),
        output_block_id=out.block_id,
        controller_atom_ids=tuple(controller_atoms),
        detail=detail,
    )
    return out, transform, start, reversed_hidden


def place_anagram(answer_atoms, fodder_blocks):
    """Map answer letters to consumed fodder block characters."""
    answer_letters = tuple(
        atom for atom in answer_atoms if atom.letter_position is not None)
    pool = [
        {
            "block": block,
            "atom": atom,
            "index": idx,
            "used": False,
        }
        for block in fodder_blocks
        for idx, atom in enumerate(block.char_atoms)
    ]
    placements = []
    for answer_atom in answer_letters:
        match = next(
            (item for item in pool
             if not item["used"]
             and item["atom"].normalized == answer_atom.normalized),
            None,
        )
        if match is None:
            raise ValueError("anagram placement missing letter:%s" % (
                answer_atom.normalized,))
        match["used"] = True
        block = match["block"]
        placements.append(AnswerPlacement(
            answer_atom_id=answer_atom.atom_id,
            answer_position=answer_atom.letter_position,
            answer_letter=answer_atom.normalized,
            source_block_id=block.block_id,
            source_char_atom_id=match["atom"].atom_id,
            source_index=match["index"],
        ))
    unused = [item for item in pool if not item["used"]]
    if unused:
        raise ValueError("anagram placement unused fodder")
    return tuple(placements)


def place_hidden(answer_atoms, source, start, reversed_hidden=False):
    """Map hidden answer letters to the source block character positions."""
    answer_letters = tuple(
        atom for atom in answer_atoms if atom.letter_position is not None)
    if start < 0 or start + len(answer_letters) > len(source.char_atoms):
        raise ValueError("hidden placement outside source")

    if reversed_hidden:
        source_chars = tuple(
            reversed(source.char_atoms[start:start + len(answer_letters)]))
    else:
        source_chars = source.char_atoms[start:start + len(answer_letters)]

    produced = "".join(atom.normalized for atom in source_chars)
    target = "".join(atom.normalized for atom in answer_letters)
    if produced != target:
        raise ValueError("hidden placement mismatch: %s != %s" % (
            produced, target))

    placements = []
    for answer_atom, source_atom in zip(answer_letters, source_chars):
        placements.append(AnswerPlacement(
            answer_atom_id=answer_atom.atom_id,
            answer_position=answer_atom.letter_position,
            answer_letter=answer_atom.normalized,
            source_block_id=source.block_id,
            source_char_atom_id=source_atom.atom_id,
            source_index=source_atom.letter_position - 1,
        ))
    return tuple(placements)


def _clean_letters(value):
    return "".join(char.upper() for char in (value or "") if char.isalpha())


def _container_insert_pos(outer, inner, target):
    for pos in range(0, len(outer) + 1):
        if outer[:pos] + inner + outer[pos:] == target:
            return pos
    return None
