"""Materialise token parses into WFW working-out records."""
from __future__ import annotations

from dataclasses import dataclass
import re

from .wfw_working import (
    WFWTransformation,
    anagram_block,
    container_block,
    hidden_block,
    place_anagram,
    place_charade,
    place_container,
    place_hidden,
    produced_block,
    reverse_block,
    source_block_from_tokens,
)


@dataclass(frozen=True)
class WFWNativeAssembly:
    parse_id: str
    operation: str
    status: str
    working_blocks: tuple
    transformations: tuple
    placements: tuple
    objections: tuple[str, ...] = ()

    def as_dict(self):
        return {
            "parse_id": self.parse_id,
            "operation": self.operation,
            "status": self.status,
            "working_blocks": [
                block.as_dict() if hasattr(block, "as_dict") else block
                for block in self.working_blocks
            ],
            "transformations": [
                transform.as_dict()
                if hasattr(transform, "as_dict") else transform
                for transform in self.transformations
            ],
            "placements": [
                placement.as_dict()
                if hasattr(placement, "as_dict") else placement
                for placement in self.placements
            ],
            "objections": list(self.objections),
        }


@dataclass(frozen=True)
class _SourceMaterial:
    output: object
    working_blocks: tuple
    transformations: tuple


def materialise_wfw_assembly(atom_context, clue_context, token_parse):
    """Return WFW working records for a mechanically verified token parse."""
    try:
        if token_parse.operation == "charade":
            return _materialise_charade(atom_context, clue_context,
                                        token_parse)
        if token_parse.operation == "reversal":
            return _materialise_reversal(atom_context, clue_context,
                                         token_parse)
        if token_parse.operation == "anagram":
            return _materialise_anagram(atom_context, clue_context,
                                        token_parse)
        if token_parse.operation in ("hidden", "hidden_reversed"):
            return _materialise_hidden(atom_context, clue_context,
                                       token_parse)
        if token_parse.operation == "container":
            return _materialise_container(atom_context, clue_context,
                                          token_parse)
        if token_parse.operation == "deletion":
            return _materialise_deletion(atom_context, clue_context,
                                         token_parse)
        if token_parse.operation == "homophone":
            return _materialise_homophone(atom_context, clue_context,
                                          token_parse)
        if token_parse.operation == "substitution":
            return _materialise_substitution(atom_context, clue_context,
                                             token_parse)
        if token_parse.operation == "double_definition":
            return _materialise_double_definition(atom_context, clue_context,
                                                  token_parse)
        if token_parse.operation.endswith("_charade"):
            return _materialise_compound_charade(atom_context, clue_context,
                                                 token_parse)
        return _unsupported(token_parse, "operation_not_materialised_yet")
    except Exception as exc:
        return _unsupported(token_parse, "materialisation_failed:%s" % exc)


def _materialise_charade(atom_context, clue_context, token_parse):
    sources = [
        block for block in token_parse.blocks
        if block.kind == "SOURCE_BLOCK"
    ]
    source_material = tuple(
        _source_material(atom_context, clue_context, token_parse, block)
        for block in sources
    )
    placement_blocks = tuple(item.output for item in source_material)
    working = tuple(
        block for item in source_material for block in item.working_blocks
    )
    transformations = tuple(
        transform
        for item in source_material
        for transform in item.transformations
    )
    placements = place_charade(atom_context.answer_atoms, placement_blocks)
    return WFWNativeAssembly(
        parse_id=token_parse.parse_id,
        operation=token_parse.operation,
        status="materialised",
        working_blocks=working,
        transformations=transformations,
        placements=placements,
    )


def _materialise_reversal(atom_context, clue_context, token_parse):
    source = _first_block(token_parse, "SOURCE_BLOCK")
    op = _first_operation_block(token_parse)
    source_material = _source_material(
        atom_context, clue_context, token_parse, source)
    start = source_material.output
    reversed_block, transform = reverse_block(
        start,
        "%s_reversed" % start.block_id,
        _controller_atoms(atom_context, clue_context, op),
    )
    placements = place_charade(atom_context.answer_atoms, (reversed_block,))
    return WFWNativeAssembly(
        parse_id=token_parse.parse_id,
        operation=token_parse.operation,
        status="materialised",
        working_blocks=source_material.working_blocks + (reversed_block,),
        transformations=source_material.transformations + (transform,),
        placements=placements,
    )


def _materialise_anagram(atom_context, clue_context, token_parse):
    sources = [
        block for block in token_parse.blocks
        if block.kind == "SOURCE_BLOCK"
    ]
    op = _first_operation_block(token_parse)
    source_material = tuple(
        _source_material(atom_context, clue_context, token_parse, block)
        for block in sources
    )
    fodder = tuple(item.output for item in source_material)
    output, transform = anagram_block(
        fodder,
        "wfw_%s_output" % _clean_id(token_parse.parse_id),
        token_parse.answer,
        _controller_atoms(atom_context, clue_context, op),
    )
    placements = place_anagram(atom_context.answer_atoms, fodder)
    return WFWNativeAssembly(
        parse_id=token_parse.parse_id,
        operation=token_parse.operation,
        status="materialised",
        working_blocks=tuple(
            block
            for item in source_material
            for block in item.working_blocks
        ) + (output,),
        transformations=tuple(
            source_transform
            for item in source_material
            for source_transform in item.transformations
        ) + (transform,),
        placements=placements,
    )


def _materialise_hidden(atom_context, clue_context, token_parse):
    source = _first_block(token_parse, "SOURCE_BLOCK")
    op = _first_operation_block(token_parse)
    source_material = _source_material(
        atom_context, clue_context, token_parse, source)
    source_work = source_material.output
    output, transform, start, reversed_hidden = hidden_block(
        source_work,
        "wfw_%s_output" % _clean_id(token_parse.parse_id),
        token_parse.answer,
        _controller_atoms(atom_context, clue_context, op),
    )
    placements = place_hidden(
        atom_context.answer_atoms, source_work, start, reversed_hidden)
    return WFWNativeAssembly(
        parse_id=token_parse.parse_id,
        operation=token_parse.operation,
        status="materialised",
        working_blocks=source_material.working_blocks + (output,),
        transformations=source_material.transformations + (transform,),
        placements=placements,
    )


def _materialise_container(atom_context, clue_context, token_parse):
    blocks = {block.block_id: block for block in token_parse.blocks}
    outer_material = _source_material(
        atom_context, clue_context, token_parse, blocks["src_outer"])
    inner_material = _source_material(
        atom_context, clue_context, token_parse, blocks["src_inner"])
    outer = outer_material.output
    inner = inner_material.output
    op = blocks.get("op_0")
    output, transform, insert_pos = container_block(
        outer,
        (inner,),
        "wfw_%s_output" % _clean_id(token_parse.parse_id),
        token_parse.answer,
        _controller_atoms(atom_context, clue_context, op),
    )
    placements = place_container(
        atom_context.answer_atoms, outer, (inner,), insert_pos)
    return WFWNativeAssembly(
        parse_id=token_parse.parse_id,
        operation=token_parse.operation,
        status="materialised",
        working_blocks=(
            outer_material.working_blocks
            + inner_material.working_blocks
            + (output,)
        ),
        transformations=(
            outer_material.transformations
            + inner_material.transformations
            + (transform,)
        ),
        placements=placements,
    )


def _materialise_deletion(atom_context, clue_context, token_parse):
    source = _first_block(token_parse, "SOURCE_BLOCK")
    op = _first_operation_block(token_parse)
    start = source_block_from_tokens(
        atom_context,
        "wfw_%s_before" % _clean_id(source.block_id),
        _wfw_indices_for_span(atom_context, clue_context, source.span),
        source.text,
        source.token or source.role or "deletion_source",
    )
    output = _source_block(atom_context, clue_context, source)
    transform = WFWTransformation(
        transform_id="wfw_%s_deletion" % _clean_id(token_parse.parse_id),
        operation="deletion",
        input_block_ids=(start.block_id,),
        output_block_id=output.block_id,
        controller_atom_ids=_controller_atoms(
            atom_context, clue_context, op),
        detail="%s trimmed to %s" % (source.text, output.value),
    )
    placements = place_charade(atom_context.answer_atoms, (output,))
    return WFWNativeAssembly(
        parse_id=token_parse.parse_id,
        operation=token_parse.operation,
        status="materialised",
        working_blocks=(start, output),
        transformations=(transform,),
        placements=placements,
    )


def _materialise_homophone(atom_context, clue_context, token_parse):
    source = _first_block(token_parse, "SOURCE_BLOCK")
    op = _first_operation_block(token_parse)
    source_material = _source_material(
        atom_context, clue_context, token_parse, source)
    heard = source_material.output
    output = produced_block(
        "wfw_%s_output" % _clean_id(token_parse.parse_id),
        token_parse.answer,
        kind="transformed_piece",
        text=token_parse.answer,
        mechanism="homophone",
        source_tokens=heard.source_token_ids,
        source_atoms=heard.source_atom_ids,
        parent_blocks=(heard.block_id,),
    )
    transform = WFWTransformation(
        transform_id="wfw_%s_homophone" % _clean_id(token_parse.parse_id),
        operation="homophone",
        input_block_ids=(heard.block_id,),
        output_block_id=output.block_id,
        controller_atom_ids=_controller_atoms(
            atom_context, clue_context, op),
        detail="%s sounds like %s" % (heard.text, token_parse.answer),
    )
    placements = place_charade(atom_context.answer_atoms, (output,))
    return WFWNativeAssembly(
        parse_id=token_parse.parse_id,
        operation=token_parse.operation,
        status="materialised",
        working_blocks=source_material.working_blocks + (output,),
        transformations=source_material.transformations + (transform,),
        placements=placements,
    )


def _materialise_substitution(atom_context, clue_context, token_parse):
    blocks = {block.block_id: block for block in token_parse.blocks}
    base_material = _source_material(
        atom_context, clue_context, token_parse, blocks["src_base"])
    insert_material = _source_material(
        atom_context, clue_context, token_parse, blocks["src_insert"])
    remove_material = _source_material(
        atom_context, clue_context, token_parse, blocks["src_remove"])
    base = base_material.output
    insert = insert_material.output
    remove = remove_material.output
    output = produced_block(
        "wfw_%s_output" % _clean_id(token_parse.parse_id),
        token_parse.answer,
        kind="transformed_piece",
        text=token_parse.answer,
        mechanism="substitution",
        source_tokens=base.source_token_ids + insert.source_token_ids,
        source_atoms=base.source_atom_ids + insert.source_atom_ids,
        parent_blocks=(base.block_id, insert.block_id, remove.block_id),
    )
    transform = WFWTransformation(
        transform_id="wfw_%s_substitution" % _clean_id(token_parse.parse_id),
        operation="substitution",
        input_block_ids=(base.block_id, insert.block_id, remove.block_id),
        output_block_id=output.block_id,
        controller_atom_ids=_controller_atoms(
            atom_context, clue_context, blocks.get("op_0")),
        detail="%s with %s replacing %s = %s" % (
            base.value, insert.value, remove.value, output.value),
    )
    placements = place_charade(atom_context.answer_atoms, (output,))
    return WFWNativeAssembly(
        parse_id=token_parse.parse_id,
        operation=token_parse.operation,
        status="materialised",
        working_blocks=(
            base_material.working_blocks
            + insert_material.working_blocks
            + remove_material.working_blocks
            + (output,)
        ),
        transformations=(
            base_material.transformations
            + insert_material.transformations
            + remove_material.transformations
            + (transform,)
        ),
        placements=placements,
    )


def _materialise_compound_charade(atom_context, clue_context, token_parse):
    piece_material = _compound_charade_piece_material(
        atom_context, clue_context, token_parse)
    placement_blocks = tuple(item.output for item in piece_material)
    working = tuple(
        block for item in piece_material for block in item.working_blocks
    )
    source_transforms = tuple(
        transform
        for item in piece_material
        for transform in item.transformations
    )
    placements = place_charade(atom_context.answer_atoms, placement_blocks)
    relation_transforms = tuple(
        _relationship_transform(
            atom_context, clue_context, block, placement_blocks)
        for block in token_parse.blocks
        if block.kind in ("OP_BLOCK", "RELATION_BLOCK")
        and not _is_source_operator(token_parse, block)
        and not _container_piece_role(block)
    )
    relation_transforms = tuple(t for t in relation_transforms if t is not None)
    return WFWNativeAssembly(
        parse_id=token_parse.parse_id,
        operation=token_parse.operation,
        status="materialised",
        working_blocks=working,
        transformations=source_transforms + relation_transforms,
        placements=placements,
    )


def _materialise_double_definition(atom_context, clue_context, token_parse):
    definitions = [
        _source_material(atom_context, clue_context, token_parse, block).output
        for block in token_parse.blocks
        if block.kind == "SOURCE_BLOCK"
    ]
    transforms = tuple(
        WFWTransformation(
            transform_id="wfw_%s_defines_whole_answer_%d" % (
                _clean_id(token_parse.parse_id), idx),
            operation="defines_whole_answer",
            input_block_ids=(block.block_id,),
            output_block_id="answer",
            detail="%s defines %s" % (block.text, token_parse.answer),
        )
        for idx, block in enumerate(definitions)
    )
    return WFWNativeAssembly(
        parse_id=token_parse.parse_id,
        operation=token_parse.operation,
        status="materialised",
        working_blocks=tuple(definitions),
        transformations=transforms,
        placements=(),
    )


def _source_block(atom_context, clue_context, parse_block):
    return source_block_from_tokens(
        atom_context,
        "wfw_%s" % _clean_id(parse_block.block_id),
        _wfw_indices_for_span(atom_context, clue_context, parse_block.span),
        parse_block.value or parse_block.text,
        parse_block.token or parse_block.role or parse_block.kind,
    )


def _source_material(atom_context, clue_context, token_parse, parse_block):
    op_block = _source_operator(token_parse, parse_block)
    if op_block is None:
        output = _source_block(atom_context, clue_context, parse_block)
        return _SourceMaterial(output, (output,), ())

    base_id = "wfw_%s_before" % _clean_id(parse_block.block_id)
    base = source_block_from_tokens(
        atom_context,
        base_id,
        _wfw_indices_for_span(atom_context, clue_context, parse_block.span),
        parse_block.input_value or parse_block.text,
        "source_text",
    )
    output = produced_block(
        "wfw_%s" % _clean_id(parse_block.block_id),
        parse_block.value or "",
        kind="source_piece",
        text=parse_block.text,
        mechanism=parse_block.token or parse_block.role,
        source_tokens=base.source_token_ids,
        source_atoms=base.source_atom_ids,
        parent_blocks=(base.block_id,),
    )
    operation = _operation_from_block(op_block) or "selection"
    transform = WFWTransformation(
        transform_id="wfw_%s_%s" % (
            _clean_id(parse_block.block_id), operation),
        operation=operation,
        input_block_ids=(base.block_id,),
        output_block_id=output.block_id,
        controller_atom_ids=_controller_atoms(
            atom_context, clue_context, op_block),
        detail="%s selects %s from %s" % (
            op_block.text, output.value, base.value),
    )
    return _SourceMaterial(output, (base, output), (transform,))


def _compound_charade_piece_material(atom_context, clue_context, token_parse):
    parse_blocks = list(token_parse.blocks)
    container_roles = {
        _container_piece_role(block): block
        for block in parse_blocks
        if block.kind in ("OP_BLOCK", "RELATION_BLOCK")
        and block.token == "CON_I"
        and _container_piece_role(block)
    }
    consumed_ids = set()
    materials = []

    for block in parse_blocks:
        if block.kind != "SOURCE_BLOCK" or not _is_answer_piece(block):
            continue
        op_block = container_roles.get(block.role)
        if op_block is not None:
            material, consumed = _container_piece_material(
                atom_context, clue_context, token_parse, block, op_block)
            materials.append(material)
            consumed_ids.update(consumed)
        else:
            materials.append(
                _source_material(atom_context, clue_context, token_parse, block)
            )
            consumed_ids.add(block.block_id)

    for block in parse_blocks:
        if block.kind != "SOURCE_BLOCK" or block.block_id in consumed_ids:
            continue
        if _is_answer_piece(block):
            materials.append(
                _source_material(atom_context, clue_context, token_parse, block)
            )
            consumed_ids.add(block.block_id)
    return tuple(materials)


def _container_piece_material(atom_context, clue_context, token_parse,
                              outer_parse, op_block):
    if outer_parse.input_value:
        outer_material = _direct_container_outer(
            atom_context, clue_context, outer_parse)
        inner_parse_blocks = _container_inner_blocks_for_piece(
            token_parse, outer_parse)
    else:
        outer_material = _shell_container_outer(
            atom_context, clue_context, token_parse, outer_parse)
        inner_parse_blocks = _container_inner_blocks_for_piece(
            token_parse, outer_parse)
    outer = outer_material.output
    inner_parse_blocks = [
        block for block in inner_parse_blocks
        if block.block_id not in {
            item.output.block_id for item in (outer_material,)
        }
    ]
    inner_material = tuple(
        _source_material(atom_context, clue_context, token_parse, block)
        for block in inner_parse_blocks
    )
    inner_outputs = tuple(item.output for item in inner_material)
    output, transform, _ = container_block(
        outer,
        inner_outputs,
        "wfw_%s" % _clean_id(outer_parse.block_id),
        outer_parse.value,
        _controller_atoms(atom_context, clue_context, op_block),
    )
    working = (
        outer_material.working_blocks
        + tuple(
            work
            for item in inner_material
            for work in item.working_blocks
        )
        + (output,)
    )
    transforms = tuple(
        source_transform
        for item in inner_material
        for source_transform in item.transformations
    ) + (transform,)
    consumed = {outer_parse.block_id}
    consumed.update(block.block_id for block in inner_parse_blocks)
    consumed.update(outer_material.consumed_source_ids)
    return _SourceMaterial(output, working, transforms), consumed


@dataclass(frozen=True)
class _ContainerOuterMaterial:
    output: object
    working_blocks: tuple
    consumed_source_ids: frozenset


def _direct_container_outer(atom_context, clue_context, outer_parse):
    outer = source_block_from_tokens(
        atom_context,
        "wfw_%s_outer" % _clean_id(outer_parse.block_id),
        _wfw_indices_for_span(atom_context, clue_context, outer_parse.span),
        outer_parse.input_value,
        outer_parse.token or outer_parse.role or "container_outer",
    )
    return _ContainerOuterMaterial(
        outer, (outer,), frozenset({outer_parse.block_id}))


def _shell_container_outer(atom_context, clue_context, token_parse,
                           outer_parse):
    shell_parse_blocks = [
        block for block in token_parse.blocks
        if block.kind == "SOURCE_BLOCK"
        and (block.role or "").startswith("container_shell_")
    ]
    shell_material = tuple(
        _source_material(atom_context, clue_context, token_parse, block)
        for block in shell_parse_blocks
    )
    shell_outputs = tuple(item.output for item in shell_material)
    value = "".join(block.value for block in shell_outputs)
    output = produced_block(
        "wfw_%s_outer" % _clean_id(outer_parse.block_id),
        value,
        kind="assembled_piece",
        text=value,
        mechanism="container_shell",
        source_tokens=tuple(
            token_id for block in shell_outputs
            for token_id in block.source_token_ids
        ),
        source_atoms=tuple(
            atom_id for block in shell_outputs
            for atom_id in block.source_atom_ids
        ),
        parent_blocks=tuple(block.block_id for block in shell_outputs),
    )
    working = tuple(
        work
        for item in shell_material
        for work in item.working_blocks
    ) + (output,)
    consumed = {outer_parse.block_id}
    consumed.update(block.block_id for block in shell_parse_blocks)
    return _ContainerOuterMaterial(output, working, frozenset(consumed))


def _is_container_inner(block):
    role = block.role or ""
    return role == "container_inner" or role.startswith("container_inner_")


def _container_inner_blocks_for_piece(token_parse, outer_parse):
    prefix = "%s_inner" % outer_parse.block_id
    scoped = [
        block for block in token_parse.blocks
        if block.kind == "SOURCE_BLOCK"
        and _is_container_inner(block)
        and block.block_id.startswith(prefix)
    ]
    if scoped:
        return scoped
    return [
        block for block in token_parse.blocks
        if block.kind == "SOURCE_BLOCK"
        and _is_container_inner(block)
    ]


def _container_piece_role(op_block):
    role = op_block.role or ""
    match = re.match(r"^(piece_\d+)_container_indicator$", role)
    return match.group(1) if match else None


def _source_operator(token_parse, source_block):
    source_role = source_block.role or ""
    if source_block.token not in ("POS_F", "HOM_F"):
        return None
    for block in token_parse.blocks:
        if block.kind not in ("OP_BLOCK", "RELATION_BLOCK"):
            continue
        if _operator_targets_source(block, source_block, source_role):
            return block
    return None


def _is_source_operator(token_parse, op_block):
    return any(
        _source_operator(token_parse, block) is op_block
        for block in token_parse.blocks
        if block.kind == "SOURCE_BLOCK"
    )


def _operator_targets_source(op_block, source_block, source_role):
    op_role = op_block.role or ""
    op_token = op_block.token or ""
    if not (op_token.startswith("POS_I_") or op_token == "HOM_I"):
        return False
    if source_role and op_role.startswith("%s_" % source_role):
        return True
    source_piece = re.search(r"piece_(\d+)", source_role)
    op_piece = re.search(r"piece_(\d+)", op_role)
    if source_piece and op_piece:
        return source_piece.group(1) == op_piece.group(1)
    return False


def _is_answer_piece(parse_block):
    if parse_block.kind != "SOURCE_BLOCK":
        return False
    role = parse_block.role or ""
    if role.startswith("dd_"):
        return False
    if "inner" in role or "insert" in role or "remove" in role:
        return False
    if role:
        return re.match(r"^piece_\d+$", role) is not None
    return parse_block.block_id.startswith("src_")


def _relationship_transform(atom_context, clue_context, parse_block, working):
    operation = _operation_from_block(parse_block)
    if operation is None:
        return None
    output_block_id = _output_block_for_relation(parse_block, working)
    return WFWTransformation(
        transform_id="wfw_%s_%s" % (
            _clean_id(parse_block.block_id), operation),
        operation=operation,
        input_block_ids=tuple(block.block_id for block in working),
        output_block_id=output_block_id,
        controller_atom_ids=_controller_atoms(
            atom_context, clue_context, parse_block),
        detail="%s marks %s" % (parse_block.text, operation),
    )


def _operation_from_block(parse_block):
    role = parse_block.role or ""
    token = parse_block.token or ""
    if "reversal" in role or token == "REV_I":
        return "reversal"
    if "anagram" in role or token == "ANA_I":
        return "anagram"
    if "deletion" in role or token.startswith("POS_I_TRIM_"):
        return "deletion"
    if "positional" in role or token.startswith("POS_I_"):
        return "selection"
    if "container" in role or token == "CON_I":
        return "container"
    if token == "HOM_I":
        return "homophone"
    return None


def _output_block_for_relation(parse_block, working):
    role = parse_block.role or ""
    for block in working:
        if block.block_id.endswith(role.split("_")[0]):
            return block.block_id
    return working[0].block_id if working else "answer"


def _controller_atoms(atom_context, clue_context, parse_block):
    if parse_block is None or parse_block.span is None:
        return ()
    atom_ids = []
    for idx in _wfw_indices_for_span(atom_context, clue_context,
                                     parse_block.span):
        atom_ids.extend(atom_context.clue_tokens[idx].atom_ids)
    return tuple(atom_ids)


def _wfw_indices_for_span(atom_context, clue_context, span):
    if span is None:
        return ()
    mapping = _clue_context_to_wfw_token_map(atom_context, clue_context)
    return tuple(
        mapping[idx] for idx in range(span[0], span[1])
        if idx in mapping
    )


def _clue_context_to_wfw_token_map(atom_context, clue_context):
    mapping = {}
    wfw_index = 0
    for clue_token in clue_context.tokens:
        target = _token_key(clue_token.text)
        if not target:
            continue
        while (wfw_index < len(atom_context.clue_tokens)
               and not _token_key(atom_context.clue_tokens[wfw_index].text)):
            wfw_index += 1
        if wfw_index >= len(atom_context.clue_tokens):
            break
        if _token_key(atom_context.clue_tokens[wfw_index].text) == target:
            mapping[clue_token.index] = wfw_index
            wfw_index += 1
    return mapping


def _first_block(token_parse, kind):
    return next(block for block in token_parse.blocks if block.kind == kind)


def _first_operation_block(token_parse):
    return next(
        (block for block in token_parse.blocks
         if block.kind in ("OP_BLOCK", "RELATION_BLOCK")),
        None,
    )


def _unsupported(token_parse, reason):
    return WFWNativeAssembly(
        parse_id=token_parse.parse_id,
        operation=token_parse.operation,
        status="review",
        working_blocks=(),
        transformations=(),
        placements=(),
        objections=(reason,),
    )


def _clean_id(value):
    return "".join(
        char.lower() if char.isalnum() else "_"
        for char in (value or "unknown")
    ).strip("_")


def _token_key(text):
    return "".join(char.upper() for char in (text or "") if char.isalnum())
