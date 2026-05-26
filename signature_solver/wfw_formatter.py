"""Prepare tokenised parses for WFW and answer-colour rendering."""
from __future__ import annotations


def format_token_parse_for_wfw(context, token_parse):
    """Return WFW-ready clue token roles and answer-letter links."""
    token_roles = _token_roles(context, token_parse)
    answer_links = _answer_links(token_parse)
    out = {
        "clue_text": context.clue_text,
        "answer": token_parse.answer,
        "tokens": [
            {
                "index": token.index,
                "text": token.text,
                "normalized": token.normalized,
                "start_char": token.start_char,
                "end_char": token.end_char,
                "roles": token_roles.get(token.index, []),
            }
            for token in context.tokens
        ],
        "blocks": [block.as_dict() for block in token_parse.blocks],
        "operations": [op.as_dict() for op in token_parse.operations],
        "answer_links": answer_links,
    }
    hidden_segments = _hidden_segments(token_parse)
    if hidden_segments:
        out["hidden_segments"] = hidden_segments
    return out


def _token_roles(context, token_parse):
    roles = {}
    for block in token_parse.blocks:
        if block.span is None:
            continue
        for idx in range(block.span[0], block.span[1]):
            if idx < 0 or idx >= len(context.tokens):
                continue
            roles.setdefault(idx, []).append({
                "block_id": block.block_id,
                "kind": block.kind,
                "role": block.role,
                "value": block.value,
            })
    return roles


def _answer_links(token_parse):
    if token_parse.operation == "container":
        return _container_answer_links(token_parse)
    if token_parse.operation in (
            "charade", "positional_charade", "reversal_charade",
            "anagram_charade", "deletion_charade", "container_charade"):
        return _charade_answer_links(token_parse)
    if token_parse.operation == "reversal":
        return _single_source_answer_links(token_parse, reversed_source=True)
    if token_parse.operation == "hidden":
        return _hidden_answer_links(token_parse, reversed_hidden=False)
    if token_parse.operation == "hidden_reversed":
        return _hidden_answer_links(token_parse, reversed_hidden=True)
    if token_parse.operation == "anagram":
        return _single_source_answer_links(token_parse, reversed_source=False)
    if token_parse.operation == "deletion":
        return _single_source_answer_links(token_parse, reversed_source=False)
    if token_parse.operation == "homophone":
        return _single_source_answer_links(token_parse, reversed_source=False)
    if token_parse.operation == "substitution":
        return _substitution_answer_links(token_parse)
    if token_parse.operation == "double_definition":
        return _double_definition_answer_links(token_parse)
    return [
        {
            "answer_index": idx,
            "letter": letter,
            "source_block": None,
            "source_role": None,
        }
        for idx, letter in enumerate(token_parse.answer)
    ]


def _charade_answer_links(token_parse):
    source_blocks = [
        block for block in token_parse.blocks
        if block.kind == "SOURCE_BLOCK" and block.role and block.role.startswith("piece_")
    ]
    source_blocks.sort(key=lambda block: int(block.role.split("_", 1)[1]))
    links = []
    answer_index = 0
    for block in source_blocks:
        value = _clean_value(block.value)
        for source_index, letter in enumerate(value):
            links.append({
                "answer_index": answer_index,
                "letter": letter,
                "source_block": block.block_id,
                "source_span": list(block.span) if block.span else None,
                "source_text": block.text,
                "source_role": block.role,
                "source_value": block.value,
                "source_value_index": source_index,
            })
            answer_index += 1
    return links if answer_index == len(token_parse.answer) else []


def _double_definition_answer_links(token_parse):
    return [
        {
            "answer_index": idx,
            "letter": letter,
            "source_block": None,
            "source_role": "double_definition",
        }
        for idx, letter in enumerate(token_parse.answer)
    ]


def _substitution_answer_links(token_parse):
    blocks = {block.block_id: block for block in token_parse.blocks}
    base = blocks.get("src_base")
    insert = blocks.get("src_insert")
    remove = blocks.get("src_remove")
    if base is None or insert is None or remove is None:
        return []
    answer = _clean_value(token_parse.answer)
    base_value = _clean_value(base.value)
    insert_value = _clean_value(insert.value)
    remove_value = _clean_value(remove.value)
    pos = base_value.find(remove_value)
    if pos < 0:
        return []
    links = []
    for idx, letter in enumerate(answer):
        if pos <= idx < pos + len(insert_value):
            source = insert
            source_index = idx - pos
        else:
            source = base
            source_index = idx
        links.append({
            "answer_index": idx,
            "letter": letter,
            "source_block": source.block_id,
            "source_span": list(source.span) if source.span else None,
            "source_text": source.text,
            "source_role": source.role,
            "source_value": source.value,
            "source_value_index": source_index,
        })
    return links


def _single_source_answer_links(token_parse, reversed_source=False):
    source = next(
        (block for block in token_parse.blocks if block.kind == "SOURCE_BLOCK"),
        None)
    if source is None:
        return []
    answer = _clean_value(token_parse.answer)
    value = _clean_value(source.value)
    if reversed_source:
        value = value[::-1]
    links = []
    for idx, letter in enumerate(answer):
        source_index = idx
        if reversed_source:
            source_index = len(answer) - idx - 1
        links.append({
            "answer_index": idx,
            "letter": letter,
            "source_block": source.block_id,
            "source_span": list(source.span) if source.span else None,
            "source_text": source.text,
            "source_role": source.role,
            "source_value": source.value,
            "source_value_index": source_index,
        })
    return links


def _hidden_answer_links(token_parse, reversed_hidden=False):
    source = next(
        (block for block in token_parse.blocks if block.kind == "SOURCE_BLOCK"),
        None)
    if source is None:
        return []
    answer = _clean_value(token_parse.answer)
    value = _clean_value(source.value)
    needle = answer[::-1] if reversed_hidden else answer
    start = value.find(needle)
    if start < 0:
        return _single_source_answer_links(token_parse, reversed_source=False)

    links = []
    for idx, letter in enumerate(answer):
        if reversed_hidden:
            source_index = start + len(answer) - idx - 1
        else:
            source_index = start + idx
        links.append({
            "answer_index": idx,
            "letter": letter,
            "source_block": source.block_id,
            "source_span": list(source.span) if source.span else None,
            "source_text": source.text,
            "source_role": source.role,
            "source_value": source.value,
            "source_value_index": source_index,
        })
    return links


def _hidden_segments(token_parse):
    if token_parse.operation not in ("hidden", "hidden_reversed"):
        return None
    source = next(
        (block for block in token_parse.blocks if block.kind == "SOURCE_BLOCK"),
        None)
    if source is None:
        return None
    answer = _clean_value(token_parse.answer)
    value = _clean_value(source.value)
    needle = answer[::-1] if token_parse.operation == "hidden_reversed" else answer
    start = value.find(needle)
    if start < 0:
        return None
    end = start + len(needle)
    display_prefix, display_hidden, display_suffix = _display_hidden_segments(
        source.text, start, end)
    return {
        "source_block": source.block_id,
        "source_text": source.text,
        "source_value": source.value,
        "prefix": value[:start],
        "hidden": value[start:end],
        "suffix": value[end:],
        "display_prefix": display_prefix,
        "display_hidden": display_hidden,
        "display_suffix": display_suffix,
        "answer_order": answer,
        "reversed": token_parse.operation == "hidden_reversed",
    }


def _display_hidden_segments(text, clean_start, clean_end):
    """Split original source text using cleaned-letter indices."""
    letters_seen = 0
    char_start = None
    char_end = None
    for idx, char in enumerate(text or ""):
        if not char.isalpha():
            continue
        if letters_seen == clean_start and char_start is None:
            char_start = idx
        letters_seen += 1
        if letters_seen == clean_end:
            char_end = idx + 1
            break
    if char_start is None or char_end is None:
        return "", text or "", ""
    return (text[:char_start], text[char_start:char_end], text[char_end:])


def _container_answer_links(token_parse):
    blocks = {block.block_id: block for block in token_parse.blocks}
    outer = blocks.get("src_outer")
    inner = blocks.get("src_inner")
    if outer is None or inner is None:
        return []

    outer_value = _clean_value(outer.value)
    inner_value = _clean_value(inner.value)
    answer = _clean_value(token_parse.answer)

    insert_pos = None
    for pos in range(1, len(outer_value)):
        if outer_value[:pos] + inner_value + outer_value[pos:] == answer:
            insert_pos = pos
            break
    if insert_pos is None:
        return []

    links = []
    for idx, letter in enumerate(answer):
        if insert_pos <= idx < insert_pos + len(inner_value):
            source = inner
            source_index = idx - insert_pos
        else:
            source = outer
            source_index = idx
            if idx >= insert_pos + len(inner_value):
                source_index = idx - len(inner_value)
        links.append({
            "answer_index": idx,
            "letter": letter,
            "source_block": source.block_id,
            "source_span": list(source.span) if source.span else None,
            "source_text": source.text,
            "source_role": source.role,
            "source_value": source.value,
            "source_value_index": source_index,
        })
    return links


def _clean_value(value):
    return "".join(c for c in (value or "").upper() if c.isalpha())
