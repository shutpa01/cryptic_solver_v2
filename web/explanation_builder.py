"""Positional explanation builder for cryptic clue answers.

Builds an HTML explanation showing exactly how each piece of the answer
is derived from the clue, with positional mapping.

Example output for SPEECHES (clue: "Talks of speed chess finishing early"):
  Definition: "Talks"
  [1-4] SPEE ← "speed" with last letter removed
  [5-8] CHES ← "chess" with last letter removed

This module is imported by web/models.py to generate user-facing explanations.
"""

import json
import re
from markupsafe import Markup


def norm(s):
    return re.sub(r"[^A-Za-z]", "", s or "").upper()


def _describe_mechanism(piece):
    """Describe how a piece gets its letters, in plain English."""
    mech = piece.get("mechanism", "")
    word = piece.get("clue_word", "")
    letters = piece.get("letters", "")
    source = piece.get("source", "")
    deleted = piece.get("deleted", "")
    indicator = piece.get("indicator", "")

    if not letters:
        return ""

    if mech == "synonym":
        if norm(letters) == norm(word):
            return f'"{word}"'  # self-synonym, just show the word
        return f'"{word}" = {letters}'

    if mech == "abbreviation":
        return f'"{word}" → {letters}'

    if mech == "literal":
        return f'"{word}"'

    if mech == "first_letter":
        return f'first letter of "{word}"'

    if mech == "last_letter":
        return f'last letter of "{word}"'

    if mech == "core_letters":
        return f'middle of "{word}"'

    if mech == "outer_letters":
        return f'edges of "{word}"'

    if mech == "alternate_letters" or mech == "alternating":
        return f'alternate letters of "{word}"'

    if mech == "deletion":
        if source and deleted:
            return f'"{source}" minus {deleted}'
        return f'"{word}" shortened'

    if mech == "reversal":
        return f'"{word}" reversed'

    if mech == "anagram_fodder":
        return f'"{word}"'

    if mech == "sound_of" or mech == "homophone":
        return f'sounds like "{word}"'

    if mech == "hidden":
        return f'hidden in "{word}"'

    if mech == "container":
        return f'"{word}"'

    # Fallback
    if word and word != letters:
        return f'"{word}" → {letters}'
    return f'"{word}"'


def build_positional_explanation(answer, definition, wordplay_type, components):
    """Build an HTML positional explanation from structured components.

    Args:
        answer: the answer word
        definition: the definition text
        wordplay_type: e.g. "charade", "anagram", "hidden"
        components: dict with ai_pieces, assembly, wordplay_type

    Returns:
        HTML string (Markup) or None if can't build
    """
    if not components:
        return None

    pieces = components.get("ai_pieces", [])
    assembly_op = components.get("assembly", {}).get("op", wordplay_type or "")

    if not pieces and assembly_op != "double_definition":
        return None

    answer_upper = norm(answer)
    lines = []

    # --- Double definition: special case ---
    if assembly_op == "double_definition":
        asm = components.get("assembly", {})
        left = asm.get("left_def", "")
        right = asm.get("right_def", "")
        if left and right:
            lines.append(f'<div class="font-medium">Double definition</div>')
            lines.append(f'<div class="ml-2">1. "{left}"</div>')
            lines.append(f'<div class="ml-2">2. "{right}"</div>')
            lines.append(f'<div class="ml-2">Both mean <b>{answer}</b></div>')
        else:
            lines.append(f'<div>Double definition = <b>{answer}</b></div>')
        return Markup("\n".join(lines))

    # --- Hidden word: special case ---
    if assembly_op in ("hidden", "hidden_reversed"):
        if pieces:
            spanning = pieces[0].get("clue_word", "")
            direction = "reversed " if "reversed" in assembly_op else ""
            from sonnet_pipeline.report import _highlight_hidden
            target = answer[::-1] if "reversed" in assembly_op else answer
            highlighted = _highlight_hidden(spanning, target)
            lines.append(f'<div><b>{answer}</b> is {direction}hidden in "{highlighted}"</div>')
        return Markup("\n".join(lines)) if lines else None

    # --- Anagram: special case ---
    if assembly_op == "anagram":
        fodder_words = [p.get("clue_word", p.get("letters", "")) for p in pieces
                        if p.get("mechanism") == "anagram_fodder"]
        non_fodder = [p for p in pieces if p.get("mechanism") != "anagram_fodder"]

        if fodder_words:
            fodder_str = " + ".join(f'"{w}"' for w in fodder_words)
            lines.append(f'<div>Anagram of {fodder_str} = <b>{answer}</b></div>')

        # If there are non-fodder pieces (composite anagram + charade)
        for p in non_fodder:
            desc = _describe_mechanism(p)
            lines.append(f'<div class="ml-2">{p.get("letters", "")} ← {desc}</div>')

        return Markup("\n".join(lines)) if lines else None

    # --- Container: special display showing insertion ---
    if assembly_op == "container" and len(pieces) >= 2:
        lines.append(f'<div class="font-medium">Container → <b>{answer}</b></div>')
        # Identify inner and outer pieces
        # Convention: first piece with "container" role or shorter piece is inner
        inner = None
        outer_pieces = []
        for p in pieces:
            if p.get("mechanism") in ("container", "hidden"):
                inner = p
            else:
                outer_pieces.append(p)
        if inner is None and len(pieces) == 2:
            # Shorter piece is typically inner
            if len(norm(pieces[0].get("letters", ""))) <= len(norm(pieces[1].get("letters", ""))):
                inner = pieces[0]
                outer_pieces = [pieces[1]]
            else:
                inner = pieces[1]
                outer_pieces = [pieces[0]]

        if inner:
            inner_desc = _describe_mechanism(inner)
            outer_desc = " + ".join(_describe_mechanism(p) for p in outer_pieces)
            outer_letters = " + ".join(p.get("letters", "") for p in outer_pieces)
            lines.append(
                f'<div class="ml-2">'
                f'<span class="font-mono font-bold">{inner.get("letters", "")}</span> '
                f'<span class="text-gray-500">({inner_desc})</span>'
                f' inside '
                f'<span class="font-mono font-bold">{outer_letters}</span> '
                f'<span class="text-gray-500">({outer_desc})</span>'
                f'</div>'
            )
        else:
            for p in pieces:
                desc = _describe_mechanism(p)
                lines.append(
                    f'<div class="ml-2">'
                    f'<span class="font-mono font-bold">{p.get("letters", "")}</span> '
                    f'<span class="text-gray-500">← {desc}</span>'
                    f'</div>'
                )
        return Markup("\n".join(lines))

    # --- Charade / reversal / other: positional breakdown ---
    pos = 0
    piece_lines = []
    for p in pieces:
        letters = norm(p.get("letters", ""))
        if not letters:
            continue
        start = pos + 1
        end = pos + len(letters)
        desc = _describe_mechanism(p)

        if start == end:
            pos_str = f'{start}'
        else:
            pos_str = f'{start}-{end}'

        piece_lines.append(
            f'<div class="ml-2">'
            f'<span class="font-mono text-xs text-gray-400 w-8 inline-block">[{pos_str}]</span> '
            f'<span class="font-mono font-bold">{p.get("letters", "")}</span> '
            f'<span class="text-gray-500">← {desc}</span>'
            f'</div>'
        )
        pos += len(letters)

    if not piece_lines:
        return None

    if assembly_op == "reversal":
        label = "Reversal"
    else:
        label = "Charade"

    lines.append(f'<div class="font-medium">{label} → <b>{answer}</b></div>')
    lines.extend(piece_lines)

    return Markup("\n".join(lines))


def build_explanation_text(answer, definition, wordplay_type, components):
    """Build a plain-text explanation (for ai_explanation field / non-HTML contexts).

    Falls back to simple format when positional isn't appropriate.
    """
    if not components:
        return None

    pieces = components.get("ai_pieces", [])
    assembly_op = components.get("assembly", {}).get("op", wordplay_type or "")

    # Double definition
    if assembly_op == "double_definition":
        return "Double definition"

    # Hidden
    if assembly_op in ("hidden", "hidden_reversed"):
        if pieces:
            spanning = pieces[0].get("clue_word", "")
            direction = "reversed " if "reversed" in assembly_op else ""
            from sonnet_pipeline.report import _highlight_hidden
            target = answer[::-1] if "reversed" in assembly_op else answer
            highlighted = _highlight_hidden(spanning, target)
            return f'{answer} is {direction}hidden in "{highlighted}"'
        return None

    # Anagram
    if assembly_op == "anagram":
        fodder = [p.get("clue_word", p.get("letters", "")) for p in pieces
                  if p.get("mechanism") == "anagram_fodder"]
        if fodder:
            return f'Anagram of {" + ".join(fodder)} = {answer}'
        return None

    # Charade / other
    if pieces:
        part_strs = []
        for p in pieces:
            letters = p.get("letters", "")
            desc = _describe_mechanism(p)
            part_strs.append(f'{letters} ({desc})')
        result = " + ".join(part_strs) + f' = {answer}'
        if definition:
            result += f'; definition: "{definition}"'
        return result

    return None
