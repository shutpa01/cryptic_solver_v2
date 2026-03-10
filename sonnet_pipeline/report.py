"""Sonnet pipeline report generator.

Extracted from pipeline_tiered.py. Takes a list of result dicts and stats,
returns a formatted report string.
"""

from datetime import datetime

import re

from .solver import OP_TO_TYPE, clean


def _piece_label(letters, ai_pieces):
    """Find the clue word attribution for a piece from the AI output."""
    letters_clean = re.sub(r"[^A-Z]", "", letters.upper())
    for p in ai_pieces:
        p_letters = re.sub(r"[^A-Z]", "", (p.get("letters") or "").upper())
        if p_letters == letters_clean:
            return p.get("clue_word", "")
    return ""


def _annotate(letters, ai_pieces):
    """Return 'LETTERS(clue_word)' if attribution found, else just 'LETTERS'."""
    label = _piece_label(letters, ai_pieces)
    if label:
        return "%s(%s)" % (letters, label)
    return letters


def _annotate_composite(letters, ai_pieces):
    """Decompose a composite string into annotated AI pieces.

    E.g. 'FFE' with pieces [F(female), FE(iron)] -> 'F(female)+FE(iron)'.
    Falls back to plain _annotate if decomposition fails.
    """
    letters_clean = re.sub(r"[^A-Z]", "", letters.upper())

    def _decompose(pos):
        if pos == len(letters_clean):
            return []
        for end in range(len(letters_clean), pos, -1):
            substr = letters_clean[pos:end]
            label = _piece_label(substr, ai_pieces)
            if label:
                rest = _decompose(end)
                if rest is not None:
                    return ["%s(%s)" % (substr, label)] + rest
        return None

    parts = _decompose(0)
    if parts and len(parts) > 1:
        return "+".join(parts)
    return _annotate(letters, ai_pieces)


def _with_charade(desc, order, key_piece, ai):
    """Wrap an operation description with any extra charade pieces from the order.

    Many operations (container, reversal, deletion) produce a single piece that
    then combines with other pieces in a charade. The 'order' field has the full
    permutation. This replaces key_piece with desc and annotates the rest.
    """
    if not order or not key_piece or len(order) <= 1:
        return desc
    parts = []
    placed = False
    for p in order:
        if p == key_piece and not placed:
            parts.append(desc)
            placed = True
        else:
            parts.append(_annotate(p, ai))
    return " + ".join(parts)


def _find_indicator(ai_pieces):
    """Find indicator word from ai_pieces (stored in deletion/reversal refinement)."""
    for p in (ai_pieces or []):
        if p.get("indicator"):
            return p["indicator"]
    return None


def _highlight_hidden(words, answer):
    """Capitalise the hidden answer letters within the source words.

    E.g. words='grim peloton', answer='IMPEL' → 'gr IM PEL oton'
    """
    if not answer:
        return words
    answer_upper = re.sub(r"[^A-Z]", "", answer.upper())
    # Build a letters-only version and find the hidden substring
    letters_only = []
    letter_positions = []
    for i, ch in enumerate(words):
        if ch.isalpha():
            letters_only.append(ch.upper())
            letter_positions.append(i)

    letters_str = "".join(letters_only)
    idx = letters_str.find(answer_upper)
    if idx < 0:
        return words

    # Build result: lowercase everything, uppercase the hidden letters
    result = list(words.lower())
    for j in range(idx, idx + len(answer_upper)):
        pos = letter_positions[j]
        result[pos] = words[pos].upper()
    return "".join(result)


def _fmt_piece(piece):
    """Format a single AI piece for display: LETTERS (clue_word).

    Adapts format based on mechanism:
    - synonym/abbreviation: LETTERS (clue_word)
    - first_letter: first letter of Clue_word
    - last_letter: last letter of clue_worD
    - anagram_fodder: CLUE WORD (raw letters)
    - hidden: hidden in "clue words"
    - literal: LETTERS (clue_word)
    """
    letters = (piece.get("letters") or "").upper()
    clue_word = piece.get("clue_word") or ""
    mechanism = (piece.get("mechanism") or "").lower()

    if mechanism == "first_letter":
        # Capitalise the first letter in the clue word to show where it comes from
        if clue_word:
            return "%s (first letter of %s)" % (letters, clue_word)
        return "%s (first letter)" % letters

    if mechanism == "last_letter":
        # Capitalise the last letter to show where it comes from
        if clue_word:
            # Highlight the last letter: "daf" -> "dafT"
            word_display = clue_word
            for i in range(len(clue_word) - 1, -1, -1):
                if clue_word[i].isalpha():
                    word_display = clue_word[:i] + clue_word[i].upper() + clue_word[i+1:]
                    break
            return "%s (last letter of %s)" % (letters, word_display)
        return "%s (last letter)" % letters

    if mechanism == "anagram_fodder":
        return clue_word.upper() if clue_word else letters

    if mechanism == "alternate_letters":
        if clue_word:
            return "%s (alternate letters of %s)" % (letters, clue_word)
        return "%s (alternate letters)" % letters

    if mechanism == "core_letters":
        if clue_word:
            return "%s (middle of %s)" % (letters, clue_word)
        return "%s (middle)" % letters

    if mechanism == "sound_of":
        if clue_word:
            return "%s (sounds like %s)" % (letters, clue_word)
        return letters

    if mechanism == "reversal":
        if clue_word:
            return "%s (reverse of %s)" % (letters, clue_word)
        return "%s (reversed)" % letters

    if mechanism == "deletion":
        # Piece-level deletion info (from refine_pieces)
        source = piece.get("source") or ""
        deleted = piece.get("deleted") or ""
        deleted_word = piece.get("deleted_word") or ""
        if source and deleted:
            if deleted_word:
                return "%s (%s minus %s/%s)" % (letters, source, deleted, deleted_word)
            return "%s (%s minus %s)" % (letters, source, deleted)
        if clue_word:
            return "%s (%s)" % (letters, clue_word)
        return letters

    # Default: synonym, abbreviation, literal
    if clue_word:
        return "%s (%s)" % (letters, clue_word)
    return letters


def _fmt_indicator(indicator_word):
    """Format an indicator reference: (indicator_word)."""
    if indicator_word:
        return "(%s)" % indicator_word
    return ""


def _describe_assembly(asm, ai_pieces=None, answer=None, indicator=None):
    """Return a user-friendly explanation of how the wordplay produces the answer.

    Format: Type (indicator) of/containing PIECE (source) + PIECE (source)
    The definition is NOT included — it's shown separately in the UI.
    """
    op = asm.get("op", "?")
    ai = ai_pieces or []

    # Fall back to indicator stored in pieces (from refine_pieces)
    if not indicator:
        indicator = _find_indicator(ai)

    ind = _fmt_indicator(indicator)

    if op == "charade":
        order = asm.get("order", [])
        if order:
            # Build pieces list, merging adjacent order elements that match a single AI piece
            parts = []
            i = 0
            while i < len(order):
                found = False
                for span in (3, 2):
                    if i + span <= len(order):
                        combined = "".join(order[i:i + span])
                        label = _piece_label(combined, ai)
                        if label:
                            # Find the full piece for formatting
                            for p in ai:
                                if clean(p.get("letters", "")) == clean(combined):
                                    parts.append(_fmt_piece(p))
                                    found = True
                                    break
                            if found:
                                i += span
                                break
                if not found:
                    # Find matching AI piece for this order element
                    matched = False
                    for p in ai:
                        if clean(p.get("letters", "")) == clean(order[i]):
                            parts.append(_fmt_piece(p))
                            matched = True
                            break
                    if not matched:
                        parts.append(order[i])
                    i += 1
            return " + ".join(parts)

    elif op == "container":
        inner = asm.get("inner", "?")
        outer = asm.get("outer", "?")
        merged = asm.get("merged_inner")
        inner_letters = merged or inner

        # Find AI pieces for inner and outer
        inner_parts = []
        if merged:
            # Multiple pieces merged as inner
            remaining = clean(merged)
            for p in ai:
                pl = clean(p.get("letters", ""))
                if pl and pl in remaining:
                    inner_parts.append(_fmt_piece(p))
                    remaining = remaining.replace(pl, "", 1)
        else:
            for p in ai:
                if clean(p.get("letters", "")) == clean(inner):
                    inner_parts.append(_fmt_piece(p))
                    break

        outer_fmt = outer
        for p in ai:
            if clean(p.get("letters", "")) == clean(outer):
                outer_fmt = _fmt_piece(p)
                break

        inner_fmt = " + ".join(inner_parts) if inner_parts else inner
        desc = "%s containing %s %s" % (outer_fmt, ind, inner_fmt)

        # Add extra charade pieces
        combined = asm.get("combined", "")
        order = asm.get("order", [])
        if order and combined and len(order) > 1:
            extra = []
            for o in order:
                if clean(o) != clean(combined):
                    for p in ai:
                        if clean(p.get("letters", "")) == clean(o):
                            extra.append(_fmt_piece(p))
                            break
                    else:
                        extra.append(o)
            if extra:
                desc = desc.strip() + " + " + " + ".join(extra)

        return desc.strip()

    elif op == "reversal":
        rev_parts = asm.get("reversed_parts")
        if rev_parts:
            parts = []
            for rp in rev_parts:
                for p in ai:
                    if clean(p.get("letters", "")) == clean(rp):
                        parts.append(_fmt_piece(p))
                        break
                else:
                    parts.append(rp)
            desc = "Reversal %s of %s" % (ind, " + ".join(parts))
        else:
            reversed_str = asm.get("reversed", "?")
            for p in ai:
                if clean(p.get("letters", "")) == clean(reversed_str):
                    reversed_str = _fmt_piece(p)
                    break
            desc = "Reversal %s of %s" % (ind, reversed_str)

            # Add charade pieces
            order = asm.get("order", [])
            gives = asm.get("gives", "")
            if order and gives and len(order) > 1:
                extra = []
                for o in order:
                    if clean(o) != clean(gives):
                        for p in ai:
                            if clean(p.get("letters", "")) == clean(o):
                                extra.append(_fmt_piece(p))
                                break
                        else:
                            extra.append(o)
                if extra:
                    desc = desc + " + " + " + ".join(extra)

        return desc.strip()

    elif op == "deletion":
        from_word = asm.get("from", "?")
        deleted = asm.get("deleted", "?")

        from_fmt = from_word
        for p in ai:
            if clean(p.get("letters", "")) == clean(from_word):
                from_fmt = _fmt_piece(p)
                break

        desc = "Deletion %s: %s minus %s" % (ind, from_fmt, deleted)

        # Add charade pieces
        order = asm.get("order", [])
        gives = asm.get("gives", "")
        if order and gives and len(order) > 1:
            extra = []
            for o in order:
                if clean(o) != clean(gives):
                    for p in ai:
                        if clean(p.get("letters", "")) == clean(o):
                            extra.append(_fmt_piece(p))
                            break
                    else:
                        extra.append(o)
            if extra:
                desc = desc + " + " + " + ".join(extra)

        return desc.strip()

    elif op == "outer_deletion":
        from_word = asm.get("from", "?")
        from_fmt = from_word
        for p in ai:
            if clean(p.get("letters", "")) == clean(from_word):
                from_fmt = _fmt_piece(p)
                break
        desc = "Outer deletion %s: strip outer letters from %s" % (ind, from_fmt)
        return desc.strip()

    elif op == "anagram":
        fodder = asm.get("fodder", [])
        if fodder:
            fodder_parts = []
            for f in fodder:
                for p in ai:
                    if clean(p.get("letters", "")) == clean(f):
                        fodder_parts.append(_fmt_piece(p))
                        break
                else:
                    fodder_parts.append(f)
            return "Anagram %s of %s" % (ind, " + ".join(fodder_parts))

    elif op in ("charade+anagram", "anagram+charade"):
        charade = asm.get("charade", [])
        anagram = asm.get("anagram", [])
        parts = []
        for c in charade:
            for p in ai:
                if clean(p.get("letters", "")) == clean(c):
                    parts.append(_fmt_piece(p))
                    break
            else:
                parts.append(c)
        if anagram:
            anagram_parts = []
            for f in anagram:
                for p in ai:
                    if clean(p.get("letters", "")) == clean(f):
                        anagram_parts.append(_fmt_piece(p))
                        break
                else:
                    anagram_parts.append(f)
            parts.append("anagram %s of %s" % (ind, "+".join(anagram_parts)))
        return " + ".join(parts) if parts else None

    elif op == "deletion+anagram":
        from_word = asm.get("from", "?")
        deleted = asm.get("deleted", "?")
        fodder = asm.get("fodder", [])

        from_fmt = from_word
        for p in ai:
            if clean(p.get("letters", "")) == clean(from_word):
                from_fmt = _fmt_piece(p)
                break

        fodder_parts = []
        for f in fodder:
            for p in ai:
                if clean(p.get("letters", "")) == clean(f):
                    fodder_parts.append(_fmt_piece(p))
                    break
            else:
                fodder_parts.append(f)

        return "Deletion + anagram %s: %s minus %s, then anagram of %s" % (
            ind, from_fmt, deleted, "+".join(fodder_parts) if fodder_parts else "?")

    elif op == "reversal_container":
        inner = asm.get("inner", "?")
        outer = asm.get("outer", "?")

        inner_fmt = inner
        outer_fmt = outer
        for p in ai:
            if clean(p.get("letters", "")) == clean(inner):
                inner_fmt = _fmt_piece(p)
            if clean(p.get("letters", "")) == clean(outer):
                outer_fmt = _fmt_piece(p)

        desc = "Reversal + container %s: reverse then %s inside %s" % (ind, inner_fmt, outer_fmt)
        return desc.strip()

    elif op == "container_reversal":
        inner = asm.get("inner", "?")
        outer = asm.get("outer", "?")

        inner_fmt = inner
        outer_fmt = outer
        for p in ai:
            if clean(p.get("letters", "")) == clean(inner):
                inner_fmt = _fmt_piece(p)
            if clean(p.get("letters", "")) == clean(outer):
                outer_fmt = _fmt_piece(p)

        desc = "Container + reversal %s: %s inside %s, then reverse" % (ind, inner_fmt, outer_fmt)
        return desc.strip()

    elif op == "hidden":
        words = asm.get("words", "?")
        highlighted = _highlight_hidden(words, answer)
        return "Hidden %s in '%s'" % (ind, highlighted)

    elif op == "hidden_reversed":
        words = asm.get("words", "?")
        highlighted = _highlight_hidden(words, answer and answer[::-1])
        return "Hidden reversed %s in '%s'" % (ind, highlighted)

    elif op == "hidden_in_word":
        word = asm.get("word", "?")
        highlighted = _highlight_hidden(word, answer)
        return "Hidden %s in '%s'" % (ind, highlighted)

    elif op == "homophone":
        sounds_like = asm.get("sounds_like", "?")
        gives = asm.get("gives", "?")
        return "Homophone %s: %s sounds like %s" % (ind, sounds_like, gives)

    elif op == "spoonerism":
        src = asm.get("source_words", ["?", "?"])
        res = asm.get("result_words", ["?", "?"])
        return "Spoonerism %s: swap initial sounds of %s and %s" % (ind, src[0], src[1])

    elif op == "substitution":
        from_word = asm.get("from", "?")
        deleted = asm.get("deleted", "?")
        del_word = asm.get("del_word", "?")
        added = asm.get("added", "?")
        add_word = asm.get("add_word", "?")

        from_fmt = from_word
        for p in ai:
            if clean(p.get("letters", "")) == clean(from_word):
                from_fmt = _fmt_piece(p)
                break

        desc = "Substitution %s: %s -- replace %s (%s) with %s (%s)" % (
            ind, from_fmt, deleted, del_word, added, add_word)
        return desc.strip()

    elif op == "double_definition":
        return "Double definition"

    elif op == "cryptic_definition":
        return "Cryptic definition"

    return None


def _actionable_quality(results):
    """Generate the actionable quality report lines and structured gaps from results.

    Returns (lines, all_gaps) where all_gaps is a list of dicts suitable for
    JSON export and interactive review.
    """
    lines = []
    all_gaps = []

    lines.append("")
    lines.append("=" * 80)
    lines.append("ACTIONABLE QUALITY REPORT")
    lines.append("=" * 80)

    assembled_results = [r for r in results if r.get("status") == "ASSEMBLED"]

    # Build clue ref lookup: clue_number -> "1A" style ref
    def _ref(r):
        d = (r.get("direction") or "")
        return "%s%s" % (r["clue_number"], d[0].upper() if d else "")
    ref_map = {r["clue_number"]: _ref(r) for r in results}

    # Track issues by category (clue_number sets)
    db_gap_clues = set()
    type_mismatch_clues = set()
    anagram_fb_clues = set()
    fabricated_clues = set()
    weak_def_clues = set()
    failed_clues = set()

    # --- DB GAPS: unvalidated pieces that could be fixed by adding DB entries ---
    # Include failed clues too — Sonnet's pieces often identify correct
    # synonym/abbreviation mappings even when assembly can't verify them
    db_gaps = []
    for r in results:
        ai = r.get("ai_output") or {}
        pieces = ai.get("pieces", [])
        answer = r.get("answer", "?")
        answer_clean = re.sub(r"[^A-Z]", "", answer.upper())
        for p in pieces:
            mech = p.get("mechanism", "")
            clue_word = (p.get("clue_word") or "").strip()
            letters = (p.get("letters") or "").strip().upper()
            letters_clean = re.sub(r"[^A-Z]", "", letters)
            if not clue_word or not letters_clean:
                continue
            if mech in ("literal", "anagram_fodder", "first_letter", "last_letter",
                        "hidden", "sound_of", "alternate_letters", "core_letters"):
                continue
            if letters_clean == answer_clean:
                continue
            if letters_clean not in answer_clean:
                continue
            clue_lower = clue_word.lower().strip(".,;:!?\"'()-")
            enrichment = r.get("enrichment", "")
            check_words = [clue_lower]
            if " " in clue_lower:
                check_words.extend(clue_lower.split())

            if mech == "synonym":
                word_found = False
                for cw in check_words:
                    for eline in enrichment.split("\n"):
                        eline_lower = eline.strip().lower()
                        if eline_lower.startswith(cw + ":") or eline_lower.startswith(cw + " "):
                            if "syn=" in eline_lower:
                                if letters_clean.lower() in eline_lower:
                                    word_found = True
                            break
                    if word_found:
                        break
                if not word_found:
                    db_gaps.append({
                        "answer": answer, "clue_word": clue_word,
                        "letters": letters_clean, "table": "synonyms_pairs",
                        "clue_number": r["clue_number"],
                        "clue": r.get("clue", ""),
                        "direction": r.get("direction", ""),
                        "score": r.get("score", 0),
                    })
                    db_gap_clues.add(r["clue_number"])
            elif mech == "abbreviation":
                word_found = False
                for cw in check_words:
                    for eline in enrichment.split("\n"):
                        eline_lower = eline.strip().lower()
                        if eline_lower.startswith(cw + ":") or eline_lower.startswith(cw + " "):
                            if "abbr=" in eline_lower:
                                if letters_clean.lower() in eline_lower:
                                    word_found = True
                            break
                    if word_found:
                        break
                if not word_found:
                    db_gaps.append({
                        "answer": answer, "clue_word": clue_word,
                        "letters": letters_clean,
                        "table": "abbreviations" if len(letters_clean) <= 3 else "synonyms_pairs",
                        "clue_number": r["clue_number"],
                        "clue": r.get("clue", ""),
                        "direction": r.get("direction", ""),
                        "score": r.get("score", 0),
                    })
                    db_gap_clues.add(r["clue_number"])

    # Also detect missing definition pairs (include failed clues — Sonnet
    # often identifies the definition even when assembly fails)
    def_gaps = []
    for r in results:
        checks = r.get("checks", {})
        def_check = checks.get("definition", "")
        if "confirmed in DB" in def_check:
            continue
        ai = r.get("ai_output") or {}
        ai_def = (ai.get("definition") or "").strip()
        if not ai_def:
            continue
        answer = r.get("answer", "?")
        def_gaps.append({
            "answer": answer, "definition": ai_def,
            "clue_number": r["clue_number"],
            "clue": r.get("clue", ""),
            "direction": r.get("direction", ""),
            "score": r.get("score", 0),
        })
        db_gap_clues.add(r["clue_number"])

    total_gaps = len(db_gaps) + len(def_gaps)
    if total_gaps:
        lines.append("")
        lines.append("DB GAPS — add these to improve future solves (%d entries)" % total_gaps)
        lines.append("-" * 60)
        for g in def_gaps:
            lines.append("  %s: \"%s\" -> %s  (missing from definition_answers_augmented)" % (
                g["answer"], g["definition"], g["answer"]))
        for g in db_gaps:
            lines.append("  %s: %s -> %s  (missing from %s)" % (
                g["answer"], g["clue_word"], g["letters"], g["table"]))
    else:
        lines.append("")
        lines.append("DB GAPS: none detected")

    # --- TYPE MISMATCHES ---
    mismatches = [r for r in assembled_results
                  if r.get("checks", {}).get("type_mismatch")]
    if mismatches:
        lines.append("")
        lines.append("TYPE MISMATCHES — AI and assembler disagree (%d)" % len(mismatches))
        lines.append("-" * 60)
        for r in mismatches:
            lines.append("  %s = %s (%d/100): %s" % (
                r["answer"], r["clue_number"], r.get("score", 0),
                r["checks"]["type_mismatch"]))
            type_mismatch_clues.add(r["clue_number"])
    else:
        lines.append("")
        lines.append("TYPE MISMATCHES: none")

    # --- ANAGRAM FALLBACKS ---
    anagram_fbs = [r for r in assembled_results
                   if r.get("checks", {}).get("anagram_fallback")]
    if anagram_fbs:
        lines.append("")
        lines.append("ANAGRAM FALLBACKS — likely wrong mechanism (%d)" % len(anagram_fbs))
        lines.append("-" * 60)
        for r in anagram_fbs:
            lines.append("  %s = %s (%d/100): %s" % (
                r["answer"], r["clue_number"], r.get("score", 0),
                r["checks"]["anagram_fallback"]))
            anagram_fb_clues.add(r["clue_number"])
    else:
        lines.append("")
        lines.append("ANAGRAM FALLBACKS: none")

    # --- FABRICATED MAPPINGS ---
    fabricated = []
    for r in assembled_results:
        asm = r.get("assembly") or {}
        issues = []
        if asm.get("gap_fill"):
            for word, yld, mech in asm["gap_fill"]:
                issues.append("%s <- \"%s\" (%s)" % (yld, word, mech))
        if asm.get("brute_gap"):
            issues.append("%s <- brute force (no clue basis)" % asm["brute_gap"])
        if issues:
            fabricated.append({
                "answer": r["answer"], "clue_number": r["clue_number"],
                "score": r.get("score", 0), "issues": issues,
            })
            fabricated_clues.add(r["clue_number"])

    if fabricated:
        lines.append("")
        lines.append("FABRICATED MAPPINGS — assembler invented these (%d clues)" % len(fabricated))
        lines.append("-" * 60)
        for f in fabricated:
            lines.append("  %s (%d/100): %s" % (
                f["answer"], f["score"], "; ".join(f["issues"])))
    else:
        lines.append("")
        lines.append("FABRICATED MAPPINGS: none")

    # --- CIRCULAR ASSEMBLIES ---
    circular_clues = set()
    circular = [r for r in assembled_results
                if r.get("checks", {}).get("circular_assembly")]
    if circular:
        lines.append("")
        lines.append("CIRCULAR ASSEMBLIES — answer used in own explanation (%d)" % len(circular))
        lines.append("-" * 60)
        for r in circular:
            lines.append("  %s = %s (%d/100): %s" % (
                r["answer"], r["clue_number"], r.get("score", 0),
                r["checks"]["circular_assembly"]))
            circular_clues.add(r["clue_number"])
    else:
        lines.append("")
        lines.append("CIRCULAR ASSEMBLIES: none")

    # --- UNVERIFIED EXPLANATIONS ---
    unverified_clues = set()
    unverified = [r for r in assembled_results
                  if r.get("checks", {}).get("single_piece_answer")
                  or r.get("checks", {}).get("zero_validation")]
    if unverified:
        lines.append("")
        lines.append("UNVERIFIED EXPLANATIONS — no real breakdown (%d)" % len(unverified))
        lines.append("-" * 60)
        for r in unverified:
            issues = []
            if r.get("checks", {}).get("single_piece_answer"):
                issues.append(r["checks"]["single_piece_answer"])
            if r.get("checks", {}).get("zero_validation"):
                issues.append(r["checks"]["zero_validation"])
            lines.append("  %s = %s (%d/100): %s" % (
                r["answer"], r["clue_number"], r.get("score", 0),
                "; ".join(issues)))
            unverified_clues.add(r["clue_number"])
    else:
        lines.append("")
        lines.append("UNVERIFIED EXPLANATIONS: none")

    # --- WEAK DEFINITIONS ---
    weak_defs = [r for r in assembled_results
                 if r.get("checks", {}).get("definition") in (
                     "not in DB, odd position", "none identified")]
    if weak_defs:
        lines.append("")
        lines.append("WEAK DEFINITIONS — not confirmed in DB (%d)" % len(weak_defs))
        lines.append("-" * 60)
        for r in weak_defs:
            ai_def = (r.get("ai_output") or {}).get("definition", "?")
            lines.append("  %s = %s (%d/100): def=\"%s\" (%s)" % (
                r["answer"], r["clue_number"], r.get("score", 0),
                ai_def, r["checks"]["definition"]))
            weak_def_clues.add(r["clue_number"])

    # --- FAILED CLUES ---
    failed = [r for r in results if r.get("status") in ("FAILED", "error")]
    if failed:
        lines.append("")
        lines.append("FAILED TO ASSEMBLE (%d)" % len(failed))
        lines.append("-" * 60)
        for r in failed:
            lines.append("  %s. %s = %s" % (
                r["clue_number"], r.get("clue", "")[:60], r["answer"]))
            failed_clues.add(r["clue_number"])

    # --- SUMMARY WITH CLUE NUMBERS ---
    total = len(results)
    all_issue_clues = (type_mismatch_clues | anagram_fb_clues | fabricated_clues
                       | circular_clues | unverified_clues
                       | weak_def_clues | failed_clues | db_gap_clues)
    clean_refs = sorted([ref_map[r["clue_number"]] for r in results
                         if r["clue_number"] not in all_issue_clues],
                        key=lambda x: (x[-1], int(''.join(c for c in x if c.isdigit()) or 0)))

    lines.append("")
    lines.append("=" * 60)
    lines.append("SCORECARD")
    lines.append("=" * 60)
    lines.append("  CLEAN:  %d/%d  %s" % (len(clean_refs), total, ", ".join(clean_refs)))

    # Show each problem category with its clue refs
    categories = [
        ("Type mismatch", type_mismatch_clues),
        ("Anagram fallback", anagram_fb_clues),
        ("Fabricated mapping", fabricated_clues),
        ("Circular assembly", circular_clues),
        ("Unverified explain.", unverified_clues),
        ("Weak definition", weak_def_clues),
        ("DB gap", db_gap_clues),
        ("Failed", failed_clues),
    ]
    problem_total = set()
    for label, clue_set in categories:
        if clue_set:
            refs = sorted([ref_map.get(cn, str(cn)) for cn in clue_set],
                          key=lambda x: (x[-1] if x[-1].isalpha() else '',
                                         int(''.join(c for c in x if c.isdigit()) or 0)))
            lines.append("  %-22s %d  %s" % (label + ":", len(refs), ", ".join(refs)))
            problem_total |= clue_set
    if not problem_total:
        lines.append("  No issues detected.")

    # Build structured gaps list for JSON export
    # DB enrichments (synonyms/abbreviations) first — they directly improve
    # assembly. Definitions second — they only improve confidence score.
    for g in db_gaps:
        gap_type = "abbreviation" if g["table"] == "abbreviations" else "synonym"
        all_gaps.append({
            "type": gap_type,
            "table": g["table"],
            "word": g["clue_word"],
            "letters": g["letters"],
            "answer": g["answer"],
            "clue_number": str(g["clue_number"]),
            "direction": g.get("direction", ""),
            "clue": g.get("clue", ""),
            "score": g.get("score", 0),
        })
    for g in def_gaps:
        all_gaps.append({
            "type": "definition",
            "table": "definition_answers_augmented",
            "definition": g["definition"],
            "answer": g["answer"],
            "clue_number": str(g["clue_number"]),
            "direction": g.get("direction", ""),
            "clue": g.get("clue", ""),
            "score": g.get("score", 0),
        })

    return lines, all_gaps


def generate_report(results, source, puzzle, stats):
    """Generate a formatted puzzle quality report."""
    lines = []

    lines.append("=" * 80)
    lines.append("PUZZLE QUALITY REPORT: %s #%s" % (source, puzzle))
    lines.append("Generated: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("Pipeline: Sonnet -> Assembler -> Enrichment Fallback")
    lines.append("=" * 80)

    # Summary
    total = stats["total"]
    assembled = stats["assembled"]
    lines.append("")
    lines.append("ASSEMBLY SUMMARY")
    lines.append("-" * 40)
    lines.append("  Total clues:      %d" % total)
    lines.append("  Assembled:        %d/%d (%d%%)" % (
        assembled, total, 100 * assembled // max(total, 1)))
    lines.append("    Sonnet:         %d" % stats["sonnet"])
    lines.append("    DB Fallback:    %d" % stats["fallback"])
    if stats.get("cached"):
        lines.append("    Cached (no API): %d" % stats["cached"])
    lines.append("  Failed:           %d" % stats["failed"])
    lines.append("")
    lines.append("CONFIDENCE SUMMARY")
    lines.append("-" * 40)

    # Build clue ref lists per bucket
    def _ref(r):
        d = (r.get("direction") or "")
        return "%s%s" % (r["clue_number"], d[0].upper() if d else "?")

    solved_refs = [_ref(r) for r in results
                   if r.get("status") == "ASSEMBLED" and r.get("score", 0) >= 80]
    medium_refs = [_ref(r) for r in results if r.get("status") == "ASSEMBLED"
                   and _ref(r) not in solved_refs and r.get("score", 0) >= 40]
    low_refs = [_ref(r) for r in results if r.get("status") == "ASSEMBLED"
                and _ref(r) not in solved_refs and r.get("score", 0) < 40]
    failed_refs = [_ref(r) for r in results if r.get("status") in ("FAILED", "error")]

    lines.append("  High (80+):   %d  %s" % (len(solved_refs), ", ".join(solved_refs)))
    lines.append("  Medium (40-79): %d  %s" % (len(medium_refs), ", ".join(medium_refs)))
    lines.append("  Low (<40):    %d  %s" % (len(low_refs), ", ".join(low_refs)))
    lines.append("  Failed:       %d  %s" % (len(failed_refs), ", ".join(failed_refs)))
    lines.append("  Average:      %.0f/100" % stats["avg_score"])
    lines.append("")
    lines.append("COST: $%.4f (Sonnet)" % stats["total_cost"])

    # ================================================================
    # ACTIONABLE QUALITY REPORT (at top for quick scanning)
    # ================================================================
    quality_lines, gaps = _actionable_quality(results)
    lines.extend(quality_lines)

    # Quick-reference table
    lines.append("")
    lines.append("QUICK REFERENCE")
    lines.append("\u2500" * 80)
    for r in results:
        d = r.get("direction", "") or ""
        d_char = d[0].upper() if d else "?"
        ref = "%s%s" % (r["clue_number"], d_char)

        clue_text = r.get("clue", "")
        enum_str = r.get("enumeration", "")
        if enum_str:
            display_clue = "%s (%s)" % (clue_text, enum_str)
        else:
            display_clue = clue_text

        ans = r.get("answer", "?")

        # Assembly description with clue word annotations
        ai_pieces = (r.get("ai_output") or {}).get("pieces", [])
        asm_desc = "---"
        if r.get("status") == "ASSEMBLED":
            asm = r.get("assembly") or {}
            desc = _describe_assembly(asm, ai_pieces, answer=ans)
            if desc:
                asm_desc = desc
            else:
                asm_desc = asm.get("op", "---")

        # Confidence tag
        if r.get("status") == "error":
            tag = "ERR"
        elif r.get("status") == "FAILED":
            tag = "FAIL"
        else:
            score = r.get("score", 0)
            if score >= 80:
                tag = "HIGH"
            elif score >= 40:
                tag = "MED"
            else:
                tag = "LOW"

        lines.append("  %4s  %-12s  %s" % (ref, ans, display_clue))
        lines.append("        %12s  %s  [%s]" % ("", asm_desc, tag))
    lines.append("")

    # Per-clue detail
    lines.append("=" * 80)
    lines.append("PER-CLUE DETAIL")
    lines.append("=" * 80)

    for r in results:
        if r.get("status") == "error":
            lines.append("")
            lines.append("  [ERROR] %s. %s = %s" % (
                r["clue_number"], r.get("clue", "?"), r["answer"]))
            lines.append("  " + "-" * 70)
            continue

        status = r["status"]
        tier = r.get("tier") or "---"
        conf = r.get("confidence", "none")
        score = r.get("score", 0)
        ai = r.get("ai_output") or {}
        asm = r.get("assembly") or {}
        checks = r.get("checks", {})
        explanation = r.get("explanation", "")

        tag = "[%s]" % conf.upper() if status == "ASSEMBLED" else "[FAILED]"
        tier_label = {
            "Sonnet": "Sonnet", "Fallback": "DB Fallback",
            "Cached+Sonnet": "Cached", "Cached+Fallback": "Cached+Fallback",
        }.get(tier, "---")

        lines.append("")
        lines.append("  %s %s. %s" % (tag, r["clue_number"], r.get("clue", "")))
        lines.append("  Answer: %s" % r["answer"])
        lines.append("  Tier: %s | Confidence: %d/100" % (tier_label, score))

        if ai:
            defn = ai.get("definition", "")
            wtype = ai.get("wordplay_type", "")
            if defn:
                lines.append("  Definition: \"%s\"" % defn)
            if wtype:
                lines.append("  Wordplay type: %s" % wtype)

            pieces = ai.get("pieces", [])
            if pieces:
                lines.append("  Components:")
                for p in pieces:
                    mech = p.get("mechanism", "?")
                    word = p.get("clue_word", "?")
                    letters = p.get("letters", "?")
                    # Show deletion/truncation detail when available
                    detail = ""
                    if p.get("source"):
                        src = p["source"]
                        deleted = p.get("deleted", "?")
                        if p.get("indicator"):
                            detail = " [%s, \"%s\"=%s]" % (src, p["indicator"], mech)
                        elif p.get("deleted_word"):
                            detail = " [%s - %s(%s)]" % (src, deleted, p["deleted_word"])
                        else:
                            detail = " [%s - %s]" % (src, deleted)
                    lines.append("    %-18s %-25s -> %s%s" % (mech, word, letters, detail))

        if asm:
            op = asm.get("op", "?")
            detail_ai_pieces = (ai or {}).get("pieces", [])
            desc = _describe_assembly(asm, detail_ai_pieces, answer=r.get("answer"))
            if desc:
                lines.append("  Assembly: %s — %s" % (op, desc))
            else:
                lines.append("  Assembly op: %s" % op)
            if asm.get("gap_fill"):
                lines.append("  Gap fill: %s" % asm["gap_fill"])
            if asm.get("brute_gap"):
                lines.append("  Brute gap: +%s" % asm["brute_gap"])
            if asm.get("note"):
                lines.append("  Note: %s" % asm["note"])
            if asm.get("anagram_fallback"):
                lines.append("  WARNING: Anagram fallback — AI suggested %s" % (
                    (ai or {}).get("wordplay_type", "different type")))
            if asm.get("source") == "enrichment_fallback":
                lines.append("  Source: enrichment fallback (DB-driven)")

        if checks:
            check_strs = []
            for k, v in checks.items():
                check_strs.append("%s=%s" % (k, v))
            lines.append("  Checks: %s" % " | ".join(check_strs))

        # Show DB lookups that were available
        enrichment = r.get("enrichment", "")
        if enrichment:
            lines.append("  DB Lookups:")
            for eline in enrichment.strip().split("\n"):
                lines.append("    %s" % eline)

        if explanation:
            lines.append("  Human explanation: %s" % explanation[:200])

        lines.append("  " + "-" * 70)

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines), gaps
