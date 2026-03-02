"""Sonnet pipeline report generator.

Extracted from pipeline_tiered.py. Takes a list of result dicts and stats,
returns a formatted report string.
"""

from datetime import datetime

import re

from .solver import OP_TO_TYPE


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


def _describe_assembly(asm, ai_pieces=None):
    """Return a human-readable description of an assembly operation."""
    op = asm.get("op", "?")
    ai = ai_pieces or []

    if op == "charade":
        order = asm.get("order", [])
        if order:
            # Merge adjacent pieces when their concatenation matches an AI piece
            # e.g. ["P", "T"] with AI piece letters="PT" → single "PT(priest vacated)"
            merged = []
            i = 0
            while i < len(order):
                # Try merging 2 or 3 adjacent pieces
                found_merge = False
                for span in (3, 2):
                    if i + span <= len(order):
                        combined = "".join(order[i:i + span])
                        label = _piece_label(combined, ai)
                        if label:
                            merged.append("%s(%s)" % (combined, label))
                            i += span
                            found_merge = True
                            break
                if not found_merge:
                    merged.append(_annotate(order[i], ai))
                    i += 1
            return " + ".join(merged)

    elif op == "container":
        inner = asm.get("inner", "?")
        outer = asm.get("outer", "?")
        merged = asm.get("merged_inner")
        if merged:
            container_desc = "%s inside %s" % (_annotate_composite(merged, ai), _annotate(outer, ai))
        else:
            container_desc = "%s inside %s" % (_annotate(inner, ai), _annotate(outer, ai))
        return _with_charade(container_desc, asm.get("order", []), asm.get("combined"), ai)

    elif op == "reversal":
        rev_parts = asm.get("reversed_parts")
        if rev_parts:
            # Full reversal of concatenated pieces: "reverse CID+AM+ON"
            parts_desc = "+".join(_annotate(p, ai) for p in rev_parts)
            return "reverse %s" % parts_desc
        rev_desc = "reverse %s" % _annotate(asm.get("reversed", "?"), ai)
        return _with_charade(rev_desc, asm.get("order", []), asm.get("gives", ""), ai)

    elif op == "deletion":
        del_desc = "delete %s from %s" % (
            asm.get("deleted", "?"), _annotate(asm.get("from", "?"), ai))
        return _with_charade(del_desc, asm.get("order", []), asm.get("gives", ""), ai)

    elif op == "outer_deletion":
        od_desc = "strip outer letters from %s" % _annotate(asm.get("from", "?"), ai)
        return _with_charade(od_desc, asm.get("order", []), asm.get("gives", ""), ai)

    elif op == "anagram":
        fodder = asm.get("fodder", [])
        if fodder:
            return "anagram of %s" % "+".join(_annotate(f, ai) for f in fodder)

    elif op in ("charade+anagram", "anagram+charade"):
        charade = asm.get("charade", [])
        anagram = asm.get("anagram", [])
        parts = []
        if charade:
            # Merge adjacent charade pieces (same logic as pure charade)
            i = 0
            while i < len(charade):
                found_merge = False
                for span in (3, 2):
                    if i + span <= len(charade):
                        combined = "".join(charade[i:i + span])
                        label = _piece_label(combined, ai)
                        if label:
                            parts.append("%s(%s)" % (combined, label))
                            i += span
                            found_merge = True
                            break
                if not found_merge:
                    parts.append(_annotate(charade[i], ai))
                    i += 1
        if anagram:
            parts.append("anagram(%s)" % "+".join(_annotate(f, ai) for f in anagram))
        if parts:
            return " + ".join(parts)

    elif op == "deletion+anagram":
        fodder = asm.get("fodder", [])
        return "delete %s from %s, anagram of %s" % (
            asm.get("deleted", "?"), _annotate(asm.get("from", "?"), ai),
            "+".join(_annotate(f, ai) for f in fodder) if fodder else "?")

    elif op == "reversal_container":
        inner = asm.get("inner", "?")
        outer = asm.get("outer", "?")
        pre_rev = asm.get("pre_reversal", "?")
        # Show which piece was reversed and the container structure
        # e.g. "reverse FAT(plump), then inside FETA(cheese)"
        rev_part = _annotate(pre_rev, ai) if pre_rev != "?" else inner
        container_desc = "reverse %s, then %s inside %s" % (
            rev_part, _annotate(inner, ai), _annotate(outer, ai))
        return _with_charade(container_desc, asm.get("order", []), asm.get("combined"), ai)

    elif op == "container_reversal":
        inner = asm.get("inner", "?")
        outer = asm.get("outer", "?")
        reversed_combined = asm.get("reversed_combined", "?")
        # E.g. "E(base) inside PEWS(benches), then reverse = SWEEP"
        container_desc = "%s inside %s, then reverse" % (
            _annotate(inner, ai), _annotate(outer, ai))
        return _with_charade(container_desc, asm.get("order", []), reversed_combined, ai)

    elif op == "hidden":
        return "hidden in '%s'" % asm.get("words", "?")

    elif op == "hidden_in_word":
        return "hidden in word '%s'" % asm.get("word", "?")

    elif op == "homophone":
        return "%s sounds like %s" % (asm.get("sounds_like", "?"), asm.get("gives", "?"))

    elif op == "spoonerism":
        src = asm.get("source_words", ["?", "?"])
        res = asm.get("result_words", ["?", "?"])
        return "spoonerism of %s %s → %s %s" % (src[0], src[1], res[0], res[1])

    elif op == "substitution":
        return "%s - %s + %s(%s)" % (
            _annotate(asm.get("from", "?"), ai), asm.get("deleted", "?"),
            asm.get("added", "?"), asm.get("add_word", "?"))

    elif op in ("double_definition", "cryptic_definition"):
        return op.replace("_", " ")

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
    db_gaps = []
    for r in assembled_results:
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

    # Also detect missing definition pairs
    def_gaps = []
    for r in assembled_results:
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
            desc = _describe_assembly(asm, ai_pieces)
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
            desc = _describe_assembly(asm, detail_ai_pieces)
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
