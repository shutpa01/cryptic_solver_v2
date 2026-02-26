"""Post-processor to align model-predicted fodder to actual clue words.

The Stage 1 model often copies phrases from explanation text (e.g. "trendy = fashionable")
instead of referencing the actual clue words ("trendy"). This module rewrites fodder fields
using reverse lookups against the enrichment DB, without changing yields, indicators, or types.

Usage:
    python classifier/align_fodder.py                          # Dry run on silver_eval_data.jsonl
    python classifier/align_fodder.py --input FILE --apply     # Write aligned JSONL to --output
    python classifier/align_fodder.py --verbose                # Per-example detail
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from itertools import combinations
from pathlib import Path

from classifier.enrich_clue import ClueEnricher

# Types where fodder should reference clue words
ALIGNABLE_TYPES = {
    "synonym", "abbreviation", "literal", "anagram",
    "hidden", "acrostic", "definition", "cryptic_definition",
}

# Types where fodder references intermediate yields — leave unchanged
PASSTHROUGH_TYPES = {
    "reversal", "deletion", "container", "homophone",
}


@dataclass
class AlignmentChange:
    component_index: int
    component_type: str
    old_fodder: str
    new_fodder: str
    method: str  # unchanged, synonym_lookup, abbreviation_lookup, anagram_scan,
                 # hidden_scan, acrostic_scan, literal_match, substring_extract,
                 # definition_lookup, full_clue, passthrough, unresolved


@dataclass
class AlignmentResult:
    fixed_output: dict
    changes: list[AlignmentChange] = field(default_factory=list)
    all_resolved: bool = True


def _clean_word(w):
    """Strip punctuation from a word for matching."""
    return re.sub(r"[.,;:!?\"'()\-]", "", w).lower()


def _tokenize_clue(clue_text):
    """Return list of (raw_word, clean_word) tuples."""
    raw_words = clue_text.split()
    return [(w, _clean_word(w)) for w in raw_words]


def _is_clue_substring(text, clue_text):
    """Check if text appears as a contiguous substring of the clue (case-insensitive)."""
    if not text:
        return False
    return text.lower() in clue_text.lower()


def _find_word_span_indices(text, tokens):
    """Find contiguous token indices whose raw words form the given text (case-insensitive).

    Returns the set of indices, or None if not found.
    """
    text_lower = text.lower().strip()
    for start in range(len(tokens)):
        for end in range(start + 1, len(tokens) + 1):
            span = " ".join(tokens[start:end][i][0] for i in range(end - start))
            if _clean_word(span) == _clean_word(text_lower):
                return set(range(start, end))
            # Also try raw comparison
            if span.lower().strip(".,;:!?\"'()-") == text_lower:
                return set(range(start, end))
    return None


class FodderAligner:
    def __init__(self, enricher: ClueEnricher):
        self.enricher = enricher
        # Build reverse synonym lookup: yield -> set of words that produce it
        self._reverse_synonyms = {}
        for word, syns in self.enricher.synonyms.items():
            for syn in syns:
                self._reverse_synonyms.setdefault(syn.upper(), set()).add(word.lower())

        # Build reverse abbreviation lookup: yield -> set of words
        self._reverse_abbreviations = {}
        for word, subs in self.enricher.abbreviations.items():
            for sub in subs:
                self._reverse_abbreviations.setdefault(sub.upper(), set()).add(word.lower())

    def align(self, model_output, clue_text, answer):
        """Align fodder fields in model_output to reference clue words.

        Args:
            model_output: dict with 'definition', 'wordplay_types', 'components'
            clue_text: the original clue string
            answer: the answer string

        Returns:
            AlignmentResult with fixed_output and list of changes
        """
        if not model_output or not model_output.get("components"):
            return AlignmentResult(fixed_output=model_output or {})

        tokens = _tokenize_clue(clue_text)
        consumed = set()  # indices of tokens already assigned

        # Pre-consume definition word indices
        definition = model_output.get("definition")
        if definition:
            def_indices = _find_word_span_indices(definition, tokens)
            if def_indices:
                consumed.update(def_indices)

        # Pre-consume indicator word indices
        for comp in model_output["components"]:
            indicator = comp.get("indicator")
            if indicator:
                ind_indices = _find_word_span_indices(indicator, tokens)
                if ind_indices:
                    consumed.update(ind_indices)

        fixed = json.loads(json.dumps(model_output))  # deep copy
        changes = []
        all_resolved = True

        for idx, comp in enumerate(fixed["components"]):
            ctype = comp.get("type", "")
            fodder = comp.get("fodder", "") or ""

            if ctype in PASSTHROUGH_TYPES:
                changes.append(AlignmentChange(
                    component_index=idx, component_type=ctype,
                    old_fodder=fodder, new_fodder=fodder,
                    method="passthrough"
                ))
                continue

            if ctype == "cryptic_definition":
                # Set fodder = entire clue text
                if fodder != clue_text:
                    comp["fodder"] = clue_text
                    changes.append(AlignmentChange(
                        component_index=idx, component_type=ctype,
                        old_fodder=fodder, new_fodder=clue_text,
                        method="full_clue"
                    ))
                else:
                    changes.append(AlignmentChange(
                        component_index=idx, component_type=ctype,
                        old_fodder=fodder, new_fodder=fodder,
                        method="unchanged"
                    ))
                continue

            # Check if fodder is already a clue substring
            if fodder and _is_clue_substring(fodder, clue_text):
                span = _find_word_span_indices(fodder, tokens)
                if span:
                    consumed.update(span)
                changes.append(AlignmentChange(
                    component_index=idx, component_type=ctype,
                    old_fodder=fodder, new_fodder=fodder,
                    method="unchanged"
                ))
                continue

            # Type-specific alignment
            yields = (comp.get("yields") or "").upper()
            new_fodder, method = self._align_component(
                ctype, fodder, yields, clue_text, tokens, consumed, answer
            )

            if new_fodder is not None and new_fodder != fodder:
                comp["fodder"] = new_fodder
                span = _find_word_span_indices(new_fodder, tokens)
                if span:
                    consumed.update(span)
                changes.append(AlignmentChange(
                    component_index=idx, component_type=ctype,
                    old_fodder=fodder, new_fodder=new_fodder,
                    method=method
                ))
            elif new_fodder is None:
                # Unresolved
                all_resolved = False
                changes.append(AlignmentChange(
                    component_index=idx, component_type=ctype,
                    old_fodder=fodder, new_fodder=fodder,
                    method="unresolved"
                ))
            else:
                changes.append(AlignmentChange(
                    component_index=idx, component_type=ctype,
                    old_fodder=fodder, new_fodder=fodder,
                    method="unchanged"
                ))

        return AlignmentResult(
            fixed_output=fixed,
            changes=changes,
            all_resolved=all_resolved
        )

    def _align_component(self, ctype, fodder, yields, clue_text, tokens, consumed, answer):
        """Try type-specific alignment. Returns (new_fodder, method) or (None, 'unresolved')."""

        available = [(i, raw, clean) for i, (raw, clean) in enumerate(tokens)
                     if i not in consumed and clean]

        if ctype == "synonym":
            result = self._align_synonym(yields, available, tokens, consumed)
            if result:
                return result
        elif ctype == "abbreviation":
            result = self._align_abbreviation(yields, available, tokens, consumed)
            if result:
                return result
        elif ctype == "literal":
            result = self._align_literal(yields, clue_text, tokens, consumed)
            if result:
                return result
        elif ctype == "anagram":
            result = self._align_anagram(yields, available, tokens, consumed)
            if result:
                return result
        elif ctype == "hidden":
            result = self._align_hidden(yields, available, tokens, consumed)
            if result:
                return result
        elif ctype == "acrostic":
            result = self._align_acrostic(yields, available, tokens, consumed)
            if result:
                return result
        elif ctype == "definition":
            result = self._align_definition(yields, answer, available, tokens, consumed)
            if result:
                return result

        # Fallback: substring extract — check if any unconsumed clue word appears
        # as a substring of the explanation-derived fodder
        result = self._fallback_substring_extract(fodder, available)
        if result:
            return result

        return None, "unresolved"

    def _align_synonym(self, yields, available, tokens, consumed):
        """Reverse-lookup: find clue word(s) where yields appears in their synonyms or abbreviations."""
        if not yields:
            return None

        # Gather candidate words from the reverse lookup
        candidate_words = set()
        rev_syns = self._reverse_synonyms.get(yields, set())
        candidate_words.update(rev_syns)
        rev_abbrs = self._reverse_abbreviations.get(yields, set())
        candidate_words.update(rev_abbrs)

        # Also check forward: clue words that have yields in their synonym list
        for i, raw, clean in available:
            if not clean:
                continue
            syns = self.enricher.synonyms.get(clean, [])
            if yields in syns:
                candidate_words.add(clean)
            abbrs = self.enricher.abbreviations.get(clean, [])
            if yields in abbrs or yields.lower() in [a.lower() for a in abbrs]:
                candidate_words.add(clean)

        # Match candidates against available clue words
        # Try single words first, then 2-word phrases, then 3-word phrases
        best = None
        best_len = 999

        # Single words
        for i, raw, clean in available:
            if clean in candidate_words:
                if best is None or len(clean) < best_len:
                    best = raw
                    best_len = len(clean)

        # 2-word phrases
        avail_indices = [i for i, _, _ in available]
        for a_idx in range(len(avail_indices)):
            for b_idx in range(a_idx + 1, len(avail_indices)):
                i, j = avail_indices[a_idx], avail_indices[b_idx]
                if j != i + 1:
                    continue  # must be contiguous
                phrase = tokens[i][1] + " " + tokens[j][1]
                if phrase in candidate_words:
                    raw_phrase = tokens[i][0] + " " + tokens[j][0]
                    return raw_phrase, "synonym_lookup"
                # Also check forward
                syns = self.enricher.synonyms.get(phrase, [])
                if yields in syns:
                    raw_phrase = tokens[i][0] + " " + tokens[j][0]
                    return raw_phrase, "synonym_lookup"

        # 3-word phrases
        for a_idx in range(len(avail_indices)):
            for b_idx in range(a_idx + 1, len(avail_indices)):
                for c_idx in range(b_idx + 1, len(avail_indices)):
                    i, j, k = avail_indices[a_idx], avail_indices[b_idx], avail_indices[c_idx]
                    if j != i + 1 or k != j + 1:
                        continue
                    phrase = tokens[i][1] + " " + tokens[j][1] + " " + tokens[k][1]
                    if phrase in candidate_words:
                        raw_phrase = tokens[i][0] + " " + tokens[j][0] + " " + tokens[k][0]
                        return raw_phrase, "synonym_lookup"
                    syns = self.enricher.synonyms.get(phrase, [])
                    if yields in syns:
                        raw_phrase = tokens[i][0] + " " + tokens[j][0] + " " + tokens[k][0]
                        return raw_phrase, "synonym_lookup"

        if best:
            return best, "synonym_lookup"

        return None

    def _align_abbreviation(self, yields, available, tokens, consumed):
        """Find clue word where yields appears in its abbreviations dict."""
        if not yields:
            return None

        for i, raw, clean in available:
            if not clean:
                continue
            abbrs = self.enricher.abbreviations.get(clean, [])
            if yields in abbrs or yields.lower() in [a.lower() for a in abbrs]:
                return raw, "abbreviation_lookup"

        # Also check reverse abbreviation lookup
        rev = self._reverse_abbreviations.get(yields, set())
        for i, raw, clean in available:
            if clean in rev:
                return raw, "abbreviation_lookup"

        return None

    def _align_literal(self, yields, clue_text, tokens, consumed):
        """Find yields as a literal substring in the clue text."""
        if not yields:
            return None

        # Check for exact word match in available tokens
        for i, (raw, clean) in enumerate(tokens):
            if i in consumed:
                continue
            if clean == yields.lower() or raw.lower().strip(".,;:!?\"'()-") == yields.lower():
                return raw, "literal_match"

        # Check as substring
        if _is_clue_substring(yields, clue_text):
            return yields.lower(), "literal_match"

        return None

    def _align_anagram(self, yields, available, tokens, consumed):
        """Sliding window: find contiguous words whose sorted letters match sorted yields."""
        if not yields:
            return None

        target = sorted(yields.lower().replace(" ", ""))
        avail_indices = sorted(i for i, _, _ in available)

        # Try windows of different sizes
        for window_size in range(1, len(avail_indices) + 1):
            # Try contiguous windows from the full token list (not just available)
            for start in range(len(tokens)):
                end = start + window_size
                if end > len(tokens):
                    break
                indices = set(range(start, end))
                # At least some of these should be available (not all consumed)
                if not indices - consumed:
                    continue

                span_words = [tokens[j][1] for j in range(start, end)]
                span_letters = sorted("".join(span_words).replace(" ", ""))
                if span_letters == target:
                    raw_span = " ".join(tokens[j][0] for j in range(start, end))
                    return raw_span, "anagram_scan"

        return None

    def _align_hidden(self, yields, available, tokens, consumed):
        """Find where yields appears hidden in concatenated clue words."""
        if not yields:
            return None

        target = yields.lower().replace(" ", "")

        # Try all possible contiguous spans
        for start in range(len(tokens)):
            for end in range(start + 1, len(tokens) + 1):
                indices = set(range(start, end))
                if not indices - consumed:
                    continue
                concat = "".join(tokens[j][1] for j in range(start, end))
                if target in concat and concat != target:
                    # The hidden word must span across words (not be a single word)
                    raw_span = " ".join(tokens[j][0] for j in range(start, end))
                    return raw_span, "hidden_scan"

        return None

    def _align_acrostic(self, yields, available, tokens, consumed):
        """Find contiguous clue words whose initials spell yields."""
        if not yields:
            return None

        target = yields.lower()
        avail_indices = sorted(i for i, _, _ in available)

        for start in range(len(tokens)):
            end = start + len(target)
            if end > len(tokens):
                break
            indices = set(range(start, end))
            if not indices - consumed:
                continue
            initials = "".join(tokens[j][1][0] if tokens[j][1] else "" for j in range(start, end))
            if initials == target:
                raw_span = " ".join(tokens[j][0] for j in range(start, end))
                return raw_span, "acrostic_scan"

        return None

    def _align_definition(self, yields, answer, available, tokens, consumed):
        """For DD components: check fodder is a clue substring, else use definition DB."""
        if not answer:
            return None

        # Check available words against definition DB
        for n in range(1, min(5, len(tokens) + 1)):
            # Try from start
            prefix = " ".join(tokens[i][0] for i in range(n))
            prefix_clean = _clean_word(prefix)
            if self.enricher.lookup_definition(prefix_clean, answer):
                # Check if any of these indices are available
                indices = set(range(n))
                if indices - consumed:
                    return prefix.strip(".,;:!?\"'()-"), "definition_lookup"

            # Try from end
            start_idx = len(tokens) - n
            suffix = " ".join(tokens[i][0] for i in range(start_idx, len(tokens)))
            suffix_clean = _clean_word(suffix)
            if self.enricher.lookup_definition(suffix_clean, answer):
                indices = set(range(start_idx, len(tokens)))
                if indices - consumed:
                    return suffix.strip(".,;:!?\"'()-"), "definition_lookup"

        # Try available word groups
        avail_indices = sorted(i for i, _, _ in available)
        for n in range(1, min(4, len(avail_indices) + 1)):
            for start in range(len(avail_indices) - n + 1):
                idxs = avail_indices[start:start + n]
                # Must be contiguous
                if idxs[-1] - idxs[0] != n - 1:
                    continue
                phrase = " ".join(tokens[i][0] for i in idxs)
                phrase_clean = _clean_word(phrase)
                if self.enricher.lookup_definition(phrase_clean, answer):
                    return phrase.strip(".,;:!?\"'()-"), "definition_lookup"

        return None

    def _fallback_substring_extract(self, fodder, available):
        """Check if any unconsumed clue word appears as a substring of the explanation fodder.

        Catches cases like "trendy = fashionable" -> "trendy".
        Requires at least 3 chars for single-word matches to avoid grabbing "a", "in", etc.
        """
        if not fodder:
            return None

        fodder_lower = fodder.lower()

        # Try multi-word matches first (longer = better)
        avail_indices = sorted(i for i, _, _ in available)
        best = None
        best_len = 0

        for n in range(min(3, len(avail_indices)), 0, -1):
            for start in range(len(avail_indices) - n + 1):
                idxs = avail_indices[start:start + n]
                # Must be contiguous
                if idxs[-1] - idxs[0] != n - 1:
                    continue
                from_available = [(i, r, c) for i, r, c in available if i in set(idxs)]
                phrase_clean = " ".join(c for _, _, c in from_available)
                if phrase_clean and len(phrase_clean) >= 3 and phrase_clean in fodder_lower:
                    raw_phrase = " ".join(r for _, r, _ in from_available)
                    if len(phrase_clean) > best_len:
                        best = raw_phrase
                        best_len = len(phrase_clean)

        if best:
            return best, "substring_extract"

        # Single word fallback — require 3+ chars to avoid "a", "in" etc.
        for i, raw, clean in available:
            if clean and len(clean) >= 3 and clean in fodder_lower:
                return raw, "substring_extract"

        return None

    def align_batch(self, examples, key="rule_parser"):
        """Align fodder for a batch of examples.

        Args:
            examples: list of dicts with 'clue', 'answer', and key containing model output
            key: the dict key containing the model output ('rule_parser' or 'structured')

        Returns:
            (fixed_examples, report_dict)
        """
        fixed = []
        total = 0
        changed = 0
        unresolved = 0
        method_counts = {}

        for ex in examples:
            model_output = ex.get(key)
            if not model_output:
                fixed.append(ex)
                continue

            result = self.align(model_output, ex["clue"], ex["answer"])
            total += 1

            new_ex = dict(ex)
            new_ex[key] = result.fixed_output
            fixed.append(new_ex)

            for c in result.changes:
                method_counts[c.method] = method_counts.get(c.method, 0) + 1
                if c.method not in ("unchanged", "passthrough"):
                    if c.method == "unresolved":
                        unresolved += 1
                    else:
                        changed += 1

            if not result.all_resolved:
                pass  # already counted above

        report = {
            "total_examples": total,
            "components_changed": changed,
            "components_unresolved": unresolved,
            "methods": method_counts,
        }
        return fixed, report


def _fodder_in_clue_rate(examples, key="rule_parser"):
    """Calculate what % of fodder fields are clue substrings."""
    total = 0
    in_clue = 0
    for ex in examples:
        model_output = ex.get(key)
        if not model_output or not model_output.get("components"):
            continue
        clue = ex["clue"]
        for comp in model_output["components"]:
            fodder = comp.get("fodder", "")
            ctype = comp.get("type", "")
            if not fodder or ctype in PASSTHROUGH_TYPES:
                continue
            total += 1
            if _is_clue_substring(fodder, clue):
                in_clue += 1
    return in_clue, total


def main():
    # Handle Windows cp1252 encoding
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Align model fodder to clue words")
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSONL file (default: data/silver_eval_data.jsonl)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file (default: <input>_aligned.jsonl)")
    parser.add_argument("--apply", action="store_true",
                        help="Write aligned output to file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-example alignment details")
    parser.add_argument("--key", type=str, default=None,
                        help="JSON key for model output (auto-detected: rule_parser or structured)")
    args = parser.parse_args()

    # Resolve input file
    base = Path(__file__).parent.parent
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = base / "data" / "silver_eval_data.jsonl"

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    # Load examples
    examples = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples from {input_path.name}")

    # Auto-detect key
    key = args.key
    if not key:
        if examples and "rule_parser" in examples[0]:
            key = "rule_parser"
        elif examples and "structured" in examples[0]:
            key = "structured"
        else:
            key = "rule_parser"
    print(f"Using key: {key}")

    # Before metrics
    before_in, before_total = _fodder_in_clue_rate(examples, key)
    if before_total:
        print(f"\nBEFORE: {before_in}/{before_total} fodder fields in clue ({100*before_in/before_total:.1f}%)")

    # Load enricher and align
    print("\nLoading enricher...")
    enricher = ClueEnricher()
    aligner = FodderAligner(enricher)

    print("Aligning...")
    fixed_examples, report = aligner.align_batch(examples, key=key)

    # After metrics
    after_in, after_total = _fodder_in_clue_rate(fixed_examples, key)
    if after_total:
        print(f"\nAFTER:  {after_in}/{after_total} fodder fields in clue ({100*after_in/after_total:.1f}%)")

    # Report
    print(f"\nAlignment report:")
    print(f"  Examples processed:     {report['total_examples']}")
    print(f"  Components changed:     {report['components_changed']}")
    print(f"  Components unresolved:  {report['components_unresolved']}")
    print(f"  Methods:")
    for method, count in sorted(report["methods"].items(), key=lambda x: -x[1]):
        print(f"    {method:24s} {count}")

    # Verbose output
    if args.verbose:
        print("\n" + "=" * 70)
        print("PER-EXAMPLE DETAILS")
        print("=" * 70)
        for i, (orig, fixed) in enumerate(zip(examples, fixed_examples)):
            orig_out = orig.get(key, {})
            fixed_out = fixed.get(key, {})
            orig_comps = orig_out.get("components", []) if orig_out else []
            fixed_comps = fixed_out.get("components", []) if fixed_out else []

            has_change = False
            for oc, fc in zip(orig_comps, fixed_comps):
                if oc.get("fodder") != fc.get("fodder"):
                    has_change = True
                    break

            if not has_change and not orig_comps:
                continue

            result = aligner.align(orig_out, orig["clue"], orig["answer"])

            print(f"\n--- [{i}] {orig['clue']} -> {orig['answer']} ---")
            if orig_out:
                print(f"  Definition: {orig_out.get('definition')}")
                print(f"  Types: {orig_out.get('wordplay_types')}")
            for c in result.changes:
                if c.method in ("unchanged", "passthrough"):
                    continue
                marker = "!" if c.method == "unresolved" else "+"
                print(f"  {marker} [{c.component_index}] {c.component_type}: "
                      f"{c.old_fodder!r} -> {c.new_fodder!r}  ({c.method})")

    # Write output
    if args.apply:
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_name(input_path.stem + "_aligned.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in fixed_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"\nWritten to {output_path}")

    enricher.close()


if __name__ == "__main__":
    main()
