"""Verifier — the referee. Reads only the form. Returns PASS/FAIL.

The verifier does not participate in solving. It checks the form against
the four rules from the spec and reports a binary verdict. The reason for
any FAIL is in the per-check breakdown — that's the diagnostic, not a
graded score.

The four rules (from memory/project_universal_form_schema.md):

    1. Assembly   walk the tree bottom-up; result must equal the answer.
    2. Bridge     every non-literal node has a known indicator in the DB
                  for that operation type, EXCEPT charade and
                  double_definition (implicitly licensed by structure).
                  Every leaf source -> value verifies in DB (synonym,
                  abbreviation) or mechanically (literal, positional).
    3. Mechanism  same as bridge, applied to the outermost node.
    4. Residue    form-claimed surface words = leaf source_words +
                  op indicators + positional indicators + definition.phrase.
                  Surface words = form-claimed + form.link_words.
                  form.link_words must all be in the global LINK_WORDS
                  allow-list.

Plus carry-overs:

    - Definition: form.definition.phrase -> form.definition.answer must
      verify in DB.
    - &lit cap: if is_and_lit, the form passes ONLY if the full clue text
      maps to the answer in DB.
    - Trivial-tree shape: a tree that is just a single synonym leaf equal
      to the answer is no wordplay; FAIL.

Verdict: PASS if every check passes; FAIL if any check fails.
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any

from .schema import (
    Form, Leaf, Op, Node,
    LEAF_OPERATIONS, OP_OPERATIONS,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REF_DB = str(PROJECT_ROOT / "data" / "cryptic_new.db")

# Global LINK_WORDS allow-list. The form's `link_words` field must be a
# subset of this set. Mirrors LINKERS in the live verifier.
LINK_WORDS = {
    "of", "in", "the", "a", "an", "to", "for", "with", "and", "or",
    "by", "from", "as", "on", "at", "but", "so", "yet", "if", "not",
    "nor", "up", "it", "its", "into", "onto", "within", "without",
    "that", "which", "when", "where", "while", "how", "why", "who",
    "this", "these", "those", "such", "one", "ones", "some", "any",
    "all", "here", "there",
    "is", "are", "be", "been", "being", "was", "were",
    "has", "have", "had", "having",
    "will", "would", "could", "should", "must", "may", "might",
    "get", "gets", "got", "getting",
    "give", "gives", "gave", "given", "giving",
    "make", "makes", "made", "making",
    "need", "needs",
    "thus", "hence", "therefore", "maybe",
    "dont", "doesnt", "didnt", "wont", "wouldnt", "cant", "isnt", "arent",
    "once",
}

# Operations exempt from the "must have an indicator" requirement of
# rule 2. Both are implicitly licensed by their structural shape — a
# charade by having multiple children, a double_definition by having
# exactly two definitional children.
INDICATOR_EXEMPT_OPS = {"charade", "double_definition"}


# --- Result types ----------------------------------------------------------

@dataclass
class Check:
    """A single check result. status is 'pass' or 'fail'.

    `enrichment_proposal` is populated on FAIL when the check failure
    corresponds to a queueable DB gap (production would propose this
    enrichment via Haiku for human review). Format mirrors the
    `pending_enrichments` table: {type, word, letters, [op_type]}.
    """
    name: str
    status: str          # "pass" or "fail"
    detail: str
    enrichment_proposal: Optional[dict] = None

    def to_dict(self) -> dict:
        d = {"check": self.name, "status": self.status,
             "detail": self.detail}
        if self.enrichment_proposal is not None:
            d["enrichment_proposal"] = self.enrichment_proposal
        return d


@dataclass
class Verdict:
    """Verifier verdict — PASS if every check passed, FAIL otherwise."""
    verdict: str         # "PASS" or "FAIL"
    checks: list[Check] = field(default_factory=list)
    assembly_value: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "assembly_value": self.assembly_value,
            "checks": [c.to_dict() for c in self.checks],
        }


# --- Verifier --------------------------------------------------------------

class FormVerifier:
    """Pure referee. Reads only the form + clue_text. Read-only DB."""

    def __init__(self, ref_db: Optional[str] = None):
        self._ref = sqlite3.connect(ref_db or REF_DB)
        self._ref.row_factory = sqlite3.Row
        self._syn_cache: dict[tuple[str, str], bool] = {}
        self._abbr_cache: dict[tuple[str, str], bool] = {}
        self._ind_cache: dict[tuple[str, str], bool] = {}
        self._def_cache: dict[tuple[str, str], bool] = {}
        self._homo_cache: dict[tuple[str, str], bool] = {}

    def close(self) -> None:
        self._ref.close()

    # --- DB lookups ------------------------------------------------------

    def is_synonym(self, word: str, target: str) -> bool:
        key = (word.lower(), target.lower())
        if key in self._syn_cache:
            return self._syn_cache[key]
        w, t = word.lower(), target.lower()
        found = False
        for w1, w2 in [(w, t), (t, w)]:
            if found:
                break
            r = self._ref.execute(
                "SELECT 1 FROM synonyms_pairs "
                "WHERE LOWER(word)=? AND LOWER(synonym)=? LIMIT 1",
                (w1, w2)).fetchone()
            if r:
                found = True
                break
            r = self._ref.execute(
                "SELECT 1 FROM definition_answers_augmented "
                "WHERE LOWER(definition)=? AND LOWER(answer)=? LIMIT 1",
                (w1, w2)).fetchone()
            if r:
                found = True
                break
        self._syn_cache[key] = found
        return found

    def is_abbreviation(self, word: str, letters: str) -> bool:
        key = (word.lower(), letters.upper())
        if key in self._abbr_cache:
            return self._abbr_cache[key]
        r = self._ref.execute(
            "SELECT 1 FROM wordplay "
            "WHERE LOWER(indicator)=? AND UPPER(substitution)=? LIMIT 1",
            (word.lower(), letters.upper())).fetchone()
        if not r:
            r = self._ref.execute(
                "SELECT 1 FROM synonyms_pairs "
                "WHERE LOWER(word)=? AND UPPER(synonym)=? LIMIT 1",
                (word.lower(), letters.upper())).fetchone()
        self._abbr_cache[key] = r is not None
        return self._abbr_cache[key]

    def is_indicator(self, word: str, op_type: str) -> bool:
        key = (word.lower(), op_type.lower())
        if key in self._ind_cache:
            return self._ind_cache[key]
        r = self._ref.execute(
            "SELECT 1 FROM indicators "
            "WHERE LOWER(word)=? AND LOWER(wordplay_type)=? LIMIT 1",
            (word.lower(), op_type.lower())).fetchone()
        # Container also accepts insertion-type indicators.
        if not r and op_type.lower() == "container":
            r = self._ref.execute(
                "SELECT 1 FROM indicators "
                "WHERE LOWER(word)=? AND LOWER(wordplay_type)='insertion' "
                "LIMIT 1",
                (word.lower(),)).fetchone()
        # Positional accepts acrostic/parts (which is how the DB tags
        # first/last letter indicators for many words).
        if not r and op_type.lower() == "positional":
            r = self._ref.execute(
                "SELECT 1 FROM indicators "
                "WHERE LOWER(word)=? AND "
                "LOWER(wordplay_type) IN ('acrostic','parts','positional') "
                "LIMIT 1",
                (word.lower(),)).fetchone()
        # Acrostic accepts 'parts' tags too (the DB uses 'parts' for many
        # multi-word positional phrases like 'at first', 'in the end').
        if not r and op_type.lower() == "acrostic":
            r = self._ref.execute(
                "SELECT 1 FROM indicators "
                "WHERE LOWER(word)=? AND "
                "LOWER(wordplay_type) IN ('acrostic','parts') "
                "LIMIT 1",
                (word.lower(),)).fetchone()
        self._ind_cache[key] = r is not None
        return self._ind_cache[key]

    def definition_matches(self, definition: str, answer: str) -> bool:
        key = (definition.lower(), answer.upper())
        if key in self._def_cache:
            return self._def_cache[key]
        d, a = definition.lower().strip(), answer.upper().strip()
        r = self._ref.execute(
            "SELECT 1 FROM definition_answers_augmented "
            "WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
            (d, a)).fetchone()
        if not r:
            r = self._ref.execute(
                "SELECT 1 FROM synonyms_pairs "
                "WHERE LOWER(word)=? AND UPPER(synonym)=? LIMIT 1",
                (d, a)).fetchone()
        if not r:
            r = self._ref.execute(
                "SELECT 1 FROM synonyms_pairs "
                "WHERE LOWER(synonym)=? AND UPPER(word)=? LIMIT 1",
                (d, a)).fetchone()
        self._def_cache[key] = r is not None
        return self._def_cache[key]

    def is_homophone(self, w1: str, w2: str) -> bool:
        a, b = w1.lower().strip(), w2.lower().strip()
        if not a or not b:
            return False
        key = tuple(sorted([a, b]))
        if key in self._homo_cache:
            return self._homo_cache[key]
        r = self._ref.execute(
            "SELECT 1 FROM homophones WHERE "
            "(LOWER(word)=? AND LOWER(homophone)=?) OR "
            "(LOWER(word)=? AND LOWER(homophone)=?) LIMIT 1",
            (a, b, b, a)).fetchone()
        self._homo_cache[key] = r is not None
        return self._homo_cache[key]

    # --- The four checks ---------------------------------------------------

    def verify(self, form: Form, clue_text: str) -> Verdict:
        checks: list[Check] = []

        # Carry-over: definition lookup
        d_ok = self.definition_matches(
            form.definition.phrase, form.definition.answer)
        checks.append(Check(
            "definition",
            "pass" if d_ok else "fail",
            f'"{form.definition.phrase}" -> {form.definition.answer}: '
            f'{"in DB" if d_ok else "not in DB"}',
            enrichment_proposal=None if d_ok else {
                "type": "definition",
                "word": form.definition.phrase,
                "letters": form.definition.answer,
            },
        ))

        # Rule 1: assembly
        assembly_value = ""
        try:
            assembly_value = compute_assembly(
                form.tree, form.definition.answer)
            ok = (assembly_value.upper().replace(" ", "")
                  == form.definition.answer.upper().replace(" ", ""))
            checks.append(Check(
                "assembly",
                "pass" if ok else "fail",
                f'tree -> {assembly_value!r} vs answer '
                f'{form.definition.answer!r}: '
                f'{"MATCH" if ok else "MISMATCH"}',
            ))
        except AssemblyError as e:
            checks.append(Check("assembly", "fail",
                                f"assembly failed: {e}"))

        # Rules 2 & 3: bridge per node, mechanism for the root
        for path, node in walk(form.tree):
            self._check_node(path, node, checks,
                             is_root=(node is form.tree))

        # Rule 4: residue
        checks.extend(self._check_residue(form, clue_text))

        # Carry-over: trivial-tree shape
        if isinstance(form.tree, Leaf) and form.tree.operation == "synonym":
            if form.tree.value.upper() == \
                    form.definition.answer.upper().replace(" ", ""):
                checks.append(Check(
                    "trivial_shape", "fail",
                    "tree is a single synonym leaf equal to the answer — "
                    "no wordplay"))

        # Carry-over: &lit cap
        if form.is_and_lit:
            and_lit_ok = self.definition_matches(
                clue_text, form.definition.answer)
            checks.append(Check(
                "and_lit_cap",
                "pass" if and_lit_ok else "fail",
                "is_and_lit: full clue maps to answer in DB"
                if and_lit_ok else
                "is_and_lit but full clue not in DB"))

        verdict = "PASS" if all(c.status == "pass" for c in checks) \
            else "FAIL"
        return Verdict(verdict=verdict, checks=checks,
                       assembly_value=assembly_value)

    # --- Per-node bridge / mechanism --------------------------------------

    def _check_node(self, path: str, node: Node,
                    checks: list[Check], is_root: bool) -> None:
        if isinstance(node, Leaf):
            self._check_leaf(path, node, checks)
            return

        # Op node — apply rule 2 (bridge for indicator) / rule 3 (mechanism)
        op = node.operation
        if op in INDICATOR_EXEMPT_OPS:
            return  # Charade / DD don't require an indicator

        check_name = (f"mechanism[{path}]" if is_root
                      else f"bridge.indicator[{path}]")

        if not node.indicator:
            checks.append(Check(
                check_name, "fail",
                f'{op} node has no indicator (rule {"3" if is_root else "2"})'))
            return

        if self._indicator_known(node.indicator, op):
            checks.append(Check(
                check_name, "pass",
                f'"{node.indicator}" is a known {op} indicator'))
        else:
            checks.append(Check(
                check_name, "fail",
                f'"{node.indicator}" is not a known {op} indicator '
                f'in the DB',
                enrichment_proposal={
                    "type": "indicator", "word": node.indicator,
                    "letters": op, "op_type": op,
                }))

    def _indicator_known(self, indicator: str, op: str) -> bool:
        # Try the full multi-word indicator first
        if self.is_indicator(indicator, op):
            return True
        # Else try each non-link word in the phrase
        for w in re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?",
                            indicator.lower()):
            if w in LINK_WORDS:
                continue
            if self.is_indicator(w, op):
                return True
        return False

    def _check_leaf(self, path: str, leaf: Leaf,
                    checks: list[Check]) -> None:
        op = leaf.operation
        if op == "literal":
            norm_src = re.sub(r"[^A-Z]", "", leaf.source_word.upper())
            norm_val = re.sub(r"[^A-Z]", "", leaf.value.upper())
            if norm_src != norm_val:
                checks.append(Check(
                    f"bridge.literal[{path}]", "fail",
                    f'"{leaf.source_word}" letters {norm_src!r} '
                    f'!= leaf value {norm_val!r}'))
            else:
                checks.append(Check(
                    f"bridge.literal[{path}]", "pass",
                    f'"{leaf.source_word}" -> {norm_val}'))
            return
        if op == "synonym":
            ok = self.is_synonym(leaf.source_word, leaf.value)
            checks.append(Check(
                f"bridge.synonym[{path}]",
                "pass" if ok else "fail",
                f'"{leaf.source_word}" -> {leaf.value}: '
                f'{"in DB" if ok else "not in DB"}',
                enrichment_proposal=None if ok else {
                    "type": "synonym",
                    "word": leaf.source_word, "letters": leaf.value,
                }))
            return
        if op == "abbreviation":
            ok = self.is_abbreviation(leaf.source_word, leaf.value)
            checks.append(Check(
                f"bridge.abbreviation[{path}]",
                "pass" if ok else "fail",
                f'"{leaf.source_word}" -> {leaf.value}: '
                f'{"in DB" if ok else "not in DB"}',
                enrichment_proposal=None if ok else {
                    "type": "abbreviation",
                    "word": leaf.source_word, "letters": leaf.value,
                }))
            return
        if op == "positional":
            mech_ok, detail = check_positional(leaf)
            checks.append(Check(
                f"bridge.positional[{path}]",
                "pass" if mech_ok else "fail",
                detail))
            # Positional can also carry an indicator (the clue word that
            # licensed the extraction). When present, verify it.
            if leaf.positional_indicator:
                if self._indicator_known(leaf.positional_indicator,
                                         "positional"):
                    checks.append(Check(
                        f"bridge.positional_indicator[{path}]", "pass",
                        f'"{leaf.positional_indicator}" is a known '
                        f'positional indicator'))
                else:
                    checks.append(Check(
                        f"bridge.positional_indicator[{path}]", "fail",
                        f'"{leaf.positional_indicator}" is not a known '
                        f'positional indicator in the DB',
                        enrichment_proposal={
                            "type": "indicator",
                            "word": leaf.positional_indicator,
                            "letters": "positional",
                            "op_type": "positional",
                        }))

    # --- Residue ----------------------------------------------------------

    def _check_residue(self, form: Form,
                       clue_text: str) -> list[Check]:
        out: list[Check] = []
        # Surface words from the clue (strip enumeration)
        clue = re.sub(r"\s*\([\d,\-\s/]+\)\s*$", "", clue_text or "")
        surface = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", clue.lower())

        # Form-claimed words: leaf source_words + op indicators +
        # positional indicators + definition.phrase
        claimed_tokens: list[str] = []
        for tok in form_claimed_words(form):
            for w in re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", tok.lower()):
                claimed_tokens.append(_normalise(w))
        claimed_counts = Counter(claimed_tokens)

        # Form's link_words (also subtracted from surface)
        link_normed = [_normalise(w.lower()) for w in form.link_words]
        link_counts = Counter(link_normed)

        # First sub-check: form.link_words subset of global LINK_WORDS
        bad_links = [w for w in form.link_words
                     if w.lower() not in LINK_WORDS]
        if bad_links:
            out.append(Check(
                "residue.link_words_subset", "fail",
                f"form.link_words contains words not in LINK_WORDS "
                f"allow-list: {bad_links}"))
        else:
            out.append(Check(
                "residue.link_words_subset", "pass",
                f"form.link_words subset of LINK_WORDS allow-list "
                f"({len(form.link_words)} words)"))

        # Second sub-check: every surface word is either claimed or in
        # form.link_words
        unaccounted: list[str] = []
        accounted = 0
        link_used = 0
        for w in surface:
            nw = _normalise(w)
            if claimed_counts[nw] > 0:
                claimed_counts[nw] -= 1
                accounted += 1
            elif link_counts[nw] > 0:
                link_counts[nw] -= 1
                link_used += 1
            else:
                unaccounted.append(w)

        if unaccounted:
            out.append(Check(
                "residue", "fail",
                f"{len(unaccounted)}/{len(surface)} unaccounted: "
                + ", ".join(unaccounted)))
        else:
            out.append(Check(
                "residue", "pass",
                f"all {len(surface)} clue words accounted "
                f"({accounted} claimed by form, "
                f"{link_used} as form.link_words)"))
        return out


# --- Assembly walker -------------------------------------------------------

class AssemblyError(Exception):
    pass


def compute_assembly(node: Node, target_answer: str) -> str:
    """Walk the tree bottom-up. Returns the produced letters.

    The walker is target-aware: charades slice the target by predicted
    child letter counts and pass each slice to the corresponding child,
    so nested anagrams can resolve to their slice of the answer.
    """
    target_clean = re.sub(r"[^A-Z]", "", target_answer.upper())

    def _walk(n: Node, target: str) -> str:
        if isinstance(n, Leaf):
            return n.value.upper().replace(" ", "")

        op = n.operation
        if op == "charade":
            results: list = []
            offset = 0
            for child in n.sources:
                child_count = _predicted_letter_count(child)
                if child_count is None:
                    # Unknown — walk with remainder of target
                    child_target = target[offset:]
                    child_result = _walk(child, child_target)
                    results.append(child_result)
                    offset += len(child_result)
                else:
                    child_target = target[offset:offset + child_count]
                    child_result = _walk(child, child_target)
                    results.append(child_result)
                    offset += child_count
            return "".join(results)

        # Children walked with full target by default; ops below override
        # if needed.
        children = [_walk(c, target) for c in n.sources]

        if op == "container":
            if len(children) != 2:
                raise AssemblyError("container needs exactly 2 children")
            outer, inner = children
            for pos in range(1, len(outer)):
                cand = outer[:pos] + inner + outer[pos:]
                if cand.upper() == target.upper():
                    return cand
            mid = len(outer) // 2
            return outer[:mid] + inner + outer[mid:]

        if op == "anagram":
            # Pool letters from all leaves under this node (anagram fodder).
            pool = _collect_leaf_letters(n)
            if sorted(pool) == sorted(target):
                return target
            # Fallback: return the pool concatenation
            return pool

        if op == "reversal":
            if len(children) != 1:
                raise AssemblyError("reversal needs exactly 1 child")
            return children[0][::-1]

        if op == "deletion":
            if len(children) != 1:
                raise AssemblyError("deletion needs exactly 1 child")
            src = children[0]
            kind = n.deletion_kind
            if not src:
                return ""
            if kind == "head":
                return src[1:]
            if kind == "tail":
                return src[:-1]
            if kind == "outer":
                return src[1:-1]
            if kind == "heart":
                if len(src) % 2:
                    return src[: len(src) // 2] + src[len(src) // 2 + 1:]
                return (src[: len(src) // 2] +
                        src[(len(src) + 1) // 2 + 1:])
            return src[:-1]

        if op == "hidden":
            joined = "".join(children)
            if target_clean in joined:
                return target_clean
            return joined

        if op == "double_definition":
            for c in children:
                if c.upper() == target_clean:
                    return target_clean
            return children[0] if children else ""

        if op == "acrostic":
            kind = n.acrostic_kind or "first"
            out = []
            for child in n.sources:
                if not isinstance(child, Leaf):
                    raise AssemblyError(
                        "acrostic children must be leaves")
                src = re.sub(r"[^A-Z]", "",
                             (child.source_word or "").upper())
                if not src:
                    raise AssemblyError(
                        f'acrostic child has empty source_word')
                if kind == "first":
                    out.append(src[0])
                elif kind == "last":
                    out.append(src[-1])
                elif kind == "middle":
                    out.append(src[len(src) // 2])
                else:
                    raise AssemblyError(
                        f'unknown acrostic_kind: {kind}')
            return "".join(out)

        if op == "unknown":
            return "".join(children)

        raise AssemblyError(f"unknown op: {op}")

    return _walk(node, target_clean)


def _predicted_letter_count(node: Node) -> Optional[int]:
    """Predict the number of letters this node will produce. Used by the
    charade walker to slice the target before recursing.

    Returns None when the count can't be predicted (hidden, double
    definition); callers should fall back to non-sliced walking.
    """
    if isinstance(node, Leaf):
        return len(re.sub(r"[^A-Z]", "", node.value.upper()))
    if not isinstance(node, Op):
        return None
    op = node.operation
    if op in ("charade", "container", "reversal"):
        counts = [_predicted_letter_count(c) for c in node.sources]
        if any(c is None for c in counts):
            return None
        return sum(counts)
    if op == "anagram":
        # Anagram preserves total letter count from its children.
        counts = [_predicted_letter_count(c) for c in node.sources]
        if any(c is None for c in counts):
            return None
        return sum(counts)
    if op == "deletion":
        kind = node.deletion_kind
        counts = [_predicted_letter_count(c) for c in node.sources]
        if any(c is None for c in counts):
            return None
        total = sum(counts)
        if kind in ("head", "tail", "heart"):
            return total - 1
        if kind == "outer":
            return total - 2
        return total - 1   # default: drop one letter
    if op == "acrostic":
        # One letter per child (default kind=first/last/middle).
        return len(node.sources)
    if op in ("hidden", "double_definition"):
        return None
    if op == "unknown":
        return None
    return None


def _collect_leaf_letters(node: Node) -> str:
    """Concatenate the letters from every Leaf under this node."""
    if isinstance(node, Leaf):
        return re.sub(r"[^A-Z]", "", node.value.upper())
    return "".join(_collect_leaf_letters(c) for c in node.sources)


# --- Positional check ------------------------------------------------------

def check_positional(leaf: Leaf) -> tuple[bool, str]:
    """Mechanically verify a positional extraction."""
    src = re.sub(r"[^A-Za-z]", "", leaf.source_word).upper()
    L = leaf.value.upper()
    n = len(L)
    kind = leaf.positional_kind
    if not src or not L:
        return False, "empty source or value"
    if kind == "first":
        ok = src.startswith(L)
    elif kind == "last":
        ok = src.endswith(L)
    elif kind == "middle":
        if len(src) < n:
            ok = False
        else:
            start = (len(src) - n) // 2
            ok = src[start:start + n] == L
    elif kind == "outer":
        if n == 2:
            ok = (L[0] == src[0] and L[-1] == src[-1])
        else:
            half = n // 2
            ok = L == src[:half] + src[-(n - half):]
    elif kind == "initial":
        ok = (n == 1 and src.startswith(L))
    elif kind == "final":
        ok = (n == 1 and src.endswith(L))
    elif kind == "odd":
        ok = "".join(src[i] for i in range(0, len(src), 2)) == L
    elif kind == "even":
        ok = "".join(src[i] for i in range(1, len(src), 2)) == L
    elif kind == "alternate":
        odd = "".join(src[i] for i in range(0, len(src), 2))
        even = "".join(src[i] for i in range(1, len(src), 2))
        ok = L in (odd, even)
    else:
        return False, f'unknown positional kind {kind!r}'
    return ok, (f'{leaf.value} as {kind} of "{leaf.source_word}": '
                f'{"YES" if ok else "NO"}')


# --- Tree helpers ----------------------------------------------------------

def walk(node: Node, prefix: str = "0") -> list[tuple[str, Node]]:
    """Pre-order walk of every node, yielding (path, node)."""
    out = [(prefix, node)]
    if isinstance(node, Op):
        for i, c in enumerate(node.sources):
            out.extend(walk(c, f"{prefix}.{i}"))
    return out


def form_claimed_words(form: Form) -> list[str]:
    """Every clue word the form claims via leaf source_word, op indicator,
    positional indicator, or definition phrase."""
    out: list[str] = [form.definition.phrase]
    for _, n in walk(form.tree):
        if isinstance(n, Leaf):
            out.append(n.source_word)
            if n.positional_indicator:
                out.append(n.positional_indicator)
        else:
            if n.indicator:
                out.append(n.indicator)
    return out


def _normalise(w: str) -> str:
    return w[:-2] if w.endswith("'s") else w


# --- CLI / smoke test ------------------------------------------------------

if __name__ == "__main__":
    from .examples import ALL_EXAMPLES
    EXAMPLE_CLUES = {
        "RIDE": "Republican fish",
        "REPAST": "Old man, among others, gets a meal",
        "REVOLTING": "Lover and nit, good back",
        "NEAPOLITAN": "Cook Antonio, mostly pale native of southern Italy",
        "FEAST": "Twist of fate's causing blow-out",
        "TAR": "Pitch that a rainstorm covers",
        "SULKY (DD)": "Light carriage being put out",
        "VINDICATION": "Start of victory sign for defence",
    }
    verifier = FormVerifier()
    try:
        for name, form in ALL_EXAMPLES.items():
            clue = EXAMPLE_CLUES.get(name, "")
            v = verifier.verify(form, clue)
            print("=" * 78)
            print(f"  {name}  ({form.definition.answer})    {v.verdict}")
            print(f"  clue: {clue}")
            for c in v.checks:
                mark = "+" if c.status == "pass" else "X"
                print(f"    [{mark}] {c.name}: {c.detail}")
            print()
    finally:
        verifier.close()
