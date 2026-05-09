"""sig_solver -> universal Form adapter.

The signature solver internally knows everything we need to build a form:
  - entry.operation              the operation type ("anagram", "container_charade", ...)
  - entry.pattern                slot ordering (e.g. ('F', 'I', 'F'))
  - assignment.indicator_indices map of indicator-token -> word index/phrase
  - assignment.lnk_indices       set of word indices used as link words
  - executor.execute_signature   returns pieces = list of (word_text, token, value)

This adapter re-implements sig_solver's outer match-and-process loop so it
has direct access to (entry, assignment), then builds a Form natively for
each successful match. No production code is modified — this file is
read-only against `signature_solver/` internals.

Design choices:
- For each definition candidate, try base_matcher / positional_matcher /
  full catalog matcher in order (same priority as solver.solve()).
- For each (entry, assignment) match, build a Form via build_form_from_match.
- Verify the Form. Return on first PASS; otherwise keep best.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from signature_solver.db import RefDB
from signature_solver.word_analyzer import analyze_phrases
from signature_solver.base_matcher import match_base
from signature_solver.matcher import match_signatures
from signature_solver.positional_matcher import match_positional
from signature_solver.base_catalog import (
    BASE_CATALOG, OPERATION_INDICATOR_TYPE,
)
from signature_solver.catalog import CATALOG
from signature_solver.positional_catalog import POSITIONAL_CATALOG
from signature_solver.solver import (
    extract_definition_candidates, _normalize_clue,
)
from signature_solver import executor
from signature_solver.tokens import (
    SYN_F, ABR_F, ANA_F, RAW, HID_F, HOM_F, POS_F, DEL_F,
    ANA_I, REV_I, CON_I, DEL_I, HID_I, HOM_I,
    POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
    POS_I_ALTERNATE, POS_I_TRIM_FIRST, POS_I_TRIM_LAST,
    POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER, POS_I_HALF,
)

from backfill_ai_exp.backfill_dd_hidden import (
    build_graph as _build_dd_graph,
    generate_dd_hypotheses,
    try_hidden,
)

from .schema import (
    Form, Definition, Leaf, Op, Node, UNKNOWN_OPERATION,
)
from .verifier import FormVerifier, Verdict, LINK_WORDS as _LINK_WORDS_SET


# Surface words used as hidden indicators. Used to attach the indicator
# to a hidden Op when try_hidden returns a hit (try_hidden doesn't
# identify the indicator itself).
_HIDDEN_INDICATOR_WORDS = {
    "in", "inside", "within", "amid", "amidst", "among", "amongst",
    "during", "across", "covers", "covering", "covered", "concealed",
    "hides", "hiding", "hidden", "harbours", "harbours", "encases",
    "contains", "containing", "of", "from", "with",
    "swallowing", "spans", "buried", "buried in", "lurking",
    "secreted", "secreted in", "trapped", "smothering", "absorbing",
    "houses", "houses some", "produced", "produces",
}
_HIDDEN_INDICATOR_PHRASES = [
    "hidden in", "concealed in", "buried in", "lurking in",
    "trapped in", "found in", "spans some of",
]


# Map sig_solver positional indicator tokens -> our positional kind names
POSITIONAL_KIND_FROM_TOKEN = {
    POS_I_FIRST:    "first",
    POS_I_LAST:     "last",
    POS_I_OUTER:    "outer",
    POS_I_MIDDLE:   "middle",
    POS_I_ALTERNATE: "alternate",
}

# Map sig_solver TRIM positional indicator tokens to deletion-kind
DELETION_KIND_FROM_TRIM_TOKEN = {
    POS_I_TRIM_FIRST:  "head",
    POS_I_TRIM_LAST:   "tail",
    POS_I_TRIM_OUTER:  "outer",
    POS_I_TRIM_MIDDLE: "heart",
}


# --- Result dataclass ------------------------------------------------------

@dataclass
class AdapterResult:
    form: Optional[Form]
    verdict: Optional[Verdict]
    operation: Optional[str]   # the catalog entry's operation name
    notes: list                # any translator-style notes


# --- Top-level entry -------------------------------------------------------

# Cached DD/hidden graph (built once per RefDB)
_DD_GRAPH = None


def _get_dd_graph(db: RefDB):
    global _DD_GRAPH
    if _DD_GRAPH is None:
        _DD_GRAPH = _build_dd_graph(db)
    return _DD_GRAPH


def solve_to_form(clue_text: str, answer: str, db: RefDB,
                  verifier: Optional[FormVerifier] = None,
                  ) -> AdapterResult:
    """Solve clue: try DD engine, then hidden engine, then sig_solver.
    Build Form for each successful match, verify, return first PASS or
    best partial.
    """
    own_verifier = verifier is None
    if own_verifier:
        verifier = FormVerifier()

    notes: list = []
    clue_words = _normalize_clue(clue_text).strip().split()
    answer_clean = answer.upper().replace(" ", "").replace("-", "")
    answer_for_form = answer.upper().strip()

    best_form: Optional[Form] = None
    best_verdict: Optional[Verdict] = None
    best_op: Optional[str] = None

    try:
        graph = _get_dd_graph(db)

        # 1. DD pre-pass
        dd_hit = generate_dd_hypotheses(
            clue_text, graph, total_len=len(answer_clean), answer=answer_clean)
        if dd_hit:
            form = _build_dd_form(dd_hit, answer_for_form)
            verdict = verifier.verify(form, clue_text)
            if verdict.verdict == "PASS":
                return AdapterResult(
                    form=form, verdict=verdict,
                    operation="double_definition", notes=notes)
            best_form, best_verdict, best_op = form, verdict, "double_definition"

        # 2. Hidden pre-pass
        hidden_hit = try_hidden(clue_text, answer_clean, graph=graph)
        if hidden_hit:
            form = _build_hidden_form(
                hidden_hit, clue_text, answer_for_form, clue_words)
            verdict = verifier.verify(form, clue_text)
            if verdict.verdict == "PASS":
                return AdapterResult(
                    form=form, verdict=verdict,
                    operation=("hidden_reversed"
                               if hidden_hit.get("direction") == "rev"
                               else "hidden"),
                    notes=notes)
            if best_verdict is None or _better(verdict, best_verdict):
                best_form, best_verdict = form, verdict
                best_op = ("hidden_reversed"
                           if hidden_hit.get("direction") == "rev"
                           else "hidden")

        # 3. sig_solver catalog matchers
        candidates = extract_definition_candidates(
            clue_words, answer_clean, db)
        if not candidates:
            candidates = [("", clue_words)]

        for def_phrase, wp_words in candidates:
            analyses, phrases = analyze_phrases(wp_words, answer_clean, db)
            for matcher_fn, cat in [
                (match_base, BASE_CATALOG),
                (match_positional, POSITIONAL_CATALOG),
                (match_signatures, CATALOG),
            ]:
                for entry, assignment in matcher_fn(
                        wp_words, analyses, phrases, cat,
                        answer_clean, db):
                    success, _expl, pieces = executor.execute_signature(
                        entry, assignment, wp_words, answer_clean)
                    if not success:
                        continue
                    form = build_form_from_match(
                        entry, assignment, pieces, wp_words,
                        def_phrase, answer_for_form, notes)
                    if form is None:
                        continue
                    verdict = verifier.verify(form, clue_text)
                    if verdict.verdict == "PASS":
                        return AdapterResult(
                            form=form, verdict=verdict,
                            operation=entry.operation, notes=notes)
                    if best_verdict is None or _better(verdict, best_verdict):
                        best_form = form
                        best_verdict = verdict
                        best_op = entry.operation

        return AdapterResult(form=best_form, verdict=best_verdict,
                             operation=best_op, notes=notes)
    finally:
        if own_verifier:
            verifier.close()


# --- Form construction for DD and hidden hits -----------------------------

def _build_dd_form(dd_hit: dict, answer: str) -> Form:
    """Build a Form for a DD hit from generate_dd_hypotheses output."""
    left = dd_hit["left_def"]
    right = dd_hit["right_def"]
    answer_clean = re.sub(r"[^A-Z]", "", answer.upper())
    tree = Op("double_definition", indicator=None, sources=[
        Leaf("synonym", left, answer_clean),
        Leaf("synonym", right, answer_clean),
    ])
    return Form(
        tree=tree,
        definition=Definition(phrase=left, answer=answer_clean),
        link_words=[],
    )


def _build_hidden_form(hidden_hit: dict, clue_text: str, answer: str,
                       clue_words: list) -> Form:
    """Build a Form for a hidden hit from try_hidden output."""
    span_words = hidden_hit.get("words") or []
    direction = hidden_hit.get("direction")
    def_phrase = hidden_hit.get("definition") or ""
    answer_clean = re.sub(r"[^A-Z]", "", answer.upper())

    # Hidden inner: literal of each spanning word
    leaves = [Leaf("literal",
                   w, re.sub(r"[^A-Z]", "", w.upper()))
              for w in span_words]
    if not leaves:
        leaves = [Leaf("literal", w, re.sub(r"[^A-Z]", "", w.upper()))
                  for w in clue_words]

    # Find a hidden indicator in the surface (clue words minus span and def)
    surface_lower = clue_text.lower()
    span_set = {w.lower() for w in span_words}
    def_set = {w.lower() for w in (def_phrase.split() if def_phrase else [])}
    indicator: Optional[str] = None
    for phrase in _HIDDEN_INDICATOR_PHRASES:
        if phrase in surface_lower:
            phrase_words = re.findall(r"[a-zA-Z]+", phrase)
            if not any(w in span_set or w in def_set
                       for w in phrase_words):
                indicator = phrase
                break
    if indicator is None:
        for w in re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", surface_lower):
            if w in span_set or w in def_set:
                continue
            if w in _HIDDEN_INDICATOR_WORDS:
                indicator = w
                break

    inner = Op("hidden", indicator=indicator, sources=leaves)
    tree = inner
    if direction == "rev":
        # Hidden reversed: wrap with reversal
        tree = Op("reversal", indicator=None, sources=[inner])

    # Try to find a reversal indicator from surface for the hidden_reversed
    # case
    if direction == "rev":
        for w in re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", surface_lower):
            if w in span_set or w in def_set:
                continue
            if w in {"back", "backward", "backwards", "reversed", "up",
                     "round", "over", "returning", "returns", "rises"}:
                tree = Op("reversal", indicator=w, sources=[inner])
                break

    return Form(
        tree=tree,
        definition=Definition(phrase=def_phrase, answer=answer_clean),
        link_words=[],
    )


def _better(v1: Verdict, v2: Verdict) -> bool:
    """v1 strictly better than v2 (more checks passed)."""
    p1 = sum(1 for c in v1.checks if c.status == "pass")
    p2 = sum(1 for c in v2.checks if c.status == "pass")
    return p1 > p2


# --- Form construction from (entry, assignment, pieces) -------------------

def build_form_from_match(entry, assignment, pieces, wp_words,
                          def_phrase: str, answer: str,
                          notes: list) -> Optional[Form]:
    """Build a universal Form from a sig_solver match."""
    op = entry.operation

    # Resolve indicator words from assignment.indicator_indices
    indicators: dict = {}
    for ind_tok, ind_idx in (assignment.get("indicator_indices") or {}).items():
        word = _index_to_word(wp_words, ind_idx)
        if word:
            indicators[ind_tok] = word

    # Build per-piece nodes
    nodes: list = []
    for word_text, tok, val in pieces:
        node = _piece_to_node(word_text, tok, val, indicators)
        if node is None:
            notes.append(f"unhandled_piece_token:{tok}")
            return None
        nodes.append(node)

    # Build the operation tree
    tree = _build_op_tree(op, nodes, indicators, pieces, notes)
    if tree is None:
        return None

    # Determine link words from assignment.lnk_indices
    lnk_idxs = assignment.get("lnk_indices") or set()
    link_words = [wp_words[i] for i in lnk_idxs
                  if 0 <= i < len(wp_words)]

    return Form(
        tree=tree,
        definition=Definition(
            phrase=def_phrase or "",
            answer=re.sub(r"[^A-Z]", "", answer.upper())),
        link_words=link_words,
    )


def _index_to_word(wp_words: list, idx) -> Optional[str]:
    if isinstance(idx, tuple):
        a, b = idx
        return " ".join(wp_words[k] for k in range(a, b))
    if isinstance(idx, int) and 0 <= idx < len(wp_words):
        return wp_words[idx]
    return None


# --- Per-piece node mapping -----------------------------------------------

def _piece_to_node(word_text: str, tok: str, val,
                   indicators: dict) -> Optional[Node]:
    """Convert one (word_text, token, value) into a Leaf or small Op.

    For ANA_F (anagram fodder), the piece's value is the rearranged letters
    of the source word. We emit a literal Leaf of the source word here;
    the parent anagram Op will rearrange. For POS_F (positional source),
    we emit a positional Leaf with the kind derived from the indicator.
    """
    if val is None:
        # Indicator/link slots — shouldn't reach here via pieces but guard
        return None
    val_str = str(val).upper().replace(" ", "")
    if tok == SYN_F:
        return Leaf("synonym", word_text, val_str)
    if tok == ABR_F:
        return Leaf("abbreviation", word_text, val_str)
    if tok == RAW:
        # Literal: source-word letters as-is
        norm = re.sub(r"[^A-Z]", "", word_text.upper())
        return Leaf("literal", word_text, norm or val_str)
    if tok == ANA_F:
        # Source word's literal letters; the parent anagram op rearranges.
        norm = re.sub(r"[^A-Z]", "", word_text.upper())
        return Leaf("literal", word_text, norm or val_str)
    if tok == HID_F:
        # Hidden span piece: literal of the source word
        norm = re.sub(r"[^A-Z]", "", word_text.upper())
        return Leaf("literal", word_text, norm or val_str)
    if tok == POS_F:
        # Positional extraction. Find which positional indicator token
        # is in `indicators` to set the kind.
        kind = None
        ind_word: Optional[str] = None
        for ptok, kind_name in POSITIONAL_KIND_FROM_TOKEN.items():
            if ptok in indicators:
                kind = kind_name
                ind_word = indicators[ptok]
                break
        # TRIM_* tokens map to deletion (handled at op level — see below)
        if kind is None:
            for ptok in DELETION_KIND_FROM_TRIM_TOKEN:
                if ptok in indicators:
                    # Treat as positional with derived kind for the leaf
                    # (the op-tree builder may wrap in deletion separately).
                    kind = "first"  # placeholder
                    ind_word = indicators[ptok]
                    break
        if kind is None:
            kind = "first"
        return Leaf("positional", word_text, val_str,
                    positional_kind=kind, positional_indicator=ind_word)
    if tok == HOM_F:
        # Homophone fodder — represent as literal for now (homophone not
        # in basic 12; will become unknown_op at parent level).
        norm = re.sub(r"[^A-Z]", "", word_text.upper())
        return Leaf("literal", word_text, norm or val_str)
    if tok == DEL_F:
        # Deletion fodder (the part being removed) — literal of word text
        norm = re.sub(r"[^A-Z]", "", word_text.upper())
        return Leaf("literal", word_text, norm or val_str)
    return None


# --- Op tree construction by operation -------------------------------------

def _build_op_tree(operation: str, leaves: list, indicators: dict,
                   pieces: list, notes: list) -> Optional[Node]:
    """Wrap leaves in the appropriate Op nodes for the operation type."""

    # Single-leaf trees
    if not leaves:
        return _unknown(operation, [])

    # --- Pure operations ---------------------------------------------------

    if operation == "charade":
        return Op("charade", indicator=None, sources=leaves) \
            if len(leaves) > 1 else leaves[0]

    if operation == "synonym":
        # Single synonym piece
        return leaves[0] if len(leaves) == 1 else \
            Op("charade", indicator=None, sources=leaves)

    if operation == "anagram":
        ind = indicators.get(ANA_I)
        if len(leaves) == 1:
            return Op("anagram", indicator=ind, sources=leaves)
        return Op("anagram", indicator=ind, sources=leaves)

    if operation == "container":
        if len(leaves) != 2:
            notes.append(f"container_arity:{len(leaves)}")
            return _unknown(operation, leaves)
        outer, inner = leaves[0], leaves[1]
        return Op("container", indicator=indicators.get(CON_I),
                  sources=[outer, inner])

    if operation == "reversal":
        if len(leaves) != 1:
            return _unknown(operation, leaves)
        return Op("reversal", indicator=indicators.get(REV_I),
                  sources=leaves)

    if operation == "hidden":
        return Op("hidden", indicator=indicators.get(HID_I),
                  sources=leaves)

    if operation == "deletion":
        if len(leaves) >= 1:
            return Op("deletion", indicator=indicators.get(DEL_I),
                      sources=[leaves[0]], deletion_kind="tail")
        return _unknown(operation, leaves)

    if operation in ("trim", "trim_charade"):
        # Trim is essentially deletion of a positional slice; map to
        # deletion for the simplest cases.
        kind = "tail"
        for tok, k in DELETION_KIND_FROM_TRIM_TOKEN.items():
            if tok in indicators:
                kind = k
                break
        if operation == "trim" and len(leaves) == 1:
            return Op("deletion", indicator=indicators.get(DEL_I),
                      sources=[leaves[0]], deletion_kind=kind)
        # trim_charade: deletion applies to one piece, charade combines all.
        # Apply deletion to the first leaf, leave others as-is.
        if len(leaves) >= 2:
            del_node = Op("deletion", indicator=indicators.get(DEL_I),
                          sources=[leaves[0]], deletion_kind=kind)
            return Op("charade", indicator=None,
                      sources=[del_node] + leaves[1:])
        return _unknown(operation, leaves)

    if operation == "acrostic":
        return Op("acrostic", indicator=indicators.get(POS_I_FIRST),
                  sources=leaves, acrostic_kind="first")

    if operation == "alternate":
        return Op("acrostic",
                  indicator=indicators.get(POS_I_ALTERNATE),
                  sources=leaves, acrostic_kind="first")  # approximate

    # --- Compound operations ----------------------------------------------

    if operation == "anagram_charade":
        # Some leaves come from ANA_F pieces (need anagram wrapper),
        # others come from SYN_F/ABR_F (used directly). Use the original
        # `pieces` list to know which are ANA_F.
        ana_indicator = indicators.get(ANA_I)
        new_children: list = []
        for leaf, (_, tok, _) in zip(leaves, pieces):
            if tok == ANA_F:
                new_children.append(
                    Op("anagram", indicator=ana_indicator,
                       sources=[leaf]))
            else:
                new_children.append(leaf)
        return Op("charade", indicator=None,
                  sources=new_children) \
            if len(new_children) > 1 else new_children[0]

    if operation == "anagram_plus":
        # Same as anagram_charade in shape
        return _build_op_tree("anagram_charade", leaves, indicators,
                              pieces, notes)

    if operation == "anagram_container":
        # Anagram inside a container, or a container of anagram-result
        ana_indicator = indicators.get(ANA_I)
        con_indicator = indicators.get(CON_I)
        # First piece is typically the anagram fodder; rest are container args
        new_children: list = []
        for leaf, (_, tok, _) in zip(leaves, pieces):
            if tok == ANA_F:
                new_children.append(
                    Op("anagram", indicator=ana_indicator,
                       sources=[leaf]))
            else:
                new_children.append(leaf)
        # Build container: first child outer, second inner (heuristic)
        if len(new_children) == 2:
            return Op("container", indicator=con_indicator,
                      sources=new_children)
        notes.append(f"anagram_container_arity:{len(new_children)}")
        return _unknown(operation, leaves)

    if operation == "container_charade":
        # Container where one of outer/inner is itself a charade.
        # For 3+ pieces: pick which goes inside vs outside (use letter-fit).
        # For now, group as: leaves[0] outer = charade(*leaves[1:-1]) or similar.
        # Simple heuristic: 3 pieces -> container(charade(p1,p2), p3).
        con_indicator = indicators.get(CON_I)
        if len(leaves) == 3:
            outer = Op("charade", indicator=None, sources=leaves[:2])
            return Op("container", indicator=con_indicator,
                      sources=[outer, leaves[2]])
        if len(leaves) == 2:
            return Op("container", indicator=con_indicator,
                      sources=leaves)
        notes.append(f"container_charade_arity:{len(leaves)}")
        return _unknown(operation, leaves)

    if operation == "container_positional":
        con_indicator = indicators.get(CON_I)
        if len(leaves) == 2:
            return Op("container", indicator=con_indicator,
                      sources=leaves)
        notes.append(f"container_positional_arity:{len(leaves)}")
        return _unknown(operation, leaves)

    if operation == "container_reversal":
        # Container of (reversed something). reversal_indicator on the
        # reversal op, container_indicator on the container op.
        rev_indicator = indicators.get(REV_I)
        con_indicator = indicators.get(CON_I)
        if len(leaves) == 2:
            inner_rev = Op("reversal", indicator=rev_indicator,
                           sources=[leaves[1]])
            return Op("container", indicator=con_indicator,
                      sources=[leaves[0], inner_rev])
        notes.append(f"container_reversal_arity:{len(leaves)}")
        return _unknown(operation, leaves)

    if operation == "reversal_charade":
        rev_indicator = indicators.get(REV_I)
        if len(leaves) == 1:
            return Op("reversal", indicator=rev_indicator, sources=leaves)
        inner_charade = Op("charade", indicator=None, sources=leaves)
        return Op("reversal", indicator=rev_indicator,
                  sources=[inner_charade])

    if operation == "hidden_reversed":
        # Hidden then reversed: reversal(hidden(*leaves))
        inner_hid = Op("hidden", indicator=indicators.get(HID_I),
                       sources=leaves)
        return Op("reversal", indicator=indicators.get(REV_I),
                  sources=[inner_hid])

    if operation == "homophone":
        # Not in basic 12 — emit unknown so the prototype surfaces it
        notes.append("homophone_not_in_basic_vocab")
        return _unknown(operation, leaves)

    if operation == "positional_charade":
        # A charade of positional leaves and possibly other pieces.
        return Op("charade", indicator=None, sources=leaves) \
            if len(leaves) > 1 else leaves[0]

    # Fallback: unknown operation
    notes.append(f"unhandled_operation:{operation}")
    return _unknown(operation, leaves)


def _unknown(flat_op: str, sources: list) -> Op:
    return Op(UNKNOWN_OPERATION, indicator=None,
              sources=list(sources), flat_op=flat_op)


# --- Batch runner ----------------------------------------------------------

def solve_puzzle(source: str, puzzle_number: str,
                 limit: Optional[int] = None) -> list:
    """Run the adapter on every clue of a puzzle. Read-only.

    Returns list of (clue_row_dict, AdapterResult).
    """
    import sqlite3
    conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, clue_number, direction, clue_text, answer, enumeration
        FROM clues
        WHERE source = ? AND puzzle_number = ?
          AND answer IS NOT NULL AND answer != ''
        ORDER BY direction, CAST(clue_number AS INTEGER)
    """, (source, str(puzzle_number))).fetchall()
    if limit:
        rows = rows[:limit]

    db = RefDB(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    verifier = FormVerifier()
    results: list = []
    try:
        for r in rows:
            try:
                ar = solve_to_form(r["clue_text"], r["answer"], db,
                                   verifier=verifier)
            except Exception as e:
                ar = AdapterResult(form=None, verdict=None,
                                   operation=None,
                                   notes=[f"exception: {e!r}"])
            results.append((dict(r), ar))
    finally:
        verifier.close()
        conn.close()
    return results


# --- CLI -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--puzzle", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print(f"sig_to_form adapter — source={args.source}, "
          f"puzzle={args.puzzle}")
    results = solve_puzzle(args.source, args.puzzle, limit=args.limit)
    n_pass = sum(1 for _, ar in results
                 if ar.verdict and ar.verdict.verdict == "PASS")
    n_fail = sum(1 for _, ar in results
                 if ar.verdict and ar.verdict.verdict == "FAIL")
    n_no_form = sum(1 for _, ar in results if ar.form is None)
    for r, ar in results:
        v = (ar.verdict.verdict if ar.verdict else "NO_FORM")
        print(f"  {r['clue_number']:>3s}{r['direction'][:1]}  "
              f"{r['answer']:<16s}  {v:<8s}  op={ar.operation}  "
              f"notes={ar.notes}")
    print()
    print(f"PASS: {n_pass}  FAIL: {n_fail}  NO_FORM: {n_no_form} "
          f"(of {len(results)})")
