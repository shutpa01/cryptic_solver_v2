"""Cascade driver — solves a single clue against the catalog.

Per the parallel-system design (PARALLEL_SYSTEM_DESIGN.md), this is
the per-clue entry point. The per-puzzle runner (separate file) calls
this for each clue.

Algorithm
---------
For each catalog template (walked by frequency, highest first):
  1. Try a STANDARD split-form interpretation. If any form clipboard-
     verifier-PASSes, return ("PASS", form, []).
  2. (Phase 2 — only if no standard PASS across all templates.) Try
     each template as an &LIT interpretation: whole clue is both
     wordplay window and definition phrase. If any form verifies
     (verdict PENDING per the &lit rule), return ("PENDING", form, []).
  3. If neither finds a verified form, return ("FAIL", None,
     enrichment_candidates) — the union of all enrichment candidates
     emitted by failed verification attempts in pass 1, deduplicated.

This implements the user's "only way to solve" rule: &lit is only
attempted when no standard interpretation works.

Cryptic-definition handling: NOT auto-attempted. CDs are produced only
by humans in the leftover review tool, not by the cascade.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple

from signature_solver.db import RefDB

from .schema import Form
from .tree_matcher import match_signature
from .clipboard_verifier import verify, EnrichmentCandidate


@dataclass
class CascadeResult:
    """The outcome of solving a single clue."""
    verdict: str               # "PASS" | "PENDING" | "FAIL"
    form: Optional[Form]       # the verified form, or None on FAIL
    signature: Optional[str]   # the catalog entry id that produced it
    enrichment_candidates: List[EnrichmentCandidate]


def solve_clue_parallel(
    clue_id: int,
    clue_text: str,
    answer: str,
    db: RefDB,
    catalog_entries: list,
    shadow_conn: Optional[sqlite3.Connection] = None,
) -> CascadeResult:
    """Solve a single clue against the catalog.

    `catalog_entries` is a list of dicts with at least `id` and
    `structure`. Walked in the order given (caller decides — typically
    highest-frequency first).

    Returns a CascadeResult. On PASS / PENDING, `enrichment_candidates`
    is empty (the form verified on its own). On FAIL, it's the union
    of unique enrichment candidates emitted by failed verification
    attempts during the standard pass, so the human review tool can
    surface them as DB-row addition candidates.
    """
    from .surface import tokenize as _tokenize
    n_clue_words = len(_tokenize(clue_text))

    # Pre-filter: each entry has a minimum word requirement based on
    # leaves and indicator slots. Skip entries that obviously can't fit.
    eligible = [e for e in catalog_entries
                if _min_words(e["structure"]) <= n_clue_words]

    # Pass 1: standard split-form interpretations
    fail_enrichments = []
    seen_keys = set()
    for entry in eligible:
        forms = match_signature(entry, clue_text, answer, db, shadow_conn)
        for f in forms:
            v = verify(f, clue_text, db, shadow_conn)
            if v.verdict == "PASS":
                return CascadeResult(
                    verdict="PASS", form=f,
                    signature=entry.get("id"),
                    enrichment_candidates=[])
            # Accumulate dedup'd enrichment candidates from FAILs
            for c in v.enrichment_candidates:
                key = (c.kind, c.source_word, c.value,
                       c.operation, c.subtype)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                fail_enrichments.append(c)

    # Pass 2: &lit fallback. Restricted by clue length AND template
    # complexity — real &lit clues are short (2–5 words) and use
    # simple structures (≤ 3 leaves typically). Beyond that, the
    # synonym-expansion combinatorics blow up against the whole-clue
    # span with no realistic chance of a true &lit form.
    if n_clue_words <= 5:
        for entry in eligible:
            if _leaf_count(entry["structure"]) > 3:
                continue
            forms = match_signature(entry, clue_text, answer, db,
                                      shadow_conn, and_lit=True)
            for f in forms:
                v = verify(f, clue_text, db, shadow_conn)
                if v.verdict == "PENDING":
                    return CascadeResult(
                        verdict="PENDING", form=f,
                        signature=entry.get("id"),
                        enrichment_candidates=[])

    # All attempts failed
    return CascadeResult(
        verdict="FAIL", form=None, signature=None,
        enrichment_candidates=fail_enrichments)


def _leaf_count(structure: dict) -> int:
    """Count leaves in a catalog template structure."""
    if structure.get("leaf"):
        return 1
    return sum(_leaf_count(c) for c in structure.get("children", []))


def _min_words(structure: dict) -> int:
    """Minimum surface-word count this template requires.

    Each leaf takes ≥ 1 word; positional leaves take 2 (indicator +
    source). Each op-node that requires an indicator adds 1.
    Container needs 3 (outer + indicator + inner)."""
    op = structure.get("op")
    is_leaf = structure.get("leaf")
    if is_leaf:
        if op == "positional":
            return 2
        return 1
    children = structure.get("children", [])
    base = sum(_min_words(c) for c in children)
    # Ops requiring an indicator
    if op in ("anagram", "hidden", "deletion", "reversal", "acrostic",
              "homophone"):
        base += 1
    elif op == "container":
        base += 1  # the indicator (children already each ≥ 1)
    # charade, double_definition, cryptic_definition: no indicator
    return base


# --- Catalog helpers -----------------------------------------------------

def load_catalog_from_shadow(shadow_conn: sqlite3.Connection,
                              min_frequency: int = 1) -> list:
    """Build a catalog-entry list from shadow_db.solves, ordered by
    signature frequency descending. Used when running the cascade
    against the catalog the parallel system has accumulated.

    Only PASS rows contribute (PENDING and FAIL excluded — see the
    design rule that PENDING items don't add to the catalog until
    human-reviewed)."""
    import json
    rows = shadow_conn.execute("""
        SELECT signature, COUNT(*) AS freq, MIN(form_json) AS sample_form
        FROM solves
        WHERE verdict = 'PASS'
        GROUP BY signature
        HAVING freq >= ?
        ORDER BY freq DESC
    """, (min_frequency,)).fetchall()
    entries = []
    for sig, freq, sample_form_json in rows:
        try:
            sample_form = json.loads(sample_form_json)
        except Exception:
            continue
        structure = _form_tree_to_structure(sample_form["tree"])
        entries.append({
            "id": sig,
            "structure": structure,
            "frequency": freq,
        })
    return entries


def _form_tree_to_structure(tree_node: dict) -> dict:
    """Convert a stored form tree (with values, source words, indicators)
    into a pure-structure dict suitable as a catalog entry. Drops
    per-clue specifics — leaves only op, leaf-flag, children, and the
    structural sub-discriminators (positional_kind, deletion_kind,
    acrostic_kind)."""
    op = tree_node["operation"]
    is_leaf = "value" in tree_node and "sources" not in tree_node
    if is_leaf:
        d = {"op": op, "leaf": True}
        if op == "positional":
            d["positional_kind"] = tree_node.get("positional_kind")
        return d
    d = {"op": op,
          "children": [_form_tree_to_structure(c)
                       for c in tree_node.get("sources", [])]}
    if op == "deletion":
        d["deletion_kind"] = tree_node.get("deletion_kind")
    if op == "acrostic":
        d["acrostic_kind"] = tree_node.get("acrostic_kind")
    return d
