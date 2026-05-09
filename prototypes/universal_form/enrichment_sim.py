"""Enrichment-cycle simulation — mimics the production stages that fire
on a verifier FAIL.

Production flow (paraphrased from the live code):

    1. Verifier returns FAIL with per-check failures.
    2. For each "DB gap" failure, a Haiku helper (haiku_definition,
       haiku_indicator, haiku_dbe) confirms/refines the proposal.
    3. enrichment_gate.already_in_reference_db dedupes against the
       canonical DB.
    4. The proposal is INSERTed into pending_enrichments.
    5. A human reviews, approves or rejects.
    6. Approved proposals end up in cryptic_new.db.
    7. Next verifier run picks them up; the same form may now PASS.

This module simulates 1-7 read-only:

    - collect_proposals(verdict)            extract DB-gap proposals
    - dedupe(proposals)                     filter via enrichment_gate
    - counterfactual_verify(form, clue,
                            verifier, proposals)
                                            re-verify treating proposals
                                            as if approved
"""
from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Import the production gate so the simulation matches production behaviour.
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from sonnet_pipeline.enrichment_gate import already_in_reference_db
except Exception:
    already_in_reference_db = None  # graceful degradation

from .schema import Form
from .verifier import FormVerifier, Verdict, Check


@dataclass
class Proposal:
    """A proposed enrichment, mirroring the pending_enrichments shape."""
    type: str            # synonym | abbreviation | definition | indicator
    word: str            # the source word
    letters: str         # the produced letters (or op_type for indicators)
    op_type: Optional[str] = None  # for indicator proposals
    dedupe_status: str = "new"   # new | already_in_db | already_pending
    source_check: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type, "word": self.word, "letters": self.letters,
            "op_type": self.op_type, "dedupe_status": self.dedupe_status,
            "source_check": self.source_check,
        }


def collect_proposals(verdict: Verdict) -> list[Proposal]:
    """Walk verdict.checks and pull out every enrichment_proposal from
    failed checks."""
    out: list[Proposal] = []
    for c in verdict.checks:
        if c.status != "fail":
            continue
        if not c.enrichment_proposal:
            continue
        p = c.enrichment_proposal
        out.append(Proposal(
            type=p["type"],
            word=p["word"],
            letters=p["letters"],
            op_type=p.get("op_type"),
            source_check=c.name,
        ))
    return out


def dedupe(proposals: list[Proposal]) -> list[Proposal]:
    """Filter via the production enrichment_gate. Each proposal's
    dedupe_status is updated in place."""
    if not already_in_reference_db:
        return proposals
    for p in proposals:
        try:
            if already_in_reference_db(p.type, p.word, p.letters):
                p.dedupe_status = "already_in_db"
            else:
                p.dedupe_status = "new"
        except Exception:
            p.dedupe_status = "new"
    return proposals


def counterfactual_verify(form: Form, clue_text: str,
                          verifier: FormVerifier,
                          proposals: list[Proposal]) -> Verdict:
    """Re-verify the form treating each NEW proposal as if it were
    approved into the DB.

    Implementation: monkey-patch the verifier's lookup methods to return
    True for any (word, letters) that matches a proposal; restore on exit.
    """
    if not proposals:
        # No DB-gap proposals — counterfactual = current verdict
        return verifier.verify(form, clue_text)

    # Build lookup sets for fast membership checks
    syn_set = set()
    abbr_set = set()
    def_set = set()
    ind_set = set()
    for p in proposals:
        if p.dedupe_status == "already_in_db":
            continue
        if p.type == "synonym":
            syn_set.add((p.word.lower(), p.letters.lower()))
            syn_set.add((p.letters.lower(), p.word.lower()))
        elif p.type == "abbreviation":
            abbr_set.add((p.word.lower(), p.letters.upper()))
        elif p.type == "definition":
            def_set.add((p.word.lower(), p.letters.upper()))
        elif p.type == "indicator":
            ind_set.add((p.word.lower(), (p.op_type or "").lower()))

    orig_is_syn = verifier.is_synonym
    orig_is_abbr = verifier.is_abbreviation
    orig_is_def = verifier.definition_matches
    orig_is_ind = verifier.is_indicator

    def cf_is_syn(word, target):
        if (word.lower(), target.lower()) in syn_set:
            return True
        return orig_is_syn(word, target)

    def cf_is_abbr(word, letters):
        if (word.lower(), letters.upper()) in abbr_set:
            return True
        return orig_is_abbr(word, letters)

    def cf_is_def(definition, answer):
        if (definition.lower(), answer.upper()) in def_set:
            return True
        return orig_is_def(definition, answer)

    def cf_is_ind(word, op_type):
        if (word.lower(), op_type.lower()) in ind_set:
            return True
        return orig_is_ind(word, op_type)

    verifier.is_synonym = cf_is_syn
    verifier.is_abbreviation = cf_is_abbr
    verifier.definition_matches = cf_is_def
    verifier.is_indicator = cf_is_ind
    # Caches were populated by previous verify() calls; clear them so the
    # counterfactual lookups re-run.
    verifier._syn_cache.clear()
    verifier._abbr_cache.clear()
    verifier._def_cache.clear()
    verifier._ind_cache.clear()

    try:
        return verifier.verify(form, clue_text)
    finally:
        verifier.is_synonym = orig_is_syn
        verifier.is_abbreviation = orig_is_abbr
        verifier.definition_matches = orig_is_def
        verifier.is_indicator = orig_is_ind
        verifier._syn_cache.clear()
        verifier._abbr_cache.clear()
        verifier._def_cache.clear()
        verifier._ind_cache.clear()
