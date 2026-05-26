"""Shared clue-level pipeline entry points."""

from __future__ import annotations

from dataclasses import dataclass

from .sig_adapter import store_signature_evidence, store_signature_result
from .solver import clean, try_spoonerism_v2


@dataclass(frozen=True)
class CluePipelineResult:
    clue_id: int
    clue_text: str
    answer: str
    answer_clean: str
    solve_result: object
    evidence_ids: dict
    stored_solution: bool = False
    solved: bool = False
    tier: str | None = None
    result_dict: dict | None = None


def run_signature_clue_pipeline(
        conn, clue_id, source, puzzle_number, clue_text, answer, ref_db,
        *, manual_roles=None, extra_catalog=None, enriched=False,
        write_db=True, store_solution=True, solver_version=None):
    """Run the shared signature/WFW pipeline for one clue.

    Scope changes live outside this function: a rerun calls it once, while a
    puzzle run calls it once per clue. The clue-level behaviour is identical.
    """
    from signature_solver.solver import solve_clue as sig_solve_clue

    answer_clean = clean(answer)
    sr = sig_solve_clue(
        clue_text, answer_clean, ref_db,
        extra_catalog=extra_catalog,
        manual_roles=manual_roles)

    if solver_version is None:
        solver_version = (
            "signature_solver_enriched_pipeline:v1"
            if enriched else "signature_solver_pipeline:v1"
        )

    evidence_ids = {}
    stored_solution = False
    if write_db:
        evidence_ids = store_signature_evidence(
            conn, clue_id, source, puzzle_number, sr, clue_text, answer_clean,
            solver_version=solver_version)
        if store_solution and sr and sr.high_confidence:
            store_signature_result(
                conn, clue_id, sr, clue_text, answer_clean,
                enriched=enriched)
            stored_solution = True

    return CluePipelineResult(
        clue_id=clue_id,
        clue_text=clue_text,
        answer=answer,
        answer_clean=answer_clean,
        solve_result=sr,
        evidence_ids=evidence_ids,
        stored_solution=stored_solution,
        solved=bool(sr and sr.high_confidence),
        tier="Signature" if sr and sr.high_confidence else None,
        result_dict=None,
    )


def run_clue_pipeline(
        conn, clue_id, source, puzzle_number, clue_number, direction,
        clue_text, answer, enumeration, existing_explanation, ref_db,
        *, dd_graph=None, manual_roles=None, extra_catalog=None,
        enriched=False, write_db=True, store_solution=True,
        solver_version=None):
    """Run the one clue pipeline used by both rerun and puzzle scope.

    The old engines are still allowed, but they are called here, behind the
    same clue-level entry point as WFW/signature evidence.
    """
    signature = run_signature_clue_pipeline(
        conn, clue_id, source, puzzle_number, clue_text, answer, ref_db,
        manual_roles=manual_roles, extra_catalog=extra_catalog,
        enriched=enriched, write_db=write_db, store_solution=False,
        solver_version=solver_version)
    sr = signature.solve_result
    answer_clean = signature.answer_clean

    legacy = _try_legacy_engines(
        conn, clue_id, source, puzzle_number, clue_number, direction,
        clue_text, answer, answer_clean, enumeration, ref_db,
        dd_graph=dd_graph, write_db=write_db)
    if legacy is not None:
        if write_db:
            _mark_current_state_solved(
                conn, clue_id, source, puzzle_number, clue_text, answer_clean,
                sr, legacy["model_version"])
        return CluePipelineResult(
            clue_id=clue_id,
            clue_text=clue_text,
            answer=answer,
            answer_clean=answer_clean,
            solve_result=sr,
            evidence_ids=signature.evidence_ids,
            stored_solution=True,
            solved=True,
            tier=legacy["tier"],
            result_dict=legacy["result_dict"],
        )

    stored_solution = False
    if store_solution and write_db and sr and sr.high_confidence:
        store_signature_result(
            conn, clue_id, sr, clue_text, answer_clean, enriched=enriched)
        stored_solution = True

    result_dict = None
    if sr and sr.high_confidence:
        from .sig_adapter import build_result_dict
        result_dict = build_result_dict(
            sr, clue_text, answer, clue_number, direction, enumeration,
            existing_explanation)

    return CluePipelineResult(
        clue_id=clue_id,
        clue_text=clue_text,
        answer=answer,
        answer_clean=answer_clean,
        solve_result=sr,
        evidence_ids=signature.evidence_ids,
        stored_solution=stored_solution,
        solved=bool(sr and sr.high_confidence),
        tier="Signature" if sr and sr.high_confidence else None,
        result_dict=result_dict,
    )


def _try_legacy_engines(
        conn, clue_id, source, puzzle_number, clue_number, direction,
        clue_text, answer, answer_clean, enumeration, ref_db, *,
        dd_graph=None, write_db=True):
    if not answer_clean:
        return None
    if dd_graph is None:
        try:
            from backfill_ai_exp.backfill_dd_hidden import build_graph
            dd_graph = build_graph(ref_db)
        except Exception:
            dd_graph = {}

    hidden = _try_hidden_engine(
        clue_text, answer, answer_clean, dd_graph)
    if hidden is not None:
        _store_legacy_result(
            conn, clue_id, source, puzzle_number, clue_number,
            hidden["wordplay_type"], hidden["definition"],
            hidden["explanation"], hidden["components"],
            "mechanical_hidden", hidden["confidence"], write_db)
        return _legacy_pipeline_result(
            "Hidden", clue_number, direction, clue_text, answer, enumeration,
            hidden, "mechanical_hidden")

    spooner = _try_spooner_engine(clue_text, answer_clean, ref_db)
    if spooner is not None:
        _store_legacy_result(
            conn, clue_id, source, puzzle_number, clue_number,
            "spoonerism", None, spooner["explanation"],
            spooner["components"], "mechanical_spoonerism",
            spooner["confidence"], write_db)
        return _legacy_pipeline_result(
            "Spoonerism", clue_number, direction, clue_text, answer,
            enumeration, spooner, "mechanical_spoonerism")

    dd = _try_dd_engine(clue_text, answer, answer_clean, dd_graph)
    if dd is not None:
        _store_legacy_result(
            conn, clue_id, source, puzzle_number, clue_number,
            "double_definition", "Double definition", dd["explanation"],
            dd["components"], "mechanical_dd", dd["confidence"], write_db)
        return _legacy_pipeline_result(
            "DD", clue_number, direction, clue_text, answer, enumeration,
            dd, "mechanical_dd")

    v1 = _try_v1_engine(clue_text, answer, answer_clean, ref_db)
    if v1 is not None:
        _store_legacy_result(
            conn, clue_id, source, puzzle_number, clue_number,
            v1["wordplay_type"], v1["definition"], v1["explanation"],
            v1["components"], "mechanical_v1", v1["confidence"], write_db)
        return _legacy_pipeline_result(
            "Mechanical", clue_number, direction, clue_text, answer,
            enumeration, v1, "mechanical_v1")

    return None


def _try_hidden_engine(clue_text, answer, answer_clean, dd_graph):
    if len(answer_clean) < 3 or not dd_graph:
        return None
    import json
    from backfill_ai_exp.backfill_dd_hidden import (
        norm_letters,
        try_hidden,
    )
    result = try_hidden(
        clue_text, answer_clean, dd_graph, len(norm_letters(answer)))
    if not result:
        return None
    op = "hidden_reversed" if result["direction"] == "reverse" else "hidden"
    hiding_words = result.get("words", "")
    from sonnet_pipeline.report import _highlight_hidden
    if op == "hidden_reversed":
        highlighted = _highlight_hidden(hiding_words, answer_clean[::-1])
        explanation = 'hidden reversed in "%s"' % highlighted
    else:
        highlighted = _highlight_hidden(hiding_words, answer_clean)
        explanation = 'hidden in "%s"' % highlighted
    pieces = [{
        "clue_word": hiding_words,
        "letters": answer_clean,
        "mechanism": "hidden",
    }]
    return {
        "wordplay_type": op,
        "definition": result.get("definition"),
        "explanation": explanation,
        "components": json.dumps({
            "ai_pieces": pieces,
            "assembly": {"op": op, "words": hiding_words},
            "wordplay_type": op,
        }),
        "confidence": 1.0,
    }


def _try_spooner_engine(clue_text, answer_clean, ref_db):
    if len(answer_clean) < 4 or "spooner" not in clue_text.lower():
        return None
    import json
    result = try_spoonerism_v2(
        answer_clean, ref_db.is_real_word, clue_text=clue_text, ref_db=ref_db)
    if not result:
        return None
    w1 = result["word1"]
    w2 = result["word2"]
    sw1 = result["swapped1"]
    sw2 = result["swapped2"]
    cw1 = result.get("clue_word1")
    cw2 = result.get("clue_word2")
    parts = []
    parts.append('"%s" = %s' % (cw1, sw1) if cw1 else sw1)
    parts.append('"%s" = %s' % (cw2, sw2) if cw2 else sw2)
    explanation = "Spoonerism: %s -> swap initials -> %s %s" % (
        " + ".join(parts), w1, w2)
    return {
        "wordplay_type": "spoonerism",
        "definition": None,
        "explanation": explanation,
        "components": json.dumps({
            "ai_pieces": [{
                "clue_word": "%s %s" % (sw1, sw2),
                "letters": answer_clean,
                "mechanism": "spoonerism",
            }],
            "assembly": result,
            "wordplay_type": "spoonerism",
        }),
        "confidence": 1.0,
    }


def _try_dd_engine(clue_text, answer, answer_clean, dd_graph):
    if len(answer_clean) < 2 or not dd_graph:
        return None
    import json
    from backfill_ai_exp.backfill_dd_hidden import (
        generate_dd_hypotheses,
        norm_letters,
    )
    result = generate_dd_hypotheses(
        clue_text, dd_graph, total_len=len(norm_letters(answer)),
        answer=answer_clean)
    if not result:
        return None
    return {
        "wordplay_type": "double_definition",
        "definition": "Double definition",
        "explanation": "Double definition",
        "components": json.dumps({
            "ai_pieces": [],
            "assembly": {
                "op": "double_definition",
                "left_def": result["left_def"],
                "right_def": result["right_def"],
            },
            "wordplay_type": "double_definition",
        }),
        "confidence": 1.0,
    }


def _try_v1_engine(clue_text, answer, answer_clean, ref_db):
    if len(answer_clean) < 3:
        return None
    import json
    import re
    from backfill_ai_exp.batch_v1_solver import (
        build_explanation_text,
        find_definition,
        solve_without_definition,
        try_acrostic,
        try_anagram,
        try_charade,
        try_container,
        try_deletion,
        try_homophone,
        try_reversal,
    )
    from backfill_ai_exp.backfill_dd_hidden import norm_letters
    from backfill_ai_exp.batch_v1_solver import strip_enumeration

    if re.search(r"\b\d+\s*(?:across|down|ac|dn)\b", clue_text, re.I):
        return None

    definition, remaining = find_definition(clue_text, answer_clean, ref_db)
    if definition is None:
        try:
            from signature_solver.haiku_definition import find_definition as hdef
            haiku_result = hdef(clue_text, answer)
            if haiku_result:
                definition, remaining = haiku_result
        except Exception:
            pass
    if remaining is None:
        remaining = strip_enumeration(clue_text).split()

    mech_result = None
    wordplay_type = None
    pieces = None

    ana = try_anagram(
        clue_text, answer_clean, ref_db,
        definition_words=definition.split() if definition else None)
    if ana:
        wordplay_type = "anagram"
        pieces = [
            {
                "clue_word": word,
                "letters": norm_letters(word).upper(),
                "mechanism": "anagram_fodder",
            }
            for word in ana["fodder_words"]
        ]
        mech_result = ana

    for try_fn, wtype in (
            (try_container, "container"),
            (try_deletion, "deletion"),
            (try_charade, "charade"),
            (try_reversal, "reversal"),
            (try_acrostic, "acrostic"),
            (try_homophone, "homophone")):
        if mech_result:
            break
        result = try_fn(remaining, answer_clean, ref_db)
        if result:
            wordplay_type = wtype
            pieces = result["pieces"]
            mech_result = result

    if not mech_result and not definition:
        inferred_def, inferred_type, inferred_pieces = solve_without_definition(
            clue_text, answer_clean, ref_db)
        if inferred_def and inferred_pieces:
            definition = inferred_def
            wordplay_type = inferred_type
            pieces = inferred_pieces
            mech_result = True

    if not mech_result or not pieces:
        return None

    explanation = build_explanation_text(
        wordplay_type, pieces, definition, answer,
        clue_text=clue_text, ref_db=ref_db,
        assembly=mech_result if isinstance(mech_result, dict) else None)
    try:
        from sonnet_pipeline.verify_explanation import ExplanationVerifier
        verifier = ExplanationVerifier()
        verification = verifier.verify(
            clue_text, answer, definition, wordplay_type, explanation)
        confidence = {
            "HIGH": 1.0,
            "MEDIUM": 0.6,
            "LOW": 0.3,
            "FAIL": 0.0,
        }.get(verification.get("verdict"), 0.0)
    except Exception:
        confidence = 0.6
    return {
        "wordplay_type": wordplay_type,
        "definition": definition,
        "explanation": explanation,
        "components": json.dumps({
            "ai_pieces": pieces,
            "assembly": {"op": wordplay_type},
            "wordplay_type": wordplay_type,
        }),
        "confidence": confidence,
    }


def _store_legacy_result(
        conn, clue_id, source, puzzle_number, clue_number, wordplay_type,
        definition, explanation, components, model_version, confidence,
        write_db):
    if not write_db:
        return
    import json
    conn.execute(
        """INSERT OR REPLACE INTO structured_explanations
           (clue_id, components, wordplay_types, definition_text, confidence,
            model_version, source, puzzle_number, clue_number)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            clue_id,
            components,
            json.dumps([wordplay_type]),
            definition,
            confidence,
            model_version,
            source,
            str(puzzle_number) if puzzle_number is not None else None,
            clue_number,
        ),
    )
    conn.execute(
        """UPDATE clues SET
               wordplay_type = COALESCE(NULLIF(wordplay_type, ''), ?),
               definition = COALESCE(NULLIF(definition, ''), ?),
               ai_explanation = COALESCE(NULLIF(ai_explanation, ''), ?),
               has_solution = 1
           WHERE id = ?""",
        (wordplay_type, definition, explanation, clue_id),
    )


def _legacy_pipeline_result(
        tier, clue_number, direction, clue_text, answer, enumeration, legacy,
        model_version):
    return {
        "tier": tier,
        "model_version": model_version,
        "result_dict": {
            "status": "ASSEMBLED",
            "tier": tier,
            "confidence": "high" if legacy["confidence"] >= 0.8 else "medium",
            "score": int(legacy["confidence"] * 100),
            "clue_number": clue_number,
            "direction": direction,
            "enumeration": enumeration,
            "clue": clue_text,
            "answer": answer,
            "explanation": legacy["explanation"],
        },
    }


def _mark_current_state_solved(
        conn, clue_id, source, puzzle_number, clue_text, answer_clean, sr,
        solver_version):
    if sr is None:
        return
    from signature_solver.atomic_parse_store import (
        upsert_solve_result_pipeline_state,
    )
    upsert_solve_result_pipeline_state(
        clue_id, source, puzzle_number, clue_text, answer_clean, sr,
        conn=conn, solver_version=solver_version,
        status_override="solved", confidence_override=100)
