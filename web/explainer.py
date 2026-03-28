"""Async explanation generator — calls Claude Sonnet to explain clues on demand.

When a user visits a clue page and the clue has an answer but incomplete hints,
this module runs the full Sonnet pipeline (API call + assembler + validation),
stores the result in the clues table, and logs it for review.
"""

import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CRYPTIC_DB = PROJECT_ROOT / "data" / "cryptic_new.db"
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"

# Reuse the same model as the batch pipeline
SONNET_MODEL = "claude-sonnet-4-20250514"

# Rate limiting: max calls per minute
MAX_CALLS_PER_MINUTE = 10

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (initialised on first API request)
# ---------------------------------------------------------------------------

_enricher = None
_homo_engine = None
_example_messages = None
_ref_db = None
_init_lock = threading.Lock()

# Rate limiter state
_call_timestamps = []
_rate_lock = threading.Lock()


def _init_pipeline():
    """Lazy-init all pipeline singletons (thread-safe)."""
    global _enricher, _homo_engine, _example_messages, _ref_db
    if _enricher is None:
        with _init_lock:
            if _enricher is None:
                from sonnet_pipeline.enricher import ClueEnricher
                from sonnet_pipeline.solver import HomophoneEngine, build_example_messages
                from signature_solver.db import RefDB
                _enricher = ClueEnricher(db_path=CRYPTIC_DB)
                _homo_engine = HomophoneEngine(db_path=str(CRYPTIC_DB))
                _example_messages = build_example_messages()
                _ref_db = RefDB()


def _check_rate_limit():
    """Return True if we're under the rate limit, False if exceeded."""
    now = time.time()
    with _rate_lock:
        _call_timestamps[:] = [t for t in _call_timestamps if now - t < 60]
        if len(_call_timestamps) >= MAX_CALLS_PER_MINUTE:
            return False
        _call_timestamps.append(now)
        return True


def _ensure_api_explanations_table(conn):
    """Create the api_explanations log table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            clue_id INTEGER NOT NULL,
            definition TEXT,
            wordplay_type TEXT,
            ai_explanation TEXT,
            previous_definition TEXT,
            previous_wordplay_type TEXT,
            previous_ai_explanation TEXT,
            raw_api_output TEXT,
            assembly_op TEXT,
            score INTEGER,
            model TEXT,
            tokens_in INTEGER,
            tokens_out INTEGER,
            api_ms INTEGER,
            assembly_ms INTEGER,
            total_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed INTEGER DEFAULT 0,
            reviewed_at TIMESTAMP,
            FOREIGN KEY (clue_id) REFERENCES clues(id)
        )
    """)
    conn.commit()


def _build_explanation_from_pieces(ai_output):
    """Build a human-readable explanation string from pipeline output.

    Uses the assembler-corrected pieces (not raw API output).
    """
    if not ai_output:
        return None

    wordplay_type = ai_output.get("wordplay_type", "")

    if wordplay_type == "double_definition":
        definition = ai_output.get("definition", "")
        return f"Double definition: {definition}"

    if wordplay_type == "cryptic_definition":
        return "The whole clue is a cryptic definition."

    pieces = ai_output.get("pieces", [])
    if not pieces:
        return None

    parts = []
    for p in pieces:
        clue_word = p.get("clue_word", "")
        letters = p.get("letters", "")
        mechanism = p.get("mechanism", "")
        if clue_word and letters:
            mech_label = mechanism.replace("_", " ")
            parts.append(f"{clue_word} \u2192 {letters} ({mech_label})")

    if not parts:
        return None

    explanation = " + ".join(parts)
    if wordplay_type:
        explanation += f"  [{wordplay_type}]"

    return explanation


def generate_explanation(clue_id):
    """Generate an explanation for a clue via the full Sonnet pipeline.

    Runs: API call -> assembler -> fallback -> validation -> scoring.

    Returns (success: bool, message: str, result: dict or None).
    Result dict contains: definition, wordplay_type, ai_explanation, timing.
    """
    if not _check_rate_limit():
        return False, "Busy \u2014 too many requests. Please try again shortly.", None

    total_start = time.time()

    # Fetch the clue
    conn = sqlite3.connect(str(CLUES_DB))
    conn.row_factory = sqlite3.Row
    try:
        clue = conn.execute(
            "SELECT * FROM clues WHERE id = ?", (clue_id,)
        ).fetchone()

        if clue is None:
            return False, "Clue not found.", None

        clue_text = clue["clue_text"]
        answer = clue["answer"]

        if not answer:
            return False, "No answer available for this clue.", None

        # Save previous values for the audit log
        prev_def = clue["definition"]
        prev_type = clue["wordplay_type"]
        prev_expl = clue["ai_explanation"]

        # Init pipeline singletons
        _init_pipeline()

        # Build enrichment
        enricher = _enricher
        enrichment = enricher.enrich(clue_text, answer)

        # Run the full pipeline: API call + assembler + validation
        from sonnet_pipeline.solver import solve_clue
        api_start = time.time()
        result = solve_clue(
            clue_text, answer, enrichment, enricher,
            _homo_engine, _example_messages,
            ref_db=_ref_db,
        )
        api_ms = int((time.time() - api_start) * 1000)

        # Time the post-API work (assembler ran inside solve_clue,
        # so api_ms includes everything — we'll log it as total pipeline time)
        tokens_in = result["tokens_in"]
        tokens_out = result["tokens_out"]
        ai_output = result["ai_output"]
        assembly = result["assembly"]
        validation = result["validation"]
        pipeline_tier = result["tier"]  # "Sonnet", "Fallback", or None
        score = validation.get("score", 0) if validation else 0

        # Extract the three fields from pipeline output
        definition = ai_output.get("definition") if ai_output else None
        wordplay_type = ai_output.get("wordplay_type") if ai_output else None
        ai_explanation = _build_explanation_from_pieces(ai_output)

        # If pipeline completely failed, still store what the API returned
        raw_api_json = json.dumps(ai_output) if ai_output else None

        assembly_op = assembly.get("op") if assembly else None

        total_ms = int((time.time() - total_start) * 1000)

        log.info(
            "clue_id=%d score=%d tier=%s op=%s api=%dms total=%dms",
            clue_id, score, pipeline_tier, assembly_op, api_ms, total_ms,
        )

        # Build assembler-style explanation and run verifier (matching pipeline)
        from sonnet_pipeline.report import _describe_assembly
        from sonnet_pipeline.verify_explanation import ExplanationVerifier
        from sonnet_pipeline.solver import OP_TO_TYPE

        # Build structured pieces (matching store_result format)
        ai_pieces = []
        if ai_output:
            for p in ai_output.get("pieces", []):
                piece = {
                    "mechanism": p.get("mechanism", "unknown"),
                    "clue_word": p.get("clue_word", ""),
                    "letters": p.get("letters", ""),
                }
                for key in ("source", "indicator", "deleted", "deleted_word"):
                    if p.get(key):
                        piece[key] = p[key]
                ai_pieces.append(piece)

        # Determine wordplay types (assembler first, AI fallback)
        asm_op = assembly.get("op", "") if assembly else ""
        asm_type = OP_TO_TYPE.get(asm_op, asm_op)
        wordplay_types = []
        if asm_type:
            wordplay_types.append(asm_type)
        if wordplay_type and wordplay_type not in wordplay_types:
            wordplay_types.append(wordplay_type)
        if not wordplay_types:
            wordplay_types = ["unknown"]

        # Build explanation from assembly (same as pipeline)
        explanation_text = _describe_assembly(assembly, ai_pieces, answer=answer) if assembly else None

        # Run mechanical verifier for confidence (same as pipeline)
        if explanation_text:
            _verifier = ExplanationVerifier()
            v_result = _verifier.verify(
                clue_text, answer, definition or "",
                wordplay_types[0], explanation_text,
            )
            confidence = v_result["score"] / 100.0 if v_result else score / 100.0
        else:
            confidence = score / 100.0

        # Build components dict (matching pipeline structure)
        components_dict = {
            "ai_pieces": ai_pieces,
            "assembly": assembly,
            "wordplay_type": wordplay_types[0],
        }

        # Use assembler explanation if available, fall back to piece-based
        store_explanation = explanation_text or ai_explanation

        # Decide what to store — even low-score results are useful as
        # "unreviewed" content, better than showing nothing
        if store_explanation or definition or wordplay_type:
            # Determine has_solution and reviewed (matching pipeline logic)
            has_def = bool(definition)
            has_type = bool(wordplay_types[0]) and wordplay_types[0] != "unknown"
            has_expl = bool(store_explanation) or bool(ai_pieces)

            # Check for existing manual review
            current_reviewed = conn.execute(
                "SELECT reviewed FROM clues WHERE id = ?", (clue_id,)
            ).fetchone()
            already_reviewed = current_reviewed and current_reviewed[0] in (1, 2)
            if not already_reviewed:
                auto_reviewed = 1 if score >= 80 else 0
            else:
                auto_reviewed = current_reviewed[0]

            if has_def and has_type and has_expl:
                has_solution = 1
            elif has_type and has_expl and score >= 80:
                has_solution = 1
            elif has_def or has_type:
                has_solution = 2
            else:
                has_solution = 0

            # Write to clues table
            conn.execute(
                """UPDATE clues
                   SET definition = ?, wordplay_type = ?, ai_explanation = ?,
                       reviewed = ?, has_solution = ?
                   WHERE id = ?""",
                (definition, wordplay_types[0], store_explanation,
                 auto_reviewed, has_solution, clue_id),
            )

            # Compute definition position
            def_start = None
            def_end = None
            if definition:
                idx = clue_text.lower().find(definition.lower())
                if idx >= 0:
                    def_start = idx
                    def_end = idx + len(definition)

            # Fetch source metadata
            clue_meta = conn.execute(
                "SELECT source, puzzle_number, clue_number FROM clues WHERE id = ?",
                (clue_id,)
            ).fetchone()
            src, pnum, cnum = clue_meta if clue_meta else (None, None, None)

            # Write to structured_explanations (full columns matching pipeline)
            conn.execute(
                "DELETE FROM structured_explanations WHERE clue_id = ?",
                (clue_id,),
            )
            conn.execute(
                """INSERT INTO structured_explanations
                   (clue_id, definition_text, definition_start, definition_end,
                    wordplay_types, components, model_version, confidence,
                    source, puzzle_number, clue_number,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                (clue_id, definition, def_start, def_end,
                 json.dumps(wordplay_types), json.dumps(components_dict),
                 "haiku_sonnet_tiered_v1", confidence,
                 src, pnum, cnum),
            )

        # Always log to api_explanations table
        _ensure_api_explanations_table(conn)
        conn.execute(
            """INSERT INTO api_explanations
               (clue_id, definition, wordplay_type, ai_explanation,
                previous_definition, previous_wordplay_type, previous_ai_explanation,
                raw_api_output, assembly_op, score, model,
                tokens_in, tokens_out, api_ms, assembly_ms, total_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                clue_id, definition, wordplay_type, ai_explanation,
                prev_def, prev_type, prev_expl,
                raw_api_json, assembly_op, score, SONNET_MODEL,
                tokens_in, tokens_out, api_ms, 0, total_ms,
            ),
        )
        conn.commit()

        if not (ai_explanation or definition or wordplay_type):
            return False, "Pipeline could not generate an explanation for this clue.", None

        return True, "OK", {
            "definition": definition,
            "wordplay_type": wordplay_type,
            "ai_explanation": ai_explanation,
            "score": score,
            "assembly_op": assembly_op,
            "pipeline_tier": pipeline_tier,
            "api_ms": api_ms,
            "total_ms": total_ms,
        }

    finally:
        conn.close()
